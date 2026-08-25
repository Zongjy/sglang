#!/usr/bin/env python3
"""Summarize SGLang Torch profiler traces into per-PP-stage costs.

The parser deliberately uses only the Chrome trace schema emitted by
``torch.profiler``.  It treats consecutive ``run_batch`` annotations as
steady-state iteration windows and measures the union of GPU activity inside
each window.  DFlash target/draft attribution is best-effort: kernels are
matched to the new record-function ranges through external/correlation ids.
"""

from __future__ import annotations

import gzip
import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

PP_RANK = re.compile(r"(?:^|[-_.])PP[-_](\d+)(?:[-_.]|$)", re.IGNORECASE)
TP_RANK = re.compile(r"(?:^|[-_.])TP[-_](\d+)(?:[-_.]|$)", re.IGNORECASE)

TARGET_MARKERS = {
    "sglang.dflash.target_verify_forward",
    "sglang.dflash.target_prefill_forward",
}
VERIFY_MARKERS = {"sglang.dflash.target_verify_forward"}
DRAFT_MARKERS = {"sglang.dflash.draft_model_forward"}
PP_WAIT_PREFIXES = (
    "scheduler.pp.wait_",
    "recv_res_dict_from_prev_stage",
)


class TraceProfileError(RuntimeError):
    pass


@dataclass(frozen=True)
class Interval:
    start_us: float
    end_us: float

    @property
    def duration_us(self) -> float:
        return max(self.end_us - self.start_us, 0.0)


@dataclass
class RankTraceProfile:
    path: str
    pp_rank: int
    tp_rank: int
    service_samples_ms: list[float]
    target_samples_ms: list[float]
    draft_samples_ms: list[float]
    wait_samples_ms: list[float]
    used_gpu_events: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def service_median_ms(self) -> float:
        return _median(self.service_samples_ms)


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _variance_of_mean(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pvariance(values) / len(values))


def _read_trace(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceProfileError(f"cannot read profiler trace {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("traceEvents"), list
    ):
        raise TraceProfileError(f"{path} is not a Chrome trace with traceEvents")
    return payload


def _rank_from_path(path: Path, pattern: re.Pattern[str], label: str) -> int:
    match = pattern.search(path.name)
    if match is None:
        raise TraceProfileError(
            f"cannot infer {label} rank from trace filename {path.name!r}"
        )
    return int(match.group(1))


def _as_interval(event: dict[str, Any]) -> Interval | None:
    if event.get("ph") != "X":
        return None
    try:
        start = float(event["ts"])
        duration = float(event.get("dur", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if duration <= 0.0:
        return None
    return Interval(start, start + duration)


def _merge(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = sorted(intervals, key=lambda item: (item.start_us, item.end_us))
    merged: list[Interval] = []
    for interval in ordered:
        if not merged or interval.start_us > merged[-1].end_us:
            merged.append(interval)
        else:
            previous = merged[-1]
            merged[-1] = Interval(
                previous.start_us, max(previous.end_us, interval.end_us)
            )
    return merged


def _overlap_us(intervals: Sequence[Interval], window: Interval) -> float:
    return sum(
        max(
            0.0,
            min(interval.end_us, window.end_us)
            - max(interval.start_us, window.start_us),
        )
        for interval in intervals
    )


def _contains(intervals: Sequence[Interval], timestamp_us: float) -> bool:
    return any(item.start_us <= timestamp_us <= item.end_us for item in intervals)


def _arg_id(args: Any, names: Sequence[str]) -> str | None:
    if not isinstance(args, dict):
        return None
    for name in names:
        value = args.get(name)
        if value is not None:
            return str(value)
    return None


def _is_gpu_event(event: dict[str, Any]) -> bool:
    category = str(event.get("cat", "")).lower()
    if category in {"kernel", "gpu_memcpy", "gpu_memset"}:
        return True
    return "kernel" in category and "cuda_runtime" not in category


def _category_for_timestamp(
    timestamp_us: float,
    target_markers: Sequence[Interval],
    draft_markers: Sequence[Interval],
) -> str | None:
    if _contains(target_markers, timestamp_us):
        return "target"
    if _contains(draft_markers, timestamp_us):
        return "draft"
    return None


def _classified_gpu_intervals(
    events: Sequence[dict[str, Any]],
    target_markers: Sequence[Interval],
    draft_markers: Sequence[Interval],
) -> tuple[list[Interval], list[Interval], list[Interval]]:
    external_category: dict[str, str] = {}
    correlation_category: dict[str, str] = {}

    for event in events:
        interval = _as_interval(event)
        if interval is None or _is_gpu_event(event):
            continue
        category = _category_for_timestamp(
            interval.start_us, target_markers, draft_markers
        )
        if category is None:
            continue
        external_id = _arg_id(event.get("args"), ("External id", "external id"))
        if external_id is not None:
            external_category[external_id] = category
        correlation = _arg_id(
            event.get("args"),
            ("correlation", "Correlation id", "correlation_id"),
        )
        if correlation is not None:
            correlation_category[correlation] = category

    for event in events:
        category_name = str(event.get("cat", "")).lower()
        if "cuda_runtime" not in category_name:
            continue
        interval = _as_interval(event)
        if interval is None:
            continue
        category = _category_for_timestamp(
            interval.start_us, target_markers, draft_markers
        )
        correlation = _arg_id(
            event.get("args"),
            ("correlation", "Correlation id", "correlation_id"),
        )
        if category is not None and correlation is not None:
            correlation_category[correlation] = category

    all_gpu: list[Interval] = []
    target_gpu: list[Interval] = []
    draft_gpu: list[Interval] = []
    for event in events:
        if not _is_gpu_event(event):
            continue
        interval = _as_interval(event)
        if interval is None:
            continue
        all_gpu.append(interval)
        args = event.get("args")
        external_id = _arg_id(args, ("External id", "external id"))
        correlation = _arg_id(args, ("correlation", "Correlation id", "correlation_id"))
        category = external_category.get(external_id or "") or correlation_category.get(
            correlation or ""
        )
        if category == "target":
            target_gpu.append(interval)
        elif category == "draft":
            draft_gpu.append(interval)
    return _merge(all_gpu), _merge(target_gpu), _merge(draft_gpu)


def parse_rank_trace(path: Path, trim_samples: int = 1) -> RankTraceProfile:
    payload = _read_trace(path)
    events = [event for event in payload["traceEvents"] if isinstance(event, dict)]
    pp_rank = _rank_from_path(path, PP_RANK, "PP")
    tp_rank = _rank_from_path(path, TP_RANK, "TP")

    run_batches = sorted(
        (
            interval
            for event in events
            if event.get("name") == "run_batch"
            if (interval := _as_interval(event)) is not None
        ),
        key=lambda item: item.start_us,
    )
    if len(run_batches) < 2:
        raise TraceProfileError(
            f"{path} has fewer than two run_batch ranges; increase --profile-steps"
        )

    target_markers = _merge(
        interval
        for event in events
        if event.get("name") in TARGET_MARKERS
        if (interval := _as_interval(event)) is not None
    )
    verify_markers = _merge(
        interval
        for event in events
        if event.get("name") in VERIFY_MARKERS
        if (interval := _as_interval(event)) is not None
    )
    draft_markers = _merge(
        interval
        for event in events
        if event.get("name") in DRAFT_MARKERS
        if (interval := _as_interval(event)) is not None
    )
    wait_intervals = _merge(
        interval
        for event in events
        if any(
            str(event.get("name", "")).startswith(prefix) for prefix in PP_WAIT_PREFIXES
        )
        if (interval := _as_interval(event)) is not None
    )
    all_gpu, target_gpu, draft_gpu = _classified_gpu_intervals(
        events, target_markers, draft_markers
    )

    windows: list[Interval] = []
    for index, batch in enumerate(run_batches[:-1]):
        windows.append(Interval(batch.start_us, run_batches[index + 1].start_us))

    # DFlash traces contain prefill and decode in the same capture. Retain only
    # windows containing target-verify work when those markers are available.
    if verify_markers:
        decode_windows = [
            window
            for window in windows
            if any(
                window.start_us <= marker.start_us < window.end_us
                for marker in verify_markers
            )
        ]
        if decode_windows:
            windows = decode_windows

    if trim_samples > 0 and len(windows) > 2 * trim_samples:
        windows = windows[trim_samples:-trim_samples]
    if not windows:
        raise TraceProfileError(f"{path} has no usable steady-state run_batch windows")

    used_gpu_events = bool(all_gpu)
    service_samples = []
    target_samples = []
    draft_samples = []
    wait_samples = []
    warnings: list[str] = []
    for window in windows:
        if used_gpu_events:
            service_us = _overlap_us(all_gpu, window)
            target_us = _overlap_us(target_gpu, window)
            draft_us = _overlap_us(draft_gpu, window)
        else:
            service_us = _overlap_us(run_batches, window)
            target_us = _overlap_us(target_markers, window)
            draft_us = _overlap_us(draft_markers, window)
        if service_us <= 0.0:
            continue
        service_samples.append(service_us / 1000.0)
        target_samples.append(target_us / 1000.0)
        draft_samples.append(draft_us / 1000.0)
        wait_samples.append(_overlap_us(wait_intervals, window) / 1000.0)

    if not service_samples:
        raise TraceProfileError(f"{path} has no positive service-time samples")
    if not used_gpu_events:
        warnings.append(
            "trace has no GPU events; service times use CPU run_batch ranges"
        )
    if target_markers and not any(value > 0 for value in target_samples):
        warnings.append(
            "DFlash target markers were found but GPU correlation was unavailable"
        )
    if draft_markers and not any(value > 0 for value in draft_samples):
        warnings.append(
            "DFlash draft markers were found but GPU correlation was unavailable"
        )

    return RankTraceProfile(
        path=str(path),
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        service_samples_ms=service_samples,
        target_samples_ms=target_samples,
        draft_samples_ms=draft_samples,
        wait_samples_ms=wait_samples,
        used_gpu_events=used_gpu_events,
        warnings=warnings,
    )


def summarize_trace_dir(
    trace_dir: Path,
    *,
    pp_size: int,
    trim_samples: int = 1,
) -> dict[str, Any]:
    paths = sorted(path for path in trace_dir.rglob("*.trace.json*") if path.is_file())
    if not paths:
        raise TraceProfileError(f"no *.trace.json[.gz] files found under {trace_dir}")

    profiles = [parse_rank_trace(path, trim_samples=trim_samples) for path in paths]
    by_pp: dict[int, list[RankTraceProfile]] = {}
    for profile in profiles:
        by_pp.setdefault(profile.pp_rank, []).append(profile)
    missing = [rank for rank in range(pp_size) if rank not in by_pp]
    if missing:
        raise TraceProfileError(f"trace set is missing PP rank(s) {missing}")

    selected: list[RankTraceProfile] = []
    for pp_rank in range(pp_size):
        # A TP stage advances at its slowest shard. Use the shard with the
        # largest median service time as the conservative stage observation.
        selected.append(max(by_pp[pp_rank], key=lambda item: item.service_median_ms))

    service_ms = [item.service_median_ms for item in selected]
    target_ms = [_median(item.target_samples_ms) for item in selected]
    draft_ms = [_median(item.draft_samples_ms) for item in selected]
    wait_ms = [_median(item.wait_samples_ms) for item in selected]
    return {
        "schema_version": 1,
        "trace_dir": str(trace_dir),
        "service_ms": service_ms,
        "service_var": [
            _variance_of_mean(item.service_samples_ms) for item in selected
        ],
        "target_ms": target_ms,
        "target_var": [_variance_of_mean(item.target_samples_ms) for item in selected],
        "draft_ms": draft_ms[-1] if draft_ms else 0.0,
        "draft_var": (
            _variance_of_mean(selected[-1].draft_samples_ms) if selected else 0.0
        ),
        "wait_ms": wait_ms,
        "wait_fraction": [
            wait / service if service > 0 else 0.0
            for wait, service in zip(wait_ms, service_ms)
        ],
        "samples": min(len(item.service_samples_ms) for item in selected),
        "selected_traces": [asdict(item) for item in selected],
        "all_traces": [asdict(item) for item in profiles],
        "warnings": [warning for item in profiles for warning in item.warnings],
    }
