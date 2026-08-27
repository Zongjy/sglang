#!/usr/bin/env python3
"""Extract intrinsic PP stage costs from Kineto Chrome traces."""

from __future__ import annotations

import gzip
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PP_RANK = re.compile(r"(?:^|[-_.])PP[-_](\d+)(?:[-_.]|$)", re.IGNORECASE)
TP_RANK = re.compile(r"(?:^|[-_.])TP[-_](\d+)(?:[-_.]|$)", re.IGNORECASE)
VERIFY_MARKER = "sglang.dflash.target_verify_forward"
P2P_KERNEL = re.compile(r"(?:nccl|rccl).*?(?:sendrecv|send|recv)", re.IGNORECASE)


class TraceProfileError(RuntimeError):
    pass


@dataclass(frozen=True)
class Interval:
    start_us: float
    end_us: float


@dataclass
class RankTraceProfile:
    pp_rank: int
    tp_rank: int
    intrinsic_samples_ms: list[float]
    target_samples_ms: list[float]
    pp_p2p_windows: int

    @property
    def intrinsic_median_ms(self) -> float:
        return float(statistics.median(self.intrinsic_samples_ms))

    @property
    def target_median_ms(self) -> float:
        return float(statistics.median(self.target_samples_ms))


def _read_trace(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceProfileError(f"cannot read profiler trace {path}: {exc}") from exc
    events = payload.get("traceEvents") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        raise TraceProfileError(f"{path} is not a Kineto Chrome trace")
    return [event for event in events if isinstance(event, dict)]


def _rank_from_path(path: Path, pattern: re.Pattern[str], label: str) -> int:
    match = pattern.search(path.name)
    if match is None:
        raise TraceProfileError(f"cannot infer {label} rank from {path.name!r}")
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
            merged[-1] = Interval(
                merged[-1].start_us, max(merged[-1].end_us, interval.end_us)
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
    return category in {"kernel", "gpu_memcpy", "gpu_memset"} or (
        "kernel" in category and "cuda_runtime" not in category
    )


def _is_pp_p2p(event: dict[str, Any]) -> bool:
    args = event.get("args")
    if not isinstance(args, dict):
        return False
    return (
        str(args.get("Process Group Description", "")).strip().casefold()
        == "pp:device"
        and str(args.get("Collective name", "")).strip().casefold()
        in {"send", "recv"}
    )


def _is_p2p_candidate(event: dict[str, Any]) -> bool:
    args = event.get("args")
    collective = args.get("Collective name") if isinstance(args, dict) else None
    return str(collective).strip().casefold() in {"send", "recv"} or bool(
        P2P_KERNEL.search(str(event.get("name", "")))
    )


def _target_gpu_intervals(
    events: Sequence[dict[str, Any]], verify_markers: Sequence[Interval]
) -> list[Interval]:
    external_ids: set[str] = set()
    correlation_ids: set[str] = set()
    for event in events:
        interval = _as_interval(event)
        if interval is None or _is_gpu_event(event):
            continue
        if not _contains(verify_markers, interval.start_us):
            continue
        external_id = _arg_id(event.get("args"), ("External id", "external id"))
        if external_id is not None:
            external_ids.add(external_id)
        correlation = _arg_id(
            event.get("args"),
            ("correlation", "Correlation id", "correlation_id"),
        )
        if correlation is not None:
            correlation_ids.add(correlation)

    for event in events:
        if "cuda_runtime" not in str(event.get("cat", "")).lower():
            continue
        interval = _as_interval(event)
        if interval is None or not _contains(verify_markers, interval.start_us):
            continue
        correlation = _arg_id(
            event.get("args"),
            ("correlation", "Correlation id", "correlation_id"),
        )
        if correlation is not None:
            correlation_ids.add(correlation)

    target_gpu: list[Interval] = []
    for event in events:
        if not _is_gpu_event(event) or _is_pp_p2p(event):
            continue
        interval = _as_interval(event)
        if interval is None:
            continue
        args = event.get("args")
        external_id = _arg_id(args, ("External id", "external id"))
        correlation = _arg_id(
            args, ("correlation", "Correlation id", "correlation_id")
        )
        if (external_id is not None and external_id in external_ids) or (
            correlation is not None and correlation in correlation_ids
        ):
            target_gpu.append(interval)
    return _merge(target_gpu)


def parse_rank_trace(path: Path, trim_samples: int = 1) -> RankTraceProfile:
    if trim_samples < 0:
        raise TraceProfileError("trim_samples cannot be negative")
    events = _read_trace(path)
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
        raise TraceProfileError(f"{path} has fewer than two run_batch ranges")

    verify_markers = _merge(
        interval
        for event in events
        if event.get("name") == VERIFY_MARKER
        if (interval := _as_interval(event)) is not None
    )
    if not verify_markers:
        raise TraceProfileError(f"{path} has no {VERIFY_MARKER} marker")

    all_gpu = [
        interval
        for event in events
        if _is_gpu_event(event)
        if (interval := _as_interval(event)) is not None
    ]
    if not all_gpu:
        raise TraceProfileError(f"{path} has no GPU events")
    ambiguous_p2p = []
    for event in events:
        if not _is_gpu_event(event) or not _is_p2p_candidate(event):
            continue
        if _is_pp_p2p(event):
            continue
        args = event.get("args")
        process_group = (
            str(args.get("Process Group Description", "")).strip().casefold()
            if isinstance(args, dict)
            else ""
        )
        if not process_group or process_group == "pp:device":
            ambiguous_p2p.append(event)
    if ambiguous_p2p:
        raise TraceProfileError(
            f"{path} has {len(ambiguous_p2p)} Send/Recv GPU events without "
            "complete process-group metadata"
        )
    pp_p2p = _merge(
        interval
        for event in events
        if _is_gpu_event(event) and _is_pp_p2p(event)
        if (interval := _as_interval(event)) is not None
    )
    intrinsic_gpu = _merge(
        interval
        for event in events
        if _is_gpu_event(event) and not _is_pp_p2p(event)
        if (interval := _as_interval(event)) is not None
    )
    target_gpu = _target_gpu_intervals(events, verify_markers)
    if not target_gpu:
        raise TraceProfileError(
            f"{path} cannot correlate {VERIFY_MARKER} to GPU events"
        )

    windows = [
        Interval(batch.start_us, run_batches[index + 1].start_us)
        for index, batch in enumerate(run_batches[:-1])
        if any(
            batch.start_us <= marker.start_us < run_batches[index + 1].start_us
            for marker in verify_markers
        )
    ]
    if trim_samples > 0 and len(windows) > 2 * trim_samples:
        windows = windows[trim_samples:-trim_samples]
    if not windows:
        raise TraceProfileError(f"{path} has no steady-state verify windows")

    intrinsic_samples: list[float] = []
    target_samples: list[float] = []
    pp_p2p_windows = 0
    for window in windows:
        intrinsic_us = _overlap_us(intrinsic_gpu, window)
        target_us = _overlap_us(target_gpu, window)
        if intrinsic_us <= 0.0 or target_us <= 0.0:
            raise TraceProfileError(
                f"{path} has a verify window without correlated GPU work"
            )
        if target_us > intrinsic_us + 1e-6:
            raise TraceProfileError(f"{path} target GPU union exceeds intrinsic union")
        pp_p2p_windows += _overlap_us(pp_p2p, window) > 0.0
        intrinsic_samples.append(intrinsic_us / 1000.0)
        target_samples.append(target_us / 1000.0)
    return RankTraceProfile(
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        intrinsic_samples_ms=intrinsic_samples,
        target_samples_ms=target_samples,
        pp_p2p_windows=pp_p2p_windows,
    )


def summarize_trace_dir(
    trace_dir: Path,
    *,
    pp_size: int,
    tp_size: int,
    trim_samples: int = 1,
) -> dict[str, Any]:
    if pp_size <= 0 or tp_size <= 0:
        raise TraceProfileError("pp_size and tp_size must be positive")
    paths = sorted(path for path in trace_dir.rglob("*.trace.json*") if path.is_file())
    if not paths:
        raise TraceProfileError(f"no Kineto traces found under {trace_dir}")
    profiles = [parse_rank_trace(path, trim_samples=trim_samples) for path in paths]

    by_rank: dict[tuple[int, int], RankTraceProfile] = {}
    for profile in profiles:
        rank = (profile.pp_rank, profile.tp_rank)
        if rank in by_rank:
            raise TraceProfileError(f"duplicate (PP, TP) trace rank {rank}")
        by_rank[rank] = profile
    expected = {
        (pp_rank, tp_rank)
        for pp_rank in range(pp_size)
        for tp_rank in range(tp_size)
    }
    actual = set(by_rank)
    if actual != expected:
        raise TraceProfileError(
            f"trace ranks do not match PP{pp_size}xTP{tp_size}: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if pp_size > 1 and any(
        profile.pp_p2p_windows != len(profile.intrinsic_samples_ms)
        for profile in profiles
    ):
        missing = sorted(
            (profile.pp_rank, profile.tp_rank)
            for profile in profiles
            if profile.pp_p2p_windows != len(profile.intrinsic_samples_ms)
        )
        raise TraceProfileError(
            f"Kineto PP Send/Recv metadata is missing from verify windows on ranks {missing}"
        )

    selected = [
        max(
            (by_rank[(pp_rank, tp_rank)] for tp_rank in range(tp_size)),
            key=lambda item: item.intrinsic_median_ms,
        )
        for pp_rank in range(pp_size)
    ]
    return {
        "intrinsic_service_ms": [item.intrinsic_median_ms for item in selected],
        "target_ms": [item.target_median_ms for item in selected],
    }
