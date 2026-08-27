#!/usr/bin/env python3
"""Extract DFlash draft-forward GPU latency from Kineto Chrome traces.

The parser intentionally uses only the Python standard library.  A draft-forward
sample is the GPU work launched by CPU events inside one
``sglang.dflash.draft_model_forward`` record-function range.  Kineto's
``External id`` and CUDA ``correlation`` fields are used instead of clipping GPU
events to the CPU range, because asynchronous kernels commonly finish after the
range has closed.

``critical_path_ms`` is the wall-clock envelope from the first associated GPU
event to the last one.  ``gpu_busy_ms`` and ``nccl_ms`` are interval unions, so
overlap between streams is counted once.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DRAFT_FORWARD_MARKER = "sglang.dflash.draft_model_forward"
DEFAULT_MARKER = DRAFT_FORWARD_MARKER

_EXTERNAL_ID_KEYS = ("external id", "external_id")
_CORRELATION_ID_KEYS = ("correlation", "correlation id", "correlation_id")
_TRACE_SUFFIXES = (".trace.json", ".trace.json.gz")
_RANK_PATTERNS = (
    re.compile(r"(?:^|[-_.])TP[-_]?(\d+)(?=[-_.]|$)", re.IGNORECASE),
    re.compile(
        r"(?:^|[-_.])(?:global[-_])?rank[-_]?(\d+)(?=[-_.]|$)",
        re.IGNORECASE,
    ),
)


class TraceParserError(RuntimeError):
    """Raised when a trace cannot produce trustworthy draft-forward samples."""


# Kept as the runner-facing spelling.
TraceParseError = TraceParserError


@dataclass(frozen=True)
class Interval:
    start_us: float
    end_us: float

    @property
    def duration_us(self) -> float:
        return self.end_us - self.start_us


@dataclass(frozen=True)
class DraftForwardSample:
    rank: int
    sample_index: int
    source: str
    marker_start_us: float
    marker_end_us: float
    gpu_start_us: float
    gpu_end_us: float
    critical_path_ms: float
    gpu_busy_ms: float
    nccl_ms: float
    gpu_event_count: int
    nccl_event_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "source": self.source,
            "marker_start_us": self.marker_start_us,
            "marker_end_us": self.marker_end_us,
            "gpu_start_us": self.gpu_start_us,
            "gpu_end_us": self.gpu_end_us,
            "critical_path_ms": self.critical_path_ms,
            "gpu_busy_ms": self.gpu_busy_ms,
            "nccl_ms": self.nccl_ms,
            "gpu_event_count": self.gpu_event_count,
            "nccl_event_count": self.nccl_event_count,
        }


@dataclass(frozen=True)
class RankTrace:
    rank: int
    source: str
    marker_name: str
    samples: tuple[DraftForwardSample, ...]

    def summary(self) -> dict[str, Any]:
        result = summarize_samples(self.samples)
        result.update(
            {
                "rank": self.rank,
                "trace_files": [self.source],
                "samples": [sample.to_dict() for sample in self.samples],
            }
        )
        return result


@dataclass
class _MarkerContext:
    interval: Interval
    pid: Any
    tid: Any
    external_ids: set[str]
    correlation_ids: set[str]
    collective_external_ids: set[str]
    collective_correlation_ids: set[str]
    id_links: set[tuple[str | None, str | None]]


def _read_trace(path: Path) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceParserError(f"cannot read Kineto trace {path}: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(
        payload.get("traceEvents"), list
    ):
        raise TraceParserError(f"{path} is not a Kineto Chrome trace")
    events = [event for event in payload["traceEvents"] if isinstance(event, dict)]
    return payload, events


def _event_start_us(event: Mapping[str, Any]) -> float | None:
    if event.get("ph") != "X":
        return None
    try:
        return float(event["ts"])
    except (KeyError, TypeError, ValueError):
        return None


def _as_interval(event: Mapping[str, Any]) -> Interval | None:
    start = _event_start_us(event)
    if start is None:
        return None
    try:
        duration = float(event.get("dur", 0.0))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0.0:
        return None
    return Interval(start, start + duration)


def _normalized_args(args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    return {str(key).strip().casefold(): value for key, value in args.items()}


def _normalize_id(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    normalized = str(value).strip()
    return normalized or None


def _arg_id(args: Any, names: Sequence[str]) -> str | None:
    normalized_args = _normalized_args(args)
    for name in names:
        if name in normalized_args:
            return _normalize_id(normalized_args[name])
    return None


def _is_gpu_event(event: Mapping[str, Any]) -> bool:
    category = str(event.get("cat", "")).strip().casefold()
    return category in {"kernel", "gpu_memcpy", "gpu_memset"} or (
        "kernel" in category and "runtime" not in category
    )


def _is_gpu_annotation(event: Mapping[str, Any]) -> bool:
    category = str(event.get("cat", "")).strip().casefold()
    return category in {"gpu_user_annotation", "gpu_annotation"}


def _is_nccl_or_collective(event: Mapping[str, Any]) -> bool:
    name = str(event.get("name", "")).casefold()
    category = str(event.get("cat", "")).casefold()
    if "nccl" in name or "rccl" in name or "allreduce" in re.sub(r"[\s_-]", "", name):
        return True
    if "nccl" in category or "rccl" in category or "collective" in category:
        return True

    args = _normalized_args(event.get("args"))
    collective = args.get("collective name")
    if collective is None:
        collective = args.get("collective_name")
    if collective is None:
        return False
    collective_name = str(collective).strip().casefold()
    return collective_name not in {"", "none", "wait"}


def _lane_matches(event: Mapping[str, Any], marker: _MarkerContext) -> bool:
    event_pid = event.get("pid")
    event_tid = event.get("tid")
    if marker.pid is not None and event_pid is not None and marker.pid != event_pid:
        return False
    return not (
        marker.tid is not None and event_tid is not None and marker.tid != event_tid
    )


def _containing_marker(
    event: Mapping[str, Any], markers: Sequence[_MarkerContext]
) -> int | None:
    timestamp = _event_start_us(event)
    if timestamp is None:
        return None
    candidates = [
        index
        for index, marker in enumerate(markers)
        if marker.interval.start_us <= timestamp <= marker.interval.end_us
        and _lane_matches(event, marker)
    ]
    if not candidates:
        return None
    # Nested annotations are possible.  Associate a launch with the narrowest
    # matching draft marker rather than counting its GPU work twice.
    return min(candidates, key=lambda index: markers[index].interval.duration_us)


def _merge(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = sorted(
        intervals, key=lambda interval: (interval.start_us, interval.end_us)
    )
    merged: list[Interval] = []
    for interval in ordered:
        if not merged or interval.start_us > merged[-1].end_us:
            merged.append(interval)
        else:
            merged[-1] = Interval(
                merged[-1].start_us, max(merged[-1].end_us, interval.end_us)
            )
    return merged


def _union_duration_us(intervals: Iterable[Interval]) -> float:
    return sum(interval.duration_us for interval in _merge(intervals))


def _coerce_rank(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return None
    return rank if rank >= 0 else None


def _infer_rank(path: Path, payload: Mapping[str, Any]) -> int:
    for pattern in _RANK_PATTERNS:
        # Profile prefixes may describe the world size (for example
        # ``dflash-tp2-...-TP-0.trace.json.gz``). The profiler-owned rank is the
        # final TP/rank component, so prefer the last match.
        matches = list(pattern.finditer(path.name))
        if matches:
            return int(matches[-1].group(1))

    distributed_info = payload.get("distributedInfo")
    if isinstance(distributed_info, dict):
        rank = _coerce_rank(distributed_info.get("rank"))
        if rank is not None:
            return rank
    raise TraceParserError(
        f"cannot infer rank from {path.name!r}; pass rank= explicitly or include "
        "TP/rank in the filename"
    )


def _build_samples(
    events: Sequence[dict[str, Any]],
    *,
    rank: int,
    source: str,
    marker_name: str,
) -> tuple[DraftForwardSample, ...]:
    marker_events = sorted(
        (
            (event, interval)
            for event in events
            if event.get("name") == marker_name
            if not _is_gpu_event(event) and not _is_gpu_annotation(event)
            if (interval := _as_interval(event)) is not None
        ),
        key=lambda item: (item[1].start_us, item[1].end_us),
    )
    if not marker_events:
        raise TraceParserError(f"{source} has no {marker_name} marker")

    markers = [
        _MarkerContext(
            interval=interval,
            pid=event.get("pid"),
            tid=event.get("tid"),
            external_ids=set(),
            correlation_ids=set(),
            collective_external_ids=set(),
            collective_correlation_ids=set(),
            id_links=set(),
        )
        for event, interval in marker_events
    ]

    # Gather identifiers from every CPU/operator/runtime event launched inside
    # each marker.  GPU annotations are excluded because their timestamps live
    # on device lanes and may overlap a later CPU marker by coincidence.
    for event in events:
        if _is_gpu_event(event) or _is_gpu_annotation(event):
            continue
        marker_index = _containing_marker(event, markers)
        if marker_index is None:
            continue
        marker = markers[marker_index]
        external_id = _arg_id(event.get("args"), _EXTERNAL_ID_KEYS)
        correlation_id = _arg_id(event.get("args"), _CORRELATION_ID_KEYS)
        if external_id is not None:
            marker.external_ids.add(external_id)
        if correlation_id is not None:
            marker.correlation_ids.add(correlation_id)
        if external_id is not None or correlation_id is not None:
            marker.id_links.add((external_id, correlation_id))
        if _is_nccl_or_collective(event):
            if external_id is not None:
                marker.collective_external_ids.add(external_id)
            if correlation_id is not None:
                marker.collective_correlation_ids.add(correlation_id)

    # A record_param_comms event often supplies only the collective External id,
    # while cudaLaunchKernel supplies the External-id-to-correlation bridge.  Do
    # this after scanning all CPU events because Chrome trace event order is not
    # part of the Kineto contract.
    for marker in markers:
        changed = True
        while changed:
            changed = False
            for external_id, correlation_id in marker.id_links:
                if not (
                    external_id in marker.collective_external_ids
                    or correlation_id in marker.collective_correlation_ids
                ):
                    continue
                if (
                    external_id is not None
                    and external_id not in marker.collective_external_ids
                ):
                    marker.collective_external_ids.add(external_id)
                    changed = True
                if (
                    correlation_id is not None
                    and correlation_id not in marker.collective_correlation_ids
                ):
                    marker.collective_correlation_ids.add(correlation_id)
                    changed = True

    external_owners: dict[str, set[int]] = {}
    correlation_owners: dict[str, set[int]] = {}
    for marker_index, marker in enumerate(markers):
        for external_id in marker.external_ids:
            external_owners.setdefault(external_id, set()).add(marker_index)
        for correlation_id in marker.correlation_ids:
            correlation_owners.setdefault(correlation_id, set()).add(marker_index)

    gpu_events_by_marker: list[list[tuple[dict[str, Any], Interval]]] = [
        [] for _ in markers
    ]
    for event in events:
        if not _is_gpu_event(event):
            continue
        interval = _as_interval(event)
        if interval is None:
            continue
        external_id = _arg_id(event.get("args"), _EXTERNAL_ID_KEYS)
        correlation_id = _arg_id(event.get("args"), _CORRELATION_ID_KEYS)
        owners: set[int] = set()
        if external_id is not None:
            owners.update(external_owners.get(external_id, ()))
        if correlation_id is not None:
            owners.update(correlation_owners.get(correlation_id, ()))
        if len(owners) > 1:
            raise TraceParserError(
                f"{source} GPU event {event.get('name', '<unnamed>')!r} "
                f"correlates to multiple {marker_name} markers"
            )
        if owners:
            gpu_events_by_marker[next(iter(owners))].append((event, interval))

    samples: list[DraftForwardSample] = []
    for sample_index, (marker, gpu_events) in enumerate(
        zip(markers, gpu_events_by_marker)
    ):
        if not gpu_events:
            raise TraceParserError(
                f"{source} cannot correlate {marker_name} marker {sample_index} "
                "to any GPU event"
            )
        gpu_intervals = [interval for _, interval in gpu_events]
        nccl_intervals: list[Interval] = []
        for event, interval in gpu_events:
            external_id = _arg_id(event.get("args"), _EXTERNAL_ID_KEYS)
            correlation_id = _arg_id(event.get("args"), _CORRELATION_ID_KEYS)
            if (
                _is_nccl_or_collective(event)
                or external_id in marker.collective_external_ids
                or correlation_id in marker.collective_correlation_ids
            ):
                nccl_intervals.append(interval)

        gpu_start_us = min(interval.start_us for interval in gpu_intervals)
        gpu_end_us = max(interval.end_us for interval in gpu_intervals)
        samples.append(
            DraftForwardSample(
                rank=rank,
                sample_index=sample_index,
                source=source,
                marker_start_us=marker.interval.start_us,
                marker_end_us=marker.interval.end_us,
                gpu_start_us=gpu_start_us,
                gpu_end_us=gpu_end_us,
                critical_path_ms=(gpu_end_us - gpu_start_us) / 1000.0,
                gpu_busy_ms=_union_duration_us(gpu_intervals) / 1000.0,
                nccl_ms=_union_duration_us(nccl_intervals) / 1000.0,
                gpu_event_count=len(gpu_events),
                nccl_event_count=len(nccl_intervals),
            )
        )
    return tuple(samples)


def parse_trace(
    path: str | Path,
    *,
    rank: int | None = None,
    marker_name: str = DRAFT_FORWARD_MARKER,
) -> RankTrace:
    """Parse every draft-forward marker in one Kineto trace.

    ``rank`` takes precedence over filename and ``distributedInfo`` inference.
    GPU timestamps are not clipped to the CPU marker.
    """

    trace_path = Path(path)
    payload, events = _read_trace(trace_path)
    resolved_rank = _coerce_rank(rank) if rank is not None else None
    if rank is not None and resolved_rank is None:
        raise TraceParserError(f"rank must be a non-negative integer, got {rank!r}")
    if resolved_rank is None:
        resolved_rank = _infer_rank(trace_path, payload)
    samples = _build_samples(
        events,
        rank=resolved_rank,
        source=str(trace_path),
        marker_name=marker_name,
    )
    return RankTrace(
        rank=resolved_rank,
        source=str(trace_path),
        marker_name=marker_name,
        samples=samples,
    )


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""

    if not values:
        raise TraceParserError("cannot compute a percentile of an empty sample set")
    if not 0.0 <= quantile <= 1.0:
        raise TraceParserError(f"quantile must be in [0, 1], got {quantile}")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise TraceParserError("cannot summarize an empty sample set")
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
    }


def summarize_samples(samples: Sequence[DraftForwardSample]) -> dict[str, Any]:
    """Summarize a non-empty set of draft-forward samples."""

    if not samples:
        raise TraceParserError("cannot summarize an empty sample set")
    return {
        "sample_count": len(samples),
        "critical_path_ms": _metric_summary(
            [sample.critical_path_ms for sample in samples]
        ),
        "gpu_busy_ms": _metric_summary([sample.gpu_busy_ms for sample in samples]),
        "nccl_ms": _metric_summary([sample.nccl_ms for sample in samples]),
    }


def _resolve_trace_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        root = Path(paths)
        if root.is_dir():
            resolved = sorted(
                path
                for path in root.rglob("*.trace.json*")
                if path.is_file() and path.name.endswith(_TRACE_SUFFIXES)
            )
        else:
            resolved = [root]
    else:
        resolved = sorted(Path(path) for path in paths)
    if not resolved:
        raise TraceParserError("no Kineto *.trace.json[.gz] files found")
    return resolved


def summarize_traces(
    paths: str | Path | Iterable[str | Path],
    *,
    marker_name: str = DRAFT_FORWARD_MARKER,
) -> dict[str, Any]:
    """Parse and summarize one or more rank traces as a JSON-ready dictionary.

    Top-level metric summaries pool all rank samples and therefore are useful for
    distribution inspection, but are not a synchronized multi-rank iteration
    latency.  ``iteration_critical_path_ms`` is explicitly estimated as the
    maximum per-rank p50 because independently exported Kineto traces cannot be
    paired reliably by timestamp.
    """

    trace_paths = _resolve_trace_paths(paths)
    rank_traces = [parse_trace(path, marker_name=marker_name) for path in trace_paths]

    samples_by_rank: dict[int, list[DraftForwardSample]] = {}
    sources_by_rank: dict[int, list[str]] = {}
    for rank_trace in rank_traces:
        samples_by_rank.setdefault(rank_trace.rank, []).extend(rank_trace.samples)
        sources_by_rank.setdefault(rank_trace.rank, []).append(rank_trace.source)

    per_rank: dict[str, Any] = {}
    all_samples: list[DraftForwardSample] = []
    for rank in sorted(samples_by_rank):
        samples = samples_by_rank[rank]
        all_samples.extend(samples)
        rank_summary = summarize_samples(samples)
        rank_summary.update(
            {
                "rank": rank,
                "trace_files": sources_by_rank[rank],
                "samples": [sample.to_dict() for sample in samples],
            }
        )
        per_rank[str(rank)] = rank_summary

    result = summarize_samples(all_samples)
    slowest_rank = max(
        samples_by_rank,
        key=lambda rank: per_rank[str(rank)]["critical_path_ms"]["p50"],
    )
    result.update(
        {
            "marker_name": marker_name,
            "trace_count": len(rank_traces),
            "rank_count": len(samples_by_rank),
            "iteration_critical_path_ms": per_rank[str(slowest_rank)][
                "critical_path_ms"
            ]["p50"],
            "iteration_critical_path_aggregation": "max_rank_p50",
            "iteration_critical_path_rank": slowest_rank,
            "per_rank": per_rank,
        }
    )
    return result
