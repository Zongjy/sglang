#!/usr/bin/env python3
"""Recommend a contiguous PP layer partition from torch-profiler traces.

The analyzer is intentionally offline: it reads one raw Chrome trace per PP
rank and never starts a server.  It separates target-layer time from fixed
stage work (scheduler preparation, result processing, final norm/head,
speculative draft work), then solves the contiguous min-max partition problem.

When layerwise NVTX ranges are present, their measured costs are used.  The
normal CUDA-graph trace does not expose individual layer boundaries, so the
analyzer falls back to a uniform per-layer cost calibrated from non-last PP
stages.  Fixed last-rank work is still measured directly in both modes.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


TRACE_SUFFIXES = (".trace.json", ".trace.json.gz")
TP_RANK_IN_FILENAME = re.compile(r"-TP-(\d+)(?:\.|-)")
PP_RANK_IN_FILENAME = re.compile(r"-PP-(\d+)(?:\.|-)")
PP_RANK_IN_PROCESS = re.compile(r"(?:scheduler_)?PP(\d+)", re.IGNORECASE)
RUN_KEY_IN_FILENAME = re.compile(
    r"^(?P<run>.+)-TP-\d+(?:-DP-\d+)?-PP-\d+"
    r"(?:-EP-\d+)?\.trace\.json(?:\.gz)?$"
)
LAYER_MODULE = re.compile(
    r"['\"]Module['\"]\s*:\s*['\"]"
    r"[^'\"]*?\.layers\.(?P<layer>\d+)['\"]"
)
PLAIN_LAYER_MODULE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)$")


class AnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class Span:
    ts_us: float
    dur_us: float
    name: str

    @property
    def end_us(self) -> float:
        return self.ts_us + self.dur_us


@dataclass(frozen=True)
class ForwardBlock:
    start_us: float
    end_us: float
    member_count: int

    @property
    def dur_us(self) -> float:
        return self.end_us - self.start_us


@dataclass
class RankTrace:
    rank: int
    tp_rank: int
    path: Path
    step_spans: list[Span]
    gpu_run_batches: list[Span]
    cpu_process_results: list[Span]
    gpu_process_results: list[Span]
    layer_spans: dict[int, list[Span]]


@dataclass
class RankMeasurement:
    rank: int
    trace: str
    current_layers: int
    target_samples: int
    draft_samples: int
    target_ms: float
    draft_ms: float | None
    run_batch_ms: float
    exposed_process_ms: float
    cycle_ms: float | None
    run_covers_target: bool
    target_coverage_by_run: float
    measured_stage_ms: float
    cadence_adjustment_ms: float = 0.0
    fixed_overhead_ms: float = 0.0


@dataclass
class PartitionCandidate:
    partition: tuple[int, ...]
    stage_ms: tuple[float, ...]
    bottleneck_ms: float


def _open_trace(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")


def _median(values: Sequence[float], what: str) -> float:
    if not values:
        raise AnalysisError(f"No samples found for {what}.")
    return float(statistics.median(values))


def _trim_by_time(items: Sequence, count: int) -> list:
    ordered = sorted(items, key=lambda item: item.start_us)
    if count <= 0 or len(ordered) <= 2 * count + 2:
        return list(ordered)
    return ordered[count:-count]


def _parse_rank_from_metadata(path: Path, events: Sequence[dict]) -> int:
    match = PP_RANK_IN_FILENAME.search(path.name)
    if match:
        return int(match.group(1))

    ranks = set()
    for event in events:
        if event.get("ph") != "M" or event.get("name") != "process_name":
            continue
        process_name = str(event.get("args", {}).get("name", ""))
        match = PP_RANK_IN_PROCESS.search(process_name)
        if match:
            ranks.add(int(match.group(1)))
    if len(ranks) != 1:
        raise AnalysisError(
            f"Cannot determine one PP rank from {path}; metadata ranks={sorted(ranks)}."
        )
    return ranks.pop()


def _parse_tp_rank(path: Path) -> int:
    match = TP_RANK_IN_FILENAME.search(path.name)
    return int(match.group(1)) if match else 0


def _layer_id_from_marker(name: str) -> int | None:
    match = LAYER_MODULE.search(name) or PLAIN_LAYER_MODULE.search(name)
    return int(match.group("layer")) if match else None


def load_rank_trace(path: Path) -> RankTrace:
    with _open_trace(path) as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents", [])
    rank = _parse_rank_from_metadata(path, events)

    step_spans: list[Span] = []
    gpu_run_batches: list[Span] = []
    cpu_process_results: list[Span] = []
    gpu_process_results: list[Span] = []
    layer_spans: dict[int, list[Span]] = {}

    for event in events:
        if event.get("ph") != "X":
            continue
        try:
            ts_us = float(event["ts"])
            dur_us = float(event["dur"])
        except (KeyError, TypeError, ValueError):
            continue
        if dur_us <= 0:
            continue

        category = event.get("cat", "")
        name = str(event.get("name", ""))
        span = Span(ts_us=ts_us, dur_us=dur_us, name=name)
        if category == "gpu_user_annotation":
            if name.startswith("step["):
                step_spans.append(span)
            elif name == "scheduler.run_batch":
                gpu_run_batches.append(span)
            elif name == "scheduler.process_batch_result":
                gpu_process_results.append(span)

            layer_id = _layer_id_from_marker(name)
            if layer_id is not None:
                layer_spans.setdefault(layer_id, []).append(span)
        elif category == "user_annotation" and name == "scheduler.process_batch_result":
            cpu_process_results.append(span)

    if not step_spans:
        raise AnalysisError(
            f"{path} has no GPU step annotations. Use raw torch-profiler traces, "
            "not a kernel-only export."
        )
    return RankTrace(
        rank=rank,
        tp_rank=_parse_tp_rank(path),
        path=path,
        step_spans=step_spans,
        gpu_run_batches=gpu_run_batches,
        cpu_process_results=cpu_process_results,
        gpu_process_results=gpu_process_results,
        layer_spans=layer_spans,
    )


def group_overlapping_steps(spans: Sequence[Span]) -> list[ForwardBlock]:
    """Collapse nested outer/inner step annotations into one forward block."""
    blocks: list[ForwardBlock] = []
    current_start: float | None = None
    current_end = 0.0
    members = 0
    for span in sorted(spans, key=lambda item: (item.ts_us, -item.dur_us)):
        if current_start is not None and span.ts_us < current_end:
            current_end = max(current_end, span.end_us)
            members += 1
            continue
        if current_start is not None:
            blocks.append(ForwardBlock(current_start, current_end, members))
        current_start = span.ts_us
        current_end = span.end_us
        members = 1
    if current_start is not None:
        blocks.append(ForwardBlock(current_start, current_end, members))
    return blocks


def _overlap_us(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _merge_intervals(spans: Iterable[Span | ForwardBlock]) -> list[tuple[float, float]]:
    intervals = sorted(
        (
            (getattr(span, "ts_us", getattr(span, "start_us", 0.0)), span.end_us)
            for span in spans
        ),
        key=lambda item: item[0],
    )
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _uncovered_duration_us(span: Span, busy: Sequence[tuple[float, float]]) -> float:
    covered = sum(
        _overlap_us(span.ts_us, span.end_us, start, end) for start, end in busy
    )
    return max(0.0, span.dur_us - covered)


def _target_coverage(targets: Sequence[ForwardBlock], runs: Sequence[Span]) -> float:
    fractions = []
    for target in targets:
        overlap = max(
            (
                _overlap_us(target.start_us, target.end_us, run.ts_us, run.end_us)
                for run in runs
            ),
            default=0.0,
        )
        fractions.append(overlap / target.dur_us)
    return _median(fractions, "target/run_batch overlap") if fractions else 0.0


def measure_ranks(
    traces: Sequence[RankTrace],
    current_partition: Sequence[int],
    target_min_ratio: float,
    trim_samples: int,
) -> tuple[list[RankMeasurement], dict[int, dict[int, float]], list[str]]:
    traces = sorted(traces, key=lambda trace: trace.rank)
    reference_blocks = group_overlapping_steps(traces[0].step_spans)
    reference_blocks = _trim_by_time(reference_blocks, trim_samples)
    reference_ms = _median(
        [block.dur_us / 1000.0 for block in reference_blocks],
        "PP0 target forward",
    )
    reference_members = max(
        1, round(statistics.median(block.member_count for block in reference_blocks))
    )

    measurements: list[RankMeasurement] = []
    layer_medians: dict[int, dict[int, float]] = {}
    warnings: list[str] = []

    for trace, layer_count in zip(traces, current_partition):
        blocks = group_overlapping_steps(trace.step_spans)
        # Target verify currently has nested outer/inner step annotations while
        # the DFlash draft has one.  This remains reliable when PP stages have
        # very different layer counts and a draft is longer than a short target
        # stage.  Older traces with only one annotation fall back to duration,
        # scaled by the current layer allocation.
        if reference_members > 1:
            targets = [
                block for block in blocks if block.member_count >= reference_members
            ]
            drafts = [
                block for block in blocks if block.member_count < reference_members
            ]
            threshold_us = None
        else:
            threshold_us = (
                reference_ms
                * 1000.0
                * layer_count
                / current_partition[0]
                * target_min_ratio
            )
            targets = [block for block in blocks if block.dur_us >= threshold_us]
            drafts = [block for block in blocks if block.dur_us < threshold_us]
        targets = _trim_by_time(targets, trim_samples)
        drafts = _trim_by_time(drafts, trim_samples)
        if len(targets) < 3:
            criterion = (
                f"member_count >= {reference_members}"
                if threshold_us is None
                else f"threshold={threshold_us / 1000.0:.2f} ms"
            )
            raise AnalysisError(
                f"PP{trace.rank} has only {len(targets)} target-like samples; "
                f"criterion={criterion}."
            )

        target_ms = _median(
            [block.dur_us / 1000.0 for block in targets],
            f"PP{trace.rank} target forward",
        )
        draft_ms = (
            _median(
                [block.dur_us / 1000.0 for block in drafts],
                f"PP{trace.rank} draft forward",
            )
            if drafts
            else None
        )
        target_starts = sorted(block.start_us for block in targets)
        cycle_ms = (
            _median(
                [
                    (right - left) / 1000.0
                    for left, right in zip(target_starts, target_starts[1:])
                    if right > left
                ],
                f"PP{trace.rank} target cadence",
            )
            if len(target_starts) >= 2
            else None
        )

        coverage = _target_coverage(targets, trace.gpu_run_batches)
        run_covers_target = coverage >= 0.75
        run_batch_ms = (
            _median(
                [span.dur_us / 1000.0 for span in trace.gpu_run_batches],
                f"PP{trace.rank} scheduler.run_batch",
            )
            if trace.gpu_run_batches
            else 0.0
        )

        gpu_service_spans: list[Span | ForwardBlock]
        if run_covers_target:
            gpu_service_spans = list(trace.gpu_run_batches)
            gpu_service_ms = run_batch_ms
        else:
            gpu_service_spans = [*trace.gpu_run_batches, *targets]
            gpu_service_ms = run_batch_ms + target_ms
        busy_intervals = _merge_intervals(gpu_service_spans)

        if trace.cpu_process_results:
            exposed_process_ms = _median(
                [
                    _uncovered_duration_us(span, busy_intervals) / 1000.0
                    for span in trace.cpu_process_results
                ],
                f"PP{trace.rank} exposed process_batch_result",
            )
        elif trace.gpu_process_results:
            exposed_process_ms = _median(
                [span.dur_us / 1000.0 for span in trace.gpu_process_results],
                f"PP{trace.rank} projected process_batch_result",
            )
            warnings.append(
                f"PP{trace.rank}: CPU process_batch_result spans are absent; "
                "using GPU-projected spans, which can underestimate fixed overhead."
            )
        else:
            exposed_process_ms = 0.0
            warnings.append(
                f"PP{trace.rank}: process_batch_result spans are absent; assuming 0 ms."
            )

        measured_stage_ms = gpu_service_ms + exposed_process_ms
        measurements.append(
            RankMeasurement(
                rank=trace.rank,
                trace=str(trace.path),
                current_layers=layer_count,
                target_samples=len(targets),
                draft_samples=len(drafts),
                target_ms=target_ms,
                draft_ms=draft_ms,
                run_batch_ms=run_batch_ms,
                exposed_process_ms=exposed_process_ms,
                cycle_ms=cycle_ms,
                run_covers_target=run_covers_target,
                target_coverage_by_run=coverage,
                measured_stage_ms=measured_stage_ms,
            )
        )

        per_layer: dict[int, float] = {}
        for layer_id, spans in trace.layer_spans.items():
            target_durations = []
            for span in spans:
                if any(
                    _overlap_us(
                        span.ts_us,
                        span.end_us,
                        target.start_us,
                        target.end_us,
                    )
                    >= 0.5 * span.dur_us
                    for target in targets
                ):
                    target_durations.append(span.dur_us / 1000.0)
            if target_durations:
                per_layer[layer_id] = _median(
                    target_durations, f"PP{trace.rank} layer {layer_id}"
                )
        layer_medians[trace.rank] = per_layer

    # Every rank observes the pipeline cadence, but only the slowest rank owns
    # it.  Using cadence as every rank's service time would turn pipeline
    # bubbles into fake work and make all partitions look equally good.  The
    # annotation-accounted slowest rank is the bottleneck candidate; floor
    # only that rank by the cross-rank median cadence to recover exposed
    # control/P2P gaps that sit between named spans.
    cadence_values = [
        measurement.cycle_ms
        for measurement in measurements
        if measurement.cycle_ms is not None
    ]
    if cadence_values:
        pipeline_cadence_ms = float(statistics.median(cadence_values))
        bottleneck = max(measurements, key=lambda item: item.measured_stage_ms)
        if pipeline_cadence_ms > bottleneck.measured_stage_ms:
            adjustment = pipeline_cadence_ms - bottleneck.measured_stage_ms
            bottleneck.cadence_adjustment_ms = adjustment
            bottleneck.measured_stage_ms = pipeline_cadence_ms
            warnings.append(
                f"PP{bottleneck.rank}: added {adjustment:.2f} ms of exposed "
                "control/P2P gap so the inferred bottleneck matches the "
                f"observed {pipeline_cadence_ms:.2f} ms pipeline cadence."
            )

    return measurements, layer_medians, warnings


def default_partition(num_layers: int, pp_size: int) -> tuple[int, ...]:
    base = num_layers // pp_size
    remainder = num_layers % pp_size
    return tuple(
        base + int(rank >= pp_size - remainder) for rank in range(pp_size)
    )


def parse_int_list(value: str, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise AnalysisError(f"Invalid {name}: {value!r}.") from exc
    if not result:
        raise AnalysisError(f"{name} cannot be empty.")
    return result


def parse_float_list(value: str, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise AnalysisError(f"Invalid {name}: {value!r}.") from exc
    if not result:
        raise AnalysisError(f"{name} cannot be empty.")
    return result


def partition_ranges(partition: Sequence[int]) -> list[tuple[int, int]]:
    ranges = []
    start = 0
    for count in partition:
        ranges.append((start, start + count))
        start += count
    return ranges


def build_layer_costs(
    num_layers: int,
    current_partition: Sequence[int],
    measurements: Sequence[RankMeasurement],
    layer_medians: dict[int, dict[int, float]],
) -> tuple[list[float], str, float, list[str]]:
    non_last_rates = [
        measurement.target_ms / measurement.current_layers
        for measurement in measurements[:-1]
        if measurement.current_layers > 0
    ]
    if not non_last_rates:
        non_last_rates = [
            measurement.target_ms / measurement.current_layers
            for measurement in measurements
            if measurement.current_layers > 0
        ]
    unit_ms = _median(non_last_rates, "per-layer target cost")
    costs = [unit_ms] * num_layers
    warnings: list[str] = []
    marked_layers = 0

    for measurement, (start, end) in zip(
        measurements, partition_ranges(current_partition)
    ):
        local_markers = {
            layer_id: cost
            for layer_id, cost in layer_medians.get(measurement.rank, {}).items()
            if start <= layer_id < end
        }
        if not local_markers:
            continue
        raw = [local_markers.get(layer_id, unit_ms) for layer_id in range(start, end)]
        raw_sum = sum(raw)
        if raw_sum <= 0:
            continue
        scale = measurement.target_ms / raw_sum
        for offset, layer_id in enumerate(range(start, end)):
            costs[layer_id] = raw[offset] * scale
        marked_layers += len(local_markers)

    if marked_layers:
        model = f"layerwise NVTX ({marked_layers}/{num_layers} layers marked)"
        if marked_layers < num_layers:
            warnings.append(
                f"Only {marked_layers}/{num_layers} target layers had layerwise markers; "
                "missing layers use the uniform calibrated cost."
            )
    else:
        model = "uniform calibrated layer cost"
        warnings.append(
            "No layerwise NVTX markers were found. The recommendation assumes target "
            "layers have equal cost; validate the best partition and its +/-1 neighbors."
        )
    return costs, model, unit_ms, warnings


def evaluate_partition(
    partition: Sequence[int], layer_costs: Sequence[float], fixed_ms: Sequence[float]
) -> PartitionCandidate:
    ranges = partition_ranges(partition)
    stage_ms = tuple(
        fixed + sum(layer_costs[start:end])
        for fixed, (start, end) in zip(fixed_ms, ranges)
    )
    return PartitionCandidate(
        partition=tuple(partition),
        stage_ms=stage_ms,
        bottleneck_ms=max(stage_ms),
    )


def optimize_partition(
    layer_costs: Sequence[float],
    fixed_ms: Sequence[float],
    min_layers: int,
    max_layers: Sequence[int],
) -> PartitionCandidate:
    num_layers = len(layer_costs)
    pp_size = len(fixed_ms)
    prefix = [0.0]
    for cost in layer_costs:
        prefix.append(prefix[-1] + cost)

    inf = math.inf
    dp = [[inf] * (num_layers + 1) for _ in range(pp_size + 1)]
    previous = [[-1] * (num_layers + 1) for _ in range(pp_size + 1)]
    dp[0][0] = 0.0

    for stage in range(1, pp_size + 1):
        remaining_stages = pp_size - stage
        min_end = stage * min_layers
        max_end = num_layers - remaining_stages * min_layers
        for end in range(min_end, max_end + 1):
            start_lo = max((stage - 1) * min_layers, end - max_layers[stage - 1])
            start_hi = end - min_layers
            for start in range(start_lo, start_hi + 1):
                if not math.isfinite(dp[stage - 1][start]):
                    continue
                load = fixed_ms[stage - 1] + prefix[end] - prefix[start]
                objective = max(dp[stage - 1][start], load)
                if objective < dp[stage][end]:
                    dp[stage][end] = objective
                    previous[stage][end] = start

    if not math.isfinite(dp[pp_size][num_layers]):
        raise AnalysisError(
            "No valid partition satisfies the min/max layer constraints."
        )

    counts = []
    end = num_layers
    for stage in range(pp_size, 0, -1):
        start = previous[stage][end]
        if start < 0:
            raise AnalysisError("Internal error while reconstructing the partition.")
        counts.append(end - start)
        end = start
    counts.reverse()
    return evaluate_partition(counts, layer_costs, fixed_ms)


def nearby_candidates(
    best: PartitionCandidate,
    layer_costs: Sequence[float],
    fixed_ms: Sequence[float],
    min_layers: int,
    max_layers: Sequence[int],
    limit: int,
) -> list[PartitionCandidate]:
    seen = {best.partition}
    frontier = {best.partition}
    for _ in range(2):
        next_frontier = set()
        for partition in frontier:
            for boundary in range(len(partition) - 1):
                for delta in (-1, 1):
                    candidate = list(partition)
                    candidate[boundary] += delta
                    candidate[boundary + 1] -= delta
                    if any(count < min_layers for count in candidate):
                        continue
                    if any(
                        count > maximum
                        for count, maximum in zip(candidate, max_layers)
                    ):
                        continue
                    item = tuple(candidate)
                    if item not in seen:
                        seen.add(item)
                        next_frontier.add(item)
        frontier = next_frontier
    candidates = [evaluate_partition(item, layer_costs, fixed_ms) for item in seen]
    candidates.sort(key=lambda candidate: candidate.bottleneck_ms)
    return candidates[:limit]


def _is_raw_trace(path: Path) -> bool:
    return path.is_file() and path.name.endswith(TRACE_SUFFIXES)


def _latest_trace_group(directory: Path) -> list[Path]:
    groups: dict[str, list[Path]] = {}
    for path in directory.iterdir():
        if not _is_raw_trace(path):
            continue
        match = RUN_KEY_IN_FILENAME.match(path.name)
        if match:
            groups.setdefault(match.group("run"), []).append(path)
    if not groups:
        raise AnalysisError(f"No per-rank PP traces found in {directory}.")
    complete_groups = {}
    for run, paths in groups.items():
        pp_ranks = sorted(
            {
                int(PP_RANK_IN_FILENAME.search(path.name).group(1))
                for path in paths
            }
        )
        if len(pp_ranks) >= 2 and pp_ranks == list(range(len(pp_ranks))):
            complete_groups[run] = paths
    if not complete_groups:
        raise AnalysisError(
            f"No complete multi-rank PP trace group found in {directory}."
        )
    _, paths = max(
        complete_groups.items(),
        key=lambda item: max(path.stat().st_mtime for path in item[1]),
    )
    return sorted(paths)


def resolve_trace_paths(inputs: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(_latest_trace_group(item))
        elif _is_raw_trace(item):
            paths.append(item)
        else:
            raise AnalysisError(f"Not a raw trace file or trace directory: {item}.")
    if not paths:
        raise AnalysisError("No trace inputs were provided.")
    return paths


def _validate_rank_set(traces: Sequence[RankTrace]) -> list[RankTrace]:
    by_rank: dict[int, list[RankTrace]] = {}
    for trace in traces:
        by_rank.setdefault(trace.rank, []).append(trace)
    expected = list(range(len(by_rank)))
    if sorted(by_rank) != expected:
        raise AnalysisError(
            f"PP ranks must be contiguous from 0; found {sorted(by_rank)}."
        )
    selected = []
    for rank in expected:
        rank_traces = by_rank[rank]
        by_tp_rank: dict[int, RankTrace] = {}
        for trace in rank_traces:
            if trace.tp_rank in by_tp_rank:
                raise AnalysisError(
                    f"Multiple traces were supplied for PP{rank}/TP{trace.tp_rank}: "
                    f"{by_tp_rank[trace.tp_rank].path} and {trace.path}."
                )
            by_tp_rank[trace.tp_rank] = trace
        # TP ranks execute the same PP-local layer range but finish at the pace
        # of their slowest collective participant.  Prefer the lane with the
        # largest target/run_batch span; deterministic TP-rank ordering breaks
        # ties in homogeneous traces.
        def service_score(item: RankTrace) -> tuple[float, float, float]:
            blocks = group_overlapping_steps(item.step_spans)
            max_members = max(block.member_count for block in blocks)
            targets = [block.dur_us for block in blocks if block.member_count == max_members]
            target_us = statistics.median(targets) if targets else 0.0
            run_us = (
                statistics.median(span.dur_us for span in item.gpu_run_batches)
                if item.gpu_run_batches
                else 0.0
            )
            return max(target_us, run_us), target_us, run_us

        selected.append(
            max(by_tp_rank.values(), key=lambda item: (service_score(item), -item.tp_rank))
        )
    return selected


def _format_partition(partition: Sequence[int]) -> str:
    return ",".join(str(count) for count in partition)


def _print_report(
    measurements: Sequence[RankMeasurement],
    current: PartitionCandidate,
    recommended: PartitionCandidate,
    candidates: Sequence[PartitionCandidate],
    cost_model: str,
    unit_ms: float,
    warnings: Sequence[str],
) -> None:
    print("\nTrace measurements")
    print(
        "rank  layers  target_ms  run_batch_ms  process_ms  gap_ms  fixed_ms  "
        "stage_ms  cadence_ms"
    )
    for measurement in measurements:
        cadence = (
            f"{measurement.cycle_ms:10.2f}"
            if measurement.cycle_ms is not None
            else "         -"
        )
        print(
            f"PP{measurement.rank:<2}  {measurement.current_layers:>6}  "
            f"{measurement.target_ms:>9.2f}  {measurement.run_batch_ms:>12.2f}  "
            f"{measurement.exposed_process_ms:>10.2f}  "
            f"{measurement.cadence_adjustment_ms:>6.2f}  "
            f"{measurement.fixed_overhead_ms:>8.2f}  "
            f"{measurement.measured_stage_ms:>8.2f}  {cadence}"
        )

    print(f"\nLayer cost model: {cost_model} (baseline {unit_ms:.3f} ms/layer)")
    print("\nPartition prediction")
    print("partition          stage_ms                          bottleneck_ms")
    for label, candidate in (("current", current), ("recommended", recommended)):
        stage_text = ", ".join(f"{value:.2f}" for value in candidate.stage_ms)
        print(
            f"{label:<11} {_format_partition(candidate.partition):<12} "
            f"[{stage_text}]  {candidate.bottleneck_ms:.2f}"
        )
    improvement = (
        (current.bottleneck_ms - recommended.bottleneck_ms)
        / current.bottleneck_ms
        * 100.0
    )
    print(f"Predicted bottleneck reduction: {improvement:.1f}%")

    print("\nCandidates to validate")
    for candidate in candidates:
        stage_text = ", ".join(f"{value:.2f}" for value in candidate.stage_ms)
        print(
            f"  {_format_partition(candidate.partition):<12} "
            f"max={candidate.bottleneck_ms:.2f} ms  stages=[{stage_text}]"
        )

    value = _format_partition(recommended.partition)
    print(f"\nexport SGLANG_PP_LAYER_PARTITION={value}")
    if warnings:
        print("\nWarnings")
        for warning in warnings:
            print(f"  - {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "traces",
        nargs="+",
        type=Path,
        help="one raw trace per PP rank, or a directory containing the traces",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        required=True,
        help="number of target transformer layers",
    )
    parser.add_argument(
        "--current-partition",
        help="comma-separated layer counts used by the traces; default is SGLang's even split",
    )
    parser.add_argument(
        "--fixed-overhead-ms",
        help="override inferred fixed stage overheads, one comma-separated value per PP rank",
    )
    parser.add_argument(
        "--min-layers-per-stage",
        type=int,
        default=1,
        help="minimum target layers on every PP stage (default: 1)",
    )
    parser.add_argument(
        "--max-layers-per-stage",
        help="optional comma-separated per-rank limits used as memory constraints",
    )
    parser.add_argument(
        "--target-min-ratio",
        type=float,
        default=0.60,
        help=(
            "forward blocks below this fraction of PP0 target time are draft "
            "blocks (default: 0.60)"
        ),
    )
    parser.add_argument(
        "--trim-samples",
        type=int,
        default=2,
        help="discard this many target/draft samples at each trace edge (default: 2)",
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=3,
        help="number of nearby partitions to print (default: 3)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="also write the measurements and recommendation as JSON",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.num_layers <= 0:
        raise AnalysisError("--num-layers must be positive.")
    if args.min_layers_per_stage <= 0:
        raise AnalysisError("--min-layers-per-stage must be positive.")
    if not 0.0 < args.target_min_ratio < 1.0:
        raise AnalysisError("--target-min-ratio must be between 0 and 1.")

    paths = resolve_trace_paths(args.traces)
    traces = _validate_rank_set([load_rank_trace(path) for path in paths])
    pp_size = len(traces)
    if pp_size < 2:
        raise AnalysisError("Automatic PP partitioning requires at least two PP ranks.")

    current_partition = (
        parse_int_list(args.current_partition, "--current-partition")
        if args.current_partition
        else default_partition(args.num_layers, pp_size)
    )
    if len(current_partition) != pp_size:
        raise AnalysisError(
            f"Current partition has {len(current_partition)} entries but traces have "
            f"{pp_size} PP ranks."
        )
    if any(count <= 0 for count in current_partition):
        raise AnalysisError("Every current partition entry must be positive.")
    if sum(current_partition) != args.num_layers:
        raise AnalysisError(
            f"Current partition sums to {sum(current_partition)}, expected "
            f"{args.num_layers}."
        )

    max_layers = (
        parse_int_list(args.max_layers_per_stage, "--max-layers-per-stage")
        if args.max_layers_per_stage
        else (args.num_layers,) * pp_size
    )
    if len(max_layers) != pp_size or any(value <= 0 for value in max_layers):
        raise AnalysisError(
            "--max-layers-per-stage must contain one positive value per PP rank."
        )

    measurements, layer_medians, warnings = measure_ranks(
        traces=traces,
        current_partition=current_partition,
        target_min_ratio=args.target_min_ratio,
        trim_samples=args.trim_samples,
    )
    layer_costs, cost_model, unit_ms, cost_warnings = build_layer_costs(
        num_layers=args.num_layers,
        current_partition=current_partition,
        measurements=measurements,
        layer_medians=layer_medians,
    )
    warnings.extend(cost_warnings)

    if args.fixed_overhead_ms:
        fixed_ms = parse_float_list(args.fixed_overhead_ms, "--fixed-overhead-ms")
        if len(fixed_ms) != pp_size or any(value < 0 for value in fixed_ms):
            raise AnalysisError(
                "--fixed-overhead-ms must contain one non-negative value per PP rank."
            )
    else:
        fixed_values = []
        for measurement, (start, end) in zip(
            measurements, partition_ranges(current_partition)
        ):
            inferred = measurement.measured_stage_ms - sum(layer_costs[start:end])
            if inferred < 0:
                warnings.append(
                    f"PP{measurement.rank}: inferred fixed overhead was {inferred:.2f} ms; "
                    "clamped to 0 ms."
                )
            fixed_values.append(max(0.0, inferred))
        fixed_ms = tuple(fixed_values)

    for measurement, fixed in zip(measurements, fixed_ms):
        measurement.fixed_overhead_ms = fixed

    current = evaluate_partition(current_partition, layer_costs, fixed_ms)
    recommended = optimize_partition(
        layer_costs=layer_costs,
        fixed_ms=fixed_ms,
        min_layers=args.min_layers_per_stage,
        max_layers=max_layers,
    )
    candidates = nearby_candidates(
        best=recommended,
        layer_costs=layer_costs,
        fixed_ms=fixed_ms,
        min_layers=args.min_layers_per_stage,
        max_layers=max_layers,
        limit=max(1, args.top_candidates),
    )
    _print_report(
        measurements=measurements,
        current=current,
        recommended=recommended,
        candidates=candidates,
        cost_model=cost_model,
        unit_ms=unit_ms,
        warnings=warnings,
    )

    result = {
        "trace_files": [str(trace.path) for trace in traces],
        "num_layers": args.num_layers,
        "pp_size": pp_size,
        "cost_model": cost_model,
        "baseline_layer_ms": unit_ms,
        "measurements": [asdict(measurement) for measurement in measurements],
        "current": asdict(current),
        "recommended": asdict(recommended),
        "candidates": [asdict(candidate) for candidate in candidates],
        "warnings": warnings,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {args.json_output}")
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except AnalysisError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
