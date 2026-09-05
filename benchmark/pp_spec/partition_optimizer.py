#!/usr/bin/env python3
"""Analytical selectors for PP layer partitions and D-Cut policies.

The candidate family is intentionally small::

    (l, ..., l, L - (P - 1) * l)

Every candidate is evaluated at one profiled execution bucket. The selected
partition is the candidate with the lowest predicted bottleneck stage time.
``optimize_joint`` extends this search to a measured D-Cut cost curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

try:
    from benchmark.pp_spec.model_layout import LayerLayout, uniform_prefix_partition
    from benchmark.pp_spec.stage_model import BucketEstimate, StageCostModel
except ImportError:  # pragma: no cover - flat script execution
    from model_layout import LayerLayout, uniform_prefix_partition
    from stage_model import BucketEstimate, StageCostModel


DEFAULT_K_BEST = 20


class OptimizerError(RuntimeError):
    pass


@dataclass(frozen=True)
class OptimizerCandidate:
    partition: tuple[int, ...]
    stage_ms: tuple[float, ...]
    bottleneck_ms: float
    bottleneck_rank: int
    # 1.0 denotes the dense/full-width verify path.
    dcut_ratio: float = 1.0

    @property
    def cycle_time_ms(self) -> float:
        return self.bottleneck_ms


@dataclass
class OptimizationResult:
    target_bucket: int
    target_bs: float
    stage_comm_ms: tuple[float, ...]
    best: OptimizerCandidate
    candidates: list[OptimizerCandidate]
    baseline_partition: tuple[int, ...]
    current: OptimizerCandidate

    @property
    def selected(self) -> OptimizerCandidate:
        return self.best

    @property
    def cycle_time_ms(self) -> float:
        return self.best.cycle_time_ms

    @property
    def recommendation(self) -> str:
        best_text = ",".join(map(str, self.best.partition))
        if self.best.partition == self.baseline_partition:
            return (
                f"keep baseline {best_text}; predicted objective cycle "
                f"{self.best.cycle_time_ms:.3f} ms"
            )
        return (
            f"switch to {best_text}; predicted objective cycle "
            f"{self.best.cycle_time_ms:.3f} ms vs baseline "
            f"{self.current.cycle_time_ms:.3f} ms"
        )

    @staticmethod
    def _candidate_dict(item: OptimizerCandidate) -> dict[str, Any]:
        return {
            "partition": list(item.partition),
            "stage_ms": list(item.stage_ms),
            "cycle_time_ms": item.cycle_time_ms,
            "bottleneck_rank": item.bottleneck_rank,
            "dcut_ratio": item.dcut_ratio,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_bucket": self.target_bucket,
            "target_bs": self.target_bs,
            "stage_comm_ms": list(self.stage_comm_ms),
            "selected": self._candidate_dict(self.best),
            "candidates": [self._candidate_dict(item) for item in self.candidates],
            "baseline_partition": list(self.baseline_partition),
            "current": self._candidate_dict(self.current),
            "recommendation": self.recommendation,
        }

    def to_report(self) -> str:
        lines = [
            f"recommendation: {self.recommendation}",
            "",
            f"target execution bucket = {self.target_bucket} (bs={self.target_bs:g})",
            f"selected = {','.join(map(str, self.best.partition))}, "
            f"predicted objective cycle = {self.best.cycle_time_ms:.3f} ms",
            f"baseline = {','.join(map(str, self.baseline_partition))}, "
            f"predicted objective cycle = {self.current.cycle_time_ms:.3f} ms",
            "",
            f"{'partition':<22} {'pred_ms':>10} {'bottleneck':>10}  stages (ms)",
        ]
        for item in self.candidates:
            stages = " | ".join(f"{value:.2f}" for value in item.stage_ms)
            lines.append(
                f"{','.join(map(str, item.partition)):<22} "
                f"{item.cycle_time_ms:>10.3f} {item.bottleneck_rank:>10}  {stages}"
            )
        return "\n".join(lines)


def _candidate_from_model(
    model: StageCostModel,
    estimate: BucketEstimate,
    partition: Sequence[int],
    stage_comm_ms: tuple[float, ...],
    layout: LayerLayout | None,
    dcut_ratio: float = 1.0,
    dcut_stage_scale: Sequence[float] | None = None,
) -> OptimizerCandidate:
    counts = tuple(int(value) for value in partition)
    stage_ms = model.predict_stages(
        counts,
        estimate=estimate,
        stage_comm_ms=stage_comm_ms,
        layout=layout,
    )
    if dcut_stage_scale is not None:
        scales = tuple(float(value) for value in dcut_stage_scale)
        if len(scales) != model.pp_size or any(value <= 0.0 for value in scales):
            raise OptimizerError("dcut_stage_scale must contain positive PP values")
        # D-Cut removes layer work. Stage fixed costs and communication floors
        # are retained, avoiding an optimistic estimate for shallow stages.
        ranges: list[tuple[int, int]] = []
        start = 0
        for count in counts:
            ranges.append((start, start + count))
            start += count
        scaled: list[float] = []
        for rank, (begin, end) in enumerate(ranges):
            layer_ms = model._layer_cost(estimate, begin, end, layout or model.layout)
            scaled.append(float(stage_ms[rank]) + (scales[rank] - 1.0) * layer_ms)
        stage_ms = tuple(scaled)
    stage_ms = tuple(float(value) for value in stage_ms)
    bottleneck_rank = max(range(len(stage_ms)), key=stage_ms.__getitem__)
    return OptimizerCandidate(
        partition=counts,
        stage_ms=stage_ms,
        bottleneck_ms=stage_ms[bottleneck_rank],
        bottleneck_rank=bottleneck_rank,
        dcut_ratio=float(dcut_ratio),
    )


def _family_partitions(
    num_layers: int,
    pp_size: int,
    min_layers: int,
    max_layers: Sequence[int] | None,
    prefix_l_range: tuple[int, int] | None,
) -> list[tuple[int, ...]]:
    if pp_size <= 0 or num_layers <= 0:
        raise OptimizerError("num_layers and pp_size must be positive")
    min_count = int(min_layers)
    if min_count <= 0:
        raise OptimizerError("min_layers must be positive")
    limits = (
        (num_layers,) * pp_size
        if max_layers is None
        else tuple(int(value) for value in max_layers)
    )
    if len(limits) != pp_size:
        raise OptimizerError("max_layers must have one entry per PP rank")
    if pp_size == 1:
        return [(num_layers,)] if min_count <= num_layers <= limits[0] else []

    lower = min_count
    upper = (num_layers - 1) // (pp_size - 1)
    if prefix_l_range is not None:
        range_lower, range_upper = map(int, prefix_l_range)
        lower = max(lower, range_lower)
        upper = min(upper, range_upper)

    partitions: list[tuple[int, ...]] = []
    for prefix_layers in range(lower, upper + 1):
        partition = uniform_prefix_partition(num_layers, pp_size, prefix_layers)
        if all(
            min_count <= count <= limits[rank]
            for rank, count in enumerate(partition)
        ):
            partitions.append(partition)
    return partitions


def _all_partitions(
    num_layers: int,
    pp_size: int,
    min_layers: int,
    max_layers: Sequence[int] | None,
) -> list[tuple[int, ...]]:
    """Enumerate all valid compositions, used for non-uniform PP tuning."""
    limits = (
        (num_layers,) * pp_size
        if max_layers is None
        else tuple(int(value) for value in max_layers)
    )
    if len(limits) != pp_size:
        raise OptimizerError("max_layers must have one entry per PP rank")
    if min_layers <= 0:
        raise OptimizerError("min_layers must be positive")
    result: list[tuple[int, ...]] = []

    def visit(rank: int, remaining: int, prefix: tuple[int, ...]) -> None:
        if rank == pp_size - 1:
            if min_layers <= remaining <= limits[rank]:
                result.append(prefix + (remaining,))
            return
        max_count = min(limits[rank], remaining - min_layers * (pp_size - rank - 1))
        for count in range(min_layers, max_count + 1):
            visit(rank + 1, remaining - count, prefix + (count,))

    visit(0, num_layers, ())
    return result


def _resolve_stage_comm(
    stage_comm_ms: Sequence[float] | None, pp_size: int
) -> tuple[float, ...]:
    if stage_comm_ms is None:
        return (0.0,) * pp_size
    resolved = tuple(float(value) for value in stage_comm_ms)
    if len(resolved) == pp_size - 1:
        resolved = (0.0, *resolved)
    if len(resolved) != pp_size or any(value < 0.0 for value in resolved):
        raise OptimizerError(
            "stage_comm_ms must contain PP or PP-1 non-negative values"
        )
    return resolved


def optimize(
    model: StageCostModel,
    estimate: BucketEstimate | None = None,
    min_layers: int = 1,
    max_layers: Sequence[int] | None = None,
    k_best: int = DEFAULT_K_BEST,
    *,
    target_bs: float | int | None = None,
    layout: LayerLayout | None = None,
    prefix_l_range: tuple[int, int] | None = None,
    stage_comm_ms: Sequence[float] | None = None,
    all_boundaries: bool = False,
) -> OptimizationResult:
    """Rank prefix-family candidates by predicted bottleneck stage time."""
    if k_best <= 0:
        raise OptimizerError("k_best must be positive")
    if target_bs is not None:
        estimate = model.estimate_for_bs(target_bs)
    elif estimate is None:
        estimate = model.target_bucket()

    active_layout = layout or model.layout
    if active_layout is not None and active_layout.num_layers != model.num_layers:
        raise OptimizerError("layout and stage model have different layer counts")
    partitions = (
        _all_partitions(model.num_layers, model.pp_size, min_layers, max_layers)
        if all_boundaries
        else _family_partitions(
            model.num_layers,
            model.pp_size,
            min_layers,
            max_layers,
            prefix_l_range,
        )
    )
    if not partitions:
        raise OptimizerError(
            "no valid prefix-uniform partition satisfies the layer limits"
        )
    resolved_stage_comm = _resolve_stage_comm(stage_comm_ms, model.pp_size)
    candidates = sorted(
        (
            _candidate_from_model(
                model,
                estimate,
                partition,
                resolved_stage_comm,
                active_layout,
            )
            for partition in partitions
        ),
        key=lambda item: (item.cycle_time_ms, item.partition),
    )
    current = _candidate_from_model(
        model,
        estimate,
        model.baseline_partition,
        resolved_stage_comm,
        active_layout,
    )
    return OptimizationResult(
        target_bucket=estimate.bucket,
        target_bs=float(estimate.bucket),
        stage_comm_ms=resolved_stage_comm,
        best=candidates[0],
        candidates=candidates[:k_best],
        baseline_partition=model.baseline_partition,
        current=current,
    )


@dataclass
class JointOptimizationResult:
    """Result of the joint PP-boundary and D-Cut search.

    ``dcut_profiles`` accepts either a scalar bottleneck measurement per ratio
    or a sequence of per-stage measurements.  Measurements are normalized to
    the full-width (ratio ``1.0``) entry and applied to layer work only.
    """

    target_bucket: int
    target_bs: float
    best: OptimizerCandidate
    candidates: list[OptimizerCandidate]
    baseline: OptimizerCandidate

    @property
    def selected(self) -> OptimizerCandidate:
        return self.best

    @property
    def cycle_time_ms(self) -> float:
        return self.best.cycle_time_ms

    def to_dict(self) -> dict[str, Any]:
        encode = OptimizationResult._candidate_dict
        return {
            "target_bucket": self.target_bucket,
            "target_bs": self.target_bs,
            "selected": encode(self.best),
            "candidates": [encode(item) for item in self.candidates],
            "baseline": encode(self.baseline),
        }

    def to_report(self) -> str:
        lines = [
            f"selected = {','.join(map(str, self.best.partition))} "
            f"dcut={self.best.dcut_ratio:g}, cycle={self.best.cycle_time_ms:.3f} ms",
            f"baseline = {','.join(map(str, self.baseline.partition))} "
            f"dcut={self.baseline.dcut_ratio:g}, cycle={self.baseline.cycle_time_ms:.3f} ms",
            "",
            f"{'partition':<22} {'dcut':>8} {'cycle_ms':>10} "
            f"{'bottleneck':>10}  stages (ms)",
        ]
        for item in self.candidates:
            stages = " | ".join(f"{value:.2f}" for value in item.stage_ms)
            lines.append(
                f"{','.join(map(str, item.partition)):<22} "
                f"{item.dcut_ratio:>8.3g} {item.cycle_time_ms:>10.3f} "
                f"{item.bottleneck_rank:>10}  {stages}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class RobustPartitionCandidate:
    """One partition evaluated over the complete runtime ratio envelope."""

    partition: tuple[int, ...]
    cycle_by_ratio: dict[float, float]
    worst_cycle_ms: float
    weighted_cycle_ms: float


@dataclass
class RobustPartitionResult:
    target_bucket: int
    target_bs: float
    best: RobustPartitionCandidate
    candidates: list[RobustPartitionCandidate]
    baseline: RobustPartitionCandidate

    @property
    def selected(self) -> RobustPartitionCandidate:
        return self.best

    def to_dict(self) -> dict[str, Any]:
        def encode(item: RobustPartitionCandidate) -> dict[str, Any]:
            return {
                "partition": list(item.partition),
                "cycle_by_ratio": {
                    str(ratio): cycle for ratio, cycle in item.cycle_by_ratio.items()
                },
                "worst_cycle_ms": item.worst_cycle_ms,
                "weighted_cycle_ms": item.weighted_cycle_ms,
            }

        return {
            "target_bucket": self.target_bucket,
            "target_bs": self.target_bs,
            "selected": encode(self.best),
            "candidates": [encode(item) for item in self.candidates],
            "baseline": encode(self.baseline),
            "runtime_dcut": "auto",
        }

    def to_report(self) -> str:
        lines = [
            f"selected = {','.join(map(str, self.best.partition))} "
            f"worst_cycle={self.best.worst_cycle_ms:.3f} ms",
            f"baseline = {','.join(map(str, self.baseline.partition))} "
            f"worst_cycle={self.baseline.worst_cycle_ms:.3f} ms",
            "",
            f"{'partition':<22} {'worst_ms':>10} {'weighted_ms':>12}  ratio cycles",
        ]
        for item in self.candidates:
            cycles = " | ".join(
                f"{ratio:g}:{cycle:.2f}" for ratio, cycle in item.cycle_by_ratio.items()
            )
            lines.append(
                f"{','.join(map(str, item.partition)):<22} "
                f"{item.worst_cycle_ms:>10.3f} {item.weighted_cycle_ms:>12.3f}  {cycles}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class MultiBucketPartitionCandidate:
    partition: tuple[int, ...]
    worst_cycle_ms: float
    weighted_cycle_ms: float
    cycle_by_bucket: dict[int, float]


@dataclass
class MultiBucketPartitionResult:
    target_buckets: tuple[int, ...]
    best: MultiBucketPartitionCandidate
    candidates: list[MultiBucketPartitionCandidate]
    baseline: MultiBucketPartitionCandidate

    @property
    def selected(self) -> MultiBucketPartitionCandidate:
        return self.best

    def to_dict(self) -> dict[str, Any]:
        def encode(item: MultiBucketPartitionCandidate) -> dict[str, Any]:
            return {
                "partition": list(item.partition),
                "cycle_by_bucket": {
                    str(bucket): cycle for bucket, cycle in item.cycle_by_bucket.items()
                },
                "worst_cycle_ms": item.worst_cycle_ms,
                "weighted_cycle_ms": item.weighted_cycle_ms,
            }

        return {
            "target_buckets": list(self.target_buckets),
            "selected": encode(self.best),
            "candidates": [encode(item) for item in self.candidates],
            "baseline": encode(self.baseline),
            "runtime_dcut": "auto",
        }

    def to_report(self) -> str:
        lines = [
            f"selected = {','.join(map(str, self.best.partition))} "
            f"worst_cycle={self.best.worst_cycle_ms:.3f} ms",
            f"baseline = {','.join(map(str, self.baseline.partition))} "
            f"worst_cycle={self.baseline.worst_cycle_ms:.3f} ms",
            "",
            f"{'partition':<22} {'worst_ms':>10} {'weighted_ms':>12}  buckets (ms)",
        ]
        for item in self.candidates:
            cycles = " | ".join(
                f"{bucket}:{cycle:.2f}" for bucket, cycle in item.cycle_by_bucket.items()
            )
            lines.append(
                f"{','.join(map(str, item.partition)):<22} "
                f"{item.worst_cycle_ms:>10.3f} {item.weighted_cycle_ms:>12.3f}  {cycles}"
            )
        return "\n".join(lines)


def _normalise_dcut_profiles(
    dcut_profiles: dict[float, float | Sequence[float]], pp_size: int
) -> list[tuple[float, tuple[float, ...]]]:
    """Convert measured D-Cut costs to per-stage scales.

    A scalar is a bottleneck cost and produces one global scale.  A sequence
    is a stage-cost vector and preserves PP asymmetry.  Requiring ratio 1.0
    makes the normalization explicit and catches accidentally mixed profiles.
    """
    if not dcut_profiles:
        raise OptimizerError("dcut_profiles must not be empty")
    raw = {float(ratio): value for ratio, value in dcut_profiles.items()}
    if 1.0 not in raw:
        raise OptimizerError("dcut_profiles must include the full ratio 1.0")

    def as_vector(value: float | Sequence[float]) -> tuple[float, ...]:
        if isinstance(value, (str, bytes)):
            raise OptimizerError("D-Cut profile values must be numbers or sequences")
        if isinstance(value, Sequence):
            result = tuple(float(item) for item in value)
            if len(result) != pp_size:
                raise OptimizerError(
                    f"D-Cut stage profile must contain {pp_size} values"
                )
        else:
            result = (float(value),) * pp_size
        if any(not math.isfinite(value) or value <= 0.0 for value in result):
            raise OptimizerError("D-Cut profile costs must be positive")
        return result

    full = as_vector(raw[1.0])
    profiles = []
    for ratio, value in raw.items():
        if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
            raise OptimizerError(f"D-Cut ratios must be in (0, 1], got {ratio}")
        measured = as_vector(value)
        profiles.append((ratio, tuple(cost / base for cost, base in zip(measured, full))))
    return sorted(profiles, key=lambda item: (item[0], item[1]))


def optimize_joint(
    model: StageCostModel,
    dcut_profiles: dict[float, float | Sequence[float]],
    estimate: BucketEstimate | None = None,
    min_layers: int = 1,
    max_layers: Sequence[int] | None = None,
    k_best: int = DEFAULT_K_BEST,
    *,
    target_bs: float | int | None = None,
    layout: LayerLayout | None = None,
    prefix_l_range: tuple[int, int] | None = None,
    stage_comm_ms: Sequence[float] | None = None,
    all_boundaries: bool = False,
) -> JointOptimizationResult:
    """Jointly select a PP partition and D-Cut ratio.

    The search is deliberately exhaustive over the small prefix partition
    family and the measured ratio grid.  This makes the policy deterministic,
    easy to inspect, and suitable for re-running after each profiling bucket.
    """
    if k_best <= 0:
        raise OptimizerError("k_best must be positive")
    if target_bs is not None:
        estimate = model.estimate_for_bs(target_bs)
    elif estimate is None:
        estimate = model.target_bucket()
    active_layout = layout or model.layout
    if active_layout is not None and active_layout.num_layers != model.num_layers:
        raise OptimizerError("layout and stage model have different layer counts")
    partitions = (
        _all_partitions(model.num_layers, model.pp_size, min_layers, max_layers)
        if all_boundaries
        else _family_partitions(
            model.num_layers, model.pp_size, min_layers, max_layers, prefix_l_range
        )
    )
    if not partitions:
        raise OptimizerError("no valid prefix-uniform partition satisfies the layer limits")
    resolved_comm = _resolve_stage_comm(stage_comm_ms, model.pp_size)
    candidates: list[OptimizerCandidate] = []
    for ratio, scales in _normalise_dcut_profiles(dcut_profiles, model.pp_size):
        for partition in partitions:
            candidates.append(
                _candidate_from_model(
                    model,
                    estimate,
                    partition,
                    resolved_comm,
                    active_layout,
                    dcut_ratio=ratio,
                    dcut_stage_scale=scales,
                )
            )
    candidates.sort(key=lambda item: (item.cycle_time_ms, item.partition, item.dcut_ratio))
    baseline = _candidate_from_model(
        model,
        estimate,
        model.baseline_partition,
        resolved_comm,
        active_layout,
        dcut_ratio=1.0,
        dcut_stage_scale=(1.0,) * model.pp_size,
    )
    return JointOptimizationResult(
        target_bucket=estimate.bucket,
        target_bs=float(estimate.bucket),
        best=candidates[0],
        candidates=candidates[:k_best],
        baseline=baseline,
    )


def optimize_partition_across_ratios(
    model: StageCostModel,
    dcut_profiles: dict[float, float | Sequence[float]],
    estimate: BucketEstimate | None = None,
    min_layers: int = 1,
    max_layers: Sequence[int] | None = None,
    k_best: int = DEFAULT_K_BEST,
    *,
    target_bs: float | int | None = None,
    layout: LayerLayout | None = None,
    prefix_l_range: tuple[int, int] | None = None,
    stage_comm_ms: Sequence[float] | None = None,
    all_boundaries: bool = False,
    ratio_weights: dict[float, float] | None = None,
) -> RobustPartitionResult:
    """Find a static partition that remains balanced as runtime ratio changes.

    ``ratio_weights`` is optional.  Without it the objective is minimax over
    the measured ratio envelope, which protects against a workload shifting
    between shallow and deep cuts.  With weights, the weighted mean is the
    primary objective while ``worst_cycle_ms`` remains visible in reports.
    """
    if k_best <= 0:
        raise OptimizerError("k_best must be positive")
    if target_bs is not None:
        estimate = model.estimate_for_bs(target_bs)
    elif estimate is None:
        estimate = model.target_bucket()
    active_layout = layout or model.layout
    if active_layout is not None and active_layout.num_layers != model.num_layers:
        raise OptimizerError("layout and stage model have different layer counts")
    partitions = (
        _all_partitions(model.num_layers, model.pp_size, min_layers, max_layers)
        if all_boundaries
        else _family_partitions(
            model.num_layers, model.pp_size, min_layers, max_layers, prefix_l_range
        )
    )
    if not partitions:
        raise OptimizerError("no valid partition satisfies the layer limits")
    resolved_comm = _resolve_stage_comm(stage_comm_ms, model.pp_size)
    profiles = _normalise_dcut_profiles(dcut_profiles, model.pp_size)
    ratios = tuple(ratio for ratio, _scales in profiles)
    if ratio_weights is None:
        weights = {ratio: 1.0 for ratio in ratios}
    else:
        weights = {float(ratio): float(weight) for ratio, weight in ratio_weights.items()}
        if set(weights) != set(ratios) or any(weight < 0.0 for weight in weights.values()):
            raise OptimizerError("ratio_weights must cover every profile with non-negative values")
        if sum(weights.values()) <= 0.0:
            raise OptimizerError("ratio_weights must contain a positive value")
    candidates: list[RobustPartitionCandidate] = []
    for partition in partitions:
        cycle_by_ratio: dict[float, float] = {}
        for ratio, scales in profiles:
            item = _candidate_from_model(
                model,
                estimate,
                partition,
                resolved_comm,
                active_layout,
                dcut_ratio=ratio,
                dcut_stage_scale=scales,
            )
            cycle_by_ratio[ratio] = item.cycle_time_ms
        worst = max(cycle_by_ratio.values())
        weighted = sum(cycle_by_ratio[r] * weights[r] for r in ratios) / sum(
            weights.values()
        )
        candidates.append(
            RobustPartitionCandidate(
                partition=tuple(partition),
                cycle_by_ratio=cycle_by_ratio,
                worst_cycle_ms=worst,
                weighted_cycle_ms=weighted,
            )
        )
    sort_key = (
        (lambda item: (item.weighted_cycle_ms, item.worst_cycle_ms, item.partition))
        if ratio_weights is not None
        else (lambda item: (item.worst_cycle_ms, item.weighted_cycle_ms, item.partition))
    )
    candidates.sort(key=sort_key)

    def evaluate_baseline() -> RobustPartitionCandidate:
        cycles = {}
        for ratio, scales in profiles:
            item = _candidate_from_model(
                model,
                estimate,
                model.baseline_partition,
                resolved_comm,
                active_layout,
                dcut_ratio=ratio,
                dcut_stage_scale=scales,
            )
            cycles[ratio] = item.cycle_time_ms
        return RobustPartitionCandidate(
            partition=model.baseline_partition,
            cycle_by_ratio=cycles,
            worst_cycle_ms=max(cycles.values()),
            weighted_cycle_ms=sum(cycles[r] * weights[r] for r in ratios)
            / sum(weights.values()),
        )

    return RobustPartitionResult(
        target_bucket=estimate.bucket,
        target_bs=float(estimate.bucket),
        best=candidates[0],
        candidates=candidates[:k_best],
        baseline=evaluate_baseline(),
    )


def optimize_partition_across_buckets(
    model: StageCostModel,
    dcut_profiles_by_bucket: dict[int, dict[float, float | Sequence[float]]],
    min_layers: int = 1,
    max_layers: Sequence[int] | None = None,
    k_best: int = DEFAULT_K_BEST,
    *,
    layout: LayerLayout | None = None,
    prefix_l_range: tuple[int, int] | None = None,
    stage_comm_ms: Sequence[float] | None = None,
    all_boundaries: bool = False,
    bucket_weights: dict[int, float] | None = None,
) -> MultiBucketPartitionResult:
    """Select one static partition across multiple batch/graph buckets."""
    if k_best <= 0:
        raise OptimizerError("k_best must be positive")
    if not dcut_profiles_by_bucket:
        raise OptimizerError("dcut_profiles_by_bucket must not be empty")
    buckets = tuple(sorted(int(bucket) for bucket in dcut_profiles_by_bucket))
    if set(buckets) != set(model.buckets):
        raise OptimizerError(
            "D-Cut profile buckets must exactly match the stage model buckets"
        )
    weights = (
        {bucket: 1.0 for bucket in buckets}
        if bucket_weights is None
        else {int(bucket): float(value) for bucket, value in bucket_weights.items()}
    )
    if set(weights) != set(buckets) or any(value < 0.0 for value in weights.values()):
        raise OptimizerError("bucket_weights must cover every bucket")
    if sum(weights.values()) <= 0.0:
        raise OptimizerError("bucket_weights must contain a positive value")
    active_layout = layout or model.layout
    if active_layout is not None and active_layout.num_layers != model.num_layers:
        raise OptimizerError("layout and stage model have different layer counts")
    partitions = (
        _all_partitions(model.num_layers, model.pp_size, min_layers, max_layers)
        if all_boundaries
        else _family_partitions(
            model.num_layers, model.pp_size, min_layers, max_layers, prefix_l_range
        )
    )
    if not partitions:
        raise OptimizerError("no valid partition satisfies the layer limits")
    resolved_comm = _resolve_stage_comm(stage_comm_ms, model.pp_size)
    normalized = {
        bucket: _normalise_dcut_profiles(profiles, model.pp_size)
        for bucket, profiles in dcut_profiles_by_bucket.items()
    }

    def evaluate(partition: Sequence[int]) -> MultiBucketPartitionCandidate:
        cycles_by_bucket: dict[int, float] = {}
        for bucket in buckets:
            estimate = model.estimate_for_bs(bucket)
            cycles = []
            for _ratio, scales in normalized[bucket]:
                item = _candidate_from_model(
                    model,
                    estimate,
                    partition,
                    resolved_comm,
                    active_layout,
                    dcut_stage_scale=scales,
                )
                cycles.append(item.cycle_time_ms)
            cycles_by_bucket[bucket] = max(cycles)
        weighted = sum(
            cycles_by_bucket[bucket] * weights[bucket] for bucket in buckets
        ) / sum(weights.values())
        return MultiBucketPartitionCandidate(
            partition=tuple(partition),
            worst_cycle_ms=max(cycles_by_bucket.values()),
            weighted_cycle_ms=weighted,
            cycle_by_bucket=cycles_by_bucket,
        )

    candidates = [evaluate(partition) for partition in partitions]
    candidates.sort(key=lambda item: (item.worst_cycle_ms, item.weighted_cycle_ms, item.partition))
    return MultiBucketPartitionResult(
        target_buckets=buckets,
        best=candidates[0],
        candidates=candidates[:k_best],
        baseline=evaluate(model.baseline_partition),
    )


def choose_dynamic_dcut_ratio(
    stage_ms: Sequence[float],
    dcut_profiles: dict[float, float | Sequence[float]],
    *,
    layer_fraction: Sequence[float] | float = 1.0,
) -> float:
    """Choose a ratio from live stage timings to reduce the PP bottleneck.

    ``stage_ms`` is an EMA or a recent synchronized sample.  ``layer_fraction``
    estimates how much of each stage is removable layer work; the remainder is
    treated as a fixed floor.  This makes the controller conservative when a
    stage is dominated by communication or an LM head, and naturally changes
    its choice as the PP bottleneck moves.
    """
    observed = tuple(float(value) for value in stage_ms)
    if not observed or any(value <= 0.0 for value in observed):
        raise OptimizerError("stage_ms must contain positive values")
    if isinstance(layer_fraction, Sequence) and not isinstance(layer_fraction, (str, bytes)):
        fractions = tuple(float(value) for value in layer_fraction)
    else:
        fractions = (float(layer_fraction),) * len(observed)
    if len(fractions) != len(observed) or any(not 0.0 <= value <= 1.0 for value in fractions):
        raise OptimizerError("layer_fraction must be in [0, 1] for every stage")
    profiles = _normalise_dcut_profiles(dcut_profiles, len(observed))
    best_ratio = profiles[-1][0]
    best_cycle = float("inf")
    for ratio, scales in profiles:
        predicted = tuple(
            base * (1.0 - fraction) + base * fraction * scale
            for base, fraction, scale in zip(observed, fractions, scales)
        )
        cycle = max(predicted)
        if (cycle, ratio) < (best_cycle, best_ratio):
            best_cycle = cycle
            best_ratio = ratio
    return best_ratio
