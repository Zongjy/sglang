#!/usr/bin/env python3
"""Analytical selector for one-dimensional PP layer partitions.

The candidate family is intentionally small::

    (l, ..., l, L - (P - 1) * l)

Every candidate is evaluated at one profiled execution bucket. The selected
partition is simply the candidate with the lowest predicted bottleneck stage
time.
"""

from __future__ import annotations

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
    current_partition: tuple[int, ...]
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
        if self.best.partition == self.current_partition:
            return (
                f"keep current {best_text}; predicted objective cycle "
                f"{self.best.cycle_time_ms:.3f} ms"
            )
        return (
            f"switch to {best_text}; predicted objective cycle "
            f"{self.best.cycle_time_ms:.3f} ms vs current "
            f"{self.current.cycle_time_ms:.3f} ms"
        )

    @staticmethod
    def _candidate_dict(item: OptimizerCandidate) -> dict[str, Any]:
        return {
            "partition": list(item.partition),
            "stage_ms": list(item.stage_ms),
            "cycle_time_ms": item.cycle_time_ms,
            "bottleneck_rank": item.bottleneck_rank,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_bucket": self.target_bucket,
            "target_bs": self.target_bs,
            "stage_comm_ms": list(self.stage_comm_ms),
            "selected": self._candidate_dict(self.best),
            "candidates": [self._candidate_dict(item) for item in self.candidates],
            "current_partition": list(self.current_partition),
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
            f"current = {','.join(map(str, self.current_partition))}, "
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
) -> OptimizerCandidate:
    counts = tuple(int(value) for value in partition)
    stage_ms = tuple(
        float(value)
        for value in model.predict_stages(
            counts,
            estimate=estimate,
            stage_comm_ms=stage_comm_ms,
            layout=layout,
        )
    )
    bottleneck_rank = max(range(len(stage_ms)), key=stage_ms.__getitem__)
    return OptimizerCandidate(
        partition=counts,
        stage_ms=stage_ms,
        bottleneck_ms=stage_ms[bottleneck_rank],
        bottleneck_rank=bottleneck_rank,
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
    partitions = _family_partitions(
        model.num_layers,
        model.pp_size,
        min_layers,
        max_layers,
        prefix_l_range,
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
        model.current_partition,
        resolved_stage_comm,
        active_layout,
    )
    return OptimizationResult(
        target_bucket=estimate.bucket,
        target_bs=float(estimate.bucket),
        stage_comm_ms=resolved_stage_comm,
        best=candidates[0],
        candidates=candidates[:k_best],
        current_partition=model.current_partition,
        current=current,
    )
