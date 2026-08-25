#!/usr/bin/env python3
"""One-dimensional PP partition selector.

The deployable family is intentionally small::

    (l, ..., l, L - (P - 1) * l)

The first ``P - 1`` stages contain the same number of target layers and the
last stage contains the residual target layers plus the draft model.  The
selector first finds the measured latency-indifference band at one execution
bucket.  Raw memory capacity refines only candidates inside that band; it is
never mixed with latency in a scalar score.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

try:
    from benchmark.pp_spec.model_layout import LayerLayout, uniform_prefix_partition
    from benchmark.pp_spec.stage_model import BucketEstimate, StageCostModel
except ImportError:  # pragma: no cover - flat script execution
    from model_layout import LayerLayout, uniform_prefix_partition
    from stage_model import BucketEstimate, StageCostModel


DEFAULT_K_BEST = 20
DEFAULT_NOISE_SIGMA = 2.0
DEFAULT_RELATIVE_TOLERANCE = 0.02


class OptimizerError(RuntimeError):
    pass


@dataclass(frozen=True)
class StageDecomposition:
    rank: int
    layers: int
    # On the last rank, fixed service includes draft proposal and all other
    # non-target-layer work.
    fixed_ms: float
    layer_ms: float
    comm_ms: float

    @property
    def total_ms(self) -> float:
        return self.fixed_ms + self.layer_ms + self.comm_ms


@dataclass(frozen=True)
class OptimizerCandidate:
    partition: tuple[int, ...]
    draft_rank: int
    stage_ms: tuple[float, ...]
    bottleneck_ms: float
    bottleneck_rank: int
    noise_std_ms: float
    decomposition: tuple[StageDecomposition, ...]
    # Capacity refines candidates only inside the latency-indifference band.
    target_feasible: bool | None = None
    memory_margin_gib: tuple[float, ...] = ()
    binding_rank: int | None = None
    binding_resource: str | None = None
    kv_tokens: int | None = None
    mamba_slots: int | None = None
    mamba_ratio: float | None = None
    memory_capacity: int | None = None
    scheduler_limit: int | None = None
    effective_limit: int | None = None
    mamba_capacity: int | None = None
    kv_capacity: int | None = None

    @property
    def cycle_time_ms(self) -> float:
        return self.bottleneck_ms


@dataclass
class OptimizationResult:
    target_bucket: int
    target_bs: float
    accept_len: float
    t_comm_ms: float
    stage_comm_ms: tuple[float, ...]
    noise_sigma: float
    best: OptimizerCandidate
    candidates: list[OptimizerCandidate]
    indifference_set: list[OptimizerCandidate]
    current_partition: tuple[int, ...]
    current: OptimizerCandidate
    keep_current: bool
    recommendation: str
    predicted_throughput_tok_s: float
    compute_optimal: tuple[int, ...] | None = None
    capacity_optimal: tuple[int, ...] | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def selected(self) -> OptimizerCandidate:
        """The deployment choice after the noise/indifference guard."""
        return self.current if self.keep_current else self.best

    @property
    def cycle_time_ms(self) -> float:
        return self.selected.cycle_time_ms

    @property
    def good_range(self) -> tuple[tuple[int, ...], ...]:
        return tuple(item.partition for item in self.indifference_set)

    @property
    def l_range(self) -> tuple[int, ...]:
        return tuple(
            item.partition[0] if len(item.partition) > 1 else 0
            for item in self.indifference_set
        )

    def _candidate_dict(self, item: OptimizerCandidate) -> dict[str, Any]:
        return {
            "partition": list(item.partition),
            "draft_rank": item.draft_rank,
            "stage_ms": list(item.stage_ms),
            "cycle_time_ms": item.cycle_time_ms,
            "bottleneck_ms": item.bottleneck_ms,
            "bottleneck_rank": item.bottleneck_rank,
            "noise_std_ms": item.noise_std_ms,
            "target_feasible": item.target_feasible,
            "memory_margin_gib": list(item.memory_margin_gib),
            "binding_rank": item.binding_rank,
            "binding_resource": item.binding_resource,
            "kv_tokens": item.kv_tokens,
            "mamba_slots": item.mamba_slots,
            "mamba_ratio": item.mamba_ratio,
            "memory_capacity": item.memory_capacity,
            "scheduler_limit": item.scheduler_limit,
            "effective_limit": item.effective_limit,
            "mamba_capacity": item.mamba_capacity,
            "kv_capacity": item.kv_capacity,
            "decomposition": [
                {
                    "rank": stage.rank,
                    "layers": stage.layers,
                    "fixed_ms": stage.fixed_ms,
                    "layer_ms": stage.layer_ms,
                    "comm_ms": stage.comm_ms,
                    "total_ms": stage.total_ms,
                }
                for stage in item.decomposition
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_bucket": self.target_bucket,
            "target_bs": self.target_bs,
            "accept_len": self.accept_len,
            "t_comm_ms": self.t_comm_ms,
            "stage_comm_ms": list(self.stage_comm_ms),
            "noise_sigma": self.noise_sigma,
            "best": self._candidate_dict(self.best),
            "selected": self._candidate_dict(self.selected),
            "candidates": [self._candidate_dict(item) for item in self.candidates],
            "indifference_set": [
                self._candidate_dict(item) for item in self.indifference_set
            ],
            "good_range": [list(item) for item in self.good_range],
            "l_range": list(self.l_range),
            "current_partition": list(self.current_partition),
            "current": self._candidate_dict(self.current),
            "keep_current": self.keep_current,
            "recommendation": self.recommendation,
            "predicted_throughput_tok_s": self.predicted_throughput_tok_s,
            "compute_optimal": (
                list(self.compute_optimal) if self.compute_optimal else None
            ),
            "capacity_optimal": (
                list(self.capacity_optimal) if self.capacity_optimal else None
            ),
            "warnings": list(self.warnings),
        }

    def to_report(self) -> str:
        selected = self.selected
        lines = [
            f"recommendation: {self.recommendation}",
            "",
            f"target execution bucket = {self.target_bucket} (bs={self.target_bs:g}), "
            f"accept_len = {self.accept_len:.3f}, "
            f"selected cycle_time = {self.best.cycle_time_ms:.3f} ms, "
            f"selected = {','.join(map(str, selected.partition))} "
            f"({selected.cycle_time_ms:.3f} ms)",
            "policy: latency-equivalent prefix partitions first; raw Mamba/KV "
            "capacity refines that set; draft fixed on the last rank",
            "",
            f"{'partition':<22} {'cycle_ms':>10} {'bottleneck':>10} "
            f"{'feasible':>9} {'C_mem':>8} {'C_m':>7} {'C_kv':>7}  stages (ms)",
        ]
        for item in self.candidates:
            stages = " | ".join(f"{value:.2f}" for value in item.stage_ms)
            marker = " *" if item in self.indifference_set else ""
            feasible = (
                "?"
                if item.target_feasible is None
                else "yes" if item.target_feasible else "no"
            )
            memory_capacity = (
                "-" if item.memory_capacity is None else str(item.memory_capacity)
            )
            mamba_capacity = (
                "-" if item.mamba_capacity is None else str(item.mamba_capacity)
            )
            kv_capacity = "-" if item.kv_capacity is None else str(item.kv_capacity)
            lines.append(
                f"{','.join(map(str, item.partition)):<22} "
                f"{item.cycle_time_ms:>10.3f} {item.bottleneck_rank:>10} "
                f"{feasible:>9} {memory_capacity:>8} {mamba_capacity:>7} "
                f"{kv_capacity:>7}  {stages}{marker}"
            )
        lines += ["", "selected stage decomposition (ms):"]
        for stage in selected.decomposition:
            lines.append(
                f"  rank {stage.rank}: layers={stage.layers} "
                f"fixed={stage.fixed_ms:.3f} layers={stage.layer_ms:.3f} "
                f"comm={stage.comm_ms:.3f} "
                f"total={stage.total_ms:.3f}"
            )
        lines += [
            "",
            "last-stage fixed residual = T_d + T_other; T_other covers "
            "draft handoff, verification/bookkeeping, sampling, graph dispatch, "
            "and non-overlapped PP synchronization",
            "good l range: "
            + ", ".join(str(item.partition[0]) for item in self.indifference_set),
            f"diagnostic throughput at target bucket ~= "
            f"{self.predicted_throughput_tok_s:.1f} tok/s (not an optimizer objective)",
            f"indifference tolerance: {self.noise_sigma:g}-sigma plus "
            f"{DEFAULT_RELATIVE_TOLERANCE:.1%} relative",
        ]
        for warning in self.warnings:
            lines.append(f"WARNING: {warning}")
        return "\n".join(lines)


def _layer_cost(
    model: StageCostModel,
    estimate: BucketEstimate,
    start: int,
    end: int,
    layout: LayerLayout | None,
) -> float:
    # Keep this helper in sync with StageCostModel's typed layer accounting.
    if (
        layout is not None
        and estimate.gdn_cost_ms is not None
        and estimate.full_cost_ms is not None
    ):
        gdn, full = layout.count_range(start, end)
        return gdn * estimate.gdn_cost_ms + full * estimate.full_cost_ms
    return (end - start) * estimate.layer_cost_ms


def _candidate_from_model(
    model: StageCostModel,
    estimate: BucketEstimate,
    partition: Sequence[int],
    t_comm_ms: float,
    stage_comm_ms: Sequence[float] | None,
    layout: LayerLayout | None,
) -> OptimizerCandidate:
    counts = tuple(int(value) for value in partition)
    stage_ms = tuple(
        float(value)
        for value in model.predict_stages(
            counts,
            estimate=estimate,
            t_comm_ms=t_comm_ms,
            stage_comm_ms=stage_comm_ms,
            layout=layout,
        )
    )
    starts: list[int] = []
    start = 0
    for count in counts:
        starts.append(start)
        start += count
    decomposition: list[StageDecomposition] = []
    for rank, (count, begin) in enumerate(zip(counts, starts)):
        layer_ms = _layer_cost(model, estimate, begin, begin + count, layout)
        if stage_comm_ms is None:
            comm = t_comm_ms if rank > 0 else 0.0
        else:
            resolved_comm = tuple(float(value) for value in stage_comm_ms)
            if len(resolved_comm) == len(counts) - 1:
                resolved_comm = (0.0, *resolved_comm)
            comm = resolved_comm[rank]
        fixed = max(stage_ms[rank] - layer_ms - comm, 0.0)
        decomposition.append(
            StageDecomposition(
                rank=rank,
                layers=count,
                fixed_ms=fixed,
                layer_ms=layer_ms,
                comm_ms=comm,
            )
        )
    bottleneck_rank = max(range(len(stage_ms)), key=stage_ms.__getitem__)
    count = counts[bottleneck_rank]
    variance = (
        count * count * estimate.layer_cost_var + estimate.fixed_var[bottleneck_rank]
    )
    return OptimizerCandidate(
        partition=counts,
        draft_rank=len(counts) - 1,
        stage_ms=stage_ms,
        bottleneck_ms=stage_ms[bottleneck_rank],
        bottleneck_rank=bottleneck_rank,
        noise_std_ms=math.sqrt(max(variance, 0.0)),
        decomposition=tuple(decomposition),
    )


def _capacity_fields(
    candidate: OptimizerCandidate,
    estimate: Any,
) -> OptimizerCandidate:
    """Attach raw memory capacity and separate runtime-limit diagnostics."""
    if estimate is None:
        return candidate
    memory_capacity = getattr(estimate, "memory_capacity", None)
    target_feasible = getattr(estimate, "target_feasible", None)
    return dataclasses.replace(
        candidate,
        target_feasible=target_feasible,
        memory_margin_gib=tuple(getattr(estimate, "memory_margin_gib", ()) or ()),
        binding_rank=getattr(estimate, "binding_rank", None),
        binding_resource=getattr(estimate, "binding_resource", None),
        kv_tokens=getattr(estimate, "kv_tokens", None),
        mamba_slots=getattr(estimate, "mamba_slots", None),
        mamba_ratio=getattr(estimate, "ratio", None),
        memory_capacity=(None if memory_capacity is None else int(memory_capacity)),
        scheduler_limit=getattr(estimate, "scheduler_limit", None),
        effective_limit=getattr(estimate, "effective_limit", None),
        mamba_capacity=getattr(estimate, "mamba_capacity", None),
        kv_capacity=getattr(estimate, "kv_capacity", None),
    )


def _is_feasible(estimate: Any) -> bool:
    if estimate is None:
        return False
    value = getattr(estimate, "target_feasible", None)
    return True if value is None else bool(value)


def _family_partitions(
    num_layers: int,
    pp_size: int,
    min_layers: int,
    max_layers: Sequence[int] | None,
    prefix_l_range: tuple[int, int] | None = None,
) -> list[tuple[int, ...]]:
    if pp_size <= 0 or num_layers <= 0:
        raise OptimizerError("num_layers and pp_size must be positive")
    limits = (
        (num_layers,) * pp_size
        if max_layers is None
        else tuple(int(value) for value in max_layers)
    )
    if len(limits) != pp_size:
        raise OptimizerError("max_layers must have one entry per PP rank")
    if pp_size == 1:
        return [(num_layers,)] if min_layers <= num_layers <= limits[0] else []
    upper = (num_layers - 1) // (pp_size - 1)
    min_count = max(int(min_layers), 1)
    lower = min_count
    if prefix_l_range is not None:
        range_lower, range_upper = map(int, prefix_l_range)
        lower = max(lower, range_lower)
        upper = min(upper, range_upper)
    partitions: list[tuple[int, ...]] = []
    for l in range(lower, upper + 1):
        partition = uniform_prefix_partition(num_layers, pp_size, l)
        if all(
            min_count <= count <= limits[rank] for rank, count in enumerate(partition)
        ):
            partitions.append(partition)
    return partitions


def optimize(
    model: StageCostModel,
    estimate: BucketEstimate | None = None,
    t_comm_ms: float = 0.0,
    min_layers: int = 1,
    max_layers: Sequence[int] | None = None,
    k_best: int = DEFAULT_K_BEST,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    capacity: Any = None,
    *,
    target_bs: float | int | None = None,
    layout: LayerLayout | None = None,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    prefix_l_range: tuple[int, int] | None = None,
    stage_comm_ms: Sequence[float] | None = None,
) -> OptimizationResult:
    """Select a prefix-uniform partition at one measured execution bucket.

    ``capacity`` is a callable ``partition -> CapacityEstimate``.  Latency is
    solved first.  Capacity is evaluated only for the latency-indifference set
    (plus the current partition); raw memory capacity then breaks ties inside
    that set.  If the whole set is infeasible, candidates are expanded in
    increasing cycle-time order until the first feasible partition is found.
    """
    if k_best <= 0:
        raise OptimizerError("k_best must be positive")
    if estimate is None:
        estimate = (
            model.estimate_for_bs(target_bs)
            if target_bs is not None
            else model.target_bucket()
        )
    elif target_bs is not None:
        estimate = model.estimate_for_bs(target_bs)
    active_layout = layout or model.layout
    if active_layout is not None and active_layout.num_layers != model.num_layers:
        raise OptimizerError("layout and stage model have different layer counts")
    partitions = _family_partitions(
        model.num_layers,
        model.pp_size,
        min_layers,
        max_layers,
        prefix_l_range=prefix_l_range,
    )
    if not partitions:
        raise OptimizerError(
            "no valid prefix-uniform partition satisfies the layer limits"
        )
    warnings: list[str] = []
    if t_comm_ms < 0:
        raise OptimizerError("t_comm_ms must be non-negative")
    resolved_stage_comm_ms = None
    if stage_comm_ms is not None:
        resolved_stage_comm_ms = tuple(float(value) for value in stage_comm_ms)
        if len(resolved_stage_comm_ms) == model.pp_size - 1:
            resolved_stage_comm_ms = (0.0, *resolved_stage_comm_ms)
        if len(resolved_stage_comm_ms) != model.pp_size or any(
            value < 0.0 for value in resolved_stage_comm_ms
        ):
            raise OptimizerError(
                "stage_comm_ms must contain PP or PP-1 non-negative values"
            )
    capacity_cache: dict[tuple[int, ...], Any] = {}

    def capacity_for(partition: tuple[int, ...]) -> Any:
        if capacity is not None and partition not in capacity_cache:
            try:
                capacity_cache[partition] = capacity(partition)
            except Exception as exc:
                warnings.append(f"capacity failed for {partition}: {exc}")
                capacity_cache[partition] = None
        return capacity_cache.get(partition)

    latency_candidates: list[OptimizerCandidate] = []
    for partition in partitions:
        latency_candidates.append(
            _candidate_from_model(
                model,
                estimate,
                partition,
                float(t_comm_ms),
                resolved_stage_comm_ms,
                active_layout,
            )
        )
    latency_candidates.sort(key=lambda item: (item.cycle_time_ms, item.partition))
    compute_best = latency_candidates[0]
    noise_band = max(
        max(float(noise_sigma), 0.0) * compute_best.noise_std_ms,
        max(float(relative_tolerance), 0.0) * max(compute_best.cycle_time_ms, 0.0),
    )
    threshold = compute_best.cycle_time_ms + noise_band
    latency_band_raw = [
        item for item in latency_candidates if item.cycle_time_ms <= threshold + 1e-12
    ]

    evaluated: dict[tuple[int, ...], OptimizerCandidate] = {}

    def with_capacity(item: OptimizerCandidate) -> OptimizerCandidate:
        if capacity is None:
            return item
        if item.partition not in evaluated:
            cap = capacity_for(item.partition)
            evaluated[item.partition] = _capacity_fields(item, cap)
        return evaluated[item.partition]

    expanded_for_capacity = False
    if capacity is None:
        best = compute_best
        indifference = sorted(latency_band_raw, key=lambda item: item.partition)
    else:
        indifference = sorted(
            [with_capacity(item) for item in latency_band_raw],
            key=lambda item: item.partition,
        )
        feasible_band = [
            item for item in indifference if _is_feasible(capacity_for(item.partition))
        ]
        if feasible_band:
            # Every item here is latency-equivalent by construction.  Prefer
            # raw memory headroom, then the lower modeled cycle time.
            best = min(
                feasible_band,
                key=lambda item: (
                    -(item.memory_capacity or 0),
                    item.cycle_time_ms,
                    item.partition,
                ),
            )
        else:
            best = None
            band_partitions = {item.partition for item in latency_band_raw}
            for raw in latency_candidates:
                if raw.partition in band_partitions:
                    continue
                candidate = with_capacity(raw)
                if _is_feasible(capacity_for(raw.partition)):
                    best = candidate
                    expanded_for_capacity = True
                    warnings.append(
                        "the latency-equivalent l range is memory-infeasible; "
                        "expanded to the lowest-latency feasible partition"
                    )
                    break
            if best is None:
                raise OptimizerError(
                    "no prefix-uniform partition is feasible at the requested "
                    "memory working point"
                )

    current_raw = _candidate_from_model(
        model,
        estimate,
        model.current_partition,
        float(t_comm_ms),
        resolved_stage_comm_ms,
        active_layout,
    )
    current_feasible = True
    if capacity is not None:
        current_cap = capacity_for(model.current_partition)
        current_feasible = _is_feasible(current_cap)
        current_raw = _capacity_fields(current_raw, current_cap)
        evaluated[model.current_partition] = current_raw
    all_candidates = [
        evaluated.get(item.partition, item) for item in latency_candidates
    ]
    current_is_deployable = model.current_partition in partitions
    current_in_band = (
        current_is_deployable
        and current_feasible
        and current_raw.cycle_time_ms <= threshold + 1e-12
    )
    if capacity is None:
        keep_current = current_in_band
    else:
        keep_current = current_in_band and (current_raw.memory_capacity or 0) >= (
            best.memory_capacity or 0
        )
    current_text = ",".join(map(str, model.current_partition))
    if keep_current:
        current_l = model.current_partition[0] if model.pp_size > 1 else 0
        recommendation = (
            f"keep current {current_text}; l={current_l} is inside the good "
            f"range (cycle {current_raw.cycle_time_ms:.3f} ms, compute best "
            f"{compute_best.cycle_time_ms:.3f} ms)"
        )
        if capacity is not None:
            recommendation += (
                f" and has raw memory capacity {current_raw.memory_capacity}"
            )
    else:
        recommendation = (
            f"switch to {','.join(map(str, best.partition))}; cycle "
            f"{best.cycle_time_ms:.3f} ms vs current "
            f"{current_raw.cycle_time_ms:.3f} ms"
        )
        if capacity is not None:
            recommendation += f", raw memory capacity {best.memory_capacity}"
            if best.partition != compute_best.partition and not expanded_for_capacity:
                recommendation += (
                    f" (latency-equivalent refinement of compute optimum "
                    f"{','.join(map(str, compute_best.partition))})"
                )
        if not current_feasible:
            recommendation += " (current partition fails the memory feasibility check)"
        elif not current_is_deployable:
            recommendation += (
                " (current partition is outside the prefix-uniform family)"
            )

    selected = current_raw if keep_current else best
    diagnostic_throughput = (
        estimate.bucket * estimate.accept_len / (selected.cycle_time_ms / 1000.0)
        if selected.cycle_time_ms > 0 and estimate.accept_len > 0
        else 0.0
    )
    shown = all_candidates[:k_best]
    if best not in shown:
        shown = [best] + shown[:-1]
    return OptimizationResult(
        target_bucket=estimate.bucket,
        target_bs=float(estimate.bucket),
        accept_len=float(estimate.accept_len),
        t_comm_ms=float(t_comm_ms),
        stage_comm_ms=(
            resolved_stage_comm_ms
            if resolved_stage_comm_ms is not None
            else tuple(
                0.0 if rank == 0 else float(t_comm_ms) for rank in range(model.pp_size)
            )
        ),
        noise_sigma=float(noise_sigma),
        best=best,
        candidates=shown,
        indifference_set=indifference,
        current_partition=model.current_partition,
        current=current_raw,
        keep_current=keep_current,
        recommendation=recommendation,
        predicted_throughput_tok_s=diagnostic_throughput,
        compute_optimal=compute_best.partition,
        capacity_optimal=(best.partition if capacity is not None else None),
        warnings=warnings,
    )
