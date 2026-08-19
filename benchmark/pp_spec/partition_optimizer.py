#!/usr/bin/env python3
"""Min-max PP partition optimizer on top of the PPM stage cost model.

Single objective (saturated-throughput): at the target working point b*,

    E_r = c(b*) * n_r + F_r(b*) + t_comm * 1[r > 0]

minimized as ``max_r E_r`` over contiguous layer partitions.  The draft block
stays on the last rank by default -- its inputs are produced locally there,
so placement is a physical constraint, not a free variable; the draft cost
D is already bundled inside F_last.  ``allow_draft_relocation=True`` opts
into an analysis-only enumeration with
``E_r = c * n_r + F_r + D * (1[r = d] - 1[r = last])``.

The DP core is a prefix-sum min-max recurrence (O(P * L^2)), generalized to
keep the top-k predecessor candidates per state so near-optimal partitions
survive for the significance analysis.  If the current partition's predicted
bottleneck lies within ``noise_sigma`` standard deviations of the best, the
recommendation is "keep current" -- never a switch within the noise band.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Sequence

try:  # importable both as benchmark.pp_spec.* and as a flat script dir
    from benchmark.pp_spec.stage_model import BucketEstimate, StageCostModel
except ImportError:  # pragma: no cover - depends on sys.path setup
    from stage_model import BucketEstimate, StageCostModel


DEFAULT_K_BEST = 20
DEFAULT_NOISE_SIGMA = 2.0


class OptimizerError(RuntimeError):
    pass


@dataclass(frozen=True)
class StageDecomposition:
    rank: int
    layers: int
    fixed_ms: float  # F_r (everything on the rank besides its layers)
    layer_ms: float  # c * n_r
    draft_ms: float  # explicit draft delta; 0 unless draft relocation analysis
    comm_ms: float

    @property
    def total_ms(self) -> float:
        return self.fixed_ms + self.layer_ms + self.draft_ms + self.comm_ms


@dataclass(frozen=True)
class OptimizerCandidate:
    partition: tuple[int, ...]
    draft_rank: int
    stage_ms: tuple[float, ...]
    bottleneck_ms: float
    bottleneck_rank: int
    noise_std_ms: float
    decomposition: tuple[StageDecomposition, ...]
    # Capacity / unified-objective fields (None when no capacity model ran).
    bs_max: int | None = None
    per_slot_bs: float | None = None  # bs_max / pp_loop_size
    cadence_at_capacity_ms: float | None = None  # cadence(p, per_slot_bs)
    throughput_tok_s: float | None = None  # bs_max*accept/(loop*cadence)
    capacity_binding_rank: int | None = None
    kv_tokens: int | None = None
    mamba_k: int | None = None
    mamba_ratio: float | None = None


@dataclass
class OptimizationResult:
    target_bucket: int
    target_bs: float
    accept_len: float
    t_comm_ms: float
    noise_sigma: float
    allow_draft_relocation: bool
    best: OptimizerCandidate  # recommended: max throughput (min bottleneck w/o capacity)
    candidates: list[OptimizerCandidate]
    indifference_set: list[OptimizerCandidate]
    current_partition: tuple[int, ...]
    current: OptimizerCandidate
    keep_current: bool
    recommendation: str
    predicted_throughput_tok_s: float
    compute_optimal: tuple[int, ...] | None = None  # min cadence at b*
    capacity_optimal: tuple[int, ...] | None = None  # max BS_max
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        def candidate_dict(item: OptimizerCandidate) -> dict:
            return {
                "partition": list(item.partition),
                "draft_rank": item.draft_rank,
                "stage_ms": list(item.stage_ms),
                "bottleneck_ms": item.bottleneck_ms,
                "bottleneck_rank": item.bottleneck_rank,
                "noise_std_ms": item.noise_std_ms,
                "bs_max": item.bs_max,
                "per_slot_bs": item.per_slot_bs,
                "cadence_at_capacity_ms": item.cadence_at_capacity_ms,
                "throughput_tok_s": item.throughput_tok_s,
                "capacity_binding_rank": item.capacity_binding_rank,
                "kv_tokens": item.kv_tokens,
                "mamba_k": item.mamba_k,
                "mamba_ratio": item.mamba_ratio,
                "decomposition": [
                    {
                        "rank": stage.rank,
                        "layers": stage.layers,
                        "fixed_ms": stage.fixed_ms,
                        "layer_ms": stage.layer_ms,
                        "draft_ms": stage.draft_ms,
                        "comm_ms": stage.comm_ms,
                        "total_ms": stage.total_ms,
                    }
                    for stage in item.decomposition
                ],
            }

        return {
            "target_bucket": self.target_bucket,
            "target_bs": self.target_bs,
            "accept_len": self.accept_len,
            "t_comm_ms": self.t_comm_ms,
            "noise_sigma": self.noise_sigma,
            "allow_draft_relocation": self.allow_draft_relocation,
            "best": candidate_dict(self.best),
            "candidates": [candidate_dict(item) for item in self.candidates],
            "indifference_set": [candidate_dict(item) for item in self.indifference_set],
            "current_partition": list(self.current_partition),
            "current": candidate_dict(self.current),
            "keep_current": self.keep_current,
            "recommendation": self.recommendation,
            "predicted_throughput_tok_s": self.predicted_throughput_tok_s,
            "compute_optimal": list(self.compute_optimal) if self.compute_optimal else None,
            "capacity_optimal": list(self.capacity_optimal) if self.capacity_optimal else None,
            "warnings": list(self.warnings),
        }

    def to_report(self) -> str:
        lines = [
            f"recommendation: {self.recommendation}",
            "",
            f"target working point b* = {self.target_bs:g} "
            f"(bucket {self.target_bucket}), accept_len = {self.accept_len:.3f}, "
            f"t_comm = {self.t_comm_ms:.3f} ms",
        ]
        if self.allow_draft_relocation:
            lines.append(
                "draft placement: enumerated over all ranks -- ANALYSIS ONLY; "
                "moving the draft requires physically relocating draft "
                "weights/KV and is not directly deployable"
            )
        else:
            lines.append(
                "draft placement: locked to the last rank (physical constraint)"
            )
        has_capacity = any(item.bs_max is not None for item in self.candidates)
        if has_capacity:
            lines += [
                "",
                f"{'partition':<18} {'pp_cycle_time':>13} {'BS_max':>6} {'slot_bs':>7} {'tok/s':>8}  stages@b* (ms)",
            ]
            for item in self.candidates:
                stages = " | ".join(f"{value:.2f}" for value in item.stage_ms)
                marker = " *" if item in self.indifference_set else ""
                cadence = item.cadence_at_capacity_ms or item.bottleneck_ms
                lines.append(
                    f"{','.join(map(str, item.partition)):<18} {cadence:>13.2f} "
                    f"{item.bs_max:>6} {item.per_slot_bs:>7.1f} "
                    f"{item.throughput_tok_s:>8.1f}  {stages}{marker}"
                )
        else:
            lines += [
                "",
                f"{'partition':<18} {'draft':>5} {'bottleneck':>10} {'margin':>8}  stages (ms)",
            ]
            for item in self.candidates:
                margin = item.bottleneck_ms - self.best.bottleneck_ms
                stages = " | ".join(f"{value:.2f}" for value in item.stage_ms)
                marker = " *" if item in self.indifference_set else ""
                lines.append(
                    f"{','.join(map(str, item.partition)):<18} {item.draft_rank:>5} "
                    f"{item.bottleneck_ms:>9.3f} {margin:>7.3f}  {stages}{marker}"
                )
        lines.append("")
        lines.append("best stage decomposition (ms):")
        for stage in self.best.decomposition:
            lines.append(
                f"  rank {stage.rank}: layers={stage.layers} F={stage.fixed_ms:.3f} "
                f"layers_ms={stage.layer_ms:.3f} draft_delta={stage.draft_ms:.3f} "
                f"comm={stage.comm_ms:.3f} total={stage.total_ms:.3f}"
            )
        lines.append("")
        lines.append(
            f"predicted throughput ~= {self.predicted_throughput_tok_s:.1f} tok/s "
            f"(BS_max x accept_len / (pp_loop_size x cadence(per_slot_bs)))"
        )
        if self.best.mamba_ratio:
            lines.append(
                f"recommended mamba ratio: {self.best.mamba_ratio:g} "
                f"(K={self.best.mamba_k}, KV_tokens={self.best.kv_tokens})"
            )
        noise = self.best.noise_std_ms * self.noise_sigma
        lines.append(
            f"significance: indifference set = bottleneck within best "
            f"+ {self.noise_sigma:g} x sigma ({noise:.3f} ms); "
            f"{len(self.indifference_set)} candidate(s) inside (*)"
        )
        for warning in self.warnings:
            lines.append(f"WARNING: {warning}")
        return "\n".join(lines)


def evaluate_candidate(
    partition: Sequence[int],
    estimate: BucketEstimate,
    draft_rank: int,
    t_comm_ms: float,
    draft_relocated: bool = False,
) -> OptimizerCandidate:
    """Predict per-stage load E_r and the bottleneck for one partition.

    With ``draft_relocated=False`` the draft is on the last rank and its cost
    is already inside F_last: E_r = c*n_r + F_r + t_comm*1[r>0].  With
    relocation analysis the explicit delta D*(1[r=draft_rank] - 1[r=last])
    moves D between stages.
    """
    pp_size = len(partition)
    last_rank = pp_size - 1
    decomposition = []
    stage_ms = []
    for rank, layers in enumerate(partition):
        fixed = estimate.fixed_ms[rank]
        layer_ms = estimate.layer_cost_ms * layers
        if draft_relocated:
            draft = 0.0
            if rank == draft_rank:
                draft += estimate.draft_ms
            if rank == last_rank:
                draft -= estimate.draft_ms
        else:
            draft = 0.0
        comm = t_comm_ms if rank > 0 else 0.0
        decomposition.append(
            StageDecomposition(
                rank=rank,
                layers=layers,
                fixed_ms=fixed,
                layer_ms=layer_ms,
                draft_ms=draft,
                comm_ms=comm,
            )
        )
        stage_ms.append(fixed + layer_ms + draft + comm)
    bottleneck_rank = max(range(pp_size), key=lambda rank: stage_ms[rank])
    # Noise of the bottleneck stage prediction, synthesized from the layer
    # cost variance and the F variance: n_r^2 * var(c) + var(F_r), plus
    # var(D) when relocation moves the draft across the bottleneck stage.
    layers = partition[bottleneck_rank]
    var = layers * layers * estimate.layer_cost_var + estimate.fixed_var[bottleneck_rank]
    if draft_relocated and (
        bottleneck_rank == draft_rank or bottleneck_rank == last_rank
    ):
        var += estimate.draft_var
    noise_std = math.sqrt(max(var, 0.0))
    return OptimizerCandidate(
        partition=tuple(partition),
        draft_rank=draft_rank,
        stage_ms=tuple(stage_ms),
        bottleneck_ms=stage_ms[bottleneck_rank],
        bottleneck_rank=bottleneck_rank,
        noise_std_ms=noise_std,
        decomposition=tuple(decomposition),
    )


def kbest_partitions(
    fixed_ms: Sequence[float],
    layer_cost_ms: float,
    num_layers: int,
    min_layers: int,
    max_layers: Sequence[int],
    k: int,
) -> list[tuple[float, tuple[int, ...]]]:
    """Top-k contiguous partitions by min-max stage load.

    Prefix-sum min-max recurrence: with prefix sums over the (uniform) layer
    costs, dp[stage][end] keeps the k smallest
    values of max(dp[stage-1][start], fixed[stage-1] + prefix[end]-prefix[start]).
    Each entry records its predecessor so the partition can be rebuilt.
    """
    pp_size = len(fixed_ms)
    layer_ms = layer_cost_ms
    # dp[stage][end] = list of (objective, start, prev_index), best first.
    dp: list[list[list[tuple[float, int, int]]]] = [
        [[] for _ in range(num_layers + 1)] for _ in range(pp_size + 1)
    ]
    dp[0][0] = [(0.0, -1, -1)]

    for stage in range(1, pp_size + 1):
        remaining_stages = pp_size - stage
        min_end = stage * min_layers
        max_end = num_layers - remaining_stages * min_layers
        for end in range(min_end, max_end + 1):
            start_lo = max((stage - 1) * min_layers, end - max_layers[stage - 1])
            start_hi = end - min_layers
            merged: list[tuple[float, int, int]] = []
            fixed = fixed_ms[stage - 1]
            for start in range(start_lo, start_hi + 1):
                prev_entries = dp[stage - 1][start]
                if not prev_entries:
                    continue
                load = fixed + layer_ms * (end - start)
                for prev_index, (prev_value, _, _) in enumerate(prev_entries):
                    merged.append((max(prev_value, load), start, prev_index))
            merged.sort(key=lambda item: item[0])
            dp[stage][end] = merged[:k]

    results = []
    for objective, start, prev_index in dp[pp_size][num_layers]:
        counts = []
        stage, end, index = pp_size, start, prev_index
        counts.append(num_layers - start)
        while stage > 1:
            _, prev_start, prev_prev_index = dp[stage - 1][end][index]
            counts.append(end - prev_start)
            stage, end, index = stage - 1, prev_start, prev_prev_index
        counts.reverse()
        results.append((objective, tuple(counts)))
    return results


def evaluate_throughput(
    candidate: OptimizerCandidate,
    model: StageCostModel,
    capacity: Any,
    accept_len: float,
    t_comm_ms: float,
) -> OptimizerCandidate:
    """Attach the unified-objective numbers to one candidate.

    Unit-correct throughput: the PPM ``bs`` is the PER-SLOT microbatch size
    (the PP loop has ``pp_loop_size`` slots sharing the global running
    requests), while ``capacity.bs_max`` is the GLOBAL max running requests:

        per_slot_bs   = BS_max_global / pp_loop_size
        throughput(p) = BS_max_global x accept_len
                        / (pp_loop_size x cadence(per_slot_bs))

    The cadence at the per-slot batch size uses the bucketed cost model with
    linear interpolation (``StageCostModel.cost_at_bs``).  ``capacity`` is a
    ``capacity_model.CapacityEstimate`` (duck-typed: bs_max / k_binding_rank
    / kv_tokens).
    """
    loop_size = max(model.pp_loop_size, 1)
    per_slot_bs = capacity.bs_max / loop_size
    c_b, fixed_b = model.cost_at_bs(per_slot_bs)
    stage_ms = tuple(
        fixed_b[rank]
        + c_b * layers
        + (t_comm_ms if rank > 0 else 0.0)
        for rank, layers in enumerate(candidate.partition)
    )
    cadence = max(stage_ms)
    throughput = (
        capacity.bs_max * accept_len / (loop_size * cadence / 1000.0)
        if cadence > 0 and accept_len > 0
        else 0.0
    )
    return dataclasses.replace(
        candidate,
        bs_max=capacity.bs_max,
        per_slot_bs=per_slot_bs,
        cadence_at_capacity_ms=cadence,
        throughput_tok_s=throughput,
        capacity_binding_rank=capacity.k_binding_rank,
        kv_tokens=capacity.kv_tokens,
        mamba_k=getattr(capacity, "k", None),
        mamba_ratio=getattr(capacity, "ratio", None),
    )


def optimize(
    model: StageCostModel,
    estimate: BucketEstimate | None = None,
    t_comm_ms: float = 0.0,
    min_layers: int = 1,
    max_layers: Sequence[int] | None = None,
    k_best: int = DEFAULT_K_BEST,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    allow_draft_relocation: bool = False,
    capacity: Any = None,
) -> OptimizationResult:
    """Solve the partition problem on the fitted model.

    The draft stays on the last rank unless ``allow_draft_relocation`` opts
    into the analysis-only placement enumeration.  ``capacity`` (optional) is
    a callable partition -> ``capacity_model.CapacityEstimate``; when given,
    candidates are ranked by the unified objective
    ``throughput(p) = BS_max(p) x accept_len / cadence(p, BS_max(p))``
    instead of the raw b* bottleneck.  If the current partition lies inside
    the significance band, the recommendation is to keep it.
    """
    if estimate is None:
        estimate = model.target_bucket()
    pp_size = model.pp_size
    num_layers = model.num_layers
    last_rank = pp_size - 1
    if max_layers is None:
        max_layers = (num_layers,) * pp_size
    if len(max_layers) != pp_size:
        raise OptimizerError("max_layers must have one entry per PP rank.")
    if pp_size < 1 or num_layers < pp_size * min_layers:
        raise OptimizerError(
            f"cannot place {num_layers} layers on {pp_size} ranks with "
            f"min_layers={min_layers}."
        )

    warnings: list[str] = []
    if t_comm_ms == 0.0:
        warnings.append(
            "t_comm is 0 (not calibrated); per-boundary communication cost is "
            "ignored. Pass a measured value from a comm micro-benchmark."
        )
    if estimate.accept_len <= 0:
        warnings.append(
            "accept_len at the target bucket is 0; the throughput prediction "
            "is not meaningful."
        )

    draft_ranks = range(pp_size) if allow_draft_relocation else (last_rank,)
    seen: set[tuple[tuple[int, ...], int]] = set()
    candidates: list[OptimizerCandidate] = []
    for draft_rank in draft_ranks:
        fixed = []
        for rank in range(pp_size):
            value = estimate.fixed_ms[rank] + (t_comm_ms if rank > 0 else 0.0)
            if allow_draft_relocation:
                if rank == draft_rank:
                    value += estimate.draft_ms
                if rank == last_rank:
                    value -= estimate.draft_ms
            fixed.append(value)
        for _, partition in kbest_partitions(
            fixed, estimate.layer_cost_ms, num_layers, min_layers, max_layers, k_best
        ):
            key = (partition, draft_rank)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                evaluate_candidate(
                    partition,
                    estimate,
                    draft_rank,
                    t_comm_ms,
                    draft_relocated=allow_draft_relocation,
                )
            )
    if not candidates:
        raise OptimizerError("no valid partition satisfies the constraints.")

    # Unified objective: attach BS_max and throughput per candidate.
    compute_optimal: tuple[int, ...] | None = None
    capacity_optimal: tuple[int, ...] | None = None
    if capacity is not None:
        candidates = [
            evaluate_throughput(
                item, model, capacity(item.partition), estimate.accept_len, t_comm_ms
            )
            for item in candidates
        ]
        compute_optimal = min(
            candidates, key=lambda item: (item.bottleneck_ms, item.partition)
        ).partition
        capacity_optimal = max(
            candidates, key=lambda item: (item.bs_max or 0, item.throughput_tok_s or 0)
        ).partition
        candidates.sort(
            key=lambda item: (-(item.throughput_tok_s or 0.0), item.partition)
        )
    else:
        candidates.sort(key=lambda item: (item.bottleneck_ms, item.partition))
    candidates = candidates[:k_best]
    best = candidates[0]

    threshold = best.bottleneck_ms + noise_sigma * best.noise_std_ms
    indifference_set = [
        item for item in candidates if item.bottleneck_ms <= threshold
    ]

    # The current deployment runs the draft on the last rank; compare it
    # against the optimum with the standard (non-relocated) load model.
    current = evaluate_candidate(
        model.current_partition, estimate, last_rank, t_comm_ms
    )
    if capacity is not None:
        current = evaluate_throughput(
            current,
            model,
            capacity(model.current_partition),
            estimate.accept_len,
            t_comm_ms,
        )

    current_text = ",".join(map(str, model.current_partition))
    if capacity is not None and current.throughput_tok_s is not None:
        # Significance band on throughput: d(tput)/tput ~= d(cadence)/cadence
        # with the cadence noise synthesized from var(c) and var(F).
        best_cadence = best.cadence_at_capacity_ms or best.bottleneck_ms
        rel_band = (
            noise_sigma * best.noise_std_ms / best_cadence
            if best_cadence > 0
            else 0.0
        )
        keep_current = current.throughput_tok_s >= best.throughput_tok_s * (
            1.0 - rel_band
        )
        if keep_current:
            recommendation = (
                f"KEEP current partition {current_text} "
                f"({current.throughput_tok_s:.1f} tok/s predicted, within the "
                f"{rel_band:.1%} significance band of the best "
                f"{best.throughput_tok_s:.1f} tok/s)"
            )
        else:
            best_text = ",".join(map(str, best.partition))
            gain = (
                (best.throughput_tok_s - current.throughput_tok_s)
                / current.throughput_tok_s
                if current.throughput_tok_s > 0
                else 0.0
            )
            recommendation = (
                f"switch to {best_text} "
                f"(predicted {best.throughput_tok_s:.1f} vs "
                f"{current.throughput_tok_s:.1f} tok/s, +{gain:.1%})"
            )
            if allow_draft_relocation and best.draft_rank != last_rank:
                recommendation += (
                    f" with draft on rank {best.draft_rank} -- ANALYSIS ONLY, "
                    "requires physically relocating draft weights/KV"
                )
        throughput = best.throughput_tok_s
    else:
        keep_current = current.bottleneck_ms <= threshold
        if keep_current:
            recommendation = (
                f"KEEP current partition {current_text} "
                f"(bottleneck {current.bottleneck_ms:.3f} ms is within the "
                f"{noise_sigma:g}-sigma indifference band of the best "
                f"{best.bottleneck_ms:.3f} ms)"
            )
        else:
            best_text = ",".join(map(str, best.partition))
            recommendation = (
                f"switch to {best_text} "
                f"(predicted bottleneck {best.bottleneck_ms:.3f} ms vs current "
                f"{current.bottleneck_ms:.3f} ms)"
            )
            if allow_draft_relocation and best.draft_rank != last_rank:
                recommendation += (
                    f" with draft on rank {best.draft_rank} -- ANALYSIS ONLY, "
                    "requires physically relocating draft weights/KV"
                )
        bottleneck_s = best.bottleneck_ms / 1000.0
        throughput = (
            estimate.bs_mean * estimate.accept_len / bottleneck_s
            if bottleneck_s > 0 and estimate.accept_len > 0
            else 0.0
        )
    return OptimizationResult(
        target_bucket=estimate.bucket,
        target_bs=estimate.bs_mean,
        accept_len=estimate.accept_len,
        t_comm_ms=t_comm_ms,
        noise_sigma=noise_sigma,
        allow_draft_relocation=allow_draft_relocation,
        best=best,
        candidates=candidates,
        indifference_set=indifference_set,
        current_partition=model.current_partition,
        current=current,
        keep_current=keep_current,
        recommendation=recommendation,
        predicted_throughput_tok_s=throughput,
        compute_optimal=compute_optimal,
        capacity_optimal=capacity_optimal,
        warnings=warnings,
    )
