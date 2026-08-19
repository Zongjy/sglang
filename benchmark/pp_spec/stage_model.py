#!/usr/bin/env python3
"""Fit the PP stage cost model from a PPM snapshot.

Cost model (all durations ms), for PP rank r and bs bucket b under the
current partition where rank r holds ``n_r`` layers:

- per-layer target cost ``c(b)``: cross-rank median of ``gpu_target / n_r``
- service time          ``S_r(b)``: snapshot ``service_ms`` mean, i.e. the
  per-iteration ``max(gpu_target + gpu_draft, wall - p2p_wait)``
- fixed stage overhead  ``F_r(b) = S_r(b) - c(b) * n_r`` -- everything on the
  rank besides its layers (on the last rank: draft + lm_head + sampling +
  CPU, bundled, not split).  Negative values inside the sigma tolerance are
  clamped to 0; beyond tolerance a warning is emitted.
- draft block cost      ``D(b)``: last-rank ``gpu_draft`` mean (reporting and
  draft-relocation analysis only; it is already inside ``F_last``)

The old ``f_r = wall - p2p_wait - gpu_target - gpu_draft`` identity was
removed: CPU blocking waits overlap the async GPU forward in a concurrent
pipeline, so the subtraction produced negative fixed overheads.  ``p2p_wait``
itself mixes pipeline imbalance with transfer time and never enters the
model.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# Cross-rank relative spread (std / mean) above which the uniform per-layer
# cost assumption is flagged.
DISPERSION_WARN_RATIO = 0.15
# Minimum samples in a (rank, bucket) cell before it contributes to the fit.
MIN_BUCKET_SAMPLES = 3
# F_r may be clamped from a negative value up to this many combined sigmas.
CLAMP_SIGMA_TOLERANCE = 2.0
# Last-rank wait fraction above which the F estimate is flagged as distorted.
LAST_RANK_WAIT_FRACTION_WARN = 0.15


class StageModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class BucketEstimate:
    """Fitted quantities for one bs bucket b."""

    bucket: int
    bs_mean: float
    samples: int
    layer_cost_ms: float  # c(b): cross-rank median of gpu_target / n_r
    layer_cost_var: float  # within-rank SE^2 + cross-rank variance
    layer_cost_rank_values: tuple[float, ...]
    layer_cost_dispersion: float  # cross-rank std / mean
    service_ms: tuple[float, ...]  # S_r(b) per rank
    service_var: tuple[float, ...]  # variance of each S_r(b) mean
    fixed_ms: tuple[float, ...]  # F_r(b) = S_r(b) - c(b) * n_r, clamped >= 0
    fixed_var: tuple[float, ...]  # variance of each F_r(b)
    draft_ms: float  # D(b): last-rank gpu_draft (reporting only)
    draft_var: float
    accept_len: float
    wait_fraction: tuple[float, ...]  # mean(p2p_wait) / mean(wall) per rank


@dataclass
class StageCostModel:
    num_layers: int
    pp_size: int
    current_partition: tuple[int, ...]
    buckets: dict[int, BucketEstimate]
    warnings: list[str] = field(default_factory=list)
    # PP microbatch slots in flight; bucket bs is the PER-SLOT microbatch
    # size, so the global running batch ~= bs * pp_loop_size.
    pp_loop_size: int = 1

    @classmethod
    def fit(
        cls,
        snapshot: Mapping[str, Any],
        current_partition: Sequence[int],
        min_samples: int = MIN_BUCKET_SAMPLES,
    ) -> "StageCostModel":
        """Fit c(b), F_r(b) and D(b) from a ppm_consumer snapshot dict."""
        partition = tuple(int(item) for item in current_partition)
        pp_size = len(partition)
        if pp_size == 0 or any(item <= 0 for item in partition):
            raise StageModelError(f"invalid current partition: {partition!r}")
        ranks_data = snapshot.get("ranks")
        if not isinstance(ranks_data, Mapping) or not ranks_data:
            raise StageModelError("snapshot has no per-rank aggregates.")

        rank_maps: dict[int, Mapping[str, Any]] = {}
        for key, value in ranks_data.items():
            rank_maps[int(key)] = value
        missing = [rank for rank in range(pp_size) if rank not in rank_maps]
        warnings: list[str] = []
        if missing:
            raise StageModelError(
                f"snapshot is missing PP rank(s) {missing}; expected "
                f"{list(range(pp_size))} from the current partition."
            )

        # Union of buckets with enough samples on at least one rank.
        bucket_ids: set[int] = set()
        for rank in range(pp_size):
            for bucket_key, cell in rank_maps[rank].get("buckets", {}).items():
                if cell.get("count", 0) >= min_samples:
                    bucket_ids.add(int(bucket_key))
        if not bucket_ids:
            raise StageModelError(
                "no work-conserving (rank, bucket) cell has enough samples; "
                "collect longer or under higher offered load."
            )

        buckets: dict[int, BucketEstimate] = {}
        for bucket in sorted(bucket_ids):
            buckets[bucket] = _fit_bucket(
                bucket, rank_maps, partition, pp_size, min_samples, warnings
            )
        pp_loop_size = int(snapshot.get("pp_loop_size") or 0)
        if pp_loop_size <= 0:
            pp_loop_size = pp_size
            warnings.append(
                "snapshot has no pp_loop_size (PPM schema v1); assuming "
                "pp_loop_size == pp_size. bs is the per-slot microbatch size "
                "and running_bs is unknown."
            )
        return cls(
            num_layers=sum(partition),
            pp_size=pp_size,
            current_partition=partition,
            buckets=buckets,
            warnings=warnings,
            pp_loop_size=pp_loop_size,
        )

    def target_bucket(self) -> BucketEstimate:
        """The saturation working point b*: the highest fitted (non-empty) bs
        bucket -- not the bucket with the most samples."""
        if not self.buckets:
            raise StageModelError("model has no fitted buckets.")
        return self.buckets[max(self.buckets)]

    def cost_at_bs(self, bs: float) -> tuple[float, tuple[float, ...]]:
        """Layer cost c and fixed overheads F_r interpolated at batch ``bs``.

        Linear interpolation in bs between the fitted bucket centers
        (bs_mean), clamped to the nearest bucket outside the range.  Used to
        evaluate the cadence of a candidate partition at its capacity
        BS_max, which generally falls between bucket centers.
        """
        if not self.buckets:
            raise StageModelError("model has no fitted buckets.")
        ordered = sorted(self.buckets.values(), key=lambda item: item.bs_mean)
        lo = ordered[0]
        hi = ordered[-1]
        if bs <= lo.bs_mean:
            return lo.layer_cost_ms, lo.fixed_ms
        if bs >= hi.bs_mean:
            return hi.layer_cost_ms, hi.fixed_ms
        for left, right in zip(ordered, ordered[1:]):
            if left.bs_mean <= bs <= right.bs_mean:
                span = right.bs_mean - left.bs_mean
                frac = (bs - left.bs_mean) / span if span > 0 else 0.0
                c = left.layer_cost_ms + frac * (
                    right.layer_cost_ms - left.layer_cost_ms
                )
                fixed = tuple(
                    f_lo + frac * (f_hi - f_lo)
                    for f_lo, f_hi in zip(left.fixed_ms, right.fixed_ms)
                )
                return c, fixed
        return hi.layer_cost_ms, hi.fixed_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "pp_size": self.pp_size,
            "pp_loop_size": self.pp_loop_size,
            "current_partition": list(self.current_partition),
            "buckets": {
                str(bucket): {
                    "bucket": est.bucket,
                    "bs_mean": est.bs_mean,
                    "samples": est.samples,
                    "layer_cost_ms": est.layer_cost_ms,
                    "layer_cost_var": est.layer_cost_var,
                    "layer_cost_rank_values": list(est.layer_cost_rank_values),
                    "layer_cost_dispersion": est.layer_cost_dispersion,
                    "service_ms": list(est.service_ms),
                    "service_var": list(est.service_var),
                    "fixed_ms": list(est.fixed_ms),
                    "fixed_var": list(est.fixed_var),
                    "draft_ms": est.draft_ms,
                    "draft_var": est.draft_var,
                    "accept_len": est.accept_len,
                    "wait_fraction": list(est.wait_fraction),
                }
                for bucket, est in sorted(self.buckets.items())
            },
            "warnings": list(self.warnings),
        }


def _cell_stats(cell: Mapping[str, Any], field_name: str) -> tuple[float, float, int]:
    """(mean, variance of the mean, count) for one Welford aggregate."""
    stats = cell.get(field_name) or {}
    count = int(stats.get("count", 0))
    mean = float(stats.get("mean", 0.0))
    var = float(stats.get("var", 0.0))
    se_sq = var / count if count > 0 else 0.0
    return mean, se_sq, count


def _fit_bucket(
    bucket: int,
    rank_maps: Mapping[int, Mapping[str, Any]],
    partition: tuple[int, ...],
    pp_size: int,
    min_samples: int,
    warnings: list[str],
) -> BucketEstimate:
    last_rank = pp_size - 1
    layer_estimates: list[float] = []
    layer_se_sq: list[float] = []
    service_ms: list[float] = []
    service_var: list[float] = []
    wait_fraction: list[float] = []
    have_cell: list[bool] = []
    bs_means: list[float] = []
    total_samples = 0
    draft_ms = 0.0
    draft_var = 0.0
    accept_len = 0.0

    for rank in range(pp_size):
        cell = rank_maps[rank].get("buckets", {}).get(str(bucket))
        if cell is None or cell.get("count", 0) < min_samples:
            service_ms.append(0.0)
            service_var.append(math.inf)
            wait_fraction.append(0.0)
            have_cell.append(False)
            continue
        have_cell.append(True)
        count = int(cell["count"])
        total_samples += count
        target, target_se, _ = _cell_stats(cell, "gpu_target_ms")
        service, service_se, _ = _cell_stats(cell, "service_ms")
        wall, _, _ = _cell_stats(cell, "wall_ms")
        p2p, _, _ = _cell_stats(cell, "p2p_wait_ms")
        bs_mean, _, _ = _cell_stats(cell, "bs")
        bs_means.append(bs_mean)

        n_r = partition[rank]
        layer_estimates.append(target / n_r)
        layer_se_sq.append(target_se / (n_r * n_r))

        service_ms.append(service)
        service_var.append(service_se)
        wait_fraction.append(p2p / wall if wall > 0 else 0.0)

        if rank == last_rank:
            draft_ms, draft_var, _ = _cell_stats(cell, "gpu_draft_ms")
            accept_len, _, _ = _cell_stats(cell, "accept_len")

    if not layer_estimates:
        raise StageModelError(f"bucket {bucket} has no usable rank cells.")

    c_median = float(statistics.median(layer_estimates))
    within_var = sum(layer_se_sq) / len(layer_se_sq)
    if len(layer_estimates) > 1:
        cross_var = float(statistics.variance(layer_estimates))
        mean_estimate = sum(layer_estimates) / len(layer_estimates)
        dispersion = (
            math.sqrt(cross_var) / mean_estimate if mean_estimate > 0 else 0.0
        )
    else:
        cross_var = 0.0
        dispersion = 0.0
        warnings.append(
            f"bucket {bucket}: only one rank contributed a layer-cost "
            "estimate; cross-rank dispersion is unknown."
        )
    if dispersion > DISPERSION_WARN_RATIO:
        warnings.append(
            f"bucket {bucket}: per-layer cost dispersion across ranks is "
            f"{dispersion:.1%} (estimates "
            f"{[round(v, 4) for v in layer_estimates]} ms); the uniform "
            "per-layer cost assumption is suspect."
        )
    layer_cost_var = within_var + cross_var

    missing = [r for r in range(pp_size) if not have_cell[r]]
    if missing:
        warnings.append(
            f"bucket {bucket}: rank(s) {missing} have too few samples; "
            "their service time falls back to the median of fitted ranks."
        )
        fitted = [service_ms[r] for r in range(pp_size) if have_cell[r]]
        fallback = float(statistics.median(fitted)) if fitted else 0.0
        fallback_var = (
            float(
                statistics.median(
                    service_var[r] for r in range(pp_size) if have_cell[r]
                )
            )
            if fitted
            else 0.0
        )
        for r in missing:
            service_ms[r] = fallback
            service_var[r] = fallback_var

    # F_r = S_r - c * n_r, clamped to 0 within the sigma tolerance.
    fixed_ms: list[float] = []
    fixed_var: list[float] = []
    for rank in range(pp_size):
        n_r = partition[rank]
        raw = service_ms[rank] - c_median * n_r
        var = service_var[rank] + n_r * n_r * layer_cost_var
        sigma = math.sqrt(max(var, 0.0))
        if raw < 0.0:
            if -raw <= CLAMP_SIGMA_TOLERANCE * sigma:
                raw = 0.0
            else:
                warnings.append(
                    f"bucket {bucket} rank {rank}: F = S - c*n is "
                    f"{raw:.3f} ms, beyond the {CLAMP_SIGMA_TOLERANCE:g}-sigma "
                    f"clamp tolerance (sigma {sigma:.3f} ms); the service "
                    "time or layer cost estimate is inconsistent."
                )
                raw = 0.0
        fixed_ms.append(raw)
        fixed_var.append(var)

    if wait_fraction[last_rank] > LAST_RANK_WAIT_FRACTION_WARN:
        warnings.append(
            f"bucket {bucket}: last-rank wait_fraction is "
            f"{wait_fraction[last_rank]:.1%}; the last rank is probably not "
            "the bottleneck or the pipeline is severely imbalanced, so the F "
            "estimate may be distorted."
        )

    return BucketEstimate(
        bucket=bucket,
        bs_mean=sum(bs_means) / len(bs_means) if bs_means else float(bucket),
        samples=total_samples,
        layer_cost_ms=c_median,
        layer_cost_var=layer_cost_var,
        layer_cost_rank_values=tuple(layer_estimates),
        layer_cost_dispersion=dispersion,
        service_ms=tuple(service_ms),
        service_var=tuple(service_var),
        fixed_ms=tuple(fixed_ms),
        fixed_var=tuple(fixed_var),
        draft_ms=draft_ms,
        draft_var=draft_var,
        accept_len=accept_len,
        wait_fraction=tuple(wait_fraction),
    )
