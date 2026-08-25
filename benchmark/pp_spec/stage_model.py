#!/usr/bin/env python3
"""Bucketed PP stage-service model.

The model intentionally has no interpolation and no throughput objective.  A
profile is keyed by an execution batch bucket and a candidate partition is
evaluated by adding its layer cost to three small fixed overheads (first,
middle, and last stage).  ``p2p_wait`` is retained as a diagnostic only.

The PPM snapshot contains a target-forward timer, a draft timer, and the
robust per-iteration ``service_ms`` aggregate.  ``service_ms`` is the right
quantity for the cycle model because CPU waits can overlap GPU work;
subtracting wall/p2p components is intentionally avoided.
"""

from __future__ import annotations

import bisect
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:
    from benchmark.pp_spec.model_layout import LayerLayout
except ImportError:  # pragma: no cover - flat script execution
    from model_layout import LayerLayout


MIN_BUCKET_SAMPLES = 3
DISPERSION_WARN_RATIO = 0.20

# ``gpu_draft_ms`` is directly observable in PPM.  The remaining fixed last
# stage residual (``T_other``) is intentionally kept as one robust term rather
# than split into noisy pseudo-measurements.  These are the operations it
# covers in the current runtime:
OTHER_OPERATION_COMPONENTS = (
    "draft input preparation and hidden-state/KV handoff",
    "target verification and accept/reject bookkeeping",
    "sampling/logits and scheduler-side output handling",
    "CUDA-graph dispatch/padding and small fixed kernel launches",
    "PP send/recv and synchronization overhead not hidden by overlap",
)


class StageModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class BucketEstimate:
    """One measured execution bucket (all times are milliseconds)."""

    bucket: int
    bs_mean: float
    samples: int
    layer_cost_ms: float
    layer_cost_var: float
    layer_cost_rank_values: tuple[float, ...]
    layer_cost_dispersion: float
    service_ms: tuple[float, ...]
    service_var: tuple[float, ...]
    fixed_ms: tuple[float, ...]
    fixed_var: tuple[float, ...]
    draft_ms: float = 0.0
    draft_var: float = 0.0
    accept_len: float = 0.0
    wait_fraction: tuple[float, ...] = ()
    # Optional typed costs.  The first implementation falls back to the
    # measured average layer cost when these are absent.
    gdn_cost_ms: float | None = None
    full_cost_ms: float | None = None
    # When an offline profile measures the complete target forward directly,
    # retain that number for auditability.  PPM-derived fits leave it unset
    # because their target timer is rank-local.
    target_total_ms: float | None = None

    @property
    def last_overhead_ms(self) -> float:
        return self.fixed_ms[-1] if self.fixed_ms else 0.0

    @property
    def other_ms(self) -> float:
        """Diagnostic ``T_other`` after the separately measured draft timer."""
        return max(self.last_overhead_ms - self.draft_ms, 0.0)

    @property
    def draft_plus_other_ms(self) -> float:
        """The fixed last-stage residual used by the cycle model."""
        return self.last_overhead_ms


@dataclass
class StageCostModel:
    num_layers: int
    pp_size: int
    current_partition: tuple[int, ...]
    buckets: dict[int, BucketEstimate]
    warnings: list[str] = field(default_factory=list)
    pp_loop_size: int = 1
    layout: LayerLayout | None = None
    capture_buckets: tuple[int, ...] = ()

    @classmethod
    def fit(
        cls,
        snapshot: Mapping[str, Any],
        current_partition: Sequence[int],
        min_samples: int = MIN_BUCKET_SAMPLES,
        *,
        layout: LayerLayout | None = None,
        capture_buckets: Sequence[int] | None = None,
    ) -> StageCostModel:
        partition = tuple(int(value) for value in current_partition)
        if min_samples <= 0:
            raise StageModelError("min_samples must be positive")
        pp_size = len(partition)
        if not partition or any(value <= 0 for value in partition):
            raise StageModelError(f"invalid current partition: {partition!r}")
        num_layers = sum(partition)
        if layout is None:
            raw_layout = snapshot.get("layout")
            raw_types = (
                raw_layout.get("layer_types") or raw_layout.get("kinds")
                if isinstance(raw_layout, Mapping)
                else None
            )
            if raw_types:
                try:
                    layout = LayerLayout.from_kinds(
                        raw_types,
                        model_type=str(raw_layout.get("model_type", "")),
                        source=str(raw_layout.get("source", "snapshot")),
                    )
                except Exception as exc:
                    raise StageModelError(
                        f"invalid snapshot layer layout: {exc}"
                    ) from exc
        if layout is not None and layout.num_layers != num_layers:
            raise StageModelError(
                f"layout has L={layout.num_layers}, partition has L={num_layers}"
            )

        ranks_data = snapshot.get("ranks")
        if not isinstance(ranks_data, Mapping) or not ranks_data:
            raise StageModelError("snapshot has no per-rank aggregates")
        rank_maps = {int(key): value for key, value in ranks_data.items()}
        invalid = [
            rank for rank, value in rank_maps.items() if not isinstance(value, Mapping)
        ]
        if invalid:
            raise StageModelError(f"invalid rank aggregate(s): {invalid}")
        missing = [rank for rank in range(pp_size) if rank not in rank_maps]
        if missing:
            raise StageModelError(
                f"snapshot is missing PP rank(s) {missing}; expected "
                f"{list(range(pp_size))}"
            )

        bucket_ids: set[int] = set()
        for rank in range(pp_size):
            for key, cell in (rank_maps[rank].get("buckets", {}) or {}).items():
                if int((cell or {}).get("count", 0)) >= min_samples:
                    bucket_ids.add(int(key))
        if not bucket_ids:
            raise StageModelError(
                "no bucket has enough work-conserving samples; increase the "
                "profile load or collection window"
            )

        warnings: list[str] = []
        buckets = {
            bucket: _fit_bucket(
                bucket,
                rank_maps,
                partition,
                pp_size,
                min_samples,
                warnings,
            )
            for bucket in sorted(bucket_ids)
        }
        loop_size = int(snapshot.get("pp_loop_size") or 0)
        if loop_size <= 0:
            raise StageModelError(
                "snapshot has no valid pp_loop_size; collect it with the current "
                "PPM producer"
            )

        if isinstance(capture_buckets, str):
            capture_values: Sequence[Any] = capture_buckets.split(",")
        else:
            capture_values = capture_buckets or ()
        resolved_capture = tuple(
            sorted({int(value) for value in capture_values if int(value) > 0})
        )
        if resolved_capture:
            missing_capture = [
                value for value in resolved_capture if value not in buckets
            ]
            if missing_capture:
                warnings.append(
                    "capture bucket(s) have no measured PPM samples: "
                    f"{missing_capture}; nearest measured bucket will be used"
                )

        return cls(
            num_layers=num_layers,
            pp_size=pp_size,
            current_partition=partition,
            buckets=buckets,
            warnings=warnings,
            pp_loop_size=loop_size,
            layout=layout,
            capture_buckets=resolved_capture,
        )

    @classmethod
    def from_bucket_profiles(
        cls,
        profiles: Mapping[int, Mapping[str, Any]],
        *,
        num_layers: int,
        pp_size: int,
        current_partition: Sequence[int],
        pp_loop_size: int = 1,
        layout: LayerLayout | None = None,
    ) -> StageCostModel:
        """Build from a compact offline profile JSON.

        Each bucket may either provide measured ``service_ms``/``fixed_ms``
        arrays, or the smaller direct profile form::

            {"target_total_ms": 12.4, "draft_ms": 2.1, "other_ms": 0.7}

        The latter implements ``t_v = target_total_ms / L`` and puts
        ``T_d + T_other`` on the last stage.  It is useful when the target and
        draft are profiled separately from the PP runtime.
        """
        partition = tuple(int(value) for value in current_partition)
        if len(partition) != int(pp_size) or any(value <= 0 for value in partition):
            raise StageModelError(f"invalid current partition: {partition!r}")
        if sum(partition) != int(num_layers):
            raise StageModelError("current partition does not sum to num_layers")
        if layout is not None and layout.num_layers != int(num_layers):
            raise StageModelError(
                "layout and offline profile have different layer counts"
            )
        buckets: dict[int, BucketEstimate] = {}
        for raw_bucket, raw in profiles.items():
            bucket = int(raw_bucket)
            service = tuple(float(v) for v in raw.get("service_ms", ()))
            fixed = tuple(float(v) for v in raw.get("fixed_ms", ()))
            if not service:
                service = tuple(float(v) for v in raw.get("stage_ms", ()))
            target_total_raw = raw.get(
                "target_total_ms",
                raw.get("target_ms", raw.get("target_total")),
            )
            target_total = (
                float(target_total_raw) if target_total_raw is not None else None
            )
            layer = float(raw.get("layer_cost_ms", raw.get("layer_ms", 0.0)))
            if target_total is None and raw.get("t_v_ms", raw.get("t_v")) is not None:
                layer = float(raw.get("t_v_ms", raw.get("t_v")))
                target_total = layer * int(num_layers)
            if target_total is not None:
                if target_total <= 0.0:
                    raise StageModelError(
                        f"profile bucket {bucket} has non-positive target_total_ms"
                    )
                # The direct offline profile is exactly the simple t_v model:
                # measure total target time once, then distribute it over L
                # target layers.  An explicitly supplied layer cost wins only
                # when the profile does not provide target_total_ms.
                if not raw.get("layer_cost_ms") and not raw.get("layer_ms"):
                    layer = target_total / int(num_layers)
            if fixed and len(fixed) != pp_size:
                raise StageModelError(
                    f"profile bucket {bucket} must contain {pp_size} fixed values"
                )
            if not service:
                if target_total is None and layer <= 0.0:
                    raise StageModelError(
                        f"profile bucket {bucket} needs service_ms/stage_ms or "
                        "a positive target_total_ms/layer_cost_ms"
                    )
                if not fixed:
                    other = float(raw.get("other_ms", 0.0))
                    draft = float(raw.get("draft_ms", 0.0))
                    if draft < 0.0 or other < 0.0:
                        raise StageModelError(
                            f"profile bucket {bucket} has negative draft/other time"
                        )
                    last_fixed = raw.get("last_fixed_ms")
                    residual = (
                        float(last_fixed) if last_fixed is not None else draft + other
                    )
                    if residual < 0.0:
                        raise StageModelError(
                            f"profile bucket {bucket} has negative last_fixed_ms"
                        )
                    fixed = tuple([0.0] * (pp_size - 1) + [residual])
                service = tuple(
                    layer * count + fixed[rank] for rank, count in enumerate(partition)
                )
            elif not fixed:
                fixed = tuple(value - layer * n for value, n in zip(service, partition))
            if len(service) != pp_size or len(fixed) != pp_size:
                raise StageModelError(
                    f"profile bucket {bucket} must contain {pp_size} stage values"
                )
            buckets[bucket] = BucketEstimate(
                bucket=bucket,
                bs_mean=float(raw.get("bs_mean", bucket)),
                samples=int(raw.get("samples", 0)),
                layer_cost_ms=layer,
                layer_cost_var=float(raw.get("layer_cost_var", 0.0)),
                layer_cost_rank_values=tuple(
                    float(value) for value in raw.get("layer_cost_rank_values", ())
                ),
                layer_cost_dispersion=float(raw.get("layer_cost_dispersion", 0.0)),
                service_ms=service,
                service_var=tuple(
                    float(v) for v in raw.get("service_var", (0.0,) * pp_size)
                ),
                fixed_ms=fixed,
                fixed_var=tuple(
                    float(v) for v in raw.get("fixed_var", (0.0,) * pp_size)
                ),
                draft_ms=float(raw.get("draft_ms", 0.0)),
                draft_var=float(raw.get("draft_var", 0.0)),
                accept_len=float(raw.get("accept_len", 0.0)),
                wait_fraction=tuple(
                    float(v) for v in raw.get("wait_fraction", (0.0,) * pp_size)
                ),
                gdn_cost_ms=(
                    float(raw["gdn_cost_ms"])
                    if raw.get("gdn_cost_ms") is not None
                    else None
                ),
                full_cost_ms=(
                    float(raw["full_cost_ms"])
                    if raw.get("full_cost_ms") is not None
                    else None
                ),
                target_total_ms=target_total,
            )
        if not buckets:
            raise StageModelError("offline profile has no buckets")
        return cls(
            num_layers=int(num_layers),
            pp_size=int(pp_size),
            current_partition=partition,
            buckets=dict(sorted(buckets.items())),
            pp_loop_size=max(int(pp_loop_size), 1),
            layout=layout,
            capture_buckets=tuple(sorted(buckets)),
        )

    def _bucket_key(self, bs: float | int) -> int:
        if not self.buckets:
            raise StageModelError("model has no fitted buckets")
        raw = int(math.ceil(float(bs)))
        choices = self.capture_buckets or tuple(sorted(self.buckets))
        index = bisect.bisect_left(choices, raw)
        if index >= len(choices):
            key = choices[-1]
        else:
            key = choices[index]
        if key in self.buckets:
            return key
        measured = tuple(sorted(self.buckets))
        measured_index = bisect.bisect_left(measured, key)
        return (
            measured[-1]
            if measured_index >= len(measured)
            else measured[measured_index]
        )

    def bucket_for_bs(self, bs: float | int) -> int:
        return self._bucket_key(bs)

    def estimate_for_bs(self, bs: float | int) -> BucketEstimate:
        return self.buckets[self._bucket_key(bs)]

    def target_bucket(self) -> BucketEstimate:
        if not self.buckets:
            raise StageModelError("model has no fitted buckets")
        return self.buckets[max(self.buckets)]

    def layer_time_ms(
        self, bs: float | int | None = None, *, bucket: int | None = None
    ) -> float:
        """Return the measured per-target-layer time at an upper bucket."""
        estimate = (
            self.buckets[int(bucket)]
            if bucket is not None
            else self.estimate_for_bs(bs if bs is not None else 1)
        )
        return estimate.layer_cost_ms

    def _layer_cost(
        self,
        estimate: BucketEstimate,
        start: int,
        end: int,
        layout: LayerLayout | None,
    ) -> float:
        if (
            layout is not None
            and estimate.gdn_cost_ms is not None
            and estimate.full_cost_ms is not None
        ):
            gdn, full = layout.count_range(start, end)
            return gdn * estimate.gdn_cost_ms + full * estimate.full_cost_ms
        return (end - start) * estimate.layer_cost_ms

    def predict_stages(
        self,
        partition: Sequence[int],
        *,
        bs: float | int | None = None,
        bucket: int | None = None,
        estimate: BucketEstimate | None = None,
        t_comm_ms: float = 0.0,
        stage_comm_ms: Sequence[float] | None = None,
        layout: LayerLayout | None = None,
    ) -> tuple[float, ...]:
        """Predict stage service times for one candidate partition."""
        counts = tuple(int(value) for value in partition)
        if len(counts) != self.pp_size or sum(counts) != self.num_layers:
            raise StageModelError(
                f"partition {counts!r} does not match P={self.pp_size}, L={self.num_layers}"
            )
        active_estimate = estimate or (
            self.buckets[int(bucket)]
            if bucket is not None
            else self.estimate_for_bs(bs if bs is not None else 1)
        )
        active_layout = layout or self.layout
        if stage_comm_ms is None:
            comm_costs = tuple(
                0.0 if rank == 0 else float(t_comm_ms) for rank in range(self.pp_size)
            )
        else:
            comm_costs = tuple(float(value) for value in stage_comm_ms)
            if len(comm_costs) == self.pp_size - 1:
                comm_costs = (0.0, *comm_costs)
            if len(comm_costs) != self.pp_size or any(
                value < 0.0 for value in comm_costs
            ):
                raise StageModelError(
                    "stage_comm_ms must contain PP or PP-1 non-negative values"
                )
        ranges: list[tuple[int, int]] = []
        start = 0
        for count in counts:
            ranges.append((start, start + count))
            start += count

        if self.pp_size > 2 and active_estimate.fixed_ms[1:-1]:
            middle = float(statistics.median(active_estimate.fixed_ms[1:-1]))
        else:
            middle = active_estimate.fixed_ms[0] if active_estimate.fixed_ms else 0.0
        values: list[float] = []
        for rank, (start, end) in enumerate(ranges):
            layer_ms = self._layer_cost(active_estimate, start, end, active_layout)
            if rank == 0:
                fixed = active_estimate.fixed_ms[0]
            elif rank == self.pp_size - 1:
                fixed = active_estimate.fixed_ms[-1]
            else:
                fixed = middle
            values.append(layer_ms + fixed + comm_costs[rank])
        return tuple(values)

    def cycle_time_ms(
        self,
        partition: Sequence[int],
        *,
        bs: float | int | None = None,
        bucket: int | None = None,
        estimate: BucketEstimate | None = None,
        t_comm_ms: float = 0.0,
        stage_comm_ms: Sequence[float] | None = None,
        layout: LayerLayout | None = None,
    ) -> float:
        """Return ``max_r stage_service_r`` for one execution bucket."""
        return max(
            self.predict_stages(
                partition,
                bs=bs,
                bucket=bucket,
                estimate=estimate,
                t_comm_ms=t_comm_ms,
                stage_comm_ms=stage_comm_ms,
                layout=layout,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "pp_size": self.pp_size,
            "pp_loop_size": self.pp_loop_size,
            "current_partition": list(self.current_partition),
            "capture_buckets": list(self.capture_buckets),
            "other_operation_components": list(OTHER_OPERATION_COMPONENTS),
            "layout": self.layout.to_dict() if self.layout is not None else None,
            "buckets": {
                str(bucket): {
                    "bucket": estimate.bucket,
                    "bs_mean": estimate.bs_mean,
                    "samples": estimate.samples,
                    "layer_cost_ms": estimate.layer_cost_ms,
                    "layer_cost_var": estimate.layer_cost_var,
                    "layer_cost_rank_values": list(estimate.layer_cost_rank_values),
                    "layer_cost_dispersion": estimate.layer_cost_dispersion,
                    "service_ms": list(estimate.service_ms),
                    "service_var": list(estimate.service_var),
                    "fixed_ms": list(estimate.fixed_ms),
                    "fixed_var": list(estimate.fixed_var),
                    "draft_ms": estimate.draft_ms,
                    "other_ms": estimate.other_ms,
                    "draft_plus_other_ms": estimate.draft_plus_other_ms,
                    "draft_var": estimate.draft_var,
                    "accept_len": estimate.accept_len,
                    "wait_fraction": list(estimate.wait_fraction),
                    "gdn_cost_ms": estimate.gdn_cost_ms,
                    "full_cost_ms": estimate.full_cost_ms,
                    "target_total_ms": estimate.target_total_ms,
                }
                for bucket, estimate in sorted(self.buckets.items())
            },
            "warnings": list(self.warnings),
        }


def _cell_stats(cell: Mapping[str, Any], name: str) -> tuple[float, float, int]:
    stats = cell.get(name) or {}
    count = int(stats.get("count", 0))
    mean = float(stats.get("mean", 0.0))
    variance = float(stats.get("var", 0.0))
    return mean, variance / count if count > 0 else 0.0, count


def _bucket_cell(rank_map: Mapping[str, Any], bucket: int) -> Mapping[str, Any] | None:
    buckets = rank_map.get("buckets", {}) or {}
    cell = buckets.get(str(bucket))
    if cell is None:
        cell = buckets.get(bucket)
    return cell if isinstance(cell, Mapping) else None


def _fit_bucket(
    bucket: int,
    rank_maps: Mapping[int, Mapping[str, Any]],
    partition: tuple[int, ...],
    pp_size: int,
    min_samples: int,
    warnings: list[str],
) -> BucketEstimate:
    target_per_layer: list[float] = []
    target_var: list[float] = []
    service = [0.0] * pp_size
    service_var = [0.0] * pp_size
    wait_fraction = [0.0] * pp_size
    bs_values: list[float] = []
    have = [False] * pp_size
    draft_ms = draft_var = accept_len = 0.0

    for rank in range(pp_size):
        cell = _bucket_cell(rank_maps[rank], bucket)
        if not isinstance(cell, Mapping) or int(cell.get("count", 0)) < min_samples:
            continue
        have[rank] = True
        target, target_se, _ = _cell_stats(cell, "gpu_target_ms")
        measured_service, service_se, _ = _cell_stats(cell, "service_ms")
        if not (cell.get("service_ms") or {}).get("count", 0):
            draft_fallback, _, _ = _cell_stats(cell, "gpu_draft_ms")
            measured_service = target + draft_fallback
        wall, _, _ = _cell_stats(cell, "wall_ms")
        p2p, _, _ = _cell_stats(cell, "p2p_wait_ms")
        bs, _, _ = _cell_stats(cell, "bs")
        if bs <= 0:
            bs = float(bucket)
        if target <= 0.0 and measured_service > 0.0:
            # Older PPM snapshots may not have the CUDA device timer.  A
            # service-only fallback is less precise (and is reported as a
            # warning) but avoids silently fitting a zero layer cost.
            draft_hint, _, _ = _cell_stats(cell, "gpu_draft_ms")
            target = max(
                measured_service - (draft_hint if rank == pp_size - 1 else 0.0), 0.0
            )
            warnings.append(
                f"bucket {bucket} rank {rank}: gpu_target_ms is missing; "
                "estimated target time from service_ms"
            )
        bs_values.append(bs)
        n_layers = partition[rank]
        target_per_layer.append(target / n_layers)
        target_var.append(target_se / (n_layers * n_layers))
        service[rank] = measured_service
        service_var[rank] = service_se
        wait_fraction[rank] = p2p / wall if wall > 0 else 0.0
        if rank == pp_size - 1:
            draft_ms, draft_var, _ = _cell_stats(cell, "gpu_draft_ms")
            accept_len, _, _ = _cell_stats(cell, "accept_len")

    if not target_per_layer:
        raise StageModelError(f"bucket {bucket} has no usable rank samples")

    layer_ms = float(statistics.median(target_per_layer))
    if layer_ms <= 0.0:
        raise StageModelError(
            f"bucket {bucket} has no positive target/service timing; "
            "enable the PPM device timer or provide an offline profile"
        )
    layer_var = sum(target_var) / len(target_var)
    if len(target_per_layer) > 1:
        dispersion = (
            statistics.pstdev(target_per_layer) / layer_ms if layer_ms > 0 else 0.0
        )
    else:
        dispersion = 0.0
        warnings.append(f"bucket {bucket}: only one rank contributed a layer estimate")
    if dispersion > DISPERSION_WARN_RATIO:
        warnings.append(
            f"bucket {bucket}: per-layer estimates differ by {dispersion:.1%}; "
            "average layer cost may be inaccurate"
        )

    present_service = [service[rank] for rank in range(pp_size) if have[rank]]
    present_var = [service_var[rank] for rank in range(pp_size) if have[rank]]
    if not present_service:
        raise StageModelError(f"bucket {bucket} has no service samples")
    for rank in range(pp_size):
        if not have[rank]:
            service[rank] = float(statistics.median(present_service))
            service_var[rank] = float(statistics.median(present_var))
            warnings.append(
                f"bucket {bucket}: rank {rank} missing; used median service time"
            )

    fixed: list[float] = []
    fixed_var: list[float] = []
    for rank, n_layers in enumerate(partition):
        raw = service[rank] - layer_ms * n_layers
        variance = max(service_var[rank] + n_layers * n_layers * layer_var, 0.0)
        if raw < 0.0:
            warnings.append(
                f"bucket {bucket} rank {rank}: negative fixed residual "
                f"{raw:.3f} ms; clamped to zero"
            )
            raw = 0.0
        fixed.append(raw)
        fixed_var.append(variance)

    sample_count = sum(
        int((_bucket_cell(rank_maps[rank], bucket) or {}).get("count", 0))
        for rank in range(pp_size)
    )
    return BucketEstimate(
        bucket=bucket,
        bs_mean=sum(bs_values) / len(bs_values) if bs_values else float(bucket),
        samples=sample_count,
        layer_cost_ms=layer_ms,
        layer_cost_var=layer_var,
        layer_cost_rank_values=tuple(target_per_layer),
        layer_cost_dispersion=dispersion,
        service_ms=tuple(service),
        service_var=tuple(service_var),
        fixed_ms=tuple(fixed),
        fixed_var=tuple(fixed_var),
        draft_ms=draft_ms,
        draft_var=draft_var,
        accept_len=accept_len,
        wait_fraction=tuple(wait_fraction),
    )
