#!/usr/bin/env python3
"""Single-bucket stage-cost model for offline adaptive PP analysis."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    from benchmark.pp_spec.model_layout import LayerLayout
except ImportError:  # pragma: no cover - flat script execution
    from model_layout import LayerLayout


class StageModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class BucketEstimate:
    """Intrinsic stage-cost inputs for one exact execution bucket."""

    bucket: int
    layer_cost_ms: float
    fixed_ms: tuple[float, ...]
    gdn_cost_ms: float
    full_cost_ms: float


@dataclass
class StageCostModel:
    """Predict stage times from one offline baseline profile."""

    num_layers: int
    pp_size: int
    baseline_partition: tuple[int, ...]
    buckets: dict[int, BucketEstimate]
    layout: LayerLayout | None = None

    _REQUIRED_PROFILE_FIELDS = frozenset(
        {
            "layer_cost_ms",
            "gdn_cost_ms",
            "full_cost_ms",
            "fixed_ms",
        }
    )

    @classmethod
    def from_bucket_profiles(
        cls,
        profiles: Mapping[int, Mapping[str, Any]],
        *,
        num_layers: int,
        pp_size: int,
        baseline_partition: Sequence[int],
        layout: LayerLayout | None = None,
    ) -> StageCostModel:
        """Build a stage model from one or more profiled buckets."""

        partition = tuple(int(value) for value in baseline_partition)
        if pp_size <= 0 or num_layers <= 0:
            raise StageModelError("num_layers and pp_size must be positive")
        if len(partition) != pp_size or any(value <= 0 for value in partition):
            raise StageModelError(f"invalid baseline partition: {partition!r}")
        if sum(partition) != num_layers:
            raise StageModelError("baseline partition does not sum to num_layers")
        if layout is not None and layout.num_layers != num_layers:
            raise StageModelError(
                "layout and offline profile have different layer counts"
            )
        if not profiles:
            raise StageModelError("offline stage model needs at least one bucket")
        estimates: dict[int, BucketEstimate] = {}
        for raw_bucket, raw in profiles.items():
            if not isinstance(raw, Mapping):
                raise StageModelError("offline bucket profile must be a mapping")
            bucket = _positive_integer(raw_bucket, "profile bucket")
            missing = sorted(cls._REQUIRED_PROFILE_FIELDS - set(raw))
            unknown = sorted(set(raw) - cls._REQUIRED_PROFILE_FIELDS)
            if missing:
                raise StageModelError(
                    f"profile bucket {bucket} is missing required fields {missing}"
                )
            if unknown:
                raise StageModelError(
                    f"profile bucket {bucket} has unsupported fields {unknown}"
                )

            fixed = _float_tuple(raw["fixed_ms"], "fixed_ms", pp_size)
            if any(value < 0.0 for value in fixed):
                raise StageModelError("fixed_ms values must be non-negative")

            layer_cost = _finite_float(raw["layer_cost_ms"], "layer_cost_ms")
            gdn_cost = _finite_float(raw["gdn_cost_ms"], "gdn_cost_ms")
            full_cost = _finite_float(raw["full_cost_ms"], "full_cost_ms")
            if layer_cost <= 0.0 or gdn_cost <= 0.0 or full_cost <= 0.0:
                raise StageModelError("all target-layer costs must be positive")
            estimates[bucket] = BucketEstimate(
                bucket=bucket,
                layer_cost_ms=layer_cost,
                fixed_ms=fixed,
                gdn_cost_ms=gdn_cost,
                full_cost_ms=full_cost,
            )
        return cls(
            num_layers=num_layers,
            pp_size=pp_size,
            baseline_partition=partition,
            buckets=estimates,
            layout=layout,
        )

    def _exact_bucket(self, value: float | int) -> int:
        bucket = _positive_integer(value, "execution bucket")
        if bucket not in self.buckets:
            measured = next(iter(self.buckets))
            raise StageModelError(
                f"execution bucket {bucket} was not profiled; expected {measured}"
            )
        return bucket

    def bucket_for_bs(self, bs: float | int) -> int:
        return self._exact_bucket(bs)

    def estimate_for_bs(self, bs: float | int) -> BucketEstimate:
        return self.buckets[self._exact_bucket(bs)]

    def target_bucket(self) -> BucketEstimate:
        return next(iter(self.buckets.values()))

    def _layer_cost(
        self,
        estimate: BucketEstimate,
        start: int,
        end: int,
        layout: LayerLayout | None,
    ) -> float:
        if layout is not None:
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
        stage_comm_ms: Sequence[float] | None = None,
        layout: LayerLayout | None = None,
    ) -> tuple[float, ...]:
        """Predict intrinsic stage work plus an explicit communication floor."""

        counts = tuple(int(value) for value in partition)
        if (
            len(counts) != self.pp_size
            or any(value <= 0 for value in counts)
            or sum(counts) != self.num_layers
        ):
            raise StageModelError(
                f"partition {counts!r} does not match "
                f"P={self.pp_size}, L={self.num_layers}"
            )
        if estimate is not None:
            if estimate.bucket not in self.buckets:
                raise StageModelError(
                    f"estimate bucket {estimate.bucket} does not belong to this model"
                )
            active_estimate = estimate
        else:
            if bs is not None and bucket is not None:
                raise StageModelError("pass either bs or bucket, not both")
            requested = bucket if bucket is not None else bs
            active_estimate = (
                self.target_bucket()
                if requested is None
                else self.estimate_for_bs(requested)
            )

        active_layout = layout or self.layout
        if active_layout is not None and active_layout.num_layers != self.num_layers:
            raise StageModelError("layout and stage model have different layer counts")
        comm_costs = _stage_comm_costs(stage_comm_ms, self.pp_size)

        ranges: list[tuple[int, int]] = []
        start = 0
        for count in counts:
            ranges.append((start, start + count))
            start += count

        if self.pp_size > 2:
            middle_fixed = float(statistics.median(active_estimate.fixed_ms[1:-1]))
        else:
            middle_fixed = active_estimate.fixed_ms[0]
        values: list[float] = []
        for rank, (start, end) in enumerate(ranges):
            layer_ms = self._layer_cost(active_estimate, start, end, active_layout)
            if rank == 0:
                fixed_ms = active_estimate.fixed_ms[0]
            elif rank == self.pp_size - 1:
                fixed_ms = active_estimate.fixed_ms[-1]
            else:
                fixed_ms = middle_fixed
            values.append(layer_ms + fixed_ms + comm_costs[rank])
        return tuple(values)

    def cycle_time_ms(
        self,
        partition: Sequence[int],
        *,
        bs: float | int | None = None,
        bucket: int | None = None,
        estimate: BucketEstimate | None = None,
        stage_comm_ms: Sequence[float] | None = None,
        layout: LayerLayout | None = None,
    ) -> float:
        return max(
            self.predict_stages(
                partition,
                bs=bs,
                bucket=bucket,
                estimate=estimate,
                stage_comm_ms=stage_comm_ms,
                layout=layout,
            )
        )

def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StageModelError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise StageModelError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, name: str) -> int:
    numeric = _finite_float(value, name)
    if not numeric.is_integer() or numeric <= 0.0:
        raise StageModelError(f"{name} must be a positive integer")
    return int(numeric)


def _float_tuple(value: Any, name: str, expected: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StageModelError(f"{name} must be a sequence")
    result = tuple(_finite_float(item, name) for item in value)
    if len(result) != expected:
        raise StageModelError(f"{name} must contain {expected} values")
    return result


def _stage_comm_costs(
    values: Sequence[float] | None, pp_size: int
) -> tuple[float, ...]:
    if values is None:
        return (0.0,) * pp_size
    costs = tuple(_finite_float(value, "stage_comm_ms") for value in values)
    if len(costs) == pp_size - 1:
        costs = (0.0, *costs)
    if len(costs) != pp_size or any(value < 0.0 for value in costs):
        raise StageModelError(
            "stage_comm_ms must contain PP or PP-1 non-negative values"
        )
    return costs
