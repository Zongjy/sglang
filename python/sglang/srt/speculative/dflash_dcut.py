from __future__ import annotations

import hashlib
import json
import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal, Optional, Sequence, Union

import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    scatter_compact_to_strided_into,
)
from sglang.srt.distributed import get_attn_tp_group, get_pp_group, get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_dp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    build_capture_verify_lens,
    round_up_grid,
)

logger = logging.getLogger(__name__)

DFlashDcutValue = Union[float, Literal["auto"]]

_AUTO_RATIOS = (0.25, 0.5, 0.75, 1.0)

_OFFLINE_PROFILE_WARMUPS = 3
_OFFLINE_PROFILE_STEPS = 5
_OFFLINE_PROFILE_SEQ_LEN = 2048
_OFFLINE_PROFILE_MAX_BS = 128


def dflash_dcut_enabled(value: DFlashDcutValue) -> bool:
    return value == "auto" or (not isinstance(value, str) and float(value) != 0.0)


def get_dflash_dcut_keep_count(*, bs: int, block_size: int, ratio: float) -> int:
    """Number of non-anchor query tokens kept by the D-Cut ratio.

    The ratio applies to all target-forward queries.  One query per request is
    the mandatory anchor, hence ``ceil(bs * block_size * ratio) - bs`` drafts.
    """
    if bs < 0:
        raise ValueError(f"bs must be non-negative, got {bs}.")
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}.")
    return min(
        bs * (block_size - 1),
        max(0, math.ceil(bs * block_size * ratio) - bs),
    )


def score_dcut_candidates(
    *,
    expected: torch.Tensor,
    costs: torch.Tensor,
) -> torch.Tensor:
    """Paper D-Cut selector utility: expected advance divided by cost."""
    if expected.shape != costs.shape:
        raise ValueError(
            f"expected and costs must share a shape, got {tuple(expected.shape)} "
            f"and {tuple(costs.shape)}."
        )
    if expected.numel() == 0:
        raise ValueError("expected/costs must be non-empty.")
    return expected.to(dtype=torch.float32) / costs.to(dtype=torch.float32).clamp_min(
        torch.finfo(torch.float32).eps
    )


def pp_pipeline_cycle_cost(
    stage_costs: torch.Tensor, microbatch_count: int
) -> torch.Tensor:
    """Forward-only PP makespan for one candidate ratio.

    A pipeline with ``M`` in-flight microbatches pays one fill/drain sum of
    stage service times and ``M-1`` steady-state cycles at the bottleneck.
    ``stage_costs`` is [PP, candidates].
    """
    if stage_costs.ndim != 2 or stage_costs.shape[0] == 0:
        raise ValueError("stage_costs must have shape [pp_size, candidates].")
    if microbatch_count < 1:
        raise ValueError(f"microbatch_count must be positive, got {microbatch_count}.")
    return stage_costs.sum(dim=0) + (int(microbatch_count) - 1) * stage_costs.amax(
        dim=0
    )


def pp_pipeline_flowshop_makespan(
    jobs: Sequence[Sequence[float]],
    transfer_costs: Optional[Sequence[float]] = None,
    job_transfer_costs: Optional[Sequence[Sequence[float]]] = None,
) -> float:
    """Simulate ordered PP microbatches through a flow-shop pipeline.

    ``jobs`` contains one per-microbatch stage-service vector.  A job can enter
    stage ``i`` only after both stage ``i`` is free and the previous stage has
    finished its work plus the adjacent transfer.  This models the marginal
    effect of changing one microbatch while preserving the other in-flight
    jobs.
    """
    if not jobs:
        return 0.0
    stage_count = len(jobs[0])
    if stage_count == 0:
        raise ValueError("PP flow-shop jobs must contain at least one stage.")
    if any(len(job) != stage_count for job in jobs):
        raise ValueError("PP flow-shop jobs must have equal stage dimensions.")
    job_transfers = None
    if job_transfer_costs is not None:
        if len(job_transfer_costs) != len(jobs):
            raise ValueError(
                "job_transfer_costs must contain one vector per PP microbatch."
            )
        job_transfers = tuple(
            tuple(float(value) for value in values) for values in job_transfer_costs
        )
        if any(len(values) != stage_count - 1 for values in job_transfers):
            raise ValueError(
                "each job transfer vector must contain one value per stage edge."
            )
        transfers = (0.0,) * max(stage_count - 1, 0)
    elif transfer_costs is None:
        transfers = (0.0,) * max(stage_count - 1, 0)
    else:
        transfers = tuple(float(value) for value in transfer_costs)
        if len(transfers) != stage_count - 1:
            raise ValueError(
                "PP flow-shop transfer_costs must contain one value per stage edge."
            )
    stage_available = [0.0] * stage_count
    for job_index, job in enumerate(jobs):
        previous_finish = 0.0
        transfers_for_job = (
            job_transfers[job_index] if job_transfers is not None else transfers
        )
        for stage, service in enumerate(job):
            start = max(
                stage_available[stage],
                previous_finish
                + (transfers_for_job[stage - 1] if stage else 0.0),
            )
            finish = start + max(0.0, float(service))
            stage_available[stage] = finish
            previous_finish = finish
    return stage_available[-1]


@dataclass(frozen=True)
class _DcutPipelineSlot:
    stage_costs: tuple[float, ...]
    expected_tokens: float
    transfer_costs: tuple[float, ...]


def dflash_dcut_batch_is_compactable(batch) -> bool:
    """Whether pruning can use the top1-only fast path without changing output."""
    if batch.has_grammar or batch.return_logprob:
        return False
    sampling_info = batch.sampling_info
    if sampling_info is None:
        return True
    if not sampling_info.is_all_greedy:
        return False
    if sampling_info.has_custom_logit_processor:
        return False
    if getattr(sampling_info, "acc_linear_penalties", None) is not None:
        return False
    penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
    if penalizer is not None and penalizer.is_required:
        return False
    if getattr(sampling_info, "grammar_mask", None) is not None:
        return False
    if getattr(sampling_info, "logit_bias", None) is not None:
        return False
    return True


@dataclass(frozen=True)
class DFlashDcutPlan:
    layout: RaggedVerifyLayout
    keep_count: int
    is_compact: bool
    candidate_index: Optional[int]


class DFlashDcutPlanner:
    """Cross-request D-Cut selector using an offline-only cost table.

    Fixed-ratio mode is entirely device-side after the host-known keep count.
    Auto mode builds a hardware-specific cost table at startup.  The selector
    follows the paper objective: expected committed tokens (including one
    mandatory anchor per request) divided by the profiled speculative-step
    cost.  Under PP, the stage cost vectors are converted to a forward
    microbatch pipeline makespan before selection.
    """

    def __init__(
        self,
        *,
        value: DFlashDcutValue,
        block_size: int,
        model_runner,
        device: torch.device,
        tp_rank: int,
        draft_cost_profiler: Optional[Callable[[int], float]] = None,
        pipeline_transfer_costs: Optional[Sequence[float]] = None,
        pipeline_transfer_models: Optional[Sequence[tuple[float, float]]] = None,
    ) -> None:
        self.value = value
        self.block_size = int(block_size)
        self.gamma = self.block_size - 1
        self.model_runner = model_runner
        self.device = device
        self.tp_rank = int(tp_rank)
        self._draft_cost_profiler = draft_cost_profiler
        self.tp_group = get_tp_group()
        self.pp_group = get_pp_group()
        self.pp_microbatch_count = 1
        if self.pp_group.world_size > 1:
            async_depth = int(getattr(get_parallel(), "pp_async_batch_depth", 0) or 0)
            self.pp_microbatch_count = max(1, self.pp_group.world_size + async_depth)
        transfer_costs = tuple(float(value) for value in (pipeline_transfer_costs or ()))
        if transfer_costs and len(transfer_costs) != self.pp_group.world_size - 1:
            raise ValueError(
                "pipeline_transfer_costs must contain one value per PP stage edge."
            )
        self._pipeline_transfer_costs = transfer_costs or (
            (0.0,) * max(self.pp_group.world_size - 1, 0)
        )
        self._pipeline_transfer_models = tuple(
            (float(alpha), float(beta))
            for alpha, beta in (pipeline_transfer_models or ())
        )
        if self._pipeline_transfer_models and len(
            self._pipeline_transfer_models
        ) != self.pp_group.world_size - 1:
            raise ValueError(
                "pipeline_transfer_models must contain one pair per PP edge."
            )
        self._dp_attention = is_dp_attention_enabled()
        # Under DP attention each rank owns an independent local batch, so the
        # auto-candidate decision is coordinated only within the attention-TP
        # group (size 1 in the pure-DP case, i.e. rank-local). The profiling
        # collectives below keep the full TP group: profiled ranks run the
        # same shapes in lockstep and step latency is the group max.
        self._select_group = (
            get_attn_tp_group() if self._dp_attention else self.tp_group
        )
        self.schedule_cfg = DSparkScheduleConfig(
            gamma=self.gamma,
            min_verify_len=1,
            max_verify_len=self.block_size,
            # A numerically underflowed confidence must still remain selectable
            # when the configured global budget asks for it.
            survival_eps=0.0,
        )
        self.schedule_cfg.validate()
        self.last_candidate_index: Optional[int] = None
        self._costs_by_bs: dict[int, list[Optional[float]]] = {}
        self._overhead_costs_by_bs: dict[int, list[Optional[float]]] = {}
        # Keep the per-PP-rank measurements as well as the reduced bottleneck
        # curve.  Runtime selection can then evaluate a candidate after the
        # bottleneck moves, instead of assuming that the stage which was slow
        # at full width remains slow after D-Cut.
        self._stage_costs_by_bs: dict[int, tuple[tuple[float, ...], ...]] = {}
        self._stage_fold_costs_by_bs: dict[int, tuple[tuple[float, ...], ...]] = {}
        self._stage_overhead_costs_by_bs: dict[
            int, tuple[tuple[float, ...], ...]
        ] = {}
        self._stage_draft_costs_by_bs: dict[int, tuple[tuple[float, ...], ...]] = {}
        self._last_local_profile_cost: Optional[tuple[float, float]] = None
        self._last_local_profile_overhead_ms = 0.0
        self._auto_index_device = torch.zeros((), dtype=torch.int64, device=device)
        self._offline_profiled = False
        self._common_capture_num_tokens: Optional[tuple[int, ...]] = None
        self._offline_keep_counts: dict[int, tuple[int, ...]] = {}
        self._warned_missing_profile_bs: set[int] = set()
        self._cost_tensors_by_bs: dict[int, torch.Tensor] = {}
        self._pp_pipeline_slots: dict[int, _DcutPipelineSlot] = {}
        self._pp_pipeline_slot_seen: dict[int, int] = {}
        self._pp_pipeline_selection_step = 0
        self._flowshop_makespan_cache: dict[tuple, tuple[float, ...]] = {}
        self._runtime_stage_cost_ema: dict[tuple[int, int], float] = {}
        self._full_hold_bs: Optional[int] = None
        self._full_hold_remaining = 0
        # Auto mode: batch sizes whose profiled best-case savings cannot pay
        # for the selection itself are pinned to full-width verify.
        self._pin_full_max_bs = 0

    def should_hold_full(self, bs: int) -> bool:
        return (
            self.pp_group.world_size > 1
            and self._full_hold_bs == int(bs)
            and self._full_hold_remaining > 0
        )

    def consume_full_hold(self) -> None:
        if self._full_hold_remaining > 0:
            self._full_hold_remaining -= 1

    def observe_runtime_stage_cost(
        self,
        *,
        bs: int,
        candidate_index: int,
        cost_ms: float,
        smoothing: float = 0.2,
    ) -> None:
        """Feed a selected candidate's measured stage wall time back into cost.

        The measurement includes compact-path work that startup model.forward
        profiling cannot see. Runtime observations only raise the offline cost
        through ``max`` during selection, so one asynchronous or noisy sample
        cannot make a candidate look unrealistically cheap.
        """
        if not self.is_auto or bs <= 0 or cost_ms <= 0.0:
            return
        if not 0 <= int(candidate_index) < len(_AUTO_RATIOS):
            return
        key = (int(bs), int(candidate_index))
        previous = self._runtime_stage_cost_ema.get(key)
        if previous is None:
            self._runtime_stage_cost_ema[key] = float(cost_ms)
        else:
            alpha = min(max(float(smoothing), 0.0), 1.0)
            self._runtime_stage_cost_ema[key] = (
                (1.0 - alpha) * previous + alpha * float(cost_ms)
            )

    def _runtime_cost_for_candidate(self, bs: int, candidate_index: int) -> Optional[float]:
        choices = [
            (abs(profile_bs - bs), cost)
            for (profile_bs, profile_candidate), cost in self._runtime_stage_cost_ema.items()
            if profile_candidate == candidate_index
        ]
        if not choices:
            return None
        return float(min(choices, key=lambda item: item[0])[1])

    def set_pipeline_transfer_models(
        self, models: Sequence[tuple[float, float]]
    ) -> None:
        """Install measured ``(alpha_ms, beta_ms_per_token)`` PP edge models."""
        models = tuple((float(alpha), float(beta)) for alpha, beta in models)
        if len(models) != self.pp_group.world_size - 1:
            raise ValueError(
                "pipeline transfer model count must equal pp_size - 1."
            )
        self._pipeline_transfer_models = models

    @property
    def is_auto(self) -> bool:
        return self.value == "auto"

    @staticmethod
    def candidate_ratios() -> tuple[float, ...]:
        return _AUTO_RATIOS

    def candidate_keep_counts(self, bs: int) -> tuple[int, ...]:
        return tuple(
            get_dflash_dcut_keep_count(bs=bs, block_size=self.block_size, ratio=ratio)
            for ratio in _AUTO_RATIOS
        )

    def is_full_pinned(self, bs: int) -> bool:
        """Whether this batch size is pinned to full-width verify.

        Only meaningful in auto mode; the pinned prefix is derived from the
        startup cost table (``_compute_pin_full_max_bs``), which is built
        from TP/PP-reduced costs, so every rank agrees on the decision.
        """
        return self.is_auto and 0 < int(bs) <= self._pin_full_max_bs

    def pinned_full_plan(self, *, bs: int) -> DFlashDcutPlan:
        """Full-width plan for a pinned batch size.

        Records the full-width candidate so ``current_keep_budget`` keeps
        publishing the full budget to the DP tier alignment.
        """
        plan = self.full_plan(bs=bs)
        self.last_candidate_index = plan.candidate_index
        return plan

    def _compute_pin_full_max_bs(self) -> None:
        """Pin small batches to full width when pruning cannot pay for itself.

        Selection itself costs a roughly constant amount of work per step
        (confidence math, the top-k schedule, a host sync for the CUDA-graph
        bucket).  When the profiled best-case step savings at a batch size --
        ``cost(ratio=1.0) - min(cost(ratio))`` -- fall below that overhead,
        running the selector can only lose time, so those batch sizes are
        pinned to the full-width plan and selection is skipped entirely.

        ``_costs_by_bs`` is reduced across TP/PP during startup profiling, so
        every rank derives the identical pinned prefix.
        """
        self._pin_full_max_bs = 0
        if not self.is_auto:
            return
        threshold_ms = float(envs.SGLANG_DFLASH_DCUT_PIN_FULL_MIN_SAVINGS_MS.get())
        if threshold_ms <= 0.0:
            return
        spreads: list[tuple[int, float]] = []
        for bs in sorted(self._costs_by_bs):
            costs = self._costs_by_bs[bs]
            if not costs or any(cost is None for cost in costs):
                continue
            spread = float(costs[-1]) - min(float(cost) for cost in costs)
            spreads.append((bs, round(spread, 3)))
            if spread < threshold_ms:
                self._pin_full_max_bs = bs
            else:
                break
        if (
            self._pin_full_max_bs > 0
            and self.tp_rank == 0
            and self.pp_group.rank_in_group == 0
        ):
            logger.info(
                "DFLASH D-Cut auto: pinning bs<=%d to full-width verify "
                "(per-step savings below the %.2f ms selector overhead); "
                "profiled spreads (bs, ms): %s",
                self._pin_full_max_bs,
                threshold_ms,
                spreads,
            )

    def current_keep_budget(self, bs: int) -> int:
        """Keep-count upper bound for the next decode step at this bs.

        Uses the last selected candidate (or full width if none yet)."""
        if bs <= 0:
            return 0
        if not self.is_auto:
            return get_dflash_dcut_keep_count(
                bs=bs,
                block_size=self.block_size,
                ratio=float(self.value),
            )
        candidate_index = self.last_candidate_index
        if candidate_index is None:
            candidate_index = len(_AUTO_RATIOS) - 1
        return self.candidate_keep_counts(bs)[candidate_index]

    def _profile_metric_for_bs(
        self,
        table: dict[int, list[Optional[float]]],
        bs: int,
        *,
        match_graph_tier: bool,
    ) -> Optional[list[float]]:
        exact = table.get(bs)
        if exact is not None and all(cost is not None for cost in exact):
            return [float(cost) for cost in exact]

        complete = {
            profile_bs: costs
            for profile_bs, costs in table.items()
            if all(cost is not None for cost in costs)
        }
        if not complete:
            return None

        requested_keeps = self.candidate_keep_counts(bs)
        resolved = []
        for candidate_index, keep_count in enumerate(requested_keeps):
            same_tier = []
            target_tier = self._graph_num_tokens(bs + keep_count)
            if match_graph_tier:
                for profile_bs, costs in complete.items():
                    profile_keeps = self._offline_keep_counts.get(
                        profile_bs, self.candidate_keep_counts(profile_bs)
                    )
                    profile_tier = self._graph_num_tokens(
                        profile_bs + profile_keeps[candidate_index]
                    )
                    if profile_tier == target_tier:
                        same_tier.append((profile_bs, costs))
            choices = same_tier or list(complete.items())
            if match_graph_tier and not same_tier:
                # Runtime batches need not line up with the small set of
                # captured request-count buckets.  Reuse the profile whose
                # candidate has the nearest packed-token bucket instead of
                # treating a sparse table as a missing table and forcing the
                # hard-coded ratio fallback.
                choices = sorted(
                    choices,
                    key=lambda item: (
                        abs(
                            self._graph_num_tokens(
                                item[0]
                                + self._offline_keep_counts.get(
                                    item[0], self.candidate_keep_counts(item[0])
                                )[candidate_index]
                            )
                            - target_tier
                        ),
                        abs(item[0] - bs),
                        item[0],
                    ),
                )
                _, nearest_costs = choices[0]
            else:
                _, nearest_costs = min(
                    choices, key=lambda item: (abs(item[0] - bs), item[0])
                )
            resolved.append(float(nearest_costs[candidate_index]))
        return resolved

    def _profile_costs_for_bs(self, bs: int) -> Optional[list[float]]:
        return self._profile_metric_for_bs(self._costs_by_bs, bs, match_graph_tier=True)

    def _profile_stage_metric_for_bs(
        self,
        table: dict[int, tuple[tuple[float, ...], ...]],
        bs: int,
        *,
        match_graph_tier: bool,
    ) -> Optional[tuple[tuple[float, ...], ...]]:
        """Resolve a stage-by-candidate table using the same bucket rules.

        The first dimension is PP rank and the second is the candidate index.
        All ranks build the same table during startup, so nearest-batch reuse is
        deterministic and preserves the existing relay contract.
        """
        exact = table.get(bs)
        if exact is not None:
            return exact
        if not table:
            return None

        resolved: list[list[float]] = []
        requested_keeps = self.candidate_keep_counts(bs)
        for candidate_index, keep_count in enumerate(requested_keeps):
            choices = []
            target_tier = self._graph_num_tokens(bs + keep_count)
            for profile_bs, stage_costs in table.items():
                if match_graph_tier:
                    profile_keeps = self._offline_keep_counts.get(
                        profile_bs, self.candidate_keep_counts(profile_bs)
                    )
                    profile_tier = self._graph_num_tokens(
                        profile_bs + profile_keeps[candidate_index]
                    )
                    if profile_tier != target_tier:
                        continue
                choices.append((abs(profile_bs - bs), profile_bs, stage_costs))
            if not choices:
                # A runtime batch may fall between captured graph buckets. Use
                # the closest candidate token bucket rather than abandoning
                # the whole PP stage table and falling back to ratio 0.75.
                for profile_bs, stage_costs in table.items():
                    profile_keeps = self._offline_keep_counts.get(
                        profile_bs, self.candidate_keep_counts(profile_bs)
                    )
                    profile_tier = self._graph_num_tokens(
                        profile_bs + profile_keeps[candidate_index]
                    )
                    choices.append(
                        (
                            abs(profile_tier - target_tier),
                            abs(profile_bs - bs),
                            profile_bs,
                            stage_costs,
                        )
                    )
                _, _, _, nearest = min(
                    choices, key=lambda item: (item[0], item[1], item[2])
                )
            else:
                _, _, nearest = min(choices, key=lambda item: (item[0], item[1]))
            if not resolved:
                resolved = [[] for _ in nearest]
            for rank, values in enumerate(nearest):
                resolved[rank].append(float(values[candidate_index]))
        return tuple(tuple(values) for values in resolved)

    def _profile_stage_costs_for_bs(
        self, bs: int
    ) -> Optional[tuple[tuple[float, ...], ...]]:
        return self._profile_stage_metric_for_bs(
            self._stage_costs_by_bs, bs, match_graph_tier=True
        )

    def _profile_stage_fold_costs_for_bs(
        self, bs: int
    ) -> Optional[tuple[tuple[float, ...], ...]]:
        return self._profile_stage_metric_for_bs(
            self._stage_fold_costs_by_bs, bs, match_graph_tier=False
        )

    def _profile_stage_overhead_costs_for_bs(
        self, bs: int
    ) -> Optional[tuple[tuple[float, ...], ...]]:
        return self._profile_stage_metric_for_bs(
            self._stage_overhead_costs_by_bs, bs, match_graph_tier=False
        )

    def _profile_stage_draft_costs_for_bs(
        self, bs: int
    ) -> Optional[tuple[tuple[float, ...], ...]]:
        return self._profile_stage_metric_for_bs(
            self._stage_draft_costs_by_bs, bs, match_graph_tier=False
        )

    def _cached_device_costs(self, bs: int, costs: list[float]) -> torch.Tensor:
        cached = self._cost_tensors_by_bs.get(bs)
        if cached is not None:
            return cached
        tensor = torch.tensor(costs, dtype=torch.float32, device=self.device)
        self._cost_tensors_by_bs[bs] = tensor
        return tensor

    def _select_pipeline_candidate(
        self,
        *,
        expected: Sequence[float],
        candidate_stage_costs: Sequence[Sequence[float]],
        candidate_transfer_costs: Sequence[Sequence[float]],
        bs: int,
        pipeline_mb_id: int,
    ) -> int:
        """Choose a ratio after inserting it into the in-flight PP schedule."""
        stage_count = self.pp_group.world_size
        if stage_count <= 1:
            raise ValueError("pipeline candidate selection requires PP > 1.")
        if len(candidate_stage_costs) != len(_AUTO_RATIOS):
            raise ValueError("candidate_stage_costs must cover every auto ratio.")
        if len(candidate_transfer_costs) != len(_AUTO_RATIOS):
            raise ValueError("candidate_transfer_costs must cover every auto ratio.")

        self._pp_pipeline_selection_step += 1
        selection_step = self._pp_pipeline_selection_step
        stale_before = selection_step - self.pp_microbatch_count
        for slot_id, seen_step in tuple(self._pp_pipeline_slot_seen.items()):
            if seen_step < stale_before:
                self._pp_pipeline_slot_seen.pop(slot_id, None)
                self._pp_pipeline_slots.pop(slot_id, None)

        # Unknown slots are conservatively modeled with the candidate under
        # consideration. Once every slot has been observed this becomes a
        # direct per-microbatch flow-shop simulation.
        slot_ids = list(range(self.pp_microbatch_count))
        if pipeline_mb_id not in slot_ids:
            slot_ids.append(int(pipeline_mb_id))
            slot_ids.sort()

        # The makespan side of the score depends only on the candidate cost
        # vectors and the other slots' state, not on this step's confidence;
        # cache it so steady-state steps skip the repeated simulation.
        cache_key = (
            tuple(slot_ids),
            int(pipeline_mb_id),
            tuple(
                tuple(float(value) for value in costs)
                for costs in candidate_stage_costs
            ),
            tuple(
                tuple(float(value) for value in costs)
                for costs in candidate_transfer_costs
            ),
            tuple(
                (slot_id, slot.stage_costs, slot.transfer_costs)
                for slot_id, slot in sorted(self._pp_pipeline_slots.items())
                if slot_id != pipeline_mb_id and slot_id in slot_ids
            ),
            self._pipeline_transfer_costs,
        )
        makespans = self._flowshop_makespan_cache.get(cache_key)
        if makespans is None:
            makespan_list = []
            for candidate_index, current_costs in enumerate(candidate_stage_costs):
                current_stage_costs = tuple(float(value) for value in current_costs)
                current_transfer_costs = tuple(
                    float(value) for value in candidate_transfer_costs[candidate_index]
                )
                jobs = []
                job_transfer_costs = []
                for slot_id in slot_ids:
                    if slot_id == pipeline_mb_id:
                        jobs.append(current_stage_costs)
                        job_transfer_costs.append(current_transfer_costs)
                        continue
                    slot = self._pp_pipeline_slots.get(slot_id)
                    if slot is None:
                        jobs.append(current_stage_costs)
                        job_transfer_costs.append(current_transfer_costs)
                    else:
                        jobs.append(slot.stage_costs)
                        job_transfer_costs.append(slot.transfer_costs)
                makespan_list.append(
                    pp_pipeline_flowshop_makespan(
                        jobs,
                        transfer_costs=self._pipeline_transfer_costs,
                        job_transfer_costs=job_transfer_costs,
                    )
                )
            makespans = tuple(makespan_list)
            if len(self._flowshop_makespan_cache) >= 256:
                self._flowshop_makespan_cache.clear()
            self._flowshop_makespan_cache[cache_key] = makespans

        scores = []
        for candidate_index in range(len(_AUTO_RATIOS)):
            current_expected = float(expected[candidate_index])
            total_expected = current_expected
            for slot_id in slot_ids:
                if slot_id == pipeline_mb_id:
                    continue
                slot = self._pp_pipeline_slots.get(slot_id)
                total_expected += (
                    slot.expected_tokens if slot is not None else current_expected
                )
            scores.append(total_expected / max(makespans[candidate_index], 1e-6))

        selected = int(torch.tensor(scores).argmax().item())
        self._pp_pipeline_slots[int(pipeline_mb_id)] = _DcutPipelineSlot(
            stage_costs=tuple(
                float(value) for value in candidate_stage_costs[selected]
            ),
            expected_tokens=float(expected[selected]),
            transfer_costs=tuple(
                float(value) for value in candidate_transfer_costs[selected]
            ),
        )
        self._pp_pipeline_slot_seen[int(pipeline_mb_id)] = selection_step
        if selected == len(_AUTO_RATIOS) - 1 and self._runtime_stage_cost_ema:
            self._full_hold_bs = int(bs)
            self._full_hold_remaining = 32
        elif self._full_hold_bs == int(bs):
            self._full_hold_bs = None
            self._full_hold_remaining = 0
        return selected

    def _stage_cost_rows_host(
        self,
        *,
        bs: int,
        stage_costs: Sequence[Sequence[float]],
        stage_fold_costs: Optional[Sequence[Sequence[float]]],
        stage_overhead_costs: Optional[Sequence[Sequence[float]]],
        stage_draft_costs: Optional[Sequence[Sequence[float]]],
        overhead_costs: Optional[Sequence[float]],
    ) -> list[list[float]]:
        """Assemble per-stage candidate cost rows as host floats.

        Every input is host data from the startup profile; the previous
        implementation round-tripped these through device tensors and read
        each element back with ``.item()`` on every decode step.
        """
        rows: list[list[float]] = []
        for rank, values in enumerate(stage_costs):
            row = [float(value) for value in values]
            if stage_fold_costs is not None and rank < len(stage_fold_costs):
                row = [
                    value + float(fold)
                    for value, fold in zip(row, stage_fold_costs[rank])
                ]
            if stage_overhead_costs is not None and rank < len(stage_overhead_costs):
                row = [
                    value + float(overhead)
                    for value, overhead in zip(row, stage_overhead_costs[rank])
                ]
            if stage_draft_costs is not None and rank < len(stage_draft_costs):
                row = [
                    value + float(draft)
                    for value, draft in zip(row, stage_draft_costs[rank])
                ]
            elif (
                stage_overhead_costs is None
                and stage_draft_costs is None
                and overhead_costs is not None
            ):
                row = [
                    value + float(overhead)
                    for value, overhead in zip(row, overhead_costs)
                ]
            for candidate_index in range(len(_AUTO_RATIOS)):
                observed = self._runtime_cost_for_candidate(bs, candidate_index)
                if observed is not None:
                    row[candidate_index] = max(row[candidate_index], observed)
            rows.append(row)
        return rows

    def _write_auto_index_local(
        self,
        *,
        confidence: torch.Tensor,
        bs: int,
        pipeline_mb_id: Optional[int] = None,
    ) -> Optional[int]:
        """Select the auto-mode candidate on this rank.

        Returns the selected index when it is resolved on the host (the PP
        per-microbatch flow-shop path); the caller may then skip the group
        broadcast and the device scalar readback when the select group is
        trivial.  Otherwise the result is written to ``_auto_index_device``
        and None is returned.
        """
        costs = self._profile_costs_for_bs(bs)
        if costs is None:
            if bs not in self._warned_missing_profile_bs and self.tp_rank == 0:
                logger.warning(
                    "DFLASH D-Cut offline cost table missing for bs=%d; "
                    "falling back to ratio 0.75.",
                    bs,
                )
                self._warned_missing_profile_bs.add(bs)
            self._auto_index_device.fill_(2)
            return None

        survival = torch.cumprod(confidence.to(torch.float32), dim=1).flatten()
        sorted_survival = torch.sort(survival, descending=True).values
        prefix_scores = torch.cumsum(sorted_survival, dim=0)
        keep_counts = self.candidate_keep_counts(bs)
        expected = torch.stack(
            tuple(
                prefix_scores[keep_count - 1] + float(bs)
                if keep_count > 0
                else prefix_scores.new_zeros(()) + float(bs)
                for keep_count in keep_counts
            )
        )

        stage_costs = self._profile_stage_costs_for_bs(bs)
        stage_fold_costs = self._profile_stage_fold_costs_for_bs(bs)
        stage_overhead_costs = self._profile_stage_overhead_costs_for_bs(bs)
        stage_draft_costs = self._profile_stage_draft_costs_for_bs(bs)
        overhead_costs = self._profile_metric_for_bs(
            self._overhead_costs_by_bs, bs, match_graph_tier=False
        )
        candidate_transfer_costs = tuple(
            tuple(
                alpha + beta * self._graph_num_tokens(bs + keep_count)
                for alpha, beta in self._pipeline_transfer_models
            )
            if self._pipeline_transfer_models
            else self._pipeline_transfer_costs
            for keep_count in keep_counts
        )
        if stage_costs is not None:
            stage_rows = self._stage_cost_rows_host(
                bs=bs,
                stage_costs=stage_costs,
                stage_fold_costs=stage_fold_costs,
                stage_overhead_costs=stage_overhead_costs,
                stage_draft_costs=stage_draft_costs,
                overhead_costs=overhead_costs,
            )
            if pipeline_mb_id is not None and self.pp_group.world_size > 1:
                # Host-side flow-shop selection: one D2H readback for the four
                # expected scores replaces the previous per-stage/per-candidate
                # .item() calls (13 scalar syncs per selection).
                selected = self._select_pipeline_candidate(
                    expected=[float(value) for value in expected.tolist()],
                    candidate_stage_costs=tuple(
                        tuple(row[candidate_index] for row in stage_rows)
                        for candidate_index in range(len(_AUTO_RATIOS))
                    ),
                    candidate_transfer_costs=candidate_transfer_costs,
                    bs=bs,
                    pipeline_mb_id=int(pipeline_mb_id),
                )
                if self._select_group.world_size > 1:
                    self._auto_index_device.fill_(selected)
                    return None
                return selected
            # The tensor rows above are [stage, candidate].
            cost_tensor = pp_pipeline_cycle_cost(
                torch.stack(
                    [
                        torch.tensor(row, dtype=torch.float32, device=self.device)
                        for row in stage_rows
                    ],
                    dim=0,
                ),
                self.pp_microbatch_count,
            )
        else:
            cost_tensor = self._cached_device_costs(bs, costs)
            runtime_costs = cost_tensor.clone()
            for candidate_index in range(len(_AUTO_RATIOS)):
                observed = self._runtime_cost_for_candidate(bs, candidate_index)
                if observed is not None:
                    runtime_costs[candidate_index] = torch.maximum(
                        runtime_costs[candidate_index],
                        runtime_costs.new_tensor(observed),
                    )
            cost_tensor = runtime_costs
        scores = score_dcut_candidates(expected=expected, costs=cost_tensor)
        self._auto_index_device.copy_(torch.argmax(scores).to(dtype=torch.int64))
        return None

    def _select_auto_candidate(
        self,
        *,
        confidence: torch.Tensor,
        bs: int,
        pipeline_mb_id: Optional[int] = None,
    ) -> int:
        group = self._select_group
        host_selected: Optional[int] = None
        if group.rank_in_group == 0:
            host_selected = self._write_auto_index_local(
                confidence=confidence,
                bs=bs,
                pipeline_mb_id=pipeline_mb_id,
            )
        if host_selected is not None and group.world_size == 1:
            # Host-resolved selection with a trivial select group: neither the
            # broadcast nor the scalar readback is needed.
            return host_selected
        group.broadcast(self._auto_index_device, src=0)
        # CUDA graph replay keys off a host bucket. This is the only hot-path
        # scalar sync required by the ratio selector.
        return int(self._auto_index_device.item())

    def _graph_num_tokens(self, total_verify_tokens: int) -> int:
        runner = self.model_runner.decode_cuda_graph_runner
        common_capture_num_tokens = getattr(self, "_common_capture_num_tokens", None)
        capture_num_tokens = (
            common_capture_num_tokens
            if common_capture_num_tokens is not None
            else (
                tuple(runner.capture_num_tokens)
                if runner is not None and runner.capture_num_tokens is not None
                else ()
            )
        )
        if (
            runner is None
            or not runner.ragged_verify_mode
            or not capture_num_tokens
            or total_verify_tokens > capture_num_tokens[-1]
        ):
            return total_verify_tokens
        return round_up_grid(total_verify_tokens, capture_num_tokens)

    def _initialize_common_capture_grid(self) -> bool:
        if self._common_capture_num_tokens is not None:
            return bool(self._common_capture_num_tokens)

        runner = self.model_runner.decode_cuda_graph_runner
        local_grid = tuple(getattr(runner, "capture_num_tokens", None) or ())
        graph_ready = torch.tensor(
            int(runner is not None and runner.ragged_verify_mode and bool(local_grid)),
            dtype=torch.int32,
            device=self.device,
        )
        if self.tp_group.world_size > 1:
            torch.distributed.all_reduce(
                graph_ready,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group.device_group,
            )
        if self.pp_group.world_size > 1:
            torch.distributed.all_reduce(
                graph_ready,
                op=torch.distributed.ReduceOp.MIN,
                group=self.pp_group.device_group,
            )
        if int(graph_ready.item()) == 0:
            self._common_capture_num_tokens = ()
            return False

        common = set(local_grid)
        if self.pp_group.world_size > 1:
            gathered: list[Optional[tuple[int, ...]]] = [
                None
            ] * self.pp_group.world_size
            torch.distributed.all_gather_object(
                gathered, local_grid, group=self.pp_group.cpu_group
            )
            for peer_grid in gathered:
                common.intersection_update(peer_grid or ())
        self._common_capture_num_tokens = tuple(sorted(common))
        return bool(self._common_capture_num_tokens)

    def _build_profile_verify_layout(
        self, *, bs: int, keep_count: int
    ) -> RaggedVerifyLayout:
        """Build a ragged verify layout that fits the captured graph buckets."""
        total_verify_tokens = bs + keep_count
        graph_num_tokens = self._graph_num_tokens(total_verify_tokens)
        verify_lens_cpu = build_capture_verify_lens(
            num_tokens=total_verify_tokens,
            # This is a live batch layout. The graph runner pads it to the
            # capture tier's synthetic slot count before replay.
            num_slots=bs,
            num_draft_tokens=self.block_size,
        )
        return RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens_cpu,
            device=self.device,
            grid=(
                self.model_runner.decode_cuda_graph_runner.capture_num_tokens
                if self.model_runner.decode_cuda_graph_runner is not None
                and self.model_runner.decode_cuda_graph_runner.capture_num_tokens
                is not None
                else [graph_num_tokens]
            ),
            graph_num_tokens_floor=graph_num_tokens,
        )

    def _build_profile_req_pool_indices(self, bs: int) -> torch.Tensor:
        """Use valid request rows while keeping row zero reserved for padding."""
        req_pool = self.model_runner.req_to_token_pool
        if req_pool is None:
            return torch.arange(bs, dtype=torch.int64, device=self.device)

        if bs > req_pool.size:
            raise RuntimeError(
                f"DFLASH D-Cut profiling batch size {bs} exceeds the request "
                f"pool capacity {req_pool.size}."
            )
        # ReqToTokenPool has size + 1 rows: zero is padding and real requests
        # occupy 1..size. Profiling runs before request admission.
        return torch.arange(1, bs + 1, dtype=torch.int64, device=self.device)

    @contextmanager
    def _isolate_replayssm_profile_slots(
        self, req_pool_indices: torch.Tensor
    ) -> Iterator[None]:
        """Give ReplaySSM profile rows distinct, temporary state slots.

        A fresh hybrid request pool maps every request row to Mamba slot zero.
        That is harmless for ordinary graph capture, but a real profile forward
        writes ReplaySSM records and concurrent rows would race in slot zero.
        Borrow free slot ids without calling alloc/free, then restore the request
        mapping even when profiling fails. The slots remain allocator-owned free
        space and are cleared normally if a real request later acquires them.
        """
        prepared = self._prepare_replayssm_profile_slots(req_pool_indices)
        with self._use_replayssm_profile_slots(prepared):
            yield

    @contextmanager
    def _use_replayssm_profile_slots(
        self,
        prepared: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ) -> Iterator[None]:
        if prepared is None:
            yield
            return

        mapping, rows, state_slots, saved_mapping = prepared
        try:
            mapping[rows] = state_slots
            yield
        finally:
            mapping[rows] = saved_mapping
            if mapping.is_cuda:
                torch.get_device_module(mapping.device).synchronize()

    def _prepare_replayssm_profile_slots(
        self, req_pool_indices: torch.Tensor
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Validate and materialize a temporary profile-slot mapping."""
        req_pool = self.model_runner.req_to_token_pool
        mamba_pool = getattr(req_pool, "mamba_pool", None)
        if not getattr(mamba_pool, "enable_linear_replayssm_spec", False):
            return None

        mapping = getattr(req_pool, "req_index_to_mamba_index_mapping", None)
        allocator = getattr(req_pool, "mamba_allocator", None)
        free_state_slots = getattr(allocator, "free_slots", None)
        if mapping is None or free_state_slots is None:
            raise RuntimeError(
                "ReplaySSM D-Cut profiling requires a request-to-Mamba mapping "
                "and a readable free-slot list."
            )

        rows = req_pool_indices.to(device=mapping.device, dtype=torch.int64).clone()
        num_rows = int(rows.numel())
        if int(torch.unique(rows).numel()) != num_rows:
            raise RuntimeError(
                "ReplaySSM D-Cut profiling requires distinct request pool rows."
            )
        if len(free_state_slots) < num_rows:
            raise RuntimeError(
                "ReplaySSM D-Cut profiling requires "
                f"{num_rows} distinct free Mamba slots, but only "
                f"{len(free_state_slots)} are available."
            )

        state_slots = free_state_slots[:num_rows].to(
            device=mapping.device, dtype=mapping.dtype
        )
        if int(torch.unique(state_slots).numel()) != num_rows:
            raise RuntimeError(
                "ReplaySSM D-Cut profiling received duplicate free Mamba slots."
            )

        saved_mapping = mapping[rows].clone()
        return mapping, rows, state_slots, saved_mapping

    def _replayssm_profile_enabled(self) -> bool:
        req_pool = self.model_runner.req_to_token_pool
        mamba_pool = getattr(req_pool, "mamba_pool", None)
        return bool(getattr(mamba_pool, "enable_linear_replayssm_spec", False))

    def _commit_replayssm_profile(self, forward_batch: ForwardBatch) -> None:
        if not self._replayssm_profile_enabled():
            return
        attn_backend = self.model_runner.attn_backend
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            raise RuntimeError(
                "ReplaySSM D-Cut profiling requires a target-verify commit backend."
            )
        verify_lens = forward_batch.spec_info.ragged_verify_layout.verify_lens[
            : forward_batch.batch_size
        ]
        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=verify_lens.to(torch.int64) - 1,
            mamba_track_indices=None,
            mamba_steps_to_track=None,
            model=self.model_runner.model,
            req_pool_indices=forward_batch.req_pool_indices,
        )

    def _max_parallel_cost_ms(self, cost_ms: float) -> float:
        """Return the bottleneck cost across both TP and PP dimensions."""
        cost = torch.tensor(cost_ms, dtype=torch.float32, device=self.device)
        if self.tp_group.world_size > 1:
            torch.distributed.all_reduce(
                cost,
                op=torch.distributed.ReduceOp.MAX,
                group=self.tp_group.device_group,
            )
        if self.pp_group.world_size > 1:
            torch.distributed.all_reduce(
                cost,
                op=torch.distributed.ReduceOp.MAX,
                group=self.pp_group.device_group,
            )
        return float(cost.item())

    def _gather_profile_stage_costs(
        self, local_costs: Sequence[float]
    ) -> tuple[tuple[float, ...], ...]:
        """Gather one candidate cost vector in PP order.

        TP ranks first reduce to the slowest shard of their local PP stage;
        the resulting vector is then gathered across the PP process group.
        Every rank executes this during startup profiling, so no hot-path
        collective is needed when the last PP rank selects the next plan.
        """
        values = torch.tensor(
            tuple(float(value) for value in local_costs),
            dtype=torch.float32,
            device=self.device,
        )
        if self.tp_group.world_size > 1:
            torch.distributed.all_reduce(
                values,
                op=torch.distributed.ReduceOp.MAX,
                group=self.tp_group.device_group,
            )
        if self.pp_group.world_size <= 1:
            return (tuple(float(value) for value in values.tolist()),)

        gathered = [torch.empty_like(values) for _ in range(self.pp_group.world_size)]
        torch.distributed.all_gather(
            gathered, values, group=self.pp_group.device_group
        )
        return tuple(
            tuple(float(value) for value in stage.tolist()) for stage in gathered
        )

    def _raise_if_parallel_profile_failed(self, error: Optional[Exception]) -> None:
        if self.tp_group.world_size == 1 and self.pp_group.world_size == 1:
            if error is not None:
                raise error
            return

        failed = torch.tensor(
            int(error is not None), dtype=torch.int32, device=self.device
        )
        if self.tp_group.world_size > 1:
            torch.distributed.all_reduce(
                failed,
                op=torch.distributed.ReduceOp.MAX,
                group=self.tp_group.device_group,
            )
        if self.pp_group.world_size > 1:
            torch.distributed.all_reduce(
                failed,
                op=torch.distributed.ReduceOp.MAX,
                group=self.pp_group.device_group,
            )
        if int(failed.item()) != 0:
            if error is not None:
                raise RuntimeError(
                    "DFLASH D-Cut profiling failed on local "
                    f"PP/TP rank ({self.pp_group.rank_in_group}, {self.tp_rank}): "
                    f"{error}"
                ) from error
            raise RuntimeError("DFLASH D-Cut profiling failed on another PP/TP rank.")

    def _build_profile_pp_proxy(
        self, graph_num_tokens: int
    ) -> Optional[PPProxyTensors]:
        if self.pp_group.world_size == 1:
            return None
        runner = self.model_runner.decode_cuda_graph_runner
        buffers = getattr(runner, "buffers", None)
        proxy_buffers = getattr(buffers, "pp_proxy_tensors", None)
        if not proxy_buffers:
            raise RuntimeError(
                "DFLASH D-Cut PP profiling requires decode graph PP proxy buffers."
            )
        return PPProxyTensors(
            {name: tensor[:graph_num_tokens] for name, tensor in proxy_buffers.items()}
        )

    def _build_profile_forward_batch(
        self,
        *,
        bs: int,
        keep_count: int,
    ) -> ForwardBatch:
        """Construct a minimal TARGET_VERIFY batch for offline profiling."""
        total_verify_tokens = bs + keep_count
        graph_num_tokens = self._graph_num_tokens(total_verify_tokens)
        layout = self._build_profile_verify_layout(bs=bs, keep_count=keep_count)

        seq_lens = torch.full(
            (bs,), _OFFLINE_PROFILE_SEQ_LEN, dtype=torch.int64, device=self.device
        )
        seq_lens_cpu = torch.full(
            (bs,), _OFFLINE_PROFILE_SEQ_LEN, dtype=torch.int32, device="cpu"
        )
        positions = torch.arange(
            graph_num_tokens, dtype=torch.int64, device=self.device
        )
        mrope_positions = positions.unsqueeze(0).repeat(3, 1)
        input_ids = torch.zeros(
            (graph_num_tokens,), dtype=torch.int64, device=self.device
        )
        out_cache_loc = torch.arange(
            graph_num_tokens, dtype=torch.int64, device=self.device
        )
        req_pool_indices = self._build_profile_req_pool_indices(bs)

        spec_info = DFlashVerifyInput(
            draft_token=input_ids,
            positions=positions,
            draft_token_num=self.block_size,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            ragged_verify_layout=layout,
        )

        # Mirror ForwardBatch.init_mlp_sync_metadata: under DP attention the
        # hand-built batch must carry the per-rank token counts the MLP-sync
        # path (eager prepare_mlp_sync_batch) and graph admission
        # (can_run_dp_cuda_graph) read. Every DP rank profiles the same
        # (bs, keep_count) grid in lockstep, so all ranks report
        # graph_num_tokens and sum/max padding agree cluster-wide.
        dp_fields: dict = {}
        if self._dp_attention:
            dp_size = get_attention_dp_size()
            global_num_tokens = [graph_num_tokens] * dp_size
            dp_fields = dict(
                original_global_num_tokens_cpu=[bs] * dp_size,
                global_num_tokens_cpu=global_num_tokens,
                global_num_tokens_gpu=torch.tensor(
                    global_num_tokens, dtype=torch.int64, device=self.device
                ),
                global_num_tokens_for_logprob_cpu=list(global_num_tokens),
                global_num_tokens_for_logprob_gpu=torch.tensor(
                    global_num_tokens, dtype=torch.int64, device=self.device
                ),
                can_run_dp_cuda_graph=True,
            )

        return ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            orig_seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(seq_lens.sum().item()),
            positions=positions,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            global_forward_mode=ForwardMode.TARGET_VERIFY,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            num_token_non_padded=torch.tensor(
                graph_num_tokens, dtype=torch.int64, device=self.device
            ),
            **dp_fields,
        )

    def _profile_compact_overhead_ms(self, *, bs: int, keep_count: int) -> float:
        """Measure planner/layout work that is absent from model.forward().

        The target verify profile captures the expensive model kernels, but it
        does not include top-k scheduling or ragged indptr construction.  Those
        operations are material at small PP batches, where the verify kernel is
        already close to its launch floor.  Measure them with the same CUDA
        event protocol and fold the result into the candidate cost table.
        """
        if (
            torch.device(self.device).type != "cuda"
            or keep_count <= 0
            or keep_count >= bs * self.gamma
        ):
            return 0.0
        try:
            confidence = torch.ones(
                (bs, self.gamma), dtype=torch.float32, device=self.device
            )
            graph_num_tokens = self._graph_num_tokens(bs + keep_count)

            for _ in range(2):
                verify_lens = ScheduleVerifyLensTopk.execute(
                    confidence=confidence,
                    budget=keep_count,
                    cfg=self.schedule_cfg,
                )
                RaggedVerifyLayout.from_verify_lens_device(
                    verify_lens=verify_lens,
                    graph_num_tokens=graph_num_tokens,
                )
            torch.get_device_module(self.device).synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(_OFFLINE_PROFILE_STEPS):
                verify_lens = ScheduleVerifyLensTopk.execute(
                    confidence=confidence,
                    budget=keep_count,
                    cfg=self.schedule_cfg,
                )
                RaggedVerifyLayout.from_verify_lens_device(
                    verify_lens=verify_lens,
                    graph_num_tokens=graph_num_tokens,
                )
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / _OFFLINE_PROFILE_STEPS
        except Exception as exc:  # pragma: no cover - CUDA/backend dependent
            logger.debug("DFLASH D-Cut compact overhead profile skipped: %s", exc)
            return 0.0

    def _profile_dcut_cost_ms(self, *, bs: int, keep_count: int) -> tuple[float, float]:
        """Average target-verify and full-prefix ReplaySSM fold latency."""
        verify_cost_ms = 0.0
        fold_cost_ms = 0.0
        preflight_error = None
        forward_batch = None
        pp_proxy_tensors = None
        prepared_slots = None
        try:
            forward_batch = self._build_profile_forward_batch(
                bs=bs, keep_count=keep_count
            )
            pp_proxy_tensors = self._build_profile_pp_proxy(
                forward_batch.input_ids.shape[0]
            )
            prepared_slots = self._prepare_replayssm_profile_slots(
                forward_batch.req_pool_indices
            )
        except Exception as exc:
            preflight_error = exc
        # Every rank reaches this checkpoint before any model collective. Local
        # forward/CUDA failures after it are not recoverable and propagate.
        self._raise_if_parallel_profile_failed(preflight_error)
        assert forward_batch is not None

        with self._use_replayssm_profile_slots(prepared_slots):
            torch.get_device_module(self.device).synchronize()
            for _ in range(_OFFLINE_PROFILE_WARMUPS):
                self.model_runner.forward(
                    forward_batch, pp_proxy_tensors=pp_proxy_tensors
                )
            torch.get_device_module(self.device).synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(_OFFLINE_PROFILE_STEPS):
                self.model_runner.forward(
                    forward_batch, pp_proxy_tensors=pp_proxy_tensors
                )
            end.record()
            torch.cuda.synchronize()
            verify_cost_ms = start.elapsed_time(end) / _OFFLINE_PROFILE_STEPS

            if self._replayssm_profile_enabled():
                # Compile the fold before timing it. Replaying the same ring into
                # a private free slot repeatedly is safe; only latency is used.
                self._commit_replayssm_profile(forward_batch)
                torch.get_device_module(self.device).synchronize()
                fold_start = torch.cuda.Event(enable_timing=True)
                fold_end = torch.cuda.Event(enable_timing=True)
                fold_start.record()
                for _ in range(_OFFLINE_PROFILE_STEPS):
                    self._commit_replayssm_profile(forward_batch)
                fold_end.record()
                torch.cuda.synchronize()
                fold_cost_ms = (
                    fold_start.elapsed_time(fold_end) / _OFFLINE_PROFILE_STEPS
                )
        # Save the local stage measurement before reducing it for the legacy
        # bottleneck table.  The startup profile gathers these local values
        # across PP ranks and keeps both representations.
        self._last_local_profile_cost = (verify_cost_ms, fold_cost_ms)
        self._last_local_profile_overhead_ms = self._profile_compact_overhead_ms(
            bs=bs, keep_count=keep_count
        )
        return self._max_parallel_cost_ms(verify_cost_ms), self._max_parallel_cost_ms(
            fold_cost_ms
        )

    def _get_dcut_profile_batch_sizes(self) -> tuple[int, ...]:
        """Batch sizes to profile, derived from captured graph token buckets."""
        runner = self.model_runner.decode_cuda_graph_runner
        pool_max_bs = (
            self.model_runner.req_to_token_pool.size
            if self.model_runner.req_to_token_pool is not None
            else 1
        )
        runtime_max_bs = int(getattr(self.model_runner, "max_running_requests", 0) or 0)
        # In the all-eager fallback there is no graph tier to derive a profile
        # shape from.  Profile the requested runtime capacity instead of the
        # (often much larger) request-pool allocation.
        max_bs = (
            runtime_max_bs if runner is None and runtime_max_bs > 0 else pool_max_bs
        )
        capture_num_tokens = self._common_capture_num_tokens or ()
        if runner is None or not capture_num_tokens:
            sizes = (min(max_bs, _OFFLINE_PROFILE_MAX_BS),)
        else:
            local_sizes = []
            for num_tokens in capture_num_tokens:
                if num_tokens % self.block_size == 0:
                    bs = num_tokens // self.block_size
                    if 0 < bs <= min(max_bs, _OFFLINE_PROFILE_MAX_BS):
                        local_sizes.append(bs)
            if not local_sizes:
                local_sizes.append(min(max_bs, _OFFLINE_PROFILE_MAX_BS))
            sizes = tuple(sorted(set(local_sizes)))

        return sizes

    def profile_dcut_cost_table(self) -> None:
        """Offline warm-up: build the auto-mode cost table."""
        if self._offline_profiled:
            return
        graph_ready = self._initialize_common_capture_grid()
        if not self.is_auto:
            self._offline_profiled = True
            return
        if not graph_ready:
            # There is no graph-shaped cost to calibrate in the all-eager
            # fallback. Under PP every stage takes the same branch so no rank
            # enters profiling collectives alone.
            self._offline_profiled = True
            if self.tp_rank == 0 and self.pp_group.rank_in_group == 0:
                logger.info(
                    "DFLASH D-Cut auto: compact decode CUDA graph is unavailable "
                    "on a PP/TP rank or has no common bucket; "
                    "using ratio 0.75 without offline profiling."
                )
            return

        try:
            profile_bs_list = self._get_dcut_profile_batch_sizes()
            if not profile_bs_list:
                self._offline_profiled = True
                if self.tp_rank == 0 and self.pp_group.rank_in_group == 0:
                    logger.warning(
                        "DFLASH D-Cut auto found no CUDA graph batch size common "
                        "to every PP stage; using ratio 0.75."
                    )
                return
            costs_by_bs: dict[int, list[tuple[int, float, float]]] = {}
            for bs in profile_bs_list:
                keep_counts = self.candidate_keep_counts(bs)
                self._offline_keep_counts[bs] = keep_counts
                local_draft_cost = (
                    float(self._draft_cost_profiler(bs))
                    if self._draft_cost_profiler is not None
                    else 0.0
                )
                draft_cost = self._max_parallel_cost_ms(local_draft_cost)
                entries: list[tuple[int, float, float]] = []
                local_verify_costs: list[float] = []
                local_fold_costs: list[float] = []
                local_overhead_costs: list[float] = []
                overhead_costs: list[float] = []
                for keep_count in keep_counts:
                    verify_cost, fold_cost = self._profile_dcut_cost_ms(
                        bs=bs, keep_count=keep_count
                    )
                    entries.append((keep_count, verify_cost, fold_cost))
                    if self._last_local_profile_cost is None:
                        raise RuntimeError(
                            "DFLASH D-Cut profiling did not produce a local cost"
                        )
                    local_verify, local_fold = self._last_local_profile_cost
                    local_verify_costs.append(local_verify)
                    local_fold_costs.append(local_fold)
                    local_overhead_costs.append(self._last_local_profile_overhead_ms)
                    overhead_costs.append(
                        self._max_parallel_cost_ms(
                            self._last_local_profile_overhead_ms
                        )
                    )
                costs_by_bs[bs] = entries
                self._overhead_costs_by_bs[bs] = overhead_costs
                self._stage_costs_by_bs[bs] = self._gather_profile_stage_costs(
                    local_verify_costs
                )
                self._stage_fold_costs_by_bs[bs] = self._gather_profile_stage_costs(
                    local_fold_costs
                )
                self._stage_overhead_costs_by_bs[bs] = (
                    self._gather_profile_stage_costs(local_overhead_costs)
                )
                local_draft_vector = [
                    local_draft_cost if self.pp_group.is_last_rank else 0.0
                ] * len(keep_counts)
                # The draft model is colocated with the last PP stage.  Keep
                # this vector separate so PP cycle modeling can add it to the
                # correct stage instead of spreading it over all stages.
                self._stage_draft_costs_by_bs[bs] = self._gather_profile_stage_costs(
                    local_draft_vector
                )
                total_costs = [
                    draft_cost + verify + fold + overhead
                    for (_, verify, fold), overhead in zip(entries, overhead_costs)
                ]
                self._costs_by_bs[bs] = total_costs
                self._cost_tensors_by_bs[bs] = torch.tensor(
                    total_costs, dtype=torch.float32, device=self.device
                )

            self._compute_pin_full_max_bs()
            self._offline_profiled = True
            if self.tp_rank == 0 and costs_by_bs:
                logger.info(
                    "DFLASH D-Cut offline cost table ready: block_size=%d %s",
                    self.block_size,
                    {
                        bs: [
                            (ratio, round(verify + fold + overhead, 4))
                            for ratio, (_keep, verify, fold), overhead in zip(
                                _AUTO_RATIOS,
                                costs_by_bs[bs],
                                self._overhead_costs_by_bs[bs],
                            )
                        ]
                        for bs in sorted(costs_by_bs)
                    },
                )
        except Exception as e:
            self._offline_profiled = True
            if self.tp_group.world_size > 1 or self.pp_group.world_size > 1:
                raise RuntimeError(
                    "DFLASH D-Cut offline profiling failed under model "
                    "parallelism; aborting startup because a rank-local fallback "
                    "cannot keep all PP stages coordinated safely."
                ) from e
            if self.tp_rank == 0:
                logger.warning(
                    "DFLASH D-Cut offline profiling failed (%s); "
                    "auto mode will fall back to ratio 0.75.",
                    e,
                )

    def load_dcut_cost_table(
        self,
        path: str,
        *,
        expected_partition: Optional[Sequence[int]] = None,
    ) -> None:
        """Load an offline-profiled D-Cut cost table instead of profiling.

        The table (benchmark/pp_spec tooling) carries per-stage step costs per
        (batch bucket, ratio).  Every PP/TP rank must load the identical file:
        the per-rank layout decisions fork if the tables diverge, so the file
        digest is compared across both groups before anything is accepted.

        Schema (version 1)::

            {
              "version": 1,
              "block_size": 16,
              "pp_size": 2,
              "pp_layer_partition": [20, 12],          # optional, validated
              "ratios": [0.25, 0.5, 0.75, 1.0],        # column order below
              "buckets": {
                "<bs>": {
                  "stage_costs_ms": [[per-ratio ...], ...],   # one row per stage
                  "step_costs_ms": [per-ratio ...]            # optional
                }
              }
            }

        ``step_costs_ms`` defaults to the per-candidate bottleneck (max over
        stages).  The stage rows map onto the auto candidate order
        ``_AUTO_RATIOS``; the file's own ``ratios`` array may use any order.
        """
        if self._offline_profiled:
            return
        # The capture-grid init is collective; run it on this path too so every
        # rank resolves the same token buckets for the loaded costs.
        graph_ready = self._initialize_common_capture_grid()
        if not self.is_auto:
            self._offline_profiled = True
            return

        raw_bytes = Path(path).read_bytes()
        digest = hashlib.md5(raw_bytes).hexdigest()
        for group in (self.tp_group, self.pp_group):
            if group.world_size > 1:
                gathered: list[Optional[str]] = [None] * group.world_size
                torch.distributed.all_gather_object(
                    gathered, digest, group=group.cpu_group
                )
                if len(set(gathered)) != 1:
                    raise RuntimeError(
                        f"DFLASH D-Cut cost table differs across ranks for "
                        f"{path!r}; every rank must load the identical file."
                    )
        try:
            payload = json.loads(raw_bytes)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid D-Cut cost table {path!r}: {exc}"
            ) from exc
        self._parse_dcut_cost_table(
            payload, source=path, expected_partition=expected_partition
        )

        self._offline_profiled = True
        self._compute_pin_full_max_bs()
        if self.tp_rank == 0 and self.pp_group.rank_in_group == 0:
            logger.info(
                "DFLASH D-Cut loaded offline cost table %s (md5=%s, %d buckets, "
                "graph_grid_ready=%s).",
                path,
                digest[:8],
                len(self._costs_by_bs),
                graph_ready,
            )

    def _parse_dcut_cost_table(
        self,
        payload: object,
        *,
        source: str,
        expected_partition: Optional[Sequence[int]],
    ) -> None:
        """Validate a cost-table payload and populate the profile tables."""

        def _fail(reason: str) -> None:
            raise RuntimeError(f"invalid D-Cut cost table {source!r}: {reason}")

        if not isinstance(payload, dict):
            _fail("top level must be a JSON object")
        if payload.get("version") != 1:
            _fail("unsupported or missing version (expected 1)")
        if int(payload.get("block_size", -1)) != self.block_size:
            _fail(
                f"block_size mismatch: file={payload.get('block_size')}, "
                f"runtime={self.block_size}"
            )
        if int(payload.get("pp_size", -1)) != self.pp_group.world_size:
            _fail(
                f"pp_size mismatch: file={payload.get('pp_size')}, "
                f"runtime={self.pp_group.world_size}"
            )
        file_partition = payload.get("pp_layer_partition")
        if file_partition is not None and expected_partition is not None:
            if tuple(int(v) for v in file_partition) != tuple(
                int(v) for v in expected_partition
            ):
                _fail(
                    f"pp_layer_partition mismatch: file={list(file_partition)}, "
                    f"runtime={list(expected_partition)}"
                )
        file_ratios = payload.get("ratios")
        if not isinstance(file_ratios, list) or not file_ratios:
            _fail("missing ratios array")
        try:
            column_of = {float(ratio): i for i, ratio in enumerate(file_ratios)}
        except (TypeError, ValueError):
            _fail("ratios must be numbers")
        if set(column_of) != set(_AUTO_RATIOS):
            _fail(
                f"ratios must cover exactly {list(_AUTO_RATIOS)}, "
                f"got {file_ratios}"
            )

        buckets = payload.get("buckets")
        if not isinstance(buckets, dict) or not buckets:
            _fail("missing buckets")
        stage_count = self.pp_group.world_size
        for raw_bs, entry in sorted(buckets.items(), key=lambda item: int(item[0])):
            try:
                bs = int(raw_bs)
            except (TypeError, ValueError):
                _fail(f"bucket key {raw_bs!r} is not an integer")
            if bs <= 0:
                _fail(f"bucket {bs} must be positive")
            stage_costs = (
                entry.get("stage_costs_ms") if isinstance(entry, dict) else None
            )
            if not isinstance(stage_costs, list) or len(stage_costs) != stage_count:
                _fail(f"bucket {bs}: stage_costs_ms must have {stage_count} rows")
            rows: list[tuple[float, ...]] = []
            for stage, row in enumerate(stage_costs):
                if not isinstance(row, list) or len(row) != len(file_ratios):
                    _fail(
                        f"bucket {bs} stage {stage}: expected "
                        f"{len(file_ratios)} ratio columns"
                    )
                values = tuple(float(row[column_of[ratio]]) for ratio in _AUTO_RATIOS)
                if any(not math.isfinite(value) or value <= 0.0 for value in values):
                    _fail(f"bucket {bs} stage {stage}: costs must be positive")
                rows.append(values)
            self._stage_costs_by_bs[bs] = tuple(rows)
            step_costs = (
                entry.get("step_costs_ms") if isinstance(entry, dict) else None
            )
            if step_costs is None:
                # Bottleneck-stage step cost, mirroring _max_parallel_cost_ms.
                step = [
                    max(row[candidate] for row in rows)
                    for candidate in range(len(_AUTO_RATIOS))
                ]
            else:
                if not isinstance(step_costs, list) or len(step_costs) != len(
                    file_ratios
                ):
                    _fail(
                        f"bucket {bs}: step_costs_ms must have "
                        f"{len(file_ratios)} columns"
                    )
                step = [float(step_costs[column_of[ratio]]) for ratio in _AUTO_RATIOS]
                if any(not math.isfinite(value) or value <= 0.0 for value in step):
                    _fail(f"bucket {bs}: step_costs_ms must be positive")
            self._costs_by_bs[bs] = step
            self._offline_keep_counts[bs] = self.candidate_keep_counts(bs)

    def plan(
        self,
        *,
        confidence: torch.Tensor,
        force_full: bool = False,
        budget_cap: Optional[int] = None,
        graph_num_tokens_floor: Optional[int] = None,
        pipeline_mb_id: Optional[int] = None,
    ) -> DFlashDcutPlan:
        """Plan one decode step's D-Cut layout.

        ``budget_cap`` clamps the selected keep count (force_full is exempt);
        the DP-attention caller passes the budget it published in the
        cross-rank tier gather so the realized layout never exceeds it.
        ``graph_num_tokens_floor`` raises the layout's graph token count to the
        caller's DP-aligned tier so every rank selects the same graph key.
        Both default to the pre-DP behavior.
        """
        if confidence.ndim != 2 or confidence.shape[1] != self.gamma:
            raise ValueError(
                "DFLASH D-Cut confidence must have shape [bs, block_size - 1], "
                f"got {tuple(confidence.shape)} for block_size={self.block_size}."
            )
        bs = int(confidence.shape[0])
        full_keep_count = bs * self.gamma
        if force_full:
            plan = self.full_plan(bs=bs)
            self.last_candidate_index = plan.candidate_index
            return plan

        candidate_index: Optional[int]
        if self.is_auto:
            candidate_index = self._select_auto_candidate(
                confidence=confidence,
                bs=bs,
                pipeline_mb_id=pipeline_mb_id,
            )
            keep_count = self.candidate_keep_counts(bs)[candidate_index]
        else:
            candidate_index = None
            keep_count = get_dflash_dcut_keep_count(
                bs=bs,
                block_size=self.block_size,
                ratio=float(self.value),
            )
        graph_num_tokens = self._graph_num_tokens(bs + keep_count)
        if budget_cap is not None and not force_full:
            keep_count = min(keep_count, budget_cap)
        self.last_candidate_index = candidate_index
        if keep_count >= full_keep_count:
            return self.full_plan(bs=bs)

        verify_lens = ScheduleVerifyLensTopk.execute(
            confidence=confidence,
            budget=keep_count,
            cfg=self.schedule_cfg,
        ).to(device=self.device, dtype=torch.int32)
        total_verify_tokens = bs + keep_count
        graph_num_tokens = self._graph_num_tokens(total_verify_tokens)
        if graph_num_tokens_floor is not None:
            graph_num_tokens = max(graph_num_tokens, graph_num_tokens_floor)
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
        )
        return DFlashDcutPlan(
            layout=layout,
            keep_count=keep_count,
            is_compact=True,
            candidate_index=candidate_index,
        )

    def full_plan(self, *, bs: int) -> DFlashDcutPlan:
        if bs <= 0:
            raise ValueError(f"DFLASH D-Cut full plan needs bs > 0, got {bs}.")
        keep_count = bs * self.gamma
        verify_lens = torch.full(
            (bs,), self.block_size, dtype=torch.int32, device=self.device
        )
        graph_num_tokens = self._graph_num_tokens(bs * self.block_size)
        return DFlashDcutPlan(
            layout=RaggedVerifyLayout.from_verify_lens_device(
                verify_lens=verify_lens,
                graph_num_tokens=graph_num_tokens,
            ),
            keep_count=keep_count,
            is_compact=False,
            candidate_index=len(_AUTO_RATIOS) - 1 if self.is_auto else None,
        )

    def plan_from_relay(
        self,
        *,
        verify_lens: torch.Tensor,
        keep_count: int,
        graph_num_tokens: int,
        candidate_index: Optional[int],
    ) -> DFlashDcutPlan:
        if verify_lens.ndim != 1 or verify_lens.numel() == 0:
            raise ValueError(
                "DFLASH D-Cut relayed verify_lens must be a non-empty 1D tensor."
            )
        bs = int(verify_lens.shape[0])
        full_keep_count = bs * self.gamma
        keep_count = int(keep_count)
        graph_num_tokens = int(graph_num_tokens)
        if not 0 <= keep_count <= full_keep_count:
            raise ValueError(
                f"DFLASH D-Cut relayed keep_count={keep_count} is outside "
                f"[0, {full_keep_count}]."
            )
        if self.is_auto:
            if candidate_index is None or not 0 <= candidate_index < len(_AUTO_RATIOS):
                raise ValueError(
                    "DFLASH D-Cut auto relay requires a candidate index in "
                    f"[0, {len(_AUTO_RATIOS) - 1}], got {candidate_index}."
                )
        elif candidate_index is not None:
            raise ValueError(
                "DFLASH D-Cut fixed-ratio relay must not carry a candidate "
                f"index, got {candidate_index}."
            )
        total_verify_tokens = bs + keep_count
        # Ragged CUDA graphs are keyed by the total packed token count.  The
        # capture bucket may be larger than this live batch's full-width block:
        # the runner pads the layout with synthetic rows before replay.  Keep
        # the relay contract strict by requiring the sender and receiver to
        # resolve the same token bucket, rather than incorrectly bounding it by
        # the live rows' full-width capacity.
        expected_graph_num_tokens = self._graph_num_tokens(total_verify_tokens)
        if graph_num_tokens != expected_graph_num_tokens:
            raise ValueError(
                "DFLASH D-Cut relayed graph bucket does not match the "
                "token capture grid: "
                f"total={total_verify_tokens}, graph={graph_num_tokens}, "
                f"expected={expected_graph_num_tokens}."
            )
        verify_lens = verify_lens.to(device=self.device, dtype=torch.int32)
        valid_lens = ((verify_lens >= 1) & (verify_lens <= self.block_size)).all() & (
            verify_lens.sum(dtype=torch.int64) == total_verify_tokens
        )
        torch._assert_async(
            valid_lens,
            "DFLASH D-Cut relayed verify_lens are inconsistent with the "
            "block size or keep count.",
        )
        return DFlashDcutPlan(
            layout=RaggedVerifyLayout.from_verify_lens_device(
                verify_lens=verify_lens,
                graph_num_tokens=graph_num_tokens,
            ),
            keep_count=keep_count,
            is_compact=keep_count < full_keep_count,
            candidate_index=candidate_index,
        )


class DFlashDcutEpilogue:
    """Graph-folded compact top1 scatter for DFlash D-Cut."""

    def __init__(self, *, max_bs: int, block_size: int, device: torch.device) -> None:
        self.max_bs = int(max_bs)
        self.block_size = int(block_size)
        max_tokens = self.max_bs * self.block_size
        self.verify_lens_buf = torch.zeros(
            (self.max_bs,), dtype=torch.int32, device=device
        )
        self.compact_top1 = torch.empty((max_tokens,), dtype=torch.int64, device=device)
        self.strided_top1 = torch.empty(
            (max_tokens, 1), dtype=torch.int64, device=device
        )

    def begin_step(self, verify_lens: torch.Tensor) -> None:
        bs = int(verify_lens.shape[0])
        # A best-effort graph configuration may capture only a prefix of the
        # scheduler's runtime capacity.  Batches above that prefix are routed
        # through the eager D-Cut scatter path, which reads the live layout
        # directly and never consumes these graph scratch buffers.
        if bs > self.max_bs:
            return
        self.verify_lens_buf[:bs].copy_(verify_lens)
        if bs < self.max_bs:
            self.verify_lens_buf[bs:].zero_()

    def capture_hook(self, runner, out, forward_batch, num_tokens: int) -> None:
        if runner.model_runner.is_draft_worker or not runner.ragged_verify_mode:
            return
        if (
            not isinstance(out, LogitsProcessorOutput)
            or out.next_token_logits is None
        ):
            return
        self(
            compact_logits=out.next_token_logits,
            bs=forward_batch.batch_size,
        )

    def __call__(
        self,
        *,
        compact_logits: torch.Tensor,
        bs: int,
    ) -> None:
        n = int(compact_logits.shape[0])
        torch.argmax(compact_logits, dim=-1, out=self.compact_top1[:n])
        verify_lens = self.verify_lens_buf[:bs]
        scatter_compact_to_strided_into(
            compact=self.compact_top1[:n].view(-1, 1),
            verify_lens=verify_lens,
            out=self.strided_top1[: bs * self.block_size],
            stride=self.block_size,
            fill_value=-1,
        )

    def read(self, bs: int) -> torch.Tensor:
        return self.strided_top1[: bs * self.block_size].view(bs, self.block_size)
