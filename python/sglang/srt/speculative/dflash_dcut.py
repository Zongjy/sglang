from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal, Optional, Union

import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    scatter_compact_to_strided_into,
)
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
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
    Auto mode builds a hardware-specific cost table at startup by running dummy
    target-verify forwards for a small grid of (batch_size, ratio) points. The
    table is used for all subsequent real steps; if it is missing for a batch
    size we fall back to the 0.75 ratio candidate.
    The mandatory anchors are handled by the keep-count formula, but (matching
    the public implementation) are not added to the selector numerator; this
    avoids over-favoring very shallow cuts when raw DFlash softmax confidence
    is under-calibrated.
    """

    def __init__(
        self,
        *,
        value: DFlashDcutValue,
        block_size: int,
        model_runner,
        device: torch.device,
        tp_rank: int,
    ) -> None:
        self.value = value
        self.block_size = int(block_size)
        self.gamma = self.block_size - 1
        self.model_runner = model_runner
        self.device = device
        self.tp_rank = int(tp_rank)
        self.tp_group = get_tp_group()
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
        self._fold_costs_by_bs: dict[int, list[Optional[float]]] = {}
        self._auto_index_device = torch.zeros((), dtype=torch.int64, device=device)
        self._offline_profiled = False
        self._offline_keep_counts: dict[int, tuple[int, ...]] = {}
        self._warned_missing_profile_bs: set[int] = set()

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
            if match_graph_tier:
                target_tier = self._graph_num_tokens(bs + keep_count)
                for profile_bs, costs in complete.items():
                    profile_keeps = self._offline_keep_counts.get(
                        profile_bs, self.candidate_keep_counts(profile_bs)
                    )
                    profile_tier = self._graph_num_tokens(
                        profile_bs + profile_keeps[candidate_index]
                    )
                    if profile_tier == target_tier:
                        same_tier.append((profile_bs, costs))
                if not same_tier:
                    return None
            choices = same_tier or list(complete.items())
            _, nearest_costs = min(
                choices, key=lambda item: (abs(item[0] - bs), item[0])
            )
            resolved.append(float(nearest_costs[candidate_index]))
        return resolved

    def _profile_costs_for_bs(self, bs: int) -> Optional[list[float]]:
        return self._profile_metric_for_bs(self._costs_by_bs, bs, match_graph_tier=True)

    def _profile_fold_costs_for_bs(self, bs: int) -> Optional[list[float]]:
        # Fold is an eager grid over requests/layers/heads, not a token-keyed
        # CUDA graph. Reuse the nearest profiled batch uniformly across ratios.
        return self._profile_metric_for_bs(
            self._fold_costs_by_bs, bs, match_graph_tier=False
        )

    def _select_auto_candidate_local(self, *, confidence: torch.Tensor, bs: int) -> int:
        costs = self._profile_costs_for_bs(bs)
        if costs is None:
            if bs not in self._warned_missing_profile_bs and self.tp_rank == 0:
                logger.warning(
                    "DFLASH D-Cut offline cost table missing for bs=%d; "
                    "falling back to ratio 0.75.",
                    bs,
                )
                self._warned_missing_profile_bs.add(bs)
            return 2

        survival = torch.cumprod(confidence.to(torch.float32), dim=1).flatten()
        sorted_survival = torch.sort(survival, descending=True).values
        prefix_scores = torch.cumsum(sorted_survival, dim=0)
        expected_tokens = []
        for keep_count in self.candidate_keep_counts(bs):
            draft_score = (
                prefix_scores[keep_count - 1]
                if keep_count > 0
                else prefix_scores.new_zeros(())
            )
            expected_tokens.append(draft_score)
        expected = torch.stack(expected_tokens)
        cost_tensor = torch.tensor(costs, dtype=torch.float32, device=self.device)
        fold_costs = self._profile_fold_costs_for_bs(bs)
        if fold_costs is not None:
            full_commit_tokens = torch.tensor(
                [bs + keep for keep in self.candidate_keep_counts(bs)],
                dtype=torch.float32,
                device=self.device,
            )
            expected_commit_tokens = expected + float(bs)
            fold_tensor = torch.tensor(
                fold_costs, dtype=torch.float32, device=self.device
            )
            cost_tensor += fold_tensor * expected_commit_tokens / full_commit_tokens
        cost_tensor.clamp_min_(torch.finfo(torch.float32).eps)
        return int(torch.argmax(expected / cost_tensor).item())

    def _select_auto_candidate(self, *, confidence: torch.Tensor, bs: int) -> int:
        if self.tp_rank == 0:
            selected = self._select_auto_candidate_local(confidence=confidence, bs=bs)
            self._auto_index_device.fill_(selected)
        self.tp_group.broadcast(self._auto_index_device, src=0)
        return int(self._auto_index_device.item())

    def _graph_num_tokens(self, total_verify_tokens: int) -> int:
        runner = self.model_runner.decode_cuda_graph_runner
        if (
            runner is None
            or not runner.ragged_verify_mode
            or runner.capture_num_tokens is None
            or total_verify_tokens > runner.capture_num_tokens[-1]
        ):
            return total_verify_tokens
        return round_up_grid(total_verify_tokens, runner.capture_num_tokens)

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

    def _max_tp_cost_ms(self, cost_ms: float) -> float:
        cost = torch.tensor(cost_ms, dtype=torch.float32, device=self.device)
        if self.tp_group.world_size > 1:
            torch.distributed.all_reduce(
                cost,
                op=torch.distributed.ReduceOp.MAX,
                group=self.tp_group.device_group,
            )
        return float(cost.item())

    def _raise_if_tp_profile_failed(self, error: Optional[Exception]) -> None:
        if self.tp_group.world_size == 1:
            if error is not None:
                raise error
            return

        failed = torch.tensor(
            int(error is not None), dtype=torch.int32, device=self.device
        )
        torch.distributed.all_reduce(
            failed,
            op=torch.distributed.ReduceOp.MAX,
            group=self.tp_group.device_group,
        )
        if int(failed.item()) != 0:
            if error is not None:
                raise RuntimeError(
                    f"DFLASH D-Cut profiling failed on TP rank {self.tp_rank}: {error}"
                ) from error
            raise RuntimeError("DFLASH D-Cut profiling failed on another TP rank.")

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
        )

    def _profile_dcut_cost_ms(self, *, bs: int, keep_count: int) -> tuple[float, float]:
        """Average target-verify and full-prefix ReplaySSM fold latency."""
        verify_cost_ms = 0.0
        fold_cost_ms = 0.0
        preflight_error = None
        forward_batch = None
        prepared_slots = None
        try:
            forward_batch = self._build_profile_forward_batch(
                bs=bs, keep_count=keep_count
            )
            prepared_slots = self._prepare_replayssm_profile_slots(
                forward_batch.req_pool_indices
            )
        except Exception as exc:
            preflight_error = exc
        # Every rank reaches this checkpoint before any model collective. Local
        # forward/CUDA failures after it are not recoverable and propagate.
        self._raise_if_tp_profile_failed(preflight_error)
        assert forward_batch is not None

        with self._use_replayssm_profile_slots(prepared_slots):
            torch.get_device_module(self.device).synchronize()
            for _ in range(_OFFLINE_PROFILE_WARMUPS):
                self.model_runner.forward(forward_batch)
            torch.get_device_module(self.device).synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(_OFFLINE_PROFILE_STEPS):
                self.model_runner.forward(forward_batch)
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
        return self._max_tp_cost_ms(verify_cost_ms), self._max_tp_cost_ms(fold_cost_ms)

    def _get_dcut_profile_batch_sizes(self) -> tuple[int, ...]:
        """Batch sizes to profile, derived from captured graph token buckets."""
        runner = self.model_runner.decode_cuda_graph_runner
        pool_max_bs = (
            self.model_runner.req_to_token_pool.size
            if self.model_runner.req_to_token_pool is not None
            else 1
        )
        runtime_max_bs = int(
            getattr(self.model_runner, "max_running_requests", 0) or 0
        )
        # In the all-eager fallback there is no graph tier to derive a profile
        # shape from.  Profile the requested runtime capacity instead of the
        # (often much larger) request-pool allocation.
        max_bs = (
            runtime_max_bs
            if runner is None and runtime_max_bs > 0
            else pool_max_bs
        )
        if runner is None or runner.capture_num_tokens is None:
            return (min(max_bs, _OFFLINE_PROFILE_MAX_BS),)
        sizes = []
        for num_tokens in runner.capture_num_tokens:
            if num_tokens % self.block_size == 0:
                bs = num_tokens // self.block_size
                if 0 < bs <= min(max_bs, _OFFLINE_PROFILE_MAX_BS):
                    sizes.append(bs)
        if not sizes:
            sizes.append(min(max_bs, _OFFLINE_PROFILE_MAX_BS))
        return tuple(sorted(set(sizes)))

    def profile_dcut_cost_table(self) -> None:
        """Offline warm-up: build the auto-mode cost table."""
        if not self.is_auto or self._offline_profiled:
            return
        if self.model_runner.decode_cuda_graph_runner is None:
            # There is no graph-shaped cost to calibrate in the all-eager
            # fallback.  Let _select_auto_candidate use its documented .75
            # fallback instead of launching a potentially oversized profile.
            self._offline_profiled = True
            if self.tp_rank == 0:
                logger.info(
                    "DFLASH D-Cut auto: decode CUDA graph disabled; "
                    "using ratio 0.75 without offline profiling."
                )
            return

        try:
            profile_bs_list = self._get_dcut_profile_batch_sizes()
            costs_by_bs: dict[int, list[tuple[int, float, float]]] = {}
            for bs in profile_bs_list:
                keep_counts = self.candidate_keep_counts(bs)
                self._offline_keep_counts[bs] = keep_counts
                entries: list[tuple[int, float, float]] = []
                for keep_count in keep_counts:
                    verify_cost, fold_cost = self._profile_dcut_cost_ms(
                        bs=bs, keep_count=keep_count
                    )
                    entries.append((keep_count, verify_cost, fold_cost))
                costs_by_bs[bs] = entries
                self._costs_by_bs[bs] = [verify for _, verify, _ in entries]
                self._fold_costs_by_bs[bs] = [fold for _, _, fold in entries]

            self._offline_profiled = True
            if self.tp_rank == 0 and costs_by_bs:
                logger.info(
                    "DFLASH D-Cut offline cost table ready: block_size=%d %s",
                    self.block_size,
                    {
                        bs: [
                            (keep, round(verify, 4), round(fold, 4))
                            for keep, verify, fold in costs_by_bs[bs]
                        ]
                        for bs in sorted(costs_by_bs)
                    },
                )
        except Exception as e:
            self._offline_profiled = True
            if self.tp_group.world_size > 1:
                raise RuntimeError(
                    "DFLASH D-Cut offline profiling failed under tensor "
                    "parallelism; aborting startup because a rank-local fallback "
                    "cannot be coordinated safely."
                ) from e
            if self.tp_rank == 0:
                logger.warning(
                    "DFLASH D-Cut offline profiling failed (%s); "
                    "auto mode will fall back to ratio 0.75.",
                    e,
                )

    def plan(
        self,
        *,
        confidence: torch.Tensor,
        force_full: bool = False,
    ) -> DFlashDcutPlan:
        if confidence.ndim != 2 or confidence.shape[1] != self.gamma:
            raise ValueError(
                "DFLASH D-Cut confidence must have shape [bs, block_size - 1], "
                f"got {tuple(confidence.shape)} for block_size={self.block_size}."
            )
        bs = int(confidence.shape[0])
        full_keep_count = bs * self.gamma
        candidate_index: Optional[int]
        if force_full:
            keep_count = full_keep_count
            candidate_index = len(_AUTO_RATIOS) - 1 if self.is_auto else None
        elif self.is_auto:
            candidate_index = self._select_auto_candidate(confidence=confidence, bs=bs)
            keep_count = self.candidate_keep_counts(bs)[candidate_index]
        else:
            candidate_index = None
            keep_count = get_dflash_dcut_keep_count(
                bs=bs,
                block_size=self.block_size,
                ratio=float(self.value),
            )
        self.last_candidate_index = candidate_index

        verify_lens = ScheduleVerifyLensTopk.execute(
            confidence=confidence,
            budget=keep_count,
            cfg=self.schedule_cfg,
        ).to(device=self.device, dtype=torch.int32)
        total_verify_tokens = bs + keep_count
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=verify_lens,
            graph_num_tokens=self._graph_num_tokens(total_verify_tokens),
        )
        return DFlashDcutPlan(
            layout=layout,
            keep_count=keep_count,
            is_compact=keep_count < full_keep_count,
            candidate_index=candidate_index,
        )


class DFlashDcutEpilogue:
    """Graph-folded compact top1 + hidden-state scatter for DFlash D-Cut."""

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
        self.strided_hidden: Optional[torch.Tensor] = None

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

    def _ensure_hidden(self, compact_hidden: torch.Tensor) -> torch.Tensor:
        if (
            self.strided_hidden is not None
            and self.strided_hidden.shape[1] == compact_hidden.shape[1]
            and self.strided_hidden.dtype == compact_hidden.dtype
        ):
            return self.strided_hidden
        assert (
            not torch.cuda.is_current_stream_capturing()
        ), "DFlashDcutEpilogue buffers must be allocated during graph warmup"
        self.strided_hidden = torch.empty(
            (self.max_bs * self.block_size, compact_hidden.shape[1]),
            dtype=compact_hidden.dtype,
            device=compact_hidden.device,
        )
        return self.strided_hidden

    def capture_hook(self, runner, out, forward_batch, num_tokens: int) -> None:
        if runner.model_runner.is_draft_worker or not runner.ragged_verify_mode:
            return
        if (
            not isinstance(out, LogitsProcessorOutput)
            or out.next_token_logits is None
            or out.hidden_states is None
        ):
            return
        self(
            compact_logits=out.next_token_logits,
            compact_hidden=out.hidden_states,
            bs=forward_batch.batch_size,
        )

    def __call__(
        self,
        *,
        compact_logits: torch.Tensor,
        compact_hidden: torch.Tensor,
        bs: int,
    ) -> None:
        n = int(compact_logits.shape[0])
        torch.argmax(compact_logits, dim=-1, out=self.compact_top1[:n])
        hidden_out = self._ensure_hidden(compact_hidden)
        verify_lens = self.verify_lens_buf[:bs]
        scatter_compact_to_strided_into(
            compact=self.compact_top1[:n].view(-1, 1),
            verify_lens=verify_lens,
            out=self.strided_top1[: bs * self.block_size],
            stride=self.block_size,
            fill_value=-1,
        )
        scatter_compact_to_strided_into(
            compact=compact_hidden,
            verify_lens=verify_lens,
            out=hidden_out[: bs * self.block_size],
            stride=self.block_size,
            fill_value=0.0,
        )

    def read(self, bs: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.strided_hidden is not None
        return (
            self.strided_top1[: bs * self.block_size].view(bs, self.block_size),
            self.strided_hidden[: bs * self.block_size],
        )
