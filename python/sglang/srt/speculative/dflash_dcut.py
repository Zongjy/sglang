from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal, Optional, Union

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

    def _profile_costs_for_bs(self, bs: int) -> Optional[list[float]]:
        exact = self._costs_by_bs.get(bs)
        if exact is not None and all(cost is not None for cost in exact):
            return [float(cost) for cost in exact]
        larger = sorted(
            profile_bs for profile_bs in self._costs_by_bs if profile_bs >= bs
        )
        for profile_bs in larger:
            costs = self._costs_by_bs[profile_bs]
            if all(cost is not None for cost in costs):
                return [float(cost) for cost in costs]
        return None

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
        max_bs = (
            self.model_runner.req_to_token_pool.size
            if self.model_runner.req_to_token_pool is not None
            else total_verify_tokens
        )
        num_slots = min(total_verify_tokens, max_bs)
        verify_lens_cpu = build_capture_verify_lens(
            num_tokens=total_verify_tokens,
            num_slots=num_slots,
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
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device=self.device)

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

    def _profile_dcut_cost_ms(self, *, bs: int, keep_count: int) -> float:
        """Average target-verify latency for one (bs, keep_count) point."""
        forward_batch = self._build_profile_forward_batch(
            bs=bs, keep_count=keep_count
        )
        torch.get_device_module(self.device).synchronize()
        self.tp_group.barrier()
        for _ in range(_OFFLINE_PROFILE_WARMUPS):
            self.model_runner.forward(forward_batch)
        torch.get_device_module(self.device).synchronize()
        self.tp_group.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(_OFFLINE_PROFILE_STEPS):
            self.model_runner.forward(forward_batch)
        end.record()
        torch.cuda.synchronize()
        self.tp_group.barrier()
        return start.elapsed_time(end) / _OFFLINE_PROFILE_STEPS

    def _get_dcut_profile_batch_sizes(self) -> tuple[int, ...]:
        """Batch sizes to profile, derived from captured graph token buckets."""
        runner = self.model_runner.decode_cuda_graph_runner
        max_bs = (
            self.model_runner.req_to_token_pool.size
            if self.model_runner.req_to_token_pool is not None
            else 1
        )
        if runner is None or runner.capture_num_tokens is None:
            return (max_bs,)
        sizes = []
        for num_tokens in runner.capture_num_tokens:
            if num_tokens % self.block_size == 0:
                bs = num_tokens // self.block_size
                if 0 < bs <= max_bs:
                    sizes.append(bs)
        sizes.append(max_bs)
        return tuple(sorted(set(sizes)))

    def profile_dcut_cost_table(self) -> None:
        """Offline warm-up: build the auto-mode cost table."""
        if not self.is_auto or self._offline_profiled:
            return

        try:
            profile_bs_list = self._get_dcut_profile_batch_sizes()
            costs_by_bs: dict[int, list[tuple[int, float]]] = {}
            for bs in profile_bs_list:
                keep_counts = self.candidate_keep_counts(bs)
                self._offline_keep_counts[bs] = keep_counts
                entries: list[tuple[int, float]] = []
                for keep_count in keep_counts:
                    cost = self._profile_dcut_cost_ms(bs=bs, keep_count=keep_count)
                    entries.append((keep_count, cost))
                costs_by_bs[bs] = entries
                self._costs_by_bs[bs] = [cost for _, cost in entries]

            self._offline_profiled = True
            if self.tp_rank == 0 and costs_by_bs:
                logger.info(
                    "DFLASH D-Cut offline cost table ready: block_size=%d %s",
                    self.block_size,
                    {
                        bs: [
                            (keep, round(cost, 4)) for keep, cost in costs_by_bs[bs]
                        ]
                        for bs in sorted(costs_by_bs)
                    },
                )
        except Exception as e:
            if self.tp_rank == 0:
                logger.warning(
                    "DFLASH D-Cut offline profiling failed (%s); "
                    "auto mode will fall back to ratio 0.75.",
                    e,
                )
            self._offline_profiled = True

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
