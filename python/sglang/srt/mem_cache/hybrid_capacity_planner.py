"""Pure memory-capacity helpers for hybrid Mamba/KV models.

This module deliberately has no torch, CUDA, distributed, or runtime-context
dependencies.  The server and offline partition tools can therefore share the
same Mamba pool arithmetic without constructing a ``ModelRunner``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3
MAMBA_CACHE_BASE_RATIO_DROP_ON_SKIP = 1
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP = 2
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY = 1
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP = 1
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_BUFFER = 1


def calculate_mamba_slots_per_request(
    *,
    disable_radix_cache: bool,
    skip_decode_lock: bool,
    extra_buffer: bool,
    extra_buffer_lazy: bool,
    disable_overlap_schedule: bool,
) -> int:
    """Return the Mamba state slots needed by one running request."""
    if disable_radix_cache:
        return 1
    if extra_buffer_lazy and (not extra_buffer or disable_overlap_schedule):
        raise ValueError("lazy Mamba extra buffer requires overlap scheduling")

    base = MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO - (
        MAMBA_CACHE_BASE_RATIO_DROP_ON_SKIP if skip_decode_lock else 0
    )
    additional = 0
    if extra_buffer:
        if disable_overlap_schedule:
            additional = MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP
        elif extra_buffer_lazy:
            additional = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY
        else:
            additional = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP
    elif skip_decode_lock:
        additional = MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_BUFFER
    return base + additional


def solve_auto_mamba_slots(
    *,
    total_available_bytes: int,
    mamba_full_memory_ratio: float,
    state_bytes_per_slot: int,
    slots_per_request: int,
    speculative_draft_tokens: int = 0,
    replay_ring_bytes_per_slot: int = 0,
    replayssm_active: bool = False,
) -> int:
    """Mirror SGLang's ratio-based ``max_mamba_cache_size`` solve.

    The returned value is the number of persistent Mamba state slots, before
    PP ranks synchronize to their minimum.  Padding and speculative
    intermediate buffers are included in the budget equation.
    """
    if total_available_bytes < 0:
        return 0
    if state_bytes_per_slot <= 0:
        return 0
    if slots_per_request <= 0:
        raise ValueError("slots_per_request must be positive")
    if speculative_draft_tokens < 0:
        raise ValueError("speculative_draft_tokens must be non-negative")
    if replay_ring_bytes_per_slot < 0:
        raise ValueError("replay_ring_bytes_per_slot must be non-negative")
    if mamba_full_memory_ratio <= 0:
        raise ValueError("mamba_full_memory_ratio must be positive")

    budget = (
        float(total_available_bytes)
        * float(mamba_full_memory_ratio)
        / (1.0 + float(mamba_full_memory_ratio))
    )
    if speculative_draft_tokens and not replayssm_active:
        draft_tokens = int(speculative_draft_tokens)
        numerator = budget - state_bytes_per_slot * (1 + draft_tokens)
        denominator = state_bytes_per_slot * (
            1 + draft_tokens / slots_per_request
        )
    else:
        per_slot = state_bytes_per_slot + replay_ring_bytes_per_slot
        numerator = budget - per_slot
        denominator = per_slot
    if numerator < 0 or denominator <= 0:
        return 0
    return max(0, int(numerator // denominator))


def _align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class HybridPoolCapacityPlan:
    """Raw memory capacity before a scheduler request limit is applied."""

    mamba_slots_per_rank: tuple[int, ...]
    mamba_slots: Optional[int]
    mamba_request_capacity: Optional[int]
    kv_tokens_per_rank: tuple[Optional[int], ...]
    kv_tokens: Optional[int]
    kv_request_capacity: Optional[int]
    request_capacity_per_rank: tuple[int, ...]
    memory_request_capacity: int
    binding_rank: Optional[int]
    binding_resource: str
    remaining_after_mamba_bytes: tuple[int, ...]
    target_margin_bytes: tuple[int, ...]


def plan_hybrid_pool_capacity(
    *,
    available_bytes_per_rank: Sequence[int],
    kv_bytes_per_token_per_rank: Sequence[int],
    mamba_state_bytes_per_slot: int,
    mamba_full_memory_ratio: float,
    slots_per_request: int,
    kv_tokens_per_request: int,
    page_size: int = 1,
    speculative_draft_tokens: int = 0,
    replay_ring_bytes_per_slot: int = 0,
    replayssm_active: bool = False,
    target_requests: Optional[int] = None,
) -> HybridPoolCapacityPlan:
    """Resolve synchronized Mamba/KV capacities for one PP partition.

    ``available_bytes_per_rank`` is the post-weight pool budget after runtime
    slack and safety reserves.  KV geometry may be zero on GDN-only stages.
    Mamba state uses the largest stage-local GDN geometry, matching SGLang's
    synchronized PP pool sizing.
    """
    available = tuple(int(value) for value in available_bytes_per_rank)
    kv_cells = tuple(int(value) for value in kv_bytes_per_token_per_rank)
    if not available or len(available) != len(kv_cells):
        raise ValueError("available and KV-cell inputs must have one value per rank")
    if any(value < 0 for value in kv_cells):
        raise ValueError("KV bytes per token must be non-negative")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if slots_per_request <= 0:
        raise ValueError("slots_per_request must be positive")
    if target_requests is not None and target_requests <= 0:
        raise ValueError("target_requests must be positive")

    has_mamba = mamba_state_bytes_per_slot > 0
    if has_mamba:
        local_slots = tuple(
            solve_auto_mamba_slots(
                total_available_bytes=max(value, 0),
                mamba_full_memory_ratio=mamba_full_memory_ratio,
                state_bytes_per_slot=mamba_state_bytes_per_slot,
                slots_per_request=slots_per_request,
                speculative_draft_tokens=speculative_draft_tokens,
                replay_ring_bytes_per_slot=replay_ring_bytes_per_slot,
                replayssm_active=replayssm_active,
            )
            for value in available
        )
        synced_slots = min(local_slots)
        mamba_capacity = synced_slots // slots_per_request
        persistent_bytes = (synced_slots + 1) * (
            mamba_state_bytes_per_slot + replay_ring_bytes_per_slot
        )
        if speculative_draft_tokens and not replayssm_active:
            intermediate = tuple(
                mamba_state_bytes_per_slot
                * (local // slots_per_request + 1)
                * speculative_draft_tokens
                for local in local_slots
            )
        else:
            intermediate = (0,) * len(available)
    else:
        local_slots = (0,) * len(available)
        synced_slots = None
        mamba_capacity = None
        persistent_bytes = 0
        intermediate = (0,) * len(available)

    remaining = tuple(
        max(0, budget - persistent_bytes - scratch)
        for budget, scratch in zip(available, intermediate)
    )
    local_kv_tokens: list[Optional[int]] = []
    for budget, cell in zip(remaining, kv_cells):
        if cell == 0:
            local_kv_tokens.append(None)
        else:
            local_kv_tokens.append(_align_down(budget // cell, page_size))
    finite_kv = [value for value in local_kv_tokens if value is not None]
    synced_kv_tokens = min(finite_kv) if finite_kv else None
    if synced_kv_tokens is not None:
        if kv_tokens_per_request <= 0:
            raise ValueError(
                "kv_tokens_per_request must be positive when KV layers are present"
            )
        kv_capacity = synced_kv_tokens // kv_tokens_per_request
    else:
        kv_capacity = None

    finite_capacities = [
        value for value in (mamba_capacity, kv_capacity) if value is not None
    ]
    if not finite_capacities:
        raise ValueError("capacity geometry has neither Mamba state nor KV bytes")
    memory_capacity = min(finite_capacities)

    per_rank_capacity: list[int] = []
    for local_tokens in local_kv_tokens:
        choices = []
        if mamba_capacity is not None:
            choices.append(mamba_capacity)
        if local_tokens is not None:
            choices.append(local_tokens // kv_tokens_per_request)
        # A stage with neither local KV cells nor a Mamba pool is non-binding.
        per_rank_capacity.append(min(choices) if choices else memory_capacity)

    mamba_binding = (
        local_slots.index(min(local_slots)) if mamba_capacity is not None else None
    )
    kv_binding = None
    if synced_kv_tokens is not None:
        kv_binding = local_kv_tokens.index(synced_kv_tokens)
    if mamba_capacity is not None and (
        kv_capacity is None or mamba_capacity < kv_capacity
    ):
        binding_resource = "mamba"
        binding_rank = mamba_binding
    elif kv_capacity is not None and (
        mamba_capacity is None or kv_capacity < mamba_capacity
    ):
        binding_resource = "kv"
        binding_rank = kv_binding
    else:
        binding_resource = "mamba+kv"
        binding_rank = mamba_binding if mamba_binding is not None else kv_binding

    margins: tuple[int, ...] = ()
    if target_requests is not None:
        target_slots = target_requests * slots_per_request
        required_mamba = (
            (target_slots + 1)
            * (mamba_state_bytes_per_slot + replay_ring_bytes_per_slot)
            if has_mamba
            else 0
        )
        if has_mamba and speculative_draft_tokens and not replayssm_active:
            required_mamba += (
                (target_requests + 1)
                * speculative_draft_tokens
                * mamba_state_bytes_per_slot
            )
        target_kv_tokens = (
            _align_up(target_requests * kv_tokens_per_request, page_size)
            if finite_kv
            else 0
        )
        margins = tuple(
            budget - required_mamba - target_kv_tokens * cell
            for budget, cell in zip(available, kv_cells)
        )

    return HybridPoolCapacityPlan(
        mamba_slots_per_rank=local_slots,
        mamba_slots=synced_slots,
        mamba_request_capacity=mamba_capacity,
        kv_tokens_per_rank=tuple(local_kv_tokens),
        kv_tokens=synced_kv_tokens,
        kv_request_capacity=kv_capacity,
        request_capacity_per_rank=tuple(per_rank_capacity),
        memory_request_capacity=memory_capacity,
        binding_rank=binding_rank,
        binding_resource=binding_resource,
        remaining_after_mamba_bytes=remaining,
        target_margin_bytes=margins,
    )
