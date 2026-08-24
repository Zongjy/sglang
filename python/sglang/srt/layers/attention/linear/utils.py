from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.runtime_context import get_exec
from sglang.srt.utils.common import rank0_log

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


def ragged_verify_dense_scatter_indices(
    *,
    query_start_loc: torch.Tensor,
    seq_len: int,
    draft_token_num: int,
) -> torch.Tensor:
    """Map packed ragged verify rows into dense ``[bs, draft_token_num]`` slots.

    Tokens outside a capped graph layout collapse into one ghost slot.  Linear
    attention convolution uses the dense view, then gathers only covered rows
    back into the packed target-forward order.
    """
    batch_size = query_start_loc.shape[0] - 1
    token_pos = torch.arange(seq_len, device=query_start_loc.device, dtype=torch.int32)
    token_slots = torch.searchsorted(query_start_loc[1:], token_pos, right=True)
    return (
        token_slots * draft_token_num
        + (token_pos - query_start_loc[token_slots]).to(torch.int64)
    ).clamp_(max=batch_size * draft_token_num)


class LinearAttnKernelBackend(Enum):
    TRITON = "triton"
    CUTEDSL = "cutedsl"
    NV_CUTEDSL = "nv_cutedsl"
    FLASHINFER = "flashinfer"
    FLASHKDA = "flashkda"
    NVIDIA_KDA = "nvidia_kda"
    PTX_KDA = "ptx_kda"
    HELION = "helion"
    CUSTOM = "custom"

    @classmethod
    def _missing_(cls, value):
        return cls.CUSTOM

    def is_triton(self):
        return self == LinearAttnKernelBackend.TRITON

    def is_cutedsl(self):
        return self == LinearAttnKernelBackend.CUTEDSL

    def is_nv_cutedsl(self):
        return self == LinearAttnKernelBackend.NV_CUTEDSL

    def is_flashinfer(self):
        return self == LinearAttnKernelBackend.FLASHINFER

    def is_flashkda(self):
        return self == LinearAttnKernelBackend.FLASHKDA

    def is_nvidia_kda(self):
        return self == LinearAttnKernelBackend.NVIDIA_KDA

    def is_ptx_kda(self):
        return self == LinearAttnKernelBackend.PTX_KDA

    def is_helion(self):
        return self == LinearAttnKernelBackend.HELION

    def is_custom(self):
        return self == LinearAttnKernelBackend.CUSTOM


class LinearAttnBackends(msgspec.Struct, frozen=True):
    """One runner's linear-attn kernel choice, per phase.

    Per runner, not per process: a target and its draft coexist and can want
    different kernels (only the runner whose model is GDN gets the SM100
    FlashInfer prefill default, and an explicit flag applies to whichever runner
    was launched with it).
    """

    decode: LinearAttnKernelBackend
    prefill: LinearAttnKernelBackend
    verify: LinearAttnKernelBackend


_PP_DEFERRED_MAMBA_COMMIT_ALGORITHMS = frozenset(
    {
        "DFLASH",
        "DSPARK",
        "EAGLE",
        "EAGLE3",
        "FROZEN_KV_MTP",
        "STANDALONE",
    }
)


def should_use_request_indexed_verify_scratch(server_args: ServerArgs) -> bool:
    """Whether target-verify scratch must survive a deferred PP commit.

    PP lanes own disjoint request slots, and a lane is not relaunched until its
    previous commit is ordered on the schedule stream. The persistent state is
    therefore protected, but positional scratch rows are shared by concurrently
    executing lanes. Keying scratch by request slot removes that cross-lane race.
    """
    algorithm = (server_args.speculative_algorithm or "").upper()
    return server_args.pp_size > 1 and algorithm in _PP_DEFERRED_MAMBA_COMMIT_ALGORITHMS


def resolve_linear_attn_backends(
    prefill_default: Optional[str] = None,
) -> LinearAttnBackends:
    """This runner's kernel choice from the published leaves.

    ``prefill_default`` is the caller's own auto-default (the SM100 GDN
    domain); an explicitly configured ``--linear-attn-prefill-backend`` wins.
    """
    mamba = get_exec().mamba
    base = mamba.linear_attn_backend
    decode = LinearAttnKernelBackend(mamba.linear_attn_decode_backend or base)
    prefill = LinearAttnKernelBackend(
        mamba.linear_attn_prefill_backend or prefill_default or base
    )

    # Unset verify follows decode (flashinfer -> its recurrent kernel, else triton).
    verify = mamba.linear_attn_verify_backend
    if verify is None:
        verify = decode.value if decode.is_flashinfer() else "triton"

    backends = LinearAttnBackends(
        decode=decode, prefill=prefill, verify=LinearAttnKernelBackend(verify)
    )
    rank0_log(
        f"Linear attention kernel backend: decode={backends.decode.value}, "
        f"prefill={backends.prefill.value}, verify={backends.verify.value}"
    )
    return backends


def build_verify_intermediate_state_indices(
    pool_size: int, server_args: ServerArgs, device
):
    """Per-request row index into the speculative intermediate scratch
    (`intermediate_ssm` / `intermediate_conv_window`) for the MTP /
    target_verify path: request slot i owns scratch row i.

    The scratch is allocated with one extra padding row (the `+1` in
    MambaPool.SpeculativeState, index `pool_size`). Warmup and MLP-sync
    batches can be padded past the pool capacity — under DP attention
    `get_eager_max_batch_size` ceil-aligns the eager warmup bs to attn_tp —
    and the verify kernels index this table positionally up to that padded
    bs. Size the table to the padded maximum and clamp every out-of-pool row
    onto the padding row: pad rows race onto one discard row, which is
    value-irrelevant (same convention as the ragged-verify ghost row).
    """
    import torch

    from sglang.srt.utils.common import get_eager_max_batch_size

    padded_bs = max(get_eager_max_batch_size(server_args, pool_size), pool_size)
    indices = torch.arange(pool_size, dtype=torch.int32, device=device)
    if padded_bs > pool_size:
        indices = torch.cat(
            [
                indices,
                torch.full(
                    (padded_bs - pool_size,),
                    pool_size,
                    dtype=torch.int32,
                    device=device,
                ),
            ]
        )
    return indices
