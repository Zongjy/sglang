"""Startup-time micro-benchmark of the PP inter-stage transport cost.

When SGLANG_PP_COMM_BENCHMARK=1, every scheduler runs a ping-pong
micro-benchmark over its own pp_group after all process groups are ready
and before the event loop starts serving. For each adjacent pair of PP
ranks (a "hop") it moves a forward-proxy message
``{"hidden_states": num_tokens x hidden_size}`` in the model dtype, sweeps
a few num_tokens points, and fits ``t_hop(num_tokens) = alpha + beta *
num_tokens``. The fitted coefficients and the derived effective bandwidth
are stored on ``scheduler.pp_comm_benchmark_result`` for the PP partition
tuner's cost model, and logged once per hop.
"""

import logging
import statistics
import time
from typing import Dict, List, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Warmup / measured ping-pong iterations per (hop, num_tokens) point.
_NUM_WARMUP_ITERS = 5
_NUM_MEASURE_ITERS = 20
# A healthy single-node NVLink hop sustains a few hundred GB/s; below this
# floor the link likely fell back to PCIe and we warn loudly.
_NVLINK_FLOOR_GBPS = 50.0


def _fit_alpha_beta(
    token_counts: List[float], median_times_ms: List[float]
) -> Tuple[float, float]:
    """Least-squares fit of ``t = alpha + beta * num_tokens``.

    Returns ``(alpha_ms, beta_ms_per_token)``. Plain Python so it stays
    unit-testable without a GPU or numpy.
    """
    if len(token_counts) != len(median_times_ms) or len(token_counts) < 2:
        raise ValueError(
            f"need >= 2 paired points, got {len(token_counts)} token_counts "
            f"and {len(median_times_ms)} times"
        )
    n = len(token_counts)
    sx = sum(token_counts)
    sy = sum(median_times_ms)
    sxx = sum(x * x for x in token_counts)
    sxy = sum(x * y for x, y in zip(token_counts, median_times_ms))
    denom = n * sxx - sx * sx
    if denom == 0:
        raise ValueError("degenerate token_counts: cannot fit a slope")
    beta = (n * sxy - sx * sy) / denom
    alpha = (sy - beta * sx) / n
    return alpha, beta


def _parse_token_counts() -> List[int]:
    raw = envs.SGLANG_PP_COMM_BENCHMARK_TOKENS.get()
    counts = sorted({int(tok.strip()) for tok in raw.split(",") if tok.strip()})
    if len(counts) < 2:
        raise ValueError(
            f"SGLANG_PP_COMM_BENCHMARK_TOKENS needs >= 2 distinct points, got {raw!r}"
        )
    return counts


def _effective_gbps(
    beta_ms_per_token: float, hidden_size: int, element_size: int
) -> float:
    """Effective bandwidth from the per-token slope of the hop latency."""
    if beta_ms_per_token <= 0:
        return 0.0
    bytes_per_token = hidden_size * element_size
    seconds_per_token = beta_ms_per_token * 1e-3
    return bytes_per_token / seconds_per_token / 1e9


def maybe_run_pp_comm_benchmark(scheduler) -> None:
    """Run the PP comm micro-benchmark if enabled; never blocks startup.

    The benchmark is a pp_group collective: every scheduler process must
    reach this call, so the enable/pp_size guards above must evaluate
    identically on all PP ranks (both are static per-launch values).
    """
    scheduler.pp_comm_benchmark_result = None
    if not envs.SGLANG_PP_COMM_BENCHMARK.get():
        return
    if scheduler.ps.pp_size <= 1:
        return
    try:
        scheduler.pp_comm_benchmark_result = _run_pp_comm_benchmark(scheduler)
    except Exception:
        logger.warning(
            "PP comm benchmark failed; continuing startup without it.",
            exc_info=True,
        )


def _run_pp_comm_benchmark(scheduler) -> Dict:
    ps = scheduler.ps
    pp_group = scheduler.pp_group
    pp_rank = pp_group.rank_in_group
    pp_size = pp_group.world_size
    hidden_size = scheduler.model_config.hidden_size
    dtype = scheduler.model_config.dtype
    element_size = torch.tensor([], dtype=dtype).element_size()
    token_counts = _parse_token_counts()

    result = {
        "hidden_size": hidden_size,
        "dtype": str(dtype),
        "token_counts": token_counts,
        "num_warmup_iters": _NUM_WARMUP_ITERS,
        "num_measure_iters": _NUM_MEASURE_ITERS,
        # Timing convention: one-way hop latency, derived as round-trip / 2
        # of a symmetric {"hidden_states"} ping-pong measured on the hop's
        # sending rank (see _benchmark_hop_as_sender).
        "timing_convention": "one_way_ms = round_trip_ms / 2",
        "hops": {},
    }

    # Hops are serialized: during hop r only ranks r and r+1 exchange
    # payloads; every other rank waits at the barrier. The barrier (CPU
    # group) re-syncs all ranks before the next hop, so there is no
    # inter-hop ordering hazard and no ring of pending sends.
    for hop in range(pp_size - 1):
        if pp_rank == hop:
            hop_result = _benchmark_hop_as_sender(
                pp_group, hop, token_counts, hidden_size, dtype, scheduler.device
            )
            result["hops"][f"{hop}->{hop + 1}"] = hop_result
            # Log once per hop: only on the measuring rank, and only for one
            # attn_tp rank so TP duplicates do not repeat the line.
            if ps.attn_tp_rank == 0:
                _log_hop_result(hop, hop_result)
        elif pp_rank == hop + 1:
            _benchmark_hop_as_receiver(
                pp_group, hop, token_counts, hidden_size, dtype, scheduler.device
            )
        pp_group.barrier()

    if ps.attn_tp_rank == 0 and ps.pp_rank == 0:
        logger.info(
            f"[pp_comm_benchmark] done: pp_size={pp_size}, "
            f"hidden_size={hidden_size}, dtype={dtype}, "
            f"token_counts={token_counts}, "
            f"element_size={element_size}B"
        )
    return result


def _make_payload(num_tokens: int, hidden_size: int, dtype, device) -> Dict:
    # Same shape/dtype as the forward proxy sent between PP stages.
    return {
        "hidden_states": torch.empty(
            num_tokens, hidden_size, dtype=dtype, device=device
        )
    }


def _benchmark_hop_as_sender(
    pp_group, hop, token_counts, hidden_size, dtype, device
) -> Dict:
    """Measure hop ``hop -> hop + 1`` from the sending rank's perspective."""
    points: List[Tuple[int, float]] = []
    for num_tokens in token_counts:
        payload = _make_payload(num_tokens, hidden_size, dtype, device)
        # Warmup also primes NCCL channel/buffer setup for this pair and
        # message size, so the measured iterations hit steady state.
        for _ in range(_NUM_WARMUP_ITERS):
            pp_group.send_tensor_dict(payload, dst=hop + 1)
            pp_group.recv_tensor_dict(src=hop + 1)
        samples_ms = []
        for _ in range(_NUM_MEASURE_ITERS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            pp_group.send_tensor_dict(payload, dst=hop + 1)
            pp_group.recv_tensor_dict(src=hop + 1)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1e3
            # The pong reuses the same shape/dtype, so both legs are
            # symmetric and round-trip / 2 approximates the one-way latency
            # of one hidden_states transfer.
            samples_ms.append(elapsed_ms / 2.0)
        points.append((num_tokens, statistics.median(samples_ms)))

    alpha_ms, beta_ms_per_token = _fit_alpha_beta(
        [p[0] for p in points], [p[1] for p in points]
    )
    element_size = torch.tensor([], dtype=dtype).element_size()
    return {
        "alpha_ms": alpha_ms,
        "beta_ms_per_token": beta_ms_per_token,
        "effective_gbps": _effective_gbps(
            beta_ms_per_token, hidden_size, element_size
        ),
        "points": [
            {"num_tokens": n, "median_one_way_ms": t} for n, t in points
        ],
    }


def _benchmark_hop_as_receiver(
    pp_group, hop, token_counts, hidden_size, dtype, device
) -> None:
    """Mirror the sender's iteration structure as the pong side of hop ``hop``."""
    total_iters = _NUM_WARMUP_ITERS + _NUM_MEASURE_ITERS
    for num_tokens in token_counts:
        payload = _make_payload(num_tokens, hidden_size, dtype, device)
        # Strictly alternating recv(ping) -> send(pong): the peer posts its
        # ops in the same matched order, so blocking p2p cannot deadlock.
        for _ in range(total_iters):
            pp_group.recv_tensor_dict(src=hop)
            pp_group.send_tensor_dict(payload, dst=hop)


def _log_hop_result(hop: int, hop_result: Dict) -> None:
    alpha_ms = hop_result["alpha_ms"]
    beta = hop_result["beta_ms_per_token"]
    gbps = hop_result["effective_gbps"]
    logger.info(
        f"[pp_comm_benchmark] hop {hop}->{hop + 1}: "
        f"alpha={alpha_ms:.4f} ms, beta={beta:.3e} ms/token, "
        f"effective_bw={gbps:.1f} GB/s, "
        f"points={hop_result['points']}"
    )
    if 0.0 < gbps < _NVLINK_FLOOR_GBPS:
        logger.warning(
            f"[pp_comm_benchmark] hop {hop}->{hop + 1} effective bandwidth "
            f"{gbps:.1f} GB/s is far below single-node NVLink expectations "
            f"(hundreds of GB/s); the pp_group link likely fell back to "
            f"PCIe. Check NVLink topology / NCCL P2P settings."
        )
