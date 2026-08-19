#!/usr/bin/env python3
"""Capacity model for PP partitions: memory pool -> mamba slots -> max BS.

A PP partition decides two balances: compute balance (stage time, see
``stage_model``) and capacity balance (per-stage memory pool -> mamba slots
K -> max running batch size, globally the minimum).  This module predicts
the capacity side for any candidate partition without starting a server.

Model (verified against six measured partitions on 2xRTX4090 / Qwen3.6-27B /
BF16 / mem_fraction_static=0.82 / mamba_full_memory_ratio=2.0 / DFlash b=8):

- ``avail_r(p)``    = pre-load free mem - weights_r(p) - draft weights (last
                      rank) + per-rank correction calibrated from the
                      baseline server log.  Layer weights come from the HF
                      safetensors index (exact per-layer bytes; the visual
                      tower loads on the last rank).  Fallback: uniform
                      per-layer weight from the baseline log + warning.
- ``slack``         = pre_load_mem * (1 - mem_fraction_static).  Verified:
                      K of 37/27, 38/26, 39/25 reproduce exactly.
- mamba budget      = rest * ratio / (1 + ratio), ratio =
                      mamba_full_memory_ratio.  Semantics per
                      ``kv_cache_configurator.py`` (_handle_max_mamba_cache):
                      "ratio:1" split of the rest memory between the mamba
                      pool and everything else.
- ``K_r``           = joint solve with the spec intermediate buffer
                      (D = speculative_num_draft_tokens = DFlash block size,
                      slots_per_request = 4 from the runtime log):
                      K = (budget - per_req*(1+D)) // (per_req*(1+D/slots)),
                      per_req = max_stage_gdn_layers(p) * GDN_SLOT_BYTES.
                      ``max_stage`` mirrors the runtime, which charges the
                      largest stage share so every rank derives the same
                      pool without a collective.  ``K = min_r K_r`` (PP sync
                      allreduce MIN).
- last-rank reserve: the last rank additionally hosts the speculative
                      draft.  Calibrated from the baseline log:
                      ``r_spec_k``  (~2.6 GiB, DFlash aux hidden states,
                      reserved before the mamba solve) and ``r_spec_kv``
                      (~3.5 GiB = aux + draft KV pool; the delta matches the
                      physical draft KV 49182 x 20480 B = 0.94 GiB exactly).
- KV tokens         = (restKV_r - (K+1)*per_req - (capped+1)*D*per_req -
                      graph_overhang) / kv_bytes_per_token_r, min over ranks.
                      ``capped`` = min(resolved_max_running, K // slots).
                      graph_overhang models the post-capture KV resize
                      (``kv_pool_runtime.py`` compute_post_capture_kv_resize)
                      when the uncapped CUDA-graph range (bs <= K//slots)
                      grows past the headroom; calibrated, 0 for bs_cap <= 9.
- ``BS_max(p)``     = min(K // slots_per_request, cuda_graph_max_bs).

Ground-truth cross-check (user-measured, uncapped runs): K within +-1 on
all six partitions, BS_max exact, KV tokens within +-5% on 32/32, 37/27,
38/26, 39/25.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


GIB = 1 << 30

# Regexes for the baseline server log (calibration sources).
LOAD_BEGIN = re.compile(r"PP(\d+)] Load weight begin\. avail mem=([0-9.]+) GB")
LOAD_END = re.compile(
    r"PP(\d+)] Load weight end\..*?type=([^,]+),.*?avail mem=([0-9.]+) GB, "
    r"mem usage=([0-9.]+) GB"
)
MAMBA_ALLOC = re.compile(r"Mamba Cache is allocated\.\s*max_mamba_cache_size: (\d+)")
MAMBA_CAPPED = re.compile(
    r"max_running_requests is capped to (\d+) by the mamba state cache "
    r"\(max_mamba_cache_size=(\d+), (\d+) state slots per request\)"
)
MAX_TOKENS = re.compile(r"max_total_num_tokens=(\d+)")
SERVER_ARGS = re.compile(r"server_args=ServerArgs\(")
# Resolved default max_running_requests from the ServerArgs dump (DFlash
# defaults to 48); it caps BS_max even when the mamba pool allows more.
SERVER_ARGS_MAX_RUNNING = re.compile(
    r"server_args=ServerArgs\(.+?max_running_requests=(\d+|None)", re.DOTALL
)

_DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "I64": 8, "I32": 4, "I16": 2, "U8": 1, "BOOL": 1,
}


class CapacityModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelStaticInfo:
    """Static per-model constants from the HF config + safetensors headers."""

    num_layers: int
    layer_types: tuple[str, ...]  # "linear_attention" (GDN) | "full_attention"
    layer_weight_bytes: tuple[int, ...]  # exact per-layer weight bytes
    first_rank_extra_bytes: int  # embed_tokens + mtp (loaded on PP rank 0)
    last_rank_extra_bytes: int  # lm_head + norm + visual tower (last rank)
    gdn_slot_bytes_per_layer: int  # mamba conv+ssm state per GDN layer / slot
    kv_bytes_per_token_per_full_layer: int
    draft_weight_bytes: int | None
    draft_kv_bytes_per_token: int | None
    weight_source: str

    def gdn_layers(self, start: int, end: int) -> int:
        return sum(
            1 for t in self.layer_types[start:end] if t != "full_attention"
        )

    def full_layers(self, start: int, end: int) -> int:
        return sum(
            1 for t in self.layer_types[start:end] if t == "full_attention"
        )

    def rank_weight_bytes(self, start: int, end: int, pp_size: int) -> tuple[int, int]:
        """(first-rank weight, last-rank weight) totals for [start, end)."""
        weights = sum(self.layer_weight_bytes[start:end])
        first = weights + self.first_rank_extra_bytes
        last = weights + self.last_rank_extra_bytes
        return first, last


def _safetensors_header_sizes(snapshot: Path) -> dict[str, int]:
    """Tensor name -> bytes for every shard of a local HF snapshot."""
    sizes: dict[str, int] = {}
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = sorted(path.name for path in snapshot.glob("*.safetensors"))
    for shard in shards:
        with (snapshot / shard).open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            numel = 1
            for dim in info["shape"]:
                numel *= dim
            sizes[name] = numel * _DTYPE_BYTES[info["dtype"]]
    return sizes


def _hf_snapshot(model_path: str) -> Path | None:
    """Resolve a HF hub id to the local cache snapshot dir, or None."""
    if Path(model_path).is_dir():
        return Path(model_path)
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_path, local_files_only=True))
    except Exception:
        return None


def load_static_info(
    model_path: str,
    draft_model_path: str | None = None,
    fallback_layer_weight_bytes: int | None = None,
) -> ModelStaticInfo:
    """Build the static model info from the local HF cache.

    Falls back to a uniform per-layer weight (from the baseline log) with a
    warning recorded in ``weight_source`` when the safetensors headers are
    not readable.
    """
    snapshot = _hf_snapshot(model_path)
    if snapshot is None:
        raise CapacityModelError(
            f"model {model_path!r} is not available in the local HF cache."
        )
    config = json.loads((snapshot / "config.json").read_text())
    text_config = config.get("text_config", config)
    layer_types = tuple(str(t) for t in text_config["layer_types"])
    num_layers = int(text_config["num_hidden_layers"])
    if len(layer_types) != num_layers:
        raise CapacityModelError("layer_types does not match num_hidden_layers.")

    # Mamba (GDN) per-slot state bytes per layer: conv + ssm, both bf16.
    dtype_bytes = 2  # --mamba-ssm-dtype bfloat16 / conv default bf16
    ssm_numel = (
        text_config["linear_num_value_heads"]
        * text_config["linear_key_head_dim"]
        * text_config["linear_value_head_dim"]
    )
    conv_dim = (
        2 * text_config["linear_num_key_heads"] * text_config["linear_key_head_dim"]
        + text_config["linear_num_value_heads"] * text_config["linear_value_head_dim"]
    )
    conv_numel = conv_dim * (text_config["linear_conv_kernel_dim"] - 1)
    gdn_slot_bytes = (ssm_numel + conv_numel) * dtype_bytes

    kv_bytes = (
        2  # K + V
        * text_config["num_key_value_heads"]
        * text_config["head_dim"]
        * dtype_bytes
    )

    draft_weight_bytes = None
    draft_kv_bytes = None
    if draft_model_path:
        draft_snapshot = _hf_snapshot(draft_model_path)
        if draft_snapshot is not None:
            draft_sizes = _safetensors_header_sizes(draft_snapshot)
            draft_weight_bytes = sum(draft_sizes.values())
            draft_config = json.loads((draft_snapshot / "config.json").read_text())
            draft_kv_bytes = (
                2
                * int(draft_config["num_hidden_layers"])
                * int(draft_config["num_key_value_heads"])
                * int(draft_config["head_dim"])
                * dtype_bytes
            )

    try:
        sizes = _safetensors_header_sizes(snapshot)
        layer_bytes = [0] * num_layers
        first_extra = 0
        last_extra = 0
        for name, nbytes in sizes.items():
            if ".layers." in name:
                layer_id = int(name.split(".layers.")[1].split(".")[0])
                layer_bytes[layer_id] += nbytes
            elif "visual" in name or "lm_head" in name or name.endswith("norm.weight"):
                # The visual tower and the head load on the last PP rank
                # (verified: baseline per-rank weight logs differ from the
                # static sums by exactly the visual tower).
                last_extra += nbytes
            else:
                first_extra += nbytes
        weight_source = "safetensors headers (exact per-layer)"
    except (OSError, KeyError, ValueError) as exc:
        if fallback_layer_weight_bytes is None:
            raise CapacityModelError(
                f"cannot read safetensors headers for {model_path!r}: {exc}"
            ) from exc
        layer_bytes = [fallback_layer_weight_bytes] * num_layers
        first_extra = 0
        last_extra = 0
        weight_source = (
            "uniform per-layer weight from the baseline log "
            "(safetensors unavailable; cross-rank layer-type weight "
            "differences are ignored)"
        )

    return ModelStaticInfo(
        num_layers=num_layers,
        layer_types=layer_types,
        layer_weight_bytes=tuple(layer_bytes),
        first_rank_extra_bytes=first_extra,
        last_rank_extra_bytes=last_extra,
        gdn_slot_bytes_per_layer=gdn_slot_bytes,
        kv_bytes_per_token_per_full_layer=kv_bytes,
        draft_weight_bytes=draft_weight_bytes,
        draft_kv_bytes_per_token=draft_kv_bytes,
        weight_source=weight_source,
    )


@dataclass(frozen=True)
class CapacityCalibration:
    """Constants calibrated from the baseline server log."""

    pre_avail_gib: float  # free GPU memory before weight load
    slack_gib: float  # pre_avail * (1 - mem_fraction_static)
    weight_adjust_gib: tuple[float, ...]  # per-rank log-vs-static correction
    draft_weight_gib: float  # 0 when unknown
    slots_per_request: int  # mamba state slots per request (4)
    resolved_max_running: int  # server default max_running_requests (48)
    r_spec_k_gib: float  # last-rank reserve before the mamba solve
    r_spec_kv_gib: float  # last-rank reserve before target KV sizing
    graph_overhang_gib: float  # per bs_cap step above GRAPH_OVERHANG_BASE
    baseline_k: int
    baseline_kv_tokens: int
    cuda_graph_max_bs: int


# Graph overhang: the post-capture KV resize (kv_pool_runtime.py) shrinks the
# pool when the uncapped graph range bs <= K//slots eats past the headroom.
# Calibrated: 37/27 (bs_cap 10) needs ~0.4 GiB, 38/26 and below need none.
GRAPH_OVERHANG_BASE = 9


def _parse_weight_logs(text: str, pp_size: int) -> dict[str, Any]:
    """Extract per-rank weight-load and capacity facts from a server log."""
    pre_avail: list[float] = []
    target_usage: dict[int, float] = {}
    post_avail: dict[int, float] = {}
    draft_usage = 0.0
    for rank_s, avail in LOAD_BEGIN.findall(text):
        pre_avail.append(float(avail))
    for rank_s, model_type, avail, usage in LOAD_END.findall(text):
        rank = int(rank_s)
        if "DFlash" in model_type or "Draft" in model_type:
            draft_usage = float(usage)
            post_avail[rank] = float(avail)
        else:
            target_usage[rank] = float(usage)
            post_avail[rank] = float(avail)
    k_values = [int(v) for v in MAMBA_ALLOC.findall(text)]
    capped = MAMBA_CAPPED.findall(text)
    tokens = [int(v) for v in MAX_TOKENS.findall(text)]
    resolved_max_running = None
    args_match = SERVER_ARGS_MAX_RUNNING.search(text)
    if args_match and args_match.group(1) != "None":
        resolved_max_running = int(args_match.group(1))
    return {
        "pre_avail": pre_avail[0] if pre_avail else None,
        "target_usage": target_usage,
        "post_avail": post_avail,
        "draft_usage": draft_usage,
        "k": min(k_values) if k_values else (int(capped[0][1]) if capped else None),
        "slots_per_request": int(capped[0][2]) if capped else 4,
        "kv_tokens": min(tokens) if tokens else None,
        "resolved_max_running": resolved_max_running,
    }


def calibrate(
    baseline_log_text: str,
    static: ModelStaticInfo,
    baseline_partition: Sequence[int],
    mem_fraction_static: float,
    mamba_full_memory_ratio: float,
    draft_tokens: int,
    cuda_graph_max_bs: int,
    resolved_max_running: int | None = None,
) -> CapacityCalibration:
    """Calibrate per-rank corrections and last-rank reserves from a log.

    ``resolved_max_running`` is the server's default max_running_requests
    (DFlash defaults to 48); when None it is parsed from the ServerArgs dump
    in the log, falling back to 48.
    """
    partition = tuple(baseline_partition)
    pp_size = len(partition)
    facts = _parse_weight_logs(baseline_log_text, pp_size)
    if facts["pre_avail"] is None or facts["k"] is None or facts["kv_tokens"] is None:
        raise CapacityModelError(
            "baseline log is missing weight-load / mamba-cache / token lines."
        )
    pre_avail = facts["pre_avail"]
    slack = pre_avail * (1.0 - mem_fraction_static)
    draft_gib = facts["draft_usage"]
    slots = facts["slots_per_request"]
    if resolved_max_running is None:
        resolved_max_running = facts["resolved_max_running"] or 48

    # Per-rank weight correction: logged usage vs static safetensors sums.
    adjust: list[float] = []
    start = 0
    for rank, count in enumerate(partition):
        static_gib = (
            sum(static.layer_weight_bytes[start : start + count])
            + (static.first_rank_extra_bytes if rank == 0 else 0)
            + (static.last_rank_extra_bytes if rank == pp_size - 1 else 0)
        ) / GIB
        logged = facts["target_usage"].get(rank)
        adjust.append((logged - static_gib) if logged is not None else 0.0)
        start += count

    # avail_r(p) model: pre_avail - logged_weight_r(p) - draft (last rank).
    # With the correction absorbed, avail_r = pre_avail - static_r - adjust_r
    # - draft.  Solve r_spec_k so the baseline K reproduces, and r_spec_kv so
    # the baseline KV token count reproduces (last rank binds in both).
    last = pp_size - 1
    max_stage_gdn = max(
        static.gdn_layers(start, start + count)
        for start, count in zip(
            _partition_starts(partition), partition
        )
    )
    per_req = max_stage_gdn * static.gdn_slot_bytes_per_layer
    ratio = mamba_full_memory_ratio
    avail_last = (
        pre_avail
        - facts["target_usage"].get(last, 0.0)
        - draft_gib
    )
    # K = (budget - per_req*(1+D)) // (per_req*(1+D/slots)) with
    # budget = (avail - slack - r_spec_k) * ratio/(1+ratio); take the K
    # interval midpoint so the estimate sits inside the floor() band.
    denom = per_req * (1.0 + draft_tokens / slots)
    numer = per_req * (1.0 + draft_tokens)
    budget_mid = ((facts["k"] + 0.5) * denom + numer) / GIB
    rest_mid = budget_mid * (1.0 + ratio) / ratio
    r_spec_k = avail_last - slack - rest_mid

    # KV: tokens = (rest_kv - (K+1)*per_req - (capped+1)*D*per_req) / kv_bytes.
    capped = min(resolved_max_running, facts["k"] // slots)
    mamba_main = (facts["k"] + 1) * per_req / GIB
    intermediate = (capped + 1) * draft_tokens * per_req / GIB
    kv_bytes = static.kv_bytes_per_token_per_full_layer * static.full_layers(
        _partition_starts(partition)[last], sum(partition)
    )
    kv_budget = facts["kv_tokens"] * kv_bytes / GIB
    r_spec_kv = avail_last - slack - mamba_main - intermediate - kv_budget

    return CapacityCalibration(
        pre_avail_gib=pre_avail,
        slack_gib=slack,
        weight_adjust_gib=tuple(adjust),
        draft_weight_gib=draft_gib,
        slots_per_request=slots,
        resolved_max_running=resolved_max_running,
        r_spec_k_gib=r_spec_k,
        r_spec_kv_gib=r_spec_kv,
        graph_overhang_gib=0.4,
        baseline_k=facts["k"],
        baseline_kv_tokens=facts["kv_tokens"],
        cuda_graph_max_bs=cuda_graph_max_bs,
    )


def _partition_starts(partition: Sequence[int]) -> tuple[int, ...]:
    starts = []
    start = 0
    for count in partition:
        starts.append(start)
        start += count
    return tuple(starts)


@dataclass(frozen=True)
class CapacityEstimate:
    partition: tuple[int, ...]
    gdn_per_rank: tuple[int, ...]
    full_per_rank: tuple[int, ...]
    avail_gib: tuple[float, ...]
    k_per_rank: tuple[int, ...]
    k: int
    k_binding_rank: int
    kv_tokens_per_rank: tuple[int, ...]
    kv_tokens: int
    kv_binding_rank: int
    bs_max: int
    capped_requests: int
    ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": list(self.partition),
            "gdn_per_rank": list(self.gdn_per_rank),
            "full_per_rank": list(self.full_per_rank),
            "avail_gib": [round(v, 3) for v in self.avail_gib],
            "k_per_rank": list(self.k_per_rank),
            "k": self.k,
            "k_binding_rank": self.k_binding_rank,
            "kv_tokens_per_rank": list(self.kv_tokens_per_rank),
            "kv_tokens": self.kv_tokens,
            "kv_binding_rank": self.kv_binding_rank,
            "bs_max": self.bs_max,
            "capped_requests": self.capped_requests,
            "ratio": self.ratio,
        }


def predict_capacity(
    partition: Sequence[int],
    static: ModelStaticInfo,
    calib: CapacityCalibration,
    mamba_full_memory_ratio: float,
    draft_tokens: int,
) -> CapacityEstimate:
    """Predict K, KV tokens and BS_max for one candidate partition."""
    partition = tuple(int(c) for c in partition)
    pp_size = len(partition)
    starts = _partition_starts(partition)
    last = pp_size - 1
    gdn = tuple(
        static.gdn_layers(start, start + count)
        for start, count in zip(starts, partition)
    )
    full = tuple(
        static.full_layers(start, start + count)
        for start, count in zip(starts, partition)
    )
    max_stage_gdn = max(gdn)
    per_req = max_stage_gdn * static.gdn_slot_bytes_per_layer

    avail: list[float] = []
    for rank, (start, count) in enumerate(zip(starts, partition)):
        weight = (
            sum(static.layer_weight_bytes[start : start + count])
            + (static.first_rank_extra_bytes if rank == 0 else 0)
            + (static.last_rank_extra_bytes if rank == last else 0)
        ) / GIB + calib.weight_adjust_gib[rank]
        value = calib.pre_avail_gib - weight
        if rank == last:
            value -= calib.draft_weight_gib
        avail.append(value)

    ratio = mamba_full_memory_ratio
    denom = per_req * (1.0 + draft_tokens / calib.slots_per_request)
    numer = per_req * (1.0 + draft_tokens)
    k_per_rank: list[int] = []
    for rank in range(pp_size):
        rest = avail[rank] - calib.slack_gib
        if rank == last:
            rest -= calib.r_spec_k_gib
        budget_bytes = rest * ratio / (1.0 + ratio) * GIB
        k_per_rank.append(max(0, int((budget_bytes - numer) // denom)))
    k = min(k_per_rank)
    k_binding = k_per_rank.index(k)

    capped = min(calib.resolved_max_running, k // calib.slots_per_request)
    bs_cap = min(k // calib.slots_per_request, calib.cuda_graph_max_bs)
    overhang = (
        max(0, bs_cap - GRAPH_OVERHANG_BASE) * calib.graph_overhang_gib
    )
    mamba_main = (k + 1) * per_req / GIB
    intermediate = (capped + 1) * draft_tokens * per_req / GIB
    tokens_per_rank: list[int] = []
    for rank in range(pp_size):
        rest = avail[rank] - calib.slack_gib - overhang
        if rank == last:
            rest -= calib.r_spec_kv_gib
        budget_gib = rest - mamba_main - intermediate
        kv_bytes = static.kv_bytes_per_token_per_full_layer * full[rank]
        tokens_per_rank.append(
            max(0, int(budget_gib * GIB / kv_bytes)) if kv_bytes > 0 else 0
        )
    tokens = min(tokens_per_rank) if tokens_per_rank else 0
    kv_binding = tokens_per_rank.index(tokens)

    # BS_max: mamba slots, the CUDA graph range, AND the scheduler's resolved
    # default max_running_requests (DFlash: 48) all cap the running batch.
    bs_max = min(
        k // calib.slots_per_request,
        calib.cuda_graph_max_bs,
        calib.resolved_max_running,
    )
    return CapacityEstimate(
        partition=partition,
        gdn_per_rank=gdn,
        full_per_rank=full,
        avail_gib=tuple(avail),
        k_per_rank=tuple(k_per_rank),
        k=k,
        k_binding_rank=k_binding,
        kv_tokens_per_rank=tuple(tokens_per_rank),
        kv_tokens=tokens,
        kv_binding_rank=kv_binding,
        bs_max=bs_max,
        capped_requests=capped,
        ratio=mamba_full_memory_ratio,
    )


def sweep_ratios(
    partition: Sequence[int],
    static: ModelStaticInfo,
    calib: CapacityCalibration,
    ratios: Sequence[float],
    draft_tokens: int,
    kv_per_req: float,
    score_fn: Any,
    kv_headroom: float = 1.2,
) -> tuple[float, CapacityEstimate, float, list[dict[str, Any]]]:
    """Scan mamba_full_memory_ratio on a log grid for one partition.

    A ratio is feasible when the KV pool still covers the working set:
    ``KV_tokens >= BS_max * kv_per_req * kv_headroom`` (kv_per_req =
    prompt + output tokens of the offered load).  Among the feasible points
    the one maximizing ``score_fn(estimate)`` (the unified throughput) wins;
    when nothing is feasible the best-scoring infeasible point is returned
    and flagged.  Returns (best_ratio, best_estimate, best_score, rows).
    """
    rows: list[dict[str, Any]] = []
    for ratio in sorted(ratios):
        est = predict_capacity(partition, static, calib, ratio, draft_tokens)
        score = score_fn(est)
        feasible = est.kv_tokens >= est.bs_max * kv_per_req * kv_headroom
        rows.append(
            {
                "ratio": ratio,
                "k": est.k,
                "bs_max": est.bs_max,
                "kv_tokens": est.kv_tokens,
                "feasible": feasible,
                "score": score,
            }
        )
    feasible_rows = [row for row in rows if row["feasible"]]
    pool = feasible_rows or rows
    best_row = max(pool, key=lambda row: (row["score"], row["ratio"]))
    best_est = predict_capacity(partition, static, calib, best_row["ratio"], draft_tokens)
    return best_row["ratio"], best_est, best_row["score"], rows
