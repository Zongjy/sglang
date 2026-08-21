#!/usr/bin/env python3
"""Offline Mamba/KV capacity planner for PP partitions.

Memory is deliberately not an optimization objective.  For a fixed global
request concurrency and a fixed per-request KV working set, this module answers
one question: can every PP rank admit the workload?

The model reuses SGLang's pure hybrid-pool arithmetic.  It predicts raw memory
capacity first and reports the configured scheduler limit separately; a low
``max_running_requests`` must never hide differences between partitions.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from sglang.srt.mem_cache.hybrid_capacity_planner import (
    calculate_mamba_slots_per_request,
    plan_hybrid_pool_capacity,
)

try:
    from benchmark.pp_spec.model_layout import LayerLayout, LayoutError
except ImportError:  # pragma: no cover - flat script execution
    from model_layout import LayerLayout, LayoutError


GIB = 1 << 30
_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "FP16": 2,
    "FLOAT16": 2,
    "F32": 4,
    "FP32": 4,
    "FLOAT32": 4,
    "F64": 8,
    "I64": 8,
    "I32": 4,
    "U8": 1,
    "BOOL": 1,
}


class CapacityModelError(RuntimeError):
    pass


LOAD_BEGIN = re.compile(r"PP(\d+)] Load weight begin\. avail mem=([0-9.]+) GB")
LOAD_END = re.compile(r"PP(\d+)] Load weight end\..*?avail mem=([0-9.]+) GB")
MAMBA_ALLOC = re.compile(r"Mamba Cache is allocated\.\s*max_mamba_cache_size: (\d+)")
MAMBA_CAPPED = re.compile(
    r"max_running_requests is capped to (\d+) by the mamba state cache "
    r"\(max_mamba_cache_size=(\d+), (\d+) state slots per request\)"
)
MAX_TOKENS = re.compile(r"max_total_num_tokens=(\d+)")
SERVER_ARGS_MAX_RUNNING = re.compile(
    r"server_args=ServerArgs\(.+?max_running_requests=(\d+|None)", re.DOTALL
)


def _server_arg(text: str, name: str) -> str | None:
    match = re.search(rf"\b{re.escape(name)}=('[^']*'|[^,)]+)", text)
    if match is None:
        return None
    return match.group(1).strip().strip("'")


def _server_arg_bool(text: str, name: str, default: bool = False) -> bool:
    value = _server_arg(text, name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _dtype_bytes(value: Any, default: int = 2) -> int:
    if value is None:
        return default
    text = str(value).replace("torch.", "").replace("bfloat16", "BF16")
    text = text.replace("float16", "F16").replace("float32", "F32").upper()
    return _DTYPE_BYTES.get(text, default)


def _safetensors_header_sizes(snapshot: Path) -> dict[str, int]:
    """Read tensor sizes without loading model weights."""
    sizes: dict[str, int] = {}
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        shards = sorted(set(weight_map.values()))
    else:
        shards = sorted(path.name for path in snapshot.glob("*.safetensors"))
    if not shards:
        raise OSError(f"no safetensors shards under {snapshot}")
    for shard in shards:
        path = snapshot / shard
        with path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            numel = 1
            for dim in info["shape"]:
                numel *= int(dim)
            sizes[name] = numel * _DTYPE_BYTES.get(str(info["dtype"]).upper(), 2)
    return sizes


def _hf_snapshot(model_path: str | Path) -> Path | None:
    path = Path(model_path)
    if path.is_dir():
        return path
    if path.is_file() and path.name == "config.json":
        return path.parent
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(str(model_path), local_files_only=True))
    except Exception:
        return None


@dataclass(frozen=True)
class ModelStaticInfo:
    """Static layer composition and byte constants."""

    num_layers: int
    layer_types: tuple[str, ...]
    layer_weight_bytes: tuple[int, ...]
    first_rank_extra_bytes: int
    last_rank_extra_bytes: int
    gdn_slot_bytes_per_layer: int
    kv_bytes_per_token_per_full_layer: int
    draft_weight_bytes: int | None
    draft_kv_bytes_per_token: int | None
    weight_source: str
    model_type: str = ""

    @property
    def layout(self) -> LayerLayout:
        try:
            return LayerLayout.from_kinds(
                self.layer_types, model_type=self.model_type, source=self.weight_source
            )
        except LayoutError as exc:
            raise CapacityModelError(str(exc)) from exc

    def gdn_layers(self, start: int, end: int) -> int:
        return self.layout.count_range(start, end)[0]

    def full_layers(self, start: int, end: int) -> int:
        return self.layout.count_range(start, end)[1]

def load_static_info(
    model_path: str,
    draft_model_path: str | None = None,
    fallback_layer_weight_bytes: int | None = None,
    state_dtype: str | None = None,
) -> ModelStaticInfo:
    """Load config/layout and static byte constants from a local model cache."""
    snapshot = _hf_snapshot(model_path)
    if snapshot is None:
        raise CapacityModelError(
            f"model {model_path!r} is not available in the local HF cache"
        )
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise CapacityModelError(f"missing config.json under {snapshot}")
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CapacityModelError(f"cannot read model config {config_path}") from exc
    text = config.get("text_config", config)
    try:
        layout = LayerLayout.from_config(config, source=str(config_path))
    except LayoutError as exc:
        raise CapacityModelError(str(exc)) from exc

    model_dtype_bytes = _dtype_bytes(
        text.get("dtype", text.get("torch_dtype")), default=2
    )
    state_dtype_bytes = _dtype_bytes(state_dtype, default=model_dtype_bytes)
    # SGLang keeps the convolution state in BF16 by default; its optional
    # environment override is not part of the HF config, so expose a config
    # hint when a model provides one and otherwise use the runtime default.
    conv_dtype_bytes = _dtype_bytes(text.get("mamba_conv_dtype"), default=2)
    # Qwen3.5/GDN state: conv state + SSM state.  These fields are absent on
    # dense models, where the state term is simply zero.
    try:
        ssm_numel = (
            int(text["linear_num_value_heads"])
            * int(text["linear_key_head_dim"])
            * int(text["linear_value_head_dim"])
        )
        conv_dim = (
            2 * int(text["linear_num_key_heads"]) * int(text["linear_key_head_dim"])
            + int(text["linear_num_value_heads"])
            * int(text["linear_value_head_dim"])
        )
        conv_numel = conv_dim * (int(text["linear_conv_kernel_dim"]) - 1)
        gdn_slot_bytes = ssm_numel * state_dtype_bytes + conv_numel * conv_dtype_bytes
    except (KeyError, TypeError, ValueError):
        gdn_slot_bytes = 0

    try:
        kv_bytes = (
            2
            * int(text["num_key_value_heads"])
            * int(text["head_dim"])
            * model_dtype_bytes
        )
    except (KeyError, TypeError, ValueError):
        kv_bytes = 0

    draft_weight_bytes: int | None = None
    draft_kv_bytes: int | None = None
    if draft_model_path:
        draft_snapshot = _hf_snapshot(draft_model_path)
        if draft_snapshot is not None:
            try:
                draft_weight_bytes = sum(_safetensors_header_sizes(draft_snapshot).values())
            except (OSError, KeyError, ValueError, struct.error):
                draft_weight_bytes = None
            draft_config_path = draft_snapshot / "config.json"
            if draft_config_path.is_file():
                try:
                    draft_config = json.loads(draft_config_path.read_text())
                except (OSError, json.JSONDecodeError):
                    draft_config = {}
                draft_text = draft_config.get("text_config", draft_config)
                try:
                    # Draft KV is resident on the last PP rank.  Count every
                    # draft layer (including sliding/full attention layers)
                    # rather than silently treating the config as one layer.
                    draft_layers = int(draft_text.get("num_hidden_layers", 1))
                    draft_kv_bytes = (
                        draft_layers
                        * 2
                        * int(draft_text["num_key_value_heads"])
                        * int(draft_text["head_dim"])
                        * _dtype_bytes(
                            draft_text.get("dtype", draft_text.get("torch_dtype")),
                            default=model_dtype_bytes,
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    draft_kv_bytes = None

    try:
        sizes = _safetensors_header_sizes(snapshot)
        layer_bytes = [0] * layout.num_layers
        first_extra = 0
        last_extra = 0
        for name, nbytes in sizes.items():
            if ".layers." in name:
                try:
                    layer_id = int(name.split(".layers.", 1)[1].split(".", 1)[0])
                except (IndexError, ValueError):
                    layer_id = -1
                if 0 <= layer_id < layout.num_layers:
                    layer_bytes[layer_id] += nbytes
                else:
                    first_extra += nbytes
            elif "visual" in name or "lm_head" in name or name.endswith("norm.weight"):
                last_extra += nbytes
            else:
                first_extra += nbytes
        weight_source = "safetensors headers"
    except (OSError, KeyError, ValueError, struct.error) as exc:
        if fallback_layer_weight_bytes is None:
            # A zero fallback is still useful for latency-only analysis, but
            # make the approximation explicit in the report.
            fallback_layer_weight_bytes = 0
        layer_bytes = [int(fallback_layer_weight_bytes)] * layout.num_layers
        first_extra = last_extra = 0
        weight_source = f"fallback per-layer weight ({exc})"

    return ModelStaticInfo(
        num_layers=layout.num_layers,
        layer_types=tuple(layout.kinds),
        layer_weight_bytes=tuple(layer_bytes),
        first_rank_extra_bytes=first_extra,
        last_rank_extra_bytes=last_extra,
        gdn_slot_bytes_per_layer=gdn_slot_bytes,
        kv_bytes_per_token_per_full_layer=kv_bytes,
        draft_weight_bytes=draft_weight_bytes,
        draft_kv_bytes_per_token=draft_kv_bytes,
        weight_source=weight_source,
        model_type=layout.model_type,
    )


@dataclass(frozen=True)
class CapacityCalibration:
    """One baseline memory calibration reused for every candidate partition."""

    pre_avail_gib: float
    slack_gib: float
    slots_per_request: int
    resolved_max_running: int
    baseline_mamba_slots: int
    baseline_kv_tokens: int
    baseline_post_avail_gib: tuple[float, ...] = ()
    baseline_weight_gib: tuple[float, ...] = ()
    baseline_tokens_per_request: int = 0
    draft_tokens: int = 0
    safety_gib: float = 1.0
    mamba_full_memory_ratio: float = 2.0
    page_size: int = 1


def _parse_weight_logs(text: str) -> dict[str, Any]:
    # A draft model may be loaded after the target and emits another
    # ``Load weight begin`` on the last rank.  The first line per rank is the
    # true pre-load budget; later lines must not reduce the baseline pool.
    begin: dict[int, float] = {}
    for rank_s, avail_s in LOAD_BEGIN.findall(text):
        begin.setdefault(int(rank_s), float(avail_s))
    post_avail: dict[int, float] = {}
    for rank_s, avail_s in LOAD_END.findall(text):
        post_avail[int(rank_s)] = float(avail_s)
    capped = MAMBA_CAPPED.findall(text)
    k_values = [int(value) for value in MAMBA_ALLOC.findall(text)]
    token_values = [int(value) for value in MAX_TOKENS.findall(text)]
    max_running = None
    match = SERVER_ARGS_MAX_RUNNING.search(text)
    if match and match.group(1) != "None":
        max_running = int(match.group(1))
    return {
        "begin": begin,
        "post_avail": post_avail,
        "mamba_slots": (
            min(k_values) if k_values else (int(capped[0][1]) if capped else None)
        ),
        "slots": int(capped[0][2]) if capped else 1,
        "kv_tokens": min(token_values) if token_values else 0,
        "max_running": max_running,
    }


def _partition_starts(partition: Sequence[int]) -> tuple[int, ...]:
    starts: list[int] = []
    start = 0
    for count in partition:
        starts.append(start)
        start += int(count)
    return tuple(starts)


def calibrate(
    baseline_log_text: str,
    static: ModelStaticInfo,
    baseline_partition: Sequence[int],
    mem_fraction_static: float = 0.82,
    mamba_full_memory_ratio: float | None = None,
    draft_tokens: int = 0,
    resolved_max_running: int | None = None,
    *,
    safety_gib: float = 1.0,
    tokens_per_request: int | None = None,
    page_size: int = 1,
) -> CapacityCalibration:
    partition = tuple(int(value) for value in baseline_partition)
    if (
        len(partition) == 0
        or any(value <= 0 for value in partition)
        or sum(partition) != static.num_layers
    ):
        raise CapacityModelError("baseline partition does not match model layers")
    if not 0.0 < float(mem_fraction_static) <= 1.0:
        raise CapacityModelError("mem_fraction_static must be in (0, 1]")
    if int(draft_tokens) < 0:
        raise CapacityModelError("draft_tokens must be non-negative")
    ratio = float(2.0 if mamba_full_memory_ratio is None else mamba_full_memory_ratio)
    if ratio <= 0:
        raise CapacityModelError("mamba_full_memory_ratio must be positive")
    if int(page_size) <= 0:
        raise CapacityModelError("page_size must be positive")
    if tokens_per_request is not None and int(tokens_per_request) < 0:
        raise CapacityModelError("tokens_per_request must be non-negative")
    facts = _parse_weight_logs(baseline_log_text)
    pre_values = list(facts["begin"].values())
    pre_avail = sum(pre_values) / len(pre_values) if pre_values else 0.0
    if pre_avail <= 0:
        # A synthetic/offline profile may not have startup memory lines.  Use a
        # large symbolic budget; the caller can still use latency-only mode.
        pre_avail = 0.0
    if _server_arg_bool(baseline_log_text, "enable_linear_replayssm_spec"):
        raise CapacityModelError(
            "offline capacity planning does not yet support ReplaySSM ring geometry"
        )
    if facts["slots"] > 1:
        slots = int(facts["slots"])
    else:
        strategy = _server_arg(
            baseline_log_text, "mamba_radix_cache_strategy"
        ) or "extra_buffer"
        try:
            slots = calculate_mamba_slots_per_request(
                disable_radix_cache=_server_arg_bool(
                    baseline_log_text, "disable_radix_cache"
                ),
                skip_decode_lock=_env_bool(
                    "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK"
                ),
                extra_buffer=strategy in {"extra_buffer", "extra_buffer_lazy"},
                extra_buffer_lazy=strategy == "extra_buffer_lazy",
                disable_overlap_schedule=_server_arg_bool(
                    baseline_log_text, "disable_overlap_schedule"
                ),
            )
        except ValueError as exc:
            raise CapacityModelError(str(exc)) from exc
    if resolved_max_running is not None and int(resolved_max_running) <= 0:
        raise CapacityModelError("resolved_max_running must be positive")
    max_running = int(
        resolved_max_running
        or facts["max_running"]
        or (facts["mamba_slots"] // slots if facts["mamba_slots"] else 0)
        or 1
    )
    starts = _partition_starts(partition)
    baseline_weight_gib: list[float] = []
    for rank, (start, count) in enumerate(zip(starts, partition)):
        value = (
            sum(static.layer_weight_bytes[start : start + count])
            + (static.first_rank_extra_bytes if rank == 0 else 0)
            + (static.last_rank_extra_bytes if rank == len(partition) - 1 else 0)
            + (
                static.draft_weight_bytes or 0
                if rank == len(partition) - 1
                else 0
            )
        ) / GIB
        baseline_weight_gib.append(value)

    if facts["post_avail"]:
        post_avail = tuple(
            float(
                facts["post_avail"].get(
                    rank, pre_avail - baseline_weight_gib[rank]
                )
            )
            for rank in range(len(partition))
        )
    else:
        # Keep the absence of memory facts distinguishable from a real
        # negative/low post-load pool.  Callers use the empty tuple to disable
        # the secondary model rather than silently rejecting every partition.
        post_avail = ()
    baseline_mamba_slots = int(facts["mamba_slots"] or max_running * slots)
    baseline_kv = int(facts["kv_tokens"] or 0)
    baseline_q = int(
        tokens_per_request
        if tokens_per_request is not None
        else (baseline_kv // max(max_running, 1) if baseline_kv else 0)
    )
    return CapacityCalibration(
        pre_avail_gib=pre_avail,
        slack_gib=pre_avail * (1.0 - float(mem_fraction_static)) if pre_avail else 0.0,
        slots_per_request=slots,
        resolved_max_running=max_running,
        baseline_mamba_slots=baseline_mamba_slots,
        baseline_kv_tokens=baseline_kv,
        baseline_post_avail_gib=post_avail,
        baseline_weight_gib=tuple(baseline_weight_gib),
        baseline_tokens_per_request=baseline_q,
        draft_tokens=int(draft_tokens),
        safety_gib=float(max(safety_gib, 0.0)),
        mamba_full_memory_ratio=ratio,
        page_size=int(page_size),
    )


@dataclass(frozen=True)
class CapacityEstimate:
    partition: tuple[int, ...]
    gdn_per_rank: tuple[int, ...]
    full_per_rank: tuple[int, ...]
    avail_gib: tuple[float, ...]
    per_request_gib: tuple[float, ...]
    capacity_per_rank: tuple[int, ...]
    memory_capacity: int
    scheduler_limit: int | None
    effective_limit: int
    binding_rank: int | None
    binding_resource: str
    target_requests: int
    target_feasible: bool
    memory_margin_gib: tuple[float, ...]
    kv_tokens_per_rank: tuple[int | None, ...]
    kv_tokens: int | None
    kv_binding_rank: int | None
    mamba_capacity: int | None
    kv_capacity: int | None
    mamba_slots_per_rank: tuple[int, ...]
    mamba_slots: int | None
    mamba_binding_rank: int | None
    ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": list(self.partition),
            "gdn_per_rank": list(self.gdn_per_rank),
            "full_per_rank": list(self.full_per_rank),
            "avail_gib": [round(value, 4) for value in self.avail_gib],
            "per_request_gib": [round(value, 6) for value in self.per_request_gib],
            "capacity_per_rank": list(self.capacity_per_rank),
            "memory_capacity": self.memory_capacity,
            "scheduler_limit": self.scheduler_limit,
            "effective_limit": self.effective_limit,
            "binding_rank": self.binding_rank,
            "binding_resource": self.binding_resource,
            "target_requests": self.target_requests,
            "target_feasible": self.target_feasible,
            "memory_margin_gib": [round(value, 4) for value in self.memory_margin_gib],
            "kv_tokens_per_rank": list(self.kv_tokens_per_rank),
            "kv_tokens": self.kv_tokens,
            "kv_binding_rank": self.kv_binding_rank,
            "mamba_capacity": self.mamba_capacity,
            "kv_capacity": self.kv_capacity,
            "mamba_slots_per_rank": list(self.mamba_slots_per_rank),
            "mamba_slots": self.mamba_slots,
            "mamba_binding_rank": self.mamba_binding_rank,
            "ratio": self.ratio,
        }


def predict_capacity(
    partition: Sequence[int],
    static: ModelStaticInfo,
    calib: CapacityCalibration,
    mamba_full_memory_ratio: float | None = None,
    draft_tokens: int | None = None,
    *,
    target_requests: int | None = None,
    tokens_per_request: int | None = None,
    graph_reserve_gib: float = 0.0,
    safety_gib: float | None = None,
) -> CapacityEstimate:
    """Predict raw Mamba/KV request capacity for one candidate partition."""
    partition = tuple(int(value) for value in partition)
    if (
        len(partition) == 0
        or any(value <= 0 for value in partition)
        or sum(partition) != static.num_layers
    ):
        raise CapacityModelError("candidate partition does not match model layers")
    last = len(partition) - 1
    starts = _partition_starts(partition)
    gdn = tuple(
        static.gdn_layers(start, start + count)
        for start, count in zip(starts, partition)
    )
    full = tuple(
        static.full_layers(start, start + count)
        for start, count in zip(starts, partition)
    )
    max_gdn = max(gdn, default=0)
    slots = max(calib.slots_per_request, 1)
    ratio = float(
        calib.mamba_full_memory_ratio
        if mamba_full_memory_ratio is None
        else mamba_full_memory_ratio
    )
    if ratio <= 0:
        raise CapacityModelError("mamba_full_memory_ratio must be positive")
    draft_tokens = int(calib.draft_tokens if draft_tokens is None else draft_tokens)
    if draft_tokens < 0:
        raise CapacityModelError("draft_tokens must be non-negative")
    if target_requests is not None and int(target_requests) <= 0:
        raise CapacityModelError("target_requests must be positive")
    target = int(target_requests or calib.resolved_max_running or 1)
    q_tokens = int(
        calib.baseline_tokens_per_request
        if tokens_per_request is None
        else max(tokens_per_request, 0)
    )
    safety = calib.safety_gib if safety_gib is None else max(float(safety_gib), 0.0)

    candidate_weights: list[float] = []
    for rank, (start, count) in enumerate(zip(starts, partition)):
        bytes_value = sum(static.layer_weight_bytes[start : start + count])
        if rank == 0:
            bytes_value += static.first_rank_extra_bytes
        if rank == last:
            bytes_value += static.last_rank_extra_bytes
            bytes_value += static.draft_weight_bytes or 0
        candidate_weights.append(bytes_value / GIB)

    if calib.baseline_post_avail_gib and calib.baseline_weight_gib:
        avail = [
            baseline_post
            - (candidate_weight - baseline_weight)
            - calib.slack_gib
            - float(graph_reserve_gib)
            - safety
            for baseline_post, candidate_weight, baseline_weight in zip(
                calib.baseline_post_avail_gib,
                candidate_weights,
                calib.baseline_weight_gib,
            )
        ]
    elif calib.pre_avail_gib > 0:
        usable = calib.pre_avail_gib - calib.slack_gib - safety - float(graph_reserve_gib)
        avail = [usable - weight for weight in candidate_weights]
    else:
        # No startup memory facts: make the estimate explicitly unusable while
        # preserving a useful object for latency-only reports.
        avail = [-safety for _ in partition]

    # SGLang sizes one synchronized Mamba pool from the largest stage-local GDN
    # share, while KV bytes/token remain rank-local and are synchronized by the
    # minimum token count.
    state_bytes_per_slot = max_gdn * static.gdn_slot_bytes_per_layer
    kv_cells = [
        full_layers * static.kv_bytes_per_token_per_full_layer
        for full_layers in full
    ]
    kv_cells[last] += static.draft_kv_bytes_per_token or 0
    try:
        plan = plan_hybrid_pool_capacity(
            available_bytes_per_rank=tuple(int(value * GIB) for value in avail),
            kv_bytes_per_token_per_rank=tuple(kv_cells),
            mamba_state_bytes_per_slot=state_bytes_per_slot,
            mamba_full_memory_ratio=ratio,
            slots_per_request=slots,
            kv_tokens_per_request=q_tokens,
            page_size=calib.page_size,
            speculative_draft_tokens=draft_tokens,
            target_requests=target,
        )
    except ValueError as exc:
        raise CapacityModelError(str(exc)) from exc

    persistent_per_request = state_bytes_per_slot * slots
    speculative_per_request = state_bytes_per_slot * draft_tokens
    per_request = tuple(
        (
            persistent_per_request
            + speculative_per_request
            + q_tokens * cell
        )
        / GIB
        for cell in kv_cells
    )
    memory_capacity = int(plan.memory_request_capacity)
    scheduler_limit = (
        int(calib.resolved_max_running) if calib.resolved_max_running > 0 else None
    )
    effective_limit = (
        min(memory_capacity, scheduler_limit)
        if scheduler_limit is not None
        else memory_capacity
    )
    margins = tuple(value / GIB for value in plan.target_margin_bytes)
    finite_kv = [
        (rank, value)
        for rank, value in enumerate(plan.kv_tokens_per_rank)
        if value is not None
    ]
    kv_binding, kv_tokens = (
        min(finite_kv, key=lambda item: item[1])
        if finite_kv
        else (None, None)
    )
    mamba_binding = (
        plan.mamba_slots_per_rank.index(min(plan.mamba_slots_per_rank))
        if plan.mamba_slots is not None
        else None
    )
    return CapacityEstimate(
        partition=partition,
        gdn_per_rank=gdn,
        full_per_rank=full,
        avail_gib=tuple(avail),
        per_request_gib=tuple(per_request),
        capacity_per_rank=plan.request_capacity_per_rank,
        memory_capacity=memory_capacity,
        scheduler_limit=scheduler_limit,
        effective_limit=effective_limit,
        binding_rank=plan.binding_rank,
        binding_resource=plan.binding_resource,
        target_requests=target,
        target_feasible=memory_capacity >= target,
        memory_margin_gib=margins,
        kv_tokens_per_rank=plan.kv_tokens_per_rank,
        kv_tokens=kv_tokens,
        kv_binding_rank=kv_binding,
        mamba_capacity=plan.mamba_request_capacity,
        kv_capacity=plan.kv_request_capacity,
        mamba_slots_per_rank=plan.mamba_slots_per_rank,
        mamba_slots=plan.mamba_slots,
        mamba_binding_rank=mamba_binding,
        ratio=ratio,
    )
