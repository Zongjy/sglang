#!/usr/bin/env python3
"""Profile, analyze, and validate an adaptive SGLang PP layer partition.

Only ``--pp-size`` and ``--tp-size`` are required.  The default workflow:

1. discovers a single-node GPU placement and the target model layer count;
2. starts an evenly partitioned baseline and applies a steady decode load;
3. collects per-iteration PP stage metrics (PPM) over ZMQ plus conservative
   memory limits from the startup log;
4. solves the unified partition objective (stage cost x capacity x mamba
   ratio); and
5. writes the recommended partition, ratio, and launch script -- optionally
   followed by a measured PP-vs-TP topology comparison (--compare-tp).

Every run is self-contained under ``tuning_runs/`` and includes logs,
measurements, a final JSON result, and a directly executable launch
script.  Child process groups are tracked explicitly; unrelated SGLang
servers are never killed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_DRAFT_MODEL = "z-lab/Qwen3.6-27B-DFlash"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "tuning_runs"
MEMORY_VALUE = re.compile(r"avail mem=([0-9]+(?:\.[0-9]+)?) GB")
TARGET_LOAD = re.compile(
    r"Load weight end\..*type=([^,]+),.*mem usage=([0-9]+(?:\.[0-9]+)?) GB"
)
FINAL_RUNNING_LIMIT = re.compile(
    r"max_total_num_tokens=\d+.*max_running_requests=(\d+)"
)
MAMBA_RUNNING_LIMIT = re.compile(
    r"max_running_requests is capped to (\d+) by the mamba state cache "
    r"\(max_mamba_cache_size=(\d+), (\d+) state slots per request\)"
)
# Decode batch, #running-req: 4, ..., accept len: 3.46, accept rate: ...
ACCEPT_LEN = re.compile(r"Decode batch.*accept len: ([0-9.]+)")


class TuningError(RuntimeError):
    pass


@dataclass(frozen=True)
class GPUInfo:
    index: int
    uuid: str
    name: str
    memory_mib: int
    pci_bus_id: str


@dataclass(frozen=True)
class ModelInfo:
    model_path: str
    model_type: str
    num_layers: int
    layer_types: tuple[str, ...]
    config_source: str


@dataclass
class ManagedProcess:
    process: subprocess.Popen
    log_handle: IO[str]
    log_path: Path

    def stop(self, timeout_s: float = 30.0) -> None:
        running = self.process.poll() is None
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if running:
            try:
                self.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=10)
        else:
            # The launcher can fail before its workers.  Give any remaining
            # members of the tracked process group time to honor SIGTERM.
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                try:
                    os.killpg(self.process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.25)
            else:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.log_handle.close()


@dataclass
class ValidationResult:
    partition: tuple[int, ...]
    status: str
    throughput_tok_s: float | None
    repeats: list[dict[str, Any]]
    server_log: str
    benchmark_logs: list[str]
    error: str | None = None
    runtime_capacity: dict[str, int] | None = None
    accept_len: float | None = None
    mamba_ratio: float | None = None


def _json_ready(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(_json_ready(payload), handle, indent=2)
        handle.write("\n")


def tail_file(path: Path, lines: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def run_output(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise TuningError(f"Command failed: {shlex.join(command)}\n{detail.strip()}") from exc
    return completed.stdout


def query_gpus() -> list[GPUInfo]:
    output = run_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            continue
        try:
            gpus.append(
                GPUInfo(
                    index=int(fields[0]),
                    uuid=fields[1],
                    name=fields[2],
                    memory_mib=int(fields[3]),
                    pci_bus_id=fields[4],
                )
            )
        except ValueError as exc:
            raise TuningError(f"Cannot parse nvidia-smi row: {line!r}") from exc
    if not gpus:
        raise TuningError("nvidia-smi did not report any GPUs.")
    return gpus


def parse_topology(text: str) -> dict[tuple[int, int], str]:
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    rows = [line.split() for line in text.splitlines() if line.strip()]
    header = next(
        (
            row
            for row in rows
            if len(row) >= 2
            and re.fullmatch(r"GPU\d+", row[0])
            and re.fullmatch(r"GPU\d+", row[1])
        ),
        None,
    )
    if header is None:
        return {}
    gpu_columns = []
    for item in header:
        if not re.fullmatch(r"GPU\d+", item):
            break
        gpu_columns.append(int(item.removeprefix("GPU")))
    links: dict[tuple[int, int], str] = {}
    for row in rows:
        if not row or not re.fullmatch(r"GPU\d+", row[0]):
            continue
        src = int(row[0].removeprefix("GPU"))
        for dst, link in zip(gpu_columns, row[1 : 1 + len(gpu_columns)]):
            links[(src, dst)] = link
    return links


def query_topology() -> dict[tuple[int, int], str]:
    try:
        return parse_topology(run_output(["nvidia-smi", "topo", "-m"]))
    except TuningError:
        return {}


def link_score(link: str) -> int:
    if link == "X":
        return 100
    if link.startswith("NV") and link[2:].isdigit():
        return 90 + int(link[2:])
    return {
        "PIX": 75,
        "PXB": 65,
        "PHB": 55,
        "NODE": 40,
        "SYS": 20,
    }.get(link, 0)


def placement_score(
    order: Sequence[GPUInfo], tp_size: int, pp_size: int, links: dict[tuple[int, int], str]
) -> int:
    score = 0
    for pp_rank in range(pp_size):
        group = order[pp_rank * tp_size : (pp_rank + 1) * tp_size]
        for left, right in itertools.combinations(group, 2):
            score += 10 * link_score(links.get((left.index, right.index), ""))
    for pp_rank in range(pp_size - 1):
        left = order[pp_rank * tp_size : (pp_rank + 1) * tp_size]
        right = order[(pp_rank + 1) * tp_size : (pp_rank + 2) * tp_size]
        for tp_rank in range(tp_size):
            score += link_score(links.get((left[tp_rank].index, right[tp_rank].index), ""))
    return score


def optimize_gpu_order(
    available: Sequence[GPUInfo], count: int, tp_size: int, pp_size: int,
    links: dict[tuple[int, int], str],
) -> list[GPUInfo]:
    if count > len(available):
        raise TuningError(
            f"Need {count} GPUs for TP{tp_size} x PP{pp_size}, "
            f"found {len(available)}."
        )
    ordered_available = sorted(available, key=lambda gpu: gpu.index)
    if not links or len(ordered_available) > 8:
        return ordered_available[:count]

    best: tuple[GPUInfo, ...] | None = None
    best_score = -1
    for order in itertools.permutations(ordered_available, count):
        score = placement_score(order, tp_size, pp_size, links)
        indices = tuple(gpu.index for gpu in order)
        if best is None or score > best_score or (
            score == best_score and indices < tuple(gpu.index for gpu in best)
        ):
            best = order
            best_score = score
    assert best is not None
    return list(best)


def resolve_requested_gpus(
    inventory: Sequence[GPUInfo], visible: str
) -> list[GPUInfo]:
    by_index = {str(gpu.index): gpu for gpu in inventory}
    by_uuid = {gpu.uuid: gpu for gpu in inventory}
    resolved = []
    for token in (item.strip() for item in visible.split(",")):
        if not token or token == "-1":
            continue
        gpu = by_index.get(token) or by_uuid.get(token)
        if gpu is None:
            matches = [item for uuid, item in by_uuid.items() if uuid.startswith(token)]
            if len(matches) == 1:
                gpu = matches[0]
        if gpu is None:
            raise TuningError(f"CUDA device {token!r} was not found by nvidia-smi.")
        if gpu in resolved:
            raise TuningError(f"CUDA device {token!r} was specified more than once.")
        resolved.append(gpu)
    return resolved


def select_gpus(args: argparse.Namespace) -> tuple[list[GPUInfo], bool, dict[tuple[int, int], str]]:
    inventory = query_gpus()
    requested = args.visible_devices
    if requested is None:
        requested = os.environ.get("CUDA_VISIBLE_DEVICES")
    explicit = requested is not None
    available = resolve_requested_gpus(inventory, requested) if explicit else list(inventory)
    world_size = args.pp_size * args.tp_size
    if len(available) < world_size:
        raise TuningError(
            f"TP{args.tp_size} x PP{args.pp_size} needs {world_size} visible GPUs, "
            f"but only {len(available)} are available."
        )
    topology = query_topology()
    selected = (
        available[:world_size]
        if explicit
        else optimize_gpu_order(available, world_size, args.tp_size, args.pp_size, topology)
    )
    return selected, explicit, topology


def inspect_model(model_path: str, trust_remote_code: bool, offline: bool) -> ModelInfo:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise TuningError("transformers is required to inspect the model config.") from exc

    source = "local cache"
    try:
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
    except OSError as local_exc:
        if offline:
            raise TuningError(
                f"Model config for {model_path!r} is not available locally."
            ) from local_exc
        source = "Hugging Face Hub"
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

    text_config = getattr(config, "text_config", config)
    num_layers = getattr(text_config, "num_hidden_layers", None)
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise TuningError(f"Cannot find num_hidden_layers in {model_path!r} config.")
    raw_layer_types = getattr(text_config, "layer_types", None) or []
    layer_types = tuple(str(item) for item in raw_layer_types)
    if layer_types and len(layer_types) != num_layers:
        layer_types = ()
    return ModelInfo(
        model_path=model_path,
        model_type=str(getattr(text_config, "model_type", type(text_config).__name__)),
        num_layers=num_layers,
        layer_types=layer_types,
        config_source=source,
    )


def default_partition(num_layers: int, pp_size: int) -> tuple[int, ...]:
    base, remainder = divmod(num_layers, pp_size)
    return tuple(base + int(rank >= pp_size - remainder) for rank in range(pp_size))


def parse_partition(text: str, pp_size: int, num_layers: int) -> tuple[int, ...]:
    try:
        partition = tuple(int(item.strip()) for item in text.split(","))
    except ValueError as exc:
        raise TuningError(f"Invalid layer partition: {text!r}.") from exc
    if len(partition) != pp_size or any(item <= 0 for item in partition):
        raise TuningError(f"Partition must contain {pp_size} positive layer counts.")
    if sum(partition) != num_layers:
        raise TuningError(
            f"Partition sums to {sum(partition)}, but the model has {num_layers} layers."
        )
    return partition


def find_server_executable(explicit: str | None) -> str:
    if explicit:
        path = shutil.which(explicit) or explicit
        if Path(path).is_file():
            return str(Path(path).resolve())
        raise TuningError(f"SGLang executable not found: {explicit!r}.")
    candidates = [shutil.which("sglang"), str(Path(sys.executable).with_name("sglang"))]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise TuningError("Cannot find the `sglang` executable next to the active Python.")


def build_server_command(
    args: argparse.Namespace,
    executable: str,
    port: int,
    mamba_ratio: float | None = None,
) -> list[str]:
    ratio = mamba_ratio if mamba_ratio is not None else args.mamba_full_memory_ratio
    command = [
        executable,
        "serve",
        "--model-path",
        args.model_path,
        "--tp-size",
        str(args.tp_size),
        "--pp-size",
        str(args.pp_size),
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        args.draft_model_path,
        "--speculative-dflash-block-size",
        str(args.block_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--cuda-graph-max-bs-decode",
        str(args.cuda_graph_max_bs),
        "--page-size",
        str(args.page_size),
        "--random-seed",
        str(args.random_seed),
        "--mamba-ssm-dtype",
        args.mamba_ssm_dtype,
        "--mamba-full-memory-ratio",
        str(ratio),
        "--enable-metrics",
        "--host",
        args.host,
        "--port",
        str(port),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.fixed_active_requests is not None:
        command.extend(
            ["--max-running-requests", str(args.fixed_active_requests)]
        )
    command.extend(args.server_args)
    return command


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def http_request(
    url: str, method: str = "GET", payload: dict[str, Any] | None = None,
    timeout_s: float = 5.0,
) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode()
    if not body.strip():
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def wait_for_server(server: ManagedProcess, base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        return_code = server.process.poll()
        if return_code is not None:
            raise TuningError(
                f"Server exited with code {return_code}; see {server.log_path}.\n"
                f"{tail_file(server.log_path)}"
            )
        try:
            http_request(f"{base_url}/health", timeout_s=3)
            return
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    raise TuningError(
        f"Server health check timed out after {timeout_s:g}s; see {server.log_path}.\n"
        f"{tail_file(server.log_path)}"
    )


def parse_accept_len(log_path: Path) -> float | None:
    """Median 'accept len' over the Decode batch lines of a server log."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None
    values = [float(v) for v in ACCEPT_LEN.findall(text)]
    return float(statistics.median(values)) if values else None


def check_accept_len_consistency(
    named_results: Sequence[tuple[str, ValidationResult]],
    tolerance: float = 0.03,
) -> list[str]:
    """Warn when compared runs disagree on mean accepted length.

    Any throughput comparison is invalid if the accept lengths differ; the
    caller prints the returned warnings but does not block.
    """
    measured = [
        (name, item)
        for name, item in named_results
        if item.status == "ok" and item.accept_len
    ]
    warnings: list[str] = []
    for (name_a, a), (name_b, b) in itertools.combinations(measured, 2):
        rel = abs(a.accept_len - b.accept_len) / max(a.accept_len, b.accept_len)
        if rel > tolerance:
            warnings.append(
                f"throughput comparison INVALID: accept_len differs "
                f"({name_a} {a.accept_len:.3f} vs {name_b} "
                f"{b.accept_len:.3f}, {rel:.1%} > {tolerance:.0%})"
            )
    return warnings


def parse_runtime_capacity(log_path: Path) -> dict[str, int]:
    """Read the resolved scheduler/Mamba capacity from a healthy server log."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return {}
    final_limits = [int(value) for value in FINAL_RUNNING_LIMIT.findall(text)]
    mamba_limits = [tuple(map(int, match)) for match in MAMBA_RUNNING_LIMIT.findall(text)]
    capacity: dict[str, int] = {}
    if final_limits:
        capacity["max_running_requests"] = min(final_limits)
    elif mamba_limits:
        capacity["max_running_requests"] = min(item[0] for item in mamba_limits)
    if mamba_limits:
        capacity["max_mamba_cache_size"] = min(item[1] for item in mamba_limits)
        capacity["mamba_slots_per_request"] = max(item[2] for item in mamba_limits)
    return capacity


def start_server(
    args: argparse.Namespace,
    executable: str,
    selected_gpus: Sequence[GPUInfo],
    partition: Sequence[int],
    phase_dir: Path,
    enable_comm_benchmark: bool = False,
    # PPM telemetry env vars; on only for the profile baseline (the sole
    # consumer) so validation/comparison servers carry no instrumentation
    # overhead.
    enable_ppm: bool = False,
    mamba_ratio: float | None = None,
) -> tuple[ManagedProcess, str, list[str]]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    port = args.port or find_free_port(args.host)
    command = build_server_command(args, executable, port, mamba_ratio=mamba_ratio)
    log_path = phase_dir / "server.log"
    log_handle = log_path.open("w")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu.index) for gpu in selected_gpus)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("SGLANG_FORCE_SHUTDOWN", "1")
    env.setdefault("SGLANG_PROFILE_WITH_STACK", "false")
    env.setdefault("SGLANG_PROFILE_RECORD_SHAPES", "false")
    if enable_ppm:
        # Per-iteration per-PP-rank stage metrics over ZMQ plus the CUDA-event
        # device timer backing gpu_target_ms/gpu_draft_ms.
        env["SGLANG_ENABLE_PP_STAGE_METRICS"] = "1"
        env["SGLANG_ENABLE_METRICS_DEVICE_TIMER"] = "1"
    if enable_comm_benchmark:
        # Startup ping-pong over the real pp_group; fits t_hop = alpha +
        # beta * num_tokens per hop into the server log.  Baseline only --
        # candidate validation servers skip it to save startup time.
        env["SGLANG_PP_COMM_BENCHMARK"] = "1"
        env["SGLANG_PP_COMM_BENCHMARK_TOKENS"] = args.comm_benchmark_tokens
    if args.pp_size > 1:
        env["SGLANG_PP_LAYER_PARTITION"] = ",".join(map(str, partition))
    else:
        env.pop("SGLANG_PP_LAYER_PARTITION", None)
    print(f"[server] {shlex.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    server = ManagedProcess(process=process, log_handle=log_handle, log_path=log_path)
    base_url = f"http://{args.host}:{port}"
    try:
        wait_for_server(server, base_url, args.server_startup_timeout_s)
    except Exception:
        server.stop()
        raise
    capacity = parse_runtime_capacity(log_path)
    resolved_active = capacity.get("max_running_requests")
    if args.fixed_active_requests is not None and resolved_active != args.fixed_active_requests:
        server.stop()
        resolved_text = "unknown" if resolved_active is None else str(resolved_active)
        raise TuningError(
            f"Requested --fixed-active-requests={args.fixed_active_requests}, but "
            f"the server resolved max_running_requests={resolved_text}; see {log_path}."
        )
    cap_suffix = f", active cap {resolved_active}" if resolved_active is not None else ""
    print(
        f"[ok] server healthy at {base_url} (pid {process.pid}{cap_suffix})",
        flush=True,
    )
    return server, base_url, command


def benchmark_command(
    args: argparse.Namespace,
    base_url: str,
    label: str,
    output_dir: Path,
    concurrency: int,
    requests: int,
    max_tokens: int,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "bench_spectre.py"),
        "--url",
        base_url,
        "--label",
        label,
        "--synthetic",
        "--tokenizer",
        args.model_path,
        "--load-points",
        f"{concurrency}:1000:{requests}",
        "--max-tokens",
        str(max_tokens),
        "--prompt-max-tokens",
        str(args.prompt_tokens),
        "--prompt-seed",
        str(args.prompt_seed),
        "--request-timeout-s",
        str(args.request_timeout_s),
        "--cooldown-s",
        "0",
        "--output-dir",
        str(output_dir),
    ]


def profile_load_requests(args: argparse.Namespace, concurrency: int) -> int:
    """Total profile-load requests at the effective (possibly over-offered)
    concurrency.

    Collection runs for --ppm-collect-s seconds; the request pool must
    outlast the window or the decode bs decays below the cap and the high-bs
    buckets starve.
    """
    return max(8 * concurrency, 64)


def profile_output_tokens(args: argparse.Namespace) -> int:
    """Per-request decode length for the profile load.

    Default 8192 so a single request outlasts the 30s+settle collection
    window.
    """
    if args.profile_output_tokens is not None:
        return args.profile_output_tokens
    return 8192


def start_profile_load(
    args: argparse.Namespace, base_url: str, phase_dir: Path, concurrency: int
) -> ManagedProcess:
    output_dir = phase_dir / "profile_load_results"
    command = benchmark_command(
        args,
        base_url,
        "profile_load",
        output_dir,
        concurrency,
        profile_load_requests(args, concurrency),
        profile_output_tokens(args),
    )
    log_path = phase_dir / "profile_load.log"
    log_handle = log_path.open("w")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return ManagedProcess(process=process, log_handle=log_handle, log_path=log_path)


def wait_for_load(load: ManagedProcess, timeout_s: float, settle_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if load.process.poll() is not None:
            raise TuningError(
                f"Profile load exited early; see {load.log_path}.\n{tail_file(load.log_path)}"
            )
        try:
            text = load.log_path.read_text(errors="replace")
        except OSError:
            text = ""
        if "=== profile_load " in text:
            time.sleep(settle_s)
            if load.process.poll() is not None:
                raise TuningError(
                    "Profile requests finished before capture began; increase "
                    "--profile-output-tokens."
                )
            return
        time.sleep(1)
    raise TuningError(
        f"Profile load did not become ready in {timeout_s:g}s; see {load.log_path}.\n"
        f"{tail_file(load.log_path)}"
    )


def run_validation(
    args: argparse.Namespace,
    base_url: str,
    partition: Sequence[int],
    phase_dir: Path,
    concurrency: int,
    server_log: Path,
) -> ValidationResult:
    phase_dir.mkdir(parents=True, exist_ok=True)
    repeats = []
    benchmark_logs = []
    requests = max(16, 2 * concurrency)
    label_base = "p" + "-".join(map(str, partition))
    for repeat in range(args.validation_repeats):
        label = f"{label_base}_r{repeat}"
        output_dir = phase_dir / "benchmark"
        command = benchmark_command(
            args,
            base_url,
            label,
            output_dir,
            concurrency,
            requests,
            args.validation_output_tokens,
        )
        log_path = phase_dir / f"benchmark_r{repeat}.log"
        benchmark_logs.append(str(log_path))
        with log_path.open("w") as handle:
            try:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    text=True,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=args.validation_timeout_s,
                )
            except subprocess.TimeoutExpired:
                return ValidationResult(
                    tuple(partition),
                    "benchmark_failed",
                    None,
                    repeats,
                    str(server_log),
                    benchmark_logs,
                    f"benchmark timed out after {args.validation_timeout_s:g}s",
                )
        if completed.returncode != 0:
            return ValidationResult(
                tuple(partition),
                "benchmark_failed",
                None,
                repeats,
                str(server_log),
                benchmark_logs,
                tail_file(log_path),
            )
        summary_path = output_dir / f"{label}_summary.json"
        try:
            summaries = json.loads(summary_path.read_text())
            summary = summaries[0]
        except (OSError, json.JSONDecodeError, IndexError) as exc:
            return ValidationResult(
                tuple(partition),
                "benchmark_failed",
                None,
                repeats,
                str(server_log),
                benchmark_logs,
                f"cannot read {summary_path}: {exc}",
            )
        repeats.append(summary)
        if summary.get("completed") != summary.get("num_requests"):
            return ValidationResult(
                tuple(partition),
                "benchmark_failed",
                None,
                repeats,
                str(server_log),
                benchmark_logs,
                "not all benchmark requests completed",
            )
    throughput = float(
        statistics.median(item["output_throughput_tok_s"] for item in repeats)
    )
    return ValidationResult(
        tuple(partition),
        "ok",
        throughput,
        repeats,
        str(server_log),
        benchmark_logs,
        runtime_capacity=parse_runtime_capacity(server_log),
        accept_len=parse_accept_len(server_log),
    )


def parse_memory_limits(
    log_path: Path,
    current_partition: Sequence[int],
    reserve_gib: float,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    target_mem: dict[int, list[float]] = {rank: [] for rank in range(len(current_partition))}
    post_load_avail: dict[int, list[float]] = {
        rank: [] for rank in range(len(current_partition))
    }
    loaded: set[tuple[int, int]] = set()
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        pp_matches = re.findall(r"\bPP(\d+)\b", line)
        if not pp_matches:
            continue
        pp_rank = int(pp_matches[-1])
        if pp_rank not in target_mem:
            continue
        tp_matches = re.findall(r"\bTP(\d+)\b", line)
        tp_rank = int(tp_matches[-1]) if tp_matches else 0
        key = (pp_rank, tp_rank)
        load_match = TARGET_LOAD.search(line)
        if load_match and "DFlash" not in load_match.group(1):
            target_mem[pp_rank].append(float(load_match.group(2)))
            loaded.add(key)
        avail_match = MEMORY_VALUE.search(line)
        if avail_match and key in loaded:
            post_load_avail[pp_rank].append(float(avail_match.group(1)))

    limits = []
    details: dict[str, Any] = {"reserve_gib": reserve_gib, "stages": []}
    num_layers = sum(current_partition)
    for rank, current_layers in enumerate(current_partition):
        memories = target_mem[rank]
        available = post_load_avail[rank]
        if not memories or not available:
            maximum = num_layers
            stage_detail = {
                "rank": rank,
                "current_layers": current_layers,
                "max_layers": maximum,
                "source": "unbounded; startup memory was not found in the log",
            }
        else:
            target_gib = float(statistics.median(memories))
            free_gib = min(available)
            gib_per_layer = target_gib / current_layers
            extra = max(0, math.floor((free_gib - reserve_gib) / gib_per_layer))
            maximum = min(num_layers, current_layers + extra)
            stage_detail = {
                "rank": rank,
                "current_layers": current_layers,
                "target_weight_gib": target_gib,
                "minimum_startup_free_gib": free_gib,
                "estimated_gib_per_layer": gib_per_layer,
                "max_layers": maximum,
                "source": "server startup log",
            }
        limits.append(max(current_layers, maximum))
        details["stages"].append(stage_detail)
    return tuple(limits), details


def collect_ppm_snapshot(args: argparse.Namespace, baseline_dir: Path) -> dict[str, Any]:
    """Collect a PPM snapshot from the live baseline server under load."""
    sys.path.insert(0, str(SCRIPT_DIR))
    import ppm_consumer

    log_path = baseline_dir / "server.log"
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    endpoints = ppm_consumer.parse_ppm_endpoints(text)
    if not endpoints:
        raise TuningError(
            "No 'PPM: ZMQ PUB bound on' lines in the baseline server log; "
            "the server did not publish stage metrics."
        )
    saturation_bs = args.ppm_saturation_bs
    if saturation_bs is None:
        if args.fixed_active_requests is not None:
            saturation_bs = args.fixed_active_requests
        else:
            saturation_bs = parse_runtime_capacity(log_path).get("max_running_requests")
    print(
        f"[ppm] collecting for {args.ppm_collect_s:g}s from {len(endpoints)} "
        f"endpoint(s) (saturation bs: {saturation_bs or 'num_queued only'})",
        flush=True,
    )
    snapshot = ppm_consumer.collect(
        endpoints,
        duration_s=args.ppm_collect_s,
        max_messages=args.ppm_max_messages,
        saturation_bs=saturation_bs,
    )
    write_json(baseline_dir / "ppm_snapshot.json", snapshot)
    print(
        f"[ok] PPM snapshot: {snapshot['messages_total']} messages, "
        f"{snapshot['messages_kept']} kept after the work-conservation filter",
        flush=True,
    )
    return snapshot


def _build_capacity_context(
    args: argparse.Namespace,
    baseline_log_text: str,
    baseline_partition: Sequence[int],
) -> dict[str, Any] | None:
    """Build the capacity model (static info + baseline calibration).

    Returns None (with a printed warning) when the model files or the log
    facts are unavailable; the analysis then falls back to the compute-only
    objective.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    import capacity_model

    try:
        static = capacity_model.load_static_info(
            args.model_path, args.draft_model_path
        )
        calibration = capacity_model.calibrate(
            baseline_log_text,
            static,
            baseline_partition,
            args.mem_fraction_static,
            args.mamba_full_memory_ratio,
            args.block_size,
            args.cuda_graph_max_bs,
        )
    except capacity_model.CapacityModelError as exc:
        print(f"[warn] capacity model disabled: {exc}", flush=True)
        return None

    def capacity_of(partition: Sequence[int]):
        return capacity_model.predict_capacity(
            partition,
            static,
            calibration,
            args.mamba_full_memory_ratio,
            args.block_size,
        )

    def composition_of(partition: Sequence[int]) -> tuple[str, ...]:
        start = 0
        parts = []
        for count in partition:
            gdn = static.gdn_layers(start, start + count)
            full = static.full_layers(start, start + count)
            parts.append(f"{gdn}G+{full}F")
            start += count
        return tuple(parts)

    def k_of(partition: Sequence[int]) -> tuple[int, ...]:
        return capacity_of(partition).k_per_rank

    return {
        "static": static,
        "calibration": calibration,
        "capacity_of": capacity_of,
        "composition_of": composition_of,
        "k_of": k_of,
        "weight_source": static.weight_source,
    }


def render_ppm_report(
    args: argparse.Namespace,
    result: Any,
    capacity_ctx: dict[str, Any] | None,
    output_dir: Path,
    meas: dict[tuple[int, ...], Any] | None = None,
    k_of: Any = None,
) -> str:
    """Render the optimizer report + decision table + sweep plot.

    ``k_of`` (optional) overrides the per-rank K source; pass the swept
    (ratio-optimized) capacity callable so the table's K0/K1 use the same
    ratio as each row's recommendation.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    import plot_partition_model as plot_mod

    rows = plot_mod.build_rows(
        result,
        composition_of=(
            capacity_ctx["composition_of"] if capacity_ctx else None
        ),
        k_of=k_of or (capacity_ctx["k_of"] if capacity_ctx else None),
        meas=meas,
    )
    table = plot_mod.render_table(rows, with_meas=bool(meas))
    text = result.to_report() + "\n\n" + table + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.txt").write_text(text)
    png = plot_mod.render_plot(
        rows,
        output_dir / "partition_sweep.png",
        title="PP partition sweep: compute vs capacity",
    )
    if png is None:
        print("[warn] matplotlib unavailable; sweep plot skipped", flush=True)
    return text


def resolve_t_comm(
    args: argparse.Namespace, baseline_log_text: str, work_tokens: float
) -> tuple[float, str]:
    """Resolve the per-boundary comm cost (ms).

    Manual --t-comm-ms wins; otherwise parse the baseline's startup comm
    micro-benchmark hops and take the worst hop at ``work_tokens`` (bs* x
    block_size hidden states per hop); otherwise fall back to 0.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    import ppm_consumer

    if args.t_comm_ms is not None:
        return args.t_comm_ms, "manual --t-comm-ms"
    hops = ppm_consumer.parse_comm_benchmark_hops(baseline_log_text)
    if hops:
        details = ", ".join(
            f"{hop}: alpha={alpha:.4f} ms, beta={beta:.3e} ms/token"
            for hop, (alpha, beta) in sorted(hops.items())
        )
        value = max(alpha + beta * work_tokens for alpha, beta in hops.values())
        return value, f"startup comm benchmark ({details})"
    print(
        "[warn] no [pp_comm_benchmark] lines in the baseline server log; "
        "t_comm falls back to 0 (uncalibrated)",
        flush=True,
    )
    return 0.0, "unavailable; fell back to 0"


def bucket_sample_distribution(snapshot: dict[str, Any]) -> dict[int, int]:
    """Kept (work-conserving) samples per bs bucket, summed over ranks."""
    counts: dict[int, int] = {}
    for rank_data in snapshot.get("ranks", {}).values():
        for bucket, cell in rank_data.get("buckets", {}).items():
            counts[int(bucket)] = counts.get(int(bucket), 0) + int(
                cell.get("count", 0)
            )
    return dict(sorted(counts.items()))


def analyze_ppm(
    args: argparse.Namespace,
    snapshot: dict[str, Any],
    current_partition: Sequence[int],
    max_layers: Sequence[int],
    output_dir: Path,
    baseline_log_text: str,
    meas: dict[tuple[int, ...], Any] | None = None,
) -> dict[str, Any]:
    """Fit stage + capacity models and solve the unified partition objective."""
    sys.path.insert(0, str(SCRIPT_DIR))
    import partition_optimizer
    import stage_model

    try:
        model = stage_model.StageCostModel.fit(snapshot, current_partition)
    except stage_model.StageModelError as exc:
        raise TuningError(str(exc)) from exc

    bucket_counts = bucket_sample_distribution(snapshot)
    print(f"[ppm] kept samples per bs bucket: {bucket_counts}", flush=True)
    work_tokens = model.target_bucket().bs_mean * args.block_size
    t_comm_ms, t_comm_source = resolve_t_comm(args, baseline_log_text, work_tokens)
    print(
        f"[t_comm] {t_comm_ms:.4f} ms at {work_tokens:g} tokens/hop "
        f"({t_comm_source})",
        flush=True,
    )

    capacity_ctx = _build_capacity_context(
        args, baseline_log_text, current_partition
    )
    capacity_callable = None
    if capacity_ctx is not None:
        import capacity_model

        ratios = tuple(
            float(item) for item in args.ratio_grid.split(",") if item.strip()
        )
        kv_per_req = args.prompt_tokens + args.validation_output_tokens
        static = capacity_ctx["static"]
        calibration = capacity_ctx["calibration"]
        accept_len = model.target_bucket().accept_len
        loop_size = max(model.pp_loop_size, 1)

        def capacity_callable(partition):  # noqa: F811
            def score(est) -> float:
                per_slot = est.bs_max / loop_size
                c_b, fixed_b = model.cost_at_bs(per_slot)
                cadence = max(
                    fixed_b[r] + c_b * n + (t_comm_ms if r > 0 else 0.0)
                    for r, n in enumerate(partition)
                )
                if cadence <= 0 or accept_len <= 0:
                    return 0.0
                return est.bs_max * accept_len / (loop_size * cadence / 1000.0)

            _, est, _, _ = capacity_model.sweep_ratios(
                partition,
                static,
                calibration,
                ratios,
                args.block_size,
                kv_per_req,
                score,
            )
            return est

    try:
        result = partition_optimizer.optimize(
            model,
            t_comm_ms=t_comm_ms,
            max_layers=max_layers,
            k_best=args.k_best,
            allow_draft_relocation=args.allow_draft_relocation,
            capacity=capacity_callable,
        )
    except partition_optimizer.OptimizerError as exc:
        raise TuningError(str(exc)) from exc

    text = render_ppm_report(
        args,
        result,
        capacity_ctx,
        output_dir,
        meas,
        k_of=(
            (lambda partition: capacity_callable(partition).k_per_rank)
            if capacity_callable is not None
            else None
        ),
    )
    print(text, flush=True)

    capacity_summary = None
    if capacity_ctx is not None:
        capacity_summary = {
            "weight_source": capacity_ctx["weight_source"],
            "baseline_k": capacity_ctx["calibration"].baseline_k,
            "baseline_kv_tokens": capacity_ctx["calibration"].baseline_kv_tokens,
            "estimates": {
                ",".join(map(str, item.partition)): capacity_ctx["capacity_of"](
                    item.partition
                ).to_dict()
                for item in result.candidates
            },
        }
    analysis = {
        "mode": "ppm",
        "num_layers": model.num_layers,
        "current_partition": list(current_partition),
        "recommended": {
            "partition": list(result.best.partition),
            "draft_rank": result.best.draft_rank,
            "bottleneck_ms": result.best.bottleneck_ms,
            "bs_max": result.best.bs_max,
            "throughput_tok_s": result.best.throughput_tok_s,
            "mamba_ratio": result.best.mamba_ratio,
        },
        "candidates": [
            {
                "partition": list(item.partition),
                "draft_rank": item.draft_rank,
                "bottleneck_ms": item.bottleneck_ms,
                "bs_max": item.bs_max,
                "throughput_tok_s": item.throughput_tok_s,
                "mamba_ratio": item.mamba_ratio,
            }
            for item in result.candidates[: args.candidate_count]
        ],
        "keep_current": result.keep_current,
        "recommendation": result.recommendation,
        "t_comm": {
            "value_ms": t_comm_ms,
            "source": t_comm_source,
            "work_tokens": work_tokens,
        },
        "ratio_grid": args.ratio_grid,
        "bucket_samples": bucket_counts,
        "compute_optimal": (
            list(result.compute_optimal) if result.compute_optimal else None
        ),
        "capacity_optimal": (
            list(result.capacity_optimal) if result.capacity_optimal else None
        ),
        "max_candidate_bs_max": max(
            (item.bs_max or 0 for item in result.candidates), default=0
        ),
        "capacity_model": capacity_summary,
        "stage_model": model.to_dict(),
        "optimization": result.to_dict(),
    }
    write_json(output_dir / "analysis.json", analysis)
    return analysis


def parse_measured_capacity(log_path: Path) -> dict[str, int]:
    """Read the measured mamba K and resolved running cap from a server log."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return {}
    measured: dict[str, int] = {}
    k_values = [int(v) for v in re.findall(r"max_mamba_cache_size: (\d+)", text)]
    if k_values:
        measured["k"] = min(k_values)
    measured.update(parse_runtime_capacity(log_path))
    return measured


TP_COMPARISON_SETTLE_S = 5.0


def run_tp_comparison(
    args: argparse.Namespace,
    run_dir: Path,
    executable: str,
    selected: Sequence[GPUInfo],
    best_partition: Sequence[int],
    baseline_partition: Sequence[int] | None = None,
    best_ratio: float | None = None,
) -> dict[str, Any]:
    """PP vs TP topology comparison, self-contained on fresh servers.

    Measures BOTH sides at their native capacity first -- the PP server at
    the recommended partition/ratio, the TP server (tp_size = pp_size *
    tp_size, pp_size = 1, no layer partition, no comm benchmark, no PPM --
    event_loop_pp does not run) at its swept ratio -- then re-measures each
    side above the common cap at --max-running-requests = min(PP_BS, TP_BS)
    with offered concurrency >= 2x that value (the side already below the
    common cap keeps its native measurement).  The accept_len consistency
    check applies to the comparison.  The TP ratio comes from a pp=1
    capacity sweep (score = BS_max under the KV constraint); the PP side
    uses the recommended partition's ratio.
    """
    tp_ratio = best_ratio
    tp_capacity_est = None
    baseline_log_path = run_dir / "baseline" / "server.log"
    if baseline_partition is not None and baseline_log_path.is_file():
        try:
            import capacity_model

            baseline_log_text = baseline_log_path.read_text(errors="replace")
            ctx = _build_capacity_context(args, baseline_log_text, baseline_partition)
            if ctx is not None:
                ratios = tuple(
                    float(item) for item in args.ratio_grid.split(",") if item.strip()
                )
                kv_per_req = args.prompt_tokens + args.validation_output_tokens
                num_layers = sum(best_partition)
                tp_ratio, tp_capacity_est, _, _ = capacity_model.sweep_ratios(
                    (num_layers,),
                    ctx["static"],
                    ctx["calibration"],
                    ratios,
                    args.block_size,
                    kv_per_req,
                    lambda est: est.bs_max,
                )
        except Exception as exc:  # ratio sweep is best-effort
            print(f"[warn] TP ratio sweep failed: {exc}; using default ratio", flush=True)
            tp_ratio = best_ratio
    tp_args = argparse.Namespace(**vars(args))
    tp_args.tp_size = args.tp_size * args.pp_size
    tp_args.pp_size = 1
    tp_args.fixed_active_requests = None
    pp_args = argparse.Namespace(**vars(args))
    pp_args.fixed_active_requests = None

    phase_dir = run_dir / "tp_comparison"
    result: dict[str, Any] = {
        "tp_size": tp_args.tp_size,
        "pp_partition": list(best_partition),
        "tp_mamba_ratio": tp_ratio,
        "tp_capacity_est": (
            tp_capacity_est.to_dict() if tp_capacity_est is not None else None
        ),
    }

    def stop_and_settle(server: ManagedProcess | None) -> None:
        """Stop a server and let the GPU fully drain before the next start.

        ManagedProcess.stop already kills the whole process group and waits;
        the extra settle covers driver-side memory release so the next
        server never sees the previous one's allocation (the TP/PP
        comparison runs three servers sequentially on the same GPUs).
        """
        if server is not None:
            server.stop()
        time.sleep(TP_COMPARISON_SETTLE_S)

    server: ManagedProcess | None = None
    try:
        # PP side at its recommended partition/ratio, native capacity.
        server, pp_url, _ = start_server(
            pp_args, executable, selected, best_partition,
            phase_dir / "pp_native", mamba_ratio=best_ratio,
        )
        pp_capacity = parse_measured_capacity(server.log_path)
        pp_cap = pp_capacity.get("max_running_requests")
        result["pp_capacity"] = pp_capacity
        pp_native = run_validation(
            pp_args, pp_url, best_partition, phase_dir / "pp_native",
            max(2 * (pp_cap or args.concurrency), 16), server.log_path,
        )
        result["pp_native"] = asdict(pp_native)
        # Exactly one server on the GPUs at any time from here on.
        stop_and_settle(server)
        server = None

        # TP side at its swept ratio, native capacity.
        server, tp_url, _ = start_server(
            tp_args, executable, selected, (), phase_dir / "tp_native",
            mamba_ratio=tp_ratio,
        )
        tp_capacity = parse_measured_capacity(server.log_path)
        tp_cap = tp_capacity.get("max_running_requests")
        result["tp_capacity"] = tp_capacity
        tp_native = run_validation(
            tp_args, tp_url, (), phase_dir / "tp_native",
            max(2 * (tp_cap or args.concurrency), 16), server.log_path,
        )
        result["tp_native"] = asdict(tp_native)
        stop_and_settle(server)
        server = None

        if (
            pp_cap
            and tp_cap
            and pp_native.status == "ok"
            and tp_native.status == "ok"
        ):
            common = min(pp_cap, tp_cap)
            concurrency = max(2 * common, 16)
            result["equal_concurrency"] = common

            if pp_cap > common:
                pp_capped_args = argparse.Namespace(**vars(pp_args))
                pp_capped_args.fixed_active_requests = common
                pp_dir = phase_dir / "pp_capped"
                server, pp_url, _ = start_server(
                    pp_capped_args, executable, selected, best_partition, pp_dir,
                    mamba_ratio=best_ratio,
                )
                pp_equal = run_validation(
                    pp_capped_args, pp_url, best_partition, pp_dir, concurrency,
                    server.log_path,
                )
                stop_and_settle(server)
                server = None
            else:
                pp_equal = pp_native
            result["pp_equal"] = asdict(pp_equal)

            if tp_cap > common:
                tp_capped_args = argparse.Namespace(**vars(tp_args))
                tp_capped_args.fixed_active_requests = common
                tp_dir = phase_dir / "tp_capped"
                server, tp_url, _ = start_server(
                    tp_capped_args, executable, selected, (), tp_dir,
                    mamba_ratio=tp_ratio,
                )
                tp_equal = run_validation(
                    tp_capped_args, tp_url, (), tp_dir, concurrency,
                    server.log_path,
                )
                stop_and_settle(server)
                server = None
            else:
                tp_equal = tp_native
            result["tp_equal"] = asdict(tp_equal)

            warnings = check_accept_len_consistency(
                [("pp", pp_equal), ("tp", tp_equal)]
            )
            result["accept_len_warnings"] = warnings
            for warning in warnings:
                print(f"[WARNING] {warning}", flush=True)
    finally:
        if server is not None:
            server.stop()

    def _tok(entry: dict[str, Any] | None) -> str:
        if not entry or entry.get("throughput_tok_s") is None:
            return "n/a"
        return f"{entry['throughput_tok_s']:.1f} tok/s"

    def _acc(entry: dict[str, Any] | None) -> str:
        if not entry or entry.get("accept_len") is None:
            return "n/a"
        return f"{entry['accept_len']:.2f}"

    lines = [
        "PP vs TP comparison:",
        f"  ratios: PP best at {best_ratio or args.mamba_full_memory_ratio:g}, "
        f"TP at {tp_ratio or args.mamba_full_memory_ratio:g}",
        f"  native capacity: PP {','.join(map(str, best_partition))} "
        f"(cap {pp_cap}) {_tok(result['pp_native'])} vs "
        f"TP{tp_args.tp_size} (cap {tp_cap}) {_tok(result['tp_native'])}",
    ]
    if "equal_concurrency" in result:
        lines.append(
            f"  equal concurrency (cap {result['equal_concurrency']}): "
            f"PP {_tok(result.get('pp_equal'))} (accept {_acc(result.get('pp_equal'))}) vs "
            f"TP {_tok(result.get('tp_equal'))} (accept {_acc(result.get('tp_equal'))})"
        )
        verdict = (
            "accept_len consistent"
            if not result.get("accept_len_warnings")
            else "accept_len DIFFERS; comparison invalid"
        )
        lines.append(f"  accept_len consistency: {verdict}")
    text = "\n".join(lines)
    print(text, flush=True)
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "tp_comparison.txt").write_text(text + "\n")
    result["summary"] = text
    return result


def create_run_dir(root: Path, tp_size: int, pp_size: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = root / f"{stamp}_tp{tp_size}_pp{pp_size}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def write_best_artifacts(
    args: argparse.Namespace,
    run_dir: Path,
    executable: str,
    selected_gpus: Sequence[GPUInfo],
    partition: Sequence[int],
    reason: str,
    mamba_ratio: float | None = None,
) -> None:
    visible = ",".join(str(gpu.index) for gpu in selected_gpus)
    command = build_server_command(
        args, executable, args.final_port, mamba_ratio=mamba_ratio
    )
    config = {
        "tp_size": args.tp_size,
        "pp_size": args.pp_size,
        "partition": tuple(partition),
        "mamba_full_memory_ratio": (
            mamba_ratio if mamba_ratio is not None else args.mamba_full_memory_ratio
        ),
        "cuda_visible_devices": visible,
        "selection_reason": reason,
        "command": command,
    }
    write_json(run_dir / "best_config.json", config)
    env_lines = [f"export CUDA_VISIBLE_DEVICES={shlex.quote(visible)}"]
    if args.pp_size > 1:
        value = ",".join(map(str, partition))
        env_lines.append(f"export SGLANG_PP_LAYER_PARTITION={shlex.quote(value)}")
    else:
        env_lines.append("unset SGLANG_PP_LAYER_PARTITION")
    (run_dir / "best_config.env").write_text("\n".join(env_lines) + "\n")
    launch = run_dir / "launch_best.sh"
    launch.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(env_lines)
        + "\nexec "
        + shlex.join(command)
        + "\n"
    )
    launch.chmod(0o755)


def format_gpu_plan(
    selected: Sequence[GPUInfo], args: argparse.Namespace, topology: dict[tuple[int, int], str]
) -> list[dict[str, Any]]:
    plan = []
    for global_rank, gpu in enumerate(selected):
        pp_rank, tp_rank = divmod(global_rank, args.tp_size)
        plan.append(
            {
                "global_rank": global_rank,
                "pp_rank": pp_rank,
                "tp_rank": tp_rank,
                "physical_gpu": gpu.index,
                "name": gpu.name,
                "memory_mib": gpu.memory_mib,
                "pci_bus_id": gpu.pci_bus_id,
            }
        )
    for left, right in zip(plan, plan[1:]):
        left["link_to_next_rank"] = topology.get(
            (left["physical_gpu"], right["physical_gpu"]), "unknown"
        )
    return plan


def run_tuner(args: argparse.Namespace) -> Path | None:
    if args.pp_size <= 0 or args.tp_size <= 0:
        raise TuningError("--pp-size and --tp-size must be positive.")
    if args.validation_repeats <= 0 or args.candidate_count <= 0:
        raise TuningError("Validation repeats and candidate count must be positive.")
    model = inspect_model(args.model_path, args.trust_remote_code, args.offline)
    if args.pp_size > model.num_layers:
        raise TuningError(
            f"PP size {args.pp_size} exceeds the model's {model.num_layers} layers."
        )
    baseline = (
        parse_partition(args.current_partition, args.pp_size, model.num_layers)
        if args.current_partition
        else default_partition(model.num_layers, args.pp_size)
    )
    concurrency = args.concurrency
    if concurrency <= 0:
        raise TuningError("--concurrency must be positive.")
    if args.fixed_active_requests is not None:
        if args.fixed_active_requests <= 0:
            raise TuningError("--fixed-active-requests must be positive.")
        if args.fixed_active_requests > concurrency:
            raise TuningError(
                "--fixed-active-requests cannot exceed the offered --concurrency."
            )
        if any(
            item == "--max-running-requests"
            or item.startswith("--max-running-requests=")
            for item in args.server_args
        ):
            raise TuningError(
                "Use --fixed-active-requests instead of passing "
                "--max-running-requests through --server-args."
            )

    if args.reanalyze:
        run_dir = args.reanalyze
        baseline_dir = run_dir / "baseline"
        snapshot_path = baseline_dir / "ppm_snapshot.json"
        log_path = baseline_dir / "server.log"
        if not snapshot_path.is_file() or not log_path.is_file():
            raise TuningError(
                f"--reanalyze needs {snapshot_path} and {log_path} from a "
                "previous ppm-mode run."
            )
        snapshot = json.loads(snapshot_path.read_text())
        baseline_log_text = log_path.read_text(errors="replace")
        saved: dict[str, Any] = {}
        result_path = run_dir / "result.json"
        if result_path.is_file():
            saved = json.loads(result_path.read_text())
        saved_settings = saved.get("settings", {})
        for key in (
            "mem_fraction_static",
            "mamba_full_memory_ratio",
            "block_size",
            "cuda_graph_max_bs",
            "model_path",
            "draft_model_path",
        ):
            if key in saved_settings:
                setattr(args, key, saved_settings[key])
        baseline = (
            tuple(saved["baseline_partition"])
            if saved.get("baseline_partition")
            else baseline
        )
        # Measured columns from the previous run's validations.
        sys.path.insert(0, str(SCRIPT_DIR))
        import plot_partition_model as plot_mod

        meas: dict[tuple[int, ...], Any] = {}
        for item in saved.get("validations", []):
            if item.get("status") != "ok":
                continue
            capacity = parse_measured_capacity(Path(item["server_log"]))
            capacity.update(item.get("runtime_capacity") or {})
            meas[tuple(item["partition"])] = plot_mod.MeasPoint(
                k=capacity.get("k"),
                bs_max=capacity.get("max_running_requests"),
                accept_len=item.get("accept_len"),
                mamba_ratio=item.get("mamba_ratio"),
            )
        out_dir = run_dir / "reanalysis"
        analysis = analyze_ppm(
            args,
            snapshot,
            baseline,
            (model.num_layers,) * args.pp_size,
            out_dir,
            baseline_log_text,
            meas=meas or None,
        )
        write_json(
            out_dir / "result.json",
            {
                "status": "reanalysis",
                "source_run": str(run_dir),
                "baseline_partition": baseline,
                "analysis": analysis,
            },
        )
        print(f"[done] reanalysis written to {out_dir}", flush=True)
        print(f"[done] sweep plot: {out_dir / 'partition_sweep.png'}", flush=True)
        return out_dir

    executable = find_server_executable(args.sglang_executable)
    selected, explicit_devices, topology = select_gpus(args)
    gpu_plan = format_gpu_plan(selected, args, topology)
    dry_plan = {
        "model": model,
        "tp_size": args.tp_size,
        "pp_size": args.pp_size,
        "world_size": args.tp_size * args.pp_size,
        "baseline_partition": baseline,
        "profile_concurrency": concurrency,
        "load_control": {
            "offered_concurrency": concurrency,
            "requested_fixed_active_requests": args.fixed_active_requests,
        },
        "explicit_device_order": explicit_devices,
        "gpu_plan": gpu_plan,
        "server_command": build_server_command(
            args, executable, args.port or args.final_port
        ),
    }
    if args.dry_run:
        print(json.dumps(_json_ready(dry_plan), indent=2))
        return None

    if args.compare_tp_only:
        # Convenience: skip PP baseline/profile/analysis and only (re-)run
        # the TP comparison of an existing run, reading the best partition
        # and ratio from its result.json.
        run_dir = args.compare_tp_only
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            raise TuningError(f"--compare-tp-only needs {result_path}.")
        saved = json.loads(result_path.read_text())
        saved_settings = saved.get("settings", {})
        for key in (
            "mem_fraction_static",
            "mamba_full_memory_ratio",
            "block_size",
            "cuda_graph_max_bs",
            "model_path",
            "draft_model_path",
        ):
            if key in saved_settings:
                setattr(args, key, saved_settings[key])
        if not saved.get("best_partition"):
            raise TuningError(f"{result_path} has no best_partition yet.")
        best = tuple(saved["best_partition"])
        saved_baseline = tuple(saved.get("baseline_partition") or best)
        best_ratio = None
        analysis = saved.get("analysis") or {}
        recommended = analysis.get("recommended") or {}
        if tuple(recommended.get("partition") or ()) == best:
            best_ratio = recommended.get("mamba_ratio")
        comparison = run_tp_comparison(
            args,
            run_dir,
            executable,
            selected,
            best,
            baseline_partition=saved_baseline,
            best_ratio=best_ratio,
        )
        saved["tp_comparison"] = comparison
        write_json(result_path, saved)
        print(
            f"[done] TP comparison written to {run_dir / 'tp_comparison'}",
            flush=True,
        )
        return run_dir

    run_dir = create_run_dir(args.output_root, args.tp_size, args.pp_size)
    state: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "model": model,
        "hardware": dry_plan,
        "baseline_partition": baseline,
    }
    write_json(run_dir / "result.json", state)
    print(f"[run] artifacts: {run_dir}", flush=True)

    if args.pp_size == 1:
        reason = "PP size is 1; no layer boundary exists to tune"
        state.update(status="complete", best_partition=baseline, selection_reason=reason)
        write_best_artifacts(args, run_dir, executable, selected, baseline, reason)
        write_json(run_dir / "result.json", state)
        print(f"[done] {reason}")
        return run_dir

    baseline_dir = run_dir / "baseline"
    server, base_url, baseline_command = start_server(
        args,
        executable,
        selected,
        baseline,
        baseline_dir,
        enable_comm_benchmark=True,
        enable_ppm=True,
    )
    ppm_snapshot: dict[str, Any] | None = None
    load: ManagedProcess | None = None
    try:
        # With abundant memory (e.g. small models) the resolved active cap
        # can exceed the offered concurrency: the queue stays empty and
        # the work-conservation filter discards every PPM sample.
        # Over-offer so the baseline is genuinely saturated during capture.
        profile_concurrency = concurrency
        baseline_cap = parse_runtime_capacity(baseline_dir / "server.log").get(
            "max_running_requests"
        )
        if baseline_cap is None:
            print(
                "[warn] max_running_requests not found in the baseline log; "
                "profile over-offer skipped and the PPM saturation fallback "
                "disabled (work-conservation relies on num_queued only)",
                flush=True,
            )
        else:
            profile_concurrency = max(concurrency, min(2 * baseline_cap, 512))
            if profile_concurrency < baseline_cap:
                raise TuningError(
                    f"Baseline max_running_requests={baseline_cap} exceeds the "
                    f"client over-offer ceiling (512); pass a higher "
                    f"--concurrency so the profile load can saturate the "
                    f"server, or the work-conservation filter will discard "
                    f"every PPM sample."
                )
            if profile_concurrency != concurrency:
                print(
                    f"[load] profile concurrency raised to "
                    f"{profile_concurrency} (baseline cap {baseline_cap})",
                    flush=True,
                )
        load = start_profile_load(args, base_url, baseline_dir, profile_concurrency)
        wait_for_load(load, args.profile_load_timeout_s, args.profile_settle_s)
        ppm_snapshot = collect_ppm_snapshot(args, baseline_dir)
        load.stop(timeout_s=10)
        load = None
    finally:
        if load is not None:
            load.stop(timeout_s=10)
        server.stop()

    state["baseline_command"] = baseline_command
    state["baseline_runtime_capacity"] = parse_runtime_capacity(
        baseline_dir / "server.log"
    )
    max_layers, memory_details = parse_memory_limits(
        baseline_dir / "server.log", baseline, args.memory_reserve_gib
    )
    state["memory_limits"] = {
        "max_layers_per_stage": max_layers,
        "details": memory_details,
    }
    write_json(run_dir / "result.json", state)

    assert ppm_snapshot is not None
    baseline_log_text = (baseline_dir / "server.log").read_text(errors="replace")
    analysis = analyze_ppm(
        args, ppm_snapshot, baseline, max_layers, baseline_dir,
        baseline_log_text,
    )
    state["analysis"] = analysis
    write_json(run_dir / "result.json", state)

    best = tuple(analysis["recommended"]["partition"])
    reason = "ppm prediction"

    state.update(
        status="complete",
        completed_at=datetime.now(timezone.utc).isoformat(),
        best_partition=best,
        selection_reason=reason,
    )
    best_ratio: float | None = None
    for item in analysis.get("candidates", []):
        if tuple(item["partition"]) == tuple(best):
            best_ratio = item.get("mamba_ratio")
            break
    if tuple(best) == tuple(baseline):
        best_ratio = (
            analysis.get("optimization", {}).get("current") or {}
        ).get("mamba_ratio")
    write_best_artifacts(
        args, run_dir, executable, selected, best, reason, mamba_ratio=best_ratio
    )
    write_json(run_dir / "result.json", state)
    if args.compare_tp:
        try:
            state["tp_comparison"] = run_tp_comparison(
                args, run_dir, executable, selected, best,
                baseline_partition=baseline,
                best_ratio=best_ratio,
            )
        except TuningError as exc:
            print(f"[warn] TP comparison failed: {exc}", flush=True)
            state["tp_comparison_error"] = str(exc)
        write_json(run_dir / "result.json", state)
    print(f"[done] best partition: {','.join(map(str, best))} ({reason})", flush=True)
    print(f"[done] launch script: {run_dir / 'launch_best.sh'}", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--draft-model-path", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--mamba-ssm-dtype", default="bfloat16")
    parser.add_argument("--mamba-full-memory-ratio", type=float, default=2.0)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="offered profiling concurrency (default: 32)",
    )
    parser.add_argument(
        "--fixed-active-requests",
        type=int,
        help=(
            "pin the baseline server's active request count via "
            "--max-running-requests; the run fails if the server cannot "
            "provide this active batch size"
        ),
    )
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument(
        "--prompt-seed",
        type=int,
        default=1234,
        help=(
            "seed of the synthetic prompt set shared by every benchmark call "
            "in one tuner run (default: 1234); keeps accept_len comparable "
            "across candidates and runs"
        ),
    )
    parser.add_argument(
        "--profile-output-tokens",
        type=int,
        help=(
            "decode length per profile-load request; default: 8192, must "
            "outlast the PPM collection window"
        ),
    )
    parser.add_argument("--profile-settle-s", type=float, default=5.0)
    parser.add_argument("--profile-load-timeout-s", type=float, default=180.0)
    parser.add_argument("--post-profile-cooldown-s", type=float, default=3.0)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument(
        "--validation-output-tokens",
        type=int,
        default=256,
        help=(
            "per-request decode length assumed by the capacity model's KV "
            "feasibility check and used by --compare-tp measurements "
            "(default: 256)"
        ),
    )
    parser.add_argument(
        "--validation-repeats",
        type=int,
        default=2,
        help="measurement repeats per --compare-tp point (default: 2)",
    )
    parser.add_argument(
        "--validation-timeout-s",
        type=float,
        default=1800.0,
        help="per-benchmark timeout for --compare-tp measurements",
    )
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--memory-reserve-gib", type=float, default=2.0)
    parser.add_argument("--server-startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="fixed tuning port; default: choose a free port")
    parser.add_argument("--final-port", type=int, default=30000)
    parser.add_argument("--visible-devices", help="override CUDA_VISIBLE_DEVICES")
    parser.add_argument(
        "--current-partition",
        help=(
            "explicit baseline partition (default: the uniform split used by "
            "a plain SGLang launch)"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--ppm-collect-s",
        type=float,
        default=30.0,
        help="PPM collection window in seconds (default: 30)",
    )
    parser.add_argument(
        "--ppm-max-messages",
        type=int,
        help="optional PPM message count that ends collection early",
    )
    parser.add_argument(
        "--ppm-saturation-bs",
        type=int,
        help=(
            "saturation threshold for the PPM work-conservation filter, in "
            "GLOBAL running requests (compared against the publisher's "
            "running_bs; per-slot bs only for legacy producers); default: "
            "--fixed-active-requests or the server's resolved "
            "max_running_requests"
        ),
    )
    parser.add_argument(
        "--t-comm-ms",
        type=float,
        help=(
            "fixed per-boundary PP communication cost in ms; default: parsed "
            "from the baseline's startup comm micro-benchmark "
            "(SGLANG_PP_COMM_BENCHMARK), falling back to 0 with a warning"
        ),
    )
    parser.add_argument(
        "--comm-benchmark-tokens",
        default="32,64,128,256,512",
        help=(
            "token sweep points for the baseline startup comm benchmark; "
            "should cover bs x block_size over the working range (default: "
            "32,64,128,256,512 for bs 4-64 at block 8)"
        ),
    )
    parser.add_argument(
        "--ratio-grid",
        default="0.25,0.5,1,2,4,8",
        help=(
            "log grid of --mamba-full-memory-ratio values swept per "
            "candidate partition (default: 0.25,0.5,1,2,4,8); the feasible "
            "point with the best unified throughput wins"
        ),
    )
    parser.add_argument(
        "--k-best",
        type=int,
        default=20,
        help="candidates kept per state in the k-best partition DP (default: 20)",
    )
    parser.add_argument(
        "--allow-draft-relocation",
        action="store_true",
        help=(
            "analysis only: let the optimizer enumerate draft block placement "
            "across ranks (default: draft locked to the last rank)"
        ),
    )
    parser.add_argument(
        "--reanalyze",
        type=Path,
        help=(
            "offline re-analysis of an existing ppm-mode run directory: reads "
            "baseline/ppm_snapshot.json + baseline/server.log (+ measured "
            "columns from legacy runs that recorded validations), refits "
            "stage/capacity models, and re-renders the report and sweep plot "
            "into <run_dir>/reanalysis/ without starting a server"
        ),
    )
    parser.add_argument(
        "--compare-tp",
        action="store_true",
        help=(
            "after PP tuning, also start a TP baseline (tp_size = pp_size x "
            "tp_size, pp_size = 1) and compare throughput at equal "
            "concurrency (min of both caps) and at each side's native capacity"
        ),
    )
    parser.add_argument(
        "--compare-tp-only",
        type=Path,
        help=(
            "skip PP baseline/profile/analysis and only run the TP "
            "comparison for an existing run directory, reading the best "
            "partition / ratio from its result.json"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not fetch a missing model config",
    )
    parser.add_argument("--sglang-executable")
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--server-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="additional arguments passed to `sglang serve`; must be last",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_tuner(args)
    except (TuningError, KeyboardInterrupt) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
