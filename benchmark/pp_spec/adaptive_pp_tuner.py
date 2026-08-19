#!/usr/bin/env python3
"""Profile, analyze, and validate an adaptive SGLang PP layer partition.

Only ``--pp-size`` and ``--tp-size`` are required.  The default workflow:

1. discovers a single-node GPU placement and the target model layer count;
2. starts an evenly partitioned baseline and captures a steady-decode trace;
3. infers stage costs and conservative memory limits from the trace and log;
4. solves the contiguous min-max layer partition problem; and
5. benchmarks the predicted candidates without profiler overhead.

Every run is self-contained under ``tuning_runs/`` and includes raw traces,
logs, measurements, a final JSON result, and a directly executable launch
script.  Child process groups are tracked explicitly; unrelated SGLang
servers are never killed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
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
TRACE_NAME = re.compile(
    r"^(?P<run>.+)-TP-(?P<tp>\d+)(?:-DP-\d+)?-PP-(?P<pp>\d+)"
    r"(?:-EP-\d+)?\.trace\.json(?:\.gz)?$"
)
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
    args: argparse.Namespace, executable: str, port: int
) -> list[str]:
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
        str(args.mamba_full_memory_ratio),
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


def resolve_comparison_active_requests(
    args: argparse.Namespace, baseline_capacity: dict[str, int]
) -> int | None:
    """Choose the common active batch used for partition validation.

    The baseline profile is already saturated by ``--concurrency``.  By
    default, candidate validation pins every server to that same effective
    active request count so partition compute cost is compared at one load.
    Capacity/service tuning remains available as an explicit opt-in.
    """
    if args.allow_variable_active_requests:
        return None
    if args.fixed_active_requests is not None:
        return args.fixed_active_requests
    baseline_limit = baseline_capacity.get("max_running_requests")
    if baseline_limit is None:
        raise TuningError(
            "Cannot derive the baseline active request limit from the server log. "
            "Use --fixed-active-requests explicitly, or opt into "
            "--allow-variable-active-requests."
        )
    return min(args.concurrency, baseline_limit)


def start_server(
    args: argparse.Namespace,
    executable: str,
    selected_gpus: Sequence[GPUInfo],
    partition: Sequence[int],
    phase_dir: Path,
) -> tuple[ManagedProcess, str, list[str]]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    port = args.port or find_free_port(args.host)
    command = build_server_command(args, executable, port)
    log_path = phase_dir / "server.log"
    log_handle = log_path.open("w")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu.index) for gpu in selected_gpus)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("SGLANG_FORCE_SHUTDOWN", "1")
    env.setdefault("SGLANG_PROFILE_WITH_STACK", "false")
    env.setdefault("SGLANG_PROFILE_RECORD_SHAPES", "false")
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
        "--request-timeout-s",
        str(args.request_timeout_s),
        "--cooldown-s",
        "0",
        "--output-dir",
        str(output_dir),
    ]


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
        concurrency,
        args.profile_output_tokens,
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


def wait_for_traces(
    trace_dir: Path, profile_id: str, expected: int, timeout_s: float
) -> list[Path]:
    deadline = time.monotonic() + timeout_s
    previous: tuple[tuple[str, int], ...] | None = None
    stable_polls = 0
    while time.monotonic() < deadline:
        paths = sorted(trace_dir.glob(f"{profile_id}-TP-*.trace.json.gz"))
        snapshot = tuple((path.name, path.stat().st_size) for path in paths)
        if len(paths) == expected and all(size > 1024 for _, size in snapshot):
            if snapshot == previous:
                stable_polls += 1
                if stable_polls >= 2:
                    return paths
            else:
                stable_polls = 0
        previous = snapshot
        time.sleep(2)
    found = sorted(path.name for path in trace_dir.glob("*.trace.json.gz"))
    raise TuningError(
        f"Expected {expected} stable trace files for {profile_id}, found {len(found)} "
        f"after {timeout_s:g}s: {found}"
    )


def representative_traces(paths: Sequence[Path], pp_size: int) -> list[Path]:
    by_pp: dict[int, dict[int, Path]] = {}
    for path in paths:
        match = TRACE_NAME.match(path.name)
        if match:
            by_pp.setdefault(int(match.group("pp")), {})[int(match.group("tp"))] = path
    if sorted(by_pp) != list(range(pp_size)):
        raise TuningError(
            f"Trace set has PP ranks {sorted(by_pp)}, expected {list(range(pp_size))}."
        )
    return [by_pp[rank][min(by_pp[rank])] for rank in range(pp_size)]


def run_validation(
    args: argparse.Namespace,
    base_url: str,
    partition: Sequence[int],
    phase_dir: Path,
    concurrency: int,
    server_log: Path,
) -> ValidationResult:
    repeats = []
    benchmark_logs = []
    requests = args.validation_requests or max(16, 2 * concurrency)
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


def analyze_traces(
    trace_inputs: Sequence[Path],
    num_layers: int,
    current_partition: Sequence[int],
    max_layers: Sequence[int],
    top_candidates: int,
    output_dir: Path,
) -> dict[str, Any]:
    sys.path.insert(0, str(SCRIPT_DIR))
    import auto_pp_partition

    namespace = argparse.Namespace(
        traces=list(trace_inputs),
        num_layers=num_layers,
        current_partition=",".join(map(str, current_partition)),
        fixed_overhead_ms=None,
        min_layers_per_stage=1,
        max_layers_per_stage=",".join(map(str, max_layers)),
        target_min_ratio=0.60,
        trim_samples=2,
        top_candidates=top_candidates,
        json_output=output_dir / "analysis.json",
    )
    report = io.StringIO()
    try:
        with contextlib.redirect_stdout(report):
            result = auto_pp_partition.run(namespace)
    except auto_pp_partition.AnalysisError as exc:
        raise TuningError(str(exc)) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.txt").write_text(report.getvalue())
    print(report.getvalue(), end="", flush=True)
    return result


def try_slim_trace(paths: Sequence[Path], output_path: Path) -> str | None:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "slim_trace.py"),
        *map(str, paths),
        "-o",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        return completed.stdout.strip()
    return None


def candidate_partitions(
    baseline: Sequence[int], analysis: dict[str, Any]
) -> list[tuple[int, ...]]:
    result = [tuple(baseline)]
    for item in analysis["candidates"]:
        partition = tuple(item["partition"])
        if partition not in result:
            result.append(partition)
    return result


def choose_best(
    validations: Sequence[ValidationResult],
    baseline: Sequence[int],
    min_improvement_pct: float,
) -> tuple[tuple[int, ...], str]:
    valid = [
        item for item in validations if item.status == "ok" and item.throughput_tok_s is not None
    ]
    if not valid:
        raise TuningError("No partition completed the validation benchmark.")
    best = max(valid, key=lambda item: item.throughput_tok_s or 0.0)
    baseline_result = next(
        (item for item in valid if item.partition == tuple(baseline)), None
    )
    if baseline_result is not None and best.partition != tuple(baseline):
        improvement = (
            (best.throughput_tok_s - baseline_result.throughput_tok_s)
            / baseline_result.throughput_tok_s
            * 100.0
        )
        if improvement < min_improvement_pct:
            return tuple(baseline), (
                f"best measured candidate improved throughput by only {improvement:.2f}%; "
                f"kept baseline because the threshold is {min_improvement_pct:.2f}%"
            )
        return best.partition, f"highest validated throughput ({improvement:.2f}% over baseline)"
    return best.partition, "highest validated throughput"


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
) -> None:
    visible = ",".join(str(gpu.index) for gpu in selected_gpus)
    command = build_server_command(args, executable, args.final_port)
    config = {
        "tp_size": args.tp_size,
        "pp_size": args.pp_size,
        "partition": tuple(partition),
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
    if args.allow_variable_active_requests and args.fixed_active_requests is not None:
        raise TuningError(
            "--allow-variable-active-requests and --fixed-active-requests are "
            "mutually exclusive."
        )
    if args.analyze_traces:
        run_dir = create_run_dir(args.output_root, args.tp_size, args.pp_size)
        analysis = analyze_traces(
            [args.analyze_traces],
            model.num_layers,
            baseline,
            (model.num_layers,) * args.pp_size,
            args.candidate_count,
            run_dir,
        )
        best = tuple(analysis["recommended"]["partition"])
        state = {
            "status": "analysis_only",
            "model": model,
            "baseline_partition": baseline,
            "analysis": analysis,
            "best_partition": best,
            "note": "Offline analysis only; candidates were not benchmarked.",
        }
        if args.pp_size > 1:
            value = ",".join(map(str, best))
            (run_dir / "best_config.env").write_text(
                f"export SGLANG_PP_LAYER_PARTITION={shlex.quote(value)}\n"
            )
        write_json(run_dir / "result.json", state)
        print(f"[done] offline analysis: {run_dir}")
        return run_dir

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
            "mode": (
                "explicit_fixed_active"
                if args.fixed_active_requests is not None
                else (
                    "variable_capacity"
                    if args.allow_variable_active_requests
                    else "baseline_fixed_active_auto"
                )
            ),
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

    run_dir = create_run_dir(args.output_root, args.tp_size, args.pp_size)
    state: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "model": model,
        "hardware": dry_plan,
        "baseline_partition": baseline,
        "validations": [],
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
        args, executable, selected, baseline, baseline_dir
    )
    raw_traces: list[Path] = []
    load: ManagedProcess | None = None
    try:
        load = start_profile_load(args, base_url, baseline_dir, concurrency)
        wait_for_load(load, args.profile_load_timeout_s, args.profile_settle_s)
        trace_dir = baseline_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        profile_id = f"adaptive_pp_{int(time.time())}"
        response = http_request(
            f"{base_url}/start_profile",
            method="POST",
            payload={
                "output_dir": str(trace_dir.resolve()),
                "profile_id": profile_id,
                "num_steps": args.profile_steps,
                "activities": ["CPU", "GPU"],
                "with_stack": False,
                "record_shapes": False,
            },
            timeout_s=30,
        )
        print(f"[profile] start response: {response}", flush=True)
        raw_traces = wait_for_traces(
            trace_dir,
            profile_id,
            args.pp_size * args.tp_size,
            args.profile_timeout_s,
        )
        print(f"[ok] captured {len(raw_traces)} rank traces", flush=True)
        load.stop(timeout_s=10)
        load = None
    finally:
        if load is not None:
            load.stop(timeout_s=10)
        server.stop()

    state["baseline_command"] = baseline_command
    state["raw_traces"] = [str(path) for path in raw_traces]
    baseline_capacity = parse_runtime_capacity(
        baseline_dir / "server.log"
    )
    state["baseline_runtime_capacity"] = baseline_capacity
    comparison_active_requests = resolve_comparison_active_requests(
        args, baseline_capacity
    )
    state["load_control"] = {
        "offered_concurrency": concurrency,
        "mode": (
            "variable_capacity"
            if comparison_active_requests is None
            else "fixed_active"
        ),
        "comparison_active_requests": comparison_active_requests,
        "source": (
            "per-partition capacity"
            if comparison_active_requests is None
            else (
                "explicit --fixed-active-requests"
                if args.fixed_active_requests is not None
                else "min(offered concurrency, uniform baseline capacity)"
            )
        ),
    }
    if comparison_active_requests is None:
        print(
            "[load] validation uses each partition's native active capacity",
            flush=True,
        )
        validation_args = args
    else:
        print(
            f"[load] validation fixed at {comparison_active_requests} active "
            f"requests with offered C{concurrency}",
            flush=True,
        )
        validation_args = argparse.Namespace(**vars(args))
        validation_args.fixed_active_requests = comparison_active_requests
    max_layers, memory_details = parse_memory_limits(
        baseline_dir / "server.log", baseline, args.memory_reserve_gib
    )
    state["memory_limits"] = {
        "max_layers_per_stage": max_layers,
        "details": memory_details,
    }
    write_json(run_dir / "result.json", state)

    analysis = analyze_traces(
        raw_traces,
        model.num_layers,
        baseline,
        max_layers,
        args.candidate_count,
        baseline_dir,
    )
    state["analysis"] = analysis
    slim_warning = try_slim_trace(raw_traces, baseline_dir / "traces" / "lean.json.gz")
    if slim_warning:
        state["slim_trace_warning"] = slim_warning
    write_json(run_dir / "result.json", state)

    partitions = candidate_partitions(baseline, analysis)
    validations: list[ValidationResult] = []
    if not args.skip_validation:
        for partition in partitions:
            phase_name = (
                "validation_baseline"
                if partition == baseline
                else "candidate_p" + "-".join(map(str, partition))
            )
            phase_dir = run_dir / phase_name
            candidate_server: ManagedProcess | None = None
            try:
                candidate_server, candidate_url, _ = start_server(
                    validation_args, executable, selected, partition, phase_dir
                )
                result = run_validation(
                    validation_args,
                    candidate_url,
                    partition,
                    phase_dir,
                    concurrency,
                    candidate_server.log_path,
                )
            except TuningError as exc:
                result = ValidationResult(
                    tuple(partition),
                    "startup_failed",
                    None,
                    [],
                    str(phase_dir / "server.log"),
                    [],
                    str(exc),
                )
            finally:
                if candidate_server is not None:
                    candidate_server.stop()
            validations.append(result)
            state["validations"] = [asdict(item) for item in validations]
            write_json(run_dir / "result.json", state)
            metric = (
                f"{result.throughput_tok_s:.2f} tok/s"
                if result.throughput_tok_s is not None
                else result.status
            )
            if result.runtime_capacity:
                metric += (
                    ", active cap "
                    f"{result.runtime_capacity.get('max_running_requests', 'unknown')}"
                )
            print(f"[candidate] {partition}: {metric}", flush=True)

        best, reason = choose_best(
            validations, baseline, args.min_improvement_pct
        )
    else:
        best = tuple(analysis["recommended"]["partition"])
        reason = "profiler prediction; candidate validation was skipped"

    state.update(
        status="complete",
        completed_at=datetime.now(timezone.utc).isoformat(),
        validations=[asdict(item) for item in validations],
        best_partition=best,
        selection_reason=reason,
    )
    write_best_artifacts(args, run_dir, executable, selected, best, reason)
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
        help="offered profiling and validation concurrency (default: 32)",
    )
    parser.add_argument(
        "--fixed-active-requests",
        type=int,
        help=(
            "override the automatically derived common active request count; "
            "the run fails if a candidate cannot provide this active batch size"
        ),
    )
    parser.add_argument(
        "--allow-variable-active-requests",
        action="store_true",
        help=(
            "compare native service throughput, allowing each partition to use "
            "its own memory-limited active request capacity"
        ),
    )
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--profile-steps", type=int, default=50)
    parser.add_argument("--profile-output-tokens", type=int, default=4096)
    parser.add_argument("--profile-settle-s", type=float, default=5.0)
    parser.add_argument("--profile-load-timeout-s", type=float, default=180.0)
    parser.add_argument("--profile-timeout-s", type=float, default=300.0)
    parser.add_argument("--post-profile-cooldown-s", type=float, default=3.0)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--validation-output-tokens", type=int, default=256)
    parser.add_argument("--validation-requests", type=int)
    parser.add_argument("--validation-repeats", type=int, default=2)
    parser.add_argument("--validation-timeout-s", type=float, default=1800.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--min-improvement-pct", type=float, default=1.0)
    parser.add_argument("--memory-reserve-gib", type=float, default=2.0)
    parser.add_argument("--server-startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="fixed tuning port; default: choose a free port")
    parser.add_argument("--final-port", type=int, default=30000)
    parser.add_argument("--visible-devices", help="override CUDA_VISIBLE_DEVICES")
    parser.add_argument("--current-partition", help="partition used by --analyze-traces")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--analyze-traces",
        type=Path,
        help="offline analysis of an existing trace directory",
    )
    parser.add_argument("--skip-validation", action="store_true")
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
