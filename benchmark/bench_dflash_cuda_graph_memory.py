#!/usr/bin/env python3
"""Measure Qwen3.5 DFlash decode CUDA graph memory.

The server profile intentionally matches ``benchmark/pp_spec/run_bench.sh``:
DFLASH block size 16, FlashInfer target/draft attention, Triton linear
attention, FP32 ReplaySSM, page size 1, and static ragged verify. Each matrix
point uses its graph batch size as ``max_running_requests``, just like the TP
path in ``run_bench.sh``.

Two graph-only settings intentionally differ from the serving benchmark:
prefill graphs and server warmup are disabled, and the KV pool is capped at
4096 tokens so unrelated cache capacity does not consume graph headroom.

Each point starts a fresh server, reads the target/draft graph counters from
``/server_info``, and tears down the full server process group. The CSV is
rewritten after every point so an interrupted run keeps all completed data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import msgspec
from msgspec.structs import replace

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DFLASH_BLOCK_SIZE = 16
ATTENTION_BACKEND = "flashinfer"
DRAFT_ATTENTION_BACKEND = "flashinfer"
LINEAR_ATTENTION_BACKEND = "triton"
MAMBA_SSM_DTYPE = "float32"
MAMBA_FULL_MEMORY_RATIO = 0.9
PAGE_SIZE = 1
MAX_TOTAL_TOKENS = 4096
RANDOM_SEED = 1
RAGGED_VERIFY_MODE = "static"
DEFAULT_BATCH_SIZES = (32, 64, 96, 128)
TARGET_COLUMN = "target_verify_incremental_gib_per_gpu"
DRAFT_COLUMN = "draft_decode_incremental_gib_per_gpu"
SERVER_ENV_OVERRIDES = {
    "PYTHONUNBUFFERED": "1",
    "SGLANG_RAGGED_VERIFY_MODE": RAGGED_VERIFY_MODE,
    "SGL_FORCE_SHUTDOWN": "1",
}


class ModelProfile(msgspec.Struct, frozen=True):
    key: str
    label: str
    target_model: str
    draft_model: str
    tp_size: int
    mem_fraction_static: float


MODEL_PROFILES = {
    "9b": ModelProfile(
        key="9b",
        label="Qwen3.5-9B",
        target_model="Qwen/Qwen3.5-9B",
        draft_model="z-lab/Qwen3.5-9B-DFlash",
        tp_size=1,
        mem_fraction_static=0.70,
    ),
    "27b": ModelProfile(
        key="27b",
        label="Qwen3.5-27B",
        target_model="Qwen/Qwen3.5-27B",
        draft_model="z-lab/Qwen3.5-27B-DFlash",
        tp_size=2,
        # FP32 ReplaySSM at BS128 needs about 10 GiB of state per TP rank.
        mem_fraction_static=0.82,
    ),
}

CSV_FIELDS = (
    "status",
    "model",
    "max_cuda_graph_bs",
    "tp_size",
    TARGET_COLUMN,
    DRAFT_COLUMN,
    "total_decode_incremental_gib_per_gpu",
)


class MeasurementError(RuntimeError):
    pass


class StopRequested(KeyboardInterrupt):
    def __init__(self, signum: int):
        super().__init__(f"received signal {signum}")
        self.exit_code = 128 + signum


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative number, got {value!r}"
        )
    return parsed


def fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError(
            f"expected a fraction in (0, 1], got {value!r}"
        )
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}")
    return parsed


def model_fraction(value: str) -> tuple[str, float]:
    key, separator, raw = value.partition("=")
    if not separator or key not in MODEL_PROFILES:
        raise argparse.ArgumentTypeError(
            f"expected MODEL=FRACTION with MODEL in {sorted(MODEL_PROFILES)}, "
            f"got {value!r}"
        )
    return key, fraction(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure Qwen3.5 DFlash target/draft CUDA graph memory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_PROFILES),
        default=list(MODEL_PROFILES),
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=positive_int,
        default=list(DEFAULT_BATCH_SIZES),
        metavar="N",
    )
    parser.add_argument(
        "--tp-sizes",
        nargs="+",
        type=positive_int,
        default=None,
        metavar="N",
        help="TP sizes to sweep for every selected model (uses model defaults when omitted).",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="CUDA_VISIBLE_DEVICES for every server (inherits when omitted).",
    )
    parser.add_argument(
        "--mem-fraction-static",
        action="append",
        type=model_fraction,
        default=None,
        metavar="MODEL=FRACTION",
        help="Override a model profile's mem fraction static (repeatable).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--startup-timeout",
        type=positive_float,
        default=600.0,
        help="Seconds allowed for model load and graph capture.",
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=nonnegative_float,
        default=60.0,
        help="Graceful shutdown timeout before SIGTERM/SIGKILL.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=nonnegative_float,
        default=10.0,
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def selected_profiles(args: argparse.Namespace) -> list[ModelProfile]:
    overrides = dict(args.mem_fraction_static or [])
    profiles = []
    for key in dict.fromkeys(args.models):
        base = MODEL_PROFILES[key]
        if key in overrides:
            base = replace(base, mem_fraction_static=overrides[key])
        tp_sizes = sorted(set(args.tp_sizes or [base.tp_size]))
        profiles.extend(replace(base, tp_size=tp_size) for tp_size in tp_sizes)
    return profiles


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    profiles: Sequence[ModelProfile],
) -> None:
    visible = args.cuda_visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return
    visible_count = len([item for item in visible.split(",") if item.strip()])
    required = max(profile.tp_size for profile in profiles)
    if visible_count < required:
        parser.error(
            f"CUDA_VISIBLE_DEVICES exposes {visible_count} device(s), "
            f"but the selected models require TP{required}"
        )


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "benchmark" / "cuda_graph_memory_results" / stamp


def prepare_output_dir(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise MeasurementError(f"output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def find_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_server_command(
    args: argparse.Namespace,
    profile: ModelProfile,
    batch_size: int,
    port: int,
) -> list[str]:
    return [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        profile.target_model,
        "--revision",
        "main",
        "--tp-size",
        str(profile.tp_size),
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        profile.draft_model,
        "--speculative-draft-model-revision",
        "main",
        "--speculative-dflash-block-size",
        str(DFLASH_BLOCK_SIZE),
        "--attention-backend",
        ATTENTION_BACKEND,
        "--speculative-draft-attention-backend",
        DRAFT_ATTENTION_BACKEND,
        "--linear-attn-backend",
        LINEAR_ATTENTION_BACKEND,
        "--mamba-ssm-dtype",
        MAMBA_SSM_DTYPE,
        "--mamba-full-memory-ratio",
        str(MAMBA_FULL_MEMORY_RATIO),
        "--enable-linear-replayssm-spec",
        "--disable-radix-cache",
        "--max-running-requests",
        str(batch_size),
        # This is a graph-memory benchmark, so keep the unrelated KV pool small
        # and constant instead of consuming all of mem_fraction_static.
        "--max-total-tokens",
        str(MAX_TOTAL_TOKENS),
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-max-bs-decode",
        str(batch_size),
        "--cuda-graph-backend-prefill",
        "disabled",
        "--mem-fraction-static",
        str(profile.mem_fraction_static),
        "--page-size",
        str(PAGE_SIZE),
        "--load-balance-method",
        "round_robin",
        "--random-seed",
        str(RANDOM_SEED),
        "--trust-remote-code",
        "--skip-server-warmup",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def server_env_overrides(args: argparse.Namespace) -> dict[str, str]:
    overrides = dict(SERVER_ENV_OVERRIDES)
    if args.cuda_visible_devices is not None:
        overrides["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return overrides


def server_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(server_env_overrides(args))
    return env


def fetch_server_info(
    opener: urllib.request.OpenerDirector, server_info_url: str
) -> dict[str, Any]:
    request = urllib.request.Request(
        server_info_url, headers={"Accept": "application/json"}
    )
    with opener.open(request, timeout=10) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise MeasurementError("/server_info did not return a JSON object")
    return payload


def tail_text(path: Path, max_bytes: int = 16 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(handle.tell() - max_bytes, 0))
            return handle.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return f"<could not read log: {exc}>"


def wait_for_server(
    process: subprocess.Popen,
    base_url: str,
    log_path: Path,
    timeout: float,
) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    server_info_url = base_url + "/server_info"
    deadline = time.monotonic() + timeout
    last_error = "server has not accepted a connection"
    while True:
        return_code = process.poll()
        if return_code is not None:
            raise MeasurementError(
                f"server exited with code {return_code} before becoming ready\n"
                f"--- log tail ---\n{tail_text(log_path)}"
            )
        try:
            return fetch_server_info(opener, server_info_url)
        except (json.JSONDecodeError, OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MeasurementError(
                f"server did not become ready within {timeout:.1f}s "
                f"(last error: {last_error})\n--- log tail ---\n{tail_text(log_path)}"
            )
        time.sleep(min(1.0, remaining))


def terminate_process_group(process: subprocess.Popen | None, timeout: float) -> None:
    if process is None:
        return
    process_group = process.pid

    def group_exists() -> bool:
        try:
            os.killpg(process_group, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def drain(sig: signal.Signals, grace: float) -> bool:
        """Signal the process group and wait; return True once it is gone."""
        try:
            os.killpg(process_group, sig)
        except ProcessLookupError:
            return True
        deadline = time.monotonic() + grace
        while group_exists() and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.2)
        return not group_exists()

    if not drain(signal.SIGINT, timeout):
        if not drain(signal.SIGTERM, 5.0):
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MeasurementError(f"{field} must be an object")
    return value


def finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementError(f"{field} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise MeasurementError(f"{field} must be finite and non-negative")
    return result


def expect(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise MeasurementError(f"resolved {field}={actual!r}, expected {expected!r}")


def extract_measurement(
    server_info: dict[str, Any],
    profile: ModelProfile,
    batch_size: int,
) -> dict[str, Any]:
    expect(server_info.get("speculative_algorithm"), "DFLASH", "algorithm")
    states = server_info.get("internal_states")
    if not isinstance(states, list) or len(states) != 1:
        count = len(states) if isinstance(states, list) else "non-list"
        raise MeasurementError(
            f"TP-only benchmark requires exactly one internal_state, got {count}"
        )
    state = mapping(states[0], "internal_states[0]")
    graph = mapping(
        mapping(state.get("memory_usage"), "memory_usage").get("graph"),
        "memory_usage.graph",
    )
    target_gib = finite_number(graph.get("target_verify"), "graph.target_verify")
    draft_gib = finite_number(graph.get("draft_decode"), "graph.draft_decode")

    cuda_graph = mapping(state.get("cuda_graph_config"), "cuda_graph_config")
    decode = mapping(cuda_graph.get("decode"), "cuda_graph_config.decode")
    prefill = mapping(cuda_graph.get("prefill"), "cuda_graph_config.prefill")
    expect(decode.get("backend"), "full", "cuda_graph_config.decode.backend")
    expect(decode.get("max_bs"), batch_size, "cuda_graph_config.decode.max_bs")
    capture_bs = decode.get("bs")
    if not isinstance(capture_bs, list) or batch_size not in capture_bs:
        raise MeasurementError(f"decode capture buckets do not contain {batch_size}")
    expect(prefill.get("backend"), "disabled", "cuda_graph_config.prefill.backend")

    return {
        "status": "ok",
        "model": profile.label,
        "max_cuda_graph_bs": batch_size,
        "tp_size": profile.tp_size,
        TARGET_COLUMN: target_gib,
        DRAFT_COLUMN: draft_gib,
        "total_decode_incremental_gib_per_gpu": round(target_gib + draft_gib, 3),
    }


def error_row(profile: ModelProfile, batch_size: int) -> dict[str, Any]:
    row = dict.fromkeys(CSV_FIELDS, "")
    row.update(
        {
            "status": "error",
            "model": profile.label,
            "max_cuda_graph_bs": batch_size,
            "tp_size": profile.tp_size,
        }
    )
    return row


def write_results(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    destination = output_dir / "results.csv"
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


def run_case(
    args: argparse.Namespace,
    profile: ModelProfile,
    batch_size: int,
    run_dir: Path,
    port: int,
) -> dict[str, Any]:
    log_path = run_dir / "server.log"
    command = build_server_command(args, profile, batch_size, port)
    process = None

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + shlex.join(command) + "\n\n")
        log_handle.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=server_environment(args),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            server_info = wait_for_server(
                process,
                f"http://127.0.0.1:{port}",
                log_path,
                args.startup_timeout,
            )
            return extract_measurement(server_info, profile, batch_size)
        finally:
            terminate_process_group(process, args.shutdown_timeout)


def run(args: argparse.Namespace, profiles: Sequence[ModelProfile]) -> int:
    batch_sizes = sorted(set(args.batch_sizes))
    matrix = [
        (profile, batch_size) for profile in profiles for batch_size in batch_sizes
    ]
    if args.dry_run:
        prefix = " ".join(
            f"{key}={value}" for key, value in server_env_overrides(args).items()
        )
        for index, (profile, batch_size) in enumerate(matrix):
            command = build_server_command(args, profile, batch_size, 30000 + index)
            print(f"{prefix} {shlex.join(command)}")
        return 0

    output_dir = prepare_output_dir(args.output_dir or default_output_dir())
    rows: list[dict[str, Any]] = []
    failed = 0
    write_results(output_dir, rows)
    print(f"Output: {output_dir}", flush=True)

    interrupted = False
    exit_code = 130
    for index, (profile, batch_size) in enumerate(matrix, start=1):
        run_dir = output_dir / f"{profile.key}_tp{profile.tp_size}_bs{batch_size}"
        run_dir.mkdir()
        port = find_open_port()
        print(
            f"[{index}/{len(matrix)}] {profile.label}, "
            f"max_cuda_graph_bs={batch_size}, TP={profile.tp_size}, "
            f"mem_fraction={profile.mem_fraction_static}",
            flush=True,
        )
        print(f"  log: {run_dir / 'server.log'}", flush=True)
        try:
            row = run_case(args, profile, batch_size, run_dir, port)
            rows.append(row)
            print(
                "  per-GPU: "
                f"target={row['target_verify_incremental_gib_per_gpu']:.3f} GiB, "
                f"draft={row['draft_decode_incremental_gib_per_gpu']:.3f} GiB, "
                f"total={row['total_decode_incremental_gib_per_gpu']:.3f} GiB",
                flush=True,
            )
        except (StopRequested, KeyboardInterrupt) as exc:
            interrupted = True
            exit_code = exc.exit_code if isinstance(exc, StopRequested) else 130
            rows.append(error_row(profile, batch_size))
            failed += 1
        except Exception as exc:  # noqa: BLE001 - record one failed matrix point.
            error = f"{type(exc).__name__}: {exc}"
            rows.append(error_row(profile, batch_size))
            failed += 1
            print(f"  ERROR: {error}", file=sys.stderr, flush=True)
        finally:
            write_results(output_dir, rows)

        if interrupted or (args.fail_fast and failed):
            break
        if args.cooldown_seconds and index < len(matrix):
            time.sleep(args.cooldown_seconds)

    print(
        f"Completed {len(rows) - failed}/{len(matrix)} matrix entries; "
        f"failures={failed}",
        flush=True,
    )
    print(f"CSV:  {output_dir / 'results.csv'}", flush=True)
    if interrupted:
        return exit_code
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    profiles = selected_profiles(args)
    validate_args(parser, args, profiles)

    def stop(signum, _frame):
        raise StopRequested(signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop)
    try:
        return run(args, profiles)
    except MeasurementError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
