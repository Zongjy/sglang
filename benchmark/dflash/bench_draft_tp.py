#!/usr/bin/env python3
"""Benchmark DFlash draft-forward latency across tensor-parallel sizes.

The benchmark deliberately reports two latency scopes:

* ``draft_forward`` is the GPU work correlated with the existing
  ``sglang.dflash.draft_model_forward`` profiler marker. By default, the runner
  keeps the sampler eager so this scope contains the native DFlash transformer
  and its TP collectives, but excludes target embedding, target ``lm_head``,
  and proposal sampling.
* ``e2e`` comes from ``sglang.benchmark.one_batch_server`` and covers the full
  speculative request. It is useful context, but it cannot isolate draft TP
  because SGLang currently applies the same ``--tp-size`` to target and draft.

Each TP size runs in a fresh subprocess so distributed/runtime state and CUDA
graphs cannot leak between configurations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

try:
    from trace_parser import DEFAULT_MARKER, TraceParseError, summarize_traces
except ImportError:  # Imported as a namespace-package module.
    from benchmark.dflash.trace_parser import (  # type: ignore[no-redef]
        DEFAULT_MARKER,
        TraceParseError,
        summarize_traces,
    )


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_DRAFT_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
DEFAULT_BATCH_SIZES = (8, 16, 32, 64)
DEFAULT_TP_SIZES = (1, 2, 4)
PROFILE_BATCH_RE = re.compile(r"(?:^|-)bs-(\d+)-il-(\d+)(?:-|$)")
CONTROLLED_CHILD_FLAGS = (
    "--model",
    "--model-path",
    "--speculative-algorithm",
    "--speculative-draft-model",
    "--speculative-draft-model-path",
    "--speculative-dflash-block-size",
    "--speculative-num-draft-tokens",
    "--tp",
    "--tp-size",
    "--tensor-parallel-size",
    "--pp",
    "--pp-size",
    "--pipeline-parallel-size",
    "--dp-size",
    "--nnodes",
    "--node-rank",
    "--dist-init-addr",
    "--batch-size",
    "--input-len",
    "--output-len",
    "--temperature",
    "--client-stream-interval",
    "--input-len-step-percentage",
    "--dataset-name",
    "--dataset-path",
    "--fixed-prompt-file",
    "--apply-chat-template",
    "--parallel-batch",
    "--enable-multi-batch",
    "--cache-hit-rate",
    "--fake-prefill",
    "--max-running-requests",
    "--cuda-graph-bs",
    "--cuda-graph-bs-decode",
    "--cuda-graph-bs-prefill",
    "--cuda-graph-config",
    "--cuda-graph-backend",
    "--cuda-graph-backend-decode",
    "--cuda-graph-backend-prefill",
    "--cuda-graph-tc-compiler",
    "--cuda-graph-max-bs",
    "--cuda-graph-max-bs-decode",
    "--cuda-graph-max-bs-prefill",
    "--disable-cuda-graph",
    "--disable-prefill-cuda-graph",
    "--disable-decode-cuda-graph",
    "--disable-cuda-graph-padding",
    "--debug-cuda-graph",
    "--attention-backend",
    "--speculative-draft-attention-backend",
    "--dtype",
    "--mem-fraction-static",
    "--page-size",
    "--disable-overlap-schedule",
    "--disable-radix-cache",
    "--profile",
    "--profile-activities",
    "--profile-start-step",
    "--profile-steps",
    "--profile-prefix",
    "--profile-output-dir",
    "--profile-by-stage",
    "--result-filename",
    "--run-name",
    "--skip-warmup",
    "--seed",
    "--random-seed",
    "--log-level",
    "--base-url",
    "--host",
    "--port",
)


class BenchmarkError(RuntimeError):
    pass


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative number, got {value!r}"
        )
    return parsed


def _contains_flag(arguments: Sequence[str], flag: str) -> bool:
    return any(value == flag or value.startswith(flag + "=") for value in arguments)


def _validate_extra_server_args(arguments: Sequence[str]) -> None:
    collisions = [
        flag for flag in CONTROLLED_CHILD_FLAGS if _contains_flag(arguments, flag)
    ]
    if collisions:
        raise BenchmarkError(
            "--extra-server-args cannot override benchmark-controlled flags: "
            + ", ".join(collisions)
        )


def _ensure_port_available(port: int) -> None:
    if not 1 <= port <= 65535:
        raise BenchmarkError(f"invalid benchmark port: {port}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise BenchmarkError(
            f"benchmark port 127.0.0.1:{port} is unavailable: {exc}"
        ) from exc


def _run_capture(command: Sequence[str], *, cwd: Path) -> str:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError) and exc.stdout:
            return exc.stdout.strip()
        return f"unavailable: {exc}"


def _visible_gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except (ImportError, RuntimeError):
        return 0


def _hardware_metadata(repo_root: Path) -> dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_gpu_count": _visible_gpu_count(),
        "gpus": _run_capture(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            cwd=repo_root,
        ),
        "topology": _run_capture(["nvidia-smi", "topo", "-m"], cwd=repo_root),
        "git_commit": _run_capture(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "git_branch": _run_capture(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root
        ),
    }


def _case_command(
    args: argparse.Namespace,
    *,
    tp_size: int,
    result_path: Path,
    profile_dir: Path,
) -> list[str]:
    max_batch = max(args.batch_sizes)
    command = [
        args.python,
        "-m",
        "sglang.benchmark.one_batch_server",
        "--model-path",
        args.model_path,
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        args.draft_model_path,
        "--speculative-dflash-block-size",
        str(args.block_size),
        "--tp-size",
        str(tp_size),
        "--batch-size",
        *(str(value) for value in args.batch_sizes),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--max-running-requests",
        str(max_batch),
        "--cuda-graph-bs-decode",
        *(str(value) for value in args.batch_sizes),
        "--disable-prefill-cuda-graph",
        "--attention-backend",
        args.attention_backend,
        "--speculative-draft-attention-backend",
        args.draft_attention_backend,
        "--dtype",
        args.dtype,
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--page-size",
        str(args.page_size),
        "--disable-overlap-schedule",
        "--disable-radix-cache",
        "--trust-remote-code",
        "--profile",
        "--profile-activities",
        "CPU",
        "GPU",
        "--profile-start-step",
        str(args.profile_start_step),
        "--profile-steps",
        str(args.profile_steps),
        "--profile-prefix",
        f"dflash-world{tp_size}-",
        "--profile-output-dir",
        str(profile_dir),
        "--result-filename",
        str(result_path),
        "--run-name",
        f"dflash-draft-tp{tp_size}",
        "--skip-warmup",
        "--seed",
        str(args.seed),
        "--random-seed",
        str(args.seed),
        "--log-level",
        args.log_level,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.base_port + tp_size),
    ]
    if args.disable_cuda_graph:
        command.append("--disable-cuda-graph")
    command.extend(args.extra_server_args)
    return command


def _stream_command(
    command: Sequence[str], *, cwd: Path, log_path: Path, env: dict[str, str]
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise BenchmarkError(
            f"benchmark subprocess exited with code {return_code}; see {log_path}"
        )


def _child_environment(
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, str]]:
    overrides = {
        "TOKENIZERS_PARALLELISM": "false",
        "SGLANG_DFLASH_EAGER_DRAFT_SAMPLER": (
            "1" if args.draft_sampler_scope == "transformer-only" else "0"
        ),
    }
    env = dict(os.environ)
    env.update(overrides)
    return env, overrides


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkError(f"invalid JSON at {path}:{line_number}: {exc}")
            if not isinstance(value, dict):
                raise BenchmarkError(f"expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _unprofiled_e2e_rows(path: Path) -> dict[int, dict[str, Any]]:
    # one_batch_server writes each case once for its normal run and once while
    # profiling. Keep the first result so Kineto overhead does not pollute E2E.
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(path):
        grouped[int(row["batch_size"])].append(row)
    return {batch_size: values[0] for batch_size, values in grouped.items()}


def _batch_from_trace_name(path: Path) -> int | None:
    match = PROFILE_BATCH_RE.search(path.name)
    return int(match.group(1)) if match else None


def _profile_traces_by_batch(profile_root: Path) -> dict[int, list[Path]]:
    traces: dict[int, list[Path]] = defaultdict(list)
    for pattern in ("*.trace.json", "*.trace.json.gz"):
        for path in profile_root.rglob(pattern):
            batch_size = _batch_from_trace_name(path)
            if batch_size is not None:
                traces[batch_size].append(path)
    return {batch_size: sorted(paths) for batch_size, paths in traces.items()}


def _metric(summary: dict[str, Any], metric: str, statistic: str) -> float | None:
    value = summary.get(metric, {}).get(statistic)
    return float(value) if value is not None else None


def _conservative_metric(
    summary: dict[str, Any], metric: str, statistic: str
) -> float | None:
    """Return the slowest-rank statistic rather than a pooled-rank percentile."""

    if metric == "critical_path_ms" and statistic == "p50":
        value = summary.get("iteration_critical_path_ms")
        if value is not None:
            return float(value)
    candidates = [
        _metric(rank_summary, metric, statistic)
        for rank_summary in (summary.get("per_rank") or {}).values()
    ]
    concrete = [value for value in candidates if value is not None]
    return max(concrete) if concrete else _metric(summary, metric, statistic)


def _critical_p50(row: dict[str, Any]) -> float | None:
    summary = row.get("draft_forward") or {}
    return _conservative_metric(summary, "critical_path_ms", "p50")


def _validate_trace_summary(summary: dict[str, Any], tp_size: int) -> None:
    expected_ranks = {str(rank) for rank in range(tp_size)}
    actual_ranks = set((summary.get("per_rank") or {}).keys())
    if actual_ranks != expected_ranks:
        raise TraceParseError(
            f"expected TP ranks {sorted(expected_ranks)}, found {sorted(actual_ranks)}"
        )
    sample_counts = {
        int(rank_summary.get("sample_count", 0))
        for rank_summary in summary["per_rank"].values()
    }
    if len(sample_counts) != 1 or next(iter(sample_counts), 0) <= 0:
        raise TraceParseError(
            f"per-rank draft sample counts differ or are empty: {sorted(sample_counts)}"
        )


def _classify(change_pct: float, threshold_pct: float) -> str:
    if change_pct > threshold_pct:
        return "slower"
    if change_pct < -threshold_pct:
        return "faster"
    return "within_threshold"


def _add_comparisons(rows: list[dict[str, Any]], threshold_pct: float) -> None:
    baselines = {
        int(row["request_batch_size"]): _critical_p50(row)
        for row in rows
        if row.get("status") == "ok" and int(row["tp_size"]) == 1
    }
    for row in rows:
        current = _critical_p50(row)
        baseline = baselines.get(int(row["request_batch_size"]))
        if current is None or baseline is None or current <= 0 or baseline <= 0:
            row["vs_tp1"] = None
            continue
        change_pct = (current / baseline - 1.0) * 100.0
        row["vs_tp1"] = {
            "speedup": baseline / current,
            "latency_change_pct": change_pct,
            "classification": _classify(change_pct, threshold_pct),
            "classification_threshold_pct": threshold_pct,
        }


def analyze_tp_directory(
    tp_dir: Path,
    *,
    tp_size: int,
    batch_sizes: Sequence[int],
    block_size: int,
    marker: str,
) -> list[dict[str, Any]]:
    e2e = _unprofiled_e2e_rows(tp_dir / "one_batch_results.jsonl")
    traces = _profile_traces_by_batch(tp_dir / "profiles")
    rows = []
    for batch_size in batch_sizes:
        case_traces = traces.get(batch_size, [])
        row: dict[str, Any] = {
            "status": "ok",
            "tp_size": tp_size,
            "request_batch_size": batch_size,
            "block_size": block_size,
            "forward_token_rows": batch_size * block_size,
            "marker": marker,
            "trace_files": [str(path) for path in case_traces],
            "e2e": e2e.get(batch_size),
        }
        if not case_traces:
            row.update(status="error", error="no profiler trace found")
        else:
            try:
                summary = summarize_traces(case_traces, marker_name=marker)
                _validate_trace_summary(summary, tp_size)
                row["draft_forward"] = summary
            except (OSError, TraceParseError, ValueError) as exc:
                row.update(status="error", error=str(exc))
        rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "status",
        "tp_size",
        "request_batch_size",
        "block_size",
        "forward_token_rows",
        "samples_per_rank",
        "rank_sample_count_total",
        "draft_critical_p50_ms",
        "draft_critical_p90_ms",
        "draft_critical_mean_ms",
        "draft_gpu_busy_p50_ms",
        "draft_nccl_p50_ms",
        "draft_nccl_share_p50_pct",
        "speedup_vs_tp1",
        "latency_change_vs_tp1_pct",
        "classification_vs_tp1",
        "e2e_latency_s",
        "output_throughput_tok_s",
        "accept_length",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            draft = row.get("draft_forward") or {}
            critical_p50 = _conservative_metric(draft, "critical_path_ms", "p50")
            nccl_p50 = _conservative_metric(draft, "nccl_ms", "p50")
            comparison = row.get("vs_tp1") or {}
            e2e = row.get("e2e") or {}
            per_rank_counts = [
                int(value.get("sample_count", 0))
                for value in (draft.get("per_rank") or {}).values()
            ]
            writer.writerow(
                {
                    "status": row.get("status"),
                    "tp_size": row.get("tp_size"),
                    "request_batch_size": row.get("request_batch_size"),
                    "block_size": row.get("block_size"),
                    "forward_token_rows": row.get("forward_token_rows"),
                    "samples_per_rank": (
                        per_rank_counts[0] if per_rank_counts else None
                    ),
                    "rank_sample_count_total": (
                        draft.get("critical_path_ms") or {}
                    ).get("count"),
                    "draft_critical_p50_ms": critical_p50,
                    "draft_critical_p90_ms": _conservative_metric(
                        draft, "critical_path_ms", "p90"
                    ),
                    "draft_critical_mean_ms": _conservative_metric(
                        draft, "critical_path_ms", "mean"
                    ),
                    "draft_gpu_busy_p50_ms": _conservative_metric(
                        draft, "gpu_busy_ms", "p50"
                    ),
                    "draft_nccl_p50_ms": nccl_p50,
                    "draft_nccl_share_p50_pct": (
                        100.0 * nccl_p50 / critical_p50
                        if nccl_p50 is not None
                        and critical_p50 is not None
                        and critical_p50 > 0
                        else None
                    ),
                    "speedup_vs_tp1": comparison.get("speedup"),
                    "latency_change_vs_tp1_pct": comparison.get("latency_change_pct"),
                    "classification_vs_tp1": comparison.get("classification"),
                    "e2e_latency_s": e2e.get("latency"),
                    "output_throughput_tok_s": e2e.get("output_throughput"),
                    "accept_length": e2e.get("acc_length"),
                    "error": row.get("error"),
                }
            )


def _print_summary(rows: Sequence[dict[str, Any]]) -> None:
    headers = (
        "TP",
        "Req BS",
        "Rows",
        "Draft p50 ms",
        "Draft p90 ms",
        "NCCL p50 ms",
        "vs TP1",
        "Verdict",
    )
    print("\n" + "  ".join(headers))
    print("  ".join("-" * len(header) for header in headers))
    for row in sorted(
        rows, key=lambda item: (item["request_batch_size"], item["tp_size"])
    ):
        draft = row.get("draft_forward") or {}
        comparison = row.get("vs_tp1") or {}

        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        speedup = comparison.get("speedup")
        print(
            "  ".join(
                (
                    str(row["tp_size"]),
                    str(row["request_batch_size"]),
                    str(row["forward_token_rows"]),
                    fmt(_conservative_metric(draft, "critical_path_ms", "p50")),
                    fmt(_conservative_metric(draft, "critical_path_ms", "p90")),
                    fmt(_conservative_metric(draft, "nccl_ms", "p50")),
                    "n/a" if speedup is None else f"{speedup:.3f}x",
                    comparison.get("classification", row.get("status", "n/a")),
                )
            )
        )


def _resolve_output_dir(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return (Path.cwd() / f"dflash_tp_latency_{stamp}").resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the native DFlash draft-forward latency for a TP x batch "
            "matrix and quantify communication overhead."
        )
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--draft-model-path", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument(
        "--tp-sizes", type=_positive_int, nargs="+", default=DEFAULT_TP_SIZES
    )
    parser.add_argument(
        "--batch-sizes", type=_positive_int, nargs="+", default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument("--block-size", type=_positive_int, default=16)
    parser.add_argument("--input-len", type=_positive_int, default=256)
    parser.add_argument("--output-len", type=_positive_int, default=512)
    parser.add_argument("--profile-start-step", type=int, default=5)
    parser.add_argument("--profile-steps", type=_positive_int, default=20)
    parser.add_argument("--attention-backend", default="flashinfer")
    parser.add_argument("--draft-attention-backend", default="flashinfer")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--page-size", type=_positive_int, default=1)
    parser.add_argument(
        "--base-port",
        type=_positive_int,
        default=31000,
        help="TP N uses 127.0.0.1:(base-port + N); occupied ports fail closed.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-level", default="warning")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir")
    parser.add_argument("--marker", default=DEFAULT_MARKER)
    parser.add_argument(
        "--classification-threshold-pct", type=_nonnegative_float, default=3.0
    )
    parser.add_argument(
        "--unavailable",
        choices=("skip", "error"),
        default="skip",
        help="What to do when a requested TP size exceeds visible GPUs.",
    )
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument(
        "--draft-sampler-scope",
        choices=("transformer-only", "production"),
        default="transformer-only",
        help=(
            "transformer-only forces the target lm_head/sampler outside the "
            "draft marker; production preserves DFlash's graph-folded sampler"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Reparse traces already present under --output-dir without launching GPUs.",
    )
    parser.add_argument(
        "--extra-server-args",
        nargs=argparse.REMAINDER,
        default=(),
        help="Additional one_batch_server/server flags. This option must be last.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.tp_sizes = tuple(dict.fromkeys(args.tp_sizes))
    args.batch_sizes = tuple(sorted(set(args.batch_sizes)))
    _validate_extra_server_args(args.extra_server_args)
    if args.analyze_only and args.dry_run:
        raise BenchmarkError("--analyze-only and --dry-run are mutually exclusive")
    if args.profile_start_step < 0:
        raise BenchmarkError("--profile-start-step must be non-negative")
    if not 0.0 < args.mem_fraction_static < 1.0:
        raise BenchmarkError("--mem-fraction-static must be between 0 and 1")
    invalid_ports = [
        args.base_port + tp_size
        for tp_size in args.tp_sizes
        if not 1 <= args.base_port + tp_size <= 65535
    ]
    if invalid_ports:
        raise BenchmarkError(f"invalid derived benchmark ports: {invalid_ports}")

    repo_root = Path(__file__).resolve().parents[2]
    if args.analyze_only:
        if not args.output_dir:
            raise BenchmarkError("--analyze-only requires --output-dir")
        output_dir = Path(args.output_dir).expanduser().resolve()
        metadata_path = output_dir / "metadata.json"
        if not metadata_path.is_file():
            raise BenchmarkError(f"missing experiment metadata: {metadata_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkError(f"cannot read {metadata_path}: {exc}") from exc
        # The recorded workload, not current CLI defaults, owns trace analysis.
        args.tp_sizes = tuple(int(value) for value in metadata["tp_sizes"])
        args.batch_sizes = tuple(int(value) for value in metadata["batch_sizes"])
        args.block_size = int(metadata["block_size"])
        args.marker = str(metadata.get("marker", DEFAULT_MARKER))
    else:
        guaranteed_steps = args.profile_start_step + args.profile_steps + 1
        minimum_output_len = args.block_size * guaranteed_steps
        if args.output_len < minimum_output_len:
            raise BenchmarkError(
                f"--output-len={args.output_len} may finish before the profiler "
                f"collects {args.profile_steps} samples after start step "
                f"{args.profile_start_step}; use at least {minimum_output_len} "
                f"for block size {args.block_size}"
            )
        hardware = _hardware_metadata(repo_root)
        visible_gpus = int(hardware["visible_gpu_count"])
        unavailable_tp_sizes = [
            tp_size for tp_size in args.tp_sizes if tp_size > visible_gpus
        ]
        if unavailable_tp_sizes and args.unavailable == "error" and not args.dry_run:
            raise BenchmarkError(
                f"TP sizes {unavailable_tp_sizes} exceed the {visible_gpus} visible GPUs"
            )
        if not args.dry_run:
            for tp_size in args.tp_sizes:
                if tp_size <= visible_gpus:
                    _ensure_port_available(args.base_port + tp_size)
        output_dir = _resolve_output_dir(args.output_dir)
        if output_dir.exists():
            if not output_dir.is_dir():
                raise BenchmarkError(f"--output-dir is not a directory: {output_dir}")
            if any(output_dir.iterdir()):
                raise BenchmarkError(
                    f"refusing to overwrite non-empty experiment directory {output_dir}"
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        transformer_only = args.draft_sampler_scope == "transformer-only"
        metadata = {
            "schema_version": 1,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_path": args.model_path,
            "draft_model_path": args.draft_model_path,
            "tp_sizes": args.tp_sizes,
            "batch_sizes": args.batch_sizes,
            "block_size": args.block_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "profile_start_step": args.profile_start_step,
            "profile_steps": args.profile_steps,
            "ports": {
                str(tp_size): args.base_port + tp_size for tp_size in args.tp_sizes
            },
            "cuda_graph_enabled": not args.disable_cuda_graph,
            "draft_sampler_scope": args.draft_sampler_scope,
            "extra_server_args": list(args.extra_server_args),
            "marker": args.marker,
            "draft_latency_definition": (
                "GPU work correlated with sglang.dflash.draft_model_forward; "
                + (
                    "sampler forced eager, so excludes target embedding, target "
                    "lm_head, and proposal sampling"
                    if transformer_only
                    else "production graph may fold target lm_head and proposal "
                    "sampling into this marker; block preparation and embedding "
                    "remain outside"
                )
            ),
            "batch_definition": (
                "request_batch_size; native DFlash draft forward processes "
                "request_batch_size * block_size token rows"
            ),
            "hardware": hardware,
        }
        _write_json(output_dir / "metadata.json", metadata)

    visible_gpus = int(metadata["hardware"]["visible_gpu_count"])
    previous_rows = {
        (int(row["tp_size"]), int(row["request_batch_size"])): row
        for row in _read_jsonl(output_dir / "summary.jsonl")
        if row.get("status") == "skipped"
    }
    rows: list[dict[str, Any]] = []
    for tp_size in args.tp_sizes:
        tp_dir = output_dir / f"tp{tp_size}"
        if args.analyze_only and not (tp_dir / "profiles").exists():
            preserved = [
                previous_rows[(tp_size, batch_size)]
                for batch_size in args.batch_sizes
                if (tp_size, batch_size) in previous_rows
            ]
            if len(preserved) == len(args.batch_sizes):
                rows.extend(preserved)
                continue
        if not args.analyze_only:
            tp_dir.mkdir(parents=True, exist_ok=True)
        unavailable = tp_size > visible_gpus
        if unavailable and not args.analyze_only and not args.dry_run:
            message = (
                f"TP={tp_size} requires {tp_size} GPUs, only {visible_gpus} visible"
            )
            if args.unavailable == "error":
                raise BenchmarkError(message)
            print(f"SKIP: {message}")
            rows.extend(
                {
                    "status": "skipped",
                    "error": message,
                    "tp_size": tp_size,
                    "request_batch_size": batch_size,
                    "block_size": args.block_size,
                    "forward_token_rows": batch_size * args.block_size,
                    "marker": args.marker,
                }
                for batch_size in args.batch_sizes
            )
            continue

        result_path = tp_dir / "one_batch_results.jsonl"
        profile_dir = tp_dir / "profiles"
        if not args.analyze_only:
            command = _case_command(
                args,
                tp_size=tp_size,
                result_path=result_path,
                profile_dir=profile_dir,
            )
            env, env_overrides = _child_environment(args)
            _write_json(
                tp_dir / "command.json",
                {"argv": command, "env_overrides": env_overrides},
            )
            print(f"\nTP={tp_size}: {shlex.join(command)}")
            if args.dry_run:
                continue
            _stream_command(
                command,
                cwd=repo_root,
                log_path=tp_dir / "run.log",
                env=env,
            )
        else:
            print(f"\nTP={tp_size}: analyzing existing traces in {profile_dir}")
        rows.extend(
            analyze_tp_directory(
                tp_dir,
                tp_size=tp_size,
                batch_sizes=args.batch_sizes,
                block_size=args.block_size,
                marker=args.marker,
            )
        )

    if args.dry_run:
        print(f"Dry run complete. Commands and metadata: {output_dir}")
        return 0

    _add_comparisons(rows, args.classification_threshold_pct)
    _write_jsonl(output_dir / "summary.jsonl", rows)
    _write_csv(output_dir / "summary.csv", rows)
    _print_summary(rows)
    print(f"\nResults: {output_dir}")
    return 1 if any(row.get("status") == "error" for row in rows) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
