#!/usr/bin/env python3
"""Collect multi-batch DFlash PP profiles and analyze them jointly."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from adaptive_pp_tuner import (
    REPO_ROOT,
    SCRIPT_DIR,
    build_parser,
    build_profile_command,
    uniform_partition,
)
from torch_trace_profile import TraceProfileError, summarize_trace_dir

RATIOS = ("0.25", "0.5", "0.75", "1.0")


def profile_directory(
    root: Path, batch_size: int, ratio: str, model_tag: str = "qwen35_9b"
) -> Path:
    suffix = "" if batch_size == 32 else f"_bs{batch_size}"
    return root / f"{model_tag}{suffix}_r{ratio}"


def default_results_dir(
    results_parent: Path, model_path: str = "Qwen/Qwen3.5-9B"
) -> Path:
    """Create a fresh consolidated run directory."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_tag = model_path.replace("/", "_")
    return results_parent / f"{model_tag}_profile_{stamp}"


def profile_arguments(
    args: argparse.Namespace,
    directory: Path,
    batch_size: int,
    ratio: str,
    profile_steps: int,
) -> list[str]:
    baseline_partition = args.baseline_partition
    if baseline_partition is None:
        baseline_partition = ",".join(
            str(value) for value in uniform_partition(args.model_path, args.pp_size)
        )
    execution_bucket = batch_size // args.pp_size
    command = [
        "profile", "--output-dir", str(directory),
        "--model-path", args.model_path,
        "--draft-model-path", args.draft_model_path,
        "--tp-size", str(args.tp_size), "--pp-size", str(args.pp_size),
        "--nnodes", str(args.nnodes),
        "--baseline-partition", baseline_partition,
        "--batch-size", str(batch_size),
        "--max-running-requests", str(batch_size),
        "--execution-bucket", str(execution_bucket),
        "--input-tokens", str(args.input_tokens),
        "--output-tokens", str(max(args.output_tokens, profile_steps)),
        "--profile-steps", str(profile_steps),
        "--block-size", str(args.block_size),
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--page-size", str(args.page_size),
        "--mamba-ssm-dtype", args.mamba_ssm_dtype,
        "--mamba-full-memory-ratio", str(args.mamba_full_memory_ratio),
        "--server-args", "--dtype", args.dtype,
        "--attention-backend", args.attention_backend,
        "--speculative-draft-attention-backend", args.draft_attention_backend,
        "--disable-radix-cache", "--disable-overlap-schedule",
        "--pp-max-micro-batch-size", str(execution_bucket),
        "--random-seed", "1", "--speculative-dflash-dcut", ratio,
        "--dist-timeout", "180",
    ]
    if args.enable_replay_ssm:
        command.extend(
            ["--linear-attn-backend", "triton", "--enable-linear-replayssm-spec"]
        )
    if args.offline:
        command.insert(1, "--offline")
    return command


def run(args: argparse.Namespace) -> None:
    root = args.results_dir.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"results directory must be absent or empty: {root}")
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
    tuner = SCRIPT_DIR / "adaptive_pp_tuner.py"
    model_tag = args.model_path.replace("/", "_")
    for batch_size in args.batch_sizes:
        profile_steps = args.profile_steps or 128
        for ratio in RATIOS:
            directory = profile_directory(root, batch_size, ratio, model_tag)
            command_args = profile_arguments(
                args, directory, batch_size, ratio, profile_steps
            )
            # Validate the same argument path used by the actual profiler.
            build_profile_command(build_parser().parse_args(command_args), directory)
            command = [sys.executable, str(tuner), *command_args]
            if not args.dry_run:
                directory.parent.mkdir(parents=True, exist_ok=True)
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=REPO_ROOT, check=True)

    costs_path = root / f"{model_tag}_dcut_costs_multi.json"
    if not args.dry_run:
        costs = {}
        for batch_size in args.batch_sizes:
            bucket_costs = {}
            for ratio in RATIOS:
                directory = profile_directory(root, batch_size, ratio, model_tag)
                summary = summarize_trace_dir(
                    directory, pp_size=args.pp_size, tp_size=args.tp_size
                )
                bucket_costs[ratio] = summary["target_ms"]
                print(f"[cost] bs={batch_size} ratio={ratio}: {summary}", flush=True)
            costs[str(batch_size // args.pp_size)] = bucket_costs
        # Single-bucket analysis retains its existing ratio -> cost input format.
        payload = next(iter(costs.values())) if len(costs) == 1 else costs
        costs_path.write_text(json.dumps(payload, indent=2) + "\n")

    command = [sys.executable, str(tuner), "analyze"]
    for batch_size in args.batch_sizes:
        command.extend(
            [
                "--profile-dir",
                str(profile_directory(root, batch_size, "1.0", model_tag)),
            ]
        )
    command.extend([
        "--dcut-profile", str(costs_path), "--all-boundaries",
        "--min-layers", str(args.min_layers), "--k-best", str(args.k_best),
        # Report the optimal partition per (bucket, ratio) cell and recommend
        # the partition that wins the most cells; also emits the runtime D-Cut
        # cost table for the selected partition.
        "--partition-selection", "per-cell",
        "--output-dir", str(root / f"{model_tag}_analysis_multi"),
    ])
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="fresh consolidated output directory; default: timestamped results directory",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-path", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--draft-model-path", default="z-lab/Qwen3.5-9B-DFlash")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--baseline-partition")
    parser.add_argument("--input-tokens", type=int, default=4000)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--mamba-ssm-dtype", default="float32")
    parser.add_argument("--mamba-full-memory-ratio", type=float, default=0.9)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--disable-replay-ssm",
        dest="enable_replay_ssm",
        action="store_false",
        default=True,
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--draft-attention-backend", default="flashinfer")
    parser.add_argument("--min-layers", type=int, default=8)
    parser.add_argument("--k-best", type=int, default=30)
    parser.add_argument(
        "--profile-steps",
        type=int,
        help="profile steps; defaults to 128",
    )
    args = parser.parse_args()
    if args.results_dir is None:
        args.results_dir = default_results_dir(SCRIPT_DIR / "results", args.model_path)
    if args.pp_size <= 0 or args.tp_size <= 0 or args.nnodes <= 0:
        parser.error("tp-size, pp-size, and nnodes must be positive")
    if len(set(args.batch_sizes)) != len(args.batch_sizes) or any(
        value <= 0 or value % args.pp_size for value in args.batch_sizes
    ):
        parser.error("batch sizes must be distinct positive multiples of pp-size")
    if args.baseline_partition is not None:
        try:
            partition = tuple(
                int(item.strip()) for item in args.baseline_partition.split(",")
            )
        except ValueError:
            parser.error("baseline-partition must contain comma-separated integers")
        if len(partition) != args.pp_size or any(value <= 0 for value in partition):
            parser.error(
                "baseline-partition must contain one positive value per PP rank"
            )
    try:
        run(args)
    except (OSError, ValueError, TraceProfileError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"[error] {exc}\n")


if __name__ == "__main__":
    main()
