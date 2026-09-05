#!/usr/bin/env python3
"""Collect Qwen3.5-9B PP2 ratio profiles and analyze several batch sizes."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from adaptive_pp_tuner import REPO_ROOT, SCRIPT_DIR, build_parser, build_profile_command
from torch_trace_profile import TraceProfileError, summarize_trace_dir

RATIOS = ("0.25", "0.5", "0.75", "1.0")


def profile_directory(root: Path, batch_size: int, ratio: str) -> Path:
    suffix = "" if batch_size == 32 else f"_bs{batch_size}"
    return root / f"qwen35_9b{suffix}_r{ratio}"


def default_results_dir(results_parent: Path) -> Path:
    """Create a fresh consolidated run directory."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return results_parent / f"Qwen_Qwen3.5-9B_profile_{stamp}"


def profile_arguments(
    directory: Path, batch_size: int, ratio: str, profile_steps: int
) -> list[str]:
    return [
        "profile", "--output-dir", str(directory),
        "--model-path", "Qwen/Qwen3.5-9B",
        "--draft-model-path", "z-lab/Qwen3.5-9B-DFlash",
        "--tp-size", "1", "--pp-size", "2", "--nnodes", "1",
        "--baseline-partition", "16,16",
        "--batch-size", str(batch_size),
        "--max-running-requests", str(batch_size),
        "--execution-bucket", str(batch_size // 2),
        "--input-tokens", "4000", "--output-tokens", str(max(256, profile_steps)),
        "--profile-steps", str(profile_steps), "--block-size", "16",
        "--mem-fraction-static", "0.7", "--page-size", "1",
        "--mamba-ssm-dtype", "float32", "--mamba-full-memory-ratio", "0.9",
        "--server-args", "--dtype", "bfloat16",
        "--attention-backend", "triton",
        "--speculative-draft-attention-backend", "flashinfer",
        "--disable-radix-cache", "--linear-attn-backend", "triton",
        "--enable-linear-replayssm-spec", "--disable-overlap-schedule",
        "--pp-max-micro-batch-size", str(batch_size // 2),
        "--random-seed", "1", "--speculative-dflash-dcut", ratio,
        "--dist-timeout", "180",
    ]


def run(args: argparse.Namespace) -> None:
    root = args.results_dir.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"results directory must be absent or empty: {root}")
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
    tuner = SCRIPT_DIR / "adaptive_pp_tuner.py"
    for batch_size in args.batch_sizes:
        profile_steps = args.profile_steps or 128
        for ratio in RATIOS:
            directory = profile_directory(root, batch_size, ratio)
            command_args = profile_arguments(directory, batch_size, ratio, profile_steps)
            # Validate the same argument path used by the actual profiler.
            build_profile_command(build_parser().parse_args(command_args), directory)
            command = [sys.executable, str(tuner), *command_args]
            if not args.dry_run:
                directory.parent.mkdir(parents=True, exist_ok=True)
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=REPO_ROOT, check=True)

    costs_path = root / "qwen35_9b_dcut_costs_multi.json"
    if not args.dry_run:
        costs = {}
        for batch_size in args.batch_sizes:
            bucket_costs = {}
            for ratio in RATIOS:
                directory = profile_directory(root, batch_size, ratio)
                summary = summarize_trace_dir(directory, pp_size=2, tp_size=1)
                bucket_costs[ratio] = summary["target_ms"]
                print(f"[cost] bs={batch_size} ratio={ratio}: {summary}", flush=True)
            costs[str(batch_size // 2)] = bucket_costs
        # Single-bucket analysis retains its existing ratio -> cost input format.
        payload = next(iter(costs.values())) if len(costs) == 1 else costs
        costs_path.write_text(json.dumps(payload, indent=2) + "\n")

    command = [sys.executable, str(tuner), "analyze"]
    for batch_size in args.batch_sizes:
        command.extend(["--profile-dir", str(profile_directory(root, batch_size, "1.0"))])
    command.extend([
        "--dcut-profile", str(costs_path), "--all-boundaries",
        "--min-layers", "8", "--k-best", "30",
        "--output-dir", str(root / "qwen35_9b_analysis_multi"),
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
    parser.add_argument(
        "--profile-steps",
        type=int,
        help="profile steps; defaults to 128",
    )
    args = parser.parse_args()
    if args.results_dir is None:
        args.results_dir = default_results_dir(SCRIPT_DIR / "results")
    if len(set(args.batch_sizes)) != len(args.batch_sizes) or any(
        value <= 0 or value % 2 for value in args.batch_sizes
    ):
        parser.error("batch sizes must be distinct positive even integers for PP2")
    try:
        run(args)
    except (OSError, ValueError, TraceProfileError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"[error] {exc}\n")


if __name__ == "__main__":
    main()
