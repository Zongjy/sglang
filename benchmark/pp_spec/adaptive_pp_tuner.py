#!/usr/bin/env python3
"""Profile and analyze DFlash PP layer boundaries with SGLang RayEngine.

The script intentionally does not launch remote nodes.  A Ray cluster owns
placement and process lifecycle; SGLang's offline benchmark owns workload and
Torch-profiler capture.  This file only builds a reproducible profile command
and turns the resulting per-rank Chrome traces into a small PP boundary model.

Typical workflow::

    python benchmark/pp_spec/adaptive_pp_tuner.py profile \
      --output-dir /shared/pp-profile-bs32 --nnodes 2 --tp-size 1 --pp-size 4 \
      --current-partition 16,16,16,16 --batch-size 32

    python benchmark/pp_spec/adaptive_pp_tuner.py analyze \
      --profile-dir /shared/pp-profile-bs32

For multi-node profiling, ``--output-dir`` must be visible at the same path on
every Ray node so that the standard SGLang profiler traces can be analyzed
together after the RayEngine shuts down.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_DRAFT_MODEL = "z-lab/Qwen3.5-9B-DFlash"
PROFILE_CONFIG = "profile.json"


class TuningError(RuntimeError):
    pass


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def parse_partition(value: str, pp_size: int) -> tuple[int, ...]:
    try:
        partition = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise TuningError(f"invalid partition {value!r}") from exc
    if len(partition) != pp_size or any(count <= 0 for count in partition):
        raise TuningError(
            f"partition must contain exactly {pp_size} positive layer counts"
        )
    return partition


def parse_int_list(value: str | None, option: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise TuningError(f"{option} must contain comma-separated integers") from exc
    if not result or any(item <= 0 for item in result):
        raise TuningError(f"{option} must contain positive integers")
    return result


def parse_float_list(value: str | None, option: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise TuningError(f"{option} must contain comma-separated numbers") from exc
    if not result or any(item < 0.0 for item in result):
        raise TuningError(f"{option} must contain non-negative numbers")
    return result


def _contains_flag(arguments: Sequence[str], name: str) -> bool:
    return any(item == name or item.startswith(name + "=") for item in arguments)


def _execution_bucket(args: argparse.Namespace) -> int:
    max_running = args.max_running_requests or args.batch_size
    if args.execution_bucket is not None:
        return args.execution_bucket
    if max_running % args.pp_size:
        raise TuningError(
            "--max-running-requests must be divisible by --pp-size when "
            "--execution-bucket is omitted"
        )
    return max_running // args.pp_size


def build_profile_command(args: argparse.Namespace, profile_dir: Path) -> list[str]:
    if args.tp_size <= 0 or args.pp_size <= 0 or args.nnodes <= 0:
        raise TuningError("--tp-size, --pp-size, and --nnodes must be positive")
    partition = parse_partition(args.current_partition, args.pp_size)
    world_size = args.tp_size * args.pp_size
    if world_size % args.nnodes != 0:
        raise TuningError(
            f"TP{args.tp_size} x PP{args.pp_size} world size {world_size} is not "
            f"divisible by nnodes={args.nnodes}"
        )
    if args.batch_size <= 0 or args.profile_steps <= 1:
        raise TuningError(
            "--batch-size must be positive and --profile-steps must exceed 1"
        )
    if args.output_tokens < args.profile_steps:
        raise TuningError(
            "--output-tokens must be at least --profile-steps so decode outlasts capture"
        )

    max_running = args.max_running_requests or args.batch_size
    if max_running <= 0:
        raise TuningError("--max-running-requests must be positive")
    execution_bucket = _execution_bucket(args)
    if execution_bucket <= 0:
        raise TuningError("--execution-bucket must be positive")
    command = [
        sys.executable,
        "-m",
        "sglang.benchmark.offline_throughput",
        "--backend",
        "engine",
        "--use-ray",
        "--nnodes",
        str(args.nnodes),
        "--model-path",
        args.model_path,
        "--tp-size",
        str(args.tp_size),
        "--pp-size",
        str(args.pp_size),
        "--pp-layer-partition",
        ",".join(map(str, partition)),
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        args.draft_model_path,
        "--speculative-dflash-block-size",
        str(args.block_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--page-size",
        str(args.page_size),
        "--mamba-ssm-dtype",
        args.mamba_ssm_dtype,
        "--mamba-full-memory-ratio",
        str(args.mamba_full_memory_ratio),
        "--max-running-requests",
        str(max_running),
        "--cuda-graph-max-bs-decode",
        str(execution_bucket),
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.input_tokens),
        "--random-output-len",
        str(args.output_tokens),
        # The random sampler interprets this as the lower/full length ratio.
        # 1.0 keeps every profiled request at the declared lengths.
        "--random-range-ratio",
        "1.0",
        "--num-prompts",
        str(args.batch_size),
        "--profile",
        "--profile-steps",
        str(args.profile_steps),
        "--profile-activities",
        "CPU",
        "GPU",
        "--result-filename",
        str(profile_dir / "benchmark.jsonl"),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    extra = list(args.server_args or ())
    controlled = (
        "--model-path",
        "--model",
        "--tp",
        "--tp-size",
        "--tensor-parallel-size",
        "--pp",
        "--pp-size",
        "--pipeline-parallel-size",
        "--pp-layer-partition",
        "--nnodes",
        "--use-ray",
        "--cuda-graph-max-bs-decode",
    )
    if any(_contains_flag(extra, name) for name in controlled):
        raise TuningError(
            "model and Ray topology options are managed by profile arguments; "
            "do not repeat them in --server-args"
        )
    if args.pp_size > 1 and not _contains_flag(extra, "--disable-overlap-schedule"):
        command.append("--disable-overlap-schedule")
    command.extend(extra)
    return command


def run_profile(args: argparse.Namespace) -> Path:
    if importlib.util.find_spec("ray") is None:
        raise TuningError(
            "Ray is required; install it with `pip install 'sglang[ray]'`"
        )
    profile_dir = args.output_dir.resolve()
    if profile_dir.exists() and any(profile_dir.iterdir()):
        raise TuningError(
            f"profile output directory must be empty or absent: {profile_dir}"
        )
    profile_dir.mkdir(parents=True, exist_ok=True)
    command = build_profile_command(args, profile_dir)
    partition = parse_partition(args.current_partition, args.pp_size)
    execution_bucket = _execution_bucket(args)
    if execution_bucket <= 0:
        raise TuningError("--execution-bucket must be positive")
    env = os.environ.copy()
    env["SGLANG_TORCH_PROFILER_DIR"] = str(profile_dir)
    env["SGLANG_PROFILE_WITH_STACK"] = "false"
    env["SGLANG_PROFILE_RECORD_SHAPES"] = "false"
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    print(f"[profile] {shlex.join(command)}", flush=True)
    log_path = profile_dir / "profile.log"
    with log_path.open("w") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise TuningError(
            f"SGLang offline profile failed with code {completed.returncode}; "
            f"see {log_path}"
        )

    traces = sorted(profile_dir.rglob("*.trace.json*"))
    if not traces:
        raise TuningError(
            "profile completed but no trace files are visible. For multi-node "
            "runs, use a profile directory shared at the same path by every Ray node."
        )
    write_json(
        profile_dir / PROFILE_CONFIG,
        {
            "model_path": args.model_path,
            "tp_size": args.tp_size,
            "pp_size": args.pp_size,
            "current_partition": list(partition),
            "execution_bucket": execution_bucket,
        },
    )
    print(f"[profile] traces written under {profile_dir}", flush=True)
    return profile_dir


def _load_profile(profile_dir: Path) -> dict[str, Any]:
    path = profile_dir / PROFILE_CONFIG
    try:
        profile = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TuningError(f"cannot read {path}: {exc}") from exc
    return profile


def _typed_layer_costs(
    layout: Any,
    partition: Sequence[int],
    target_ms: Sequence[float],
) -> tuple[float, float, float]:
    fit_ranks = (
        range(1, len(partition) - 1)
        if len(partition) > 2
        else range(len(partition))
    )
    per_layer = [
        target_ms[rank] / partition[rank]
        for rank in fit_ranks
        if target_ms[rank] > 0.0 and partition[rank] > 0
    ]
    conservative_cost = max(per_layer, default=0.0)
    if layout is None or not per_layer:
        return conservative_cost, conservative_cost, conservative_cost

    rows: list[tuple[int, int, float]] = []
    start = 0
    for rank, (count, observed) in enumerate(zip(partition, target_ms)):
        gdn, full = layout.count_range(start, start + count)
        start += count
        if rank in fit_ranks and observed > 0.0:
            rows.append((gdn, full, observed))
    gg = sum(gdn * gdn for gdn, _full, _value in rows)
    ff = sum(full * full for _gdn, full, _value in rows)
    gf = sum(gdn * full for gdn, full, _value in rows)
    gy = sum(gdn * value for gdn, _full, value in rows)
    fy = sum(full * value for _gdn, full, value in rows)
    determinant = gg * ff - gf * gf
    if determinant <= 1e-9:
        return conservative_cost, conservative_cost, conservative_cost
    gdn_cost = (gy * ff - fy * gf) / determinant
    full_cost = (fy * gg - gy * gf) / determinant
    if gdn_cost <= 0.0 or full_cost <= 0.0:
        return conservative_cost, conservative_cost, conservative_cost
    weighted = sum(gdn * gdn_cost + full * full_cost for gdn, full, _ in rows) / sum(
        gdn + full for gdn, full, _ in rows
    )
    return weighted, gdn_cost, full_cost


def _intrinsic_service_observation(summary: Mapping[str, Any]) -> list[float]:
    intrinsic = [float(value) for value in summary["intrinsic_service_ms"]]
    if not intrinsic:
        raise TuningError("trace summary has no intrinsic stage times")
    return intrinsic


def _target_observation(summary: Mapping[str, Any]) -> list[float]:
    target = [float(value) for value in summary["target_ms"]]
    if not target or any(value <= 0.0 for value in target):
        raise TuningError("trace summary has no positive target stage times")
    return target


def _bucket_profile(
    summary: Mapping[str, Any],
    partition: Sequence[int],
    layout: Any,
    typed_costs: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    intrinsic_service = _intrinsic_service_observation(summary)
    target = _target_observation(summary)
    if len(intrinsic_service) != len(partition) or len(target) != len(partition):
        raise TuningError("trace stage count does not match the baseline partition")
    layer, gdn, full = typed_costs or _typed_layer_costs(
        layout, partition, target
    )
    fixed: list[float] = []
    start = 0
    for rank, (count, observed) in enumerate(zip(partition, intrinsic_service)):
        if layout is not None:
            gdn_count, full_count = layout.count_range(start, start + count)
            predicted_layers = gdn_count * gdn + full_count * full
        else:
            predicted_layers = count * layer
        start += count
        fixed.append(max(observed - predicted_layers, 0.0))

    raw = {
        "fixed_ms": fixed,
        "layer_cost_ms": layer,
        "gdn_cost_ms": gdn,
        "full_cost_ms": full,
    }
    return raw


def run_analysis(args: argparse.Namespace) -> Path:
    sys.path.insert(0, str(SCRIPT_DIR))
    import partition_optimizer
    import stage_model
    import torch_trace_profile
    from model_layout import LayerLayout

    profile_dir = args.profile_dir.resolve()
    if args.trim_samples < 0:
        raise TuningError("--trim-samples cannot be negative")
    profile = _load_profile(profile_dir)
    pp_size = int(profile["pp_size"])
    tp_size = int(profile["tp_size"])
    try:
        partition = tuple(int(value) for value in profile["current_partition"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TuningError("baseline profile has an invalid partition") from exc
    if len(partition) != pp_size or any(value <= 0 for value in partition):
        raise TuningError("baseline profile partition does not match pp_size")
    model_path = str(profile["model_path"])
    if len(set(partition[:-1])) > 1:
        raise TuningError(
            "the thin optimizer currently searches (l,...,l,residual); the first "
            "PP stages in the baseline partition must have the same layer count"
        )

    try:
        layout = LayerLayout.from_model_path(model_path, local_files_only=True)
    except Exception as exc:
        raise TuningError(f"cannot load model layer layout: {exc}") from exc

    bucket = int(profile["execution_bucket"])
    try:
        summary = torch_trace_profile.summarize_trace_dir(
            profile_dir,
            pp_size=pp_size,
            tp_size=tp_size,
            trim_samples=args.trim_samples,
        )
    except torch_trace_profile.TraceProfileError as exc:
        raise TuningError(str(exc)) from exc

    typed_costs = _typed_layer_costs(
        layout,
        partition,
        _target_observation(summary),
    )
    raw = _bucket_profile(
        summary,
        partition,
        layout,
        typed_costs=typed_costs,
    )
    profiles = {bucket: raw}

    try:
        model = stage_model.StageCostModel.from_bucket_profiles(
            profiles,
            num_layers=sum(partition),
            pp_size=pp_size,
            current_partition=partition,
            layout=layout,
        )
    except stage_model.StageModelError as exc:
        raise TuningError(str(exc)) from exc
    max_layers = parse_int_list(args.max_layers_per_rank, "--max-layers-per-rank")
    if args.min_layers <= 0:
        raise TuningError("--min-layers must be positive")
    if max_layers is not None and len(max_layers) != pp_size:
        raise TuningError("--max-layers-per-rank needs one value per PP rank")
    stage_comm_ms = parse_float_list(args.stage_comm_ms, "--stage-comm-ms")
    if stage_comm_ms is not None and len(stage_comm_ms) not in (
        pp_size - 1,
        pp_size,
    ):
        raise TuningError("--stage-comm-ms needs PP or PP-1 values")
    prefix_range = None
    if args.boundary_radius is not None:
        if args.boundary_radius < 0:
            raise TuningError("--boundary-radius cannot be negative")
        center = partition[0]
        prefix_range = (
            max(1, center - args.boundary_radius),
            center + args.boundary_radius,
        )
    try:
        result = partition_optimizer.optimize(
            model,
            target_bs=bucket,
            min_layers=args.min_layers,
            max_layers=max_layers,
            k_best=args.k_best,
            layout=layout,
            prefix_l_range=prefix_range,
            stage_comm_ms=stage_comm_ms,
        )
    except (partition_optimizer.OptimizerError, stage_model.StageModelError) as exc:
        raise TuningError(str(exc)) from exc
    output_dir = (args.output_dir or (profile_dir / "analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "analysis.json", result.to_dict())
    report = result.to_report()
    (output_dir / "analysis.txt").write_text(report + "\n")
    selected = result.selected.partition
    (output_dir / "recommended.args").write_text(
        "--pp-layer-partition " + ",".join(map(str, selected)) + "\n"
    )
    print(report, flush=True)
    print(f"[analyze] artifacts written to {output_dir}", flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="capture a RayEngine profile")
    profile.add_argument("--output-dir", type=Path, required=True)
    profile.add_argument("--model-path", default=DEFAULT_MODEL)
    profile.add_argument("--draft-model-path", default=DEFAULT_DRAFT_MODEL)
    profile.add_argument("--tp-size", type=int, default=1)
    profile.add_argument("--pp-size", type=int, required=True)
    profile.add_argument("--nnodes", type=int, default=1)
    profile.add_argument("--current-partition", required=True)
    profile.add_argument("--batch-size", type=int, default=32)
    profile.add_argument(
        "--execution-bucket",
        type=int,
        help=(
            "decode execution bucket represented by this run; default: "
            "ceil(batch-size / pp-size)"
        ),
    )
    profile.add_argument("--input-tokens", type=int, default=256)
    profile.add_argument("--output-tokens", type=int, default=128)
    profile.add_argument("--profile-steps", type=int, default=32)
    profile.add_argument("--max-running-requests", type=int)
    profile.add_argument("--block-size", type=int, default=16)
    profile.add_argument("--page-size", type=int, default=1)
    profile.add_argument("--mem-fraction-static", type=float, default=0.75)
    profile.add_argument("--mamba-ssm-dtype", default="float32")
    profile.add_argument("--mamba-full-memory-ratio", type=float, default=0.9)
    profile.add_argument("--offline", action="store_true")
    profile.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        default=True,
    )
    profile.add_argument(
        "--server-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="additional ServerArgs passed to offline_throughput; must be last",
    )

    analyze = subparsers.add_parser("analyze", help="analyze existing traces")
    analyze.add_argument("--profile-dir", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path)
    analyze.add_argument(
        "--stage-comm-ms",
        help=(
            "optional explicit communication floor per PP rank (P values) or "
            "per boundary (P-1 values), added after trace PP Send/Recv is excluded"
        ),
    )
    analyze.add_argument("--trim-samples", type=int, default=1)
    analyze.add_argument("--boundary-radius", type=int, default=8)
    analyze.add_argument(
        "--min-layers",
        type=int,
        default=1,
        help="minimum target layers on every PP rank",
    )
    analyze.add_argument("--max-layers-per-rank")
    analyze.add_argument("--k-best", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "profile":
            run_profile(args)
        else:
            run_analysis(args)
    except (OSError, subprocess.SubprocessError, TuningError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
