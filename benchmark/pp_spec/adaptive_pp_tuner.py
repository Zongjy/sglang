#!/usr/bin/env python3
"""Profile and analyze DFlash PP layer boundaries with SGLang Engine/RayEngine.

For multi-node runs, a Ray cluster owns placement and process lifecycle.  For a
single-node run, the regular Engine launches PP workers directly.  SGLang's
offline benchmark owns workload and Torch-profiler capture; this file builds a
reproducible profile command and analyzes the resulting per-rank traces.

Typical workflow::

    python benchmark/pp_spec/adaptive_pp_tuner.py profile \
      --output-dir /shared/pp-profile-bs32 --nnodes 2 --tp-size 1 --pp-size 4 \
      --batch-size 32

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

try:
    from benchmark.pp_spec.model_layout import LayerLayout, LayoutError
except ImportError:  # pragma: no cover - flat script execution
    from model_layout import LayerLayout, LayoutError


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


def uniform_partition(model_path: str, pp_size: int) -> tuple[int, ...]:
    """Total model layers split evenly across PP stages (remainder up front)."""

    def _layout() -> LayerLayout:
        try:
            return LayerLayout.from_model_path(model_path)  # local cache first
        except LayoutError:
            from huggingface_hub import hf_hub_download

            config_path = Path(hf_hub_download(model_path, "config.json"))
            return LayerLayout.from_config(json.loads(config_path.read_text()))

    layout = _layout()
    base, rem = divmod(layout.num_layers, pp_size)
    return tuple(base + (rank < rem) for rank in range(pp_size))


def resolve_partition(args: argparse.Namespace) -> tuple[int, ...]:
    if args.baseline_partition is None:
        try:
            return uniform_partition(args.model_path, args.pp_size)
        except Exception as exc:
            raise TuningError(
                f"cannot derive uniform partition for {args.model_path!r}; "
                "pass --baseline-partition explicitly"
            ) from exc
    return parse_partition(args.baseline_partition, args.pp_size)


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
    partition = resolve_partition(args)
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
    # Ray is required for multi-node placement.  On a single node the regular
    # Engine launches the PP workers directly and avoids an extra actor layer.
    if args.nnodes > 1:
        command.append("--use-ray")
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
    if args.nnodes > 1 and importlib.util.find_spec("ray") is None:
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
    partition = resolve_partition(args)
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
            "baseline_partition": list(partition),
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


def _load_dcut_profiles(path: Path, pp_size: int) -> dict[float, Any]:
    """Read a ratio -> cost map used by the joint PP/D-Cut optimizer."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TuningError(f"cannot read D-Cut profile {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise TuningError("D-Cut profile must be a JSON object mapping ratio to cost")
    profiles: dict[float, Any] = {}
    for ratio, value in raw.items():
        try:
            key = float(ratio)
        except (TypeError, ValueError) as exc:
            raise TuningError(f"invalid D-Cut ratio {ratio!r}") from exc
        if isinstance(value, list) and len(value) != pp_size:
            raise TuningError(
                f"D-Cut stage cost for ratio {key:g} needs {pp_size} values"
            )
        profiles[key] = value
    return profiles


def _load_dcut_profiles_by_bucket(
    path: Path, buckets: Sequence[int], pp_size: int
) -> dict[int, dict[float, Any]]:
    """Load ratio costs either as one shared map or as bucket-keyed maps."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TuningError(f"cannot read D-Cut profile {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise TuningError("D-Cut profile must be a JSON object")
    nested = all(isinstance(value, Mapping) for value in raw.values())
    if nested:
        result: dict[int, dict[float, Any]] = {}
        for bucket in buckets:
            value = raw.get(str(bucket), raw.get(bucket))
            if value is None:
                raise TuningError(f"D-Cut profile is missing bucket {bucket}")
            result[bucket] = _parse_dcut_profile_mapping(value, pp_size, bucket)
        return result
    shared = _parse_dcut_profile_mapping(raw, pp_size, None)
    return {int(bucket): shared for bucket in buckets}


def _parse_dcut_profile_mapping(
    raw: Mapping[Any, Any], pp_size: int, bucket: int | None
) -> dict[float, Any]:
    profiles: dict[float, Any] = {}
    for ratio, value in raw.items():
        try:
            key = float(ratio)
        except (TypeError, ValueError) as exc:
            label = f" for bucket {bucket}" if bucket is not None else ""
            raise TuningError(f"invalid D-Cut ratio {ratio!r}{label}") from exc
        if isinstance(value, list) and len(value) != pp_size:
            raise TuningError(
                f"D-Cut stage cost for ratio {key:g} needs {pp_size} values"
            )
        profiles[key] = value
    return profiles


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


def _run_multi_bucket_analysis(args: argparse.Namespace, profile_dirs: Sequence[Path]) -> Path:
    """Analyze several execution buckets and select one robust partition."""
    sys.path.insert(0, str(SCRIPT_DIR))
    import partition_optimizer
    import stage_model
    import torch_trace_profile
    from model_layout import LayerLayout

    if args.dcut_profile is None:
        raise TuningError("--dcut-profile is required when --profile-dir is repeated")
    first_profile = _load_profile(profile_dirs[0])
    pp_size = int(first_profile["pp_size"])
    tp_size = int(first_profile["tp_size"])
    baseline = tuple(int(value) for value in first_profile["baseline_partition"])
    model_path = str(first_profile["model_path"])
    try:
        layout = LayerLayout.from_model_path(model_path, local_files_only=True)
    except Exception as exc:
        raise TuningError(f"cannot load model layer layout: {exc}") from exc

    bucket_profiles: dict[int, dict[str, Any]] = {}
    for profile_dir in profile_dirs:
        profile = _load_profile(profile_dir)
        if int(profile["pp_size"]) != pp_size or int(profile["tp_size"]) != tp_size:
            raise TuningError("all profiles must use the same PP and TP sizes")
        partition = tuple(int(value) for value in profile["baseline_partition"])
        if partition != baseline:
            raise TuningError("all profiles must use the same baseline partition")
        bucket = int(profile["execution_bucket"])
        if bucket in bucket_profiles:
            raise TuningError(f"duplicate execution bucket {bucket}")
        try:
            summary = torch_trace_profile.summarize_trace_dir(
                profile_dir,
                pp_size=pp_size,
                tp_size=tp_size,
                trim_samples=args.trim_samples,
            )
        except torch_trace_profile.TraceProfileError as exc:
            raise TuningError(str(exc)) from exc
        bucket_profiles[bucket] = _bucket_profile(
            summary, partition, layout, typed_costs=_typed_layer_costs(
                layout, partition, _target_observation(summary)
            )
        )

    try:
        model = stage_model.StageCostModel.from_bucket_profiles(
            bucket_profiles,
            num_layers=sum(baseline),
            pp_size=pp_size,
            baseline_partition=baseline,
            layout=layout,
        )
    except stage_model.StageModelError as exc:
        raise TuningError(str(exc)) from exc
    max_layers = parse_int_list(args.max_layers_per_rank, "--max-layers-per-rank")
    stage_comm_ms = parse_float_list(args.stage_comm_ms, "--stage-comm-ms")
    if stage_comm_ms is not None and len(stage_comm_ms) not in (pp_size - 1, pp_size):
        raise TuningError("--stage-comm-ms needs PP or PP-1 values")
    prefix_range = None
    if args.boundary_radius is not None and not args.all_boundaries:
        prefix_range = (
            max(1, baseline[0] - args.boundary_radius),
            baseline[0] + args.boundary_radius,
        )
    dcut = _load_dcut_profiles_by_bucket(
        args.dcut_profile, tuple(bucket_profiles), pp_size
    )
    try:
        result = partition_optimizer.optimize_partition_across_buckets(
            model,
            dcut,
            min_layers=args.min_layers,
            max_layers=max_layers,
            k_best=args.k_best,
            layout=layout,
            prefix_l_range=prefix_range,
            stage_comm_ms=stage_comm_ms,
            all_boundaries=args.all_boundaries,
        )
    except (partition_optimizer.OptimizerError, stage_model.StageModelError) as exc:
        raise TuningError(str(exc)) from exc
    output_dir = (args.output_dir or (profile_dirs[0] / "analysis_multi")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "analysis.json", result.to_dict())
    (output_dir / "analysis.txt").write_text(result.to_report() + "\n")
    (output_dir / "recommended.args").write_text(
        "--pp-layer-partition "
        + ",".join(map(str, result.selected.partition))
        + " --speculative-dflash-dcut auto\n"
    )
    print(result.to_report(), flush=True)
    print(f"[analyze] artifacts written to {output_dir}", flush=True)
    return output_dir


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

    profile_dirs = args.profile_dir
    if not isinstance(profile_dirs, list):
        profile_dirs = [profile_dirs]
    profile_dirs = [Path(path).resolve() for path in profile_dirs]
    if len(profile_dirs) > 1:
        return _run_multi_bucket_analysis(args, profile_dirs)
    profile_dir = profile_dirs[0]
    if args.trim_samples < 0:
        raise TuningError("--trim-samples cannot be negative")
    profile = _load_profile(profile_dir)
    pp_size = int(profile["pp_size"])
    tp_size = int(profile["tp_size"])
    try:
        partition = tuple(int(value) for value in profile["baseline_partition"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TuningError("baseline profile has an invalid partition") from exc
    if len(partition) != pp_size or any(value <= 0 for value in partition):
        raise TuningError("baseline profile partition does not match pp_size")
    model_path = str(profile["model_path"])
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
            baseline_partition=partition,
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
    if args.boundary_radius is not None and not args.all_boundaries:
        if args.boundary_radius < 0:
            raise TuningError("--boundary-radius cannot be negative")
        center = partition[0]
        prefix_range = (
            max(1, center - args.boundary_radius),
            center + args.boundary_radius,
        )
    try:
        common_kwargs = dict(
            target_bs=bucket,
            min_layers=args.min_layers,
            max_layers=max_layers,
            k_best=args.k_best,
            layout=layout,
            prefix_l_range=prefix_range,
            stage_comm_ms=stage_comm_ms,
            all_boundaries=args.all_boundaries,
        )
        if args.dcut_profile is None:
            result = partition_optimizer.optimize(model, **common_kwargs)
        else:
            dcut_profiles = _load_dcut_profiles(args.dcut_profile, pp_size)
            # Runtime D-Cut is adaptive.  Pick a static PP layout that stays
            # balanced across the complete ratio envelope instead of tuning
            # the layout to one ratio that may not be selected next step.
            result = partition_optimizer.optimize_partition_across_ratios(
                model, dcut_profiles, **common_kwargs
            )
    except (partition_optimizer.OptimizerError, stage_model.StageModelError) as exc:
        raise TuningError(str(exc)) from exc
    output_dir = (args.output_dir or (profile_dir / "analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "analysis.json", result.to_dict())
    report = result.to_report()
    (output_dir / "analysis.txt").write_text(report + "\n")
    selected = result.selected.partition
    recommended = "--pp-layer-partition " + ",".join(map(str, selected))
    if hasattr(result.selected, "dcut_ratio"):
        if result.selected.dcut_ratio != 1.0:
            recommended += f" --speculative-dflash-dcut {result.selected.dcut_ratio:g}"
    elif args.dcut_profile is not None:
        recommended += " --speculative-dflash-dcut auto"
    (output_dir / "recommended.args").write_text(recommended + "\n")
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
    profile.add_argument(
        "--baseline-partition",
        help=(
            "comma-separated PP layer counts to profile; default: uniform, "
            "i.e. total model layers split evenly across PP stages"
        ),
    )
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
    analyze.add_argument(
        "--profile-dir",
        type=Path,
        action="append",
        required=True,
        help="profile directory; repeat for multiple execution buckets",
    )
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
    analyze.add_argument(
        "--all-boundaries",
        action="store_true",
        help="enumerate all valid PP compositions instead of only prefix-uniform ones",
    )
    analyze.add_argument(
        "--dcut-profile",
        type=Path,
        help=(
            "JSON ratio -> measured bottleneck cost, or ratio -> per-stage costs; "
            "enables joint PP partition and D-Cut selection"
        ),
    )
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
