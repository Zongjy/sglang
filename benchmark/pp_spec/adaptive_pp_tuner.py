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
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_DRAFT_MODEL = "z-lab/Qwen3.5-9B-DFlash"
PROFILE_MANIFEST = "profile_manifest.json"


class TuningError(RuntimeError):
    pass


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
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
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.input_tokens),
        "--random-output-len",
        str(args.output_tokens),
        "--random-range-ratio",
        "0",
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
    execution_bucket = args.execution_bucket or math.ceil(
        args.batch_size / args.pp_size
    )
    if execution_bucket <= 0:
        raise TuningError("--execution-bucket must be positive")
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "profile_dir": profile_dir,
        "model_path": args.model_path,
        "draft_model_path": args.draft_model_path,
        "tp_size": args.tp_size,
        "pp_size": args.pp_size,
        "nnodes": args.nnodes,
        "current_partition": partition,
        "batch_size": args.batch_size,
        "execution_bucket": execution_bucket,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "profile_steps": args.profile_steps,
        "block_size": args.block_size,
        "command": command,
    }
    write_json(profile_dir / PROFILE_MANIFEST, manifest)

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
        manifest.update(status="failed", returncode=completed.returncode)
        write_json(profile_dir / PROFILE_MANIFEST, manifest)
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
    manifest.update(
        status="complete",
        completed_at=datetime.now(timezone.utc).isoformat(),
        traces=[str(path.relative_to(profile_dir)) for path in traces],
    )
    write_json(profile_dir / PROFILE_MANIFEST, manifest)
    print(f"[profile] traces written under {profile_dir}", flush=True)
    return profile_dir


def _load_manifest(profile_dir: Path) -> dict[str, Any]:
    path = profile_dir / PROFILE_MANIFEST
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TuningError(f"cannot read {path}: {exc}") from exc
    if manifest.get("status") != "complete":
        raise TuningError(f"profile is not complete: {path}")
    return manifest


def _typed_layer_costs(
    layout: Any,
    observations: Sequence[tuple[Sequence[int], Sequence[float]]],
) -> tuple[float, float, float, list[str]]:
    per_layer = []
    for partition, target_ms in observations:
        ranks = (
            range(1, len(partition) - 1)
            if len(partition) > 2
            else range(len(partition))
        )
        for rank in ranks:
            value = target_ms[rank]
            count = partition[rank]
            if value > 0.0 and count > 0:
                per_layer.append(value / count)
    fallback = float(sorted(per_layer)[len(per_layer) // 2]) if per_layer else 0.0
    warnings: list[str] = []
    if observations and len(observations[0][0]) <= 2:
        warnings.append(
            "PP<=2 has no middle stage; endpoint overhead cannot be separated "
            "completely from typed layer costs"
        )
    if layout is None or not per_layer:
        return fallback, fallback, fallback, warnings

    rows: list[tuple[int, int, float]] = []
    for partition, target_ms in observations:
        fit_ranks = (
            range(1, len(partition) - 1)
            if len(partition) > 2
            else range(len(partition))
        )
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
        warnings.append(
            "baseline stages do not provide independent GDN/full compositions; "
            "using one average per-layer cost"
        )
        return fallback, fallback, fallback, warnings
    gdn_cost = (gy * ff - fy * gf) / determinant
    full_cost = (fy * gg - gy * gf) / determinant
    if gdn_cost <= 0.0 or full_cost <= 0.0:
        warnings.append(
            "typed least-squares fit produced a non-positive layer cost; using "
            "one average per-layer cost"
        )
        return fallback, fallback, fallback, warnings
    weighted = sum(gdn * gdn_cost + full * full_cost for gdn, full, _ in rows) / sum(
        gdn + full for gdn, full, _ in rows
    )
    return weighted, gdn_cost, full_cost, warnings


def _target_observation(summary: Mapping[str, Any]) -> list[float]:
    service = [float(value) for value in summary["service_ms"]]
    target = [float(value) for value in summary.get("target_ms", ())]
    if any(value > 0.0 for value in target):
        return target
    target = list(service)
    if target:
        # The last stage owns DFlash proposal/materialization. Exclude it from
        # a fallback layer fit and recover its residual after fitting the other
        # ranks or neighboring partition profiles.
        target[-1] = 0.0
    return target


def _bucket_profile(
    summary: Mapping[str, Any],
    partition: Sequence[int],
    layout: Any,
    accept_len: float,
    typed_costs: tuple[float, float, float, list[str]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    service = [float(value) for value in summary["service_ms"]]
    target = _target_observation(summary)
    draft = float(summary.get("draft_ms", 0.0))
    layer, gdn, full, warnings = typed_costs or _typed_layer_costs(
        layout, [(partition, target)]
    )
    warnings = list(warnings)
    fixed: list[float] = []
    start = 0
    for rank, (count, observed) in enumerate(zip(partition, service)):
        if layout is not None:
            gdn_count, full_count = layout.count_range(start, start + count)
            predicted_layers = gdn_count * gdn + full_count * full
        else:
            predicted_layers = count * layer
        start += count
        residual = observed - predicted_layers
        if residual < 0.0:
            warnings.append(
                f"rank {rank} fixed residual {residual:.3f} ms was clamped to zero"
            )
            residual = 0.0
        fixed.append(residual)

    raw = {
        "service_ms": service,
        "service_var": summary.get("service_var", [0.0] * len(service)),
        "fixed_ms": fixed,
        "fixed_var": summary.get("service_var", [0.0] * len(service)),
        "layer_cost_ms": layer,
        "gdn_cost_ms": gdn,
        "full_cost_ms": full,
        "draft_ms": draft,
        "draft_var": float(summary.get("draft_var", 0.0)),
        "accept_len": accept_len,
        "wait_fraction": summary.get("wait_fraction", [0.0] * len(service)),
        "samples": int(summary.get("samples", 0)),
    }
    return raw, warnings


def run_analysis(args: argparse.Namespace) -> Path:
    sys.path.insert(0, str(SCRIPT_DIR))
    import partition_optimizer
    import stage_model
    import torch_trace_profile
    from model_layout import LayerLayout

    profile_dirs = [path.resolve() for path in args.profile_dir]
    if args.trim_samples < 0:
        raise TuningError("--trim-samples cannot be negative")
    if args.target_batch_size is not None and args.target_batch_size <= 0:
        raise TuningError("--target-batch-size must be positive")
    if args.accept_len < 0.0:
        raise TuningError("--accept-len cannot be negative")
    manifests = [_load_manifest(path) for path in profile_dirs]
    first = manifests[0]
    pp_size = int(first["pp_size"])
    partition = (
        parse_partition(args.current_partition, pp_size)
        if args.current_partition
        else tuple(int(value) for value in first["current_partition"])
    )
    model_path = args.model_path or str(first["model_path"])
    num_layers = sum(partition)
    for manifest in manifests:
        if int(manifest["pp_size"]) != pp_size:
            raise TuningError("all profiles must use the same pp_size")
        profile_partition = tuple(int(value) for value in manifest["current_partition"])
        if len(profile_partition) != pp_size or sum(profile_partition) != num_layers:
            raise TuningError(
                "all profile partitions must cover the same model layers and pp_size"
            )
        if str(manifest["model_path"]) != str(first["model_path"]):
            raise TuningError("all profiles must use the same model")
    if len(set(partition[:-1])) > 1:
        raise TuningError(
            "the thin optimizer currently searches (l,...,l,residual); the first "
            "PP stages in --current-partition must have the same layer count"
        )

    layout = None
    layout_warning = None
    try:
        layout = LayerLayout.from_model_path(model_path, local_files_only=True)
    except Exception as exc:
        layout_warning = (
            f"model layout unavailable; treating all layers equally ({exc})"
        )

    observations: dict[int, list[dict[str, Any]]] = {}
    warnings: list[str] = []
    for profile_dir, manifest in zip(profile_dirs, manifests):
        bucket = int(manifest.get("execution_bucket", manifest["batch_size"]))
        profile_partition = tuple(int(value) for value in manifest["current_partition"])
        try:
            summary = torch_trace_profile.summarize_trace_dir(
                profile_dir,
                pp_size=pp_size,
                trim_samples=args.trim_samples,
            )
        except torch_trace_profile.TraceProfileError as exc:
            raise TuningError(str(exc)) from exc
        observations.setdefault(bucket, []).append(
            {
                "profile_dir": profile_dir,
                "partition": profile_partition,
                "summary": summary,
            }
        )
        warnings.extend(summary.get("warnings", ()))
    if layout_warning:
        warnings.append(layout_warning)

    profiles: dict[int, dict[str, Any]] = {}
    for bucket, bucket_observations in sorted(observations.items()):
        baseline = next(
            (
                item
                for item in bucket_observations
                if tuple(item["partition"]) == partition
            ),
            None,
        )
        if baseline is None:
            raise TuningError(
                f"execution bucket {bucket} has no profile for baseline "
                f"partition {partition}"
            )
        typed_costs = _typed_layer_costs(
            layout,
            [
                (
                    item["partition"],
                    _target_observation(item["summary"]),
                )
                for item in bucket_observations
            ],
        )
        raw, bucket_warnings = _bucket_profile(
            baseline["summary"],
            partition,
            layout,
            args.accept_len,
            typed_costs=typed_costs,
        )
        profiles[bucket] = raw
        warnings.extend(bucket_warnings)

    try:
        model = stage_model.StageCostModel.from_bucket_profiles(
            profiles,
            num_layers=sum(partition),
            pp_size=pp_size,
            current_partition=partition,
            pp_loop_size=args.pp_loop_size or pp_size,
            layout=layout,
        )
    except stage_model.StageModelError as exc:
        raise TuningError(str(exc)) from exc
    model.warnings.extend(warnings)
    max_layers = parse_int_list(args.max_layers_per_rank, "--max-layers-per-rank")
    if args.min_layers <= 0:
        raise TuningError("--min-layers must be positive")
    if max_layers is not None and len(max_layers) != pp_size:
        raise TuningError("--max-layers-per-rank needs one value per PP rank")
    target_bucket = args.target_batch_size or max(profiles)
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
            target_bs=target_bucket,
            t_comm_ms=args.t_comm_ms,
            min_layers=args.min_layers,
            max_layers=max_layers,
            k_best=args.k_best,
            layout=layout,
            prefix_l_range=prefix_range,
            stage_comm_ms=stage_comm_ms,
        )
    except (partition_optimizer.OptimizerError, stage_model.StageModelError) as exc:
        raise TuningError(str(exc)) from exc

    validation = []
    for bucket, bucket_observations in sorted(observations.items()):
        estimate = model.estimate_for_bs(bucket)
        for item in bucket_observations:
            candidate_partition = tuple(item["partition"])
            predicted_stages = model.predict_stages(
                candidate_partition,
                estimate=estimate,
                t_comm_ms=args.t_comm_ms,
                stage_comm_ms=result.stage_comm_ms,
                layout=layout,
            )
            measured_stages = tuple(
                float(value) + result.stage_comm_ms[rank]
                for rank, value in enumerate(item["summary"]["service_ms"])
            )
            predicted_cycle = max(predicted_stages)
            measured_cycle = max(measured_stages)
            relative_error = (
                abs(predicted_cycle - measured_cycle) / measured_cycle
                if measured_cycle > 0.0
                else 0.0
            )
            validation.append(
                {
                    "bucket": bucket,
                    "partition": candidate_partition,
                    "predicted_stage_ms": predicted_stages,
                    "measured_stage_ms": measured_stages,
                    "predicted_cycle_ms": predicted_cycle,
                    "measured_cycle_ms": measured_cycle,
                    "relative_error": relative_error,
                }
            )
            if candidate_partition != partition and relative_error > 0.10:
                model.warnings.append(
                    f"profiled partition {candidate_partition} at bucket {bucket} "
                    f"has {relative_error:.1%} model error; collect another nearby "
                    "partition before deployment"
                )
    result.warnings.extend(model.warnings)

    output_dir = (args.output_dir or (profile_dirs[0] / "analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_profiles": profile_dirs,
        "model_path": model_path,
        "current_partition": partition,
        "target_batch_size": target_bucket,
        "trace_observations": observations,
        "profile_validation": validation,
        "stage_model": model.to_dict(),
        "optimization": result.to_dict(),
        "modeling": {
            "objective": "minimize measured post-overlap bottleneck stage time",
            "candidate_family": "(l,...,l,L-(P-1)l)",
            "boundary_radius": args.boundary_radius,
            "memory_constraint": (
                "max_layers_per_rank" if max_layers is not None else "not calibrated"
            ),
            "stage_comm_ms": stage_comm_ms,
            "inspiration": [
                "HexGen: compute + communication with hard memory constraints",
                "Tessera: profile post-overlap costs before partition selection",
            ],
        },
    }
    write_json(output_dir / "analysis.json", payload)
    report = result.to_report()
    profiled_selected = [
        item
        for item in validation
        if tuple(item["partition"]) == result.selected.partition
        and int(item["bucket"]) == int(result.target_bucket)
    ]
    if profiled_selected:
        item = profiled_selected[0]
        report += (
            "\nprofiled selected partition: measured cycle "
            f"{item['measured_cycle_ms']:.3f} ms, model error "
            f"{item['relative_error']:.1%}\n"
        )
    if max_layers is None:
        report += (
            "\nWARNING: no per-rank memory constraint was supplied; validate the "
            "recommended partition before deployment.\n"
        )
    (output_dir / "analysis.txt").write_text(report + "\n")
    selected = result.selected.partition
    (output_dir / "recommended.env").write_text(
        "SGLANG_PP_LAYER_PARTITION=" + ",".join(map(str, selected)) + "\n"
    )
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
    profile.add_argument("--block-size", type=int, default=8)
    profile.add_argument("--page-size", type=int, default=1)
    profile.add_argument("--mem-fraction-static", type=float, default=0.82)
    profile.add_argument("--mamba-ssm-dtype", default="bfloat16")
    profile.add_argument("--mamba-full-memory-ratio", type=float, default=2.0)
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
    analyze.add_argument("--profile-dir", type=Path, action="append", required=True)
    analyze.add_argument("--output-dir", type=Path)
    analyze.add_argument("--model-path")
    analyze.add_argument(
        "--current-partition",
        help="baseline partition to optimize; default: first profile manifest",
    )
    analyze.add_argument("--target-batch-size", type=int)
    analyze.add_argument("--pp-loop-size", type=int)
    analyze.add_argument("--accept-len", type=float, default=1.0)
    analyze.add_argument("--t-comm-ms", type=float, default=0.0)
    analyze.add_argument(
        "--stage-comm-ms",
        help=(
            "comma-separated exposed communication cost per PP rank (P values) "
            "or per boundary (P-1 values); overrides --t-comm-ms"
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
