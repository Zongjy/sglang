#!/usr/bin/env python3
"""Post-process torch-profiler chrome traces into one lean, inspectable file.

Takes one trace per rank (as written by /start_profile) and emits a single
json.gz fit for dragging into ui.perfetto.dev:

* drops python stacks, generic cpu ops, and memcpy/memset noise;
* keeps GPU kernels, GPU-side stage annotations, selected PP/scheduler CPU
  spans, and blocking CUDA synchronization calls;
* shortens mangled kernel names (template soup -> the config token);
* merges all ranks onto the shared wall clock via baseTimeNanoseconds, so
  TP/PP ranks line up on one timeline;
* synthesizes DRAFT / VERIFY / POST phase blocks per scheduler.run_batch by
  clustering the step[] annotations (the draft forward is the leading
  cluster outside PP and the trailing cluster for PP+DFlash);
* synthesizes INTER_STEP_GAP spans for PP traces, including GPU busy/idle time
  and the overlapping control-plane activity. --no-gaps disables this.

Output is typically ~5x smaller than the raw traces and contains only the
rows worth reading: kernels + a handful of named spans.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Optional

CLUSTER_GAP_US = 500.0
BLOCKING_CUDA_RUNTIME = {
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cudaStreamSynchronize",
}
CONTROL_PREFIXES = (
    "scheduler.",
    "gloo:",
    "recv_",
    "send_",
    "process_",
    "get_",
    "run_batch",
    "copy_result_to_cpu",
)


def load(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def shorten(name: str, limit: int = 56) -> str:
    """Strip args / template soup, keep the distinguishing config token."""
    n = name.split("(", 1)[0].strip()
    if n.startswith("void "):
        n = n[5:]
    if "<" in n:
        head, _, rest = n.partition("<")
        first_arg = rest.split(",", 1)[0].split("<", 1)[0]
        n = f"{head}<{first_arg}" if first_arg else head
    return n[:limit] or name[:limit]


def cluster(items, gap_us: float = CLUSTER_GAP_US):
    clusters = []
    for item in items:
        previous_end = (
            max(event["ts"] + event["dur"] for event in clusters[-1])
            if clusters
            else 0
        )
        if clusters and item["ts"] - previous_end <= gap_us:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters


def merge_overlapping_steps(items) -> list[dict]:
    groups = []
    for item in sorted(items, key=lambda event: event["ts"]):
        start = float(item["ts"])
        end = start + float(item["dur"])
        if groups and start < groups[-1]["end"]:
            groups[-1]["end"] = max(groups[-1]["end"], end)
            groups[-1]["members"].append(item)
        else:
            groups.append({"start": start, "end": end, "members": [item]})
    return groups


def interval_union_us(intervals) -> float:
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def control_kind(event: dict) -> str | None:
    name = str(event.get("name", ""))
    if ".wait_result_d2h" in name:
        return "D2H_WAIT"
    if event.get("cat") == "cuda_runtime":
        return "CUDA_SYNC"
    if "process_batch_result" in name:
        return "RESULT_PROCESS"
    if (
        name.startswith(("gloo:", "recv_", "send_"))
        or ".recv_" in name
        or ".send_" in name
        or ".wait_send" in name
        or ".exchange_previous_output" in name
    ):
        return "PP_COMM"
    if "get_next_batch" in name or "recv_requests" in name or "process_input" in name:
        return "SCHEDULER"
    if name.endswith("run_batch") or name == "run_batch":
        return "HOST_LAUNCH"
    return None


def synthesize_inter_step_gaps(
    annotations: list[dict],
    controls: list[dict],
    kernels: list[dict],
    pid: int,
    tid: int,
    min_gap_us: float,
) -> list[dict]:
    steps = [item for item in annotations if item["name"].startswith("step[")]
    groups = merge_overlapping_steps(steps)
    if len(groups) < 2:
        return []
    max_members = max(len(group["members"]) for group in groups)

    def phase(group: dict) -> str:
        return "TARGET" if len(group["members"]) == max_members else "DRAFT"

    out = []
    kind_order = (
        "PP_COMM",
        "D2H_WAIT",
        "CUDA_SYNC",
        "RESULT_PROCESS",
        "SCHEDULER",
        "HOST_LAUNCH",
    )
    for left, right in zip(groups, groups[1:]):
        lo, hi = left["end"], right["start"]
        duration = hi - lo
        if duration < min_gap_us:
            continue
        busy_intervals = []
        for kernel in kernels:
            start = float(kernel["ts"])
            end = start + float(kernel["dur"])
            if start < hi and end > lo:
                busy_intervals.append((max(lo, start), min(hi, end)))
        busy_us = interval_union_us(busy_intervals)

        kinds = set()
        top_controls: dict[str, float] = {}
        for event in controls:
            start = float(event["ts"])
            end = start + float(event.get("dur", 0))
            overlap = max(0.0, min(hi, end) - max(lo, start))
            if overlap < 10.0:
                continue
            name = str(event.get("name", ""))
            top_controls[name] = max(top_controls.get(name, 0.0), overlap)
            kind = control_kind(event)
            if kind:
                kinds.add(kind)
        ordered_kinds = [kind for kind in kind_order if kind in kinds]
        if not ordered_kinds:
            ordered_kinds = ["UNATTRIBUTED_HOST_WAIT"]
        top = sorted(top_controls.items(), key=lambda item: -item[1])[:8]
        out.append(
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": f"INTER_STEP_GAP {phase(left)}->{phase(right)}",
                "pid": pid,
                "tid": tid,
                "ts": lo,
                "dur": duration,
                "args": {
                    "derived": True,
                    "duration_ms": round(duration / 1000.0, 3),
                    "gpu_busy_ms": round(busy_us / 1000.0, 3),
                    "gpu_idle_ms": round((duration - busy_us) / 1000.0, 3),
                    "gpu_busy_pct": round(100.0 * busy_us / duration, 1),
                    "overlapping_activity": ", ".join(ordered_kinds),
                    "top_control_spans": ", ".join(
                        f"{name}={overlap / 1000.0:.3f}ms"
                        for name, overlap in top
                    ),
                },
            }
        )
    return out


def synthesize_phases(
    annotations, pid: int, tid: int, pp_mode: bool = False
) -> list[dict]:
    """Emit DRAFT/VERIFY/POST gpu annotations for each run_batch window."""
    run_batches = sorted(
        (a for a in annotations if a["name"] == "scheduler.run_batch" and a["dur"] > 1000),
        key=lambda a: a["ts"],
    )
    steps = sorted(
        (a for a in annotations if a["name"].startswith("step[")), key=lambda a: a["ts"]
    )
    if not run_batches or not steps:
        return []

    out = []
    for batch in run_batches:
        lo, hi = batch["ts"], batch["ts"] + batch["dur"]
        inner = [s for s in steps if lo <= s["ts"] < hi]
        if len(inner) < 2:
            continue
        groups = cluster(inner)
        if pp_mode and len(groups) >= 2:
            first_start = min(item["ts"] for item in groups[0])
            verify_end = max(
                item["ts"] + item["dur"] for item in groups[-2]
            )
            draft_start = min(item["ts"] for item in groups[-1])
            draft_end = max(item["ts"] + item["dur"] for item in groups[-1])
            draft_duration = draft_end - draft_start
            verify_duration = verify_end - first_start
            if draft_duration < 0.4 * batch["dur"] and draft_duration < verify_duration:
                out.append(
                    {"ph": "X", "cat": "gpu_user_annotation", "name": "VERIFY",
                     "pid": pid, "tid": tid, "ts": first_start,
                     "dur": verify_end - first_start, "args": {}}
                )
                if verify_end < draft_start:
                    out.append(
                        {"ph": "X", "cat": "gpu_user_annotation", "name": "POST",
                         "pid": pid, "tid": tid, "ts": verify_end,
                         "dur": draft_start - verify_end, "args": {}}
                    )
                out.append(
                    {"ph": "X", "cat": "gpu_user_annotation", "name": "DRAFT",
                     "pid": pid, "tid": tid, "ts": draft_start,
                     "dur": draft_end - draft_start, "args": {}}
                )
                continue
        # DRAFT = leading cluster when it is clearly the small block-diffusion
        # forward (short, holds a minority of the step spans); otherwise the
        # whole window is one verify and no split is emitted.
        split = len(groups) >= 2 and groups[0][-1]["ts"] + groups[0][-1]["dur"] - lo < 0.4 * batch["dur"]
        if not split:
            out.append(
                {"ph": "X", "cat": "gpu_user_annotation", "name": "VERIFY",
                 "pid": pid, "tid": tid, "ts": lo, "dur": batch["dur"], "args": {}}
            )
            continue
        draft_end = groups[0][-1]["ts"] + groups[0][-1]["dur"]
        verify_end = groups[-1][-1]["ts"] + groups[-1][-1]["dur"]
        out.append({"ph": "X", "cat": "gpu_user_annotation", "name": "DRAFT",
                    "pid": pid, "tid": tid, "ts": lo, "dur": draft_end - lo, "args": {}})
        out.append({"ph": "X", "cat": "gpu_user_annotation", "name": "VERIFY",
                    "pid": pid, "tid": tid, "ts": draft_end, "dur": verify_end - draft_end, "args": {}})
        if verify_end < hi:
            out.append({"ph": "X", "cat": "gpu_user_annotation", "name": "POST",
                        "pid": pid, "tid": tid, "ts": verify_end, "dur": hi - verify_end, "args": {}})
    return out


def slim_file(
    path: Path,
    file_index: int,
    phases: bool,
    min_kernel_us: float = 0.0,
    gaps: bool = True,
    min_gap_us: float = 100.0,
) -> tuple[list[dict], str]:
    data = load(path)
    base_ns = data.get("baseTimeNanoseconds", 0)
    shift_us = base_ns / 1000.0
    pid_offset = 1000 * file_index

    kept = []
    annotations = []
    controls = []
    kernels = []
    tid_of_annotation: Optional[int] = None
    dropped = 0
    for event in data.get("traceEvents", []):
        cat = event.get("cat", "")
        if event.get("ph") == "M" and event.get("name") == "process_name":
            event = dict(event)
            event["pid"] = event["pid"] + pid_offset
            event["args"] = {
                **event.get("args", {}),
                "name": f"{event['args'].get('name', 'proc')} [{path.stem[-12:]}]",
            }
            kept.append(event)
            continue
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        keep = (
            cat in {"kernel", "gpu_user_annotation"}
            or (cat == "user_annotation" and name.startswith(CONTROL_PREFIXES))
            or (cat == "cuda_runtime" and name in BLOCKING_CUDA_RUNTIME)
        )
        if not keep:
            continue
        if cat == "kernel" and min_kernel_us > 0 and event.get("dur", 0) < min_kernel_us:
            dropped += 1
            continue
        event = dict(event)
        event["pid"] = event["pid"] + pid_offset
        event["ts"] = event["ts"] + shift_us
        if cat == "kernel":
            event["name"] = shorten(event["name"])
            kernels.append(event)
        elif cat == "gpu_user_annotation":
            if tid_of_annotation is None:
                tid_of_annotation = event.get("tid", 0)
            annotations.append(event)
        else:
            controls.append(event)
        kept.append(event)

    if phases and tid_of_annotation is not None and annotations:
        kept.extend(
            synthesize_phases(
                annotations,
                annotations[0]["pid"],
                tid_of_annotation,
                pp_mode="-PP-" in path.name,
            )
        )
    if gaps and tid_of_annotation is not None and annotations and "-PP-" in path.name:
        kept.extend(
            synthesize_inter_step_gaps(
                annotations,
                controls,
                kernels,
                annotations[0]["pid"],
                tid_of_annotation,
                min_gap_us,
            )
        )

    label = (
        f"{path.name}: kept {len(kept)} events"
        + (f" (dropped {dropped} kernels < {min_kernel_us}us)" if dropped else "")
        + f" (pid offset {pid_offset}, base +{shift_us:.0f}us)"
    )
    return kept, label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--no-phases", action="store_true", help="skip DRAFT/VERIFY/POST synthesis")
    parser.add_argument(
        "--no-gaps",
        action="store_true",
        help="skip derived PP INTER_STEP_GAP spans",
    )
    parser.add_argument(
        "--min-gap-us",
        type=float,
        default=100.0,
        help="minimum PP inter-step gap to annotate (default: 100us)",
    )
    parser.add_argument(
        "--min-kernel-us",
        type=float,
        default=0.0,
        help="drop GPU kernels shorter than this many microseconds (default keep all)",
    )
    args = parser.parse_args()

    all_events = []
    for index, path in enumerate(args.inputs):
        events, label = slim_file(
            path,
            index,
            phases=not args.no_phases,
            min_kernel_us=args.min_kernel_us,
            gaps=not args.no_gaps,
            min_gap_us=args.min_gap_us,
        )
        print(label)
        all_events.extend(events)

    with gzip.open(args.output, "wt") as handle:
        json.dump({"traceEvents": all_events}, handle)
    print(f"wrote {len(all_events)} events -> {args.output}")


if __name__ == "__main__":
    main()
