#!/usr/bin/env python3
"""Post-process torch-profiler chrome traces into one lean, inspectable file.

Takes one trace per rank (as written by /start_profile) and emits a single
json.gz fit for dragging into ui.perfetto.dev:

* drops the noise -- python stacks, cpu ops, cuda runtime, memcpy/memset
  events (these dominate the raw file);
* keeps GPU kernels and GPU-side stage annotations (gpu_user_annotation);
* shortens mangled kernel names (template soup -> the config token);
* merges all ranks onto the shared wall clock via baseTimeNanoseconds, so
  TP/PP ranks line up on one timeline;
* synthesizes DRAFT / VERIFY / POST phase blocks per scheduler.run_batch by
  clustering the step[] annotations (the draft forward is the leading
  cluster, verify the long tail, everything after it is accept + draft-KV
  materialization). --no-phases disables this.

Output is typically ~5x smaller than the raw traces and contains only the
rows worth reading: kernels + a handful of named spans.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Optional

KEEP_CATS = {"kernel", "gpu_user_annotation"}
CLUSTER_GAP_US = 500.0


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
        if clusters and item["ts"] - (clusters[-1][-1]["ts"] + clusters[-1][-1]["dur"]) <= gap_us:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters


def synthesize_phases(annotations, pid: int, tid: int) -> list[dict]:
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
    path: Path, file_index: int, phases: bool, min_kernel_us: float = 0.0
) -> tuple[list[dict], str]:
    data = load(path)
    base_ns = data.get("baseTimeNanoseconds", 0)
    shift_us = base_ns / 1000.0
    pid_offset = 1000 * file_index

    kept = []
    annotations = []
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
        if cat not in KEEP_CATS or event.get("ph") != "X":
            continue
        if cat == "kernel" and min_kernel_us > 0 and event.get("dur", 0) < min_kernel_us:
            dropped += 1
            continue
        event = dict(event)
        event["pid"] = event["pid"] + pid_offset
        event["ts"] = event["ts"] + shift_us
        if cat == "kernel":
            event["name"] = shorten(event["name"])
        else:
            if tid_of_annotation is None:
                tid_of_annotation = event.get("tid", 0)
            annotations.append(event)
        kept.append(event)

    if phases and tid_of_annotation is not None and annotations:
        kept.extend(synthesize_phases(annotations, annotations[0]["pid"], tid_of_annotation))

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
        "--min-kernel-us",
        type=float,
        default=0.0,
        help="drop GPU kernels shorter than this many microseconds (default keep all)",
    )
    args = parser.parse_args()

    all_events = []
    for index, path in enumerate(args.inputs):
        events, label = slim_file(
            path, index, phases=not args.no_phases, min_kernel_us=args.min_kernel_us
        )
        print(label)
        all_events.extend(events)

    with gzip.open(args.output, "wt") as handle:
        json.dump({"traceEvents": all_events}, handle)
    print(f"wrote {len(all_events)} events -> {args.output}")


if __name__ == "__main__":
    main()
