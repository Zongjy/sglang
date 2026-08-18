#!/usr/bin/env python3
"""High-level breakdown of a (merged) torch-profiler chrome trace.

Prints per-process GPU busy %, stage-span averages (draft/verify/...),
NCCL time share and top kernels -- no timeline reading required.
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def merge_intervals(intervals):
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top-kernels", type=int, default=12)
    args = parser.parse_args()

    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt") as handle:
        data = json.load(handle)
    events = data.get("traceEvents", [])
    base_ns = data.get("baseTimeNanoseconds", 0)

    pid_name = {}
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            pid_name[event.get("pid")] = event["args"].get("name", str(event.get("pid")))

    kernels = defaultdict(list)  # pid -> [(ts, dur, name)]
    spans = defaultdict(list)  # pid -> [(ts, dur, name)] -- user_annotation only
    gpu_spans = defaultdict(list)  # pid -> [(ts, dur, name)] -- gpu_user_annotation
    for event in events:
        if event.get("ph") != "X":
            continue
        ts = event.get("ts", 0) + base_ns / 1000
        dur = event.get("dur", 0)
        cat = event.get("cat", "")
        if cat == "kernel":
            kernels[event["pid"]].append((ts, dur, event["name"]))
        elif cat == "user_annotation" and dur > 0:
            spans[event["pid"]].append((ts, dur, event["name"]))
        elif cat == "gpu_user_annotation" and dur > 0:
            gpu_spans[event["pid"]].append((ts, dur, event["name"]))

    for pid, _ in sorted(kernels.items(), key=lambda kv: pid_name.get(kv[0], "?")):
        label = pid_name.get(pid, f"pid {pid}")
        klist = kernels[pid]
        lo = min(ts for ts, _, _ in klist)
        hi = max(ts + dur for ts, dur, _ in klist)
        window = hi - lo
        busy = sum(end - start for start, end in merge_intervals([(ts, ts + d) for ts, d, _ in klist]))
        nccl = sum(d for _, d, n in klist if "nccl" in n.lower())
        print(f"\n=== {label} ===")
        print(f"GPU window: {window/1e6:.2f}s  busy: {busy/1e6:.2f}s ({busy/window*100:.1f}%)  "
              f"nccl: {nccl/1e6:.2f}s ({nccl/window*100:.1f}%)")

        # stage spans (user_annotation = CPU-side range, gpu_user_annotation
        # = the same range projected onto the GPU timeline)
        by_name = defaultdict(list)
        for ts, dur, name in gpu_spans[pid]:
            by_name[name].append((ts, dur))
        interesting = [(n, t) for n, t in by_name.items() if len(t) >= 5]
        interesting.sort(key=lambda kv: -sum(d for _, d in kv[1]))
        print("stage spans on the GPU timeline (count / avg ms / total ms):")
        for name, times in interesting[:12]:
            total = sum(d for _, d in times)
            print(f"  {name[:70]:<70} {len(times):>5}  {total/len(times)/1e3:>8.2f}  {total/1e6:>8.1f}")

        totals = defaultdict(float)
        for _, dur, name in klist:
            totals[name] += dur
        print(f"top kernels by total GPU time:")
        for name, total in sorted(totals.items(), key=lambda kv: -kv[1])[: args.top_kernels]:
            print(f"  {total/1e6:>7.1f}ms {total/busy*100:>5.1f}%  {name[:80]}")


if __name__ == "__main__":
    main()
