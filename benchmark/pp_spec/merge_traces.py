#!/usr/bin/env python3
"""Merge multiple torch-profiler chrome traces (one per rank) into one file.

Each input contributes its events with timestamps rebased onto the machine's
shared wall clock via the trace's baseTimeNanoseconds, so ranks line up
(sanity check: matching NCCL collectives across ranks must overlap).
PIDs are already distinct across ranks; a suffix is added to process names.
"""

import argparse
import gzip
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument(
        "--rank-offset-ms",
        default=None,
        help="manual per-file shift, e.g. 'trace1.json=12.5' (file -> ms); "
        "use when auto alignment via base time is off",
    )
    args = parser.parse_args()

    manual = {}
    if args.rank_offset_ms:
        for chunk in args.rank_offset_ms.split(","):
            name, _, ms = chunk.partition("=")
            manual[name] = float(ms) * 1000.0  # ms -> us

    merged_events = []
    for path in args.inputs:
        data = load_json(path)
        base_ns = data.get("baseTimeNanoseconds", 0)
        shift_us = base_ns / 1000.0 + manual.get(path.name, 0.0)
        ts_lo, ts_hi = None, None
        for event in data.get("traceEvents", []):
            ts = event.get("ts")
            if isinstance(ts, (int, float)):
                event["ts"] = ts + shift_us
                ts_lo = ts if ts_lo is None else min(ts_lo, ts)
                ts_hi = ts if ts_hi is None else max(ts_hi, ts)
            if event.get("ph") == "M" and event.get("name") == "process_name":
                event["args"] = {
                    **event.get("args", {}),
                    "name": f"{event['args'].get('name', 'proc')} [{path.stem}]",
                }
            merged_events.append(event)
        if ts_lo is not None:
            print(
                f"{path.name}: {len(data.get('traceEvents', []))} events, "
                f"span {(ts_hi - ts_lo) / 1e6:.2f}s "
                f"(merged-window start {(ts_lo + shift_us) / 1e6:.3f}s)"
            )

    out = {"traceEvents": merged_events}
    target = args.output
    opener = gzip.open if target.suffix == ".gz" else open
    with opener(target, "wt") as handle:
        json.dump(out, handle)
    print(f"wrote {len(merged_events)} events -> {target}")


if __name__ == "__main__":
    main()
