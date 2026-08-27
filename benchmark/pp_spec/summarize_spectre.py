#!/usr/bin/env python3
"""Collect SPECTRE point summaries into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_rows(run_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(run_dir.glob("*/*/*_summary.json")):
        config = path.relative_to(run_dir).parts[0]
        for raw in json.loads(path.read_text()):
            rows.append({"label": config, **raw})
    return sorted(rows, key=lambda row: (row["label"], row["max_concurrency"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    run_dir = parser.parse_args().run_dir.resolve()
    rows = load_rows(run_dir)
    if not rows:
        raise SystemExit(f"no summaries found under {run_dir}")

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    output = run_dir / "summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
