#!/usr/bin/env python3
"""Collect all sweep summaries in a SPECTRE run into one CSV.

Expected result layout::

    RUN_DIR/
      CONFIG_NAME/
        LOAD_POINT/
          CONFIG_NAME_r1_summary.json

The first directory below RUN_DIR becomes the series label. Each configuration
must have exactly one complete row per concurrency; this keeps different
repeats or QPS values from being silently mixed in one sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


REQUIRED_FIELDS = {
    "max_concurrency",
    "num_requests",
    "completed",
    "output_throughput_tok_s",
    "ttft_p50_s",
    "ttft_p99_s",
    "tpot_p99_s",
}


class SummaryError(RuntimeError):
    pass


def _load_rows(run_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(run_dir.glob("*/*/*_summary.json"))
    if not paths:
        raise SummaryError(f"no per-point '*_summary.json' files found under {run_dir}")

    rows: list[dict[str, Any]] = []
    seen_points: dict[tuple[str, int], Path] = {}
    for path in paths:
        relative = path.relative_to(run_dir)
        config = relative.parts[0]
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SummaryError(f"cannot read summary {path}: {exc}") from exc
        if not isinstance(payload, list) or not payload:
            raise SummaryError(f"summary {path} must contain a non-empty JSON list")

        for index, raw_row in enumerate(payload):
            if not isinstance(raw_row, dict):
                raise SummaryError(f"summary {path} row {index} is not an object")
            missing = REQUIRED_FIELDS.difference(raw_row)
            if missing:
                raise SummaryError(
                    f"summary {path} row {index} is missing field(s): {sorted(missing)}"
                )
            try:
                concurrency = int(raw_row["max_concurrency"])
                requested = int(raw_row["num_requests"])
                completed = int(raw_row["completed"])
            except (TypeError, ValueError) as exc:
                raise SummaryError(
                    f"summary {path} row {index} has invalid request counts"
                ) from exc
            if concurrency <= 0 or requested <= 0 or completed < 0:
                raise SummaryError(
                    f"summary {path} row {index} has non-positive concurrency/requests"
                )
            if completed != requested:
                raise SummaryError(
                    f"summary {path} is incomplete: completed={completed}, "
                    f"num_requests={requested}"
                )

            point = (config, concurrency)
            if point in seen_points:
                raise SummaryError(
                    f"duplicate point config={config!r}, concurrency={concurrency}: "
                    f"{seen_points[point]} and {path}. Use one repeat and one QPS "
                    "per config/concurrency when plotting a concurrency sweep."
                )
            seen_points[point] = path

            row = dict(raw_row)
            row["label"] = config
            rows.append(row)

    rows.sort(key=lambda row: (str(row["label"]), int(row["max_concurrency"])))
    return rows


def _fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    names = ["label"]
    seen = {"label"}
    for row in rows:
        for name in row:
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def _write_csv(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="sweep run directory, e.g. results/Qwen/Qwen3-8B_20260825_120000",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"[summary] run directory not found: {run_dir}")

    output_path = run_dir / "summary.csv"
    try:
        rows = _load_rows(run_dir)
        _write_csv(rows, output_path)
    except SummaryError as exc:
        raise SystemExit(f"[summary] {exc}") from exc

    configs = sorted({str(row["label"]) for row in rows})
    print(
        f"[summary] wrote {len(rows)} point(s) across {len(configs)} "
        f"configuration(s) to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
