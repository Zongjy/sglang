#!/usr/bin/env python3
"""Plot stacked DFlash CUDA graph memory grouped by tensor parallel size."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import msgspec
from bench_dflash_cuda_graph_memory import CSV_FIELDS, DRAFT_COLUMN, TARGET_COLUMN

EDGE_COLOR = "#30343B"
GRID_COLOR = "#D9DEE3"
TARGET_COLOR = "#1994C9"
DRAFT_COLOR = "#DD3924"
TP_HATCHES = ("", "///", "\\\\", "xx", "..", "++")


class PlotError(RuntimeError):
    pass


class Measurement(msgspec.Struct, frozen=True):
    model: str
    tp_size: int
    max_batch_size: int
    target_gib: float
    draft_gib: float

    @property
    def total_gib(self) -> float:
        return self.target_gib + self.draft_gib


def positive_int(value: str, field: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PlotError(f"row {row_number}: {field} must be an integer") from exc
    if parsed <= 0:
        raise PlotError(f"row {row_number}: {field} must be positive")
    return parsed


def nonnegative_float(value: str, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PlotError(f"row {row_number}: {field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise PlotError(f"row {row_number}: {field} must be finite and non-negative")
    return parsed


def positive_cli_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def load_measurements(path: Path) -> list[Measurement]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PlotError(f"CSV has no header: {path}")
        missing = [name for name in CSV_FIELDS if name not in reader.fieldnames]
        if missing:
            raise PlotError(f"CSV is missing required columns: {', '.join(missing)}")

        measurements = []
        for row_number, row in enumerate(reader, start=2):
            if row["status"].strip() != "ok":
                continue
            model = row["model"].strip()
            if not model:
                raise PlotError(f"row {row_number}: model must not be empty")
            batch_size = positive_int(
                row["max_cuda_graph_bs"], "max_cuda_graph_bs", row_number
            )
            tp_size = positive_int(row["tp_size"], "tp_size", row_number)
            target_gib = nonnegative_float(
                row[TARGET_COLUMN], TARGET_COLUMN, row_number
            )
            draft_gib = nonnegative_float(row[DRAFT_COLUMN], DRAFT_COLUMN, row_number)
            measurements.append(
                Measurement(
                    model=model,
                    tp_size=tp_size,
                    max_batch_size=batch_size,
                    target_gib=target_gib,
                    draft_gib=draft_gib,
                )
            )

    if not measurements:
        raise PlotError("CSV contains no successful measurements")
    return measurements


def group_measurements(
    measurements: Sequence[Measurement],
) -> dict[str, list[Measurement]]:
    groups: dict[str, list[Measurement]] = {}
    for measurement in measurements:
        groups.setdefault(measurement.model, []).append(measurement)
    for points in groups.values():
        points.sort(key=lambda point: (point.max_batch_size, point.tp_size))
    return groups


def select_tp_sizes(
    measurements: Sequence[Measurement], tp_sizes: Sequence[int] | None
) -> list[Measurement]:
    if tp_sizes is None:
        return list(measurements)
    selected = set(tp_sizes)
    filtered = [point for point in measurements if point.tp_size in selected]
    if not filtered:
        available = sorted({point.tp_size for point in measurements})
        raise PlotError(
            f"CSV contains no requested TP sizes {sorted(selected)}; "
            f"available TP sizes: {available}"
        )
    return filtered


def add_value_labels(axis: Any, bars: Any, values: Sequence[float]) -> None:
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{value:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=EDGE_COLOR,
        )


def draw_group(
    axis: Any,
    model: str,
    points: Sequence[Measurement],
    hatch_by_tp: dict[int, str],
    max_value: float,
) -> None:
    batch_sizes = sorted({point.max_batch_size for point in points})
    positions = list(range(len(batch_sizes)))
    position_by_batch = {
        batch_size: position for position, batch_size in enumerate(batch_sizes)
    }
    tp_sizes = sorted({point.tp_size for point in points})
    cluster_width = 0.76 if len(tp_sizes) > 1 else 0.52
    width = cluster_width / len(tp_sizes)

    for tp_index, tp_size in enumerate(tp_sizes):
        offset = -cluster_width / 2 + width * (tp_index + 0.5)
        tp_points = [point for point in points if point.tp_size == tp_size]
        x_values = [
            position_by_batch[point.max_batch_size] + offset for point in tp_points
        ]
        target_values = [point.target_gib for point in tp_points]
        draft_values = [point.draft_gib for point in tp_points]
        total_values = [point.total_gib for point in tp_points]
        hatch = hatch_by_tp[tp_size]
        axis.bar(
            x_values,
            target_values,
            width * 0.9,
            color=TARGET_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.6,
            hatch=hatch,
        )
        total_bars = axis.bar(
            x_values,
            draft_values,
            width * 0.9,
            bottom=target_values,
            color=DRAFT_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.6,
            hatch=hatch,
        )
        add_value_labels(axis, total_bars, total_values)

    axis.set_xticks(positions, [str(batch_size) for batch_size in batch_sizes])
    axis.set_title(model)
    axis.set_xlabel("Max CUDA graph batch size")
    axis.set_ylim(0, max(max_value * 1.18, 1.0))
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def render_chart(measurements: Sequence[Measurement], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    groups = group_measurements(measurements)
    group_items = list(groups.items())
    column_count = min(2, len(group_items))
    row_count = math.ceil(len(group_items) / column_count)
    figure, axes_grid = plt.subplots(
        row_count,
        column_count,
        figsize=(6.2 * column_count, 4.6 * row_count),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_grid.flat)
    all_tp_sizes = sorted({point.tp_size for point in measurements})
    hatch_by_tp = {
        tp_size: TP_HATCHES[index % len(TP_HATCHES)]
        for index, tp_size in enumerate(all_tp_sizes)
    }
    max_value = max(point.total_gib for point in measurements)

    for axis, (model, points) in zip(axes, group_items):
        draw_group(axis, model, points, hatch_by_tp, max_value)

    for axis in axes[len(group_items) :]:
        axis.set_visible(False)
    for row_index in range(row_count):
        axes[row_index * column_count].set_ylabel(
            "CUDA graph capture memory (GiB per GPU)"
        )

    legend_handles = [
        Patch(facecolor=TARGET_COLOR, edgecolor=EDGE_COLOR, label="Target verify"),
        Patch(facecolor=DRAFT_COLOR, edgecolor=EDGE_COLOR, label="Draft decode"),
    ]
    if len(all_tp_sizes) > 1:
        legend_handles.extend(
            Patch(
                facecolor="white",
                edgecolor=EDGE_COLOR,
                hatch=hatch_by_tp[tp_size],
                label=f"TP{tp_size}",
            )
            for tp_size in all_tp_sizes
        )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(len(legend_handles), 6),
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot grouped DFlash target/draft CUDA graph memory bars."
    )
    parser.add_argument("results_csv", type=Path, help="Benchmark results.csv path.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: cuda_graph_memory.png beside the CSV).",
    )
    parser.add_argument(
        "--tp-sizes",
        nargs="+",
        type=positive_cli_int,
        default=None,
        metavar="N",
        help="Only plot these TP sizes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = args.results_csv.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_path.with_name("cuda_graph_memory.png")
    )
    try:
        measurements = select_tp_sizes(load_measurements(input_path), args.tp_sizes)
        render_chart(measurements, output_path)
    except PlotError as exc:
        parser.error(str(exc))
    print(f"Chart: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
