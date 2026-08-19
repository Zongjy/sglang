#!/usr/bin/env python3
"""Decision table and plot for the unified PP partition objective.

One row per candidate boundary: stage times and pp_cycle_time from the cost
model, K / BS_max from the capacity model.  After validation, measured
columns (meas_K, accept) are appended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # importable both as benchmark.pp_spec.* and as a flat script dir
    from benchmark.pp_spec.partition_optimizer import OptimizationResult
except ImportError:  # pragma: no cover - depends on sys.path setup
    from partition_optimizer import OptimizationResult


@dataclass
class MeasPoint:
    """Measured validation numbers for one partition."""

    k: int | None = None
    bs_max: int | None = None
    accept_len: float | None = None
    mamba_ratio: float | None = None


def uniform_partition(pp_size: int, total_layers: int) -> tuple[int, ...]:
    """The even split SGLang launches by default (``get_pp_indices``: the
    last ``total_layers % pp_size`` stages get one extra layer)."""
    base, extra = divmod(total_layers, pp_size)
    return tuple(
        base + 1 if rank >= pp_size - extra else base for rank in range(pp_size)
    )


@dataclass
class Row:
    partition: tuple[int, ...]
    composition: tuple[str, ...]  # per-stage "GGg+AAa" layer-type counts
    stage_ms: tuple[float, ...]
    pp_cycle_time_ms: float
    k_per_rank: tuple[int | None, ...]
    bs_max: int | None
    mamba_ratio: float | None
    bottleneck_rank: int
    capacity_binding_rank: int | None
    annotations: list[str] = field(default_factory=list)
    meas: MeasPoint | None = None


def build_rows(
    result: OptimizationResult,
    composition_of: Any = None,
    k_of: Any = None,
    meas: Mapping[tuple[int, ...], MeasPoint] | None = None,
) -> list[Row]:
    """Build table rows from an OptimizationResult.

    ``composition_of(partition)`` -> per-stage "27G+9F" strings (from the
    capacity model's layer types); ``k_of(partition)`` -> per-rank K values.
    Both optional so the table degrades gracefully without a capacity model.
    """
    rows: list[Row] = []
    seen: dict[tuple[int, ...], Row] = {}
    items = list(result.candidates)
    if result.current.partition not in {item.partition for item in items}:
        items.append(result.current)
    uniform = uniform_partition(
        len(result.current_partition), sum(result.current_partition)
    )
    for item in items:
        partition = item.partition
        annotations: list[str] = []
        if partition == uniform:
            annotations.append("uniform")
        if partition == result.best.partition:
            annotations.append("recommended")
        composition = (
            tuple(composition_of(partition))
            if composition_of is not None
            else ("",) * len(partition)
        )
        k_per_rank = (
            tuple(k_of(partition))
            if k_of is not None
            else (None,) * len(partition)
        )
        row = Row(
            partition=partition,
            composition=composition,
            stage_ms=item.stage_ms,
            pp_cycle_time_ms=item.cadence_at_capacity_ms or item.bottleneck_ms,
            k_per_rank=k_per_rank,
            bs_max=item.bs_max,
            mamba_ratio=item.mamba_ratio,
            bottleneck_rank=item.bottleneck_rank,
            capacity_binding_rank=item.capacity_binding_rank,
            annotations=annotations,
            meas=(meas or {}).get(partition),
        )
        seen[partition] = row
    rows = sorted(seen.values(), key=lambda row: row.partition)
    return rows


def render_table(rows: Sequence[Row], with_meas: bool = False) -> str:
    """Render the decision table as text (stdout + analysis.txt)."""
    has_ratio = any(row.mamba_ratio for row in rows)
    header = (
        f"{'boundary':<9} {'layers(G+F)':<17} "
        f"{'stage0_ms':>9} {'stage1_ms':>9} {'pp_cycle_time':>13} "
        f"{'K0':>4} {'K1':>4} {'BS':>3}"
    )
    if has_ratio:
        header += f" {'ratio':>5}"
    if with_meas:
        header += f" {'meas_K':>6} {'accept':>6}"
    header += "  marks"
    lines = [header, "-" * len(header)]
    for row in rows:
        boundary = f"{row.partition[0]}/{sum(row.partition) - row.partition[0]}"
        stage_cells = []
        for rank, value in enumerate(row.stage_ms[:2]):
            cell = f"{value:9.2f}"
            if rank == row.bottleneck_rank:
                cell += "\u2020"  # compute bottleneck stage
            else:
                cell += " "
            stage_cells.append(cell)
        k_cells = []
        for rank, k in enumerate(row.k_per_rank[:2]):
            text = "-" if k is None else str(k)
            cell = f"{text:>4}"
            if rank == row.capacity_binding_rank and k is not None:
                cell += "\u2021"  # capacity binding stage
            else:
                cell += " "
            k_cells.append(cell)
        bs = "-" if row.bs_max is None else str(row.bs_max)
        line = (
            f"{boundary:<9} {'/'.join(row.composition):<17} "
            f"{stage_cells[0]} {stage_cells[1] if len(stage_cells) > 1 else ' ':>9} "
            f"{row.pp_cycle_time_ms:>13.2f} "
            f"{k_cells[0]}{k_cells[1] if len(k_cells) > 1 else ' ':>4} "
            f"{bs:>3}"
        )
        if has_ratio:
            line += f" {row.mamba_ratio:>5g}" if row.mamba_ratio else f" {'-':>5}"
        if with_meas:
            meas_k = "-" if row.meas is None or row.meas.k is None else str(row.meas.k)
            if (
                row.meas is not None
                and row.meas.k is not None
                and row.meas.mamba_ratio is not None
                and row.mamba_ratio is not None
                and abs(row.meas.mamba_ratio - row.mamba_ratio) > 1e-9
            ):
                # The measurement used a different ratio than this row's
                # prediction: annotate honestly instead of comparing K
                # numbers across ratios.
                meas_k = f"{row.meas.k}@{row.meas.mamba_ratio:g}"
            meas_accept = (
                "-"
                if row.meas is None or row.meas.accept_len is None
                else f"{row.meas.accept_len:.2f}"
            )
            line += f" {meas_k:>6} {meas_accept:>6}"
        line += "  " + ",".join(row.annotations)
        lines.append(line)
    lines.append("")
    lines.append("\u2020 compute bottleneck stage   \u2021 capacity binding stage")
    if with_meas and has_ratio:
        lines.append(
            "meas columns used each validation's own mamba ratio; 'K@r' "
            "marks measurements taken at a different ratio than the prediction"
        )
    return "\n".join(lines)


def render_plot(
    rows: Sequence[Row],
    output_path: Path,
    title: str = "PP partition: compute vs capacity",
) -> Path | None:
    """Render the boundary sweep plot; returns the PNG path or None.

    matplotlib is imported lazily with the Agg backend so the table still
    works when matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    ordered = sorted(rows, key=lambda row: row.partition[0])
    x = [row.partition[0] for row in ordered]
    stage0 = [row.stage_ms[0] for row in ordered]
    stage1 = [row.stage_ms[1] if len(row.stage_ms) > 1 else 0.0 for row in ordered]
    cycle = [row.pp_cycle_time_ms for row in ordered]
    bs_max = [row.bs_max or 0 for row in ordered]

    fig, ax_ms = plt.subplots(figsize=(11, 6.5))
    ax_ms.plot(x, stage0, lw=1.2, color="tab:blue", alpha=0.8, label="stage0 ms")
    ax_ms.plot(x, stage1, lw=1.2, color="tab:orange", alpha=0.8, label="stage1 ms")
    ax_ms.plot(x, cycle, lw=2.6, color="tab:red", label="pp_cycle_time = max(stage)")
    ax_ms.set_xlabel("boundary (PP0 layers)")
    ax_ms.set_ylabel("stage time (ms)")
    ax_ms.grid(alpha=0.25)

    ax_bs = ax_ms.twinx()
    ax_bs.step(x, bs_max, where="mid", lw=1.6, color="tab:green", label="BS_max")
    ax_bs.set_ylabel("BS_max (requests)", color="tab:green")
    ax_bs.tick_params(axis="y", labelcolor="tab:green")
    ax_bs.set_ylim(bottom=0)

    def boundary_of(row: Row) -> int:
        return row.partition[0]

    vlines = [
        ("uniform", "gray", ":"),
        ("recommended", "tab:purple", "--"),
    ]
    seen_labels = set()
    for row in ordered:
        for name, color, style in vlines:
            if name in row.annotations:
                label = name if name not in seen_labels else None
                seen_labels.add(name)
                ax_ms.axvline(boundary_of(row), color=color, ls=style, lw=1.4, label=label)

    handles1, labels1 = ax_ms.get_legend_handles_labels()
    handles2, labels2 = ax_bs.get_legend_handles_labels()
    ax_ms.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        fontsize=9,
        ncol=3,
    )
    ax_ms.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
