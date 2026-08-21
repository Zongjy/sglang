#!/usr/bin/env python3
"""Small report/table/plot helpers for the prefix-boundary selector."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:
    from benchmark.pp_spec.partition_optimizer import OptimizationResult
except ImportError:  # pragma: no cover - flat script execution
    from partition_optimizer import OptimizationResult


@dataclass
class Row:
    partition: tuple[int, ...]
    composition: tuple[str, ...]
    stage_ms: tuple[float, ...]
    pp_cycle_time_ms: float
    feasible: bool | None
    memory_margin_gib: tuple[float, ...]
    memory_capacity: int | None
    bottleneck_rank: int
    annotations: list[str] = field(default_factory=list)


def uniform_partition(pp_size: int, total_layers: int) -> tuple[int, ...]:
    """The ordinary even split used as a visual reference."""
    base, extra = divmod(total_layers, pp_size)
    return tuple(
        base + int(rank >= pp_size - extra) for rank in range(pp_size)
    )


def build_rows(
    result: OptimizationResult,
    composition_of: Any = None,
) -> list[Row]:
    """Build one row per candidate plus the current partition."""
    items = list(result.candidates)
    if result.current.partition not in {item.partition for item in items}:
        items.append(result.current)
    uniform = uniform_partition(
        len(result.current_partition), sum(result.current_partition)
    )
    rows: dict[tuple[int, ...], Row] = {}
    for item in items:
        partition = item.partition
        annotations: list[str] = []
        if partition == uniform:
            annotations.append("uniform")
        if partition == result.selected.partition:
            annotations.append("recommended")
        composition = (
            tuple(composition_of(partition))
            if composition_of is not None
            else ("",) * len(partition)
        )
        rows[partition] = Row(
            partition=partition,
            composition=composition,
            stage_ms=item.stage_ms,
            pp_cycle_time_ms=item.cycle_time_ms,
            feasible=item.target_feasible,
            memory_margin_gib=item.memory_margin_gib,
            memory_capacity=item.memory_capacity,
            bottleneck_rank=item.bottleneck_rank,
            annotations=annotations,
        )
    return sorted(rows.values(), key=lambda row: row.partition)


def render_table(rows: Sequence[Row]) -> str:
    """Render a compact cycle-time decision table."""
    max_stages = max((len(row.stage_ms) for row in rows), default=0)
    header = f"{'partition':<18} {'composition':<30} {'cycle_ms':>10} {'feasible':>9}"
    if any(row.memory_capacity is not None for row in rows):
        header += f" {'C_mem':>8}"
    for rank in range(max_stages):
        header += f" {('s' + str(rank) + '_ms'):>9}"
    header += "  marks"
    lines = [header, "-" * len(header)]
    for row in rows:
        composition = "/".join(row.composition)
        feasible = "-" if row.feasible is None else ("yes" if row.feasible else "no")
        line = (
            f"{','.join(map(str, row.partition)):<18} {composition:<30} "
            f"{row.pp_cycle_time_ms:>10.3f} {feasible:>9}"
        )
        if any(item.memory_capacity is not None for item in rows):
            capacity = "-" if row.memory_capacity is None else row.memory_capacity
            line += f" {capacity:>8}"
        for rank in range(max_stages):
            value = row.stage_ms[rank] if rank < len(row.stage_ms) else None
            marker = "†" if rank == row.bottleneck_rank else " "
            line += f" {'-' if value is None else f'{value:.2f}' + marker:>9}"
        line += "  " + ",".join(row.annotations)
        lines.append(line)
    lines.append("")
    lines.append("† cycle bottleneck stage")
    lines.append(
        "capacity refines only the latency-equivalent range; it is not mixed into cycle_ms"
    )
    return "\n".join(lines)


def render_plot(
    rows: Sequence[Row],
    output_path: Path,
    title: str = "PP prefix-boundary cycle time",
) -> Path | None:
    """Render cycle and stage service times; returns None without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: row.partition[0])
    x = [row.partition[0] for row in ordered]
    cycle = [row.pp_cycle_time_ms for row in ordered]
    fig, ax = plt.subplots(figsize=(11, 6.0))
    ax.plot(x, cycle, lw=2.5, color="tab:red", label="cycle=max(stage)")
    max_stages = max(len(row.stage_ms) for row in ordered)
    for rank in range(max_stages):
        values = [
            row.stage_ms[rank] if rank < len(row.stage_ms) else float("nan")
            for row in ordered
        ]
        ax.plot(x, values, lw=1.1, alpha=0.65, label=f"stage {rank}")
    for row in ordered:
        if "uniform" in row.annotations:
            ax.axvline(row.partition[0], color="gray", ls=":", lw=1.2, label="uniform")
        if "recommended" in row.annotations:
            ax.axvline(
                row.partition[0], color="tab:purple", ls="--", lw=1.4, label="recommended"
            )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best", fontsize=9)
    ax.set_xlabel("l (layers on each of the first P-1 stages)")
    ax.set_ylabel("service / cycle time (ms)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
