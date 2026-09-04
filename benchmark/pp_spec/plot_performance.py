"""Plot throughput / TPOT / TTFT vs concurrency from results/summary.csv.

Usage: python plot_performance.py [path/to/summary.csv]

Expected columns:
    label,max_concurrency,output_throughput_tok_s,ttft_p50_s,ttft_p99_s,tpot_p99_s
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "results/summary.csv"
OUT_DIR = Path(CSV_PATH).resolve().parent

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "pdf.fonttype": 42,  # embed TrueType so text stays selectable/vector
})


def series_style(label: str):
    """Return a stable style for a result label.

    Check D-Cut before the generic TP / PP-auto checks: ``tp2_dcut_auto`` and
    ``pp2_auto_dcut_auto`` contain those substrings as well, but should stay
    visually distinguishable in the aggregate plots.
    """

    normalized = label.lower()
    if "dcut" in normalized:
        return dict(color="tab:purple", linestyle="--", marker="^")
    if "tp" in normalized:
        return dict(color="tab:green", linestyle="-", marker="s")
    if "dp" in normalized:
        return dict(color="tab:red", linestyle="-.", marker="D")
    if "uniform" in normalized:
        return dict(color="tab:blue", linestyle="-.", marker="^")
    if "auto" in normalized:
        return dict(color="tab:blue", linestyle="-", marker="s")
    return dict(color="tab:orange", linestyle="-", marker="o")


def load_rows():
    by_label = defaultdict(list)
    required_metrics = {
        "max_concurrency",
        "output_throughput_tok_s",
        "tpot_p99_s",
        "ttft_p50_s",
    }
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("label"):
                continue
            r = {
                key: float(value)
                for key, value in row.items()
                if key and key != "label" and value and value.strip()
            }
            r["label"] = row["label"]
            if (
                r.get("completed") != r.get("num_requests")
                or not required_metrics.issubset(r)
            ):
                print(
                    f"skipping incomplete result: {r['label']} "
                    f"C={r.get('max_concurrency', 'unknown')}",
                    file=sys.stderr,
                )
                continue
            by_label[r["label"]].append(r)
    for series in by_label.values():
        series.sort(key=lambda r: r["max_concurrency"])
    return by_label


def plot_metric(by_label, col: str, ylabel: str, output_path: Path):
    """Render one metric for a selected set of series."""

    if not by_label:
        return

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    scale = 1.0 if col == "output_throughput_tok_s" else 1000.0
    for label, series in by_label.items():
        ax.plot(
            [r["max_concurrency"] for r in series],
            [r[col] * scale for r in series],
            markersize=7,
            linewidth=2,
            label=label,
            **series_style(label),
        )
    ax.set_xscale("log")
    # label ticks with the actual concurrency values (4, 8, 16, ...) instead of 10^n
    conc = sorted({r["max_concurrency"] for series in by_label.values() for r in series})
    ax.set_xticks(conc)
    ax.set_xticklabels([int(c) if c == int(c) else c for c in conc])
    ax.minorticks_off()
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=1.0, edgecolor="black")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"saved {output_path}")


def main():
    by_label = load_rows()

    # one figure per metric; latency columns are in seconds -> ms
    metrics = [
        ("output_throughput_tok_s", "Throughput (tok/s)", "throughput.pdf"),
        ("tpot_p99_s", "TPOT p99 (ms)", "tpot.pdf"),
        ("ttft_p50_s", "TTFT p50 (ms)", "ttft.pdf"),
    ]
    for col, ylabel, outfile in metrics:
        plot_metric(by_label, col, ylabel, OUT_DIR / outfile)


if __name__ == "__main__":
    main()
