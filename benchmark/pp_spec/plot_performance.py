"""Plot throughput / TPOT / TTFT vs concurrency from results/summary.csv.

Usage: python plot_performance.py [path/to/summary.csv]

Expected columns:
    label,max_concurrency,output_throughput_tok_s,ttft_p50_s,ttft_p99_s,tpot_p99_s

Label conventions: label containing "tp" -> TP; "uniform" -> PP(uniform); else PP(auto).
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

# label -> matplotlib style
def series_style(label: str):
    if "tp" in label.lower():
        return dict(color="tab:green", linestyle="-", marker="s")
    if "uniform" in label.lower():
        return dict(color="tab:blue", linestyle="--", marker="^")
    return dict(color="tab:blue", linestyle="-", marker="s")


def load_rows():
    by_label = defaultdict(list)
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "label",
            "max_concurrency",
            "output_throughput_tok_s",
            "ttft_p50_s",
            "tpot_p99_s",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{CSV_PATH} is missing required column(s): {sorted(missing)}"
            )
        for line_number, row in enumerate(reader, start=2):
            if not row.get("label"):
                continue
            r = {}
            for key, value in row.items():
                if not key or key == "label" or value is None or not value.strip():
                    continue
                try:
                    r[key] = float(value)
                except ValueError as exc:
                    raise ValueError(
                        f"{CSV_PATH}:{line_number}: invalid {key}={value!r}"
                    ) from exc
            r["label"] = row["label"]
            by_label[r["label"]].append(r)
    for series in by_label.values():
        series.sort(key=lambda r: r["max_concurrency"])
    return by_label


def main():
    by_label = load_rows()

    # one figure per metric; latency columns are in seconds -> ms
    for col, ylabel, outfile in [
        ("output_throughput_tok_s", "Throughput (tok/s)", "throughput.pdf"),
        ("tpot_p99_s", "TPOT p99 (ms)", "tpot.pdf"),
        ("ttft_p50_s", "TTFT p50 (ms)", "ttft.pdf"),
    ]:
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
        output_path = OUT_DIR / outfile
        fig.savefig(output_path)
        plt.close(fig)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
