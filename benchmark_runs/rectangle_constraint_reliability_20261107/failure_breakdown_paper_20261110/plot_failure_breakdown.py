"""Plot paired, 100% stacked outcomes for the paper's DoN comparison."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

OUT = Path(__file__).resolve().parent
DONS = tuple(index / 100 for index in range(80, 88))
METHODS = ("cc_sf", "cc_mf")
OUTCOMES = (
    ("feature_switching", "Feature-switching", "#D48143", ""),
    ("feature_persistent", "Feature-persistent", "#5196B4", "///"),
    ("progress_stagnation", "Progress stagnation", "#B4CAA8", "..."),
    ("other_failures", "Other failures", "#D5D5D5", "\\\\"),
    ("successful_trials", "Success", "#EDF0F1", ""),
)
INK = "#25292E"
EDGE = "#686F76"


def load_counts():
    with (OUT / "failure_breakdown.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_group = {(float(row["don"]), row["controller"]): row for row in rows}
    groups = [(don, method) for don in DONS for method in METHODS]
    if len(rows) != len(groups) or set(by_group) != set(groups):
        raise ValueError("Expected one SF and one MF row for each DoN from 0.80 to 0.87")

    sample_sizes = []
    outcomes = []
    for don, method in groups:
        row = by_group[don, method]
        counts = [int(row[field]) for field, *_ in OUTCOMES]
        n = int(row["completed_trials"])
        rerun = method == "cc_mf" and don <= 0.83
        expected_n = 5 if rerun else 50
        expected_cohort = "random_seed_rerun" if rerun else "original_sweep"
        if row["cohort"] != expected_cohort:
            raise ValueError(f"Unexpected cohort: {row}")
        if n != expected_n or min(counts) < 0 or sum(counts) != n:
            raise ValueError(f"Outcome counts or denominator do not reconcile: {row}")
        sample_sizes.append(n)
        outcomes.append(counts)
    return np.array(sample_sizes), np.array(outcomes)


def main():
    sample_sizes, counts = load_counts()
    percentages = 100 * counts / sample_sizes[:, None]
    np.testing.assert_allclose(percentages.sum(axis=1), 100)
    centers = np.arange(len(DONS))
    x = (centers[:, None] + [-0.166, 0.166]).flatten()
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": EDGE,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "ytick.major.width": 0.6,
        "hatch.linewidth": 0.35,
        "pdf.fonttype": 42,
    }):
        fig, ax = plt.subplots(figsize=(7.10, 3.9))
        fig.subplots_adjust(left=0.105, right=0.985, bottom=0.258, top=0.775)
        bottom = np.zeros(len(x))
        handles = []
        for index, (_, label, color, hatch) in enumerate(OUTCOMES):
            heights = percentages[:, index]
            visible = heights > 0
            bars = ax.bar(
                x[visible], heights[visible], width=0.307, bottom=bottom[visible],
                color=color, edgecolor=EDGE, linewidth=0.54, hatch=hatch, zorder=3,
            )
            np.testing.assert_allclose([bar.get_height() for bar in bars], heights[visible])
            bottom += heights
            handles.append(Patch(facecolor=color, edgecolor=EDGE, linewidth=0.54,
                                 hatch=hatch, label=label))
        for location, n in zip(x, sample_sizes):
            if n == 5:
                ax.text(location, 101.7, "n=5", ha="center", va="bottom",
                        color=INK, fontsize=7.4, fontweight="bold")
        for location, don in zip(centers, DONS):
            ax.text(location, -0.116, f"{don:.2f}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, fontweight="bold")
        ax.set_xticks(x, ["SF", "MF"] * len(DONS))
        ax.tick_params(axis="x", labelsize=7.6, length=0, pad=4)
        ax.tick_params(axis="y", labelsize=8, length=3, pad=4)
        ax.set_xlim(-0.49, len(DONS) - 0.51)
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.set_ylabel("Share of trials", labelpad=7)
        ax.set_xlabel("Degree of narrowness (DoN)")
        ax.xaxis.set_label_coords(0.5, -0.205)
        ax.grid(axis="y", color="#DDE0E3", linewidth=0.55)
        ax.set_axisbelow(True)

        fig.text(0.105, 0.972, "Trial outcomes by degree of narrowness",
                 va="top", fontsize=12, fontweight="bold")
        # Legend columns read vertically. Success remains last in the stack and legend.
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.103, 0.901),
                   ncols=3, frameon=False, fontsize=8.2, handlelength=1.5,
                   handletextpad=0.5, columnspacing=2.2, borderaxespad=0)
        fig.text(0.105, 0.067,
                 "CC-MF at DoN 0.80–0.83 uses a separate rerun with new seeds (n = 5).\n"
                 "All other bars use the original sweep (n = 50 per controller and DoN).",
                 va="top", fontsize=7.7, color="#515962")

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        artists = [
            *fig.texts, *fig.legends, *ax.texts, ax.xaxis.label, ax.yaxis.label,
            *ax.get_xticklabels(), *ax.get_yticklabels(),
        ]
        for artist in artists:
            extent = artist.get_window_extent(renderer)
            if not (fig.bbox.contains(extent.x0, extent.y0) and
                    fig.bbox.contains(extent.x1, extent.y1)):
                raise ValueError(f"Artist outside canvas: {artist!r}")
        for suffix in ("pdf", "png"):
            fig.savefig(OUT / f"failure_breakdown.{suffix}", dpi=300, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
