"""Plot failure modes using the classified DoN sweep totals."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

OUT = Path(__file__).resolve().parent
MODES = (
    ("feature_switching", "Feature-switching", "#D48143", ""),
    ("feature_persistent", "Feature-persistent", "#5196B4", "///"),
    ("progress_stagnation", "Progress stagnation", "#B4CAA8", "..."),
    ("other_failures", "Other failures", "#D5D5D5", "\\\\"),
)
INK = "#25292E"
CONTROLLERS = {"cc_sf": "CC-SF", "cc_mf": "CC-MF"}


def load_counts():
    """Check that all sweep bins exist and that the stacks reconcile."""
    bins = {method: {} for method in CONTROLLERS}
    count_fields = ["completed_trials", "successful_trials", "directed_switching"]
    count_fields += [mode[0] for mode in MODES]
    with (OUT / "failure_modes_by_don.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            method_bins = bins[row["controller"]]
            don = float(row["don"])
            if don in method_bins:
                raise ValueError(f"Duplicate controller and DoN: {row['controller']}, {don}")
            counts = {field: int(row[field]) for field in count_fields}
            failures = sum(counts[field] for field, *_ in MODES)
            if min(counts.values()) < 0 or counts["completed_trials"] != 50:
                raise ValueError(f"Unexpected trial counts: {row}")
            if failures + counts["successful_trials"] != counts["completed_trials"]:
                raise ValueError(f"Outcomes do not add to the denominator: {row}")
            if counts["directed_switching"] > counts["feature_switching"]:
                raise ValueError(f"Directed switches exceed total switching: {row}")
            method_bins[don] = counts
    dons = [index / 100 for index in range(70, 88)]
    if any(sorted(method_bins) != dons for method_bins in bins.values()):
        raise ValueError("Expected one row per controller and DoN from 0.70 to 0.87")
    return dons, bins


def main():
    dons, bins = load_counts()
    config = {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#858A90",
        "axes.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "hatch.linewidth": 0.35,
        "pdf.fonttype": 42,
    }
    with plt.rc_context(config):
        fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.25), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.118, right=0.976, bottom=0.192, top=0.766, hspace=0.34)
        x = np.arange(len(dons))
        for index, (ax, (method, name)) in enumerate(zip(axes, CONTROLLERS.items())):
            rows = [bins[method][don] for don in dons]
            denominator = np.array([row["completed_trials"] for row in rows])
            bottom = np.zeros(len(dons))
            for key, _, color, hatch in MODES:
                values = 100 * np.array([row[key] for row in rows]) / denominator
                nonzero = values > 0
                bars = ax.bar(
                    x[nonzero], values[nonzero], bottom=bottom[nonzero], width=0.66,
                    color=color, edgecolor=INK, linewidth=0.35, hatch=hatch, zorder=3,
                )
                np.testing.assert_allclose(
                    [bar.get_height() for bar in bars], values[nonzero],
                )
                bottom += values
            directed = 100 * np.array([row["directed_switching"] for row in rows]) / denominator
            has_directed = directed > 0
            ax.plot(x[has_directed], directed[has_directed], linestyle="none", marker="D",
                    markerfacecolor=INK, markeredgecolor="white", markeredgewidth=0.45,
                    markersize=4.5, zorder=4)
            ax.set_title(f"({chr(97 + index)})  {name}", loc="left", fontsize=9.5,
                         fontweight="bold", pad=6)
            ax.set_xlim(-0.55, len(x) - 0.45)
            ax.set_ylim(0, 105)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.tick_params(length=2.5, pad=4, labelsize=8)
            ax.grid(axis="y", color="#D9DEE2", linewidth=0.55)
            ax.set_axisbelow(True)

        axes[-1].set_xticks(x, [f"{don:.2f}" for don in dons])
        axes[-1].set_xlabel("Degree of narrowness (DoN)", labelpad=7)
        fig.supylabel("Share of all trials", x=0.025, y=0.483, fontsize=9)
        fig.text(0.118, 0.967, "Failure modes across corridor narrowness",
                 va="top", fontsize=12, fontweight="bold")
        fig.text(0.118, 0.918, "Rectangle maze benchmark, 50 trials per controller and DoN",
                 va="top", fontsize=8.4, color="#515962")
        handles = [Patch(facecolor=color, edgecolor=INK, linewidth=0.35, hatch=hatch, label=label)
                   for _, label, color, hatch in MODES]
        handles.append(Line2D([], [], linestyle="none", marker="D", color=INK, markersize=4.5,
                              label="Directed switching subset"))
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.113, 0.882),
                   ncols=3, fontsize=8.0, handletextpad=0.45, handlelength=1.45,
                   columnspacing=2.0, frameon=False, borderaxespad=0)
        fig.text(0.118, 0.085,
                 "Switching includes one within-obstacle feature change.\n"
                 "Diamonds mark the subset where the previous obstacle's clearance increased and stayed\n"
                 "at or above the 0.05 mm guard while a different obstacle entered it on the terminal transition.",
                 va="top", fontsize=7.3, color="#515962")

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        artists = [*fig.texts, *fig.legends]
        for ax in axes:
            artists.extend([ax.xaxis.label, ax.title, *ax.get_xticklabels(), *ax.get_yticklabels()])
        for artist in artists:
            extent = artist.get_window_extent(renderer)
            if not (fig.bbox.contains(extent.x0, extent.y0) and
                    fig.bbox.contains(extent.x1, extent.y1)):
                raise ValueError(f"Artist outside canvas: {artist!r}")
        for suffix in ("pdf", "png"):
            fig.savefig(OUT / f"failure_modes_by_don.{suffix}", dpi=300, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
