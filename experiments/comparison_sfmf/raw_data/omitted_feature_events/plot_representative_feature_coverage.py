"""Plot one recorded feature-coverage case; no controller or replay is run.

Run with Python 3 and matplotlib. The default input is the adjacent
representative_cases.json. PNG, vector PDF, caption, and provenance JSON are
written alongside this script unless --output-prefix is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


SAMPLE_ID = "126_cc_sf_5_261_first_geometric_event"
CAPTION_LINES = [
    "Probabilities are frozen local-model estimates using nominal feature geometry and each state's planning covariance;",
    "they are not measured collision rates. The MF value evaluates the unrelaxed affine row at the SF solution, using the",
    "same nominal state, without MF reoptimization. Negative actual signed PSDF indicates geometric overlap. This is one case,",
    "not a guarantee that MF would prevent the event. Feature identities are held fixed from the nominal state to the SF solution.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path, default=directory / "representative_cases.json")
    parser.add_argument(
        "--output-prefix", type=Path,
        default=directory / "representative_feature_coverage_126mm_seed5_cycle261",
    )
    args = parser.parse_args()
    source_bytes = args.input.read_bytes()
    source = json.loads(source_bytes)
    matching = [record for record in source["cases"] if record["sample_id"] == SAMPLE_ID]
    if len(matching) != 1:
        raise ValueError(f"Expected exactly one {SAMPLE_ID}; found {len(matching)}")
    record = matching[0]
    nominal = record["nominal"]["fresh"]
    solution = record["solution"]["frozen_nominal_geometry"]
    if nominal["selected_slot"] != solution["selected_slot"]:
        raise ValueError("Selected feature identity changed in frozen evaluation")
    if nominal["worst_omitted_slot"] != solution["worst_omitted_slot"]:
        raise ValueError("This figure requires the same omitted feature at both states")
    if not record["nominal_mf_domain_active"]:
        raise ValueError("MF counterfactual requires an active nominal domain")

    epsilon = 0.1  # The experiment's common chance_epsilon.
    selected = [nominal["selected_probability"], solution["selected_probability"]]
    omitted = [nominal["worst_omitted_probability"], solution["worst_omitted_probability"]]
    sf_h_mm = 1000.0 * solution["sf_rti_affine_h_m"]
    mf_psi = solution["mf_rti_affine_quantile_without_slack"]
    actual_phi_mm = 1000.0 * record["actual_next"]["fresh"]["signed_psdf_m"]
    gap_mm = 1000.0 * nominal["worst_omitted_distance_gap_m"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(11.0, 5.7), facecolor="white")
    grid = fig.add_gridspec(1, 2, left=0.075, right=0.985, bottom=0.30,
                           top=0.79, wspace=0.24, width_ratios=(1, 1.11))
    ax = fig.add_subplot(grid[0, 0])
    table_ax = fig.add_subplot(grid[0, 1])
    blue, orange, ink, muted = "#176B97", "#D56B28", "#25313D", "#56616D"
    fig.suptitle("Representative feature-coverage failure", x=0.075, y=0.975,
                 ha="left", fontsize=17, fontweight="semibold", color=ink)
    fig.text(0.075, 0.912,
             "Robot width 126 mm  |  CC-SF, seed 5  |  cycle 261  |  26.1–26.2 s",
             ha="left", fontsize=10.5, color=muted)

    positions = [0, 1]
    width = 0.30
    selected_bars = ax.bar([x - width / 2 for x in positions], selected, width,
                           color=blue, label="SF-selected feature")
    omitted_bars = ax.bar([x + width / 2 for x in positions], omitted, width,
                          color=orange, label="Unselected adjacent feature")
    for bars in (selected_bars, omitted_bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.014,
                    f"{bar.get_height():.4f}", ha="center", va="bottom",
                    fontsize=9, color=ink)
    ax.axhline(epsilon, color="#A84348", linestyle=(0, (4, 3)), linewidth=1.3)
    ax.text(0.98, epsilon + 0.014, r"$\epsilon=0.10$",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=9.5, color="#A84348", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    ax.set_xlim(-0.52, 1.58)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(positions, ["Nominal state", "SF solution"])
    ax.set_ylabel("Local-model violation probability")
    ax.yaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_title("(a) Two fixed feature identities", loc="left", pad=16, color=ink)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E3E7EB", linewidth=0.7)
    ax.tick_params(axis="both", colors=muted, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C1C9D0")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.015, 1.035), frameon=False,
              fontsize=9, handlelength=1.3, labelspacing=0.45)
    ax.text(0, -0.22, f"Nominal distance gap between these features: {gap_mm:.5f} mm",
            transform=ax.transAxes, fontsize=9, color=muted, ha="left")

    table_ax.set_axis_off()
    table_ax.set_title("(b) Constraint and geometry checks", loc="left", pad=16, color=ink)
    rows = [
        ["SF affine constraint\nat the SF solution", f"{sf_h_mm:+.3f} mm", "Passes\n$h\\geq 0$"],
        ["MF affine constraint\nat the same SF solution", f"{mf_psi:+.3f}\nquantile units", "Violates\n$\\psi\\geq 0$"],
        ["Actual next pose\nsigned PSDF", f"{actual_phi_mm:+.3f} mm", "Overlap\n$\\phi<0$"],
    ]
    table = table_ax.table(cellText=rows, colLabels=["Diagnostic", "Value", "Result"],
                          colWidths=[0.52, 0.25, 0.23], cellLoc="left", colLoc="left",
                          bbox=[0, 0.10, 1, 0.86])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row, col), cell in table.get_celld().items():
        cell.PAD = 0.08
        cell.set_linewidth(0.65)
        cell.set_edgecolor("#DFE5EA")
        cell.get_text().set_color(ink)
        if row == 0:
            cell.set_facecolor("#EAF0F4")
            cell.get_text().set_fontweight("semibold")
            cell.set_height(cell.get_height() * 0.6)
        else:
            cell.set_facecolor("white" if row % 2 else "#F8FAFC")
            if col == 1:
                cell.get_text().set_fontweight("semibold")
                cell.get_text().set_color(blue if row == 1 else "#A84348")
    table_ax.text(0, -0.045, "MF: counterfactual, unrelaxed row; no reoptimization.",
                  transform=table_ax.transAxes, fontsize=9, color=muted, ha="left")

    for index, line in enumerate(CAPTION_LINES):
        fig.text(0.075, 0.166 - index * 0.031, line, ha="left", va="top",
                 fontsize=8.7, color=muted)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=240)
    fig.savefig(prefix.with_suffix(".pdf"), metadata={
        "Title": "Representative feature-coverage failure: CC-SF 126 mm seed 5",
        "Subject": "Frozen local-model probabilities and counterfactual MF constraint",
    })
    plt.close(fig)
    prefix.with_suffix(".caption.txt").write_text(" ".join(CAPTION_LINES) + "\n", encoding="utf-8")
    metadata = {
        "sample_id": SAMPLE_ID,
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "epsilon": epsilon,
        "selected_feature_slot": nominal["selected_slot"],
        "omitted_feature_slot": nominal["worst_omitted_slot"],
        "selected_probabilities_nominal_then_solution": selected,
        "omitted_probabilities_nominal_then_solution": omitted,
        "sf_affine_h_mm": sf_h_mm,
        "mf_counterfactual_unrelaxed_affine_quantile": mf_psi,
        "actual_next_signed_psdf_mm": actual_phi_mm,
        "nominal_feature_distance_gap_mm": gap_mm,
    }
    prefix.with_suffix(".provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(prefix.with_suffix(".png"))
    print(prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
