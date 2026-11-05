"""Plot archived CC-SF/CC-MF constraint residuals without running a controller.

Use the original experiment's Python environment (NumPy, matplotlib, torch,
CasADi, PyYAML). The archived PSDF/covariance algebra is evaluated offline;
no OCP, acados solver, simulation, or benchmark is initialized.

Example:
    python plot_constraint_margins.py --width-mm 70 --seed 0 --stage 10
    python plot_constraint_margins.py --width-mm 125 --successful-pairs

The default stage is the configured robust horizon endpoint (1 s ahead for
this experiment). Progress is the current cycle's path_s, not distance traveled
or the selected predicted state's progress. Each panel has its own units.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from utils.offline_cc_margin import load_trial


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = ROOT / "benchmark_runs" / (
    "rectangle_width070_130mm_step005_10trials_20261103"
)
COLORS = {"nominal": "#087fbb", "postsolve_affine": "#d84c36",
          "postsolve_fresh": "#343434"}


def finite_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "min": float(values.min()),
            "median": float(np.median(values)), "max": float(values.max())}


def make_summary(trial, mask):
    """Report native-unit residuals; never compare cross-unit amplitudes."""
    nominal = np.asarray(trial["nominal"], float)
    affine = np.asarray(trial["postsolve_affine"], float)
    fresh = np.asarray(trial["postsolve_fresh"], float)
    valid = mask & np.asarray(trial["solver_success"], bool)
    result = {"method": trial["method"], "stage": trial["stage"],
              "seed": trial["metadata"]["seed"],
              "native_unit": "m" if trial["method"] == "cc_sf" else "standard_normal_quantile",
              "cycles_in_view": int(mask.sum()),
              "successful_cycles_in_view": int(valid.sum())}
    result["safe_stop_cycles_in_view"] = int(np.sum(mask & trial["safe_stop"]))
    # Absolute feasibility tolerances are explicitly different native units.
    tolerance = 1e-6 if trial["method"] == "cc_sf" else max(
        1e-4, 10 * (trial["metadata"].get("mf_nominal_max_abs_error") or 0.0))
    result["negative_residual_tolerance_native"] = tolerance
    for key, values in (("nominal", nominal), ("postsolve_affine", affine),
                        ("postsolve_fresh", fresh)):
        sample = values[valid & np.isfinite(values)]
        result[key] = finite_summary(sample)
        result[key]["negative_count"] = int(np.sum(sample < -tolerance))
        result[key]["nonnegative_count_with_tolerance"] = int(np.sum(sample >= -tolerance))
    paired = valid & np.isfinite(affine) & np.isfinite(fresh)
    result["fresh_minus_affine"] = finite_summary((fresh-affine)[paired])
    result["paired_cycles"] = int(paired.sum())
    result["affine_feasible_but_fresh_negative_count"] = int(np.sum(
        paired & (affine >= -tolerance) & (fresh < -tolerance)))
    slack = np.asarray(trial["slack"], float)
    result["slack"] = finite_summary(slack[valid])
    result["affine_plus_slack"] = finite_summary((affine + slack)[valid])
    if trial["method"] == "cc_mf":
        result["nominal_mf_inactive_cycles_in_view"] = int(np.sum(mask & ~trial["nominal_active"]))
        result["fresh_margin_unavailable_successful_cycles"] = int(np.sum(valid & ~np.isfinite(fresh)))
    return result


def save_rows(path, trials):
    fields = ["method", "seed", "stage", "cycle", "time_s", "current_s_m", "predicted_s_stage1_m",
              "selected_predicted_s_m",
              "solver_success", "solve_mode", "safe_stop", "nominal_active", "fresh_active", "nominal_valid", "nominal", "postsolve_affine",
              "postsolve_fresh", "slack", "native_unit"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trial in trials:
            for i, progress in enumerate(trial["s"]):
                row = {"method": trial["method"], "stage": trial["stage"], "cycle": i,
                       "seed": trial["metadata"]["seed"],
                       "time_s": float(trial["time"][i]), "current_s_m": float(progress),
                       "predicted_s_stage1_m": float(trial["predicted_s"][i]),
                       "selected_predicted_s_m": float(trial["selected_predicted_s"][i]),
                       "solver_success": bool(trial["solver_success"][i]),
                       "solve_mode": str(trial["solve_mode"][i]),
                       "safe_stop": bool(trial["safe_stop"][i]),
                       "nominal_active": bool(trial["nominal_active"][i]),
                       "fresh_active": bool(trial["fresh_active"][i]),
                       "nominal_valid": bool(trial["nominal_valid"][i]),
                       "native_unit": "m" if trial["method"] == "cc_sf" else "standard_normal_quantile"}
                for key in ("nominal", "postsolve_affine", "postsolve_fresh", "slack"):
                    value = float(trial[key][i])
                    row[key] = value if np.isfinite(value) else ""
                writer.writerow(row)


def plot(trials, output, width_mm, seeds, stage, s_range, dpi, successful_pairs=False):
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                         "axes.labelsize": 9, "axes.titlesize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none"})
    columns = len(seeds)
    fig, axes = plt.subplots(2, columns, figsize=(7.05, 5.7), sharex=True,
                             sharey="row", squeeze=False)
    fig.subplots_adjust(left=.10, right=.96, bottom=.18, top=.73,
                        hspace=.47, wspace=.12)
    summaries = []
    for trial in trials:
        row = 0 if trial["method"] == "cc_sf" else 1
        column = seeds.index(trial["metadata"]["seed"])
        ax = axes[row, column]
        progress = np.asarray(trial["s"], float)
        mask = np.ones(len(progress), bool)
        if s_range is not None:
            mask &= (progress >= s_range[0]) & (progress <= s_range[1])
        if not mask.any():
            raise ValueError(f"No {trial['method']} cycles in requested s range")
        success = np.asarray(trial["solver_success"], bool)
        scale = 1000.0 if trial["method"] == "cc_sf" else 1.0
        for key, style, linewidth, zorder in (
            ("nominal", "-", 1.05, 4),
            ("postsolve_affine", "-", 1.0, 5),
            ("postsolve_fresh", "--", .95, 6),
        ):
            values = np.asarray(trial[key], float) * scale
            # Keep chronological order and preserve missing/failed-cycle gaps.
            values = np.where(mask & success, values, np.nan)
            ax.plot(progress, values, style, color=COLORS[key], lw=linewidth,
                    zorder=zorder)
        for event_s in progress[mask & trial["safe_stop"]]:
            ax.axvline(event_s, color="#aa7a18", lw=.8, ls=":", zorder=2)
            ax.plot(event_s, .98, marker="v", markersize=4, color="#aa7a18",
                    transform=ax.get_xaxis_transform(), clip_on=False)
        if trial["method"] == "cc_mf":
            inactive_s = progress[mask & ~trial["nominal_active"]]
            ax.plot(inactive_s, np.full(len(inactive_s), .025), linestyle="none",
                    marker="|", markersize=4, color="#888888", alpha=.8,
                    transform=ax.get_xaxis_transform(), zorder=7)
        ax.axhline(0, color="#777777", lw=.75, zorder=3)
        ax.grid(axis="y", color="#e5e5e5", lw=.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(.6)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        method = "CC-SF" if row == 0 else "CC-MF"
        panel = row * columns + column
        stop_count = int(np.sum(mask & trial["safe_stop"]))
        status = trial["outcome"]["status"]
        ax.set_title(f"({chr(97+panel)}) {method} · seed {seeds[column]}\n"
                     f"{status}; safe stops in view: {stop_count}",
                     loc="left", pad=7, fontsize=8.5)
        if column == 0:
            ax.set_ylabel("Distance margin [mm]" if row == 0 else "Inverse-CDF margin [–]")
        summaries.append(make_summary(trial, mask))
    lo = min(float(np.min(t["s"])) for t in trials)
    hi = max(float(np.max(t["s"])) for t in trials)
    for ax in axes[-1]:
        ax.set_xlim(s_range if s_range else (min(0, lo), hi))
        ax.set_xlabel(r"Current progress $s$ [m]")
    handles = [Line2D([], [], color=COLORS["nominal"], lw=1.2),
               Line2D([], [], color=COLORS["postsolve_affine"], lw=1.2),
               Line2D([], [], color=COLORS["postsolve_fresh"], lw=1.1, ls="--"),
               Line2D([], [], color="#aa7a18", lw=.8, ls=":", marker="v", markersize=4),
               Line2D([], [], color="#888888", ls="none", marker="|", markersize=6)]
    fig.legend(handles, ["Nominal at initial guess", "QP constraint at solution",
                         "Geometry re-evaluation at solution", "Safe stop (progress location)",
                         "MF inactive at nominal (rug)"],
               loc="upper center", bbox_to_anchor=(.55, .92), frameon=False,
               ncol=2, handlelength=2, columnspacing=1.5, fontsize=8)
    cohort = "All matched successful seeds" if successful_pairs else "Selected seed"
    fig.suptitle(f"Robot width {width_mm} mm · prediction stage k = {stage}\n{cohort}: "
                 + ", ".join(map(str, seeds)), y=.985, fontsize=10)
    fig.text(.10, .025,
             "Offline reconstruction. Gaps retain unavailable/failed cycles; MF domain-inactive rows are not feasible samples.\n"
             "QP constraints use frozen nominal geometry; geometry is refreshed for the dashed curve. MF slack is excluded.\n"
             "Zero denotes the unrelaxed chance boundary; these matched successful trials omit the failed trials.",
             fontsize=7.1, color="#444444", va="bottom")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=dpi, facecolor="white")
    plt.close(fig)
    return summaries


def width_audit(experiment, width_mm):
    """Keep the full trial population separate from the successful-pair figure."""
    rows = []
    for method in ("cc_sf", "cc_mf"):
        for folder in sorted((experiment / f"width_{width_mm:03d}mm" / method).glob("seed_*")):
            outcome = json.loads((folder / "outcome.json").read_text())
            with (folder / "cycles.csv").open(newline="") as handle:
                cycles = list(csv.DictReader(handle))
            stops = sum(row["solve_mode"].startswith("safe_stop") for row in cycles)
            rows.append({"method": method, "seed": int(folder.name.split("_")[1]),
                         "status": outcome["status"], "cycles": len(cycles),
                         "safe_stop_cycles": stops, "source": str(folder)})
    summary = []
    for method in ("cc_sf", "cc_mf"):
        all_rows = [row for row in rows if row["method"] == method]
        successful = [row for row in all_rows if row["status"] == "success"]
        summary.append({"method": method, "trials": len(all_rows),
                        "successful_trials": len(successful),
                        "safe_stop_cycles_all_trials": sum(r["safe_stop_cycles"] for r in all_rows),
                        "control_cycles_all_trials": sum(r["cycles"] for r in all_rows),
                        "safe_stop_cycles_successful_trials": sum(r["safe_stop_cycles"] for r in successful)})
    return {"rows": rows, "summary": summary,
            "count_definition": "Safe-stop control cycles, not independent stop episodes."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--width-mm", type=int, default=70)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--successful-pairs", action="store_true",
                        help="Plot every seed where both SF and MF trials succeeded.")
    parser.add_argument("--stage", type=int, default=10)
    parser.add_argument("--s-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/paper_constraint_margins")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    if args.s_range and args.s_range[0] >= args.s_range[1]:
        parser.error("--s-range requires MIN < MAX")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = width_audit(args.experiment.resolve(), args.width_mm)
    seeds = [args.seed]
    if args.successful_pairs:
        successful_sets = [{r["seed"] for r in audit["rows"]
                            if r["method"] == method and r["status"] == "success"}
                           for method in ("cc_sf", "cc_mf")]
        seeds = sorted(set.intersection(*successful_sets))
        if not seeds:
            parser.error("No matched successful seeds exist at the requested width")
    trials = [load_trial(args.experiment.resolve(), method, args.width_mm, seed, args.stage)
              for method in ("cc_sf", "cc_mf") for seed in seeds]
    cohort_name = "success_pairs" if args.successful_pairs else f"seed{args.seed}"
    stem = f"constraint_margins_width{args.width_mm:03d}_{cohort_name}_stage{args.stage:02d}"
    if args.s_range:
        stem += (f"_s{args.s_range[0]:g}_{args.s_range[1]:g}").replace(".", "p")
    output = args.output_dir / stem
    save_rows(output.with_suffix(".csv"), trials)
    summaries = plot(trials, output, args.width_mm, seeds, args.stage, args.s_range,
                     args.dpi, args.successful_pairs)
    provenance = {"experiment": str(args.experiment.resolve()), "width_mm": args.width_mm,
                  "seeds": seeds, "stage": args.stage, "s_range": args.s_range,
                  "successful_pairs_only": args.successful_pairs, "width_population": audit,
                  "residual_formulas": {
                      "cc_sf": "phi - d_min - clip(gamma * sqrt(q.T @ Sigma @ q), floor, cap)",
                      "cc_mf": "-Phi_inverse(R) - Phi_inverse(1 - epsilon), R = sum_j Phi((d_min - d_j) / sigma_j)",
                      "mf_extensions": "R <= float64.tiny is tail-clipped; R > 0.5 uses the archived linear extension.",
                      "qp": "The nominal row's first-order approximation evaluated at the saved postsolve state, before MF slack.",
                  },
                  "summary": summaries,
                  "trials": [trial["metadata"] for trial in trials],
                  "interpretation": [
                      "Success-conditioned margin plots do not represent failed trials; see the full width population counts.",
                      "SF and MF have different residual definitions and units; amplitudes are not comparable.",
                      "Fresh nonlinear residuals reevaluate geometry at the solved state; QP rows freeze nominal geometry.",
                      "Nominal and QP residuals are offline reconstructions where validated; no synthetic feature-switch markers.",
                  ]}
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"figure": str(output.with_suffix(".png")), "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
