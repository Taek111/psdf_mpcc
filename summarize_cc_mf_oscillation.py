"""Summarize the recorded ablations and render reproducible scientific figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_cc_mf_oscillation import read_csv, column, write_csv


EXPERIMENTS = (
    ("baseline", "Baseline"),
    ("frozen", "Frozen MF covariance"),
    ("proximal1_pilot", "Proximal 1 (pilot)"),
    ("proximal001_pilot", "Proximal 0.01 (pilot)"),
    ("sigmafloor_pilot", "Initial sigma floor (pilot)"),
    ("penalty_pilot", "Input penalty x10 (pilot)"),
    ("trust_pilot", "Relative input trust box (pilot)"),
    ("candidate", "Proximal 0.003 + guard 1.5 mm"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path(
        "benchmark_runs/cc_mf_oscillation_20261030"))
    args = parser.parse_args()
    output = Path(str(args.prefix) + "_comparison")
    output.mkdir(exist_ok=True)
    summaries, rows = {}, []
    for suffix, label in EXPERIMENTS:
        directory = Path(str(args.prefix) + "_" + suffix)
        document = json.loads((directory / "oscillation_analysis/summary.json").read_text())
        summaries[suffix] = document
        aggregate, trials = document["aggregate"], document["trials"]
        covariance = read_csv(directory / "oscillation_analysis/covariance_audit_cycles.csv")
        ratios = np.array([float(row["stage1_heading_sigma_exact_over_predicted"])
                           for row in covariance
                           if row.get("successful_control_cycle") == "True"
                           and not row.get("audit_error")])
        rows.append({
            "experiment": suffix, "label": label,
            "trials": aggregate["completed_trials"],
            "success": sum(t["status"] == "success" for t in trials),
            "stuck": sum(t["failure_reason"].startswith("stuck") for t in trials),
            "stage0_stop": sum("safe_stop_stage0" in t["failure_reason"] for t in trials),
            "timeout": sum(t["status"] == "timeout" for t in trials),
            "other_failures": sum(t["status"] != "success"
                                  and not t["failure_reason"].startswith("stuck")
                                  and "safe_stop_stage0" not in t["failure_reason"]
                                  and t["status"] != "timeout" for t in trials),
            "sampled_penetration_trials": sum(
                t["geometry"]["signed_polygon_clearance_m"]["min"] < -1e-8 for t in trials),
            "minimum_signed_clearance_mm": min(
                t["geometry"]["signed_polygon_clearance_m"]["min"] for t in trials) * 1000,
            "max_consecutive_dual_reversal_pairs": max(
                t["whole_run"]["dual_reversal_longest_pairs"] for t in trials),
            "max_consecutive_dual_reversal_pairs_all_amplitudes": max(
                t["whole_run"]["dual_reversal_all_amplitudes_longest_pairs"] for t in trials),
            "v_reversals_per_s": aggregate["v_reversals_per_s"],
            "omega_reversals_per_s": aggregate["omega_reversals_per_s"],
            "heading_sigma_exact_over_predicted_median": float(np.median(ratios)),
            "heading_sigma_exact_over_predicted_p95": float(np.quantile(ratios, .95)),
            "solver_mean_ms": aggregate["solver_successful_time_s"]["mean"] * 1000,
            "solver_p95_ms": aggregate["solver_successful_time_s"]["p95"] * 1000,
            "controller_without_setup_mean_ms": aggregate["controller_without_first_cycle_time_s"]["mean"] * 1000,
            "controller_without_setup_p95_ms": aggregate["controller_without_first_cycle_time_s"]["p95"] * 1000,
            "controller_without_setup_deadline_overruns": sum(
                t["controller_without_first_cycle_overrun_count"] for t in trials),
            "source": str(directory.resolve()),
        })
    write_csv(output / "comparison.csv", rows)
    (output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")

    # Verify every paired seed has exactly the same realized starting pose.
    baseline_outcomes = read_csv(Path(str(args.prefix) + "_baseline/results/trial_results.csv"))
    starts = {int(t["start_seed"]): [float(t[k]) for k in
              ("initial_x", "initial_y", "initial_theta")] for t in baseline_outcomes}
    for suffix, _ in EXPERIMENTS:
        for trial in read_csv(Path(str(args.prefix) + f"_{suffix}/results/trial_results.csv")):
            np.testing.assert_array_equal(starts[int(trial["start_seed"])],
                [float(trial[k]) for k in ("initial_x", "initial_y", "initial_theta")])
    (output / "verification.json").write_text(json.dumps({
        "matched_start_poses": True,
        "total_runs": sum(row["trials"] for row in rows),
        "note": "Pilot seeds 0-2; full batches seeds 0-9. No independence claim across repeated seeds. "
                "Penetration checked at recorded poses, not continuously. Timings are exploratory observations.",
    }, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"baseline": "#C77824", "candidate": "#2267A5"}
    labels = {"baseline": "Baseline (seed 1)", "candidate": "Proximal + guard (seed 1)"}
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True, constrained_layout=True)
    for suffix in ("baseline", "candidate"):
        trial = next(t for t in summaries[suffix]["trials"] if t["seed"] == 1)
        history = read_csv(trial["history_path"])
        time = column(history, "time")
        linestyle = "--" if suffix == "baseline" else "-"
        for ax, field in zip(axes[:2], ("v", "omega")):
            ax.plot(time, column(history, field), color=colors[suffix], linestyle=linestyle,
                    linewidth=1.05, label=labels[suffix])
        audits = read_csv(Path(str(args.prefix) + f"_{suffix}/oscillation_analysis/covariance_audit_cycles.csv"))
        chosen = [r for r in audits if int(r["seed"]) == 1
                  and r.get("successful_control_cycle") == "True" and not r.get("audit_error")]
        axes[2].plot(column(chosen, "time"),
                     column(chosen, "stage1_heading_sigma_exact_over_predicted"),
                     color=colors[suffix], linestyle=linestyle, linewidth=1.05)
    axes[0].set_ylabel("Linear input v [m/s]")
    axes[1].set_ylabel("Angular input omega [rad/s]")
    axes[2].set_ylabel("Heading sigma\nexact / RTI prediction")
    axes[2].set_xlabel("Simulation time [s]")
    axes[2].set_yscale("log")
    axes[2].axhline(1, color="#333333", linewidth=.8, linestyle=":")
    axes[0].legend(loc="upper right", frameon=False)
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].set_title("Matched start pose: input history and offline covariance audit")
    fig.savefig(output / "seed1_controls.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    for ax, suffix in zip(axes, ("baseline", "candidate")):
        directory = Path(str(args.prefix) + "_" + suffix)
        geometry = json.loads((directory / "environment_geometry.json").read_text())
        for polygon in geometry["obstacles"]:
            ax.add_patch(plt.Polygon(polygon, color="#D7D9DC"))
        for trial in summaries[suffix]["trials"]:
            history = read_csv(trial["history_path"])
            x, y = column(history, "x"), column(history, "y")
            ax.plot(x, y, color=colors[suffix], linewidth=.9, alpha=.7)
            ax.scatter(x[-1], y[-1], marker="o" if trial["status"] == "success" else "x",
                       color=colors[suffix], s=22)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        success_count = sum(t["status"] == "success" for t in summaries[suffix]["trials"])
        label = "Baseline" if suffix == "baseline" else "Proximal + guard"
        ax.set_title(f"{label}: {success_count}/10 reached goal")
    fig.suptitle("All 10 trials: circle = reached goal; cross = failed or timed out")
    fig.savefig(output / "trajectories.png", dpi=180)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
