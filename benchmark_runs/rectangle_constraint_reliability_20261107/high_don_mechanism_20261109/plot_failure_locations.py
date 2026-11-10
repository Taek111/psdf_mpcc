"""All 50 seeds at DoN .85: first-corner outcomes and recorded failure locations."""

from collections import Counter
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from analyze_high_don import ROOT, OUTPUT, SUITE, save_csv


def main():
    records = [r for r in csv.DictReader((OUTPUT / "trial_terminal_audit.csv").open()) if float(r["don"]) == .85]
    membership, summary = [], []
    for method in ("cc_sf", "cc_mf"):
        rows = [r for r in records if r["source"] == method]
        for row in rows:
            trace = SUITE / "widths" / row["trace"].replace("\\", "/").split("/widths/", 1)[1]
            raw = np.load(trace)
            poses, times = raw["poses"], raw["pose_times_s"]
            inside = 0 <= float(row["final_x_m"]) <= .2 and .45 <= float(row["final_y_m"]) <= .65
            stage = "goal_success" if row["status"] == "success" else ("first_corner_failure" if inside else "later_failure")
            crossed = np.flatnonzero(poses[:, 0] >= .2)
            if stage == "later_failure" and not len(crossed):
                raise ValueError("Failure outside first-corner ROI did not pass the stated center-x threshold")
            membership.append({
                "don": .85, "source": method, "seed": int(row["seed"]), "outcome_group": stage,
                "trial_status": row["status"], "denominator_trials": 50,
                "roi_x_min_m": 0, "roi_x_max_m": .2, "roi_y_min_m": .45, "roi_y_max_m": .65,
                "initial_x_m": poses[0, 0], "initial_y_m": poses[0, 1],
                "terminal_x_m": poses[-1, 0], "terminal_y_m": poses[-1, 1], "terminal_heading_rad": poses[-1, 2],
                "first_guard_time_s": row["first_guard_current_pose_time_s"],
                "terminal_clearance_mm": row["terminal_true_clearance_mm"],
                "recorded_center_passed_x_0p20": bool(len(crossed)),
                "first_recorded_center_x_ge_0p20_time_s": float(times[crossed[0]]) if len(crossed) else None,
            })
        selected = [r for r in membership if r["source"] == method]
        counts = Counter(r["outcome_group"] for r in selected)
        summary.append({
            "don": .85, "source": method, "attempted_trials": 50,
            "first_corner_failure_trials": counts["first_corner_failure"],
            "later_failure_trials": counts["later_failure"], "goal_success_trials": counts["goal_success"],
            "recorded_center_passed_x_0p20_trials": sum(r["recorded_center_passed_x_0p20"] for r in selected),
            "roi_definition": "inclusive recorded stopped center x in [0,0.20]m, y in [0.45,0.65]m",
            "passage_definition": "at least one actual recorded robot-center sample with x>=0.20m; not whole-footprint passage",
            "later_failure_definition": "failed outside ROI after the recorded center had reached x>=0.20m",
        })
    save_csv("first_corner_roi_trial_membership.csv", membership)
    save_csv("first_corner_roi_summary.csv", summary)
    geometry = json.loads((SUITE / "widths/width_127p5mm/runs/config03/trial_0011/data/raw_trace_metadata.json").read_text())["geometry"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(14, 7.3))
    bars = fig.add_axes([.12, .68, .81, .18])
    maps = [fig.add_axes([.065, .20, .405, .37]), fig.add_axes([.56, .20, .405, .37])]
    colors = {"first_corner_failure": "#BA5A2A", "later_failure": "#767C86", "goal_success": "#246B9D"}
    labels = {"first_corner_failure": "First-corner failure", "later_failure": "Later failure", "goal_success": "Goal reached"}
    for index, row in enumerate(summary):
        start = 0
        values = [row["first_corner_failure_trials"], row["later_failure_trials"], row["goal_success_trials"]]
        for name, value in zip(colors, values):
            bars.barh(index, value, left=start, height=.53, color=colors[name], edgecolor="white", linewidth=1)
            if value:
                bars.text(start + value / 2, index, str(value), ha="center", va="center", color="white", fontsize=12, fontweight="bold")
            start += value
    bars.set(yticks=[0, 1], yticklabels=["CC-SF", "CC-MF"], xlim=(0, 50), ylim=(1.6, -.6),
             xticks=[0, 10, 20, 30, 40, 50], xlabel="Trials (same 50 seed IDs per controller)")
    bars.tick_params(axis="y", length=0, pad=18)
    bars.spines[["top", "right", "left"]].set_visible(False)
    bars.grid(axis="x", color="#EAEAEA", lw=.5)
    bars.set_axisbelow(True)
    from matplotlib.patches import Patch
    bars.legend(handles=[Patch(color=colors[k], label=labels[k]) for k in colors], loc="lower center",
                bbox_to_anchor=(.5, 1.03), ncol=3, frameon=False)
    for ax, method, name, row in zip(maps, ("cc_sf", "cc_mf"), ("CC-SF", "CC-MF"), summary):
        for polygon in geometry["obstacle_polygons"]:
            ax.add_patch(Polygon(polygon, closed=True, facecolor="#E3E5E7", edgecolor="#B8BBC0", lw=.45))
        ax.add_patch(Rectangle((0, .45), .2, .2, fill=False, edgecolor="#BA5A2A", linestyle=(0, (4, 2)), lw=1.3))
        group = [r for r in membership if r["source"] == method]
        ax.scatter([r["initial_x_m"] for r in group], [r["initial_y_m"] for r in group],
                   s=8, color="#6C6174", alpha=.45, marker=".", label="Recorded starts")
        # This A* endpoint was source-verified in the original benchmark audit.
        ax.scatter([1.81875], [.09375], marker="*", s=95, color="#202020", label="Goal")
        for outcome in ("first_corner_failure", "later_failure"):
            failures = [r for r in group if r["outcome_group"] == outcome]
            ax.scatter([r["terminal_x_m"] for r in failures], [r["terminal_y_m"] for r in failures],
                       s=43, marker="x", color=colors[outcome], linewidths=1.2, alpha=.8,
                       label=labels[outcome] if failures else None)
        ax.annotate(f"First corner: {row['first_corner_failure_trials']}/50", xy=(.10, .55), xytext=(.25, .83),
                    color=colors["first_corner_failure"], fontsize=10,
                    arrowprops={"arrowstyle": "->", "color": colors["first_corner_failure"], "lw": .8})
        ax.set(xlim=(-.03, 1.98), ylim=(-.03, .93), xlabel="World x (m)", ylabel="World y (m)")
        ax.set_aspect("equal")
        ax.set_title(f"{name}: {row['goal_success_trials']}/50 goals; "
                     f"{row['first_corner_failure_trials'] + row['later_failure_trials']}/50 guard-stop failures", loc="left", pad=10)
        ax.grid(color="#EEEEEE", lw=.4)
    maps[1].legend(loc="upper right", fontsize=8.5, frameon=True)
    fig.suptitle("DoN 0.85 · where the paired benchmark outcomes diverge", fontsize=15, y=.98)
    fig.text(.5, .925, "All 50 trials per controller. Markers show actual stopped robot centers; overlapping markers are not jittered.",
             ha="center", fontsize=10, color="#505050")
    fig.text(.065, .12, "First-corner ROI: stopped center x ∈ [0, 0.20] m and y ∈ [0.45, 0.65] m. "
             "SF: 16 first-corner failures + 2 later failures + 32 goals. MF: 0 + 5 + 45.", fontsize=10)
    fig.text(.065, .08, "The recorded center reaches x ≥ 0.20 m in 34/50 SF trials and 50/50 MF trials. "
             "All later failures occur after that crossing; this is not a whole-footprint passage definition.", fontsize=9, color="#505050")
    fig.text(.065, .045, "Observed locations and outcomes establish a spatial difference; they do not isolate feature coverage, slack, "
             "or solver linearization as its cause. Successful runs are not treated as censored failures.", fontsize=9, color="#505050")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"cohort_failure_locations_don085.{extension}", dpi=240, facecolor="white")
    plt.close(fig)
    print(OUTPUT / "cohort_failure_locations_don085.png")


if __name__ == "__main__":
    main()
