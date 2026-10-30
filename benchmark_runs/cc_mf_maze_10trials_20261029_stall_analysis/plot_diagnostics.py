"""Scientific diagnostic figures from the completed trials; no simulation."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
summary = json.loads((BASE / "analysis_summary.json").read_text())
trials = summary["trials"]
cov = pd.read_csv(BASE / "covariance_cycle_metrics.csv")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})
BLUE, ORANGE, GRAY = "#17669c", "#ba541c", "#707070"

# Same axes expose differences in progress across seeds without selecting tails.
fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True,
                         constrained_layout=True)
for ax, trial in zip(axes.flat, trials):
    pose = pd.read_csv(trial["pose_sdf_path"])
    t = pose.timestep.to_numpy() * 0.1
    failed = trial["status"] == "failure"
    color = ORANGE if failed else BLUE if trial["status"] == "success" else GRAY
    ax.plot(t, pose.path_s, color=color, lw=1.5)
    marker = "x" if failed else "o" if trial["status"] == "success" else "s"
    ax.plot(t[-1], pose.path_s.iloc[-1], marker, color=color, ms=6)
    if failed:
        ax.axvspan(trial["final_time_s"] - 20, trial["final_time_s"],
                   color=ORANGE, alpha=0.09)
    ax.set_title(f"Seed {trial['seed']}: {trial['status']}\n"
                 f"{trial['final_time_s']:.1f} s", fontsize=11)
    ax.set(xlim=(0, 60), ylim=(0, 3.05), xticks=[0, 20, 40, 60])
    ax.grid(alpha=0.2)
for ax in axes[-1]:
    ax.set_xlabel("Simulation time (s)")
for ax in axes[:, 0]:
    ax.set_ylabel("Path progress s (m)")
fig.suptitle("CC-MF: path progress in all 10 trials; shaded = final 20 s of stalled trials")
fig.savefig(BASE / "progress_all_trials.png", dpi=150)
plt.close(fig)

stalled = [trial for trial in trials if trial["status"] == "failure"
           and "stuck_for" in trial["failure_reason"]]
selected = stalled[:2]
fig, axes = plt.subplots(3, len(selected), figsize=(12, 8), squeeze=False,
                         constrained_layout=True)
for column, trial in enumerate(selected):
    data = cov[cov.trial == trial["trial"]].copy()
    data = data[data.time >= data.time.max() - 1.2 - 1e-8]
    time = data.time.to_numpy()
    axes[0, column].plot(time, data.v * 1000, "o-", color=BLUE, ms=3)
    axes[0, column].set_ylabel("Applied v (mm/s)")
    axes[0, column].set_title(f"Seed {trial['seed']}: final two-step limit cycle")
    axes[1, column].plot(time, data.omega, "o-", color=BLUE, ms=3)
    axes[1, column].set_ylabel("Applied omega (rad/s)")
    axes[2, column].plot(time, data.stage1_exact_sigma_psi * 1000,
                         "o-", color=ORANGE, ms=3, label="Exact nonlinear transition")
    axes[2, column].plot(time, data.stage1_predicted_sigma_psi * 1000,
                         "s--", color=BLUE, ms=3, label="RTI predicted state")
    axes[2, column].set_ylabel("Next heading std (mrad)")
    axes[2, column].set_xlabel("Control-cycle time (s)")
    axes[2, column].legend(loc="best", fontsize=9)
    for ax in axes[:, column]:
        ax.grid(alpha=0.2)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.set_xlim(time[0], time[-1])
fig.suptitle("Repeated input reversal and underestimated covariance (status 0 on every shown cycle)")
fig.savefig(BASE / "stall_mechanism.png", dpi=150)
plt.close(fig)

geometry = json.loads((BASE / "environment_geometry.json").read_text())
fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
polygons = geometry.get("obstacles", geometry.get("obstacle_polygons", []))
for polygon in polygons:
    if isinstance(polygon, dict):
        polygon = polygon.get("vertices", polygon.get("polygon"))
    ax.add_patch(Polygon(np.asarray(polygon), closed=True,
                         facecolor="0.87", edgecolor="0.5", lw=0.6))
for trial in trials:
    history = pd.read_csv(trial["history_path"])
    failed = trial["status"] == "failure"
    color = ORANGE if failed else BLUE if trial["status"] == "success" else GRAY
    ax.plot(history.x, history.y, color=color, alpha=0.5, lw=1)
    if failed:
        ax.plot(history.x.iloc[-1], history.y.iloc[-1], "x", color=color, ms=7)
        ax.annotate(f"seed {trial['seed']}", (history.x.iloc[-1], history.y.iloc[-1]),
                    xytext=(4, 8 + 10 * (trial["trial"] % 2)),
                    textcoords="offset points", color=color)
start, goal = np.asarray(geometry["start"]), np.asarray(geometry["goal"])
ax.scatter([start[0]], [start[1]], marker="o", color=GRAY, label="Start", s=35)
ax.scatter([goal[0]], [goal[1]], marker="*", color=GRAY, label="Goal", s=90)
ax.plot([], [], color=BLUE, label="Successful trial")
ax.plot([], [], "x-", color=ORANGE, label="Stalled trial")
ax.plot([], [], color=GRAY, label="Timeout trial")
bounds = np.asarray(geometry["bounds"])
ax.set(xlim=(bounds[0, 0] - .03, bounds[1, 0] + .03),
       ylim=(bounds[0, 1] - .03, bounds[1, 1] + .03),
       xlabel="World x (m)", ylabel="World y (m)", aspect="equal",
       title="Recorded robot-center trajectories; rectangle footprint 0.15 x 0.12 m")
ax.legend(loc="lower right", fontsize=9)
fig.savefig(BASE / "trajectories_all_trials.png", dpi=150)
plt.close(fig)
