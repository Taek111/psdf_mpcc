"""Scientific plots and exact polygon-clearance checks of recorded failures."""

import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/"source_snapshot"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from sim.simulation_mpc import simulation_mpc
from sim.start_perturbation import pose_clearance
from models.dd import DifferentialDriveRectangleGeometry

data = json.loads((BASE/"failure_analysis.json").read_text())
cycles = pd.read_csv(BASE/"failure_cycle_metrics.csv")
failed = [trial for trial in data["original_trials"] if trial["status"] == "failure"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})
BLUE, RED, GRAY = "#17669c", "#bf4c24", "#777777"
fig, axes = plt.subplots(4, 3, figsize=(12, 8), constrained_layout=True)
for col, trial in enumerate(failed):
    seed = trial["seed"]
    w = cycles[cycles.seed == seed].copy()
    w = w[w.time >= w.time.max()-3.9-1e-8]
    t = w.time.to_numpy() - w.time.max()
    axes[0, col].plot(t, w.v*1000, "o-", ms=2.5, lw=1, color=BLUE)
    axes[0, col].set_title(f"Trial {trial['trial']} / seed {seed}")
    axes[1, col].plot(t, w.omega, "o-", ms=2.5, lw=1, color=BLUE)
    axes[2, col].plot(t, w.stage1_exact_sigma_psi*1000, "o-", ms=2.5, lw=1,
                      color=RED, label="Nonlinear covariance update")
    axes[2, col].plot(t, w.stage1_predicted_sigma_psi*1000, "s--", ms=2.5, lw=1,
                      color=BLUE, label="RTI predicted state")
    axes[2, col].axhline(.25, color=GRAY, lw=.8, ls=":", label="Initial heading std")
    axes[3, col].plot(t, (w.current_s0-w.current_s0.iloc[0])*1000, "o-", ms=2.5, lw=1, color=BLUE)
    for ax in axes[:, col]:
        ax.grid(alpha=.18)
        ax.set_xlim(t[0], t[-1])
    axes[3, col].set_xlabel("Time relative to last control cycle (s)")
    axes[2, col].legend(fontsize=7, loc="upper right")
for ax, label in zip(axes[:, 0], ["Applied v (mm/s)", "Applied omega (rad/s)",
                                 "Next-step heading std (mrad)", "Path progress change (mm)"]):
    ax.set_ylabel(label)
fig.suptitle("CC-SF stalled trials: final 4 s (every shown solver status is 0)", fontsize=13)
fig.savefig(BASE/"failure_mechanism.png", dpi=160)
plt.close(fig)

start, goal, grid, obstacles = simulation_mpc().create_env("maze")
polygons = [np.asarray(region.get_ccw_vertices(), float) for region in obstacles]
footprint = [np.asarray(region.get_ccw_vertices(), float)
             for region in DifferentialDriveRectangleGeometry(.15, .12, 0.).equiv_rep()]
geometry = {"start": start.tolist(), "goal": goal.tolist(), "bounds": grid[0],
            "obstacles": [polygon.tolist() for polygon in polygons]}
(BASE/"diagnostics/environment_geometry.json").write_text(json.dumps(geometry, indent=2)+"\n")
clearances = []
fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
for polygon in polygons:
    ax.add_patch(Polygon(polygon, closed=True, facecolor="#e3e3e3", edgecolor="#999999", lw=.5))
for trial in data["original_trials"]:
    h = pd.read_csv(next((BASE/f"results/runs/config01/trial_{trial['trial']:04d}/data").glob("trial_history*")))
    bad = trial["status"] == "failure"
    ax.plot(h.x, h.y, color=RED if bad else BLUE, lw=.9, alpha=.65 if bad else .35)
    if bad:
        selected = h[h.time >= trial["window_start_s"]-1e-8]
        values = [pose_clearance(pose, footprint, polygons, grid[0])
                  for pose in selected[["x","y","theta"]].to_numpy()]
        clearances.append({"seed": trial["seed"], "samples": len(values),
                           "minimum_polygon_clearance_mm": float(min(values)*1000),
                           "final_polygon_clearance_mm": float(values[-1]*1000)})
        pose = h[["x","y","theta"]].iloc[-1].to_numpy()
        theta = pose[2]
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        ax.add_patch(Polygon(footprint[0]@rotation.T+pose[:2], closed=True,
                             edgecolor=RED, facecolor="none", lw=1.2))
        ax.plot(h.x.iloc[-1], h.y.iloc[-1], "x", color=RED, ms=6)
        ax.annotate(f"seed {trial['seed']}", tuple(pose[:2]), xytext=(4, 15+10*(trial["seed"]==5)),
                    textcoords="offset points", color=RED, fontsize=10)
ax.plot([], [], color=BLUE, label="Successful trial")
ax.plot([], [], color=RED, label="Stalled trial / final footprint")
ax.scatter(*start[:2], color=GRAY, s=25, label="Start")
ax.scatter(*goal[:2], color=GRAY, marker="*", s=70, label="Goal")
ax.set(xlabel="World x (m)", ylabel="World y (m)", aspect="equal",
       xlim=(-.02, 1.97), ylim=(-.02, .92), title="CC-SF: all 10 original trajectories")
ax.legend(fontsize=9, loc="lower left")
fig.savefig(BASE/"failure_locations.png", dpi=160)
plt.close(fig)
(BASE/"polygon_clearance_audit.json").write_text(json.dumps(clearances, indent=2)+"\n")
print(json.dumps(clearances, indent=2))
