"""Summarize completed diagnostics and plot executed control histories."""

import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
import numpy as np
from shapely.geometry import Polygon
import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "source_snapshot"))
torch.set_num_threads(6)
torch.set_num_interop_threads(1)
from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
from models.dd import DifferentialDriveRectangleGeometry
from sim.simulation_mpc import simulation_mpc


def load_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def finite_stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))} if len(values) else None


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


outcomes = load_csv(BASE / "results" / "trial_results.csv")
start, goal, grid, obstacles = simulation_mpc().create_env("maze")
obstacle_polygons = [Polygon(obs.get_ccw_vertices()) for obs in obstacles]
geometry = DifferentialDriveRectangleGeometry(0.15, 0.12, 0.0)
vertices = geometry._region.get_ccw_vertices()
optimizer = CCMFOptimizer()
optimizer.param = CCMFOptimizerParam()
optimizer.N = optimizer.param.horizon
optimizer.initialize_augmented_psdf(SimpleNamespace(
    _geometry=SimpleNamespace(_geometries=[geometry])), device="cuda")
optimizer.update_obstacles(obstacles)

all_rows = []
summaries = []
metrics = []
risk_replay = []
for trial, outcome in enumerate(outcomes, 1):
    with (BASE / f"telemetry_trial{trial:02d}.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    all_rows.append(rows)
    inputs = np.array([row["inputs"][0] for row in rows])
    modes = [row["mode"] for row in rows]
    nonlinear_unsafe = 0
    affine_unsafe = 0
    nonlinear_risk_unsafe = 0
    nonlinear_soft_unsafe = 0
    affine_errors = []
    risk_errors = []
    min_clearance = np.inf
    physical_collisions = 0
    mask_switches = []
    for index, row in enumerate(rows):
        pose = np.asarray(row["pose"])
        c, s = np.cos(pose[2]), np.sin(pose[2])
        rotation = np.array([[c, -s], [s, c]])
        poly = Polygon(vertices @ rotation.T + pose[:2])
        boundary = min(poly.bounds[0] - grid[0][0][0],
                       poly.bounds[1] - grid[0][0][1],
                       grid[0][1][0] - poly.bounds[2],
                       grid[0][1][1] - poly.bounds[3])
        clearance = min(boundary, *(poly.distance(obs) for obs in obstacle_polygons))
        min_clearance = min(min_clearance, clearance)
        physical_collisions += int(boundary < 0 or any(poly.intersects(obs) for obs in obstacle_polygons))
        d = row["diagnostics"]
        if index:
            mask_switches.append(np.count_nonzero(
                np.array(row["nominal"]["mf_mask"][1:]) !=
                np.array(rows[index - 1]["nominal"]["mf_mask"][1:])))
        if d is None:
            continue
        guard_affine = np.array(d["guard_affine_residual"], float)[1:]
        guard_exact = np.array(d["guard_fresh_exact_psdf_residual"], float)[1:]
        risk_affine = np.array(d["mf_affine_residual"], float)[1:]
        risk_frozen = np.array(d["mf_icdf_residual_frozen"], float)[1:]
        fresh_risk = np.array(d["mf_probability_sum_fresh"], float)[1:]
        mask = np.array(d["mf_mask"], bool)[1:]
        if row["status"] == 0:
            nonlinear_unsafe += int(np.min(guard_exact) < -0.0002)
            affine_unsafe += int(np.min(guard_affine) < -1e-6)
            nonlinear_risk_unsafe += int(np.any(fresh_risk[mask] > 0.2 + 1e-6))
            same_slack = np.array(d["mf_lower_slack"], float)[1:]
            fresh_residual = np.array(d["mf_fresh_raw_residual"], float)[1:]
            fresh_mask = np.array(d["mf_fresh_mask"], bool)[1:]
            nonlinear_soft_unsafe += int(np.any(
                mask & fresh_mask & (risk_affine + same_slack >= -1e-6)
                & (fresh_residual + same_slack < -1e-3)))
            affine_errors.extend((guard_affine - guard_exact).tolist())
            risk_errors.extend((risk_affine[mask] - risk_frozen[mask]).tolist())
        metric = {
            "trial": trial, "time": row["time"], "mode": row["mode"],
            "status": row["status"], "v": inputs[index, 0], "omega": inputs[index, 1],
            "psdf_mm": d["exact_current_psdf"] * 1000,
            "predicted_exact_psdf_min_mm": d["minimum_exact_psdf_predicted_nodes"] * 1000,
            "guard_affine_min_mm": np.min(guard_affine) * 1000,
            "active_mf_rows": np.count_nonzero(mask),
            "nlp_eq_residual": d["solver"]["nlp_residuals"][1],
            "max_mf_slack": np.nanmax(np.array(d["mf_lower_slack"], float)),
        }
        metrics.append(metric)
        # Isolate covariance-dynamics error: keep the solver's predicted poses
        # fixed, replacing only its covariance by the exact process rollout.
        if row["status"] == 0 and (index % 20 == 0 or index >= len(rows) - 40):
            states = np.array(row["states"], float)[:, :8]
            optimizer.state = SimpleNamespace(_x=pose)
            corrected = states.copy()
            corrected[:, 4:8] = optimizer._propagate_covariance_state_batch(
                np.asarray(row["inputs"]), corrected[:, 3])
            reevaluated = optimizer._compute_constraint_affine_data(corrected)
            risk_replay.append({
                "trial": trial, "time": row["time"],
                "solver_cov_probability": d["mf_probability_sum_fresh"],
                "exact_cov_probability": reevaluated["probability_sum"],
                "solver_cov_residual": d["mf_icdf_residual_fresh"],
                "exact_cov_residual": reevaluated["residual"],
                "stage1_omega": inputs[index, 1], "stage1_v": inputs[index, 0],
            })
    summaries.append({
        "trial": trial, "seed": trial - 1, "status": outcome["status"],
        "final_time_s": float(outcome["final_time"]), "steps": len(rows),
        "safe_stop_cycles": sum(mode != "sqp_rti" for mode in modes),
        "successful_solver_cycles": sum(row["status"] == 0 for row in rows),
        "omega_reversals_ge_0_05": int(np.sum(
            (inputs[1:, 1] * inputs[:-1, 1] < 0)
            & (abs(inputs[1:, 1]) >= 0.05) & (abs(inputs[:-1, 1]) >= 0.05))),
        "omega_rate_saturated_transitions": int(np.sum(np.isclose(
            abs(np.diff(inputs[:, 1])), 0.35, atol=1e-5))),
        "min_precontrol_psdf_mm": min(row["diagnostics"]["exact_current_psdf"]
                                        for row in rows if row["diagnostics"]) * 1000,
        "min_precontrol_polygon_clearance_mm": min_clearance * 1000,
        "precontrol_polygon_collision_count": physical_collisions,
        "successful_cycles_predicted_horizon_has_penetration": nonlinear_unsafe,
        "successful_cycles_affine_guard_violation": affine_unsafe,
        "successful_cycles_fresh_mf_sum_gt_epsilon": nonlinear_risk_unsafe,
        "successful_cycles_affine_soft_feasible_but_fresh_soft_violated": nonlinear_soft_unsafe,
        "guard_affine_minus_exact_m": finite_stats(affine_errors),
        "mf_affine_minus_frozen_quantile": finite_stats(risk_errors),
        "mean_mf_row_mask_changes_per_cycle": float(np.mean(mask_switches)),
    })

with (BASE / "cycle_metrics.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
    writer.writeheader()
    writer.writerows(metrics)
(BASE / "diagnostic_summary.json").write_text(json.dumps(plain(summaries), indent=2) + "\n")
(BASE / "risk_replay.json").write_text(json.dumps(plain(risk_replay), indent=2) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(13, 11), constrained_layout=True)
gs = fig.add_gridspec(4, 2, height_ratios=[1.4, 1, 1, 1])
ax_path = fig.add_subplot(gs[0, :])
for obs in obstacle_polygons:
    ax_path.add_patch(PolygonPatch(np.array(obs.exterior.coords), color="#d7dce2"))
colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
for trial, (rows, color) in enumerate(zip(all_rows, colors), 1):
    poses = np.array([r["pose"] for r in rows])
    ax_path.plot(poses[:, 0], poses[:, 1], color=color, lw=1.5,
                 label=f"Trial {trial} / seed {trial - 1}")
    ax_path.scatter(*poses[-1, :2], color=color, marker="o" if trial == 1 else "x", s=45)
ax_path.scatter(*goal[:2], marker="*", color="black", s=110, label="Goal")
ax_path.set(xlim=(0, 1.95), ylim=(0, 0.9), xlabel="x [m]", ylabel="y [m]",
            title="Executed paths: 1 arrival, 4 stuck-check terminations")
ax_path.set_aspect("equal")
ax_path.legend(ncol=3, loc="lower center", fontsize=9)
ax_speed = fig.add_subplot(gs[1, :])
for trial, (rows, color) in enumerate(zip(all_rows, colors), 1):
    ax_speed.plot([r["time"] for r in rows],
                  [r["inputs"][0][0] * 1000 for r in rows], color=color, lw=1, label=str(trial))
ax_speed.set(xlabel="Simulation time [s]", ylabel="v [mm/s]", title="Applied forward commands")
ax_speed.axhline(0, color="black", lw=0.5)
ax_speed.grid(alpha=0.2)
for column, trial in enumerate((2, 3)):
    rows = all_rows[trial - 1]
    rows = [row for row in rows if row["time"] >= rows[-1]["time"] - 2.5]
    times = [row["time"] for row in rows]
    ax = fig.add_subplot(gs[2, column])
    ax.step(times, [row["inputs"][0][1] for row in rows], where="post", color=colors[trial-1])
    ax.axhline(0, color="black", lw=0.5)
    ax.set(ylabel="omega [rad/s]", title=f"Trial {trial}: alternating turns, solver status 0")
    ax.grid(alpha=0.2)
    ax = fig.add_subplot(gs[3, column])
    ax.step(times, [row["inputs"][0][0] * 1000 for row in rows], where="post", color=colors[trial-1])
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="Simulation time [s]", ylabel="v [mm/s]", title="Small forward / reverse commands")
    ax.grid(alpha=0.2)
fig.suptitle("CC-MF | rectangle 0.15 x 0.12 m | DD | maze | seeds 0-4", fontsize=15)
fig.savefig(BASE / "oscillation_diagnostics.png", dpi=170)
print(json.dumps(plain(summaries), indent=2))
print(f"Saved {len(risk_replay)} covariance-only risk replays and diagnostic figure.")
