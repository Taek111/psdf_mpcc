"""Audit original stalls and passive replays using the saved controller sources."""

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

BASE = Path(__file__).resolve().parent
SNAPSHOT = BASE / "source_snapshot"
sys.path.insert(0, str(SNAPSHOT))

import casadi as ca
import numpy as np
import pandas as pd

from control.cc_covariance import covariance_from_log_cholesky, propagate_log_cholesky, symbolic_log_cholesky_transition
from models.dd_dynamics import DifferentialDriveDynamics

PARAM = SimpleNamespace(**json.loads((BASE / "resolved_cc_sf_parameters.json").read_text()))
DT = PARAM.tf / PARAM.horizon
XI = ca.MX.sym("xi", 4)
U = ca.MX.sym("u", 2)
NEXT = symbolic_log_cholesky_transition(XI, U[0], U[1], 0, DT, PARAM)
TRANSITION = ca.Function("audit_transition", [XI, U], [NEXT, ca.jacobian(NEXT, XI), ca.jacobian(NEXT, U)])


def stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return {"min": float(values.min()), "median": float(np.median(values)),
            "mean": float(values.mean()), "max": float(values.max())} if len(values) else None


def margin(xi, terms):
    cov = covariance_from_log_cholesky(xi)
    q_l = np.sqrt(np.maximum(terms[:, 1], 0))
    q_theta = np.where(terms[:, 3] < 0, -1., 1.) * np.sqrt(np.maximum(terms[:, 2], 0))
    variance = terms[:, 0] * cov[:, 0, 0] + (np.exp(xi[:, 1])*q_l + xi[:, 3]*q_theta)**2 + np.exp(2*xi[:, 2])*terms[:, 2]
    return PARAM.chance_gamma * np.sqrt(np.maximum(variance, 0))


def audit(record):
    x, u = np.asarray(record["states"]), np.asarray(record["inputs"])
    nominal = np.asarray(record["nominal_states"])
    nominal_u = np.asarray(record["nominal_inputs"])
    p = np.asarray(record["params"])
    psdf = p[:, slice(*record["psdf_param_slice"])]
    terms = p[:, slice(*record["risk_tensor_param_slice"])]
    affine_phi = psdf[:, 3] + np.sum(psdf[:, 4:7] * (x[:, :3] - psdf[:, :3]), axis=1)
    predicted_margin = margin(x[:, 4:8], terms)
    frozen_h = affine_phi - predicted_margin - PARAM.d_min
    local = np.empty_like(x[1:, 4:8])
    rollout = np.empty_like(x[:, 4:8]); rollout[0] = x[0, 4:8]
    for k in range(len(u)):
        if k < 10:
            local[k] = propagate_log_cholesky(x[k, 4:8], *u[k, :2], 0., DT, PARAM)
            rollout[k+1] = propagate_log_cholesky(rollout[k], *u[k, :2], 0., DT, PARAM)
        else:
            local[k] = x[k, 4:8]; rollout[k+1] = rollout[k]
    rollout_margin = margin(rollout, terms)
    sigma = np.sqrt(np.diagonal(covariance_from_log_cholesky(x[:, 4:8]), axis1=-2, axis2=-1))
    exact_sigma = np.sqrt(np.diagonal(covariance_from_log_cholesky(rollout), axis1=-2, axis2=-1))
    physical_next = np.stack([DifferentialDriveDynamics.forward_dynamics(x[k, :3], u[k, :2], DT) for k in range(len(u))])
    fresh = np.asarray(record["fresh_psdf_params"])
    val, jx, ju = TRANSITION(nominal[0, 4:8], nominal_u[0, :2])
    tangent = np.asarray(val).ravel() + np.asarray(jx) @ (x[0, 4:8]-nominal[0, 4:8]) + np.asarray(ju) @ (u[0, :2]-nominal_u[0, :2])
    tight = np.flatnonzero(abs(frozen_h[:-1]) < 1e-5)
    return {
        "seed": record["seed"], "time": record["time"], "mode": record["mode"], "status": record["status"],
        "x": x[0, 0], "y": x[0, 1], "theta": x[0, 2], "v": u[0, 0], "omega": u[0, 1], "v_s": u[0, 2],
        "current_s0": record["current_s0"], "predicted_s1": x[1, 3], "predicted_sN": x[-1, 3],
        "max_covariance_log_defect": abs(x[1:, 4:7]-local[:, :3]).max(),
        "stage1_log_cov_defect": abs(x[1, 4:7]-local[0, :3]).max(),
        "stage1_covariance_tangent_error": abs(x[1, 4:8]-tangent).max(),
        "stage1_predicted_sigma_psi": sigma[1, 2], "stage1_exact_sigma_psi": exact_sigma[1, 2],
        "stage1_sigma_ratio": sigma[1, 2] / exact_sigma[1, 2],
        "max_sigma_psi_underestimate_factor": np.max(exact_sigma[1:, 2]/sigma[1:, 2]),
        "max_xy_dynamic_defect": abs(x[1:, :2]-physical_next[:, :2]).max(),
        "max_theta_dynamic_defect": abs(x[1:, 2]-physical_next[:, 2]).max(),
        "max_progress_dynamic_defect": abs(x[1:, 3]-x[:-1, 3]-DT*u[:, 2]).max(),
        "min_installed_nonlinear_h": frozen_h[:-1].min(),
        "min_installed_affine_phi": affine_phi[:-1].min(),
        "min_fresh_phi_at_predicted_poses": fresh[:-1, 3].min(),
        "min_h_with_exact_covariance_same_affine_geometry": (affine_phi-rollout_margin-PARAM.d_min)[:-1].min(),
        "max_predicted_margin": predicted_margin[:-1].max(), "max_exact_covariance_margin": rollout_margin[:-1].max(),
        "current_phi": fresh[0, 3], "tight_stages": ",".join(map(str, tight.tolist())),
        "nominal_v0": nominal_u[0, 0], "nominal_omega0": nominal_u[0, 1],
        "stage1_log_sigma_psi_derivative_omega": np.asarray(ju)[2, 1],
        "horizon_pose_span_m": np.linalg.norm(x[-1, :2]-x[0, :2]),
        "horizon_heading_span_rad": np.ptp(x[:, 2]),
    }


original_rows = pd.read_csv(BASE / "safe_stop_summary.csv")
events = pd.read_csv(BASE / "safe_stop_events.csv")
original = []
for _, row in original_rows.iterrows():
    directory = Path(row.output_root_dir)/"data"
    h = pd.read_csv(next(directory.glob("trial_history*")))
    q = pd.read_csv(next(directory.glob("pose_sdf*")))
    end = float(h.time.iloc[-1]); start = end-20.
    w = h[(h.time > start+1e-8) & (h.time <= end+1e-8)]
    z = q[(q.timestep*DT >= start-1e-8)]
    endpoints = h[(h.time >= start-1e-8)].copy()
    last = h[h.time > end-4.+1e-8]
    xy = endpoints[["x", "y"]].to_numpy()
    v, omega = last.v.to_numpy(), last.omega.to_numpy()
    selected_events = events[(events.trial_number == row.trial_number) & (events.time_before_control_s >= start-1e-8)]
    original.append({
        "trial": int(row.trial_number), "seed": int(row.start_seed), "status": row.status,
        "window_start_s": start, "end_s": end, "failure_reason": str(row.failure_reason),
        "last20_successful_steps": int(w.solver_success.sum()), "last20_steps": len(w),
        "last20_safe_stops": len(selected_events),
        "last20_net_displacement_m": float(np.linalg.norm(xy[-1]-xy[0])),
        "last20_traveled_distance_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        "last20_abs_speed_mm_s": stats(abs(w.v)*1000),
        "last4_v_mm_s": stats(last.v*1000), "last4_omega_rad_s": stats(last.omega),
        "last4_v_reversals": int(np.count_nonzero(v[:-1]*v[1:] < 0)),
        "last4_omega_reversals": int(np.count_nonzero(omega[:-1]*omega[1:] < 0)),
        "last4_two_step_return_error_um": stats(np.linalg.norm(last[["x", "y"]].to_numpy()[2:]-last[["x", "y"]].to_numpy()[:-2], axis=1)*1e6),
        "last20_current_psdf_mm": stats(z.sdf_value*1000),
        "last20_path_progress_m": float(z.path_s.iloc[-1]-z.path_s.iloc[0]),
        "end_xy": xy[-1].tolist(),
        "last20_slice_changes": int(np.count_nonzero(np.diff(z.slice_start_idx))),
        "last20_distance_s_to_slice_end_min_m": float((z.slice_end_s-z.path_s).min()),
    })

cycles = []
reproductions = []
for path in sorted((BASE/"diagnostics").glob("telemetry_seed*.jsonl")):
    seed = int(path.stem.split("seed")[1])
    outcome_path = BASE / f"diagnostics/outcome_seed{seed}.json"
    if not outcome_path.exists():
        continue
    with path.open() as handle:
        records = [json.loads(line) for line in handle]
    hpath, = (BASE / f"diagnostics/seed{seed}/data").glob("trial_history*")
    actual = pd.read_csv(hpath)
    baseline = pd.read_csv(next((BASE/f"results/runs/config01/trial_{seed+1:04d}/data").glob("trial_history*")))
    assert len(actual) == len(baseline)
    delta = abs(actual[["x","y","theta","v","omega"]].to_numpy()-baseline[["x","y","theta","v","omega"]].to_numpy())
    reproductions.append({"seed": seed, "steps": len(actual), "max_pose_input_difference": float(delta.max()),
                          "status_match": actual.solver_success.equals(baseline.solver_success)})
    cycles.extend(audit(record) for record in records)

pd.DataFrame(cycles).to_csv(BASE/"failure_cycle_metrics.csv", index=False)
replay_summary = []
metrics = ["stage1_sigma_ratio", "max_sigma_psi_underestimate_factor", "max_covariance_log_defect",
           "stage1_covariance_tangent_error", "min_installed_nonlinear_h",
           "min_h_with_exact_covariance_same_affine_geometry", "max_xy_dynamic_defect",
           "max_theta_dynamic_defect", "max_progress_dynamic_defect", "stage1_log_sigma_psi_derivative_omega",
           "max_predicted_margin", "max_exact_covariance_margin", "horizon_pose_span_m"]
frame = pd.DataFrame(cycles)
if len(frame):
    for seed, group in frame.groupby("seed"):
        for duration in [20., 4.]:
            w = group[group.time >= group.time.max()-duration+DT-1e-8]
            replay_summary.append({"seed": int(seed), "window_s": duration, "cycles": len(w),
                                   "modes": dict(Counter(w["mode"])),
                                   **{metric: stats(w[metric]) for metric in metrics}})
output = {"original_trials": original, "reproduction": reproductions, "replay_windows": replay_summary}
(BASE/"failure_analysis.json").write_text(json.dumps(output, indent=2) + "\n")
for trial in original:
    if trial["status"] != "success":
        print(json.dumps(trial, indent=2))
print("REPRODUCTION", json.dumps(reproductions))
for item in replay_summary:
    if item["window_s"] == 4:
        print(json.dumps(item, indent=2))
