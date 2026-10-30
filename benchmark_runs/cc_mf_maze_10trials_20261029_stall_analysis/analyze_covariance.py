"""Offline covariance audit of the saved CC-MF telemetry; no controller solves.

Uses this batch's source snapshot and resolved parameters. JSONL is streamed;
an incomplete trailing line is ignored. Rerun after all trials finish to refresh
the outputs. Last-20s and last-4s windows end at the final logged control time.
"""

from __future__ import annotations

import csv
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import casadi as ca
import numpy as np


OUTPUT = Path(__file__).resolve().parent
SNAPSHOT = OUTPUT / "source_snapshot"
sys.path.insert(0, str(SNAPSHOT))
SPEC = importlib.util.spec_from_file_location(
    "batch_snapshot_covariance", SNAPSHOT / "control/cc_covariance.py")
covariance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(covariance)
from models.dd_dynamics import DifferentialDriveDynamics

PARAMETERS = json.loads((OUTPUT / "resolved_cc_mf_parameters.json").read_text())
PARAM = SimpleNamespace(**PARAMETERS)
DT = float(PARAM.tf) / int(PARAM.horizon)
ROBUST_HORIZON = covariance.resolve_robust_horizon(PARAM)
XI = ca.MX.sym("offline_xi", 4)
CONTROL = ca.MX.sym("offline_u", 2)
NEXT_XI = covariance.symbolic_log_cholesky_transition(
    XI, CONTROL[0], CONTROL[1], 0.0, DT, PARAM)
TRANSITION_AND_JACOBIAN = ca.Function(
    "offline_covariance_transition_jacobian", [XI, CONTROL],
    [NEXT_XI, ca.jacobian(NEXT_XI, XI), ca.jacobian(NEXT_XI, CONTROL)])


def stats(values):
    values = np.asarray([np.nan if value is None else value for value in values],
                        dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {"count": int(len(values)), "min": float(values.min()),
            "median": float(np.median(values)),
            "mean": float(values.mean()), "p95": float(np.quantile(values, .95)),
            "max": float(values.max())}


def finite_min(values):
    result = stats(values)
    return result.get("min")


def sigma(xi):
    matrix = covariance.covariance_from_log_cholesky(xi)
    return np.sqrt(np.diagonal(matrix, axis1=-2, axis2=-1))


def valid_float(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def audit_record(record, previous):
    states = np.asarray(record["states"], dtype=float)
    inputs = np.asarray(record["inputs"], dtype=float)
    xi = states[:, 4:8]
    nominal = record.get("nominal") or {}
    nominal_states = np.asarray(nominal.get("nominal_state"), dtype=float)
    nominal_inputs = np.asarray(record.get("nominal_inputs"), dtype=float)
    local_next = np.empty_like(xi[1:])
    rollout = np.empty_like(xi)
    rollout[0] = xi[0]
    for stage in range(len(inputs)):
        if stage < ROBUST_HORIZON:
            local_next[stage] = covariance.propagate_log_cholesky(
                xi[stage], *inputs[stage, :2], 0.0, DT, PARAM)
            rollout[stage + 1] = covariance.propagate_log_cholesky(
                rollout[stage], *inputs[stage, :2], 0.0, DT, PARAM)
        else:
            local_next[stage] = xi[stage]
            rollout[stage + 1] = rollout[stage]
    local_defect = xi[1:] - local_next
    physical_next = np.stack([
        DifferentialDriveDynamics.forward_dynamics(states[k, :3], u[:2], DT)
        for k, u in enumerate(inputs)])
    physical_defect = states[1:, :3] - physical_next
    progress_defect = states[1:, 3] - states[:-1, 3] - DT * inputs[:, 2]
    predicted_sigma, exact_sigma = sigma(xi), sigma(rollout)
    sigma_ratio = predicted_sigma / exact_sigma
    diag = record.get("diagnostics") or {}
    residuals = (diag.get("solver") or {}).get("nlp_residuals")
    mode = record["mode"]
    successful = record.get("status") == 0 and mode == "sqp_rti"
    pose = np.asarray(record["pose"], dtype=float)
    row = {
        "time": float(record["time"]), "solve_step": record["solve_step"],
        "mode": mode, "status": record.get("status"), "success": successful,
        "x": float(pose[0]), "y": float(pose[1]), "theta": float(pose[2]),
        "v": float(inputs[0, 0]), "omega": float(inputs[0, 1]),
        "v_s": float(inputs[0, 2]), "s0": float(record["current_s0"]),
        "abs_v": float(abs(inputs[0, 0])),
        "abs_omega": float(abs(inputs[0, 1])),
        "delta_omega": (float(inputs[0, 1]) - previous["omega"]
                        if previous else None),
        "omega_sign_flip": bool(previous and previous["omega"] * inputs[0, 1] < 0
                                and abs(previous["omega"]) > .005
                                and abs(inputs[0, 1]) > .005),
        "omega_sign_flip_gt_0p05": bool(previous and previous["omega"] * inputs[0, 1] < 0
                                        and abs(previous["omega"]) > .05
                                        and abs(inputs[0, 1]) > .05),
        "v_sign_flip": bool(previous and previous["v"] * inputs[0, 0] < 0
                            and abs(previous["v"]) > .0001
                            and abs(inputs[0, 0]) > .0001),
        "current_phi_m": valid_float(diag.get("exact_current_psdf")),
        "max_local_log_factor_defect": float(abs(local_defect[:, :3]).max()),
        "max_local_cross_factor_defect": float(abs(local_defect[:, 3]).max()),
        "max_rollout_log_factor_defect": float(abs(xi[:, :3] - rollout[:, :3]).max()),
        "max_local_xy_defect_m": float(abs(physical_defect[:, :2]).max()),
        "max_local_theta_defect_rad": float(abs(physical_defect[:, 2]).max()),
        "stage1_xy_defect_m": float(abs(physical_defect[0, :2]).max()),
        "stage1_theta_defect_rad": float(abs(physical_defect[0, 2])),
        "max_progress_defect_m": float(abs(progress_defect).max()),
        "stage1_heading_sigma_predicted_over_exact": float(sigma_ratio[1, 2]),
        "stage1_heading_sigma_underestimate_factor": float(1.0 / sigma_ratio[1, 2]),
        "horizon_max_heading_sigma_underestimate_factor": float(
            1.0 / sigma_ratio[1:, 2].min()),
        "nlp_equality_residual": valid_float(residuals[1]) if residuals else None,
        "fresh_mf_min": finite_min(diag.get("mf_icdf_residual_fresh") or []),
        "first_interval_min_psdf_m": valid_float(
            diag.get("minimum_exact_psdf_first_interval_substeps")),
        "predicted_nodes_min_psdf_m": valid_float(
            diag.get("minimum_exact_psdf_predicted_nodes")),
    }
    for prefix, values in (("stage1_predicted_sigma", predicted_sigma[1]),
                           ("stage1_exact_sigma", exact_sigma[1])):
        row.update({f"{prefix}_{axis}": float(value)
                    for axis, value in zip(("f", "l", "psi"), values)})
    tangent = None
    if nominal_states.shape == (len(inputs) + 1, 8) and nominal_inputs.shape == inputs.shape:
        nominal_xi0, nominal_u0 = nominal_states[0, 4:8], nominal_inputs[0, :2]
        next_value, jac_x, jac_u = TRANSITION_AND_JACOBIAN(nominal_xi0, nominal_u0)
        tangent = (np.asarray(next_value).ravel()
                   + np.asarray(jac_x) @ (xi[0] - nominal_xi0)
                   + np.asarray(jac_u) @ (inputs[0, :2] - nominal_u0))
        row.update({
            "nominal_v0": float(nominal_u0[0]),
            "nominal_omega0": float(nominal_u0[1]),
            "stage1_tangent_max_abs_error": (
                float(abs(xi[1] - tangent).max()) if successful else None),
            "stage1_nominal_sigma_psi": float(sigma(nominal_states[1, 4:8])[2]),
            "stage1_nominal_covariance_rollout_error": float(abs(
                np.asarray(next_value).ravel() - nominal_states[1, 4:8]).max()),
            "stage1_log_heading_derivative_omega": float(np.asarray(jac_u)[2, 1]),
        })
    else:
        row.update({key: None for key in (
            "nominal_v0", "nominal_omega0", "stage1_tangent_max_abs_error",
            "stage1_nominal_sigma_psi", "stage1_nominal_covariance_rollout_error",
            "stage1_log_heading_derivative_omega")})
    row["direct_max_equality_defect"] = max(row[key] for key in (
        "max_local_log_factor_defect", "max_local_cross_factor_defect",
        "max_local_xy_defect_m", "max_local_theta_defect_rad", "max_progress_defect_m"))
    row["reported_minus_direct_equality_abs_error"] = (
        abs(row["nlp_equality_residual"] - row["direct_max_equality_defect"])
        if row["nlp_equality_residual"] is not None else None)
    return row


METRICS = (
    "abs_v", "omega", "abs_omega", "current_phi_m", "max_local_log_factor_defect",
    "max_rollout_log_factor_defect", "stage1_heading_sigma_predicted_over_exact",
    "stage1_heading_sigma_underestimate_factor",
    "horizon_max_heading_sigma_underestimate_factor", "stage1_tangent_max_abs_error",
    "stage1_nominal_covariance_rollout_error", "max_local_xy_defect_m",
    "stage1_xy_defect_m", "max_progress_defect_m", "nlp_equality_residual",
    "reported_minus_direct_equality_abs_error", "predicted_nodes_min_psdf_m",
)


def aggregate(rows):
    successful = [row for row in rows if row["success"]]
    return {
        "cycles": len(rows), "successful_rti_cycles": len(successful),
        "time_range": [rows[0]["time"], rows[-1]["time"]] if rows else [],
        "mode_counts": dict(Counter(row["mode"] for row in rows)),
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
        "omega_sign_flips": sum(row["omega_sign_flip"] for row in rows),
        "omega_sign_flip_amplitude_threshold_rad_s": 0.005,
        "omega_sign_flips_gt_0p05": sum(row["omega_sign_flip_gt_0p05"] for row in rows),
        "v_sign_flips": sum(row["v_sign_flip"] for row in rows),
        "mean_abs_v_mm_s": (1000 * float(np.mean([row["abs_v"] for row in rows])))
                             if rows else None,
        "net_xy_displacement_mm": (1000 * float(np.hypot(
            rows[-1]["x"] - rows[0]["x"], rows[-1]["y"] - rows[0]["y"])))
            if len(rows) > 1 else None,
        "metrics_successful_rti_only": {key: stats([row[key] for row in successful])
                                        for key in METRICS},
    }


def representative_transitions(rows, failed_stall):
    if not failed_stall:
        return []
    candidates = []
    for i in range(1, len(rows) - 1):
        previous, middle, following = rows[i - 1:i + 2]
        if (middle["time"] < rows[-1]["time"] - 20
                or not all(row["success"] for row in (previous, middle, following))
                or not middle["omega_sign_flip"] or not following["omega_sign_flip"]):
            continue
        score = float(np.median([row["stage1_heading_sigma_underestimate_factor"]
                                 for row in (previous, middle, following)]))
        candidates.append((score, i))
    selected, used = [], set()
    for score, i in sorted(candidates, reverse=True):
        if any(j in used for j in range(i - 1, i + 2)):
            continue
        keys = ("time", "v", "omega", "nominal_v0", "nominal_omega0", "delta_omega",
                "stage1_predicted_sigma_psi", "stage1_exact_sigma_psi",
                "stage1_nominal_sigma_psi", "stage1_heading_sigma_underestimate_factor",
                "stage1_tangent_max_abs_error", "stage1_log_heading_derivative_omega",
                "nlp_equality_residual", "max_local_log_factor_defect",
                "stage1_xy_defect_m", "current_phi_m", "status", "mode")
        selected.append({"ranking_median_stage1_heading_underestimate_factor": score,
                         "transitions": [{key: row[key] for key in keys}
                                         for row in rows[i - 1:i + 2]]})
        used.update(range(i - 2, i + 3))
        if len(selected) == 3:
            break
    return selected


def terminal_transition(rows, failed_stall):
    if not failed_stall or len(rows) < 3:
        return None
    keys = ("time", "v", "omega", "nominal_v0", "nominal_omega0", "delta_omega",
            "stage1_predicted_sigma_psi", "stage1_exact_sigma_psi",
            "stage1_nominal_sigma_psi", "stage1_heading_sigma_underestimate_factor",
            "stage1_tangent_max_abs_error", "stage1_log_heading_derivative_omega",
            "nlp_equality_residual", "max_local_log_factor_defect",
            "stage1_xy_defect_m", "current_phi_m", "status", "mode")
    return [{key: row[key] for key in keys} for row in rows[-3:]]


def main():
    outcomes_path = OUTPUT / "results/trial_results.csv"
    outcomes = {}
    if outcomes_path.exists():
        with outcomes_path.open(newline="") as handle:
            outcomes = {int(row["trial_number"]): row for row in csv.DictReader(handle)}
    summary = {
        "method": __doc__, "parameter_source": "resolved_cc_mf_parameters.json",
        "dynamics_source": "source_snapshot/control/cc_covariance.py",
        "dt_s": DT, "robust_horizon": ROBUST_HORIZON,
        "interpretation_limits": [
            "Positive sigma ratios compare optimized model states with exact model rollout, not measured estimator covariance.",
            "A covariance tangent match establishes the recorded RTI approximation; it alone does not prove that modifying it cures stalls.",
            "Metrics of solver candidates do not imply the physical plant followed their full prediction horizon.",
            "Per-trial windows on an in-progress run may change; outcome_recorded identifies completed runs."],
        "trials": {},
    }
    csv_rows = []
    for path in sorted(OUTPUT.glob("telemetry_trial*.jsonl")):
        trial = int(path.stem.removeprefix("telemetry_trial"))
        rows, incomplete = [], 0
        with path.open() as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    incomplete += 1
                    continue
                row = audit_record(record, rows[-1] if rows else None)
                rows.append(row)
        if not rows:
            continue
        outcome = outcomes.get(trial)
        failed_stall = bool(outcome and "stuck_for_" in outcome["failure_reason"])
        trial_summary = {
            "outcome_recorded": outcome is not None,
            "outcome": outcome, "ignored_incomplete_lines": incomplete,
            "all": aggregate(rows),
            "last20sec": aggregate([row for row in rows
                                     if row["time"] > rows[-1]["time"] - 20 + 1e-8]),
            "last4sec": aggregate([row for row in rows
                                    if row["time"] > rows[-1]["time"] - 4 + 1e-8]),
            "representative_repeated_sign_reversals": representative_transitions(rows, failed_stall),
            "terminal_three_transitions": terminal_transition(rows, failed_stall),
        }
        summary["trials"][path.stem] = trial_summary
        csv_rows.extend({"trial": trial, **row} for row in rows)
    (OUTPUT / "covariance_analysis.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if csv_rows:
        with (OUTPUT / "covariance_cycle_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(json.dumps({name: {
        "outcome_recorded": trial["outcome_recorded"],
        "status": (trial["outcome"] or {}).get("status"),
        "cycles": trial["all"]["cycles"],
        "last20sec_omega_sign_flips": trial["last20sec"]["omega_sign_flips"],
        "last4sec_omega_sign_flips": trial["last4sec"]["omega_sign_flips"],
        "last20sec_stage1_sigma_underestimate_factor": trial["last20sec"][
            "metrics_successful_rti_only"]["stage1_heading_sigma_underestimate_factor"],
        "tangent_error": trial["all"]["metrics_successful_rti_only"][
            "stage1_tangent_max_abs_error"],
    } for name, trial in summary["trials"].items()}, indent=2))


if __name__ == "__main__":
    main()
