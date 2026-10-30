"""Audit recorded RTI covariance states against the saved nonlinear dynamics.

This reads telemetry only and performs no navigation or controller solves.
Run with the pp Python environment after (or during) the diagnostic batch.
An incomplete trailing JSONL line is ignored and recorded in the output.
"""

from __future__ import annotations

import ast
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import casadi as ca


OUTPUT = Path(__file__).resolve().parent
SNAPSHOT = OUTPUT / "source_snapshot"
sys.path.insert(0, str(SNAPSHOT))
spec = importlib.util.spec_from_file_location(
    "snapshot_cc_covariance", SNAPSHOT / "control/cc_covariance.py"
)
covariance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(covariance)
from models.dd_dynamics import DifferentialDriveDynamics


def source_parameters():
    """Resolve self assignments from the saved parameter initializers."""
    param = SimpleNamespace()
    for filename, class_name in (
        ("control/mpcc_optimizer.py", "MPCCOptimizerParam"),
        ("control/cc_mf_optimizer.py", "CCMFOptimizerParam"),
    ):
        tree = ast.parse((SNAPSHOT / filename).read_text())
        cls = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == class_name)
        init = next(node for node in cls.body
                    if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        for statement in init.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not all(isinstance(target, ast.Attribute)
                       and isinstance(target.value, ast.Name)
                       and target.value.id == "self" for target in statement.targets):
                continue
            value = eval(compile(ast.Expression(statement.value), filename, "eval"),
                         {"np": np, "self": param})
            for target in statement.targets:
                setattr(param, target.attr, value)
    return param


PARAM = source_parameters()
DT = PARAM.tf / PARAM.horizon
ROBUST_HORIZON = covariance.resolve_robust_horizon(PARAM)
XI_SYM = ca.MX.sym("audit_xi", 4)
INPUT_SYM = ca.MX.sym("audit_input", 2)
NEXT_XI_SYM = covariance.symbolic_log_cholesky_transition(
    XI_SYM, INPUT_SYM[0], INPUT_SYM[1], 0.0, DT, PARAM)
COVARIANCE_LINEARIZATION = ca.Function(
    "audit_covariance_linearization", [XI_SYM, INPUT_SYM],
    [NEXT_XI_SYM, ca.jacobian(NEXT_XI_SYM, INPUT_SYM)])


def finite_max(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if len(arr) else None


def finite_min(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.min()) if len(arr) else None


def stats(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return ({"count": int(len(arr)), "min": float(arr.min()),
             "median": float(np.median(arr)), "p95": float(np.quantile(arr, .95)),
             "max": float(arr.max())} if len(arr) else {"count": 0})


def correlation(rows, key_a, key_b):
    data = np.asarray([[r[key_a], r[key_b]] for r in rows], dtype=float)
    if len(data) < 3 or not np.all(np.std(data, axis=0) > 0):
        return None
    return float(np.corrcoef(data.T)[0, 1])


def audit_record(record, previous):
    states = np.asarray(record["states"], dtype=float)
    inputs = np.asarray(record["inputs"], dtype=float)
    xi = states[:, 4:8]
    n = len(inputs)
    local_next = np.empty_like(xi[1:])
    rollout = np.empty_like(xi)
    rollout[0] = xi[0]
    for k in range(n):
        if k < ROBUST_HORIZON:
            local_next[k] = covariance.propagate_log_cholesky(
                xi[k], inputs[k, 0], inputs[k, 1], 0.0, DT, PARAM)
            rollout[k + 1] = covariance.propagate_log_cholesky(
                rollout[k], inputs[k, 0], inputs[k, 1], 0.0, DT, PARAM)
        else:
            local_next[k] = xi[k]
            rollout[k + 1] = rollout[k]
    local_defects = xi[1:] - local_next
    physical_next = np.stack([
        DifferentialDriveDynamics.forward_dynamics(states[k, :3], inputs[k, :2], DT)
        for k in range(n)
    ])
    physical_defects = states[1:, :3] - physical_next
    progress_defects = states[1:, 3] - states[:-1, 3] - DT * inputs[:, 2]
    predicted_sigma = np.sqrt(np.diagonal(
        covariance.covariance_from_log_cholesky(xi), axis1=-2, axis2=-1))
    exact_sigma = np.sqrt(np.diagonal(
        covariance.covariance_from_log_cholesky(rollout), axis1=-2, axis2=-1))
    sigma_ratio = predicted_sigma / exact_sigma
    diag = record.get("diagnostics") or {}
    nominal_u0 = np.array([previous["v"], previous["omega"]]) if previous else np.zeros(2)
    nominal_xi1, jacobian_u0 = COVARIANCE_LINEARIZATION(xi[0], nominal_u0)
    nominal_xi1 = np.asarray(nominal_xi1).ravel()
    jacobian_u0 = np.asarray(jacobian_u0)
    tangent_xi1 = nominal_xi1 + jacobian_u0 @ (inputs[0, :2] - nominal_u0)
    nominal_sigma_psi = float(np.sqrt(
        covariance.covariance_from_log_cholesky(nominal_xi1)[2, 2]))
    result = {
        "time": record["time"], "step": record["solve_step"],
        "mode": record["mode"], "status": record["status"],
        "pose": record["pose"], "v": float(inputs[0, 0]),
        "omega": float(inputs[0, 1]), "v_s": float(inputs[0, 2]),
        "abs_omega": float(abs(inputs[0, 1])),
        "abs_delta_omega": (abs(float(inputs[0, 1]) - previous["omega"])
                            if previous else 0.0),
        "omega_sign_flip": bool(previous and previous["omega"] * inputs[0, 1] < 0
                                and abs(previous["omega"]) > .005
                                and abs(inputs[0, 1]) > .005),
        "low_speed": bool(abs(inputs[0, 0]) < .01),
        "current_phi": diag.get("exact_current_psdf"),
        "max_local_log_factor_defect": float(np.abs(local_defects[:, :3]).max()),
        "max_local_cross_factor_defect": float(np.abs(local_defects[:, 3]).max()),
        "max_local_xy_defect_m": float(np.abs(physical_defects[:, :2]).max()),
        "max_local_theta_defect_rad": float(np.abs(physical_defects[:, 2]).max()),
        "stage1_xy_defect_m": float(np.abs(physical_defects[0, :2]).max()),
        "stage1_theta_defect_rad": float(abs(physical_defects[0, 2])),
        "max_progress_defect_m": float(np.abs(progress_defects).max()),
        "max_rollout_log_factor_defect": float(np.abs(xi[:, :3] - rollout[:, :3]).max()),
        "stage1_predicted_sigma": predicted_sigma[1].tolist(),
        "stage1_exact_sigma": exact_sigma[1].tolist(),
        "stage1_heading_sigma_ratio": float(sigma_ratio[1, 2]),
        "stage1_heading_log_factor_defect": float(local_defects[0, 2]),
        "stage1_nominal_input_v": float(nominal_u0[0]),
        "stage1_nominal_input_omega": float(nominal_u0[1]),
        "stage1_nominal_sigma_psi": nominal_sigma_psi,
        "stage1_xi_psi_tangent_prediction": float(tangent_xi1[2]),
        "stage1_log_covariance_vs_tangent_max_abs": (
            float(np.abs(xi[1, :3] - tangent_xi1[:3]).max())
            if record["mode"] == "sqp_rti" else None),
        "stage1_nominal_xi_psi_derivative_v": float(jacobian_u0[2, 0]),
        "stage1_nominal_xi_psi_derivative_omega": float(jacobian_u0[2, 1]),
        "min_heading_sigma_ratio": float(sigma_ratio[1:, 2].min()),
        "max_heading_sigma_ratio": float(sigma_ratio[1:, 2].max()),
        "max_heading_sigma_underestimate_factor": float(1 / sigma_ratio[1:, 2].min()),
        "min_xyz_sigma_ratio": sigma_ratio[1:].min(axis=0).tolist(),
        "max_xyz_sigma_ratio": sigma_ratio[1:].max(axis=0).tolist(),
        "max_local_log_defect_stage": int(np.unravel_index(
            np.abs(local_defects[:, :3]).argmax(), local_defects[:, :3].shape)[0] + 1),
        "fresh_mf_min": finite_min(diag.get("mf_icdf_residual_fresh", [])),
        "nominal_stage1_probability": (record.get("nominal", {}).get("probability_sum")
                                       or [None, None])[1],
        "nominal_stage1_mf_residual": (record.get("nominal", {}).get("residual")
                                       or [None, None])[1],
        "mf_affine_minus_frozen_max": finite_max(
            diag.get("mf_icdf_affine_minus_frozen", [])),
        "nlp_equality_residual": ((diag.get("solver") or {}).get("nlp_residuals")
                                  or [None, None])[1],
    }
    return result


def summarize(rows):
    metrics = ("max_local_log_factor_defect", "max_rollout_log_factor_defect",
               "stage1_heading_sigma_ratio", "min_heading_sigma_ratio",
               "max_heading_sigma_underestimate_factor", "max_local_xy_defect_m",
               "max_local_theta_defect_rad", "stage1_xy_defect_m",
               "stage1_log_covariance_vs_tangent_max_abs")
    last4sec = [r for r in rows if r["time"] >= rows[-1]["time"] - 4 + 1e-8]
    eq_comparison = [abs(r["nlp_equality_residual"] - max(
        r["max_local_log_factor_defect"], r["max_local_cross_factor_defect"],
        r["max_local_xy_defect_m"], r["max_local_theta_defect_rad"],
        r["max_progress_defect_m"])) for r in rows
        if r["nlp_equality_residual"] is not None]
    return {
        "cycles": len(rows), "last_time": rows[-1]["time"] if rows else None,
        "mode_counts": dict(Counter(r["mode"] for r in rows)),
        "status_counts": dict(Counter(str(r["status"]) for r in rows)),
        "metrics": {key: stats([r[key] for r in rows]) for key in metrics},
        "angular_sign_flips": sum(r["omega_sign_flip"] for r in rows),
        "equality_residual_minus_direct_max_dynamics_defect": stats(eq_comparison),
        "last4sec": {
            "cycles": len(last4sec),
            "angular_sign_flips": sum(r["omega_sign_flip"] for r in last4sec),
            "v": stats([r["v"] for r in last4sec]),
            "omega": stats([r["omega"] for r in last4sec]),
            "stage1_heading_underestimate_factor": stats([
                1 / r["stage1_heading_sigma_ratio"] for r in last4sec]),
            **{key: stats([r[key] for r in last4sec]) for key in metrics},
        },
        "correlations": {
            key: correlation(rows, "max_local_log_factor_defect", key)
            for key in ("abs_omega", "abs_delta_omega")
        },
        "group_comparisons": {
            group: {key: stats([r[key] for r in rows if r[flag]]) for key in metrics}
            for group, flag in (("omega_flip_cycles", "omega_sign_flip"),
                                ("low_speed_cycles", "low_speed"))
        },
        "worst_local_defects": sorted(rows, key=lambda r: r[metrics[0]], reverse=True)[:8],
        "worst_stage1_heading_underestimates": sorted(
            rows, key=lambda r: r["stage1_heading_sigma_ratio"])[:8],
        "first_omega_flips": [r for r in rows if r["omega_sign_flip"]][:10],
    }


def main():
    output = {
        "method": "Read telemetry only. Compare every optimized covariance transition"
                  " to saved exact propagate_log_cholesky, plus an exact full rollout"
                  " from each cycle's initial covariance under the optimized inputs.",
        "units": {"sigma": "m,m,rad", "log_factor_defect": "dimensionless",
                  "cross_factor_defect": "rad", "time": "s"},
        "parameters": {name: getattr(PARAM, name) for name in (
            "horizon", "tf", "robust_horizon", "sigma_f0", "sigma_l0", "sigma_psi0",
            "q_f0", "q_l0", "q_psi0", "alpha_f", "alpha_v", "alpha_kappa",
            "beta_v", "beta_kappa", "beta_omega")},
        "trials": {},
    }
    for filename in sorted(OUTPUT.glob("telemetry_trial*.jsonl")):
        rows, invalid = [], 0
        with filename.open() as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    continue
                row = audit_record(record, rows[-1] if rows else None)
                rows.append(row)
        summary = summarize(rows)
        summary["ignored_incomplete_lines"] = invalid
        summary["cycles_data"] = rows
        output["trials"][filename.stem] = summary
    destination = OUTPUT / "covariance_analysis.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    csv_rows = []
    for trial, summary in output["trials"].items():
        for row in summary["cycles_data"]:
            csv_row = {"trial": trial, **{key: value for key, value in row.items()
                                        if not isinstance(value, list)}}
            for prefix in ("stage1_predicted_sigma", "stage1_exact_sigma"):
                for axis, value in zip(("f", "l", "psi"), row[prefix]):
                    csv_row[f"{prefix}_{axis}"] = value
            csv_rows.append(csv_row)
    with (OUTPUT / "covariance_cycle_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    compact = {key: {k: v for k, v in value.items()
                     if k not in ("cycles_data", "worst_local_defects",
                                  "worst_stage1_heading_underestimates",
                                  "first_omega_flips", "group_comparisons")}
               for key, value in output["trials"].items()}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
