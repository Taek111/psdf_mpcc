"""Reconstruct a saved single-RTI stage-1 iterate without running a solver.

The QP stage-0 equality fixes the current state. Its stage-1 dynamics therefore
give f(x_current, u_previous) + f_u (u_candidate - u_previous). Retained,
unshifted warm starts make the preceding cycle's stage-1 iterate the next
cycle's nominal geometry. A scalar curvature fit is cross-validated against
all four logged log-Cholesky coordinates and the saved nominal MF sum.
"""

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import casadi as ca
import numpy as np
from scipy.optimize import brentq, least_squares
import torch

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
ROOT = RUN.parent.parent
sys.path.insert(0, str(ROOT))

from control.cc_covariance import (
    covariance_from_log_cholesky, initialize_cc_noise_params,
    log_cholesky_from_covariance, symbolic_log_cholesky_transition,
)
from models.dd_dynamics import symbolic_dd_step
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_pose(row):
    return np.array([float(row[key]) for key in ("x", "y", "theta")])


def read_u(row):
    return np.array([float(row[key]) for key in ("v", "omega")])


def decode_xi(row):
    sf, sl, sp, cross = [float(row[key]) for key in ("sigma_f", "sigma_l", "sigma_psi", "P_lpsi")]
    matrix = np.array([[sf*sf, 0, 0], [0, sl*sl, cross], [0, cross, sp*sp]])
    return log_cholesky_from_covariance(matrix)


def make_functions():
    pose = ca.SX.sym("pose", 3)
    u = ca.SX.sym("u", 2)
    next_pose = symbolic_dd_step(pose, u[0], u[1], 0.1)
    pose_fun = ca.Function("pose_fun", [pose, u], [next_pose, ca.jacobian(next_pose, u)])
    param = SimpleNamespace(q_f0=0, q_l0=0, q_psi0=0)
    initialize_cc_noise_params(param)
    xi = ca.SX.sym("xi", 4)
    kappa = ca.SX.sym("kappa")
    next_xi = symbolic_log_cholesky_transition(xi, u[0], u[1], kappa, 0.1, param)
    cov_fun = ca.Function("cov_fun", [xi, u, kappa], [next_xi, ca.jacobian(next_xi, u)])
    return pose_fun, cov_fun


def fit_curvature(cov_fun, xi0, ubar, u, xi_logged):
    def predicted(kappa):
        base, jac = cov_fun(xi0, ubar, float(kappa))
        return np.array(base).ravel()+np.array(jac)@(u-ubar)
    def residual_l(kappa):
        return float(predicted(kappa)[1]-xi_logged[1])
    grid = np.r_[0.0, np.geomspace(1e-8, 1e6, 1401)]
    roots = []
    values = [residual_l(value) for value in grid]
    for a, b, fa, fb in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if fa*fb < 0:
            root = brentq(residual_l, a, b, xtol=1e-13)
            if not any(abs(root-old) < 1e-8 for old in roots):
                roots.append(root)
    scale = np.array([1, 1, 1, 1e-6])
    if not roots:
        fitted = least_squares(lambda value: (predicted(value[0])-xi_logged)/scale,
                               [1.0], bounds=(0, 1e6), xtol=1e-14, ftol=1e-14, gtol=1e-14)
        roots = [float(fitted.x[0])]
    candidates = []
    for root in roots:
        xi_pred = predicted(root)
        candidates.append({"abs_kappa": float(root), "xi_reconstructed": xi_pred.tolist(),
                           "xi_error": (xi_pred-xi_logged).tolist(),
                           "weighted_error_norm": float(np.linalg.norm((xi_pred-xi_logged)/scale))})
    candidates.sort(key=lambda row: row["weighted_error_norm"])
    return candidates


def wrapper(dtype, obstacles, robot_vertices):
    obj = AugmentedPSDFWrapper(torch.tensor(robot_vertices, dtype=dtype), K_max=20, E_max=100, device="cpu")
    obj.update_edge_clusters([torch.tensor(obs, dtype=dtype) for obs in obstacles],
                             [torch.tensor(np.roll(obs, -1, axis=0), dtype=dtype) for obs in obstacles])
    return obj


def as_numpy(value):
    return value.detach().cpu().numpy()


def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("geom_analysis", RUN/"failure_analysis/geometry_analysis.py")
    geom = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geom)
    _, _, _, obstacles = geom.load_environment()
    robot = geom.rectangle(-.075, .075, -.05, .05)
    pose_fun, cov_fun = make_functions()
    xi0 = log_cholesky_from_covariance(np.diag(np.array([.0002, .00025, .00025])**2))
    wrappers = {"float32": wrapper(torch.float32, obstacles, robot), "float64": wrapper(torch.float64, obstacles, robot)}
    rows = []
    requested_cycles = [(trial, k) for trial, last in ((1, 35), (2, 29)) for k in range(last-6, last+1)]
    for trial, k in requested_cycles:
        data = RUN/f"runs/config01/trial_{trial:04d}/data"
        history = read_csv(next(data.glob("trial_history_*.csv")))
        poses = read_csv(next(data.glob("pose_sdf_*.csv")))
        cycles = read_csv(data/"cycle_raw.csv")
        covariances = read_csv(data/"covariance_trajectory.csv")
        xpre, ubar, u = read_pose(poses[k]), read_u(history[k-1]), read_u(history[k])
        xpre_prev, ubar_prev, u_prev = read_pose(poses[k-1]), read_u(history[k-2]), read_u(history[k-1])
        mean_base_previous, mean_jac_previous = pose_fun(xpre_prev, ubar_prev)
        xbar1 = np.array(mean_base_previous).ravel()+np.array(mean_jac_previous)@(u_prev-ubar_prev)
        mean_base, mean_jac = pose_fun(xpre, ubar)
        xcand1 = np.array(mean_base).ravel()+np.array(mean_jac)@(u-ubar)
        actual1 = read_pose(history[k])
        cov1 = next(row for row in covariances if row["stage"] == "1" and abs(float(row["time"])-k*.1)<1e-8)
        xi_logged = decode_xi(cov1)
        candidates = fit_curvature(cov_fun, xi0, ubar, u, xi_logged)
        fitted = candidates[0]
        abs_kappa = fitted["abs_kappa"]
        xi_nom, _ = cov_fun(xi0, ubar, abs_kappa)
        xi_nom = np.array(xi_nom).ravel()
        xi_actual, _ = cov_fun(xi0, u, abs_kappa)
        xi_actual = np.array(xi_actual).ravel()
        nominal_probability_logged = json.loads(cycles[k]["mf_probability_sum"])[1]
        slack = json.loads(cycles[k]["mf_slack"])[1]
        zbar = np.r_[xbar1, 0.0, xi_nom]
        zcand = np.r_[xcand1, 0.0, xi_logged]
        zactual = np.r_[actual1, 0.0, xi_actual]
        record = {
            "trial": trial, "seed": trial-1, "control_timestep": k, "time_s": k*.1,
            "current_pose": xpre.tolist(), "nominal_input0": ubar.tolist(), "candidate_input0": u.tolist(),
            "reconstructed_nominal_pose_stage1": xbar1.tolist(),
            "reconstructed_candidate_pose_stage1": xcand1.tolist(), "actual_next_pose": actual1.tolist(),
            "nonlinear_mean_dynamics_defect_m_rad": (actual1-xcand1).tolist(),
            "mean_xy_defect_mm": float(np.linalg.norm(actual1[:2]-xcand1[:2])*1000),
            "candidate_xi1_from_logged_covariance": xi_logged.tolist(),
            "curvature_fits": candidates, "fitted_abs_curvature": abs_kappa,
            "nominal_xi_stage1": xi_nom.tolist(), "nonlinear_xi_stage1_for_candidate_input": xi_actual.tolist(),
            "nominal_covariance_stage1": covariance_from_log_cholesky(xi_nom).tolist(),
            "candidate_covariance_stage1": covariance_from_log_cholesky(xi_logged).tolist(),
            "nonlinear_covariance_stage1_for_candidate_input": covariance_from_log_cholesky(xi_actual).tolist(),
            "logged_nominal_probability_stage1": nominal_probability_logged, "logged_mf_slack_stage1": slack,
            "psdf_evaluations": {},
        }
        for name, psdf in wrappers.items():
            zt = torch.tensor(zbar[None], dtype=psdf.A.dtype)
            phi, gradient, mf_a, mf_c = [as_numpy(item)[0] for item in psdf.forward_mf(zt, .2, .001)]
            nominal_probability = .2-float(mf_a@zbar+mf_c)
            guard_residual = float(phi+gradient@(xcand1-xbar1)-.001)
            mf_residual = float(mf_a@zcand+mf_c)
            exact_phis, exact_gradients = psdf(torch.tensor(np.stack([xbar1, xcand1, actual1]), dtype=psdf.A.dtype))
            cluster_phi, _ = psdf.forward_clusters(torch.tensor(np.stack([xbar1, xcand1, actual1]), dtype=psdf.A.dtype))
            candidate_probability_eval = psdf.forward_mf(torch.tensor(zcand[None], dtype=psdf.A.dtype), .2, .001)
            _, _, candidate_a, candidate_c = [as_numpy(item)[0] for item in candidate_probability_eval]
            candidate_probability = .2-float(candidate_a@zcand+candidate_c)
            actual_probability_eval = psdf.forward_mf(torch.tensor(zactual[None], dtype=psdf.A.dtype), .2, .001)
            _, _, actual_a, actual_c = [as_numpy(item)[0] for item in actual_probability_eval]
            actual_probability = .2-float(actual_a@zactual+actual_c)
            record["psdf_evaluations"][name] = {
                "phi_nominal_mm": float(phi)*1000, "gradient_nominal": gradient.tolist(),
                "mf_A_nominal": mf_a.tolist(), "mf_c_nominal": float(mf_c),
                "nominal_probability_reconstructed": nominal_probability,
                "nominal_probability_log_error": nominal_probability-nominal_probability_logged,
                "guard_affine_residual_at_candidate_mm": guard_residual*1000,
                "guard_affine_prediction_at_candidate_mm": (guard_residual+.001)*1000,
                "mf_affine_residual_at_candidate_before_slack": mf_residual,
                "mf_affine_residual_at_candidate_after_slack": mf_residual+slack,
                "mf_active_in_original_qp": bool(phi > .001),
                "effective_qp_mf_residual_after_slack": mf_residual+slack if phi > .001 else .2+slack,
                "exact_phi_nominal_candidate_actual_mm": (as_numpy(exact_phis)*1000).tolist(),
                "closest_obstacle_nominal_candidate_actual": np.argmin(as_numpy(cluster_phi), axis=1).tolist(),
                "raw_forward_mf_probability_sum_at_candidate_geometry": candidate_probability,
                "raw_forward_mf_probability_sum_at_actual_geometry_and_nonlinear_covariance": actual_probability,
                "candidate_geometry_in_separated_mf_domain": bool(as_numpy(exact_phis)[1] > .001),
                "actual_geometry_in_separated_mf_domain": bool(as_numpy(exact_phis)[2] > .001),
                "mf_pose_change_term": float(mf_a[:3]@(xcand1-xbar1)),
                "mf_covariance_change_term": float(mf_a[4:]@(xi_logged-xi_nom)),
            }
        record["reconstruction_validated"] = bool(
            fitted["weighted_error_norm"] < 1e-3 and
            abs(record["psdf_evaluations"]["float32"]["nominal_probability_log_error"]) < 2e-4 and
            record["psdf_evaluations"]["float32"]["guard_affine_residual_at_candidate_mm"] >= -.1 and
            record["psdf_evaluations"]["float32"]["effective_qp_mf_residual_after_slack"] >= -1e-4
        )
        rows.append(record)
    failures = [row for row in rows if (row["trial"], row["control_timestep"]) in ((1, 35), (2, 29))]
    payload = {"method": __doc__, "caveat": "Reconstructed, not originally logged, candidate mean states; validates the retained-warm-start single-RTI dynamics using covariance consistency and nominal MF sum. Curvature sign is not identifiable because only absolute curvature enters covariance. Raw forward_mf probability sums outside the separated domain are not meaningful safety probabilities; feature masking can yield zero even for colliding geometry.", "trials": failures, "preceding_and_failure_cycles": rows}
    (OUT/"stage1_reconstruction.json").write_text(json.dumps(payload, indent=2))
    table = []
    for row in rows:
        evals = row["psdf_evaluations"]["float32"]
        table.append({"trial": row["trial"], "seed": row["seed"], "time_s": row["time_s"],
                      "reconstruction_validated": row["reconstruction_validated"],
                      "fitted_abs_curvature": row["fitted_abs_curvature"],
                      "mean_xy_defect_mm": row["mean_xy_defect_mm"],
                      **{key: value for key, value in evals.items() if isinstance(value, (float, int))}})
    with (OUT/"stage1_reconstruction.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
