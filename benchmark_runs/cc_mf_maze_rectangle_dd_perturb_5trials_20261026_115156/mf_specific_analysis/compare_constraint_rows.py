"""Compare MF and SF tangents at reconstructed saved stage-1 iterates.

This performs no solver call or closed-loop experiment. SF is evaluated on the
MF iterate to compare local rows, not to predict the input an SF optimizer
would choose. Probabilities and gradients remain frozen at the nominal pose.
"""

import csv
import importlib.util
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch


OUT = Path(__file__).resolve().parent
RUN = OUT.parent
EPSILON = 0.2
D_MIN = 0.001
GAMMA = NormalDist().inv_cdf(1 - EPSILON)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numpy(tensor):
    return tensor.detach().cpu().numpy()


def std_and_gradient(xi, q):
    lf, ll, lt = np.exp(xi[:3])
    cross = xi[3]
    a = lf * q[..., 0]
    b0 = ll * q[..., 1]
    b = b0 + cross * q[..., 2]
    c = lt * q[..., 2]
    sigma = np.sqrt(a*a + b*b + c*c)
    gradient = np.stack((a*a, b0*b, c*c, q[..., 2]*b), axis=-1) / sigma[..., None]
    return sigma, gradient


def evaluate(record, wrapper, pose_function, covariance_function, xi_initial):
    nominal_pose = np.asarray(record["reconstructed_nominal_pose_stage1"])
    candidate_pose = np.asarray(record["reconstructed_candidate_pose_stage1"])
    nominal_xi = np.asarray(record["nominal_xi_stage1"])
    candidate_xi = np.asarray(record["candidate_xi1_from_logged_covariance"])
    zbar = np.r_[nominal_pose, 0.0, nominal_xi]
    zcandidate = np.r_[candidate_pose, 0.0, candidate_xi]
    delta_pose = candidate_pose - nominal_pose
    delta_xi = candidate_xi - nominal_xi
    tensor = torch.tensor(zbar[None], dtype=wrapper.A.dtype)
    active = wrapper.active_clusters
    phi, gradient, distances, q, mask = wrapper.psdf._forward_impl(
        wrapper.A[:active], wrapper.B[:active], wrapper.mask[:active],
        tensor[:, :3], return_features=True,
    )
    phi = float(numpy(phi)[0])
    gradient = numpy(gradient)[0]
    slots = np.flatnonzero(numpy(mask)[0])
    d = numpy(distances)[0, slots].astype(float)
    q = numpy(q)[0, slots].astype(float)
    # Recompute feature quantities in float64 after dtype-specific geometry.
    # Exact implementation row is also evaluated separately below.
    sigma, dsigma = std_and_gradient(nominal_xi, q)
    zeta = (D_MIN - d) / sigma
    probability = np.array([NormalDist().cdf(float(value)) for value in zeta])
    pdf = np.exp(-0.5*zeta*zeta) / math.sqrt(2*math.pi)
    c, s = np.cos(nominal_pose[2]), np.sin(nominal_pose[2])
    rotation = np.array([[c, -s], [s, c]])
    qworld = q.copy()
    qworld[:, :2] = q[:, :2] @ rotation.T
    feature_pose_gradient = pdf[:, None] / sigma[:, None] * qworld
    feature_xi_gradient = (pdf*zeta/sigma)[:, None] * dsigma
    feature_pose_change = feature_pose_gradient @ delta_pose
    feature_xi_change = feature_xi_gradient @ delta_xi
    feature_affine_probability = probability-feature_pose_change-feature_xi_change
    mf_result = wrapper.forward_mf(tensor, EPSILON, D_MIN)
    mf_a = numpy(mf_result[2])[0].astype(float)
    mf_c = float(numpy(mf_result[3])[0])
    mf_h = float(mf_a @ zbar + mf_c)
    mf_pose = float(mf_a[:3] @ delta_pose)
    mf_xi = float(mf_a[4:] @ delta_xi)
    affine_probability_sum_from_row = EPSILON-(mf_h+mf_pose+mf_xi)
    affine_probability_decomposition_error = float(feature_affine_probability.sum()-affine_probability_sum_from_row)
    assert abs(affine_probability_decomposition_error) < 2e-4
    slack = record["logged_mf_slack_stage1"]
    mf_active = bool(phi > 0.001 and np.isfinite(mf_a).all())
    # Same nominal PSDF-minimizing feature and fixed body-frame gradient as SF.
    qsf = gradient.copy()
    qsf[:2] = rotation.T @ gradient[:2]
    sigma_sf, gradient_sf = std_and_gradient(nominal_xi, qsf)
    sigma_sf = float(sigma_sf)
    sf_h = phi - D_MIN - GAMMA*sigma_sf
    sf_pose = float(gradient @ delta_pose)
    sf_xi = float(-GAMMA * gradient_sf @ delta_xi)
    candidate_sigma_fixed_gradient = float(std_and_gradient(candidate_xi, qsf)[0])
    _, pose_jacobian = pose_function(record["current_pose"], record["nominal_input0"])
    _, covariance_jacobian = covariance_function(
        xi_initial, record["nominal_input0"], record["fitted_abs_curvature"]
    )
    pose_jacobian = np.asarray(pose_jacobian)
    covariance_jacobian = np.asarray(covariance_jacobian)
    mf_input_pose = mf_a[:3] @ pose_jacobian
    mf_input_xi = mf_a[4:] @ covariance_jacobian
    mf_input_gradient = mf_input_pose + mf_input_xi
    sf_input_pose = gradient @ pose_jacobian
    sf_input_xi = -GAMMA * gradient_sf @ covariance_jacobian
    sf_input_gradient = sf_input_pose + sf_input_xi
    # acados slack penalty is z_l*s_l + 0.5*Z_l*s_l^2; stage weight dt=0.1.
    # This is one active stage's local pressure, not the total QP gradient.
    mf_slack_cost_input_gradient = -0.1*(100.0 + slack)*mf_input_gradient
    control_cost_input_gradient = 0.2*np.array([1.0, 0.01])*np.asarray(record["candidate_input0"])
    input_change = np.asarray(record["candidate_input0"])-np.asarray(record["nominal_input0"])

    def velocity_threshold(row_at_candidate, input_gradient, hold_candidate_omega):
        if abs(input_gradient[0]) < 1e-12:
            return None
        base_row = row_at_candidate - input_gradient @ input_change
        omega_change = input_change[1] if hold_candidate_omega else 0.0
        return float(record["nominal_input0"][0] - (
            base_row + input_gradient[1]*omega_change
        ) / input_gradient[0])

    mf_threshold_nominal_omega = velocity_threshold(mf_h+mf_pose+mf_xi, mf_input_gradient, False)
    mf_threshold_candidate_omega = velocity_threshold(mf_h+mf_pose+mf_xi, mf_input_gradient, True)
    sf_threshold_nominal_omega = velocity_threshold(sf_h+sf_pose+sf_xi, sf_input_gradient, False)
    sf_threshold_candidate_omega = velocity_threshold(sf_h+sf_pose+sf_xi, sf_input_gradient, True)
    best_mf_input = np.where(mf_input_gradient >= 0.0, [.7, 1.2], [-.7, -1.2])
    best_mf_row = float(mf_h+mf_pose+mf_xi+mf_input_gradient @ (
        best_mf_input-np.asarray(record["candidate_input0"])
    ))
    denom = np.linalg.norm(feature_pose_gradient, axis=1).sum()
    denom_xy = np.linalg.norm(feature_pose_gradient[:, :2], axis=1).sum()
    cancellation_ratio = float(np.linalg.norm(feature_pose_gradient.sum(0))/denom) if denom else None
    cancellation_xy = float(np.linalg.norm(feature_pose_gradient[:, :2].sum(0))/denom_xy) if denom_xy else None
    groups = {}
    for i in range(len(slots)):
        # Coincident feature rows only; not a proof of statistically duplicate
        # events when covariance/feature correspondence changes.
        key = tuple(np.round(np.r_[d[i], q[i]], 7))
        groups.setdefault(key, []).append(i)
    duplicate_groups = [values for values in groups.values() if len(values) > 1]
    feature_rows = []
    for i in np.argsort(-probability)[:8]:
        feature_rows.append({
            "slot": int(slots[i]), "d_m": float(d[i]), "sigma_m": float(sigma[i]),
            "standardized_separation": float(-zeta[i]), "probability": float(probability[i]),
            "q_body": q[i].tolist(), "pose_gradient_world": feature_pose_gradient[i].tolist(),
            "xi_gradient": feature_xi_gradient[i].tolist(),
            "pose_change_contribution": float(feature_pose_gradient[i] @ delta_pose),
            "xi_change_contribution": float(feature_xi_gradient[i] @ delta_xi),
            "affine_probability": float(feature_affine_probability[i]),
        })
    all_feature_probability_rows = [
        {"slot": int(slot), "nominal_probability": float(probability[i]),
         "pose_h_change": float(feature_pose_change[i]),
         "xi_h_change": float(feature_xi_change[i]),
         "affine_probability": float(feature_affine_probability[i])}
        for i, slot in enumerate(slots)
    ]
    diagnostic_clipped_sum = float(np.maximum(feature_affine_probability, 0.0).sum())
    return {
        "nominal_phi_mm": phi*1000,
        "mf_active": mf_active,
        "logged_nominal_probability": record["logged_nominal_probability_stage1"],
        "reconstructed_nominal_probability": EPSILON-mf_h,
        "probability_log_error": EPSILON-mf_h-record["logged_nominal_probability_stage1"],
        "feature_probability_sum_float64": float(probability.sum()),
        "feature_affine_probability_sum": float(feature_affine_probability.sum()),
        "feature_affine_probability_sum_from_aggregate_row": affine_probability_sum_from_row,
        "feature_affine_probability_decomposition_error": affine_probability_decomposition_error,
        "negative_affine_probability_count_raw": int(np.sum(feature_affine_probability < 0.0)),
        "negative_affine_probability_count_below_negative_1e_minus8": int(np.sum(feature_affine_probability < -1e-8)),
        "negative_affine_probability_count_below_negative_1e_minus6": int(np.sum(feature_affine_probability < -1e-6)),
        "negative_affine_probability_sum": float(feature_affine_probability[feature_affine_probability < 0.0].sum()),
        "affine_probability_zero_clipped_sum_diagnostic_only": diagnostic_clipped_sum,
        "clipped_sum_excess_over_epsilon_and_logged_slack_diagnostic_only": diagnostic_clipped_sum-EPSILON-slack,
        "all_feature_affine_probabilities": all_feature_probability_rows,
        "mf_nominal_h_probability_units": mf_h,
        "mf_pose_change_probability_units": mf_pose,
        "mf_xi_change_probability_units": mf_xi,
        "mf_candidate_row_before_slack_probability_units": mf_h+mf_pose+mf_xi,
        "mf_logged_slack_probability_units": slack,
        "mf_candidate_row_after_slack_probability_units": mf_h+mf_pose+mf_xi+slack,
        "mf_pose_gradient": mf_a[:3].tolist(), "mf_xi_gradient": mf_a[4:].tolist(),
        "mf_condensed_input_pose_gradient": mf_input_pose.tolist(),
        "mf_condensed_input_covariance_gradient": mf_input_xi.tolist(),
        "mf_condensed_input_gradient": mf_input_gradient.tolist(),
        "mf_single_active_stage_slack_cost_input_gradient": mf_slack_cost_input_gradient.tolist(),
        "mf_effective_single_stage_slack_cost_input_gradient": (mf_slack_cost_input_gradient if mf_active else np.zeros(2)).tolist(),
        "mf_zero_slack_v_threshold_omega_nominal": mf_threshold_nominal_omega,
        "mf_zero_slack_v_threshold_omega_candidate": mf_threshold_candidate_omega,
        "mf_velocity_inequality": ">=" if mf_input_gradient[0] > 0 else "<=",
        "mf_maximum_raw_row_under_first_input_box": best_mf_row,
        "mf_first_input_box_maximizer": best_mf_input.tolist(),
        "mf_minimum_slack_from_first_input_box": max(0.0, -best_mf_row) if mf_active else 0.0,
        "stage0_physical_control_cost_input_gradient": control_cost_input_gradient.tolist(),
        "feature_gradient_reconstruction_error": (np.r_[feature_pose_gradient.sum(0), feature_xi_gradient.sum(0)]-np.r_[mf_a[:3], mf_a[4:]]).tolist(),
        "pose_gradient_cancellation_ratio_mixed_units": cancellation_ratio,
        "xy_gradient_cancellation_ratio": cancellation_xy,
        "valid_feature_count": len(slots),
        "coincident_feature_group_count": len(duplicate_groups),
        "coincident_feature_extra_count": sum(len(group)-1 for group in duplicate_groups),
        "coincident_feature_probability_sum": float(sum(probability[i] for group in duplicate_groups for i in group)),
        "coincident_feature_groups": [[int(slots[i]) for i in group] for group in duplicate_groups],
        "top_features_by_probability": feature_rows,
        "sf_nominal_sigma_mm": sigma_sf*1000,
        "sf_condensed_input_pose_gradient_m": sf_input_pose.tolist(),
        "sf_condensed_input_covariance_gradient_m": sf_input_xi.tolist(),
        "sf_condensed_input_gradient_m": sf_input_gradient.tolist(),
        "sf_v_threshold_omega_nominal": sf_threshold_nominal_omega,
        "sf_v_threshold_omega_candidate": sf_threshold_candidate_omega,
        "sf_velocity_inequality": ">=" if sf_input_gradient[0] > 0 else "<=",
        "sf_nominal_h_mm": sf_h*1000,
        "sf_pose_change_mm": sf_pose*1000,
        "sf_xi_change_mm": sf_xi*1000,
        "sf_candidate_tangent_row_mm": (sf_h+sf_pose+sf_xi)*1000,
        "sf_tangent_accepts_candidate": bool(sf_h+sf_pose+sf_xi >= -1e-6),
        "sf_linearized_sigma_at_candidate_mm": (sigma_sf+float(gradient_sf @ delta_xi))*1000,
        "sf_frozen_gradient_sigma_at_candidate_exact_mm": candidate_sigma_fixed_gradient*1000,
        "sf_frozen_geometry_nonlinear_margin_row_mm": (phi-D_MIN+sf_pose-GAMMA*candidate_sigma_fixed_gradient)*1000,
        "guard_candidate_affine_residual_mm": (phi-D_MIN+sf_pose)*1000,
    }


def main():
    torch.set_num_threads(1)
    reconstruct = load_module("reconstruct_stage1", OUT / "reconstruct_stage1.py")
    geometry = load_module("geometry_analysis", RUN / "failure_analysis/geometry_analysis.py")
    _, _, _, obstacles = geometry.load_environment()
    robot = geometry.rectangle(-.075, .075, -.05, .05)
    wrappers = {"float32": reconstruct.wrapper(torch.float32, obstacles, robot),
                "float64": reconstruct.wrapper(torch.float64, obstacles, robot)}
    reconstructed = json.loads((OUT / "stage1_reconstruction.json").read_text())
    pose_function, covariance_function = reconstruct.make_functions()
    xi_initial = np.r_[np.log([.0002, .00025, .00025]), 0.0]
    def compare_record(record):
        assert record["reconstruction_validated"]
        comparisons = {
            dtype: evaluate(record, wrapper, pose_function, covariance_function, xi_initial)
            for dtype, wrapper in wrappers.items()
        }
        assert abs(comparisons["float32"]["probability_log_error"]) < 2e-4
        return {"trial": record["trial"], "seed": record["seed"],
                "time_s": record["time_s"], "comparisons": comparisons}

    records = [compare_record(record) for record in reconstructed["trials"]]
    sequence = [compare_record(record) for record in reconstructed.get("preceding_and_failure_cycles", [])]
    payload = {"method": __doc__, "gamma": GAMMA, "trials": records,
               "preceding_and_failure_cycles": sequence,
               "limits": [
                   "SF row is a local counterfactual on the MF nominal/candidate, not an SF closed-loop replay.",
                   "Mixed pose-gradient norm combines position and heading units; XY cancellation is provided separately.",
                   "Feature coincidence means equal nominal d and q to seven decimal places; it does not prove identical random events globally.",
                   "MF residuals/slacks have probability units; SF/guard residuals have distance units.",
                   "Raw MF coefficients and zero-slack velocity thresholds are counterfactual when mf_active is false; the original QP has a trivial inactive row.",
                   "Velocity thresholds hold either nominal or candidate omega fixed, eliminate only the first-step dynamics, and ignore all other horizon constraints and costs.",
                   "Post-step exact PSDF is negative; probability values outside the separated-feature domain cannot certify safety.",
                   "Negative feature affine probabilities are linearization artifacts, not physical probabilities. Clipping them at zero here is a diagnostic only and does not replace exact probability evaluation or rerun the QP.",
                   "Feature affine values decompose the single summed MF affine row; the solver does not impose or optimize these as separate feature constraints.",
               ]}
    (OUT / "row_comparison.json").write_text(json.dumps(payload, indent=2))
    csv_rows = []
    for record in sequence or records:
        flat = {key: value for key, value in record["comparisons"]["float32"].items()
                if isinstance(value, (bool, int, float, str)) or value is None}
        for rank, feature in enumerate(record["comparisons"]["float32"]["top_features_by_probability"][:3], 1):
            for key in ("slot", "d_m", "sigma_m", "probability", "pose_change_contribution", "xi_change_contribution", "affine_probability"):
                flat[f"feature{rank}_{key}"] = feature[key]
        csv_rows.append({"trial": record["trial"], "seed": record["seed"], "time_s": record["time_s"], **flat})
    with (OUT / "row_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps([row for row in csv_rows if (row["trial"], round(row["time_s"], 1))
                      in {(1, 3.5), (2, 2.9)}], indent=2))


if __name__ == "__main__":
    main()
