"""Reconstruct one-step RTI equalities from saved inputs; never run a solver."""

import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_snapshot"
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))

import casadi as ca
import numpy as np
import torch
from control.cc_covariance import (
    covariance_from_log_cholesky, log_cholesky_from_covariance,
    propagate_log_cholesky, symbolic_log_cholesky_transition,
)
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.dd_dynamics import DifferentialDriveDynamics, symbolic_dd_step


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def encode(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def functions(param):
    x = ca.MX.sym("state", 7)
    u = ca.MX.sym("input", 2)
    mapped = ca.vertcat(symbolic_dd_step(x[:3], u[0], u[1], .1),
        symbolic_log_cholesky_transition(x[3:], u[0], u[1], 0., .1, param))
    return ca.Function("offline_dynamics", [x, u], [mapped, ca.jacobian(mapped, u)])


def xi_from_row(row):
    P = np.diag([float(row[key]) ** 2 for key in ("sigma_f", "sigma_l", "sigma_psi")])
    P[1, 2] = P[2, 1] = float(row["P_lpsi"])
    return log_cholesky_from_covariance(P)


def main():
    summaries, details, validations = [], [], []
    for width in [.115, .120, .125]:
        folder = ROOT / f"width_{round(width * 1000):03d}mm"
        geometry = json.loads((folder / "geometry.json").read_text())
        param = SimpleNamespace(**json.loads((folder / "optimizer_parameters.json").read_text()))
        dynamics = functions(param)
        wrapper = AugmentedPSDFWrapper(
            torch.tensor(geometry["footprint"], dtype=torch.float32, device="cuda"),
            K_max=20, E_max=100, device="cuda")
        edges = [np.asarray(vertices) for vertices in geometry["obstacles"]]
        wrapper.update_edge_clusters(edges, [np.roll(vertices, -1, axis=0) for vertices in edges])

        def phi_at(pose):
            t = torch.as_tensor(np.atleast_2d(pose), dtype=torch.float32, device="cuda")
            phi, grad = wrapper(t)
            clusters, _ = wrapper.forward_clusters(t)
            return float(phi[0].item()), grad[0].cpu().numpy().astype(float), int(clusters[0].argmin().item())

        for trial in rows(folder / "trial_results.csv"):
            seed = int(trial["start_seed"])
            data = folder / "runs/config01" / f"trial_{seed + 1:04d}" / "data"
            history_path = data / f"trial_history_{trial['current_name']}.csv"
            history = rows(history_path)
            cycles = rows(data / f"cc_mf_icdf_cycle_{trial['current_name']}.csv")
            covariance = rows(data / f"cc_mf_icdf_covariance_{trial['current_name']}.csv")
            cov = {(round(float(row["time"]) / .1), int(row["stage"])): row for row in covariance}
            n = len(history)
            poses = np.array([[float(trial[key]) for key in ("initial_x", "initial_y", "initial_theta")]]
                + [[float(row[key]) for key in ("x", "y", "theta")] for row in history])
            inputs = np.array([[float(row[key]) for key in ("v", "omega")] for row in history])
            stopped = [i for i, row in enumerate(cycles) if row["solve_mode"] == "safe_stop_stage0"]
            if stopped:
                event = stopped[0] - 1
            else:
                # Match the tightest recorded signed distance for successful trials.
                closest = min(range(1, n), key=lambda i: float(cycles[i]["psdf"]))
                event = closest - 1
            assert event >= 2 and all(row["solve_mode"] == "sqp_rti" for row in cycles[:event + 1])
            replay_errors, covariance_errors = [], []
            reconstructed = {}
            for i in range(1, event + 1):
                xi0 = xi_from_row(cov[i, 0])
                base, jac = dynamics(np.r_[poses[i], xi0], inputs[i - 1])
                predicted = np.asarray(base).reshape(-1) + np.asarray(jac) @ (inputs[i] - inputs[i - 1])
                recorded_P = covariance_from_log_cholesky(xi_from_row(cov[i, 1]))
                predicted_P = covariance_from_log_cholesky(predicted[3:])
                covariance_errors.append(float(np.max(np.abs(recorded_P - predicted_P))))
                replay = DifferentialDriveDynamics.forward_dynamics(poses[i], inputs[i], .1)
                replay_errors.append(float(np.max(np.abs(replay - poses[i + 1]))))
                reconstructed[i] = predicted
            anchor = reconstructed[event - 1][:3]
            predicted = reconstructed[event][:3]
            actual = poses[event + 1]
            xi0 = xi_from_row(cov[event, 0])
            nominal_xi1 = propagate_log_cholesky(xi0, *inputs[event - 1], 0., .1, param)
            nominal = np.r_[anchor, 0., nominal_xi1]
            returned = np.r_[predicted, 0., xi_from_row(cov[event, 1])]
            actual_consistent = np.r_[actual, 0., propagate_log_cholesky(xi0, *inputs[event], 0., .1, param)]
            result = wrapper.forward_mf_icdf(nominal[None], param.chance_epsilon, param.d_min,
                stage_requested=np.array([True]), d_col=param.d_col, return_frozen_features=True)
            nominal_phi = float(result.phi[0].item())
            gradient = result.gradient[0].cpu().numpy().astype(float)
            A = result.A[0].cpu().numpy()
            c = float(result.c[0].item())
            required = bool(result.required[0].item())
            recorded_residual = json.loads(cycles[event]["mf_icdf_residual_nominal"])[1]
            recorded_probability = json.loads(cycles[event]["mf_probability_sum"])[1]
            recorded_active = bool(json.loads(cycles[event]["mf_active"])[1])
            assert required == recorded_active
            if required:
                np.testing.assert_allclose(float(result.residual[0].item()), recorded_residual, atol=2e-6, rtol=2e-6)
                np.testing.assert_allclose(float(result.probability_sum[0].item()), recorded_probability, atol=2e-7, rtol=2e-6)
            predicted_phi, _, predicted_cluster = phi_at(predicted)
            actual_phi, _, actual_cluster = phi_at(actual)
            anchor_phi, _, anchor_cluster = phi_at(anchor)
            np.testing.assert_allclose(nominal_phi, anchor_phi, rtol=0, atol=1e-8)
            np.testing.assert_allclose(actual_phi, float(cycles[event + 1]["psdf"]), rtol=0, atol=1e-8)
            affine_phi = nominal_phi + gradient @ (predicted - anchor)
            affine_actual_phi = nominal_phi + gradient @ (actual - anchor)
            slack = json.loads(cycles[event]["mf_slack"])[1]
            frozen = wrapper.evaluate_mf_icdf_frozen(result.frozen_features, returned[None],
                param.chance_epsilon, param.d_min, d_col=param.d_col)
            fresh = wrapper.forward_mf_icdf(actual_consistent[None], param.chance_epsilon, param.d_min,
                stage_requested=np.array([True]), d_col=param.d_col)
            record = {
                "width_m": width, "seed": seed, "trial_status": trial["status"],
                "event_type": "first_threshold_crossing" if stopped else "closest_successful_pose",
                "time_before_s": event * .1, "time_after_s": (event + 1) * .1,
                "main_solver_status": cycles[event]["solver_status"],
                "signed_psdf_before_mm": float(cycles[event]["psdf"]) * 1000,
                "nominal_stage1_phi_mm": nominal_phi * 1000,
                "predicted_affine_phi_mm": affine_phi * 1000,
                "predicted_exact_psdf_mm": predicted_phi * 1000,
                "actual_exact_psdf_mm": actual_phi * 1000,
                "hard_row_residual_mm": (affine_phi - param.d_col) * 1000,
                "affine_phi_at_actual_mm": affine_actual_phi * 1000,
                "taylor_overestimate_at_predicted_mm": (affine_phi - predicted_phi) * 1000,
                "dynamics_translation_defect_mm": float(np.linalg.norm(predicted[:2] - actual[:2])) * 1000,
                "dynamics_heading_defect_rad": float(predicted[2] - actual[2]),
                "nominal_to_predicted_translation_mm": float(np.linalg.norm(predicted[:2] - anchor[:2])) * 1000,
                "nominal_to_predicted_heading_rad": float(predicted[2] - anchor[2]),
                "nominal_cluster": anchor_cluster, "predicted_cluster": predicted_cluster,
                "actual_cluster": actual_cluster, "mf_active_stage1": required,
                "mf_nominal_residual": float(result.residual[0].item()) if required else None,
                "mf_stage1_slack": slack,
                "mf_affine_residual_unrelaxed": float(A @ returned + c) if required else None,
                "mf_affine_residual_after_slack": float(A @ returned + c + slack) if required else None,
                "mf_frozen_residual_at_returned": float(frozen.residual[0].item()) if required else None,
                "fresh_actual_mf_active": bool(fresh.required[0].item()),
                "max_covariance_reconstruction_error": max(covariance_errors),
                "max_rk4_replay_error": max(replay_errors),
            }
            summaries.append(record)
            details.append({**record, "before_pose": poses[event], "anchor_pose": anchor,
                "predicted_pose": predicted, "actual_pose": actual, "previous_input": inputs[event - 1],
                "applied_input": inputs[event], "nominal_gradient": gradient,
                "nominal_xi1": nominal_xi1, "returned_xi1": returned[4:],
                "mf_A": A, "mf_c": c, "history_path": str(history_path)})
            validations.append({"width_m": width, "seed": seed,
                "history_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
                "main_status_zero_on_crossing": cycles[event]["solver_status"] == "0",
                "native_nominal_mf_reproduced": True,
                "native_stage1_returned_covariance_reproduced": max(covariance_errors) < 1e-9,
                "actual_next_psdf_matches_following_cycle": True})
            print(json.dumps(record, default=encode), flush=True)
    with (OUTPUT / "one_step_reconstruction.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (OUTPUT / "one_step_details.json").write_text(json.dumps(details, indent=2, default=encode) + "\n")
    source_hashes = json.loads((ROOT / "source_hashes.json").read_text())
    assert all(hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
               for name, digest in source_hashes.items())
    (OUTPUT / "reconstruction_validation.json").write_text(json.dumps({
        "new_solver_or_navigation_runs": 0, "sources_unchanged": True,
        "checks": validations,
        "limitations": ["Stage-1 poses are reconstructed from one-QP linearized dynamics, not directly logged.",
            "Covariance states are independently checked against the recorded returned covariance.",
            "Feature/obstacle identity identifies polygon clusters, not all internal vertex/edge branch identities."]},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
