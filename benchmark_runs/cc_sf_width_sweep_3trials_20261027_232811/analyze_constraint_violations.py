"""Offline reconstruction of first-overlap steps; no simulation or solver run.

The next cycle's pre-solve stage-1 pose retains the previous returned pose:
the controller does not shift pose/input guesses, and covariance regeneration
changes only the covariance block. At successful main-solver steps we can also
reconstruct stage-1 log-Cholesky variables from the linearized stage-0 dynamics,
since stage 0 is fixed and its nominal input is the preceding applied input.
"""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import casadi as ca
import numpy as np
import torch
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_snapshot"
OUTPUT = ROOT / "clearance_diagnosis"
OUTPUT.mkdir(exist_ok=True)
sys.path.insert(0, str(SOURCE))

from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.dd_dynamics import DifferentialDriveDynamics, symbolic_dd_step


spec = importlib.util.spec_from_file_location("frozen_covariance", SOURCE / "control/cc_covariance.py")
covariance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(covariance)


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def make_functions(param):
    pose = ca.MX.sym("pose", 3)
    xi = ca.MX.sym("xi", 4)
    control = ca.MX.sym("input", 2)
    state = ca.vertcat(pose, xi)
    next_state = ca.vertcat(
        symbolic_dd_step(pose, control[0], control[1], .1),
        covariance.symbolic_log_cholesky_transition(xi, control[0], control[1], 0., .1, param),
    )
    dynamics = ca.Function("dd_and_covariance", [state, control],
                           [next_state, ca.jacobian(next_state, control)])
    terms = ca.MX.sym("risk_terms", 4)
    q_l = ca.sqrt(ca.fmax(terms[1], 0.))
    q_theta = ca.if_else(terms[3] < 0., -1., 1.) * ca.sqrt(ca.fmax(terms[2], 0.))
    variance = (ca.exp(2. * xi[0]) * terms[0]
                + (ca.exp(xi[1]) * q_l + xi[3] * q_theta) ** 2
                + ca.exp(2. * xi[2]) * terms[2])
    margin = param.chance_gamma * ca.sqrt(variance)
    margin = ca.fmin(margin, param.risk_margin_cap)
    risk = ca.Function("sf_margin", [xi, terms], [margin, ca.jacobian(margin, xi)])
    return dynamics, risk


first_overlaps = {}
for row in rows(ROOT / "geometry_violations.csv"):
    first_overlaps.setdefault((float(row["width_m"]), int(row["seed"])), row)

results, detail = [], []
for width in [.120, .125, .130]:
    folder = ROOT / f"width_{round(width * 1000):03d}mm"
    geometry = json.loads((folder / "geometry.json").read_text())
    footprint = np.asarray(geometry["footprint"])
    obstacles = [Polygon(vertices) for vertices in geometry["obstacles"]]
    param = SimpleNamespace(**json.loads((folder / "optimizer_parameters.json").read_text()))
    dynamics, risk = make_functions(param)
    wrapper = AugmentedPSDFWrapper(torch.tensor(footprint, dtype=torch.float32, device="cuda"),
                                   K_max=20, E_max=100, device="cuda")
    edges = [np.asarray(vertices) for vertices in geometry["obstacles"]]
    wrapper.update_edge_clusters(edges, [np.roll(vertices, -1, axis=0) for vertices in edges])
    nominal = {}
    with (folder / "nominal_risk.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "fast":
                nominal[(row["seed"], row["cycle"])] = row

    def geometry_at(pose):
        pose32 = torch.as_tensor(np.atleast_2d(pose), dtype=torch.float32, device="cuda")
        phi, gradient = wrapper(pose32)
        clusters, _ = wrapper.forward_clusters(pose32)
        return (float(phi[0].item()), gradient[0].cpu().numpy().astype(float),
                int(clusters[0].argmin().item()))

    def local_gradient(pose, gradient):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return np.array([c * gradient[0] + s * gradient[1],
                         -s * gradient[0] + c * gradient[1], gradient[2]])

    for seed in range(3):
        first = first_overlaps[(width, seed)]
        cycle = int(first["sample_index"]) - 1
        assert cycle > 0
        current = nominal[(seed, cycle)]
        following = nominal[(seed, cycle + 1)]
        assert current["status"] == 0
        history = rows(folder / "runs/config01" / f"trial_{seed + 1:04d}" / "data" /
                       f"trial_history_MPC_cc_sf_rectangle_maze_startseed{seed}.csv")
        before = np.asarray(current["states"][0][:3])
        after = np.array([float(history[cycle][key]) for key in ("x", "y", "theta")])
        predicted = np.asarray(following["states"][1][:3])
        anchor = np.asarray(current["states"][1][:3])
        anchor32 = anchor.astype(np.float32).astype(float)
        applied = np.array([float(history[cycle][key]) for key in ("v", "omega")])
        previous_input = np.array([float(history[cycle - 1][key]) for key in ("v", "omega")])
        xi0 = np.asarray(current["states"][0][4:8])
        xi_bar1 = np.asarray(current["states"][1][4:8])
        old_phi, gradient, old_cluster = geometry_at(anchor)
        predicted_phi, _, predicted_cluster = geometry_at(predicted)
        after_phi, _, after_cluster = geometry_at(after)
        assert abs(old_phi - current["phi_m"][1]) < 1e-7
        assert abs(predicted_phi - following["phi_m"][1]) < 1e-7
        assert abs(after_phi - following["phi_m"][0]) < 1e-7
        q = local_gradient(anchor, gradient)
        check_terms = np.array([q[0] ** 2, q[1] ** 2, q[2] ** 2, q[1] * q[2]])
        terms = np.asarray(current["risk_terms"][1])
        np.testing.assert_allclose(check_terms, terms, rtol=2e-4, atol=1e-7)

        nominal_map, input_jacobian = dynamics(np.r_[before, xi0], previous_input)
        nominal_map = np.asarray(nominal_map).reshape(-1)
        input_jacobian = np.asarray(input_jacobian)
        np.testing.assert_allclose(nominal_map[3:], xi_bar1, rtol=0, atol=1e-11)
        reconstructed = nominal_map + input_jacobian @ (applied - previous_input)
        pose_reconstruction_error = float(np.max(np.abs(reconstructed[:3] - predicted)))
        np.testing.assert_allclose(reconstructed[:3], predicted, rtol=0, atol=1e-8)
        reconstructed_xi = reconstructed[3:]
        physical = DifferentialDriveDynamics.forward_dynamics(before, applied, .1)
        np.testing.assert_array_equal(physical, after)
        regenerated_xi = covariance.propagate_log_cholesky(xi0, *applied, 0., .1, param)
        np.testing.assert_allclose(regenerated_xi, following["states"][1][4:8], rtol=0, atol=1e-10)
        margin_bar, margin_jacobian = risk(xi_bar1, terms)
        margin_bar = float(margin_bar)
        np.testing.assert_allclose(margin_bar, current["risk_margin_m"][1], rtol=0, atol=1e-12)
        margin_qp = margin_bar + float(np.asarray(margin_jacobian).reshape(-1) @
                                       (reconstructed_xi - xi_bar1))
        margin_at_returned_xi = float(risk(reconstructed_xi, terms)[0])
        margin_consistent = float(risk(regenerated_xi, terms)[0])
        affine_predicted = old_phi + gradient @ (predicted - anchor32)
        affine_after = old_phi + gradient @ (after - anchor32)
        qp_residual = affine_predicted - param.d_min - margin_qp

        cosine, sine = np.cos(after[2]), np.sin(after[2])
        polygon = Polygon(footprint @ np.array([[cosine, sine], [-sine, cosine]]) + after[:2])
        area = max(polygon.intersection(obstacle).area for obstacle in obstacles)
        assert area > 0
        record = {
            "width_m": width, "seed": seed, "input_cycle": cycle,
            "time_before_s": cycle * .1, "time_after_s": (cycle + 1) * .1,
            "fast_status": current["status"], "obstacle_clusters": len(edges),
            "nominal_cluster": old_cluster, "predicted_cluster": predicted_cluster,
            "actual_cluster": after_cluster,
            "nominal_stage1_phi_mm": old_phi * 1000,
            "affine_phi_at_predicted_mm": affine_predicted * 1000,
            "actual_psdf_at_predicted_mm": predicted_phi * 1000,
            "affine_phi_at_actual_mm": affine_after * 1000,
            "actual_psdf_at_actual_mm": after_phi * 1000,
            "nominal_margin_mm": margin_bar * 1000,
            "linearized_qp_margin_mm": margin_qp * 1000,
            "margin_at_reconstructed_returned_xi_mm": margin_at_returned_xi * 1000,
            "margin_with_consistent_covariance_mm": margin_consistent * 1000,
            "reconstructed_stage1_qp_residual_mm": qp_residual * 1000,
            "nonlinear_sf_residual_at_predicted_mm": (predicted_phi - param.d_min -
                                                        margin_at_returned_xi) * 1000,
            "geometry_taylor_overestimate_mm": (affine_predicted - predicted_phi) * 1000,
            "dynamics_translation_defect_mm": float(np.linalg.norm(predicted[:2] - after[:2])) * 1000,
            "nominal_to_returned_translation_mm": float(np.linalg.norm(predicted[:2] - anchor[:2])) * 1000,
            "nominal_to_returned_heading_rad": float(predicted[2] - anchor[2]),
            "rk4_replay_max_error": float(np.max(np.abs(physical - after))),
            "linearized_pose_reconstruction_max_error": pose_reconstruction_error,
            "actual_intersection_area_m2": area,
        }
        results.append(record)
        detail.append({**record, "before_pose": before.tolist(), "nominal_stage1_pose": anchor.tolist(),
            "returned_stage1_pose": predicted.tolist(), "actual_after_pose": after.tolist(),
            "nominal_stage1_gradient_world": gradient.tolist(),
            "nominal_input": previous_input.tolist(), "returned_input": applied.tolist(),
            "nominal_stage1_xi": xi_bar1.tolist(), "reconstructed_returned_stage1_xi": reconstructed_xi.tolist(),
            "consistent_stage1_xi": regenerated_xi.tolist(), "risk_terms": terms.tolist()})
        print(json.dumps(record), flush=True)

with (OUTPUT / "first_overlap_reconstruction.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
(OUTPUT / "first_overlap_details.json").write_text(json.dumps(detail, indent=2) + "\n")
hashes = json.loads((ROOT / "source_hashes.json").read_text())
assert all(hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
           for name, digest in hashes.items())
validation = {"new_navigation_or_solver_runs": 0, "analyzed_first_overlap_events": len(results),
    "frozen_sources_unchanged": True,
    "maximum_rk4_replay_error": max(row["rk4_replay_max_error"] for row in results),
    "maximum_linearized_pose_reconstruction_error": max(
        row["linearized_pose_reconstruction_max_error"] for row in results),
    "original_phi_risk_terms_and_margin_reproduced": True,
    "fresh_predicted_and_actual_psdf_match_original_following_cycle_logs": True,
    "limitations": ["Returned stage-1 covariance is reconstructed from the one-QP equality model, not directly logged.",
        "The directly recovered returned poses and independent intersection areas do not rely on that covariance reconstruction."]}
(OUTPUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
