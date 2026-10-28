"""Audit competing distance branches against independent polygon geometry.

Only evaluates saved poses; no controller, simulation, solver, or benchmark run.
"""

import csv
import json
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import Polygon
import torch


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "clearance_diagnosis"
sys.path.insert(0, str(ROOT / "source_snapshot"))
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper


def transform(footprint, pose):
    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
    return footprint @ np.array([[cosine, sine], [-sine, cosine]]) + pose[:2]


def independent_signed(robot, obstacle):
    """Euclidean separation or negative minimum SAT translation for convex sets."""
    edges = np.vstack((np.roll(robot, -1, axis=0) - robot,
                       np.roll(obstacle, -1, axis=0) - obstacle))
    axes = np.column_stack((-edges[:, 1], edges[:, 0]))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    first, second = robot @ axes.T, obstacle @ axes.T
    overlaps = np.minimum(first.max(0) - second.min(0), second.max(0) - first.min(0))
    if (overlaps < 0).any():
        return Polygon(robot).distance(Polygon(obstacle))
    return -float(np.min(overlaps))


details = json.loads((OUTPUT / "first_overlap_details.json").read_text())
audits = []
primary = None
for record in details:
    width = record["width_m"]
    geometry = json.loads((ROOT / f"width_{round(width * 1000):03d}mm/geometry.json").read_text())
    footprint = np.asarray(geometry["footprint"])
    obstacles = [np.asarray(vertices) for vertices in geometry["obstacles"]]
    wrapper = AugmentedPSDFWrapper(torch.tensor(footprint, dtype=torch.float64),
                                   K_max=20, E_max=4, device="cpu")
    wrapper.update_edge_clusters(obstacles, [np.roll(vertices, -1, axis=0) for vertices in obstacles])
    poses = np.array([record["nominal_stage1_pose"], record["returned_stage1_pose"],
                      record["actual_after_pose"]])
    distances, gradients = wrapper.forward_clusters(torch.tensor(poses))
    distances, gradients = distances.numpy(), gradients.numpy()
    reference = np.array([[independent_signed(transform(footprint, pose), obstacle)
                           for obstacle in obstacles] for pose in poses])
    geometry_error = float(np.max(np.abs(distances - reference)))
    assert geometry_error < 1e-7
    active = int(distances[0].argmin())
    fd = []
    for axis, epsilon in enumerate([1e-7, 1e-7, 1e-6]):
        increment = np.zeros(3)
        increment[axis] = epsilon
        plus = independent_signed(transform(footprint, poses[0] + increment), obstacles[active])
        minus = independent_signed(transform(footprint, poses[0] - increment), obstacles[active])
        fd.append((plus - minus) / (2 * epsilon))
    fd = np.asarray(fd)
    gradient_error = np.abs(fd - gradients[0, active])
    assert np.max(gradient_error) < 1e-4
    float32_error = max(abs(float(np.min(distances[index])) * 1000 - expected) for index, expected in
        [(0, record["nominal_stage1_phi_mm"]), (1, record["actual_psdf_at_predicted_mm"]),
         (2, record["actual_psdf_at_actual_mm"])])
    assert float32_error < .001
    audits.append({"width_m": width, "seed": record["seed"], "active_cluster": active,
        "independent_signed_distance_max_error_mm": geometry_error * 1000,
        "float32_vs_float64_max_error_mm": float32_error,
        "analytic_gradient": gradients[0, active].tolist(),
        "independent_finite_difference_gradient": fd.tolist(),
        "absolute_gradient_error": gradient_error.tolist()})

    if width == .120 and record["seed"] == 0:
        delta = poses[1] - poses[0]
        translated_mm = float(np.linalg.norm(delta[:2])) * 1000
        tangent1 = distances[0, 1] + gradients[0, 1] @ delta
        tangent4 = distances[0, 4] + gradients[0, 4] @ delta
        low, high = 0., 1.
        for _ in range(45):
            middle = (low + high) / 2
            vertices = transform(footprint, poses[0] + middle * delta)
            difference = independent_signed(vertices, obstacles[1]) - independent_signed(vertices, obstacles[4])
            if difference <= 0:
                low = middle
            else:
                high = middle
        crossing = (low + high) / 2
        primary = {"width_m": width, "seed": 0,
            "per_obstacle_signed_distances_mm": {
                name: {"obstacle1": float(distances[index, 1] * 1000),
                       "obstacle4": float(distances[index, 4] * 1000)}
                for index, name in enumerate(["anchor", "returned", "actual"])},
            "anchor_competitor_gap_mm": float((distances[0, 4] - distances[0, 1]) * 1000),
            "obstacle1_tangent_at_returned_mm": float(tangent1 * 1000),
            "obstacle1_smooth_linearization_error_mm": float((tangent1 - distances[1, 1]) * 1000),
            "obstacle4_tangent_at_returned_mm": float(tangent4 * 1000),
            "obstacle4_smooth_linearization_error_mm": float((tangent4 - distances[1, 4]) * 1000),
            "returned_motion_translation_mm": translated_mm,
            "returned_motion_heading_deg": float(np.rad2deg(delta[2])),
            "first_minimum_switch_fraction": crossing,
            "translation_until_minimum_switch_mm": crossing * translated_mm,
            "original_float32_taylor_to_actual_total_gap_mm": record["affine_phi_at_predicted_mm"] - record["actual_psdf_at_actual_mm"],
            "original_float32_geometry_gap_mm": record["geometry_taylor_overestimate_mm"],
            "original_float32_kinematics_signed_distance_change_mm": record["actual_psdf_at_predicted_mm"] - record["actual_psdf_at_actual_mm"],
            "wall_x_m": .6,
            "rightmost_robot_x_m": {name: float(transform(footprint, pose)[:, 0].max())
                                      for name, pose in zip(["anchor", "returned", "actual"], poses)},
            "obstacle1_gradient_world": gradients[0, 1].tolist(),
            "obstacle4_gradient_world": gradients[0, 4].tolist()}
        fractions = np.unique(np.r_[np.linspace(0, 1, 501), crossing])
        line = poses[0] + fractions[:, None] * delta
        branch_distances, _ = wrapper.forward_clusters(torch.tensor(line))
        branch_distances = branch_distances.numpy()
        assert set(branch_distances.argmin(1)) == {1, 4}
        samples = []
        for fraction, values in zip(fractions, branch_distances):
            pose = poses[0] + fraction * delta
            samples.append({"fraction": float(fraction), "translation_from_anchor_mm": float(fraction * translated_mm),
                "obstacle1_phi_mm": float(values[1] * 1000), "obstacle4_phi_mm": float(values[4] * 1000),
                "global_phi_mm": float(values.min() * 1000),
                "frozen_tangent_mm": float((distances[0, 1] + gradients[0, 1] @ (pose - poses[0])) * 1000),
                "active_obstacle": int(values.argmin())})
        with (OUTPUT / "primary_branch_path.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)

result = {"new_navigation_or_solver_runs": 0, "events_audited": len(audits),
    "independent_distance_checks": len(audits) * 3 * 17,
    "primary_case": primary, "gradient_and_geometry_audits": audits,
    "max_independent_distance_error_mm": max(row["independent_signed_distance_max_error_mm"] for row in audits),
    "max_float32_vs_float64_error_mm": max(row["float32_vs_float64_max_error_mm"] for row in audits),
    "max_analytic_gradient_error": max(max(row["absolute_gradient_error"]) for row in audits)}
(OUTPUT / "branch_audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if key != "gradient_and_geometry_audits"}, indent=2))
