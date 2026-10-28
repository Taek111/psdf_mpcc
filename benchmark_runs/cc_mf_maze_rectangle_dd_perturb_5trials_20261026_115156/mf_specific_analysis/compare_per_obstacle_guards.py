"""Compare geometry planes on identical reconstructed MF state pairs.

This evaluates the per-polygon affine PSDF geometry component used by current
CC-ECS. It does not run ECS, reconstruct an ECS solution, or predict its full
QP result. In particular ECS covariance margins and their SQP linearization
are not included in this geometry-only comparison.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

import reconstruct_stage1 as reconstruction


OUT = Path(__file__).resolve().parent
RUN = OUT.parent
ROOT = RUN.parent.parent


def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("geometry_analysis", RUN/"failure_analysis/geometry_analysis.py")
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    _, _, _, obstacles = geometry.load_environment()
    robot = geometry.rectangle(-.075, .075, -.05, .05)
    data = json.loads((OUT/"stage1_reconstruction.json").read_text())
    results = []
    for record in data["trials"]:
        nominal = np.array(record["reconstructed_nominal_pose_stage1"])
        candidate = np.array(record["reconstructed_candidate_pose_stage1"])
        actual = np.array(record["actual_next_pose"])
        row = {"trial": record["trial"], "seed": record["seed"], "time_s": record["time_s"],
               "nominal_pose": nominal.tolist(), "candidate_pose": candidate.tolist(),
               "actual_next_pose": actual.tolist(), "evaluations": {}}
        for precision, dtype in (("float32", torch.float32), ("float64", torch.float64)):
            psdf = reconstruction.wrapper(dtype, obstacles, robot)
            pose_tensor = torch.tensor(np.stack([nominal, candidate, actual]), dtype=dtype)
            cluster_phi, cluster_gradient = psdf.forward_clusters(pose_tensor)
            cluster_phi = cluster_phi.detach().cpu().numpy().astype(float)
            cluster_gradient = cluster_gradient.detach().cpu().numpy().astype(float)
            phi, gradient = psdf(pose_tensor[:1])
            phi = float(phi[0])
            gradient = gradient[0].detach().cpu().numpy().astype(float)
            per_polygon_affine_phi = cluster_phi[0]+cluster_gradient[0]@(candidate-nominal)
            per_polygon_geometry_residual = per_polygon_affine_phi-.001
            selected_index = int(np.argmin(cluster_phi[0]))
            global_min_affine_phi = phi+gradient@(candidate-nominal)
            obstacle_rows = []
            for index in range(len(obstacles)):
                obstacle_rows.append({
                    "obstacle_index": index, "vertices": obstacles[index].tolist(),
                    "is_nominal_global_min": index == selected_index,
                    "nominal_psdf_mm": cluster_phi[0, index]*1000,
                    "world_gradient_at_nominal": cluster_gradient[0, index].tolist(),
                    "affine_psdf_at_mf_candidate_mm": per_polygon_affine_phi[index]*1000,
                    "geometry_residual_at_mf_candidate_mm": per_polygon_geometry_residual[index]*1000,
                    "exact_psdf_at_mf_candidate_mm": cluster_phi[1, index]*1000,
                    "exact_psdf_at_actual_next_pose_mm": cluster_phi[2, index]*1000,
                })
            row["evaluations"][precision] = {
                "nominal_global_min_obstacle_index": selected_index,
                "mf_global_min_guard_affine_psdf_mm": float(global_min_affine_phi*1000),
                "mf_global_min_guard_geometry_residual_mm": float((global_min_affine_phi-.001)*1000),
                "per_obstacle_geometry_violations": [item for item in obstacle_rows if item["geometry_residual_at_mf_candidate_mm"] < -.1],
                "minimum_per_obstacle_geometry_residual_mm": float(per_polygon_geometry_residual.min()*1000),
                "obstacle_rows": obstacle_rows,
            }
        results.append(row)
    source_files = ["control/cc_ecs_optimizer.py", "models/augmented_psdf_wrapper.py", "models/augmented_psdf.py"]
    payload = {
        "scope": __doc__,
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files},
        "implementation_evidence": {
            "per_obstacle_geometry_evaluation": "control/cc_ecs_optimizer.py:_evaluate_cluster_geometry uses forward_clusters over all obstacle polygons.",
            "affine_plane": "_set_ecs_parameters forms one plane [gradient, distance-gradient@nominal_pose] per obstacle at every future stage.",
            "full_row": "The current ECS full row additionally subtracts d_min and a symbolic covariance support margin. Only plane-d_min is evaluated here.",
            "default_covariance_difference": "Current CCECSOptimizerParam sets all process-noise coefficients to zero, unlike CC-MF. No assumption of identical covariance models is made.",
        },
        "d_min_m": .001,
        "geometric_violation_tolerance_m": .0001,
        "limitations": "This is a geometry-component cross-evaluation on MF's reconstructed nominal/candidate pair, not an ECS counterfactual trajectory or proof about the full ECS QP. Per-obstacle planes also remain local approximations within each polygon.",
        "trials": results,
    }
    (OUT/"per_obstacle_guard_comparison.json").write_text(json.dumps(payload, indent=2))
    for record in results:
        evaluation = record["evaluations"]["float32"]
        print(json.dumps({"trial": record["trial"],
                          "global_guard_residual_mm": evaluation["mf_global_min_guard_geometry_residual_mm"],
                          "per_obstacle_violations": evaluation["per_obstacle_geometry_violations"]}, indent=2))


if __name__ == "__main__":
    main()
