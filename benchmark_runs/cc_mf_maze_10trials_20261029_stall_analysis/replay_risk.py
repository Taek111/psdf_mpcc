"""Offline MF risk replay on tail samples of completed stalled trials.

No navigation or optimizer solve is run. CPU evaluation avoids contending with
the active batch's GPU. Covariance-only replay keeps optimized poses fixed;
integrated replay additionally propagates the saved controls through DD dynamics.
"""

from collections import deque
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "source_snapshot"))
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
from control.cc_covariance import covariance_from_log_cholesky
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.dd import DifferentialDriveRectangleGeometry
from models.dd_dynamics import DifferentialDriveDynamics
from sim.simulation_mpc import simulation_mpc


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


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def masked_summary(data, slack=None):
    mask = np.asarray(data["mf_mask"], dtype=bool)
    mask[0] = False
    probability = np.asarray(data["probability_sum"], dtype=float)
    residual = np.asarray(data["residual"], dtype=float)
    evaluated = mask & np.isfinite(probability) & np.isfinite(residual)
    result = {
        "active_rows": int(mask.sum()),
        "active_row_stages": np.flatnonzero(mask),
        "mf_sum_max": float(probability[evaluated].max()) if evaluated.any() else None,
        "mf_sum_gt_epsilon_stages": np.flatnonzero(evaluated & (probability > .2 + 1e-6)),
        "quantile_residual_min": float(residual[evaluated].min()) if evaluated.any() else None,
        "guard_residual_min_m": float(np.min(data["phi"][1:] - .0002)),
        "stage1_mf_sum": probability[1],
        "stage1_mf_residual": residual[1],
        "stage1_phi_m": float(data["phi"][1]),
    }
    if slack is not None:
        effective = residual + slack
        result["quantile_with_saved_slack_min"] = float(effective[evaluated].min()) if evaluated.any() else None
        result["fresh_soft_violation_stages"] = np.flatnonzero(evaluated & (effective < -1e-6))
    return result


def main():
    start, goal, grid, obstacles = simulation_mpc().create_env("maze")
    geometry = DifferentialDriveRectangleGeometry(.15, .12, 0.)
    vertices = geometry._region.get_ccw_vertices()
    write_json(BASE / "environment_geometry.json", {
        "environment": "maze", "source": "source_snapshot/sim/simulation_mpc.py:create_env",
        "scale_m": .15, "corridor_width_m": .15,
        "bounds": grid[0], "cell_size_m": grid[1], "start": start, "goal": goal,
        "footprint_vertices_local": vertices,
        "footprint_length_m": .15, "footprint_width_m": .12, "rear_dist_m": 0.,
        "obstacles": [obs.get_ccw_vertices() for obs in obstacles],
    })
    param = CCMFOptimizerParam()
    for key, value in json.loads((BASE / "resolved_cc_mf_parameters.json").read_text()).items():
        setattr(param, key, value)
    # Retain same float32 geometry buffers and float64 MF arithmetic as the
    # online wrapper, using the CPU. Compare original replay with online cache.
    optimizer = CCMFOptimizer()
    optimizer.param = param
    optimizer.N = int(param.horizon)
    optimizer.device = torch.device("cpu")
    optimizer.psdf_wrapper = AugmentedPSDFWrapper(
        torch.tensor(vertices, dtype=torch.float32), K_max=20, E_max=100, device="cpu")
    optimizer.update_obstacles(obstacles)
    dt = float(param.tf) / int(param.horizon)
    with (BASE / "results/trial_results.csv").open(newline="") as handle:
        outcomes = list(csv.DictReader(handle))
    examples = []
    for outcome in outcomes:
        if not outcome["failure_reason"].startswith("stuck_"):
            continue
        trial = int(outcome["trial_number"])
        with (BASE / f"telemetry_trial{trial:02d}.jsonl").open() as handle:
            tail = deque((json.loads(line) for line in handle), maxlen=4)
        for record in tail:
            if record["status"] != 0 or record["diagnostics"] is None:
                continue
            optimizer.state = SimpleNamespace(_x=np.asarray(record["pose"]))
            optimized = np.asarray(record["states"], dtype=float)[:, :8]
            inputs = np.asarray(record["inputs"], dtype=float)
            original = optimizer._compute_constraint_affine_data(optimized)
            corrected = optimized.copy()
            corrected[:, 4:8] = optimizer._propagate_covariance_state_batch(inputs, corrected[:, 3])
            exact_cov = optimizer._compute_constraint_affine_data(corrected)
            integrated = corrected.copy()
            for stage in range(optimizer.N):
                integrated[stage + 1, :3] = DifferentialDriveDynamics.forward_dynamics(
                    integrated[stage, :3], inputs[stage, :2], dt)
            integrated_risk = optimizer._compute_constraint_affine_data(integrated)
            d = record["diagnostics"]
            slack = np.asarray(d["mf_lower_slack"], dtype=float)
            mask = np.asarray(d["mf_mask"], bool)
            affine_effective = np.asarray(d["mf_affine_residual"], float) + slack
            logged_probability = np.asarray(d["mf_probability_sum_fresh"], float)
            original_probability = np.asarray(original["probability_sum"], float)
            overlap = np.isfinite(logged_probability) & np.isfinite(original_probability)
            original_sigma = np.sqrt(np.diagonal(covariance_from_log_cholesky(optimized[:, 4:8]), axis1=-2, axis2=-1))
            corrected_sigma = np.sqrt(np.diagonal(covariance_from_log_cholesky(corrected[:, 4:8]), axis1=-2, axis2=-1))
            example = {
                "trial": trial, "seed": int(outcome["start_seed"]),
                "time_s": record["time"], "step": record["solve_step"],
                "status": record["status"], "mode": record["mode"],
                "applied_v_m_s": inputs[0, 0], "applied_omega_rad_s": inputs[0, 1],
                "nominal_u0": np.asarray(record["nominal_inputs"])[0, :2],
                "affine_guard_min_m": np.nanmin(np.asarray(d["guard_affine_residual"], float)),
                "affine_soft_effective_min": np.nanmin(affine_effective[mask]),
                "max_saved_slack": np.nanmax(slack),
                "cpu_vs_logged_mf_sum_max_abs_error": float(np.max(np.abs(logged_probability[overlap] - original_probability[overlap]))),
                "optimized_covariance_at_optimized_poses": masked_summary(original, slack),
                "exact_covariance_at_optimized_poses": masked_summary(exact_cov, slack),
                "exact_integrated_controls_and_covariance": masked_summary(integrated_risk, slack),
                "stage1_optimized_sigma_f_l_psi": original_sigma[1],
                "stage1_exact_sigma_f_l_psi": corrected_sigma[1],
                "stage1_heading_sigma_ratio_optimized_over_exact": original_sigma[1, 2] / corrected_sigma[1, 2],
                "stage1_optimized_pose_minus_integrated_pose": optimized[1, :3] - integrated[1, :3],
                "max_predicted_xy_pose_difference_m": np.max(np.linalg.norm(optimized[:, :2] - integrated[:, :2], axis=1)),
                "probability_sum_arrays": {
                    "optimized": original["probability_sum"],
                    "exact_covariance_only": exact_cov["probability_sum"],
                    "exact_integrated": integrated_risk["probability_sum"],
                },
                "quantile_residual_arrays": {
                    "optimized": original["residual"],
                    "exact_covariance_only": exact_cov["residual"],
                    "exact_integrated": integrated_risk["residual"],
                },
            }
            examples.append(example)
    optimizer.cleanup()
    result = {
        "method": __doc__,
        "scope": "Last four successful cycles of each completed movement-stall trial; no new solver calls.",
        "device": "cpu; float32 geometry buffers, float64 MF calculations",
        "chance_epsilon": .2, "d_col_m": .0002,
        "probability_note": "MF probability_sum is a feature probability sum/risk surrogate, not a measured collision probability.",
        "completed_trial_count": len(outcomes), "examples": examples,
    }
    write_json(BASE / "risk_replay.json", result)
    print(json.dumps({"completed_trials": len(outcomes), "examples": len(examples),
                      "trials": sorted({r["trial"] for r in examples}),
                      "max_cpu_vs_logged_error": max((r["cpu_vs_logged_mf_sum_max_abs_error"] for r in examples), default=None)}, indent=2))


if __name__ == "__main__":
    main()
