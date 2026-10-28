"""Offline analysis of recorded CC-MF runs; does not run a solver or simulation.

Run with the recorded Python environment. Writes only failure_timeline.csv,
covariance_evidence.csv and log_evidence.json next to this script.
"""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


OUTPUT = Path(__file__).resolve().parent
RUN = OUTPUT.parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT))

from control.cc_covariance import (  # noqa: E402
    covariance_from_log_cholesky,
    initialize_cc_noise_params,
    log_cholesky_from_covariance,
    propagate_log_cholesky,
)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUTPUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_values(values):
    return [value for value in values if value is not None and math.isfinite(value)]


def main():
    environment = json.loads((RUN / "environment.json").read_text())
    covariance_path = ROOT / "control/cc_covariance.py"
    covariance_hash = hashlib.sha256(covariance_path.read_bytes()).hexdigest()
    recorded_covariance_hash = environment["source_sha256"]["control/cc_covariance.py"]
    if covariance_hash != recorded_covariance_hash:
        raise RuntimeError("Current covariance source differs from the recorded run.")

    dt = 0.1
    initial_sigmas = np.array([0.0002, 0.00025, 0.00025])
    param = SimpleNamespace(q_f0=0.0, q_l0=0.0, q_psi0=0.0)
    initialize_cc_noise_params(param)
    initial_covariance = np.diag(initial_sigmas**2)
    xi0 = log_cholesky_from_covariance(initial_covariance)
    timeline = []
    covariance_rows = []
    trial_summaries = []

    for directory in sorted((RUN / "runs/config01").glob("trial_*")):
        trial = int(directory.name.split("_")[-1])
        data_dir = directory / "data"
        history_path = next(data_dir.glob("trial_history_*.csv"))
        pose_path = next(data_dir.glob("pose_sdf_*.csv"))
        history = read_csv(history_path)
        poses = read_csv(pose_path)
        cycles = read_csv(data_dir / "cycle_raw.csv")
        covariance = read_csv(data_dir / "covariance_trajectory.csv")
        start = json.loads(next(data_dir.glob("start_pose_*.json")).read_text())
        seed = start["seed"]
        stage_count = len(covariance) // len(cycles)
        assert stage_count == 21 and len(covariance) == stage_count * len(cycles)
        assert len(history) == len(poses) == len(cycles)
        alignment_error = max(
            abs(float(poses[k][axis]) - float(history[k - 1][axis]))
            for k in range(1, len(poses)) for axis in ("x", "y", "theta")
        )
        assert alignment_error == 0.0
        assert all(float(c["psdf"]) == float(p["sdf_value"]) for c, p in zip(cycles, poses))
        failures = [k for k, row in enumerate(history) if row["solver_success"] != "True"]
        first_failure = failures[0] if failures else None
        first_negative = next((k for k, row in enumerate(poses) if float(row["sdf_value"]) < 0), None)
        first_low = next((k for k, row in enumerate(poses) if float(row["sdf_value"]) < 0.001), None)
        nominal_probabilities = [json.loads(row["mf_probability_sum"]) for row in cycles]
        candidate_slacks = [json.loads(row["mf_slack"]) for row in cycles]
        first7 = [row for row in history if float(row["time"]) <= 7.0 + 1e-9]
        first7_pre = [row for k, row in enumerate(poses) if float(cycles[k]["time"]) < 7.0 - 1e-9]
        summary = {
            "trial": trial,
            "seed": seed,
            "status": "failure" if failures else "success",
            "initial_pose": start["initial_pose"],
            "initial_clearance_m": start["initial_clearance"],
            "navigation_time_s": float(history[-1]["time"]),
            "first_failed_cycle": first_failure,
            "first_below_1mm_cycle": first_low,
            "first_negative_psdf_cycle": first_negative,
            "safe_stop_count": sum(row["solve_mode"].startswith("safe_stop") for row in cycles),
            "min_logged_precontrol_psdf_m": min(float(row["sdf_value"]) for row in poses),
            "first_7s_max_abs_applied_v_mps": max(abs(float(row["v"])) for row in first7),
            "first_7s_min_logged_precontrol_psdf_m": min(float(row["sdf_value"]) for row in first7_pre),
            "max_candidate_mf_slack": max(value for row in candidate_slacks for value in finite_values(row)),
            "cycles_candidate_mf_slack_above_1e_4": sum(any(value > 1e-4 for value in finite_values(row)) for row in candidate_slacks),
            "max_nominal_mf_probability_sum": max(value for row in nominal_probabilities for value in finite_values(row)),
            "cycles_nominal_stage1_probability_sum_above_0_2": sum(row[1] is not None and row[1] > 0.2 for row in nominal_probabilities),
            "max_logged_candidate_sigma_l_m": max(float(row["sigma_l"]) for row in covariance),
            "pre_vs_previous_post_pose_max_abs_error": alignment_error,
        }
        trial_summaries.append(summary)
        if first_failure is None:
            continue

        # Last five successful inputs and the first safe-stop input.
        for k in range(max(0, first_failure - 5), first_failure + 1):
            row, pose, cycle = history[k], poses[k], cycles[k]
            slacks = finite_values(candidate_slacks[k])
            probabilities = finite_values(nominal_probabilities[k])
            timeline.append({
                "trial": trial,
                "seed": seed,
                "cycle_index": k,
                "source_csv_line": k + 2,
                "control_start_time_s": float(cycle["time"]),
                "plant_post_time_s": float(row["time"]),
                "solver_success": row["solver_success"],
                "solve_mode": cycle["solve_mode"],
                "pre_psdf_m": float(pose["sdf_value"]),
                "post_psdf_m_from_next_precontrol_log": float(poses[k + 1]["sdf_value"]),
                "pre_x_m": float(pose["x"]),
                "pre_y_m": float(pose["y"]),
                "pre_theta_rad": float(pose["theta"]),
                "post_x_m": float(row["x"]),
                "post_y_m": float(row["y"]),
                "post_theta_rad": float(row["theta"]),
                "applied_v_mps": float(row["v"]),
                "applied_omega_radps": float(row["omega"]),
                "delta_v_from_previous_cycle_mps": float(row["v"]) - float(history[k - 1]["v"]),
                "delta_omega_from_previous_cycle_radps": float(row["omega"]) - float(history[k - 1]["omega"]),
                "candidate_mf_slack_max": max(slacks) if slacks else None,
                "candidate_mf_slack_stage1": candidate_slacks[k][1],
                "candidate_mf_slack_stage2": candidate_slacks[k][2],
                "nominal_mf_probability_sum_stage1": nominal_probabilities[k][1],
                "nominal_mf_probability_sum_max": max(probabilities) if probabilities else None,
                "slice_start_index": pose["slice_start_idx"],
                "slice_start_s": pose["slice_start_s"],
            })

        k = first_failure - 1
        v, omega = float(history[k]["v"]), float(history[k]["omega"])
        previous_v = float(history[k - 1]["v"])
        previous_omega = float(history[k - 1]["omega"])
        candidate_stage0 = covariance[k * stage_count]
        candidate_stage1 = covariance[k * stage_count + 1]
        assert int(candidate_stage0["stage"]) == 0
        assert int(candidate_stage1["stage"]) == 1
        assert np.allclose([float(candidate_stage0[key]) for key in ("sigma_f", "sigma_l", "sigma_psi")], initial_sigmas, atol=1e-15, rtol=1e-10)
        assert np.allclose([float(candidate_stage0["v"]), float(candidate_stage0["omega"])], [v, omega])
        # Curvature is unavailable in CSV. With nonnegative coefficients and |kappa|,
        # kappa=0 is a lower bound for lateral/heading variance and exact for forward.
        expected_xi = propagate_log_cholesky(xi0, v, omega, 0.0, dt, param)
        expected_sigmas = np.sqrt(np.diag(covariance_from_log_cholesky(expected_xi)))
        logged_sigmas = np.array([float(candidate_stage1[key]) for key in ("sigma_f", "sigma_l", "sigma_psi")])

        # Physical inputs are not shifted before the next RTI preparation, so the
        # previous cycle's stage-0 input is the stage-0 nominal input. This formula
        # reproduces the affine dynamics row for xi_f with fixed initial covariance.
        nominal_forward_variance = (
            initial_sigmas[0]**2
            + (dt * previous_omega)**2 * initial_sigmas[1]**2
            + dt * param.alpha_f * previous_v**2
        )
        nominal_xi_f = 0.5 * math.log(nominal_forward_variance)
        gradient_v = dt * param.alpha_f * previous_v / nominal_forward_variance
        gradient_omega = dt**2 * previous_omega * initial_sigmas[1]**2 / nominal_forward_variance
        tangent_xi_f = nominal_xi_f + gradient_v * (v - previous_v) + gradient_omega * (omega - previous_omega)
        tangent_sigma_f = math.exp(tangent_xi_f)
        covariance_rows.append({
            "trial": trial,
            "seed": seed,
            "cycle_index": k,
            "control_start_time_s": float(cycles[k]["time"]),
            "covariance_stage1_csv_line": k * stage_count + 3,
            "v_mps": v,
            "omega_radps": omega,
            "previous_v_mps": previous_v,
            "previous_omega_radps": previous_omega,
            "candidate_sigma_f_m": logged_sigmas[0],
            "candidate_sigma_l_m": logged_sigmas[1],
            "candidate_sigma_psi_rad": logged_sigmas[2],
            "nonlinear_sigma_f_exact_m": expected_sigmas[0],
            "nonlinear_sigma_l_lower_bound_m": expected_sigmas[1],
            "nonlinear_sigma_psi_lower_bound_rad": expected_sigmas[2],
            "candidate_to_exact_sigma_f_ratio": logged_sigmas[0] / expected_sigmas[0],
            "candidate_to_lower_bound_sigma_l_ratio": logged_sigmas[1] / expected_sigmas[1],
            "candidate_to_lower_bound_sigma_psi_ratio": logged_sigmas[2] / expected_sigmas[2],
            "tangent_nominal_xi_f": nominal_xi_f,
            "tangent_d_xi_f_d_v": gradient_v,
            "tangent_d_xi_f_d_omega": gradient_omega,
            "tangent_predicted_xi_f": tangent_xi_f,
            "tangent_predicted_sigma_f_m": tangent_sigma_f,
            "tangent_vs_logged_xi_f_error": tangent_xi_f - math.log(logged_sigmas[0]),
            "nonlinear_minus_candidate_xi_f_defect": float(expected_xi[0]) - math.log(logged_sigmas[0]),
        })

    write_csv("failure_timeline.csv", timeline)
    write_csv("covariance_evidence.csv", covariance_rows)
    evidence = {
        "analysis_type": "offline CSV analysis; no solver calls, new trials, or controller changes",
        "run_directory": str(RUN),
        "covariance_source_sha256": covariance_hash,
        "covariance_source_matches_run": covariance_hash == recorded_covariance_hash,
        "dt_s": dt,
        "initial_sigmas": initial_sigmas.tolist(),
        "noise_parameters": vars(param),
        "timing_alignment": "pose_sdf/cycle_raw are pre-control; history is post-update. pre[i] equals history[i-1] exactly for all trials.",
        "log_semantics": {
            "mf_probability_sum": "Nominal geometry/covariance probability sum from _last_nominal_probability_sum, not nonlinear candidate re-evaluation. It is a Boole sum that may exceed one; approximately -2.98e-9 is floating-point residue and has not been clipped in this analysis.",
            "mf_slack": "Optimized candidate lower slack; safe-stop has null, not zero, because its constructed plan has no optimized slack.",
            "covariance": "Reconstructed from returned solver trajectory. No factor clipping or minimum variance floor is applied by covariance_from_log_cholesky; logging uses sqrt(max(diagonal,0)), which does not affect these strictly positive entries.",
            "psdf_minimum": "Minimum over logged pre-control poses, not a continuous-time geometric minimum. The final post-update state of a successful trial has no subsequent pre-control PSDF row.",
        },
        "covariance_interpretation": {
            "kappa_zero": "Exact forward sigma; lateral and marginal heading sigmas are lower bounds since curvature enters as nonnegative |kappa| process noise.",
            "tangent_reproduction": "Uses preceding cycle stage-0 input as nominal. Forward log-variance tangent predicts both observed collapsed stage-1 sigma_f values to floating-point precision.",
            "limitation": "Candidate pose horizons and exact nonlinear guard/chance residuals are not stored. This identifies an affine covariance dynamics defect, but cannot quantify each mechanism's contribution to the collision or certify a tuning remedy.",
        },
        "source_references": {
            "covariance_transition": "control/cc_covariance.py:171-219",
            "nominal_probability_log": "control/cc_mf_optimizer.py:1446-1448 (current source; compare archived source for line stability)",
            "candidate_covariance_log": "control/cc_mf_optimizer.py:1453-1491 (current source)",
        },
        "trials": trial_summaries,
        "failure_covariance": covariance_rows,
    }
    (OUTPUT / "log_evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False))
    print(json.dumps({"timeline_rows": len(timeline), "covariance_rows": covariance_rows, "trials": trial_summaries}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
