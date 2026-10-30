"""Reproduce the covariance tangent across the observed trial-2 limit cycle.

No solver is instantiated: CasADi differentiates the saved covariance map.
The nominal input is the retained previous cycle's input. The next cycle's
nominal covariance is reconstructed according to _prepare_and_solve's exact
rollout, not directly logged as a separate nominal covariance telemetry field.
"""

import json
from pathlib import Path

import numpy as np

import analysis_covariance as audit


OUTPUT = Path(__file__).resolve().parent
records = [json.loads(line) for line in
           (OUTPUT / "telemetry_trial02.jsonl").read_text().splitlines()]
examples = []
for target_time in (18.6, 18.7, 18.8):
    k = min(range(len(records)), key=lambda i: abs(records[i]["time"] - target_time))
    previous, record, next_record = records[k - 1:k + 2]
    states = np.asarray(record["states"], dtype=float)
    nominal_input = np.asarray(previous["inputs"][0][:2], dtype=float)
    actual_input = np.asarray(record["inputs"][0][:2], dtype=float)
    nominal_next, jacobian = audit.COVARIANCE_LINEARIZATION(states[0, 4:8], nominal_input)
    nominal_next = np.asarray(nominal_next).ravel()
    jacobian = np.asarray(jacobian)
    tangent_next = nominal_next + jacobian @ (actual_input - nominal_input)
    exact_next = audit.covariance.propagate_log_cholesky(
        states[0, 4:8], *actual_input, 0.0, audit.DT, audit.PARAM)
    next_nominal_covariance = audit.covariance.propagate_log_cholesky(
        np.asarray(next_record["states"])[0, 4:8],
        *actual_input, 0.0, audit.DT, audit.PARAM)

    def sigmas(xi):
        return np.sqrt(np.diag(audit.covariance.covariance_from_log_cholesky(xi))).tolist()

    physical_next = audit.DifferentialDriveDynamics.forward_dynamics(
        states[0, :3], actual_input, audit.DT)
    examples.append({
        "trial": 2, "time": record["time"], "status": record["status"],
        "mode": record["mode"], "nominal_u0": nominal_input.tolist(),
        "actual_u0": actual_input.tolist(),
        "delta_u0": (actual_input - nominal_input).tolist(),
        "nominal_stage1_sigma_f_l_psi": sigmas(nominal_next),
        "optimized_stage1_sigma_f_l_psi": sigmas(states[1, 4:8]),
        "exact_stage1_sigma_f_l_psi": sigmas(exact_next),
        "next_cycle_time": next_record["time"],
        "next_cycle_reconstructed_nominal_stage1_sigma_f_l_psi": sigmas(next_nominal_covariance),
        "optimized_xi_stage1": states[1, 4:8].tolist(),
        "tangent_xi_stage1": tangent_next.tolist(),
        "exact_xi_stage1": exact_next.tolist(),
        "optimized_minus_tangent": (states[1, 4:8] - tangent_next).tolist(),
        "optimized_minus_exact": (states[1, 4:8] - exact_next).tolist(),
        "nominal_xi_transition_jacobian_wrt_v_omega": jacobian.tolist(),
        "physical_stage1_pose_defect": (states[1, :3] - physical_next).tolist(),
        "reported_nlp_equality_residual": record["diagnostics"]["solver"]["nlp_residuals"][1],
        "current_nominal_stage1_probability": record["nominal"]["probability_sum"][1],
        "next_nominal_stage1_probability": next_record["nominal"]["probability_sum"][1],
    })

result = {
    "method": __doc__,
    "parameter_source": str(OUTPUT / "source_snapshot/control/cc_mf_optimizer.py"),
    "covariance_map_source": str(OUTPUT / "source_snapshot/control/cc_covariance.py"),
    "sigma_units": ["m", "m", "rad"],
    "examples": examples,
    "interpretation": "The optimized stage-1 covariance matches the prior-input tangent"
                      " to machine precision while violating the nonlinear map. Each"
                      " angular sign flip changes omega by approximately its 0.35 rad/s"
                      " per-step rate limit. The omega-squared noise tangent treats"
                      " reversal as a large uncertainty reduction. The following cycle"
                      " restores the true uncertainty in nominal preprocessing.",
}
(OUTPUT / "covariance_mechanism_examples.json").write_text(
    json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"maximum_tangent_error": max(
    max(abs(value) for value in example["optimized_minus_tangent"])
    for example in examples), "examples": len(examples)}, indent=2))
