"""Reproduce the CSV/source audit for this inverse-CDF validation run."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import ndtri
import yaml


RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[1]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


config = yaml.safe_load((RUN / "navigation_config.yaml").read_text())
epsilon = config["cc_mf_icdf"]["chance_epsilon"]
trials = read_csv(RUN / "navigation/trial_results.csv")
navigation = []
for trial in trials:
    data = RUN / "navigation/runs" / trial["configuration_id"] / "trial_0001/data"
    cycles = read_csv(data / f"cc_mf_icdf_cycle_{trial['current_name']}.csv")
    branches = Counter()
    residual_error = []
    for row in cycles:
        probability = np.asarray(json.loads(row["mf_probability_sum"]), dtype=float)
        residual = np.asarray(json.loads(row["mf_icdf_residual_nominal"]), dtype=float)
        active = np.asarray(json.loads(row["mf_active"]), dtype=bool)
        expected = np.where(
            probability > .5,
            -np.sqrt(2 * np.pi) * (probability - .5) + ndtri(epsilon),
            -ndtri(np.clip(probability, np.finfo(float).tiny, .5)) + ndtri(epsilon),
        )
        residual_error.extend(np.abs(expected[active] - residual[active]).tolist())
        branches.update(
            branch for branch, enabled in zip(json.loads(row["mf_branch"]), active)
            if enabled
        )
        assert row["residual_kind"] == "inverse_cdf"
        assert row["slack_unit"] == "standard_normal_quantile"
        assert not active[0] and np.isnan(residual[0])
    maximum_error = max(residual_error)
    assert maximum_error < 1e-12
    assert trial["status"] == "success"
    assert all(row["solver_status"] == "0" for row in cycles)
    assert all(row["solve_mode"] == "sqp_rti" for row in cycles)
    navigation.append({
        "environment": trial["maze_type"],
        "status": trial["status"],
        "simulated_time_s": float(trial["final_time"]),
        "successful_solver_steps": len(cycles),
        "safe_stop_count": 0,
        "minimum_logged_pose_psdf_m": min(float(row["psdf"]) for row in cycles),
        "active_branch_counts": dict(branches),
        "max_residual_log_error_vs_scipy": maximum_error,
    })

expected_hashes = json.loads((RUN / "navigation_source_hashes.json").read_text())
changed = [
    name for name, digest in expected_hashes.items()
    if hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() != digest
]
assert not changed, changed
new_test_log = (RUN / "verification/new_tests.log").read_text()
assert "Ran 34 tests" in new_test_log and "\nOK\n" in new_test_log
summary = {
    "new_tests": {"run": 34, "passed": 34, "actual_cuda_test_included": True},
    "combined_suite": {
        "run": 214,
        "new_tests_in_that_run": 32,
        "note": "Two additional math tests were added afterward; all 34 new tests were rerun.",
        "preexisting_failures": [
            "test_mpcc_inheritance noise defaults: alpha_kappa actual 0.0003 vs expected 0.006 (MF/SF subtests)",
        ],
        "preexisting_errors": [
            "test_cc_ecs_integration missing config/config_nmpc_cc_ecs_rectangle_all4.yaml",
        ],
    },
    "native_smoke": {
        "simulated_time_s": 3,
        "successful_solver_steps": 30,
        "debug_mf": True,
        "status": "timeout at deliberately configured 3 s limit",
        "environment_issue_resolved": "Added existing acados/lib to LD_LIBRARY_PATH after initial load failure.",
    },
    "navigation": navigation,
    "navigation_sources_changed_during_run": changed,
    "scope": "One nominal start per environment; localization noise and start perturbation disabled. No success-rate or performance-improvement claim.",
    "preservation_report": "preservation.json",
    "native_codegen_cleanup": "No psdf_mpcc_cc_mf_icdf_* directories remained in /tmp after the runs.",
}
(RUN / "verification/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
