import math
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from run_optimizer_benchmark import (
    build_cc_mf_scale_config,
    compute_metrics,
    load_summary,
    select_best_scale,
)
from sim.simulation import Robot


class _FakeController:
    def generate_control_input(self, system, global_path, local_trajectory, obstacles):
        time.sleep(0.001)
        return np.array([0.1, 0.0])

    def logging(self, logger):
        return None


class _FakeSystem:
    pass


class OptimizerBenchmarkTest(unittest.TestCase):
    def test_load_summary_rejects_removed_names_without_changing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "all_runs.csv"
            for optimizer_type in (
                "rmpcc", "rmpcc_pv", "mpcc", "acados", "casadi",
                "dcbf_casadi", "minkowski_cbf",
            ):
                with self.subTest(optimizer_type=optimizer_type):
                    original = (
                        "run_id,optimizer_type\n"
                        f"legacy_run,{optimizer_type}\n"
                    )
                    path.write_text(original, encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, "new --output-root"):
                        load_summary(path)

                    self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_load_summary_accepts_renamed_controllers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "all_runs.csv"
            path.write_text(
                "run_id,optimizer_type\nsf_run,cc_sf\nmf_run,cc_mf\n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_summary(path),
                [
                    {"run_id": "sf_run", "optimizer_type": "cc_sf"},
                    {"run_id": "mf_run", "optimizer_type": "cc_mf"},
                ],
            )

    def test_load_summary_returns_empty_for_absent_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "all_runs.csv"

            self.assertEqual(load_summary(path), [])
            self.assertFalse(path.exists())

    def test_scale_config_matches_existing_default_at_point_three(self):
        config = build_cc_mf_scale_config(0.3)
        self.assertAlmostEqual(config["covariance_growth_scale"], 0.3)
        self.assertAlmostEqual(config["alpha_f"], 0.0006)
        self.assertAlmostEqual(config["alpha_v"], 0.00012)
        self.assertAlmostEqual(config["alpha_kappa"], 0.003)
        self.assertAlmostEqual(config["beta_v"], 0.006)
        self.assertAlmostEqual(config["beta_kappa"], 0.0024)
        self.assertAlmostEqual(config["beta_omega"], 0.0024)

    def test_controller_boundary_records_full_call_time(self):
        robot = Robot(_FakeSystem())
        robot.set_controller(_FakeController())
        robot._global_path = []
        robot._local_trajectory = []

        robot.run_controller([])

        self.assertEqual(len(robot._controller_logger._computation_times), 1)
        self.assertGreater(robot._controller_logger._computation_times[0], 0.0)

    def test_metrics_cover_deadline_clearance_smoothness_and_failures(self):
        trial_rows = [
            {
                "controller_computation_time": "0.05",
                "solver_computation_time": "0.04",
                "v": "0.0",
                "omega": "0.0",
                "solver_success": "True",
            },
            {
                "controller_computation_time": "0.10",
                "solver_computation_time": "0.08",
                "v": "1.0",
                "omega": "-1.0",
                "solver_success": "False",
            },
            {
                "controller_computation_time": "0.20",
                "solver_computation_time": "0.18",
                "v": "1.0",
                "omega": "1.0",
                "solver_success": "True",
            },
        ]
        pose_rows = [{"sdf_value": "-0.1"}, {"sdf_value": "0.1"}]
        outcome = {
            "status": "success",
            "goal_reached": True,
            "final_time": 1.5,
            "distance_to_goal": 0.0,
        }

        metrics = compute_metrics(
            trial_rows,
            pose_rows,
            outcome,
            {"safe_stop_count": 2},
        )

        self.assertEqual(metrics["deadline_miss_count"], 1)
        self.assertAlmostEqual(metrics["deadline_miss_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["clearance_q05"], -0.09)
        self.assertAlmostEqual(metrics["clearance_min"], -0.1)
        self.assertTrue(metrics["collision"])
        self.assertAlmostEqual(metrics["rms_delta_v"], math.sqrt(0.5))
        self.assertAlmostEqual(metrics["rms_delta_omega"], math.sqrt(2.5))
        self.assertEqual(metrics["solver_failure_count"], 1)
        self.assertAlmostEqual(metrics["solver_failure_rate"], 1.0 / 3.0)
        self.assertEqual(metrics["safe_stop_count"], 2)
        self.assertAlmostEqual(metrics["driving_time"], 1.5)

    def test_selection_is_safety_then_clearance_first(self):
        base = {
            "optimizer_type": "cc_mf",
            "status": "success",
            "collision": "False",
            "raw_logs_complete": "True",
            "solver_failure_count": "0",
            "safe_stop_count": "0",
            "driving_time": "10.0",
        }
        faster = {
            **base,
            "run_id": "fast",
            "covariance_growth_scale": "0.1",
            "clearance_q05": "0.01",
            "deadline_miss_rate": "0.0",
            "controller_time_p95": "0.05",
        }
        safer = {
            **base,
            "run_id": "safe",
            "covariance_growth_scale": "0.5",
            "clearance_q05": "0.02",
            "deadline_miss_rate": "1.0",
            "controller_time_p95": "0.20",
        }

        selected = select_best_scale([faster, safer])

        self.assertEqual(selected["run_id"], "safe")


if __name__ == "__main__":
    unittest.main()
