import math
import time
import unittest

import numpy as np

from run_optimizer_benchmark import (
    build_rmpcc_scale_config,
    compute_metrics,
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
    def test_scale_config_matches_existing_default_at_point_three(self):
        config = build_rmpcc_scale_config(0.3)
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
            "optimizer_type": "rmpcc",
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
