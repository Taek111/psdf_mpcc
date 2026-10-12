import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from sim.simulation_mpc import TrialFailureChecker, simulation_mpc


class CCMFPSDFLoggingTest(unittest.TestCase):
    @staticmethod
    def _cached_optimizer(value):
        return SimpleNamespace(
            get_current_psdf=Mock(return_value=value),
            psdf_wrapper=Mock(side_effect=AssertionError("Unexpected PSDF evaluation")),
        )

    def test_logging_reuses_cached_current_psdf_without_forward(self):
        optimizer = self._cached_optimizer(-0.012)

        value = simulation_mpc._get_current_psdf_for_logging(optimizer, [1., 2., 3.])

        self.assertEqual(value, -0.012)
        optimizer.get_current_psdf.assert_called_once_with()
        optimizer.psdf_wrapper.assert_not_called()

    def test_unavailable_or_invalid_cache_never_falls_back_to_forward(self):
        for cached_value in (None, np.nan, np.inf, -np.inf, "invalid"):
            with self.subTest(cached_value=cached_value):
                optimizer = self._cached_optimizer(cached_value)

                value = simulation_mpc._get_current_psdf_for_logging(optimizer, [0., 0., 0.])

                self.assertTrue(np.isnan(value))
                optimizer.psdf_wrapper.assert_not_called()

    def test_cache_invalidation_does_not_reuse_previous_step(self):
        optimizer = self._cached_optimizer(None)
        optimizer.get_current_psdf.side_effect = [0.025, None]

        first = simulation_mpc._get_current_psdf_for_logging(optimizer, [0., 0., 0.])
        second = simulation_mpc._get_current_psdf_for_logging(optimizer, [0.1, 0., 0.])

        self.assertEqual(first, 0.025)
        self.assertTrue(np.isnan(second))
        optimizer.psdf_wrapper.assert_not_called()

    def test_cache_error_does_not_trigger_forward_or_interrupt_logging(self):
        optimizer = self._cached_optimizer(None)
        optimizer.get_current_psdf.side_effect = RuntimeError("No prepared step")

        value = simulation_mpc._get_current_psdf_for_logging(optimizer, [0., 0., 0.])

        self.assertTrue(np.isnan(value))
        optimizer.psdf_wrapper.assert_not_called()

    def test_legacy_optimizers_keep_existing_psdf_forward(self):
        for tuple_output in (False, True):
            with self.subTest(tuple_output=tuple_output):
                phi = torch.tensor([0.025])
                output = (phi, torch.zeros((1, 3))) if tuple_output else phi
                optimizer = SimpleNamespace(device="cpu", psdf_wrapper=Mock(return_value=output))
                current_pose = np.array([1., 2., 3.])

                value = simulation_mpc._get_current_psdf_for_logging(optimizer, current_pose)

                self.assertAlmostEqual(value, 0.025)
                optimizer.psdf_wrapper.assert_called_once()
                pose_tensor = optimizer.psdf_wrapper.call_args.args[0]
                torch.testing.assert_close(
                    pose_tensor, torch.tensor(current_pose, dtype=torch.float32).unsqueeze(0),
                )

    def test_optimizer_without_wrapper_reports_nan(self):
        value = simulation_mpc._get_current_psdf_for_logging(SimpleNamespace(), [0., 0., 0.])

        self.assertTrue(np.isnan(value))


class CCMFSafetyStopFailureCheckerTest(unittest.TestCase):
    @staticmethod
    def _checker_and_simulation(statuses):
        checker = TrialFailureChecker(
            movement_window_sec=0.0,
            movement_threshold=0.03,
            max_consecutive_solver_failures=2,
        )
        controller = SimpleNamespace(get_last_solver_status_info=Mock(side_effect=statuses))
        simulation = SimpleNamespace(_robot=SimpleNamespace(_controller=controller))
        return checker, simulation

    @staticmethod
    def _safety_stop():
        return {
            "success": False,
            "solver_called": False,
            "raw_status": "safe_stop_stage0",
            "status_code": None,
        }

    @staticmethod
    def _solver_failure():
        # Legacy optimizers do not need to expose solver_called.
        return {"success": False, "raw_status": 4, "status_code": 4}

    def test_repeated_safety_stops_terminate_without_claiming_solver_failure(self):
        checker, simulation = self._checker_and_simulation([self._safety_stop()] * 2)

        self.assertIsNone(checker(simulation))
        self.assertEqual(checker(simulation), {
            "reason": "consecutive_unsuccessful_control_steps(2): safe_stop_stage0",
        })

    def test_mixed_streak_retains_safety_stop_information_in_either_order(self):
        for statuses in (
            [self._safety_stop(), self._solver_failure()],
            [self._solver_failure(), self._safety_stop()],
        ):
            with self.subTest(statuses=statuses):
                checker, simulation = self._checker_and_simulation(statuses)

                self.assertIsNone(checker(simulation))
                self.assertEqual(checker(simulation), {
                    "reason": (
                        "consecutive_unsuccessful_control_steps(2): "
                        f"{statuses[-1]['raw_status']}"
                    ),
                })

    def test_solver_only_streak_retains_legacy_failure_reason(self):
        checker, simulation = self._checker_and_simulation([self._solver_failure()] * 2)

        self.assertIsNone(checker(simulation))
        self.assertEqual(checker(simulation), {"reason": "consecutive_solver_failures(2): 4"})

    def test_success_resets_counter_and_safety_stop_marker(self):
        checker, simulation = self._checker_and_simulation([
            self._safety_stop(),
            {"success": True, "solver_called": True, "raw_status": 0},
            self._solver_failure(),
            self._solver_failure(),
        ])

        for _ in range(3):
            self.assertIsNone(checker(simulation))
        self.assertEqual(checker(simulation), {"reason": "consecutive_solver_failures(2): 4"})


if __name__ == "__main__":
    unittest.main()
