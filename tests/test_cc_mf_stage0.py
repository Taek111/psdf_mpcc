"""Regression checks for cached PSDF and pre-solve stage-0 safety rejection."""

import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
from control.controller import BaseController
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.dd_dynamics import DifferentialDriveDynamics
from sim.simulation_mpc import simulation_mpc
from sim.trial_metrics import collect_successful_solver_steps


class _CountingPSDF(AugmentedPSDFWrapper):
    def __init__(self):
        vertices = torch.tensor(
            [[-.075, -.06], [.075, -.06], [.075, .06], [-.075, .06]],
            dtype=torch.float64,
        )
        super().__init__(vertices, K_max=1, E_max=4)
        self.mf_batches = []
        self.pose_batches = []

    def forward_mf(self, z_bar, epsilon, d_min):
        self.mf_batches.append(z_bar.detach().clone())
        return super().forward_mf(z_bar, epsilon, d_min)

    def forward(self, poses):
        self.pose_batches.append(poses.detach().clone())
        return super().forward(poses)


class _Solver:
    def __init__(self):
        self.states = np.zeros((4, 8))
        self.inputs = np.tile([.7, 1., .5], (3, 1))
        self.status = 0
        self.solve_calls = 0
        self.fail_install = False

    def get(self, stage, field):
        if field == "x":
            return self.states[stage].copy()
        if field == "u":
            return self.inputs[stage].copy()
        if field == "sl":
            return np.zeros(1)
        raise KeyError(field)

    def set(self, stage, field, value):
        if field == "x":
            self.states[stage] = value
        elif field == "u":
            if self.fail_install:
                raise RuntimeError("installation failed")
            self.inputs[stage] = value

    def solve(self):
        self.solve_calls += 1
        self.status = 0
        return 0

    def reset(self, **kwargs):
        # A reset can erase native status. It is not evidence of solver success.
        self.status = 0
        self.states.fill(0.)
        self.inputs.fill(0.)


class CCMFStageZeroTest(unittest.TestCase):
    def setUp(self):
        self.param = CCMFOptimizerParam()
        self.param.horizon = 3
        self.param.tf = .3
        self.optimizer = CCMFOptimizer()
        self.optimizer.param = self.param
        self.optimizer.N, self.optimizer.nx, self.optimizer.nu = 3, 8, 3
        self.optimizer.state = SimpleNamespace(_x=np.zeros(3), _u=np.array([.3, .5]))
        self.optimizer._system_dynamics = DifferentialDriveDynamics
        self.optimizer.solver = _Solver()
        self.wrapper = _CountingPSDF()
        self.optimizer.psdf_wrapper = self.wrapper
        self.optimizer._set_constraint_parameters = Mock()
        self.optimizer.compute_diagnostics = Mock(return_value=None)
        self.optimizer.recovery_infeasible = Mock(side_effect=AssertionError("No backup expected"))
        self.set_wall(.08)

    def tearDown(self):
        # This test never owns generated solvers or code artifacts.
        self.optimizer.cleanup = lambda: None

    def set_wall(self, left):
        a = torch.tensor([[left, -.5], [.3, -.5], [.3, .5], [left, .5]], dtype=torch.float64)
        self.wrapper.update_edge_clusters([a], [torch.roll(a, -1, 0)])

    def solve(self):
        with redirect_stdout(io.StringIO()) as output:
            solution = self.optimizer.solve_nlp()
        return solution, output.getvalue()

    def test_normal_cycle_and_csv_need_only_one_batch_including_measured_pose(self):
        self.optimizer.solver.states[0, :3] = [4., 5., 1.]
        solution, output = self.solve()
        phi = simulation_mpc._get_current_psdf_for_logging(self.optimizer, np.zeros(3))

        self.assertEqual(len(self.wrapper.mf_batches), 1)
        self.assertEqual(len(self.wrapper.pose_batches), 0)
        self.assertEqual(self.wrapper.mf_batches[0].shape, (4, 8))
        np.testing.assert_array_equal(self.wrapper.mf_batches[0][0, :3].numpy(), np.zeros(3))
        self.assertAlmostEqual(phi, .005)
        self.assertEqual(self.optimizer.get_cycle_log_rows()[0]["psdf"], phi)
        self.assertTrue(solution.stats()["success"])
        self.assertEqual(self.optimizer.solver.solve_calls, 1)
        self.optimizer.compute_diagnostics.assert_not_called()
        self.assertEqual(output, "")

    def test_unsafe_stage_zero_stops_even_with_masked_mf_and_stale_success_status(self):
        for left in (.045, .0755):
            with self.subTest(left=left):
                self.set_wall(left)
                solution, _ = self.solve()
                self.assertFalse(self.optimizer._constraint_data["mf_mask"][0])
                self.assertEqual(self.optimizer.solver.solve_calls, 0)
                self.assertEqual(self.optimizer.solver.status, 0)
                self.assertEqual(solution.stats(), {
                    "return_status": "safe_stop_stage0", "status_code": None,
                    "success": False, "solver_called": False,
                })
                # Reuse the established bounded braking behavior, not a nominal input.
                np.testing.assert_allclose(solution.get_input_trajectory()[:, 0], [.2, 0., 0.])
                self.optimizer.recovery_infeasible.assert_not_called()
                self.optimizer.compute_diagnostics.assert_not_called()
                row = self.optimizer.get_cycle_log_rows()[-1]
                self.assertEqual(row["solve_mode"], "safe_stop_stage0")
                self.assertIsNone(row["solver_status"])
                self.assertLess(row["psdf"], self.param.d_col)
                self.assertEqual(json.loads(row["mf_slack"]), [None] * 4)
        self.assertEqual(len(self.wrapper.mf_batches), 2)
        self.assertEqual(len(self.wrapper.pose_batches), 0)
        self.assertEqual(self.optimizer._safe_stop_count, 2)
        self.assertEqual(self.optimizer._stage0_guard_stop_count, 2)
        self.assertEqual(self.optimizer._main_solver_failure_count, 0)
        self.assertEqual(self.optimizer.solver_times, [0., 0.])

    def test_guard_works_with_row_mf_disabled(self):
        self.param.use_row_mf = False
        self.set_wall(.045)
        solution, _ = self.solve()
        self.assertFalse(solution.stats()["solver_called"])
        self.assertEqual(len(self.wrapper.mf_batches), 0)
        self.assertEqual(len(self.wrapper.pose_batches), 1)
        self.assertEqual(self.wrapper.pose_batches[0].shape, (4, 3))

    def test_disabled_obstacle_constraints_do_not_stop_or_log_sentinel_distance(self):
        self.param.use_obstacle_constraint = False
        self.set_wall(.045)
        solution, _ = self.solve()
        self.assertTrue(solution.stats()["success"])
        self.assertEqual(len(self.wrapper.mf_batches), 0)
        self.assertEqual(len(self.wrapper.pose_batches), 0)
        self.assertIsNone(self.optimizer.get_current_psdf())
        self.assertTrue(np.isnan(simulation_mpc._get_current_psdf_for_logging(self.optimizer, np.zeros(3))))
        self.assertIsNone(self.optimizer.get_cycle_log_rows()[0]["psdf"])

    def test_guard_accepts_equality_at_threshold(self):
        phi, _ = self.wrapper(torch.zeros((1, 3), dtype=torch.float64))
        self.param.d_col = float(phi[0])
        self.wrapper.pose_batches.clear()
        solution, _ = self.solve()
        self.assertTrue(solution.stats()["success"])
        self.assertEqual(len(self.wrapper.pose_batches), 0)

    def test_new_state_or_failed_preparation_clears_cached_distance(self):
        self.solve()
        self.assertIsNotNone(self.optimizer.get_current_psdf())
        self.optimizer.set_state(self.optimizer.state)
        self.assertIsNone(self.optimizer.get_current_psdf())
        self.solve()
        self.optimizer._compute_constraint_affine_data = Mock(side_effect=RuntimeError("PSDF unavailable"))
        with self.assertRaisesRegex(RuntimeError, "PSDF unavailable"):
            self.solve()
        self.assertIsNone(self.optimizer.get_current_psdf())

    def test_debug_diagnostics_are_opt_in_and_skipped_when_solver_did_not_run(self):
        self.param.debug_mf = True
        self.solve()
        self.optimizer.compute_diagnostics.assert_called_once()
        self.optimizer.compute_diagnostics.reset_mock()
        self.set_wall(.045)
        self.solve()
        self.optimizer.compute_diagnostics.assert_not_called()
        self.assertEqual(len(self.wrapper.pose_batches), 0)

    def test_failed_stop_installation_raises_instead_of_returning_old_input(self):
        self.set_wall(.045)
        self.optimizer.solver.fail_install = True
        with self.assertRaisesRegex(RuntimeError, "stage-0 safe-stop plan"):
            self.solve()
        self.assertEqual(self.optimizer.solver.solve_calls, 0)
        self.assertEqual(self.optimizer.solver_times, [])

    def test_backup_preparation_exception_restores_main_psdf_cache(self):
        self.solve()
        phi = self.optimizer.get_current_psdf()
        self.optimizer.backup_solver = _Solver()

        def fail_prepare(_):
            self.optimizer._current_psdf = None
            raise RuntimeError("backup preparation failed")

        self.optimizer._prepare_and_solve = fail_prepare
        with redirect_stdout(io.StringIO()):
            result = self.optimizer._try_backup_recovery()
        self.assertIsNone(result)
        self.assertEqual(self.optimizer.get_current_psdf(), phi)
        self.assertEqual(simulation_mpc._get_current_psdf_for_logging(self.optimizer, np.zeros(3)), phi)

    def test_skipped_stop_does_not_enter_successful_solver_time_statistics(self):
        self.set_wall(.045)
        solution, _ = self.solve()
        controller = BaseController(self.optimizer, self.param)
        controller._opt_sol = solution
        controller._update_last_solver_status_info()
        stopped = controller.get_last_solver_status_info()
        self.assertFalse(stopped["success"])
        self.assertIsNone(stopped["status_code"])
        self.assertFalse(stopped["solver_called"])

        self.set_wall(.08)
        solution, _ = self.solve()
        controller._opt_sol = solution
        controller._update_last_solver_status_info()
        robot = SimpleNamespace(
            _controller=controller, _system=SimpleNamespace(_dt=.1),
            _controller_logger=SimpleNamespace(_solver_status_infos=[stopped, controller.get_last_solver_status_info()]),
        )
        steps = collect_successful_solver_steps(SimpleNamespace(robot=robot))
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["timestep"], 1)


if __name__ == "__main__":
    unittest.main()
