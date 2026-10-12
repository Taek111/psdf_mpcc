"""ECS algebra and controller contracts; no generated solver is required.

These tests are prepared for an explicitly requested validation run. They use
symbolic models and fake solvers, not navigation trials or acados codegen.
"""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import casadi as ca
import numpy as np
from scipy.stats import chi2

from control.cc_covariance import (
    covariance_from_log_cholesky,
    propagate_log_cholesky,
)
from control.cc_ecs_optimizer import (
    CCECSOptimizer, CCECSOptimizerParam, PROCESS_NOISE_FIELDS,
)
from control.controller import BaseController
from models.dd_dynamics import DifferentialDriveDynamics
from models.geometry_utils import RectangleRegion


class _Solver:
    def __init__(self, states, controls, status=0, on_solve=None):
        self.states = np.asarray(states).copy()
        self.controls = np.asarray(controls).copy()
        self.parameters = {}
        self.status = 0
        self.result_status = status
        self.on_solve = on_solve

    def get(self, stage, field):
        if field == "x":
            return self.states[stage].copy()
        if field == "u":
            return self.controls[stage].copy()
        raise KeyError(field)

    def set(self, stage, field, value):
        if field == "x":
            self.states[stage] = value
        elif field == "u":
            self.controls[stage] = value
        elif field == "p":
            self.parameters[stage] = np.asarray(value).copy()
        elif field not in ("lbx", "ubx"):
            raise KeyError(field)

    def solve(self):
        if self.on_solve is not None:
            self.on_solve(self)
        self.status = self.result_status
        return self.status

    def reset(self, **kwargs):
        # Exercise the failure status restoration after writing a safe stop.
        self.status = 0


class CCECSTest(unittest.TestCase):
    def setUp(self):
        self.param = CCECSOptimizerParam()
        self.param.horizon = 3
        self.param.tf = 0.3
        self.param.augmented_psdf_device = "cpu"
        self.optimizer = CCECSOptimizer()
        self.addCleanup(self.optimizer.cleanup)
        self.optimizer.param = self.param
        self.optimizer.state = SimpleNamespace(_x=np.zeros(3), _u=np.zeros(2))
        self.optimizer._system_dynamics = DifferentialDriveDynamics
        self.optimizer._configure_cluster_capacity(3)
        self.optimizer.setup_ocp(self.param, None)
        self.optimizer.add_obstacle_avoidance_constraint(self.param, None, None)
        self.optimizer._evaluate_cluster_geometry = self._plane_geometry

    @staticmethod
    def _plane_geometry(poses):
        gradients = np.array([[1.0, 0.0, 0.2], [0.0, 1.0, -0.3]])
        distances = np.array([0.8, 0.9])[None, :] + poses @ gradients.T
        return distances, np.tile(gradients, (len(poses), 1, 1))

    def _states(self, covariance_state=None):
        if covariance_state is None:
            covariance_state = self.optimizer._build_initial_covariance_state(self.param)
        states = np.zeros((self.param.horizon + 1, 8))
        states[:, 3] = np.arange(len(states)) * 0.02
        states[:, 4:] = covariance_state
        return states

    def _solver(self, status=0, on_solve=None, covariance_state=None):
        return _Solver(
            self._states(covariance_state),
            np.tile([0.2, 0.3, 0.2], (self.param.horizon, 1)),
            status=status, on_solve=on_solve,
        )

    def _constraint_function(self):
        model = self.optimizer.ocp.model
        return ca.Function(
            "ecs_rows", [model.x, model.p],
            [model.con_h_expr, model.con_h_expr_e,
             ca.jacobian(model.con_h_expr, model.x), model.con_h_expr_0],
        )

    def test_noise_free_covariance_matches_reduced_A_P_AT(self):
        self.assertTrue(all(getattr(self.param, name) == 0 for name in PROCESS_NOISE_FIELDS))
        xi = np.r_[np.log([0.03, 0.07, 0.02]), -0.04]
        p = covariance_from_log_cholesky(xi)
        dt = self.param.tf / self.param.horizon
        model = self.optimizer.ocp.model
        dynamics = ca.Function("ecs_step", [model.x, model.u, model.p], [model.disc_dyn_expr])
        parameters = self.optimizer.ocp.parameter_values.copy()
        parameters[self.optimizer.covariance_propagation_param_slice] = 1.0
        for v, omega in ((0.0, 0.0), (0.2, 0.7), (-0.3, -0.4)):
            with self.subTest(v=v, omega=omega):
                a = np.array([[1.0, dt * omega, 0.0],
                              [-dt * omega, 1.0, dt * v], [0.0, 0.0, 1.0]])
                expected = a @ p @ a.T
                expected[0, 1] = expected[1, 0] = 0.0
                expected[0, 2] = expected[2, 0] = 0.0
                numeric = propagate_log_cholesky(xi, v, omega, 8.0, dt, self.param)
                np.testing.assert_allclose(covariance_from_log_cholesky(numeric), expected)
                state = np.r_[np.zeros(4), xi]
                symbolic = np.asarray(dynamics(state, [v, omega, 0.1], parameters)).ravel()
                np.testing.assert_allclose(symbolic[4:], numeric)
                if v == omega == 0:
                    np.testing.assert_allclose(numeric, xi)
                else:
                    self.assertGreater(np.linalg.norm(symbolic[4:] - xi), 0)

    def test_covariance_and_margin_depend_on_control_without_process_noise(self):
        model = self.optimizer.ocp.model
        q = ca.DM([0.0, 1.0, 0.4])
        next_xi = model.disc_dyn_expr[4:8]
        margin = self.optimizer._symbolic_projected_std(next_xi, q)
        evaluate = ca.Function(
            "ecs_input_margin", [model.x, model.u, model.p],
            [ca.jacobian(next_xi, model.u), ca.jacobian(margin, model.u)],
        )
        state = np.r_[np.zeros(4), np.log([0.03, 0.07, 0.02]), -0.04]
        covariance_jac, margin_jac = evaluate(
            state, [0.2, 0.3, 0.1], self.optimizer.ocp.parameter_values
        )
        self.assertGreater(np.linalg.norm(covariance_jac), 0.0)
        self.assertGreater(np.linalg.norm(margin_jac), 0.0)

    def test_all_process_noise_overrides_are_rejected(self):
        for name in PROCESS_NOISE_FIELDS:
            with self.subTest(name=name):
                param = copy.deepcopy(self.param)
                setattr(param, name, 1e-6)
                with self.assertRaisesRegex(ValueError, name):
                    self.optimizer._validate_param(param)

    def test_horizon_budget_quantile_and_invalid_inputs(self):
        param = copy.deepcopy(self.param)
        param.horizon = 20
        param.chance_epsilon = 0.20
        epsilon, beta = self.optimizer._risk_allocation(param)
        self.assertAlmostEqual(epsilon, 0.01)
        self.assertAlmostEqual(beta, 3.368214175218727, places=10)
        self.assertAlmostEqual(chi2.cdf(beta**2, df=3), 0.99)
        for invalid in (0.0, 1.0, -0.1, np.nan, np.inf, [0.2], True):
            with self.subTest(invalid=invalid):
                param.chance_epsilon = invalid
                with self.assertRaises(ValueError):
                    self.optimizer._risk_allocation(param)

    def test_frame_conversion_and_signed_cross_covariance(self):
        poses = np.array([[0.0, 0.0, 0.8]])
        gradients = np.array([[[0.6, -0.8, 0.2], [0.0, 1.0, -0.4]]])
        local = self.optimizer._cluster_local_gradients(poses, gradients)
        xi = np.array([[*np.log([0.03, 0.07, 0.02]), -0.04]])
        cosine, sine = np.cos(0.8), np.sin(0.8)
        transform = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
        covariance = covariance_from_log_cholesky(xi)[0]
        world = transform @ covariance @ transform.T
        expected_variance = np.einsum("ki,ij,kj->k", gradients[0], world, gradients[0])
        _, beta = self.optimizer._risk_allocation(self.param)
        np.testing.assert_allclose(
            self.optimizer._compute_margins(xi, local)[0], beta * np.sqrt(expected_variance),
        )

    def test_hard_rows_terminal_padding_and_live_covariance_derivatives(self):
        xi = np.r_[np.log([0.03, 0.07, 0.02]), -0.04]
        states = self._states(xi)
        solver = self._solver(covariance_state=xi)
        data = self.optimizer._compute_constraint_data(states)
        self.optimizer._set_ecs_parameters(solver, states[:, 3], data)
        evaluate = self._constraint_function()
        margins = self.optimizer._compute_margins(states[:, 4:], data["gradients_local"])
        for k in range(1, self.param.horizon + 1):
            rows, terminal, jac, initial = evaluate(states[k], solver.parameters[k])
            expected = data["distances"][k] - self.param.d_min - margins[k]
            np.testing.assert_allclose(np.asarray(rows).ravel()[:2], expected)
            np.testing.assert_allclose(rows, terminal)
            self.assertAlmostEqual(float(rows[2]), 1.0)
            self.assertTrue(np.isfinite(np.asarray(jac)).all())
            self.assertGreater(np.linalg.norm(np.asarray(jac)[:2, 4:]), 0.0)
            np.testing.assert_array_equal(np.asarray(jac)[2], np.zeros(8))
            np.testing.assert_array_equal(np.asarray(initial).ravel(), np.ones(3))
        initial_rows, _, _, _ = evaluate(states[0], solver.parameters[0])
        np.testing.assert_allclose(np.asarray(initial_rows).ravel(), np.ones(3))
        self.assertEqual(len(self.optimizer.ocp.constraints.idxsh), 0)
        self.assertEqual(len(self.optimizer.ocp.constraints.idxsh_e), 0)
        # Covariance is a live state; increasing it must tighten the same rows.
        enlarged = states[1].copy()
        enlarged[4:7] += np.log(2.0)
        enlarged[7] *= 2.0
        larger, _, _, _ = evaluate(enlarged, solver.parameters[1])
        np.testing.assert_allclose(
            np.asarray(larger).ravel()[:2], data["distances"][1] - self.param.d_min - 2 * margins[1],
        )

    def test_farther_obstacle_can_be_more_restrictive_and_duplicates_do_not_change_beta(self):
        gradients = np.tile(np.array([[1., 0., 0.], [0., 1., 0.]]), (4, 1, 1))
        distances = np.tile([0.05, 0.10], (4, 1))
        self.optimizer._evaluate_cluster_geometry = lambda poses: (distances, gradients)
        states = self._states(np.r_[np.log([0.001, 0.1, 0.001]), 0.0])
        data = self.optimizer._compute_constraint_data(states)
        margin = self.optimizer._compute_margins(states[:, 4:], data["gradients_local"])
        residual = distances - self.param.d_min - margin
        self.assertGreater(residual[1, 0], 0)
        self.assertLess(residual[1, 1], 0)
        self.optimizer._evaluate_cluster_geometry = lambda poses: (
            distances[:, [0, 1, 1]], gradients[:, [0, 1, 1]],
        )
        duplicated = self.optimizer._compute_constraint_data(states)
        self.assertEqual(data["beta"], duplicated["beta"])
        self.assertEqual(data["epsilon_stage"], duplicated["epsilon_stage"])
        np.testing.assert_array_equal(
            self.optimizer._compute_margins(states[:, 4:], duplicated["gradients_local"])[:, :2], margin,
        )

    def test_preparation_freezes_covariance_tail_but_keeps_terminal_ecs_active(self):
        self.param.robust_horizon = 1
        solver = self._solver()
        self.assertEqual(self.optimizer._prepare_and_solve(solver), 0)
        self.assertFalse(np.array_equal(solver.states[0, 4:], solver.states[1, 4:]))
        np.testing.assert_array_equal(solver.states[1:, 4:], np.tile(solver.states[1, 4:], (3, 1)))
        _, beta = self.optimizer._risk_allocation(self.param)
        for k in range(4):
            parameters = solver.parameters[k]
            self.assertEqual(parameters[self.optimizer.covariance_propagation_param_slice][0], float(k < 1))
            self.assertEqual(parameters[self.optimizer.risk_margin_scale_param_slice][0], beta if k else 0)

    def test_overlap_keeps_hard_rows_active_and_large_margin_is_not_capped(self):
        gradients = np.tile([[1.0, 0.0, 0.0]], (4, 1, 1))
        self.optimizer._evaluate_cluster_geometry = lambda poses: (
            np.full((len(poses), 1), -0.05), gradients,
        )
        states = self._states(np.r_[np.log([1.0, 1.0, 1.0]), 0.0])
        solver = self._solver()
        data = self.optimizer._compute_constraint_data(states)
        self.optimizer._set_ecs_parameters(solver, states[:, 3], data)
        rows, _, _, _ = self._constraint_function()(states[1], solver.parameters[1])
        expected_margin = data["beta"]
        self.assertGreater(expected_margin, 2.0)
        self.assertAlmostEqual(float(rows[0]), -0.05 - self.param.d_min - expected_margin)
        self.assertGreater(solver.parameters[1][self.optimizer.risk_margin_scale_param_slice][0], 0.0)

    def test_empty_obstacles_are_finite_inactive_rows_and_have_no_constraint_log(self):
        self.optimizer._evaluate_cluster_geometry = lambda poses: (
            np.empty((len(poses), 0)), np.empty((len(poses), 0, 3)),
        )
        solver = self._solver()
        self.optimizer.solver = solver
        self.optimizer.solve_nlp()
        rows, _, jac, _ = self._constraint_function()(solver.states[-1], solver.parameters[3])
        np.testing.assert_allclose(np.asarray(rows).ravel(), np.ones(3))
        self.assertTrue(np.isfinite(np.asarray(jac)).all())
        self.assertEqual(self.optimizer.get_constraint_log_rows(), [])
        self.assertEqual(len(self.optimizer.get_covariance_log_rows()), 4)
        self.assertIsNone(self.optimizer.get_cycle_log_rows()[0]["minimum_fixed_row_residual"])

    def test_initial_update_capacity_overflow_and_invalid_geometry(self):
        # Simulator invokes this before setup(), when self.param may not exist.
        fresh = CCECSOptimizer()
        self.addCleanup(fresh.cleanup)
        obstacle = RectangleRegion(0.5, 0.7, -0.1, 0.1)
        fresh.update_obstacles([obstacle])
        fresh.param = self.param
        system = SimpleNamespace(_geometry=SimpleNamespace(_geometries=[
            SimpleNamespace(_region=RectangleRegion(-0.05, 0.05, -0.03, 0.03)),
        ]))
        fresh.initialize_ped_model(system, [obstacle], device="cpu")
        self.assertEqual(fresh.obstacle_capacity, 1)
        self.assertEqual(fresh.psdf_wrapper.E_max, 4)
        with self.assertRaisesRegex(ValueError, "capacity"):
            fresh.update_obstacles([obstacle, obstacle])
        fresh.update_obstacles([])
        distances, gradients = fresh._evaluate_cluster_geometry(np.zeros((2, 3)))
        self.assertEqual(distances.shape, (2, 0))
        self.assertEqual(gradients.shape, (2, 0, 3))
        with self.assertRaises(ValueError):
            fresh._evaluate_cluster_geometry(np.array([[np.nan, 0., 0.]]))

    def test_solver_only_time_and_backup_status(self):
        clock = SimpleNamespace(value=0.0)
        def main_call(solver):
            clock.value += 0.25
        def backup_call(solver):
            clock.value += 0.75
        self.optimizer.solver = self._solver(status=4, on_solve=main_call)
        self.optimizer.backup_solver = self._solver(on_solve=backup_call)
        original_parameters = self.optimizer._set_ecs_parameters
        original_logs = self.optimizer._record_experiment_logs
        def prepare(*args):
            clock.value += 8.0
            return original_parameters(*args)
        def logs(*args):
            clock.value += 16.0
            return original_logs(*args)
        with (
            patch("control.cc_ecs_optimizer.time.perf_counter", side_effect=lambda: clock.value),
            patch("control.cc_ecs_optimizer.report_solver_failure"),
            patch.object(self.optimizer, "_set_ecs_parameters", side_effect=prepare),
            patch.object(self.optimizer, "_record_experiment_logs", side_effect=logs),
        ):
            solution = self.optimizer.solve_nlp()
            self.assertEqual(solution.stats()["return_status"], "success")
            self.assertIs(solution.solver, self.optimizer.backup_solver)
            self.assertEqual(self.optimizer.solver_times, [1.0])
            self.optimizer.solver.result_status = 0
            self.optimizer.solve_nlp()
            self.assertEqual(self.optimizer.solver_times, [1.0, 0.25])
        self.assertEqual(self.optimizer.get_cycle_log_rows()[0]["solver_status"], 0)
        self.assertEqual(self.optimizer.get_cycle_log_rows()[0]["main_status"], 4)

    def test_failed_backup_safe_stop_preserves_failure_and_main_fixed_rows(self):
        def shift_failed_nominal(solver):
            solver.states[1:, 2] = 0.8
        self.optimizer.solver = self._solver(status=4, on_solve=shift_failed_nominal)
        self.optimizer.backup_solver = self._solver(status=2)
        with patch("control.cc_ecs_optimizer.report_solver_failure"):
            solution = self.optimizer.solve_nlp()
        self.assertIs(solution.solver, self.optimizer.solver)
        self.assertEqual(solution.solver.status, 4)
        self.assertEqual(solution.stats()["return_status"], "failure")
        controller = BaseController(self.optimizer, self.param)
        controller._opt_sol = solution
        controller._update_last_solver_status_info()
        self.assertFalse(controller.get_last_solver_status_info()["success"])
        diagnostics = self.optimizer.get_last_constraint_diagnostics()
        self.assertEqual(diagnostics["solve_mode"], "safe_stop")
        for k in range(1, 4):
            rows, _, _, _ = self._constraint_function()(
                solution.solver.states[k], solution.solver.parameters[k],
            )
            np.testing.assert_allclose(
                np.asarray(rows).ravel()[:2], diagnostics["fixed_row_residual"][k],
            )
        np.testing.assert_array_equal(solution.get_input_trajectory()[2], np.zeros(3))

    def test_geometry_errors_and_solver_exceptions_are_not_hidden(self):
        solver = self._solver(on_solve=lambda _: (_ for _ in ()).throw(RuntimeError("solver error")))
        self.optimizer.solver = solver
        with self.assertRaisesRegex(RuntimeError, "solver error"):
            self.optimizer.solve_nlp()
        self.assertEqual(self.optimizer.solver_times, [])
        with patch.object(self.optimizer, "_evaluate_cluster_geometry", side_effect=ValueError("bad geometry")):
            with self.assertRaisesRegex(ValueError, "bad geometry"):
                self.optimizer.solve_nlp()

    def test_compiled_settings_require_reset_but_runtime_risk_and_covariance_do_not(self):
        self.optimizer._is_initialized = True
        self.optimizer._compiled_param_signature = self.optimizer._build_signature(self.param)
        system = SimpleNamespace(_state=self.optimizer.state, _dynamics=DifferentialDriveDynamics)
        changed = copy.deepcopy(self.param)
        changed.tf = 0.6
        with self.assertRaisesRegex(ValueError, "reset"):
            self.optimizer.setup(changed, system, None, None)
        changed = copy.deepcopy(self.param)
        changed.chance_epsilon = 0.1
        changed.sigma_f0 = 0.003
        changed.robust_horizon = 1
        self.optimizer.setup(changed, system, None, None)
        self.assertIs(self.optimizer.param, changed)
        changed.tf = 0.6
        with self.assertRaisesRegex(ValueError, "reset"):
            self.optimizer._prepare_and_solve(self._solver())


if __name__ == "__main__":
    unittest.main()
