import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

import casadi as ca
import numpy as np

from control.analytic_psdf_casadi import AnalyticPSDFCasADi
from control.cc_sf_optimizer import CCSFOptimizer, CCSFOptimizerParam


class _NominalSolver:
    def __init__(self, states, inputs, status=0, on_solve=None):
        self.states = np.asarray(states, dtype=float).copy()
        self.inputs = np.asarray(inputs, dtype=float).copy()
        self.parameters = {}
        self.status = 0
        self.result_status = status
        self.on_solve = on_solve

    def get(self, stage, field):
        if field == "x":
            return self.states[stage].copy()
        if field == "u":
            return self.inputs[stage].copy()
        raise KeyError(field)

    def set(self, stage, field, value):
        if field == "x":
            self.states[stage] = value
        elif field == "u":
            self.inputs[stage] = value
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
        self.status = 0


class CCSFLogCholeskyTest(unittest.TestCase):
    def setUp(self):
        self.param = CCSFOptimizerParam()
        self.param.horizon = 3
        self.param.tf = 0.3
        self.param.robust_horizon = 1
        self.optimizer = CCSFOptimizer()
        self.optimizer.param = self.param
        self.optimizer.N = self.param.horizon
        self.optimizer.nx = 8
        self.optimizer.nu = 3
        self.optimizer.state = SimpleNamespace(
            _x=np.array([0.0, 0.0, 0.0]), _u=np.zeros(2)
        )

    def tearDown(self):
        # These tests never own generated solver files.
        self.optimizer.cleanup = lambda: None

    def _runtime_solver(self, status=0, on_solve=None):
        states = np.zeros((self.param.horizon + 1, 8))
        states[:, 4:] = self.optimizer._build_initial_covariance_state(self.param)
        return _NominalSolver(
            states, np.tile([0.2, 0.3, 0.2], (self.param.horizon, 1)),
            status=status, on_solve=on_solve,
        )

    def test_debug_mode_is_opt_in_and_disabling_it_clears_cached_diagnostics(self):
        self.assertFalse(self.param.debug_mode)
        self.optimizer.solver = self._runtime_solver()
        self.optimizer.psdf_param_dim = 7
        self.optimizer.ped_model = SimpleNamespace(
            get_params=Mock(return_value=np.tile(
                [0., 0., 0., 1., 1., 0., 0.], (self.param.horizon + 1, 1),
            )),
        )
        with (
            patch.object(self.optimizer, "_prepare_and_solve", return_value=0),
            redirect_stdout(io.StringIO()),
        ):
            self.optimizer.solve_nlp()
            self.optimizer.ped_model.get_params.assert_not_called()
            self.assertIsNone(self.optimizer.get_last_risk_margin_trajectory())

            self.param.debug_mode = True
            self.optimizer.solve_nlp()
            self.optimizer.ped_model.get_params.assert_called_once()
            self.assertIsNotNone(self.optimizer.get_last_risk_margin_trajectory())

            self.param.debug_mode = False
            self.optimizer.solve_nlp()
            self.optimizer.ped_model.get_params.assert_called_once()
            self.assertIsNone(self.optimizer.get_last_risk_margin_trajectory())

    def test_failed_solve_stops_immediately_and_preserves_failure_status(self):
        self.optimizer.create_model(self.param)
        for previous_speed, backup_status in ((-0.6, None), (0.6, 2)):
            with self.subTest(previous_speed=previous_speed, backup_status=backup_status):
                self.optimizer.state._x = np.array([0.2, -0.1, 0.4])
                self.optimizer.state._u = np.array([previous_speed, 0.9])
                self.optimizer._system_dynamics = None
                self.optimizer.solver = self._runtime_solver(status=4)
                self.optimizer.backup_solver = (
                    None if backup_status is None else self._runtime_solver(status=backup_status)
                )
                with (
                    patch("control.cc_sf_optimizer.report_solver_failure"),
                    redirect_stdout(io.StringIO()),
                ):
                    solution = self.optimizer.solve_nlp()
                self.assertEqual(solution.stats()["return_status"], "failure")
                self.assertEqual(solution.solver.status, 4 if backup_status is None else backup_status)
                np.testing.assert_array_equal(solution.get_input_trajectory(), np.zeros((3, 3)))
                expected = np.r_[self.optimizer.state._x, self.optimizer._current_s0]
                np.testing.assert_array_equal(
                    solution.get_state_trajectory()[:4].T, np.tile(expected, (4, 1)),
                )
        self.assertEqual(self.optimizer.get_runtime_stats()["safe_stop_count"], 2)

    def test_failed_stop_installation_cannot_return_a_stale_control(self):
        self.optimizer.create_model(self.param)
        self.optimizer.solver = self._runtime_solver(status=4)
        with (
            patch.object(self.optimizer.solver, "set", side_effect=RuntimeError("cannot install")),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(RuntimeError):
                self.optimizer.recovery_infeasible(4)
        self.assertEqual(self.optimizer.get_runtime_stats()["safe_stop_count"], 0)

    def test_solver_time_sums_recovery_calls_and_excludes_preparation_and_diagnostics(self):
        self.optimizer.create_model(self.param)
        self.param.debug_mode = True
        clock = SimpleNamespace(value=0.0)

        def main_call(_):
            clock.value += 0.25

        def backup_call(_):
            clock.value += 0.75

        self.optimizer.solver = self._runtime_solver(status=4, on_solve=main_call)
        self.optimizer.backup_solver = self._runtime_solver(on_solve=backup_call)
        original_path = self.optimizer._build_stage_path_parameter
        original_risk = self.optimizer._compute_stage_risk_data

        def path(*args, **kwargs):
            clock.value += 8.0
            return original_path(*args, **kwargs)

        def risk(*args, **kwargs):
            clock.value += 16.0
            return original_risk(*args, **kwargs)

        with (
            patch("control.cc_sf_optimizer.time.perf_counter", side_effect=lambda: clock.value),
            patch("control.cc_sf_optimizer.report_solver_failure"),
            patch.object(self.optimizer, "_build_stage_path_parameter", side_effect=path),
            patch.object(self.optimizer, "_compute_stage_risk_data", side_effect=risk),
            redirect_stdout(io.StringIO()),
        ):
            solution = self.optimizer.solve_nlp()
            self.assertIs(solution.solver, self.optimizer.backup_solver)
            self.assertEqual(solution.stats()["return_status"], "success")
            self.assertEqual(self.optimizer.solver_times, [1.0])
            self.optimizer.solver.result_status = 0
            self.optimizer.solve_nlp()
        self.assertEqual(self.optimizer.solver_times, [1.0, 0.25])
        self.assertGreater(clock.value, 100.0)

    def test_projected_margin_matches_factor_norm_for_signed_gradients(self):
        xi = np.array([np.log(0.03), np.log(0.07), np.log(0.02), -0.04])
        gradients = np.array(
            [[0.6, 0.8, -0.2], [0.6, -0.8, -0.2], [1.0, 0.0, 0.3],
             [0.0, -1.0, 0.4], [0.0, 0.0, 0.0]]
        )
        states = np.tile(xi, (len(gradients), 1))
        terms = self.optimizer._extract_projected_variance_terms(gradients)
        margins = self.optimizer._compute_margin_from_risk_terms(states, terms, self.param)
        factor = np.array([[0.03, 0.0, 0.0], [0.0, 0.07, 0.0], [0.0, -0.04, 0.02]])
        expected = self.param.chance_gamma * np.linalg.norm(gradients @ factor, axis=1)
        np.testing.assert_allclose(margins, expected, rtol=1e-14, atol=1e-16)

    def test_initial_margin_and_zero_gradient_have_finite_derivatives(self):
        # The bridge can build symbolic expressions without a numeric PSDF backend.
        self.optimizer.ped_model = AnalyticPSDFCasADi(None, name="test_sf_factor_margin")
        self.optimizer.setup_ocp(self.param, None)
        self.optimizer.add_obstacle_avoidance_constraint(self.param, None, None)
        model = self.optimizer.ocp.model
        self.assertEqual(self.optimizer.ocp.dims.nh_0, 1)
        self.assertEqual(self.optimizer.ocp.dims.nh, 1)
        self.assertEqual(self.optimizer.ocp.dims.nh_e, 0)
        self.assertEqual(model.con_h_expr_e.numel(), 0)
        self.assertEqual(self.optimizer.ocp.constraints.lh_e.size, 0)
        self.assertEqual(self.optimizer.ocp.constraints.uh_e.size, 0)
        self.assertEqual(self.optimizer.N, self.param.horizon)
        self.assertEqual(self.optimizer.ocp.solver_options.tf, self.param.tf)
        evaluate = ca.Function(
            "sf_initial_factor_margin",
            [model.x, model.p],
            [model.con_h_expr, model.con_h_expr_0,
             ca.jacobian(model.con_h_expr_0, model.x)],
        )
        state = np.r_[np.zeros(4), np.log([0.03, 0.07, 0.02]), -0.04]
        for gradient in (np.zeros(3), np.array([0.6, -0.8, 0.2])):
            with self.subTest(gradient=gradient):
                terms = self.optimizer._extract_projected_variance_terms(gradient)
                parameters = self.optimizer._build_stage_param(
                    np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                    terms,
                    1.0,
                    psdf_stage=np.r_[np.zeros(3), 0.8, gradient],
                    stage_idx=0,
                )
                stage_value, initial_value, jacobian = evaluate(state, parameters)
                margin = self.optimizer._compute_margin_from_risk_terms(
                    state[None, 4:], terms[None, :], self.param
                )[0]
                self.assertAlmostEqual(float(initial_value), 0.8 - self.param.d_min - margin)
                self.assertAlmostEqual(float(stage_value), float(initial_value))
                self.assertTrue(np.isfinite(np.asarray(jacobian)).all())
                if not np.any(gradient):
                    np.testing.assert_array_equal(np.asarray(jacobian), np.zeros((1, 8)))
                else:
                    self.assertGreater(np.linalg.norm(np.asarray(jacobian)[0, 4:]), 0.0)

    def test_preparation_regenerates_frozen_tail_and_keeps_risk_active(self):
        self.optimizer.create_model(self.param)
        states = np.zeros((self.param.horizon + 1, 8))
        states[:, 3] = np.arange(self.param.horizon + 1) * 0.02
        states[:, 4:] = [np.log(0.5), np.log(0.6), np.log(0.7), 0.1]
        solver = _NominalSolver(states, np.tile([0.2, 0.3, 0.2], (self.param.horizon, 1)))
        self.optimizer.solver = solver
        self.assertEqual(self.optimizer._prepare_and_solve(solver), 0)
        initial = self.optimizer._build_initial_covariance_state(self.param)
        np.testing.assert_array_equal(solver.states[0, 4:], initial)
        self.assertFalse(np.array_equal(solver.states[1, 4:], initial))
        np.testing.assert_array_equal(
            solver.states[1:, 4:], np.tile(solver.states[1, 4:], (self.param.horizon, 1))
        )
        for stage in range(self.param.horizon + 1):
            parameters = solver.parameters[stage]
            self.assertEqual(
                parameters[self.optimizer.covariance_propagation_param_slice][0],
                float(stage < self.param.robust_horizon),
            )
            self.assertEqual(parameters[self.optimizer.risk_margin_scale_param_slice][0], 1.0)

    def test_removed_stage_window_cannot_disable_future_or_terminal_margin(self):
        self.assertFalse(hasattr(self.param, "risk_margin_active_start_step"))
        self.assertFalse(hasattr(self.param, "risk_margin_active_end_step"))
        # A caller retaining old Python attributes must not reinstate a window.
        self.param.risk_margin_active_start_step = self.param.horizon + 1
        self.param.risk_margin_active_end_step = 0
        np.testing.assert_array_equal(
            self.optimizer._build_risk_margin_scale_batch(self.param.horizon + 1, self.param),
            np.ones(self.param.horizon + 1),
        )
        np.testing.assert_array_equal(
            self.optimizer.get_risk_margin_active_stage_mask(),
            np.ones(self.param.horizon + 1, dtype=bool),
        )

    def test_global_risk_disable_keeps_deterministic_initial_psdf(self):
        self.param.use_risk_margin = False
        self.optimizer.ped_model = AnalyticPSDFCasADi(None, name="test_sf_risk_disabled")
        self.optimizer.setup_ocp(self.param, None)
        self.optimizer.add_obstacle_avoidance_constraint(self.param, None, None)
        model = self.optimizer.ocp.model
        evaluate = ca.Function(
            "sf_global_risk_disabled",
            [model.x, model.p],
            [model.con_h_expr, model.con_h_expr_0,
             ca.jacobian(model.con_h_expr_0, model.x)],
        )
        gradient = np.array([0.6, -0.8, 0.2])
        terms = self.optimizer._extract_projected_variance_terms(gradient)
        parameters = self.optimizer._build_stage_param(
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            terms, 1.0,
            psdf_stage=np.r_[np.zeros(3), 0.8, gradient],
            stage_idx=0,
        )
        state = np.r_[np.zeros(4), np.log([0.03, 0.07, 0.02]), -0.04]
        stage_value, initial_value, jacobian = evaluate(state, parameters)
        self.assertAlmostEqual(float(stage_value), 0.8 - self.param.d_min)
        self.assertAlmostEqual(float(initial_value), 0.8 - self.param.d_min)
        np.testing.assert_array_equal(np.asarray(jacobian)[0, 4:], np.zeros(4))
        np.testing.assert_array_equal(
            self.optimizer.get_risk_margin_active_stage_mask(),
            np.zeros(self.param.horizon + 1, dtype=bool),
        )
        # Risk disable does not remove the four Log-Cholesky states.
        self.assertEqual(int(model.x.numel()), 8)

    def test_global_obstacle_disable_reports_no_active_margin(self):
        self.param.use_obstacle_constraint = False
        np.testing.assert_array_equal(
            self.optimizer.get_risk_margin_active_stage_mask(),
            np.zeros(self.param.horizon + 1, dtype=bool),
        )


if __name__ == "__main__":
    unittest.main()
