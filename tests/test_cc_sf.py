import unittest
from types import SimpleNamespace

import casadi as ca
import numpy as np

from control.analytic_psdf_casadi import AnalyticPSDFCasADi
from control.cc_sf_optimizer import CCSFOptimizer, CCSFOptimizerParam


class _NominalSolver:
    def __init__(self, states, inputs):
        self.states = np.asarray(states, dtype=float).copy()
        self.inputs = np.asarray(inputs, dtype=float).copy()
        self.parameters = {}

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
        return 0


class CCSFLogCholeskyTest(unittest.TestCase):
    def setUp(self):
        self.param = CCSFOptimizerParam()
        self.param.horizon = 3
        self.param.tf = 0.3
        self.param.robust_horizon = 1
        self.param.debug_risk = False
        self.optimizer = CCSFOptimizer()
        self.optimizer.param = self.param
        self.optimizer.N = self.param.horizon
        self.optimizer.nx = 8
        self.optimizer.nu = 3
        self.optimizer.state = SimpleNamespace(
            _x=np.array([0.0, 0.0, 0.0]), _u=np.zeros(2)
        )

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

    def test_terminal_margin_and_zero_gradient_have_finite_derivatives(self):
        # The bridge can build symbolic expressions without a numeric PSDF backend.
        self.optimizer.ped_model = AnalyticPSDFCasADi(None, name="test_sf_factor_margin")
        self.optimizer.setup_ocp(self.param, None)
        self.optimizer.add_obstacle_avoidance_constraint(self.param, None, None)
        model = self.optimizer.ocp.model
        self.assertEqual(self.optimizer.ocp.dims.nh_e, 1)
        evaluate = ca.Function(
            "sf_terminal_factor_margin",
            [model.x, model.p],
            [model.con_h_expr, model.con_h_expr_e,
             ca.jacobian(model.con_h_expr_e, model.x)],
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
                    stage_idx=self.param.horizon,
                )
                stage_value, terminal_value, jacobian = evaluate(state, parameters)
                margin = self.optimizer._compute_margin_from_risk_terms(
                    state[None, 4:], terms[None, :], self.param
                )[0]
                self.assertAlmostEqual(float(terminal_value), 0.8 - self.param.d_min - margin)
                self.assertAlmostEqual(float(stage_value), float(terminal_value))
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

    def test_global_risk_disable_keeps_deterministic_terminal_psdf(self):
        self.param.use_risk_margin = False
        self.optimizer.ped_model = AnalyticPSDFCasADi(None, name="test_sf_risk_disabled")
        self.optimizer.setup_ocp(self.param, None)
        self.optimizer.add_obstacle_avoidance_constraint(self.param, None, None)
        model = self.optimizer.ocp.model
        evaluate = ca.Function(
            "sf_global_risk_disabled",
            [model.x, model.p],
            [model.con_h_expr, model.con_h_expr_e,
             ca.jacobian(model.con_h_expr_e, model.x)],
        )
        gradient = np.array([0.6, -0.8, 0.2])
        terms = self.optimizer._extract_projected_variance_terms(gradient)
        parameters = self.optimizer._build_stage_param(
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            terms, 1.0,
            psdf_stage=np.r_[np.zeros(3), 0.8, gradient],
            stage_idx=self.param.horizon,
        )
        state = np.r_[np.zeros(4), np.log([0.03, 0.07, 0.02]), -0.04]
        stage_value, terminal_value, jacobian = evaluate(state, parameters)
        self.assertAlmostEqual(float(stage_value), 0.8 - self.param.d_min)
        self.assertAlmostEqual(float(terminal_value), 0.8 - self.param.d_min)
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
