"""Validate the optional covariance RTI proximal cost without a solver build."""

import math
import unittest
from unittest.mock import Mock, patch

import casadi as ca
import numpy as np

from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam


class CCMFCovarianceProximalTest(unittest.TestCase):
    def setUp(self):
        self.param = CCMFOptimizerParam()
        # Zero-cost fixtures exercise explicit opt-out from the tuned default.
        self.param.covariance_proximal_weight = 0.0
        self.param.horizon = 4
        self.param.robust_horizon = 3
        self.param.tf = .4
        self.param.a_max = self.param.alpha_max = None
        self.param.r_delta_v = self.param.r_delta_omega = 0.0
        self.optimizer = CCMFOptimizer()
        self.optimizer.param = self.param
        self.optimizer.N = self.param.horizon
        self.anchor = np.array([math.log(.02), math.log(.025), math.log(.03), .004])
        self.path = np.array([0., 0., 1., 0., 0., 0.])
        self.guard = np.array([0., 0., 0., 1.])
        self.mf = np.r_[np.zeros(8), 1.]

    def _parameters(self, stage=1, anchor=None):
        return self.optimizer._build_stage_param(
            self.path, self.guard, self.mf, stage,
            nominal_covariance=self.anchor if anchor is None else anchor,
        )

    @staticmethod
    def _function(model):
        return ca.Function("proximal_test", [model.x, model.u, model.p], [
            model.cost_expr_ext_cost,
            model.cost_expr_ext_cost_e,
            ca.gradient(model.cost_expr_ext_cost, model.x),
            ca.hessian(model.cost_expr_ext_cost, model.x)[0],
            model.disc_dyn_expr,
        ])

    def test_zero_weight_preserves_original_parameter_layout_and_cost(self):
        self.assertEqual(self.param.covariance_proximal_weight, 0.0)
        model = self.optimizer.create_model(self.param)
        self.assertEqual(model.p.numel(), 20)
        self.assertEqual(self._parameters().shape, (20,))
        covariance_gradient = ca.Function(
            "zero_gradient", [model.x, model.u, model.p],
            [ca.gradient(model.cost_expr_ext_cost, model.x)[4:8]],
        )
        value = covariance_gradient(np.r_[np.zeros(4), self.anchor], np.zeros(3), self._parameters())
        np.testing.assert_array_equal(np.asarray(value).ravel(), np.zeros(4))

    def test_cost_is_zero_at_anchor_and_has_expected_positive_hessian(self):
        baseline = self._function(self.optimizer.create_model(self.param))
        self.param.covariance_proximal_weight = 2.5
        model = self.optimizer.create_model(self.param)
        enabled = self._function(model)
        self.assertEqual(model.p.numel(), 26)
        p = self._parameters()
        x = np.r_[np.zeros(4), self.anchor]
        u = np.array([.02, .1, .05])
        before = baseline(x, u, p[:20])
        after = enabled(x, u, p)
        np.testing.assert_allclose(after[0], before[0], atol=1e-14)
        np.testing.assert_allclose(after[1], before[1], atol=1e-14)
        np.testing.assert_allclose(after[2], before[2], atol=1e-14)
        np.testing.assert_array_equal(after[4], before[4])
        heading_std = math.hypot(.03, .004)
        expected = np.diag([5., 5., 5., 5. / heading_std**2])
        difference = np.asarray(after[3]) - np.asarray(before[3])
        np.testing.assert_allclose(difference[4:8, 4:8], expected, rtol=1e-13)
        self.assertGreaterEqual(np.linalg.eigvalsh(difference).min(), 0.)
        x[6] -= 2.
        changed = enabled(x, u, p)
        self.assertAlmostEqual(float(changed[0] - before[0]), 10.)
        self.assertAlmostEqual(float(changed[1] - before[1]), 10.)

    def test_stage_schedule_excludes_initial_and_frozen_tail(self):
        self.param.covariance_proximal_weight = 1.
        self.optimizer.create_model(self.param)
        proximal = self.optimizer.covariance_proximal_param_slice
        expected_std = math.hypot(.03, .004)
        for stage, active in enumerate((0., 1., 1., 1., 0.)):
            with self.subTest(stage=stage):
                values = self._parameters(stage)[proximal]
                np.testing.assert_array_equal(values[:4], self.anchor)
                self.assertAlmostEqual(values[4], 1. / expected_std)
                self.assertEqual(values[5], active)
        self.param.robust_horizon = self.param.horizon
        self.optimizer.create_model(self.param)
        self.assertEqual(self._parameters(self.param.horizon)[-1], 1.)

    def test_constraint_parameters_use_existing_nominal_covariance_only(self):
        self.param.covariance_proximal_weight = 1.
        self.optimizer.create_model(self.param)
        nominal = np.tile(np.r_[np.zeros(4), self.anchor], (5, 1))
        nominal[:, 6] += np.arange(5) * .1
        data = {
            "nominal_state": nominal,
            "guard_A": np.zeros((5, 8)), "guard_c": np.ones(5),
            "mf_A": np.zeros((5, 8)), "mf_c": np.ones(5),
            "epsilon": np.full(5, .2), "mf_mask": np.ones(5, dtype=bool),
        }
        solver = Mock()
        with patch.object(self.optimizer, "_build_stage_path_parameter", return_value=self.path), \
                patch.object(self.optimizer, "_propagate_covariance_state_batch") as rollout:
            self.optimizer._set_constraint_parameters(solver, np.zeros(5), data)
            rollout.assert_not_called()
        self.assertEqual(solver.set.call_count, 5)
        proximal = self.optimizer.covariance_proximal_param_slice
        for stage, call in enumerate(solver.set.call_args_list):
            self.assertEqual(call.args[:2], (stage, "p"))
            np.testing.assert_array_equal(call.args[2][proximal][:4], nominal[stage, 4:8])

    def test_cross_factor_normalization_has_finite_floor(self):
        self.param.covariance_proximal_weight = 1.
        self.optimizer.create_model(self.param)
        anchor = self.anchor.copy()
        anchor[2:] = [math.log(1e-12), 0.]
        self.assertEqual(self._parameters(anchor=anchor)[-2], 1e6)

    def test_invalid_weight_fails_before_model_or_solver_setup(self):
        for weight in (-.1, math.nan, math.inf, True, "1", None, [1.]):
            with self.subTest(weight=weight):
                self.param.covariance_proximal_weight = weight
                with self.assertRaisesRegex(ValueError, "covariance_proximal_weight"):
                    self.optimizer.create_model(self.param)
                with self.assertRaisesRegex(ValueError, "covariance_proximal_weight"):
                    self.optimizer.setup(self.param, Mock(), Mock(), Mock())


if __name__ == "__main__":
    unittest.main()
