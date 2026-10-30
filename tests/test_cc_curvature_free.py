"""Curvature-free CC preparation contracts without solver generation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from control.cc_covariance import propagate_log_cholesky
from control.cc_ecs_optimizer import CCECSOptimizer, CCECSOptimizerParam
from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
from control.cc_sf_optimizer import CCSFOptimizer, CCSFOptimizerParam
from utils.spline_path import (
    build_cubic_spline_path_data,
    curvature,
    stage_parameter,
)


CC_TYPES = (
    (CCSFOptimizer, CCSFOptimizerParam),
    (CCECSOptimizer, CCECSOptimizerParam),
    (CCMFOptimizer, CCMFOptimizerParam),
)


class CCCurvatureFreeTest(unittest.TestCase):
    def make_optimizer(self, optimizer_type, param_type):
        optimizer = optimizer_type()
        self.addCleanup(optimizer.cleanup)
        optimizer.param = param_type()
        optimizer.param.horizon = 4
        optimizer.param.tf = 0.4
        optimizer.param.robust_horizon = 2
        optimizer.N = optimizer.param.horizon
        optimizer.state = SimpleNamespace(_x=np.array([0.1, -0.2, 0.3]), _u=np.zeros(2))
        optimizer.reference_path_data = build_cubic_spline_path_data(
            np.array([[0.0, 0.0], [0.5, 0.3], [1.0, -0.2], [1.5, 0.1]])
        )
        return optimizer

    def test_cc_stage_parameter_keeps_zero_compatibility_slot(self):
        for optimizer_type, param_type in CC_TYPES:
            with self.subTest(optimizer=optimizer_type.__name__):
                optimizer = self.make_optimizer(optimizer_type, param_type)
                progress = float(optimizer.reference_path_data["s_breaks"][1])
                expected = stage_parameter(
                    optimizer.reference_path_data, progress,
                    optimizer.param.tangent_reg_delta, optimizer.param.fd_eps,
                    include_curvature=False,
                )
                with patch(
                    "utils.spline_path.curvature",
                    side_effect=AssertionError("unused curvature must not be evaluated"),
                ):
                    actual = optimizer._build_stage_path_parameter(progress)
                self.assertEqual(actual.shape, (6,))
                np.testing.assert_array_equal(actual[:5], expected)
                self.assertEqual(actual[5], 0.0)

    def test_covariance_rollout_needs_no_path_evaluation_and_matches_curved_reference(self):
        inputs = np.array([
            [0.1, 0.2, 0.1], [0.2, -0.3, 0.2],
            [0.3, 0.1, 0.3], [-0.2, 0.4, 0.1],
        ])
        for optimizer_type, param_type in CC_TYPES:
            with self.subTest(optimizer=optimizer_type.__name__):
                optimizer = self.make_optimizer(optimizer_type, param_type)
                progress = np.linspace(0.0, optimizer.reference_path_data["s_breaks"][-1], 5)
                expected = []
                factor = optimizer._build_initial_covariance_state(optimizer.param)
                for stage, s_value in enumerate(progress):
                    expected.append(factor.copy())
                    if stage < optimizer.param.robust_horizon:
                        factor = propagate_log_cholesky(
                            factor, inputs[stage, 0], inputs[stage, 1],
                            curvature(optimizer.reference_path_data, s_value),
                            0.1, optimizer.param,
                        )
                with patch.object(
                    optimizer, "_build_stage_path_parameter",
                    side_effect=AssertionError("covariance must not evaluate path parameters"),
                ):
                    actual = optimizer._propagate_covariance_state_batch(inputs, progress)
                np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
                np.testing.assert_array_equal(actual[2:], np.tile(actual[2], (3, 1)))


if __name__ == "__main__":
    unittest.main()
