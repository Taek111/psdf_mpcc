"""Inverse-CDF controller contracts without native solver generation."""

import json
import math
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from acados_template import AcadosOcp

from control.cc_mf_icdf_optimizer import (
    CCMFICDFOptimizer,
    CCMFICDFOptimizerParam,
    MFInverseCDFEvaluationError,
)
from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
from models.mf_inverse_cdf import compute_mf_icdf, evaluate_mf_icdf_frozen


class _FeatureWrapper:
    """Supply frozen synthetic features through the real numerical kernel."""

    def __init__(self, *, failed_stage=None, phi=0.2, no_obstacles=False):
        self.A = torch.zeros((1, 1, 2), dtype=torch.float32)
        self.device = self.A.device
        self.active_clusters = 0 if no_obstacles else 1
        self.failed_stage = failed_stage
        self.phi = phi
        self.icdf_calls = 0
        self.pose_calls = 0
        self.last_result = None

    def __call__(self, poses):
        self.pose_calls += 1
        gradient = torch.zeros((len(poses), 3), dtype=torch.float64)
        gradient[:, 0] = 1.0
        phi = torch.as_tensor(self.phi, dtype=torch.float64).expand(len(poses))
        return phi, gradient

    def forward_mf_icdf(self, z_bar, epsilon, d_min, **kwargs):
        self.icdf_calls += 1
        z_bar = torch.as_tensor(z_bar, dtype=torch.float64)
        size = len(z_bar)
        d = torch.full((size, 1), 0.01, dtype=torch.float64)
        q = torch.zeros((size, 1, 3), dtype=torch.float64)
        q[..., 0] = 1.0
        mask = torch.ones((size, 1), dtype=torch.bool)
        if self.failed_stage is not None:
            mask[self.failed_stage] = False
        gradient = torch.zeros((size, 3), dtype=torch.float64)
        gradient[:, 0] = 1.0
        self.last_result = compute_mf_icdf(
            torch.as_tensor(self.phi, dtype=torch.float64).expand(size),
            gradient, d, q, mask, z_bar, epsilon, d_min,
            no_obstacles=self.active_clusters == 0, **kwargs,
        )
        return self.last_result

    evaluate_mf_icdf_frozen = staticmethod(evaluate_mf_icdf_frozen)


class CCMFInverseCDFControllerTest(unittest.TestCase):
    def setUp(self):
        self.param = CCMFICDFOptimizerParam()
        self.param.horizon = 3
        self.param.tf = 0.3
        self.param.chance_epsilon = 0.12
        self.optimizer = CCMFICDFOptimizer()
        self.optimizer.param = self.param
        self.optimizer.N, self.optimizer.nx, self.optimizer.nu = 3, 8, 3
        self.optimizer.psdf_wrapper = _FeatureWrapper()
        self.z = np.tile(
            [0.0, 0.0, 0.3, 0.0, math.log(.02), math.log(.025), math.log(.03), .004],
            (4, 1),
        )

    def tearDown(self):
        self.optimizer.solver = None
        self.optimizer.backup_solver = None

    def _solver(self):
        solver = Mock()
        solver.get.side_effect = lambda stage, field: (
            self.z[stage].copy() if field == "x" else
            np.array([.25]) if field == "sl" else np.zeros(3)
        )
        return solver

    def _mock_preparation(self, solver):
        self.optimizer._get_current_s_bounds = Mock(return_value=(0., 1.))
        self.optimizer._get_augmented_state_bounds = Mock(
            return_value=(np.arange(8), -np.ones(8), np.ones(8)),
        )
        self.optimizer._get_effective_v_s_max = Mock(return_value=1.)
        self.optimizer._predict_stage_s_values = Mock(return_value=np.zeros(4))
        self.optimizer._propagate_covariance_state_batch = Mock(
            return_value=self.z[:, 4:].copy(),
        )
        self.optimizer.solver = solver

    def test_negative_residual_is_valid_and_guard_is_unchanged(self):
        data = self.optimizer._compute_constraint_affine_data(self.z)
        np.testing.assert_array_equal(data["mf_mask"], [False, True, True, True])
        np.testing.assert_array_equal(data["mf_A"][0], np.zeros(8))
        self.assertEqual(data["mf_c"][0], 1.)
        residual = np.einsum("ij,ij->i", data["mf_A"], self.z) + data["mf_c"]
        self.assertTrue(np.all(residual[1:] < 0))
        np.testing.assert_allclose(residual[1:], data["residual"][1:])
        np.testing.assert_array_equal(data["mf_A"][:, 3], np.zeros(4))
        guard = np.einsum("ij,ij->i", data["guard_A"], self.z) + data["guard_c"]
        np.testing.assert_allclose(guard, .2 - self.param.d_col)
        self.assertEqual(self.optimizer.psdf_wrapper.icdf_calls, 1)
        self.assertEqual(self.optimizer.psdf_wrapper.pose_calls, 0)

    def test_d_col_controls_guard_and_mf_gate_after_parameter_change(self):
        self.assertEqual(self.param.d_col, 0.0002)
        self.assertEqual(self.param.d_min, 0.0002)
        self.assertFalse(hasattr(self.param, "d_mf_mask"))
        self.optimizer.psdf_wrapper.phi = 0.0003
        for threshold, active in ((0.0004, False), (0.0002, True)):
            with self.subTest(d_col=threshold):
                self.param.d_col = threshold
                data = self.optimizer._compute_constraint_affine_data(self.z)
                np.testing.assert_array_equal(data["mf_mask"], [False] + [active] * 3)
                guard = np.einsum("ij,ij->i", data["guard_A"], self.z) + data["guard_c"]
                np.testing.assert_allclose(guard, 0.0003 - threshold)
                self.assertEqual(self.param.d_min, 0.0002)

    def test_stage_window_terminal_and_strict_domain_masks(self):
        self.param.mf_active_start_step = 2
        self.param.mf_active_end_step = 2
        self.param.mf_active_terminal = False
        data = self.optimizer._compute_constraint_affine_data(self.z)
        np.testing.assert_array_equal(data["mf_mask"], [False, False, True, False])
        self.optimizer.psdf_wrapper.phi = self.param.d_col
        data = self.optimizer._compute_constraint_affine_data(self.z)
        self.assertFalse(data["mf_mask"].any())
        np.testing.assert_array_equal(data["mf_c"], np.ones(4))

    def test_disabled_row_skips_covariance_evaluation_and_retains_guard(self):
        self.param.use_row_mf = False
        data = self.optimizer._compute_constraint_affine_data(self.z)
        self.assertFalse(data["mf_mask"].any())
        self.assertEqual(self.optimizer.psdf_wrapper.icdf_calls, 0)
        self.assertEqual(self.optimizer.psdf_wrapper.pose_calls, 1)
        self.assertTrue(np.isnan(data["probability_sum"]).all())
        self.param.use_row_mf = True
        self.param.use_obstacle_constraint = False
        data = self.optimizer._compute_constraint_affine_data(self.z)
        self.assertFalse(data["mf_mask"].any())
        self.assertEqual(self.optimizer.psdf_wrapper.icdf_calls, 0)
        np.testing.assert_array_equal(data["guard_A"], np.zeros((4, 8)))

    def test_no_obstacles_are_distinct_from_missing_required_features(self):
        self.optimizer.psdf_wrapper = _FeatureWrapper(no_obstacles=True)
        data = self.optimizer._compute_constraint_affine_data(self.z)
        self.assertFalse(data["mf_mask"].any())
        np.testing.assert_array_equal(data["probability_sum"], np.zeros(4))
        self.assertTrue(np.isnan(data["residual"]).all())
        self.optimizer.psdf_wrapper = _FeatureWrapper(failed_stage=2)
        with self.assertRaisesRegex(MFInverseCDFEvaluationError, "stage 2: no_valid_features"):
            self.optimizer._compute_constraint_affine_data(self.z)

    def test_epsilon_validation_is_isolated_from_probability_mode(self):
        for value in (0., np.finfo(np.float64).tiny, .5, .9, float("nan")):
            self.param.chance_epsilon = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.optimizer._normalize_epsilon_values(4)
        self.param.chance_epsilon = [.1, .15, .2, .25]
        np.testing.assert_array_equal(
            self.optimizer._normalize_epsilon_values(4), self.param.chance_epsilon,
        )
        baseline = CCMFOptimizer()
        baseline.param = CCMFOptimizerParam()
        baseline.param.chance_epsilon = .9
        np.testing.assert_array_equal(baseline._normalize_epsilon_values(4), [.9] * 4)

    def test_setup_validates_before_native_solver_creation(self):
        self.param.chance_epsilon = .5
        with patch.object(CCMFOptimizer, "setup") as base_setup:
            with self.assertRaises(ValueError):
                self.optimizer.setup(self.param, Mock(), Mock(), Mock())
        base_setup.assert_not_called()

    def test_active_failure_precedes_qp_and_clears_previous_metadata(self):
        solver = self._solver()
        self._mock_preparation(solver)
        self.optimizer.psdf_wrapper = _FeatureWrapper(failed_stage=2)
        self.optimizer._set_constraint_parameters = Mock()
        self.optimizer._constraint_data = {"stale": True}
        self.optimizer._current_psdf = 42.
        self.optimizer._last_nominal_probability_sum = np.ones(4)
        with self.assertRaises(MFInverseCDFEvaluationError):
            self.optimizer._prepare_and_solve(solver)
        self.optimizer._set_constraint_parameters.assert_not_called()
        solver.solve.assert_not_called()
        self.assertIsNone(self.optimizer._constraint_data)
        self.assertIsNone(self.optimizer._current_psdf)
        self.assertIsNone(self.optimizer._last_nominal_probability_sum)
        self.assertEqual(self.optimizer.psdf_wrapper.icdf_calls, 1)

    def test_batch_failure_has_no_stagewise_probability_fallback(self):
        wrapper = self.optimizer.psdf_wrapper
        wrapper.forward_mf_icdf = Mock(side_effect=RuntimeError("synthetic failure"))
        with self.assertRaisesRegex(MFInverseCDFEvaluationError, "synthetic failure"):
            self.optimizer._compute_constraint_affine_data(self.z)
        wrapper.forward_mf_icdf.assert_called_once()
        self.assertEqual(wrapper.pose_calls, 0)

    def test_nonfinite_guard_is_fatal_even_when_mf_domain_would_be_inactive(self):
        self.optimizer.psdf_wrapper.phi = float("nan")
        with self.assertRaisesRegex(MFInverseCDFEvaluationError, "PSDF guard geometry"):
            self.optimizer._compute_constraint_affine_data(self.z)

    def test_backup_and_diagnostic_failures_are_not_suppressed(self):
        self.optimizer.solver = Mock()
        self.optimizer.backup_solver = Mock()
        self.optimizer._copy_primal_guess = Mock()
        error = MFInverseCDFEvaluationError("evaluation failure")
        self.optimizer._prepare_and_solve = Mock(side_effect=error)
        with self.assertRaises(MFInverseCDFEvaluationError):
            self.optimizer._try_backup_recovery()
        self.optimizer._read_diagnostic_states = Mock(return_value=self.z)
        self.optimizer._build_constraint_diagnostics = Mock(side_effect=error)
        with self.assertRaises(MFInverseCDFEvaluationError):
            self.optimizer.compute_diagnostics(Mock(), 0, "sqp_rti")

    def test_fresh_and_frozen_diagnostics_preserve_nominal_metadata(self):
        self.param.debug_mf = True
        nominal = self.optimizer._compute_constraint_affine_data(self.z)
        self.optimizer._constraint_data = nominal
        probability = self.optimizer._nominal_probability_sum(nominal, self.z)
        self.optimizer._last_nominal_probability_sum = probability.copy()
        moved = self.z.copy()
        moved[:, 0] += .002
        moved[:, 4] += .1
        fresh = self.optimizer._compute_constraint_affine_data(moved)
        affine = np.einsum("ij,ij->i", nominal["mf_A"], moved) + nominal["mf_c"]
        diagnostics = self.optimizer._extend_constraint_diagnostics(
            {"mf_affine_residual": affine}, moved, fresh,
        )
        self.assertIs(self.optimizer._constraint_data, nominal)
        np.testing.assert_array_equal(self.optimizer._last_nominal_probability_sum, probability)
        np.testing.assert_array_equal(diagnostics["mf_probability_sum_nominal"], probability)
        self.assertFalse(np.array_equal(fresh["probability_sum"], probability))
        self.assertFalse(np.allclose(
            diagnostics["mf_icdf_residual_frozen"][1:],
            diagnostics["mf_icdf_residual_fresh"][1:],
        ))
        self.assertTrue(np.isnan(diagnostics["mf_icdf_affine_minus_frozen"][0]))
        np.testing.assert_allclose(
            diagnostics["mf_icdf_affine_minus_frozen"][1:],
            affine[1:] - diagnostics["mf_icdf_residual_frozen"][1:],
        )

    def test_probability_log_uses_raw_sum_and_quantile_slack_units(self):
        data = self.optimizer._compute_constraint_affine_data(self.z)
        self.optimizer._constraint_data = data
        self.optimizer._last_nominal_probability_sum = self.optimizer._nominal_probability_sum(
            data, self.z,
        )
        self.optimizer._record_experiment_logs("sqp_rti", 0, self._solver(), None)
        row = self.optimizer.get_cycle_log_rows()[-1]
        self.assertEqual(set(row), set(self.optimizer.get_cycle_log_fields()))
        np.testing.assert_allclose(json.loads(row["mf_probability_sum"]), data["probability_sum"])
        self.assertEqual(row["residual_kind"], "inverse_cdf")
        self.assertEqual(row["slack_unit"], "standard_normal_quantile")
        self.assertEqual(json.loads(row["mf_icdf_residual_nominal"])[0], None)
        self.assertEqual(json.loads(row["mf_slack"])[1:], [.25] * 3)
        wrong_probability = data["epsilon"][1:] - data["residual"][1:]
        self.assertFalse(np.allclose(wrong_probability, data["probability_sum"][1:]))

    def test_constraint_log_schema_keeps_probability_and_residual_separate(self):
        data = self.optimizer._compute_constraint_affine_data(self.z)
        self.optimizer._constraint_data = data
        mf = np.einsum("ij,ij->i", data["mf_A"], self.z) + data["mf_c"]
        guard = np.einsum("ij,ij->i", data["guard_A"], self.z) + data["guard_c"]
        diagnostics = {
            "constraint_stages": np.arange(1, 4),
            "mf_mask": data["mf_mask"], "mf_valid": data["mf_valid"],
            "guard_affine_residual": guard, "mf_affine_residual": mf,
            "mf_lower_slack": np.ones(4), "mf_activation_phi": data["phi"],
            "d_col": self.param.d_col,
            "mf_domain_eligible": data["phi"] > self.param.d_col,
            "solver": {"qp_fixed_row_lower_residual": np.column_stack((guard, mf + 1))},
        }
        self.optimizer._extend_constraint_diagnostics(diagnostics, self.z, data)
        self.optimizer._record_constraint_log(diagnostics)
        row = self.optimizer.get_constraint_log_rows()[0]
        self.assertEqual(set(row), set(self.optimizer.get_constraint_log_fields()))
        self.assertEqual(row["mf_probability_sum_nominal"], data["probability_sum"][1])
        self.assertEqual(row["mf_icdf_residual_nominal"], data["residual"][1])
        self.assertIsNone(row["mf_icdf_residual_frozen"])
        self.assertEqual(row["slack_unit"], "standard_normal_quantile")

    def test_parameter_layout_has_trivial_stage_zero_and_nine_mf_values(self):
        self.optimizer.create_model(self.param)
        self.optimizer._build_stage_path_parameter = Mock(return_value=np.zeros(6))
        data = self.optimizer._compute_constraint_affine_data(self.z)
        solver = Mock()
        self.optimizer._set_constraint_parameters(solver, np.zeros(4), data)
        rows = {call.args[0]: call.args[2] for call in solver.set.call_args_list}
        self.assertEqual(set(rows), {0, 1, 2, 3})
        np.testing.assert_array_equal(
            rows[0][self.optimizer.mf_param_slice], [0.] * 8 + [1.],
        )
        np.testing.assert_array_equal(
            rows[0][self.optimizer.guard_param_slice], [0., 0., 0., 1.],
        )
        np.testing.assert_array_equal(
            rows[1][self.optimizer.mf_param_slice],
            np.r_[data["mf_A_raw"][1], data["mf_c_raw"][1]],
        )

    def test_model_dimensions_and_hard_guard_soft_mf_layout_are_preserved(self):
        baseline = CCMFOptimizer()
        model = self.optimizer.create_model(self.param)
        baseline_model = baseline.create_model(CCMFOptimizerParam())
        self.assertEqual(model.name, "differential_drive_cc_mf_icdf")
        self.assertEqual(baseline_model.name, "differential_drive_cc_mf")
        self.assertEqual(int(model.x.shape[0]), 8)
        self.assertEqual(int(model.u.shape[0]), 3)
        self.assertEqual(self.optimizer.mf_param_dim, 9)
        self.assertNotEqual(self.optimizer._codegen_name, baseline._codegen_name)
        self.optimizer.ocp = AcadosOcp()
        self.optimizer.ocp.model = model
        self.param.use_obstacle_constraint = False
        self.optimizer.add_obstacle_avoidance_constraint(self.param, [])
        self.assertEqual(self.optimizer.ocp.dims.nh, 2)
        self.assertEqual(self.optimizer.ocp.dims.nsh, 1)
        np.testing.assert_array_equal(self.optimizer.ocp.constraints.idxsh, [1])
        np.testing.assert_array_equal(self.optimizer.ocp.constraints.idxsh_e, [1])


if __name__ == "__main__":
    unittest.main()
