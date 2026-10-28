"""CC-MF-IMP routing and fail-stop contracts without native solver builds."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from control import normalize_optimizer_type, SUPPORTED_OPTIMIZER_TYPES
from control.cc_mf_imp_errors import MFImplicitRootError
from control.cc_mf_imp_optimizer import CCMFImpOptimizer, CCMFImpOptimizerParam
from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
from control.controller import BaseController
from models.mf_implicit import compute_mf_imp
from sim.simulation_mpc import simulation_mpc


class _FeatureWrapper:
    """Supply signed geometry and synthetic frozen features in one batch."""

    def __init__(self, *, failed_stage=None, phi=0.2):
        self.A = torch.zeros((1, 1, 2), dtype=torch.float32)
        self.device = self.A.device
        self.active_clusters = 1
        self.failed_stage = failed_stage
        self.phi = float(phi)
        self.implicit_calls = 0
        self.pose_calls = 0
        self.last_result = None

    def __call__(self, poses):
        self.pose_calls += 1
        size = len(poses)
        gradient = torch.zeros((size, 3), dtype=torch.float64)
        gradient[:, 0] = 1.0
        return torch.full((size,), self.phi, dtype=torch.float64), gradient

    def forward_mf_imp(self, z_bar, epsilon, d_min, **kwargs):
        self.implicit_calls += 1
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
        self.last_result = compute_mf_imp(
            torch.full((size,), self.phi, dtype=torch.float64),
            gradient, d, q, mask, z_bar, epsilon, d_min, **kwargs,
        )
        return self.last_result


class CCMFImplicitControllerTest(unittest.TestCase):
    def setUp(self):
        self.param = CCMFImpOptimizerParam()
        self.param.horizon = 3
        self.param.tf = 0.3
        self.param.chance_epsilon = 0.12
        self.optimizer = CCMFImpOptimizer()
        self.optimizer.param = self.param
        self.optimizer.N, self.optimizer.nx, self.optimizer.nu = 3, 8, 3
        self.optimizer.psdf_wrapper = _FeatureWrapper()
        self.z = np.tile(
            [0.0, 0.0, 0.3, 0.0, math.log(.02), math.log(.025), math.log(.03), .004],
            (4, 1),
        )

    def tearDown(self):
        # No native solver or generated workspace is owned by these tests.
        self.optimizer.solver = None
        self.optimizer.backup_solver = None

    def test_distance_rows_keep_negative_valid_margins_and_meter_sentinel(self):
        data = self.optimizer._compute_constraint_affine_data(self.z)
        np.testing.assert_array_equal(data['mf_mask'], [False, True, True, True])
        self.assertEqual(float(data['mf_c'][0]), 1.0)
        np.testing.assert_array_equal(data['mf_A'][0], np.zeros(8))
        residuals = np.einsum('ij,ij->i', data['mf_A'], self.z) + data['mf_c']
        self.assertTrue(np.all(residuals[1:] < 0.0))
        self.assertEqual(self.optimizer.psdf_wrapper.implicit_calls, 1)
        self.assertEqual(self.optimizer.psdf_wrapper.pose_calls, 0)

    def test_configuration_rejects_float_iteration_limit_before_solver_setup(self):
        self.param.mf_imp_root_max_iter = 12.0
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            self.optimizer._validate_implicit_params(self.param)

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

    def test_stage_window_and_terminal_policy_precede_root_request(self):
        self.param.mf_active_start_step = 2
        self.param.mf_active_end_step = 2
        self.param.mf_active_terminal = False
        data = self.optimizer._compute_constraint_affine_data(self.z)
        np.testing.assert_array_equal(data['mf_mask'], [False, False, True, False])
        np.testing.assert_array_equal(
            self.optimizer.psdf_wrapper.last_result.mf_required.numpy(),
            [False, False, True, False],
        )
        np.testing.assert_array_equal(data['mf_c'][[0, 1, 3]], np.ones(3))

    def test_domain_gate_is_strict_and_skips_root(self):
        self.optimizer.psdf_wrapper = _FeatureWrapper(phi=float(self.param.d_col))
        data = self.optimizer._compute_constraint_affine_data(self.z)
        self.assertFalse(data['mf_mask'].any())
        self.assertFalse(bool(self.optimizer.psdf_wrapper.last_result.mf_required.any()))
        np.testing.assert_array_equal(data['mf_c'], np.ones(4))

    def test_master_or_mf_switch_does_not_request_roots(self):
        for attribute in ('use_obstacle_constraint', 'use_row_mf'):
            with self.subTest(attribute=attribute):
                setattr(self.param, attribute, False)
                data = self.optimizer._compute_constraint_affine_data(self.z)
                self.assertFalse(data['mf_mask'].any())
                self.assertEqual(self.optimizer.psdf_wrapper.implicit_calls, 0)
                if attribute == 'use_row_mf':
                    self.assertEqual(self.optimizer.psdf_wrapper.pose_calls, 1)
                np.testing.assert_array_equal(data['mf_c'], np.ones(4))
                setattr(self.param, attribute, True)

    def test_required_feature_failure_never_retries_geometry_or_runs_qp(self):
        self.optimizer.psdf_wrapper = _FeatureWrapper(failed_stage=2)
        solver = Mock()
        self.optimizer.solver = solver
        self.optimizer._set_constraint_parameters = Mock()
        self.optimizer.recovery_infeasible = Mock()
        self.optimizer._get_current_s_bounds = Mock(return_value=(0.0, 1.0))
        self.optimizer._get_augmented_state_bounds = Mock(
            return_value=(np.arange(8), -np.ones(8), np.ones(8)),
        )
        self.optimizer._get_effective_v_s_max = Mock(return_value=1.0)
        self.optimizer._predict_stage_s_values = Mock(return_value=np.zeros(4))
        self.optimizer._propagate_covariance_state_batch = Mock(return_value=self.z[:, 4:].copy())
        solver.get.side_effect = lambda stage, field: (
            self.z[stage].copy() if field == 'x' else np.zeros(3)
        )
        with self.assertRaises(MFImplicitRootError):
            self.optimizer._prepare_and_solve(solver)
        self.assertEqual(self.optimizer.psdf_wrapper.implicit_calls, 1)
        self.assertEqual(self.optimizer.psdf_wrapper.pose_calls, 0)
        self.optimizer._set_constraint_parameters.assert_not_called()
        self.optimizer.recovery_infeasible.assert_not_called()
        solver.solve.assert_not_called()
        self.assertFalse(any(call.args[1] == 'p' for call in solver.set.call_args_list))

    def test_fresh_diagnostic_evaluation_does_not_replace_nominal_metrics(self):
        nominal = self.optimizer._compute_constraint_affine_data(self.z)
        self.optimizer._constraint_data = nominal
        probability = self.optimizer._nominal_probability_sum(nominal, self.z)
        self.optimizer._last_nominal_probability_sum = probability.copy()
        moved = self.z.copy()
        moved[:, 4] += 0.1
        fresh = self.optimizer._compute_constraint_affine_data(moved)
        self.assertIs(self.optimizer._constraint_data, nominal)
        np.testing.assert_array_equal(self.optimizer._last_nominal_probability_sum, probability)
        self.assertFalse(np.array_equal(fresh['probability_sum'], probability))

    def test_batch_exception_is_fail_stop_without_stagewise_retry(self):
        wrapper = self.optimizer.psdf_wrapper
        wrapper.forward_mf_imp = Mock(side_effect=RuntimeError('synthetic GPU failure'))
        with self.assertRaises(MFImplicitRootError) as raised:
            self.optimizer._compute_constraint_affine_data(self.z)
        self.assertIn('synthetic GPU failure', str(raised.exception))
        self.assertEqual(wrapper.forward_mf_imp.call_count, 1)
        self.assertEqual(wrapper.pose_calls, 0)

    def test_root_failure_latches_until_explicit_reset(self):
        failure = MFImplicitRootError('synthetic root failure', context='nominal')
        with patch.object(CCMFOptimizer, 'solve_nlp', side_effect=failure) as base_solve:
            for _ in range(2):
                with self.assertRaises(MFImplicitRootError):
                    self.optimizer.solve_nlp()
            self.assertEqual(base_solve.call_count, 1)
            self.optimizer.reset()
            with self.assertRaises(MFImplicitRootError):
                self.optimizer.solve_nlp()
            self.assertEqual(base_solve.call_count, 2)

    def test_pre_qp_failure_log_does_not_reuse_previous_status_time_or_slack(self):
        self.optimizer._cycle_solver_time = 99.0
        self.optimizer._last_main_status = 0
        self.optimizer.psdf_wrapper = _FeatureWrapper(failed_stage=2)

        def fail_preprocessing(optimizer):
            optimizer._cycle_solver_time = 0.0
            optimizer._current_psdf = None
            optimizer._compute_constraint_affine_data(self.z)

        with patch.object(CCMFOptimizer, 'solve_nlp', new=fail_preprocessing):
            with self.assertRaises(MFImplicitRootError):
                self.optimizer.solve_nlp()
        row = self.optimizer.get_cycle_log_rows()[-1]
        self.assertEqual(row['solve_mode'], 'implicit_root_failure')
        self.assertIsNone(row['solver_status'])
        self.assertIsNone(row['solver_time_s'])
        self.assertIsNone(row['psdf'])
        self.assertEqual(row['mf_imp_slack_m'], '[]')
        self.assertTrue(self.optimizer.get_implicit_log_rows())

    def test_backup_root_failure_is_not_converted_to_safe_stop(self):
        self.optimizer.solver = Mock()
        self.optimizer.backup_solver = Mock()
        self.optimizer._copy_primal_guess = Mock()
        self.optimizer._prepare_and_solve = Mock(
            side_effect=MFImplicitRootError('backup root failure', context='backup'),
        )
        with self.assertRaises(MFImplicitRootError):
            self.optimizer._try_backup_recovery()

    def test_debug_root_failure_is_not_suppressed(self):
        self.optimizer._read_diagnostic_states = Mock(return_value=self.z)
        self.optimizer._build_constraint_diagnostics = Mock(
            side_effect=MFImplicitRootError('debug root failure', context='debug_fresh'),
        )
        with self.assertRaises(MFImplicitRootError):
            self.optimizer.compute_diagnostics(Mock(), 0, 'sqp_rti')

    def test_base_controller_rethrows_without_reusing_previous_solution(self):
        optimizer = Mock()
        failure = MFImplicitRootError('synthetic root failure')
        optimizer.solve_nlp.side_effect = failure
        controller = BaseController(optimizer, self.param)
        stale_solution = Mock()
        controller._opt_sol = stale_solution
        with self.assertRaises(MFImplicitRootError):
            controller.generate_control_input(Mock(), Mock(), Mock(), Mock())
        stale_solution.value.assert_not_called()
        self.assertFalse(controller.get_last_solver_status_info()['success'])
        self.assertIsNone(controller.get_last_solver_status_info()['status_code'])

    def test_model_identity_and_dimensions_are_independent(self):
        baseline = CCMFOptimizer()
        implicit_model = self.optimizer.create_model(self.param)
        baseline_model = baseline.create_model(CCMFOptimizerParam())
        self.assertEqual(implicit_model.name, 'differential_drive_cc_mf_imp')
        self.assertEqual(baseline_model.name, 'differential_drive_cc_mf')
        self.assertEqual(int(implicit_model.x.shape[0]), 8)
        self.assertEqual(int(implicit_model.u.shape[0]), 3)
        self.assertEqual(self.optimizer.mf_param_dim, 9)
        self.assertNotEqual(self.optimizer._codegen_name, baseline._codegen_name)


class CCMFImplicitRoutingTest(unittest.TestCase):
    def test_alias_is_narrow_and_output_names_are_canonical(self):
        self.assertIn('cc_mf_imp', SUPPORTED_OPTIMIZER_TYPES)
        self.assertEqual(normalize_optimizer_type('cc-mf-imp'), 'cc_mf_imp')
        self.assertEqual(normalize_optimizer_type('cc_mf'), 'cc_mf')
        self.assertEqual(normalize_optimizer_type('other-method'), 'other-method')
        for name in ('cc_mf_imp', 'cc-mf-imp'):
            self.assertEqual(simulation_mpc._resolve_optimizer_variant(name), 'cc_mf_imp')
            self.assertIn('cc_mf_imp', simulation_mpc.build_output_name('rectangle', 'maze', name))

    def test_parameter_defaults_do_not_share_mutable_state_with_baseline(self):
        implicit = CCMFImpOptimizerParam()
        baseline = CCMFOptimizerParam()
        baseline.mf_slack_linear = 12345.0
        self.assertEqual(implicit.mf_slack_linear, 100.0)
        self.assertEqual(implicit.mf_slack_quadratic, 1.0)
        self.assertEqual(implicit.mf_imp_root_max_iter, 12)
        self.assertEqual(implicit.mf_imp_root_atol_m, 1e-8)

    def test_failure_snapshot_export_error_does_not_hide_root_failure(self):
        simulation = simulation_mpc.__new__(simulation_mpc)
        optimizer = Mock()
        optimizer.save_implicit_failure_snapshot.side_effect = OSError('snapshot storage unavailable')
        simulation.robot = SimpleNamespace(
            _system=SimpleNamespace(_time=1.5),
            _controller=SimpleNamespace(_optimizer=optimizer),
        )
        simulation.current_name = 'MPC_cc_mf_imp_rectangle_maze'
        simulation._get_output_dir = Mock(return_value='/tmp')
        simulation.save_cc_mf_imp_history_to_csv = Mock()
        simulation.save_trial_history_to_csv = Mock()
        simulation.save_pose_sdf_to_csv = Mock()
        failure = MFImplicitRootError('original root failure', context='nominal')
        with patch('pathlib.Path.open', side_effect=OSError('outcome storage unavailable')):
            simulation._preserve_implicit_failure(failure)
        self.assertIn('original root failure', simulation.last_run_outcome['failure_reason'])
        self.assertEqual(simulation.last_run_outcome['status'], 'error')
        simulation.save_cc_mf_imp_history_to_csv.assert_called_once()


if __name__ == '__main__':
    unittest.main()
