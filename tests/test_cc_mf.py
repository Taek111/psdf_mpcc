import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import casadi as ca
import numpy as np
import torch

from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
from control.cc_sf_optimizer import CCSFOptimizer, CCSFOptimizerParam
from control.cc_covariance import (
    covariance_from_log_cholesky,
)
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from sim.simulation_mpc import simulation_mpc


class _FakeState:
    def __init__(self, pose):
        self._x = np.asarray(pose, dtype=float)
        self._u = np.zeros(2, dtype=float)


class _FakeDynamics:
    @staticmethod
    def forward_dynamics(x, u, timestep):
        return np.array(
            [
                x[0] + timestep * u[0] * np.cos(x[2]),
                x[1] + timestep * u[0] * np.sin(x[2]),
                x[2] + timestep * u[1],
            ],
            dtype=float,
        )


class _FakeSolver:
    def __init__(self, states, inputs_):
        self.states = np.asarray(states, dtype=float).copy()
        self.inputs = np.asarray(inputs_, dtype=float).copy()

    def get(self, stage, field):
        if field == "x":
            return self.states[stage].copy()
        if field == "u":
            return self.inputs[stage].copy()
        raise KeyError(field)
    def set(self, stage, field, value):
        if field == "x":
            self.states[stage] = np.asarray(value, dtype=float)
            return
        if field == "u":
            self.inputs[stage] = np.asarray(value, dtype=float)
            return
        raise KeyError(field)


class _StagewiseFallbackPSDFWrapper:
    def __init__(self, failed_stage):
        self.A = torch.zeros((1, 1, 2), dtype=torch.float64)
        self.device = self.A.device
        self.active_clusters = 1
        self.failed_stage = int(failed_stage)
        self.mf_call_sizes = []
        self.mf_single_stages = []
        self.pose_call_sizes = []

    @staticmethod
    def expected_phi(poses):
        return 0.20 + 0.05 * poses[:, 0]

    @staticmethod
    def expected_gradient(poses):
        gradient = torch.zeros(
            (poses.shape[0], 3),
            dtype=poses.dtype,
            device=poses.device,
        )
        gradient[:, 0] = 1.0
        gradient[:, 1] = 0.25
        gradient[:, 2] = -0.10
        return gradient

    @staticmethod
    def expected_mf_A(stage, dtype=torch.float64):
        return (stage + 1.0) * torch.arange(1, 9, dtype=dtype) / 20.0

    @staticmethod
    def expected_mf_c(stage):
        return -0.05 * (stage + 1.0)

    def __call__(self, poses):
        poses_t = torch.as_tensor(poses, dtype=self.A.dtype, device=self.device)
        self.pose_call_sizes.append(int(poses_t.shape[0]))
        return self.expected_phi(poses_t), self.expected_gradient(poses_t)

    def forward_mf(self, z_bar, epsilon, d_min):
        del epsilon, d_min
        z_bar_t = torch.as_tensor(z_bar, dtype=self.A.dtype, device=self.device)
        self.mf_call_sizes.append(int(z_bar_t.shape[0]))
        if z_bar_t.shape[0] > 1:
            raise RuntimeError("synthetic batched MF failure")

        stage = int(round(float(z_bar_t[0, 0].item())))
        self.mf_single_stages.append(stage)
        if stage == self.failed_stage:
            raise ValueError("synthetic single-stage MF failure")

        phi = self.expected_phi(z_bar_t[:, :3])
        gradient = self.expected_gradient(z_bar_t[:, :3])
        A_mf = self.expected_mf_A(stage, dtype=z_bar_t.dtype).unsqueeze(0)
        c_mf = torch.tensor(
            [self.expected_mf_c(stage)],
            dtype=z_bar_t.dtype,
            device=z_bar_t.device,
        )
        return phi, gradient, A_mf, c_mf


class CCMFMultiFeatureTest(unittest.TestCase):
    def setUp(self):
        self.param = CCMFOptimizerParam()
        self.param.horizon = 3
        self.param.tf = 0.3
        self.optimizer = CCMFOptimizer()
        self.optimizer.param = self.param
        self.optimizer.N = self.param.horizon
        self.optimizer.nx = 8
        self.optimizer.nu = 3
        self.optimizer.reference_path_data = None
        self.optimizer._current_s0 = 0.05
        self.optimizer.state = _FakeState([0.25, -0.1, 0.2])
        self.optimizer._system_dynamics = _FakeDynamics()

    def tearDown(self):
        self.optimizer.ocp = None

    def test_default_covariance_growth_scale_is_point_six(self):
        self.assertAlmostEqual(self.param.covariance_growth_scale, 0.6)
        expected = {
            "alpha_f": 0.0012,
            "alpha_v": 0.00024,
            "alpha_kappa": 0.006,
            "beta_v": 0.012,
            "beta_kappa": 0.0048,
            "beta_omega": 0.0048,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(
                getattr(self.param, key),
                value,
                msg=f"unexpected default for {key}",
            )

    def _setup_constraint_ocp(self):
        self.optimizer.setup_ocp(self.param)
        verts = torch.tensor(
            [[-0.1, -0.05], [0.1, -0.05], [0.1, 0.05], [-0.1, 0.05]],
            dtype=torch.float32,
        )
        self.optimizer.psdf_wrapper = AugmentedPSDFWrapper(
            verts,
            K_max=2,
            E_max=4,
        )
        self.optimizer.add_obstacle_avoidance_constraint(self.param, None)

    def test_sf_and_mf_models_share_discrete_covariance_dynamics_and_mpcc_cost(self):
        pv_param = CCSFOptimizerParam()
        pv_param.horizon = self.param.horizon
        pv_param.tf = self.param.tf
        pv_optimizer = CCSFOptimizer()

        mf_model = self.optimizer.create_model(self.param)
        pv_model = pv_optimizer.create_model(pv_param)
        state = np.array(
            [0.2, -0.1, 0.3, 0.4, np.log(0.01), np.log(0.02),
             np.log(0.03), -0.004],
            dtype=float,
        )
        control = np.array([0.35, -0.25, 0.3], dtype=float)
        path_param = np.array([0.1, -0.2, 0.8, 0.6, 0.35, -0.4])
        mf_param = np.zeros(int(mf_model.p.shape[0]), dtype=float)
        pv_param_value = np.zeros(int(pv_model.p.shape[0]), dtype=float)
        mf_param[:6] = path_param
        pv_param_value[:6] = path_param
        mf_param[self.optimizer.covariance_propagation_param_slice] = 1.0
        pv_param_value[pv_optimizer.covariance_propagation_param_slice] = 1.0

        mf_eval = ca.Function(
            "test_mf_common_model",
            [mf_model.x, mf_model.u, mf_model.p],
            [mf_model.disc_dyn_expr, mf_model.cost_expr_ext_cost],
        )
        pv_eval = ca.Function(
            "test_pv_common_model",
            [pv_model.x, pv_model.u, pv_model.p],
            [pv_model.disc_dyn_expr, pv_model.cost_expr_ext_cost],
        )
        mf_dynamics, mf_cost = mf_eval(state, control, mf_param)
        pv_dynamics, pv_cost = pv_eval(state, control, pv_param_value)

        np.testing.assert_allclose(
            np.asarray(mf_dynamics),
            np.asarray(pv_dynamics),
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            np.asarray(mf_cost),
            np.asarray(pv_cost),
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            self.optimizer._build_initial_covariance_state(self.param),
            pv_optimizer._build_initial_covariance_state(pv_param),
        )

    def test_discrete_frozen_tail_has_no_covariance_input_sensitivity(self):
        state = np.array(
            [0.2, -0.1, 0.3, 0.4, np.log(0.01), np.log(0.02),
             np.log(0.03), -0.004],
            dtype=float,
        )
        control = np.array([0.35, -0.25, 0.3], dtype=float)
        for optimizer_type, param_type in (
            (CCMFOptimizer, CCMFOptimizerParam),
            (CCSFOptimizer, CCSFOptimizerParam),
        ):
            with self.subTest(optimizer=optimizer_type.__name__):
                optimizer = optimizer_type()
                param = param_type()
                param.horizon = 3
                param.tf = 0.3
                optimizer.param = param
                model = optimizer.create_model(param)
                evaluate = ca.Function(
                    f"frozen_dynamics_{optimizer_type.__name__}",
                    [model.x, model.u, model.p],
                    [model.disc_dyn_expr, ca.jacobian(model.disc_dyn_expr, model.u)],
                )
                parameters = np.zeros(int(model.p.shape[0]), dtype=float)
                parameters[:6] = [0.0, 0.0, 1.0, 0.0, 0.0, -0.4]
                parameters[optimizer.covariance_propagation_param_slice] = 0.0
                frozen, frozen_jacobian = evaluate(state, control, parameters)
                frozen = np.asarray(frozen).reshape(-1)
                frozen_jacobian = np.asarray(frozen_jacobian)
                np.testing.assert_array_equal(frozen[4:], state[4:])
                np.testing.assert_array_equal(frozen_jacobian[4:], np.zeros((4, 3)))
                self.assertGreater(np.linalg.norm(frozen[:3] - state[:3]), 0.0)
                self.assertAlmostEqual(frozen[3], state[3] + 0.1 * control[2])

                parameters[optimizer.covariance_propagation_param_slice] = 1.0
                propagated, propagated_jacobian = evaluate(state, control, parameters)
                propagated = np.asarray(propagated).reshape(-1)
                propagated_jacobian = np.asarray(propagated_jacobian)
                np.testing.assert_allclose(propagated[:4], frozen[:4])
                self.assertGreater(np.linalg.norm(propagated[4:] - state[4:]), 0.0)
                self.assertGreater(np.linalg.norm(propagated_jacobian[4:, :2]), 0.0)
                np.testing.assert_array_equal(propagated_jacobian[4:, 2], np.zeros(4))

    def test_nominal_covariance_rollout_freezes_at_robust_horizon(self):
        inputs = np.array([[0.1, 0.2, 0.1], [0.2, -0.3, 0.2], [0.3, 0.1, 0.3]])
        stage_progress = np.linspace(0.0, 0.3, 4)
        for optimizer_type, param_type in (
            (CCMFOptimizer, CCMFOptimizerParam),
            (CCSFOptimizer, CCSFOptimizerParam),
        ):
            with self.subTest(optimizer=optimizer_type.__name__):
                optimizer = optimizer_type()
                param = param_type()
                param.horizon = 3
                param.tf = 0.3
                param.robust_horizon = 1
                optimizer.param = param
                optimizer.N = param.horizon
                optimizer.state = _FakeState([0.0, 0.0, 0.3])
                optimizer.reference_path_data = None
                frozen = optimizer._propagate_covariance_state_batch(inputs, stage_progress)
                np.testing.assert_allclose(
                    frozen[0], optimizer._build_initial_covariance_state(param)
                )
                self.assertGreater(np.linalg.norm(frozen[1] - frozen[0]), 0.0)
                np.testing.assert_array_equal(frozen[1:], np.tile(frozen[1], (3, 1)))

                changed_tail_inputs = inputs.copy()
                changed_tail_inputs[1:, :2] *= -2.0
                changed_tail_inputs[:, 2] *= 2.0
                np.testing.assert_array_equal(
                    optimizer._propagate_covariance_state_batch(
                        changed_tail_inputs, stage_progress
                    ),
                    frozen,
                )

                param.robust_horizon = param.horizon
                full = optimizer._propagate_covariance_state_batch(inputs, stage_progress)
                np.testing.assert_allclose(full[:2], frozen[:2])
                self.assertGreater(np.linalg.norm(full[-1] - frozen[-1]), 0.0)
                covariances = covariance_from_log_cholesky(full)
                self.assertTrue((np.linalg.eigvalsh(covariances) > 0.0).all())

    def test_log_cholesky_states_have_no_covariance_entry_bounds(self):
        for optimizer_type, param_type in (
            (CCMFOptimizer, CCMFOptimizerParam),
            (CCSFOptimizer, CCSFOptimizerParam),
        ):
            with self.subTest(optimizer=optimizer_type.__name__):
                indices, lower, upper = optimizer_type()._get_augmented_state_bounds(
                    param_type(), -0.2, 2.0
                )
                np.testing.assert_array_equal(indices, [3])
                np.testing.assert_allclose(lower, [-0.2])
                np.testing.assert_allclose(upper, [2.0])

    def test_parameter_layout_and_exact_two_raw_affine_rows(self):
        self._setup_constraint_ocp()

        self.assertEqual(self.optimizer.path_param_dim, 6)
        self.assertEqual(self.optimizer.guard_param_dim, 4)
        self.assertEqual(self.optimizer.mf_param_dim, 9)
        self.assertEqual(self.optimizer.path_param_slice, slice(0, 6))
        self.assertEqual(self.optimizer.guard_param_slice, slice(6, 10))
        self.assertEqual(self.optimizer.mf_param_slice, slice(10, 19))
        self.assertEqual(self.optimizer.covariance_propagation_param_slice, slice(19, 20))
        self.assertEqual(self.optimizer.ocp.dims.np, 20)
        self.assertEqual(int(self.optimizer.ocp.model.p.shape[0]), 20)
        self.assertEqual(self.optimizer.ocp.solver_options.integrator_type, "DISCRETE")
        self.assertEqual(self.optimizer.ocp.dims.nh, 2)
        self.assertEqual(self.optimizer.ocp.dims.nh_e, 2)

        path_constraint = ca.Function(
            "test_cc_mf_path_constraints",
            [self.optimizer.ocp.model.x, self.optimizer.ocp.model.p],
            [self.optimizer.ocp.model.con_h_expr],
        )
        terminal_constraint = ca.Function(
            "test_cc_mf_terminal_constraints",
            [self.optimizer.ocp.model.x, self.optimizer.ocp.model.p],
            [self.optimizer.ocp.model.con_h_expr_e],
        )
        x_value = np.arange(8, dtype=float) / 10.0
        path_value = np.array([1.0, -2.0, 0.6, 0.8, 0.3, -0.4])
        guard_value = np.array([0.4, -0.2, 0.1, -0.3])
        mf_value = np.concatenate(
            (np.arange(1, 9, dtype=float) / 20.0, np.array([-0.25]))
        )
        p_value = self.optimizer._build_stage_param(
            path_value,
            guard_value,
            mf_value,
        )

        np.testing.assert_allclose(p_value[:6], path_value)
        np.testing.assert_allclose(p_value[6:10], guard_value)
        np.testing.assert_allclose(p_value[10:19], mf_value)
        self.assertEqual(p_value[19], 1.0)
        self.param.robust_horizon = 1
        for stage in range(self.param.horizon + 1):
            scheduled = self.optimizer._build_stage_param(
                path_value, guard_value, mf_value, stage_idx=stage
            )
            self.assertEqual(scheduled[19], float(stage < self.param.robust_horizon))

        expected = np.array(
            [
                guard_value[:3] @ x_value[:3] + guard_value[3],
                mf_value[:8] @ x_value + mf_value[8],
            ]
        )
        actual_path = np.asarray(path_constraint(x_value, p_value)).reshape(-1)
        actual_terminal = np.asarray(terminal_constraint(x_value, p_value)).reshape(-1)
        np.testing.assert_allclose(actual_path, expected, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(actual_terminal, expected, atol=1e-12, rtol=0.0)

    def test_no_stage_zero_constraint(self):
        self._setup_constraint_ocp()

        self.assertEqual(self.optimizer.ocp.model.con_h_expr_0, [])
        np.testing.assert_array_equal(
            self.optimizer.ocp.constraints.lh_0,
            np.array([]),
        )
        np.testing.assert_array_equal(
            self.optimizer.ocp.constraints.uh_0,
            np.array([]),
        )
        np.testing.assert_array_equal(
            self.optimizer.ocp.constraints.idxsh_0,
            np.array([]),
        )

        self.optimizer.ocp.make_consistent()
        self.assertEqual(self.optimizer.ocp.dims.nh_0, 0)
        self.assertEqual(self.optimizer.ocp.dims.nsh_0, 0)
        self.assertEqual(self.optimizer.ocp.dims.ns_0, 0)
        self.assertEqual(self.optimizer.ocp.dims.nh, 2)
        self.assertEqual(self.optimizer.ocp.dims.nsh, 1)
        self.assertEqual(self.optimizer.ocp.dims.ns, 1)
        self.assertEqual(self.optimizer.ocp.dims.nh_e, 2)
        self.assertEqual(self.optimizer.ocp.dims.nsh_e, 1)
        self.assertEqual(self.optimizer.ocp.dims.ns_e, 1)

    def test_hard_guard_and_soft_mf_slack_configuration(self):
        self._setup_constraint_ocp()

        soft_row_indices = np.array([1], dtype=np.int64)
        np.testing.assert_array_equal(
            self.optimizer.ocp.constraints.idxsh,
            soft_row_indices,
        )
        np.testing.assert_array_equal(
            self.optimizer.ocp.constraints.idxsh_e,
            soft_row_indices,
        )
        self.assertNotIn(0, self.optimizer.ocp.constraints.idxsh)
        np.testing.assert_allclose(self.optimizer.ocp.constraints.lsh, np.zeros(1))
        np.testing.assert_allclose(self.optimizer.ocp.constraints.ush, np.zeros(1))
        np.testing.assert_allclose(self.optimizer.ocp.constraints.lsh_e, np.zeros(1))
        np.testing.assert_allclose(self.optimizer.ocp.constraints.ush_e, np.zeros(1))
        np.testing.assert_allclose(self.optimizer.ocp.constraints.lh, np.zeros(2))
        np.testing.assert_allclose(self.optimizer.ocp.constraints.lh_e, np.zeros(2))
        self.assertTrue((self.optimizer.ocp.constraints.uh > 0.0).all())
        self.assertTrue((self.optimizer.ocp.constraints.uh_e > 0.0).all())

        expected_linear = np.array([self.param.mf_slack_linear])
        expected_quadratic = np.array([self.param.mf_slack_quadratic])
        np.testing.assert_allclose(self.optimizer.ocp.cost.zl, expected_linear)
        np.testing.assert_allclose(self.optimizer.ocp.cost.Zl, expected_quadratic)
        np.testing.assert_allclose(self.optimizer.ocp.cost.zu, expected_linear)
        np.testing.assert_allclose(self.optimizer.ocp.cost.Zu, expected_quadratic)
        np.testing.assert_allclose(self.optimizer.ocp.cost.zl_e, expected_linear)
        np.testing.assert_allclose(self.optimizer.ocp.cost.Zl_e, expected_quadratic)
        np.testing.assert_allclose(self.optimizer.ocp.cost.zu_e, expected_linear)
        np.testing.assert_allclose(self.optimizer.ocp.cost.Zu_e, expected_quadratic)
        np.testing.assert_allclose(expected_linear, [1e2])
        np.testing.assert_allclose(expected_quadratic, [1e0])
        self.assertEqual(self.optimizer.ocp.cost.zl.shape, (1,))
        self.assertEqual(self.optimizer.ocp.cost.Zl.shape, (1,))
        self.assertFalse(hasattr(self.param, "guard_slack_linear"))
        self.assertFalse(hasattr(self.param, "guard_slack_quadratic"))
        self.assertFalse(hasattr(self.optimizer, "_set_stage_slack_penalties"))

        # Row M is disabled through stage parameters, not by changing generated
        # solver dimensions.  Row G therefore remains hard under either setting.
        self.param.use_row_mf = False
        self.assertEqual(self.optimizer.ocp.dims.nh, 2)
        self.assertEqual(self.optimizer.ocp.dims.nh_e, 2)
        np.testing.assert_array_equal(
            self.optimizer.ocp.constraints.idxsh,
            soft_row_indices,
        )
        np.testing.assert_array_equal(
            self.optimizer.ocp.constraints.idxsh_e,
            soft_row_indices,
        )

    def test_lower_slack_diagnostics_map_single_solver_slack_to_mf_row(self):
        class _SlackSolver:
            @staticmethod
            def get(stage, field):
                if field != "sl":
                    raise KeyError(field)
                return np.array([0.01 * stage], dtype=float)

        lower_slacks = self.optimizer._read_lower_slack_trajectory(_SlackSolver())
        self.assertTrue(np.isnan(lower_slacks[0]).all())
        np.testing.assert_allclose(lower_slacks[1:, 0], np.zeros(3))
        np.testing.assert_allclose(lower_slacks[1:, 1], [0.01, 0.02, 0.03])

    def test_guard_coefficients_and_stagewise_mf_mask(self):
        self.param.use_row_mf = True
        self.param.d_col = 0.03
        self.param.d_mf_mask = 0.05
        epsilon = float(self.param.chance_epsilon)
        z_bar = np.zeros((4, 8), dtype=float)
        z_bar[:, :3] = np.array(
            [
                [0.25, -0.10, 0.20],
                [0.40, -0.05, 0.15],
                [0.55, 0.00, 0.10],
                [0.70, 0.05, 0.05],
            ]
        )
        z_bar[:, 4:7] = np.log(0.01)

        # Stage 0 is penetrating. Stages 1 and 3 remain valid separated-domain
        # MF stages; stage 2 has invalid MF coefficients only.
        phi = np.array([-0.02, 0.20, 0.25, 0.30])
        gradient = np.array(
            [
                [1.0, 0.0, 0.1],
                [0.8, 0.2, 0.0],
                [0.6, 0.3, -0.1],
                [0.4, 0.4, -0.2],
            ]
        )
        mf_A_raw = np.arange(32, dtype=float).reshape(4, 8) / 50.0
        mf_A_raw[2, 4] = np.nan
        mf_c_raw = np.array([-0.1, -0.2, -0.3, -0.4])

        self.optimizer._compute_mf_affine_data = lambda _: (
            phi.copy(),
            gradient.copy(),
            mf_A_raw.copy(),
            mf_c_raw.copy(),
        )
        data = self.optimizer._compute_constraint_affine_data(z_bar)

        expected_guard_A = np.zeros((4, 8), dtype=float)
        expected_guard_A[:, :3] = gradient
        expected_guard_c = (
            phi
            - np.einsum("ij,ij->i", gradient, z_bar[:, :3])
            - self.param.d_col
        )
        np.testing.assert_allclose(data["guard_A"], expected_guard_A)
        np.testing.assert_allclose(data["guard_c"], expected_guard_c)

        np.testing.assert_allclose(data["phi"], phi)
        np.testing.assert_allclose(data["gradient"], gradient)
        np.testing.assert_allclose(data["mf_A_raw"], mf_A_raw, equal_nan=True)
        np.testing.assert_allclose(data["mf_c_raw"], mf_c_raw)
        np.testing.assert_array_equal(data["mf_valid"], [True, True, False, True])
        np.testing.assert_array_equal(data["mf_mask"], [False, True, False, True])
        np.testing.assert_allclose(data["epsilon"], np.full(4, epsilon))

        np.testing.assert_allclose(data["mf_A"][0], np.zeros(8))
        np.testing.assert_allclose(data["mf_A"][1], mf_A_raw[1])
        np.testing.assert_allclose(data["mf_A"][2], np.zeros(8))
        np.testing.assert_allclose(data["mf_A"][3], mf_A_raw[3])
        np.testing.assert_allclose(data["mf_c"], [epsilon, -0.2, epsilon, -0.4])

        # A penetrating measured stage must not globally disable later valid MF
        # rows. Row normalization and its stage-dependent penalty data are absent.
        self.assertTrue(data["mf_mask"][1])
        self.assertTrue(data["mf_mask"][3])
        self.assertNotIn("guard_scale", data)
        self.assertNotIn("mf_scale", data)

    def test_mf_mask_uses_strict_phi_threshold(self):
        self.param.use_row_mf = True
        self.param.d_mf_mask = 0.05
        z_bar = np.zeros((4, 8), dtype=float)
        phi = np.array(
            [
                0.10,
                self.param.d_mf_mask,
                np.nextafter(self.param.d_mf_mask, np.inf),
                np.nextafter(self.param.d_mf_mask, -np.inf),
            ]
        )
        gradient = np.tile(np.array([1.0, 0.0, 0.0]), (4, 1))
        mf_A_raw = np.ones((4, 8), dtype=float)
        mf_c_raw = -np.ones(4, dtype=float)
        self.optimizer._compute_mf_affine_data = lambda _: (
            phi.copy(),
            gradient.copy(),
            mf_A_raw.copy(),
            mf_c_raw.copy(),
        )

        data = self.optimizer._compute_constraint_affine_data(z_bar)

        np.testing.assert_array_equal(data["mf_valid"], np.ones(4, dtype=bool))
        np.testing.assert_array_equal(data["mf_mask"], [False, False, True, False])

    def test_mf_active_window_separates_intermediate_and_terminal_stages(self):
        self.param.use_row_mf = True
        self.param.d_mf_mask = 0.0
        self.param.mf_active_start_step = 1
        self.param.mf_active_end_step = 1
        self.param.mf_active_terminal = False
        z_bar = np.zeros((4, 8), dtype=float)
        phi = np.full(4, 0.2, dtype=float)
        gradient = np.tile(np.array([1.0, 0.0, 0.0]), (4, 1))
        mf_A_raw = np.ones((4, 8), dtype=float)
        mf_c_raw = -np.ones(4, dtype=float)
        self.optimizer._compute_mf_affine_data = lambda _: (
            phi.copy(),
            gradient.copy(),
            mf_A_raw.copy(),
            mf_c_raw.copy(),
        )

        data = self.optimizer._compute_constraint_affine_data(z_bar)
        np.testing.assert_array_equal(
            data["mf_stage_mask"],
            [False, True, False, False],
        )
        np.testing.assert_array_equal(
            data["mf_mask"],
            [False, True, False, False],
        )

        self.param.mf_active_end_step = 20
        data = self.optimizer._compute_constraint_affine_data(z_bar)
        np.testing.assert_array_equal(
            data["mf_stage_mask"],
            [False, True, True, False],
        )

        self.param.mf_active_terminal = True
        data = self.optimizer._compute_constraint_affine_data(z_bar)
        np.testing.assert_array_equal(
            data["mf_stage_mask"],
            [False, True, True, True],
        )

    def test_row_mf_option_masks_only_mf_row_and_preserves_hard_guard(self):
        self.param.use_row_mf = True
        self.assertTrue(self.param.use_row_mf)
        self.assertFalse(hasattr(self.param, "use_row_g"))
        self.param.d_mf_mask = 0.0
        z_bar = np.zeros((4, 8), dtype=float)
        z_bar[:, :3] = np.array(
            [
                [0.25, -0.10, 0.20],
                [0.40, -0.05, 0.15],
                [0.55, 0.00, 0.10],
                [0.70, 0.05, 0.05],
            ]
        )
        # Include a penetrating future stage to ensure disabling Row M cannot
        # silently replace the geometric guard with a feasible constant.
        phi = np.array([0.10, -0.02, 0.25, 0.30])
        gradient = np.tile(np.array([0.8, 0.2, -0.1]), (4, 1))
        mf_A_raw = np.arange(32, dtype=float).reshape(4, 8) / 50.0
        mf_c_raw = np.array([-0.1, -0.2, -0.3, -0.4])
        calls = {"mf": 0, "psdf": 0}

        def compute_mf(_):
            calls["mf"] += 1
            return (
                phi.copy(),
                gradient.copy(),
                mf_A_raw.copy(),
                mf_c_raw.copy(),
            )

        def compute_exact(poses):
            calls["psdf"] += 1
            self.assertEqual(np.asarray(poses).shape, (4, 3))
            return phi.copy(), gradient.copy()

        self.optimizer._compute_mf_affine_data = compute_mf
        self.optimizer._compute_exact_psdf = compute_exact

        enabled = self.optimizer._compute_constraint_affine_data(z_bar)
        self.param.use_row_mf = False
        disabled = self.optimizer._compute_constraint_affine_data(z_bar)

        self.assertFalse(self.param.use_row_mf)
        self.assertTrue(enabled["row_mf_enabled"])
        self.assertFalse(disabled["row_mf_enabled"])
        self.assertEqual(calls, {"mf": 1, "psdf": 1})
        self.assertGreater(np.linalg.norm(enabled["guard_A"]), 0.0)
        np.testing.assert_allclose(disabled["guard_A"], enabled["guard_A"])
        np.testing.assert_allclose(disabled["guard_c"], enabled["guard_c"])

        expected_guard_residual = phi - self.param.d_col
        enabled_guard_residual = (
            np.einsum("ij,ij->i", enabled["guard_A"], z_bar)
            + enabled["guard_c"]
        )
        disabled_guard_residual = (
            np.einsum("ij,ij->i", disabled["guard_A"], z_bar)
            + disabled["guard_c"]
        )
        np.testing.assert_allclose(enabled_guard_residual, expected_guard_residual)
        np.testing.assert_allclose(disabled_guard_residual, expected_guard_residual)
        self.assertLess(disabled_guard_residual[1], 0.0)

        self.assertTrue(np.isnan(disabled["mf_A_raw"]).all())
        self.assertTrue(np.isnan(disabled["mf_c_raw"]).all())
        np.testing.assert_array_equal(disabled["mf_valid"], np.zeros(4, dtype=bool))
        np.testing.assert_array_equal(disabled["mf_mask"], np.zeros(4, dtype=bool))
        np.testing.assert_allclose(disabled["mf_A"], np.zeros((4, 8)))
        np.testing.assert_allclose(
            disabled["mf_c"],
            np.full(4, self.param.chance_epsilon),
        )

        # The independent master switch still disables all obstacle processing,
        # even if a previously initialized PSDF wrapper remains attached.
        self.param.use_obstacle_constraint = False
        globally_disabled = self.optimizer._compute_constraint_affine_data(z_bar)
        self.assertEqual(calls, {"mf": 1, "psdf": 1})
        np.testing.assert_allclose(globally_disabled["guard_A"], np.zeros((4, 8)))
        np.testing.assert_allclose(
            globally_disabled["guard_c"],
            np.full(4, 1000.0 - self.param.d_col),
        )

    def test_batch_mf_failure_falls_back_and_masks_only_failed_stage(self):
        self.param.use_row_mf = True
        self.param.d_col = 0.03
        self.param.d_mf_mask = 0.0
        epsilon = float(self.param.chance_epsilon)
        failed_stage = 2
        wrapper = _StagewiseFallbackPSDFWrapper(failed_stage)
        self.optimizer.psdf_wrapper = wrapper

        z_bar = np.zeros((4, 8), dtype=float)
        z_bar[:, 0] = np.arange(4, dtype=float)
        z_bar[:, 1] = np.array([-0.2, -0.1, 0.0, 0.1])
        z_bar[:, 2] = np.array([0.3, 0.2, 0.1, 0.0])
        z_bar[:, 4:7] = np.log(0.01)

        data = self.optimizer._compute_constraint_affine_data(z_bar)

        self.assertEqual(wrapper.mf_call_sizes, [4, 1, 1, 1, 1])
        self.assertEqual(wrapper.mf_single_stages, [0, 1, 2, 3])
        self.assertEqual(wrapper.pose_call_sizes, [4])

        expected_phi = 0.20 + 0.05 * z_bar[:, 0]
        expected_gradient = np.tile(np.array([1.0, 0.25, -0.10]), (4, 1))
        np.testing.assert_allclose(data["phi"], expected_phi)
        np.testing.assert_allclose(data["gradient"], expected_gradient)
        self.assertTrue(np.isfinite(data["guard_A"]).all())
        self.assertTrue(np.isfinite(data["guard_c"]).all())

        expected_guard_c = (
            expected_phi
            - np.einsum("ij,ij->i", expected_gradient, z_bar[:, :3])
            - self.param.d_col
        )
        np.testing.assert_allclose(data["guard_c"], expected_guard_c)
        np.testing.assert_array_equal(data["mf_valid"], [True, True, False, True])
        np.testing.assert_array_equal(data["mf_mask"], [False, True, False, True])

        for stage in (1, 3):
            expected_A = wrapper.expected_mf_A(stage).numpy()
            np.testing.assert_allclose(data["mf_A_raw"][stage], expected_A)
            np.testing.assert_allclose(data["mf_A"][stage], expected_A)
            self.assertAlmostEqual(
                data["mf_c"][stage],
                wrapper.expected_mf_c(stage),
            )

        np.testing.assert_allclose(
            data["mf_A_raw"][0], wrapper.expected_mf_A(0).numpy()
        )
        np.testing.assert_allclose(data["mf_A"][0], np.zeros(8))
        self.assertAlmostEqual(data["mf_c"][0], epsilon)

        self.assertTrue(np.isnan(data["mf_A_raw"][failed_stage]).all())
        self.assertTrue(np.isnan(data["mf_c_raw"][failed_stage]))
        np.testing.assert_allclose(data["mf_A"][failed_stage], np.zeros(8))
        self.assertAlmostEqual(data["mf_c"][failed_stage], epsilon)

    def test_infeasibility_classification_excludes_inactive_mf_row(self):
        diagnostics = {
            "guard_affine_residual": np.array([np.nan, -0.01, 0.02, 0.03]),
            "mf_affine_residual": np.array([np.nan, 0.20, 0.20, 0.20]),
            "mf_mask": np.zeros(4, dtype=bool),
            "solver": {
                "qp_fixed_row_lower_residual": np.array(
                    [
                        [np.nan, np.nan],
                        [-0.01, 0.20],
                        [0.02, 0.20],
                        [0.03, 0.20],
                    ]
                ),
                "nlp_residuals": np.zeros(4),
            },
            "bounds": {
                "state_stages": np.arange(1, 4),
                "state_names": np.array(["s"]),
                "state_lower_residual": np.ones((3, 1)),
                "state_upper_residual": np.ones((3, 1)),
                "input_stages": np.arange(3),
                "input_names": np.array(["v", "omega", "v_s"]),
                "input_lower_residual": np.ones((3, 3)),
                "input_upper_residual": np.ones((3, 3)),
            },
        }

        summary = self.optimizer._classify_infeasibility_candidates(diagnostics)

        self.assertEqual(len(summary["candidates"]), 1)
        self.assertEqual(
            summary["candidates"][0]["constraint"],
            "row_g_hard_guard",
        )
        np.testing.assert_array_equal(
            summary["inactive_mf_violation_stages"],
            np.array([], dtype=int),
        )

    def test_solve_skips_diagnostics_when_debug_is_disabled(self):
        states = np.zeros((4, 8), dtype=float)
        inputs_ = np.zeros((3, 3), dtype=float)
        self.optimizer.solver = _FakeSolver(states, inputs_)
        self.optimizer._prepare_and_solve = lambda _: 0
        self.param.debug_mf = False
        self.param.debug_infeasibility = False
        calls = []
        self.optimizer.compute_diagnostics = lambda *args, **kwargs: calls.append(
            (args, kwargs)
        )

        solution = self.optimizer.solve_nlp()

        self.assertEqual(calls, [])
        self.assertIsNone(self.optimizer.get_last_constraint_diagnostics())
        self.assertEqual(solution.get_state_trajectory().shape, (8, 4))
        self.assertEqual(solution.get_input_trajectory().shape, (3, 3))

    def _setup_solver_timing(self, outcomes):
        """Keep the real solve lifecycle while charging fake work to a clock."""
        clock = {"now": 100.0}
        states = np.zeros((4, 8), dtype=float)
        states[:, 4:7] = np.log([0.01, 0.02, 0.03])
        inputs_ = np.zeros((3, 3), dtype=float)

        def make_solver(solve_outcomes):
            solver = _FakeSolver(states, inputs_)
            solver.status = -1
            attempts = iter(solve_outcomes)

            def solve():
                status, duration = next(attempts)
                clock["now"] += duration
                if isinstance(status, Exception):
                    raise status
                solver.status = status
                return status

            def reset(**_):
                clock["now"] += 17.0

            solver.solve = solve
            solver.reset = reset
            return solver

        def constraint_data(_):
            clock["now"] += 11.0
            return {
                "phi": np.ones(4),
                "mf_A_raw": np.zeros((4, 8)),
                "mf_c_raw": np.zeros(4),
                "epsilon": np.full(4, self.param.chance_epsilon),
                "mf_valid": np.zeros(4, dtype=bool),
                "mf_mask": np.zeros(4, dtype=bool),
            }

        def set_parameters(*_):
            clock["now"] += 13.0

        def diagnostics(*_, **__):
            clock["now"] += 19.0
            return None

        record_logs = self.optimizer._record_experiment_logs

        def record(*args):
            record_logs(*args)
            clock["now"] += 23.0

        self.optimizer.solver = make_solver(outcomes)
        self.optimizer._compute_constraint_affine_data = constraint_data
        self.optimizer._compute_exact_psdf = lambda _: (1.0, np.zeros(3))
        self.optimizer._set_constraint_parameters = set_parameters
        self.optimizer.compute_diagnostics = diagnostics
        self.optimizer._record_experiment_logs = record
        self.optimizer._report_solver_failure = lambda _: None
        self.optimizer._system_dynamics.nominal_safe_controller = (
            lambda pose, *_: (pose.copy(), np.zeros(2))
        )
        self.param.debug_mf = True
        self.param.debug_infeasibility = True
        return clock, make_solver

    def test_solver_timing_excludes_preparation_diagnostics_and_logging(self):
        clock, _ = self._setup_solver_timing([(0, 0.125), (0, 0.375)])

        with patch(
            "control.cc_mf_optimizer.time.perf_counter",
            side_effect=lambda: clock["now"],
        ) as timer, redirect_stdout(io.StringIO()):
            self.optimizer.solve_nlp()
            self.optimizer.solve_nlp()

        np.testing.assert_allclose(self.optimizer.solver_times, [0.125, 0.375])
        self.assertEqual(timer.call_count, 4)
        self.assertGreater(clock["now"] - 100.0, 100.0)
        self.assertEqual(
            [row["solver_status"] for row in self.optimizer.get_cycle_log_rows()],
            [0, 0],
        )

    def test_recovered_cycle_sums_main_and_backup_solver_time_once(self):
        clock, make_solver = self._setup_solver_timing([(1, 0.125)])
        backup_solver = make_solver([(0, 0.375)])
        self.optimizer.backup_solver = backup_solver

        with patch(
            "control.cc_mf_optimizer.time.perf_counter",
            side_effect=lambda: clock["now"],
        ) as timer, redirect_stdout(io.StringIO()):
            solution = self.optimizer.solve_nlp()

        self.assertEqual(self.optimizer.solver_times, [0.5])
        self.assertEqual(timer.call_count, 4)
        self.assertIs(solution.solver, backup_solver)
        self.assertEqual(solution.stats()["return_status"], "success")
        self.assertEqual(self.optimizer._last_main_status, 1)
        self.assertEqual(self.optimizer._last_backup_status, 0)
        self.assertEqual(self.optimizer._backup_feasible_qp_success_count, 1)
        self.assertEqual(self.optimizer._safe_stop_count, 0)
        cycle = self.optimizer.get_cycle_log_rows()[0]
        self.assertEqual(cycle["solve_mode"], "backup_feasible_qp")
        self.assertEqual(cycle["solver_status"], 1)

    def test_solver_exception_does_not_append_or_leak_into_next_cycle(self):
        clock, _ = self._setup_solver_timing(
            [(RuntimeError("solver failed"), 0.125), (0, 0.375)]
        )

        with patch(
            "control.cc_mf_optimizer.time.perf_counter",
            side_effect=lambda: clock["now"],
        ), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "solver failed"):
                self.optimizer.solve_nlp()
            self.assertEqual(self.optimizer.solver_times, [])
            self.assertEqual(self.optimizer.get_cycle_log_rows(), [])
            self.optimizer.solve_nlp()

        self.assertEqual(self.optimizer.solver_times, [0.375])
        self.assertEqual(len(self.optimizer.get_cycle_log_rows()), 1)

    def test_failed_cycle_keeps_status_and_time_paired_after_safe_stop(self):
        clock, make_solver = self._setup_solver_timing([(1, 0.125)])
        self.optimizer.backup_solver = make_solver([(2, 0.375)])

        with patch(
            "control.cc_mf_optimizer.time.perf_counter",
            side_effect=lambda: clock["now"],
        ) as timer, redirect_stdout(io.StringIO()):
            solution = self.optimizer.solve_nlp()

        self.assertEqual(self.optimizer.solver_times, [0.5])
        self.assertEqual(timer.call_count, 4)
        self.assertIs(solution.solver, self.optimizer.solver)
        self.assertEqual(solution.stats()["return_status"], "failure")
        self.assertEqual(self.optimizer._last_main_status, 1)
        self.assertEqual(self.optimizer._last_backup_status, 2)
        self.assertEqual(self.optimizer._main_solver_failure_count, 1)
        self.assertEqual(self.optimizer._backup_solver_failure_count, 1)
        self.assertEqual(self.optimizer._safe_stop_count, 1)
        cycle = self.optimizer.get_cycle_log_rows()[0]
        self.assertEqual(cycle["solve_mode"], "safe_stop")
        self.assertEqual(cycle["solver_status"], 1)

    def test_diagnostic_reporting_failure_does_not_interrupt_control(self):
        states = np.zeros((4, 8), dtype=float)
        solver = _FakeSolver(states, np.zeros((3, 3), dtype=float))
        expected = {"snapshot": "complete"}
        self.optimizer._build_constraint_diagnostics = (
            lambda *args, **kwargs: expected
        )

        def fail_to_report(_):
            raise RuntimeError("formatting failed")

        self.optimizer._print_constraint_diagnostics = fail_to_report

        result = self.optimizer.compute_diagnostics(
            solver,
            0,
            "sqp_rti",
            report=True,
        )

        self.assertIs(result, expected)
        self.assertEqual(self.optimizer.get_last_constraint_diagnostics(), expected)

    def test_failed_solve_diagnostics_run_before_recovery(self):
        states = np.zeros((4, 8), dtype=float)
        inputs_ = np.zeros((3, 3), dtype=float)
        solver = _FakeSolver(states, inputs_)
        self.optimizer.solver = solver
        self.optimizer._prepare_and_solve = lambda _: 1
        self.optimizer._last_boole_risk_visualization_data = {
            "nominal_poses": np.zeros((4, 3)),
            "boole_risk_sum": np.full(4, 0.02),
            "epsilon": np.full(4, 0.05),
            "mf_mask": np.array([False, True, True, True]),
        }
        self.optimizer._report_solver_failure = lambda _: None
        self.param.debug_mf = False
        self.param.debug_infeasibility = True
        events = []

        def compute(*args, **kwargs):
            events.append(("diagnostics", kwargs["failed"]))
            return None

        def recover(status):
            events.append(("recovery", status))
            return solver, status, "safe_stop"

        self.optimizer.compute_diagnostics = compute
        self.optimizer.recovery_infeasible = recover

        self.optimizer.solve_nlp()

        self.assertEqual(events, [("diagnostics", True), ("recovery", 1)])
        self.assertIsNone(
            self.optimizer.get_last_boole_risk_visualization_data()
        )

    def test_failed_diagnostic_report_error_does_not_block_recovery(self):
        states = np.zeros((4, 8), dtype=float)
        inputs_ = np.zeros((3, 3), dtype=float)
        solver = _FakeSolver(states, inputs_)
        self.optimizer.solver = solver
        self.optimizer._prepare_and_solve = lambda _: 1
        self.optimizer._report_solver_failure = lambda _: None
        self.param.debug_mf = False
        self.param.debug_infeasibility = True
        self.optimizer.compute_diagnostics = lambda *args, **kwargs: {
            "snapshot": "failed"
        }

        def fail_to_report(_):
            raise RuntimeError("formatting failed")

        self.optimizer._print_infeasibility_diagnostics = fail_to_report
        recovery_statuses = []

        def recover(status):
            recovery_statuses.append(status)
            return solver, status, "safe_stop"

        self.optimizer.recovery_infeasible = recover
        output = io.StringIO()
        with redirect_stdout(output):
            self.optimizer.solve_nlp()

        self.assertEqual(recovery_statuses, [1])
        self.assertIn(
            "could not report failed diagnostics",
            output.getvalue(),
        )

    def test_constraint_log_keeps_only_feasibility_fields(self):
        diagnostics = {
            "constraint_stages": np.arange(1, 4),
            "d_mf_mask": 0.1,
            "mf_activation_phi": np.array([0.0, 0.2, 0.05, 0.3]),
            "mf_domain_eligible": np.array([False, True, False, True]),
            "mf_valid": np.array([False, True, True, True]),
            "mf_mask": np.array([False, True, False, True]),
            "guard_affine_residual": np.array([np.nan, 0.01, -0.02, 0.03]),
            "mf_affine_residual": np.array([np.nan, -0.01, 0.2, 0.04]),
            "mf_lower_slack": np.array([np.nan, 0.02, 0.0, 0.0]),
            "solver": {
                "qp_fixed_row_lower_residual": np.array(
                    [
                        [np.nan, np.nan],
                        [0.01, 0.01],
                        [-0.02, 0.2],
                        [0.03, 0.04],
                    ]
                )
            },
        }

        self.optimizer._record_constraint_log(diagnostics)

        rows = self.optimizer.get_constraint_log_rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual(set(rows[0]), set(self.optimizer.CONSTRAINT_LOG_FIELDS))
        self.assertTrue(rows[0]["mf_active"])
        self.assertFalse(rows[0]["row_m_feasible_before_slack"])
        self.assertTrue(rows[0]["row_m_feasible_after_slack"])
        self.assertFalse(rows[1]["mf_active"])
        self.assertIsNone(rows[1]["row_m_residual_before_slack"])
        self.assertIsNone(rows[1]["row_m_feasible_after_slack"])
        self.assertFalse(rows[1]["row_g_feasible"])

    def test_compact_cycle_and_covariance_logs_use_eight_fields(self):
        states = np.zeros((4, 8), dtype=float)
        states[:, 4:7] = np.log([0.01, 0.02, 0.03])
        states[:, 7] = -0.004
        inputs_ = np.array(
            [
                [0.1, 0.01, 0.1],
                [0.2, 0.02, 0.2],
                [0.3, 0.03, 0.3],
            ],
            dtype=float,
        )
        solver = _FakeSolver(states, inputs_)
        self.optimizer._last_nominal_probability_sum = np.array(
            [np.nan, 0.1, 0.2, 0.3],
            dtype=float,
        )
        self.optimizer._last_main_status = 0
        diagnostics = {
            "exact_current_psdf": 0.012,
            "mf_lower_slack": np.array([np.nan, 0.01, 0.02, 0.03]),
        }

        self.optimizer._record_experiment_logs(
            "sqp_rti",
            0,
            solver,
            diagnostics,
        )

        cycle_rows = self.optimizer.get_cycle_log_rows()
        self.assertEqual(len(cycle_rows), 1)
        self.assertEqual(
            set(cycle_rows[0]),
            set(self.optimizer.CYCLE_LOG_FIELDS),
        )
        self.assertEqual(len(self.optimizer.CYCLE_LOG_FIELDS), 8)
        self.assertEqual(
            json.loads(cycle_rows[0]["mf_probability_sum"]),
            [None, 0.1, 0.2, 0.3],
        )

        covariance_rows = self.optimizer.get_covariance_log_rows()
        self.assertEqual(len(covariance_rows), 4)
        self.assertEqual(
            set(covariance_rows[0]),
            set(self.optimizer.COVARIANCE_LOG_FIELDS),
        )
        self.assertEqual(len(self.optimizer.COVARIANCE_LOG_FIELDS), 8)
        np.testing.assert_allclose(
            [
                covariance_rows[0]["sigma_f"],
                covariance_rows[0]["sigma_l"],
                covariance_rows[0]["sigma_psi"],
            ],
            [0.01, 0.02, np.sqrt(0.03**2 + 0.004**2)],
        )
        self.assertIsNone(covariance_rows[-1]["v"])
        self.assertIsNone(covariance_rows[-1]["omega"])

    def test_compact_runtime_summary_omits_solver_mode(self):
        self.param.debug_mf = True
        states = np.zeros((4, 8), dtype=float)
        solver = _FakeSolver(states, np.zeros((3, 3), dtype=float))
        diagnostics = {
            "guard_affine_residual": np.array([np.nan, 0.01, 0.02, 0.03]),
            "minimum_exact_psdf_predicted_nodes": 0.012,
            "mf_mask": np.array([False, True, False, True]),
            "mf_lower_slack": np.array([np.nan, 0.0, 0.0, 0.002]),
            "solver": {
                "qp_fixed_row_lower_residual": np.array(
                    [
                        [np.nan, np.nan],
                        [0.01, 0.004],
                        [0.02, 0.2],
                        [0.03, 0.006],
                    ]
                )
            },
        }
        output = io.StringIO()

        with redirect_stdout(output):
            self.optimizer._finish_solve(
                "sqp_rti",
                0,
                solver,
                diagnostics=diagnostics,
            )

        summary = output.getvalue().strip()
        self.assertIn("[CC-MF 0000] OK", summary)
        self.assertIn("PSDFmin=+0.012", summary)
        self.assertIn("G=OK(+0.01)", summary)
        self.assertIn("MF=OK(active=2/3,min=+0.004,slack=+0.002)", summary)
        self.assertNotIn("mode=", summary)
        self.assertEqual(len(summary.splitlines()), 1)

    def test_prepare_regenerates_covariance_and_freezes_tail(self):
        self.param.robust_horizon = 1
        states = np.zeros((4, 8), dtype=float)
        states[:, 0] = [0.0, 1.0, 2.0, 3.0]
        states[:, 1] = [0.0, 0.1, 0.2, 0.3]
        states[:, 2] = 0.1
        states[:, 3] = [0.0, 0.1, 0.2, 0.3]
        states[:, 4:8] = 99.0
        # Large or negative stale factor guesses must be replaced by a new
        # rollout from the measured covariance before MF rows are assembled.
        states[1, 4:7] = [-1.0, -2.0, -3.0]
        inputs_ = np.array(
            [
                [0.1, 0.01, 0.1],
                [0.2, 0.02, 0.2],
                [0.3, 0.03, 0.3],
            ],
            dtype=float,
        )
        solver = _FakeSolver(states, inputs_)
        solver.solve = lambda: 0
        self.optimizer.solver = solver

        num_stages = self.optimizer.N + 1
        self.optimizer._compute_constraint_affine_data = lambda _: {
            "phi": np.ones(num_stages),
            "mf_A_raw": np.zeros((num_stages, self.optimizer.nx)),
            "mf_c_raw": np.zeros(num_stages),
            "epsilon": np.full(num_stages, self.param.chance_epsilon),
            "mf_valid": np.zeros(num_stages, dtype=bool),
            "mf_mask": np.zeros(num_stages, dtype=bool),
        }
        self.optimizer._compute_exact_psdf = lambda _: (1.0, np.zeros(3))
        captured = {}
        self.optimizer._set_constraint_parameters = (
            lambda _, stage_s_values, __: captured.update(
                stage_s_values=np.asarray(stage_s_values).copy()
            )
        )

        status = self.optimizer._prepare_and_solve(solver)

        self.assertEqual(status, 0)
        np.testing.assert_allclose(solver.states[0, :3], self.optimizer.state._x)
        np.testing.assert_allclose(
            solver.states[0, 4:8],
            self.optimizer._build_initial_covariance_state(self.param),
        )
        expected_covariance = self.optimizer._propagate_covariance_state_batch(
            inputs_, captured["stage_s_values"]
        )
        np.testing.assert_allclose(
            solver.states[:, 4:8], expected_covariance,
        )
        np.testing.assert_array_equal(
            solver.states[1:, 4:8], np.tile(solver.states[1, 4:8], (3, 1))
        )
        np.testing.assert_allclose(solver.inputs, inputs_)
        self.assertEqual(captured["stage_s_values"].shape, (num_stages,))

    def test_optimizer_routing_and_independent_classes(self):
        self.assertFalse(self.param.enable_backup_solver)
        self.assertEqual(simulation_mpc._resolve_optimizer_variant("cc_mf"), "cc_mf")
        self.assertEqual(
            simulation_mpc._resolve_optimizer_variant("cc_sf"),
            "cc_sf",
        )
        self.assertIsNot(CCMFOptimizer, CCSFOptimizer)
        self.assertIsNot(CCMFOptimizerParam, CCSFOptimizerParam)
        self.assertTrue(CCSFOptimizerParam().use_projected_variance_margin)


if __name__ == "__main__":
    unittest.main()
