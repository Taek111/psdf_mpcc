"""SF/MF covariance proximal parity without solver generation or navigation."""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import casadi as ca
import numpy as np

from control.cc_ecs_optimizer import CCECSOptimizer, CCECSOptimizerParam, PROCESS_NOISE_FIELDS
from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
from control.cc_sf_optimizer import CCSFOptimizer, CCSFOptimizerParam


class _Solver:
    def __init__(self, states, inputs):
        self.states = states.copy()
        self.inputs = inputs.copy()
        self.parameters = {}
        self.status = 0

    def get(self, stage, field):
        return (self.states if field == "x" else self.inputs)[stage].copy()

    def set(self, stage, field, value):
        value = np.asarray(value, dtype=float).copy()
        if field == "x":
            self.states[stage] = value
        elif field == "u":
            self.inputs[stage] = value
        elif field == "p":
            self.parameters[stage] = value

    def solve(self):
        return 0


class CCSFCovarianceProximalTest(unittest.TestCase):
    def setUp(self):
        self.anchor = np.array([math.log(.02), math.log(.025), math.log(.03), -.004])
        self.path = np.array([0., 0., 1., 0., 0., 0.])
        self.psdf = np.array([.1, .2, .3, .4, .5, .6, .7])

    def make_model(self, kind="sf", weight=0.01, memory=False):
        param_type = CCSFOptimizerParam if kind == "sf" else CCMFOptimizerParam
        optimizer_type = CCSFOptimizer if kind == "sf" else CCMFOptimizer
        param = param_type()
        param.horizon, param.tf, param.robust_horizon = 4, .4, 3
        param.covariance_proximal_weight = weight
        if not memory:
            param.a_max = param.alpha_max = None
            param.r_delta_v = param.r_delta_omega = 0.0
        optimizer = optimizer_type()
        self.addCleanup(optimizer.cleanup)
        optimizer.param = param
        optimizer.N = param.horizon
        if kind == "sf":
            model = optimizer.create_model(param, ca.MX.sym("test_psdf", 7))
        else:
            model = optimizer.create_model(param)
        optimizer.nx, optimizer.nu = int(model.x.numel()), 3
        return optimizer, param, model

    def parameters(self, optimizer, stage=1, anchor=None):
        anchor = self.anchor if anchor is None else anchor
        if isinstance(optimizer, CCMFOptimizer):
            return optimizer._build_stage_param(
                self.path, np.array([0., 0., 0., 1.]), np.r_[np.zeros(8), 1.],
                stage, nominal_covariance=anchor,
            )
        return optimizer._build_stage_param(
            self.path, np.zeros(4), 1., psdf_stage=self.psdf,
            stage_idx=stage, nominal_covariance=anchor,
        )

    @staticmethod
    def evaluate(model):
        return ca.Function("proximal_parity", [model.x, model.u, model.p], [
            model.cost_expr_ext_cost, model.cost_expr_ext_cost_e,
            ca.gradient(model.cost_expr_ext_cost, model.x),
            ca.hessian(model.cost_expr_ext_cost, model.x)[0],
            model.disc_dyn_expr,
        ])

    def test_default_and_zero_weight_preserve_psdf_parameter_layout(self):
        self.assertEqual(CCSFOptimizerParam().covariance_proximal_weight, .01)
        active, _, enabled_model = self.make_model(weight=.01)
        inactive, _, disabled_model = self.make_model(weight=0.)
        self.assertEqual(enabled_model.p.numel(), 25)
        self.assertEqual(disabled_model.p.numel(), 19)
        self.assertEqual(active.psdf_param_slice, inactive.psdf_param_slice)
        self.assertEqual(active.covariance_propagation_param_slice,
                         inactive.covariance_propagation_param_slice)
        for optimizer in (active, inactive):
            p = self.parameters(optimizer)
            np.testing.assert_array_equal(p[optimizer.psdf_param_slice], self.psdf)
            self.assertEqual(p[optimizer.covariance_propagation_param_slice][0], 1.)
        values = inactive._build_stage_param(self.path, np.zeros(4), 1., stage_idx=1)
        self.assertEqual(values.size, 19)  # Zero weight needs no covariance anchor.
        self.assertEqual(active.create_model(active.param).p.numel(), 18)
        self.assertEqual(inactive.create_model(inactive.param).p.numel(), 12)

    def test_cost_increment_and_derivatives_match_mf_and_closed_form(self):
        for memory in (False, True):
            increments = []
            for kind in ("sf", "mf"):
                with self.subTest(memory=memory, method=kind):
                    off, _, base_model = self.make_model(kind, 0., memory)
                    on, _, model = self.make_model(kind, 2.5, memory)
                    before, after = self.evaluate(base_model), self.evaluate(model)
                    x = np.r_[np.array([.1, -.2, .3, .4]), self.anchor]
                    if memory:
                        x = np.r_[x, [.12, -.1]]
                    u = np.array([.02, .1, .05])
                    base_p, p = self.parameters(off), self.parameters(on)
                    baseline, centered = before(x, u, base_p), after(x, u, p)
                    np.testing.assert_allclose(centered[0], baseline[0], atol=1e-14)
                    np.testing.assert_allclose(centered[1], baseline[1], atol=1e-14)
                    np.testing.assert_allclose(centered[2], baseline[2], atol=1e-14)
                    delta = np.array([.2, -.3, .4, .002])
                    x[4:8] += delta
                    baseline, changed = before(x, u, base_p), after(x, u, p)
                    inv_std = 1. / math.hypot(math.exp(self.anchor[2]), self.anchor[3])
                    weights = np.array([1., 1., 1., inv_std**2])
                    expected_cost = 2.5 * np.dot(weights * delta, delta)
                    expected_gradient = np.zeros_like(x)
                    expected_gradient[4:8] = 5. * weights * delta
                    expected_hessian = np.zeros((x.size, x.size))
                    expected_hessian[4:8, 4:8] = np.diag(5. * weights)
                    difference = [np.asarray(a) - np.asarray(b)
                                  for a, b in zip(changed, baseline)]
                    self.assertAlmostEqual(float(difference[0].item()), expected_cost)
                    self.assertAlmostEqual(float(difference[1].item()), expected_cost)
                    np.testing.assert_allclose(difference[2].ravel(), expected_gradient,
                                               rtol=1e-13, atol=1e-13)
                    np.testing.assert_allclose(difference[3], expected_hessian,
                                               rtol=1e-13, atol=1e-13)
                    np.testing.assert_array_equal(difference[4], 0.)
                    increments.append(difference)
            for sf, mf in zip(*increments):
                np.testing.assert_allclose(sf, mf, rtol=1e-13, atol=1e-13)

    def test_mask_normalization_and_terminal_match_mf(self):
        sf, sf_param, _ = self.make_model()
        mf, mf_param, _ = self.make_model("mf")
        for stage, active in enumerate((0., 1., 1., 1., 0.)):
            sf_p, mf_p = self.parameters(sf, stage), self.parameters(mf, stage)
            values = sf_p[sf.covariance_proximal_param_slice]
            np.testing.assert_array_equal(values, mf_p[mf.covariance_proximal_param_slice])
            np.testing.assert_array_equal(values[:4], self.anchor)
            self.assertAlmostEqual(values[4], 1. / math.hypot(.03, -.004))
            self.assertEqual(values[5], active)
            self.assertEqual(sf_p[sf.covariance_propagation_param_slice][0], float(stage < 3))
        sf_param.robust_horizon = mf_param.robust_horizon = 4
        self.assertEqual(self.parameters(sf, 4)[-1], 1.)
        self.assertEqual(self.parameters(mf, 4)[-1], 1.)
        tiny = self.anchor.copy()
        tiny[2:] = [math.log(1e-12), 0.]
        self.assertEqual(self.parameters(sf, anchor=tiny)[-2], 1e6)

    def test_invalid_weights_and_anchors_fail_before_solver_setup(self):
        optimizer, param, _ = self.make_model()
        for weight in (-.1, math.nan, math.inf, True, "1", None, [1.]):
            with self.subTest(weight=weight):
                param.covariance_proximal_weight = weight
                with self.assertRaisesRegex(ValueError, "covariance_proximal_weight"):
                    optimizer.create_model(param)
                with patch.object(optimizer, "initialize_ped_model") as geometry:
                    with self.assertRaisesRegex(ValueError, "covariance_proximal_weight"):
                        optimizer.setup(param, Mock(), Mock(), Mock())
                    geometry.assert_not_called()
        param.covariance_proximal_weight = .01
        optimizer.create_model(param)
        for anchor in (None, [0.] * 3, np.zeros((2, 2)), [0., 0., math.nan, 0.]):
            with self.subTest(anchor=anchor):
                with self.assertRaisesRegex(ValueError, "Nominal proximal covariance"):
                    optimizer._build_stage_param(self.path, np.zeros(4), 1.,
                                                 nominal_covariance=anchor)

    def test_preparation_uses_one_regenerated_nominal_for_main_and_backup(self):
        optimizer, _, _ = self.make_model()
        nominal = np.tile(np.r_[np.zeros(4), self.anchor], (5, 1))
        nominal[:, 4:7] += np.arange(5)[:, None] * .1
        nominal[:, 7] += np.arange(5) * .002
        stage_s = np.arange(5) * .1
        optimizer.state = None
        optimizer._cycle_solver_time = 0.
        for label in ("main", "backup"):
            with self.subTest(solver=label):
                solver = _Solver(nominal, np.tile([.2, .3, .2], (4, 1)))
                with (
                    patch.object(optimizer, "_get_current_s_bounds", return_value=(0., 2.)),
                    patch.object(optimizer, "_get_augmented_state_bounds", return_value=([], [], [])),
                    patch.object(optimizer, "_get_effective_v_s_max", return_value=1.),
                    patch.object(optimizer, "_predict_stage_s_values", return_value=stage_s),
                    patch.object(optimizer, "_regenerate_covariance_guess", return_value=nominal) as rollout,
                    patch.object(optimizer, "_compute_stage_risk_data",
                                 return_value=(None, np.zeros((5, 3, 3)), np.zeros((5, 4)))),
                    patch.object(optimizer, "_log_stage_risk_heads"),
                    patch.object(optimizer, "_build_stage_path_parameter", return_value=self.path),
                ):
                    self.assertEqual(optimizer._prepare_and_solve(solver), 0)
                    rollout.assert_called_once_with(solver, stage_s)
                self.assertEqual(set(solver.parameters), set(range(5)))
                for stage, p in solver.parameters.items():
                    actual = p[optimizer.covariance_proximal_param_slice]
                    np.testing.assert_array_equal(actual[:4], nominal[stage, 4:8])
                    self.assertEqual(actual[-1], float(1 <= stage <= 3))

    def test_warm_start_keeps_proximal_inactive_until_preparation(self):
        optimizer, param, model = self.make_model()
        nominal = np.tile(np.r_[np.zeros(4), self.anchor], (5, 1))
        solver = _Solver(nominal, np.zeros((4, 3)))
        optimizer.ocp = SimpleNamespace(parameter_values=np.zeros(int(model.p.numel())))
        optimizer.state = SimpleNamespace(_x=np.zeros(3), _u=np.zeros(2))
        system = SimpleNamespace(_dynamics=SimpleNamespace(
            nominal_safe_controller=Mock(return_value=(np.zeros(3), np.zeros(2)))))
        with (
            patch.object(optimizer, "_get_effective_v_s_max", return_value=1.),
            patch.object(optimizer, "_build_initial_covariance_state", return_value=self.anchor),
            patch.object(optimizer, "_clamp_s", side_effect=lambda s: s),
            patch.object(optimizer, "_regenerate_covariance_guess", return_value=nominal) as rollout,
        ):
            optimizer.add_warm_start(param, system, solver)
            rollout.assert_called_once_with(solver)
        self.assertEqual(set(solver.parameters), set(range(5)))
        for stage, p in solver.parameters.items():
            self.assertEqual(p.size, 25)
            np.testing.assert_array_equal(p[optimizer.covariance_proximal_param_slice], np.zeros(6))
            self.assertEqual(p[optimizer.covariance_propagation_param_slice][0], float(stage < 3))

    def test_cleanup_and_zero_weight_rebuild_clear_proximal_layout(self):
        optimizer, param, _ = self.make_model()
        optimizer.cleanup()
        self.assertEqual(optimizer.covariance_proximal_param_slice, slice(0, 0))
        param.covariance_proximal_weight = 0.
        model = optimizer.create_model(param, ca.MX.sym("rebuilt_psdf", 7))
        self.assertEqual(model.p.numel(), 19)
        self.assertEqual(optimizer.covariance_proximal_param_slice, slice(19, 19))
        self.assertEqual(optimizer._build_stage_param(self.path, np.zeros(4), 1.).size, 19)

    def test_ecs_shares_proximal_cost_and_preserves_noise_free_dynamics(self):
        param = CCECSOptimizerParam()
        self.assertEqual(param.covariance_proximal_weight, .01)
        self.assertTrue(all(getattr(param, name) == 0.0 for name in PROCESS_NOISE_FIELDS))
        optimizer = CCECSOptimizer()
        self.addCleanup(optimizer.cleanup)
        optimizer.param = param
        optimizer._configure_cluster_capacity(2)
        model = optimizer.create_model(param)
        self.assertEqual(model.p.numel(), 28)  # 8 + 7*K + 6, with K=2.
        self.assertEqual(optimizer.covariance_proximal_param_slice, slice(22, 28))
        p = optimizer._build_stage_param(self.path, np.zeros(6), 1.,
                                         psdf_stage=np.zeros(8), stage_idx=1,
                                         nominal_covariance=self.anchor)
        state = np.r_[np.zeros(4), self.anchor, np.zeros(2)]
        state[4] += .2
        state[7] += .001
        evaluate = self.evaluate(model)
        active = evaluate(state, np.zeros(3), p)
        p[optimizer.covariance_proximal_param_slice.stop - 1] = 0.0
        inactive = evaluate(state, np.zeros(3), p)
        inverse_std = 1.0 / np.hypot(np.exp(self.anchor[2]), self.anchor[3])
        expected = param.covariance_proximal_weight * (.2**2 + (.001 * inverse_std)**2)
        self.assertAlmostEqual(float(active[0] - inactive[0]), expected)
        self.assertAlmostEqual(float(active[1] - inactive[1]), expected)
        np.testing.assert_array_equal(active[4], inactive[4])

        optimizer.cleanup()
        self.assertEqual(optimizer.covariance_proximal_param_slice, slice(0, 0))
        param.covariance_proximal_weight = 0.0
        optimizer._configure_cluster_capacity(2)
        model = optimizer.create_model(param)
        self.assertEqual(model.p.numel(), 22)
        self.assertEqual(optimizer.covariance_proximal_param_slice, slice(22, 22))
        self.assertEqual(optimizer._build_stage_param(
            self.path, np.zeros(6), 1., psdf_stage=np.zeros(8), stage_idx=1,
        ).size, 22)


if __name__ == "__main__":
    unittest.main()
