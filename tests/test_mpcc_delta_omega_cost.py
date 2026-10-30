"""Command smoothing without solver generation or navigation runs."""

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import casadi as ca
import numpy as np

from control.cc_ecs_optimizer import CCECSOptimizer, CCECSOptimizerParam
from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
from control.cc_sf_optimizer import CCSFOptimizer, CCSFOptimizerParam
from control.mpcc_optimizer import MPCCOptimizer, MPCCOptimizerParam
from models.dd_dynamics import DifferentialDriveDynamics
from sim.simulation_mpc import simulation_mpc


CONTROLLERS = (
    (MPCCOptimizer, MPCCOptimizerParam, 4),
    (CCSFOptimizer, CCSFOptimizerParam, 8),
    (CCMFOptimizer, CCMFOptimizerParam, 8),
    (CCECSOptimizer, CCECSOptimizerParam, 8),
)
SMOOTHING_AXES = ((0, "r_delta_v"), (1, "r_delta_omega"))


class MPCCDeltaOmegaCostTest(unittest.TestCase):
    def make_optimizer(self, optimizer_type, param_type, **overrides):
        optimizer = optimizer_type()
        self.addCleanup(optimizer.cleanup)
        param = param_type()
        param.horizon, param.tf = 3, 0.6
        if hasattr(param, "robust_horizon"):
            param.robust_horizon = 3
        for name, value in overrides.items():
            setattr(param, name, value)
        optimizer.param = param
        optimizer.state = SimpleNamespace(
            _x=np.array([0.3, -0.2, 0.7]), _u=np.array([0.12, -0.4]),
        )
        optimizer._system_dynamics = DifferentialDriveDynamics
        return optimizer, param

    @staticmethod
    def setup_ocp(optimizer, param):
        if isinstance(optimizer, CCMFOptimizer):
            optimizer.setup_ocp(param)
        else:
            optimizer.setup_ocp(param, None)

    @staticmethod
    def state_and_parameters(model, base_nx):
        state = np.array([0.3, -0.2, 0.7, 0.4])
        if base_nx == 8:
            state = np.r_[state, np.log([0.03, 0.02, 0.01]), -0.004]
        state = np.r_[state, [0.12, -0.4]]
        parameters = np.zeros(int(model.p.numel()))
        parameters[:5] = [0.2, -0.1, 0.8, 0.6, 0.3]
        return state, parameters

    @staticmethod
    def evaluate_cost(model):
        variables = ca.vertcat(model.u, model.x)
        hessian, _ = ca.hessian(model.cost_expr_ext_cost, variables)
        return ca.Function(
            "delta_input_cost", [model.x, model.u, model.p],
            [model.cost_expr_ext_cost, model.cost_expr_ext_cost_e,
             ca.gradient(model.cost_expr_ext_cost, variables), hessian],
        )

    def test_cost_derivatives_penalize_command_differences(self):
        control = np.array([0.27, 0.3, 0.19])
        for optimizer_type, param_type, base_nx in CONTROLLERS:
            for r_delta_v, r_delta_omega in ((0.25, 0.0), (0.0, 0.25), (0.75, 0.25)):
                with self.subTest(controller=optimizer_type.__name__,
                                  weights=(r_delta_v, r_delta_omega)):
                    optimizer, param = self.make_optimizer(
                        optimizer_type, param_type,
                        r_delta_v=r_delta_v, r_delta_omega=r_delta_omega,
                    )
                    baseline, baseline_param = self.make_optimizer(
                        optimizer_type, param_type, r_delta_v=0.0, r_delta_omega=0.0,
                    )
                    model = optimizer.create_model(param)
                    base_model = baseline.create_model(baseline_param)
                    state, parameters = self.state_and_parameters(model, base_nx)
                    actual = self.evaluate_cost(model)(state, control, parameters)
                    original = self.evaluate_cost(base_model)(state, control, parameters)
                    expected_cost = 0.0
                    expected_gradient = np.zeros(3 + base_nx + 2)
                    expected_hessian = np.zeros((len(expected_gradient),) * 2)
                    for axis, weight_name in SMOOTHING_AXES:
                        weight = getattr(param, weight_name)
                        difference = control[axis] - state[base_nx + axis]
                        expected_cost += weight * difference**2
                        # A difference penalty must couple the current command
                        # to its memory, rather than penalizing each separately.
                        memory_index = 3 + base_nx + axis
                        expected_gradient[axis] = 2 * weight * difference
                        expected_gradient[memory_index] = -expected_gradient[axis]
                        expected_hessian[axis, axis] = 2 * weight
                        expected_hessian[memory_index, memory_index] = 2 * weight
                        expected_hessian[axis, memory_index] = -2 * weight
                        expected_hessian[memory_index, axis] = -2 * weight
                    self.assertAlmostEqual(float(actual[0] - original[0]), expected_cost)
                    self.assertAlmostEqual(float(actual[1]), float(original[1]))
                    np.testing.assert_allclose(
                        np.asarray(actual[2] - original[2]).ravel(),
                        expected_gradient, atol=1e-14,
                    )
                    np.testing.assert_allclose(
                        actual[3] - original[3], expected_hessian, atol=1e-14,
                    )

    def test_alternating_commands_cost_more_than_constant_commands_over_horizon(self):
        for optimizer_type, param_type, base_nx in CONTROLLERS:
            for axis, weight_name in SMOOTHING_AXES:
                with self.subTest(controller=optimizer_type.__name__, axis=axis):
                    optimizer, param = self.make_optimizer(
                        optimizer_type, param_type, **{weight_name: 0.1},
                    )
                    baseline, baseline_param = self.make_optimizer(
                        optimizer_type, param_type, **{weight_name: 0.0},
                    )
                    model = optimizer.create_model(param)
                    base_model = baseline.create_model(baseline_param)
                    evaluate = self.evaluate_cost(model)
                    original = self.evaluate_cost(base_model)
                    transition = ca.Function(
                        "delta_input_memory", [model.x, model.u, model.p],
                        [model.disc_dyn_expr],
                    )
                    costs = []
                    for commands in ([0.175] * 3, [-0.175, 0.175, -0.175]):
                        state, parameters = self.state_and_parameters(model, base_nx)
                        optimizer.state._u = np.array([0.12, -0.4])
                        optimizer.state._u[axis] = 0.175
                        state[-2:] = optimizer._with_input_memory([])
                        total = 0.0
                        for command in commands:
                            control = np.array([0.3, 0.3, 0.4])
                            control[axis] = command
                            total += float(evaluate(state, control, parameters)[0]
                                           - original(state, control, parameters)[0])
                            state = np.asarray(transition(state, control, parameters)).ravel()
                            np.testing.assert_array_equal(state[-2:], control[:2])
                        costs.append(total)
                    self.assertAlmostEqual(costs[0], 0.0)
                    self.assertAlmostEqual(costs[1], 3 * getattr(param, weight_name) * 0.35**2)

    def test_cost_keeps_applied_input_memory_when_rate_limits_are_disabled(self):
        for optimizer_type, param_type, base_nx in CONTROLLERS:
            for r_delta_v, r_delta_omega in ((0.1, 0.0), (0.0, 0.1), (0.1, 0.1)):
                with self.subTest(controller=optimizer_type.__name__,
                                  weights=(r_delta_v, r_delta_omega)):
                    optimizer, param = self.make_optimizer(
                        optimizer_type, param_type, a_max=None, alpha_max=None,
                        r_delta_v=r_delta_v, r_delta_omega=r_delta_omega,
                    )
                    self.setup_ocp(optimizer, param)
                    self.assertEqual(optimizer.nx, base_nx + 2)
                    self.assertEqual(optimizer.ocp.solver_options.integrator_type, "DISCRETE")
                    self.assertEqual(optimizer.ocp.constraints.C.size, 0)
                    self.assertEqual(optimizer.ocp.constraints.D.size, 0)
                    np.testing.assert_array_equal(
                        optimizer.ocp.constraints.lbx_0[-2:], optimizer.state._u,
                    )
                    np.testing.assert_array_equal(
                        optimizer.ocp.constraints.ubx_0[-2:], optimizer.state._u,
                    )
                    optimizer.state._u = np.array([-0.3, 0.6])
                    np.testing.assert_array_equal(optimizer._with_input_memory([]), [-0.3, 0.6])

    def test_zero_weight_and_no_limits_restore_unaugmented_model(self):
        for optimizer_type, param_type, base_nx in CONTROLLERS:
            with self.subTest(controller=optimizer_type.__name__):
                optimizer, param = self.make_optimizer(
                    optimizer_type, param_type,
                    r_delta_v=0.0, r_delta_omega=0.0, a_max=None, alpha_max=None,
                )
                self.setup_ocp(optimizer, param)
                self.assertEqual(optimizer.nx, base_nx)
                self.assertIsNone(optimizer._input_memory_slice)
                expected_integrator = "ERK" if base_nx == 4 else "DISCRETE"
                self.assertEqual(optimizer.ocp.solver_options.integrator_type,
                                 expected_integrator)

    def test_config_override_and_invalid_weights(self):
        for optimizer_type, param_type, _ in CONTROLLERS:
            with self.subTest(controller=optimizer_type.__name__, override=True):
                param = param_type()
                with redirect_stdout(io.StringIO()) as output:
                    simulation_mpc._apply_param_overrides(
                        param, {"r_delta_v": 0.4, "r_delta_omega": 0.3}, "delta_input_test",
                    )
                self.assertEqual(param.r_delta_v, 0.4)
                self.assertEqual(param.r_delta_omega, 0.3)
                self.assertNotIn("Unknown", output.getvalue())
            for _, weight_name in SMOOTHING_AXES:
                for invalid in (-0.1, np.nan, np.inf, [0.1], True, None, "0.1"):
                    with self.subTest(controller=optimizer_type.__name__,
                                      weight=weight_name, invalid=invalid):
                        optimizer, param = self.make_optimizer(
                            optimizer_type, param_type, **{weight_name: invalid},
                        )
                        with self.assertRaisesRegex(ValueError, weight_name):
                            optimizer.create_model(param)


if __name__ == "__main__":
    unittest.main()
