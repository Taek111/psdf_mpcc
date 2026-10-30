"""Input slew limits across the prediction horizon, without navigation runs."""

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


class _GuessSolver:
    def __init__(self, states, inputs):
        self.states = np.asarray(states, dtype=float).copy()
        self.inputs = np.asarray(inputs, dtype=float).copy()
        self.fields = {}
        self.status = 0

    def get(self, stage, field):
        return (self.states if field == "x" else self.inputs)[stage].copy()

    def set(self, stage, field, value):
        if field == "x":
            self.states[stage] = value
        elif field == "u":
            self.inputs[stage] = value
        else:
            self.fields[stage, field] = np.asarray(value, dtype=float).copy()

    def solve(self):
        self.status = 0
        return 0

    def reset(self, **kwargs):
        self.states.fill(0.0)
        self.inputs.fill(0.0)
        self.status = 0


class MPCCInputRateTest(unittest.TestCase):
    def make_optimizer(self, optimizer_type, param_type, **limits):
        optimizer = optimizer_type()
        self.addCleanup(optimizer.cleanup)
        param = param_type()
        # Isolate hard rate limits from the optional command-change cost.
        param.r_delta_v = 0.0
        param.r_delta_omega = 0.0
        param.horizon, param.tf = 3, 0.6
        if hasattr(param, "robust_horizon"):
            param.robust_horizon = param.horizon
        for name, value in limits.items():
            setattr(param, name, value)
        optimizer.param = param
        optimizer.state = SimpleNamespace(
            _x=np.array([0.3, -0.2, 0.7]), _u=np.array([0.12, -0.4, 0.63]),
        )
        optimizer._system_dynamics = DifferentialDriveDynamics
        return optimizer, param

    @staticmethod
    def setup_ocp(optimizer, param):
        if isinstance(optimizer, CCMFOptimizer):
            optimizer.setup_ocp(param)
        else:
            optimizer.setup_ocp(param, None)

    def test_parameter_defaults_and_disabled_dimensions_and_integrator(self):
        default_controllers = CONTROLLERS
        for optimizer_type, param_type, base_nx in default_controllers:
            with self.subTest(controller=optimizer_type.__name__):
                optimizer, param = self.make_optimizer(optimizer_type, param_type)
                expected_limits = (0.5, 5.0, -1.0, 1.0)
                self.assertEqual(
                    (param.a_max, param.alpha_max, param.omegamin, param.omegamax),
                    expected_limits,
                )
                self.assertEqual((param.vmin, param.vmax, param.v_s_target),
                                 (-1.0, 1.0, 1.0))
                param.a_max = param.alpha_max = None
                self.setup_ocp(optimizer, param)
                self.assertEqual(optimizer.nx, base_nx)
                self.assertEqual(optimizer.nu, 3)
                expected_integrator = "ERK" if base_nx == 4 else "DISCRETE"
                self.assertEqual(optimizer.ocp.solver_options.integrator_type, expected_integrator)
                self.assertEqual(optimizer.ocp.constraints.C.size, 0)

    def test_enabled_dynamics_remember_input_without_changing_tracking_or_covariance(self):
        control = np.array([0.27, -0.6, 0.19])
        for optimizer_type, param_type, base_nx in CONTROLLERS:
            with self.subTest(controller=optimizer_type.__name__):
                optimizer, param = self.make_optimizer(
                    optimizer_type, param_type, a_max=0.5, alpha_max=2.0,
                )
                baseline, baseline_param = self.make_optimizer(
                    optimizer_type, param_type, a_max=None, alpha_max=None,
                )
                old_model = baseline.create_model(baseline_param)
                model = optimizer.create_model(param)
                self.assertEqual(int(model.x.numel()), base_nx + 2)
                self.assertEqual(int(model.u.numel()), 3)
                self.assertEqual(optimizer._input_memory_slice, slice(base_nx, base_nx + 2))
                self.assertEqual(int(model.p.numel()), int(old_model.p.numel()))
                state = np.array([0.3, -0.2, 0.7, 0.4])
                if base_nx == 8:
                    state = np.r_[state, np.log([0.03, 0.02, 0.01]), -0.004]
                full_state = np.r_[state, [0.12, -0.4]]
                parameters = np.zeros(int(model.p.numel()))
                parameters[:5] = [0.2, -0.1, 0.8, 0.6, 0.3]
                if base_nx == 8:
                    parameters[optimizer.covariance_propagation_param_slice] = 1.0
                evaluate = ca.Function(
                    "rate_memory_model", [model.x, model.u, model.p],
                    [model.disc_dyn_expr, model.cost_expr_ext_cost,
                     model.cost_expr_ext_cost_e],
                )
                transition, stage_cost, terminal_cost = evaluate(full_state, control, parameters)
                transition = np.asarray(transition).ravel()
                np.testing.assert_array_equal(transition[-2:], control[:2])
                expected_pose = DifferentialDriveDynamics.forward_dynamics(
                    state[:3], control[:2], param.tf / param.horizon,
                )
                np.testing.assert_allclose(transition[:3], expected_pose, atol=1e-14)
                self.assertAlmostEqual(transition[3], state[3] + 0.2 * control[2])
                old_costs = ca.Function(
                    "base_rate_cost", [old_model.x, old_model.u, old_model.p],
                    [old_model.cost_expr_ext_cost, old_model.cost_expr_ext_cost_e],
                )(state, control, parameters)
                np.testing.assert_allclose(stage_cost, old_costs[0], atol=1e-14)
                np.testing.assert_allclose(terminal_cost, old_costs[1], atol=1e-14)
                if base_nx == 8:
                    old_transition = ca.Function(
                        "base_rate_transition", [old_model.x, old_model.u, old_model.p],
                        [old_model.disc_dyn_expr],
                    )(state, control, parameters)
                    np.testing.assert_allclose(
                        transition[:base_nx], np.asarray(old_transition).ravel(), atol=1e-14,
                    )

    def test_hard_rate_rows_use_previous_input_at_every_horizon_interval(self):
        for optimizer_type, param_type, base_nx in CONTROLLERS:
            with self.subTest(controller=optimizer_type.__name__):
                optimizer, param = self.make_optimizer(
                    optimizer_type, param_type, a_max=0.5, alpha_max=2.0,
                )
                self.setup_ocp(optimizer, param)
                self.assertEqual(optimizer.nx, base_nx + 2)
                self.assertEqual(optimizer.ocp.solver_options.integrator_type, "DISCRETE")
                constraints = optimizer.ocp.constraints
                np.testing.assert_array_equal(constraints.idxbx_0, np.arange(base_nx + 2))
                np.testing.assert_array_equal(constraints.lbx_0[-2:], optimizer.state._u[:2])
                np.testing.assert_array_equal(constraints.ubx_0, constraints.lbx_0)
                np.testing.assert_allclose(constraints.lg, [-0.1, -0.4])
                np.testing.assert_allclose(constraints.ug, [0.1, 0.4])
                self.assertEqual(constraints.C.shape, (2, base_nx + 2))
                self.assertEqual(constraints.D.shape, (2, 3))
                # A large change only at the last interval must still violate
                # the same hard rows; virtual progress speed has no slew limit.
                controls = np.array([[0.17, -0.2, 0.0], [0.22, 0.0, 0.7], [0.4, 0.1, 0.1]])
                previous = optimizer.state._u[:2].copy()
                for stage, control in enumerate(controls):
                    state = constraints.lbx_0.copy()
                    state[-2:] = previous
                    row_value = constraints.C @ state + constraints.D @ control
                    np.testing.assert_allclose(row_value, control[:2] - previous)
                    if stage < 2:
                        self.assertTrue(np.all(row_value >= constraints.lg))
                        self.assertTrue(np.all(row_value <= constraints.ug))
                    else:
                        self.assertGreater(row_value[0], constraints.ug[0])
                    previous = control[:2]
                np.testing.assert_array_equal(constraints.D[:, 2], [0.0, 0.0])

    def test_one_limit_does_not_constrain_the_other_input_rate(self):
        for optimizer_type, param_type, base_nx in CONTROLLERS:
            for field, control_index in (("a_max", 0), ("alpha_max", 1)):
                with self.subTest(controller=optimizer_type.__name__, limit=field):
                    limits = {"a_max": None, "alpha_max": None, field: 0.5}
                    optimizer, param = self.make_optimizer(optimizer_type, param_type, **limits)
                    self.setup_ocp(optimizer, param)
                    constraints = optimizer.ocp.constraints
                    self.assertEqual(constraints.C.shape, (1, base_nx + 2))
                    state = constraints.lbx_0.copy()
                    state[-2:] = [0.1, -0.4]
                    control = np.array([0.15, -0.35, 0.7])
                    np.testing.assert_allclose(
                        constraints.C @ state + constraints.D @ control,
                        [control[control_index] - state[-2 + control_index]],
                    )
                    self.assertEqual(constraints.D[0, 1 - control_index], 0.0)
                    self.assertAlmostEqual(constraints.ug[0], 0.1)

    def test_memory_guesses_follow_actual_applied_input_and_previous_stage_controls(self):
        inputs = np.array([[0.2, 0.3, 0.4], [0.25, 0.1, 0.2], [0.3, -0.1, 0.5]])
        for optimizer_type, param_type, base_nx in CONTROLLERS:
            with self.subTest(controller=optimizer_type.__name__):
                optimizer, param = self.make_optimizer(
                    optimizer_type, param_type, a_max=0.5, alpha_max=2.0,
                )
                self.setup_ocp(optimizer, param)
                states = np.tile(optimizer.ocp.constraints.lbx_0, (4, 1))
                states[:, -2:] = 100.0  # Stale optimized memory is discarded.
                original_base = states[:, :base_nx].copy()
                solver = _GuessSolver(states, inputs)
                optimizer._refresh_input_memory_guess(solver)
                np.testing.assert_array_equal(solver.states[0, -2:], optimizer.state._u[:2])
                np.testing.assert_array_equal(solver.states[1:, -2:], inputs[:, :2])
                np.testing.assert_array_equal(solver.states[:, :base_nx], original_base)
                # The measured/applied command can differ from both last
                # solution and retained warm start on the next control cycle.
                optimizer.state._u = np.array([-0.05, 0.8])
                optimizer._refresh_input_memory_guess(solver)
                np.testing.assert_array_equal(solver.states[0, -2:], [-0.05, 0.8])
                augmented = optimizer._with_input_memory(np.arange(base_nx, dtype=float))
                np.testing.assert_array_equal(augmented[-2:], [-0.05, 0.8])

    def test_mf_safety_rows_do_not_treat_input_memory_as_covariance(self):
        optimizer, param = self.make_optimizer(
            CCMFOptimizer, CCMFOptimizerParam, a_max=0.5, alpha_max=2.0,
        )
        self.setup_ocp(optimizer, param)
        with redirect_stdout(io.StringIO()):
            optimizer.add_obstacle_avoidance_constraint(param, None, update_obstacle_data=False)
        model = optimizer.ocp.model
        parameters = optimizer.ocp.parameter_values.copy()
        parameters[optimizer.mf_param_slice] = np.arange(1, 10)
        evaluate = ca.Function(
            "mf_memory_safety", [model.x, model.p],
            [model.con_h_expr, ca.jacobian(model.con_h_expr, model.x)],
        )
        state = optimizer.ocp.constraints.lbx_0.copy()
        residual, jacobian = evaluate(state, parameters)
        state[-2:] = [-0.7, 1.2]
        changed_residual, _ = evaluate(state, parameters)
        np.testing.assert_array_equal(residual, changed_residual)
        np.testing.assert_array_equal(np.asarray(jacobian)[:, -2:], np.zeros((2, 2)))

    def test_runtime_preparation_updates_memory_from_applied_input_each_cycle(self):
        variants = CONTROLLERS
        inputs = np.array([[0.2, 0.3, 0.4], [0.25, 0.1, 0.2], [0.3, -0.1, 0.5]])
        for optimizer_type, param_type, _ in variants:
            with self.subTest(controller=optimizer_type.__name__):
                optimizer, param = self.make_optimizer(
                    optimizer_type, param_type, a_max=0.5, alpha_max=2.0,
                )
                param.use_obstacle_constraint = False
                self.setup_ocp(optimizer, param)
                states = np.tile(optimizer.ocp.constraints.lbx_0, (4, 1))
                states[:, -2:] = 100.0
                optimizer.solver = solver = _GuessSolver(states, inputs)
                for applied_input in ([0.12, -0.4], [-0.05, 0.8]):
                    optimizer.state._u = np.asarray(applied_input)
                    with redirect_stdout(io.StringIO()):
                        if optimizer_type is MPCCOptimizer:
                            optimizer.solve_nlp()
                        else:
                            self.assertEqual(optimizer._prepare_and_solve(solver), 0)
                    np.testing.assert_array_equal(solver.fields[0, "lbx"][-2:], applied_input)
                    np.testing.assert_array_equal(solver.fields[0, "ubx"][-2:], applied_input)
                    np.testing.assert_array_equal(solver.states[0, -2:], applied_input)
                    np.testing.assert_array_equal(solver.states[1:, -2:], inputs[:, :2])

    def test_warm_start_and_emergency_stop_have_consistent_memory(self):
        system = SimpleNamespace(_dynamics=DifferentialDriveDynamics)
        inputs = np.tile([0.2, 0.3, 0.4], (3, 1))
        for optimizer_type, param_type, _ in CONTROLLERS:
            with self.subTest(controller=optimizer_type.__name__):
                optimizer, param = self.make_optimizer(
                    optimizer_type, param_type, a_max=0.5, alpha_max=2.0,
                )
                self.setup_ocp(optimizer, param)
                states = np.tile(optimizer.ocp.constraints.lbx_0, (4, 1))
                optimizer.solver = solver = _GuessSolver(states, inputs)
                with redirect_stdout(io.StringIO()):
                    optimizer.add_warm_start(param, system)
                np.testing.assert_array_equal(solver.states[0, -2:], optimizer.state._u[:2])
                np.testing.assert_array_equal(solver.states[1:, -2:], solver.inputs[:, :2])
                if optimizer_type is MPCCOptimizer:
                    continue  # Deterministic MPCC has no existing stop-recovery path.
                with redirect_stdout(io.StringIO()):
                    _, status, mode = optimizer.recovery_infeasible(4)
                self.assertEqual((status, mode), (4, "safe_stop"))
                self.assertEqual(solver.status, 4)
                np.testing.assert_array_equal(solver.inputs, np.zeros((3, 3)))
                np.testing.assert_array_equal(solver.states[0, -2:], optimizer.state._u[:2])
                np.testing.assert_array_equal(solver.states[1:, -2:], np.zeros((3, 2)))

    def test_each_controller_parameter_accepts_config_overrides(self):
        for _, param_type, _ in CONTROLLERS:
            with self.subTest(parameter=param_type.__name__):
                param = param_type()
                with redirect_stdout(io.StringIO()) as output:
                    simulation_mpc._apply_param_overrides(
                        param, {"a_max": 0.5, "alpha_max": 2.0}, "rate_test",
                    )
                self.assertEqual((param.a_max, param.alpha_max), (0.5, 2.0))
                self.assertNotIn("Unknown", output.getvalue())

    def test_zero_limit_freezes_input_and_invalid_limits_are_rejected(self):
        for optimizer_type, param_type, _ in CONTROLLERS:
            with self.subTest(controller=optimizer_type.__name__, limit="zero"):
                optimizer, param = self.make_optimizer(
                    optimizer_type, param_type, a_max=0.0, alpha_max=0.0,
                )
                self.setup_ocp(optimizer, param)
                np.testing.assert_array_equal(optimizer.ocp.constraints.lg, [0.0, 0.0])
                np.testing.assert_array_equal(optimizer.ocp.constraints.ug, [0.0, 0.0])
            for field in ("a_max", "alpha_max"):
                for invalid in (-0.1, np.nan, np.inf, [0.5], True):
                    with self.subTest(controller=optimizer_type.__name__, field=field, invalid=invalid):
                        optimizer, param = self.make_optimizer(
                            optimizer_type, param_type, **{field: invalid},
                        )
                        with self.assertRaisesRegex(ValueError, field):
                            optimizer.create_model(param)


if __name__ == "__main__":
    unittest.main()
