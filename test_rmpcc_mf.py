import unittest

import casadi as ca
import numpy as np
import torch

from control.rmpcc_optimizer import RMPCCOptimizer, RMPCCOptimizerParam
from control.rmpcc_pv_optimizer import RMPCCPVOptimizer, RMPCCPVOptimizerParam
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


class RMPCCMultiFeatureTest(unittest.TestCase):
    def setUp(self):
        self.param = RMPCCOptimizerParam()
        self.param.horizon = 3
        self.param.tf = 0.3
        self.optimizer = RMPCCOptimizer()
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

    def test_mf_parameter_layout_and_terminal_constraint(self):
        self.optimizer.setup_ocp(self.param, None)
        verts = torch.tensor(
            [[-0.1, -0.05], [0.1, -0.05], [0.1, 0.05], [-0.1, 0.05]],
            dtype=torch.float32,
        )
        self.optimizer.psdf_wrapper = AugmentedPSDFWrapper(
            verts,
            K_max=2,
            E_max=4,
        )
        self.optimizer.add_obstacle_avoidance_constraint(self.param, None, None)

        self.assertEqual(self.optimizer.ocp.dims.np, 15)
        self.assertEqual(int(self.optimizer.ocp.model.p.shape[0]), 15)
        self.assertEqual(self.optimizer.ocp.dims.nh, 1)
        self.assertEqual(self.optimizer.ocp.dims.nh_e, 1)

        constraint = ca.Function(
            "test_mf_constraint",
            [self.optimizer.ocp.model.x, self.optimizer.ocp.model.p],
            [self.optimizer.ocp.model.con_h_expr],
        )
        x_value = np.arange(8, dtype=float) / 10.0
        p_value = np.zeros(15, dtype=float)
        A_value = np.arange(1, 9, dtype=float) / 20.0
        c_value = -0.3
        p_value[6:14] = A_value
        p_value[14] = c_value
        actual = float(constraint(x_value, p_value))
        self.assertAlmostEqual(actual, A_value @ x_value + c_value, places=12)

    def test_shifted_nominal_stage_alignment_and_covariance_repropagation(self):
        states = np.zeros((4, 8), dtype=float)
        states[:, 0] = [0.0, 1.0, 2.0, 3.0]
        states[:, 1] = [0.0, 0.1, 0.2, 0.3]
        states[:, 2] = 0.1
        states[:, 3] = [0.0, 0.1, 0.2, 0.3]
        states[:, 4:8] = 99.0
        inputs_ = np.array(
            [
                [0.1, 0.01, 0.1],
                [0.2, 0.02, 0.2],
                [0.3, 0.03, 0.3],
            ],
            dtype=float,
        )
        solver = _FakeSolver(states, inputs_)

        self.optimizer._has_shift_source = False
        first_states, first_inputs = self.optimizer._build_shifted_nominal(
            solver,
            0.0,
            10.0,
        )
        np.testing.assert_allclose(first_states[1, :3], states[1, :3])
        np.testing.assert_allclose(first_inputs, inputs_)
        np.testing.assert_allclose(first_states[0, :3], self.optimizer.state._x)
        self.assertFalse(np.any(first_states[:, 4:8] == 99.0))

        self.optimizer._has_shift_source = True
        shifted_states, shifted_inputs = self.optimizer._build_shifted_nominal(
            solver,
            0.0,
            10.0,
        )
        np.testing.assert_allclose(shifted_states[1, :3], states[2, :3])
        np.testing.assert_allclose(shifted_states[2, :3], states[3, :3])
        np.testing.assert_allclose(shifted_inputs[0], inputs_[1])
        np.testing.assert_allclose(shifted_inputs[-1], inputs_[-1])
        expected_terminal_pose = _FakeDynamics.forward_dynamics(
            states[-1, :3],
            inputs_[-1, :2],
            0.1,
        )
        np.testing.assert_allclose(shifted_states[-1, :3], expected_terminal_pose)
        self.assertAlmostEqual(shifted_states[-1, 3], 0.33, places=12)
        expected_covariance = self.optimizer._propagate_covariance_state_batch(
            shifted_inputs,
            shifted_states[:, 3],
        )
        np.testing.assert_allclose(shifted_states[:, 4:8], expected_covariance)

    def test_optimizer_routing_and_independent_classes(self):
        self.assertFalse(self.param.enable_backup_solver)
        self.assertEqual(simulation_mpc._resolve_optimizer_variant("rmpcc"), "rmpcc")
        self.assertEqual(
            simulation_mpc._resolve_optimizer_variant("rmpcc_pv"),
            "rmpcc_pv",
        )
        self.assertIsNot(RMPCCOptimizer, RMPCCPVOptimizer)
        self.assertIsNot(RMPCCOptimizerParam, RMPCCPVOptimizerParam)
        self.assertTrue(RMPCCPVOptimizerParam().use_projected_variance_margin)


if __name__ == "__main__":
    unittest.main()
