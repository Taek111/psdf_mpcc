"""Check plant/MPC integration against analytic motion and actual model maps."""

import gc
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import casadi as ca
import numpy as np

from control.cc_covariance import symbolic_dd_step
from models.dd import (
    DifferentialDriveDynamics,
    DifferentialDriveStates,
    DifferentialDriveSystem,
    LocalizationErrorModel,
    LocalizedDifferentialDriveSystem,
)
from models.geometry_utils import DifferentialDriveDynamics as GeometryDynamics


def exact_constant_input(pose, control, dt):
    """Analytic circular arc for a zero-order-held differential-drive input."""
    theta = pose[2]
    v, omega = control[:2]
    angle = omega * dt
    distance = v * dt * np.sinc(angle / (2.0 * np.pi))
    return pose + np.array([
        distance * np.cos(theta + 0.5 * angle),
        distance * np.sin(theta + 0.5 * angle),
        angle,
    ])


class DifferentialDriveRK4Test(unittest.TestCase):
    def test_straight_stationary_and_pure_rotation_match_exact_solution(self):
        pose = np.array([0.7, -0.3, 0.8])
        for control in ([0.0, 0.0], [0.0, -1.2], [0.6, 0.0], [-0.6, 0.0]):
            with self.subTest(control=control):
                np.testing.assert_allclose(
                    DifferentialDriveDynamics.forward_dynamics(pose, control, 0.1),
                    exact_constant_input(pose, control, 0.1),
                    rtol=0.0, atol=2e-15,
                )

    def test_turning_motion_has_fourth_order_global_accuracy(self):
        pose = np.array([0.7, -0.3, 0.8])
        control = np.array([0.6, -1.2])
        target = exact_constant_input(pose, control, 1.0)
        errors = []
        for dt in (0.1, 0.05):
            state = pose.copy()
            for _ in range(round(1.0 / dt)):
                state = DifferentialDriveDynamics.forward_dynamics(state, control, dt)
            errors.append(np.linalg.norm(state[:2] - target[:2]))
        self.assertGreater(errors[0] / errors[1], 15.0)
        self.assertLess(errors[0] / errors[1], 17.0)
        self.assertLess(errors[0], 1e-7)

    def test_numpy_casadi_and_cc_maps_match_for_signed_inputs(self):
        self.assertIs(GeometryDynamics, DifferentialDriveDynamics)
        rng = np.random.default_rng(12)
        for dt in (0.02, 0.1, 0.2):
            discrete = DifferentialDriveDynamics.forward_dynamics_opt(dt)
            pose = ca.MX.sym("pose", 3)
            control = ca.MX.sym("control", 2)
            cc_discrete = ca.Function(
                "cc_pose_step", [pose, control],
                [symbolic_dd_step(pose, control[0], control[1], dt)],
            )
            for _ in range(64):
                state = rng.uniform([-2, -2, -np.pi], [2, 2, np.pi])
                action = rng.uniform([-0.7, -1.2], [0.7, 1.2])
                expected = DifferentialDriveDynamics.forward_dynamics(state, action, dt)
                for step in (discrete, cc_discrete):
                    np.testing.assert_allclose(
                        np.asarray(step(state, action)).reshape(3), expected,
                        rtol=0.0, atol=2e-15,
                    )

    def test_normal_and_localized_plants_use_rk4(self):
        pose = np.array([0.7, -0.3, 0.8])
        action = np.array([-0.5, 1.0])
        exact = exact_constant_input(pose, action, 0.1)
        systems = (
            DifferentialDriveSystem(
                DifferentialDriveStates(pose.copy()), None, DifferentialDriveDynamics(),
            ),
            LocalizedDifferentialDriveSystem(
                DifferentialDriveStates(pose.copy()), None, DifferentialDriveDynamics(),
                LocalizationErrorModel(),
            ),
        )
        for system in systems:
            system.update(action)
            state = system.get_true_state() if hasattr(system, "get_true_state") else system.get_state()
            np.testing.assert_allclose(state, exact, rtol=0.0, atol=2e-9)
            np.testing.assert_allclose(system.get_state(), state, rtol=0.0, atol=2e-15)
            self.assertAlmostEqual(system._time, 0.1)


class MPCIntegrationTest(unittest.TestCase):
    def setUp(self):
        # Some legacy optimizer destructors delete codegen files in the cwd.
        self.previous_directory = os.getcwd()
        self.directory = tempfile.TemporaryDirectory(prefix="psdf-rk4-models-")
        os.chdir(self.directory.name)

    def tearDown(self):
        gc.collect()
        os.chdir(self.previous_directory)
        self.directory.cleanup()

    @staticmethod
    def make_erk_ocps():
        from control.psdf_optimizer import PSDFOptimizer, PSDFOptimizerParam

        param = PSDFOptimizerParam()
        optimizer = PSDFOptimizer()
        optimizer.set_state(SimpleNamespace(_x=np.zeros(3), _u=np.zeros(2)))
        optimizer.setup_ocp(param, np.zeros((param.horizon + 1, 3)))
        return [optimizer.ocp]

    def test_erk_ocps_explicitly_select_one_classical_rk4_step(self):
        for ocp in self.make_erk_ocps():
            with self.subTest(model=ocp.model.name):
                self.assertEqual(ocp.solver_options.integrator_type, "ERK")
                self.assertEqual(ocp.solver_options.sim_method_num_stages, 4)
                self.assertEqual(ocp.solver_options.sim_method_num_steps, 1)
                self.assertAlmostEqual(ocp.solver_options.tf / ocp.dims.N, 0.1)

    def test_cc_augmented_model_pose_steps_match_the_plant(self):
        from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
        from control.cc_sf_optimizer import CCSFOptimizer, CCSFOptimizerParam

        rng = np.random.default_rng(13)
        for optimizer_type, param_type in (
            (CCMFOptimizer, CCMFOptimizerParam), (CCSFOptimizer, CCSFOptimizerParam),
        ):
            param = param_type()
            optimizer = optimizer_type()
            optimizer.setup_ocp(param) if optimizer_type is CCMFOptimizer else optimizer.setup_ocp(param, None)
            model = optimizer.ocp.model
            self.assertEqual(optimizer.ocp.solver_options.integrator_type, "DISCRETE")
            dt = param.tf / param.horizon
            self.assertAlmostEqual(dt, 0.1)
            step = ca.Function("augmented_step", [model.x, model.u, model.p], [model.disc_dyn_expr])
            for _ in range(64):
                state = np.r_[rng.uniform(-2.0, 2.0, 4), np.log([0.02, 0.03, 0.04]), -0.01]
                control = rng.uniform([-0.7, -1.2, 0.0], [0.7, 1.2, 0.7])
                actual = np.asarray(step(state, control, optimizer.ocp.parameter_values)).reshape(-1)
                expected = DifferentialDriveDynamics.forward_dynamics(state[:3], control, dt)
                np.testing.assert_allclose(actual[:3], expected, rtol=0.0, atol=2e-15)

    def test_simulation_rejects_a_prediction_timestep_mismatch(self):
        from sim.simulation_mpc import simulation_mpc

        with self.assertRaisesRegex(ValueError, "prediction timestep.*must match"):
            simulation_mpc().mpc_test(
                "maze", "rectangle", optimizer_type="cc_mf",
                config={"cc_mf": {"horizon": 20, "tf": 4.0}},
            )

    def test_simulation_passes_the_shared_rk4_map_to_casadi_mpcs(self):
        from sim.simulation_mpc import simulation_mpc

        class SimulationReady(Exception):
            pass

        pose = np.array([0.7, -0.3, 0.8])
        control = np.array([-0.5, 1.0])
        for method in ("obca", "uobca", "dcbf", "sipoc"):
            with self.subTest(method=method):
                simulation = simulation_mpc()
                # Inspect the actual wiring before any planning or solver run.
                with patch("sim.simulation_mpc.SingleAgentSimulation", side_effect=SimulationReady):
                    with self.assertRaises(SimulationReady):
                        simulation.mpc_test("maze", "rectangle", optimizer_type=method)
                plant = simulation.robot._system
                optimizer = simulation.robot._controller._optimizer
                predicted = np.asarray(optimizer.dynamics_opt(pose, control)).reshape(3)
                expected = plant._dynamics.forward_dynamics(pose, control, plant._dt)
                np.testing.assert_allclose(predicted, expected, rtol=0.0, atol=2e-15)

    @unittest.skipUnless(os.environ.get("PSDF_TEST_ACADOS_INTEGRATION") == "1",
                         "set PSDF_TEST_ACADOS_INTEGRATION=1 to build acados integrators")
    def test_actual_acados_erk_integrators_match_the_plant(self):
        from acados_template import AcadosSim, AcadosSimSolver

        rng = np.random.default_rng(14)
        for ocp in self.make_erk_ocps():
            sim = AcadosSim()
            sim.model = ocp.model
            sim.parameter_values = ocp.parameter_values
            sim.code_export_directory = os.path.join(self.directory.name, ocp.model.name)
            sim.solver_options.integrator_type = ocp.solver_options.integrator_type
            sim.solver_options.num_stages = ocp.solver_options.sim_method_num_stages
            sim.solver_options.num_steps = ocp.solver_options.sim_method_num_steps
            sim.solver_options.T = ocp.solver_options.tf / ocp.dims.N
            solver = AcadosSimSolver(sim, json_file=f"{ocp.model.name}.json", verbose=False)
            maximum_error = 0.0
            for _ in range(64):
                state = rng.uniform(-2.0, 2.0, int(sim.model.x.numel()))
                control = rng.uniform(-1.0, 1.0, int(sim.model.u.numel()))
                solver.set("x", state)
                solver.set("u", control)
                self.assertEqual(solver.solve(), 0)
                actual = solver.get("x")[:3]
                expected = DifferentialDriveDynamics.forward_dynamics(state[:3], control, sim.solver_options.T)
                maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
            print(f"{ocp.model.name}: acados/plant maximum error = {maximum_error:.3e} m/rad")
            self.assertLessEqual(maximum_error, 2e-14)


if __name__ == "__main__":
    unittest.main()
