"""Deterministic MPCC factory, CLI, and logging without solver generation."""

import io
import unittest
from contextlib import ExitStack, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from control import DEFAULT_OPTIMIZER_TYPE, SUPPORTED_OPTIMIZER_TYPES
from control.mpcc_optimizer import AcadosSolution, MPCCOptimizerParam
from sim.logger import ControllerLogger
from sim.simulation_mpc import simulation_mpc
from sim.trial_metrics import collect_successful_solver_steps
from test_nmpc import build_output_name, build_parser


class _FactoryComplete(Exception):
    """Stop before navigation when inspecting factory parameters."""


class MPCCIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.simulation = simulation_mpc()
        self.pose = np.zeros(3)
        self.system = SimpleNamespace(
            _dt=0.1, _time=0.1,
            _state=SimpleNamespace(_x=self.pose, _u=np.zeros(2)),
            get_state=lambda: self.pose.copy(),
        )
        self.robot = Mock()
        self.robot._system = self.system
        self.robot._controller_logger = ControllerLogger()
        self.robot.set_controller.side_effect = lambda value: setattr(self.robot, "_controller", value)
        self.optimizer = SimpleNamespace(
            reset=Mock(), setup=Mock(), solver_times=[0.012], update_obstacles=Mock(),
            get_progress_info=lambda: {"path_s": 0.2, "predicted_s": 0.25},
        )
        self.spline = object()
        environment = (self.pose, np.array([1.0, 0.0, 0.0]), (((0, 0), (2, 2)), 0.1), [])
        for replacement in (
            patch.object(self.simulation, "create_env", return_value=environment),
            patch("sim.simulation_mpc.sample_start_pose", return_value=(self.pose, None)),
            patch("sim.simulation_mpc.DifferentialDriveSystem", return_value=self.system),
            patch("sim.simulation_mpc.Robot", return_value=self.robot),
            patch("sim.simulation_mpc.AstarLoSPathGenerator"),
            patch("sim.simulation_mpc.ConstantSpeedTrajectoryGenerator"),
            patch("control.mpcc_optimizer.MPCCOptimizer", return_value=self.optimizer),
        ):
            self.stack.enter_context(replacement)
        self.planner = self.stack.enter_context(
            patch("sim.simulation_mpc.SplineReferenceGenerator", return_value=self.spline)
        )

    def capture_factory(self, config):
        output = io.StringIO()
        with (
            patch("sim.simulation_mpc.BaseController", side_effect=_FactoryComplete) as controller,
            patch("sim.simulation_mpc.SingleAgentSimulation") as navigation,
            redirect_stdout(output),
        ):
            with self.assertRaises(_FactoryComplete):
                self.simulation.mpc_test(
                    "maze", "rectangle", optimizer_type="mpcc",
                    dynamics_type="differential_drive", path_planner="astar", config=config,
                )
            navigation.assert_not_called()
        actual_optimizer, params = controller.call_args.args
        self.assertIs(actual_optimizer, self.optimizer)
        self.optimizer.reset.assert_called_once_with()
        self.robot.set_local_planner.assert_called_with(self.spline)
        return params, output.getvalue()

    def test_factory_keeps_mpcc_defaults_and_ignores_cc_and_detection_overrides(self):
        params, output = self.capture_factory({
            "cc_mf": {"horizon": 99, "q_s": 9.0, "vmax": 99.0},
            "cc_sf": {"horizon": 99, "q_s": 8.0},
            "psdf_detection": {"window_width": 123.0},
        })
        self.assertIsInstance(params, MPCCOptimizerParam)
        self.assertEqual((params.horizon, params.tf), (20, 2.0))
        self.assertEqual((params.q_s, params.q_s_ref, params.vmax), (1.0, 2.0, 0.7))
        np.testing.assert_array_equal(params.mat_Qe, np.diag([1.0, 10.0]))
        np.testing.assert_array_equal(params.mat_Re, np.diag([1.0, 0.01]))
        self.assertIn("MPCC ignores legacy 'psdf_detection'", output)
        self.assertEqual(params.detection_window_width, 4.0)
        planner = self.planner.call_args.kwargs
        self.assertEqual(planner["num_horizon"], 20)
        self.assertAlmostEqual(planner["local_path_timestep"], 0.1)
        self.assertAlmostEqual(planner["reference_speed"], 1.4)
        self.assertAlmostEqual(planner["window_length"], 5.0)
        self.assertEqual(planner["sample_count"], 20)
        self.assertEqual(planner["spline_bc_type"], "natural")

    def test_factory_applies_only_its_own_overrides_before_spline_creation(self):
        params, _ = self.capture_factory({
            "mpcc": {"horizon": 3, "tf": 0.3, "d_min": 0.006, "v_s_max": 0.4},
        })
        self.assertEqual(params.horizon, 3)
        self.assertAlmostEqual(params.d_min, 0.006)
        self.assertEqual(self.planner.call_args.kwargs["num_horizon"], 3)
        self.assertAlmostEqual(self.planner.call_args.kwargs["local_path_timestep"], 0.1)
        self.assertAlmostEqual(self.planner.call_args.kwargs["reference_speed"], 0.28)
        self.assertAlmostEqual(self.planner.call_args.kwargs["window_length"], 0.8)

    def test_controller_status_progress_and_successful_timings_reach_logging(self):
        for status in (0, 4):
            with self.subTest(status=status):
                self.robot._controller_logger = ControllerLogger()
                solver = SimpleNamespace(
                    status=status,
                    get=lambda stage, field: np.zeros(4 if field == "x" else 3),
                )
                solution = AcadosSolution(solver, 1, {"x": "x", "u": "u"})
                self.optimizer.solve_nlp = Mock(return_value=solution)

                def one_cycle(duration):
                    controller = self.robot._controller
                    controller.generate_control_input(self.system, None, {}, [])
                    controller.logging(self.robot._controller_logger)
                    return {"status": "success" if status == 0 else "failure", "final_time": 0.1}

                with (
                    patch(
                        "sim.simulation_mpc.SingleAgentSimulation",
                        return_value=SimpleNamespace(run_navigation=one_cycle),
                    ),
                    patch.object(self.simulation, "save_trial_history_to_csv") as save_trial,
                    patch.object(self.simulation, "save_pose_sdf_to_csv") as save_pose,
                    redirect_stdout(io.StringIO()),
                ):
                    self.simulation.mpc_test("maze", "rectangle", optimizer_type="mpcc", config={})
                self.assertEqual(self.robot._controller.get_last_solver_status_info()["status_code"], status)
                row = self.simulation.pose_sdf_data[0]
                self.assertEqual(row["solver_status"], status)
                self.assertEqual((row["path_s"], row["predicted_s"]), (0.2, 0.25))
                save_trial.assert_called_once_with("trial_history_MPC_mpcc_rectangle_maze")
                save_pose.assert_called_once_with("pose_sdf_MPC_mpcc_rectangle_maze")
                steps = collect_successful_solver_steps(self.simulation)
                self.assertEqual(len(steps), 1 if status == 0 else 0)
                if status == 0:
                    self.assertEqual(steps[0]["solver_status_code"], 0)
                    self.assertEqual(steps[0]["solver_computation_time_s"], 0.012)


class MPCCCliRoutingTest(unittest.TestCase):
    def test_cli_registers_mpcc_without_changing_the_default(self):
        self.assertIn("mpcc", SUPPORTED_OPTIMIZER_TYPES)
        self.assertEqual(DEFAULT_OPTIMIZER_TYPE, "psdf")
        parser = build_parser()
        self.assertEqual(parser.parse_args([]).optimizer_type, "psdf")
        args = parser.parse_args([
            "--single", "--optimizer-type", "mpcc", "--maze-type", "maze",
            "--robot-shape", "rectangle", "--dynamics-type", "differential_drive",
        ])
        self.assertEqual(simulation_mpc._resolve_optimizer_variant(args.optimizer_type), "mpcc")
        name = build_output_name(args, interrupted=True)
        self.assertEqual(name, "MPC_mpcc_rectangle_maze_interrupted")
        self.assertEqual(simulation_mpc()._normalize_animation_method(name), "mpcc")


if __name__ == "__main__":
    unittest.main()
