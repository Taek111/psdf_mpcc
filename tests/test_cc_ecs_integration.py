"""CC-ECS simulator wiring and CSV contracts without a solver or navigation run."""

import argparse
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from sim.simulation_mpc import simulation_mpc
from test_nmpc import build_output_name


class _FactoryComplete(Exception):
    """Stop setup at the controller boundary before any simulation starts."""


class CCECSIntegrationTest(unittest.TestCase):
    def test_factory_uses_only_ecs_overrides_and_matching_spline(self):
        simulation = simulation_mpc()
        optimizer = SimpleNamespace(reset=Mock())
        robot = Mock()
        spline = object()
        pose = np.array([0.0, 0.0, 0.0])
        environment = (
            pose, np.array([1.0, 0.0, 0.0]), (((0, 0), (2, 2)), 0.1), [],
        )
        config = {
            "cc_mf": {"horizon": 99, "chance_epsilon": 0.9, "vmin": -99.0},
            "cc_ecs": {
                "horizon": 3, "tf": 0.3, "d_min": 0.006, "v_s_max": 0.4,
            },
        }
        with (
            patch.object(simulation, "create_env", return_value=environment),
            patch(
                "sim.simulation_mpc.sample_start_pose", return_value=(pose, None),
            ),
            patch(
                "sim.simulation_mpc.DifferentialDriveSystem",
                return_value=SimpleNamespace(_dt=0.1),
            ),
            patch("sim.simulation_mpc.Robot", return_value=robot),
            patch("sim.simulation_mpc.AstarLoSPathGenerator"),
            patch("sim.simulation_mpc.ConstantSpeedTrajectoryGenerator"),
            patch(
                "sim.simulation_mpc.SplineReferenceGenerator", return_value=spline,
            ) as planner,
            patch("control.cc_ecs_optimizer.CCECSOptimizer", return_value=optimizer),
            patch(
                "sim.simulation_mpc.BaseController", side_effect=_FactoryComplete,
            ) as controller,
            patch("sim.simulation_mpc.SingleAgentSimulation") as navigation,
        ):
            with self.assertRaises(_FactoryComplete):
                simulation.mpc_test(
                    "maze", "rectangle", optimizer_type="cc_ecs", config=config,
                )

        actual_optimizer, params = controller.call_args.args
        self.assertIs(actual_optimizer, optimizer)
        self.assertEqual(params.horizon, 3)
        self.assertAlmostEqual(params.d_min, 0.006)
        self.assertAlmostEqual(params.chance_epsilon, 0.20)
        self.assertAlmostEqual(params.vmin, -0.7)
        self.assertEqual(planner.call_args.kwargs["num_horizon"], 3)
        self.assertAlmostEqual(planner.call_args.kwargs["local_path_timestep"], 0.1)
        robot.set_local_planner.assert_called_with(spline)
        optimizer.reset.assert_called_once_with()
        navigation.assert_not_called()

    def test_output_names_match_cli_simulator_and_animation_routing(self):
        args = argparse.Namespace(
            optimizer_type="cc_ecs", robot_shape="rectangle", maze_type="maze",
            output_suffix="trial7",
        )
        name = build_output_name(args, interrupted=True)
        self.assertEqual(name, "MPC_cc_ecs_rectangle_maze_trial7_interrupted")
        self.assertEqual(
            name,
            simulation_mpc.build_output_name(
                "rectangle", "maze", "cc_ecs", interrupted=True,
                extra_suffix="trial7",
            ),
        )
        self.assertEqual(
            simulation_mpc()._normalize_animation_method(name), "cc_ecs",
        )

    def test_ecs_exports_preserve_each_optimizer_schema(self):
        optimizer = SimpleNamespace()
        samples = {
            "constraint": (
                ["stage", "obstacle", "ecs_residual"],
                {"stage": 1, "obstacle": 2, "ecs_residual": 0.012},
            ),
            "covariance": (
                ["stage", "sigma_f"], {"stage": 1, "sigma_f": 0.02},
            ),
            "cycle": (
                ["solve_mode", "solver_status"],
                {"solve_mode": "safe_stop", "solver_status": 4},
            ),
        }
        for name, (fields, row) in samples.items():
            setattr(optimizer, f"get_{name}_log_fields", Mock(return_value=fields))
            setattr(optimizer, f"get_{name}_log_rows", Mock(return_value=[row]))
        simulation = simulation_mpc()
        simulation.robot = SimpleNamespace(
            _controller=SimpleNamespace(_optimizer=optimizer),
        )
        simulation.current_name = "MPC_cc_ecs_rectangle_maze"
        with tempfile.TemporaryDirectory() as directory:
            simulation.output_root_dir = directory
            simulation.save_cc_ecs_history_to_csv()
            for name, (fields, row) in samples.items():
                file_kind = "constraints" if name == "constraint" else name
                filename = f"cc_ecs_{file_kind}_{simulation.current_name}.csv"
                path = Path(directory) / "data" / filename
                with path.open(newline="", encoding="utf-8") as csv_file:
                    reader = csv.DictReader(csv_file)
                    self.assertEqual(reader.fieldnames, fields)
                    expected = {key: str(value) for key, value in row.items()}
                    self.assertEqual(list(reader), [expected])
            self.assertEqual(len(list((Path(directory) / "data").iterdir())), 3)

    def test_missing_or_empty_logs_do_not_create_output_files(self):
        simulation = simulation_mpc()
        simulation.current_name = "MPC_cc_ecs_rectangle_maze"
        optimizer = SimpleNamespace(
            get_constraint_log_fields=lambda: ["stage"],
            get_constraint_log_rows=lambda: [],
        )
        simulation.robot = SimpleNamespace(
            _controller=SimpleNamespace(_optimizer=optimizer),
        )
        with tempfile.TemporaryDirectory() as directory:
            simulation.output_root_dir = directory
            simulation.save_cc_ecs_history_to_csv()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_example_config_has_four_isolated_ecs_runs(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "config" / "config_nmpc_cc_ecs_rectangle_all4.yaml"
        )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(
            [entry["maze_type"] for entry in config["test_configs"]],
            ["s_path", "straight_corridor", "maze", "oblique_maze"],
        )
        for entry in config["test_configs"]:
            self.assertEqual(entry["optimizer_type"], "cc_ecs")
            self.assertEqual(entry["robot_shape"], "rectangle")
            self.assertEqual(entry["dynamics_type"], "differential_drive")
            self.assertEqual(entry["path_planner"], "astar")
        self.assertFalse(config["defaults"]["generate_animation"])
        self.assertFalse(config["defaults"]["generate_plots"])
        self.assertNotIn("cc_mf", config)


if __name__ == "__main__":
    unittest.main()
