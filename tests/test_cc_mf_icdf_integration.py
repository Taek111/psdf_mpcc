"""Inverse-CDF factory isolation and exported log contracts."""

import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from control import DEFAULT_OPTIMIZER_TYPE
from control.cc_mf_icdf_optimizer import CCMFICDFOptimizerParam
from sim.simulation_mpc import simulation_mpc
from test_nmpc import build_output_name, build_parser


class _FactoryComplete(Exception):
    """Stop at the controller boundary before starting navigation."""


class CCMFICDFIntegrationTest(unittest.TestCase):
    def test_factory_uses_only_icdf_overrides_before_building_planner(self):
        simulation = simulation_mpc()
        optimizer = SimpleNamespace(reset=Mock())
        robot = Mock()
        spline = object()
        pose = np.zeros(3)
        environment = (
            pose, np.array([1., 0., 0.]), (((0, 0), (2, 2)), .1), [],
        )
        config = {
            "cc_mf": {"horizon": 99, "chance_epsilon": .9, "vmin": -99.},
            "cc_mf_imp": {"horizon": 88, "mf_slack_linear": 999.},
            "cc_mf_icdf": {"horizon": 3, "tf": .3, "d_min": .006},
        }
        with (
            patch.object(simulation, "create_env", return_value=environment),
            patch("sim.simulation_mpc.sample_start_pose", return_value=(pose, None)),
            patch("sim.simulation_mpc.DifferentialDriveSystem",
                  return_value=SimpleNamespace(_dt=.1)),
            patch("sim.simulation_mpc.Robot", return_value=robot),
            patch("sim.simulation_mpc.AstarLoSPathGenerator"),
            patch("sim.simulation_mpc.ConstantSpeedTrajectoryGenerator"),
            patch("sim.simulation_mpc.SplineReferenceGenerator",
                  return_value=spline) as planner,
            patch("control.cc_mf_icdf_optimizer.CCMFICDFOptimizer",
                  return_value=optimizer),
            patch("sim.simulation_mpc.BaseController",
                  side_effect=_FactoryComplete) as controller,
            patch("sim.simulation_mpc.SingleAgentSimulation") as navigation,
        ):
            with self.assertRaises(_FactoryComplete):
                simulation.mpc_test(
                    "maze", "rectangle", optimizer_type="cc_mf_icdf", config=config,
                )
        selected, params = controller.call_args.args
        self.assertIs(selected, optimizer)
        self.assertIsInstance(params, CCMFICDFOptimizerParam)
        self.assertEqual(params.horizon, 3)
        self.assertAlmostEqual(params.tf, .3)
        self.assertAlmostEqual(params.d_min, .006)
        self.assertAlmostEqual(params.chance_epsilon, .2)
        self.assertAlmostEqual(params.vmin, -.7)
        self.assertAlmostEqual(params.mf_slack_linear, 100.)
        self.assertEqual(planner.call_args.kwargs["num_horizon"], 3)
        self.assertAlmostEqual(planner.call_args.kwargs["local_path_timestep"], .1)
        robot.set_local_planner.assert_called_with(spline)
        optimizer.reset.assert_called_once_with()
        navigation.assert_not_called()

    def test_cli_and_output_names_keep_icdf_distinct_from_existing_mf(self):
        args = build_parser().parse_args(["--single", "--optimizer-type", "cc_mf_icdf"])
        args.output_suffix = "trial7"
        name = build_output_name(args)
        self.assertEqual(name, "MPC_cc_mf_icdf_rectangle_s_path_trial7")
        self.assertEqual(simulation_mpc()._normalize_animation_method(name), "cc_mf_icdf")
        self.assertEqual(build_parser().parse_args([]).optimizer_type, DEFAULT_OPTIMIZER_TYPE)
        self.assertEqual(DEFAULT_OPTIMIZER_TYPE, "psdf")

    def test_csv_export_retains_probability_and_quantile_as_distinct_fields(self):
        optimizer = SimpleNamespace()
        samples = {
            "constraint": {
                "probability_sum": .12, "residual": .3,
                "residual_unit": "standard_normal_quantile", "tail_clipped": False,
            },
            "covariance": {"stage": 1, "sigma_f": .02},
            "cycle": {"solver_status": 0, "slack_unit": "standard_normal_quantile"},
        }
        for name, row in samples.items():
            setattr(optimizer, f"get_{name}_log_fields", Mock(return_value=list(row)))
            setattr(optimizer, f"get_{name}_log_rows", Mock(return_value=[row]))
        simulation = simulation_mpc()
        simulation.robot = SimpleNamespace(_controller=SimpleNamespace(_optimizer=optimizer))
        simulation.current_name = "MPC_cc_mf_icdf_rectangle_maze"
        with tempfile.TemporaryDirectory() as directory:
            simulation.output_root_dir = directory
            simulation.save_cc_mf_icdf_history_to_csv()
            for name, row in samples.items():
                filename = "constraints" if name == "constraint" else name
                path = Path(directory) / "data" / (
                    f"cc_mf_icdf_{filename}_{simulation.current_name}.csv"
                )
                with path.open(newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(reader.fieldnames, list(row))
                    actual = list(reader)
                self.assertEqual(actual, [{key: str(value) for key, value in row.items()}])

    def test_example_is_independent_and_disables_visualization(self):
        path = Path(__file__).resolve().parents[1] / "config/config_cc_mf_icdf.yaml"
        config = yaml.safe_load(path.read_text())
        self.assertNotIn("cc_mf", config)
        self.assertNotIn("cc_mf_imp", config)
        self.assertEqual(config["test_configs"][0]["optimizer_type"], "cc_mf_icdf")
        self.assertFalse(config["defaults"]["generate_animation"])
        self.assertFalse(config["defaults"]["generate_plots"])


if __name__ == "__main__":
    unittest.main()
