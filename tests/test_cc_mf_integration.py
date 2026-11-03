"""Shared MPCC factory settings and canonical CC-MF log exports."""

import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from control import DEFAULT_OPTIMIZER_TYPE
from control.cc_ecs_optimizer import CCECSOptimizerParam
from control.cc_mf_optimizer import CCMFOptimizerParam
from control.cc_sf_optimizer import CCSFOptimizerParam
from control.mpcc_optimizer import MPCCOptimizerParam
from sim.simulation_mpc import simulation_mpc
from test_nmpc import build_output_name, build_parser


class _FactoryComplete(Exception):
    """Stop at the controller boundary before starting navigation."""


class CCMFIntegrationTest(unittest.TestCase):
    def test_factories_apply_shared_settings_and_only_their_method_block_before_planning(self):
        controllers = (
            ("mpcc", "MPCCOptimizer", MPCCOptimizerParam),
            ("cc_sf", "CCSFOptimizer", CCSFOptimizerParam),
            ("cc_ecs", "CCECSOptimizer", CCECSOptimizerParam),
            ("cc_mf", "CCMFOptimizer", CCMFOptimizerParam),
        )
        for method, class_name, param_type in controllers:
            with self.subTest(controller=method):
                simulation = simulation_mpc()
                optimizer = SimpleNamespace(reset=Mock())
                robot = Mock()
                spline = object()
                pose = np.zeros(3)
                environment = (
                    pose, np.array([1., 0., 0.]), (((0, 0), (2, 2)), .1), [],
                )
                config = {
                    name: {"horizon": 99, "vmin": -99., "chance_epsilon": .4}
                    for name, _, _ in controllers
                    if name != "mpcc"
                }
                config["mpcc"] = {
                    "horizon": 3, "tf": .3, "d_min": .006,
                    "a_max": .25, "robust_horizon": 3, "chance_epsilon": .12,
                    "mat_Qe": [[2., 0.], [0., 12.]],
                }
                if method != "mpcc":
                    config[method] = {}
                if method == "cc_mf":
                    config[method]["mf_slack_linear"] = 42.
                elif method == "cc_sf":
                    config[method]["risk_margin_cap"] = 1.3
                elif method == "cc_ecs":
                    config[method]["debug_mode"] = True
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
                    patch(f"control.{method}_optimizer.{class_name}",
                          return_value=optimizer),
                    patch("sim.simulation_mpc.BaseController",
                          side_effect=_FactoryComplete) as controller,
                    patch("sim.simulation_mpc.SingleAgentSimulation") as navigation,
                ):
                    with self.assertRaises(_FactoryComplete):
                        simulation.mpc_test(
                            "maze", "rectangle", optimizer_type=method, config=config,
                        )
                selected, params = controller.call_args.args
                self.assertIs(selected, optimizer)
                self.assertIsInstance(params, param_type)
                self.assertIsInstance(params, MPCCOptimizerParam)
                self.assertEqual(params.horizon, 3)
                self.assertAlmostEqual(params.tf, .3)
                self.assertAlmostEqual(params.d_min, .006)
                self.assertAlmostEqual(params.a_max, .25)
                np.testing.assert_array_equal(params.mat_Qe, [[2., 0.], [0., 12.]])
                self.assertIsInstance(params.mat_Qe, np.ndarray)
                self.assertAlmostEqual(params.vmin, param_type().vmin)
                if method != "mpcc":
                    self.assertAlmostEqual(params.chance_epsilon, .12)
                    self.assertEqual(params.robust_horizon, 3)
                else:
                    self.assertFalse(hasattr(params, "chance_epsilon"))
                if method == "cc_mf":
                    self.assertAlmostEqual(params.mf_slack_linear, 42.)
                elif method == "cc_sf":
                    self.assertAlmostEqual(params.risk_margin_cap, 1.3)
                elif method == "cc_ecs":
                    self.assertTrue(params.debug_mode)
                self.assertEqual(planner.call_args.kwargs["num_horizon"], 3)
                self.assertAlmostEqual(planner.call_args.kwargs["local_path_timestep"], .1)
                robot.set_local_planner.assert_called_with(spline)
                optimizer.reset.assert_called_once_with()
                navigation.assert_not_called()

    def test_cli_and_output_names_use_canonical_mf(self):
        args = build_parser().parse_args(["--single", "--optimizer-type", "cc_mf"])
        args.output_suffix = "trial7"
        name = build_output_name(args)
        self.assertEqual(name, "MPC_cc_mf_rectangle_s_path_trial7")
        self.assertEqual(simulation_mpc()._normalize_animation_method(name), "cc_mf")
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
        simulation.current_name = "MPC_cc_mf_rectangle_maze"
        with tempfile.TemporaryDirectory() as directory:
            simulation.output_root_dir = directory
            simulation.save_cc_mf_history_to_csv()
            for name, row in samples.items():
                filename = "constraints" if name == "constraint" else name
                path = Path(directory) / "data" / (
                    f"cc_mf_{filename}_{simulation.current_name}.csv"
                )
                with path.open(newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(reader.fieldnames, list(row))
                    actual = list(reader)
                self.assertEqual(actual, [{key: str(value) for key, value in row.items()}])


if __name__ == "__main__":
    unittest.main()
