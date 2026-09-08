import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from models.dd import DifferentialDriveMultipleGeometry, DifferentialDriveRectangleGeometry
from models.geometry_utils import RectangleRegion
from sim.simulation_mpc import simulation_mpc
from sim.start_perturbation import (
    convex_polygon_clearance,
    resolve_start_perturbation,
    sample_start_pose,
)
from test_nmpc import apply_start_perturbation_options, build_parser, run_single_test


@contextlib.contextmanager
def working_directory(path):
    previous = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def scenario_geometry(maze_type, robot_shape):
    """Capture the actual production footprint/environment before optimizer setup."""
    captured = {}

    class GeometryCaptured(Exception):
        pass

    def capture(pose, geometry, obstacles, bounds, settings):
        captured.update(pose=pose, geometry=geometry, obstacles=obstacles, bounds=bounds)
        raise GeometryCaptured

    with patch("sim.simulation_mpc.sample_start_pose", side_effect=capture):
        with contextlib.suppress(GeometryCaptured):
            simulation_mpc().mpc_test(maze_type, robot_shape)
    return captured


class StartPerturbationTest(unittest.TestCase):
    def setUp(self):
        self.geometry = DifferentialDriveMultipleGeometry()
        self.geometry.add_geometry(DifferentialDriveRectangleGeometry(0.15, 0.09, 0.0))
        self.bounds = ((0.0, 0.0), (1.0, 1.0))

    def test_disabled_is_exact_noop_without_geometry_or_rng(self):
        nominal = np.array([0.3, 0.2, 4.0])
        with patch("numpy.random.default_rng", side_effect=AssertionError("RNG used")):
            sampled, info = sample_start_pose(nominal, None, None, None)
        np.testing.assert_array_equal(sampled, nominal)
        self.assertIsNot(sampled, nominal)
        self.assertIsNone(info)

    def test_distance_handles_contact_containment_crossings_and_corner_gaps(self):
        box = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        self.assertAlmostEqual(convex_polygon_clearance(box, box + [2., 2.]), np.sqrt(2))
        self.assertAlmostEqual(convex_polygon_clearance(box, box + [1.3, 0.]), 0.3)
        self.assertEqual(convex_polygon_clearance(box, box + [1., 0.]), 0.0)
        self.assertEqual(convex_polygon_clearance(box, 0.2 * box + [0.4, 0.4]), 0.0)
        horizontal = np.array([[-2., -.1], [2., -.1], [2., .1], [-2., .1]])
        self.assertEqual(convex_polygon_clearance(horizontal, horizontal[:, ::-1]), 0.0)

    def test_seed_reproduces_pose_independently_of_global_rng(self):
        settings = {"enabled": True, "seed": 42, "heading_std_deg": 1.0}
        first, info = sample_start_pose([.5, .5, 0.], self.geometry, [], self.bounds, settings)
        # Simulate unrelated localization/planner draws without altering global state.
        state = np.random.get_state()
        try:
            np.random.seed(781)
            np.random.normal(size=100)
            second, _ = sample_start_pose([.5, .5, 0.], self.geometry, [], self.bounds, settings)
        finally:
            np.random.set_state(state)
        np.testing.assert_array_equal(first, second)
        third, _ = sample_start_pose(
            [.5, .5, 0.], self.geometry, [], self.bounds, {**settings, "seed": 43}
        )
        self.assertFalse(np.array_equal(first, third))
        self.assertEqual(info["seed"], 42)

    def test_rejects_gaussian_tail_and_unsafe_footprint_before_accepting(self):
        # Center is outside the obstacle, but the second sample's footprint overlaps it.
        obstacle = RectangleRegion(.425, .8, .2, .8)
        with patch("numpy.random.default_rng") as generator:
            generator.return_value.normal.side_effect = [
                np.array([4., 0., 0.]), np.array([2., 0., 0.]), np.array([-.5, 0., 0.]),
            ]
            pose, info = sample_start_pose(
                [.3, .5, 0.], self.geometry, [obstacle], self.bounds,
                {"enabled": True, "seed": 0, "position_std": .04},
            )
        self.assertEqual(info["attempts"], 3)
        self.assertAlmostEqual(pose[0], .28)
        self.assertGreaterEqual(info["initial_clearance"], .01)

    def test_fails_explicitly_when_no_safe_start_exists(self):
        with self.assertRaisesRegex(ValueError, "Could not sample a safe start in 3 attempts"):
            sample_start_pose(
                [.5, .5, 0.], self.geometry, [RectangleRegion(0., 1., 0., 1.)],
                self.bounds, {"enabled": True, "seed": 0, "max_attempts": 3},
            )

    def test_default_samples_safe_and_bounded_in_all_production_scenarios(self):
        for maze in ("s_path", "straight_corridor", "parking", "maze", "oblique_maze"):
            for shape in ("rectangle", "triangle", "pentagon"):
                case = scenario_geometry(maze, shape)
                for seed in range(32):
                    with self.subTest(maze=maze, shape=shape, seed=seed):
                        pose, info = sample_start_pose(
                            case["pose"], case["geometry"], case["obstacles"], case["bounds"],
                            {"enabled": True, "seed": seed},
                        )
                        self.assertGreaterEqual(info["initial_clearance"], .01)
                        self.assertTrue(np.all(np.abs(pose[:2] - case["pose"][:2]) <= .015))
                        self.assertEqual(pose[2], case["pose"][2])

    def test_heading_noise_wraps_and_map_boundary_is_checked(self):
        pose, info = sample_start_pose(
            [.09, .5, np.pi], self.geometry, [], self.bounds,
            {"enabled": True, "seed": 4, "heading_std_deg": 2.0},
        )
        self.assertTrue(-np.pi <= pose[2] < np.pi)
        self.assertGreaterEqual(info["initial_clearance"], .01)
        self.assertLessEqual(abs(info["delta_pose"][2]), np.deg2rad(6.0))

    def test_invalid_parameters_are_rejected(self):
        for settings in (
            {"seed": -1}, {"seed": 1.5}, {"max_attempts": 0}, {"max_attempts": 1.5},
            {"position_std": -1}, {"heading_std_deg": float("nan")},
            {"max_sigma": 0}, {"min_clearance": float("inf")}, {"enabled": "false"},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                resolve_start_perturbation(settings)

    def test_cli_overrides_yaml_without_mutating_it(self):
        config = {
            "start_perturbation": {"enabled": True, "position_std": .002},
            "test_configs": {"start_perturbation": {"enabled": True, "seed": 99}},
        }
        original = copy.deepcopy(config)
        args = build_parser().parse_args(["--no-perturb-start", "--start-seed", "7"])
        runtime = apply_start_perturbation_options(config, args)
        self.assertEqual(config, original)
        self.assertFalse(runtime["start_perturbation"]["enabled"])
        self.assertEqual(runtime["start_perturbation"]["position_std"], .002)
        self.assertEqual(runtime["test_configs"]["start_perturbation"], {"enabled": False, "seed": 7})

    def test_config_batch_shares_generated_seed_and_honors_per_test_seed(self):
        seen = []

        def fake_run(sim, *args):
            seen.append(args[-1]["start_perturbation"])
            return [], []

        config = {
            "start_perturbation": {"enabled": True},
            "defaults": {"generate_animation": False, "generate_plots": False},
            "test_configs": [
                {"optimizer_type": "psdf"}, {"optimizer_type": "dcbf"},
                {"optimizer_type": "acados", "start_perturbation": {"seed": 12}},
            ],
        }
        with patch.object(simulation_mpc, "mpc_test", fake_run), contextlib.redirect_stdout(io.StringIO()):
            simulation_mpc().run_tests_from_config(config)
        self.assertIsInstance(seen[0]["seed"], int)
        self.assertEqual(seen[0]["seed"], seen[1]["seed"])
        self.assertEqual(seen[2]["seed"], 12)

    def test_single_run_passes_perturbed_pose_to_plant_and_saves_metadata(self):
        from sim.simulation import SingleAgentSimulation

        actual_initial_poses = []

        def fake_navigation(sim, duration):
            robot = sim._robot
            actual_initial_poses.append(robot._system.get_state().copy())
            robot._controller._optimizer.solver_times = [0.001]
            return {"status": "test"}

        # Optimizer reset/cleanup touches code-generation files in the cwd.
        with tempfile.TemporaryDirectory() as output_dir, working_directory(output_dir):
            for method in ("psdf", "dcbf", "acados", "obca"):
                args = build_parser().parse_args([
                    "--single", "--maze-type", "maze", "--optimizer-type", method,
                    "--perturb-start", "--start-seed", "42", "--no-animation", "--no-plots",
                ])
                with patch.object(SingleAgentSimulation, "run_navigation", fake_navigation):
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = run_single_test(args, {"output_root_dir": output_dir})
                info = result["start_perturbation"]
                np.testing.assert_array_equal(actual_initial_poses[-1], info["initial_pose"])
                self.assertTrue(result["current_name"].endswith("_startseed42"))
                path = Path(output_dir) / "data" / f"start_pose_{result['current_name']}.json"
                stored = json.loads(path.read_text())
                self.assertEqual(stored["initial_pose"], info["initial_pose"])
                self.assertEqual(stored["seed"], 42)
                if method == "obca":
                    self.assertTrue(result["current_name"].startswith("MPC_OBCA_"))
        np.testing.assert_array_equal(actual_initial_poses[0], actual_initial_poses[1])
        np.testing.assert_array_equal(actual_initial_poses[0], actual_initial_poses[2])


if __name__ == "__main__":
    unittest.main()
