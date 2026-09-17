import contextlib
import copy
import csv
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from control.controller import BaseController
from sim.simulation import Robot, SingleAgentSimulation
from sim.simulation_mpc import TrialFailureChecker, simulation_mpc
from sim.trial_metrics import TrialMetricsRecorder, collect_successful_solver_steps
from test_nmpc import build_parser, run_single_test, run_trial_batch


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def identity(config_id="config01", method="psdf"):
    return {
        "configuration_id": config_id, "maze_type": "maze",
        "optimizer_type": method, "robot_shape": "rectangle",
        "dynamics_type": "differential_drive", "path_planner": "astar",
    }


def set_simulation_state(sim, times, successes):
    statuses = [
        {"success": success, "raw_status": str(success), "status_code": None}
        for success in successes
    ]
    optimizer = SimpleNamespace(solver_times=list(times))
    optimizer.cleanup = Mock(side_effect=optimizer.solver_times.clear)
    robot = SimpleNamespace(
        _controller=SimpleNamespace(_optimizer=optimizer),
        _controller_logger=SimpleNamespace(_solver_status_infos=statuses),
        _system=SimpleNamespace(_dt=0.1, _time=len(times) * 0.1),
    )
    sim.robot = robot
    sim.sim = SimpleNamespace(_robot=robot)
    sim.initial_pose = np.array([0.2, 0.3, 0.0])
    sim.current_name = "test_run"
    return optimizer


class TrialMetricsTest(unittest.TestCase):
    def test_only_successful_finite_step_times_are_collected(self):
        sim = simulation_mpc()
        set_simulation_state(
            sim,
            [0.01, 20.0, 0.03, float("nan"), float("inf"), -1, 0, 0.8, 7],
            [True, False, None, True, True, True, True, True],
        )
        steps = collect_successful_solver_steps(sim)
        self.assertEqual([row["timestep"] for row in steps], [0, 6, 7])
        self.assertEqual([row["solver_computation_time_s"] for row in steps], [0.01, 0, 0.8])
        self.assertAlmostEqual(steps[-1]["simulation_time_s"], 0.8)
        self.assertEqual(collect_successful_solver_steps(simulation_mpc()), [])

    def test_incremental_csvs_and_pooled_statistics_include_steps_from_failed_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrialMetricsRecorder(directory)
            sim = simulation_mpc()
            set_simulation_state(sim, [0.01], [True])
            recorder.record(identity(), 1, {"enabled": True, "seed": 7}, {
                "status": "success", "goal_reached": True,
                "initial_pose": sim.initial_pose,
                "successful_solver_steps": collect_successful_solver_steps(sim),
            })
            self.assertEqual(len(read_rows(recorder.trial_path)), 1)
            self.assertEqual(read_rows(recorder.summary_path)[0]["success_rate"], "1.0")

            set_simulation_state(sim, [0.03, 0.05, 0.07, 100], [True, True, True, False])
            recorder.record(identity(), 2, {"enabled": True, "seed": 8}, {
                "status": "error", "failure_reason": "solver failed,\nsecond line",
                "successful_solver_steps": collect_successful_solver_steps(sim),
            })
            trials = read_rows(recorder.trial_path)
            summary = read_rows(recorder.summary_path)[0]
            self.assertEqual(len(trials), 2)
            self.assertEqual(trials[1]["failure_reason"], "solver failed,\nsecond line")
            self.assertEqual(summary["successful_trials"], "1")
            self.assertEqual(summary["failed_trials"], "1")
            self.assertEqual(summary["successful_solver_steps"], "4")
            self.assertAlmostEqual(float(summary["success_rate"]), 0.5)
            self.assertAlmostEqual(float(summary["solver_time_mean_s"]), 0.04)
            self.assertAlmostEqual(float(summary["solver_time_p95_s"]), 0.067)
            timing_path = next(Path(directory).glob("computation_times_*.csv"))
            timings = read_rows(timing_path)
            self.assertEqual(len(timings), 4)
            self.assertEqual(timings[1]["trial_success"], "0")
            self.assertEqual(timings[1]["start_seed"], "8")
            self.assertEqual(timing_path.read_text().count("configuration_id,"), 1)

    def test_empty_failed_trial_and_interruption_have_no_fabricated_times(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrialMetricsRecorder(directory)
            recorder.record(identity(), 1, {}, {"status": "error"})
            recorder.record(identity(), 2, {}, {"status": "interrupted"})
            summary = read_rows(recorder.summary_path)[0]
            self.assertEqual(summary["recorded_trials"], "2")
            self.assertEqual(summary["completed_trials"], "1")
            self.assertEqual(summary["interrupted_trials"], "1")
            self.assertEqual(summary["success_rate"], "0.0")
            self.assertEqual(summary["solver_time_mean_s"], "")
            self.assertEqual(summary["solver_time_p95_s"], "")
            self.assertEqual(read_rows(next(Path(directory).glob("computation_times_*.csv"))), [])
            original = recorder.trial_path.read_bytes()
            with self.assertRaises(FileExistsError):
                TrialMetricsRecorder(directory)
            self.assertEqual(recorder.trial_path.read_bytes(), original)

    def test_distinct_configurations_are_not_merged(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrialMetricsRecorder(directory)
            recorder.record(identity(), 1, {}, {"status": "success"})
            recorder.record(identity("config02"), 1, {}, {"status": "timeout"})
            self.assertEqual(len(list(Path(directory).glob("computation_times_*.csv"))), 2)
            self.assertEqual([r["success_rate"] for r in read_rows(recorder.summary_path)], ["1.0", "0.0"])

    def test_casadi_reported_success_and_exception_override_stale_status(self):
        controller = BaseController(SimpleNamespace(), None)
        controller._opt_sol = SimpleNamespace(stats=lambda: {
            "return_status": "Solved_To_Acceptable_Level", "success": True,
        })
        controller._update_last_solver_status_info()
        self.assertIs(controller.get_last_solver_status_info()["success"], True)
        controller._optimizer.solver = SimpleNamespace(status=0)
        controller._update_last_solver_status_info(exc=RuntimeError("failed"))
        self.assertIs(controller.get_last_solver_status_info()["success"], False)


class GoalReachedTest(unittest.TestCase):
    @staticmethod
    def robot(state, goal=(0.0, 0.0, 0.0)):
        robot = Robot(SimpleNamespace(get_state=lambda: np.asarray(state)))
        robot._global_path = [np.asarray(goal)]
        return robot

    def test_l1_position_threshold_includes_boundary_and_ignores_heading_by_default(self):
        for state, expected in (
            ((0.012, -0.008, np.pi), True),  # L1 = 20 mm.
            ((0.02, 0.0, np.pi), True),
            ((0.015, 0.01, 0.0), False),  # L2 < 20 mm, L1 > 20 mm.
            ((0.020001, 0.0, 0.0), False),
        ):
            with self.subTest(state=state):
                self.assertEqual(self.robot(state).is_goal_reached(), expected)

    def test_explicit_angle_tolerance_is_preserved(self):
        robot = self.robot((0.005, 0.005, np.pi))
        self.assertTrue(robot.is_goal_reached())
        self.assertFalse(robot.is_goal_reached(angle_tolerance=0.2))

    def test_logged_distance_uses_same_planner_goal_and_l1_metric(self):
        robot = self.robot((0.015, 0.01, 0.0), goal=(10.0, 10.0))
        robot._global_planner = SimpleNamespace(
            get_path_poses=lambda: [np.zeros(3)],
        )
        simulation = SingleAgentSimulation(robot, [], np.zeros(3))
        self.assertAlmostEqual(simulation._distance_to_goal(), 0.025)
        self.assertFalse(robot.is_goal_reached())
        robot._global_path = []
        self.assertFalse(robot.is_goal_reached())

    def test_success_criteria_defaults_to_20mm_without_angle_condition(self):
        criteria = simulation_mpc()._build_success_criteria({})
        self.assertEqual(criteria["position_tolerance"], 0.02)
        self.assertIsNone(criteria["angle_tolerance"])


class TrialFailureCheckerTest(unittest.TestCase):
    def test_stuck_detection_keeps_boundary_sample_after_repeated_time_steps(self):
        checker = TrialFailureChecker(4.0, 0.03, 20)
        system = SimpleNamespace(_time=0.0)
        robot = SimpleNamespace(
            _system=system,
            _get_navigation_state=lambda: np.zeros(3),
        )
        simulation = SimpleNamespace(_robot=robot)

        for _ in range(40):
            system._time += 0.1
            self.assertIsNone(checker(simulation))

        system._time += 0.1
        failure = checker(simulation)
        self.assertIsNotNone(failure)
        self.assertIn("stuck_for_4.00s", failure["reason"])


class TrialBatchTest(unittest.TestCase):
    @staticmethod
    def config():
        return {
            "defaults": {"generate_animation": False, "generate_plots": False, "simulation_time": 0.3},
            "start_perturbation": {"enabled": True, "seed": 10},
            "test_configs": [
                {"maze_type": maze, "optimizer_type": method, "robot_shape": "rectangle"}
                for maze in ("maze", "oblique_maze") for method in ("psdf", "dcbf", "obca")
            ],
        }

    def test_two_trials_for_all_six_configs_preserve_partial_failure_and_paired_seeds(self):
        seen = []
        optimizers = []

        def fake_test(sim, **kwargs):
            seen.append(copy.deepcopy(kwargs))
            optimizers.append(set_simulation_state(sim, [0.01, 9.0, 0.03], [True, False, True]))
            sim.output_root_dir = kwargs["config"]["output_root_dir"]
            if len(seen) == 2:
                raise RuntimeError("solver exception after earlier successful steps")
            sim.last_run_outcome = {"status": "success", "goal_reached": True}

        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(["--trials", "2", "--results-dir", directory])
            config = self.config()
            original = copy.deepcopy(config)
            with patch.object(simulation_mpc, "mpc_test", fake_test), contextlib.redirect_stdout(io.StringIO()):
                rows = run_trial_batch(args, config)
            self.assertEqual(config, original)
            self.assertEqual(len(rows), 12)
            self.assertEqual([run["config"]["start_perturbation"]["seed"] for run in seen], [10] * 6 + [11] * 6)
            self.assertEqual(len({run["config"]["output_root_dir"] for run in seen}), 12)
            self.assertEqual([row["successful_solver_steps"] for row in rows], [2] * 12)
            self.assertEqual(rows[1]["status"], "error")
            self.assertIn("solver exception", rows[1]["failure_reason"])
            self.assertEqual(len(read_rows(Path(directory) / "trial_results.csv")), 12)
            timing_paths = list(Path(directory).glob("computation_times_*.csv"))
            self.assertEqual(len(timing_paths), 6)
            self.assertTrue(all(len(read_rows(path)) == 4 for path in timing_paths))
            for optimizer in optimizers:
                optimizer.cleanup.assert_called_once()
                self.assertEqual(optimizer.solver_times, [])

    def test_interrupted_trial_is_saved_and_stops_whole_batch(self):
        calls = []

        def fake_test(sim, **kwargs):
            calls.append(kwargs)
            set_simulation_state(sim, [0.02], [True])
            if len(calls) == 2:
                raise KeyboardInterrupt
            sim.last_run_outcome = {"status": "success"}

        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(["--trials", "100", "--results-dir", directory])
            with patch.object(simulation_mpc, "mpc_test", fake_test), contextlib.redirect_stdout(io.StringIO()):
                rows = run_trial_batch(args, self.config())
            self.assertEqual(len(calls), 2)
            self.assertEqual(rows[-1]["status"], "interrupted")
            self.assertEqual(rows[-1]["successful_solver_steps"], 1)
            self.assertEqual(len(read_rows(Path(directory) / "trial_results.csv")), 2)
            summary = read_rows(Path(directory) / "summary.csv")[-1]
            self.assertEqual(summary["completed_trials"], "0")
            self.assertEqual(summary["success_rate"], "")

    def test_psdf_consecutive_failure_limit_resets_on_success_and_batch_continues(self):
        from control.psdf_optimizer import PSDFOptimizer

        solvers = []

        def fake_test(sim, **kwargs):
            optimizer = PSDFOptimizer()
            optimizer.ped_model = None
            optimizer.N = 1
            optimizer.setup = Mock()
            solver = Mock(status=0)
            # The first trial recovers after 19 failures, then fails 20 times.
            # The next trial recovers from one failure and reaches the goal.
            statuses = iter([3] * 19 + [0] + [3] * 20 if not solvers else [3, 0])

            def fake_solve():
                solver.status = next(statuses)
                return solver.status

            solver.solve.side_effect = fake_solve
            solver.get.side_effect = lambda stage, field: np.zeros(3 if field == "x" else 2)
            optimizer.solver = solver
            solvers.append(solver)

            system = SimpleNamespace(
                _dt=0.1, _time=0.0, get_state=lambda: np.zeros(3), logging=Mock(),
            )

            def update_system(control_input):
                system._time += system._dt

            system.update = update_system
            robot = Robot(system)
            robot.set_controller(BaseController(optimizer, None))
            robot._global_path = [np.array([1.0, 1.0])]
            robot._local_trajectory = []
            robot.run_global_planner = Mock()
            robot.run_local_planner = Mock()
            robot.is_goal_reached = Mock(return_value=False)
            if len(solvers) == 2:
                robot.is_goal_reached.side_effect = [False, True]
            sim.robot = robot
            sim.sim = SingleAgentSimulation(
                robot, [], np.array([1.0, 1.0, 0.0]),
                failure_checker=sim._build_trial_failure_checker(kwargs["config"]),
            )
            sim.last_run_outcome = sim.sim.run_navigation(kwargs["simulation_time"])

        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(["--trials", "2", "--results-dir", directory])
            config = self.config()
            config["test_configs"] = config["test_configs"][:1]
            config["defaults"]["simulation_time"] = 6.0
            config["trial_failure_criteria"] = {
                "enabled": True, "movement_window_sec": 4.0,
                "movement_threshold": 0.03, "max_consecutive_solver_failures": 20,
            }
            with patch.object(simulation_mpc, "mpc_test", fake_test), contextlib.redirect_stdout(io.StringIO()):
                rows = run_trial_batch(args, config)

            self.assertEqual([row["status"] for row in rows], ["failure", "success"])
            self.assertEqual([row["success"] for row in rows], [0, 1])
            self.assertIn("consecutive_solver_failures(20)", rows[0]["failure_reason"])
            self.assertEqual([row["successful_solver_steps"] for row in rows], [1, 1])
            self.assertEqual([solver.solve.call_count for solver in solvers], [40, 2])
            self.assertAlmostEqual(rows[0]["final_time"], 4.0)
            timings = read_rows(next(Path(directory).glob("computation_times_*.csv")))
            self.assertEqual(len(timings), 2)
            self.assertEqual([row["trial_success"] for row in timings], ["0", "1"])
            self.assertEqual(read_rows(Path(directory) / "summary.csv")[0]["success_rate"], "0.5")

    def test_single_config_cli_override_and_disabled_perturbation(self):
        seen = []

        def fake_test(sim, **kwargs):
            seen.append(copy.deepcopy(kwargs))
            sim.last_run_outcome = {"status": "timeout", "failure_reason": "timeout"}

        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args([
                "--single", "--trials", "2", "--optimizer-type", "obca",
                "--no-perturb-start", "--start-seed", "42", "--no-animation", "--no-plots",
                "--results-dir", directory,
            ])
            with patch.object(simulation_mpc, "mpc_test", fake_test), contextlib.redirect_stdout(io.StringIO()):
                rows = run_trial_batch(args, self.config())
            self.assertEqual(len(rows), 2)
            self.assertEqual([run["optimizer_type"] for run in seen], ["obca", "obca"])
            self.assertEqual([row["start_seed"] for row in rows], [42, 43])
            self.assertTrue(all(not row["start_perturbation_enabled"] for row in rows))
            self.assertEqual(len({run["config"]["output_root_dir"] for run in seen}), 2)

    def test_completed_navigation_stays_successful_after_output_error(self):
        def fake_test(sim, **kwargs):
            set_simulation_state(sim, [0.01], [True])
            sim.last_run_outcome = {"status": "success", "failure_reason": None}
            raise OSError("output failed")
        args = build_parser().parse_args(["--single", "--no-animation", "--no-plots"])
        with patch.object(simulation_mpc, "mpc_test", fake_test), contextlib.redirect_stdout(io.StringIO()):
            result = run_single_test(args, {}, collect_metrics=True)
        self.assertEqual(result["status"], "success")
        self.assertIn("output failed", result["error"])

    def test_invalid_trial_count_is_rejected_before_simulation(self):
        args = build_parser().parse_args(["--trials", "0"])
        with patch.object(simulation_mpc, "mpc_test") as simulate:
            with self.assertRaisesRegex(ValueError, "positive integer"):
                run_trial_batch(args, self.config())
            simulate.assert_not_called()


if __name__ == "__main__":
    unittest.main()

