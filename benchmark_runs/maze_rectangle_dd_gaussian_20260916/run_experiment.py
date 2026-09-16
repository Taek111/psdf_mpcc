"""Paired five-seed benchmark through test_nmpc.run_single_test.

WSL: /usr/bin/python3 run_experiment.py --distribution safe --workers 6
The repository's controller, planner, and termination logic are unchanged.
"""

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
METHODS = ("dcbf", "obca", "psdf")
SEEDS = tuple(range(42, 47))
SOURCE_FILES = (
    "test_nmpc.py", "sim/simulation_mpc.py", "sim/simulation.py",
    "sim/start_perturbation.py", "control/dcbf_optimizer.py",
    "control/obca_optimizer.py", "control/psdf_optimizer.py",
    "control/controller.py", "models/dd.py", "models/psdf.py",
    "models/psdf_wrapper.py", "planning/path_generator/search_path_generator.py",
    "planning/trajectory_generator/constant_speed_generator.py", "config/config.yaml",
)


def source_hashes():
    return {p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()
            for p in SOURCE_FILES}


def write_json(path, data):
    def convert(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        return str(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=convert), encoding="utf-8")
    temporary.replace(path)


def runtime_environment():
    env = dict(os.environ)
    env.update(MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               ACADOS_SOURCE_DIR="/home/taek111/projects/acados")
    env["LD_LIBRARY_PATH"] = "/home/taek111/projects/acados/lib:" + env.get("LD_LIBRARY_PATH", "")
    return env


def run_worker(options):
    # Avoid duplicate evaluation when an additional worker starts a queued trial.
    import fcntl
    folder = ROOT / options.distribution / options.method / ("seed%d" % options.seed)
    folder.mkdir(parents=True, exist_ok=True)
    trial_lock = (folder / "trial.lock").open("a")
    fcntl.flock(trial_lock, fcntl.LOCK_EX)
    if (folder / "result.json").exists():
        print("ALREADY COMPLETE", options.method, options.seed, flush=True)
        return
    sys.path.insert(0, str(PROJECT))
    # Installed solver packages use Python 3.8; the CLI needs Python 3.9's action.
    if not hasattr(argparse, "BooleanOptionalAction"):
        spec = importlib.util.spec_from_file_location(
            "argparse_compat310", "/home/taek111/anaconda3/envs/pp/lib/python3.10/argparse.py")
        compat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compat)
        argparse.BooleanOptionalAction = compat.BooleanOptionalAction

    import numpy as np
    import torch
    import yaml
    import casadi
    import test_nmpc
    from control.controller import BaseController
    from sim.start_perturbation import sample_start_pose, pose_clearance
    from models.dd import DifferentialDriveMultipleGeometry, DifferentialDriveRectangleGeometry

    torch.set_num_threads(1)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    folder = ROOT / options.distribution / options.method / ("seed%d" % options.seed)
    folder.mkdir(parents=True, exist_ok=True)
    os.chdir(folder)
    started = time.perf_counter()
    before_hashes = source_hashes()
    config = yaml.safe_load((PROJECT / "config/config.yaml").read_text())
    config.update(output_root_dir=str(folder), output_suffix="gaussian_20260916",
                  localization_error={"enabled": False})
    config["test_configs"] = [dict(maze_type="maze", robot_shape="rectangle",
                                  optimizer_type=options.method,
                                  dynamics_type="differential_drive", path_planner="astar")]
    config["defaults"].update(simulation_time=60.0, generate_animation=False,
                              generate_plots=False)
    config["start_perturbation"] = dict(
        enabled=True, seed=options.seed, position_std=0.02,
        heading_std_deg=float(np.rad2deg(0.05)), max_sigma=3.0,
        min_clearance=0.01, max_attempts=1000)
    config["benchmark_sampling"] = options.distribution
    (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    cli = ["--single", "--config", str(folder / "config.yaml"), "--maze-type", "maze",
           "--robot-shape", "rectangle", "--dynamics-type", "differential_drive",
           "--optimizer-type", options.method, "--path-planner", "astar",
           "--simulation-time", "60", "--no-animation", "--no-plots"]
    args = test_nmpc.build_parser().parse_args(cli)
    sim = test_nmpc.simulation_mpc()
    nominal, goal, grid, obstacles = sim.create_env("maze")
    geometry = DifferentialDriveMultipleGeometry()
    geometry.add_geometry(DifferentialDriveRectangleGeometry(length=.15, width=.09, rear_dist=0.0))
    footprints = [g.get_ccw_vertices() for g in geometry.equiv_rep()]
    obstacle_polygons = [o.get_ccw_vertices() for o in obstacles]
    raw_delta = np.random.default_rng(options.seed).normal(size=3) * [.02, .02, .05]
    raw_pose = nominal + raw_delta
    raw_clearance = pose_clearance(raw_pose, footprints, obstacle_polygons, grid[0])
    if options.distribution == "safe":
        initial_pose, start_info = sample_start_pose(
            nominal, geometry, obstacles, grid[0], config["start_perturbation"])
    else:
        initial_pose = raw_pose.copy()
        initial_pose[2] = (initial_pose[2] + np.pi) % (2 * np.pi) - np.pi
        start_info = dict(config["start_perturbation"], nominal_pose=nominal,
                          initial_pose=initial_pose, delta_pose=raw_delta,
                          initial_clearance=raw_clearance, attempts=1,
                          sampling="untruncated_unconditioned_gaussian",
                          max_sigma=None, min_clearance=None)
        # Process-local sampling adapter; no controller or repository source changes.
        module = sys.modules[test_nmpc.simulation_mpc.__module__]
        module.sample_start_pose = lambda *a, **kw: (initial_pose.copy(), dict(start_info))
    write_json(folder / "initial_pose.json", dict(
        accepted=start_info, raw_pose=raw_pose, raw_delta=raw_delta, raw_clearance=raw_clearance))

    original_generate = BaseController.generate_control_input
    def generate_with_progress(self, system, global_path, reference, obstacles):
        write_json(folder / "progress.json", dict(
            simulation_time=float(system._time), state=system.get_state(),
            last_solver_status=self.get_last_solver_status_info(),
            wall_seconds=time.perf_counter() - started))
        return original_generate(self, system, global_path, reference, obstacles)
    BaseController.generate_control_input = generate_with_progress

    captured = {}
    original_cleanup = test_nmpc.cleanup_simulation
    def capture_and_cleanup(test_sim):
        try:
            robot = test_sim.robot
            if robot is not None:
                optimizer = robot._controller._optimizer
                states = np.asarray([test_sim.initial_pose, *robot._system_logger._xs], dtype=float)
                inputs = np.asarray(robot._system_logger._us, dtype=float).reshape(-1, 2)
                clearances = np.asarray([
                    pose_clearance(x, footprints, obstacle_polygons, grid[0]) for x in states])
                statuses = robot._controller_logger._solver_status_infos
                captured.update(
                    final_pose=robot._system.get_state(), completed_steps=len(inputs),
                    captured_final_time=float(robot._system._time),
                    minimum_sampled_clearance=float(clearances.min()),
                    contact_or_collision_samples=int(np.count_nonzero(clearances <= 1e-9)),
                    solver_failure_steps=sum(s.get("success") is False for s in statuses),
                    optimizer_class=type(optimizer).__name__, parameters=vars(robot._controller._param),
                    global_path=robot._global_path,
                    goal_pose=goal, goal_angle_checked=(robot._global_path.shape[1] >= 3))
                np.savez_compressed(folder / "trajectory.npz", states=states, inputs=inputs,
                                    clearance=clearances, times=np.arange(len(states)) * .1)
                if test_sim.last_run_outcome is None:
                    test_sim.save_trial_history_to_csv("partial_trial_history")
        except Exception:
            captured["capture_error"] = traceback.format_exc()
        finally:
            write_json(folder / "diagnostics.json", captured)
            original_cleanup(test_sim)
    test_nmpc.cleanup_simulation = capture_and_cleanup
    result = {}
    try:
        print("COMMAND: python3 test_nmpc.py", " ".join(cli), flush=True)
        if options.distribution == "raw" and raw_clearance <= 0:
            result = dict(status="failure", failure_reason="initial_collision_or_out_of_bounds",
                          goal_reached=False, final_time=0.0, start_perturbation=start_info)
            captured.update(completed_steps=0, minimum_sampled_clearance=raw_clearance,
                            contact_or_collision_samples=1)
        else:
            result = test_nmpc.run_single_test(args, config=config) or {}
    except Exception as exc:
        traceback.print_exc()
        result = dict(status="exception", failure_reason=str(exc),
                      exception_type=type(exc).__name__, goal_reached=False)
    finally:
        result.update(captured)
        result.update(method=options.method, seed=options.seed, trial=options.seed - 41,
                      distribution=options.distribution, initial_pose=initial_pose,
                      start_perturbation=start_info, wall_seconds=time.perf_counter()-started,
                      cli=cli, python=sys.version, numpy=np.__version__, torch=torch.__version__,
                      casadi=casadi.__version__, source_sha256=before_hashes,
                      source_unchanged=before_hashes == source_hashes())
        result["navigation_success"] = result.get("status") == "success"
        result["collision_free_success"] = (result["navigation_success"] and
                                             result.get("contact_or_collision_samples") == 0)
        write_json(folder / "result.json", result)
        print("RESULT", options.method, options.seed, result["status"],
              result.get("failure_reason"), flush=True)


def run_batch(options):
    output = ROOT / options.distribution
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", dict(
        seeds=SEEDS, methods=METHODS, standard_deviation=[.02, .02, .05],
        units=["m", "m", "rad"], distribution=options.distribution,
        simulation_time=60., planner="astar", source_sha256=source_hashes(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=PROJECT, text=True),
        workers=options.workers))
    def launch(method, seed):
        folder = output / method / ("seed%d" % seed)
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "result.json").exists():
            return method, seed, "already_complete"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--method", method,
                   "--seed", str(seed), "--distribution", options.distribution]
        with (folder / "run.log").open("w") as log:
            process = subprocess.Popen(command, cwd=folder, env=runtime_environment(),
                                       stdout=log, stderr=subprocess.STDOUT)
            write_json(folder / "process.json", dict(pid=process.pid, command=command))
            print("START", method, seed, "PID", process.pid, flush=True)
            return method, seed, process.wait()
    with concurrent.futures.ThreadPoolExecutor(max_workers=options.workers) as pool:
        futures = [pool.submit(launch, method, seed) for seed in SEEDS for method in METHODS]
        for future in concurrent.futures.as_completed(futures):
            print("END", *future.result(), flush=True)
    print("ALL TRIALS FINISHED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", choices=("raw", "safe"), required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    options = parser.parse_args()
    if options.method:
        run_worker(options)
    else:
        run_batch(options)
