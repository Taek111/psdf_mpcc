"""Run test_nmpc's single-test entry point and retain diagnostic evidence.

Run with the installed WSL solver Python: python3 run_experiment.py
Only the Python 3.8 CLI compatibility shim and logging are added; controller
parameters and simulation termination criteria retain repository defaults.
"""
import argparse
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

if len(sys.argv) == 1:
    processes = []
    for method in ("dcbf",):
        folder = ROOT / method
        folder.mkdir(exist_ok=True)
        env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   ACADOS_SOURCE_DIR="/home/taek111/projects/acados",
                   LD_LIBRARY_PATH="/home/taek111/projects/acados/lib:"
                   + os.environ.get("LD_LIBRARY_PATH", ""))
        log = (folder / "run.log").open("w")
        process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), method],
            cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        processes.append((method, process, log))
        print("START", method, "PID", process.pid, flush=True)
    for method, process, log in processes:
        print("END", method, "exit", process.wait(), flush=True)
        log.close()
    sys.exit(0)

method = sys.argv[1]
folder = ROOT / method
assert method == "dcbf", "This runner is for the DCBF-only retest."
source_file = PROJECT / "control/dcbf_optimizer.py"
source_bytes = source_file.read_bytes()
source_hash = hashlib.sha256(source_bytes).hexdigest()
assert source_bytes == (ROOT / "dcbf_optimizer_snapshot.py").read_bytes()
(ROOT / "source_diff.patch").write_bytes(subprocess.check_output(
    ["git", "diff", "--", "control/dcbf_optimizer.py"], cwd=PROJECT))
sys.path.insert(0, str(PROJECT))
if not hasattr(argparse, "BooleanOptionalAction"):
    spec = importlib.util.spec_from_file_location(
        "argparse_compat310",
        "/home/taek111/anaconda3/envs/pp/lib/python3.10/argparse.py",
    )
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    argparse.BooleanOptionalAction = compat.BooleanOptionalAction

import numpy as np
import torch
import casadi
import yaml
import test_nmpc
from control.controller import BaseController

config = yaml.safe_load((PROJECT / "config/config.yaml").read_text())
config.update(output_root_dir=str(folder), output_suffix="retest_20260910")
config["start_perturbation"]["enabled"] = False
config["test_configs"] = [dict(maze_type="maze", robot_shape="rectangle",
                                optimizer_type=method,
                                dynamics_type="differential_drive",
                                path_planner="astar")]
config["defaults"].update(simulation_time=60.0, generate_animation=False)
(folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
cli = ["--single", "--config", str(folder / "config.yaml"), "--maze-type", "maze",
       "--robot-shape", "rectangle", "--dynamics-type", "differential_drive",
       "--optimizer-type", method, "--path-planner", "astar", "--simulation-time",
       "60", "--no-perturb-start", "--no-animation"]
args = test_nmpc.build_parser().parse_args(cli)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


original_generate = BaseController.generate_control_input


def generate_with_progress(self, system, global_path, reference, obstacles):
    progress = dict(simulation_time=float(system._time), state=system.get_state(),
                    reference=reference, last_status=self.get_last_solver_status_info())
    (folder / "progress.json").write_text(json.dumps(progress, default=json_default))
    return original_generate(self, system, global_path, reference, obstacles)


BaseController.generate_control_input = generate_with_progress
original_cleanup = test_nmpc.cleanup_simulation


def capture_and_cleanup(sim):
    captured = {}
    try:
        if sim.robot is not None:
            robot = sim.robot
            optimizer = robot._controller._optimizer
            captured.update(
                optimizer_module=type(optimizer).__module__,
                initial_pose=sim.initial_pose, final_state=robot._system.get_state(),
                final_time=robot._system._time,
                last_solver_status=robot._controller.get_last_solver_status_info(),
                parameters=vars(robot._controller._param), outcome=sim.last_run_outcome,
                global_path=getattr(robot, "_global_path", None),
                solver_statuses=robot._controller_logger._solver_status_infos,
            )
            np.savez_compressed(
                folder / "diagnostics.npz",
                states=np.asarray(robot._system_logger._xs),
                inputs=np.asarray(robot._system_logger._us),
                predicted_states=np.asarray(robot._controller_logger._xtrajs),
                predicted_inputs=np.asarray(robot._controller_logger._utrajs),
                references=np.asarray(robot._local_planner_logger._trajs),
                controller_times=np.asarray(robot._controller_logger._computation_times),
                solver_times=np.asarray(optimizer.solver_times),
            )
            if sim.last_run_outcome is None:
                sim.save_trial_history_to_csv("partial_trial_history")
    except Exception:
        captured["capture_error"] = traceback.format_exc()
    finally:
        (folder / "diagnostics.json").write_text(
            json.dumps(captured, indent=2, default=json_default))
        original_cleanup(sim)


test_nmpc.cleanup_simulation = capture_and_cleanup
started = time.perf_counter()
print("COMMAND: python3 test_nmpc.py", " ".join(cli), flush=True)
print("RUNTIME:", sys.version, "numpy", np.__version__, "torch", torch.__version__,
      "casadi", casadi.__version__, flush=True)
result = {}
exit_code = 0
try:
    result = test_nmpc.run_single_test(args, config=config) or {}
except Exception as exc:
    traceback.print_exc()
    result = dict(status="exception", failure_reason=str(exc),
                  exception_type=type(exc).__name__)
    exit_code = 1
finally:
    result.update(wall_seconds=time.perf_counter() - started, method=method, cli=cli,
                  optimizer_source_sha256=source_hash,
                  optimizer_source_unchanged_during_run=(
                      hashlib.sha256(source_file.read_bytes()).hexdigest() == source_hash),
                  source_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip())
    (folder / "result.json").write_text(json.dumps(result, indent=2, default=json_default))
    print("RESULT:", json.dumps(result, default=json_default), flush=True)
sys.exit(exit_code)

