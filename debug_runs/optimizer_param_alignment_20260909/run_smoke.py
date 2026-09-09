"""Run the existing test_nmpc single-test API with raw diagnostics preserved."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

RUN_ROOT = Path(__file__).resolve().parent
PROJECT = RUN_ROOT.parents[1]
if len(sys.argv) == 1:
    for method in ("psdf", "dcbf", "obca"):
        folder = RUN_ROOT / method
        folder.mkdir(exist_ok=True)
        env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   ACADOS_SOURCE_DIR="/home/taek111/projects/acados",
                   LD_LIBRARY_PATH="/home/taek111/projects/acados/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))
        print("START", method, flush=True)
        with (folder / "run.log").open("w") as log:
            completed = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), method], cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
        print("END", method, "exit", completed.returncode, flush=True)
    sys.exit(0)

method = sys.argv[1]
folder = RUN_ROOT / method
sys.path.insert(0, str(PROJECT))
# Python 3.8 contains the existing solver dependencies. Borrow only the
# stdlib BooleanOptionalAction from the installed Python 3.9, without
# changing project source or any control/simulation behavior.
if not hasattr(argparse, "BooleanOptionalAction"):
    spec = importlib.util.spec_from_file_location("argparse_compat39", "/usr/lib/python3.9/argparse.py")
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    argparse.BooleanOptionalAction = compat.BooleanOptionalAction

import numpy as np
import yaml
import test_nmpc

config = yaml.safe_load((PROJECT / "config/config.yaml").read_text())
config["output_root_dir"] = str(folder)
config["output_suffix"] = "20260909"
config["start_perturbation"]["enabled"] = False
(folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
cli = ["--single", "--config", str(folder / "config.yaml"), "--maze-type", "maze", "--robot-shape", "rectangle", "--dynamics-type", "differential_drive", "--optimizer-type", method, "--path-planner", "astar", "--simulation-time", "0.3", "--no-perturb-start", "--no-animation", "--no-plots"]
args = test_nmpc.build_parser().parse_args(cli)

captured = {}
def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)

original_cleanup = test_nmpc.cleanup_simulation
def capture_and_cleanup(sim):
    try:
        if sim.robot is not None:
            robot = sim.robot
            optimizer = robot._controller._optimizer
            captured.update(optimizer_class=type(optimizer).__name__, optimizer_module=type(optimizer).__module__, initial_pose=sim.initial_pose, final_state=robot._system.get_state(), final_time=robot._system._time,
                            last_solver_status=robot._controller.get_last_solver_status_info(), parameters=vars(robot._controller._param),
                            outcome=sim.last_run_outcome, global_path=getattr(robot, "_global_path", None))
            arrays = dict(states=np.asarray(robot._system_logger._xs), inputs=np.asarray(robot._system_logger._us),
                          predicted_states=np.asarray(robot._controller_logger._xtrajs), predicted_inputs=np.asarray(robot._controller_logger._utrajs),
                          references=np.asarray(robot._local_planner_logger._trajs), controller_times=np.asarray(robot._controller_logger._computation_times),
                          solver_times=np.asarray(optimizer.solver_times))
            np.savez_compressed(folder / "diagnostics.npz", **arrays)
            captured["solver_statuses"] = robot._controller_logger._solver_status_infos
            if sim.last_run_outcome is None:
                sim.save_trial_history_to_csv("partial_trial_history")
    except Exception:
        captured["capture_error"] = traceback.format_exc()
    finally:
        (folder / "diagnostics.json").write_text(json.dumps(captured, indent=2, default=json_default))
        original_cleanup(sim)
test_nmpc.cleanup_simulation = capture_and_cleanup
started = time.perf_counter()
print("COMMAND: python3 test_nmpc.py", " ".join(cli), flush=True)
print("RUNTIME:", sys.version, flush=True)
result = {}
exit_code = 0
try:
    result = test_nmpc.run_single_test(args, config=config) or {}
except Exception as exc:
    traceback.print_exc()
    result = dict(status="exception", failure_reason=str(exc), exception_type=type(exc).__name__)
    exit_code = 1
finally:
    result.update(wall_seconds=time.perf_counter() - started, method=method, cli=cli, source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip())
    (folder / "result.json").write_text(json.dumps(result, indent=2, default=json_default))
    print("RESULT:", json.dumps(result, default=json_default), flush=True)
sys.exit(exit_code)
