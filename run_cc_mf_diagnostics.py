"""Run sequential CC-MF trials from preserved sources with passive telemetry.

Use the project's CUDA/acados Python environment. Each invocation requires a
fresh output directory; controller behavior is unchanged by the observations.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_width_benchmark import capture_run_log, runtime_environment, snapshot_sources


ROOT = Path(__file__).resolve().parent


def plain(value):
    """Serialize NumPy values and explicitly mark unavailable numbers as null."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/config_benchmark.yaml")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--simulation-time", type=float, default=60.0)
    parser.add_argument("--movement-window", type=float, default=10.0)
    parser.add_argument(
        "--preserve-config", action="store_true",
        help="Keep all YAML scenario, start-pose, termination and output settings; "
             "only add diagnostics and select the requested trial count.",
    )
    parser.add_argument("--rti-passes", type=int, default=1,
                        help="Experimental repeated relinearization per control cycle (default: 1)")
    args = parser.parse_args()
    if args.rti_passes < 1 or args.trials < 1:
        parser.error("--trials and --rti-passes must be positive")
    runtime_environment()
    output = args.results_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = snapshot_sources(output)
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    sys.path.insert(0, str(snapshot))

    import numpy as np
    import torch
    import yaml

    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
    from sim.simulation_mpc import simulation_mpc, DifferentialDriveRectangleGeometry
    from test_nmpc import main as run_trials

    config = yaml.safe_load(args.config.read_text())
    original_config = plain(config)
    config["trials"] = args.trials
    if args.preserve_config:
        expected = {
            "maze_type": "maze", "robot_shape": "rectangle", "optimizer_type": "cc_mf",
            "dynamics_type": "differential_drive", "path_planner": "astar",
        }
        scenarios = config.get("test_configs", [])
        if len(scenarios) != 1 or any(scenarios[0].get(k) != v for k, v in expected.items()):
            raise ValueError("Preserved diagnostic config must contain one CC-MF/DD/rectangle/maze/A* scenario")
    else:
        config["test_configs"] = [{
            "maze_type": "maze", "robot_shape": "rectangle", "optimizer_type": "cc_mf",
            "dynamics_type": "differential_drive", "path_planner": "astar",
        }]
    config.setdefault("rectangle_geometry", {"length": 0.15, "width": 0.127})
    if not args.preserve_config:
        config["start_perturbation"] = {
            "enabled": True, "seed": args.start_seed, "position_std": 0.02,
            "heading_std_deg": 2.8647889756541165, "max_sigma": 3.0,
            "min_clearance": 0.01, "max_attempts": 1000,
        }
        config["localization_error"] = {"enabled": False}
        config.setdefault("defaults", {}).update(
            simulation_time=args.simulation_time, generate_animation=False, generate_plots=False,
        )
        config["trial_failure_criteria"] = {
            "enabled": True, "movement_window_sec": args.movement_window,
            "movement_threshold": 0.03, "max_consecutive_solver_failures": 20,
        }
    config.setdefault("cc_mf", {}).update(debug_mf=True, debug_infeasibility=True)
    config["diagnostic_refinement"] = {"rti_passes": args.rti_passes}
    config_path = output / "diagnostic_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2(args.config, output / "input_config.yaml")
    write_json(output / "config_changes.json", {
        key: {"original": original_config.get(key), "diagnostic": config.get(key)}
        for key in original_config.keys() | config.keys()
        if original_config.get(key) != config.get(key)
    })
    parameters = CCMFOptimizerParam()
    for key, value in config["cc_mf"].items():
        if not hasattr(parameters, key):
            raise ValueError(f"Unknown CC-MF parameter: {key}")
        setattr(parameters, key, value)
    write_json(output / "resolved_cc_mf_parameters.json", vars(parameters))
    write_json(output / "environment.json", {
        "python": sys.executable, "python_version": sys.version, "torch": torch.__version__,
        "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
        "ACADOS_SOURCE_DIR": os.environ.get("ACADOS_SOURCE_DIR"),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "baseline_config": str(args.config.resolve()), "cli": sys.argv,
        "trials": args.trials,
        "start_seed": config.get("start_perturbation", {}).get("seed"),
        "preserve_config": args.preserve_config,
        "source_root": str(snapshot),
        "note": "Post-solve observations only. Controller latency includes diagnostics and JSON writes; solver timing excludes them.",
    })
    (output / "working_tree_status.txt").write_text(subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True,
    ))
    start, goal, grid, obstacles = simulation_mpc().create_env("maze")
    dimensions = config["rectangle_geometry"]
    geometry = DifferentialDriveRectangleGeometry(dimensions["length"], dimensions["width"], 0.0)
    write_json(output / "environment_geometry.json", {
        "environment": "maze", "bounds": grid[0], "start": start, "goal": goal,
        "footprint_length_m": dimensions["length"], "footprint_width_m": dimensions["width"],
        "footprint_vertices_local": geometry._region.get_ccw_vertices(),
        "obstacles": [obstacle.get_ccw_vertices() for obstacle in obstacles],
    })

    original_prepare = CCMFOptimizer._prepare_and_solve
    original_finish = CCMFOptimizer._finish_solve
    instance_ids, nominal_inputs, attempts = {}, {}, {}

    def prepare_with_telemetry(optimizer, solver):
        attempts[optimizer] = []
        for pass_number in range(args.rti_passes):
            nominal_inputs[optimizer] = np.asarray([
                solver.get(stage, "u") for stage in range(optimizer.N)
            ], dtype=float).copy()
            elapsed_before = optimizer._cycle_solver_time
            status = original_prepare(optimizer, solver)
            attempts[optimizer].append({
                "pass": pass_number + 1, "status": status,
                "solver_time_s": optimizer._cycle_solver_time - elapsed_before,
            })
            if status is None or status != 0:
                break
        return status

    def finish_with_telemetry(optimizer, mode, status, solver, diagnostics):
        original_finish(optimizer, mode, status, solver, diagnostics)
        if optimizer not in instance_ids:
            instance_ids[optimizer] = len(instance_ids) + 1
            write_json(output / f"path_trial{instance_ids[optimizer]:02d}.json", optimizer.reference_path_data)
        states, inputs = optimizer._read_solver_trajectory(solver)
        record = {
            "time": optimizer._get_cycle_log_time(), "solve_step": optimizer._solve_count,
            "mode": mode, "status": status, "main_status": optimizer._last_main_status,
            "pose": optimizer.state._x, "current_s0": optimizer._current_s0,
            "states": states, "inputs": inputs, "nominal_inputs": nominal_inputs.get(optimizer),
            "rti_attempts": attempts.get(optimizer),
            "nominal": {key: (optimizer._constraint_data or {}).get(key) for key in (
                "nominal_state", "phi", "guard_A", "guard_c", "mf_A", "mf_c", "mf_mask",
                "residual", "probability_sum", "branch", "status", "tail_clipped",
            )},
            "diagnostics": diagnostics,
            "failed_diagnostics": optimizer._last_failed_constraint_diagnostics,
        }
        path = output / f"telemetry_trial{instance_ids[optimizer]:02d}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(plain(record), allow_nan=False) + "\n")

    CCMFOptimizer._prepare_and_solve = prepare_with_telemetry
    CCMFOptimizer._finish_solve = finish_with_telemetry
    sys.argv = ["test_nmpc.py", "-c", str(config_path), "--trials", str(args.trials),
                "--results-dir", str(output / "results")]
    if not args.preserve_config:
        sys.argv.extend(["--no-animation", "--no-plots"])
    print(f"Running {args.trials} sequential trials; detailed log: {output / 'run.log'}", flush=True)
    hashes = json.loads((output / "source_hashes.json").read_text())
    try:
        with (output / "run.log").open("w", buffering=1) as log, capture_run_log(log):
            run_trials()
    finally:
        write_json(output / "source_changed_during_batch.json", [
            key for key, value in hashes.items() if not (ROOT / key).exists()
            or hashlib.sha256((ROOT / key).read_bytes()).hexdigest() != value
        ])
        CCMFOptimizer._prepare_and_solve = original_prepare
        CCMFOptimizer._finish_solve = original_finish
    print((output / "results/summary.csv").read_text(), flush=True)


if __name__ == "__main__":
    main()
