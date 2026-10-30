#!/usr/bin/env python3
"""Sequential rectangle-width benchmark with complete control-cycle timing.

Run with the project's acados/CUDA Python environment. Source files and resolved
parameters are preserved with the raw results; report generation is separate.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
METHODS = ("mpcc", "cc_sf", "cc_ecs", "cc_mf")
WIDTHS = (0.100, 0.105, 0.110, 0.115, 0.120)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def capture_run_log(log):
    """Capture Python and native compiler/solver output in the same run log."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    try:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        with redirect_stdout(log), redirect_stderr(log):
            yield
    finally:
        log.flush()
        for descriptor, backup in zip((1, 2), saved):
            os.dup2(backup, descriptor)
            os.close(backup)


def runtime_environment():
    acados = Path(os.environ.get("ACADOS_SOURCE_DIR", ROOT.parent / "acados"))
    library = str(acados / "lib")
    paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if library not in paths:
        environment = dict(os.environ)
        environment["ACADOS_SOURCE_DIR"] = str(acados)
        environment["LD_LIBRARY_PATH"] = ":".join([library, *filter(None, paths)])
        os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)


def snapshot_sources(destination):
    snapshot = destination / "source_snapshot"
    sources = [ROOT / name for name in ("AGENTS.md", "test_nmpc.py",
                                        "run_width_benchmark.py")]
    sources.extend(ROOT / name for name in (
        "run_deadlock_width_benchmark.py", "analyze_deadlock_width_benchmark.py",
        "recover_width_benchmark_animations.py",
    ) if (ROOT / name).is_file())
    for directory in ("control", "models", "planning", "sim", "utils", "config"):
        sources.extend(path for path in (ROOT / directory).rglob("*")
                       if path.is_file() and path.suffix in (".py", ".yaml")
                       and "__pycache__" not in path.parts)
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(destination / "source_hashes.json", hashes)
    return snapshot


def run_benchmark(output, trials, *, resume=False, generate_animation=False,
                  only_width=None, only_method=None, only_seed=None,
                  widths=WIDTHS, current_defaults=False):
    if resume:
        snapshot = output / "source_snapshot"
        if not snapshot.is_dir():
            raise FileNotFoundError(f"Missing preserved sources: {snapshot}")
    else:
        output.mkdir(parents=True, exist_ok=False)
        snapshot = snapshot_sources(output)
    # Freeze controller implementations even if the working tree changes later.
    sys.path.insert(0, str(snapshot))
    import casadi
    import matplotlib
    import numpy as np
    import shapely
    import torch
    import yaml
    from shapely.geometry import Polygon

    from sim.simulation import Robot
    from sim.simulation_mpc import simulation_mpc
    from sim.trial_metrics import collect_successful_solver_steps
    from control.cc_covariance import ZERO_INITIAL_STD
    from test_nmpc import collect_optimizer_runtime_stats, cleanup_simulation

    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the current MPCC implementations")

    def serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [serializable(item) for item in value]
        return value

    original_local = Robot.run_local_planner
    original_control = Robot.run_controller
    original_system = Robot.run_system
    checkpoint = None

    def timed_local(robot):
        if checkpoint is not None and not getattr(robot, "_benchmark_checkpoint_started", False):
            checkpoint.write(json.dumps({"initial_pose": serializable(robot._system.get_state()),
                "global_path": serializable(robot._global_path)}) + "\n")
            robot._benchmark_checkpoint_started = True
        torch.cuda.synchronize()
        robot._benchmark_cycle_started = time.perf_counter()
        return original_local(robot)

    def timed_control(robot, obstacles):
        try:
            return original_control(robot, obstacles)
        finally:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - robot._benchmark_cycle_started
            if not hasattr(robot, "_benchmark_cycle_times"):
                robot._benchmark_cycle_times = []
                robot._benchmark_cycle_modes = []
            robot._benchmark_cycle_times.append(elapsed)
            optimizer = robot._controller._optimizer
            robot._benchmark_cycle_modes.append(getattr(optimizer, "_last_solve_mode", "normal"))

    def checkpointed_system(robot):
        result = original_system(robot)
        if checkpoint is not None:
            local = getattr(robot._local_planner_logger, "_trajs", [])
            predicted = getattr(robot._controller_logger, "_xtrajs", [])
            checkpoint.write(json.dumps({"pose": serializable(robot._system.get_state()),
                "time_s": float(robot._system._time),
                "local_path": serializable(local[-1]) if local else None,
                "optimized_traj": serializable(predicted[-1]) if predicted else None}) + "\n")
        return result

    Robot.run_local_planner = timed_local
    Robot.run_controller = timed_control
    Robot.run_system = checkpointed_system

    manifest = {
        "started_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "widths_m": widths, "methods": METHODS, "trials_per_configuration": trials,
        "seeds": list(range(trials)), "python": sys.version,
        "initial_sigmas": {"sigma_f0": 0.0002, "sigma_l0": 0.0, "sigma_psi0": 0.0},
        "zero_sigma_numerical_std": ZERO_INITIAL_STD,
        "platform": platform.platform(), "torch": torch.__version__,
        "casadi": casadi.__version__, "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(), "torch_threads": torch.get_num_threads(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "timing": "perf_counter + CUDA synchronization; local planner through "
                  "controller and controller logging; first cycle excluded "
                  "from aggregates; all later calls included regardless of solver status",
        "clearance": "exact polygon distance, zero at overlap/contact, "
                     "minimum with signed map-boundary margin; initial and every post-step pose",
        "current_defaults": current_defaults,
        "crash_checkpoints": current_defaults,
        "output_suffix": "widthbench_" + datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S"),
        "animation_directory": str(ROOT / "animations"),
    }
    if current_defaults:
        from control.mpcc_optimizer import MPCCOptimizerParam
        from control.cc_sf_optimizer import CCSFOptimizerParam
        from control.cc_ecs_optimizer import CCECSOptimizerParam
        from control.cc_mf_optimizer import CCMFOptimizerParam
        parameter_types = (MPCCOptimizerParam, CCSFOptimizerParam,
                           CCECSOptimizerParam, CCMFOptimizerParam)
        manifest["resolved_defaults"] = {
            method: serializable(vars(param_type()))
            for method, param_type in zip(METHODS, parameter_types)
        }
        manifest["initial_sigmas"] = {
            method: {field: params[field] for field in ("sigma_f0", "sigma_l0", "sigma_psi0")}
            for method, params in manifest["resolved_defaults"].items() if method != "mpcc"
        }
    if resume:
        manifest = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
        manifest["animation_directory"] = str(ROOT / "animations")
        manifest["remaining_trial_process_isolation"] = True
        harness_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest["continuation_harness_sha256"] = harness_hash
    write_json(output / "experiment.json", manifest)
    all_trials = []
    if resume:
        with (output / "trial_results.csv").open(newline="", encoding="utf-8") as handle:
            all_trials = list(csv.DictReader(handle))
    if generate_animation:
        manifest.setdefault("animation_from_trial", len(all_trials) + 1)
        write_json(output / "experiment.json", manifest)
    recorded = {(float(row["width_m"]), row["method"], int(row["seed"])) for row in all_trials}
    widths = tuple(manifest["widths_m"])
    total = len(widths) * len(METHODS) * trials
    for width in widths:
        # Rotate controller order between seeds, keeping execution sequential.
        for seed in range(trials):
            order = METHODS[seed % len(METHODS):] + METHODS[:seed % len(METHODS)]
            for method in order:
                if (width, method, seed) in recorded:
                    continue
                if only_width is not None and width != only_width:
                    continue
                if only_method is not None and method != only_method:
                    continue
                if only_seed is not None and seed != only_seed:
                    continue
                run_dir = output / f"width_{round(width * 1000):03d}mm" / method / f"seed_{seed}"
                run_dir.mkdir(parents=True)
                config = {
                    "rectangle_geometry": {"length": 0.15, "width": width},
                    "start_perturbation": {"enabled": True, "seed": seed,
                        "position_std": 0.02, "heading_std_deg": float(np.rad2deg(0.05)),
                        "max_sigma": 3.0, "min_clearance": manifest.get(
                            "min_clearance_by_width_m", {}).get(f"{width:.3f}", 0.01),
                        "max_attempts": 1000},
                    "localization_error": {"enabled": False},
                    "trial_failure_criteria": {"enabled": True,
                        "movement_window_sec": 4.0, "movement_threshold": 0.03,
                        "max_consecutive_solver_failures": 20},
                    "defaults": {"simulation_time": 60.0, "generate_animation": generate_animation,
                                 "generate_plots": False},
                    "output_root_dir": str(run_dir),
                    "output_suffix": f"width{round(width * 1000)}mm_seed{seed}",
                }
                if method != "mpcc" and not manifest.get("current_defaults", False):
                    config[method] = {"sigma_f0": 0.0002, "sigma_l0": 0.0,
                                      "sigma_psi0": 0.0}
                overrides = manifest.get("optimizer_overrides", {}).get(method)
                if overrides is not None:
                    config[method] = dict(overrides)
                if manifest.get("output_suffix"):
                    config["output_suffix"] += "_" + manifest["output_suffix"]
                (run_dir / "config.yaml").write_text(
                    yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                print(f"START {len(all_trials) + 1}/{total} width={width:.3f} "
                      f"method={method} seed={seed}", flush=True)
                simulation = simulation_mpc()
                if manifest.get("crash_checkpoints", False):
                    checkpoint = (run_dir / "animation_checkpoint.jsonl").open(
                        "w", encoding="utf-8", buffering=1)
                outcome = None
                error = ""
                started = time.perf_counter()
                interrupted = False
                with (run_dir / "run.log").open("w", encoding="utf-8", buffering=1) as log:
                    with capture_run_log(log):
                        try:
                            simulation.mpc_test("maze", "rectangle", method,
                                "differential_drive", "astar", 60.0, config)
                            outcome = dict(simulation.last_run_outcome or {})
                            if generate_animation:
                                def animation_path(name, method_name=None):
                                    directory = ROOT / "animations" / method
                                    directory.mkdir(parents=True, exist_ok=True)
                                    return str(directory / f"{name}.mp4")
                                simulation._get_animation_output_path = animation_path
                                name = simulation.current_name
                                simulation.animate_world(simulation.sim, animation_name=name,
                                    maze_type="maze", method_name=method, return_html=False,
                                    frame_skip=manifest.get("animation_frame_skip", 1))
                                outcome["animation_path"] = animation_path(name)
                        except KeyboardInterrupt:
                            interrupted = True
                            outcome = {"status": "interrupted", "failure_reason": "keyboard_interrupt"}
                        except Exception as exc:
                            traceback.print_exc()
                            error = f"{type(exc).__name__}: {exc}"
                            outcome = dict(simulation.last_run_outcome or {})
                            outcome.setdefault("status", "error")
                            outcome.setdefault("failure_reason", error)
                        finally:
                            robot = getattr(simulation, "robot", None)
                            optimizer = getattr(getattr(robot, "_controller", None), "_optimizer", None)
                            stats = collect_optimizer_runtime_stats(simulation)
                            if robot is not None and hasattr(robot, "_system"):
                                outcome.setdefault("final_time", float(robot._system._time))
                            if simulation.sim is not None:
                                simulation.save_trial_history_to_csv("trial_history")
                            params = getattr(getattr(robot, "_controller", None), "_param", None)
                            if params is not None:
                                write_json(run_dir / "optimizer_parameters.json", serializable(vars(params)))
                            cycles = []
                            times = getattr(robot, "_benchmark_cycle_times", [])
                            statuses = getattr(getattr(robot, "_controller_logger", None), "_solver_status_infos", [])
                            controller_times = getattr(getattr(robot, "_controller_logger", None), "_computation_times", [])
                            for index, elapsed in enumerate(times):
                                status = statuses[index] if index < len(statuses) else {}
                                cycles.append({"step": index, "cycle_time_s": elapsed,
                                    "included_in_aggregate": int(index > 0),
                                    "solve_mode": robot._benchmark_cycle_modes[index],
                                    "controller_time_s": controller_times[index],
                                    "solver_success": status.get("success"),
                                    "solver_status": status.get("raw_status"),
                                    "solver_status_code": status.get("status_code")})
                            write_csv(run_dir / "cycles.csv", cycles)
                            successful_solver_steps = collect_successful_solver_steps(simulation)
                            write_csv(run_dir / "successful_solver_steps.csv", successful_solver_steps)
                            states = list(getattr(getattr(robot, "_system_logger", None), "_xs", []))
                            initial = getattr(simulation, "initial_pose", None)
                            poses = ([np.asarray(initial)] if initial is not None else []) + states
                            samples = []
                            if poses:
                                _, _, grid, obstacles = simulation.create_env("maze")
                                obstacle_shapes = [Polygon(item.get_ccw_vertices()) for item in obstacles]
                                half_length, half_width = 0.075, width / 2
                                local = np.array([[-half_length, -half_width],
                                    [half_length, -half_width], [half_length, half_width],
                                    [-half_length, half_width]])
                                lower, upper = np.asarray(grid[0], dtype=float)
                                for index, pose in enumerate(poses):
                                    pose = np.asarray(pose, dtype=float).reshape(-1)[:3]
                                    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
                                    rotation = np.array([[cosine, -sine], [sine, cosine]])
                                    vertices = local @ rotation.T + pose[:2]
                                    polygon = Polygon(vertices)
                                    clearance = min(float(shapely.distance(polygon, obstacle_shapes).min()),
                                        float((vertices - lower).min()), float((upper - vertices).min()))
                                    samples.append({"pose_index": index, "simulation_time_s": index * 0.1,
                                        "x": float(pose[0]), "y": float(pose[1]), "theta": float(pose[2]),
                                        "clearance_m": clearance})
                            write_csv(run_dir / "clearance.csv", samples)
                            outcome["initial_pose"] = serializable(initial)
                            outcome["start_perturbation"] = serializable(simulation.start_perturbation_info)
                            outcome["runtime_stats"] = serializable(stats)
                            outcome["error"] = error
                            write_json(run_dir / "outcome.json", serializable(outcome))
                            cleanup_simulation(simulation)
                if checkpoint is not None:
                    checkpoint.close()
                    checkpoint = None
                row = {"width_m": width, "method": method, "trial": seed + 1,
                    "seed": seed, "success": int(outcome.get("status") == "success"),
                    "status": outcome.get("status"), "failure_reason": outcome.get("failure_reason"),
                    "final_time_s": outcome.get("final_time"),
                    "safe_stop_count": stats.get("safe_stop_count", 0),
                    "safe_stop_supported": method != "mpcc",
                    "steps": len(times), "clearance_samples": len(samples),
                    "wall_time_s": time.perf_counter() - started,
                    "run_dir": str(run_dir.relative_to(output)), "error": error}
                all_trials.append(row)
                write_csv(output / "trial_results.csv", all_trials)
                write_json(output / "progress.json", {"completed": len(all_trials), "total": total, "last": row})
                print(f"DONE {len(all_trials)}/{total} {method} w={width:.3f} seed={seed} "
                      f"{row['status']} t={row['final_time_s']} stop={row['safe_stop_count']} "
                      f"wall={row['wall_time_s']:.1f}s reason={row['failure_reason']}", flush=True)
                del optimizer, robot, simulation
                gc.collect()
                torch.cuda.empty_cache()
                matplotlib.pyplot.close("all")
                if interrupted:
                    return
    if len(all_trials) == total:
        manifest["finished_at_kst"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
        write_json(output / "experiment.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generate-animation", action="store_true")
    parser.add_argument("--only-width", type=float)
    parser.add_argument("--only-method", choices=METHODS)
    parser.add_argument("--only-seed", type=int)
    parser.add_argument("--widths", type=float, nargs="+", default=WIDTHS)
    parser.add_argument("--current-defaults", action="store_true",
                        help="Keep current controller defaults, including initial covariance")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    if any(not 0 < width < float("inf") for width in args.widths):
        parser.error("--widths must be finite positive values")
    runtime_environment()
    run_benchmark(args.results_dir.resolve(), args.trials, resume=args.resume,
                  generate_animation=args.generate_animation,
                  only_width=args.only_width, only_method=args.only_method,
                  only_seed=args.only_seed, widths=tuple(args.widths),
                  current_defaults=args.current_defaults)


if __name__ == "__main__":
    main()
