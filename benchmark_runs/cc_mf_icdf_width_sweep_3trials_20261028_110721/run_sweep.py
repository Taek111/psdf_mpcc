"""Sequential CC-MF-ICDF width experiment using an immutable source snapshot."""

import argparse
import csv
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = ROOT / "source_snapshot"
WIDTHS = [.105, .110, .115, .120, .125, .130]
STOP_MODES = ("safe_stop", "safe_stop_stage0")


def stamp():
    return dt.datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2,
                              default=lambda value: value.tolist()) + "\n")


def check_sources():
    hashes = json.loads((ROOT / "source_hashes.json").read_text())
    changed = [name for name, digest in hashes.items()
               if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != digest]
    assert not changed, changed
    return changed


def prepare():
    assert not SOURCE.exists(), "Use a new experiment directory."
    SOURCE.mkdir()
    for directory in ("control", "models", "planning", "sim", "utils", "config"):
        for path in (PROJECT / directory).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".yaml"):
                destination = SOURCE / path.relative_to(PROJECT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    for name in ("test_nmpc.py", "method_description.md",
                 "inverse_cdf_mf_residual_revised.md", "AGENTS.md"):
        shutil.copy2(PROJECT / name, SOURCE / name)
    hashes = {str(path.relative_to(SOURCE)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(SOURCE.rglob("*")) if path.is_file()}
    dump(ROOT / "source_hashes.json", hashes)
    dump(ROOT / "experiment.json", {
        "created_at_kst": stamp(), "controller": "cc_mf_icdf",
        "environment": "maze", "shape": "rectangle", "length_m": .15,
        "widths_m": WIDTHS, "trials_per_width": 3, "seeds": [0, 1, 2],
        "initial_clearance_m": .001,
        "initial_condition_reference": "CC-SF width sweep in benchmark_results.md",
        "early_stop": "Skip larger widths after 0/3 goal completions at one width.",
        "source_root": str(SOURCE), "controller_source_edits": False,
    })


def child(width):
    check_sources()
    sys.path.insert(0, str(SOURCE))
    os.chdir(SOURCE)
    import casadi
    import numpy as np
    import torch
    import yaml
    import test_nmpc
    from control.cc_mf_icdf_optimizer import CCMFICDFOptimizer, CCMFICDFOptimizerParam
    from models.dd import DifferentialDriveRectangleGeometry
    from sim.start_perturbation import sample_start_pose

    module = importlib.import_module("sim.simulation_mpc")
    assert Path(module.__file__).resolve().is_relative_to(SOURCE)
    assert Path(test_nmpc.__file__).resolve().is_relative_to(SOURCE)
    assert torch.cuda.is_available() and torch.get_num_threads() == 6
    folder = ROOT / f"width_{round(width * 1000):03d}mm"
    folder.mkdir(exist_ok=False)
    config = yaml.safe_load((SOURCE / "config/config_cc_mf_icdf.yaml").read_text())
    config.update(trials=3, start_perturbation={
        "enabled": True, "seed": 0, "position_std": .02,
        "heading_std_deg": float(np.rad2deg(.05)), "max_sigma": 3.,
        "min_clearance": .001, "max_attempts": 1000,
    })
    (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    # Override only this module's geometry constructor, preserving the actual
    # geometry type and all controller, simulator, and dynamics source files.
    def rectangle_for_run(length, width=None, rear_dist=0.):
        assert length == .15 and rear_dist == 0.
        return DifferentialDriveRectangleGeometry(length, selected_width, rear_dist)

    selected_width = width
    module.DifferentialDriveRectangleGeometry = rectangle_for_run
    footprint = rectangle_for_run(.15)
    start, goal, grid, obstacles = module.simulation_mpc().create_env("maze")
    dump(folder / "geometry.json", {
        "bounds": grid[0], "nominal_start": start, "goal": goal, "dt_s": .1,
        "footprint": footprint.equiv_rep()[0].get_ccw_vertices(),
        "obstacles": [obstacle.get_ccw_vertices() for obstacle in obstacles],
    })
    starts = []
    for seed in range(3):
        pose, details = sample_start_pose(start, footprint, obstacles, grid[0],
                                         {**config["start_perturbation"], "seed": seed})
        starts.append({"seed": seed, "pose": pose, "details": details})
    dump(folder / "expected_start_poses.json", starts)
    params = CCMFICDFOptimizerParam()
    module.simulation_mpc._apply_param_overrides(params, config["cc_mf_icdf"], "CC-MF-ICDF")
    dump(folder / "optimizer_parameters.json", vars(params))

    context = {"seed": None, "cycle": 0}
    original_solve = CCMFICDFOptimizer.solve_nlp
    original_trial = test_nmpc.run_single_test
    with (folder / "cycles.csv").open("w", newline="") as stream:
        fields = ["seed", "cycle", "time_before_s", "mode", "main_status",
                  "backup_status", "safe_stop", "cumulative_safe_stops",
                  "exact_current_phi_m"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        def observe_solve(self):
            solution = original_solve(self)
            data = self._constraint_data or {}
            writer.writerow({
                "seed": context["seed"], "cycle": context["cycle"],
                "time_before_s": context["cycle"] * .1,
                "mode": self._last_solve_mode, "main_status": self._last_main_status,
                "backup_status": self._last_backup_status,
                "safe_stop": int(self._last_solve_mode in STOP_MODES),
                "cumulative_safe_stops": self._safe_stop_count,
                "exact_current_phi_m": data.get("exact_current_phi"),
            })
            context["cycle"] += 1
            return solution

        def observe_trial(trial_args, trial_config=None, **kwargs):
            context.update(seed=trial_args.start_seed, cycle=0)
            result = original_trial(trial_args, trial_config, **kwargs)
            stream.flush()
            dump(folder / f"runtime_seed{context['seed']}.json", {
                key: value for key, value in result.items()
                if key != "successful_solver_steps"
            })
            print(f"WIDTH_RESULT width={width:.3f} seed={context['seed']} "
                  f"status={result.get('status')} safe_stop={result.get('safe_stop_count')}",
                  flush=True)
            return result

        CCMFICDFOptimizer.solve_nlp = observe_solve
        test_nmpc.run_single_test = observe_trial
        metadata = {
            "created_at_kst": stamp(), "width_m": width, "length_m": .15,
            "python": sys.executable, "python_version": sys.version,
            "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "casadi": casadi.__version__,
            "gpu": torch.cuda.get_device_name(0), "torch_num_threads": torch.get_num_threads(),
            "thread_environment": {name: os.environ.get(name) for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "execution_source_root": str(SOURCE), "observation_only_hooks": True,
        }
        dump(folder / "environment.json", metadata)
        args = test_nmpc.build_parser().parse_args([
            "-c", str(folder / "config.yaml"), "--trials", "3",
            "--results-dir", str(folder), "--no-animation", "--no-plots",
        ])
        begun = time.perf_counter()
        try:
            test_nmpc.run_trial_batch(args, config=config)
        finally:
            metadata.update(finished_at_kst=stamp(), wall_time_s=time.perf_counter() - begun,
                            snapshot_changed=check_sources())
            dump(folder / "environment.json", metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=float, choices=WIDTHS)
    args = parser.parse_args()
    if args.width is not None:
        child(args.width)
        return
    prepare()
    env = dict(os.environ)
    env.update(ACADOS_SOURCE_DIR="/home/taek111/projects/acados",
               LD_LIBRARY_PATH="/home/taek111/projects/acados/lib",
               MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.pop(name, None)
    records = []
    skip_reason = None
    for width in WIDTHS:
        if skip_reason:
            records.append({"width_m": width, "status": "skipped", "reason": skip_reason})
        else:
            label = f"width_{round(width * 1000):03d}mm"
            print(f"Starting {label}: three sequential trials, seeds 0-2", flush=True)
            records.append({"width_m": width, "status": "running"})
            dump(ROOT / "sweep_progress.json", records)
            with (ROOT / f"{label}.run.log").open("w") as log:
                completed = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                            "--width", str(width)], cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                records[-1].update(status="execution_error", returncode=completed.returncode)
                dump(ROOT / "sweep_progress.json", records)
                raise RuntimeError(f"{label}: inspect its run log")
            with (ROOT / label / "trial_results.csv").open(newline="") as stream:
                trials = list(csv.DictReader(stream))
            assert len(trials) == 3
            outcomes = [row["status"] for row in trials]
            assert all(status in ("success", "failure", "timeout") for status in outcomes), outcomes
            successes = outcomes.count("success")
            records[-1].update(status="completed", successes=successes, trial_statuses=outcomes)
            for line in (ROOT / f"{label}.run.log").read_text().splitlines():
                if "WIDTH_RESULT" in line:
                    print(line, flush=True)
            if successes == 0:
                skip_reason = f"All three trials at width {width:.3f} m failed to reach the goal."
                print(skip_reason + " Skipping larger widths.", flush=True)
        dump(ROOT / "sweep_progress.json", records)
    dump(ROOT / "source_integrity.json", {"finished_at_kst": stamp(),
        "sources_changed_during_run": check_sources()})
    print("Requested sweep finished.", flush=True)


if __name__ == "__main__":
    main()
