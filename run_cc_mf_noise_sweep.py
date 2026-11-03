"""Sequential, source-frozen one-factor-at-a-time CC-MF noise experiments.

Initialize once, then run whole jobs. Completed jobs are skipped; incomplete
trial directories are never resumed or overwritten. Each job has ten trials.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone

from run_width_benchmark import runtime_environment, snapshot_sources


ROOT = Path(__file__).resolve().parent
GRID = {
    "q_f0": (1e-7, 1e-6),
    "q_l0": (1e-7, 1e-6),
    "q_psi0": (1e-6, 1e-5),
    "beta_omega": (0.0032, 0.0016),
    "beta_kappa": (0.0032, 0.0016),
    "alpha_kappa": (0.0002, 0.0001),
}


def plain(value):
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def initialize(args):
    import yaml

    output = args.results_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = snapshot_sources(output)
    (snapshot / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    hashes = json.loads((output / "source_hashes.json").read_text())
    hashes[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(output / "source_hashes.json", hashes)
    sys.path.insert(0, str(snapshot))
    from control.cc_mf_optimizer import CCMFOptimizerParam

    input_bytes = args.config.read_bytes()
    (output / "input_config.yaml").write_bytes(input_bytes)
    base = yaml.safe_load(input_bytes)
    base["trials"] = 10
    base.setdefault("rectangle_geometry", {"length": 0.15, "width": 0.127})
    base.setdefault("defaults", {}).update(generate_animation=False, generate_plots=False)
    base.setdefault("cc_mf", {}).update(debug_mf=False, debug_infeasibility=False)
    parameters = CCMFOptimizerParam()
    for key, value in base["cc_mf"].items():
        if not hasattr(parameters, key):
            raise ValueError(f"Unknown parameter {key}")
        setattr(parameters, key, value)
    write_json(output / "baseline_parameters.json", vars(parameters))
    (output / "configs").mkdir()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(snapshot), "trials_per_job": 10,
        "seeds": list(range(10)), "environments": args.environments,
        "grid": GRID, "jobs": [],
        "reference_run": str(ROOT / "benchmark_runs/cc_mf_benchmark_stop_analysis_20261102"),
        "output_changes": "Animations, plots and optional post-solve GPU diagnostics disabled for all jobs, including fresh baseline.",
        "rules": "Each candidate changes one resolved noise parameter from baseline. Sequential execution. No partial-trial resume.",
    }
    for environment in args.environments:
        variants = [(None, None)] + [(key, value) for key, values in GRID.items() for value in values]
        for number, (parameter, value) in enumerate(variants):
            suffix = "baseline" if parameter is None else f"{parameter}_{value:g}".replace("-", "m").replace(".", "p")
            identity = f"{environment}_{number:02d}_{suffix}"
            config = copy.deepcopy(base)
            config["start_perturbation"]["seed"] = 0
            config["test_configs"] = [{
                "maze_type": environment, "robot_shape": "rectangle",
                "optimizer_type": "cc_mf", "dynamics_type": "differential_drive",
                "path_planner": "astar",
            }]
            if parameter is not None:
                config["cc_mf"][parameter] = value
            config_path = output / "configs" / f"{identity}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            manifest["jobs"].append({
                "id": identity, "environment": environment, "parameter": parameter,
                "value": value,
                "baseline_value": getattr(parameters, parameter) if parameter else None,
                "config_path": str(config_path),
                "output_dir": str(output / "jobs" / identity),
            })
    write_json(output / "sweep_manifest.json", manifest)
    print(f"Prepared {len(manifest['jobs'])} jobs, {10 * len(manifest['jobs'])} drives: {output}")


def worker(args):
    output = args.results_dir.resolve()
    manifest = json.loads((output / "sweep_manifest.json").read_text())
    job = next(job for job in manifest["jobs"] if job["id"] == args.job)
    destination = Path(job["output_dir"])
    sys.path.insert(0, manifest["source_root"])
    import numpy as np
    import torch
    import yaml

    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    from control.cc_mf_optimizer import CCMFOptimizerParam
    from sim.simulation_mpc import simulation_mpc, DifferentialDriveRectangleGeometry
    from test_nmpc import main as run_trials

    config = yaml.safe_load(Path(job["config_path"]).read_text())
    parameters = CCMFOptimizerParam()
    for key, value in config.get("cc_mf", {}).items():
        if not hasattr(parameters, key):
            raise ValueError(f"Unknown parameter {key}")
        setattr(parameters, key, value)
    resolved = plain(vars(parameters))
    baseline = json.loads((output / "baseline_parameters.json").read_text())
    differences = {key: {"baseline": baseline.get(key), "actual": value}
                   for key, value in resolved.items() if baseline.get(key) != value}
    expected = set() if job["parameter"] is None else {job["parameter"]}
    if set(differences) != expected:
        raise ValueError(f"Not an OFAT experiment: {differences}")
    write_json(destination / "resolved_cc_mf_parameters.json", resolved)
    write_json(destination / "parameter_changes.json", differences)
    write_json(destination / "environment.json", {
        "python": sys.executable, "torch": torch.__version__, "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(), "torch_threads": torch.get_num_threads(),
        "source_root": manifest["source_root"], "cli": sys.argv,
    })
    start, goal, grid, obstacles = simulation_mpc().create_env(job["environment"])
    dimensions = config["rectangle_geometry"]
    geometry = DifferentialDriveRectangleGeometry(dimensions["length"], dimensions["width"], 0.0)
    write_json(destination / "environment_geometry.json", {
        "environment": job["environment"], "bounds": grid[0], "start": start, "goal": goal,
        "footprint_length_m": dimensions["length"], "footprint_width_m": dimensions["width"],
        "footprint_vertices_local": geometry._region.get_ccw_vertices(),
        "obstacles": [obstacle.get_ccw_vertices() for obstacle in obstacles],
    })
    sys.argv = ["test_nmpc.py", "-c", job["config_path"], "--trials", "10",
                "--results-dir", str(destination / "results"), "--no-animation", "--no-plots"]
    run_trials()


def run(args):
    output = args.results_dir.resolve()
    manifest = json.loads((output / "sweep_manifest.json").read_text())
    hashes = json.loads((output / "source_hashes.json").read_text())
    source = Path(manifest["source_root"])
    mismatches = [key for key, digest in hashes.items()
                  if hashlib.sha256((source / key).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise ValueError(f"Frozen sources changed: {mismatches}")
    started = 0
    for job in manifest["jobs"]:
        if args.only and job["id"] not in args.only:
            continue
        destination = Path(job["output_dir"])
        completion = destination / "completion.json"
        if completion.exists() and json.loads(completion.read_text()).get("complete"):
            continue
        if destination.exists():
            raise FileExistsError(f"Incomplete job exists; use a new job ID, never overwrite: {destination}")
        destination.mkdir(parents=True)
        print(f"START {job['id']} (10 trials)", flush=True)
        command = [sys.executable, "-u", str(source / Path(__file__).name), "worker",
                   "--results-dir", str(output), "--job", job["id"]]
        with (destination / "run.log").open("w", buffering=1) as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                returncode = child.wait()
            except KeyboardInterrupt:
                child.send_signal(signal.SIGINT)
                child.wait()
                raise
        result_path = destination / "results/trial_results.csv"
        rows = list(csv.DictReader(result_path.open())) if result_path.exists() else []
        complete = returncode == 0 and len(rows) == 10 and not any(row["status"] == "interrupted" for row in rows)
        record = {
            "complete": complete, "returncode": returncode, "trials": len(rows),
            "successes": sum(int(row["success"]) for row in rows),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(completion, record)
        print(f"DONE {job['id']}: {record['successes']}/{len(rows)} success, rc={returncode}", flush=True)
        if not complete:
            raise RuntimeError(f"Job did not complete: {job['id']}; see {destination / 'run.log'}")
        started += 1
        if args.max_jobs is not None and started >= args.max_jobs:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run", "worker"))
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/config_benchmark.yaml")
    parser.add_argument("--environments", nargs="+", default=["maze"], choices=("maze", "oblique_maze"))
    parser.add_argument("--job")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    runtime_environment()
    {"init": initialize, "run": run, "worker": worker}[args.action](args)


if __name__ == "__main__":
    main()
