"""Sequential, frozen-source adaptive CC-MF noise combination experiments."""

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone

from run_width_benchmark import runtime_environment, snapshot_sources


ROOT = Path(__file__).resolve().parent
NOISE_KEYS = {
    "q_f0", "q_l0", "q_psi0", "alpha_f", "alpha_v", "alpha_kappa",
    "beta_v", "beta_kappa", "beta_omega",
}


def plain(value):
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def add_job(output, manifest, label, overrides, cohort, seed_start,
            kind="candidate", reused_output=None):
    import yaml

    baseline = json.loads((output / "baseline_parameters.json").read_text())
    if not overrides or set(overrides) - NOISE_KEYS:
        raise ValueError(f"Only explicit noise overrides are allowed: {overrides}")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
               for value in overrides.values()):
        raise ValueError("Noise overrides must be finite and nonnegative")
    overrides = {key: value for key, value in overrides.items() if baseline[key] != value}
    identity = f"maze_{cohort}_{label}"
    if any(job["id"] == identity for job in manifest["jobs"]):
        raise ValueError(f"Job already exists: {identity}")
    seeds = list(range(seed_start, seed_start + 10))
    if cohort in manifest["cohorts"] and manifest["cohorts"][cohort] != seeds:
        raise ValueError("A cohort must use the same seeds")
    if kind == "reference" and any(job["cohort"] == cohort and job["kind"] == kind
                                    for job in manifest["jobs"]):
        raise ValueError("A cohort must have exactly one reference")
    manifest["cohorts"][cohort] = seeds
    config = yaml.safe_load((output / "base_config.yaml").read_text())
    config["start_perturbation"]["seed"] = seed_start
    config["cc_mf"].update(overrides)
    config_path = output / "configs" / f"{identity}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    job = {
        "id": identity, "environment": "maze", "cohort": cohort, "kind": kind,
        "seeds": seeds, "overrides": overrides,
        "parameter": ", ".join(f"{key}={value:g}" for key, value in overrides.items()),
        "value": None, "baseline_value": None,
        "config_path": str(config_path),
        "output_dir": str(reused_output or output / "jobs" / identity),
        "reused": reused_output is not None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest["jobs"].append(job)
    return job


def initialize(args):
    import yaml

    output, prior = args.results_dir.resolve(), args.reference_run.resolve()
    previous_hashes = json.loads((prior / "source_hashes.json").read_text())
    changed = [key for key, digest in previous_hashes.items()
               if not (ROOT / key).exists()
               or hashlib.sha256((ROOT / key).read_bytes()).hexdigest() != digest]
    if changed:
        raise ValueError(f"Previous reference is not comparable; sources changed: {changed}")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = snapshot_sources(output)
    script = Path(__file__)
    (snapshot / script.name).write_bytes(script.read_bytes())
    hashes = json.loads((output / "source_hashes.json").read_text())
    hashes[script.name] = hashlib.sha256(script.read_bytes()).hexdigest()
    write_json(output / "source_hashes.json", hashes)
    (output / "input_config.yaml").write_bytes((prior / "input_config.yaml").read_bytes())
    (output / "baseline_parameters.json").write_bytes((prior / "baseline_parameters.json").read_bytes())
    base = yaml.safe_load((output / "input_config.yaml").read_text())
    base["trials"] = 10
    base["rectangle_geometry"] = {"length": 0.15, "width": 0.127}
    base.setdefault("defaults", {}).update(generate_animation=False, generate_plots=False)
    base.setdefault("cc_mf", {}).update(debug_mf=False, debug_infeasibility=False)
    base["test_configs"] = [{"maze_type": "maze", "robot_shape": "rectangle",
                             "optimizer_type": "cc_mf", "dynamics_type": "differential_drive",
                             "path_planner": "astar"}]
    (output / "base_config.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    (output / "configs").mkdir()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(snapshot), "reference_run": str(prior),
        "trials_per_job": 10, "environments": ["maze"], "cohorts": {}, "jobs": [],
        "output_changes": "Animations, plots and optional post-solve diagnostics disabled uniformly.",
        "rules": "Sequential ten-trial jobs; explicit multi-parameter overrides against the original benchmark; no partial-job resume.",
        "design": "Compare two beta_omega/alpha_kappa combinations first, then select additional combinations from their results. Confirm a selected improvement on new seeds against the prior 80% setting.",
    }
    reference = prior / "jobs/maze_07_beta_omega_0p0032"
    add_job(output, manifest, "reference80", {"beta_omega": 0.0032}, "search", 0,
            kind="reference", reused_output=reference)
    add_job(output, manifest, "01_joint_moderate",
            {"beta_omega": 0.0032, "alpha_kappa": 0.0002}, "search", 0)
    add_job(output, manifest, "02_joint_strong",
            {"beta_omega": 0.0032, "alpha_kappa": 0.0001}, "search", 0)
    write_json(output / "sweep_manifest.json", manifest)
    write_json(output / "reference_reuse.json", {
        "prior_root": str(prior), "reference_output": str(reference),
        "all_prior_source_hashes_match_current": True,
        "prior_source_hashes": previous_hashes,
        "reused_trials": 10, "new_initial_trials": 20,
    })
    print(f"Prepared 2 new combinations (20 drives), reusing 10 reference trials: {output}")


def append(args):
    output = args.results_dir.resolve()
    manifest = json.loads((output / "sweep_manifest.json").read_text())
    if not args.label or args.overrides is None:
        raise ValueError("append requires --label and --overrides JSON")
    job = add_job(output, manifest, args.label, json.loads(args.overrides),
                  args.cohort, args.seed_start, "reference" if args.reference else "candidate")
    write_json(output / "sweep_manifest.json", manifest)
    print(f"Added {job['id']}: {job['overrides']}, seeds {job['seeds']}")


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
    for key, value in config["cc_mf"].items():
        if not hasattr(parameters, key):
            raise ValueError(f"Unknown CC-MF parameter {key}")
        setattr(parameters, key, value)
    resolved = plain(vars(parameters))
    baseline = json.loads((output / "baseline_parameters.json").read_text())
    differences = {key: value for key, value in resolved.items() if baseline.get(key) != value}
    if differences != job["overrides"]:
        raise ValueError(f"Unexpected parameter differences: {differences}")
    write_json(destination / "resolved_cc_mf_parameters.json", resolved)
    write_json(destination / "parameter_changes.json", {
        key: {"baseline": baseline[key], "actual": value} for key, value in differences.items()})
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
    source = Path(manifest["source_root"])
    hashes = json.loads((output / "source_hashes.json").read_text())
    if any(hashlib.sha256((source / key).read_bytes()).hexdigest() != digest
           for key, digest in hashes.items()):
        raise ValueError("Frozen sources changed")
    started = 0
    for job in manifest["jobs"]:
        if args.only and job["id"] not in args.only:
            continue
        destination = Path(job["output_dir"])
        completion = destination / "completion.json"
        if completion.exists() and json.loads(completion.read_text()).get("complete"):
            continue
        if destination.exists():
            raise FileExistsError(f"Incomplete job exists; use a new ID: {destination}")
        if job["reused"]:
            raise ValueError("A reused reference must already be complete")
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
        complete = (returncode == 0 and len(rows) == 10
                    and {int(row["start_seed"]) for row in rows} == set(job["seeds"])
                    and not any(row["status"] == "interrupted" for row in rows))
        record = {"complete": complete, "returncode": returncode, "trials": len(rows),
                  "successes": sum(int(row["success"]) for row in rows),
                  "finished_utc": datetime.now(timezone.utc).isoformat()}
        write_json(completion, record)
        print(f"DONE {job['id']}: {record['successes']}/{len(rows)} success, rc={returncode}", flush=True)
        if not complete:
            raise RuntimeError(f"Job incomplete; inspect {destination / 'run.log'}")
        started += 1
        if args.max_jobs is not None and started >= args.max_jobs:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "append", "run", "worker"))
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path,
                        default=ROOT / "benchmark_runs/cc_mf_noise_sweep_20261102")
    parser.add_argument("--job")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--label")
    parser.add_argument("--overrides")
    parser.add_argument("--cohort", default="search")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    runtime_environment()
    {"init": initialize, "append": append, "run": run, "worker": worker}[args.action](args)


if __name__ == "__main__":
    main()
