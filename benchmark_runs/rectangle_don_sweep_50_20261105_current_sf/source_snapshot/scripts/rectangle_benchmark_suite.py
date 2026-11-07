#!/usr/bin/env python3
"""Prepare frozen rectangle benchmark inputs and run one width at a time.

--plan-only writes inspectable inputs without importing a controller, creating
an acados solver, or running any simulation. Normal execution uses the frozen
source snapshot rather than the possibly changing working checkout.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo

import yaml

from rectangle_benchmark_reuse import (
    describe_reuse,
    materialize_reused_width,
    validate_reuse_environment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("mpcc", "cc_ecs", "cc_sf", "cc_mf")
CORRIDOR_WIDTH = Decimal("0.150")
WIDTHS = {
    "don": ("0.075", "0.1125"),
    "don_sweep": tuple(str(Decimal(level) / 100 * CORRIDOR_WIDTH) for level in range(70, 88)),
}
TRIALS = {"don": 100, "don_sweep": 50}
SOURCE_DIRS = ("control", "models", "planning", "sim", "utils", "scripts")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def width_id(width: str) -> str:
    millimetres = Decimal(width) * 1000
    whole = int(millimetres)
    fraction = format(millimetres - whole, "f").rstrip("0")
    suffix = "p" + fraction[2:] if fraction not in ("", "0", "0.") else ""
    return f"width_{whole:03d}{suffix}mm"


def source_files(root: Path) -> list[Path]:
    paths = [root / "test_nmpc.py", root / "AGENTS.md", root / "benchmark_results_old.md"]
    paths.append(root / "scripts" / "rectangle_benchmark_baseline.json")
    for directory in SOURCE_DIRS:
        paths.extend(sorted((root / directory).rglob("*.py")))
        paths.extend(sorted((root / directory).rglob("*.sh")))
    paths.extend(sorted((root / "config").glob("*.yaml")))
    paths.extend(sorted((root / "docs").glob("rectangle_benchmark*.md")))
    return sorted(set(path for path in paths if path.is_file()))


def snapshot_hashes(snapshot: Path) -> dict:
    files = {
        str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files(snapshot)
    }
    aggregate = "".join(f"{name}\0{digest}\n" for name, digest in sorted(files.items()))
    return {
        "schema_version": 1, "algorithm": "sha256",
        "aggregate_sha256": hashlib.sha256(aggregate.encode("utf-8")).hexdigest(),
        "files": files,
    }


def freeze_source(origin: Path, destination: Path, reuse: bool) -> dict:
    required = (
        "test_nmpc.py", "sim/start_perturbation.py", "utils/solver_timing.py",
        "utils/benchmark_trace.py", "scripts/rectangle_benchmark_baseline.json",
    )
    if any(not (origin / name).is_file() for name in required):
        raise ValueError(f"Source snapshot lacks required benchmark sources: {origin}")
    if reuse:
        manifest_path = origin.parent / "source_hashes.json"
        if not manifest_path.is_file():
            raise ValueError(f"Reused source snapshot needs its hash manifest: {manifest_path}")
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
        if snapshot_hashes(origin) != expected:
            raise ValueError("Reused source snapshot has changed since it was recorded")
    destination.mkdir()
    for source in source_files(origin):
        target = destination / source.relative_to(origin)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return snapshot_hashes(destination)


def make_config(width: str, trials: int) -> dict:
    return {
        "trials": trials,
        "rectangle_geometry": {"length": 0.150, "width": float(width)},
        "mpcc": {"use_debug": False},
        "benchmark_logging": {"enabled": True},
        "start_perturbation": {
            "enabled": True, "seed": 0, "position_std": 0.02,
            "heading_std_deg": 2.8647889756541165, "max_sigma": 3.0,
            "min_clearance": 0.001, "max_attempts": 1000,
        },
        "localization_error": {"enabled": False},
        "defaults": {
            "simulation_time": 80.0, "generate_animation": False,
            "generate_plots": False,
        },
        "success_criteria": {"position_tolerance": 0.02},
        "trial_failure_criteria": {
            "enabled": True, "movement_window_sec": 10.0,
            "movement_threshold": 0.03, "max_consecutive_solver_failures": 20,
        },
        "test_configs": [{
            "maze_type": "maze", "robot_shape": "rectangle",
            "optimizer_type": method, "dynamics_type": "differential_drive",
            "path_planner": "astar",
        } for method in METHODS],
    }


METADATA_PROGRAM = r'''
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np
import torch
from control.mpcc_optimizer import MPCCOptimizerParam
from control.cc_ecs_optimizer import CCECSOptimizerParam
from control.cc_sf_optimizer import CCSFOptimizerParam
from control.cc_mf_optimizer import CCMFOptimizerParam

root = Path(sys.argv[1])
params = {}
for name, cls in (
    ('mpcc', MPCCOptimizerParam), ('cc_ecs', CCECSOptimizerParam),
    ('cc_sf', CCSFOptimizerParam), ('cc_mf', CCMFOptimizerParam),
):
    value = cls()
    value.use_debug = False
    if value.horizon != 20 or abs(value.tf / value.horizon - .1) > 1e-12:
        raise RuntimeError(f'{name}: expected horizon=20 and dt=0.1')
    if value.d_min != .0001 or value.a_max != .3 or value.r_delta_omega != .1:
        raise RuntimeError(f'{name}: baseline d_min/a_max/r_delta_omega changed')
    if name != 'mpcc' and (value.sigma_f0, value.sigma_l0, value.sigma_psi0) != (.0002,.00025,.00025):
        raise RuntimeError(f'{name}: baseline initial covariance changed')
    params[name] = vars(value)
def serialize(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)
params = json.loads(json.dumps(params, default=serialize, allow_nan=False))
(root / 'optimizer_parameters.json').write_text(
    json.dumps(params, indent=2, allow_nan=False)+'\n', encoding='utf-8')
baseline = json.loads((Path.cwd() / 'scripts' / 'rectangle_benchmark_baseline.json').read_text(encoding='utf-8'))
def numerical(values):
    return {key:value for key,value in values.items()
            if key != 'use_debug' and not key.startswith('debug')
            and key not in ('constraint_debug_tolerance','first_interval_substeps')}
differences = {}
for name, value in params.items():
    old, new = numerical(baseline['parameters'][name]), numerical(value)
    changed = {key:{'baseline':old.get(key),'current':new.get(key)}
               for key in old.keys() | new.keys()
               if key not in old or key not in new or old[key] != new[key]}
    if changed: differences[name] = changed
(root / 'baseline_validation.json').write_text(
    json.dumps({'matched':not differences,'differences':differences},indent=2)+'\n',encoding='utf-8')
if differences:
    raise RuntimeError('Non-debug controller parameters differ from 2026-11-03 baseline; see baseline_validation.json')
versions = {}
for name in ('numpy','torch','casadi','scipy','PyYAML','matplotlib'):
    try: versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name] = None
cuda = torch.cuda.is_available()
environment = {
    'python': sys.version, 'python_executable': sys.executable,
    'platform': platform.platform(), 'processor': platform.processor(),
    'packages': versions, 'torch_num_threads': torch.get_num_threads(),
    'torch_num_interop_threads': torch.get_num_interop_threads(),
    'torch_cuda_version': torch.version.cuda, 'cuda_available': cuda,
    'gpu_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
    'environment': {name: os.environ.get(name) for name in (
        'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS',
        'PYTHONHASHSEED','MPLBACKEND','ACADOS_SOURCE_DIR','CUDA_VISIBLE_DEVICES')},
}
(root / 'environment.json').write_text(json.dumps(environment,indent=2)+'\n',encoding='utf-8')
if not cuda:
    raise RuntimeError('These controllers require CUDA; no CUDA device is available')
'''


def child_environment(threads: int) -> dict:
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = str(threads)
    environment.update(
        PYTHONHASHSEED="0", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1",
    )
    return environment


def width_result_summary(directory: Path, trials: int) -> dict:
    result = directory / "trial_results.csv"
    if not result.is_file():
        return {"complete": False, "trace_export_error_trials": 0}
    with result.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(method, seed) for method in METHODS for seed in range(trials)}
    actual = [(row["optimizer_type"], int(row["start_seed"])) for row in rows]
    complete = (
        len(actual) == len(expected) and set(actual) == expected
        and all(row["status"] in ("success", "failure", "timeout", "error") for row in rows)
    )
    return {
        "complete": complete,
        "trace_export_error_trials": sum(
            bool(row.get("raw_trace_error"))
            or not row.get("raw_trace_path")
            or not row.get("raw_trace_metadata_path")
            for row in rows
        ),
    }


def run_child(command: list[str], cwd: Path, environment: dict, log: Path) -> int:
    with log.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(
            command, cwd=cwd, env=environment, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            return child.wait()
        except KeyboardInterrupt:
            # Let test_nmpc persist the interrupted trial and clean acados files.
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait()
            except KeyboardInterrupt:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
            return 130


def request_stop(signum, frame):
    """Route process-manager termination through normal child cleanup."""
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=(*WIDTHS, "success"))
    parser.add_argument("--results-dir", type=Path, default=None, help="Fresh directory; existing paths are rejected")
    parser.add_argument("--source-snapshot", type=Path, default=None, help="Reuse a prior suite's verified source_snapshot directory")
    parser.add_argument(
        "--reuse-results", type=Path, default=None,
        help="For don_sweep only, copy DoN=.75 seeds0-49 from a completed first suite after strict compatibility checks",
    )
    parser.add_argument("--plan-only", action="store_true", help="Write frozen sources/configs/manifest only; do not import or run controllers")
    args = parser.parse_args()
    # Retain the old wrapper/CLI spelling as an alias for the revised B plan.
    if args.experiment == "success":
        args.experiment = "don_sweep"
    if args.reuse_results is not None and args.experiment != "don_sweep":
        parser.error("--reuse-results applies only to the second (don_sweep) experiment")
    trials = TRIALS[args.experiment]
    signal.signal(signal.SIGTERM, request_stop)
    threads = int(os.environ.get("BENCHMARK_NUM_THREADS", "6"))
    if threads < 1:
        parser.error("BENCHMARK_NUM_THREADS must be a positive integer")
    created = datetime.now(ZoneInfo("Asia/Seoul"))
    root = args.results_dir or (
        PROJECT_ROOT / "benchmark_runs" /
        f"rectangle_{args.experiment}_{trials}trials_{created:%Y%m%d_%H%M%S_%f}"
    )
    root = root.expanduser().resolve()
    reuse_root = args.reuse_results.expanduser().resolve() if args.reuse_results else None
    origin = (
        args.source_snapshot.expanduser().resolve() if args.source_snapshot else
        reuse_root / "source_snapshot" if reuse_root else PROJECT_ROOT
    )
    lock_root = PROJECT_ROOT / "benchmark_runs"
    lock_root.mkdir(exist_ok=True)
    with (lock_root / ".rectangle_benchmark.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another rectangle suite is already running from this checkout")
        root.mkdir(parents=True, exist_ok=False)
        for directory in ("configs", "logs", "widths"):
            (root / directory).mkdir()
        hashes = freeze_source(
            origin, root / "source_snapshot",
            args.source_snapshot is not None or reuse_root is not None,
        )
        write_json(root / "source_hashes.json", hashes)
        widths = []
        for width in WIDTHS[args.experiment]:
            name = width_id(width)
            config = f"configs/{name}.yaml"
            (root / config).write_text(yaml.safe_dump(make_config(width, trials), sort_keys=False), encoding="utf-8")
            widths.append({
                "width_m": float(width), "nominal_don": float(Decimal(width) / CORRIDOR_WIDTH),
                "width_id": name, "config_path": config, "results_dir": f"widths/{name}",
            })
        reuse = describe_reuse(reuse_root, widths, trials, METHODS) if reuse_root else None
        reused_trials = len(METHODS) * trials if reuse is not None else 0
        manifest = {
            "schema_version": 1, "experiment": args.experiment,
            "created_at_kst": created.isoformat(), "plan_only": args.plan_only,
            "trials_per_configuration": trials, "seeds": list(range(trials)),
            "methods": list(METHODS), "widths": widths,
            "expected_trials": len(widths) * len(METHODS) * trials,
            "planned_new_trials": len(widths) * len(METHODS) * trials - reused_trials,
            "reused_trials": reused_trials,
            "nominal_don_definition": "robot_width_m / nominal_corridor_width_m",
            "nominal_corridor_width_m": float(CORRIDOR_WIDTH),
            "source_origin": str(origin), "source_snapshot": "source_snapshot",
            "source_hashes": "source_hashes.json", "source_aggregate_sha256": hashes["aggregate_sha256"],
            "baseline_parameters": "source_snapshot/scripts/rectangle_benchmark_baseline.json",
            "reference": "source_snapshot/benchmark_results_old.md#Rectangle 너비별 주행 benchmark (0.005 m 간격)",
            "computation_time_scope": "solve_nlp", "computation_time_unit": "s",
            "successful_solver_steps_only": True, "include_first_step": True,
            "use_debug": False, "clearance_evaluation": "offline_only",
            "benchmark_num_threads": threads, "python_executable": sys.executable,
            "host_platform": platform.platform(),
            "runtime_metadata": "deferred_until_execution" if args.plan_only else "environment.json and optimizer_parameters.json",
        }
        write_json(root / "suite_manifest.json", manifest)
        state = {
            "status": "prepared", "completed_widths": [], "current_width": None,
            "trace_export_error_trials": {}, "trace_data_complete": None,
            "reused_widths": [],
        }
        write_json(root / "suite_state.json", state)
        print(f"Suite: {root}\nPlanned trials: {manifest['expected_trials']}", flush=True)
        if args.plan_only:
            print("Plan only: no controller imports or simulations were executed.", flush=True)
            return 0
        environment = child_environment(threads)
        snapshot = root / "source_snapshot"
        try:
            subprocess.run(
                [sys.executable, "-c", METADATA_PROGRAM, str(root)],
                cwd=snapshot, env=environment, check=True,
            )
            if reuse is not None:
                # Validate and copy every selected trace before starting ANY B run.
                validate_reuse_environment(root, reuse)
                reused_width = next(width for width in widths if "reuse" in width)
                materialize_reused_width(root, reused_width, METHODS)
                state["reused_widths"].append(reused_width["width_id"])
                write_json(root / "suite_state.json", state)
            for width in widths:
                name = width["width_id"]
                state.update(status="running", current_width=name)
                write_json(root / "suite_state.json", state)
                if "reuse" in width:
                    print(f"Reusing {name}: four controllers x {trials} selected trials; no simulation", flush=True)
                    code = 0
                else:
                    command = [
                        sys.executable, "-u", str(snapshot / "test_nmpc.py"),
                        "-c", str(root / width["config_path"]), "--trials", str(trials),
                        "--results-dir", str(root / width["results_dir"]),
                        "--no-use-debug", "--no-animation", "--no-plots",
                    ]
                    print(f"Running {name}: four controllers x {trials} trials; log={root / 'logs' / (name + '.log')}", flush=True)
                    code = run_child(command, snapshot, environment, root / "logs" / f"{name}.log")
                summary = width_result_summary(root / width["results_dir"], trials)
                state["trace_export_error_trials"][name] = summary["trace_export_error_trials"]
                if code != 0 or not summary["complete"]:
                    state.update(status="interrupted" if code == 130 else "incomplete", return_code=code)
                    write_json(root / "suite_state.json", state)
                    print(f"Suite stopped at {name}; inspect its log and trial_results.csv. Use a fresh root for a rerun.", file=sys.stderr)
                    return code or 1
                state["completed_widths"].append(name)
                write_json(root / "suite_state.json", state)
                print(f"Completed {name}", flush=True)
        except KeyboardInterrupt:
            state.update(status="interrupted", return_code=130)
            write_json(root / "suite_state.json", state)
            return 130
        except Exception as error:
            state.update(status="error", error=f"{type(error).__name__}: {error}")
            write_json(root / "suite_state.json", state)
            raise
        state.update(
            status="complete", current_width=None,
            trace_data_complete=not any(state["trace_export_error_trials"].values()),
        )
        write_json(root / "suite_state.json", state)
        print(f"Suite complete: {root}", flush=True)
        if not state["trace_data_complete"]:
            print(
                "Outcome rows are complete, but some raw traces are missing; "
                "inspect suite_state.json trace_export_error_trials before clearance analysis.",
                file=sys.stderr,
            )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
