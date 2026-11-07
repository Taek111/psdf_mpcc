"""Parallel scheduling of the unchanged offline helper's exact trial calculation.

Only offline calculation is parallelized. Source trials, source snapshots and the
project helper are read-only. The original analyze()/main() retain every cohort,
provenance, validation, aggregation, coverage and strict-exit rule. Ten results
are independently recalculated serially and required to be exactly equal.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Avoid unnecessary BLAS thread pools inside independent offline processes.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import concurrent.futures
import csv
import datetime
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/analyze_rectangle_clearance.py"
SUITES = [
    ROOT / "benchmark_runs/rectangle_don_100_20261105_current_sf",
    ROOT / "benchmark_runs/rectangle_don_sweep_50_20261105_current_sf",
]
WORKERS = 12
WORKER_HELPER = None
WORKER_CLEARANCE = {}


def import_helper():
    spec = importlib.util.spec_from_file_location("unchanged_clearance", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initialize_worker():
    global WORKER_HELPER
    WORKER_HELPER = import_helper()
    for suite in SUITES:
        WORKER_CLEARANCE[str(suite)] = WORKER_HELPER.load_clearance_function(suite)[0]


def task_key(npz, metadata, trial, width_m):
    return (
        str(Path(npz).resolve()), str(Path(metadata).resolve()), float(width_m),
        json.dumps(trial, sort_keys=True, ensure_ascii=False),
    )


def calculate_worker(task):
    suite, width, npz, metadata, trial = task
    key = task_key(npz, metadata, trial, width)
    try:
        result = WORKER_HELPER.calculate_trial(
            Path(npz), Path(metadata), trial, width, WORKER_CLEARANCE[suite]
        )
        return key, True, result
    except (OSError, ValueError, KeyError, TypeError) as error:
        # Preserve the original exception instance and message through pickle.
        return key, False, error


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run():
    helper = import_helper()
    helper_digest = helper.sha256(HELPER_PATH)
    original_calculate_trial = helper.calculate_trial
    tasks = []
    for suite in SUITES:
        manifest = helper.read_json(suite / "suite_manifest.json")
        for width in manifest["widths"]:
            width_root = (suite / width["results_dir"]).resolve()
            result_path = width_root / "trial_results.csv"
            if not result_path.exists():
                continue  # Original analyze() records this missing cohort.
            with result_path.open(newline="", encoding="utf-8-sig") as stream:
                for trial in csv.DictReader(stream):
                    npz, metadata = helper.trace_paths(width_root, trial)
                    tasks.append((str(suite), float(width["width_m"]), str(npz), str(metadata), trial))
    print(f"Precomputing {len(tasks)} original calculate_trial() calls with {WORKERS} workers", flush=True)
    cache = {}
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
    ) as executor:
        futures = [executor.submit(calculate_worker, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            key, available, value = future.result()
            if key in cache:
                raise ValueError("Duplicate exact precomputation key")
            cache[key] = (available, value)
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                print(f"Precomputed {completed}/{len(tasks)} trials", flush=True)

    # Use first trial per A width/method, plus two methods at maximum B width.
    validation_tasks = []
    chosen = set()
    for task in tasks:
        suite, width, _, _, trial = task
        group = (suite, width, trial["optimizer_type"])
        eligible = suite == str(SUITES[0]) or (
            suite == str(SUITES[1]) and np.isclose(width, 0.1305, rtol=0, atol=1e-12)
            and trial["optimizer_type"] in {"mpcc", "cc_sf"}
        )
        if eligible and group not in chosen:
            chosen.add(group)
            validation_tasks.append(task)
    validation_rows = []
    for task in validation_tasks:
        suite, width, npz, metadata, trial = task
        pose_clearance, _ = helper.load_clearance_function(Path(suite))
        key = task_key(npz, metadata, trial, width)
        available, parallel_value = cache[key]
        if not available:
            raise RuntimeError(f"Serial validation selected an invalid trace: {parallel_value}")
        serial_value = original_calculate_trial(Path(npz), Path(metadata), trial, width, pose_clearance)
        equal = bool(np.array_equal(serial_value[0], parallel_value[0]))
        receipts_equal = serial_value[1] == parallel_value[1]
        validation_rows.append({
            "suite_root": suite, "width_m": width, "optimizer_type": trial["optimizer_type"],
            "start_seed": trial["start_seed"], "status": trial["status"],
            "pose_samples": len(serial_value[0]), "arrays_exactly_equal": equal,
            "receipts_exactly_equal": receipts_equal,
            "max_absolute_difference_m": float(np.max(np.abs(serial_value[0] - parallel_value[0]))),
            "raw_trace_path": npz,
        })
        if not equal or not receipts_equal:
            raise ValueError("Serial and parallel clearance validation differed")
    validation_path = OUT / "audit/clearance_parallel_serial_validation.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(validation_rows[0]))
        writer.writeheader()
        writer.writerows(validation_rows)
    print(f"Exactly equal serial validation: {len(validation_rows)} trials", flush=True)

    requested_keys = []

    def lookup_calculate_trial(npz, metadata, trial, width, pose_clearance):
        key = task_key(npz, metadata, trial, width)
        requested_keys.append(key)
        available, value = cache[key]
        if not available:
            raise value
        return value

    # The helper entry point is unchanged; only completed exact calculations
    # are supplied from memory when analyze() reaches the same trial call.
    helper.calculate_trial = lookup_calculate_trial
    helper_arguments = [str(HELPER_PATH), *map(str, SUITES), "--output-dir", str(OUT / "offline_clearance"), "--strict"]
    sys.argv = helper_arguments
    exit_code = helper.main()
    if helper.sha256(HELPER_PATH) != helper_digest:
        raise RuntimeError("Project helper changed during analysis")
    write_json(OUT / "audit/clearance_parallel_equivalence.json", {
        "helper_path": str(HELPER_PATH), "helper_sha256": helper_digest,
        "workers": WORKERS, "tasks_precomputed": len(tasks),
        "helper_requested_calculations": len(requested_keys),
        "calculation_requests_match_precomputation": set(requested_keys) == set(cache),
        "worker_calculation_errors": [
            {"raw_trace_path": key[0], "type": type(value).__name__, "message": str(value)}
            for key, (available, value) in cache.items() if not available
        ],
        "serial_validation_trials": len(validation_rows),
        "serial_validation_exactly_equal": all(row["arrays_exactly_equal"] and row["receipts_exactly_equal"] for row in validation_rows),
        "strict_helper_exit_code": exit_code,
        "implementation_difference": "Scheduling only: unchanged calculate_trial plus archived scalar pose_clearance executed in process workers, then supplied to unchanged helper.main/analyze via an in-memory lookup; no geometry vectorization or tolerance change.",
        "source_result_content_files_modified": False,
        "bytecode_writes_disabled": sys.dont_write_bytecode and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "simulation_executed": False,
    })
    return exit_code


if __name__ == "__main__":
    receipt_path = OUT / "audit/clearance_parallel_execution.json"
    receipt = {
        "command": [sys.executable, str(Path(__file__).resolve())],
        "equivalent_helper_arguments": [str(HELPER_PATH), *map(str, SUITES), "--output-dir", str(OUT / "offline_clearance"), "--strict"],
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "running", "workers": WORKERS,
        "bytecode_writes_disabled": sys.dont_write_bytecode and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "prior_serial_receipt": str(OUT / "audit/clearance_execution.json"),
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(receipt_path, receipt)
    start = time.perf_counter()
    try:
        code = run()
    except BaseException as error:
        code = 1
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
    receipt.update(status="completed", exit_code=code,
                   elapsed_seconds=time.perf_counter()-start,
                   finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(receipt_path, receipt)
    raise SystemExit(code)
