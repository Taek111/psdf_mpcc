"""Run the ten matched width-125 trials with only the CC-SF source replaced.

All simulation/model sources and parameters come from the selected reference
experiment. Each seed executes sequentially in a separate process. A fresh
results directory is required; the reference is never modified.
"""

import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from run_width_benchmark import ROOT, runtime_environment, write_csv, write_json


FIELDS = (
    "width_m", "method", "trial", "seed", "success", "status", "failure_reason",
    "final_time_s", "safe_stop_count", "safe_stop_supported", "steps",
    "clearance_samples", "wall_time_s", "run_dir", "error",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--reference-run", type=Path, default=ROOT / (
        "benchmark_runs/rectangle_width070_130mm_step005_10trials_20261103"))
    args = parser.parse_args()
    runtime_environment()
    reference = args.reference_run.resolve()
    output = args.results_dir.resolve()
    baseline = json.loads((reference / "experiment.json").read_text())
    reference_hashes = json.loads((reference / "source_hashes.json").read_text())
    for name, expected in reference_hashes.items():
        if digest(reference / "source_snapshot" / name) != expected:
            raise ValueError(f"Reference snapshot changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    shutil.copytree(reference / "source_snapshot", snapshot,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # The current harness supports a methods subset. It does not change the plant.
    replacements = ("control/cc_sf_optimizer.py", "run_width_benchmark.py")
    for name in replacements:
        shutil.copy2(ROOT / name, snapshot / name)
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    hashes = {str(path.relative_to(snapshot)): digest(path)
              for path in snapshot.rglob("*") if path.is_file()}
    write_json(output / "source_hashes.json", hashes)
    changes = {name: {"reference": expected, "current": hashes.get(name)}
               for name, expected in reference_hashes.items()
               if hashes.get(name) != expected}
    if set(changes) - set(replacements):
        raise ValueError(f"Unexpected source changes: {changes}")
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    parameters = copy.deepcopy(baseline["resolved_defaults"]["cc_sf"])
    parameters["d_col"] = baseline["resolved_defaults"]["cc_mf"]["d_col"]
    manifest = copy.deepcopy(baseline)
    for field in ("finished_at_kst", "continuation_harness_sha256"):
        manifest.pop(field, None)
    manifest.update(
        started_at_kst=now.isoformat(), widths_m=[0.125], methods=["cc_sf"],
        trials_per_configuration=10, requested_trials=10, seeds=list(range(10)),
        trial_process_isolation=True, current_defaults=True, crash_checkpoints=True,
        generate_animation=False, simulation_limit_s=80.0,
        min_clearance_by_width_m={"0.125": 0.001},
        output_suffix="sfguard_" + now.strftime("%Y%m%d_%H%M%S"),
        reference_run=str(reference), source_changes_from_reference=changes,
        optimizer_overrides={"cc_sf": parameters},
        ablation={
            "name": "CC-SF + hard PSDF guard, future-stage chance constraints",
            "guard_clearance_m": parameters["d_col"],
            "chance_stages": list(range(1, 21)),
            "guard_stages": list(range(1, 21)),
            "stage0_policy": "No OCP safety row; exact PSDF < d_col triggers zero-input stop",
            "chance_row": "Original hard Gaussian distance margin; no slack or domain gate",
            "changed_controller_source": "control/cc_sf_optimizer.py",
            "unchanged_simulation_sources": True,
        },
    )
    manifest["resolved_defaults"]["cc_sf"] = parameters
    write_json(output / "experiment.json", manifest)
    with (output / "trial_results.csv").open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
    write_json(output / "progress.json", {"completed": 0, "total": 10})
    environment = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6",
                       MKL_NUM_THREADS="6")
    integrity = {"source_changes": changes, "start_pose_matches": []}
    for seed in range(10):
        command = [sys.executable, "-u", str(snapshot / "run_width_benchmark.py"),
                   "--results-dir", str(output), "--trials", "10", "--resume",
                   "--only-width", "0.125", "--only-method", "cc_sf",
                   "--only-seed", str(seed)]
        print(f"Starting matched CC-SF + Guard seed {seed}/9", flush=True)
        started = time.perf_counter()
        with (output / "workers.log").open("a", buffering=1) as log:
            result = subprocess.run(command, cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        with (output / "trial_results.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        recorded = any(int(row["seed"]) == seed for row in rows)
        if result.returncode != 0 and not recorded:
            row = dict.fromkeys(FIELDS, "")
            row.update(width_m=0.125, method="cc_sf", trial=seed + 1, seed=seed,
                       success=0, status="error", safe_stop_supported=True,
                       failure_reason=f"worker_exit_{result.returncode}",
                       wall_time_s=time.perf_counter() - started,
                       run_dir=f"width_125mm/cc_sf/seed_{seed}",
                       error=f"Isolated worker exited with code {result.returncode}")
            rows.append(row)
            write_csv(output / "trial_results.csv", rows)
            write_json(output / "progress.json", {"completed": len(rows),
                       "total": 10, "last": row})
        elif result.returncode != 0 or not recorded:
            raise RuntimeError(f"Unexpected worker exit: {result.returncode}")
        run = Path(f"width_125mm/cc_sf/seed_{seed}")
        if (output / run / "outcome.json").exists():
            actual = json.loads((output / run / "outcome.json").read_text())
            original = json.loads((reference / run / "outcome.json").read_text())
            actual_pose, expected_pose = actual.get("initial_pose"), original["initial_pose"]
            if actual_pose is None or len(actual_pose) != len(expected_pose):
                raise ValueError(f"Missing initial pose for seed {seed}")
            error = max(abs(a - b) for a, b in zip(actual_pose, expected_pose))
            integrity["start_pose_matches"].append({"seed": seed, "max_abs_error": error})
            if error > 1e-12:
                raise ValueError(f"Start pose mismatch for seed {seed}: {error}")
        write_json(output / "ablation_integrity.json", integrity)
        print(json.dumps(json.loads((output / "progress.json").read_text()),
                         ensure_ascii=False), flush=True)
    manifest = json.loads((output / "experiment.json").read_text())
    manifest["finished_at_kst"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    write_json(output / "experiment.json", manifest)


if __name__ == "__main__":
    main()
