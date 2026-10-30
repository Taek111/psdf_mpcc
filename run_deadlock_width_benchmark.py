#!/usr/bin/env python3
"""Run the requested 70--130 mm sweep sequentially with isolated trial processes."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_width_benchmark import (
    METHODS, ROOT, run_benchmark, runtime_environment, write_csv, write_json,
)


WIDTHS = tuple(value / 1000 for value in range(70, 131, 10))
FIELDS = (
    "width_m", "method", "trial", "seed", "success", "status", "failure_reason",
    "final_time_s", "safe_stop_count", "safe_stop_supported", "steps",
    "clearance_samples", "wall_time_s", "run_dir", "error",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()
    runtime_environment()
    output = args.results_dir.resolve()
    # Initialize the snapshot and manifest without running a trial.
    run_benchmark(output, 10, generate_animation=True, only_seed=-1,
                  widths=WIDTHS, current_defaults=True)
    with (output / "trial_results.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
    manifest = json.loads((output / "experiment.json").read_text())
    manifest["trial_process_isolation"] = True
    manifest["requested_trials"] = 280
    manifest["failure_criteria"] = {"movement_window_sec": 4.0,
        "movement_threshold_m": 0.03, "max_consecutive_solver_failures": 20}
    manifest["simulation_limit_s"] = 60.0
    manifest["animation_frame_skip"] = 5  # Same as test_nmpc.py's CLI default.
    # User-approved exception: 130 mm cannot be perturbed with 10 mm on each side
    # of the 150 mm starting corridor.
    manifest["min_clearance_by_width_m"] = {"0.130": 0.001}
    write_json(output / "experiment.json", manifest)
    write_json(output / "progress.json", {"completed": 0, "total": 280})
    environment = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6",
                       MKL_NUM_THREADS="6")
    for width in WIDTHS:
        for seed in range(10):
            order = METHODS[seed % 4:] + METHODS[:seed % 4]
            for method in order:
                command = [sys.executable, "-u", str(ROOT / "run_width_benchmark.py"),
                    "--results-dir", str(output), "--trials", "10", "--resume",
                    "--generate-animation", "--only-width", str(width),
                    "--only-method", method, "--only-seed", str(seed)]
                started = time.perf_counter()
                with (output / "workers.log").open("a", buffering=1) as log:
                    result = subprocess.run(command, cwd=ROOT, env=environment,
                                            stdout=log, stderr=subprocess.STDOUT)
                if result.returncode != 0:
                    # Preserve native failure as an error; never silently retry a trial.
                    with (output / "trial_results.csv").open(newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    if any(float(row["width_m"]) == width and row["method"] == method
                           and int(row["seed"]) == seed for row in rows):
                        raise RuntimeError(f"Worker failed after recording its trial: {result.returncode}")
                    row = dict.fromkeys(FIELDS, "")
                    row.update(width_m=width, method=method, trial=seed + 1, seed=seed,
                        success=0, status="error", failure_reason=f"worker_exit_{result.returncode}",
                        safe_stop_supported=method != "mpcc",
                        wall_time_s=time.perf_counter() - started,
                        run_dir=f"width_{round(width * 1000):03d}mm/{method}/seed_{seed}",
                        error=f"Isolated trial exited with code {result.returncode}")
                    rows.append(row)
                    write_csv(output / "trial_results.csv", rows)
                    write_json(output / "progress.json",
                               {"completed": len(rows), "total": 280, "last": row})
                    print(f"ERROR width={width} method={method} seed={seed} "
                          f"exit={result.returncode}", flush=True)
                progress = json.loads((output / "progress.json").read_text())
                print(json.dumps(progress, ensure_ascii=False), flush=True)
        analyzer = ROOT / "analyze_deadlock_width_benchmark.py"
        if analyzer.exists():
            subprocess.run([sys.executable, str(analyzer), "--results-dir", str(output)],
                           cwd=ROOT, check=True, env=environment)
    manifest = json.loads((output / "experiment.json").read_text())
    from datetime import datetime
    from zoneinfo import ZoneInfo
    manifest["finished_at_kst"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    write_json(output / "experiment.json", manifest)


if __name__ == "__main__":
    main()
