#!/usr/bin/env python3
"""Run sequential, isolated maze trials using current benchmark defaults."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import yaml

from run_width_benchmark import (
    METHODS, ROOT, run_benchmark, runtime_environment, write_csv, write_json,
)


WIDTHS = tuple(mm / 1000 for mm in range(70, 131, 5))
TRIALS = 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--widths", nargs="+", type=float, default=WIDTHS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--reference-run", type=Path)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    if any(not math.isfinite(width) or width <= 0 for width in args.widths):
        parser.error("--widths must contain finite positive values")
    if len(set(args.widths)) != len(args.widths) or len(set(args.methods)) != len(args.methods):
        parser.error("Duplicate widths or methods")
    runtime_environment()
    widths, methods, repeats = tuple(args.widths), tuple(args.methods), args.trials
    output = args.results_dir.resolve()
    config_path = ROOT / "config/config_benchmark.yaml"
    config = yaml.safe_load(config_path.read_text())
    if args.reference_run:
        reference = args.reference_run.resolve()
        reference_hashes = json.loads((reference / "source_hashes.json").read_text())
        for name, digest in reference_hashes.items():
            if name.split("/")[0] in ("control", "models", "planning", "sim", "utils", "config") or name == "test_nmpc.py":
                assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    run_benchmark(output, repeats, only_seed=-1, widths=widths,
                  current_defaults=True)
    manifest = json.loads((output / "experiment.json").read_text())
    total = len(widths) * len(methods) * repeats
    manifest.update(
        methods=methods,
        requested_trials=total, trial_process_isolation=True,
        simulation_limit_s=config["defaults"]["simulation_time"],
        failure_criteria=config["trial_failure_criteria"],
        generate_animation=False,
        configuration_source="config/config_benchmark.yaml",
    )
    manifest["min_clearance_by_width_m"] = {
        f"{width:.3f}": config["start_perturbation"]["min_clearance"]
        for width in widths
    }
    if args.reference_run:
        baseline = json.loads((reference / "experiment.json").read_text())
        assert all(manifest["resolved_defaults"][method] == baseline["resolved_defaults"][method]
                   for method in methods)
        manifest["reference_run"] = str(reference.relative_to(ROOT))
        manifest["simulation_sources_match_reference"] = True
    write_json(output / "experiment.json", manifest)
    fields = (
        "width_m", "method", "trial", "seed", "success", "status", "failure_reason",
        "final_time_s", "safe_stop_count", "safe_stop_supported", "steps",
        "clearance_samples", "wall_time_s", "run_dir", "error",
    )
    with (output / "trial_results.csv").open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
    write_json(output / "progress.json", {"completed": 0, "total": total})
    environment = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6", MKL_NUM_THREADS="6")
    for width in widths:
        for seed in range(repeats):
            order = methods[seed % len(methods):] + methods[:seed % len(methods)]
            for method in order:
                command = [sys.executable, "-u", str(output / "source_snapshot/run_width_benchmark.py"),
                           "--results-dir", str(output), "--trials", str(repeats), "--resume",
                           "--only-width", str(width), "--only-method", method,
                           "--only-seed", str(seed)]
                # The snapshot harness resolves ROOT from its own location. Its
                # imports and sources are frozen; git commit metadata is inherited.
                started = time.perf_counter()
                with (output / "workers.log").open("a", buffering=1) as log:
                    result = subprocess.run(command, cwd=ROOT, env=environment,
                                            stdout=log, stderr=subprocess.STDOUT)
                with (output / "trial_results.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                recorded = any(float(row["width_m"]) == width and row["method"] == method
                               and int(row["seed"]) == seed for row in rows)
                if result.returncode != 0 and not recorded:
                    row = dict.fromkeys(fields, "")
                    row.update(width_m=width, method=method, trial=seed + 1, seed=seed,
                               success=0, status="error",
                               failure_reason=f"worker_exit_{result.returncode}",
                               safe_stop_supported=method != "mpcc",
                               wall_time_s=time.perf_counter() - started,
                               run_dir=f"width_{round(width * 1000):03d}mm/{method}/seed_{seed}",
                               error=f"Isolated trial exited with code {result.returncode}")
                    rows.append(row)
                    write_csv(output / "trial_results.csv", rows)
                    write_json(output / "progress.json",
                               {"completed": len(rows), "total": total, "last": row})
                elif result.returncode != 0 or not recorded:
                    raise RuntimeError(f"Unexpected worker exit: {result.returncode}")
                print(json.dumps(json.loads((output / "progress.json").read_text()),
                                 ensure_ascii=False), flush=True)
        subprocess.run([sys.executable, str(ROOT / "analyze_width_sweep.py"),
                        "--results-dir", str(output)], cwd=ROOT, env=environment, check=True)
    manifest = json.loads((output / "experiment.json").read_text())
    manifest["finished_at_kst"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    write_json(output / "experiment.json", manifest)


if __name__ == "__main__":
    main()
