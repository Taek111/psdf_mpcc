"""Execute the user's final six-width list sequentially, with early stopping."""

import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
WIDTHS = [.105, .110, .115, .120, .125, .130]


def save_progress(records):
    (ROOT / "sweep_progress.json").write_text(json.dumps({
        "updated_at_kst": dt.datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "requested_widths_m": WIDTHS, "trials_per_width": 3,
        "seeds": [0, 1, 2], "records": records,
    }, indent=2) + "\n")


def main():
    baseline = ROOT / "width_100mm/environment.json"
    assert baseline.exists() and "finished_at_kst" in json.loads(baseline.read_text()), \
        "The baseline process must finish before starting this sequential sweep."
    metadata_path = ROOT / "experiment.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(requested_widths_m=WIDTHS, reference_width_m=.100,
                    final_user_width_list="0.105, 0.11, 0.115, 0.12, 0.125, 0.13 m")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
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
            save_progress(records)
            continue
        label = f"width_{round(width * 1000):03d}mm"
        print(f"Starting {label}: three trials, seeds 0-2", flush=True)
        records.append({"width_m": width, "status": "running"})
        save_progress(records)
        with (ROOT / f"{label}.run.log").open("w") as log:
            result = subprocess.run([sys.executable, "-u", str(ROOT / "run_benchmark.py"),
                                     "--width", str(width)], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            records[-1].update(status="execution_error", returncode=result.returncode)
            save_progress(records)
            raise RuntimeError(f"{label}: inspect {label}.run.log")
        with (ROOT / label / "trial_results.csv").open(newline="") as stream:
            trials = list(csv.DictReader(stream))
        assert len(trials) == 3
        successes = sum(row["status"] == "success" for row in trials)
        outcomes = [row["status"] for row in trials]
        records[-1].update(status="completed", successes=successes, trial_statuses=outcomes)
        save_progress(records)
        for line in (ROOT / f"{label}.run.log").read_text().splitlines():
            if "WIDTH_RESULT" in line:
                print(line, flush=True)
        if any(status not in ("success", "failure", "timeout") for status in outcomes):
            raise RuntimeError(f"Unexpected trial status: {outcomes}")
        if successes == 0:
            skip_reason = f"All three trials at width {width:.3f} m failed to reach the goal."
            print(skip_reason + " Skipping remaining larger widths.", flush=True)
    print("Requested sweep finished.", flush=True)


if __name__ == "__main__":
    main()
