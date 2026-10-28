"""Rebuild evaluation metrics from this experiment's retained CSVs."""

import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "count": len(values),
        "min": float(np.min(values)) if len(values) else None,
        "mean": float(np.mean(values)) if len(values) else None,
        "p95": float(np.percentile(values, 95)) if len(values) else None,
        "max": float(np.max(values)) if len(values) else None,
    }


def main():
    geometry = json.loads((ROOT / "geometry.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "clearance_helper", ROOT / "source_snapshot/sim/start_perturbation.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    footprint = np.asarray(geometry["footprint"])
    obstacles = [np.asarray(vertices) for vertices in geometry["obstacles"]]
    results = []
    for batch in sorted(ROOT.iterdir()):
        trial_path = batch / "trial_results.csv"
        if not trial_path.exists():
            continue
        trials = []
        for trial in read_csv(trial_path):
            data = Path(trial["output_root_dir"]) / "data"
            history = read_csv(next(data.glob("trial_history_*.csv")))
            pose_sdf = read_csv(next(data.glob("pose_sdf_*.csv")))
            cycles = read_csv(next(data.glob("cc_mf_imp_cycle_*.csv")))
            roots = read_csv(next(data.glob("cc_mf_imp_roots_*.csv")))
            poses = [[float(trial[k]) for k in ("initial_x", "initial_y", "initial_theta")]]
            poses += [[float(row[k]) for k in ("x", "y", "theta")] for row in history]
            clearance = [
                helper.pose_clearance(pose, [footprint], obstacles, geometry["bounds"])
                for pose in poses
            ]
            slacks = [
                value for row in cycles
                for value in json.loads(row["mf_imp_slack_m"])
                if value is not None
            ]
            required_roots = [row for row in roots if row["mf_active"] == "True"]
            trials.append({
                "trial_number": int(trial["trial_number"]),
                "seed": int(trial["start_seed"]),
                "status": trial["status"],
                "failure_reason": trial["failure_reason"],
                "final_time_s": float(trial["final_time"]),
                "min_logged_psdf_m": min(float(row["sdf_value"]) for row in pose_sdf),
                "min_sampled_geometry_clearance_m": min(clearance),
                "sampled_contact_or_overlap_count": sum(value <= 0 for value in clearance),
                "solver_failure_steps": sum(row["solver_success"] == "False" for row in history),
                "solve_modes": dict(Counter(row["solve_mode"] for row in cycles)),
                "slack_m": distribution(slacks),
                "required_root_count": len(required_roots),
                "invalid_required_roots": sum(row["root_valid"] != "True" for row in required_roots),
                "root_status_counts": dict(Counter(row["root_status"] for row in roots)),
                "control_cycle_time_s": distribution(
                    [float(row["control_cycle_time_s"]) for row in cycles if row["control_cycle_time_s"]]
                ),
                "preprocessing_time_s": distribution(
                    [float(row["preprocessing_time_s"]) for row in cycles if row["preprocessing_time_s"]]
                ),
            })
        results.append({
            "batch": batch.name,
            "summary": read_csv(batch / "summary.csv"),
            "trials": trials,
        })
    report = {
        "geometry_clearance_note": (
            "Exact polygon distance at initial and executed logged poses, including map boundaries. "
            "Contact/overlap maps to zero; continuous inter-step collision checking is not performed."
        ),
        "timing_note": (
            "summary.csv pools only successful acados solve calls, including the first step. "
            "Control cycle and preprocessing times are logged separately."
        ),
        "batches": results,
    }
    (ROOT / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for result in results:
        for trial in result["trials"]:
            print(result["batch"], json.dumps(trial))


if __name__ == "__main__":
    main()
