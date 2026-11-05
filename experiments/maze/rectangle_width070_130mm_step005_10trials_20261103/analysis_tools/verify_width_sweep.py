#!/usr/bin/env python3
"""Independently reconcile all reported metrics using standard-library math."""

import argparse
import csv
import json
import math
from pathlib import Path


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentile(values, fraction):
    data = sorted(values)
    position = (len(data) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return data[lower] + (data[upper] - data[lower]) * (position - lower)


def verify(output):
    trials = rows(output / "trial_results.csv")
    summaries = rows(output / "benchmark_summary.csv")
    manifest = json.loads((output / "experiment.json").read_text())
    expected = len(manifest["widths_m"]) * len(manifest["methods"]) * manifest["trials_per_configuration"]
    assert len(trials) == expected
    assert len(summaries) == len(manifest["widths_m"]) * len(manifest["methods"])
    for summary in summaries:
        group = [trial for trial in trials
                 if float(trial["width_m"]) == float(summary["width_m"])
                 and trial["method"] == summary["method"]]
        assert len(group) == manifest["trials_per_configuration"]
        assert len({int(trial["seed"]) for trial in group}) == len(group)
        times, clearance = [], []
        successes = [trial for trial in group if trial["success"] == "1"]
        for trial in group:
            run = output / trial["run_dir"]
            if not (run / "outcome.json").exists():
                continue
            cycles = rows(run / "cycles.csv")
            times.extend(float(row["cycle_time_s"]) for row in cycles if row["included_in_aggregate"] == "1")
            clearance.extend(float(row["clearance_m"]) for row in rows(run / "clearance.csv"))
        calculated = {
            "completed_trials": len(group), "successful_trials": len(successes),
            "success_rate": len(successes) / len(group),
            "safe_stop_count": sum(int(trial["safe_stop_count"] or 0) for trial in group),
            "clearance_samples": len(clearance), "timing_samples": len(times),
            "mean_traversal_time_s": math.fsum(float(trial["final_time_s"]) for trial in successes) / len(successes) if successes else None,
            "mean_clearance_mm": math.fsum(clearance) / len(clearance) * 1000 if clearance else None,
            "p05_clearance_mm": percentile(clearance, 0.05) * 1000 if clearance else None,
            "mean_computation_time_ms": math.fsum(times) / len(times) * 1000 if times else None,
            "p95_computation_time_ms": percentile(times, 0.95) * 1000 if times else None,
        }
        for field, value in calculated.items():
            if value is None:
                assert not summary[field], (summary["width_m"], summary["method"], field)
            else:
                assert math.isclose(value, float(summary[field]), rel_tol=1e-10, abs_tol=1e-10), (summary["width_m"], summary["method"], field)
    result = {"verified_trials": len(trials), "verified_configurations": len(summaries),
              "independent_implementation": "stdlib fsum and sorted linear interpolation; all groups",
              "all_metrics_reconcile": True}
    (output / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    verify(parser.parse_args().results_dir.resolve())
