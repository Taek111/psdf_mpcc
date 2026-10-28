"""Compare archived runs using the same polygon clearance and sample scope."""

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BENCHMARKS = ROOT.parent


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def stats(values):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    ordered = sorted(values)
    rank = 0.05 * (len(ordered) - 1)
    low, high = math.floor(rank), math.ceil(rank)
    independently_interpolated = ordered[low] + (rank - low) * (ordered[high] - ordered[low])
    assert math.isclose(independently_interpolated, np.percentile(values, 5), abs_tol=1e-14)
    assert math.isclose(math.fsum(values) / len(values), float(np.mean(values)), abs_tol=1e-14)
    return {
        "samples": len(values),
        "mean_mm": float(np.mean(values) * 1000),
        "p05_mm": float(np.percentile(values, 5) * 1000),
        "min_mm": float(np.min(values) * 1000),
    }


def main():
    geometry = json.loads((ROOT / "geometry.json").read_text())
    previous = BENCHMARKS / "maze_rectangle_dd_perturb_four_methods_5trials_20261026_165452"
    assert geometry == json.loads((previous / "geometry.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "clearance_helper", ROOT / "source_snapshot/sim/start_perturbation.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    footprint = np.asarray(geometry["footprint"])
    obstacles = [np.asarray(vertices) for vertices in geometry["obstacles"]]
    batches = [
        ("cc_mf_imp", ROOT / "baseline"),
        ("cc_sf_four_methods", previous),
        ("cc_sf_safe_stop", BENCHMARKS / "cc_sf_safe_stop_5trials_20261026_173620"),
        ("mpcc", previous),
        ("cc_ecs", previous),
    ]
    summaries, per_trial, inputs, starts = [], [], {}, {}
    for name, batch in batches:
        method = "cc_sf" if name.startswith("cc_sf_") else name
        selected = [trial for trial in rows(batch / "trial_results.csv")
                    if trial["optimizer_type"] == method]
        assert len(selected) == 5
        pooled = []
        times = []
        starts[name] = []
        for trial in selected:
            assert trial["status"] == "success"
            assert (trial["maze_type"], trial["robot_shape"], trial["dynamics_type"], trial["path_planner"]) == (
                "maze", "rectangle", "differential_drive", "astar"
            )
            initial = [float(trial[key]) for key in ("initial_x", "initial_y", "initial_theta")]
            starts[name].append([int(trial["start_seed"]), *initial])
            history_path = Path(trial["output_root_dir"]) / "data" / f"trial_history_{trial['current_name']}.csv"
            inputs[str(history_path)] = hashlib.sha256(history_path.read_bytes()).hexdigest()
            history = rows(history_path)
            assert len(history) == round(float(trial["final_time"]) / geometry["dt_s"])
            poses = [initial] + [[float(row[key]) for key in ("x", "y", "theta")] for row in history]
            values = [helper.pose_clearance(pose, [footprint], obstacles, geometry["bounds"])
                      for pose in poses]
            pooled.extend(values)
            times.append(float(trial["final_time"]))
            per_trial.append({"batch": name, "seed": int(trial["start_seed"]), **stats(values)})
        summaries.append({"batch": name, "source": str(batch), "trials": 5,
                          "successes": 5, "mean_traversal_time_s": float(np.mean(times)), **stats(pooled)})
    assert all(value == starts["cc_mf_imp"] for value in starts.values())
    archived_summary = {row["optimizer_type"]: row for row in rows(previous / "benchmark_summary.csv")}
    for summary in summaries:
        name = summary["batch"]
        if name == "cc_mf_imp":
            continue
        old = archived_summary["cc_sf" if name.startswith("cc_sf_") else name]
        for metric, archived_metric in (("mean_mm", "clearance_m_mean"),
                                         ("p05_mm", "clearance_m_p05"),
                                         ("min_mm", "clearance_m_min")):
            assert math.isclose(summary[metric], 1000 * float(old[archived_metric]), abs_tol=1e-10)
        assert math.isclose(summary["mean_traversal_time_s"], float(old["traversal_time_mean_s"]), abs_tol=1e-10)
    result = {
        "definition": "Exact rectangle footprint distance to obstacles and map boundaries at initial pose and every executed logged pose (dt=0.1 s), including final pose. Pool all five trials; percentile uses linear interpolation. No solver-success sample filtering.",
        "same_geometry_and_initial_poses": True,
        "summaries": summaries,
        "per_trial": per_trial,
        "source_history_sha256": inputs,
    }
    (ROOT / "clearance_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (ROOT / "clearance_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({"summaries": summaries, "per_trial": per_trial}, indent=2))


if __name__ == "__main__":
    main()
