"""Compare this 0.08 m rectangle batch against retained 0.10 m runs."""

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OLD_FOUR = ROOT.parent / "maze_rectangle_dd_perturb_four_methods_5trials_20261026_165452"
OLD_IMP = ROOT.parent / "cc_mf_imp_maze_rectangle_dd_5trials_20261027_180434"


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    current = rows(ROOT / "benchmark_summary.csv")
    old = {row["optimizer_type"]: row for row in rows(OLD_FOUR / "benchmark_summary.csv")}
    old_imp = next(row for row in rows(OLD_IMP / "clearance_comparison.csv") if row["batch"] == "cc_mf_imp")
    old["cc_mf_imp"] = {
        "clearance_m_mean": float(old_imp["mean_mm"]) / 1000,
        "clearance_m_p05": float(old_imp["p05_mm"]) / 1000,
        "clearance_m_min": float(old_imp["min_mm"]) / 1000,
        "traversal_time_mean_s": float(old_imp["mean_traversal_time_s"]),
        "successful_trials": old_imp["successes"],
    }
    samples = rows(ROOT / "clearance_samples.csv")
    current_trials = rows(ROOT / "trial_results.csv")
    old_trials = rows(OLD_FOUR / "trial_results.csv") + rows(OLD_IMP / "baseline/trial_results.csv")
    initial_poses = {}
    comparison = []
    for row in current:
        method = row["optimizer_type"]
        baseline = old[method]
        values = [float(sample["clearance_m"]) for sample in samples if sample["optimizer_type"] == method]
        assert math.isclose(math.fsum(values) / len(values), float(row["clearance_m_mean"]), abs_tol=1e-14)
        ordered = sorted(values)
        index = 0.05 * (len(values) - 1)
        lo, hi = math.floor(index), math.ceil(index)
        p05 = ordered[lo] + (index - lo) * (ordered[hi] - ordered[lo])
        assert math.isclose(p05, float(row["clearance_m_p05"]), abs_tol=1e-14)
        old_by_seed = {int(trial["start_seed"]): trial for trial in old_trials if trial["optimizer_type"] == method}
        pose_checks = []
        for trial in current_trials:
            if trial["optimizer_type"] != method:
                continue
            seed = int(trial["start_seed"])
            pose = [float(trial[key]) for key in ("initial_x", "initial_y", "initial_theta")]
            old_pose = [float(old_by_seed[seed][key]) for key in ("initial_x", "initial_y", "initial_theta")]
            pose_checks.append({"seed": seed, "pose_w008": pose, "pose_w010": old_pose,
                                "matched": bool(np.allclose(pose, old_pose, rtol=0, atol=1e-14))})
        initial_poses[method] = pose_checks
        result = {"method": method, "width_m": 0.08,
                  "successes": int(row["successful_trials"]), "trials": int(row["trials"])}
        for metric, output, scale in (
            ("clearance_m_mean", "clearance_mean_mm", 1000),
            ("clearance_m_p05", "clearance_p05_mm", 1000),
            ("clearance_m_min", "clearance_min_mm", 1000),
            ("traversal_time_mean_s", "traversal_mean_s", 1),
        ):
            value = float(row[metric])
            prior = float(baseline[metric])
            result[output] = value * scale
            result["w010_" + output] = prior * scale
            result["delta_" + output] = (value - prior) * scale
        comparison.append(result)
    geometry = json.loads((ROOT / "geometry.json").read_text())
    vertices = np.asarray(geometry["footprint"])
    assert np.allclose(np.ptp(vertices, axis=0), [0.15, 0.08])
    result = {
        "geometry": {"length_m": 0.15, "width_m": 0.08},
        "clearance_definition": "Exact polygon distance at initial and every executed logged pose (dt=0.1 s), including final pose; pooled across all five trials. P05 uses linear interpolation.",
        "comparison": comparison,
        "initial_pose_comparison": initial_poses,
        "same_start_poses_between_widths": all(pose["matched"] for group in initial_poses.values() for pose in group),
        "sources": {"width_008": str(ROOT), "width_010_four_methods": str(OLD_FOUR), "width_010_cc_mf_imp": str(OLD_IMP)},
    }
    with (ROOT / "width_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    (ROOT / "width_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
