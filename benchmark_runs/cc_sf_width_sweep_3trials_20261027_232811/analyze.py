"""Recompute exact clearance and validate safe-stop counts from raw histories."""

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
WIDTHS = [.105, .110, .115, .120, .125, .130]


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def distribution(values):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    mean, p05 = float(np.mean(values)), float(np.percentile(values, 5))
    assert math.isclose(mean, math.fsum(values) / len(values), abs_tol=1e-14)
    ordered = sorted(values)
    rank = .05 * (len(values) - 1)
    low, high = math.floor(rank), math.ceil(rank)
    independent = ordered[low] + (rank - low) * (ordered[high] - ordered[low])
    assert math.isclose(p05, independent, abs_tol=1e-14)
    return {"mean_mm": mean * 1000, "p05_mm": p05 * 1000,
            "min_mm": float(np.min(values)) * 1000, "samples": len(values)}


def stop_intervals(cycles):
    intervals = []
    start = None
    for index in range(len(cycles) + 1):
        stopped = index < len(cycles) and cycles[index]["mode"] == "safe_stop"
        if stopped and start is None:
            start = index
        elif not stopped and start is not None:
            intervals.append((start, index))
            start = None
    return intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "frozen_clearance", ROOT / "source_snapshot/sim/start_perturbation.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    try:
        from shapely.geometry import Polygon
    except ImportError:
        Polygon = None

    summaries, trial_metrics, samples, events, validations = [], [], [], [], []
    nominal_diagnostics, input_hashes = [], {}
    common_poses = {}
    parameter_reference = None
    geometry_reference = None
    for width in [.100] + WIDTHS:
        folder = ROOT / f"width_{round(width * 1000):03d}mm"
        environment_path = folder / "environment.json"
        if not environment_path.exists():
            continue
        environment = json.loads(environment_path.read_text())
        if "finished_at_kst" not in environment:
            continue
        assert not environment["snapshot_changed"]
        config = __import__("yaml").safe_load((folder / "config.yaml").read_text())
        assert config["start_perturbation"]["min_clearance"] == (.01 if width == .1 else .001)
        parameters = json.loads((folder / "optimizer_parameters.json").read_text())
        if parameter_reference is None:
            parameter_reference = parameters
        assert parameters == parameter_reference
        geometry = json.loads((folder / "geometry.json").read_text())
        footprint = np.asarray(geometry["footprint"], dtype=float)
        np.testing.assert_allclose(np.ptp(footprint, axis=0), [.15, width], rtol=0, atol=1e-14)
        fixed_geometry = {key: value for key, value in geometry.items() if key != "footprint"}
        if geometry_reference is None:
            geometry_reference = fixed_geometry
        assert fixed_geometry == geometry_reference
        obstacles = [np.asarray(vertices) for vertices in geometry["obstacles"]]
        shapely_obstacles = [Polygon(vertices) for vertices in obstacles] if Polygon else []
        expected = {entry["seed"]: entry for entry in
                    json.loads((folder / "expected_start_poses.json").read_text())}
        trials = rows(folder / "trial_results.csv")
        assert len(trials) == 3 and {int(row["start_seed"]) for row in trials} == {0, 1, 2}
        all_cycles = rows(folder / "cycles.csv")
        width_values, width_modes = [], Counter()
        width_safe, width_episodes, width_fast_failed, width_recovered = [], [], [], []
        width_contacts = 0
        solver_values = []
        for trial in trials:
            assert trial["status"] in ("success", "failure", "timeout")
            seed = int(trial["start_seed"])
            runtime = json.loads((folder / f"runtime_seed{seed}.json").read_text())
            pose0 = np.array([float(trial[key]) for key in
                              ("initial_x", "initial_y", "initial_theta")])
            np.testing.assert_allclose(pose0, expected[seed]["pose"], rtol=0, atol=1e-14)
            if seed not in common_poses:
                common_poses[seed] = pose0
            np.testing.assert_allclose(pose0, common_poses[seed], rtol=0, atol=1e-14)
            data = folder / "runs" / trial["configuration_id"] / \
                f"trial_{int(trial['trial_number']):04d}" / "data"
            history_path = data / f"trial_history_{trial['current_name']}.csv"
            history = rows(history_path)
            input_hashes[str(history_path.relative_to(ROOT))] = hashlib.sha256(
                history_path.read_bytes()).hexdigest()
            psdf = rows(data / f"pose_sdf_{trial['current_name']}.csv")
            cycles = [row for row in all_cycles if int(row["seed"]) == seed]
            n = round(float(trial["final_time"]) / .1)
            assert len(history) == len(psdf) == len(cycles) == n
            assert runtime["plant_input_apply_count"] == n
            poses = [pose0] + [np.array([float(row[key]) for key in ("x", "y", "theta")])
                               for row in history]
            values = []
            independent_checks = 0
            for index, pose in enumerate(poses):
                value = helper.pose_clearance(pose, [footprint], obstacles, geometry["bounds"])
                values.append(value)
                if Polygon and (index % 10 == 0 or value <= 0 or index == len(poses) - 1):
                    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
                    rotation = np.array([[cosine, -sine], [sine, cosine]])
                    vertices = footprint @ rotation.T + pose[:2]
                    shape = Polygon(vertices)
                    lower, upper = np.asarray(geometry["bounds"])
                    alternative = min(float(np.min(vertices - lower)),
                                      float(np.min(upper - vertices)),
                                      min(shape.distance(other) for other in shapely_obstacles))
                    assert math.isclose(value, alternative, abs_tol=1e-12)
                    independent_checks += 1
                samples.append({"width_m": width, "seed": seed, "sample_index": index,
                                "time_s": index * .1, "x": pose[0], "y": pose[1],
                                "theta": pose[2], "clearance_m": value})
            assert math.isclose(values[0], expected[seed]["details"]["initial_clearance"],
                                abs_tol=1e-12)
            for index, (row, cycle) in enumerate(zip(history, cycles)):
                assert int(cycle["cycle"]) == int(row["timestep"]) == index
                assert math.isclose(float(row["time"]), (index + 1) * .1, abs_tol=1e-9)
                np.testing.assert_allclose([float(psdf[index][key]) for key in
                                           ("x", "y", "theta")], poses[index],
                                          rtol=0, atol=1e-12)
                stopped = cycle["mode"] == "safe_stop"
                assert int(cycle["safe_stop"]) == int(stopped)
                assert (row["solver_success"] == "True") == (not stopped)
                if stopped:
                    assert float(row["v"]) == float(row["omega"]) == 0.0
                else:
                    solver_values.append(float(row["solver_computation_time"]))
            modes = Counter(row["mode"] for row in cycles)
            stopped = modes["safe_stop"]
            recovered = modes["backup_feasible_qp"]
            assert runtime["safe_stop_count"] == stopped
            assert runtime["backup_feasible_qp_success_count"] == recovered
            assert int(trial["successful_solver_steps"]) == n - stopped
            intervals = stop_intervals(cycles)
            assert sum(end - start for start, end in intervals) == stopped
            for start, end in intervals:
                events.append({"width_m": width, "seed": seed,
                               "start_time_s": start * .1, "end_time_s": end * .1,
                               "steps": end - start, "duration_s": (end - start) * .1,
                               "first_fast_status": cycles[start]["fast_status"],
                               "first_backup_status": cycles[start]["backup_status"]})
            fast_failed = sum(int(row["fast_status"]) != 0 for row in cycles)
            contacts = sum(value <= 0 for value in values)
            stat = distribution(values)
            trial_metrics.append({"width_m": width, "role": "reference" if width == .1 else "requested",
                "seed": seed, "status": trial["status"], "failure_reason": trial["failure_reason"],
                "final_time_s": float(trial["final_time"]), "control_steps": n,
                **{f"clearance_{key}": value for key, value in stat.items()},
                "safe_stop_steps": stopped, "safe_stop_episodes": len(intervals),
                "safe_stop_duration_s": stopped * .1,
                "max_consecutive_safe_stop_steps": max((b - a for a, b in intervals), default=0),
                "fast_failure_steps": fast_failed, "backup_recovered_steps": recovered,
                "contact_or_boundary_violation_samples": contacts})
            validations.append({"width_m": width, "seed": seed,
                "initial_and_final_pose_included": True, "same_seed_pose_across_widths": True,
                "history_cycles_runtime_counts_match": True, "safe_stop_inputs_are_zero": True,
                "independent_shapely_clearance_samples_checked": independent_checks,
                "solve_modes": dict(modes),
                "fast_status_counts": dict(Counter(row["fast_status"] for row in cycles)),
                "backup_status_counts": dict(Counter(row["backup_status"] for row in cycles
                                                     if row["backup_status"] != ""))})
            width_values.extend(values)
            width_modes.update(modes)
            width_safe.append(stopped)
            width_episodes.append(len(intervals))
            width_fast_failed.append(fast_failed)
            width_recovered.append(recovered)
            width_contacts += contacts

        saved_solver = [float(row["solver_computation_time_s"]) for row in
                        rows(next(folder.glob("computation_times_*.csv")))]
        np.testing.assert_array_equal(solver_values, saved_solver)
        stat = distribution(width_values)
        successes = sum(row["status"] == "success" for row in trials)
        summaries.append({"width_m": width, "role": "reference" if width == .1 else "requested",
            "trials": 3, "successes": successes, "success_rate": successes / 3,
            **{f"clearance_{key}": value for key, value in stat.items()},
            "safe_stop_steps": sum(width_safe), "safe_stop_steps_seed012": "/".join(map(str, width_safe)),
            "safe_stop_episodes": sum(width_episodes),
            "safe_stop_episodes_seed012": "/".join(map(str, width_episodes)),
            "fast_failure_steps": sum(width_fast_failed),
            "backup_recovered_steps": sum(width_recovered),
            "contact_or_boundary_violation_samples": width_contacts,
            "solver_successful_steps": len(solver_values)})

        with (folder / "nominal_risk.jsonl").open() as stream:
            margin_summary = {}
            for line in stream:
                row = json.loads(line)
                seed = int(row["seed"])
                entry = margin_summary.setdefault((seed, row["kind"]), {
                    "margins": [], "stage0_margins": [], "residuals": [], "failed": 0,
                })
                entry["margins"].extend(row["risk_margin_m"])
                entry["stage0_margins"].append(row["risk_margin_m"][0])
                entry["residuals"].extend(row["nominal_residual_m"])
                entry["failed"] += row["status"] != 0
            for (seed, kind), entry in margin_summary.items():
                nominal_diagnostics.append({"width_m": width, "seed": seed, "kind": kind,
                    "failed_attempts": entry["failed"],
                    "nominal_margin_max_mm": max(entry["margins"]) * 1000,
                    "stage0_margin_min_mm": min(entry["stage0_margins"]) * 1000,
                    "stage0_margin_max_mm": max(entry["stage0_margins"]) * 1000,
                    "nominal_residual_min_mm": min(entry["residuals"]) * 1000})

    progress = json.loads((ROOT / "sweep_progress.json").read_text())
    requested_completed = {row["width_m"] for row in summaries if row["role"] == "requested"}
    if not args.partial:
        assert len(progress["records"]) == 6
        for record in progress["records"]:
            assert record["status"] in ("completed", "skipped")
            if record["status"] == "completed":
                assert record["width_m"] in requested_completed
    for name, records in (("benchmark_summary.csv", summaries), ("benchmark_trials.csv", trial_metrics),
                          ("clearance_samples.csv", samples), ("safe_stop_events.csv", events),
                          ("nominal_risk_summary.csv", nominal_diagnostics)):
        save_csv(ROOT / name, records)
    source_hashes = json.loads((ROOT / "source_hashes.json").read_text())
    changed = [name for name, digest in source_hashes.items()
               if hashlib.sha256((ROOT / "source_snapshot" / name).read_bytes()).hexdigest() != digest]
    assert not changed
    validation = {"partial": args.partial, "checks": validations,
        "frozen_sources_changed": changed, "source_history_sha256": input_hashes,
        "shapely_available": Polygon is not None,
        "common_seed_poses": {seed: pose.tolist() for seed, pose in common_poses.items()},
        "pooled_clearance_mean_and_p05_independently_recomputed": True,
        "pilot_results_excluded": True, "progress": progress}
    (ROOT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
