"""Pool all recorded poses and reconcile safe stops against independent logs."""

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
import yaml


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
WIDTHS = [.115, .120, .125, .130]
STOP_MODES = ("safe_stop", "safe_stop_stage0")
SF_REFERENCE = PROJECT / "benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811"


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(name, records, fields=None):
    with (ROOT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def distribution(values):
    assert values and np.isfinite(values).all()
    mean = float(np.mean(values))
    p05 = float(np.percentile(values, 5))
    assert math.isclose(mean, math.fsum(values) / len(values), abs_tol=1e-14)
    ordered = sorted(values)
    rank = .05 * (len(values) - 1)
    low, high = math.floor(rank), math.ceil(rank)
    independent = ordered[low] + (rank - low) * (ordered[high] - ordered[low])
    assert math.isclose(p05, independent, abs_tol=1e-14)
    return {"mean_mm": mean * 1000, "p05_mm": p05 * 1000,
            "min_mm": min(values) * 1000, "samples": len(values)}


def intervals(cycles):
    result, start = [], None
    for index in range(len(cycles) + 1):
        stopped = index < len(cycles) and cycles[index]["mode"] in STOP_MODES
        if stopped and start is None:
            start = index
        elif not stopped and start is not None:
            result.append((start, index))
            start = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "frozen_clearance", ROOT / "source_snapshot/sim/start_perturbation.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    summaries, trials_out, samples, events, validations, violations = [], [], [], [], [], []
    inputs, common_starts = {}, {}
    common_geometry = common_params = None
    for width in WIDTHS:
        folder = ROOT / f"width_{round(width * 1000):03d}mm"
        metadata_file = folder / "environment.json"
        if not metadata_file.exists():
            continue
        metadata = json.loads(metadata_file.read_text())
        if "finished_at_kst" not in metadata:
            continue
        assert metadata["snapshot_changed"] == []
        config = yaml.safe_load((folder / "config.yaml").read_text())
        assert config["trials"] == 3
        assert config["start_perturbation"]["min_clearance"] == .001
        params = json.loads((folder / "optimizer_parameters.json").read_text())
        assert params["d_col"] == params["d_min"] == .0002
        assert "d_mf_mask" not in params
        if common_params is None:
            common_params = params
        assert params == common_params
        geometry = json.loads((folder / "geometry.json").read_text())
        footprint = np.asarray(geometry["footprint"])
        np.testing.assert_allclose(np.ptp(footprint, axis=0), [.15, width], atol=1e-14, rtol=0)
        fixed_geometry = {key: value for key, value in geometry.items() if key != "footprint"}
        if common_geometry is None:
            common_geometry = fixed_geometry
        assert fixed_geometry == common_geometry
        obstacles = [np.asarray(vertices) for vertices in geometry["obstacles"]]
        shapes = [Polygon(vertices) for vertices in obstacles]
        lower, upper = np.asarray(geometry["bounds"])
        expected = {entry["seed"]: entry for entry in
                    json.loads((folder / "expected_start_poses.json").read_text())}
        sf_expected = {entry["seed"]: entry for entry in json.loads(
            (SF_REFERENCE / folder.name / "expected_start_poses.json").read_text())}
        trials = rows(folder / "trial_results.csv")
        assert len(trials) == 3 and {int(row["start_seed"]) for row in trials} == {0, 1, 2}
        all_cycles = rows(folder / "cycles.csv")
        width_values, width_metrics, solver_times = [], [], []
        for trial in trials:
            assert trial["status"] in ("success", "failure", "timeout")
            seed = int(trial["start_seed"])
            runtime = json.loads((folder / f"runtime_seed{seed}.json").read_text())
            stats = runtime["optimizer_runtime_stats"]
            pose0 = np.array([float(trial[key]) for key in
                              ("initial_x", "initial_y", "initial_theta")])
            for target in (expected[seed]["pose"], sf_expected[seed]["pose"],
                           common_starts.setdefault(seed, pose0)):
                np.testing.assert_allclose(pose0, target, atol=1e-14, rtol=0)
            data = folder / "runs" / trial["configuration_id"] / \
                f"trial_{int(trial['trial_number']):04d}" / "data"
            history_path = data / f"trial_history_{trial['current_name']}.csv"
            history = rows(history_path)
            native = rows(data / f"cc_mf_icdf_cycle_{trial['current_name']}.csv")
            pose_sdf = rows(data / f"pose_sdf_{trial['current_name']}.csv")
            cycles = [row for row in all_cycles if int(row["seed"]) == seed]
            n = round(float(trial["final_time"]) / .1)
            assert len(history) == len(native) == len(pose_sdf) == len(cycles) == n
            assert stats["plant_input_apply_count"] == runtime["plant_input_apply_count"] == n
            poses = [pose0] + [np.array([float(row[key]) for key in ("x", "y", "theta")])
                               for row in history]
            values, overlap_count, boundary_count, contact_count = [], 0, 0, 0
            for index, pose in enumerate(poses):
                value = helper.pose_clearance(pose, [footprint], obstacles, geometry["bounds"])
                values.append(value)
                cosine, sine = np.cos(pose[2]), np.sin(pose[2])
                vertices = footprint @ np.array([[cosine, sine], [-sine, cosine]]) + pose[:2]
                shape = Polygon(vertices)
                boundary = min(float(np.min(vertices - lower)), float(np.min(upper - vertices)))
                distances = [shape.distance(other) for other in shapes]
                alternative = min(boundary, min(distances))
                assert math.isclose(value, alternative, abs_tol=1e-12)
                touched = [i for i, distance in enumerate(distances) if distance == 0]
                area = max((shape.intersection(shapes[i]).area for i in touched), default=0.)
                overlap = area > 1e-16
                contact = bool(touched) and not overlap
                overlap_count += int(overlap)
                contact_count += int(contact)
                boundary_count += int(boundary < 0)
                sample = {"width_m": width, "seed": seed, "sample_index": index,
                          "time_s": index * .1, "x": pose[0], "y": pose[1], "theta": pose[2],
                          "clearance_m": value}
                samples.append(sample)
                if value <= 0:
                    violations.append({**sample, "obstacle_overlap": overlap,
                        "contact_only": contact, "boundary_violation": boundary < 0,
                        "intersection_area_m2": area})
            assert values[0] >= .001
            assert math.isclose(values[0], expected[seed]["details"]["initial_clearance"],
                                abs_tol=1e-12)
            cumulative = 0
            for index, (history_row, cycle, native_row) in enumerate(zip(history, cycles, native)):
                assert int(cycle["cycle"]) == int(history_row["timestep"]) == index
                assert math.isclose(float(history_row["time"]), (index + 1) * .1, abs_tol=1e-9)
                assert math.isclose(float(native_row["time"]), index * .1, abs_tol=1e-9)
                assert cycle["mode"] == native_row["solve_mode"]
                assert cycle["main_status"] == native_row["solver_status"]
                np.testing.assert_allclose([float(pose_sdf[index][key]) for key in
                                           ("x", "y", "theta")], poses[index], atol=1e-12, rtol=0)
                stopped = cycle["mode"] in STOP_MODES
                assert int(cycle["safe_stop"]) == int(stopped)
                assert (history_row["solver_success"] == "True") == (not stopped)
                cumulative += int(stopped)
                assert int(cycle["cumulative_safe_stops"]) == cumulative
                if stopped:
                    assert float(history_row["v"]) == float(history_row["omega"]) == 0
                    assert np.all(np.asarray(json.loads(native_row["u0"])) == 0)
                    np.testing.assert_array_equal(poses[index], poses[index + 1])
                else:
                    solver_times.append(float(history_row["solver_computation_time"]))
            modes = Counter(cycle["mode"] for cycle in cycles)
            stopped = sum(modes[mode] for mode in STOP_MODES)
            assert stopped == stats["safe_stop_count"] == runtime["safe_stop_count"]
            assert modes["safe_stop_stage0"] == stats["stage0_guard_stop_count"]
            assert modes["backup_feasible_qp"] == stats["backup_feasible_qp_success_count"]
            assert sum(row["main_status"] not in ("", "0") for row in cycles) == \
                stats["main_solver_failure_count"]
            assert int(trial["successful_solver_steps"]) == n - stopped
            spans = intervals(cycles)
            assert sum(end - start for start, end in spans) == stopped
            for start, end in spans:
                events.append({"width_m": width, "seed": seed, "start_time_s": start * .1,
                    "end_time_s": end * .1, "steps": end - start, "duration_s": (end - start) * .1,
                    "modes": "/".join(sorted({row["mode"] for row in cycles[start:end]})),
                    "first_main_status": cycles[start]["main_status"],
                    "initial_signed_psdf_m": cycles[start]["exact_current_phi_m"],
                    "initial_geometric_clearance_m": values[start]})
            metric = {"width_m": width, "seed": seed, "status": trial["status"],
                "failure_reason": trial["failure_reason"], "final_time_s": float(trial["final_time"]),
                "control_steps": n, **{f"clearance_{key}": value for key, value in distribution(values).items()},
                "safe_stop_steps": stopped, "stage0_guard_stop_steps": modes["safe_stop_stage0"],
                "solver_failure_stop_steps": modes["safe_stop"], "safe_stop_episodes": len(spans),
                "safe_stop_duration_s": stopped * .1,
                "max_consecutive_safe_stop_steps": max((b - a for a, b in spans), default=0),
                "obstacle_overlap_samples": overlap_count, "obstacle_contact_only_samples": contact_count,
                "boundary_violation_samples": boundary_count,
                "nonpositive_clearance_samples": sum(value <= 0 for value in values),
                "below_d_col_samples": sum(value < params["d_col"] - 1e-12 for value in values)}
            trials_out.append(metric)
            width_metrics.append(metric)
            width_values.extend(values)
            validations.append({"width_m": width, "seed": seed, "solve_modes": dict(modes),
                "native_logs_runtime_histories_reconciled": True, "safe_stop_inputs_zero": True,
                "initial_and_final_poses_included": True,
                "same_seed_pose_across_widths_and_sf_reference": True,
                "independent_shapely_samples_checked": len(values)})
            for path in (history_path, data / f"cc_mf_icdf_cycle_{trial['current_name']}.csv",
                         folder / f"runtime_seed{seed}.json"):
                inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        saved_times = [float(row["solver_computation_time_s"]) for row in
                       rows(next(folder.glob("computation_times_*.csv")))]
        np.testing.assert_array_equal(solver_times, saved_times)
        summaries.append({"width_m": width, "trials": 3,
            "successes": sum(row["status"] == "success" for row in width_metrics),
            **{f"clearance_{key}": value for key, value in distribution(width_values).items()},
            "safe_stop_steps_seed012": "/".join(str(row["safe_stop_steps"]) for row in width_metrics),
            **{key: sum(row[key] for row in width_metrics) for key in (
                "safe_stop_steps", "stage0_guard_stop_steps", "solver_failure_stop_steps",
                "safe_stop_episodes", "obstacle_overlap_samples", "obstacle_contact_only_samples",
                "boundary_violation_samples", "nonpositive_clearance_samples", "below_d_col_samples")},
            "positive_clearance_goal_trials": sum(row["status"] == "success"
                and row["clearance_min_mm"] > 0 for row in width_metrics),
            "solver_successful_steps": len(saved_times)})
    progress = json.loads((ROOT / "sweep_progress.json").read_text())
    if not args.partial:
        assert len(progress) == len(WIDTHS)
        assert all(record["status"] in ("completed", "skipped") for record in progress)
        assert {row["width_m"] for row in summaries} == {
            row["width_m"] for row in progress if row["status"] == "completed"}
        for index, record in enumerate(progress):
            if record["status"] == "skipped":
                assert any(previous.get("successes") == 0 for previous in progress[:index])
    hashes = json.loads((ROOT / "source_hashes.json").read_text())
    changed = [name for name, digest in hashes.items()
               if hashlib.sha256((ROOT / "source_snapshot" / name).read_bytes()).hexdigest() != digest]
    assert not changed
    if summaries:
        save("benchmark_summary.csv", summaries)
        save("benchmark_trials.csv", trials_out)
        save("clearance_samples.csv", samples)
    save("safe_stop_events.csv", events, ["width_m", "seed", "start_time_s", "end_time_s", "steps",
        "duration_s", "modes", "first_main_status", "initial_signed_psdf_m", "initial_geometric_clearance_m"])
    save("geometry_violations.csv", violations, ["width_m", "seed", "sample_index", "time_s", "x",
        "y", "theta", "clearance_m", "obstacle_overlap", "contact_only", "boundary_violation",
        "intersection_area_m2"])
    (ROOT / "validation.json").write_text(json.dumps({
        "partial": args.partial, "checks": validations, "frozen_sources_changed": changed,
        "input_sha256": inputs, "mean_p05_independently_recomputed": True,
        "common_start_poses": {seed: pose.tolist() for seed, pose in common_starts.items()},
        "progress": progress}, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
