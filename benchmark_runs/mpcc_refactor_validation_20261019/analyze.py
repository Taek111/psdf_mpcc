"""Analyze saved nominal validation trials without creating any optimizer.

Python 3.8 compatible. Only simulation_mpc.create_env and geometry constructors
are used; no simulation, solver, generated-code cleanup or PSDF evaluation runs.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def jsonable(value):
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def command(*arguments):
    try:
        return subprocess.check_output(arguments, cwd=str(ROOT), stderr=subprocess.STDOUT,
                                       timeout=10).decode().strip()
    except (OSError, subprocess.SubprocessError) as error:
        return "Unavailable: {}".format(error)


def stats(values):
    values = np.asarray([float(value) if value else np.nan for value in values], dtype=float)
    finite = values[np.isfinite(values) & (values >= 0)]
    return {
        "count": len(values), "valid_count": len(finite),
        "mean_s": float(np.mean(finite)) if len(finite) else None,
        "median_s": float(np.median(finite)) if len(finite) else None,
        "p95_s": float(np.percentile(finite, 95)) if len(finite) else None,
        "max_s": float(np.max(finite)) if len(finite) else None,
    }


def strict_overlap(first, second, tolerance=1e-10):
    """SAT with unit axes: contact alone is not strict interior overlap."""
    edges = np.vstack((np.roll(first, -1, axis=0) - first,
                       np.roll(second, -1, axis=0) - second))
    axes = np.column_stack((-edges[:, 1], edges[:, 0]))
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > 1e-12] / lengths[lengths > 1e-12, None]
    a, b = first @ axes.T, second @ axes.T
    depths = np.minimum(a.max(axis=0), b.max(axis=0)) - np.maximum(a.min(axis=0), b.min(axis=0))
    return bool(np.all(depths > tolerance))


def analyze_trial(trial, results, output, simulation, footprints, pose_clearance):
    if trial["robot_shape"] != "rectangle" or trial["dynamics_type"] != "differential_drive":
        raise ValueError("This analyzer is limited to the requested rectangle/DD validation")
    trial_name = "{}_trial_{:04d}".format(trial["configuration_id"], int(trial["trial_number"]))
    data = results / "runs" / trial["configuration_id"] / (
        "trial_{:04d}".format(int(trial["trial_number"]))) / "data"
    history_path = data / ("trial_history_{}.csv".format(trial["current_name"]))
    rows = read_csv(history_path)
    initial = [float(trial["initial_" + key]) for key in ("x", "y", "theta")]
    poses = np.asarray([initial] + [[float(row[key]) for key in ("x", "y", "theta")]
                                  for row in rows])
    if not np.all(np.isfinite(poses)):
        raise ValueError("Nonfinite pose in {}".format(history_path))
    times = [0.0] + [float(row["time"]) for row in rows]
    _, goal, grid, regions = simulation.create_env(trial["maze_type"])
    obstacles = [np.asarray(region.get_ccw_vertices(), dtype=float) for region in regions]
    clearances, overlaps = [], []
    for index, pose in enumerate(poses):
        clearances.append(pose_clearance(pose, footprints, obstacles, grid[0]))
        cosine, sine = np.cos(pose[2]), np.sin(pose[2])
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        intersecting = [number for number, obstacle in enumerate(obstacles)
                        if any(strict_overlap(polygon @ rotation.T + pose[:2], obstacle)
                               for polygon in footprints)]
        overlaps.append(intersecting)
    with (output / (trial_name + "_clearances.csv")).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "time_s", "x", "y", "theta", "polygon_clearance_m",
                         "strict_overlap_obstacle_indices"])
        for index, (pose, clearance) in enumerate(zip(poses, clearances)):
            writer.writerow([index, times[index], *pose, clearance, json.dumps(overlaps[index])])
    minimum_index = int(np.argmin(clearances))
    report = {
        "trial": trial,
        "history_path": str(history_path),
        "history_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        "initial_pose": poses[0], "final_pose": poses[-1], "scenario_goal_pose": goal,
        "completed_plant_steps": len(rows), "sample_count": len(poses),
        "status_counts": {key: dict(Counter(row.get(key, "") for row in rows))
                          for key in ("solver_raw_status", "solver_status_code", "solver_success")},
        "minimum_polygon_clearance_m": clearances[minimum_index],
        "minimum_clearance_sample_index": minimum_index,
        "minimum_clearance_time_s": times[minimum_index],
        "minimum_clearance_pose": poses[minimum_index],
        "contact_or_overlap_sample_indices": [i for i, value in enumerate(clearances) if value <= 0],
        "outside_map_sample_indices": [i for i, value in enumerate(clearances) if value < -1e-10],
        "strict_overlap_samples": [{"sample_index": i, "time_s": times[i], "obstacle_indices": indices}
                                   for i, indices in enumerate(overlaps) if indices],
        "footprint_polygons_local": footprints, "map_bounds": grid[0],
        "obstacle_count": len(obstacles),
        "controller_total_time_all_steps": stats([row["controller_computation_time"] for row in rows]),
        "controller_total_time_excluding_first": stats([row["controller_computation_time"] for row in rows[1:]]),
        "controller_first_step_time_s": float(rows[0]["controller_computation_time"]) if rows else None,
        "solver_time_all_steps": stats([row["solver_computation_time"] for row in rows]),
        "solver_time_successful_steps": stats([row["solver_computation_time"] for row in rows
                                               if row["solver_success"].lower() == "true"]),
    }
    psdf_path = data / ("pose_sdf_{}.csv".format(trial["current_name"]))
    if psdf_path.exists():
        psdf_rows = read_csv(psdf_path)
        values = np.asarray([float(row["sdf_value"]) for row in psdf_rows])
        finite = values[np.isfinite(values)]
        report["logged_psdf"] = {
            "path": str(psdf_path), "count": len(values), "finite_count": len(finite),
            "minimum_m": float(np.min(finite)) if len(finite) else None,
            "negative_sample_count": int(np.sum(values < 0)),
            "scope": "Existing controller logging at pre-plant poses; may omit final pose or reuse optimizer cache. Not recomputed by this analyzer.",
        }
    return report


def plot_trajectories(report, simulation, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    figure, axes = plt.subplots(1, len(report["trials"]), figsize=(13, 4.5), squeeze=False)
    for axis, trial in zip(axes[0], report["trials"]):
        _, goal, grid, regions = simulation.create_env(trial["trial"]["maze_type"])
        for region in regions:
            axis.add_patch(Polygon(region.get_ccw_vertices(), facecolor="0.82", edgecolor="0.55"))
        rows = read_csv(Path(trial["history_path"]))
        poses = np.asarray([trial["initial_pose"]] + [
            [float(row[key]) for key in ("x", "y", "theta")] for row in rows
        ])
        axis.plot(poses[:, 0], poses[:, 1], color="#2463a0", linewidth=1.8)
        axis.scatter(*poses[0, :2], color="#26834a", s=28, label="Start", zorder=3)
        axis.scatter(*goal[:2], marker="*", color="#111111", s=75, label="Scenario goal", zorder=3)
        closest = np.asarray(trial["minimum_clearance_pose"])
        cosine, sine = np.cos(closest[2]), np.sin(closest[2])
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        for polygon in trial["footprint_polygons_local"]:
            axis.add_patch(Polygon(np.asarray(polygon) @ rotation.T + closest[:2],
                                   facecolor="#cc474733", edgecolor="#b23232", linewidth=1.2))
        axis.set_title("{} | {:.1f} s\nMinimum gap: {:.2f} mm".format(
            trial["trial"]["optimizer_type"].upper(), float(trial["trial"]["final_time"]),
            1000 * trial["minimum_polygon_clearance_m"]))
        axis.set(xlim=(grid[0][0][0], grid[0][1][0]), ylim=(grid[0][0][1], grid[0][1][1]),
                 xlabel="x [m]", ylabel="y [m]")
        axis.set_aspect("equal")
        axis.legend(loc="upper right", fontsize=7)
    figure.suptitle("DD kinematics / maze / 0.15 x 0.07 m rectangle", fontsize=13)
    figure.text(0.5, 0.025, "One nominal run per method. Red footprint: closest saved sample. Sampling interval: 0.1 s.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    figure.savefig(output / "trajectories.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "current")
    parser.add_argument("--output-dir", type=Path, default=HERE / "posthoc")
    args = parser.parse_args()
    results, output = args.results_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(ROOT))
    from models.dd import DifferentialDriveRectangleGeometry
    from sim.simulation_mpc import simulation_mpc
    from sim.start_perturbation import pose_clearance
    import torch
    import acados_template

    # Match mpc_test's rectangle construction exactly; no robot or optimizer exists.
    geometry = DifferentialDriveRectangleGeometry(length=0.15, width=0.07, rear_dist=0.0)
    footprints = [np.asarray(region.get_ccw_vertices(), dtype=float) for region in geometry.equiv_rep()]
    simulation = simulation_mpc()
    report = {
        "analyzed_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_environment": {"python": sys.version, "python_executable": sys.executable,
                                 "platform": platform.platform()},
        "checkout_commit": command("git", "rev-parse", "HEAD"),
        "checkout_status": command("git", "status", "--short"),
        "gpu_and_driver": command("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"),
        "pytorch_cuda_runtime": torch.version.cuda,
        "acados_template_path": acados_template.__file__,
        "acados_source_commit": command("git", "-C", os.environ["ACADOS_SOURCE_DIR"], "rev-parse", "HEAD")
            if os.environ.get("ACADOS_SOURCE_DIR") else None,
        "results_dir": str(results),
        "scope": "Initial pose plus every saved post-plant-step pose. No optimizer is instantiated. Sampled checks do not guarantee continuous collision freedom between control steps.",
        "geometry_definition": "Minimum Euclidean footprint-to-obstacle distance and map boundary clearance. Obstacle contact/overlap is zero; negative values indicate map violations. Strict obstacle overlap uses normalized-axis SAT with tolerance 1e-10 m.",
        "timing_definitions": {
            "controller_total": "perf_counter around generate_control_input, including setup/PSDF preparation/recovery and first-step code generation; excludes subsequent controller logging and plant update.",
            "solver_cc_mf": "perf_counter around acados solve calls only; main plus backup solve durations per cycle.",
            "solver_mpcc_cc_sf": "Legacy time.time interval including solver preparation, with SF recovery/diagnostics; not directly comparable to MF's solver-only interval.",
            "statistics": "Seconds; pooled saved steps, finite nonnegative values; NumPy linear percentile. First step is included unless explicitly excluded.",
        },
        "versions_note": "Versions below describe the analyzer interpreter; execute this script in the same environment used for the CLI run.",
        "package_versions": {}, "source_sha256": {}, "trials": [],
    }
    for package in ("numpy", "torch", "casadi", "scipy", "PyYAML", "acados_template"):
        try:
            report["package_versions"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["package_versions"][package] = None
    for relative in ("control/mpcc_optimizer.py", "control/cc_mf_optimizer.py", "control/cc_sf_optimizer.py",
                     "control/cc_covariance.py", "sim/simulation_mpc.py", "sim/start_perturbation.py",
                     "models/dd.py", "models/geometry_utils.py", "config/config_mpcc_refactor_validation.yaml"):
        report["source_sha256"][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    manifest = results / "run_manifest.yaml"
    report["run_manifest_path"] = str(manifest)
    report["run_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    report["trial_results_sha256"] = hashlib.sha256((results / "trial_results.csv").read_bytes()).hexdigest()
    for trial in read_csv(results / "trial_results.csv"):
        report["trials"].append(analyze_trial(trial, results, output, simulation, footprints, pose_clearance))
    destination = output / "analysis.json"
    destination.write_text(json.dumps(jsonable(report), indent=2, allow_nan=False), encoding="utf-8")
    plot_trajectories(report, simulation, output)
    print("Saved {}".format(destination), flush=True)
    for trial in report["trials"]:
        print("{}: {} steps, min clearance {:.9f} m, {} strict overlap samples".format(
            trial["trial"]["optimizer_type"], trial["completed_plant_steps"],
            trial["minimum_polygon_clearance_m"], len(trial["strict_overlap_samples"])), flush=True)


if __name__ == "__main__":
    main()
