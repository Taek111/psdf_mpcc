"""Validate matched starts/settings and polygon clearance from saved poses."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import Polygon
import yaml

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]
BASELINE = PROJECT / "benchmark_runs/cc_mf_maze_5trials_20261029_diagnostics"
sys.path.insert(0, str(PROJECT))
from models.dd import DifferentialDriveRectangleGeometry
from sim.simulation_mpc import simulation_mpc
from control.cc_mf_optimizer import CCMFOptimizerParam


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


config = yaml.safe_load((BASELINE / "diagnostic_config.yaml").read_text())
baseline_outcomes = rows(BASELINE / "results/trial_results.csv")[:3]
_, _, grid, obstacles = simulation_mpc().create_env("maze")
polygons = [Polygon(obstacle.get_ccw_vertices()) for obstacle in obstacles]
vertices = DifferentialDriveRectangleGeometry(.15, .12, .0)._region.get_ccw_vertices()
validation = {"batches": {}, "persisted_cc_mf_weights": {}}
for batch in sorted((BASE / "batches").iterdir()):
    if not batch.is_dir():
        continue
    batch_config = yaml.safe_load((batch / "diagnostic_config.yaml").read_text())
    weight = batch_config["cc_mf"].pop("r_delta_omega")
    outcomes = rows(batch / "results/trial_results.csv")
    result = {
        "weight": weight, "same_config_except_weight": batch_config == config,
        "three_completed_trials": len(outcomes) == 3,
        "same_initial_poses": all(
            all(row[key] == old[key] for key in ("initial_x", "initial_y", "initial_theta", "start_seed"))
            for row, old in zip(outcomes, baseline_outcomes)),
        "source_changes_from_baseline": json.loads((batch / "source_changes_from_baseline.json").read_text()),
        "source_changed_during_batch": json.loads((batch / "source_changed_during_batch.json").read_text()),
        "geometry": [],
    }
    source_hashes = json.loads((batch / "source_hashes.json").read_text())
    result["snapshot_hashes_valid"] = all(
        hashlib.sha256((batch / "source_snapshot" / key).read_bytes()).hexdigest() == value
        for key, value in source_hashes.items())
    for row in outcomes:
        trial = int(row["trial_number"])
        history = rows(next((batch / "results/runs/config01" / f"trial_{trial:04d}/data").glob("trial_history*.csv")))
        poses = [[float(row[key]) for key in ("initial_x", "initial_y", "initial_theta")]]
        poses += [[float(step[key]) for key in ("x", "y", "theta")] for step in history]
        minimum = np.inf
        collisions = 0
        for x, y, theta in poses:
            cosine, sine = np.cos(theta), np.sin(theta)
            rotation = np.array([[cosine, -sine], [sine, cosine]])
            polygon = Polygon(vertices @ rotation.T + [x, y])
            clearance = min(polygon.distance(obstacle) for obstacle in polygons)
            boundary = min(polygon.bounds[0] - grid[0][0][0],
                           polygon.bounds[1] - grid[0][0][1],
                           grid[0][1][0] - polygon.bounds[2],
                           grid[0][1][1] - polygon.bounds[3])
            minimum = min(minimum, clearance, boundary)
            collisions += int(boundary < 0 or any(polygon.intersects(obstacle) for obstacle in polygons))
        result["geometry"].append({
            "seed": int(row["start_seed"]), "pose_count": len(poses),
            "min_polygon_clearance_mm": minimum * 1000,
            "sampled_pose_collision_count": collisions,
        })
    validation["batches"][batch.name] = result

for name in ("config/config.yaml", "config/config_benchmark.yaml"):
    loaded = yaml.safe_load((PROJECT / name).read_text())
    param = CCMFOptimizerParam()
    simulation_mpc._apply_param_overrides(param, loaded["cc_mf"], "CC-MF")
    validation["persisted_cc_mf_weights"][name] = param.r_delta_omega
    assert param.r_delta_omega == .03

(BASE / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
print(json.dumps(validation, indent=2))
