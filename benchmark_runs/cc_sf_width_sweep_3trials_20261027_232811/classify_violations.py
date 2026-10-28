"""Distinguish obstacle intersection from map-boundary violations."""

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parent
with (ROOT / "clearance_samples.csv").open() as stream:
    samples = list(csv.DictReader(stream))
geometries = {}
counts = Counter()
details = []
for row in samples:
    width, seed = float(row["width_m"]), int(row["seed"])
    if width not in geometries:
        geometry = json.loads((ROOT / f"width_{round(width * 1000):03d}mm/geometry.json").read_text())
        geometries[width] = (np.asarray(geometry["footprint"]),
                             [Polygon(vertices) for vertices in geometry["obstacles"]],
                             np.asarray(geometry["bounds"]))
    if float(row["clearance_m"]) > 0:
        continue
    footprint, obstacles, (lower, upper) = geometries[width]
    x, y, theta = [float(row[key]) for key in ("x", "y", "theta")]
    cosine, sine = np.cos(theta), np.sin(theta)
    vertices = footprint @ np.array([[cosine, sine], [-sine, cosine]]) + [x, y]
    robot = Polygon(vertices)
    boundary_margin = min(float(np.min(vertices - lower)), float(np.min(upper - vertices)))
    touched = [index for index, obstacle in enumerate(obstacles) if robot.distance(obstacle) == 0]
    area = max((robot.intersection(obstacles[index]).area for index in touched), default=0.0)
    overlap = area > 1e-16
    boundary_violation = boundary_margin < 0
    assert touched or boundary_margin <= 0
    counts[(width, seed, "obstacle_overlap_samples")] += int(overlap)
    counts[(width, seed, "obstacle_contact_only_samples")] += int(bool(touched) and not overlap)
    counts[(width, seed, "boundary_violation_samples")] += int(boundary_violation)
    details.append({**row, "obstacle_overlap": overlap,
                    "obstacle_contact_only": bool(touched) and not overlap,
                    "boundary_violation": boundary_violation,
                    "boundary_margin_mm": boundary_margin * 1000,
                    "max_intersection_area_m2": area,
                    "obstacle_indices": "/".join(map(str, touched))})


def save_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


summary = []
with (ROOT / "benchmark_trials.csv").open() as stream:
    for row in csv.DictReader(stream):
        width, seed = float(row["width_m"]), int(row["seed"])
        summary.append({"width_m": width, "seed": seed, "goal_reached": row["status"] == "success",
            "clearance_nonpositive_samples": int(row["contact_or_boundary_violation_samples"]),
            **{metric: counts[(width, seed, metric)] for metric in
               ("obstacle_overlap_samples", "obstacle_contact_only_samples", "boundary_violation_samples")}})
save_csv(ROOT / "geometry_violations.csv", details)
save_csv(ROOT / "geometry_violations_by_trial.csv", summary)
print(json.dumps(summary, indent=2))
