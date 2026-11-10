"""Exact sampled clearances for selected recorded SF/MF success-discordant pairs."""

import csv
import json
from pathlib import Path

import numpy as np

from analyze_high_don import ROOT, OUTPUT, SUITE, load_geometry_module, save_csv


def main():
    module = load_geometry_module()
    selected = list(csv.DictReader((OUTPUT / "selected_pairs.csv").open()))
    terminal = list(csv.DictReader((OUTPUT / "trial_terminal_audit.csv").open()))
    samples, passages = [], []
    for example in selected:
        don, seed = float(example["don"]), int(example["seed"])
        traces = {}
        for method in ("cc_sf", "cc_mf"):
            row = next(r for r in terminal if float(r["don"]) == don and int(r["seed"]) == seed and r["source"] == method)
            data = SUITE / "widths" / row["trace"].replace("\\", "/").split("/widths/", 1)[1]
            raw = np.load(data)
            metadata = json.loads(data.with_name("raw_trace_metadata.json").read_text(encoding="utf-8"))
            poses, times, inputs = raw["poses"], raw["pose_times_s"], raw["applied_inputs"]
            geometry = metadata["geometry"]
            footprints = [np.array(p) for p in geometry["footprint_polygons"]]
            obstacles = [np.array(p) for p in geometry["obstacle_polygons"]]
            unique, reverse = np.unique(poses, axis=0, return_inverse=True)
            clearances = np.array([module.pose_clearance(p, footprints, obstacles, geometry["bounds"]) for p in unique])[reverse]
            progress = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1))]
            statuses = metadata["solver_status_infos"]
            region = (poses[:, 0] <= .2) & (poses[:, 1] <= .65)
            crossed = np.flatnonzero(poses[:, 0] >= .2)
            traces[method] = {"poses": poses, "times": times, "inputs": inputs, "clearances": clearances,
                              "progress": progress, "metadata": metadata}
            np.savez_compressed(OUTPUT / f"selected_don{don:.2f}_{method}_seed{seed}.npz", poses=poses, times=times,
                                inputs=inputs, true_clearances_m=clearances, travelled_distance_m=progress)
            passages.append({
                "don": don, "seed": seed, "source": method, "outcome": row["status"],
                "pose_samples": len(poses), "clearance_min_mm": 1000 * clearances.min(),
                "nonpositive_clearance_samples": int(np.sum(clearances <= 0)),
                "first_corner_region": "recorded center x<=0.20m and y<=0.65m, including stopped samples",
                "first_corner_samples": int(region.sum()),
                "first_corner_min_clearance_mm": 1000 * clearances[region].min() if region.any() else None,
                "first_x_ge_0p20_time_s": float(times[crossed[0]]) if len(crossed) else None,
                "final_travelled_distance_m": progress[-1],
            })
            for k, pose in enumerate(poses):
                samples.append({
                    "don": don, "seed": seed, "source": method, "pose_index": k, "time_s": times[k],
                    "x_m": pose[0], "y_m": pose[1], "heading_rad": pose[2], "true_clearance_mm": 1000 * clearances[k],
                    "travelled_distance_m": progress[k], "first_corner_region": bool(region[k]),
                    "current_control_status": statuses[k]["raw_status"] if k < len(statuses) else "trial_ended",
                    "input_v_m_s": inputs[k, 0] if k < len(inputs) else None,
                    "input_omega_rad_s": inputs[k, 1] if k < len(inputs) else None,
                })
        sf, mf = traces["cc_sf"], traces["cc_mf"]
        target = sf["poses"][-1]
        distance = np.linalg.norm(mf["poses"][:, :2] - target[:2], axis=1)
        near = int(np.argmin(distance))
        delta = mf["poses"][near, 2] - target[2]
        passages[-1].update({
            "nearest_sf_stopping_center_time_s": float(mf["times"][near]),
            "nearest_sf_stopping_center_distance_mm": 1000 * distance[near],
            "nearest_sf_stopping_heading_difference_rad": float(np.arctan2(np.sin(delta), np.cos(delta))),
            "nearest_sf_stopping_center_clearance_mm": 1000 * mf["clearances"][near],
        })
    save_csv("selected_pair_recorded_samples.csv", samples)
    save_csv("selected_pair_passage_summary.csv", passages)
    print(json.dumps(passages, indent=2))


if __name__ == "__main__":
    main()
