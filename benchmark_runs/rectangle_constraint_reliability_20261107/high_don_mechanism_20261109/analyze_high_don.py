"""Inspect existing paired SF/MF outcomes and terminal guard-stop traces."""

from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
SUITE = ROOT / "benchmark_runs/rectangle_constraint_reliability_20261107/rectangle_don_sweep_50_20261105_current_sf"
WIDTHS = ((.83, "124p5"), (.84, "126"), (.85, "127p5"), (.86, "129"), (.87, "130p5"))


def load_geometry_module():
    path = SUITE / "source_snapshot/sim/start_perturbation.py"
    spec = importlib.util.spec_from_file_location("archived_clearance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_csv(name, rows):
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    geometry_module = load_geometry_module()
    records, pairs, examples = [], [], []
    for don, width in WIDTHS:
        folder = SUITE / "widths" / f"width_{width}mm"
        all_rows = list(csv.DictReader((folder / "trial_results.csv").open(encoding="utf-8-sig")))
        by_method = {method: {int(r["start_seed"]): r for r in all_rows if r["optimizer_type"] == method}
                     for method in ("cc_sf", "cc_mf")}
        successes = {m: {seed for seed, r in rows.items() if r["status"] == "success"} for m, rows in by_method.items()}
        groups = {
            "both_success": successes["cc_sf"] & successes["cc_mf"],
            "mf_only_success": successes["cc_mf"] - successes["cc_sf"],
            "sf_only_success": successes["cc_sf"] - successes["cc_mf"],
            "neither_success": set(range(50)) - successes["cc_sf"] - successes["cc_mf"],
        }
        pairs.append({"don": don, **{key: len(value) for key, value in groups.items()},
                      **{key + "_seeds": " ".join(map(str, sorted(value))) for key, value in groups.items()}})
        group_by_seed = {seed: key for key, value in groups.items() for seed in value}
        for method, trial_rows in by_method.items():
            for seed, row in trial_rows.items():
                data = folder / "runs" / row["configuration_id"] / f"trial_{int(row['trial_number']):04d}" / "data"
                raw = np.load(data / "raw_trace.npz")
                meta = json.loads((data / "raw_trace_metadata.json").read_text(encoding="utf-8"))
                poses, inputs = raw["poses"], raw["applied_inputs"]
                times = raw["pose_times_s"]
                statuses = meta["solver_status_infos"]
                trace_hash = hashlib.sha256((data / "raw_trace.npz").read_bytes()).hexdigest()
                if trace_hash != meta["raw_trace_sha256"]:
                    raise ValueError(f"Trace hash mismatch: {data}")
                if not (len(poses) == len(inputs) + 1 == len(times) and len(statuses) == len(inputs)):
                    raise ValueError(f"Trace alignment mismatch: {data}")
                if not all(np.isfinite(raw[key]).all() for key in raw.files):
                    raise ValueError(f"Non-finite trace: {data}")
                if meta["outcome"]["status"] != row["status"]:
                    raise ValueError(f"Outcome mismatch: {data}")
                geom = meta["geometry"]
                footprints = [np.array(p) for p in geom["footprint_polygons"]]
                obstacles = [np.array(p) for p in geom["obstacle_polygons"]]
                clearance = lambda pose: geometry_module.pose_clearance(pose, footprints, obstacles, geom["bounds"])
                guard_steps = [k for k, status in enumerate(statuses) if status["raw_status"] == "safe_stop_stage0"]
                first = guard_steps[0] if guard_steps else None
                stats = meta["outcome"]["optimizer_runtime_stats"]
                last = poses[-1]
                cosine, sine = np.cos(last[2]), np.sin(last[2])
                rotation = np.array([[cosine, -sine], [sine, cosine]])
                world = footprints[0] @ rotation.T + last[:2]
                obstacle_gaps = [geometry_module.convex_polygon_clearance(world, obstacle) for obstacle in obstacles]
                item = {
                    "don": don, "source": method, "seed": seed, "paired_group": group_by_seed[seed],
                    "status": row["status"], "failure_reason": row["failure_reason"],
                    "final_time_s": float(times[-1]), "final_x_m": last[0], "final_y_m": last[1], "final_theta_rad": last[2],
                    "distance_to_goal_m": float(row["distance_to_goal"]),
                    "terminal_true_clearance_mm": 1000 * clearance(last), "terminal_nearest_obstacle_index": int(np.argmin(obstacle_gaps)),
                    "d_col_mm": 1000 * meta["optimizer_parameters"]["d_col"],
                    "main_solver_failure_count": stats["main_solver_failure_count"],
                    "safe_stop_count": stats["safe_stop_count"], "stage0_guard_stop_count": stats["stage0_guard_stop_count"],
                    "backup_recovery_count": stats["backup_feasible_qp_success_count"],
                    "actual_guard_status_count": len(guard_steps),
                    "first_guard_step": first, "first_guard_current_pose_time_s": float(times[first]) if first is not None else None,
                    "first_guard_post_integrated_time_s": float(times[first + 1]) if first is not None else None,
                    "first_guard_current_clearance_mm": 1000 * clearance(poses[first]) if first is not None else None,
                    "pre_guard_previous_pose_clearance_mm": 1000 * clearance(poses[first - 1]) if first is not None and first > 0 else None,
                    "pre_guard_previous_solver_success": bool(statuses[first - 1]["success"]) if first is not None and first > 0 else None,
                    "pre_guard_previous_v": float(inputs[first - 1, 0]) if first is not None and first > 0 else None,
                    "pre_guard_previous_omega": float(inputs[first - 1, 1]) if first is not None and first > 0 else None,
                    "guard_to_end_max_pose_change": float(np.max(np.abs(poses[first:] - poses[first]))) if first is not None else None,
                    "guard_to_end_max_applied_input": float(np.max(np.abs(inputs[first:]))) if first is not None else None,
                    "trace": str((data / "raw_trace.npz").relative_to(ROOT)),
                    "trace_sha256": trace_hash,
                    "metadata_sha256": hashlib.sha256((data / "raw_trace_metadata.json").read_bytes()).hexdigest(),
                }
                records.append(item)
        candidates = [r for r in records if r["don"] == don and r["source"] == "cc_sf" and r["paired_group"] == "mf_only_success"]
        if candidates:
            median = float(np.median([r["final_time_s"] for r in candidates]))
            selected = min(candidates, key=lambda r: (round(abs(r["final_time_s"] - median), 10), r["seed"]))
            examples.append({"don": don, "seed": selected["seed"], "selection": "MF-only successful pair with SF failure time closest to its group median; ties choose smallest seed",
                             "group_size": len(candidates), "median_sf_failure_time_s": median, "selected_sf_failure_time_s": selected["final_time_s"],
                             "selected_sf_guard_time_s": selected["first_guard_current_pose_time_s"],
                             "selected_sf_terminal_clearance_mm": selected["terminal_true_clearance_mm"]})
    save_csv("trial_terminal_audit.csv", records)
    save_csv("paired_outcomes.csv", pairs)
    save_csv("selected_pairs.csv", examples)
    summaries = []
    for don, _ in WIDTHS:
        for method in ("cc_sf", "cc_mf"):
            failed = [r for r in records if r["don"] == don and r["source"] == method and r["status"] != "success"]
            clearances = [r["terminal_true_clearance_mm"] for r in failed]
            guards = [r for r in failed if r["first_guard_step"] is not None]
            summaries.append({
                "don": don, "source": method, "failed_trials": len(failed),
                "failure_reasons": dict(Counter(r["failure_reason"] for r in failed)),
                "guard_termination_trials": len(guards),
                "terminal_clearance_min_median_max_mm": [min(clearances), float(np.median(clearances)), max(clearances)] if clearances else [],
                "terminal_clearance_nonpositive_count": sum(x <= 0 for x in clearances),
                "terminal_map_boundary_intrusion_count": sum(x < 0 for x in clearances),
                "terminal_obstacle_contact_or_overlap_count": sum(x == 0 for x in clearances),
                "terminal_positive_gap_guard_count": sum(x > 0 for x in clearances),
                "terminal_clearance_below_dcol_count": sum(r["terminal_true_clearance_mm"] < r["d_col_mm"] for r in failed),
                "previous_solve_success_count": sum(r["pre_guard_previous_solver_success"] is True for r in guards),
                "zero_motion_after_first_guard_count": sum(r["guard_to_end_max_pose_change"] == 0 for r in guards),
                "terminal_nearest_obstacle_counts": dict(Counter(r["terminal_nearest_obstacle_index"] for r in failed)),
                "no_main_solver_failure_in_trial_count": sum(r["main_solver_failure_count"] == 0 for r in failed),
                "first_guard_before_10s_count": sum(r["first_guard_current_pose_time_s"] < 10 for r in guards),
                "first_guard_median_time_s": float(np.median([r["first_guard_current_pose_time_s"] for r in guards])) if guards else None,
            })
    summary = {
        "scope": "Existing raw poses, applied inputs, statuses and archived exact geometry only; no simulation or optimizer executed.",
        "first_guard_time_definition": "raw pose_times[first_guard_step], the current pose inspected before zero-input integration; final failure occurs after repeated guard-stop steps.",
        "conclusions_are_observational": "Measured guard termination mechanism is not an attribution to feature switching, slack use, or unlogged optimized candidates.",
        "paired_groups": pairs, "selected_examples": examples, "summaries": summaries,
        "inspected_trial_count": len(records),
        "source_hashes_manifest": str((SUITE / "source_hashes.json").relative_to(ROOT)),
        "source_aggregate_sha256": json.loads((SUITE / "source_hashes.json").read_text(encoding="utf-8"))["aggregate_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUTPUT / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(examples, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
