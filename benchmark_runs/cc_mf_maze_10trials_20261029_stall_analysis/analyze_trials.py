"""Summarize only completed trials, offline; safe to rerun during the batch."""

import argparse
import csv
from collections import Counter
import json
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent
DT = 0.1
TOL = 1e-6


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def array(rows, field):
    return np.asarray([float(row[field]) if row.get(field) else np.nan
                       for row in rows], dtype=float)


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return None if not len(values) else {
        "min": float(values.min()), "median": float(np.median(values)),
        "mean": float(values.mean()), "max": float(values.max()),
    }


def streak(mask):
    longest = current = 0
    for item in mask:
        current = current + 1 if item else 0
        longest = max(longest, current)
    return longest


def intervals(mask):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return list(zip(starts, ends))


def trajectory_window(history, poses, start, end):
    """State endpoints inclusive; commands in (start,end], not [start,end]."""
    post_t = array(history, "time")
    initial = np.array([float(poses[0][field]) for field in ("x", "y", "theta")])
    states = np.vstack((initial, np.column_stack([
        array(history, field) for field in ("x", "y", "theta")
    ])))
    state_t = np.r_[0.0, post_t]
    state_mask = (state_t >= start - 1e-8) & (state_t <= end + 1e-8)
    control_mask = (post_t > start + 1e-8) & (post_t <= end + 1e-8)
    window_states = states[state_mask]
    v = array(history, "v")[control_mask]
    omega = array(history, "omega")[control_mask]
    rows = [row for row, keep in zip(history, control_mask) if keep]
    pre_t = array(poses, "timestep") * DT
    pose_rows = [row for row, time in zip(poses, pre_t)
                 if start - 1e-8 <= time <= end + 1e-8]
    xy = window_states[:, :2]
    heading = np.unwrap(window_states[:, 2])
    displacement = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) else None
    travel = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    result = {
        "start_time_s": float(start), "end_time_s": float(end),
        "state_samples": len(window_states), "control_samples": len(rows),
        "duration_s": float(state_t[state_mask][-1] - state_t[state_mask][0]),
        "xy_start_m": xy[0], "xy_end_m": xy[-1],
        "net_displacement_m": displacement, "accumulated_travel_m": travel,
        "xy_bounding_box_m": np.ptp(xy, axis=0),
        "heading_range_rad": float(np.ptp(heading)),
        "accumulated_abs_heading_change_rad": float(np.abs(np.diff(heading)).sum()),
        "v_m_s": stats(v), "abs_v_m_s": stats(abs(v)),
        "omega_rad_s": stats(omega), "abs_omega_rad_s": stats(abs(omega)),
        "speed_below_1_mm_s_fraction": float(np.mean(abs(v) < 0.001)) if len(v) else None,
        "omega_sign_reversals_both_at_least_0_05_rad_s": int(np.count_nonzero(
            (omega[:-1] * omega[1:] < 0)
            & (np.minimum(abs(omega[:-1]), abs(omega[1:])) >= 0.05))),
        "omega_sign_reversals_both_at_least_0_001_rad_s": int(np.count_nonzero(
            (omega[:-1] * omega[1:] < 0)
            & (np.minimum(abs(omega[:-1]), abs(omega[1:])) >= 0.001))),
        "v_sign_reversals_both_at_least_0_01_mm_s": int(np.count_nonzero(
            (v[:-1] * v[1:] < 0)
            & (np.minimum(abs(v[:-1]), abs(v[1:])) >= 0.00001))),
        "one_step_xy_change_m": stats(np.linalg.norm(np.diff(xy, axis=0), axis=1)),
        "two_step_xy_return_error_m": stats(np.linalg.norm(xy[2:] - xy[:-2], axis=1)),
        "two_step_heading_return_error_rad": stats(abs(heading[2:] - heading[:-2])),
        "returned_solver_status_counts": dict(Counter(row["solver_status_code"] for row in rows)),
        "max_consecutive_unsuccessful_steps": streak([
            row["solver_success"] != "True" for row in rows]),
    }
    if pose_rows:
        for field in ("path_s", "predicted_s", "planner_projected_s_raw",
                      "planner_projected_s", "slice_start_s", "slice_end_s"):
            vals = array(pose_rows, field)
            result[field] = {"start": vals[0], "end": vals[-1],
                             "range": stats(vals), "end_minus_start": vals[-1] - vals[0],
                             "max_abs_step": float(np.max(abs(np.diff(vals)))) if len(vals) > 1 else 0}
        result["slice_start_indices"] = sorted(set(
            int(float(row["slice_start_idx"])) for row in pose_rows if row["slice_start_idx"]))
        result["slice_change_count"] = int(np.count_nonzero(np.diff(array(pose_rows, "slice_start_idx"))))
        result["path_s_distance_to_slice_end_m"] = stats(
            array(pose_rows, "slice_end_s") - array(pose_rows, "path_s"))
        result["current_psdf_m"] = stats(array(pose_rows, "sdf_value"))
        result["last_pose_row_time_s"] = float(pose_rows[-1]["timestep"]) * DT
    return result


def finite_min(values):
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    return float(finite.min()) if len(finite) else np.nan


def finite_max(values):
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    return float(finite.max()) if len(finite) else np.nan


def compact_cycle(record, epsilon):
    result = {key: record.get(key) for key in (
        "time", "solve_step", "mode", "status", "main_status", "current_s0", "pose")}
    result["v"] = record["inputs"][0][0]
    result["omega"] = record["inputs"][0][1]
    d = record.get("diagnostics")
    if not d:
        return result
    guard = np.asarray(d["guard_affine_residual"], float)[1:]
    exact_guard = np.asarray(d["guard_fresh_exact_psdf_residual"], float)[1:]
    mf = np.asarray(d["mf_affine_residual"], float)[1:]
    slack = np.asarray(d["mf_lower_slack"], float)[1:]
    active = np.asarray(d["mf_mask"], bool)[1:]
    fresh_active = np.asarray(d["mf_fresh_mask"], bool)[1:]
    fresh = np.asarray(d["mf_fresh_raw_residual"], float)[1:]
    fresh_prob = np.asarray(d.get("mf_probability_sum_fresh", np.full(len(active)+1, np.nan)), float)[1:]
    phi = np.asarray(d["exact_psdf_predicted_nodes"], float)
    interval = np.asarray(d["first_interval_substep_exact_psdf"], float)
    mf_effective = mf + slack
    result.update({
        "exact_current_psdf_m": d["exact_current_psdf"],
        "min_exact_predicted_psdf_m": finite_min(phi[1:]),
        "stage1_exact_psdf_m": phi[1],
        "min_first_interval_psdf_m": finite_min(interval),
        "d_col_m": d["d_col"],
        "min_affine_guard_residual_m": finite_min(guard),
        "min_exact_guard_residual_m": finite_min(exact_guard),
        "guard_tight_stages_abs_residual_le_10um": [int(x + 1) for x in np.flatnonzero(abs(guard) <= 1e-5)],
        "guard_affine_violated": bool(np.any(guard < -TOL)),
        "guard_exact_violated": bool(np.any(exact_guard < -TOL)),
        "predicted_node_penetration": bool(np.any(phi[1:] < -TOL)),
        "first_interval_guard_violated": bool(np.any(interval < d["d_col"] - TOL)),
        "first_interval_penetration": bool(np.any(interval < -TOL)),
        "active_mf_rows": int(active.sum()),
        "min_affine_mf_residual_quantile": finite_min(mf[active]),
        "min_affine_mf_plus_slack_quantile": finite_min(mf_effective[active]),
        "max_mf_slack_quantile": finite_max(slack[active]),
        "mf_slack_positive_stages_gt_1e_5": [int(x+1) for x in np.flatnonzero(active & (slack > 1e-5))],
        "mf_affine_tight_stages_abs_residual_le_1e_5": [int(x+1) for x in np.flatnonzero(active & (abs(mf) <= 1e-5))],
        "affine_mf_before_slack_violated": bool(np.any(mf[active] < -TOL)),
        "affine_mf_after_slack_violated": bool(np.any(mf_effective[active] < -TOL)),
        "fresh_mf_after_same_slack_violated_on_shared_active_rows": bool(np.any(
            (fresh + slack)[active & fresh_active] < -1e-3)),
        "fresh_mf_probability_sum_gt_epsilon_on_installed_rows": bool(np.any(fresh_prob[active] > epsilon + TOL)),
        "max_fresh_mf_probability_sum_on_installed_rows": finite_max(fresh_prob[active]),
        "nlp_residuals": d.get("solver", {}).get("nlp_residuals"),
    })
    for key in ("mf_icdf_residual_frozen", "mf_icdf_residual_fresh"):
        if key in d:
            result["min_" + key] = finite_min(np.asarray(d[key], float)[1:][active])
    return result


def telemetry_summary(cycles, start=None, end=None):
    selected = [row for row in cycles if (start is None or row["time"] >= start - 1e-8)
                and (end is None or row["time"] < end - 1e-8)]
    successful = [row for row in selected if row.get("status") == 0]
    diagnosed = [row for row in successful if "guard_affine_violated" in row]
    result = {"cycles": len(selected), "successful_cycles": len(successful),
              "diagnosed_successful_cycles": len(diagnosed),
              "mode_counts": dict(Counter(row["mode"] for row in selected)),
              "main_status_counts": dict(Counter(str(row.get("main_status")) for row in selected)),
              "returned_status_counts": dict(Counter(str(row.get("status")) for row in selected))}
    for key in ("guard_affine_violated", "guard_exact_violated", "predicted_node_penetration",
                "first_interval_guard_violated", "first_interval_penetration",
                "affine_mf_before_slack_violated", "affine_mf_after_slack_violated",
                "fresh_mf_after_same_slack_violated_on_shared_active_rows",
                "fresh_mf_probability_sum_gt_epsilon_on_installed_rows"):
        result["successful_cycles_" + key] = sum(bool(row.get(key)) for row in diagnosed)
    for key in ("exact_current_psdf_m", "min_exact_predicted_psdf_m", "stage1_exact_psdf_m",
                "min_first_interval_psdf_m", "min_affine_guard_residual_m", "active_mf_rows",
                "max_mf_slack_quantile", "max_fresh_mf_probability_sum_on_installed_rows"):
        result[key] = stats([row.get(key, np.nan) for row in diagnosed])
    result["successful_cycles_with_tight_guard_rows"] = sum(bool(row.get(
        "guard_tight_stages_abs_residual_le_10um")) for row in diagnosed)
    result["successful_cycles_with_positive_mf_slack"] = sum(bool(row.get(
        "mf_slack_positive_stages_gt_1e_5")) for row in diagnosed)
    result["successful_cycles_with_tight_mf_rows"] = sum(bool(row.get(
        "mf_affine_tight_stages_abs_residual_le_1e_5")) for row in diagnosed)
    return result


def analyze_trial(outcome, epsilon):
    number = int(outcome["trial_number"])
    directory = BASE / "results" / "runs" / outcome["configuration_id"] / f"trial_{number:04d}" / "data"
    histories = list(directory.glob("trial_history_*.csv"))
    poses_files = list(directory.glob("pose_sdf_*.csv"))
    if not histories or not poses_files:
        return {"trial": number, "status": outcome["status"], "analysis_error": "missing trajectory CSV"}
    history, poses = read_csv(histories[0]), read_csv(poses_files[0])
    end = float(history[-1]["time"])
    result = {"trial": number, "seed": int(outcome["start_seed"]),
              "status": outcome["status"], "failure_reason": outcome["failure_reason"],
              "final_time_s": float(outcome["final_time"]), "steps": len(history),
              "distance_to_goal_l1_m": float(outcome["distance_to_goal"]),
              "history_path": str(histories[0]), "pose_sdf_path": str(poses_files[0]),
              "returned_status_counts": dict(Counter(row["solver_status_code"] for row in history)),
              "max_consecutive_unsuccessful_steps": streak([row["solver_success"] != "True" for row in history])}
    result["whole_run"] = trajectory_window(history, poses, 0.0, end)
    # Identify a settled period-two suffix without calling a controller.
    xy = np.column_stack([array(history, field) for field in ("x", "y")])
    angle = np.unwrap(array(history, "theta"))
    v = array(history, "v")
    omega = array(history, "omega")
    repeated = ((np.linalg.norm(xy[2:] - xy[:-2], axis=1) < 1e-6)
                & (abs(angle[2:] - angle[:-2]) < 1e-4)
                & (v[2:] * v[1:-1] < 0)
                & (omega[2:] * omega[1:-1] < 0)
                & (np.minimum(abs(v[2:]), abs(v[1:-1])) > 1e-5))
    suffix_length = 0
    for flag in repeated[::-1]:
        if not flag:
            break
        suffix_length += 1
    result["settled_period_two_suffix"] = {
        "comparison_tolerance_xy_m": 1e-6,
        "comparison_tolerance_heading_rad": 1e-4,
        "consecutive_return_comparisons": suffix_length,
        "detected_at_least_1s": suffix_length >= 10,
        "suffix_start_time_s": float(array(history, "time")[len(history)-suffix_length-2])
        if suffix_length else None,
        "period_s": 2*DT if suffix_length >= 10 else None,
    }
    for window in (20.0, 4.0):
        label = f"last_{int(window)}s"
        result[label] = trajectory_window(history, poses, max(0.0, end-window), end)
        result[label]["full_requested_window_available"] = end >= window - 1e-8
    # Show every >=1s contiguous command interval with speed below 1cm/s.
    speed = abs(array(history, "v"))
    result["low_speed_intervals_abs_v_lt_0_01_m_s"] = [
        trajectory_window(history, poses, a*DT, (b+1)*DT)
        for a,b in intervals(speed < .01) if b-a+1 >= 10
    ]
    cycles = []
    telemetry = BASE / f"telemetry_trial{number:02d}.jsonl"
    if telemetry.exists():
        with telemetry.open(encoding="utf-8") as handle:
            for line in handle:
                cycles.append(compact_cycle(json.loads(line), epsilon))
        result["telemetry"] = telemetry_summary(cycles)
        for window in (20.0, 4.0):
            result[f"last_{int(window)}s"]["telemetry"] = telemetry_summary(cycles, max(0., end-window), end)
        # Cycle snapshots are scalar summaries; no giant raw horizon arrays.
        result["representative_cycles"] = {
            "last": cycles[-1],
            "max_mf_slack": max(cycles, key=lambda row: row.get("max_mf_slack_quantile") or -np.inf),
            "min_current_psdf": min(cycles, key=lambda row: row.get("exact_current_psdf_m", np.inf)),
        }
    result["motion_classification"] = (
        "near_stationary_translation" if result["last_20s"]["accumulated_travel_m"] <= .03
        else "motion_with_low_net_displacement" if result["last_20s"]["net_displacement_m"] <= .03
        else "moving_over_final_20s")
    (directory.parent / "analysis_summary.json").write_text(json.dumps(plain(result), indent=2) + "\n")
    return result, cycles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    outcomes_path = BASE / "results" / "trial_results.csv"
    outcomes = read_csv(outcomes_path) if outcomes_path.exists() else []
    parameters = json.loads((BASE / "resolved_cc_mf_parameters.json").read_text())
    summaries, cycle_rows = [], []
    for outcome in outcomes:
        analyzed = analyze_trial(outcome, parameters["chance_epsilon"])
        if isinstance(analyzed, tuple):
            summary, cycles = analyzed
            cycle_rows.extend({"trial": summary["trial"], **row} for row in cycles)
        else:
            summary = analyzed
        summaries.append(summary)
    payload = {"completed_trials_analyzed": len(summaries), "notes": [
        "Offline-only analysis of trial_results rows; no navigation or solver calls.",
        "history row k contains post-step state at (k+1)dt and applied input u_k; pose/telemetry use pre-step time kdt.",
        "Movement verdict is endpoint XY displacement, not accumulated distance or heading change.",
        "Telemetry window [a,b) matches applied controls advancing state endpoints [a,b].",
        "Current/first-interval PSDF and future predicted PSDF are reported separately.",
        "MF slack uses standard-normal quantile units; probability sums are feature probability sums.",
        "A successful SQP-RTI QP does not certify nonlinear constraints or an optimal stationary solution.",
    ], "trials": summaries}
    (BASE / "analysis_summary.json").write_text(json.dumps(plain(payload), indent=2) + "\n")
    table = []
    for summary in summaries:
        if "analysis_error" in summary:
            continue
        tail = summary["last_20s"]
        telemetry = summary.get("telemetry", {})
        table.append({
            "trial": summary["trial"], "seed": summary["seed"], "status": summary["status"],
            "failure_reason": summary["failure_reason"], "final_time_s": summary["final_time_s"],
            "steps": summary["steps"], "motion_classification": summary["motion_classification"],
            "tail20_net_displacement_m": tail["net_displacement_m"],
            "tail20_accumulated_travel_m": tail["accumulated_travel_m"],
            "tail20_heading_variation_rad": tail["accumulated_abs_heading_change_rad"],
            "tail20_mean_abs_v_m_s": tail["abs_v_m_s"]["mean"],
            "tail4_mean_abs_v_m_s": summary["last_4s"]["abs_v_m_s"]["mean"],
            "tail20_omega_reversals": tail["omega_sign_reversals_both_at_least_0_05_rad_s"],
            "tail20_path_s_change_m": tail["path_s"]["end_minus_start"],
            "tail20_min_path_s_distance_to_slice_end_m": tail["path_s_distance_to_slice_end_m"]["min"],
            "max_failure_streak": summary["max_consecutive_unsuccessful_steps"],
            "solver_mode_counts": json.dumps(telemetry.get("mode_counts")),
            "min_current_psdf_m": (telemetry.get("exact_current_psdf_m") or {}).get("min"),
            "min_first_interval_psdf_m": (telemetry.get("min_first_interval_psdf_m") or {}).get("min"),
            "successful_cycles_guard_exact_violated": telemetry.get("successful_cycles_guard_exact_violated"),
            "successful_cycles_affine_guard_violated": telemetry.get("successful_cycles_guard_affine_violated"),
            "successful_cycles_fresh_mf_after_same_slack_violated": telemetry.get("successful_cycles_fresh_mf_after_same_slack_violated_on_shared_active_rows"),
        })
    if table:
        with (BASE / "analysis_trials.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(plain(table))
    if cycle_rows:
        keys = list(dict.fromkeys(key for row in cycle_rows for key in row))
        with (BASE / "analysis_cycles.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows({key: json.dumps(plain(value)) if isinstance(value, (dict, list))
                              else plain(value) for key, value in row.items()} for row in cycle_rows)
    print(json.dumps({"completed_trials_analyzed": len(summaries), "trials": table}, indent=2))


if __name__ == "__main__":
    main()
