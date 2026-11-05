"""Summarize a completed paired SF/MF width sweep from saved logs only.

No simulation, solver, geometry replay, or tests are run. The default command
requires a completed experiment manifest and all requested trial-result rows.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def numeric(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def truth(value):
    return str(value).lower() in ("1", "true")


def integer_id(value):
    value = numeric(value)
    return int(value) if value is not None and value >= 0 and value.is_integer() else None


def valid_pose(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    return all(numeric(v) is not None for v in value)


def binary_flag(value):
    if not isinstance(value, (bool, int, float)) or not np.isfinite(value) or value not in (0, 1):
        return None
    return bool(value)


def dist(values):
    values = np.array([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95)),
            "min": float(values.min()), "max": float(values.max())}


def hash_file(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def parse_vector(row, name):
    value = row.get(name)
    return json.loads(value) if value else []


def failure_type(status, reason):
    text = str(reason or "").lower()
    if status == "success":
        return "success"
    if "safe_stop_stage0" in text:
        return "stage0_guard_stop_limit"
    if "consecutive_solver_failures" in text:
        return "solver_failure_limit"
    if "movement" in text or "stagnation" in text or "stuck" in text:
        return "movement_stagnation"
    if "timeout" in text or status == "timeout":
        return "timeout"
    if status == "error":
        return "error"
    return str(status or "unknown")


def geometric_events(rows, flags, *, event_type, source, dt, cycles, threshold=None):
    """Record episode onsets without equating zero polygon distance to penetration."""
    events = []
    cycle_ids = [integer_id(r.get("step")) for r in cycles]
    cycle_ids_valid = bool(cycles) and all(i is not None for i in cycle_ids) and len(set(cycle_ids)) == len(cycle_ids)
    cycle_lookup = dict(zip(cycle_ids, cycles)) if cycle_ids_valid else {}
    sample_key = "pose_index" if source == "clearance" else "timestep"
    sample_ids = [integer_id(r.get(sample_key)) for r in rows]
    sample_ids_valid = all(i is not None for i in sample_ids) and len(set(sample_ids)) == len(sample_ids)
    i = 0
    while i < len(rows):
        if not flags[i]:
            i += 1
            continue
        start = i
        while i < len(rows) and flags[i] and (i == start or (
                sample_ids_valid and sample_ids[i] == sample_ids[i - 1] + 1)):
            i += 1
        row = rows[start]
        # pose_sdf samples are pre-controller poses; clearance sample j is the
        # initial pose for j=0 and the result of controller call j-1 otherwise.
        cycle = sample_ids[start] if sample_ids_valid else None
        recorded_time = numeric(row.get("simulation_time_s")) if source == "clearance" else None
        time_verified = source != "clearance" or (cycle is not None and dt is not None
                                                   and recorded_time is not None
                                                   and abs(recorded_time - cycle * dt) <= 1e-6)
        mapping_verified = cycle_ids_valid and sample_ids_valid and time_verified
        previous = cycle - 1 if cycle is not None else None
        current_cycle = cycle_lookup.get(cycle) if mapping_verified else None
        previous_cycle = cycle_lookup.get(previous) if mapping_verified else None
        event = {"event_type": event_type, "source": source, "sample_start": start,
                 "sample_last": i - 1, "sample_count": i - start,
                 "sample_id": cycle, "cycle_mapping_verified": mapping_verified,
                 "time_s": recorded_time if source == "clearance" else (cycle * dt if cycle is not None and dt is not None else None),
                 "x": numeric(row.get("x")), "y": numeric(row.get("y")), "theta": numeric(row.get("theta")),
                 "signed_psdf_m": numeric(row.get("sdf_value")),
                 "clearance_m": numeric(row.get("clearance_m")), "threshold_m": threshold,
                 "preceding_controller_step": previous if previous_cycle is not None else None,
                 "preceding_solve_mode": previous_cycle.get("solve_mode") if previous_cycle else None,
                 "current_controller_step": cycle if current_cycle is not None else None,
                 "current_solve_mode": current_cycle.get("solve_mode") if current_cycle else None,
                 "current_solver_status_code": current_cycle.get("solver_status_code") if current_cycle else None,
                 "progress_m": numeric(row.get("path_s"))}
        events.append(event)
    return events


def mf_coverage(rows, params):
    n = int(params["horizon"])
    start = int(params.get("mf_active_start_step", 1))
    end = params.get("mf_active_end_step")
    end = n - 1 if end is None else int(end)
    requested = list(range(max(1, start), min(n - 1, end) + 1))
    if params.get("mf_active_terminal", True):
        requested.append(n)
    activation_enabled = bool(params.get("use_obstacle_constraint", True)) and bool(params.get("use_row_mf", True))
    if not activation_enabled:
        requested = []
    count = Counter({name: 0 for name in (
        "logged_cycles", "fallback_cycles_excluded", "nonfallback_logged_cycles",
        "missing_activation_cycles", "valid_activation_cycles", "requested_future_stage_instances",
        "active_future_stage_instances", "all_requested_future_rows_inactive_cycles",
        "any_requested_future_row_inactive_cycles",
        "nominal_probability_sum_gt_epsilon_active_stage_instances",
        "any_active_nominal_probability_sum_gt_epsilon_cycles", "invalid_activation_cycles",
        "unknown_requested_activation_flags", "disabled_or_empty_requested_window_cycles")})
    stage_active = Counter()
    branch_counts = Counter()
    slack = []
    probability = []
    epsilon = float(params["chance_epsilon"])
    thresholds = (1e-6, 1e-4, 1e-2)
    for row in rows:
        count["logged_cycles"] += 1
        if str(row.get("solve_mode", "")).startswith("safe_stop"):
            count["fallback_cycles_excluded"] += 1
            continue
        count["nonfallback_logged_cycles"] += 1
        if not requested:
            count["disabled_or_empty_requested_window_cycles"] += 1
            continue
        active = parse_vector(row, "mf_active")
        values = parse_vector(row, "mf_slack")
        risks = parse_vector(row, "mf_probability_sum")
        branches = parse_vector(row, "mf_branch")
        if len(active) != n + 1:
            count["missing_activation_cycles"] += 1
            continue
        requested_flags = {i: binary_flag(active[i]) for i in requested}
        unknown = sum(value is None for value in requested_flags.values())
        if unknown:
            count["unknown_requested_activation_flags"] += unknown
            count["invalid_activation_cycles"] += 1
            continue
        count["valid_activation_cycles"] += 1
        count["requested_future_stage_instances"] += len(requested)
        active_ids = [i for i in requested if requested_flags[i]]
        count["active_future_stage_instances"] += len(active_ids)
        count["all_requested_future_rows_inactive_cycles"] += not active_ids
        count["any_requested_future_row_inactive_cycles"] += len(active_ids) < len(requested)
        stage_active.update(active_ids)
        cycle_slack = []
        cycle_risk_exceed = False
        for i in active_ids:
            if i < len(branches) and branches[i] is not None:
                branch_counts[str(branches[i])] += 1
            value = numeric(values[i]) if i < len(values) else None
            if value is not None:
                slack.append(value)
                cycle_slack.append(value)
                for threshold in thresholds:
                    count[f"active_slack_gt_{threshold:g}_stage_instances"] += value > threshold
            risk = numeric(risks[i]) if i < len(risks) else None
            if risk is not None:
                probability.append(risk)
                count["nominal_probability_sum_gt_epsilon_active_stage_instances"] += risk > epsilon
                cycle_risk_exceed |= risk > epsilon
        for threshold in thresholds:
            count[f"any_active_slack_gt_{threshold:g}_cycles"] += any(v > threshold for v in cycle_slack)
        count["any_active_nominal_probability_sum_gt_epsilon_cycles"] += cycle_risk_exceed
    valid = count["valid_activation_cycles"]
    result = {"counts": dict(count), "requested_stages": requested, "activation_enabled": activation_enabled,
              "active_count_by_stage": dict(stage_active),
              "active_fraction_by_stage": {i: stage_active[i] / valid if valid else None for i in requested},
              "branch_counts_on_active_rows": dict(branch_counts),
              "active_slack_quantile_units": dist(slack),
              "active_nominal_probability_sum": dist(probability),
              "slack_thresholds_quantile_units": list(thresholds),
              "probability_definition": "Logged pre-solve nominal Boole sum, not postsolve chance feasibility"}
    return result, slack, probability


def load_trial(root, row):
    width = int(round(float(row["width_m"]) * 1000))
    method, seed = row["method"], int(row["seed"])
    directory = root / f"width_{width:03d}mm/{method}/seed_{seed}"
    paths = {"outcome": directory / "outcome.json", "parameters": directory / "optimizer_parameters.json",
             "cycles": directory / "cycles.csv", "clearance": directory / "clearance.csv"}
    poses_files = list((directory / "data").glob("pose_sdf_*.csv"))
    if len(poses_files) == 1:
        paths["poses"] = poses_files[0]
    outcome = read_json(paths["outcome"], {})
    params = read_json(paths["parameters"], {})
    cycles = read_csv(paths["cycles"]) if paths["cycles"].exists() else []
    poses = read_csv(paths["poses"]) if "poses" in paths else []
    clearance = read_csv(paths["clearance"]) if paths["clearance"].exists() else []
    runtime = outcome.get("optimizer_runtime_stats", outcome.get("runtime_stats", {}))
    status = row.get("status") or outcome.get("status")
    reached = truth(row.get("success"))
    dt = params["tf"] / params["horizon"] if params else None
    initial = outcome.get("initial_pose", outcome.get("start_perturbation", {}).get("initial_pose"))
    if initial is None and poses:
        initial = [float(poses[0][key]) for key in ("x", "y", "theta")]
    stops = [str(r.get("solve_mode", "")).startswith("safe_stop") for r in cycles]
    stage0 = sum(r.get("solve_mode") == "safe_stop_stage0" for r in cycles)
    qp = sum(r.get("solve_mode") == "safe_stop" for r in cycles)
    modes = Counter(r.get("solve_mode") for r in cycles)
    codes = Counter("not_called" if r.get("solve_mode") == "safe_stop_stage0"
                    else str(r.get("solver_status_code") or "unknown") for r in cycles)
    episodes = []
    i = 0
    while i < len(cycles):
        if not stops[i]:
            i += 1
            continue
        start = i
        while i < len(cycles) and stops[i]:
            i += 1
        pose = poses[start] if start < len(poses) else {}
        episodes.append({"start_step": start, "last_step": i - 1, "steps": i - start,
                         "start_time_s": start * dt if dt is not None else None,
                         "duration_s": (i - start) * dt if dt is not None else None,
                         "progress_start_m": numeric(pose.get("path_s")),
                         "psdf_start_m": numeric(pose.get("sdf_value")),
                         "terminal": i == len(cycles),
                         "next_cycle_success": truth(cycles[i].get("solver_success")) if i < len(cycles) else None,
                         "mode_counts": dict(Counter(r["solve_mode"] for r in cycles[start:i]))})
    clear_values = [numeric(r.get("clearance_m")) for r in clearance]
    phi_values = [numeric(r.get("sdf_value")) for r in poses]
    valid_clearance_count = sum(v is not None for v in clear_values)
    valid_phi_count = sum(v is not None for v in phi_values)
    d_col = numeric(params.get("d_col"))
    event_specs = [
        (poses, [v is not None and v < 0 for v in phi_values], "current_psdf_negative", "pose_sdf", 0.0),
        (poses, [v is not None and v < -1e-6 for v in phi_values], "current_psdf_below_minus_1um", "pose_sdf", -1e-6),
        (clearance, [v == 0 for v in clear_values], "exact_clearance_zero_contact_or_overlap", "clearance", 0.0),
        (clearance, [v is not None and v < 0 for v in clear_values], "negative_clearance_map_boundary", "clearance", 0.0),
    ]
    if d_col is not None:
        event_specs.append((poses, [v is not None and v < d_col for v in phi_values],
                            "current_psdf_below_d_col", "pose_sdf", d_col))
    geometry_events = [event for data, flags, event_type, source, threshold in event_specs
                       for event in geometric_events(data, flags, event_type=event_type, source=source,
                                                     dt=dt, cycles=cycles, threshold=threshold)]
    timings = [numeric(r.get("cycle_time_s")) for r in cycles if truth(r.get("included_in_aggregate"))]
    recorded_stop_count = sum(stops) if cycles else runtime.get("safe_stop_count")
    final_time = numeric(row.get("final_time_s"))
    if final_time is None:
        final_time = numeric(outcome.get("final_time"))
    item = {"width_mm": width, "method": method, "seed": seed, "status": status,
            "success": reached, "failure_reason": row.get("failure_reason") or outcome.get("failure_reason"),
            "failure_type": failure_type(status, row.get("failure_reason") or outcome.get("failure_reason")),
            "final_time_s": final_time,
            "directory": str(directory), "initial_pose": initial, "cycle_count": len(cycles),
            "cycle_log_available": paths["cycles"].exists(), "pose_log_count": len(poses),
            "pose_log_available": valid_phi_count > 0, "clearance_log_available": valid_clearance_count > 0,
            "current_psdf_valid_samples": valid_phi_count, "clearance_valid_samples": valid_clearance_count,
            "current_psdf_missing_value_samples": len(phi_values) - valid_phi_count,
            "clearance_missing_value_samples": len(clear_values) - valid_clearance_count,
            "cycle_pose_counts_match": len(cycles) == len(poses) if cycles and poses else None,
            "clearance_covers_initial_and_each_recorded_call": valid_clearance_count == len(cycles) + 1 if cycles else None,
            "solver_attempted_cycles": len(cycles) - stage0 if cycles else None,
            "safe_stop_steps": recorded_stop_count, "safe_stop_episodes": len(episodes) if cycles else None,
            "terminal_safe_stop_episodes": sum(e["terminal"] for e in episodes) if cycles else None,
            "longest_safe_stop_episode_steps": max([e["steps"] for e in episodes], default=0) if cycles else None,
            "stage0_guard_stop_steps": stage0 if cycles else None,
            "qp_failure_stop_steps": qp if cycles else None,
            "other_safe_stop_steps": sum(stops) - stage0 - qp if cycles else None,
            "safe_stop_steps_per_100_controller_calls": 100 * sum(stops) / len(cycles) if cycles else None,
            "qp_failure_steps_per_100_solver_attempts": 100 * qp / (len(cycles) - stage0) if len(cycles) > stage0 else None,
            "mode_counts": dict(modes), "status_code_counts": dict(codes), "episodes": episodes,
            "runtime_stats": runtime, "parameters": params,
            "runtime_stop_count_matches_cycles": runtime.get("safe_stop_count") == sum(stops) if cycles else None,
            "outcome_matches_trial_csv": bool(outcome.get("goal_reached")) == reached if outcome else None,
            "clearance_m": dist(clear_values),
            "clearance_zero_samples": sum(v == 0 for v in clear_values if v is not None) if valid_clearance_count else None,
            "clearance_negative_boundary_samples": sum(v < 0 for v in clear_values if v is not None) if valid_clearance_count else None,
            "current_signed_psdf_m": dist(phi_values),
            "current_psdf_negative_samples": sum(v < 0 for v in phi_values if v is not None) if valid_phi_count else None,
            "current_psdf_below_minus_1um_samples": sum(v < -1e-6 for v in phi_values if v is not None) if valid_phi_count else None,
            "current_psdf_below_d_col_samples": sum(v < d_col for v in phi_values if v is not None) if d_col is not None and valid_phi_count else None,
            "current_psdf_within_1um_of_zero_samples": sum(abs(v) <= 1e-6 for v in phi_values if v is not None) if valid_phi_count else None,
            "first_geometric_events": {name: next((e for e in geometry_events if e["event_type"] == name), None)
                                       for name in ("current_psdf_negative", "current_psdf_below_minus_1um", "current_psdf_below_d_col",
                                                    "exact_clearance_zero_contact_or_overlap", "negative_clearance_map_boundary")},
            "geometric_events": geometry_events,
            "cycle_time_s_excluding_first": dist(timings),
            "_clearance": clear_values, "_phi": phi_values, "_timing": timings, "_mf_slack": [], "_mf_probability": []}
    if method == "cc_mf":
        mf_files = list((directory / "data").glob("cc_mf_cycle_*.csv"))
        if len(mf_files) == 1 and params:
            paths["mf_cycles"] = mf_files[0]
            coverage, slack, probability = mf_coverage(read_csv(mf_files[0]), params)
            item.update(mf_coverage=coverage, _mf_slack=slack, _mf_probability=probability)
    item["input_sources"] = [hash_file(path) for path in paths.values() if path.exists()]
    return item


def aggregate(trials):
    completed = [r for r in trials if r["status"] != "interrupted"]
    def total(key):
        return sum(r[key] for r in completed if r.get(key) is not None)
    n, calls = len(completed), total("cycle_count")
    attempts = total("solver_attempted_cycles")
    stops_with_cycle_logs = sum(r["safe_stop_steps"] for r in completed
                               if r["cycle_count"] and r["safe_stop_steps"] is not None)
    result = {"width_mm": trials[0]["width_mm"], "method": trials[0]["method"],
              "completed_trials": n, "successes": sum(r["success"] for r in completed),
              "failures_or_errors": sum(not r["success"] for r in completed),
              "success_rate": sum(r["success"] for r in completed) / n if n else None,
              "failure_type_counts": dict(Counter(r["failure_type"] for r in completed if not r["success"])),
              "exact_failure_reason_counts": dict(Counter(str(r["failure_reason"]) for r in completed if not r["success"])),
              "controller_calls_recorded": calls, "solver_attempts_recorded": attempts,
              "safe_stop_count_unavailable_trials": sum(r["safe_stop_steps"] is None for r in completed),
              "episode_count_unavailable_trials": sum(r["safe_stop_episodes"] is None for r in completed),
              **{key: total(key) for key in ("safe_stop_steps", "safe_stop_episodes", "terminal_safe_stop_episodes",
                                            "stage0_guard_stop_steps", "qp_failure_stop_steps", "other_safe_stop_steps",
                                            "clearance_zero_samples", "clearance_negative_boundary_samples",
                                            "current_psdf_negative_samples", "current_psdf_below_minus_1um_samples",
                                            "current_psdf_below_d_col_samples")},
              "trials_with_current_psdf_negative": sum((r["current_psdf_negative_samples"] or 0) > 0 for r in completed),
              "trials_with_current_psdf_below_minus_1um": sum((r["current_psdf_below_minus_1um_samples"] or 0) > 0 for r in completed),
              "trials_with_clearance_zero_contact_or_overlap": sum((r["clearance_zero_samples"] or 0) > 0 for r in completed),
              "trials_with_negative_clearance_boundary": sum((r["clearance_negative_boundary_samples"] or 0) > 0 for r in completed),
              "trials_with_known_current_psdf": sum(r["pose_log_available"] for r in completed),
              "trials_with_known_clearance": sum(r["clearance_log_available"] for r in completed),
              "current_psdf_unavailable_trials": sum(not r["pose_log_available"] for r in completed),
              "clearance_unavailable_trials": sum(not r["clearance_log_available"] for r in completed),
              "geometric_event_counts_are_lower_bounds": any(not r["pose_log_available"] or not r["clearance_log_available"]
                                                             or r["current_psdf_missing_value_samples"]
                                                             or r["clearance_missing_value_samples"]
                                                             or r["cycle_pose_counts_match"] is not True
                                                             or r["clearance_covers_initial_and_each_recorded_call"] is not True
                                                             for r in completed),
              "longest_safe_stop_episode_steps": max([r["longest_safe_stop_episode_steps"] for r in completed
                                                      if r["longest_safe_stop_episode_steps"] is not None], default=None),
              "safe_stop_steps_per_100_controller_calls": 100 * stops_with_cycle_logs / calls if calls else None,
              "qp_failure_steps_per_100_solver_attempts": 100 * total("qp_failure_stop_steps") / attempts if attempts else None,
              "successful_traversal_time_s": dist([r["final_time_s"] for r in completed if r["success"]]),
              "clearance_m_pooled": dist([v for r in completed for v in r["_clearance"]]),
              "current_signed_psdf_m_pooled": dist([v for r in completed for v in r["_phi"]]),
              "cycle_time_s_pooled_excluding_first": dist([v for r in completed for v in r["_timing"]])}
    coverage = [r["mf_coverage"] for r in completed if "mf_coverage" in r]
    if coverage:
        counts, active, branches = Counter(), Counter(), Counter()
        for item in coverage:
            counts.update(item["counts"])
            active.update(item["active_count_by_stage"])
            branches.update(item["branch_counts_on_active_rows"])
        valid = counts["valid_activation_cycles"]
        requested = coverage[0]["requested_stages"]
        result["mf_coverage"] = {"trials_with_mf_log": len(coverage), "counts": dict(counts),
                                 "active_count_by_stage": dict(active),
                                 "active_fraction_by_stage": {i: active[i] / valid if valid else None for i in requested},
                                 "branch_counts_on_active_rows": dict(branches),
                                 "active_slack_quantile_units": dist([v for r in completed for v in r["_mf_slack"]]),
                                 "active_nominal_probability_sum": dist([v for r in completed for v in r["_mf_probability"]])}
    return result


def pair_trials(trials, widths, seeds):
    lookup = {(r["width_mm"], r["method"], r["seed"]): r for r in trials}
    pairs, summaries = [], []
    for width in widths:
        local = []
        for seed in seeds:
            sf, mf = (lookup[(width, method, seed)] for method in ("cc_sf", "cc_mf"))
            pose_error = None
            if valid_pose(sf["initial_pose"]) and valid_pose(mf["initial_pose"]):
                pose_error = float(np.max(np.abs(np.asarray(sf["initial_pose"], dtype=float)
                                                - np.asarray(mf["initial_pose"], dtype=float))))
            interrupted = sf["status"] == "interrupted" or mf["status"] == "interrupted"
            both = sf["success"] and mf["success"] and not interrupted
            category = ("excluded_interrupted" if interrupted else "both_success" if both else
                        "sf_only_success" if sf["success"] else "mf_only_success" if mf["success"] else "both_fail")
            item = {"width_mm": width, "seed": seed, "paired_category": category,
                    "start_pose_max_abs_difference": pose_error,
                    "start_pose_match_within_1e_12": pose_error <= 1e-12 if pose_error is not None else None,
                    "sf_success": sf["success"], "mf_success": mf["success"],
                    "sf_status": sf["status"], "mf_status": mf["status"],
                    "sf_failure_reason": sf["failure_reason"], "mf_failure_reason": mf["failure_reason"],
                    "sf_safe_stop_steps": sf["safe_stop_steps"], "mf_safe_stop_steps": mf["safe_stop_steps"],
                    "sf_final_time_s": sf["final_time_s"], "mf_final_time_s": mf["final_time_s"],
                    "common_success_traversal_delta_mf_minus_sf_s": mf["final_time_s"] - sf["final_time_s"]
                    if both and sf["final_time_s"] is not None and mf["final_time_s"] is not None else None}
            local.append(item)
        summaries.append({"width_mm": width, "pair_count": len(local),
                          "categories": dict(Counter(r["paired_category"] for r in local)),
                          "starts_checked": sum(r["start_pose_max_abs_difference"] is not None for r in local),
                          "starts_verified_within_1e_12": sum(r["start_pose_match_within_1e_12"] is True for r in local),
                          "start_mismatches": sum(r["start_pose_match_within_1e_12"] is False for r in local),
                          "max_start_pose_difference": max([r["start_pose_max_abs_difference"] for r in local
                                                             if r["start_pose_max_abs_difference"] is not None], default=None),
                          "common_success_traversal_delta_mf_minus_sf_s": dist([r["common_success_traversal_delta_mf_minus_sf_s"] for r in local]),
                          "common_success_sf_time_s": dist([r["sf_final_time_s"] for r in local if r["paired_category"] == "both_success"]),
                          "common_success_mf_time_s": dist([r["mf_final_time_s"] for r in local if r["paired_category"] == "both_success"])})
        pairs.extend(local)
    return pairs, summaries


def flat(value, prefix=""):
    out = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(item, dict):
            out.update(flat(item, name))
        elif isinstance(item, list):
            out[name] = json.dumps(item, separators=(",", ":"))
        else:
            out[name] = item
    return out


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.results_dir.resolve()
    manifest = read_json(root / "experiment.json")
    if not manifest.get("finished_at_kst"):
        raise RuntimeError("Analyze only after the driving batch has finished.")
    widths = [int(round(float(w) * 1000)) for w in manifest["widths_m"]]
    methods = manifest["methods"]
    if set(methods) != {"cc_sf", "cc_mf"}:
        raise ValueError("This comparison expects exactly cc_sf and cc_mf.")
    seeds = manifest["seeds"]
    rows = read_csv(root / "trial_results.csv")
    keys = [(int(round(float(r["width_m"]) * 1000)), r["method"], int(r["seed"])) for r in rows]
    expected = {(w, m, s) for w in widths for m in methods for s in seeds}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Trial-result rows do not match the complete requested width/method/seed population.")
    trials = [load_trial(root, row) for row in rows]
    summary = [aggregate([r for r in trials if r["width_mm"] == width and r["method"] == method])
               for width in widths for method in methods]
    pairs, paired_summary = pair_trials(trials, widths, seeds)
    episodes = [{"width_mm": r["width_mm"], "method": r["method"], "seed": r["seed"], **e}
                for r in trials for e in r["episodes"]]
    geometry_events = [{"width_mm": r["width_mm"], "method": r["method"], "seed": r["seed"],
                       "trial_directory": r["directory"], **e} for r in trials for e in r["geometric_events"]]
    for item in trials:
        for key in ("_clearance", "_phi", "_timing", "_mf_slack", "_mf_probability"):
            item.pop(key)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "experiment": str(root),
              "scope": {"widths_mm": widths, "methods": methods, "seeds": seeds, "requested_trials": len(expected)},
              "summary": summary, "paired_summary": paired_summary, "paired_trials": pairs, "trials": trials,
              "integrity": {"requested_rows_present": len(rows), "duplicate_rows": len(keys) - len(set(keys)),
                            "pairs_with_checked_initial_poses": sum(r["start_pose_max_abs_difference"] is not None for r in pairs),
                            "pairs_with_verified_initial_poses": sum(r["start_pose_match_within_1e_12"] is True for r in pairs),
                            "pair_initial_pose_mismatches": sum(r["start_pose_match_within_1e_12"] is False for r in pairs),
                            "pair_start_pose_max_abs_difference": max([r["start_pose_max_abs_difference"] for r in pairs
                                                                        if r["start_pose_max_abs_difference"] is not None], default=None),
                            "cycle_log_missing_trials": sum(not r["cycle_log_available"] for r in trials),
                            "stop_counter_mismatch_trials": sum(r["runtime_stop_count_matches_cycles"] is False for r in trials),
                            "outcome_csv_mismatch_trials": sum(r["outcome_matches_trial_csv"] is False for r in trials)},
              "execution_environment": read_json(root / "execution_environment.json", {}),
              "manifest": manifest,
              "definitions_and_limits": [
                  "Success is recorded goal reach, not a probabilistic safety certificate. Completed failure/error/timeout trials remain in denominators; interruptions are excluded.",
                  "Stop steps and stop episodes are different units. An episode is a contiguous sequence of fallback controller calls; terminal means it reaches the final recorded call.",
                  "Per100 controller-call rates include stage0 prechecks. Solver-attempt rates exclude safe_stop_stage0. Early termination still changes exposure and these rates do not replace paired trial outcomes.",
                  "Traversal time averaged over each method's successful trials can have selection bias. Common-success paired times are also reported.",
                  "Clearance statistics pool initial and postintegration pose samples from successes and failures. Exact zero conflates contact and overlap; negative values indicate map-boundary intrusion.",
                  "Current signed PSDF is the measured pose logged before each controller call, not a predicted-horizon pose. Negative PSDF and below-minus1um counts are distinct from zero polygon clearance. The final postintegration pose may lack a signed-PSDF sample if no next controller call occurs.",
                  "A negative signed PSDF indicates overlap according to the geometric implementation; near-zero float32 values should be interpreted with the separate1um sensitivity count. First geometric events and associated stop modes are provided without attributing causality.",
                  "MF coverage excludes fallback cycles because those have no optimized slack and may retain old nominal diagnostics. Activation is stage-local; one inactive stage is not whole-horizon guard-only operation.",
                  "Activation coverage accepts only finite numeric0/1 flags on requested stages. Unknown flags, disabled rows and empty stage windows are separately counted rather than treated as inactive.",
                  "MF probabilities are pre-solve nominal Boole sums. Positive optimized slack or nominal R>epsilon is descriptive, and zero slack is not a nonlinear chance-constraint certificate.",
                  "The SF implementation retains hard, ungated chance rows whereas MF rows are soft and domain-gated. This sweep compares complete controllers and does not isolate feature aggregation.",
                  "No local feature-switch sensitivity or solver-failure causality is inferred from these aggregate logs. Each width has only ten paired seeds.",
                  "Geometric counts use known finite observations; missing logs are unavailable, not collision-free. Known-data trial denominators and lower-bound flags accompany aggregate counts. Event/cycle metadata is attached only after sample IDs (and clearance simulation times) are verified.",
              ],
              "analysis_source": hash_file(Path(__file__)),
              "top_level_sources": [hash_file(root / name) for name in ("experiment.json", "trial_results.csv", "source_hashes.json")
                                    if (root / name).exists()]}
    output = (args.output_dir or root / "transition_analysis").resolve()
    output.mkdir(exist_ok=True, parents=True)
    with (output / "comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    write_csv(output / "width_summary.csv", [flat(r) for r in summary])
    write_csv(output / "paired_trials.csv", pairs)
    write_csv(output / "paired_summary.csv", [flat(r) for r in paired_summary])
    excluded = {"parameters", "input_sources", "episodes", "mf_coverage", "geometric_events", "first_geometric_events"}
    write_csv(output / "trial_metrics.csv", [flat({k: v for k, v in r.items() if k not in excluded}) for r in trials])
    write_csv(output / "safe_stop_episodes.csv", [flat(r) for r in episodes])
    write_csv(output / "geometric_events.csv", geometry_events)
    write_csv(output / "mf_coverage.csv", [flat({"width_mm": r["width_mm"], "seed": r["seed"], **r["mf_coverage"]})
                                           for r in trials if "mf_coverage" in r])
    print(json.dumps({"output": str(output), "integrity": report["integrity"],
                      "summary": [{k: r[k] for k in ("width_mm", "method", "successes", "completed_trials",
                                                     "safe_stop_steps", "safe_stop_episodes", "stage0_guard_stop_steps",
                                                     "qp_failure_stop_steps")} for r in summary]}, indent=2))


if __name__ == "__main__":
    main()
