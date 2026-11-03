"""Offline CC-MF trial analysis; never runs a solver or a driving experiment.

Usage: python analyze_cc_mf_oscillation.py <batch-or-results-dir>
Optional --geometry JSON verifies signed polygon clearance at recorded poses.
Resolved parameters are required only for the optional exact covariance audit.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def column(rows, name):
    return np.asarray([float(row[name]) if row.get(name) else np.nan for row in rows])


def statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "p99": float(np.quantile(values, .99)), "min": float(values.min()),
            "max": float(values.max())}


def longest_streak(mask):
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def sign_flips(values, threshold):
    return ((values[:-1] * values[1:] < 0)
            & (np.minimum(abs(values[:-1]), abs(values[1:])) > threshold))


def motion_metrics(states, controls, dt):
    duration = len(controls) * dt
    v, omega = controls.T if len(controls) else (np.array([]), np.array([]))
    vf, wf = sign_flips(v, .001), sign_flips(omega, .01)
    dual_all = sign_flips(v, 0) & sign_flips(omega, 0)
    deltas = np.diff(controls, axis=0)
    xy = states[:, :2]
    return {
        "duration_s": duration, "control_samples": len(controls),
        "net_displacement_m": float(np.linalg.norm(xy[-1] - xy[0])),
        "accumulated_travel_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        "heading_total_variation_rad": float(abs(np.diff(np.unwrap(states[:, 2]))).sum()),
        "v_total_variation_m_s": float(abs(deltas[:, 0]).sum()),
        "omega_total_variation_rad_s": float(abs(deltas[:, 1]).sum()),
        "v_delta_rms_m_s": float(np.sqrt(np.mean(deltas[:, 0] ** 2))) if len(deltas) else None,
        "omega_delta_rms_rad_s": float(np.sqrt(np.mean(deltas[:, 1] ** 2))) if len(deltas) else None,
        "mean_abs_v_m_s": float(np.mean(abs(v))) if len(v) else None,
        "mean_abs_omega_rad_s": float(np.mean(abs(omega))) if len(omega) else None,
        "v_reversal_count": int(vf.sum()), "omega_reversal_count": int(wf.sum()),
        "v_reversals_per_s": float(vf.sum() / duration) if duration else None,
        "omega_reversals_per_s": float(wf.sum() / duration) if duration else None,
        "dual_reversal_count": int((vf & wf).sum()),
        "dual_reversal_longest_pairs": longest_streak(vf & wf),
        "dual_reversal_all_amplitudes_count": int(dual_all.sum()),
        "dual_reversal_all_amplitudes_longest_pairs": longest_streak(dual_all),
        "lag2_xy_return_error_m": statistics(np.linalg.norm(xy[2:] - xy[:-2], axis=1)),
    }


def signed_convex_clearance(first, second):
    """Euclidean separation outside, negative SAT translation depth inside."""
    edges_a = np.roll(first, -1, axis=0) - first
    edges_b = np.roll(second, -1, axis=0) - second
    edges = np.vstack((edges_a, edges_b))
    axes = np.column_stack((-edges[:, 1], edges[:, 0]))
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-12] / norms[norms > 1e-12, None]
    a, b = first @ axes.T, second @ axes.T
    gaps = np.maximum(b.min(axis=0) - a.max(axis=0), a.min(axis=0) - b.max(axis=0))
    if np.max(gaps) <= 0:
        return float(np.max(gaps))

    def point_edge_distance(points, vertices, segments):
        delta = points[:, None, :] - vertices[None, :, :]
        fraction = np.sum(delta * segments[None, :, :], axis=2) / np.maximum(
            np.sum(segments ** 2, axis=1), 1e-24)
        difference = delta - np.clip(fraction, 0, 1)[:, :, None] * segments
        return float(np.linalg.norm(difference, axis=2).min())

    return min(point_edge_distance(first, second, edges_b),
               point_edge_distance(second, first, edges_a))


def geometry_metrics(states, geometry, dt):
    vertices = np.asarray(geometry["footprint_vertices_local"], dtype=float)
    obstacles = [np.asarray(p.get("vertices", p.get("polygon")) if isinstance(p, dict)
                            else p, dtype=float) for p in geometry["obstacles"]]
    lower, upper = np.asarray(geometry["bounds"], dtype=float)
    clearance = []
    for x, y, heading in states:
        c, s = np.cos(heading), np.sin(heading)
        polygon = vertices @ np.array([[c, s], [-s, c]]) + [x, y]
        boundary = min((polygon.min(axis=0) - lower).min(),
                       (upper - polygon.max(axis=0)).min())
        clearance.append(min([boundary] + [signed_convex_clearance(polygon, p)
                                           for p in obstacles]))
    clearance = np.asarray(clearance)
    minimum_index = int(np.argmin(clearance))
    penetration_indices = np.flatnonzero(clearance < -1e-9)
    return {"signed_polygon_clearance_m": statistics(clearance),
            "minimum_pose_index": minimum_index,
            "minimum_pose_time_s": minimum_index * dt,
            "minimum_pose_xytheta": states[minimum_index].tolist(),
            "sampled_pose_penetration_count": len(penetration_indices),
            "first_penetration_time_s": int(penetration_indices[0]) * dt
            if len(penetration_indices) else None,
            "sampled_pose_contact_or_penetration_count": int(np.sum(clearance <= 1e-9))}


def covariance_audit(path, parameters, dt, trial, seed, source):
    # Import only the covariance algebra, without initializing controllers/GPU.
    spec = importlib.util.spec_from_file_location("offline_covariance", source)
    covariance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(covariance)
    param = SimpleNamespace(**parameters)
    robust_horizon = covariance.resolve_robust_horizon(param)
    cycles = defaultdict(list)
    for row in read_csv(path):
        cycles[float(row["time"])].append(row)
    result = []
    for time, rows in cycles.items():
        rows.sort(key=lambda row: int(row["stage"]))
        if len(rows) < 2:
            continue
        try:
            matrices = np.zeros((len(rows), 3, 3))
            for k, row in enumerate(rows):
                matrices[k] = np.diag([float(row[key]) ** 2
                                       for key in ("sigma_f", "sigma_l", "sigma_psi")])
                matrices[k, 1, 2] = matrices[k, 2, 1] = float(row["P_lpsi"])
            predicted = covariance.log_cholesky_from_covariance(matrices)
            exact = [predicted[0]]
            for k, row in enumerate(rows[:-1]):
                exact.append(covariance.propagate_log_cholesky(
                    exact[-1], float(row["v"]), float(row["omega"]), 0., dt, param)
                             if k < robust_horizon else exact[-1].copy())
            exact = np.asarray(exact)
            exact_sigma = np.sqrt(covariance.covariance_from_log_cholesky(exact)[:, 2, 2])
            predicted_sigma = np.sqrt(matrices[:, 2, 2])
            ratio = exact_sigma[1:] / predicted_sigma[1:]
            result.append({"trial": trial, "seed": seed, "time": time,
                           "stage1_heading_sigma_exact_over_predicted": float(ratio[0]),
                           "horizon_heading_sigma_max_exact_over_predicted": float(ratio.max()),
                           "max_log_factor_rollout_error": float(abs(exact[:, :3] - predicted[:, :3]).max())})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            result.append({"trial": trial, "seed": seed, "time": time, "audit_error": str(exc)})
    return result


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--geometry", type=Path)
    args = parser.parse_args()
    base = args.run_dir.resolve()
    if not (base / "trial_results.csv").exists():
        base /= "results"
    output = args.output_dir or base.parent / "oscillation_analysis"
    output.mkdir(parents=True, exist_ok=True)
    parameter_path = args.parameters or base.parent / "resolved_cc_mf_parameters.json"
    parameters = json.loads(parameter_path.read_text()) if parameter_path.exists() else None
    covariance_source = base.parent / "source_snapshot/control/cc_covariance.py"
    if not covariance_source.exists():
        covariance_source = Path(__file__).parent / "control/cc_covariance.py"
    geometry_path = args.geometry or base.parent / "environment_geometry.json"
    geometry = json.loads(geometry_path.read_text()) if geometry_path.exists() else None
    outcomes = read_csv(base / "trial_results.csv")
    all_solver, all_controller, steady_controller, summaries, flat, audits = [], [], [], [], [], []
    total_delta_squares = np.zeros(2)
    total_delta_count = 0
    for outcome in outcomes:
        trial = int(outcome["trial_number"])
        seed = int(outcome["start_seed"]) if outcome.get("start_seed") else None
        directory = base / "runs" / outcome["configuration_id"] / f"trial_{trial:04d}" / "data"
        history_paths = sorted(directory.glob("trial_history_*.csv"))
        if not history_paths:
            continue
        rows = read_csv(history_paths[0])
        if not rows:
            continue
        times = column(rows, "time")
        dt = float(np.median(np.diff(np.r_[0., times])))
        states = np.vstack(([float(outcome[key]) for key in ("initial_x", "initial_y", "initial_theta")],
                            np.column_stack([column(rows, k) for k in ("x", "y", "theta")])))
        controls = np.column_stack([column(rows, k) for k in ("v", "omega")])
        success = np.asarray([row.get("solver_success") == "True" for row in rows])
        solver = column(rows, "solver_computation_time")[success]
        controller = column(rows, "controller_computation_time")
        all_solver.extend(solver)
        all_controller.extend(controller)
        steady_controller.extend(controller[1:])
        delta = np.diff(controls, axis=0)
        total_delta_squares += (delta ** 2).sum(axis=0)
        total_delta_count += len(delta)
        summary = {"trial": trial, "seed": seed, "status": outcome["status"],
                   "failure_reason": outcome["failure_reason"], "final_time_s": float(times[-1]),
                   "dt_s": dt, "history_path": str(history_paths[0]),
                   "whole_run": motion_metrics(states, controls, dt),
                   "solver_successful_time_s": statistics(solver),
                   "controller_all_time_s": statistics(controller),
                   "controller_without_first_cycle_time_s": statistics(controller[1:]),
                   "controller_deadline_overrun_count": int(np.sum(controller > dt)),
                   "controller_without_first_cycle_overrun_count": int(np.sum(controller[1:] > dt)),
                   "solver_status_counts": dict(Counter(row["solver_status_code"] for row in rows)),
                   "unsuccessful_control_cycles": int((~success).sum()),
                   "max_consecutive_unsuccessful_control_cycles": longest_streak(~success)}
        for seconds in (4, 20):
            first = int(np.searchsorted(times, times[-1] - seconds + 1e-8, side="right"))
            window = motion_metrics(states[first:], controls[first:], dt)
            window["window_complete"] = bool(times[-1] >= seconds - 1e-8)
            summary[f"tail{seconds}"] = window
        pose_files = sorted(directory.glob("pose_sdf_*.csv"))
        if pose_files:
            pose_rows = read_csv(pose_files[0])
            sdf = column(pose_rows, "sdf_value")
            summary["logged_precontrol_psdf_m"] = statistics(sdf)
            summary["logged_precontrol_negative_psdf_count"] = int(np.sum(sdf < 0))
        if geometry is not None:
            summary["geometry"] = geometry_metrics(states, geometry, dt)
        cov_files = sorted(directory.glob("cc_mf_covariance_*.csv"))
        if parameters is not None and cov_files:
            cycle_audits = covariance_audit(cov_files[0], parameters, dt, trial, seed, covariance_source)
            for audit in cycle_audits:
                cycle_index = int(round(audit["time"] / dt))
                audit["successful_control_cycle"] = bool(
                    cycle_index < len(success) and success[cycle_index])
            audits.extend(cycle_audits)
            for key, threshold in (("covariance", -np.inf), ("tail4_covariance", times[-1] - 4 - 1e-8)):
                chosen = [r for r in cycle_audits if r["time"] >= threshold
                          and "audit_error" not in r and r["successful_control_cycle"]]
                summary[key] = {"cycles": len(chosen), "stage1_heading_sigma_exact_over_predicted":
                                statistics([r["stage1_heading_sigma_exact_over_predicted"] for r in chosen]),
                                "max_log_factor_rollout_error":
                                statistics([r["max_log_factor_rollout_error"] for r in chosen])}
            summary["covariance_audit_errors"] = sum("audit_error" in r for r in cycle_audits)
        summaries.append(summary)
        record = {k: summary[k] for k in ("trial", "seed", "status", "failure_reason", "final_time_s")}
        for prefix in ("whole_run", "tail4", "tail20"):
            record.update({f"{prefix}_{key}": value for key, value in summary[prefix].items()
                           if not isinstance(value, dict)})
        for key in ("solver_successful_time_s", "controller_all_time_s", "controller_without_first_cycle_time_s"):
            record.update({f"{key}_{stat}": value for stat, value in summary[key].items()})
        record["min_logged_psdf_m"] = summary.get("logged_precontrol_psdf_m", {}).get("min")
        record["min_signed_polygon_clearance_m"] = summary.get("geometry", {}).get("signed_polygon_clearance_m", {}).get("min")
        flat.append(record)
    duration = sum(r["whole_run"]["duration_s"] for r in summaries)
    completed = sum(r["status"] != "interrupted" for r in outcomes)
    aggregate = {"recorded_trials": len(outcomes), "analyzed_trials": len(summaries),
                 "completed_trials": completed, "status_counts": dict(Counter(r["status"] for r in outcomes)),
                 "success_rate": sum(r["status"] == "success" for r in outcomes) / completed if completed else None,
                 "simulation_duration_s": duration,
                 "solver_successful_time_s": statistics(all_solver),
                 "controller_all_time_s": statistics(all_controller),
                 "controller_without_first_cycle_time_s": statistics(steady_controller)}
    for name in ("v", "omega"):
        count = sum(r["whole_run"][f"{name}_reversal_count"] for r in summaries)
        aggregate[f"{name}_reversal_count"] = count
        aggregate[f"{name}_reversals_per_s"] = count / duration if duration else None
    aggregate["input_delta_rms_v_omega"] = (np.sqrt(total_delta_squares / total_delta_count).tolist()
                                                   if total_delta_count else None)
    document = {"source": str(base), "parameters_source": str(parameter_path) if parameters else None,
                "covariance_algebra_source": str(covariance_source),
                "geometry_source": str(geometry_path) if geometry else None,
                "definitions": {"thresholds": "Both adjacent |v| > .001 m/s; |omega| > .01 rad/s.",
                                "latency": "Pooled cycles. Successful solver calls only; controller includes failures and setup.",
                                "geometry": "Recorded initial/postcontrol poses only; no continuous-time guarantee.",
                                "covariance": "Offline exact rollout from stage0 using logged full horizon inputs; snapshot covariance algebra when available, saved resolved parameters. Summary statistics use successful cycles only.",
                                "tail_windows": "Controls after window start, inclusive state endpoints; 20s displacement matches failure checker."},
                "aggregate": aggregate, "trials": summaries}
    (output / "summary.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    write_csv(output / "trial_summary.csv", flat)
    write_csv(output / "covariance_audit_cycles.csv", audits)
    print(json.dumps(aggregate, indent=2, allow_nan=False))
    print(f"Analysis saved in {output}")


if __name__ == "__main__":
    main()
