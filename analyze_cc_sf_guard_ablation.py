"""Compare completed SF+Guard trials with archived SF and MF, without solving.

Only existing CSV/JSON/checkpoint inputs are read. Optional stage-zero auditing
uses archived geometry and initial covariance; it never instantiates a solver.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "benchmark_runs/rectangle_width070_130mm_step005_10trials_20261103"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def distribution(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "p05": None, "p95": None, "min": None}
    return {"n": len(values), "mean": float(values.mean()),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)), "min": float(values.min())}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stage_zero_audit(experiment, directory, poses, params):
    import torch
    import yaml
    from utils.offline_cc_margin import _maze_polygons, _snapshot_imports

    torch.set_num_threads(6)
    config = yaml.safe_load((directory / "config.yaml").read_text())
    length, width = (config["rectangle_geometry"][key] for key in ("length", "width"))
    pose_array = np.array([[float(r[k]) for k in ("x", "y", "theta")] for r in poses])
    with _snapshot_imports(experiment / "source_snapshot") as (module, covariance, _):
        vertices = torch.tensor([[-length / 2, -width / 2], [length / 2, -width / 2],
                                 [length / 2, width / 2], [-length / 2, width / 2]],
                                dtype=torch.float32)
        wrapper = module.AugmentedPSDFWrapper(vertices, K_max=20, E_max=100, device="cpu")
        polygons = _maze_polygons(experiment / "source_snapshot/sim/simulation_mpc.py")
        wrapper.update_edge_clusters([torch.tensor(p, dtype=torch.float32) for p in polygons],
                                     [torch.tensor(np.roll(p, -1, axis=0), dtype=torch.float32)
                                      for p in polygons])
        with torch.no_grad():
            phi, gradient = wrapper(torch.tensor(pose_array, dtype=torch.float32))
        phi = phi.numpy().reshape(-1).astype(float)
        gradient = gradient.numpy().astype(float)
        q = gradient.copy()
        c, s = np.cos(pose_array[:, 2]), np.sin(pose_array[:, 2])
        q[:, 0] = c * gradient[:, 0] + s * gradient[:, 1]
        q[:, 1] = -s * gradient[:, 0] + c * gradient[:, 1]
        param = SimpleNamespace(**params)
        xi = np.array([covariance.log_cholesky_from_covariance(
            covariance.initial_local_covariance(param, SimpleNamespace(_x=pose)))
            for pose in pose_array])
        variance = ((np.exp(xi[:, 0]) * q[:, 0]) ** 2
                    + (np.exp(xi[:, 1]) * q[:, 1] + xi[:, 3] * q[:, 2]) ** 2
                    + (np.exp(xi[:, 2]) * q[:, 2]) ** 2)
        gamma = params.get("chance_gamma")
        if gamma is None:
            gamma = NormalDist().inv_cdf(1 - params["chance_epsilon"])
        risk = np.clip(gamma * np.sqrt(variance), params.get("risk_margin_floor", 0),
                       params.get("risk_margin_cap", np.inf))
        h = phi - params["d_min"] - risk
        error = float(np.max(np.abs(phi - [float(r["sdf_value"]) for r in poses])))
        return h, {"definition": "Fresh SF residual at recorded cycle-initial pose and reset covariance",
                   "geometry": "Archived CPU float32; original run uses CUDA float32",
                   "max_phi_replay_error_m": error, "h_m": distribution(h.tolist()),
                   "negative_below_minus_1um_cycles": int(np.sum(h < -1e-6)),
                   "near_zero_within_1um_cycles": int(np.sum(np.abs(h) <= 1e-6)),
                   "negative_is_not_a_solver_infeasibility_certificate": True}


def load_trial(experiment, label, method, width, seed, audit):
    directory = experiment / f"width_{width:03d}mm" / method / f"seed_{seed}"
    outcome_path = directory / "outcome.json"
    if not outcome_path.exists():
        return None
    outcome = json.loads(outcome_path.read_text())
    cycles = read_csv(directory / "cycles.csv")
    params = json.loads((directory / "optimizer_parameters.json").read_text())
    pose_files = list((directory / "data").glob("pose_sdf_*.csv"))
    poses = read_csv(pose_files[0]) if len(pose_files) == 1 else []
    dt = params["tf"] / params["horizon"]
    safe = [str(r.get("solve_mode", "")).startswith("safe_stop") for r in cycles]
    success = [str(r.get("solver_success", "")).lower() == "true" for r in cycles]
    h, stage_audit = (None, None)
    if audit and poses and len(poses) == len(cycles):
        h, stage_audit = stage_zero_audit(experiment, directory, poses, params)
        stage_audit["safe_stop_negative_below_minus_1um_cycles"] = int(
            sum(stopped and value < -1e-6 for stopped, value in zip(safe, h)))
        stage_audit["nonstop_negative_below_minus_1um_cycles"] = int(
            sum(not stopped and value < -1e-6 for stopped, value in zip(safe, h)))
    episodes = []
    index = 0
    while index < len(cycles):
        if not safe[index]:
            index += 1
            continue
        start = index
        while index < len(cycles) and safe[index]:
            index += 1
        last = index - 1
        row = poses[start] if start < len(poses) else {}
        episode = {"start_step": start, "last_step": last, "steps": index - start,
                   "start_time_s": start * dt, "end_time_s": index * dt,
                   "duration_s": (index - start) * dt,
                   "start_progress_m": number(row.get("path_s")),
                   "start_psdf_m": number(row.get("sdf_value")),
                   "modes": dict(Counter(r["solve_mode"] for r in cycles[start:index])),
                   "status_codes": dict(Counter(str(r.get("solver_status_code"))
                                                for r in cycles[start:index])),
                   "terminal_episode": index == len(cycles),
                   "next_cycle_solver_success": success[index] if index < len(cycles) else None,
                   "stage0_sf_h_start_m": float(h[start]) if h is not None else None,
                   "stage0_sf_h_min_m": float(np.min(h[start:index])) if h is not None else None}
        episodes.append(episode)
    clearance = read_csv(directory / "clearance.csv") if (directory / "clearance.csv").exists() else []
    clear_values = [number(r.get("clearance_m")) for r in clearance]
    history_path = directory / "data/trial_history.csv"
    history = read_csv(history_path) if history_path.exists() else []
    controls = np.array([[float(r[k]) for k in ("v", "omega")] for r in history])
    deltas = np.abs(np.diff(controls, axis=0)) if len(controls) > 1 else np.empty((0, 2))
    timings = [number(r.get("cycle_time_s")) for r in cycles
               if r.get("included_in_aggregate") == "1"]
    runtime = outcome.get("optimizer_runtime_stats", outcome.get("runtime_stats", {}))
    initial = outcome.get("initial_pose", outcome.get("start_perturbation", {}).get("initial_pose"))
    input_paths = [outcome_path, directory / "cycles.csv", directory / "optimizer_parameters.json",
                   directory / "config.yaml", *pose_files]
    input_paths += [p for p in (history_path, directory / "clearance.csv") if p.exists()]
    result = {"label": label, "method": method, "seed": seed, "directory": str(directory),
              "status": outcome.get("status"), "goal_reached": bool(outcome.get("goal_reached")),
              "failure_reason": outcome.get("failure_reason"), "final_time_s": outcome.get("final_time"),
              "initial_pose": initial, "cycle_count": len(cycles), "safe_stop_steps": sum(safe),
              "safe_stop_episodes": len(episodes),
              "longest_safe_stop_episode_steps": max([e["steps"] for e in episodes], default=0),
              "stage0_guard_stop_steps": sum(r.get("solve_mode") == "safe_stop_stage0" for r in cycles),
              "solver_failure_stop_steps": sum(r.get("solve_mode") == "safe_stop" for r in cycles),
              "safe_stop_steps_per_100_cycles": 100 * sum(safe) / len(cycles) if cycles else None,
              "mode_counts": dict(Counter(r.get("solve_mode") for r in cycles)),
              "status_counts": dict(Counter(str(r.get("solver_status_code")) for r in cycles)),
              "runtime_stats": runtime, "episodes": episodes,
              "logged_stop_counter_matches_cycles": runtime.get("safe_stop_count") == sum(safe),
              "clearance_m": distribution(clear_values), "cycle_time_s_excluding_first": distribution(timings),
              "clearance_zero_samples": sum(v == 0 for v in clear_values if v is not None),
              "clearance_negative_samples": sum(v < 0 for v in clear_values if v is not None),
              "absolute_control_delta_v_m_s": distribution(deltas[:, 0].tolist()),
              "absolute_control_delta_omega_rad_s": distribution(deltas[:, 1].tolist()),
              "stage0_sf_reconstruction": stage_audit, "parameters": params,
              "input_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in input_paths}}
    mf_files = list((directory / "data").glob("cc_mf_cycle_*.csv"))
    if method == "cc_mf" and len(mf_files) == 1:
        result["input_hashes"][str(mf_files[0])] = hashlib.sha256(mf_files[0].read_bytes()).hexdigest()
        rows = read_csv(mf_files[0])
        all_inactive, stage_active, valid = 0, Counter(), 0
        maximum_slack = 0.0
        for row in rows:
            active = json.loads(row.get("mf_active") or "[]")
            slack = json.loads(row.get("mf_slack") or "[]")
            if not active or row.get("solve_mode", "").startswith("safe_stop"):
                continue
            valid += 1
            all_inactive += not any(active[1:])
            stage_active.update(i for i, value in enumerate(active) if value)
            maximum_slack = max([maximum_slack] + [float(v) for v in slack if v is not None])
        result["mf_telemetry"] = {"evaluated_nonstop_cycles": valid,
                                  "all_future_rows_inactive_cycles": all_inactive,
                                  "active_count_by_stage": dict(stage_active),
                                  "maximum_recorded_slack_quantile": maximum_slack,
                                  "interpretation": "Stage-local inactivity is distinct from all-future-row inactivity; slack does not certify nonlinear chance feasibility."}
    result["_clearance_values"] = clear_values
    result["_timings"] = timings
    return result


def summarize(trials):
    completed = [r for r in trials if r["status"] != "interrupted"]
    n = len(completed)
    steps = sum(r["cycle_count"] for r in completed)
    return {"completed_trials": n, "successes": sum(r["goal_reached"] for r in completed),
            "failed_or_error_trials": sum(not r["goal_reached"] for r in completed),
            "success_rate": sum(r["goal_reached"] for r in completed) / n if n else None,
            "safe_stop_steps": sum(r["safe_stop_steps"] for r in completed),
            "safe_stop_episodes": sum(r["safe_stop_episodes"] for r in completed),
            "longest_safe_stop_episode_steps": max([r["longest_safe_stop_episode_steps"] for r in completed], default=0),
            "terminal_safe_stop_episodes": sum(e["terminal_episode"] for r in completed for e in r["episodes"]),
            "stage0_guard_stop_steps": sum(r["stage0_guard_stop_steps"] for r in completed),
            "solver_failure_stop_steps": sum(r["solver_failure_stop_steps"] for r in completed),
            "recorded_cycles": steps,
            "safe_stop_steps_per_100_cycles": 100 * sum(r["safe_stop_steps"] for r in completed) / steps if steps else None,
            "successful_traversal_time_s": distribution([r["final_time_s"] for r in completed if r["goal_reached"]]),
            "clearance_m_pooled": distribution([v for r in completed for v in r["_clearance_values"]]),
            "clearance_zero_samples": sum(r["clearance_zero_samples"] for r in completed),
            "clearance_negative_samples": sum(r["clearance_negative_samples"] for r in completed),
            "cycle_time_s_pooled_excluding_first": distribution([v for r in completed for v in r["_timings"]])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", "--new-results", dest="new_results", type=Path, required=True)
    parser.add_argument("--reference-run", "--baseline", dest="baseline", type=Path, default=BASELINE)
    parser.add_argument("--width-mm", type=int, default=125)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--stage0-audit", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    groups = [("original_sf", args.baseline, "cc_sf"), ("original_mf", args.baseline, "cc_mf"),
              ("sf_guard", args.new_results, "cc_sf")]
    all_trials, summaries = [], {}
    for label, experiment, method in groups:
        trials = [load_trial(experiment.resolve(), label, method, args.width_mm, seed, args.stage0_audit)
                  for seed in range(args.trials)]
        trials = [r for r in trials if r is not None]
        summaries[label] = summarize(trials)
        all_trials.extend(trials)
    paired = []
    for seed in range(args.trials):
        rows = {r["label"]: r for r in all_trials if r["seed"] == seed}
        if len(rows) < 3:
            continue
        poses = [r["initial_pose"] for r in rows.values()]
        old_params, new_params = rows["original_sf"]["parameters"], rows["sf_guard"]["parameters"]
        paired.append({"seed": seed, "initial_pose_max_difference":
                       float(np.max(np.abs(np.array(poses) - poses[0]))) if all(p is not None for p in poses) else None,
                       "sf_parameter_differences": {key: {"original": old_params.get(key), "guard": new_params.get(key)}
                                                    for key in sorted(set(old_params) | set(new_params))
                                                    if old_params.get(key) != new_params.get(key)},
                       "outcomes": {label: {k: r[k] for k in ("goal_reached", "status", "safe_stop_steps", "safe_stop_episodes", "failure_reason")}
                                    for label, r in rows.items()}})
    for trial in all_trials:
        trial.pop("_clearance_values")
        trial.pop("_timings")
    report = {"summary": summaries, "paired_trials": paired, "trials": all_trials,
              "analysis_source": {"path": str(Path(__file__).resolve()),
                                  "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "limitations": ["This ablation changes stage-zero handling and the terminal stage as well as adding Guard; it does not isolate feature aggregation.",
                              "SF chance rows remain hard and ungated, whereas MF chance rows are soft and domain-gated.",
                              "With shared geometry, d_min >= d_col and nonnegative risk, a hard SF row implies the Guard row; future Guard is algebraically redundant.",
                              "Safe-stop episodes are consecutive fallback cycles, not independent trials. Exposure-normalized stop counts remain affected by early termination.",
                              "Fresh stage-zero residual reconstruction uses archived CPU float32 and is descriptive, not a certificate of solver infeasibility or failure causality.",
                              "All recorded trials, including failed trials, enter stop/clearance/timing summaries. Successful traversal time is success-conditioned.",
                              "Feature switching immediately before solver failures is not identified by these aggregate logs; the separate boundary audit concerns local residual sensitivity."]}
    environment_path = args.new_results / "execution_environment.json"
    if environment_path.exists():
        report["new_run_execution_environment"] = json.loads(environment_path.read_text())
        report["new_run_execution_environment_source"] = {
            "path": str(environment_path.resolve()),
            "sha256": hashlib.sha256(environment_path.read_bytes()).hexdigest(),
        }
    output = (args.output_dir or args.new_results / "sf_guard_comparison").resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "comparison.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    trial_rows = []
    episode_rows = []
    for trial in all_trials:
        scalar_keys = ("label", "method", "seed", "directory", "status", "goal_reached",
                       "failure_reason", "final_time_s", "cycle_count", "safe_stop_steps",
                       "safe_stop_episodes", "longest_safe_stop_episode_steps",
                       "stage0_guard_stop_steps", "solver_failure_stop_steps",
                       "safe_stop_steps_per_100_cycles", "clearance_zero_samples",
                       "clearance_negative_samples")
        row = {k: trial[k] for k in scalar_keys}
        for key, prefix in (("clearance_m", "clearance_m"),
                            ("cycle_time_s_excluding_first", "cycle_time_s"),
                            ("absolute_control_delta_v_m_s", "abs_delta_v_m_s"),
                            ("absolute_control_delta_omega_rad_s", "abs_delta_omega_rad_s")):
            row.update({f"{prefix}_{k}": v for k, v in trial[key].items()})
        trial_rows.append(row)
        for episode in trial["episodes"]:
            episode_rows.append({"label": trial["label"], "seed": trial["seed"],
                                 **{k: json.dumps(v) if isinstance(v, dict) else v
                                    for k, v in episode.items()}})
    summary_rows = []
    for label, values in summaries.items():
        row = {"label": label}
        for key, value in values.items():
            if isinstance(value, dict):
                row.update({f"{key}_{k}": v for k, v in value.items()})
            else:
                row[key] = value
        summary_rows.append(row)
    write_csv(output / "comparison_trials.csv", trial_rows)
    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "safe_stop_episodes.csv", episode_rows)
    print(json.dumps({"path": str(path), "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
