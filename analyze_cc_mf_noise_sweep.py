"""Offline, incremental analysis of run_cc_mf_noise_sweep.py outputs.

No controller, dynamics, PSDF, or solver modules are imported. Completed trial
CSV rows are the denominator; interrupted trials are excluded. Geometry checks
cover the recorded initial pose and every post-control pose, including the last
pose. They do not certify safety between samples or under localization error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


CACHE_VERSION = 2
PENETRATION_TOLERANCE_M = 1e-9


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def truth(value: Any) -> bool:
    return str(value).lower() in {"1", "true"}


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, handle.getvalue())


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Accept paths relative to the manifest, or repository-relative paths.
    relative = root / path
    return relative if relative.exists() or not path.exists() else path.resolve()


def signed_clearances(states: np.ndarray, geometry: dict) -> np.ndarray:
    """Vectorized convex polygon distance; negative SAT translation depth.

    Obstacles must be convex, as in the benchmark maze generators. Outside
    contact this is Euclidean edge/vertex distance, not a separating-axis gap.
    """
    vertices = np.asarray(geometry["footprint_vertices_local"], dtype=float)
    lower, upper = np.asarray(geometry["bounds"], dtype=float)
    cos, sin = np.cos(states[:, 2]), np.sin(states[:, 2])
    rotation = np.stack((cos, -sin, sin, cos), axis=1).reshape(-1, 2, 2)
    robots = np.einsum("mij,vj->mvi", rotation, vertices) + states[:, None, :2]
    clearance = np.minimum((robots.min(axis=1) - lower).min(axis=1),
                           (upper - robots.max(axis=1)).min(axis=1))
    robot_edges = np.roll(robots, -1, axis=1) - robots
    robot_normals = np.stack((-robot_edges[..., 1], robot_edges[..., 0]), axis=2)
    robot_normals /= np.linalg.norm(robot_normals, axis=2, keepdims=True)
    for raw in geometry["obstacles"]:
        obstacle = np.asarray(raw.get("vertices", raw.get("polygon"))
                              if isinstance(raw, dict) else raw, dtype=float)
        edges = np.roll(obstacle, -1, axis=0) - obstacle
        cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
        if np.any(cross > 1e-12) and np.any(cross < -1e-12):
            raise ValueError("Independent clearance audit requires convex obstacles")
        normals = np.stack((-edges[:, 1], edges[:, 0]), axis=1)
        normals = normals[np.linalg.norm(normals, axis=1) > 1e-12]
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        axes = np.concatenate((robot_normals,
                               np.broadcast_to(normals, (len(states),) + normals.shape)), axis=1)
        a = np.einsum("mvi,mai->mva", robots, axes)
        b = np.einsum("vi,mai->mva", obstacle, axes)
        gap = np.maximum(b.min(axis=1) - a.max(axis=1),
                         a.min(axis=1) - b.max(axis=1)).max(axis=1)
        outside = gap > 0
        distance = gap.copy()
        if np.any(outside):
            p = robots[outside]
            delta = p[:, :, None, :] - obstacle[None, None, :, :]
            fraction = np.sum(delta * edges, axis=3) / np.maximum(np.sum(edges**2, axis=1), 1e-24)
            difference = delta - np.clip(fraction, 0, 1)[..., None] * edges
            first = np.linalg.norm(difference, axis=3).min(axis=(1, 2))
            segments = robot_edges[outside]
            delta = obstacle[None, :, None, :] - p[:, None, :, :]
            fraction = np.sum(delta * segments[:, None, :, :], axis=3) / np.maximum(
                np.sum(segments**2, axis=2)[:, None, :], 1e-24)
            difference = delta - np.clip(fraction, 0, 1)[..., None] * segments[:, None, :, :]
            second = np.linalg.norm(difference, axis=3).min(axis=(1, 2))
            distance[outside] = np.minimum(first, second)
        clearance = np.minimum(clearance, distance)
    return clearance


def trial_metrics(row: dict, job: dict, root: Path, cache_dir: Path) -> dict:
    job_dir = resolve_path(job["output_dir"], root)
    run_dir = resolve_path(row["output_root_dir"], root) if row.get("output_root_dir") else (
        job_dir / "results" / "runs" / row["configuration_id"] / f"trial_{int(row['trial_number']):04d}")
    history_files = sorted((run_dir / "data").glob("trial_history*.csv"))
    sdf_files = sorted((run_dir / "data").glob("pose_sdf*.csv"))
    geometry_path = job_dir / "environment_geometry.json"
    files = history_files + sdf_files + ([geometry_path] if geometry_path.exists() else [])
    fingerprint = hashlib.sha256(json.dumps({
        "version": CACHE_VERSION, "row": row,
        "files": [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files],
    }, sort_keys=True).encode()).hexdigest()
    cache_path = cache_dir / f"{job['id']}_trial_{row['trial_number']}.json"
    if cache_path.exists():
        cached = read_json(cache_path)
        if cached.get("fingerprint") == fingerprint:
            return cached["metrics"]
    history = read_csv(history_files[0]) if len(history_files) == 1 else []
    sdf = read_csv(sdf_files[0]) if len(sdf_files) == 1 else []
    result = {
        "history_samples": len(history), "geometry_audited": False,
        "min_clearance_m": None, "min_clearance_time_s": None,
        "final_clearance_m": None, "penetrating_samples": None,
        "first_penetration_time_s": None, "history_final_time_matches": None,
        "history_timeline_complete": False, "geometry_audit_complete": False,
        "sampled_penetration_free_goal": False,
    }
    issues = []
    if len(history_files) != 1:
        issues.append(f"expected_one_history_found_{len(history_files)}")
    if history and geometry_path.exists():
        states = np.asarray([[float(row[key]) for key in ("initial_x", "initial_y", "initial_theta")]] +
                            [[float(step[key]) for key in ("x", "y", "theta")] for step in history])
        times = np.asarray([0.] + [float(step["time"]) for step in history])
        if not np.all(np.isfinite(states)) or not np.all(np.diff(times) > 0):
            issues.append("invalid_pose_or_time_sequence")
        else:
            clearances = signed_clearances(states, read_json(geometry_path))
            index = int(np.argmin(clearances))
            penetration = np.flatnonzero(clearances < -PENETRATION_TOLERANCE_M)
            final_time = number(row.get("final_time"))
            final_matches = final_time is not None and math.isclose(times[-1], final_time, abs_tol=1e-8)
            step_indices = [number(step.get("timestep")) for step in history]
            timeline_complete = (step_indices == list(range(len(history)))
                                 and np.allclose(np.diff(times), times[1], rtol=0., atol=1e-8)
                                 and final_matches)
            if not final_matches:
                issues.append("history_final_time_mismatch_or_missing")
            if not timeline_complete:
                issues.append("incomplete_history_timeline")
            result.update({
                "geometry_audited": True, "min_clearance_m": float(clearances[index]),
                "min_clearance_time_s": float(times[index]),
                "final_clearance_m": float(clearances[-1]),
                "penetrating_samples": len(penetration),
                "first_penetration_time_s": float(times[penetration[0]]) if len(penetration) else None,
                "history_final_time_matches": final_matches,
                "history_timeline_complete": bool(timeline_complete),
                "geometry_audit_complete": bool(timeline_complete),
                "sampled_penetration_free_goal": truth(row["success"]) and len(penetration) == 0 and bool(timeline_complete),
            })
    elif not geometry_path.exists():
        issues.append("missing_environment_geometry")
    raw_status = [step.get("solver_raw_status", "") for step in history]
    status_codes = [number(step.get("solver_status_code")) for step in history]
    veto_times = [float(step["time"]) for step in history if step.get("solver_raw_status") == "safe_stop_stage0"]
    successful_times = [number(step.get("solver_computation_time")) for step in history
                        if truth(step.get("solver_success"))]
    sdf_values = [number(step.get("sdf_value")) for step in sdf]
    result.update({
        "stage0_veto_steps": raw_status.count("safe_stop_stage0"),
        "first_stage0_veto_post_control_time_s": min(veto_times) if veto_times else None,
        "nonzero_solver_status_steps": sum(code is not None and code != 0 for code in status_codes),
        "unsuccessful_control_steps": sum(not truth(step.get("solver_success")) for step in history),
        "unknown_status_steps": sum(not raw for raw in raw_status),
        "min_logged_psdf_m": min((v for v in sdf_values if v is not None), default=None),
        "audit_issues": ";".join(issues),
        "successful_solver_times_s": [t for t in successful_times if t is not None and t >= 0],
    })
    atomic_text(cache_path, json.dumps({"fingerprint": fingerprint, "metrics": result}, indent=2))
    return result


def wilson(successes: int, count: int) -> tuple[float | None, float | None]:
    if not count:
        return None, None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return max(0., center-half), min(1., center+half)


def parameter_audit(job: dict, baseline: dict | None, root: Path) -> dict:
    result = {"ofat_valid": None, "parameter_diff_json": "", "parameter_audit_issue": ""}
    current_path = resolve_path(job["output_dir"], root) / "resolved_cc_mf_parameters.json"
    baseline_path = (resolve_path(baseline["output_dir"], root) / "resolved_cc_mf_parameters.json"
                     if baseline else None)
    if not current_path.exists() or baseline_path is None or not baseline_path.exists():
        result["parameter_audit_issue"] = "resolved_parameters_pending"
        return result
    current, original = read_json(current_path), read_json(baseline_path)
    diff = {key: {"baseline": original.get(key), "variant": current.get(key)}
            for key in set(current) | set(original) if current.get(key) != original.get(key)}
    parameter = job.get("parameter")
    is_baseline = parameter in (None, "baseline")
    valid = (not diff) if is_baseline else (
        set(diff) == {parameter} and current.get(parameter) == job.get("value")
        and original.get(parameter) == job.get("baseline_value"))
    result.update({"ofat_valid": valid, "parameter_diff_json": json.dumps(diff, sort_keys=True),
                   "parameter_audit_issue": "" if valid else "unexpected_resolved_parameter_diff"})
    return result


def analyze(root: Path, output: Path) -> tuple[list[dict], list[dict], list[dict]]:
    manifest = read_json(root / "sweep_manifest.json")
    jobs = manifest["jobs"]
    expected = int(manifest.get("trials_per_job", manifest.get("trials", 10)))
    expected_seeds = {str(seed) for seed in manifest.get("seeds", range(expected))}
    baselines = {job["environment"]: job for job in jobs if job.get("parameter") in (None, "baseline")}
    trials, by_job, timings, partial_rows = [], {}, {}, {}
    for job in jobs:
        rows = read_csv(resolve_path(job["output_dir"], root) / "results" / "trial_results.csv")
        by_job[job["id"]], timings[job["id"]] = [], []
        partial_rows[job["id"]] = 0
        for row in rows:
            if (None in row or any(value is None for value in row.values())
                    or not row.get("trial_number") or not row.get("status")):
                partial_rows[job["id"]] += 1
                continue  # Ignore an incomplete append while the runner is active.
            result = {"job_id": job["id"], "environment": job["environment"],
                      "parameter": job.get("parameter"), "value": job.get("value"),
                      **{key: row.get(key, "") for key in (
                          "trial_number", "start_seed", "initial_x", "initial_y", "initial_theta",
                          "status", "failure_reason", "final_time", "distance_to_goal", "output_root_dir")},
                      "success": int(truth(row.get("success"))),
                      "completed": row["status"] != "interrupted"}
            outcome_issues = []
            if row["status"] not in {"success", "failure", "timeout", "error", "interrupted"}:
                outcome_issues.append("unknown_trial_status")
            if str(row.get("success", "")).lower() not in {"0", "1", "true", "false"}:
                outcome_issues.append("invalid_success_flag")
            if bool(result["success"]) != (row["status"] == "success"):
                outcome_issues.append("inconsistent_status_success_flag")
            if result["success"] and number(result["final_time"]) is None:
                outcome_issues.append("invalid_goal_time")
            result["outcome_data_valid"] = not outcome_issues
            result["outcome_audit_issues"] = ";".join(outcome_issues)
            try:
                metrics = trial_metrics(row, job, root, output / ".analysis_cache")
            except (OSError, ValueError, KeyError, TypeError) as error:
                metrics = {"geometry_audited": False, "geometry_audit_complete": False,
                           "history_timeline_complete": False,
                           "sampled_penetration_free_goal": False,
                           "audit_issues": f"{type(error).__name__}: {error}"}
            if outcome_issues:
                metrics["audit_issues"] = ";".join(filter(None, [metrics.get("audit_issues"), *outcome_issues]))
            times = metrics.pop("successful_solver_times_s", [])
            if result["completed"]:
                timings[job["id"]].extend(times)
            result.update(metrics)
            trials.append(result)
            by_job[job["id"]].append(result)
    pairs = []
    summaries = []
    for job in jobs:
        baseline = baselines.get(job["environment"])
        base_rows = by_job.get(baseline["id"], []) if baseline else []
        base_by_seed = {row["start_seed"]: row for row in base_rows if row["completed"]}
        rows = by_job[job["id"]]
        completed = [row for row in rows if row["completed"]]
        job_pairs = []
        for row in completed:
            other = base_by_seed.get(row["start_seed"])
            if other is None:
                continue
            pose_keys = ("initial_x", "initial_y", "initial_theta")
            valid_poses = all(number(item[key]) is not None for item in (row, other) for key in pose_keys)
            pose_match = valid_poses and all(number(row[key]) == number(other[key]) for key in pose_keys)
            paired = {"job_id": job["id"], "environment": job["environment"],
                      "start_seed": row["start_seed"], "exact_initial_pose_match": pose_match,
                      "pose_audit_issue": "" if valid_poses else "missing_or_nonfinite_initial_pose",
                      "baseline_success": other["success"], "variant_success": row["success"],
                      "success_delta": row["success"] - other["success"],
                      "baseline_status": other["status"], "variant_status": row["status"],
                      "baseline_min_clearance_m": other.get("min_clearance_m"),
                      "variant_min_clearance_m": row.get("min_clearance_m"),
                      "baseline_final_time_s": other["final_time"], "variant_final_time_s": row["final_time"]}
            pairs.append(paired)
            job_pairs.append(paired)
        count = len(completed)
        successes = sum(row["success"] for row in completed)
        low, high = wilson(successes, count)
        statuses = Counter(row["status"] for row in completed)
        successful_times = [number(row["final_time"]) for row in completed if row["success"]
                            and number(row["final_time"]) is not None]
        clearance = [row["min_clearance_m"] for row in completed if row.get("geometry_audited")]
        audit = parameter_audit(job, baseline, root)
        unique_seeds = len({row["start_seed"] for row in completed})
        expected_seed_match = {row["start_seed"] for row in completed} == expected_seeds
        pose_matches = len(job_pairs) == expected and all(p["exact_initial_pose_match"] for p in job_pairs)
        fully_complete = count == expected and unique_seeds == expected and expected_seed_match
        outcome_data_valid = all(row["outcome_data_valid"] for row in completed)
        baseline_outcomes_valid = all(row["outcome_data_valid"] for row in base_rows if row["completed"])
        comparison_eligible = (fully_complete and pose_matches and audit["ofat_valid"] is True
                               and outcome_data_valid and baseline_outcomes_valid)
        geometry_complete = sum(bool(row.get("geometry_audit_complete")) for row in completed)
        baseline_geometry_complete = sum(bool(row.get("geometry_audit_complete")) for row in base_rows if row["completed"])
        summary = {
            "job_id": job["id"], "environment": job["environment"],
            "parameter": job.get("parameter"), "value": job.get("value"),
            "baseline_value": job.get("baseline_value"),
            "completed_trials": count, "expected_trials": expected,
            "unique_completed_seeds": unique_seeds, "fully_complete": fully_complete,
            "expected_seed_set_matches": expected_seed_match,
            "partial_rows_ignored": partial_rows[job["id"]],
            "outcome_data_valid": outcome_data_valid,
            "successful_trials": successes, "success_rate": successes/count if count else None,
            "wilson95_low": low, "wilson95_high": high,
            "failure_trials": statuses["failure"], "timeout_trials": statuses["timeout"],
            "error_trials": statuses["error"], "interrupted_trials": len(rows) - count,
            "stage0_veto_trials": sum(row.get("stage0_veto_steps", 0) > 0 for row in completed),
            "stage0_veto_steps": sum(row.get("stage0_veto_steps", 0) for row in completed),
            "nonzero_solver_status_steps": sum(row.get("nonzero_solver_status_steps", 0) for row in completed),
            "movement_failure_trials": sum("movement" in row["failure_reason"].lower()
                                           or "stuck" in row["failure_reason"].lower() for row in completed),
            "goal_time_mean_s": float(np.mean(successful_times)) if successful_times else None,
            "goal_time_median_s": float(np.median(successful_times)) if successful_times else None,
            "min_clearance_m": min(clearance) if clearance else None,
            "geometry_audited_trials": len(clearance),
            "geometry_complete_trials": geometry_complete,
            "sampled_penetration_trials": sum(row.get("penetrating_samples", 0) is not None
                                             and row.get("penetrating_samples", 0) > 0 for row in completed),
            "sampled_penetration_free_goal_trials": sum(row.get("sampled_penetration_free_goal", False) for row in completed),
            "paired_trials": len(job_pairs), "all_paired_initial_poses_match": pose_matches,
            "paired_improved_trials": sum(p["success_delta"] > 0 for p in job_pairs),
            "paired_regressed_trials": sum(p["success_delta"] < 0 for p in job_pairs),
            "paired_success_delta": sum(p["success_delta"] for p in job_pairs),
            "successful_solver_steps": len(timings[job["id"]]),
            "solver_time_mean_s": float(np.mean(timings[job["id"]])) if timings[job["id"]] else None,
            "solver_time_p95_s": float(np.percentile(timings[job["id"]], 95)) if timings[job["id"]] else None,
            **audit,
            "eligible_for_comparison": comparison_eligible,
            "safety_comparison_eligible": comparison_eligible and geometry_complete == expected
                                          and baseline_geometry_complete == expected,
            "trial_audit_issues": ";".join(f"seed{row['start_seed']}:{row['audit_issues']}"
                                           for row in completed if row.get("audit_issues")),
            "output_dir": job["output_dir"],
        }
        summaries.append(summary)
    return summaries, trials, pairs


def plot_svg(summaries: list[dict]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    environments = list(dict.fromkeys(row["environment"] for row in summaries))
    fig, axes = plt.subplots(len(environments), 1, figsize=(12, max(3.5, 4.1*len(environments))), squeeze=False)
    for axis, environment in zip(axes.flat, environments):
        rows = [row for row in summaries if row["environment"] == environment]
        x = np.arange(len(rows))
        rates = np.asarray([100*row["success_rate"] if row["success_rate"] is not None else 0 for row in rows])
        colors = ["#334155" if row["parameter"] in (None, "baseline") else
                  "#0f766e" if row["eligible_for_comparison"] else "#cbd5e1" for row in rows]
        axis.bar(x, rates, color=colors, width=.65)
        for index, row in enumerate(rows):
            if row["completed_trials"]:
                axis.errorbar(index, rates[index], yerr=[[max(0., rates[index] - 100*row["wilson95_low"])],
                    [max(0., 100*row["wilson95_high"] - rates[index])]], color="#172554", capsize=3, linewidth=1)
            axis.text(index, min(102, rates[index]+3),
                      f"{row['successful_trials']}/{row['completed_trials']}", ha="center", fontsize=8)
        labels = ["baseline" if row["parameter"] in (None, "baseline") else
                  f"{row['parameter']}\n{row['value']:g}" for row in rows]
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.set_ylim(0, 110)
        axis.set_ylabel("Goal success (%)")
        axis.set_title(f"{environment}: observed successes / completed trials (95% Wilson intervals)")
        baseline = next((row for row in rows if row["parameter"] in (None, "baseline")), None)
        if baseline and baseline["success_rate"] is not None:
            axis.axhline(100*baseline["success_rate"], color="#a16207", linestyle=":", linewidth=1,
                         label=f"fresh baseline: {100*baseline['success_rate']:.0f}%")
            axis.legend(loc="upper right", frameon=False, fontsize=8)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    fig.tight_layout()
    handle = io.StringIO()
    fig.savefig(handle, format="svg", metadata={"Date": None})
    plt.close(fig)
    return handle.getvalue()


def recommendation_html(summaries: list[dict]) -> str:
    """Describe completed, auditable candidates without extrapolating outcomes."""
    paragraphs = []
    for environment in dict.fromkeys(row["environment"] for row in summaries):
        rows = [row for row in summaries if row["environment"] == environment]
        baseline = next((row for row in rows if row["parameter"] in (None, "baseline")), None)
        remaining = sum(not row["fully_complete"] for row in rows)
        candidates = [row for row in rows if row["parameter"] not in (None, "baseline")
                      and row["fully_complete"] and row["eligible_for_comparison"]
                      and row["safety_comparison_eligible"]]
        prefix = f"<strong>{html.escape(environment)}</strong>: "
        if baseline is None or not baseline["eligible_for_comparison"] or not candidates:
            paragraphs.append(f"<p>{prefix}완료와 비교 검증을 모두 통과한 변경 후보를 기다리고 있습니다. "
                              f"미완료 조건 {remaining}개.</p>")
            continue
        def rank(row: dict) -> tuple[int, int]:
            return row["successful_trials"], -row["sampled_penetration_trials"]
        best_rank = max(map(rank, candidates))
        best = [row for row in candidates if rank(row) == best_rank]
        scope = "현재까지" if remaining else "완료한 sweep에서"
        baseline_text = (f"baseline {baseline['successful_trials']}/{baseline['completed_trials']} 성공, "
                         f"기록 시점 침투 {baseline['sampled_penetration_trials']}개 trial")
        descriptions = []
        for row in best:
            label = html.escape(f"{row['parameter']} = {row['value']:g}")
            retained = " (baseline 성공 seed 모두 유지)" if row["paired_regressed_trials"] == 0 else ""
            descriptions.append(
                f"<code>{label}</code>: {row['successful_trials']}/{row['completed_trials']} 성공, "
                f"동일 seed 개선 +{row['paired_improved_trials']} / 악화 −{row['paired_regressed_trials']}{retained}, "
                f"기록 시점 침투 {row['sampled_penetration_trials']}개 trial")
        conclusion = (" 관측 성공수는 baseline보다 높습니다." if best_rank[0] > baseline["successful_trials"]
                      else " 관측 성공수가 baseline을 넘는 후보는 아직 없습니다.")
        paragraphs.append(f"<p>{prefix}{scope} 최상위 {'공동 ' if len(best) > 1 else ''}변경 후보는 "
                          f"{' ; '.join(descriptions)}입니다. {baseline_text}."
                          f"{conclusion} 미완료 조건 {remaining}개.</p>")
        alternative = next((row for row in candidates if row["parameter"] == "alpha_kappa"
                            and row["value"] == 0.0002 and row not in best), None)
        if alternative is not None:
            paragraphs.append(f"<p>{html.escape(environment)}의 별도 비교 후보 "
                              f"<code>alpha_kappa = {alternative['value']:g}</code>는 "
                              f"{alternative['successful_trials']}/{alternative['completed_trials']} 성공, "
                              f"기록 시점 침투 {alternative['sampled_penetration_trials']}개 trial, "
                              f"movement 기준 실패 {alternative['movement_failure_trials']}개 trial입니다.</p>")
    if paragraphs:
        paragraphs.append("<p>순위는 완료·초기 pose·단일 변수 변경·전체 pose 기록 검증을 통과한 후보만 대상으로, "
                          "성공수가 많은 순서로 정하며 동률이면 기록 시점 침투 trial이 적은 후보를 앞에 둡니다. "
                          "두 수치가 모두 같으면 공동 순위입니다. 이는 관측 표본의 순위이며 통계적 일반화나 안전 보장이 아닙니다.</p>")
    return "".join(paragraphs)


def experiment_conditions_html(root: Path, summaries: list[dict]) -> str:
    """Describe saved configs and provenance, rather than assumed defaults."""
    import yaml

    manifest = read_json(root / "sweep_manifest.json")
    original = yaml.safe_load((root / "input_config.yaml").read_text())
    configs = [yaml.safe_load(resolve_path(job["config_path"], root).read_text()) for job in manifest["jobs"]]
    baseline_job = next(job for job in manifest["jobs"] if job.get("parameter") in (None, "baseline"))
    baseline_config = yaml.safe_load(resolve_path(baseline_job["config_path"], root).read_text())
    geometry = read_json(resolve_path(baseline_job["output_dir"], root) / "environment_geometry.json")
    identity = baseline_config["test_configs"][0]
    defaults = baseline_config["defaults"]
    failure = baseline_config["trial_failure_criteria"]
    preserved = all(
        config.get("start_perturbation") == original.get("start_perturbation")
        and config.get("localization_error") == original.get("localization_error")
        and config.get("trial_failure_criteria") == original.get("trial_failure_criteria")
        and config.get("defaults", {}).get("simulation_time") == original.get("defaults", {}).get("simulation_time")
        and all(config.get("cc_mf", {}).get(key) == value for key, value in original.get("cc_mf", {}).items()
                if key != job.get("parameter"))
        for job, config in zip(manifest["jobs"], configs))
    disabled = all(config.get("defaults", {}).get("generate_animation") is False
                   and config.get("defaults", {}).get("generate_plots") is False
                   and config.get("cc_mf", {}).get("debug_mf") is False
                   and config.get("cc_mf", {}).get("debug_infeasibility") is False for config in configs)
    environments = ", ".join(map(str, manifest["environments"]))
    seeds = manifest["seeds"]
    seeds_text = (f"{seeds[0]}–{seeds[-1]}" if seeds and seeds == list(range(seeds[0], seeds[-1]+1))
                  else ", ".join(map(str, seeds)))
    preservation = ("기준 설정의 시뮬레이션 제한·시작 perturbation·localization error·종료 기준과 기존 CC-MF 설정을 "
                    "보존했습니다(각 후보가 변경한 noise 변수 한 개 제외)." if preserved
                    else "기준 설정과 조건 차이가 있으므로 저장된 config를 확인해야 합니다.")
    dynamics = "DD (differential_drive)" if identity["dynamics_type"] == "differential_drive" else identity["dynamics_type"]
    planner = "A*" if identity["path_planner"] == "astar" else identity["path_planner"]
    conditions = (
        f"<p>실험 조건은 저장한 <a href='input_config.yaml'>config/config_benchmark.yaml 기준 설정</a>과 "
        f"각 실행 config에서 읽었습니다. {preservation} 환경은 <strong>{html.escape(environments)}</strong>"
        f"{'만' if len(manifest['environments']) == 1 else ''} 실행했습니다. "
        f"{html.escape(identity['robot_shape'])} footprint "
        f"{geometry['footprint_length_m']:.3f} × {geometry['footprint_width_m']:.3f} m, "
        f"{html.escape(dynamics)}, {html.escape(planner)}, seed {html.escape(seeds_text)}, "
        f"조건별 {manifest['trials_per_job']}회입니다. "
        f"명목 시뮬레이션 제한 {defaults['simulation_time']:g} s, "
        f"movement 판정 {failure['movement_window_sec']:g} s / {1000*failure['movement_threshold']:g} mm, "
        f"연속 unsuccessful control step {failure['max_consecutive_solver_failures']}회에서 실패 처리합니다.</p>")
    if disabled:
        conditions += ("<p>모든 조건에서 animation·plot·선택적 solve 후 진단(debug_mf/debug_infeasibility)을 "
                       "동일하게 껐습니다. stage0 veto는 연속 unsuccessful control step에 포함됩니다.")
    else:
        conditions += "<p>출력 및 선택적 진단 설정은 저장한 각 실행 config를 참고하세요."
    reproduction_path = root / "baseline_reproduction.json"
    baseline_row = next((row for row in summaries if row["job_id"] == baseline_job["id"]), None)
    if (reproduction_path.exists() and baseline_row and baseline_row["fully_complete"]
            and read_json(reproduction_path).get("all_ten_baseline_outcomes_and_starts_exactly_match") is True):
        conditions += (f" 새 baseline {baseline_row['completed_trials']}개의 결과와 초기 pose가 기존 로그와 정확히 일치하여 "
                       f"성공률 {100*baseline_row['success_rate']:.0f}% "
                       f"({baseline_row['successful_trials']}/{baseline_row['completed_trials']})를 재현했습니다.")
    return conditions + "</p>"


def final_selection_html(root: Path, summaries: list[dict]) -> str:
    path = root / "recommendation.json"
    if not path.exists():
        return ""
    selection = read_json(path)
    by_job = {row["job_id"]: row for row in summaries}
    primary = by_job.get(selection["primary_job"])
    if (primary is None or not primary["safety_comparison_eligible"]
            or primary["successful_trials"] != selection["recommended_successes"]
            or sum(row["completed_trials"] for row in summaries) != selection["total_drives"]):
        return "<p>저장한 최종 선택과 현재 집계의 일치 확인이 필요합니다.</p>"
    label = html.escape(f"{selection['parameter']} = {selection['recommended_value']:g}")
    reason = html.escape(selection["reason_ko"])
    paragraph = (f"<p><strong>최종 선택: <code>{label}</code></strong> "
                 f"(기준값 {selection['baseline_value']:g}). "
                 f"{selection['total_drives']}회 완료 후, "
                 f"{primary['successful_trials']}/{primary['completed_trials']} 성공 조건 중 선택했습니다. "
                 f"{reason} 동일 seed 개선 +{primary['paired_improved_trials']} / "
                 f"악화 −{primary['paired_regressed_trials']}입니다.")
    alternative = by_job.get(selection.get("alternative_same_success_job"))
    if alternative and primary["min_clearance_m"] is not None and alternative["min_clearance_m"] is not None:
        paragraph += (f" 기록 시점 최대 침투 깊이는 선택 후보 "
                      f"{max(0., -1000*primary['min_clearance_m']):.3f} mm, "
                      f"{html.escape(alternative['parameter'])}={alternative['value']:g} "
                      f"{max(0., -1000*alternative['min_clearance_m']):.3f} mm입니다.")
    paragraph += " 기록 시점 침투가 남아 있으므로 이 선택은 안전 보장이 아닙니다.</p>"
    links = []
    for export in selection.get("exports", []):
        row = by_job.get(export["job_id"])
        if row is None or export.get("resolved_parameters_equal_to_executed_job") is not True:
            continue
        filename = export["file"]
        if not (root / filename).exists():
            continue
        title = ("권장 YAML" if export["job_id"] == selection["primary_job"]
                 else f"대안 {row['parameter']}={row['value']:g} YAML")
        links.append(f"<a href='{html.escape(filename, quote=True)}'>{html.escape(title)}</a>")
    links += ["<a href='recommendation.json'>최종 선택 근거 JSON</a>",
              f"<a href='final_validation.json'>{selection['total_drives']}회 최종 검증 JSON</a>"]
    return paragraph + "<p>" + " · ".join(links) + "</p>"


def make_report(summaries: list[dict], svg: str, conditions: str = "", selection: str = "") -> str:
    def display(value: Any, digits: int = 3) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    rows = []
    for row in summaries:
        label = "baseline" if row["parameter"] in (None, "baseline") else f"{row['parameter']} = {row['value']:g}"
        row_class = "complete" if row["eligible_for_comparison"] else "pending"
        rows.append(f"<tr class='{row_class}'><td>{html.escape(row['environment'])}</td>"
                    f"<td>{html.escape(label)}</td><td>{row['successful_trials']}/{row['completed_trials']}</td>"
                    f"<td>{display(100*row['success_rate'] if row['success_rate'] is not None else None, 0)}%</td>"
                    f"<td>+{row['paired_improved_trials']} / −{row['paired_regressed_trials']}</td>"
                    f"<td>{row['stage0_veto_trials']} / {row['stage0_veto_steps']}</td>"
                    f"<td>{row['nonzero_solver_status_steps']}</td>"
                    f"<td>{row['movement_failure_trials']} / {row['timeout_trials']} / {row['error_trials']}</td>"
                    f"<td>{display(1000*row['min_clearance_m'] if row['min_clearance_m'] is not None else None)}</td>"
                    f"<td>{row['sampled_penetration_trials']}</td>"
                    f"<td>{row['sampled_penetration_free_goal_trials']}</td>"
                    f"<td>{display(row['goal_time_mean_s'], 1)}</td>"
                    f"<td>{'OK' if row['eligible_for_comparison'] else 'pending / inspect CSV'} / "
                    f"{'OK' if row['safety_comparison_eligible'] else 'pending / inspect CSV'}</td></tr>")
    completed = sum(row["completed_trials"] for row in summaries)
    expected = sum(row["expected_trials"] for row in summaries)
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>CC-MF process-noise OFAT sweep</title><style>
body{{font:15px/1.55 system-ui,sans-serif;max-width:1450px;margin:32px auto;padding:0 20px;color:#172033}}
h1{{font-size:28px}}p{{max-width:1100px}}.plot svg{{max-width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #dbe2ea;text-align:left}}
th{{background:#f1f5f9}}.pending{{color:#687589}}.table{{overflow:auto}}code{{background:#f1f5f9;padding:2px 4px}}
</style><h1>CC-MF process noise: 한 변수씩 비교</h1>
{selection}
{recommendation_html(summaries)}
{conditions}
<p>완료 {completed}/{expected} trials. 성공은 기존 목표 도달 판정이며 failure·timeout·error를 분모에 포함합니다.
interrupted는 제외합니다. 회색 막대는 아직 10회 또는 비교 검증이 완료되지 않은 조건입니다.</p>
<p>같은 환경의 baseline과 seed별 초기 pose가 float 값까지 동일한지, resolved parameter가 지정된 변수 하나만
달라졌는지 검증합니다. 표본 10회의 성공률과 Wilson 95% 구간은 탐색 결과이며, 여러 후보 중 최댓값을 고르면
개선량이 낙관적일 수 있습니다. 독립 seed 검증 없이 일반화된 개선이나 통계적 유의성을 주장하지 않습니다.</p>
<div class="plot">{svg}</div><div class="table"><table><thead><tr>
<th>환경</th><th>변경</th><th>성공/완료</th><th>성공률</th><th>paired 개선/악화</th>
<th>stage0 veto trials/steps</th><th>nonzero solver steps</th><th>movement/timeout/error</th>
<th>최소 clearance mm</th><th>침투 trials</th><th>침투 없는 성공</th><th>성공 시간 s</th><th>목표/안전 지표 비교 검증</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>clearance는 실제 footprint와 지도 polygon으로 계산한 기록 시점의 signed distance입니다. 초기 pose와 모든
post-control pose(마지막 pose 포함)를 사용합니다. −1 nm 미만을 침투로 집계하며, 침투 없는 성공은 별도 지표입니다.
마지막 시점 불일치나 중간 기록 누락이 있으면 침투 없는 성공 집계 및 안전 지표 비교 자격에서 제외합니다.
안전 지표 비교 자격은 해당 후보와 baseline의 10개 trial 모두에 완전한 pose 기록이 있는지도 요구합니다.
이는 샘플 사이의 연속시간 안전이나 위치추정 오차에 대한 안전 보장이 아닙니다. goal success만 높고 침투가 늘어난 후보는
안전한 개선으로 해석할 수 없습니다. PSDF 로그의 최솟값도 trial CSV에 별도로 보존합니다.</p>
<p>stage0 veto는 <code>solver_raw_status=safe_stop_stage0</code>인 solver 호출 생략이며, 실제 nonzero solver status와
분리합니다. solver 시간은 모든 완료 trial에서 성공한 step을 합쳐 계산합니다. 성공 시간 평균은 성공 trial만의 조건부
평균이므로 성공 seed 구성이 다른 후보 간 속도 비교에는 주의가 필요합니다.</p>
<p><a href="sweep_summary.csv">조건별 요약 CSV</a> · <a href="trial_summary.csv">trial별 CSV</a> ·
<a href="paired_comparisons.csv">동일 seed 비교 CSV</a> · <a href="sweep_summary.json">JSON</a> ·
<a href="math_audit/audit.json">수식·파라미터 감사 JSON</a> ·
<a href="baseline_reproduction.json">기존 baseline 재현 검증 JSON</a> ·
<a href="input_config.yaml">기준 config YAML</a> · <a href="source_hashes.json">고정한 실행 소스 hash</a></p></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    root = args.sweep_root.resolve()
    output = (args.output_dir or root).resolve()
    summaries, trials, pairs = analyze(root, output)
    write_csv(output / "sweep_summary.csv", summaries)
    write_csv(output / "trial_summary.csv", trials)
    write_csv(output / "paired_comparisons.csv", pairs)
    atomic_text(output / "sweep_summary.json", json.dumps(summaries, indent=2, ensure_ascii=False))
    svg = "" if args.no_plots else plot_svg(summaries)
    if svg:
        atomic_text(output / "success_rates.svg", svg)
    try:
        conditions = experiment_conditions_html(root, summaries)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        conditions = f"<p>실험 조건 메타데이터 확인 미완료: {html.escape(str(error))}</p>"
    try:
        selection = final_selection_html(root, summaries)
    except (OSError, ValueError, KeyError, TypeError) as error:
        selection = f"<p>최종 선택 메타데이터 확인 미완료: {html.escape(str(error))}</p>"
    atomic_text(output / "analysis.html", make_report(summaries, svg, conditions, selection))
    source_bytes = Path(__file__).read_bytes()
    atomic_text(output / "analysis_source.py", source_bytes.decode("utf-8"))
    atomic_text(output / "analysis_provenance.json", json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "sweep_manifest_sha256": hashlib.sha256((root / "sweep_manifest.json").read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "geometry_cache_version": CACHE_VERSION,
        "penetration_tolerance_m": PENETRATION_TOLERANCE_M,
        "recorded_pose_safety_only": True,
    }, indent=2))
    for row in summaries:
        print(f"{row['job_id']}: {row['successful_trials']}/{row['completed_trials']} "
              f"veto={row['stage0_veto_trials']} penetration={row['sampled_penetration_trials']} "
              f"paired=+{row['paired_improved_trials']}/-{row['paired_regressed_trials']} "
              f"eligible={row['eligible_for_comparison']}")


if __name__ == "__main__":
    main()
