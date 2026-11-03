"""Offline analysis of CC-MF noise combinations and independent seed cohorts.

Only CSV, JSON, NumPy and plotting code is used; this never imports a controller
or launches a drive. The validated polygon/history audit is reused from the
one-factor analyzer. Search and validation cohorts are never pooled or paired.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

import analyze_cc_mf_noise_sweep as shared


def manifest_path(root: Path) -> Path:
    preferred = root / "combination_manifest.json"
    return preferred if preferred.exists() else root / "sweep_manifest.json"


def label(job: dict) -> str:
    if not isinstance(job.get("overrides", {}), dict):
        return "invalid overrides (inspect manifest)"
    values = ", ".join(f"{key}={value:g}" if isinstance(value, (int, float))
                       else f"{key}={value}" for key, value in job.get("overrides", {}).items())
    return values or "original parameters"


def expected_seeds(job: dict, manifest: dict) -> set[str]:
    raw = job.get("seeds", manifest.get("seeds", range(10)))
    return {str(seed) for seed in raw} if isinstance(raw, (list, tuple, range)) else set()


def seed_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        integer = int(str(value))
        return integer if str(integer) == str(value).strip() else None
    except (ValueError, TypeError):
        return None


def seed_display(value: Any) -> int | str:
    parsed = seed_number(value)
    return parsed if parsed is not None else str(value)


def seed_declaration_audit(job: dict, manifest: dict) -> dict:
    count = manifest.get("trials_per_job", 10)
    count_valid = isinstance(count, int) and not isinstance(count, bool) and count > 0
    expected_count = count if count_valid else 10
    raw = job.get("seeds", manifest.get("seeds", list(range(expected_count))))
    valid_list = isinstance(raw, (list, tuple))
    values = [seed_number(value) for value in raw] if valid_list else []
    issues = []
    if not count_valid:
        issues.append("invalid_manifest_trials_per_job")
    if (not valid_list or len(values) != expected_count or any(value is None for value in values)
            or len(set(values)) != expected_count):
        issues.append("job_seed_count_or_values_invalid")
    cohort_seeds = manifest.get("cohorts", {}).get(job.get("cohort", "search"))
    if cohort_seeds is not None:
        cohort_values = [seed_number(value) for value in cohort_seeds] if isinstance(cohort_seeds, list) else []
        if (len(cohort_values) != expected_count or any(value is None for value in cohort_values)
                or len(set(cohort_values)) != expected_count or set(values) != set(cohort_values)):
            issues.append("job_seeds_differ_from_manifest_cohort")
    return {"seed_declaration_valid": not issues, "seed_declaration_issue": ";".join(issues),
            "expected_trials": expected_count}


def parameter_audit(job: dict, original: dict, root: Path) -> dict:
    result = {"parameters_valid": False, "parameter_diff_json": "",
              "parameter_audit_issue": "resolved_parameters_pending"}
    path = shared.resolve_path(job["output_dir"], root) / "resolved_cc_mf_parameters.json"
    if not path.exists():
        return result
    actual = shared.read_json(path)
    overrides = job.get("overrides", {})
    if not isinstance(overrides, dict):
        result["parameter_audit_issue"] = "overrides_must_be_a_mapping"
        return result
    expected = {**original, **overrides}
    diff = {key: {"original": original.get(key), "actual": actual.get(key)}
            for key in set(original) | set(actual) if original.get(key) != actual.get(key)}
    valid = set(overrides) <= set(original) and actual == expected
    result.update(parameters_valid=valid,
                  parameter_diff_json=json.dumps(diff, sort_keys=True),
                  parameter_audit_issue="" if valid else "resolved_parameters_differ_from_declared_combination")
    return result


def completion_audit(job: dict, root: Path, count: int) -> dict:
    path = shared.resolve_path(job["output_dir"], root) / "completion.json"
    record = shared.read_json(path) if path.exists() else {}
    valid = (record.get("complete") is True and record.get("returncode") == 0
             and record.get("trials") == count)
    return {"runner_complete": valid, "runner_returncode": record.get("returncode")}


def collect_trials(job: dict, root: Path, output: Path) -> tuple[list[dict], list[float], int]:
    raw_rows = shared.read_csv(shared.resolve_path(job["output_dir"], root) / "results/trial_results.csv")
    rows, solver_times, partial = [], [], 0
    for raw in raw_rows:
        if (None in raw or any(value is None for value in raw.values())
                or not raw.get("trial_number") or not raw.get("status")):
            partial += 1
            continue
        row = {"job_id": job["id"], "environment": job["environment"],
               "cohort": job.get("cohort", "search"), "kind": job.get("kind", "candidate"),
               "overrides_json": json.dumps(job.get("overrides", {}), sort_keys=True),
               **{key: raw.get(key, "") for key in (
                   "trial_number", "start_seed", "initial_x", "initial_y", "initial_theta",
                   "status", "failure_reason", "final_time", "distance_to_goal", "output_root_dir")},
               "success": int(shared.truth(raw.get("success"))),
               "completed": raw["status"] != "interrupted"}
        issues = []
        if seed_number(row["start_seed"]) is None:
            issues.append("invalid_start_seed")
        if row["status"] not in {"success", "failure", "timeout", "error", "interrupted"}:
            issues.append("unknown_trial_status")
        if str(raw.get("success", "")).lower() not in {"0", "1", "true", "false"}:
            issues.append("invalid_success_flag")
        if bool(row["success"]) != (row["status"] == "success"):
            issues.append("inconsistent_status_success_flag")
        if row["success"] and shared.number(row["final_time"]) is None:
            issues.append("invalid_goal_time")
        row["outcome_data_valid"] = not issues
        row["outcome_audit_issues"] = ";".join(issues)
        try:
            metrics = shared.trial_metrics(raw, job, root, output / ".combination_analysis_cache")
        except (OSError, ValueError, KeyError, TypeError) as error:
            metrics = {"geometry_audited": False, "geometry_audit_complete": False,
                       "history_timeline_complete": False, "sampled_penetration_free_goal": False,
                       "audit_issues": f"{type(error).__name__}: {error}"}
        metrics["audit_issues"] = ";".join(filter(None, [metrics.get("audit_issues"), *issues]))
        times = metrics.pop("successful_solver_times_s", [])
        if row["completed"]:
            solver_times.extend(times)
        row.update(metrics)
        rows.append(row)
    return rows, solver_times, partial


def analyze(root: Path, output: Path) -> tuple[list[dict], list[dict], list[dict]]:
    manifest = shared.read_json(manifest_path(root))
    jobs = manifest["jobs"]
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate job IDs in combination manifest")
    original = shared.read_json(root / "baseline_parameters.json")
    references = {}
    for job in jobs:
        if job.get("kind") == "reference":
            group = (job["environment"], job.get("cohort", "search"))
            if group in references:
                raise ValueError(f"More than one reference for {group}")
            references[group] = job
    data = {job["id"]: collect_trials(job, root, output) for job in jobs}
    audits = {job["id"]: parameter_audit(job, original, root) for job in jobs}
    seed_audits = {job["id"]: seed_declaration_audit(job, manifest) for job in jobs}
    completed = {key: [row for row in values[0] if row["completed"]] for key, values in data.items()}

    def complete_job(job: dict) -> bool:
        rows = completed[job["id"]]
        seeds = expected_seeds(job, manifest)
        return (seed_audits[job["id"]]["seed_declaration_valid"]
                and len(rows) == seed_audits[job["id"]]["expected_trials"]
                and {row["start_seed"] for row in rows} == seeds
                and len({row["start_seed"] for row in rows}) == len(rows)
                and all(row["outcome_data_valid"] for row in rows)
                and audits[job["id"]]["parameters_valid"]
                and completion_audit(job, root, len(rows))["runner_complete"])

    summaries, pairs = [], []
    for job in jobs:
        identity = job["id"]
        cohort = job.get("cohort", "search")
        reference = references.get((job["environment"], cohort))
        rows = completed[identity]
        expected = expected_seeds(job, manifest)
        reference_rows = completed[reference["id"]] if reference else []
        reference_by_seed = {row["start_seed"]: row for row in reference_rows}
        job_pairs = []
        for row in rows:
            other = reference_by_seed.get(row["start_seed"])
            if other is None:
                continue
            keys = ("initial_x", "initial_y", "initial_theta")
            valid_pose = all(shared.number(item.get(key)) is not None for item in (row, other) for key in keys)
            match = valid_pose and all(shared.number(row[key]) == shared.number(other[key]) for key in keys)
            both_success = bool(row["success"] and other["success"])
            a, b = shared.number(other["final_time"]), shared.number(row["final_time"])
            pair = {
                "job_id": identity, "reference_job_id": reference["id"], "environment": job["environment"],
                "cohort": cohort, "start_seed": row["start_seed"], "exact_initial_pose_match": match,
                "reference_success": other["success"], "candidate_success": row["success"],
                "success_delta": row["success"] - other["success"],
                "reference_status": other["status"], "candidate_status": row["status"],
                "reference_min_clearance_m": other.get("min_clearance_m"),
                "candidate_min_clearance_m": row.get("min_clearance_m"),
                "reference_final_time_s": a, "candidate_final_time_s": b,
                "both_success": both_success,
                "common_success_time_delta_s": b-a if both_success and a is not None and b is not None else None,
            }
            pairs.append(pair)
            job_pairs.append(pair)
        count = len(rows)
        successes = sum(row["success"] for row in rows)
        statuses = Counter(row["status"] for row in rows)
        low, high = shared.wilson(successes, count)
        exact_pairs = len(job_pairs) == len(expected) and all(pair["exact_initial_pose_match"] for pair in job_pairs)
        other_cohort_seeds = set().union(*(expected_seeds(other, manifest) for other in jobs
            if other["environment"] == job["environment"] and other.get("cohort", "search") != cohort))
        independent_seeds = not (expected & other_cohort_seeds) if cohort == "validation" else True
        eligible = (complete_job(job) and reference is not None and complete_job(reference)
                    and expected == expected_seeds(reference, manifest) and exact_pairs and independent_seeds)
        geometry_complete = sum(bool(row.get("geometry_audit_complete")) for row in rows)
        safety_eligible = (eligible and geometry_complete == len(expected)
                           and all(row.get("geometry_audit_complete") for row in reference_rows))
        goal_times = [shared.number(row["final_time"]) for row in rows if row["success"]
                      and shared.number(row["final_time"]) is not None]
        clearances = [row["min_clearance_m"] for row in rows if row.get("geometry_audited")]
        success_clearances = [row["min_clearance_m"] for row in rows if row["success"] and row.get("geometry_audited")]
        common = [pair for pair in job_pairs if pair["common_success_time_delta_s"] is not None]
        timings = data[identity][1]
        summaries.append({
            "job_id": identity, "environment": job["environment"], "cohort": cohort,
            "kind": job.get("kind", "candidate"), "label": label(job),
            "reused": bool(job.get("reused", False)),
            "overrides_json": json.dumps(job.get("overrides", {}), sort_keys=True),
            "reference_job_id": reference["id"] if reference else None,
            "completed_trials": count, **seed_audits[identity],
            "expected_seeds_json": json.dumps(sorted(expected, key=lambda value: (seed_number(value) is None, seed_display(value)))),
            "expected_seed_set_matches": {row["start_seed"] for row in rows} == expected,
            "independent_validation_seeds": independent_seeds,
            "unique_completed_seeds": len({row["start_seed"] for row in rows}),
            "fully_complete": bool(complete_job(job)), "partial_rows_ignored": data[identity][2],
            "successful_trials": successes, "success_rate": successes/count if count else None,
            "wilson95_low": low, "wilson95_high": high,
            "success_seeds_json": json.dumps([seed_display(row["start_seed"]) for row in rows if row["success"]]),
            "failure_trials": statuses["failure"], "timeout_trials": statuses["timeout"],
            "error_trials": statuses["error"], "interrupted_trials": len(data[identity][0])-count,
            "stage0_veto_trials": sum(row.get("stage0_veto_steps", 0) > 0 for row in rows),
            "stage0_veto_steps": sum(row.get("stage0_veto_steps", 0) for row in rows),
            "nonzero_solver_status_steps": sum(row.get("nonzero_solver_status_steps", 0) for row in rows),
            "movement_failure_trials": sum("movement" in row["failure_reason"].lower()
                                           or "stuck" in row["failure_reason"].lower() for row in rows),
            "geometry_complete_trials": geometry_complete,
            "sampled_penetration_trials": sum(bool(row.get("penetrating_samples")) for row in rows),
            "sampled_penetration_free_goal_trials": sum(bool(row.get("sampled_penetration_free_goal")) for row in rows),
            "min_clearance_m": min(clearances) if clearances else None,
            "successful_trial_min_clearance_m": min(success_clearances) if success_clearances else None,
            "goal_time_mean_s": float(np.mean(goal_times)) if goal_times else None,
            "paired_trials": len(job_pairs), "all_paired_initial_poses_match": exact_pairs,
            "paired_improved_trials": sum(pair["success_delta"] > 0 for pair in job_pairs),
            "paired_regressed_trials": sum(pair["success_delta"] < 0 for pair in job_pairs),
            "paired_success_delta": sum(pair["success_delta"] for pair in job_pairs),
            "gained_seeds_json": json.dumps([seed_display(pair["start_seed"]) for pair in job_pairs if pair["success_delta"] > 0]),
            "lost_seeds_json": json.dumps([seed_display(pair["start_seed"]) for pair in job_pairs if pair["success_delta"] < 0]),
            "common_success_trials": len(common),
            "common_success_reference_time_mean_s": float(np.mean([p["reference_final_time_s"] for p in common])) if common else None,
            "common_success_candidate_time_mean_s": float(np.mean([p["candidate_final_time_s"] for p in common])) if common else None,
            "common_success_time_delta_mean_s": float(np.mean([p["common_success_time_delta_s"] for p in common])) if common else None,
            "successful_solver_steps": len(timings),
            "solver_time_mean_s": float(np.mean(timings)) if timings else None,
            "solver_time_p95_s": float(np.percentile(timings, 95)) if timings else None,
            **audits[identity], **completion_audit(job, root, count),
            "eligible_for_comparison": bool(eligible), "safety_comparison_eligible": bool(safety_eligible),
            "trial_audit_issues": ";".join(f"seed{row['start_seed']}:{row['audit_issues']}" for row in rows if row.get("audit_issues")),
            "output_dir": job["output_dir"],
        })
    return summaries, [row for values in data.values() for row in values[0]], pairs


def plot_svg(summaries: list[dict]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(dict.fromkeys((row["environment"], row["cohort"]) for row in summaries))
    if not groups:
        return ""
    figure, axes = plt.subplots(len(groups), 1, figsize=(11, max(3.4, 3.4*len(groups))), squeeze=False)
    for axis, (environment, cohort) in zip(axes.flat, groups):
        rows = [row for row in summaries if (row["environment"], row["cohort"]) == (environment, cohort)]
        for index, row in enumerate(rows):
            rate = 100*(row["success_rate"] or 0.)
            color = "#334155" if row["kind"] == "reference" else "#0f766e" if row["eligible_for_comparison"] else "#cbd5e1"
            axis.bar(index, rate, color=color, width=.65)
            if row["completed_trials"]:
                axis.errorbar(index, rate, yerr=[[max(0., rate-100*row["wilson95_low"])],
                              [max(0., 100*row["wilson95_high"]-rate)]], capsize=3, color="#172554", linewidth=1)
            axis.text(index, rate+3, f"{row['successful_trials']}/{row['completed_trials']}", ha="center", fontsize=8)
        axis.set_xticks(range(len(rows)), [row["job_id"] for row in rows], rotation=25, ha="right", fontsize=8)
        axis.set_ylim(0, 112)
        axis.set_ylabel("Goal success (%)")
        axis.set_title(f"{environment} / {cohort}: observed successes (Wilson 95% intervals)")
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.tight_layout()
    stream = io.StringIO()
    figure.savefig(stream, format="svg", metadata={"Date": None})
    plt.close(figure)
    return stream.getvalue()


def frozen_selection_report(summaries: list[dict], root: Path) -> str:
    path = root / "selection_before_validation.json"
    if not path.exists():
        return ""
    record = shared.read_json(path)
    manifest = shared.read_json(manifest_path(root))
    selected_job = next((job for job in manifest["jobs"] if job["id"] == record.get("selected_search_job")), None)
    selected_row = next((row for row in summaries if row["job_id"] == record.get("selected_search_job")), None)
    if selected_job is None or selected_row is None:
        return "<p>선택 기록의 search job이 현재 manifest에 없습니다. 선택 기록을 점검해야 합니다.</p>"
    config = shared.resolve_path(selected_job["config_path"], root)
    config_matches = (config.exists() and hashlib.sha256(config.read_bytes()).hexdigest()
                      == record.get("selected_config_sha256"))
    choice_matches = (record.get("overrides") == selected_job.get("overrides") and config_matches
                      and record.get("validation_has_not_started") is True)
    rows = []
    for cohort in ("search", "validation"):
        candidate = selected_row if cohort == "search" else next((row for row in summaries
            if row["cohort"] == cohort and row["kind"] == "candidate"
            and json.loads(row["overrides_json"]) == record.get("overrides")), None)
        reference = next((row for row in summaries if row["cohort"] == cohort
                          and row["kind"] == "reference" and row["environment"] == selected_row["environment"]), None)
        seeds = record.get(f"{cohort}_seeds", [])
        seed_text = f"{min(seeds)}–{max(seeds)}" if seeds else "—"
        def counts(row: dict | None) -> str:
            return "미시작" if row is None else f"{row['successful_trials']}/{row['completed_trials']} 완료"
        comparable = (choice_matches and candidate is not None and reference is not None
                      and candidate["safety_comparison_eligible"]
                      and {str(value) for value in seeds} == set(json.loads(candidate["expected_seeds_json"])))
        comparison = (f"+{candidate['paired_improved_trials']} / −{candidate['paired_regressed_trials']}"
                      if comparable else "진행 중 / 비교 미확정")
        penetration = str(candidate["sampled_penetration_trials"]) if comparable else "—"
        rows.append(f"<tr><td>{cohort}</td><td>{html.escape(seed_text)}</td><td>{counts(reference)}</td>"
                    f"<td>{counts(candidate)}</td><td>{comparison}</td><td>{penetration}</td></tr>")
    final_links = []
    for filename, title in (("recommended_config.yaml", "최종 추천 YAML"),
                            ("recommendation.json", "최종 선택 근거"),
                            ("final_validation.json", "최종 검증 기록")):
        if (root / filename).exists():
            final_links.append(f'<a href="{filename}">{title}</a>')
    return (f"<p><b>검증 전 고정한 조합:</b> <code>{html.escape(selected_row['label'])}</code>. "
            f"선택 시각 {html.escape(str(record.get('selected_utc', '')))}; "
            f"선택 기록·설정 해시 {'일치' if choice_matches else '불일치 — 점검 필요'}. "
            "선택 후 validation seed의 결과를 보고 계수를 다시 고르지 않습니다. "
            '<a href="selection_before_validation.json">선택 고정 기록</a></p>'
            '<div class="table"><table><thead><tr><th>구분</th><th>seed</th><th>기존 80% 설정</th>'
            '<th>선택한 조합</th><th>같은 seed 개선/악화</th><th>선택 조합 침투 trial</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div><p>" + " · ".join(final_links) + "</p>")


def make_report(summaries: list[dict], svg: str, root: Path) -> str:
    def fmt(value: Any, places: int = 2) -> str:
        return "—" if value is None else f"{value:.{places}f}"

    rows, leaders = [], []
    groups = list(dict.fromkeys((row["environment"], row["cohort"]) for row in summaries))
    for environment, cohort in groups:
        group = [row for row in summaries if (row["environment"], row["cohort"]) == (environment, cohort)]
        candidates = [row for row in group if row["kind"] != "reference" and row["safety_comparison_eligible"]]
        reference = next((row for row in group if row["kind"] == "reference"), None)
        pending = sum(not row["fully_complete"] for row in group)
        if candidates and cohort != "validation":
            ranked = sorted(candidates, key=lambda row: (-row["successful_trials"], row["sampled_penetration_trials"], row["paired_regressed_trials"]))
            best = ranked[0]
            leaders.append(f"<p><b>{html.escape(environment)} / {html.escape(cohort)}</b>: "
                f"{'현재까지 ' if pending else ''}완료 후보 최고는 <code>{html.escape(best['label'])}</code>, "
                f"{best['successful_trials']}/{best['completed_trials']} 성공입니다. "
                f"같은 cohort 기준 {reference['successful_trials']}/{reference['completed_trials']} 대비 "
                f"추가 성공 {best['gained_seeds_json']}, 상실 {best['lost_seeds_json']}; "
                f"관측 침투 {best['sampled_penetration_trials']}회. 미완료 조건 {pending}개.</p>")
        elif cohort != "validation":
            leaders.append(f"<p>{html.escape(environment)} / {html.escape(cohort)}: 비교 검증 완료 후보를 기다리는 중입니다.</p>")
    for row in summaries:
        css = "" if row["safety_comparison_eligible"] else " class='pending'"
        rows.append(f"<tr{css}><td>{html.escape(row['environment'])}<br>{html.escape(row['cohort'])}</td>"
                    f"<td><b>{html.escape(row['job_id'])}</b><br>{html.escape(row['label'])}</td>"
                    f"<td>{row['successful_trials']}/{row['completed_trials']}</td>"
                    f"<td>+{row['paired_improved_trials']} / −{row['paired_regressed_trials']}</td>"
                    f"<td>{row['stage0_veto_trials']} / {row['movement_failure_trials']} / {row['timeout_trials']} / {row['error_trials']}</td>"
                    f"<td>{row['sampled_penetration_trials']}</td><td>{row['sampled_penetration_free_goal_trials']}</td>"
                    f"<td>{fmt(None if row['min_clearance_m'] is None else 1000*row['min_clearance_m'], 3)}</td>"
                    f"<td>{fmt(row['goal_time_mean_s'])}</td><td>{fmt(row['common_success_time_delta_mean_s'])}</td>"
                    f"<td>{'OK' if row['eligible_for_comparison'] else 'pending / inspect'} / "
                    f"{'OK' if row['safety_comparison_eligible'] else 'pending / inspect'}</td></tr>")
    total = sum(row["completed_trials"] for row in summaries)
    expected = sum(row["expected_trials"] for row in summaries)
    reused = sum(row["completed_trials"] for row in summaries if row["reused"])
    new_expected = sum(row["expected_trials"] for row in summaries if not row["reused"])
    references = sorted({row["label"] for row in summaries if row["kind"] == "reference"})
    reference_labels = "; ".join(html.escape(value) for value in references)
    selection_report = frozen_selection_report(summaries, root)
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>CC-MF noise combinations</title>
<style>body{{font:15px/1.55 system-ui,sans-serif;max-width:1500px;margin:30px auto;padding:0 20px;color:#172033}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #dbe2ea;text-align:left}}
th,code{{background:#f1f5f9}}.pending{{color:#687589}}.table{{overflow:auto}}.plot svg{{max-width:100%;height:auto}}</style>
<h1>CC-MF noise 조합: 탐색과 독립 seed 확인</h1>
<p>분석 대상 완료 {total}/{expected}회: 기존 로그 재사용 {reused}회, 신규 주행 {total-reused}/{new_expected}회.
각 조건은 기존 benchmark에서 manifest에 명시한 noise 계수 조합만 변경합니다.
기준 조합은 <code>{reference_labels}</code>입니다. Search와 validation은 서로 다른 seed 집합으로 따로 집계하며,
후보는 같은 환경·cohort의 기준과만 초기 pose를 일치시켜 비교합니다.</p>
{selection_report}
{''.join(leaders)}
<p>순위는 완료·전체 geometry 검증 조건만 대상으로 성공수, 관측 침투 횟수, 기준 성공 상실 횟수 순으로 정합니다.
선택에 사용한 search 성공률은 일반화 성능의 증명이 아니며, validation 결과도 10개 seed의 작은 표본입니다.
후보 전체의 평균 성공 시간은 성공 seed 구성이 달라질 수 있으므로 공통 성공 seed의 시간 차이를 별도로 제공합니다.</p>
<div class="plot">{svg}</div><div class="table"><table><thead><tr><th>환경/cohort</th><th>조합</th>
<th>성공/완료</th><th>기준 대비 개선/악화</th><th>veto/저이동/timeout/error</th><th>침투 trial</th>
<th>침투 없는 성공</th><th>최소 clearance mm</th><th>성공 시간 s</th><th>공통 성공 시간차 s</th><th>목표/geometry 검증</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>침투는 독립 polygon 계산으로 초기 pose와 모든 기록된 post-control pose(최종 pose 포함)를 검사한 결과입니다.
−1 nm보다 작은 clearance를 침투로 집계합니다. 전체 시간축·끝시간이 완전하지 않으면 안전 지표 비교에서 제외합니다.
이는 샘플 사이의 연속시간 안전 또는 위치추정 오차에 대한 보장이 아닙니다. 실패·timeout·error는 성공률 분모에 포함하고
interrupted는 제외합니다. Stage0 veto는 solve 호출 생략이며 실제 nonzero solver status와 별도 집계됩니다.</p>
<p><a href="combination_summary.csv">조건별 CSV</a> · <a href="trial_summary.csv">trial별 CSV</a> ·
<a href="paired_comparisons.csv">동일 seed 비교</a> · <a href="combination_summary.json">요약 JSON</a> ·
<a href="{manifest_path(root).name}">실험 manifest</a> · <a href="source_hashes.json">실행 소스 해시</a></p></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    root = args.results_dir.resolve()
    output = (args.output_dir or root).resolve()
    summaries, trials, pairs = analyze(root, output)
    shared.write_csv(output / "combination_summary.csv", summaries)
    shared.write_csv(output / "trial_summary.csv", trials)
    shared.write_csv(output / "paired_comparisons.csv", pairs)
    shared.atomic_text(output / "combination_summary.json", json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False))
    svg = "" if args.no_plots else plot_svg(summaries)
    if svg:
        shared.atomic_text(output / "success_rates.svg", svg)
    shared.atomic_text(output / "analysis.html", make_report(summaries, svg, root))
    sources = {"analyze_cc_mf_noise_combinations.py": Path(__file__),
               "analyze_cc_mf_noise_sweep.py": Path(shared.__file__)}
    source_hashes = {}
    for filename, path in sources.items():
        payload = path.read_bytes()
        shared.atomic_text(output / "analysis_sources" / filename, payload.decode())
        source_hashes[filename] = hashlib.sha256(payload).hexdigest()
    shared.atomic_text(output / "analysis_provenance.json", json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(), "source_hashes": source_hashes,
        "manifest_sha256": hashlib.sha256(manifest_path(root).read_bytes()).hexdigest(),
        "input_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                         for name in ("baseline_parameters.json", "selection_before_validation.json",
                                      "reference_reuse.json", "recommendation.json", "final_validation.json")
                         if (root / name).exists()},
        "numpy_version": np.__version__, "geometry_cache_version": shared.CACHE_VERSION,
        "penetration_tolerance_m": shared.PENETRATION_TOLERANCE_M,
        "independent_cohorts_not_pooled": True, "recorded_pose_safety_only": True,
    }, indent=2))
    for row in summaries:
        print(f"{row['job_id']}: {row['successful_trials']}/{row['completed_trials']} "
              f"veto={row['stage0_veto_trials']} penetration={row['sampled_penetration_trials']} "
              f"paired=+{row['paired_improved_trials']}/-{row['paired_regressed_trials']} "
              f"eligible={row['eligible_for_comparison']} safety={row['safety_comparison_eligible']}")


if __name__ == "__main__":
    main()
