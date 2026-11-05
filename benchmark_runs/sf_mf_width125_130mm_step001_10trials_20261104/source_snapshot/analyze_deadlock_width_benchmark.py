#!/usr/bin/env python3
"""Aggregate saved trials and append the new sweep to benchmark_results.md."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import yaml

from run_width_benchmark import METHODS, ROOT, write_csv, write_json


LABELS = {"mpcc": "MPCC", "cc_sf": "CC-SF", "cc_ecs": "CC-ECS", "cc_mf": "CC-MF"}
TUNING = {"vmin": -1.0, "vmax": 1.0, "v_s_target": 1.0, "omegamin": -1.0,
          "omegamax": 1.0, "a_max": 0.5, "alpha_max": 5.0}


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def statistic(values, percentile=None, scale=1.0):
    if not values:
        return None
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Non-finite observations")
    return float(np.mean(data) if percentile is None else np.percentile(data, percentile)) * scale


def formatted(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def analyze(output):
    manifest = json.loads((output / "experiment.json").read_text())
    trials = read_csv(output / "trial_results.csv")
    widths = manifest["widths_m"]
    expected = {(width, method, seed) for width in widths for method in METHODS for seed in range(10)}
    seen, poses, integrity, videos = set(), {}, [], []
    groups = {(width, method): {"trials": [], "clearances": [], "times": [], "solver_times": [],
                               "missing": 0, "animations": 0, "native_safe_stops_confirmed": 0}
              for width in widths for method in METHODS}
    for trial in trials:
        width, method, seed = float(trial["width_m"]), trial["method"], int(trial["seed"])
        key = width, method, seed
        assert key in expected and key not in seen, f"Unexpected/duplicate trial: {key}"
        seen.add(key)
        assert (trial["status"] == "success") == bool(int(trial["success"]))
        group = groups[width, method]
        group["trials"].append(trial)
        run = output / trial["run_dir"]
        if not (run / "outcome.json").exists():
            group["missing"] += 1
            if (run / "run.log").exists():
                group["native_safe_stops_confirmed"] += (run / "run.log").read_text().count(
                    "Applied zero-input safe-stop plan after infeasible CC-SF solve.")
            integrity.append({"trial": key, "issue": "missing trial telemetry after worker failure"})
            partial = run / "partial_animation.json"
            animation = None
            status = trial["status"]
            if partial.exists():
                metadata = json.loads(partial.read_text())
                animation = Path(metadata["animation_path"])
                assert animation.is_file() and animation.stat().st_size > 0
                group["animations"] += 1
                status += f" (partial {metadata['source']})"
            videos.append((width, method, seed, status, animation))
            continue
        outcome = json.loads((run / "outcome.json").read_text())
        params = json.loads((run / "optimizer_parameters.json").read_text())
        config = yaml.safe_load((run / "config.yaml").read_text())
        assert config["rectangle_geometry"] == {"length": 0.15, "width": width}
        for field, value in TUNING.items():
            assert params[field] == value, (key, field, params[field])
        defaults = manifest["resolved_defaults"][method]
        for field in ("sigma_f0", "sigma_l0", "sigma_psi0", "covariance_growth_scale",
                      "r_delta_v", "r_delta_omega", "d_min"):
            if field in defaults:
                assert params[field] == defaults[field], (key, field)
        assert outcome["status"] == trial["status"]
        assert math.isclose(float(trial["final_time_s"]), outcome["final_time"], abs_tol=1e-9)
        if trial["status"] == "success":
            assert outcome["goal_reached"] and outcome["distance_to_goal"] <= 0.02
        pose = np.array(outcome["initial_pose"])
        pose_key = width, seed
        if pose_key in poses:
            np.testing.assert_allclose(pose, poses[pose_key], rtol=0, atol=1e-12)
        poses[pose_key] = pose
        minimum_clearance = manifest.get("min_clearance_by_width_m", {}).get(f"{width:.3f}", 0.01)
        assert config["start_perturbation"]["min_clearance"] == minimum_clearance
        assert outcome["start_perturbation"]["initial_clearance"] >= minimum_clearance - 1e-12
        cycles = read_csv(run / "cycles.csv")
        clearance = read_csv(run / "clearance.csv")
        assert len(cycles) == int(trial["steps"])
        assert len(clearance) == int(trial["clearance_samples"]) == len(cycles) + 1
        for index, cycle in enumerate(cycles):
            assert int(cycle["step"]) == index
            assert int(cycle["included_in_aggregate"]) == int(index > 0)
            assert float(cycle["cycle_time_s"]) > 0
        for index, sample in enumerate(clearance):
            assert int(sample["pose_index"]) == index
            assert math.isclose(float(sample["simulation_time_s"]), index * 0.1, abs_tol=1e-9)
        np.testing.assert_allclose([float(clearance[0][axis]) for axis in ("x", "y", "theta")], pose)
        group["times"].extend(float(cycle["cycle_time_s"]) for cycle in cycles[1:])
        group["clearances"].extend(float(sample["clearance_m"]) for sample in clearance)
        solver_steps = read_csv(run / "successful_solver_steps.csv")
        group["solver_times"].extend(float(step["solver_computation_time_s"]) for step in solver_steps)
        animation = Path(outcome["animation_path"]) if outcome.get("animation_path") else None
        if animation is not None and animation.is_file() and animation.stat().st_size > 0:
            group["animations"] += 1
        else:
            integrity.append({"trial": key, "issue": "missing animation"})
        if trial["error"]:
            integrity.append({"trial": key, "issue": trial["error"]})
        videos.append((width, method, seed, trial["status"], animation))
    summaries = []
    for (width, method), group in groups.items():
        rows = group["trials"]
        if not rows:
            continue
        successes = [row for row in rows if int(row["success"])]
        summaries.append({"width_m": width, "method": method, "completed_trials": len(rows),
            "successful_trials": len(successes), "success_rate": len(successes) / len(rows),
            "mean_clearance_mm": statistic(group["clearances"], scale=1000),
            "p05_clearance_mm": statistic(group["clearances"], 5, 1000),
            "mean_computation_time_ms": statistic(group["times"], scale=1000),
            "p95_computation_time_ms": statistic(group["times"], 95, 1000),
            "mean_traversal_time_s": statistic([float(row["final_time_s"]) for row in successes]),
            "safe_stop_count": sum(int(row["safe_stop_count"] or 0) for row in rows)
                               + group["native_safe_stops_confirmed"],
            "safe_stop_count_is_lower_bound": bool(group["missing"]),
            "clearance_samples": len(group["clearances"]), "timing_samples": len(group["times"]),
            "missing_telemetry_trials": group["missing"], "saved_animations": group["animations"],
        })
    write_csv(output / "benchmark_summary.csv", summaries)
    if len(trials) == 280:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
        colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
        for method, marker, color in zip(METHODS, ("o", "s", "^", "D"), colors):
            rows = [row for row in summaries if row["method"] == method]
            ax.plot([row["width_m"] * 1000 for row in rows],
                    [row["success_rate"] * 100 for row in rows],
                    marker=marker, color=color, label=LABELS[method], linewidth=1.7,
                    markerfacecolor="none")
        ax.set(xlabel="Robot width (mm)", ylabel="Goal-reaching success rate (%)",
               xticks=[width * 1000 for width in widths], yticks=range(0, 101, 20), ylim=(-4, 104))
        ax.grid(alpha=0.25)
        ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
        fig.savefig(output / "success_rate.png", dpi=160)
        plt.close(fig)
    hashes = json.loads((output / "source_hashes.json").read_text())
    mismatches = [name for name, digest in hashes.items()
                  if hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() != digest]
    assert not mismatches, mismatches
    write_json(output / "report_integrity.json", {
        "completed_trials": len(trials), "expected_trials": len(expected),
        "saved_animations": sum(group["animations"] for group in groups.values()),
        "unique_trial_keys": len(seen), "matched_start_pose_groups": len(poses),
        "snapshot_hashes_match": True, "issues": integrity,
    })
    index = ["# Deadlock tuning width sweep animations", "",
             "| Width (m) | Controller | Seed | Status | Animation |", "|---:|---|---:|---|---|"]
    for width, method, seed, status, animation in videos:
        link = f"[MP4]({os.path.relpath(animation, output)})" if animation else "—"
        index.append(f"| {width:.3f} | {LABELS[method]} | {seed} | {status} | {link} |")
    (output / "animation_index.md").write_text("\n".join(index) + "\n")
    write_report(output, manifest, trials, summaries, integrity)
    print(f"Report updated: {len(trials)}/280 trials; "
          f"{sum(group['animations'] for group in groups.values())} animations")


def write_report(output, manifest, trials, summaries, issues):
    relative = output.relative_to(ROOT).as_posix()
    finished = len(trials) == 280
    movement_window_sec = manifest.get("failure_criteria", {}).get("movement_window_sec", 4.0)
    lines = ["## Deadlock tuning 적용: robot width 0.070–0.130m, 각 10 trials", "",
        f"- 실행일: {manifest['started_at_kst'][:10]} (KST). "
        f"{'완료' if finished else '진행 중'}: **{len(trials)}/280 trials**. "
        "Rectangle 길이 0.150m, maze, DD, A*. 같은 PC에서 순차 실행했다.",
        "- MPCC 공통 기본값: **vmin=-1.0m/s, vmax=1.0m/s, v_s_target=1.0m/s, "
        "omegamin=-1.0rad/s, omegamax=1.0rad/s, a_max=0.5m/s², alpha_max=5.0rad/s²**. "
        "CC-SF·CC-ECS·CC-MF가 이 값을 상속한다.",
        "- 폭 0.070/0.080/0.090/0.100/0.110/0.120/0.130m × 네 controller × seed 0–9, "
        "조합별 10회다. x/y 표준편차 0.02m, heading 표준편차 0.05rad, ±3σ 제한, "
        "초기 clearance는 **0.070–0.120m에서 ≥10mm, 0.130m에서만 ≥1mm**다. "
        "폭 0.130m는 0.150m 시작 통로에서 10mm 여유와 무작위 자세 교란을 동시에 만족할 수 없어 "
        "사용자 지정으로 완화했다. 같은 폭·seed에서 시작 자세가 동일하다. Localization error는 끈다.",
        "- 이번 실행은 위의 과거 실험과 달리 **현재 controller 기본 불확실성**을 유지했다. "
        "CC-SF·CC-MF: (σf,σl,σψ)=(0.0002m,0.00025m,0.000025rad), covariance_growth_scale=0.8. "
        "CC-ECS: (0.0002m,0.00025m,0.00025rad), process noise=0. "
        "d_min은 MPCC·CC-SF·CC-ECS 1mm, CC-MF 0.2mm이며, r_delta_v=r_delta_omega=0.05다. "
        "과거 표와 tuning만 다른 통제 비교로 해석할 수 없다.",
        "- dt=0.1s, horizon=20, trial 제한 60s. 목표 위치 L1 오차 ≤ 0.02m이면 성공이다. "
        f"{movement_window_sec:g}초 동안 위치 변화 ≤ 0.03m 또는 solver 연속 20회 실패면 실패다. "
        "접촉·겹침 자체는 별도 실패 조건이 아니다.",
        "- Success Rate는 성공/완료 trial이며 timeout·failure·error도 분모에 포함한다. "
        "Mean Traversal time은 성공 trial만의 평균이다. "
        "Mean/P05 clearance는 초기 자세와 매 적분 후 실제 rectangle의 정확한 polygon 거리 및 "
        "지도 경계 margin 중 최솟값을 전체 성공·실패 trial에 걸쳐 합쳐 계산한다. "
        "접촉·겹침은 0, 지도 경계 침범은 음수다.",
        "- Mean/P95 computation time은 local reference 생성부터 controller 전처리·solver·복구·내부 로깅까지 "
        "한 사이클의 wall-clock 시간이다. perf_counter와 CUDA 동기화를 사용하며 각 trial의 첫 사이클을 제외한다. "
        "이후 실패·safe stop step도 모두 포함한다. Global path·plant 적분·파일 및 animation 저장은 제외한다. "
        "평균과 분위수는 전체 유효 step을 합쳐 계산하며 trial별 통계의 평균이 아니다.",
        "- Safe stop은 zero-input fallback step의 합계다. MPCC에는 이 fallback이 없어 0으로 표시한다. "
        "계측 누락이 있으면 확인된 하한에 ≥를 붙인다.",
        "- Animation은 기존 animate_world()/FFMpegWriter로 animations/<controller>/에 저장한다. "
        "10fps, 100 DPI, libx264 ultrafast/CRF28이며 별도 재인코딩은 하지 않는다. "
        + ("처음 4개는 모든 frame, 이후는 test_nmpc.py 기본값 frame_skip=5다."
           if manifest.get("animation_frame_skip_note") else "test_nmpc.py 기본값 frame_skip=5다."),
        f"- 실행 환경: Python {manifest['python'].split()[0]}, PyTorch {manifest['torch']}, "
        f"CasADi {manifest['casadi']}, {manifest['gpu']}, Torch CPU thread {manifest['torch_threads']}개.",
        f"- [실행 원본·설정]({relative}/experiment.json), [trial 결과]({relative}/trial_results.csv), "
        f"[폭별 집계]({relative}/benchmark_summary.csv), [animation 목록]({relative}/animation_index.md).", ""]
    if not finished:
        lines += ["**아래 비율은 현재 완료한 trial 기준의 중간 결과다. 조합별 최종 분모는 10이다.**", ""]
    else:
        partial_sources = []
        for trial in trials:
            partial = output / trial["run_dir"] / "partial_animation.json"
            if partial.exists():
                partial_sources.append(json.loads(partial.read_text())["source"])
        if partial_sources:
            saved = sum(row["saved_animations"] for row in summaries)
            lines += [f"저장 영상 {saved}개 중 {saved - len(partial_sources)}개는 원래 trial 종료까지의 영상이다. "
                f"Native crash trial은 원본 checkpoint 부분 영상 {partial_sources.count('checkpoint')}개와 "
                f"동일 조건·seed의 별도 재실행 부분 영상 {partial_sources.count('replay')}개로 구분했다. "
                "재실행은 위 280회 집계에 추가하지 않았다.", ""]
        all_success_widths = [width for width in manifest["widths_m"]
            if all(row["successful_trials"] == 10 for row in summaries if row["width_m"] == width)]
        if all_success_widths:
            lines += ["네 controller가 모두 10/10 성공한 폭은 "
                      + ", ".join(f"{width:.3f}m" for width in all_success_widths) + "다.", ""]
        widest = max(manifest["widths_m"])
        lines += [f"폭 {widest:.3f}m의 성공 횟수는 " + ", ".join(
            f"{LABELS[row['method']]} {row['successful_trials']}/10"
            for row in summaries if row["width_m"] == widest) + "이다.", ""]
        if (output / "success_rate.png").exists():
            lines += [f"![Robot width별 목표 도달 성공률]({relative}/success_rate.png)", ""]
    for width in manifest["widths_m"]:
        rows = [row for row in summaries if row["width_m"] == width]
        if not rows:
            continue
        lines += [f"### Robot width {width:.3f}m — deadlock tuning", "",
            "| Controller | Success Rate | Mean clearance (mm) | P05 clearance (mm) | "
            "Mean computation time (ms/step) | P95 computation time (ms/step) | Mean Traversal time (s) | Safe stop (회) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            safe = ("≥" if row["safe_stop_count_is_lower_bound"] else "") + str(row["safe_stop_count"])
            values = [LABELS[row["method"]],
                f"{row['success_rate']:.0%} ({row['successful_trials']}/{row['completed_trials']})",
                *[formatted(row[field]) for field in ("mean_clearance_mm", "p05_clearance_mm",
                    "mean_computation_time_ms", "p95_computation_time_ms")],
                formatted(row["mean_traversal_time_s"], 2), safe]
            lines.append("| " + " | ".join(values) + " |")
        negative = [row for row in rows if any(
            row[field] is not None and row[field] < 0
            for field in ("mean_clearance_mm", "p05_clearance_mm"))]
        if negative:
            lines += ["", "음수 clearance는 실제 footprint의 지도 경계 침범을 뜻한다. "
                      "해당 값과 실패 중의 모든 적분 후 자세도 집계에 포함했다."]
        failures = [row for row in trials if float(row["width_m"]) == width and not int(row["success"])]
        lines += ["", "**실패 trial**", ""]
        if not failures:
            lines += ["없음.", ""]
        else:
            lines += ["| Controller | Trial (seed) | 종료 시간 (s) | 실패 사유 |",
                      "|---|---:|---:|---|"]
            for row in failures:
                time = formatted(float(row["final_time_s"]) if row["final_time_s"] else None, 1)
                reason = row["failure_reason"].replace("|", "\\|").replace("\n", " ")
                if reason == "worker_exit_-11":
                    reason = "Native crash (SIGSEGV, exit -11); step 계측 저장 전 종료"
                lines.append(f"| {LABELS[row['method']]} | {row['trial']} ({row['seed']}) | {time} | {reason} |")
            lines.append("")
        missing_rows = [row for row in rows if row["missing_telemetry_trials"]]
        if missing_rows:
            lines += ["Native crash로 종료한 trial도 성공률 분모에 포함했다. "
                "해당 trial의 clearance·계산 시간·종료 시각은 저장되지 않아 다른 지표에서 제외했다. "
                "부분 영상은 checkpoint 또는 동일 설정·seed의 별도 재실행으로 만들었으며, "
                "재실행 영상은 원래 trial의 유실된 궤적을 증명하지 않는다. "
                "Animation 목록에 partial checkpoint/replay를 구분했다.", ""]
    if issues:
        lines += [f"계측·animation 저장 이슈 {len(issues)}건은 "
                  f"[검증 기록]({relative}/report_integrity.json)에 명시했다.", ""]
    text = "\n".join(lines)
    # The standalone copy has links relative to its own output directory.
    (output / "benchmark_results.md").write_text(
        "# Deadlock tuning width benchmark\n\n" + text.replace(relative + "/", ""), encoding="utf-8")
    target = ROOT / "benchmark_results.md"
    existing = target.read_text(encoding="utf-8")
    start = f"<!-- deadlock-width:{output.name}:start -->"
    end = f"<!-- deadlock-width:{output.name}:end -->"
    section = start + "\n\n" + text + "\n" + end
    if start in existing:
        before, remaining = existing.split(start, 1)
        _, after = remaining.split(end, 1)
        existing = before + section + after
    else:
        existing = existing.rstrip() + "\n\n" + section + "\n"
    target.write_text(existing, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    analyze(parser.parse_args().results_dir.resolve())
