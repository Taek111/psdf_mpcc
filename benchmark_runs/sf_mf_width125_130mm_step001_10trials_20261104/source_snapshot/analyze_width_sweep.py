#!/usr/bin/env python3
"""Validate and report the saved rectangle-width sweep without rerunning trials."""

import argparse
from collections import Counter
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


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def measure(values, percentile=None, scale=1.0):
    if not values:
        return None
    data = np.asarray(values, dtype=float)
    assert np.isfinite(data).all()
    return float(np.mean(data) if percentile is None else np.percentile(data, percentile)) * scale


def display(value, precision=3):
    return "—" if value is None else f"{value:.{precision}f}"


def analyze(output):
    manifest = json.loads((output / "experiment.json").read_text())
    trials = read_csv(output / "trial_results.csv")
    widths, repeats = manifest["widths_m"], manifest["trials_per_configuration"]
    methods = manifest.get("methods", METHODS)
    expected = {(width, method, seed) for width in widths for method in methods
                for seed in range(repeats)}
    seen, poses, issues = set(), {}, []
    groups = {(width, method): {"trials": [], "clearance": [], "times": [], "missing": 0}
              for width in widths for method in methods}
    for trial in trials:
        width, method, seed = float(trial["width_m"]), trial["method"], int(trial["seed"])
        key = width, method, seed
        assert key in expected and key not in seen, key
        seen.add(key)
        assert (trial["status"] == "success") == bool(int(trial["success"]))
        group = groups[width, method]
        group["trials"].append(trial)
        run = output / trial["run_dir"]
        if not (run / "outcome.json").exists():
            group["missing"] += 1
            issues.append({"trial": key, "issue": "worker failure; missing telemetry"})
            continue
        outcome = json.loads((run / "outcome.json").read_text())
        params = json.loads((run / "optimizer_parameters.json").read_text())
        config = yaml.safe_load((run / "config.yaml").read_text())
        assert config["rectangle_geometry"] == {"length": 0.15, "width": width}
        assert config["defaults"]["simulation_time"] == manifest["simulation_limit_s"]
        assert config["trial_failure_criteria"] == manifest["failure_criteria"]
        assert not config["localization_error"]["enabled"]
        defaults = manifest["resolved_defaults"][method]
        for field, value in defaults.items():
            # Some numpy arrays are converted internally; JSON preserves them.
            assert field in params and params[field] == value, (key, field)
        assert outcome["status"] == trial["status"]
        assert math.isclose(float(trial["final_time_s"]), outcome["final_time"], abs_tol=1e-9)
        if trial["status"] == "success":
            assert outcome["goal_reached"] and outcome["distance_to_goal"] <= 0.02
        pose = np.array(outcome["initial_pose"])
        if (width, seed) in poses:
            np.testing.assert_allclose(pose, poses[width, seed], rtol=0, atol=1e-12)
        poses[width, seed] = pose
        minimum = manifest["min_clearance_by_width_m"][f"{width:.3f}"]
        assert config["start_perturbation"]["min_clearance"] == minimum
        assert outcome["start_perturbation"]["initial_clearance"] >= minimum - 1e-12
        cycles, clearance = read_csv(run / "cycles.csv"), read_csv(run / "clearance.csv")
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
        safe_stops = sum(cycle["solve_mode"] in ("safe_stop", "safe_stop_stage0") for cycle in cycles)
        assert safe_stops == int(trial["safe_stop_count"]), (key, safe_stops)
        assert safe_stops == outcome["runtime_stats"].get("safe_stop_count", 0)
        group["times"].extend(float(cycle["cycle_time_s"]) for cycle in cycles[1:])
        group["clearance"].extend(float(sample["clearance_m"]) for sample in clearance)
        if trial["error"]:
            issues.append({"trial": key, "issue": trial["error"]})
    summaries = []
    for (width, method), group in groups.items():
        rows = group["trials"]
        if not rows:
            continue
        successes = [row for row in rows if int(row["success"])]
        summaries.append({
            "width_m": width, "method": method, "completed_trials": len(rows),
            "successful_trials": len(successes), "success_rate": len(successes) / len(rows),
            "mean_clearance_mm": measure(group["clearance"], scale=1000),
            "p05_clearance_mm": measure(group["clearance"], 5, 1000),
            "mean_computation_time_ms": measure(group["times"], scale=1000),
            "p95_computation_time_ms": measure(group["times"], 95, 1000),
            "mean_traversal_time_s": measure([float(row["final_time_s"]) for row in successes]),
            "safe_stop_count": sum(int(row["safe_stop_count"] or 0) for row in rows),
            "safe_stop_count_is_lower_bound": bool(group["missing"]),
            "clearance_samples": len(group["clearance"]), "timing_samples": len(group["times"]),
            "missing_telemetry_trials": group["missing"],
        })
    write_csv(output / "benchmark_summary.csv", summaries)
    hashes = json.loads((output / "source_hashes.json").read_text())
    assert all(hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() == digest
               for name, digest in hashes.items())
    write_json(output / "report_integrity.json", {
        "completed_trials": len(trials), "expected_trials": len(expected),
        "unique_trial_keys": len(seen), "matched_start_pose_groups": len(poses),
        "snapshot_hashes_match": True, "safe_stop_modes_reconcile": True,
        "issues": issues,
    })
    write_report(output, manifest, trials, summaries, issues, len(expected))
    print(f"Validated and reported {len(trials)}/{len(expected)} trials")


def write_report(output, manifest, trials, summaries, issues, expected):
    relative = output.relative_to(ROOT).as_posix()
    complete = len(trials) == expected
    methods = manifest.get("methods", METHODS)
    repeats = manifest["trials_per_configuration"]
    width_text = ", ".join(f"{width:.3f}" for width in manifest["widths_m"])
    method_text = "·".join(LABELS[method] for method in methods)
    lines = [f"# Rectangle 너비별 주행 benchmark — {method_text}", "",
             f"실행일 {manifest['started_at_kst'][:10]} (KST). "
             f"**{len(trials)}/{expected} trials {'완료' if complete else '진행 중'}**.", "",
             "maze / differential_drive (DD) / A*, rectangle 길이 0.150 m 고정. "
             f"너비 {width_text} m의 {len(manifest['widths_m'])}개 너비에서 {method_text}를 "
             f"각 {repeats}회(seed 0–{repeats - 1}) 같은 PC에서 순차 실행했다. "
             "Localization error는 끄고 animation은 생성하지 않았다.", "",
             "설정은 `config/config_benchmark.yaml`의 현재 공통 조건을 사용했다. "
             f"trial 제한 {manifest['simulation_limit_s']:g} s, "
             f"이동 정체 창 {manifest['failure_criteria']['movement_window_sec']:g} s, "
             f"이동 임계값 {manifest['failure_criteria']['movement_threshold']:g} m, "
             f"연속 실패 한도 {manifest['failure_criteria']['max_consecutive_solver_failures']}회다. "
             "목표 위치 L1 오차 ≤0.02 m가 성공 판정이다. 접촉·겹침 자체는 별도 실패 조건이 아니다.", "",
             "시작 자세에 x/y 표준편차 각각 0.02 m, heading 표준편차 0.05 rad, ±3σ의 Gaussian 교란을 적용했다. "
             "모든 너비에서 초기 clearance ≥1 mm이며 같은 너비·seed의 방법들은 시작 자세가 같다. "
             "과거 보고서의 일부 너비(≥10 mm)와 초기 여유 조건이 다르다.", "",
             "컨트롤러는 실행 시작 시점의 코드를 고정하고 기본 파라미터를 사용했다. "
             + (f"[앞선 실험]({manifest['reference_run']}/experiment.json)과 시뮬레이션 코드 및 "
                "해당 controller의 전체 파라미터가 동일한 것을 확인했다."
                if manifest.get("reference_run") else
                "이전 실행과 파라미터도 다르므로 과거 결과와 직접적인 통제 비교는 할 수 없다."), "",
             "| Controller | d_min (mm) | a_max (m/s²) | r_delta_omega | σf / σl / σψ (m / m / rad) |",
             "|---|---:|---:|---:|---|"]
    for method in methods:
        params = manifest["resolved_defaults"][method]
        sigmas = " / ".join(f"{params[name]:g}" for name in ("sigma_f0", "sigma_l0", "sigma_psi0")) if method != "mpcc" else "—"
        lines.append(f"| {LABELS[method]} | {params['d_min'] * 1000:g} | {params['a_max']:g} | "
                     f"{params['r_delta_omega']:g} | {sigmas} |")
    lines += ["", "## 지표 정의", "",
              "- Clearance: 초기 및 매 적분 후 실제 rectangle과 장애물의 정확한 polygon 거리, 지도 경계 margin 중 최솟값. "
              "접촉·겹침은 0, 지도 경계 침범은 음수다. 성공·실패 trial의 전체 표본을 합쳐 mean/P05를 계산한다.",
              "- Computation time: local reference 생성부터 controller 전처리·파라미터 갱신·solver·복구·내부 로깅까지의 "
              "제어 사이클 wall-clock 시간(perf_counter + CUDA 동기화). 각 trial 첫 사이클은 제외하고 이후 실패·fallback도 포함한다. "
              "경로 계획·plant 적분·결과 저장은 제외한다. 전체 유효 step을 합쳐 mean/P95를 계산한다.",
              "- Traversal time: 성공 trial의 시뮬레이션 시간 평균. 성공 0건은 —다.",
              "- Safe stop: zero-input fallback 제어 step 합계(연속 정지도 매 step 집계). "
              "MPCC는 해당 fallback이 없어 0이다. 계측 유실 시 확인 가능한 하한을 ≥로 표시한다.", "",
              "## 성공률", "", "| Width (m) | " + " | ".join(LABELS[method] for method in methods) + " |",
              "|---:|" + "---:|" * len(methods)]
    for width in manifest["widths_m"]:
        rows = {row["method"]: row for row in summaries if row["width_m"] == width}
        if rows:
            lines.append("| " + f"{width:.3f} | " + " | ".join(
                f"{rows[method]['successful_trials']}/{rows[method]['completed_trials']}" if method in rows else "—"
                for method in methods) + " |")
    if complete:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for method, marker, color in zip(METHODS, ("o", "s", "^", "D"),
                                         ("#0072B2", "#D55E00", "#009E73", "#CC79A7")):
            if method not in methods:
                continue
            rows = [row for row in summaries if row["method"] == method]
            for ax, field, scale in ((axes[0], "success_rate", 100), (axes[1], "safe_stop_count", 1)):
                ax.plot([row["width_m"] * 1000 for row in rows],
                        [row[field] * scale for row in rows], marker=marker, color=color,
                        label=LABELS[method], markerfacecolor="none", linewidth=1.5)
        for ax, ylabel in zip(axes, ("Goal-reaching success rate (%)", f"Safe-stop steps ({repeats} trials)")):
            ax.set(xlabel="Robot width (mm)", ylabel=ylabel,
                   xticks=[width * 1000 for width in manifest["widths_m"]])
            ax.tick_params(axis="x", rotation=45)
            ax.grid(alpha=0.25)
        axes[0].set_ylim(-4, 104)
        axes[1].set_ylim(bottom=0)
        axes[0].legend(frameon=False)
        fig.savefig(output / "width_sweep.png", dpi=160)
        plt.close(fig)
        lines += ["", f"![너비별 성공률과 safe stop]({relative}/width_sweep.png)"]
    lines += ["", "## 상세 결과", "",
              "| Width (m) | Controller | 성공/완료 | Mean clearance (mm) | P05 clearance (mm) | "
              "Mean computation (ms/step) | P95 computation (ms/step) | Mean traversal (s) | Safe stop (step) |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        values = [f"{row['width_m']:.3f}", LABELS[row["method"]],
                  f"{row['successful_trials']}/{row['completed_trials']}",
                  *[display(row[field]) for field in ("mean_clearance_mm", "p05_clearance_mm",
                                                     "mean_computation_time_ms", "p95_computation_time_ms")],
                  display(row["mean_traversal_time_s"], 2),
                  ("≥" if row["safe_stop_count_is_lower_bound"] else "") + str(row["safe_stop_count"])]
        lines.append("| " + " | ".join(values) + " |")
    failures = [row for row in trials if not int(row["success"])]
    lines += ["", "## 실패 기록", "",
              f"목표 미도달 trial {len(failures)}건. 성공률 분모에는 failure·timeout·error를 포함한다.", ""]
    if failures:
        lines += ["| Width (m) | Controller | Status | Failure reason | Count |", "|---:|---|---|---|---:|"]
        counts = Counter((row["width_m"], row["method"], row["status"],
                          row["failure_reason"].split(" (")[0]) for row in failures)
        for (width, method, status, reason), count in sorted(counts.items()):
            lines.append(f"| {float(width):.3f} | {LABELS[method]} | {status} | {reason} | {count} |")
        lines += ["", "<details>", "<summary>Trial별 실패 상세</summary>", "",
                  "| Width (m) | Controller | Seed | End time (s) | Failure reason |", "|---:|---|---:|---:|---|"]
        for row in failures:
            reason = row["failure_reason"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {float(row['width_m']):.3f} | {LABELS[row['method']]} | {row['seed']} | "
                         f"{display(float(row['final_time_s']) if row['final_time_s'] else None, 1)} | {reason} |")
        lines += ["", "</details>"]
    if issues:
        lines += ["", f"계측 이슈 {len(issues)}건. 유실된 trial도 성공률 분모에 포함하며, "
                  "해당 clearance·계산 시간 표본은 제외하고 safe stop은 하한으로 표시한다."]
    lines += ["", "## 실행 자료", "",
              f"Python {manifest['python'].split()[0]}, PyTorch {manifest['torch']}, CasADi {manifest['casadi']}, "
              f"{manifest['gpu']}, Torch CPU threads {manifest['torch_threads']}.", "",
              f"[실행 조건·전체 파라미터]({relative}/experiment.json) · "
              f"[Trial 결과]({relative}/trial_results.csv) · "
              f"[{len(manifest['widths_m']) * len(methods)}개 조합 집계]({relative}/benchmark_summary.csv) · "
              f"[검증 기록]({relative}/report_integrity.json)", ""]
    text = "\n".join(lines)
    standalone = text.replace(relative + "/", "")
    if manifest.get("reference_run"):
        reference = manifest["reference_run"]
        standalone = standalone.replace(
            reference + "/", os.path.relpath(ROOT / reference, output) + "/")
    (output / "benchmark_results.md").write_text(standalone, encoding="utf-8")
    target = ROOT / "benchmark_results.md"
    existing = target.read_text(encoding="utf-8")
    start, end = f"<!-- width-sweep:{output.name}:start -->", f"<!-- width-sweep:{output.name}:end -->"
    section = start + "\n\n" + text + "\n" + end
    if start in existing:
        before, rest = existing.split(start, 1)
        _, after = rest.split(end, 1)
        existing = before + section + after
    else:
        existing = section + "\n\n---\n\n" + existing
    target.write_text(existing, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    analyze(parser.parse_args().results_dir.resolve())
