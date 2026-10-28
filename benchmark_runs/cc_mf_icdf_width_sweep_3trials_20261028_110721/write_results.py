"""Append the validated width sweep to the requested existing Markdown file."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
DESTINATION = PROJECT / "benchmark_results.md"
HEADING = "## CC-MF-ICDF 폭 증가 실험 (2026-10-28)"


def rows(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def link(label, name):
    return f"[{label}]({ROOT / name})"


def main():
    validation = json.loads((ROOT / "validation.json").read_text())
    assert not validation["partial"] and not validation["frozen_sources_changed"]
    assert validation["mean_p05_independently_recomputed"]
    summary = rows("benchmark_summary.csv")
    trials = rows("benchmark_trials.csv")
    events = rows("safe_stop_events.csv")
    progress = json.loads((ROOT / "sweep_progress.json").read_text())
    executed = len(trials)
    goals = sum(row["status"] == "success" for row in trials)
    stops = sum(int(row["safe_stop_steps"]) for row in trials)
    episodes = sum(int(row["safe_stop_episodes"]) for row in trials)
    stage0 = sum(int(row["stage0_guard_stop_steps"]) for row in trials)
    solver = sum(int(row["solver_failure_stop_steps"]) for row in trials)
    skipped = [row for row in progress if row["status"] == "skipped"]
    samples = sum(int(row["clearance_samples"]) for row in trials)
    overlap = sum(int(row["obstacle_overlap_samples"]) for row in trials)
    boundary = sum(int(row["boundary_violation_samples"]) for row in trials)
    lines = [HEADING, "", f"폭 0.105–0.130 m를 0.005 m 간격으로 증가시키며 폭당 3회를 요청한 새 실험이다. "
             f"실제로 **{executed} trials를 실행해 목표 도달 {goals}/{executed}, "
             f"safe-stop step {stops}회, 연속 safe-stop 구간 {episodes}개**를 기록했다.", ""]
    if skipped:
        failed_width = next(row["width_m"] for row in progress if row.get("successes") == 0)
        lines += [f"**{failed_width:.3f} m에서 3회 모두 목표 도달에 실패**해 "
                  + " / ".join(f"{row['width_m']:.3f} m" for row in skipped)
                  + "는 실행하지 않았다. 미실행 폭의 측정값은 빈 값으로 표시했다.", ""]
    else:
        lines += ["한 폭에서 3회 모두 목표 도달에 실패한 경우가 없어 여섯 폭을 모두 실행했다.", ""]
    lines += ["### 조건과 측정 기준", "",
        "- Rectangle 길이 0.15 m, maze, DD, A*, seed 0–2, dt=0.1 s, trial당 최대 60 s. Localization error 및 animation/plot은 비활성화했다.",
        "- 기존 CC-SF 폭 증가 실험과 맞춰 시작 자세 교란은 x/y 표준편차 20 mm, heading 표준편차 0.05 rad, 각 성분 ±3σ, 초기 clearance ≥1 mm로 설정했다. 실행한 모든 폭에서 seed별 시작 자세가 서로 같고 기존 CC-SF 실험의 자세와도 일치함을 검증했다. 상단 과거 5-trial 실험의 초기 clearance ≥10 mm 조건과 구분한다.",
        "- 실패 판정은 연속 unsuccessful control step 20회 또는 4 s 동안 이동량 ≤0.03 m이다. Goal 판정은 기존 simulation 판정을 그대로 사용했다. 한 폭의 세 trial이 모두 목표 도달에 실패하면 이후 더 큰 폭을 생략하는 규칙을 적용했다.",
        "- CC-MF-ICDF는 d_min=d_col=d_mf_mask=0.001 m, chance_epsilon=0.20, horizon=20 (2 s), robust horizon=10, MF slack linear=100 / quadratic=1을 유지했다. `enable_backup_solver=false`이므로 QP 실패 시 기본 복구는 safe stop이다. 전체 설정은 "
        + link("optimizer parameters", "width_105mm/optimizer_parameters.json") + "에 보관했다.",
        "- `method_description.md`의 nominal geometry·affine hard PSDF guard·soft MF row 구성을 확인했다. ICDF 변환의 상세 정의는 "
        f"[inverse_cdf_mf_residual_revised.md]({PROJECT / 'inverse_cdf_mf_residual_revised.md'})를 함께 확인했다. "
        "확률합 R≤1/2의 유효 정의역에서 residual은 Φ⁻¹(ε)−Φ⁻¹(R)이며, R>1/2에서는 구현된 선형 확장 branch를 사용한다. Slack 단위는 standard-normal quantile이며 거리 단위가 아니다. Stage 0 OCP row는 비활성이고 현재 자세의 exact signed PSDF<d_col이면 solve 전에 `safe_stop_stage0`으로 들어간다. 미래 stage에는 affine guard가 적용되므로 solver 성공을 exact nonlinear clearance 보장으로 해석하지 않는다.",
        "- **Clearance**는 실제 rectangle polygon과 모든 장애물 polygon의 거리 및 지도 경계 margin 중 최솟값이다. 초기 자세와 모든 적분 후 자세(최종 자세 포함)를 사용했다. 실패 trial의 정지 step도 포함한 전체 표본을 합쳐 mean/p05/min을 계산했으며, trial별 통계를 평균하지 않았다. 장애물 접촉·겹침 거리는 0, 지도 경계 침범은 음수다. 아래 최소값은 signed PSDF가 아니며 장애물 침투 깊이를 나타내지 않는다.",
        "- 조기 종료된 실패 trial과 정지 자세의 반복 표본도 포함하므로, 폭별 평균·분위수는 서로 동일한 경로 구간만을 비교한 값이 아니다. 겹침 표본 수도 같은 겹친 자세에서 정지한 동안의 반복 기록을 포함하며, 독립적인 충돌 사건 수가 아니다.",
        "- **Safe stop 횟수는 zero input을 적용한 control step 수**이며 `safe_stop`과 `safe_stop_stage0`를 모두 포함한다. 연속 20 step은 20회·2.0 s·연속 구간 1개다. 폭별 seed 표기는 0 / 1 / 2 순서다.",
        "- 실행 소스를 새 폴더에 고정하고 같은 Python 3.10.13 / PyTorch 2.4.1 / CasADi 3.7.0 / RTX 3060, PyTorch CPU thread 6개에서 순차 실행했다. Controller 동작은 수정하지 않았다.",
        "", "### 폭별 결과", "",
        "| Robot width (m) | 목표 도달 | Mean clearance (mm) | P05 (mm) | Min (mm) | Safe-stop step: seed 0 / 1 / 2 | 연속 정지 구간 | 겹침 표본 | 경계 침범 표본 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    by_width = {float(row["width_m"]): row for row in summary}
    for record in progress:
        width = record["width_m"]
        if record["status"] == "skipped":
            lines.append(f"| {width:.3f} | 미실행 | — | — | — | — | — | — | — |")
            continue
        row = by_width[width]
        lines.append(f"| {width:.3f} | {row['successes']}/3 | "
            f"{float(row['clearance_mean_mm']):.3f} | {float(row['clearance_p05_mm']):.3f} | "
            f"{float(row['clearance_min_mm']):.3f} | {row['safe_stop_steps_seed012']} | "
            f"{row['safe_stop_episodes']} | {row['obstacle_overlap_samples']} | {row['boundary_violation_samples']} |")
    lines += ["", f"총 {samples:,}개 자세에서 장애물 겹침 {overlap}개, 지도 경계 침범 {boundary}개가 관측됐다. "
        "겹침과 경계 침범 표본은 중복될 수 있다. 기록된 모든 자세에서 clearance>0을 유지하면서 목표에 도달한 trial 수는 "
        + " / ".join(f"{float(row['width_m']):.3f} m: {row['positive_clearance_goal_trials']}/3"
                     for row in summary) + "이다. "
        "폭당 3회 관측이며, 저장된 자세 사이의 연속시간 충돌이나 실제 확률적 안전성을 검증한 결과는 아니다.",
        "", "### Trial별 결과", "",
        "| Width (m) | Seed | 종료 상태 | 종료 시간 (s) | Mean clearance (mm) | Min (mm) | Safe-stop step | Stage-0 guard / QP 실패 |",
        "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in trials:
        status = {"success": "목표 도달", "failure": "실패", "timeout": "시간 초과"}[row["status"]]
        lines.append(f"| {float(row['width_m']):.3f} | {row['seed']} | {status} | "
            f"{float(row['final_time_s']):.1f} | {float(row['clearance_mean_mm']):.3f} | "
            f"{float(row['clearance_min_mm']):.3f} | {row['safe_stop_steps']} | "
            f"{row['stage0_guard_stop_steps']} / {row['solver_failure_stop_steps']} |")
    lines += ["", "### Safe stop과 실패 사유", "",
              f"Safe-stop step {stops}회 중 stage-0 guard에 의한 정지는 {stage0}회, QP 실패 후 정지는 {solver}회였다.", ""]
    for event in events:
        phi = event["initial_signed_psdf_m"]
        phi_note = f", 시작 signed PSDF {float(phi) * 1000:.3f} mm" if phi else ""
        status_note = f", main status={event['first_main_status']}" if event["first_main_status"] else ""
        lines.append(f"- {float(event['width_m']):.3f} m, seed {event['seed']}: "
            f"{float(event['start_time_s']):.1f}–{float(event['end_time_s']):.1f} s에 "
            f"{event['steps']} step 연속 정지 (`{event['modes']}`{status_note}{phi_note}).")
    for trial in trials:
        if trial["status"] != "success":
            lines.append(f"- {float(trial['width_m']):.3f} m, seed {trial['seed']}의 종료 사유: "
                         f"`{trial['failure_reason']}`.")
    if not events:
        lines.append("- 실행한 모든 trial에서 safe stop은 발생하지 않았다.")
    overlapped_trials = [row for row in trials if int(row["obstacle_overlap_samples"]) > 0]
    if overlapped_trials:
        lines += ["", "**" + ", ".join(f"{float(row['width_m']):.3f} m seed {row['seed']}"
            for row in overlapped_trials)
            + "에서는 실제 장애물 겹침이 생긴 뒤 stage-0 guard가 정지시켰다.** "
            "따라서 이 결과의 safe stop을 이전 step의 충돌을 예방했다는 뜻으로 해석할 수 없다. "
            "더 큰 폭의 생략은 이번 3-trial 실패에 따른 실행 규칙이며, 물리적 통과 가능성의 한계를 확정한 결과는 아니다."]
    lines += ["", "### 원본 자료와 검증", "",
        "- " + link("폭별 집계", "benchmark_summary.csv") + ", "
        + link("trial별 집계", "benchmark_trials.csv") + ", "
        + link("모든 자세의 clearance", "clearance_samples.csv") + ".",
        "- " + link("정지 구간", "safe_stop_events.csv") + ", "
        + link("접촉·겹침·경계 침범 자세", "geometry_violations.csv") + ", "
        + link("실행·생략 내역", "sweep_progress.json") + ". 각 폭 폴더에 config, geometry, trial 결과, native cycle/history CSV, 관측 cycle CSV, runtime JSON 및 실행 로그를 보관했다.",
        "- " + link("검증 결과", "validation.json") + ": source 사본 불변성, 폭 외 geometry·optimizer parameter 동일성, seed별 시작 자세 일치, 초기·최종 자세 포함, pooled mean/p05 독립 재계산을 확인했다. "
        f"Shapely로 {samples:,}개 전체 자세의 polygon 거리를 별도 대조했다. Safe-stop step은 native cycle 로그·관측 로그·누적 runtime counter·적용 zero input 및 실제 정지 자세와 일치했고, 성공 step 수는 solver CSV와 대조했다.",
        "- 재현용 " + link("순차 실행 스크립트", "run_sweep.py") + " 및 "
        + link("집계·검증 스크립트", "analyze.py") + ", "
        + link("고정 소스 해시", "source_hashes.json") + "를 보관했다. 재실행에는 새 결과 폴더를 사용한다.", ""]
    content = "\n".join(lines)
    before = DESTINATION.read_bytes()
    assert HEADING.encode() not in before, "This experiment is already documented."
    DESTINATION.write_bytes(before + b"\n" + content.encode())
    assert DESTINATION.read_bytes().startswith(before)
    (ROOT / "report_integrity.json").write_text(json.dumps({
        "previous_content_sha256": hashlib.sha256(before).hexdigest(),
        "previous_content_preserved_byte_for_byte": True,
        "report_path": str(DESTINATION), "appended_heading": HEADING,
        "executed_trials": executed, "goals": goals, "safe_stop_steps": stops,
    }, indent=2) + "\n")
    print(content)


if __name__ == "__main__":
    main()
