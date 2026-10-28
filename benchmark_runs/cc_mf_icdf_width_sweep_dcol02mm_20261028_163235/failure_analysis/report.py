"""Plot and document the offline reconstruction without altering trial data."""

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = ROOT.parents[1]
SOURCE = ROOT / "source_snapshot"
REPORT = PROJECT / "benchmark_results.md"
HEADING = "## CC-MF-ICDF 성공·실패 분리 원인 분석 (0.2 mm 기준, 2026-10-28)"


def link(label, path, line=None):
    return f"[{label}]({path}{':' + str(line) if line else ''})"


def main():
    records = json.loads((HERE / "one_step_details.json").read_text())
    failed = [row for row in records if row["trial_status"] == "failure"]
    assert len(failed) == 7
    validation = json.loads((HERE / "reconstruction_validation.json").read_text())
    assert validation["sources_unchanged"] and validation["new_solver_or_navigation_runs"] == 0
    assert all(check["native_stage1_returned_covariance_reproduced"] for check in validation["checks"])
    assert all(row["main_solver_status"] == "0" and row["mf_active_stage1"] for row in failed)
    assert all(row["hard_row_residual_mm"] >= -1e-6 for row in failed)
    assert all(row["mf_affine_residual_after_slack"] >= -1e-6 for row in failed)
    assert all(row["mf_frozen_residual_at_returned"] < 0 for row in failed)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), layout="constrained")
    groups = [([row for row in failed if row["predicted_exact_psdf_mm"] < 0],
               "A. Geometry approximation: 4 overlap cases"),
              ([row for row in failed if row["predicted_exact_psdf_mm"] >= .2],
               "B. Dynamics defect: 3 near-margin cases")]
    for ax, (group, title) in zip(axes, groups):
        x = np.arange(len(group))
        for offset, key, label, color, hatch in [
            (-.25, "predicted_affine_phi_mm", "Affine guard at RTI pose", "#999999", "//"),
            (0, "predicted_exact_psdf_mm", "Exact PSDF at RTI pose", "#2764a0", ""),
            (.25, "actual_exact_psdf_mm", "Exact PSDF after plant step", "#c33d38", "")]:
            values = [row[key] for row in group]
            bars = ax.bar(x + offset, values, width=.24, label=label, color=color, hatch=hatch)
            ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], fontsize=8,
                         padding=2, rotation=90 if ax is axes[0] else 0)
        ax.axhline(.2, color="black", linestyle="--", linewidth=1.2, label="Required: 0.200 mm")
        ax.axhline(0, color="black", linewidth=.6)
        ax.set_xticks(x, [f"{round(row['width_m'] * 1000)} / {row['seed']}" for row in group])
        ax.set_xlabel("Robot width (mm) / seed")
        ax.set_ylabel("Signed PSDF (mm)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(y=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle("First crossing below d_col: approximate feasibility versus actual motion\n"
                 "RTI poses are reconstructed from saved one-QP dynamics; plant poses are recorded", fontsize=12)
    figure = HERE / "failure_mechanisms.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)

    max_cov_error = max(row["max_covariance_reconstruction_error"] for row in records)
    lines = [HEADING, "",
        "**주요 원인은 solver의 수치 실패나 MF 비활성화가 아니라, 한 번의 SQP-RTI에서 사용하는 근사 안전 모델과 실제 다음 자세의 차이다.** "
        "실패 7건 중 실제 겹침이 생긴 4건은 geometry Taylor 모델이 다른 장애물 쪽의 위험을 놓쳤고, "
        "겹침 없이 멈춘 3건은 예측 자세와 실제 RK4 적분 자세의 차이로 0.2 mm 기준을 밑돌았다. "
        "그 뒤 `safe_stop_stage0`가 zero input을 계속 적용해 같은 자세에 머물며 20연속 unsuccessful step으로 종료했다.", "",
        "### 로그에서 직접 확인한 사실", "",
        "- 9개 trial 전체에서 main solver failure는 0회다. 실패 7건의 마지막 이동 입력은 모두 `solver_status=0`으로 반환됐고, 미래 stage 1 MF row도 모두 활성화돼 있었다.",
        "- 위험 입력 cycle의 최대 MF slack은 약 7.8e−17–2.0e−12 quantile, 전체 9 trial 최대는 2.953e−10 quantile였다. 큰 slack으로 제약을 풀어준 사례가 아니다.",
        "- 실패 후 20 step은 solver가 20번 infeasible을 반환한 것이 아니다. 현재 exact signed PSDF<0.2 mm를 감지해 solve를 건너뛴 정지 step이다. DD에서 zero input은 자세를 바꾸지 않으므로 같은 veto가 반복된다.", "",
        "### 최초 기준 위반을 만든 한 step의 복원", "",
        "아래 값은 기존 benchmark의 unsigned polygon clearance와 다른 **signed PSDF**다. 양수는 분리, 음수는 겹침을 나타낸다. "
        "`근사 guard`는 nominal tangent를 복원한 RTI 예측 자세에 대입한 값이고, `예측 자세 실제 PSDF`는 같은 예측 자세에서 geometry를 새로 평가한 값이다. "
        "`적분 후 실제 PSDF`는 저장된 실제 자세를 재평가해 다음 cycle의 로그와 대조했다. 요구값은 0.200 mm다.", "",
        "| Width (m) | Seed | 위험 입력 시각 (s) | 근사 guard (mm) | 예측 자세 실제 PSDF (mm) | 적분 후 실제 PSDF (mm) | 주된 위반 요인 |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for row in failed:
        cause = "장애물 branch 전환·기하 근사" if row["predicted_exact_psdf_mm"] < 0 else "동역학 선형화 오차"
        lines.append(f"| {row['width_m']:.3f} | {row['seed']} | {row['time_before_s']:.1f} | "
            f"{row['predicted_affine_phi_mm']:.3f} | {row['predicted_exact_psdf_mm']:.3f} | "
            f"{row['actual_exact_psdf_mm']:.3f} | {cause} |")
    lines += ["", f"![실패 직전 근사 모델과 실제 거리 비교]({figure})", "",
        "그림의 좌우 패널은 y축 범위가 다르다. 점선은 공통 기준 0.200 mm다.", "",
        "**4개 겹침 사례:** 근사 guard는 1.394–4.646 mm로 충분히 안전하다고 평가했지만, 복원한 예측 자세를 geometry로 재평가하면 −0.982–−0.191 mm였다. "
        "4건 모두 nominal의 최소 장애물과 예측 자세의 최소 장애물이 바뀌었다(0부터 시작하는 cluster ID: 4→3, 1→4, 1→4, 4→2). "
        "기하 모델의 과대평가는 2.376–4.836 mm로, 예측–적분 위치 차이 0.060–0.248 mm보다 크다. "
        "따라서 이 4건은 실제 plant에 입력을 적용하기 전의 예측 자세부터 이미 겹침이었다. "
        "하나의 최소 branch의 tangent가 전체 최소 PSDF의 안전한 하한은 아니어서, slack 없는 hard row도 이 위험을 막지 못했다. "
        "Cluster 전환을 확인했지만 내부 vertex/edge ID 전환을 모두 직접 추적한 것은 아니다.", "",
        "**3개 비겹침 정지 사례:** 폭 0.115 m seed 0/1 및 0.125 m seed 2는 예측 자세의 실제 PSDF가 각각 0.620 / 0.567 / 0.204 mm로 기준 이상이었다. "
        "그러나 반환 입력을 현재 자세에 RK4로 적용한 자세는 예측과 각각 0.460 / 0.430 / 0.090 mm 달랐고, 실제 PSDF는 0.161 / 0.138 / 0.137 mm가 됐다. "
        "SQP-RTI가 한 번 푼 선형화된 동역학 equality의 예측 자세와 비선형 RK4 결과가 일치하지 않은 사례다. "
        "저장된 입력을 기존 plant RK4로 다시 평가하면 실제 다음 자세가 오차 0으로 재현돼, 별도의 plant 모델이나 localization noise 탓으로 볼 근거는 없다.", "",
        "MF row 자체도 같은 한계를 보인다. 7건 모두 복원한 반환 상태에서 affine MF row는 거의 0 이상의 residual과 미미한 slack으로 충족됐지만, "
        "그 반환 상태의 covariance와 frozen nominal features로 비선형 ICDF residual을 다시 평가하면 음수였다. "
        "즉 MF가 켜져 있고 slack이 0에 가까워도 한 번의 affine RTI row 충족이 원래 residual 충족을 보장하지 않는다.", "",
        "### 왜 seed에 따라 성공과 실패가 갈리는가", "",
        "시작 자세는 seed마다 다르며 localization error는 꺼져 있다. 이후 nominal pose/input을 한 칸 shift하거나 현재 자세에서 다시 rollout하지 않고 이전 solver 궤적을 유지하므로, "
        "작은 초기 자세 차이가 경로 추종·회전 입력·최소 거리 branch·다음 nominal을 다르게 만든다. "
        "이번 실패는 corner 통과 입력과 최종 여유가 그 차이에 민감한 상황에서 발생했다. 특정 초기 교란 성분 하나가 단독 원인임을 입증한 대조 실험은 아니다.", "",
        "- 폭 0.120 m seed 1 실패 직전과 성공 seed 2의 최근접 자세는 위치 차이 1.082 mm, heading 차이 약 0.00431 rad였다. "
        "실패 입력은 v=0.08833 m/s, ω=0.18201 rad/s로 PSDF를 3.922→−1.036 mm로 바꿨다. "
        "성공 trial은 그 근처에서 v=0.04499 m/s, ω=0.71713 rad/s로 2.874→3.140 mm를 유지했다. "
        "현재 pose가 비슷해도 retained nominal/covariance와 최적 입력이 같지는 않다.",
        "- 폭 0.125 m seed 1은 v=0.00163 m/s로 거의 제자리에서 ω=−1.15755 rad/s로 회전한 한 step만으로 2.570→−0.527 mm가 됐다. "
        "Rectangle에서는 전진 속도가 작아도 회전하는 모서리가 벽 쪽으로 들어갈 수 있다.",
        "- 성공 seed 2의 최소 polygon clearance는 폭 0.115 m에서 0.35495 mm, 폭 0.120 m에서 0.23200 mm였다. "
        "후자는 기준보다 약 32 μm 높을 뿐이다. 그 최소점에서도 예측–적분 위치 차이가 약 0.923 mm였으므로, 성공을 일관된 동역학 정확성이나 넉넉한 안전 여유의 증거로 볼 수 없다. "
        "성공 trial은 이번 기록에서 기준 아래로 내려가는 입력 조합을 피한 것이며, 폭당 3회로 일반적인 성공 확률이나 특정 seed의 우수성을 판정할 수 없다.", "",
        "### 실패가 회복되지 않는 코드 경로와 수정 우선순위", "",
        "1. 고정 nominal의 affine guard와 affine ICDF row로 한 번의 SQP-RTI를 푼다.\n"
        "2. status=0인 입력을 fresh one-step geometry 검사 없이 적용한다.\n"
        "3. 다음 cycle에서 current PSDF<d_col을 발견하면 solve를 건너뛰고 zero input을 적용한다.\n"
        "4. 같은 자세에서 20회 반복해 trial을 종료한다.", "",
        "이 chain에 직접 대응하는 우선순위는 **반환 u0를 실제 RK4로 한 step 전파한 뒤 exact PSDF를 재평가하고, 기준 미달이면 적용 전에 거부·재선형화하는 검사**다. "
        "그다음 현재 자세에서 shifted-input rollout으로 nominal을 만들고, 가까운 여러 obstacle/feature row 또는 trust region·재선형화 반복을 함께 검토할 수 있다. "
        "이미 기준 아래인 상태를 탈출시키는 정책은 별도의 회복 문제다. Stage-0 정지를 끄거나 d_col을 더 낮추는 것만으로 앞선 기하·동역학 근사 오차는 해결되지 않는다. "
        "이 분석에서는 controller를 수정하거나 새 주행/solver를 실행하지 않았다.", "",
        "### 재현성과 분석 범위", "",
        "Nominal/returned pose 전 궤적은 직접 저장돼 있지 않아, stage-0가 현재 자세로 고정되고 nominal u0가 직전 반환 u0라는 코드 정책을 이용해 "
        "`x1_RTI = f(x0, u_prev) + B(x0, u_prev)·(u_current−u_prev)`로 stage-1 pose를 복원했다. "
        "직전 cycle의 복원 stage-1 pose가 이번 nominal anchor가 된다. 이와 같은 equality로 계산한 반환 covariance를 별도 저장된 covariance CSV와 대조했고, "
        f"분석 cycle들까지의 최대 covariance-entry 차이는 {max_cov_error:.3e}였다. "
        "Nominal stage-1 MF residual/probability와 active mask도 9개 대표 step 모두 재현했다. "
        "이 검증은 복원의 일관성을 뒷받침하지만 직접 저장된 pose와의 대조를 대신하지는 않는다. "
        "성공 trial의 비교 step은 최소 clearance 지점이며 실패 trial의 최초 위반 step과 동일 위치/동일 시각 대조는 아니다.", "",
        "- " + link("최초 위반/성공 최소점 수치", HERE / "one_step_reconstruction.csv") + ", "
        + link("자세·입력·기울기 상세", HERE / "one_step_details.json") + ", "
        + link("복원 검증", HERE / "reconstruction_validation.json") + ", "
        + link("오프라인 복원 스크립트", HERE / "reconstruct.py") + ".",
        "- " + link("실패/성공 telemetry·최근접 위치 비교", HERE / "telemetry/comparison.json") + ".",
        "- 코드 근거: " + link("affine guard 정의", SOURCE / "control/cc_mf_icdf_optimizer.py", 171) + ", "
        + link("hard row와 MF slack", SOURCE / "control/cc_mf_optimizer.py", 798) + ", "
        + link("nominal 재사용과 covariance 재전파", SOURCE / "control/cc_mf_optimizer.py", 1003) + ", "
        + link("현재 상태 veto", SOURCE / "control/cc_mf_optimizer.py", 1104) + ", "
        + link("zero-input 정지 계획", SOURCE / "control/cc_mf_optimizer.py", 1180) + ", "
        + link("성공 입력 반환", SOURCE / "control/controller.py", 94) + ".", ""]
    prior = REPORT.read_bytes()
    assert HEADING.encode() not in prior
    REPORT.write_bytes(prior + b"\n" + "\n".join(lines).encode())
    (HERE / "report_validation.json").write_text(json.dumps({
        "old_report_sha256": hashlib.sha256(prior).hexdigest(),
        "old_content_preserved": REPORT.read_bytes().startswith(prior),
        "all_crossing_main_status_zero": True, "all_crossing_stage1_mf_active": True,
        "all_reconstructed_guard_residuals_nonnegative_with_1e_minus6_mm_tolerance": True,
        "all_reconstructed_affine_mf_rows_feasible_with_1e_minus6_tolerance": True,
        "all_returned_frozen_mf_residuals_negative": True,
        "report_heading": HEADING,
        "report_line": prior.decode().count("\n") + 2,
    }, indent=2) + "\n")
    print(HEADING)
    print(f"Report line: {prior.decode().count(chr(10)) + 2}")


if __name__ == "__main__":
    main()
