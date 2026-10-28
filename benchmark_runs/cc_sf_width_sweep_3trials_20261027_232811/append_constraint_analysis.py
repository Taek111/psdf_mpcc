"""Append the code/log diagnosis to the existing benchmark report."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = ROOT / "source_snapshot"
OUTPUT = ROOT / "clearance_diagnosis"
HEADING = "## CC-SF hard constraint와 실제 clearance 위반 원인 분석"


def code(name, line, label):
    return f"[{label}]({SOURCE / name}:{line})"


def artifact(name, label):
    return f"[{label}]({OUTPUT / name})"


with (OUTPUT / "first_overlap_reconstruction.csv").open() as stream:
    records = list(csv.DictReader(stream))
validation = json.loads((OUTPUT / "validation.json").read_text())
assert validation["analyzed_first_overlap_events"] == 9
assert validation["new_navigation_or_solver_runs"] == 0
assert all(float(row["reconstructed_stage1_qp_residual_mm"]) >= -1e-6 for row in records)

text = [HEADING, "",
    "**원인은 hard row의 비활성화나 slack이 아니라, hard row가 실제 polygon separation이 아닌 고정된 최소-feature Taylor 모델과 한 번의 SQP-RTI QP에 적용된다는 점이다.** 실제 입력을 적용하기 전 exact one-step geometry를 검사하는 단계가 없어, 그 QP가 성공하면 실제 겹침으로 이어지는 입력도 적용된다. 새 주행·solver 실행 없이 기존 0.120 / 0.125 / 0.130 m 로그의 첫 겹침 9건을 복원해 확인했다.", "",
    "### 1. 실제로 enforce하는 hard constraint", "",
    code("control/analytic_psdf_casadi.py", 38, "AnalyticPSDFCasADi.__call__") + "은 exact PSDF 대신 다음 affine 식을 반환한다.", "",
    "$$", r"\hat\phi_i(p)=\phi(\bar p_i)+\bar g_i^\top(p-\bar p_i).", "$$", "",
    code("control/cc_sf_optimizer.py", 854, "presolve parameter 갱신") + "에서 nominal pose의 PSDF 값과 gradient를 한 번 계산해 `model.p`에 넣는다. " + code("control/cc_sf_optimizer.py", 702, "hard SF row") + "는 이 affine 값으로 `hat_phi - risk_margin - d_min >= 0`을 설정한다. Row에는 slack이 없고, 미래 stage 1–20에서 활성화된다. 다만 risk margin과 covariance dynamics는 비선형이므로 SQP-RTI가 실제 QP에서 적용하는 것은 다시 다음과 같은 1차 row다.", "",
    "$$", r"\hat h_i^{\rm QP}=\hat\phi_i(p_i)-d_{\min}-\left[r_i(\bar\xi_i)+\nabla_\xi r_i(\bar\xi_i)^\top(\xi_i-\bar\xi_i)\right]\ge0.", "$$", "",
    "이 부등식에는 모든 장애물의 exact distance나 비선형 polygon overlap 판정이 들어 있지 않다. 최소 PSDF는 여러 feature/장애물 branch의 최솟값이므로, 한 branch에서 얻은 tangent가 다른 branch까지 포함한 PSDF의 안전한 하한인 것은 아니다. Nominal의 최소 장애물에서 멀어지거나 그 옆을 따라 움직이면서 다른 장애물로 들어가도 이 단일 row는 만족할 수 있다. 같은 장애물에서도 활성 vertex/edge가 바뀌거나 회전하면 같은 문제가 생긴다.", "",
    "### 2. 0.120 m, seed 0: 6.3 → 6.4 s의 수치 증거", "",
    "Nominal stage-1의 최소 장애물은 1번이었다. Solver가 반환한 자세와 실제 적분 후 자세는 4번 장애물과 겹친다. 전체 장애물 17개가 모두 wrapper에 들어 있으므로 obstacle capacity/ROI 누락 사례가 아니다.", "",
    "| 항목 | 값 (mm) |", "|---|---:|",
    "| Nominal stage-1 PSDF | +3.988508 |",
    "| 반환된 예측 자세의 Taylor PSDF | +3.766252 |",
    "| 복원한 QP risk margin | +1.621063 |",
    "| QP의 요구 clearance: d_min + margin | +2.621063 |",
    "| 복원한 hard QP residual | **+1.145189** |",
    "| 반환된 예측 자세에서 재평가한 exact signed PSDF | **−4.091114** |",
    "| 실제 입력 RK4 적분 후 signed PSDF | **−4.495174** |", "",
    "즉 solver가 hard QP row를 어긴 것이 아니라 **잘못 안전하다고 보는 근사 row를 만족한 것**이다. 같은 예측 자세에서 Taylor 모델이 signed PSDF를 7.857366 mm 과대평가한다. 실제 signed PSDF는 원래 PSDF 함수를 해당 자세에서 새로 평가한 값이며, Shapely 교차 면적으로도 겹침을 확인했다. 앞선 benchmark 표의 polygon clearance는 겹침을 0으로 기록하므로, 그 통계와 위 signed PSDF의 음수 크기를 직접 비교하면 안 된다.", "",
    "![첫 겹침에서의 footprint와 Taylor/실제 signed PSDF 비교](" + str(OUTPUT / "first_overlap_geometry.png") + ")", "",
    "### 3. 9개 trial의 첫 겹침 대조", "",
    "아래 QP residual은 로그의 nominal state와 applied input 및 선형화된 stage-0 equality로 복원했다. 반환 pose는 다음 cycle의 pre-solve stage-1 pose에서 회수했다. Covariance는 다음 cycle에 재생성되어 덮어쓰므로 반환 covariance를 직접 읽은 값은 아니며, QP의 선형 covariance transition으로 재구성한 값이다.", "",
    "| Width (m) | Seed | 겹침 시작 (s) | 최소 장애물: nominal → 반환 | 반환 Taylor PSDF (mm) | 반환 exact PSDF (mm) | 복원 QP residual (mm) | 예측–실제 위치차 (mm) |",
    "|---|---:|---:|---|---:|---:|---:|---:|"]
for row in records:
    text.append(f"| {float(row['width_m']):.3f} | {row['seed']} | {float(row['time_after_s']):.1f} | "
                f"{row['nominal_cluster']} → {row['predicted_cluster']} | "
                f"{float(row['affine_phi_at_predicted_mm']):.3f} | "
                f"{float(row['actual_psdf_at_predicted_mm']):.3f} | "
                f"{float(row['reconstructed_stage1_qp_residual_mm']):.3f} | "
                f"{float(row['dynamics_translation_defect_mm']):.3f} |")
text.extend(["",
    "9건 모두 `fast_status=0`, `safe_stop=0`으로 겹침을 시작했고, 복원한 hard QP row는 모두 비음수였다. 반환된 예측 자세 9개 모두 exact PSDF가 d_min=1 mm보다 작았으며 8개는 이미 음수였다. 최소 장애물은 8/9건에서 바뀌었다. 같은 장애물 1번이 유지된 0.130 m seed 1도 Taylor PSDF 7.888 mm 대비 exact PSDF −1.719 mm로, 한 장애물 내의 geometry branch/회전 선형화 오차가 있다. 이 로그 분석만으로 내부 vertex/edge ID까지 직접 확정한 것은 아니다.", "",
    "### 4. RTI 동역학 오차와 warm start가 추가로 작용한다", "",
    code("models/dd_dynamics.py", 6, "plant RK4") + "와 " + code("control/cc_sf_optimizer.py", 260, "OCP의 symbolic DD step") + "은 같은 RK4식을 사용한다. 저장된 실제 상태에 적용 입력을 다시 대입하면 실제 다음 상태와 부동소수점 값까지 일치했다. 따라서 별도의 plant 모델, localization error 또는 잘못된 입력 stage를 적용한 문제로 볼 근거가 없다.", "",
    "하지만 SQP-RTI는 이 RK4의 선형화된 동역학을 한 번 풀기 때문에 `x1_returned`와 `f_RK4(x_current, u0_returned)`가 정확히 같을 필요는 없다. 첫 겹침들에서 위치차는 0.011–0.544 mm였고, geometry Taylor 과대평가 3.334–9.606 mm보다 작았다. 0.125 m seed 0은 반환 pose의 실제 PSDF가 +0.177 mm였으나, 실제 plant pose에서는 −0.075 mm가 되어 이 작은 동역학 오차가 접촉 여부까지 바꿨다. 반환 pose부터 이미 d_min=1 mm를 충족하지 못했다.", "",
    code("control/cc_sf_optimizer.py", 818, "초기 상태·guess 갱신") + "은 stage 0만 현재 자세로 바꾸고 stage 1 이후 pose/input을 한 칸 shift하거나 현재 상태에서 다시 rollout하지 않는다. " + code("control/cc_sf_optimizer.py", 539, "guess 재생성") + "은 covariance만 재계산한다. 이 구조에서는 stage-1 geometry anchor가 직전 예측 시각에 머물고, QP 출력이 anchor에서 크게 움직일 수 있다. 0.120 m seed 0은 stage-1 위치가 anchor에서 12.952 mm 이동했고, 0.125 m 세 seed는 heading이 약 0.12 rad 바뀌었다. 이를 제한하는 geometry trust region이나 같은 cycle 내 exact PSDF 재선형화 반복은 없다. 이는 큰 근사 오차를 허용하는 요인이지, shift만 하면 exact 안전성이 보장된다는 뜻은 아니다.", "",
    "Risk/covariance도 별도 근사 오차가 있다. 0.130 m seed 1의 첫 겹침에서 복원한 **QP 선형 risk margin은 −0.857 mm**였다. 같은 반환 covariance에 비선형 margin 식을 적용하면 +0.144 mm, 반환 입력으로 covariance를 정확히 전파한 값에서는 +1.912 mm였다(geometry gradient는 해당 stage의 nominal 값으로 고정). 원래 margin의 비음수성이 그 1차 QP 근사까지 자동으로 유지되는 것은 아니다. 다만 9건 모두 geometry 오차만으로 d_min 위반을 확인할 수 있어, 이 추가 오차를 모든 겹침의 단독 원인으로 주장하지 않는다.", "",
    "### 5. Safe stop이 발생하지 않는 직접적인 코드 경로", "",
    code("control/cc_sf_optimizer.py", 1021, "복구 진입 조건") + "은 `status != 0`일 때만 복구 solver와 safe stop으로 들어간다. `status == 0`인 해에 대해 exact PSDF, 비선형 dynamics defect 또는 실제 다음 자세의 충돌 여부로 거부하는 분기가 없다. `debug_mode=false`라 반환 궤적의 fresh-risk 진단도 생략된다. Debug를 켜도 그 분기는 값 계산/출력이며 안전 입력 검증 조건은 아니다.", "",
    code("control/controller.py", 101, "입력 적용") + "은 성공 상태를 기록한 뒤 반환된 stage-0 입력을 그대로 내보낸다. " + code("sim/simulation.py", 156, "simulation loop") + "는 이를 적분하고 목표/정체/연속 solver 실패를 확인한다. Exact collision을 trial 실패나 safe stop으로 바꾸는 검사가 없다. 이 때문에 겹침이 생겨도 계속 주행하고 목표 도달로 기록될 수 있다.", "",
    "[설치된 acados SQP-RTI 소스](/home/taek111/projects/acados/acados/ocp_nlp/ocp_nlp_sqp_rti.c:621)도 QP 처리 후 status=0을 설정하며 exact nonlinear collision의 만족을 판정하지 않는다. 이 버전은 QP MAXITER도 같은 경로로 허용하지만 원본 로그에는 내부 qp_status가 없어, 이번 9건에 MAXITER가 실제 발생했다고 해석하지는 않았다. 이번 복원에서는 hard QP row 자체가 만족되는 사례를 확인했다.", "",
    "생성 solver의 `nh_0=0`도 현재 pose의 충돌이 즉시 constraint 실패로 연결되지 않는 배경이다. 다만 stage-1 hard row가 실제로 있어도 첫 겹침을 막지 못한 사례이므로, stage-0 row 부재만으로 진입 원인을 설명할 수 없다. Stage-0 row 추가만으로 이전 입력이 만드는 충돌을 예방하지도 못한다.", "",
    "### 6. 수정한다면 우선 다뤄야 할 지점", "",
    "1. **입력 적용 전 exact one-step 검사**: 현재 실제/추정 자세와 반환 u0로 동일 RK4를 계산하고 full geometry의 PSDF를 재평가한다. d_col 미달이면 main status=0이어도 해당 입력을 거부해 재선형화·복구 경로로 보낸다. 이미 겹친 자세에서 zero input을 주는 것은 새 충돌 예방과 다른 문제이므로 별도 복원 정책이 필요하다. 연속시간 안전을 요구하면 step 사이의 swept trajectory도 검사해야 한다.",
    "2. **기하 근사의 유효 범위 관리**: 현재 상태에서 shifted-input rollout으로 nominal을 만들고, 여러 가까운 장애물/feature의 hard row, trust region, exact geometry 재평가 후 재선형화를 함께 고려한다. 여러 row만 추가해도 회전/동역학 근사 오차는 남으므로 exact acceptance 검사를 대체하지 않는다.",
    "3. **수렴/물리 일관성 검사**: SQP-RTI status와 함께 비선형 dynamics defect, fresh exact PSDF, covariance propagation consistency를 기록한다. 단순히 solver를 full SQP로 바꾸더라도 `model.p`의 고정 affine PSDF를 갱신하지 않으면 다른 장애물로 들어가는 문제는 남을 수 있다.", "",
    "이번 요청에서는 controller 동작을 수정하지 않았다. 분석의 핵심은 minimum-feature 전환이 risk margin만 바꾸는 문제가 아니라, **hard row의 거리 모델 자체를 낙관적으로 만들 수 있음**을 로그에서 직접 확인했다는 점이다.", "",
    "원본/재현 자료: " + artifact("first_overlap_reconstruction.csv", "9건의 수치 대조") + ", " + artifact("first_overlap_details.json", "pose·gradient·input·covariance 상세") + ", " + artifact("validation.json", "재구성 검증") + ", [오프라인 분석 스크립트](" + str(ROOT / "analyze_constraint_violations.py") + "). 원본 nominal PSDF/risk terms/margin을 재현했고, 선형 동역학으로 복원한 pose와 다음 cycle에 보존된 반환 pose의 최대 차이는 1.73e-14였다. QP covariance는 직접 로그가 아닌 위 선형 equality를 이용한 복원값이라는 한계가 있다.", ""])

section = "\n".join(text)
destination = PROJECT / "benchmark_results.md"
existing = destination.read_text()
assert HEADING not in existing
destination.write_text(existing.rstrip() + "\n\n" + section)
print(f"Appended analysis to {destination}")
