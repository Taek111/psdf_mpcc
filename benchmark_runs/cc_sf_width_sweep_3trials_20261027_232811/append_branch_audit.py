"""Add the independent distance/gradient audit to the benchmark analysis."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "clearance_diagnosis"
PROJECT = ROOT.parents[1]
audit = json.loads((OUTPUT / "branch_audit.json").read_text())
p = audit["primary_case"]
heading = "## 8 mm의 차이를 같은 feature의 선형화 오차로 볼 수 있는가"
parts = [heading, "",
    "추가 분해 결과, 0.120 m seed 0 사례에서 **선택된 장애물 1번 branch 자체의 매끄러운 선형화 오차는 약 0.015 mm**였다. 앞서 약 7.857 mm를 ‘geometry Taylor 오차’로 묶어 표현한 값의 대부분은 다른 장애물 branch가 최소가 된 영향이다. 큰 kinematics 곡률 오차나 PSDF gradient 구현 오류로 8 mm가 생겼다고 해석할 필요는 없다.", "",
    "### 장애물별 거리로 분리한 결과", "",
    "| 평가 위치 | 장애물 1번 signed distance (mm) | 장애물 4번 signed distance (mm) | 전체 최소 |",
    "|---|---:|---:|---|",
    "| Nominal anchor | +3.988483 | +4.019862 | 1번 |",
    "| 반환된 QP 예측 자세 | +3.751371 | −4.091155 | 4번 |",
    "| 실제 입력 적분 후 자세 | +4.267927 | −4.495209 | 4번 |", "",
    "독립 float64 분석에서 장애물 1번의 tangent 예측은 +3.766251 mm였고, 같은 장애물의 실제 거리는 +3.751371 mm였다. 차이는 **0.014880 mm**다. 반면 빠진 장애물 4번의 tangent를 계산하면 **−4.091217 mm**, 그 장애물의 실제 거리는 −4.091155 mm였다. 4번의 거리도 자체 선형화는 정확했지만, 그 row가 QP에 들어가지 않았다. 이 추가 수치는 원본 실행의 float32 값을 재해석하기 위한 float64 geometry 감사 결과라 마지막 자릿수에서 차이가 있다.", "",
    "즉 함수는 `phi(p)=min(d1(p),d4(p),...)`인데, QP가 anchor에서 선택한 `d1`의 tangent를 계속 사용했다. 실제 최소가 `d4`로 바뀐 이후 그 tangent는 전체 최소 거리의 근사로서 크게 낙관적이다. PSDF 값은 연속이어도 gradient가 branch 전환점에서 바뀌므로, 한 매끄러운 branch 안에서의 작은 2차 Taylor remainder와 다른 현상이다.", "",
    f"Anchor의 두 거리 차이는 **{p['anchor_competitor_gap_mm']:.6f} mm**에 불과했다. Anchor→QP 해 선분을 따라 약 **{p['translation_until_minimum_switch_mm']:.6f} mm** 이동하면 최소 장애물이 바뀐다. 그런데 실제 QP 위치 수정량은 **{p['returned_motion_translation_mm']:.6f} mm**, heading 수정량은 **{p['returned_motion_heading_deg']:.6f}°**였다. 이 선분은 branch 변화 확인을 위한 pose 보간이며 실제 robot의 시간 궤적이라는 뜻은 아니다. 약 13 mm 위치 수정에 비해 약 4 mm의 여유는 작다.", "",
    "![기존 최소 branch의 tangent와 실제 최소 거리의 분리](" + str(OUTPUT / "distance_branch_switch.png") + ")", "",
    "### 3.766 mm → −4.495 mm 차이의 분해", "",
    "원본 float32 로그와 일치하는 값으로 다음과 같이 나뉜다.", "",
    "| 단계 | Signed PSDF/근사값 (mm) | 차이 (mm) |", "|---|---:|---:|",
    "| QP 반환 자세에서 기존 tangent 예측 | +3.766252 | |",
    "| 같은 QP 반환 자세에서 전체 PSDF 재평가 | −4.091114 | 7.857366 |",
    "| 입력을 실제 RK4로 적분한 자세의 PSDF | −4.495174 | 0.404060 |", "",
    "합계 8.261426 mm 중 7.857366 mm는 동일한 반환 자세에서 발생한 geometry 모델 차이다. 나머지 0.404060 mm는 RTI 예측 자세와 실제 입력 rollout 자세 사이의 signed-distance 변화다. 뒤의 위치 오차 벡터 norm은 0.516655 mm로, signed-distance 변화와 같은 값일 필요가 없다.", "",
    "### 다른 원인 후보에 대한 대조", "",
    "- **거리 함수/겹침 판정 오류**: 9개 첫 겹침 사례의 anchor·반환·실제 자세에서 17개 장애물을 각각 대조했다(459개 거리). 독립 Shapely separation 및 SAT minimum-translation 계산과 float64 PSDF가 일치했다. 이 특정 사례는 4번 장애물의 `x=0.600 m` 벽과 robot의 오른쪽 끝 `x=0.604495209 m`만으로도 4.495209 mm 침범을 확인할 수 있다.",
    f"- **Gradient 또는 좌표계 오류**: 위 9개 anchor의 활성 장애물 gradient를 독립 polygon 거리의 central finite difference와 대조했다. 세 성분별 절대 오차의 최댓값은 {audit['max_analytic_gradient_error']:.3g}였다(translation 미분과 heading 미분은 단위가 다르므로 이를 하나의 물리 오차로 해석하지 않는다). 이 위치들에서 mm급 오차를 설명하는 gradient 오류는 관측되지 않았다. 전체 pose 공간의 모든 branch에 대한 검증을 의미하지는 않는다.",
    f"- **float32 반올림 또는 단위 변환**: 원본 float32 PSDF와 같은 geometry의 float64 PSDF 차이는 최대 {audit['max_float32_vs_float64_error_mm']:.6f} mm였다. 8 mm 차이를 설명하는 크기가 아니다.",
    "- **Plant 모델·입력 적용·시점 불일치**: 앞선 분석에서 저장 입력으로 동일 RK4를 재계산한 상태가 로그와 정확히 일치했고, 다음 cycle의 원본 PSDF도 복원값과 일치했다. 현재 확보한 기록에는 다른 입력을 적용했거나 한 step을 잘못 맞춘 흔적이 없다.",
    "- **추가 구현 요인**: pose/input warm start를 한 칸 shift하고 현재 상태에서 rollout하는 처리가 없고, trust region·same-cycle geometry 재선형화·exact one-step acceptance 검사도 없다. 이 조합은 거의 동률인 feature 중 하나에만 만든 tangent에서 QP가 멀리 이동하고도 그대로 입력을 내보내게 한다. 다른 사례에서는 covariance/risk QP 근사도 추가 오차를 만들지만, 위 8 mm 차이는 장애물별 거리만으로 이미 설명된다.", "",
    "재현 자료: [branch 감사 수치](" + str(OUTPUT / "branch_audit.json") + "), [보간 선분의 장애물별 거리](" + str(OUTPUT / "primary_branch_path.csv") + "), [독립 감사 스크립트](" + str(ROOT / "audit_distance_branches.py") + "). 새 주행이나 solver 실행은 수행하지 않았다.", ""]

report = PROJECT / "benchmark_results.md"
existing = report.read_text()
assert heading not in existing
report.write_text(existing.rstrip() + "\n\n" + "\n".join(parts))
print(f"Appended independent branch audit to {report}")
