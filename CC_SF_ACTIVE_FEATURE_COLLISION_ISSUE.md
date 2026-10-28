# CC-SF: active feature tangent 고정에 따른 실제 clearance 위반

분석 대상: 2026-10-27 rectangle maze 폭별 실험 당시의 소스와 로그.

## 원인

CC-SF는 nominal 자세에서 모든 장애물의 PSDF를 평가하지만, 각 예측 stage의 hard constraint에는 **그중 최소 거리를 주는 active feature의 tangent 하나**를 사용한다. 이 tangent는 다른 feature까지 포함한 최소 거리의 안전한 하한을 보장하지 않는다.

QP 해에서 최소 feature가 바뀌면 기존 tangent를 만족하면서 다른 장애물과 겹칠 수 있다. 실험 당시 구현은 solver status가 0이면 exact one-step clearance 검사 없이 입력을 적용하며, status가 0이 아닐 때만 복구·safe stop으로 진입한다. 따라서 **hard QP row 만족과 실제 무충돌은 서로 다른 판정**이다.

## 대표 데이터

조건: corridor 폭 0.150 m, robot 길이 0.150 m·폭 0.120 m, DD, dt=0.1 s, seed 0. 첫 겹침은 **6.3 → 6.4 s** step에서 발생했다. 아래는 장애물별 signed distance이며, 음수는 겹침을 뜻한다.

| 평가 자세 | 장애물 1번 (mm) | 장애물 4번 (mm) | 최소 장애물 |
|---|---:|---:|---|
| Nominal anchor | +3.988 | +4.020 | 1번 |
| QP가 반환한 예측 자세 | +3.751 | −4.091 | 4번 |
| 실제 입력 적분 후 자세 | +4.268 | −4.495 | 4번 |

- QP의 기존 tangent 값은 **+3.766 mm**였다. 복원한 stage-1 QP row는 `3.766 − (d_min 1.000 + risk margin 1.621) = +1.145 mm`로 만족했다. 실제 solver status는 **0**, 해당 step의 safe stop은 **0회**였다. Risk margin과 QP residual은 저장된 nominal 상태·입력과 선형화된 동역학으로 복원한 값이다.
- 선택했던 1번 장애물만 비교하면 tangent **3.766 mm**와 실제 거리 **3.751 mm**의 차이는 **0.015 mm**다. 큰 오차의 대부분은 매끄러운 동일 feature의 선형화 오차가 아니라 **최소 feature 전환을 놓친 영향**이다.
- Nominal에서 두 장애물의 거리 차이는 **0.031 mm**였지만, QP 해는 anchor에서 **12.952 mm** 이동했다. 기존 tangent의 유효 범위를 제한하는 trust region이나 같은 cycle 내 geometry 재선형화 반복은 없었다.
- 전체 `+3.766 → −4.495 mm` 차이 **8.261 mm**는 동일한 QP 예측 자세에서의 geometry 근사 차이 **7.857 mm**와, 실제 입력 rollout으로 추가된 signed-distance 변화 **0.404 mm**로 나뉜다.

장애물 4번의 벽은 `x=0.600000 m`, 실제 robot의 오른쪽 끝은 `x=0.604495 m`였다. 독립 polygon 계산으로도 약 **4.495 mm 침범**을 확인했다.

## 추가 확인 및 개선 방향

- 폭 0.120 / 0.125 / 0.130 m의 첫 겹침 9건 모두 solver status가 0이었고, 8건에서 nominal과 반환 자세의 최소 장애물이 바뀌었다. 같은 장애물 안의 feature 전환·회전도 별도 오차를 만들 수 있다.
- 9개 사례의 거리와 nominal-anchor gradient 대조에서는 이 현상을 설명할 큰 구현 오류가 발견되지 않았다. Float32/float64 거리 차이는 최대 **0.00005 mm**였고, 기록된 입력의 RK4 재계산은 실제 상태 로그와 일치했다.
- 우선 **입력 적용 전 실제 one-step rollout의 exact PSDF 검사**를 추가하고, 위반 입력을 재선형화·복구 경로로 보내야 한다. 가까운 복수 feature의 제약, trust region, warm-start shift/rollout도 검토 대상이다. 복수 row 추가만으로 비선형 geometry·동역학 오차가 모두 제거되지는 않는다.

이 문서는 기존 로그의 오프라인 분석 결과이며, 새 주행 실험이나 controller 수정은 포함하지 않는다.
