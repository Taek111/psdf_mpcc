# CC-MF-ICDF의 기하·동역학 선형화 오차와 충돌 이슈

작성일: 2026-10-28

## 1. 요약

**근사 제약을 만족한 solver 입력이 실제 이동에서는 clearance 기준을 위반했다.**
실패 7건의 위험 입력은 모두 solver status=0으로 반환됐고, stage-1 MF row도 활성화돼 있었다.
MF slack은 거의 0이었다. 주요 원인은 solver의 수치 실패나 큰 slack 허용이 아니라 **기하·동역학 근사의 오차**다.

기준 위반 후에는 `safe_stop_stage0`가 zero input을 반복 적용한다.
자세가 변하지 않아 같은 정지 판정이 계속되고, 20연속 unsuccessful control step으로 trial이 종료된다.
따라서 정지는 최종 종료 경로이며, 먼저 분석해야 할 문제는 **정지 직전 입력이 clearance를 무너뜨린 이유**다.

## 2. 실험 조건과 결과

- Controller: CC-MF-ICDF, 환경: maze, DD, A*.
- Rectangle 길이 0.15 m, 폭 0.115 / 0.120 / 0.125 m에서 각 3회, seed 0–2.
- `d_col = d_min = 0.0002 m` = 0.2 mm. 별도 `d_mf_mask`는 제거했다.
- `d_col`은 미래 nominal MF 활성화(`phi > d_col`), hard PSDF guard 최소 거리, stage-0 정지(`phi < d_col`)에 공통 사용한다.
- `SQP_RTI`, `dt=0.1 s`, horizon=20, 예측 시간 2 s, robust horizon=10.
- 시작 자세 교란: xy 표준편차 20 mm, heading 표준편차 0.05 rad, ±3σ 제한, 초기 clearance ≥1 mm. Localization error는 껐다.

| Robot width (m) | 목표 도달 | 최소 geometric clearance (mm) | Safe-stop step: seed 0 / 1 / 2 |
|---|---:|---:|---:|
| 0.115 | 1/3 | 0.138 | 20 / 20 / 0 |
| 0.120 | 1/3 | 0.000 | 20 / 20 / 0 |
| 0.125 | 0/3 | 0.000 | 20 / 20 / 20 |

총 9회 중 성공 2회, 실패 7회다. 0.125 m에서 모두 실패해 0.130 m는 생략했다.
실패는 실제 겹침 4건과, 겹침 없이 0.2 mm 기준을 밑돈 정지 3건으로 나뉜다.
위 geometric clearance는 장애물 겹침을 0으로 기록한다. 아래 **signed PSDF**는 음수로 겹침을 나타내므로 두 지표를 구분해야 한다.

## 3. 확인한 실패 메커니즘

### A. PSDF 기하 근사 오차: 실제 겹침 4건

해당 trial은 0.120 m seed 0/1, 0.125 m seed 0/1이다.

- 복원한 예측 자세에서 affine guard는 1.394–4.646 mm로 기준을 만족했다.
- 같은 예측 자세의 실제 signed PSDF는 −0.982–−0.191 mm로 이미 겹쳤다.
- 거리 과대평가는 2.376–4.836 mm였다. 4건 모두 nominal과 예측 자세의 최소 장애물이 달랐다.
- 장애물 branch 전환과 회전이 함께 있으므로, 오차 전체를 feature 전환 하나만의 영향으로 단정하지는 않는다.

예: **0.120 m seed 1**은 근사 guard가 **3.677 mm**였지만, 예측 자세의 실제 PSDF는 **−0.849 mm**, 실제 적분 후에는 **−1.036 mm**였다.

현재 hard guard는 nominal PSDF의 tangent를 강제한다.

```text
phi_hat(x) = phi(x_bar) + grad_phi(x_bar) · (x - x_bar)
phi_hat(x) >= d_col
```

이 tangent가 다른 장애물과 feature까지 포함한 실제 최소 PSDF의 안전한 하한이라는 보장은 없다.
**Hard라는 것은 이 근사식을 slack 없이 만족한다는 뜻이며, 실제 polygon separation 보장과는 다르다.**

### B. 동역학 선형화 오차: 비겹침 정지 3건

해당 trial은 0.115 m seed 0/1, 0.125 m seed 2다.

| Width (m) | Seed | 예측 자세의 실제 PSDF (mm) | 실제 적분 후 PSDF (mm) | 예측–실제 위치 차이 (mm) |
|---|---:|---:|---:|---:|
| 0.115 | 0 | 0.620 | 0.161 | 0.460 |
| 0.115 | 1 | 0.567 | 0.138 | 0.430 |
| 0.125 | 2 | 0.204 | 0.137 | 0.090 |

예측 자세는 기준 이상이었지만, 반환 입력을 현재 자세에 실제 RK4로 적용한 결과는 기준 미달이었다.
특히 0.125 m seed 2는 예측 여유부터 약 4 μm에 불과했다.

0.115 m seed 0/1에서는 한 0.1 s cycle에 각속도가 약 **−0.48 → +1.09 rad/s**로 급변했다.
현재 SQP-RTI가 한 번 푼 선형 동역학 equality와 실제 비선형 적분 사이의 차이가 기준 위반을 만들었다.
저장 입력을 기존 plant RK4로 다시 계산하면 실제 다음 자세가 오차 0으로 재현됐다.
별도의 plant 모델을 사용하거나 localization noise가 추가돼 생긴 차이는 아니다.

### C. 성공 trial도 여유가 작음

성공 seed 2의 최소 geometric clearance는 0.115 m에서 **0.355 mm**, 0.120 m에서 **0.232 mm**였다.
후자는 기준 대비 여유가 **32 μm**뿐이다.
성공한 0.120 m trial의 최소점에서도 예측–실제 위치 차이는 약 0.923 mm였다.
따라서 오차 크기 하나보다 **오차 방향, 당시 clearance, 기하 근사 오차의 조합**이 기준 위반 여부를 결정한다.

Seed별 초기 자세 차이는 이후 nominal 궤적과 회전·전진 입력을 다르게 만든다.
현재 구현은 미래 pose/input을 shift하거나 현재 자세에서 다시 rollout하지 않고 이전 solver 궤적을 유지한다.
특정 초기 교란 성분이 단독 원인이라는 대조 검증은 하지 않았으며, 폭당 3회로 일반적인 성공 확률을 판정할 수는 없다.

## 4. 개선 후보

아래 항목과 수치는 **아직 주행으로 검증하지 않은 제안**이다.

| 후보 | 기대 효과 | 한계·필요 작업 |
|---|---|---|
| 현재 자세에서 shifted-input nonlinear rollout | 현재 상태와 일관된 nominal을 만들어 선형화 기준 개선 | 초기 guess 갱신 코드 필요. 이것만으로 실제 안전을 보장하지 않음 |
| 동일 cycle 내 geometry·MF 재선형화 2–3회 | 새 후보의 장애물/feature와 covariance에서 제약을 다시 구성 | 같은 고정 QP를 반복하는 것과 다름. 계산 시간 증가 |
| 입력 변화량 제한 또는 trust region | 큰 입력·회전 보정이 만드는 근사 오차 억제 | 위치뿐 아니라 heading도 관리해야 함. 지나치게 좁으면 정체 가능 |
| 가까운 여러 장애물의 hard guard | 하나의 최소 장애물 tangent가 다른 장애물 위험을 놓치는 문제 완화 | 각 row의 회전·feature 근사 오차는 남음 |
| 입력 적용 전 exact RK4·PSDF 검사 | 위험 입력을 적용하기 전에 거부하고 다시 계산 | 검사만 추가하면 정지가 늘 수 있어 재선형화·복구 절차와 결합 필요 |

입력 변화량 제한의 시험 시작값은 `|Δomega| ≤ 0.3 rad/s/cycle`, `|Δv| ≤ 0.02 m/s/cycle` 정도를 고려할 수 있다.
이는 검증된 actuator 사양이나 최적 튜닝값이 아니며, 현재 독립 설정에는 없어 제약 추가가 필요하다.
Nominal 기준 trust region은 각 반복의 보정 크기를 제한하는 것이고, 직전 적용 입력 대비 변화량 제한은 시간에 따른 입력 변화를 제한하는 것이므로 구분해야 한다.

Exact 검사는 현재 자세와 반환 `u0`를 사용해 RK4로 다음 자세를 계산하고 full-geometry PSDF를 평가하는 방식이다.
기준 미달이면 적용 전에 거부하고 같은 cycle에서 재선형화해 다시 푼다.
Step 중간 시점도 검사하면 endpoint 사이의 충돌을 다룰 수 있지만, 유한 개 표본 검사만으로 연속시간 안전이 자동 보장되는 것은 아니다.

## 5. 설정만 바꿔 먼저 확인할 후보

### Full SQP로 동역학 영향 분리

```yaml
cc_mf_icdf:
  nlp_solver_type: SQP
  tol: 1.0e-6
```

현재 `SQP_RTI`, `tol=1e-4`에서 바꿔 비선형 동역학 불일치가 감소하는지 확인하는 비교안이다.
다른 설정은 유지하고, 성공률과 함께 solver 시간·예측–실제 차이를 측정해야 한다.
모드와 tolerance의 영향을 구분하려면 `RTI / tol=1e-4`, `SQP / tol=1e-4`, `SQP / tol=1e-6`를 단계적으로 비교한다.

**Full SQP 내부에서도 PSDF tangent와 MF 계수는 외부에서 전달한 값으로 고정된다.**
따라서 full SQP만으로 기하 근사 오차가 해결되는 것은 아니며, geometry를 새로 평가하는 외부 반복과 구분해야 한다.
이번 위험 step은 affine guard 자체를 이미 만족했으므로 tolerance만 낮추는 것도 핵심 해결책은 아니다.

### 제어 간격 감소

`dt=0.1 → 0.05 s`로 줄이고 2 s 예측 시간을 유지하려면 horizon을 `20 → 40`으로 늘리는 안이다.
현재 simulator는 plant dt와 `tf/horizon`의 일치를 검사하므로 **YAML의 horizon만 바꾸면 안 된다.**
Plant 주기와 관련 로깅도 함께 바꾸고, 기존 1 s covariance 전파 시간을 유지하려면 robust horizon도 `10 → 20`으로 맞춰야 한다.
계산량과 실제 제어 주기 내 처리 가능성을 함께 확인해야 한다.

### 단독 해결책으로 보기 어려운 설정

- `vmax`만 낮추기: 실패 직전 전진 속도는 대부분 0.002–0.088 m/s로 이미 낮았다. 회전만으로도 rectangle 모서리가 벽으로 들어갈 수 있다.
- `d_col`만 높이거나 낮추기: 요구 여유와 정지 시점을 바꾸지만 기하·동역학 근사 오차 자체를 없애지는 않는다.
- MF slack penalty 증가 또는 epsilon 감소: risk row 요구를 바꾸지만 이번 실패는 slack이 거의 0인 상황에서도 발생했다.

## 6. 권장 확인 순서와 평가 지표

1. 동일 seed·폭에서 full SQP와 기존 RTI를 비교해 동역학 오차 영향을 확인한다.
2. 현재 자세 기준 nonlinear rollout과 geometry 재선형화를 추가한다.
3. 입력 적용 전 exact 검사와 재계산 절차를 결합한다.
4. 남는 기하 오차에 대해 복수 장애물 guard와 trust region을 평가한다.

측정할 항목은 목표 도달률, 실제 겹침 trial 수, 최소 geometric clearance, safe-stop step/구간 수, 예측–실제 위치 차이, affine–실제 PSDF 차이, 전체 controller cycle 시간이다.
정지만 늘려 충돌 수가 감소한 경우와, 안전 여유를 유지하며 목표 도달한 경우를 구분해야 한다.

## 7. 분석 범위

예측 자세 전체가 직접 저장돼 있지 않아, 저장된 입력과 stage-0 고정 상태를 이용해 한 번의 QP 동역학 equality로 stage-1 pose를 복원했다.
반환 covariance CSV와 nominal MF residual/probability를 대조했으며, covariance-entry 최대 차이는 약 `6.6e-14`였다.
이는 복원의 일관성을 뒷받침하는 수치이며 pose 오차 상한은 아니다.
실제 적분 후 자세와 PSDF는 저장 기록에 대조했다.

이 문서는 기존 0.2 mm 기준 실험과 오프라인 복원에 근거한다. 문서 작성 과정에서 controller 수정이나 새 주행·solver 실행은 하지 않았다.
원본 실행 식별자는 `cc_mf_icdf_width_sweep_dcol02mm_20261028_163235`다.
