# Rectangle 폭별 controller benchmark — Deadlock tuning 적용 후

<!-- deadlock-width:rectangle_width070_130mm_deadlock_10trials_20261030_0116:start -->

2026-10-30(KST)에 수행한 Deadlock tuning 적용 후 실험을 정리한다. 폭 7종과 controller 4종을 조합별 10회씩 실행했으며, 총 **280/280 trials**를 완료했다.

## 1. 실험 조건

### 1.1. 기본 구성

| 항목 | 조건 |
|---|---|
| 실행일 | 2026-10-30 (KST) |
| 환경 / 동역학 / Global planner | maze / differential drive (DD) / A* |
| Robot footprint | Rectangle, 길이 0.150m 고정 |
| Robot 폭 | 0.070 / 0.080 / 0.090 / 0.100 / 0.110 / 0.120 / 0.130m |
| Controller | MPCC, CC-SF, CC-ECS, CC-MF |
| 반복 횟수 | 폭 × controller 조합별 10회, seed 0–9, 총 280회 |
| 실행 방식 | 같은 PC에서 순차 실행 |
| 제어 주기 / Prediction horizon | dt=0.1s / 20 step |
| Trial 제한 | 시뮬레이션 시간 60s |
| Localization error | 비활성화 |
| 실행 환경 | Python 3.10.13, PyTorch 2.4.1, CasADi 3.7.0, NVIDIA GeForce RTX 3060, Torch CPU thread 6개 |

### 1.2. 시작 자세

- x/y Gaussian 표준편차는 각각 0.02m, heading 표준편차는 0.05rad이며, 각 성분은 ±3σ로 제한했다.
- 같은 폭·seed에서는 네 controller에 동일한 시작 자세를 사용했다.
- 초기 clearance는 **폭 0.070–0.120m에서 ≥10mm, 폭 0.130m에서만 ≥1mm**다.
- 폭 0.130m에서는 0.150m 시작 통로에서 10mm 여유와 무작위 자세 교란을 동시에 만족할 수 없어 사용자 지정으로 기준을 완화했다.

### 1.3. Controller 설정

Deadlock tuning을 적용한 MPCC 공통 기본값을 CC-SF·CC-ECS·CC-MF가 상속했다.

| 공통 파라미터 | 값 |
|---|---:|
| vmin / vmax | -1.0 / 1.0m/s |
| v_s_target | 1.0m/s |
| omegamin / omegamax | -1.0 / 1.0rad/s |
| a_max | 0.5m/s² |
| alpha_max | 5.0rad/s² |
| r_delta_v / r_delta_omega | 0.05 / 0.05 |

초기 불확실성과 안전 거리는 실험 시점의 controller 기본값을 사용했다.

| Controller | σf (m) | σl (m) | σψ (rad) | Covariance 설정 | d_min (mm) |
|---|---:|---:|---:|---|---:|
| MPCC | — | — | — | 불확실성 모델 없음 | 1.0 |
| CC-SF | 0.0002 | 0.00025 | 0.000025 | covariance_growth_scale=0.8 | 1.0 |
| CC-ECS | 0.0002 | 0.00025 | 0.00025 | process noise=0 | 1.0 |
| CC-MF | 0.0002 | 0.00025 | 0.000025 | covariance_growth_scale=0.8 | 0.2 |

전체 파라미터는 [실행 원본·설정](benchmark_runs/rectangle_width070_130mm_deadlock_10trials_20261030_0116/experiment.json)에 기록되어 있다.

### 1.4. 종료 기준

| 판정 | 기준 |
|---|---|
| 성공 | 목표 위치의 L1 오차 ≤ 0.02m |
| 이동 정체 실패 | 4초 동안 위치 변화 ≤ 0.03m |
| 연속 제어 실패 | Solver 실패 또는 unsuccessful control step이 연속 20회 발생 |
| Timeout | 시뮬레이션 시간 60s 이내 목표 미도달 |

접촉·겹침 자체는 별도의 실패 조건이 아니다. 따라서 성공률은 목표 도달 비율을 뜻한다.

## 2. 지표 정의

| 지표 | 정의 |
|---|---|
| Success Rate | 성공 trial / 완료 trial. Timeout·failure·error도 분모에 포함한다. |
| Mean Traversal time (s) | 성공 trial만의 평균 시뮬레이션 시간. 성공이 없으면 —로 표시한다. |
| Mean / P05 clearance (mm) | 실제 rectangle과 모든 장애물 사이의 정확한 polygon 거리 및 지도 경계 margin 중 최솟값의 평균 / 5백분위수 |
| Mean / P95 computation time (ms/step) | 한 제어 사이클의 wall-clock 시간 평균 / 95백분위수 |
| Safe stop (step) | 조합별 10 trials에서 zero-input fallback이 발동한 제어 step의 합계. 연속 정지도 매 step을 센다. |

**Clearance 집계:** 초기 자세와 매 적분 후 자세를 포함하며, 성공·실패 trial의 전체 표본을 합쳐 계산했다. 접촉·겹침은 0이고, 음수는 실제 footprint의 지도 경계 침범을 뜻한다.

**계산 시간 측정:** local reference 생성부터 controller 전처리·파라미터 갱신·solver·복구·호출 내부 로깅까지 포함했다. `time.perf_counter()`와 CUDA 동기화를 사용하고, 각 trial의 첫 사이클은 제외했다. 이후 실패·safe stop step은 모두 포함했다. Global path 생성·plant 적분·파일 및 animation 저장은 제외했다.

Clearance와 계산 시간의 평균·분위수는 전체 유효 표본을 합쳐 계산했으며, trial별 통계의 평균이 아니다. MPCC에는 zero-input fallback이 없어 Safe stop을 0으로 표시한다. 계측 누락으로 확인된 하한만 알 수 있으면 ≥를 붙인다.

## 3. 실험 결과

### 3.1. 성공률 요약

각 셀은 성공률과 성공 횟수/완료 횟수를 나타낸다.

| Robot 폭 (m) | MPCC | CC-SF | CC-ECS | CC-MF |
|---|---:|---:|---:|---:|
| 0.070 | 100% (10/10) | 100% (10/10) | 100% (10/10) | 100% (10/10) |
| 0.080 | 100% (10/10) | 100% (10/10) | 100% (10/10) | 100% (10/10) |
| 0.090 | 100% (10/10) | 100% (10/10) | 100% (10/10) | 100% (10/10) |
| 0.100 | 100% (10/10) | 100% (10/10) | 100% (10/10) | 100% (10/10) |
| 0.110 | 100% (10/10) | 100% (10/10) | 100% (10/10) | 100% (10/10) |
| 0.120 | 0% (0/10) | 30% (3/10) | 0% (0/10) | 90% (9/10) |
| 0.130 | 0% (0/10) | 0% (0/10) | 0% (0/10) | 0% (0/10) |

- **폭 0.070–0.110m:** 네 controller 모두 각 폭에서 10/10 성공했으며 Safe stop은 0회였다.
- **폭 0.120m:** CC-MF 90%(9/10), CC-SF 30%(3/10), MPCC·CC-ECS 0%(0/10)였다.
- **폭 0.130m:** 네 controller 모두 0/10이었다. 이 폭은 초기 clearance 기준을 1mm로 완화한 조건이다.

![Robot 폭별 목표 도달 성공률](benchmark_runs/rectangle_width070_130mm_deadlock_10trials_20261030_0116/success_rate.png)

### 3.2. 폭별 상세 결과

계산 시간은 제어 사이클 기준이며, 주행 시간은 성공 trial만의 평균이다. 실패 상세와 계측 누락은 4절에 정리했다.

#### Robot 폭 0.070m

| Controller | 성공률 | 평균 clearance (mm) | P05 clearance (mm) | 평균 계산 시간 (ms/step) | P95 계산 시간 (ms/step) | 평균 주행 시간 (s) | Safe stop (step) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPCC | 100% (10/10) | 8.896 | 1.594 | 27.305 | 35.805 | 22.94 | 0 |
| CC-SF | 100% (10/10) | 11.926 | 3.108 | 32.453 | 43.944 | 24.43 | 0 |
| CC-ECS | 100% (10/10) | 9.852 | 2.406 | 30.366 | 37.590 | 24.38 | 0 |
| CC-MF | 100% (10/10) | 11.483 | 2.410 | 32.472 | 45.393 | 26.34 | 0 |

#### Robot 폭 0.080m

| Controller | 성공률 | 평균 clearance (mm) | P05 clearance (mm) | 평균 계산 시간 (ms/step) | P95 계산 시간 (ms/step) | 평균 주행 시간 (s) | Safe stop (step) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPCC | 100% (10/10) | 8.930 | 1.630 | 27.056 | 34.318 | 23.92 | 0 |
| CC-SF | 100% (10/10) | 11.648 | 2.892 | 31.566 | 42.139 | 25.74 | 0 |
| CC-ECS | 100% (10/10) | 9.791 | 2.395 | 30.176 | 35.938 | 25.65 | 0 |
| CC-MF | 100% (10/10) | 10.938 | 2.238 | 31.713 | 41.277 | 28.79 | 0 |

#### Robot 폭 0.090m

| Controller | 성공률 | 평균 clearance (mm) | P05 clearance (mm) | 평균 계산 시간 (ms/step) | P95 계산 시간 (ms/step) | 평균 주행 시간 (s) | Safe stop (step) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPCC | 100% (10/10) | 8.588 | 1.479 | 28.988 | 44.807 | 25.50 | 0 |
| CC-SF | 100% (10/10) | 10.724 | 2.628 | 30.993 | 40.887 | 27.44 | 0 |
| CC-ECS | 100% (10/10) | 9.359 | 2.356 | 31.117 | 41.466 | 27.27 | 0 |
| CC-MF | 100% (10/10) | 9.974 | 2.018 | 34.992 | 56.929 | 31.82 | 0 |

#### Robot 폭 0.100m

| Controller | 성공률 | 평균 clearance (mm) | P05 clearance (mm) | 평균 계산 시간 (ms/step) | P95 계산 시간 (ms/step) | 평균 주행 시간 (s) | Safe stop (step) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPCC | 100% (10/10) | 7.430 | 1.321 | 27.296 | 36.293 | 26.55 | 0 |
| CC-SF | 100% (10/10) | 8.529 | 2.469 | 31.093 | 40.820 | 29.98 | 0 |
| CC-ECS | 100% (10/10) | 8.170 | 2.056 | 30.126 | 35.184 | 29.18 | 0 |
| CC-MF | 100% (10/10) | 7.948 | 1.812 | 32.842 | 43.251 | 34.45 | 0 |

#### Robot 폭 0.110m

| Controller | 성공률 | 평균 clearance (mm) | P05 clearance (mm) | 평균 계산 시간 (ms/step) | P95 계산 시간 (ms/step) | 평균 주행 시간 (s) | Safe stop (step) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPCC | 100% (10/10) | 5.058 | 1.134 | 27.548 | 36.867 | 29.18 | 0 |
| CC-SF | 100% (10/10) | 6.303 | 2.310 | 31.286 | 41.060 | 33.93 | 0 |
| CC-ECS | 100% (10/10) | 5.390 | 1.983 | 30.887 | 37.979 | 37.27 | 0 |
| CC-MF | 100% (10/10) | 6.205 | 1.630 | 32.791 | 42.589 | 39.55 | 0 |

#### Robot 폭 0.120m

| Controller | 성공률 | 평균 clearance (mm) | P05 clearance (mm) | 평균 계산 시간 (ms/step) | P95 계산 시간 (ms/step) | 평균 주행 시간 (s) | Safe stop (step) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPCC | 0% (0/10) | -1.524 | -47.121 | 27.498 | 37.655 | — | 0 |
| CC-SF | 30% (3/10) | 4.868 | 0.692 | 38.093 | 131.626 | 40.87 | ≥124 |
| CC-ECS | 0% (0/10) | 4.245 | 1.897 | 29.454 | 35.764 | — | 0 |
| CC-MF | 90% (9/10) | 4.865 | 1.246 | 32.584 | 42.602 | 41.79 | 2 |

CC-SF의 clearance·계산 시간은 계측이 저장된 8 trials로 집계했고, Safe stop은 확인된 하한이다. MPCC의 음수 clearance는 지도 경계 침범을 포함한 값이다.

#### Robot 폭 0.130m

| Controller | 성공률 | 평균 clearance (mm) | P05 clearance (mm) | 평균 계산 시간 (ms/step) | P95 계산 시간 (ms/step) | 평균 주행 시간 (s) | Safe stop (step) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPCC | 0% (0/10) | 0.954 | 0.000 | 26.502 | 32.470 | — | 0 |
| CC-SF | 0% (0/10) | 2.564 | 0.083 | 77.578 | 149.598 | — | 179 |
| CC-ECS | 0% (0/10) | 2.780 | 1.796 | 29.836 | 37.017 | — | 0 |
| CC-MF | 0% (0/10) | 1.601 | -1.966 | 32.297 | 44.271 | — | 177 |

CC-MF의 음수 P05 clearance는 지도 경계 침범을 포함한 값이다.

## 4. 실패 상세 및 계측 누락

### 4.1. 실패 사유별 집계

폭 0.070–0.110m에서는 실패가 없었다. 폭 0.120m의 실패는 28건, 폭 0.130m의 실패는 40건이다. 아래 수치는 실패 trial 수이며, Safe stop step 수와는 다르다.

| Robot 폭 (m) | Controller | Solver 연속 실패 | 이동 정체 | Stage-0 safe stop 연속 발생 | Native crash | 합계 |
|---|---|---:|---:|---:|---:|---:|
| 0.120 | MPCC | 9 | 1 | 0 | 0 | 10 |
| 0.120 | CC-SF | 5 | 0 | 0 | 2 | 7 |
| 0.120 | CC-ECS | 0 | 10 | 0 | 0 | 10 |
| 0.120 | CC-MF | 0 | 1 | 0 | 0 | 1 |
| 0.130 | MPCC | 10 | 0 | 0 | 0 | 10 |
| 0.130 | CC-SF | 8 | 2 | 0 | 0 | 10 |
| 0.130 | CC-ECS | 0 | 10 | 0 | 0 | 10 |
| 0.130 | CC-MF | 0 | 2 | 8 | 0 | 10 |

### 4.2. Trial별 실패 기록

Trial 번호는 seed+1이다. 종료 시간은 시뮬레이션 시간이며, —는 저장되지 않은 값을 뜻한다.

<details>
<summary>Robot 폭 0.120m — 실패 28건</summary>

| Controller | Trial (seed) | 종료 시간 (s) | 실패 사유 |
|---|---:|---:|---|
| MPCC | 1 (0) | 9.9 | Solver 연속 20회 실패 |
| MPCC | 2 (1) | 9.9 | Solver 연속 20회 실패 |
| MPCC | 3 (2) | 10.0 | Solver 연속 20회 실패 |
| MPCC | 4 (3) | 14.0 | 4초간 위치 변화 ≤ 0.03m (실측 0.0287m) |
| MPCC | 5 (4) | 8.2 | Solver 연속 20회 실패 |
| MPCC | 6 (5) | 10.0 | Solver 연속 20회 실패 |
| MPCC | 7 (6) | 9.9 | Solver 연속 20회 실패 |
| MPCC | 8 (7) | 8.2 | Solver 연속 20회 실패 |
| MPCC | 9 (8) | 9.9 | Solver 연속 20회 실패 |
| MPCC | 10 (9) | 9.9 | Solver 연속 20회 실패 |
| CC-SF | 1 (0) | 14.2 | Solver 연속 20회 실패 |
| CC-SF | 2 (1) | — | Native crash (SIGSEGV, exit -11); step 계측 저장 전 종료 |
| CC-SF | 4 (3) | 17.6 | Solver 연속 20회 실패 |
| CC-SF | 5 (4) | 14.3 | Solver 연속 20회 실패 |
| CC-SF | 6 (5) | 14.7 | Solver 연속 20회 실패 |
| CC-SF | 7 (6) | — | Native crash (SIGSEGV, exit -11); step 계측 저장 전 종료 |
| CC-SF | 8 (7) | 17.0 | Solver 연속 20회 실패 |
| CC-ECS | 1 (0) | 11.1 | 4초간 위치 변화 ≤ 0.03m (실측 0.0299m) |
| CC-ECS | 2 (1) | 11.0 | 4초간 위치 변화 ≤ 0.03m (실측 0.0287m) |
| CC-ECS | 3 (2) | 11.3 | 4초간 위치 변화 ≤ 0.03m (실측 0.0286m) |
| CC-ECS | 4 (3) | 11.1 | 4초간 위치 변화 ≤ 0.03m (실측 0.0299m) |
| CC-ECS | 5 (4) | 11.2 | 4초간 위치 변화 ≤ 0.03m (실측 0.0289m) |
| CC-ECS | 6 (5) | 11.1 | 4초간 위치 변화 ≤ 0.03m (실측 0.0287m) |
| CC-ECS | 7 (6) | 11.0 | 4초간 위치 변화 ≤ 0.03m (실측 0.0287m) |
| CC-ECS | 8 (7) | 11.3 | 4초간 위치 변화 ≤ 0.03m (실측 0.0289m) |
| CC-ECS | 9 (8) | 11.2 | 4초간 위치 변화 ≤ 0.03m (실측 0.0286m) |
| CC-ECS | 10 (9) | 21.1 | 4초간 위치 변화 ≤ 0.03m (실측 0.0290m) |
| CC-MF | 10 (9) | 12.0 | 4초간 위치 변화 ≤ 0.03m (실측 0.0289m) |

</details>

<details>
<summary>Robot 폭 0.130m — 실패 40건</summary>

| Controller | Trial (seed) | 종료 시간 (s) | 실패 사유 |
|---|---:|---:|---|
| MPCC | 1 (0) | 10.9 | Solver 연속 20회 실패 |
| MPCC | 2 (1) | 10.9 | Solver 연속 20회 실패 |
| MPCC | 3 (2) | 10.6 | Solver 연속 20회 실패 |
| MPCC | 4 (3) | 10.9 | Solver 연속 20회 실패 |
| MPCC | 5 (4) | 4.0 | Solver 연속 20회 실패 |
| MPCC | 6 (5) | 11.2 | Solver 연속 20회 실패 |
| MPCC | 7 (6) | 10.9 | Solver 연속 20회 실패 |
| MPCC | 8 (7) | 10.8 | Solver 연속 20회 실패 |
| MPCC | 9 (8) | 5.2 | Solver 연속 20회 실패 |
| MPCC | 10 (9) | 12.8 | Solver 연속 20회 실패 |
| CC-SF | 1 (0) | 5.5 | 4초간 위치 변화 ≤ 0.03m (실측 0.0270m) |
| CC-SF | 2 (1) | 2.4 | Solver 연속 20회 실패 |
| CC-SF | 3 (2) | 6.4 | 4초간 위치 변화 ≤ 0.03m (실측 0.0290m) |
| CC-SF | 4 (3) | 2.2 | Solver 연속 20회 실패 |
| CC-SF | 5 (4) | 4.8 | Solver 연속 20회 실패 |
| CC-SF | 6 (5) | 2.0 | Solver 연속 20회 실패 |
| CC-SF | 7 (6) | 5.0 | Solver 연속 20회 실패 |
| CC-SF | 8 (7) | 4.9 | Solver 연속 20회 실패 |
| CC-SF | 9 (8) | 2.6 | Solver 연속 20회 실패 |
| CC-SF | 10 (9) | 4.9 | Solver 연속 20회 실패 |
| CC-ECS | 1 (0) | 7.7 | 4초간 위치 변화 ≤ 0.03m (실측 0.0295m) |
| CC-ECS | 2 (1) | 8.0 | 4초간 위치 변화 ≤ 0.03m (실측 0.0289m) |
| CC-ECS | 3 (2) | 8.0 | 4초간 위치 변화 ≤ 0.03m (실측 0.0285m) |
| CC-ECS | 4 (3) | 7.4 | 4초간 위치 변화 ≤ 0.03m (실측 0.0291m) |
| CC-ECS | 5 (4) | 7.7 | 4초간 위치 변화 ≤ 0.03m (실측 0.0284m) |
| CC-ECS | 6 (5) | 8.4 | 4초간 위치 변화 ≤ 0.03m (실측 0.0290m) |
| CC-ECS | 7 (6) | 7.9 | 4초간 위치 변화 ≤ 0.03m (실측 0.0284m) |
| CC-ECS | 8 (7) | 7.8 | 4초간 위치 변화 ≤ 0.03m (실측 0.0290m) |
| CC-ECS | 9 (8) | 7.0 | 4초간 위치 변화 ≤ 0.03m (실측 0.0295m) |
| CC-ECS | 10 (9) | 7.4 | 4초간 위치 변화 ≤ 0.03m (실측 0.0281m) |
| CC-MF | 1 (0) | 4.8 | Stage-0 safe stop 20 step 연속 |
| CC-MF | 2 (1) | 5.6 | Stage-0 safe stop 20 step 연속 |
| CC-MF | 3 (2) | 6.3 | Stage-0 safe stop 20 step 연속 |
| CC-MF | 4 (3) | 17.9 | 4초간 위치 변화 ≤ 0.03m (실측 0.0270m) |
| CC-MF | 5 (4) | 5.1 | Stage-0 safe stop 20 step 연속 |
| CC-MF | 6 (5) | 6.4 | Stage-0 safe stop 20 step 연속 |
| CC-MF | 7 (6) | 6.9 | Stage-0 safe stop 20 step 연속 |
| CC-MF | 8 (7) | 8.0 | 4초간 위치 변화 ≤ 0.03m (실측 0.0294m) |
| CC-MF | 9 (8) | 7.8 | Stage-0 safe stop 20 step 연속 |
| CC-MF | 10 (9) | 4.9 | Stage-0 safe stop 20 step 연속 |

</details>

### 4.3. 계측 누락

폭 0.120m의 CC-SF에서 **Trial 2(seed 1), Trial 7(seed 6)**이 Native crash(SIGSEGV, exit -11)로 종료했다.

- 두 trial은 성공률 분모에 포함했다.
- Clearance·계산 시간·종료 시각은 저장되지 않아 해당 지표 집계에서 제외했다.
- Safe stop은 확인된 하한인 ≥124 step으로 표시했다.
- 이 2건은 [검증 기록](benchmark_runs/rectangle_width070_130mm_deadlock_10trials_20261030_0116/report_integrity.json)에 명시되어 있다.

## 5. 원본 자료 및 Animation

### 5.1. 원본 자료

| 자료 | 내용 |
|---|---|
| [실행 원본·설정](benchmark_runs/rectangle_width070_130mm_deadlock_10trials_20261030_0116/experiment.json) | 실행 환경, controller 파라미터, 실험 구성 |
| [Trial 결과](benchmark_runs/rectangle_width070_130mm_deadlock_10trials_20261030_0116/trial_results.csv) | Trial별 성공 여부, 실패 사유, seed, 시작 자세 |
| [폭별 집계](benchmark_runs/rectangle_width070_130mm_deadlock_10trials_20261030_0116/benchmark_summary.csv) | 폭·controller 조합 28개에 대한 지표 |
| [Animation 목록](benchmark_runs/rectangle_width070_130mm_deadlock_10trials_20261030_0116/animation_index.md) | 원본 영상과 부분 checkpoint/replay 영상 구분 |
| [검증 기록](benchmark_runs/rectangle_width070_130mm_deadlock_10trials_20261030_0116/report_integrity.json) | 완료 횟수, 시작 자세 일치 여부, 계측 누락 |

### 5.2. Animation 저장 및 복구

기존 `animate_world()`/`FFMpegWriter`로 `animations/<controller>/`에 저장했다. 설정은 10fps, 100 DPI, libx264 ultrafast/CRF28이며 별도 재인코딩은 하지 않았다. 처음 4개 trial은 모든 frame, 이후는 `test_nmpc.py` 기본값인 frame_skip=5를 사용했다.

저장 영상 280개의 구성은 다음과 같다.

| 영상 구분 | 개수 |
|---|---:|
| 원래 trial 종료까지 저장된 영상 | 278 |
| Native crash trial의 원본 checkpoint 부분 영상 | 1 |
| 동일 조건·seed로 별도 재실행한 부분 영상 | 1 |

재실행은 280회 실험 집계에 추가하지 않았다. 재실행 영상은 원래 trial에서 유실된 궤적을 증명하지 않으며, Animation 목록에 partial checkpoint/replay를 구분했다.

<!-- deadlock-width:rectangle_width070_130mm_deadlock_10trials_20261030_0116:end -->
