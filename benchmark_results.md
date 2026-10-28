# Rectangle width별 controller benchmark 결과

작성일: 2026-10-27 (KST). 기존 실험 로그의 집계에 동일 조건으로 새로 실행한 CC-MF-ICDF 결과를 추가했다. 기존 controller의 수치는 유지했다.

## 실험 조건

- 환경: maze, 동역학: differential drive (DD), global planner: A*.
- Rectangle 길이: 0.15 m. 폭: 0.10 m / 0.08 m.
- Controller: MPCC, CC-SF, CC-ECS, CC-MF-IMP, CC-MF-ICDF. 각 조합 5 trials, seed 0–4.
- 시작 자세 교란: x/y 표준편차 0.02 m, heading 표준편차 0.05 rad, 각 성분 ±3σ 제한, 초기 clearance ≥ 0.01 m.
- Trial당 시뮬레이션 제한 60초, dt = 0.1초. Localization error 비활성화.
- 실패 기준: 4초 동안 이동량 0.03 m 미만 또는 solver 연속 실패 20회.
- 10개 조합 모두 목표 도달 5/5 (100%). CC-MF-IMP와 CC-MF-ICDF의 slack penalty는 linear=100, quadratic=1을 유지했다. Slack의 척도는 각각 거리와 standard-normal quantile이므로 같은 숫자가 동일한 목적함수를 뜻하지는 않는다.
- CC-MF-ICDF는 기존 실험과 같은 conda 환경(Python 3.10.13, PyTorch 2.4.1, CasADi 3.7.0), RTX 3060, PyTorch CPU thread 6개로 순차 실행했다. 두 폭 모두 기존 실험과 geometry 및 seed별 시작 자세가 일치함을 확인했다.

## 주요 결과: clearance · solver computation time · traversal time

Computation time은 성공한 control step의 acados `solve()` 내부 호출 시간이다. 단위는 clearance **mm**, computation time **ms/step**, traversal time **시뮬레이션 초(s)**이다.

### Rectangle width 0.10 m

| Controller | Mean clearance (mm) | P05 clearance (mm) | Mean computation time (ms/step) | P95 computation time (ms/step) | Mean traversal time (s) |
|---|---:|---:|---:|---:|---:|
| MPCC | 7.869 | 1.340 | 1.412 | 0.676 | 26.16 |
| CC-SF | 8.352 | 2.362 | 0.709 | 0.846 | 28.94 |
| CC-ECS | 8.464 | 2.047 | 1.151 | 1.389 | 28.76 |
| CC-MF-IMP | 8.158 | 2.238 | 0.912 | 1.097 | 31.14 |
| CC-MF-ICDF | 7.991 | 2.176 | 0.849 | 1.073 | 31.18 |

### Rectangle width 0.08 m

| Controller | Mean clearance (mm) | P05 clearance (mm) | Mean computation time (ms/step) | P95 computation time (ms/step) | Mean traversal time (s) |
|---|---:|---:|---:|---:|---:|
| MPCC | 8.749 | 1.638 | 0.614 | 0.937 | 24.18 |
| CC-SF | 10.861 | 2.714 | 0.670 | 0.836 | 25.68 |
| CC-ECS | 9.669 | 2.422 | 1.115 | 1.359 | 25.46 |
| CC-MF-IMP | 10.740 | 2.598 | 0.913 | 1.122 | 27.10 |
| CC-MF-ICDF | 10.822 | 2.603 | 0.852 | 1.141 | 27.48 |

## 집계 기준과 해석

- **Clearance**: rectangle footprint와 장애물 사이의 정확한 polygon 거리 및 지도 경계까지의 margin 중 최솟값. 초기 자세와 모든 적분 후 자세(최종 자세 포함)를 사용하고, 5 trials의 전체 샘플을 합쳐 mean/p05를 계산했다. Trial별 평균이나 p05를 다시 평균한 값이 아니다. P05는 하위 5% 분위수이며 선형 보간을 사용했다. 샘플 사이의 연속 궤적에 대한 충돌 검사는 포함하지 않는다.
- **Solver computation time**: `time.perf_counter()`로 측정한 `solve()` 호출 시간. 첫 step을 포함한 성공 step 전체를 합쳐 mean/p95를 계산했다. 복구 solver가 호출되면 해당 step의 호출 시간을 합산한다. Solver setup/코드 생성/컴파일, PSDF 및 inverse-CDF/implicit-root 전처리, 파라미터 갱신, 경로 계획, 출력 시간은 제외한다.
- **Traversal time**: 기존 목표 도달 판정까지의 시뮬레이션 시간이며, 성공한 5 trials의 산술평균이다. 실제 실행에 걸린 wall-clock 시간과 다르다.
- **폭 0.10 m MPCC의 mean > p95**: 첫 step의 solver 호출 약 1,113.927 ms가 평균에 반영됐다. 해당 이상치를 제거하지 않았으며, 원본 1,308 solver step을 그대로 집계한 값이다.
- **폭 간 시작 자세 차이**: 같은 폭에서는 다섯 controller가 seed별 동일한 시작 자세를 사용했다. 폭 0.08 m에서는 seed 4의 첫 후보 자세가 초기 clearance 조건을 통과해, 폭 0.10 m에서 재샘플링한 자세와 다르다. 폭 간 비교는 이 차이를 포함한다.
- **CC-MF-ICDF 검증 범위**: 폭 0.10 m / 0.08 m의 clearance 표본은 각각 1,564 / 1,379개, 성공 solver step은 1,559 / 1,374개다. 최소 clearance는 각각 1.588 / 2.164 mm이며 저장된 자세에서 접촉·경계 침범과 safe stop은 없었다. 환경당 5회 결과이며 연속시간 안전성이나 통계적 우위를 입증하는 결과는 아니다.

## 참고: controller 호출 시간 (첫 step 제외)

위 주요 표의 solver 시간에는 PSDF/inverse-CDF/implicit-root 전처리 비용이 들어가지 않는다. 아래 표는 로깅된 **controller 호출 전체 시간**을 사용한다. 호출 내부의 전처리, 파라미터 갱신, solver 및 복구 처리를 포함하고, global/local 경로 계획은 제외한다. 초기화·코드 생성·컴파일의 영향을 줄이기 위해 **각 trial의 첫 호출만 제외**하고 나머지 step을 합쳐 집계했다. 따라서 주요 표의 solver 시간과 표본 범위도 다르다.

### Rectangle width 0.10 m

| Controller | Mean controller time (ms/step) | P95 controller time (ms/step) | Steps |
|---|---:|---:|---:|
| MPCC | 36.442 | 54.744 | 1303 |
| CC-SF | 41.118 | 59.433 | 1442 |
| CC-ECS | 41.832 | 55.677 | 1433 |
| CC-MF-IMP | 84.504 | 121.337 | 1552 |
| CC-MF-ICDF | 44.325 | 61.100 | 1554 |

### Rectangle width 0.08 m

| Controller | Mean controller time (ms/step) | P95 controller time (ms/step) | Steps |
|---|---:|---:|---:|
| MPCC | 40.269 | 67.250 | 1204 |
| CC-SF | 42.051 | 61.390 | 1279 |
| CC-ECS | 40.534 | 55.914 | 1268 |
| CC-MF-IMP | 81.748 | 117.042 | 1350 |
| CC-MF-ICDF | 45.922 | 65.557 | 1369 |

## 원본 자료

- CC-MF-ICDF — 두 폭의 [전체 지표 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_widths_5trials_20261027_185126/benchmark_summary.csv), [trial별 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_widths_5trials_20261027_185126/benchmark_trials.csv), [집계·시작 자세 검증](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_widths_5trials_20261027_185126/validation.json). 폭 0.10 m의 [trial 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_widths_5trials_20261027_185126/frozen/width_010/trial_results.csv)·[solver 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_widths_5trials_20261027_185126/frozen/width_010/summary.csv), 폭 0.08 m의 [trial 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_widths_5trials_20261027_185126/frozen/width_008/trial_results.csv)·[solver 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_widths_5trials_20261027_185126/frozen/width_008/summary.csv). 2026-10-27 실행 자료를 사용했다.
- CC-MF-ICDF 측정에는 실험 시작 시 보관한 소스 사본을 사용했다. 작업 디렉터리의 공통 코드 변경이 감지된 최초 실행은 표에서 제외하고, 소스를 고정해 두 폭 모두 재실행했다. [공통 소스 비교](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_widths_5trials_20261027_185126/reference_source_comparison.json)와 [실행 중 소스 불변성](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_widths_5trials_20261027_185126/source_integrity.json)을 확인했다. 재현용 [실행 스크립트](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_widths_5trials_20261027_185126/run_benchmark.py)와 [집계 스크립트](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_widths_5trials_20261027_185126/analyze.py)를 보관했다.
- 폭 0.10 m — MPCC / CC-SF / CC-ECS: [clearance·traversal·시간 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/maze_rectangle_dd_perturb_four_methods_5trials_20261026_165452/benchmark_summary.csv), [solver 시간 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/maze_rectangle_dd_perturb_four_methods_5trials_20261026_165452/summary.csv), [trial 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/maze_rectangle_dd_perturb_four_methods_5trials_20261026_165452/trial_results.csv). 2026-10-26 실행 자료를 사용했다.
- 폭 0.10 m — CC-MF-IMP: [clearance 비교 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_imp_maze_rectangle_dd_5trials_20261027_180434/clearance_comparison.csv), [solver 시간 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_imp_maze_rectangle_dd_5trials_20261027_180434/baseline/summary.csv), [trial 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_imp_maze_rectangle_dd_5trials_20261027_180434/baseline/trial_results.csv). 2026-10-27 실행 자료를 사용했다.
- 폭 0.08 m — 네 controller: [전체 지표 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/maze_rectangle_w008_dd_four_methods_5trials_20261027_181526/benchmark_summary.csv), [solver 시간 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/maze_rectangle_w008_dd_four_methods_5trials_20261027_181526/summary.csv), [trial 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/maze_rectangle_w008_dd_four_methods_5trials_20261027_181526/trial_results.csv), [검증 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/maze_rectangle_w008_dd_four_methods_5trials_20261027_181526/validation.json). 2026-10-27 실행 자료를 사용했다.
- 폭 0.10 m CC-SF는 MPCC·CC-ECS와 함께 실행한 four-method batch의 시간 통계를 사용했다. 별도로 수행한 CC-SF safe-stop batch와 clearance/주행 시간은 같지만, 실행 시간 측정값은 다르므로 혼합하지 않았다.
- Controller 시간은 위 trial 결과가 가리키는 각 `runs/.../data/trial_history_*.csv`의 `controller_computation_time`에서 재계산했다. 모든 solver mean/p95는 해당 실험의 `computation_times_*.csv`와 원본 `summary.csv`에 일치함을 확인했다.

## CC-SF 폭 증가 실험 (2026-10-27)

사용자가 최종 지정한 여섯 폭에서 각 3회, 총 **18 trials**를 새로 실행했다. **목표 도달 16/18, safe-stop step 23회, 연속 safe-stop 구간 4개**였다. 0.120 m 이상에서는 장애물 겹침이 관측됐으므로, 목표 도달을 무충돌 성공으로 해석하면 안 된다. 한 폭에서 3회 모두 목표 도달에 실패한 경우는 없어 여섯 폭을 모두 실행했다.

### 실험 조건과 측정 정의

- 환경은 기존 maze이며 직선 corridor 폭 0.15 m, rectangle 길이 0.15 m를 유지했다. Robot width는 0.105 / 0.110 / 0.115 / 0.120 / 0.125 / 0.130 m이다. DD, A*, seed 0–2, dt=0.1 s, trial당 최대 60 s, localization error 비활성화 조건이다.
- 시작 자세 교란은 x/y 표준편차 0.02 m, heading 표준편차 0.05 rad, 각 성분 ±3σ 제한이다. **사용자 선택에 따라 초기 clearance 조건을 여섯 폭 모두 ≥ 0.001 m로 통일했다.** 0.130 m에서는 정렬된 중앙 자세의 양측 여유가 각 10 mm이므로 기존 ≥10 mm 조건과 연속 Gaussian 자세 교란을 함께 만족하는 후보를 생성하지 못한다. 최종 실험에서는 seed별 시작 자세가 여섯 폭에서 정확히 일치함을 검증했다.
- CC-SF는 기존 기본값 d_min=0.001 m, chance_epsilon=0.20, gamma=0.841621, horizon=20 (2 s), robust horizon=10을 유지했다. 초기 covariance는 매 cycle sigma_f0=0.0002 m, sigma_l0=0.00025 m, sigma_psi0=0.00025 rad로 설정된다. 전체 설정은 [optimizer parameters](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/width_105mm/optimizer_parameters.json)에 보관했다.
- 실패 기준은 solver 연속 실패 20회 또는 4 s 동안 이동량 ≤0.03 m이다. 목표 도달은 기존 판정이다. 실행 소스를 사본으로 고정하고 같은 Python 3.10.13 / PyTorch 2.4.1 / CasADi 3.7.0 / RTX 3060 환경, PyTorch CPU thread 6개에서 trial을 순차 실행했다. Animation/plot은 비활성화했다.
- **Clearance**는 기존 표와 동일하게 rectangle–장애물 polygon 거리와 지도 경계 margin의 최솟값이다. 초기 및 모든 적분 후 자세(최종 자세 포함)를 사용하며, 실패 trial도 포함한 전체 표본을 합쳐 mean/p05/min을 계산했다. 장애물 접촉·겹침의 polygon 거리는 0으로 집계되고, **음수는 지도 경계 침범**을 뜻한다. 따라서 음수의 크기를 일반적인 장애물 침투 깊이로 해석하면 안 된다. 장애물 겹침은 Shapely 교차 면적으로 별도 검증했다.
- **Safe stop 횟수는 zero-input을 적용한 control step 수**이다. 연속 20 step은 20회·2.0 s이며 연속 정지 구간은 1개이다. 아래 seed별 횟수는 seed 0 / 1 / 2 순서다. 기록된 자세 사이의 연속시간 충돌은 검사하지 않았다.
- 초기 clearance ≥10 mm로 수행하다 조건 변경 전에 중단한 예비 실험은 [별도 폴더](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/pilot_initial_clearance10mm)에 보존하고 최종 집계에서 제외했다. 비교용 0.100 m 3회는 아래 본 실험 18회에 포함하지 않는다.

### 폭별 결과

| Robot width (m) | 목표 도달 | Mean clearance (mm) | P05 (mm) | Min (mm) | Safe-stop step: seed 0 / 1 / 2 | 장애물 겹침 표본 |
|---|---:|---:|---:|---:|---:|---:|
| 0.105 | 3/3 | 7.425 | 2.240 | 1.863 | 0/0/0 | 0 |
| 0.110 | 2/3 | 6.040 | 1.832 | 0.822 | 20/0/0 | 0 |
| 0.115 | 3/3 | 5.150 | 1.987 | 0.117 | 0/0/0 | 0 |
| 0.120 | 3/3 | 4.334 | 1.680 | -0.251 | 0/0/0 | 13 |
| 0.125 | 3/3 | 3.857 | 0.754 | -3.704 | 0/1/0 | 37 |
| 0.130 | 2/3 | 2.290 | 0.000 | -11.802 | 0/1/1 | 178 |

비교용 **0.100 m**는 목표 도달 3/3, safe stop 0/0/0회, mean/p05/min clearance는 각각 8.571 / 2.362 / 1.961 mm였다. 이 기준 실행의 초기 clearance 조건은 ≥10 mm였으며, 실제 seed별 시작 자세는 최종 여섯 폭과 같았다. 상단의 과거 5-trial 표와는 표본 수가 다르다.

**0.120 / 0.125 / 0.130 m의 모든 trial에서 장애물 겹침이 있었다.** 지도 경계 침범 표본은 각각 1 / 8 / 38개이며 장애물 겹침 표본과 중복될 수 있다. 기록된 모든 자세에서 clearance>0을 유지하면서 목표에 도달한 trial은 폭별로 3 / 2 / 3 / 0 / 0 / 0회였다. 0.115 m도 최소 clearance가 0.117 mm로 d_min=1 mm보다 작아, safe stop이 없다는 사실만으로 실제 geometric margin 준수를 판단할 수 없다.

### Trial별 결과

| Width (m) | Seed | 종료 상태 | 종료 시간 (s) | Mean clearance (mm) | Min (mm) | Safe-stop step | 겹침 표본 |
|---|---:|---|---:|---:|---:|---:|---:|
| 0.105 | 0 | 목표 도달 | 30.0 | 7.483 | 1.866 | 0 | 0 |
| 0.105 | 1 | 목표 도달 | 30.2 | 7.357 | 1.866 | 0 | 0 |
| 0.105 | 2 | 목표 도달 | 30.0 | 7.435 | 1.863 | 0 | 0 |
| 0.110 | 0 | 실패 | 26.0 | 6.215 | 0.822 | 20 | 0 |
| 0.110 | 1 | 목표 도달 | 32.0 | 5.945 | 1.466 | 0 | 0 |
| 0.110 | 2 | 목표 도달 | 31.8 | 5.992 | 1.402 | 0 | 0 |
| 0.115 | 0 | 목표 도달 | 33.2 | 5.225 | 1.406 | 0 | 0 |
| 0.115 | 1 | 목표 도달 | 33.8 | 5.072 | 0.587 | 0 | 0 |
| 0.115 | 2 | 목표 도달 | 33.2 | 5.155 | 0.117 | 0 | 0 |
| 0.120 | 0 | 목표 도달 | 36.2 | 4.329 | 0.000 | 0 | 4 |
| 0.120 | 1 | 목표 도달 | 36.1 | 4.333 | -0.251 | 0 | 6 |
| 0.120 | 2 | 목표 도달 | 36.0 | 4.340 | 0.000 | 0 | 3 |
| 0.125 | 0 | 목표 도달 | 40.7 | 3.891 | -3.704 | 0 | 9 |
| 0.125 | 1 | 목표 도달 | 40.7 | 3.872 | -1.141 | 1 | 15 |
| 0.125 | 2 | 목표 도달 | 40.5 | 3.809 | -3.634 | 0 | 13 |
| 0.130 | 0 | 실패 | 17.0 | 2.080 | -2.881 | 0 | 41 |
| 0.130 | 1 | 목표 도달 | 47.9 | 2.328 | -4.584 | 1 | 72 |
| 0.130 | 2 | 목표 도달 | 44.3 | 2.330 | -11.802 | 1 | 65 |

### Safe stop과 실패 사유

- **0.110 m, seed 0**: simulation time 24.0–26.0 s에 safe stop 20 step이 연속 발생했고 `consecutive_solver_failures(20)`로 종료했다. 나머지 두 seed는 safe stop 없이 목표에 도달했다.
- **0.125 m, seed 1**: 10.6–10.7 s에 1 step 정지 후 재개해 목표에 도달했다.
- **0.130 m, seed 1 / 2**: 각각 6.7–6.8 s / 22.2–22.3 s에 1 step 정지 후 재개해 목표에 도달했다. **seed 0의 실패는 safe stop 없이** 17.0 s에 `stuck_for_4.00s_disp_0.0281m_le_0.0300m` 기준으로 종료된 것이다.
- Safe-stop step 23회 모두 주 SQP-RTI는 acados status=4 (**QP solver failure**), 복구 SQP_WITH_FEASIBLE_QP는 status=2 (**maximum iterations**)를 반환했다. 복구 solver의 성공 횟수는 0회였다. 이 반환 코드는 비선형 문제의 수학적 infeasibility를 증명하지 않는다. [정지 구간](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/safe_stop_events.csv), [상태·margin 전후 기록](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/failure_diagnostics.json)에 근거를 보관했다.

### Minimum-feature margin 가설과의 관계

[method_description.md §4.1 및 §5.5](/home/taek111/projects/psdf_mpcc/method_description.md:290)의 SF margin은 다음과 같다.

$$
r^{\mathrm{SF}}=\Phi^{-1}(1-\varepsilon)\sqrt{q^\top\tilde\Sigma q},\qquad q=T(\theta)^\top g,\qquad h^{\mathrm{SF}}=\phi-d_{\min}-r^{\mathrm{SF}}.
$$

최소 feature 전환으로 gradient가 바뀌면 projected variance와 margin이 불연속일 수 있다. 다만 전환 전후의 projected variance가 같으면 margin도 같으며, 특히 q가 -q로 바뀌는 경우 quadratic form은 변하지 않는다. 따라서 최소 feature 선택만으로 항상 margin 점프나 infeasibility가 생기는 것은 아니다.

이번 실험에서는 0.110 m에서 지속적인 solver 실패가 나타났지만 0.115 / 0.120 m에서는 safe stop이 없었고, 더 큰 폭의 정지 3건은 다음 step에 복구됐다. 실패 빈도는 폭에 대해 단조롭지 않았다. 0.110 m seed 0의 첫 정지 직전/시작 시점에서 현재 자세의 진단용 margin은 **0.210407 → 0.210406 mm**로 급증하지 않았다. 미래 stage의 feature 전환이나 RTI nominal geometry의 영향까지 배제하는 결과는 아니다.

생성된 실제 solver JSON은 **nh_0=0, nh=nh_e=1, slack=0**이었다. 즉 stage 0에는 obstacle row가 없고, 미래 stage 1–20에 hard SF row가 적용된다. 따라서 stage-0 residual이 음수라는 기록만으로 ‘고정 초기 상태 제약 위반 때문에 infeasible’이라고 해석할 수 없다. [주·복구 solver의 stage 정책](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/solver_stage_policy.json)을 확인했다. 저장된 자세에서의 겹침도 존재하므로, SQP-RTI가 반환한 성공 및 affine constraint 처리를 exact nonlinear clearance 보장과 동일시할 수 없다.

이 결과는 폭별 현상 측정이다. Feature ID의 전환 추적이나 risk margin을 끄는 대조 실험은 수행하지 않았으므로 **‘feature switch → margin 불연속 → solver 실패’의 인과관계는 이번 실험만으로 확정하지 않았다.**

### 원본 자료와 검증

- [폭별 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/benchmark_summary.csv), [trial별 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/benchmark_trials.csv), [모든 자세의 clearance](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/clearance_samples.csv).
- [겹침·경계 침범 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/geometry_violations_by_trial.csv), [위반 자세와 교차 면적](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/geometry_violations.csv), [nominal risk 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/nominal_risk_summary.csv). 각 폭 폴더에 `trial_results.csv`, `cycles.csv`, `nominal_risk.jsonl`, runtime JSON, 원본 history 및 실행 로그를 보관했다.
- [검증 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/validation.json): source 사본 불변성, 폭 외 geometry·optimizer parameter 동일성, seed별 시작 자세 동일성, 초기·최종 자세 포함, mean/p05 독립 재계산, safe-stop 누적수·step 로그·zero-input 및 solver CSV 대조를 통과했다. Shapely로 모든 비양수 clearance 표본을 포함한 총 945개 자세의 거리를 독립 검증했다(비교용 기준 실행 포함).
- 재현용 [순차 실행](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/run_sweep.py)·[단일 폭 실행·관측](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/run_benchmark.py)·[clearance 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/analyze.py)·[겹침 분류](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/classify_violations.py)·[정지 시점 분석](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/diagnose_stops.py)을 보관했다.

## CC-SF hard constraint와 실제 clearance 위반 원인 분석

**원인은 hard row의 비활성화나 slack이 아니라, hard row가 실제 polygon separation이 아닌 고정된 최소-feature Taylor 모델과 한 번의 SQP-RTI QP에 적용된다는 점이다.** 실제 입력을 적용하기 전 exact one-step geometry를 검사하는 단계가 없어, 그 QP가 성공하면 실제 겹침으로 이어지는 입력도 적용된다. 새 주행·solver 실행 없이 기존 0.120 / 0.125 / 0.130 m 로그의 첫 겹침 9건을 복원해 확인했다.

### 1. 실제로 enforce하는 hard constraint

[AnalyticPSDFCasADi.__call__](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/source_snapshot/control/analytic_psdf_casadi.py:38)은 exact PSDF 대신 다음 affine 식을 반환한다.

$$
\hat\phi_i(p)=\phi(\bar p_i)+\bar g_i^\top(p-\bar p_i).
$$

[presolve parameter 갱신](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/source_snapshot/control/cc_sf_optimizer.py:854)에서 nominal pose의 PSDF 값과 gradient를 한 번 계산해 `model.p`에 넣는다. [hard SF row](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/source_snapshot/control/cc_sf_optimizer.py:702)는 이 affine 값으로 `hat_phi - risk_margin - d_min >= 0`을 설정한다. Row에는 slack이 없고, 미래 stage 1–20에서 활성화된다. 다만 risk margin과 covariance dynamics는 비선형이므로 SQP-RTI가 실제 QP에서 적용하는 것은 다시 다음과 같은 1차 row다.

$$
\hat h_i^{\rm QP}=\hat\phi_i(p_i)-d_{\min}-\left[r_i(\bar\xi_i)+\nabla_\xi r_i(\bar\xi_i)^\top(\xi_i-\bar\xi_i)\right]\ge0.
$$

이 부등식에는 모든 장애물의 exact distance나 비선형 polygon overlap 판정이 들어 있지 않다. 최소 PSDF는 여러 feature/장애물 branch의 최솟값이므로, 한 branch에서 얻은 tangent가 다른 branch까지 포함한 PSDF의 안전한 하한인 것은 아니다. Nominal의 최소 장애물에서 멀어지거나 그 옆을 따라 움직이면서 다른 장애물로 들어가도 이 단일 row는 만족할 수 있다. 같은 장애물에서도 활성 vertex/edge가 바뀌거나 회전하면 같은 문제가 생긴다.

### 2. 0.120 m, seed 0: 6.3 → 6.4 s의 수치 증거

Nominal stage-1의 최소 장애물은 1번이었다. Solver가 반환한 자세와 실제 적분 후 자세는 4번 장애물과 겹친다. 전체 장애물 17개가 모두 wrapper에 들어 있으므로 obstacle capacity/ROI 누락 사례가 아니다.

| 항목 | 값 (mm) |
|---|---:|
| Nominal stage-1 PSDF | +3.988508 |
| 반환된 예측 자세의 Taylor PSDF | +3.766252 |
| 복원한 QP risk margin | +1.621063 |
| QP의 요구 clearance: d_min + margin | +2.621063 |
| 복원한 hard QP residual | **+1.145189** |
| 반환된 예측 자세에서 재평가한 exact signed PSDF | **−4.091114** |
| 실제 입력 RK4 적분 후 signed PSDF | **−4.495174** |

즉 solver가 hard QP row를 어긴 것이 아니라 **잘못 안전하다고 보는 근사 row를 만족한 것**이다. 같은 예측 자세에서 Taylor 모델이 signed PSDF를 7.857366 mm 과대평가한다. 실제 signed PSDF는 원래 PSDF 함수를 해당 자세에서 새로 평가한 값이며, Shapely 교차 면적으로도 겹침을 확인했다. 앞선 benchmark 표의 polygon clearance는 겹침을 0으로 기록하므로, 그 통계와 위 signed PSDF의 음수 크기를 직접 비교하면 안 된다.

![첫 겹침에서의 footprint와 Taylor/실제 signed PSDF 비교](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/clearance_diagnosis/first_overlap_geometry.png)

### 3. 9개 trial의 첫 겹침 대조

아래 QP residual은 로그의 nominal state와 applied input 및 선형화된 stage-0 equality로 복원했다. 반환 pose는 다음 cycle의 pre-solve stage-1 pose에서 회수했다. Covariance는 다음 cycle에 재생성되어 덮어쓰므로 반환 covariance를 직접 읽은 값은 아니며, QP의 선형 covariance transition으로 재구성한 값이다.

| Width (m) | Seed | 겹침 시작 (s) | 최소 장애물: nominal → 반환 | 반환 Taylor PSDF (mm) | 반환 exact PSDF (mm) | 복원 QP residual (mm) | 예측–실제 위치차 (mm) |
|---|---:|---:|---|---:|---:|---:|---:|
| 0.120 | 0 | 6.4 | 1 → 4 | 3.766 | -4.091 | 1.145 | 0.517 |
| 0.120 | 1 | 6.6 | 1 → 4 | 3.489 | -4.467 | 0.867 | 0.544 |
| 0.120 | 2 | 10.5 | 4 → 3 | 3.784 | -2.480 | 1.260 | 0.389 |
| 0.125 | 0 | 6.5 | 1 → 0 | 3.853 | 0.177 | 1.751 | 0.287 |
| 0.125 | 1 | 6.5 | 1 → 0 | 3.266 | -0.088 | 0.991 | 0.294 |
| 0.125 | 2 | 6.7 | 1 → 0 | 3.335 | -0.237 | 1.251 | 0.286 |
| 0.130 | 0 | 2.4 | 14 → 0 | 2.654 | -2.948 | 0.000 | 0.058 |
| 0.130 | 1 | 0.4 | 1 → 1 | 7.888 | -1.719 | 7.745 | 0.027 |
| 0.130 | 2 | 2.2 | 14 → 0 | 2.872 | -0.462 | 0.000 | 0.011 |

9건 모두 `fast_status=0`, `safe_stop=0`으로 겹침을 시작했고, 복원한 hard QP row는 모두 비음수였다. 반환된 예측 자세 9개 모두 exact PSDF가 d_min=1 mm보다 작았으며 8개는 이미 음수였다. 최소 장애물은 8/9건에서 바뀌었다. 같은 장애물 1번이 유지된 0.130 m seed 1도 Taylor PSDF 7.888 mm 대비 exact PSDF −1.719 mm로, 한 장애물 내의 geometry branch/회전 선형화 오차가 있다. 이 로그 분석만으로 내부 vertex/edge ID까지 직접 확정한 것은 아니다.

### 4. RTI 동역학 오차와 warm start가 추가로 작용한다

[plant RK4](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/source_snapshot/models/dd_dynamics.py:6)와 [OCP의 symbolic DD step](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/source_snapshot/control/cc_sf_optimizer.py:260)은 같은 RK4식을 사용한다. 저장된 실제 상태에 적용 입력을 다시 대입하면 실제 다음 상태와 부동소수점 값까지 일치했다. 따라서 별도의 plant 모델, localization error 또는 잘못된 입력 stage를 적용한 문제로 볼 근거가 없다.

하지만 SQP-RTI는 이 RK4의 선형화된 동역학을 한 번 풀기 때문에 `x1_returned`와 `f_RK4(x_current, u0_returned)`가 정확히 같을 필요는 없다. 첫 겹침들에서 위치차는 0.011–0.544 mm였고, geometry Taylor 과대평가 3.334–9.606 mm보다 작았다. 0.125 m seed 0은 반환 pose의 실제 PSDF가 +0.177 mm였으나, 실제 plant pose에서는 −0.075 mm가 되어 이 작은 동역학 오차가 접촉 여부까지 바꿨다. 반환 pose부터 이미 d_min=1 mm를 충족하지 못했다.

[초기 상태·guess 갱신](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/source_snapshot/control/cc_sf_optimizer.py:818)은 stage 0만 현재 자세로 바꾸고 stage 1 이후 pose/input을 한 칸 shift하거나 현재 상태에서 다시 rollout하지 않는다. [guess 재생성](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/source_snapshot/control/cc_sf_optimizer.py:539)은 covariance만 재계산한다. 이 구조에서는 stage-1 geometry anchor가 직전 예측 시각에 머물고, QP 출력이 anchor에서 크게 움직일 수 있다. 0.120 m seed 0은 stage-1 위치가 anchor에서 12.952 mm 이동했고, 0.125 m 세 seed는 heading이 약 0.12 rad 바뀌었다. 이를 제한하는 geometry trust region이나 같은 cycle 내 exact PSDF 재선형화 반복은 없다. 이는 큰 근사 오차를 허용하는 요인이지, shift만 하면 exact 안전성이 보장된다는 뜻은 아니다.

Risk/covariance도 별도 근사 오차가 있다. 0.130 m seed 1의 첫 겹침에서 복원한 **QP 선형 risk margin은 −0.857 mm**였다. 같은 반환 covariance에 비선형 margin 식을 적용하면 +0.144 mm, 반환 입력으로 covariance를 정확히 전파한 값에서는 +1.912 mm였다(geometry gradient는 해당 stage의 nominal 값으로 고정). 원래 margin의 비음수성이 그 1차 QP 근사까지 자동으로 유지되는 것은 아니다. 다만 9건 모두 geometry 오차만으로 d_min 위반을 확인할 수 있어, 이 추가 오차를 모든 겹침의 단독 원인으로 주장하지 않는다.

### 5. Safe stop이 발생하지 않는 직접적인 코드 경로

[복구 진입 조건](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/source_snapshot/control/cc_sf_optimizer.py:1021)은 `status != 0`일 때만 복구 solver와 safe stop으로 들어간다. `status == 0`인 해에 대해 exact PSDF, 비선형 dynamics defect 또는 실제 다음 자세의 충돌 여부로 거부하는 분기가 없다. `debug_mode=false`라 반환 궤적의 fresh-risk 진단도 생략된다. Debug를 켜도 그 분기는 값 계산/출력이며 안전 입력 검증 조건은 아니다.

[입력 적용](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/source_snapshot/control/controller.py:101)은 성공 상태를 기록한 뒤 반환된 stage-0 입력을 그대로 내보낸다. [simulation loop](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/source_snapshot/sim/simulation.py:156)는 이를 적분하고 목표/정체/연속 solver 실패를 확인한다. Exact collision을 trial 실패나 safe stop으로 바꾸는 검사가 없다. 이 때문에 겹침이 생겨도 계속 주행하고 목표 도달로 기록될 수 있다.

[설치된 acados SQP-RTI 소스](/home/taek111/projects/acados/acados/ocp_nlp/ocp_nlp_sqp_rti.c:621)도 QP 처리 후 status=0을 설정하며 exact nonlinear collision의 만족을 판정하지 않는다. 이 버전은 QP MAXITER도 같은 경로로 허용하지만 원본 로그에는 내부 qp_status가 없어, 이번 9건에 MAXITER가 실제 발생했다고 해석하지는 않았다. 이번 복원에서는 hard QP row 자체가 만족되는 사례를 확인했다.

생성 solver의 `nh_0=0`도 현재 pose의 충돌이 즉시 constraint 실패로 연결되지 않는 배경이다. 다만 stage-1 hard row가 실제로 있어도 첫 겹침을 막지 못한 사례이므로, stage-0 row 부재만으로 진입 원인을 설명할 수 없다. Stage-0 row 추가만으로 이전 입력이 만드는 충돌을 예방하지도 못한다.

### 6. 수정한다면 우선 다뤄야 할 지점

1. **입력 적용 전 exact one-step 검사**: 현재 실제/추정 자세와 반환 u0로 동일 RK4를 계산하고 full geometry의 PSDF를 재평가한다. d_col 미달이면 main status=0이어도 해당 입력을 거부해 재선형화·복구 경로로 보낸다. 이미 겹친 자세에서 zero input을 주는 것은 새 충돌 예방과 다른 문제이므로 별도 복원 정책이 필요하다. 연속시간 안전을 요구하면 step 사이의 swept trajectory도 검사해야 한다.
2. **기하 근사의 유효 범위 관리**: 현재 상태에서 shifted-input rollout으로 nominal을 만들고, 여러 가까운 장애물/feature의 hard row, trust region, exact geometry 재평가 후 재선형화를 함께 고려한다. 여러 row만 추가해도 회전/동역학 근사 오차는 남으므로 exact acceptance 검사를 대체하지 않는다.
3. **수렴/물리 일관성 검사**: SQP-RTI status와 함께 비선형 dynamics defect, fresh exact PSDF, covariance propagation consistency를 기록한다. 단순히 solver를 full SQP로 바꾸더라도 `model.p`의 고정 affine PSDF를 갱신하지 않으면 다른 장애물로 들어가는 문제는 남을 수 있다.

이번 요청에서는 controller 동작을 수정하지 않았다. 분석의 핵심은 minimum-feature 전환이 risk margin만 바꾸는 문제가 아니라, **hard row의 거리 모델 자체를 낙관적으로 만들 수 있음**을 로그에서 직접 확인했다는 점이다.

원본/재현 자료: [9건의 수치 대조](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/clearance_diagnosis/first_overlap_reconstruction.csv), [pose·gradient·input·covariance 상세](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/clearance_diagnosis/first_overlap_details.json), [재구성 검증](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/clearance_diagnosis/validation.json), [오프라인 분석 스크립트](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/analyze_constraint_violations.py). 원본 nominal PSDF/risk terms/margin을 재현했고, 선형 동역학으로 복원한 pose와 다음 cycle에 보존된 반환 pose의 최대 차이는 1.73e-14였다. QP covariance는 직접 로그가 아닌 위 선형 equality를 이용한 복원값이라는 한계가 있다.

## 8 mm의 차이를 같은 feature의 선형화 오차로 볼 수 있는가

추가 분해 결과, 0.120 m seed 0 사례에서 **선택된 장애물 1번 branch 자체의 매끄러운 선형화 오차는 약 0.015 mm**였다. 앞서 약 7.857 mm를 ‘geometry Taylor 오차’로 묶어 표현한 값의 대부분은 다른 장애물 branch가 최소가 된 영향이다. 큰 kinematics 곡률 오차나 PSDF gradient 구현 오류로 8 mm가 생겼다고 해석할 필요는 없다.

### 장애물별 거리로 분리한 결과

| 평가 위치 | 장애물 1번 signed distance (mm) | 장애물 4번 signed distance (mm) | 전체 최소 |
|---|---:|---:|---|
| Nominal anchor | +3.988483 | +4.019862 | 1번 |
| 반환된 QP 예측 자세 | +3.751371 | −4.091155 | 4번 |
| 실제 입력 적분 후 자세 | +4.267927 | −4.495209 | 4번 |

독립 float64 분석에서 장애물 1번의 tangent 예측은 +3.766251 mm였고, 같은 장애물의 실제 거리는 +3.751371 mm였다. 차이는 **0.014880 mm**다. 반면 빠진 장애물 4번의 tangent를 계산하면 **−4.091217 mm**, 그 장애물의 실제 거리는 −4.091155 mm였다. 4번의 거리도 자체 선형화는 정확했지만, 그 row가 QP에 들어가지 않았다. 이 추가 수치는 원본 실행의 float32 값을 재해석하기 위한 float64 geometry 감사 결과라 마지막 자릿수에서 차이가 있다.

즉 함수는 `phi(p)=min(d1(p),d4(p),...)`인데, QP가 anchor에서 선택한 `d1`의 tangent를 계속 사용했다. 실제 최소가 `d4`로 바뀐 이후 그 tangent는 전체 최소 거리의 근사로서 크게 낙관적이다. PSDF 값은 연속이어도 gradient가 branch 전환점에서 바뀌므로, 한 매끄러운 branch 안에서의 작은 2차 Taylor remainder와 다른 현상이다.

Anchor의 두 거리 차이는 **0.031379 mm**에 불과했다. Anchor→QP 해 선분을 따라 약 **0.051520 mm** 이동하면 최소 장애물이 바뀐다. 그런데 실제 QP 위치 수정량은 **12.952330 mm**, heading 수정량은 **-0.065142°**였다. 이 선분은 branch 변화 확인을 위한 pose 보간이며 실제 robot의 시간 궤적이라는 뜻은 아니다. 약 13 mm 위치 수정에 비해 약 4 mm의 여유는 작다.

![기존 최소 branch의 tangent와 실제 최소 거리의 분리](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/clearance_diagnosis/distance_branch_switch.png)

### 3.766 mm → −4.495 mm 차이의 분해

원본 float32 로그와 일치하는 값으로 다음과 같이 나뉜다.

| 단계 | Signed PSDF/근사값 (mm) | 차이 (mm) |
|---|---:|---:|
| QP 반환 자세에서 기존 tangent 예측 | +3.766252 | |
| 같은 QP 반환 자세에서 전체 PSDF 재평가 | −4.091114 | 7.857366 |
| 입력을 실제 RK4로 적분한 자세의 PSDF | −4.495174 | 0.404060 |

합계 8.261426 mm 중 7.857366 mm는 동일한 반환 자세에서 발생한 geometry 모델 차이다. 나머지 0.404060 mm는 RTI 예측 자세와 실제 입력 rollout 자세 사이의 signed-distance 변화다. 뒤의 위치 오차 벡터 norm은 0.516655 mm로, signed-distance 변화와 같은 값일 필요가 없다.

### 다른 원인 후보에 대한 대조

- **거리 함수/겹침 판정 오류**: 9개 첫 겹침 사례의 anchor·반환·실제 자세에서 17개 장애물을 각각 대조했다(459개 거리). 독립 Shapely separation 및 SAT minimum-translation 계산과 float64 PSDF가 일치했다. 이 특정 사례는 4번 장애물의 `x=0.600 m` 벽과 robot의 오른쪽 끝 `x=0.604495209 m`만으로도 4.495209 mm 침범을 확인할 수 있다.
- **Gradient 또는 좌표계 오류**: 위 9개 anchor의 활성 장애물 gradient를 독립 polygon 거리의 central finite difference와 대조했다. 세 성분별 절대 오차의 최댓값은 1.34e-05였다(translation 미분과 heading 미분은 단위가 다르므로 이를 하나의 물리 오차로 해석하지 않는다). 이 위치들에서 mm급 오차를 설명하는 gradient 오류는 관측되지 않았다. 전체 pose 공간의 모든 branch에 대한 검증을 의미하지는 않는다.
- **float32 반올림 또는 단위 변환**: 원본 float32 PSDF와 같은 geometry의 float64 PSDF 차이는 최대 0.000049 mm였다. 8 mm 차이를 설명하는 크기가 아니다.
- **Plant 모델·입력 적용·시점 불일치**: 앞선 분석에서 저장 입력으로 동일 RK4를 재계산한 상태가 로그와 정확히 일치했고, 다음 cycle의 원본 PSDF도 복원값과 일치했다. 현재 확보한 기록에는 다른 입력을 적용했거나 한 step을 잘못 맞춘 흔적이 없다.
- **추가 구현 요인**: pose/input warm start를 한 칸 shift하고 현재 상태에서 rollout하는 처리가 없고, trust region·same-cycle geometry 재선형화·exact one-step acceptance 검사도 없다. 이 조합은 거의 동률인 feature 중 하나에만 만든 tangent에서 QP가 멀리 이동하고도 그대로 입력을 내보내게 한다. 다른 사례에서는 covariance/risk QP 근사도 추가 오차를 만들지만, 위 8 mm 차이는 장애물별 거리만으로 이미 설명된다.

재현 자료: [branch 감사 수치](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/clearance_diagnosis/branch_audit.json), [보간 선분의 장애물별 거리](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/clearance_diagnosis/primary_branch_path.csv), [독립 감사 스크립트](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/audit_distance_branches.py). 새 주행이나 solver 실행은 수행하지 않았다.

## CC-MF-ICDF 폭 증가 실험 (2026-10-28)

폭 0.105–0.130 m를 0.005 m 간격으로 증가시키며 폭당 3회를 요청한 새 실험이다. 실제로 **12 trials를 실행해 목표 도달 7/12, safe-stop step 101회, 연속 safe-stop 구간 6개**를 기록했다.

**0.120 m에서 3회 모두 목표 도달에 실패**해 0.125 m / 0.130 m는 실행하지 않았다. 미실행 폭의 측정값은 빈 값으로 표시했다.

### 조건과 측정 기준

- Rectangle 길이 0.15 m, maze, DD, A*, seed 0–2, dt=0.1 s, trial당 최대 60 s. Localization error 및 animation/plot은 비활성화했다.
- 기존 CC-SF 폭 증가 실험과 맞춰 시작 자세 교란은 x/y 표준편차 20 mm, heading 표준편차 0.05 rad, 각 성분 ±3σ, 초기 clearance ≥1 mm로 설정했다. 실행한 모든 폭에서 seed별 시작 자세가 서로 같고 기존 CC-SF 실험의 자세와도 일치함을 검증했다. 상단 과거 5-trial 실험의 초기 clearance ≥10 mm 조건과 구분한다.
- 실패 판정은 연속 unsuccessful control step 20회 또는 4 s 동안 이동량 ≤0.03 m이다. Goal 판정은 기존 simulation 판정을 그대로 사용했다. 한 폭의 세 trial이 모두 목표 도달에 실패하면 이후 더 큰 폭을 생략하는 규칙을 적용했다.
- CC-MF-ICDF는 d_min=d_col=d_mf_mask=0.001 m, chance_epsilon=0.20, horizon=20 (2 s), robust horizon=10, MF slack linear=100 / quadratic=1을 유지했다. `enable_backup_solver=false`이므로 QP 실패 시 기본 복구는 safe stop이다. 전체 설정은 [optimizer parameters](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_3trials_20261028_110721/width_105mm/optimizer_parameters.json)에 보관했다.
- `method_description.md`의 nominal geometry·affine hard PSDF guard·soft MF row 구성을 확인했다. ICDF 변환의 상세 정의는 [inverse_cdf_mf_residual_revised.md](/home/taek111/projects/psdf_mpcc/inverse_cdf_mf_residual_revised.md)를 함께 확인했다. 확률합 R≤1/2의 유효 정의역에서 residual은 Φ⁻¹(ε)−Φ⁻¹(R)이며, R>1/2에서는 구현된 선형 확장 branch를 사용한다. Slack 단위는 standard-normal quantile이며 거리 단위가 아니다. Stage 0 OCP row는 비활성이고 현재 자세의 exact signed PSDF<d_col이면 solve 전에 `safe_stop_stage0`으로 들어간다. 미래 stage에는 affine guard가 적용되므로 solver 성공을 exact nonlinear clearance 보장으로 해석하지 않는다.
- **Clearance**는 실제 rectangle polygon과 모든 장애물 polygon의 거리 및 지도 경계 margin 중 최솟값이다. 초기 자세와 모든 적분 후 자세(최종 자세 포함)를 사용했다. 실패 trial의 정지 step도 포함한 전체 표본을 합쳐 mean/p05/min을 계산했으며, trial별 통계를 평균하지 않았다. 장애물 접촉·겹침 거리는 0, 지도 경계 침범은 음수다. 아래 최소값은 signed PSDF가 아니며 장애물 침투 깊이를 나타내지 않는다.
- 조기 종료된 실패 trial과 정지 자세의 반복 표본도 포함하므로, 폭별 평균·분위수는 서로 동일한 경로 구간만을 비교한 값이 아니다. 겹침 표본 수도 같은 겹친 자세에서 정지한 동안의 반복 기록을 포함하며, 독립적인 충돌 사건 수가 아니다.
- **Safe stop 횟수는 zero input을 적용한 control step 수**이며 `safe_stop`과 `safe_stop_stage0`를 모두 포함한다. 연속 20 step은 20회·2.0 s·연속 구간 1개다. 폭별 seed 표기는 0 / 1 / 2 순서다.
- 실행 소스를 새 폴더에 고정하고 같은 Python 3.10.13 / PyTorch 2.4.1 / CasADi 3.7.0 / RTX 3060, PyTorch CPU thread 6개에서 순차 실행했다. Controller 동작은 수정하지 않았다.

### 폭별 결과

| Robot width (m) | 목표 도달 | Mean clearance (mm) | P05 (mm) | Min (mm) | Safe-stop step: seed 0 / 1 / 2 | 연속 정지 구간 | 겹침 표본 | 경계 침범 표본 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.105 | 3/3 | 7.171 | 2.237 | 1.643 | 0/0/0 | 0 | 0 | 0 |
| 0.110 | 3/3 | 6.001 | 1.867 | 1.070 | 0/0/0 | 0 | 0 | 0 |
| 0.115 | 1/3 | 5.351 | 0.473 | 0.000 | 20/20/0 | 2 | 21 | 0 |
| 0.120 | 0/3 | 4.601 | 0.000 | 0.000 | 20/20/21 | 4 | 42 | 0 |
| 0.125 | 미실행 | — | — | — | — | — | — | — |
| 0.130 | 미실행 | — | — | — | — | — | — | — |

총 3,053개 자세에서 장애물 겹침 63개, 지도 경계 침범 0개가 관측됐다. 겹침과 경계 침범 표본은 중복될 수 있다. 기록된 모든 자세에서 clearance>0을 유지하면서 목표에 도달한 trial 수는 0.105 m: 3/3 / 0.110 m: 3/3 / 0.115 m: 1/3 / 0.120 m: 0/3이다. 폭당 3회 관측이며, 저장된 자세 사이의 연속시간 충돌이나 실제 확률적 안전성을 검증한 결과는 아니다.

### Trial별 결과

| Width (m) | Seed | 종료 상태 | 종료 시간 (s) | Mean clearance (mm) | Min (mm) | Safe-stop step | Stage-0 guard / QP 실패 |
|---|---:|---|---:|---:|---:|---:|---:|
| 0.105 | 0 | 목표 도달 | 33.3 | 7.121 | 1.878 | 0 | 0 / 0 |
| 0.105 | 1 | 목표 도달 | 33.7 | 7.122 | 1.643 | 0 | 0 / 0 |
| 0.105 | 2 | 목표 도달 | 32.9 | 7.271 | 1.905 | 0 | 0 / 0 |
| 0.110 | 0 | 목표 도달 | 34.5 | 6.045 | 1.261 | 0 | 0 / 0 |
| 0.110 | 1 | 목표 도달 | 33.7 | 5.966 | 1.070 | 0 | 0 / 0 |
| 0.110 | 2 | 목표 도달 | 35.5 | 5.992 | 1.079 | 0 | 0 / 0 |
| 0.115 | 0 | 실패 | 8.1 | 5.421 | 0.000 | 20 | 20 / 0 |
| 0.115 | 1 | 실패 | 17.0 | 5.616 | 0.473 | 20 | 20 / 0 |
| 0.115 | 2 | 목표 도달 | 35.3 | 5.206 | 1.499 | 0 | 0 / 0 |
| 0.120 | 0 | 실패 | 17.5 | 4.909 | 0.431 | 20 | 20 / 0 |
| 0.120 | 1 | 실패 | 12.7 | 4.333 | 0.000 | 20 | 20 / 0 |
| 0.120 | 2 | 실패 | 9.9 | 4.402 | 0.000 | 21 | 20 / 1 |

### Safe stop과 실패 사유

Safe-stop step 101회 중 stage-0 guard에 의한 정지는 100회, QP 실패 후 정지는 1회였다.

- 0.115 m, seed 0: 6.1–8.1 s에 20 step 연속 정지 (`safe_stop_stage0`, 시작 signed PSDF -1.292 mm).
- 0.115 m, seed 1: 15.0–17.0 s에 20 step 연속 정지 (`safe_stop_stage0`, 시작 signed PSDF 0.473 mm).
- 0.120 m, seed 0: 15.5–17.5 s에 20 step 연속 정지 (`safe_stop_stage0`, 시작 signed PSDF 0.431 mm).
- 0.120 m, seed 1: 10.7–12.7 s에 20 step 연속 정지 (`safe_stop_stage0`, 시작 signed PSDF -7.050 mm).
- 0.120 m, seed 2: 6.1–6.2 s에 1 step 연속 정지 (`safe_stop`, main status=4, 시작 signed PSDF 3.166 mm).
- 0.120 m, seed 2: 7.9–9.9 s에 20 step 연속 정지 (`safe_stop_stage0`, 시작 signed PSDF -0.764 mm).
- 0.115 m, seed 0의 종료 사유: `consecutive_unsuccessful_control_steps(20): safe_stop_stage0`.
- 0.115 m, seed 1의 종료 사유: `consecutive_unsuccessful_control_steps(20): safe_stop_stage0`.
- 0.120 m, seed 0의 종료 사유: `consecutive_unsuccessful_control_steps(20): safe_stop_stage0`.
- 0.120 m, seed 1의 종료 사유: `consecutive_unsuccessful_control_steps(20): safe_stop_stage0`.
- 0.120 m, seed 2의 종료 사유: `consecutive_unsuccessful_control_steps(20): safe_stop_stage0`.

**0.115 m seed 0, 0.120 m seed 1, 0.120 m seed 2에서는 실제 장애물 겹침이 생긴 뒤 stage-0 guard가 정지시켰다.** 따라서 이 결과의 safe stop을 이전 step의 충돌을 예방했다는 뜻으로 해석할 수 없다. 더 큰 폭의 생략은 이번 3-trial 실패에 따른 실행 규칙이며, 물리적 통과 가능성의 한계를 확정한 결과는 아니다.

### 원본 자료와 검증

- [폭별 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_3trials_20261028_110721/benchmark_summary.csv), [trial별 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_3trials_20261028_110721/benchmark_trials.csv), [모든 자세의 clearance](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_3trials_20261028_110721/clearance_samples.csv).
- [정지 구간](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_3trials_20261028_110721/safe_stop_events.csv), [접촉·겹침·경계 침범 자세](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_3trials_20261028_110721/geometry_violations.csv), [실행·생략 내역](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_3trials_20261028_110721/sweep_progress.json). 각 폭 폴더에 config, geometry, trial 결과, native cycle/history CSV, 관측 cycle CSV, runtime JSON 및 실행 로그를 보관했다.
- [검증 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_3trials_20261028_110721/validation.json): source 사본 불변성, 폭 외 geometry·optimizer parameter 동일성, seed별 시작 자세 일치, 초기·최종 자세 포함, pooled mean/p05 독립 재계산을 확인했다. Shapely로 3,053개 전체 자세의 polygon 거리를 별도 대조했다. Safe-stop step은 native cycle 로그·관측 로그·누적 runtime counter·적용 zero input 및 실제 정지 자세와 일치했고, 성공 step 수는 solver CSV와 대조했다.
- 재현용 [순차 실행 스크립트](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_3trials_20261028_110721/run_sweep.py) 및 [집계·검증 스크립트](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_3trials_20261028_110721/analyze.py), [고정 소스 해시](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_3trials_20261028_110721/source_hashes.json)를 보관했다. 재실행에는 새 결과 폴더를 사용한다.

## CC-MF-ICDF 0.2 mm 기준 재실험: 폭 0.115 m부터 (2026-10-28)

폭 0.115–0.130 m를 0.005 m 간격으로 증가시키며 폭당 3회를 요청한 새 실험이다. 실제로 **9 trials를 실행해 목표 도달 2/9, safe-stop step 140회, 연속 safe-stop 구간 7개**를 기록했다.

**0.125 m에서 3회 모두 목표 도달에 실패**해 0.130 m는 실행하지 않았다. 미실행 폭의 측정값은 빈 값으로 표시했다.

### 조건과 측정 기준

- Rectangle 길이 0.15 m, maze, DD, A*, seed 0–2, dt=0.1 s, trial당 최대 60 s. Localization error 및 animation/plot은 비활성화했다.
- 기존 CC-SF 폭 증가 실험과 맞춰 시작 자세 교란은 x/y 표준편차 20 mm, heading 표준편차 0.05 rad, 각 성분 ±3σ, 초기 clearance ≥1 mm로 설정했다. 실행한 모든 폭에서 seed별 시작 자세가 서로 같고 기존 CC-SF 실험의 자세와도 일치함을 검증했다. 상단 과거 5-trial 실험의 초기 clearance ≥10 mm 조건과 구분한다.
- 실패 판정은 연속 unsuccessful control step 20회 또는 4 s 동안 이동량 ≤0.03 m이다. Goal 판정은 기존 simulation 판정을 그대로 사용했다. 한 폭의 세 trial이 모두 목표 도달에 실패하면 이후 더 큰 폭을 생략하는 규칙을 적용했다.
- CC-MF-ICDF는 d_min=d_col=0.0002 m (0.2 mm), d_mf_mask 제거 및 d_col로 MF 활성화 기준 통합, chance_epsilon=0.20, horizon=20 (2 s), robust horizon=10, MF slack linear=100 / quadratic=1을 유지했다. `enable_backup_solver=false`이므로 QP 실패 시 기본 복구는 safe stop이다. 전체 설정은 [optimizer parameters](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/width_115mm/optimizer_parameters.json)에 보관했다.
- `method_description.md`의 nominal geometry·affine hard PSDF guard·soft MF row 구성을 확인했다. ICDF 변환의 상세 정의는 [inverse_cdf_mf_residual_revised.md](/home/taek111/projects/psdf_mpcc/inverse_cdf_mf_residual_revised.md)를 함께 확인했다. 확률합 R≤1/2의 유효 정의역에서 residual은 Φ⁻¹(ε)−Φ⁻¹(R)이며, R>1/2에서는 구현된 선형 확장 branch를 사용한다. Slack 단위는 standard-normal quantile이며 거리 단위가 아니다. Stage 0 OCP row는 비활성이고 현재 자세의 exact signed PSDF<d_col이면 solve 전에 `safe_stop_stage0`으로 들어간다. 미래 stage에는 affine guard가 적용되므로 solver 성공을 exact nonlinear clearance 보장으로 해석하지 않는다.
- **Clearance**는 실제 rectangle polygon과 모든 장애물 polygon의 거리 및 지도 경계 margin 중 최솟값이다. 초기 자세와 모든 적분 후 자세(최종 자세 포함)를 사용했다. 실패 trial의 정지 step도 포함한 전체 표본을 합쳐 mean/p05/min을 계산했으며, trial별 통계를 평균하지 않았다. 장애물 접촉·겹침 거리는 0, 지도 경계 침범은 음수다. 아래 최소값은 signed PSDF가 아니며 장애물 침투 깊이를 나타내지 않는다.
- 조기 종료된 실패 trial과 정지 자세의 반복 표본도 포함하므로, 폭별 평균·분위수는 서로 동일한 경로 구간만을 비교한 값이 아니다. 겹침 표본 수도 같은 겹친 자세에서 정지한 동안의 반복 기록을 포함하며, 독립적인 충돌 사건 수가 아니다.
- **Safe stop 횟수는 zero input을 적용한 control step 수**이며 `safe_stop`과 `safe_stop_stage0`를 모두 포함한다. 연속 20 step은 20회·2.0 s·연속 구간 1개다. 폭별 seed 표기는 0 / 1 / 2 순서다.
- 실행 소스를 새 폴더에 고정하고 같은 Python 3.10.13 / PyTorch 2.4.1 / CasADi 3.7.0 / RTX 3060, PyTorch CPU thread 6개에서 순차 실행했다. 이번 실행 중 controller를 추가 수정하지 않았다. 앞선 실험은 1 mm 및 별도 d_mf_mask를 사용했고, 이번 실험은 통합 d_col과 0.2 mm 기준을 사용했다.

### 폭별 결과

| Robot width (m) | 목표 도달 | Mean clearance (mm) | P05 (mm) | Min (mm) | Safe-stop step: seed 0 / 1 / 2 | 연속 정지 구간 | 겹침 표본 | 경계 침범 표본 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.115 | 1/3 | 4.920 | 0.161 | 0.138 | 20/20/0 | 2 | 0 | 0 |
| 0.120 | 1/3 | 3.872 | 0.000 | 0.000 | 20/20/0 | 2 | 42 | 0 |
| 0.125 | 0/3 | 3.402 | 0.000 | 0.000 | 20/20/20 | 3 | 42 | 0 |
| 0.130 | 미실행 | — | — | — | — | — | — | — |

총 1,705개 자세에서 장애물 겹침 84개, 지도 경계 침범 0개가 관측됐다. 겹침과 경계 침범 표본은 중복될 수 있다. 기록된 모든 자세에서 clearance>0을 유지하면서 목표에 도달한 trial 수는 0.115 m: 1/3 / 0.120 m: 1/3 / 0.125 m: 0/3이다. 폭당 3회 관측이며, 저장된 자세 사이의 연속시간 충돌이나 실제 확률적 안전성을 검증한 결과는 아니다.

### Trial별 결과

| Width (m) | Seed | 종료 상태 | 종료 시간 (s) | Mean clearance (mm) | Min (mm) | Safe-stop step | Stage-0 guard / QP 실패 |
|---|---:|---|---:|---:|---:|---:|---:|
| 0.115 | 0 | 실패 | 22.4 | 5.256 | 0.161 | 20 | 20 / 0 |
| 0.115 | 1 | 실패 | 21.3 | 4.921 | 0.138 | 20 | 20 / 0 |
| 0.115 | 2 | 목표 도달 | 35.6 | 4.708 | 0.355 | 0 | 0 / 0 |
| 0.120 | 0 | 실패 | 13.7 | 4.029 | 0.000 | 20 | 20 / 0 |
| 0.120 | 1 | 실패 | 8.7 | 3.662 | 0.000 | 20 | 20 / 0 |
| 0.120 | 2 | 목표 도달 | 37.8 | 3.863 | 0.232 | 0 | 0 / 0 |
| 0.125 | 0 | 실패 | 8.5 | 3.385 | 0.000 | 20 | 20 / 0 |
| 0.125 | 1 | 실패 | 13.1 | 3.329 | 0.000 | 20 | 20 / 0 |
| 0.125 | 2 | 실패 | 8.5 | 3.531 | 0.137 | 20 | 20 / 0 |

### Safe stop과 실패 사유

Safe-stop step 140회 중 stage-0 guard에 의한 정지는 140회, QP 실패 후 정지는 0회였다.

- 0.115 m, seed 0: 20.4–22.4 s에 20 step 연속 정지 (`safe_stop_stage0`, 시작 signed PSDF 0.161 mm).
- 0.115 m, seed 1: 19.3–21.3 s에 20 step 연속 정지 (`safe_stop_stage0`, 시작 signed PSDF 0.138 mm).
- 0.120 m, seed 0: 11.7–13.7 s에 20 step 연속 정지 (`safe_stop_stage0`, 시작 signed PSDF -1.061 mm).
- 0.120 m, seed 1: 6.7–8.7 s에 20 step 연속 정지 (`safe_stop_stage0`, 시작 signed PSDF -1.036 mm).
- 0.125 m, seed 0: 6.5–8.5 s에 20 step 연속 정지 (`safe_stop_stage0`, 시작 signed PSDF -0.150 mm).
- 0.125 m, seed 1: 11.1–13.1 s에 20 step 연속 정지 (`safe_stop_stage0`, 시작 signed PSDF -0.527 mm).
- 0.125 m, seed 2: 6.5–8.5 s에 20 step 연속 정지 (`safe_stop_stage0`, 시작 signed PSDF 0.137 mm).
- 0.115 m, seed 0의 종료 사유: `consecutive_unsuccessful_control_steps(20): safe_stop_stage0`.
- 0.115 m, seed 1의 종료 사유: `consecutive_unsuccessful_control_steps(20): safe_stop_stage0`.
- 0.120 m, seed 0의 종료 사유: `consecutive_unsuccessful_control_steps(20): safe_stop_stage0`.
- 0.120 m, seed 1의 종료 사유: `consecutive_unsuccessful_control_steps(20): safe_stop_stage0`.
- 0.125 m, seed 0의 종료 사유: `consecutive_unsuccessful_control_steps(20): safe_stop_stage0`.
- 0.125 m, seed 1의 종료 사유: `consecutive_unsuccessful_control_steps(20): safe_stop_stage0`.
- 0.125 m, seed 2의 종료 사유: `consecutive_unsuccessful_control_steps(20): safe_stop_stage0`.

**0.120 m seed 0, 0.120 m seed 1, 0.125 m seed 0, 0.125 m seed 1에서는 실제 장애물 겹침이 생긴 뒤 stage-0 guard가 정지시켰다.** 따라서 이 결과의 safe stop을 이전 step의 충돌을 예방했다는 뜻으로 해석할 수 없다. 더 큰 폭의 생략은 이번 3-trial 실패에 따른 실행 규칙이며, 물리적 통과 가능성의 한계를 확정한 결과는 아니다.

### 애니메이션

기존 `animate_world` 코드로 저장된 실제 주행 자세를 재생했다. 초기·최종 자세와 safe-stop 정지 구간을 포함하며, 폭당 seed 0–2 각각 MP4로 저장했다. 100 dpi, 830×500 px, 10 fps, frame_skip=1을 사용했다. A* 경로는 동일 시작 자세와 지도에서 재생성했으며, CSV에 저장되지 않은 local/predicted trajectory는 표시하지 않았다.

| Width (m) | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 0.115 | [MP4 · 실패](/home/taek111/projects/psdf_mpcc/animations/cc_mf_icdf/MPC_cc_mf_icdf_rectangle_maze_width115mm_seed0_dcol02mm_20261028_163235.mp4) | [MP4 · 실패](/home/taek111/projects/psdf_mpcc/animations/cc_mf_icdf/MPC_cc_mf_icdf_rectangle_maze_width115mm_seed1_dcol02mm_20261028_163235.mp4) | [MP4 · 목표 도달](/home/taek111/projects/psdf_mpcc/animations/cc_mf_icdf/MPC_cc_mf_icdf_rectangle_maze_width115mm_seed2_dcol02mm_20261028_163235.mp4) |
| 0.120 | [MP4 · 실패](/home/taek111/projects/psdf_mpcc/animations/cc_mf_icdf/MPC_cc_mf_icdf_rectangle_maze_width120mm_seed0_dcol02mm_20261028_163235.mp4) | [MP4 · 실패](/home/taek111/projects/psdf_mpcc/animations/cc_mf_icdf/MPC_cc_mf_icdf_rectangle_maze_width120mm_seed1_dcol02mm_20261028_163235.mp4) | [MP4 · 목표 도달](/home/taek111/projects/psdf_mpcc/animations/cc_mf_icdf/MPC_cc_mf_icdf_rectangle_maze_width120mm_seed2_dcol02mm_20261028_163235.mp4) |
| 0.125 | [MP4 · 실패](/home/taek111/projects/psdf_mpcc/animations/cc_mf_icdf/MPC_cc_mf_icdf_rectangle_maze_width125mm_seed0_dcol02mm_20261028_163235.mp4) | [MP4 · 실패](/home/taek111/projects/psdf_mpcc/animations/cc_mf_icdf/MPC_cc_mf_icdf_rectangle_maze_width125mm_seed1_dcol02mm_20261028_163235.mp4) | [MP4 · 실패](/home/taek111/projects/psdf_mpcc/animations/cc_mf_icdf/MPC_cc_mf_icdf_rectangle_maze_width125mm_seed2_dcol02mm_20261028_163235.mp4) |

영상은 [애니메이션 manifest](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/animations.json)에 저장 경로·해상도·프레임 수·원본 history 해시를 기록했다. 실행한 trial 전체를 렌더링했으며 생략한 0.130 m의 영상은 생성하지 않았다.

### 원본 자료와 검증

- [폭별 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/benchmark_summary.csv), [trial별 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/benchmark_trials.csv), [모든 자세의 clearance](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/clearance_samples.csv).
- [정지 구간](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/safe_stop_events.csv), [접촉·겹침·경계 침범 자세](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/geometry_violations.csv), [실행·생략 내역](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/sweep_progress.json). 각 폭 폴더에 config, geometry, trial 결과, native cycle/history CSV, 관측 cycle CSV, runtime JSON 및 실행 로그를 보관했다.
- [검증 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/validation.json): source 사본 불변성, 폭 외 geometry·optimizer parameter 동일성, seed별 시작 자세 일치, 초기·최종 자세 포함, pooled mean/p05 독립 재계산을 확인했다. Shapely로 1,705개 전체 자세의 polygon 거리를 별도 대조했다. Safe-stop step은 native cycle 로그·관측 로그·누적 runtime counter·적용 zero input 및 실제 정지 자세와 일치했고, 성공 step 수는 solver CSV와 대조했다.
- 재현용 [순차 실행 스크립트](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/run_sweep.py) 및 [집계·검증 스크립트](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/analyze.py), [고정 소스 해시](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/source_hashes.json)를 보관했다. 재실행에는 새 결과 폴더를 사용한다.

## CC-SF 안전 제약 stage 0–19 재실험 (2026-10-28)

**0.105 m는 3/3 목표 도달·safe stop 0회, 0.110 m는 0/3 목표 도달·각 seed에서 safe stop 20 step이었다.** 기존에 허용한 전원 실패 시 생략 규칙에 따라 **0.115 / 0.120 / 0.125 / 0.130 m는 실행하지 않았다.** 총 6 trials를 완료했으며, 기록된 모든 자세에서 장애물 접촉·겹침 및 지도 경계 침범은 없었다. 실행한 6개 trial의 MP4도 저장했다.

### 변경과 실험 조건

- **N=20, dt=0.1 s, tf=2 s를 유지**했다. 상태는 x0–x20, 입력은 u0–u19이며, hard SF 안전 제약 20개의 적용 위치만 기존 stage 1–20에서 **stage 0–19**로 옮겼다. 현재 상태 x0는 고정하고 terminal state x20의 안전 제약은 제외했다. Horizon을 19개 제어 구간으로 줄인 실험이 아니다.
- 주·복구 solver의 실제 생성 JSON 12개 모두 **nh_0=1, nh=1, nh_e=0, slack=0**을 확인했다. Stage 0은 별도의 사전 검사로 처리하지 않고 OCP hard row에 포함했다.
- 이전 width sweep의 고정 소스에서 `control/cc_sf_optimizer.py`의 제약 stage 부분만 변경했다. 폭 이외의 설정은 동일하다: corridor 0.150 m, robot 길이 0.150 m, DD·A*, seed 0–2, 초기 clearance ≥1 mm, localization error 비활성화, d_min=1 mm, epsilon=0.20, robust horizon=10, 최대 60 s. 실행한 각 폭·seed의 시작 자세는 이전 실험과 정확히 같다.
- Clearance는 초기 및 모든 적분 후 자세를 포함한 polygon 거리와 지도 경계 margin의 최솟값이다. 실패 trial 및 정지 중 반복 자세도 포함해 pooled mean/p05/min을 계산했다. Safe-stop 횟수는 zero input을 적용한 **control step 수**이며, 연속 20 step=2 s=정지 구간 1개이다.

### 폭별 결과

| Width (m) | 목표 도달 | Mean clearance (mm) | P05 (mm) | Min (mm) | Safe-stop step: seed 0 / 1 / 2 | 겹침 표본 |
|---|---:|---:|---:|---:|---:|---:|
| 0.105 | 3/3 | 7.012 | 2.219 | 1.870 | 0/0/0 | 0 |
| 0.110 | 0/3 | 5.989 | 0.952 | 0.952 | 20/20/20 | 0 |
| 0.115 | 생략 | — | — | — | — | — |
| 0.120 | 생략 | — | — | — | — | — |
| 0.125 | 생략 | — | — | — | — | — |
| 0.130 | 생략 | — | — | — | — | — |

0.105 m의 seed 0 / 1 / 2 목표 도달 시간은 **29.1 / 29.2 / 28.9 s**였다. 기존 stage 1–20 실험의 0.110 m는 목표 도달 2/3, safe stop 20/0/0 step이었지만, 이번 stage 0–19 설정에서는 세 seed 모두 더 일찍 정지했다. 두 설정은 terminal row 유무도 다르며, 실패로 주행 길이가 달라지므로 pooled 평균 clearance만으로 안전성을 비교하지 않는다.

### 0.110 m의 정지 원인

정지 직전의 현재 상태에서 $h_0=\phi(x_0)-d_{\min}-r_0$가 음수였다. 아래 phi는 로그의 signed PSDF이며, polygon 거리와 최대 0.000031 mm 차이였다. 단위는 mm이다.

| Seed | Safe-stop 구간 (s) | phi(x0) | Risk margin r0 | d_min+r0 | h0 | 종료 사유 |
|---|---:|---:|---:|---:|---:|---|
| 0 | 4.2–6.2 | 0.983508 | 0.209877 | 1.209877 | -0.226369 | solver 연속 실패 20회 |
| 1 | 4.3–6.3 | 1.056027 | 0.209900 | 1.209900 | -0.153874 | solver 연속 실패 20회 |
| 2 | 4.1–6.1 | 0.952340 | 0.209871 | 1.209871 | -0.257531 | solver 연속 실패 20회 |

- 실패 60 step 모두 주 SQP-RTI **status=4**, 복구 SQP_WITH_FEASIBLE_QP **status=2**였고, 복구 성공은 0회였다. Stage-0 residual이 음수인 60 step 모두 safe stop으로 이어졌으며, 음수인데 solver 성공으로 통과한 step은 없었다.
- x0의 pose와 초기 covariance는 등식으로 고정된다. 따라서 현재 h0<0인 상태는 미래 입력으로 해소할 수 없고, zero input으로 멈춘 후 같은 조건이 유지되어 20회 연속 실패로 종료했다. Seed 1은 실제 거리가 1 mm보다 커도 risk margin을 더한 약 1.210 mm보다 작아 정지했다.
- **이번 6 trials에서는 겹치기 전에 멈췄지만, stage 0 추가가 다음 입력 적용 후의 무충돌을 보장하는 것은 아니다.** 첫 정지 바로 전 cycle은 세 seed 모두 solver status=0이었고 h0가 양수였으나, 입력 적용 후 현재 margin 조건을 위반했다. 고정 tangent와 비선형 입력 적용에 따른 차이는 그대로 남아 있다. 기록된 자세 사이의 연속시간 충돌은 검사하지 않았다.

### Animation

기존 `simulation_mpc.animate_world`를 그대로 사용했다(기본 렌더링 설정, 10 fps, frame_skip=5, HTML 변환 생략). 별도 고화질 렌더러는 추가하지 않았다. 각 영상은 해당 trial의 마지막 정지 또는 목표 도달 자세까지 포함한다.

| Width (m) | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 0.105 | [MP4](/home/taek111/projects/psdf_mpcc/animations/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/MPC_cc_sf_rectangle_maze_width105mm_seed0_stage0-19.mp4) | [MP4](/home/taek111/projects/psdf_mpcc/animations/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/MPC_cc_sf_rectangle_maze_width105mm_seed1_stage0-19.mp4) | [MP4](/home/taek111/projects/psdf_mpcc/animations/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/MPC_cc_sf_rectangle_maze_width105mm_seed2_stage0-19.mp4) |
| 0.110 | [MP4](/home/taek111/projects/psdf_mpcc/animations/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/MPC_cc_sf_rectangle_maze_width110mm_seed0_stage0-19.mp4) | [MP4](/home/taek111/projects/psdf_mpcc/animations/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/MPC_cc_sf_rectangle_maze_width110mm_seed1_stage0-19.mp4) | [MP4](/home/taek111/projects/psdf_mpcc/animations/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/MPC_cc_sf_rectangle_maze_width110mm_seed2_stage0-19.mp4) |

### 검증과 원본 자료

- 관련 단위 테스트 **10개 통과**. 소스 사본 불변성, 기존 시작 자세 일치, runtime·cycle·history의 safe-stop 수, zero input 및 정지 중 자세 유지, solver 시간 CSV와 성공 step 수를 대조했다. 전체 **1,064개 자세**의 clearance를 Shapely로 독립 확인했다. MP4 6개의 유효한 영상 stream·프레임 수·파일 크기도 확인했다.
- [폭별 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/benchmark_summary.csv), [Trial별 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/benchmark_trials.csv), [정지 시점 데이터](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stage0_stop_onsets.csv), [생략 내역](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/sweep_progress.json), [검증](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stage0_validation.json), [Animation 목록](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/animations.csv), [생성 solver의 stage 정책](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/solver_stage_policy.json).
- 관측용 로깅 함수의 인자 오류로 첫 사전 실행이 주행 전에 종료된 기록은 `setup_observer_error_excluded/`에 분리했다. 실제 적용 입력 0회였으며 위 6 trials와 통계에서 제외했다. 수정 후 새 폭별 출력 폴더에서 재실행했다.
- 재현용 [순차 실행](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/run_sweep.py)·[집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/analyze.py)·[stage-0 검증](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/audit_stage0.py) 스크립트를 보관했다. 재실행에는 새 결과 폴더를 사용한다.

## CC-MF-ICDF 성공·실패 분리 원인 분석 (0.2 mm 기준, 2026-10-28)

**주요 원인은 solver의 수치 실패나 MF 비활성화가 아니라, 한 번의 SQP-RTI에서 사용하는 근사 안전 모델과 실제 다음 자세의 차이다.** 실패 7건 중 실제 겹침이 생긴 4건은 geometry Taylor 모델이 다른 장애물 쪽의 위험을 놓쳤고, 겹침 없이 멈춘 3건은 예측 자세와 실제 RK4 적분 자세의 차이로 0.2 mm 기준을 밑돌았다. 그 뒤 `safe_stop_stage0`가 zero input을 계속 적용해 같은 자세에 머물며 20연속 unsuccessful step으로 종료했다.

### 로그에서 직접 확인한 사실

- 9개 trial 전체에서 main solver failure는 0회다. 실패 7건의 마지막 이동 입력은 모두 `solver_status=0`으로 반환됐고, 미래 stage 1 MF row도 모두 활성화돼 있었다.
- 위험 입력 cycle의 최대 MF slack은 약 7.8e−17–2.0e−12 quantile, 전체 9 trial 최대는 2.953e−10 quantile였다. 큰 slack으로 제약을 풀어준 사례가 아니다.
- 실패 후 20 step은 solver가 20번 infeasible을 반환한 것이 아니다. 현재 exact signed PSDF<0.2 mm를 감지해 solve를 건너뛴 정지 step이다. DD에서 zero input은 자세를 바꾸지 않으므로 같은 veto가 반복된다.

### 최초 기준 위반을 만든 한 step의 복원

아래 값은 기존 benchmark의 unsigned polygon clearance와 다른 **signed PSDF**다. 양수는 분리, 음수는 겹침을 나타낸다. `근사 guard`는 nominal tangent를 복원한 RTI 예측 자세에 대입한 값이고, `예측 자세 실제 PSDF`는 같은 예측 자세에서 geometry를 새로 평가한 값이다. `적분 후 실제 PSDF`는 저장된 실제 자세를 재평가해 다음 cycle의 로그와 대조했다. 요구값은 0.200 mm다.

| Width (m) | Seed | 위험 입력 시각 (s) | 근사 guard (mm) | 예측 자세 실제 PSDF (mm) | 적분 후 실제 PSDF (mm) | 주된 위반 요인 |
|---|---:|---:|---:|---:|---:|---|
| 0.115 | 0 | 20.3 | 0.200 | 0.620 | 0.161 | 동역학 선형화 오차 |
| 0.115 | 1 | 19.2 | 0.200 | 0.567 | 0.138 | 동역학 선형화 오차 |
| 0.120 | 0 | 11.6 | 1.394 | -0.982 | -1.061 | 장애물 branch 전환·기하 근사 |
| 0.120 | 1 | 6.6 | 3.677 | -0.849 | -1.036 | 장애물 branch 전환·기하 근사 |
| 0.125 | 0 | 6.4 | 4.646 | -0.191 | -0.150 | 장애물 branch 전환·기하 근사 |
| 0.125 | 1 | 11.0 | 2.548 | -0.310 | -0.527 | 장애물 branch 전환·기하 근사 |
| 0.125 | 2 | 6.4 | 0.200 | 0.204 | 0.137 | 동역학 선형화 오차 |

![실패 직전 근사 모델과 실제 거리 비교](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/failure_analysis/failure_mechanisms.png)

그림의 좌우 패널은 y축 범위가 다르다. 점선은 공통 기준 0.200 mm다.

**4개 겹침 사례:** 근사 guard는 1.394–4.646 mm로 충분히 안전하다고 평가했지만, 복원한 예측 자세를 geometry로 재평가하면 −0.982–−0.191 mm였다. 4건 모두 nominal의 최소 장애물과 예측 자세의 최소 장애물이 바뀌었다(0부터 시작하는 cluster ID: 4→3, 1→4, 1→4, 4→2). 기하 모델의 과대평가는 2.376–4.836 mm로, 예측–적분 위치 차이 0.060–0.248 mm보다 크다. 따라서 이 4건은 실제 plant에 입력을 적용하기 전의 예측 자세부터 이미 겹침이었다. 하나의 최소 branch의 tangent가 전체 최소 PSDF의 안전한 하한은 아니어서, slack 없는 hard row도 이 위험을 막지 못했다. Cluster 전환을 확인했지만 내부 vertex/edge ID 전환을 모두 직접 추적한 것은 아니다.

**3개 비겹침 정지 사례:** 폭 0.115 m seed 0/1 및 0.125 m seed 2는 예측 자세의 실제 PSDF가 각각 0.620 / 0.567 / 0.204 mm로 기준 이상이었다. 그러나 반환 입력을 현재 자세에 RK4로 적용한 자세는 예측과 각각 0.460 / 0.430 / 0.090 mm 달랐고, 실제 PSDF는 0.161 / 0.138 / 0.137 mm가 됐다. SQP-RTI가 한 번 푼 선형화된 동역학 equality의 예측 자세와 비선형 RK4 결과가 일치하지 않은 사례다. 저장된 입력을 기존 plant RK4로 다시 평가하면 실제 다음 자세가 오차 0으로 재현돼, 별도의 plant 모델이나 localization noise 탓으로 볼 근거는 없다.

MF row 자체도 같은 한계를 보인다. 7건 모두 복원한 반환 상태에서 affine MF row는 거의 0 이상의 residual과 미미한 slack으로 충족됐지만, 그 반환 상태의 covariance와 frozen nominal features로 비선형 ICDF residual을 다시 평가하면 음수였다. 즉 MF가 켜져 있고 slack이 0에 가까워도 한 번의 affine RTI row 충족이 원래 residual 충족을 보장하지 않는다.

### 왜 seed에 따라 성공과 실패가 갈리는가

시작 자세는 seed마다 다르며 localization error는 꺼져 있다. 이후 nominal pose/input을 한 칸 shift하거나 현재 자세에서 다시 rollout하지 않고 이전 solver 궤적을 유지하므로, 작은 초기 자세 차이가 경로 추종·회전 입력·최소 거리 branch·다음 nominal을 다르게 만든다. 이번 실패는 corner 통과 입력과 최종 여유가 그 차이에 민감한 상황에서 발생했다. 특정 초기 교란 성분 하나가 단독 원인임을 입증한 대조 실험은 아니다.

- 폭 0.120 m seed 1 실패 직전과 성공 seed 2의 최근접 자세는 위치 차이 1.082 mm, heading 차이 약 0.00431 rad였다. 실패 입력은 v=0.08833 m/s, ω=0.18201 rad/s로 PSDF를 3.922→−1.036 mm로 바꿨다. 성공 trial은 그 근처에서 v=0.04499 m/s, ω=0.71713 rad/s로 2.874→3.140 mm를 유지했다. 현재 pose가 비슷해도 retained nominal/covariance와 최적 입력이 같지는 않다.
- 폭 0.125 m seed 1은 v=0.00163 m/s로 거의 제자리에서 ω=−1.15755 rad/s로 회전한 한 step만으로 2.570→−0.527 mm가 됐다. Rectangle에서는 전진 속도가 작아도 회전하는 모서리가 벽 쪽으로 들어갈 수 있다.
- 성공 seed 2의 최소 polygon clearance는 폭 0.115 m에서 0.35495 mm, 폭 0.120 m에서 0.23200 mm였다. 후자는 기준보다 약 32 μm 높을 뿐이다. 그 최소점에서도 예측–적분 위치 차이가 약 0.923 mm였으므로, 성공을 일관된 동역학 정확성이나 넉넉한 안전 여유의 증거로 볼 수 없다. 성공 trial은 이번 기록에서 기준 아래로 내려가는 입력 조합을 피한 것이며, 폭당 3회로 일반적인 성공 확률이나 특정 seed의 우수성을 판정할 수 없다.

### 실패가 회복되지 않는 코드 경로와 수정 우선순위

1. 고정 nominal의 affine guard와 affine ICDF row로 한 번의 SQP-RTI를 푼다.
2. status=0인 입력을 fresh one-step geometry 검사 없이 적용한다.
3. 다음 cycle에서 current PSDF<d_col을 발견하면 solve를 건너뛰고 zero input을 적용한다.
4. 같은 자세에서 20회 반복해 trial을 종료한다.

이 chain에 직접 대응하는 우선순위는 **반환 u0를 실제 RK4로 한 step 전파한 뒤 exact PSDF를 재평가하고, 기준 미달이면 적용 전에 거부·재선형화하는 검사**다. 그다음 현재 자세에서 shifted-input rollout으로 nominal을 만들고, 가까운 여러 obstacle/feature row 또는 trust region·재선형화 반복을 함께 검토할 수 있다. 이미 기준 아래인 상태를 탈출시키는 정책은 별도의 회복 문제다. Stage-0 정지를 끄거나 d_col을 더 낮추는 것만으로 앞선 기하·동역학 근사 오차는 해결되지 않는다. 이 분석에서는 controller를 수정하거나 새 주행/solver를 실행하지 않았다.

### 재현성과 분석 범위

Nominal/returned pose 전 궤적은 직접 저장돼 있지 않아, stage-0가 현재 자세로 고정되고 nominal u0가 직전 반환 u0라는 코드 정책을 이용해 `x1_RTI = f(x0, u_prev) + B(x0, u_prev)·(u_current−u_prev)`로 stage-1 pose를 복원했다. 직전 cycle의 복원 stage-1 pose가 이번 nominal anchor가 된다. 이와 같은 equality로 계산한 반환 covariance를 별도 저장된 covariance CSV와 대조했고, 분석 cycle들까지의 최대 covariance-entry 차이는 6.573e-14였다. Nominal stage-1 MF residual/probability와 active mask도 9개 대표 step 모두 재현했다. 이 검증은 복원의 일관성을 뒷받침하지만 직접 저장된 pose와의 대조를 대신하지는 않는다. 성공 trial의 비교 step은 최소 clearance 지점이며 실패 trial의 최초 위반 step과 동일 위치/동일 시각 대조는 아니다.

- [최초 위반/성공 최소점 수치](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/failure_analysis/one_step_reconstruction.csv), [자세·입력·기울기 상세](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/failure_analysis/one_step_details.json), [복원 검증](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/failure_analysis/reconstruction_validation.json), [오프라인 복원 스크립트](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/failure_analysis/reconstruct.py).
- [실패/성공 telemetry·최근접 위치 비교](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/failure_analysis/telemetry/comparison.json).
- 코드 근거: [affine guard 정의](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/source_snapshot/control/cc_mf_icdf_optimizer.py:171), [hard row와 MF slack](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/source_snapshot/control/cc_mf_optimizer.py:798), [nominal 재사용과 covariance 재전파](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/source_snapshot/control/cc_mf_optimizer.py:1003), [현재 상태 veto](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/source_snapshot/control/cc_mf_optimizer.py:1104), [zero-input 정지 계획](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/source_snapshot/control/cc_mf_optimizer.py:1180), [성공 입력 반환](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_icdf_width_sweep_dcol02mm_20261028_163235/source_snapshot/control/controller.py:94).

## 0.110 m: zero-input 후 정지 지속과 d_min 미만 진입 원인 (2026-10-28 추가 분석)

**0.110 m의 원인은 앞서 분석한 0.120 m의 최소 장애물 전환 사례와 다르다.** Zero input으로 미래 margin은 줄었지만 현재 stage-0 margin은 남아 복구되지 않았다. 그 위치에 진입한 직전 QP에서는 **covariance 동역학을 선형화한 결과 불확실성이 과소평가되고, risk-margin 접선이 음수가 되어 phi<1 mm인 해도 hard row를 만족**했다. 이번 세 입력에서는 phi tangent와 물리 kinematics 오차가 오히려 실제 clearance를 늘리는 방향이었다. 앞 절의 phi·kinematics 설명만으로는 이 현상을 설명할 수 없다.

### 1. Zero input인데 seed 1이 복구되지 않은 이유

입력 의존 noise는 이후 covariance 전파에 반영된다. 매 cycle의 현재 covariance는 설정 초기값(표준편차: forward 0.2 mm, lateral 0.25 mm, heading 0.25 mrad)으로 다시 고정된다. 따라서 현재 $r_0$는 현재 covariance와 gradient로 결정되며, 현재 입력을 0으로 바꿔도 변하지 않는다. 정지한다고 초기 위치·자세 불확실성까지 없어지지는 않는다.

Seed 1의 관측값은 다음과 같다. 단위는 mm이다.

| 값 | 첫 정지 시도 k43, t=4.3 s | Zero input 후 k44, t=4.4 s |
|---|---:|---:|
| phi(x0) | 1.056027 | 1.056027 |
| Stage-0 margin r0 | 0.209900 | 0.209900 |
| Stage-1 nominal margin | 1.560160 | 0.209900 |
| Stage-10 nominal margin | 3.743221 | 0.209900 |
| h0 = phi − 1 mm − r0 | −0.153874 | −0.153874 |

미래 margin은 실제로 크게 감소했다. 하지만 seed 1에서 허용되는 margin은 phi−d_min=**0.056027 mm**뿐이므로, 남은 **0.209900 mm**로도 현재 hard row를 위반한다. 이 잔여 margin은 `risk_margin_floor` 설정 때문이 아니라 비영 초기 covariance 때문이며 해당 파라미터는 0이다. Pose와 covariance를 포함한 x0 전체가 등식으로 고정되어 미래 입력으로 h0<0을 해소할 수 없다. k43–62의 fast/backup 총 40회 시도에서 동일한 phi0·r0·h0가 유지됐고, fast status=4 / backup status=2가 반복됐다.

코드 근거: [초기 covariance 및 x0 고정](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/source_snapshot/control/cc_sf_optimizer.py:812), [covariance 재전파](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/source_snapshot/control/cc_sf_optimizer.py:521), [zero-input plan 설치](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/source_snapshot/control/cc_sf_optimizer.py:943).

### 2. phi가 1 mm보다 작은 곳에 진입한 직접 원인

첫 정지 직전의 성공 입력 3개를 분석했다. 표의 QP phi는 고정 tangent 평가값, 재평가 phi는 QP가 반환한 다음 pose에서 다시 계산한 PSDF, 적용 후 phi는 실제 RK4 적분 후 PSDF이다. 모든 수치는 mm이며 세 입력 모두 solver status=0이었다.

| Seed | 입력 적용 구간 (s) | QP phi | QP risk 접선 | QP hard residual | 다음 pose 재평가 phi | 적용 후 phi |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4.1→4.2 | 0.839771 | -0.774102 | 0.613873 | 0.968312 | 0.983508 |
| 1 | 4.2→4.3 | 0.912194 | -0.779120 | 0.691314 | 1.041426 | 1.056027 |
| 2 | 4.0→4.1 | 0.805809 | -0.706762 | 0.512571 | 0.937196 | 0.952340 |

원래 nonlinear margin $r(\xi)=\gamma\|L(\xi)^Tq\|$는 비음수지만, QP는 그 접선 $r_{lin}=\bar r+\nabla_\xi r(\bar\xi)^T(\xi-\bar\xi)$을 사용한다. 이 접선의 비음수성은 보장되지 않는다.

예를 들어 seed 0은 다음처럼 QP row를 만족했다.

$$
h_{QP}=0.839771-1.000000-(-0.774102)=+0.613873\;\mathrm{mm}.
$$

**QP 내부에서 이미 phi가 1 mm보다 작았고, 음수 risk 접선이 이를 허용했다.** 원래 margin 자체가 음수가 된 것은 아니다. 같은 반환 covariance에서 nonlinear margin을 평가하면 seed별 **+0.114551 / +0.113762 / +0.125595 mm**이며, 재평가한 nonlinear SF residual은 세 seed 모두 음수였다.

### 3. Covariance와 margin 선형화가 크게 어긋난 경로

Seed 0의 nominal 각속도는 **−0.028147 rad/s**, 반환 입력은 **+0.636302 rad/s**로, 한 RTI step에서 0을 가로질러 **+0.664448 rad/s** 바뀌었다. 다른 두 seed도 같은 방향의 부호 전환이 있었다. Covariance의 process noise는 `|v*omega|`와 `omega**2`를 포함한다. 음수 omega에서 만든 국소 선형 모델을 큰 양수 omega까지 외삽하면 실제로 증가하는 회전 noise를 감소하는 방향으로 예측할 수 있다.

| Seed 0의 다음-stage 값 | Nominal anchor | QP 반환 covariance | 적용 입력을 원래 covariance 식으로 전파 |
|---|---:|---:|---:|
| Lateral 표준편차 (mm) | 0.634930 | 0.051732 | 1.548167 |
| 동일 frozen q에서 nonlinear margin (mm) | 0.583912 | 0.114551 | 1.584790 |

QP의 covariance는 같은 입력을 nonlinear map에 넣은 값보다 지나치게 작다. 이 잘못된 covariance 이동을 따라 margin 접선도 **−0.774102 mm**까지 내려갔다. 위 마지막 열은 동일 frozen gradient에서 재전파한 **계획 margin**이며, 다음 cycle에 초기 covariance로 reset하여 계산하는 r0=0.209877 mm와는 다른 값이다.

반면 seed 0의 거리 tangent는 0.839771 mm로 다음 pose의 정확한 PSDF 0.968312 mm보다 **0.128541 mm 작게** 계산했다. Kinematics 차이도 적용 후 거리를 **0.015196 mm 늘렸다**(pose 위치 차이 0.017903 mm). 나머지 두 seed도 같은 부호다. 세 사례 모두 최소 장애물 index는 anchor·반환 pose·적용 후 pose에서 **0**으로 같았다. 따라서 이 사례의 d_min 진입 원인을 최소 장애물 전환이나 거리 과대평가로 설명해서는 안 된다.

코드 근거: [입력 의존 noise](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/source_snapshot/control/cc_covariance.py:180), [Log-Cholesky 전이](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/source_snapshot/control/cc_covariance.py:229), [nonlinear margin 및 hard row](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/source_snapshot/control/cc_sf_optimizer.py:674).

### 검증 범위와 시사점

- 반환 states·inputs를 직접 저장한 로그를 사용했다. Linearized dynamics equality는 최대 3.95e−15 오차로, 실제 RK4 적용 결과는 로그와 정확히 일치했다. 153개 obstacle–pose 거리를 독립 polygon geometry로 대조했고 float32/64 PSDF 차이는 최대 0.000037 mm였다. Production CasADi constraint·dynamics 식으로 margin 접선과 covariance 계산도 독립 확인했다. 새 주행이나 최적화 solver 실행은 하지 않았다.
- Stage 0은 현재 위반을 검출하지만 직전 입력의 진입을 예방하지는 않는다. 수정 방향은 독립적인 deterministic phi≥d_min row로 음수 risk 접선의 안전거리 완화를 막고, covariance·입력 변화 제한 또는 재선형화, 입력 적용 전 nonlinear pose·covariance 안전성 검사를 함께 검토하는 것이다. Deterministic row도 affine phi 모델에 걸리므로 exact geometry 검사를 대체하지 않는다. 이 분석에서 controller 동작은 변경하지 않았다.
- [오차 분해 데이터](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stop_diagnosis/entry_decomposition.csv), [반환 상태·gradient 상세](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stop_diagnosis/entry_details.json), [정지 후 margin 데이터](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stop_margin_audit.json), [production 식 대조](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stop_diagnosis/production_expression_review.json), [검증 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stop_diagnosis/validation.json), [재현 스크립트](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stop_diagnosis/analyze_entry.py).
