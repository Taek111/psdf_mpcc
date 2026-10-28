# Rectangle width별 controller benchmark 결과

작성일: 2026-10-27 (KST). 기존 실험 로그를 집계했으며 이 문서 작성을 위한 추가 시뮬레이션은 실행하지 않았다.

## 실험 조건

- 환경: maze, 동역학: differential drive (DD), global planner: A*.
- Rectangle 길이: 0.15 m. 폭: 0.10 m / 0.08 m.
- Controller: MPCC, CC-SF, CC-ECS, CC-MF-IMP. 각 조합 5 trials, seed 0–4.
- 시작 자세 교란: x/y 표준편차 0.02 m, heading 표준편차 0.05 rad, 각 성분 ±3σ 제한, 초기 clearance ≥ 0.01 m.
- Trial당 시뮬레이션 제한 60초, dt = 0.1초. Localization error 비활성화.
- 실패 기준: 4초 동안 이동량 0.03 m 미만 또는 solver 연속 실패 20회.
- 8개 조합 모두 목표 도달 5/5 (100%). CC-MF-IMP slack penalty는 linear=100, quadratic=1을 유지했다.

## 주요 결과: clearance · solver computation time · traversal time

Computation time은 성공한 control step의 acados `solve()` 내부 호출 시간이다. 단위는 clearance **mm**, computation time **ms/step**, traversal time **시뮬레이션 초(s)**이다.

### Rectangle width 0.10 m

| Controller | Mean clearance (mm) | P05 clearance (mm) | Mean computation time (ms/step) | P95 computation time (ms/step) | Mean traversal time (s) |
|---|---:|---:|---:|---:|---:|
| MPCC | 7.869 | 1.340 | 1.412 | 0.676 | 26.16 |
| CC-SF | 8.352 | 2.362 | 0.709 | 0.846 | 28.94 |
| CC-ECS | 8.464 | 2.047 | 1.151 | 1.389 | 28.76 |
| CC-MF-IMP | 8.158 | 2.238 | 0.912 | 1.097 | 31.14 |

### Rectangle width 0.08 m

| Controller | Mean clearance (mm) | P05 clearance (mm) | Mean computation time (ms/step) | P95 computation time (ms/step) | Mean traversal time (s) |
|---|---:|---:|---:|---:|---:|
| MPCC | 8.749 | 1.638 | 0.614 | 0.937 | 24.18 |
| CC-SF | 10.861 | 2.714 | 0.670 | 0.836 | 25.68 |
| CC-ECS | 9.669 | 2.422 | 1.115 | 1.359 | 25.46 |
| CC-MF-IMP | 10.740 | 2.598 | 0.913 | 1.122 | 27.10 |

## 집계 기준과 해석

- **Clearance**: rectangle footprint와 장애물 사이의 정확한 polygon 거리 및 지도 경계까지의 margin 중 최솟값. 초기 자세와 모든 적분 후 자세(최종 자세 포함)를 사용하고, 5 trials의 전체 샘플을 합쳐 mean/p05를 계산했다. Trial별 평균이나 p05를 다시 평균한 값이 아니다. P05는 하위 5% 분위수이며 선형 보간을 사용했다. 샘플 사이의 연속 궤적에 대한 충돌 검사는 포함하지 않는다.
- **Solver computation time**: `time.perf_counter()`로 측정한 `solve()` 호출 시간. 첫 step을 포함한 성공 step 전체를 합쳐 mean/p95를 계산했다. 복구 solver가 호출되면 해당 step의 호출 시간을 합산한다. Solver setup/코드 생성/컴파일, PSDF 및 implicit-root 전처리, 파라미터 갱신, 경로 계획, 출력 시간은 제외한다.
- **Traversal time**: 기존 목표 도달 판정까지의 시뮬레이션 시간이며, 성공한 5 trials의 산술평균이다. 실제 실행에 걸린 wall-clock 시간과 다르다.
- **폭 0.10 m MPCC의 mean > p95**: 첫 step의 solver 호출 약 1,113.927 ms가 평균에 반영됐다. 해당 이상치를 제거하지 않았으며, 원본 1,308 solver step을 그대로 집계한 값이다.
- **폭 간 시작 자세 차이**: 같은 폭에서는 네 controller가 seed별 동일한 시작 자세를 사용했다. 폭 0.08 m에서는 seed 4의 첫 후보 자세가 초기 clearance 조건을 통과해, 폭 0.10 m에서 재샘플링한 자세와 다르다. 폭 간 비교는 이 차이를 포함한다.

## 참고: controller 호출 시간 (첫 step 제외)

위 주요 표의 solver 시간에는 PSDF/implicit-root 전처리 비용이 들어가지 않는다. 아래 표는 로깅된 **controller 호출 전체 시간**을 사용한다. 호출 내부의 전처리, 파라미터 갱신, solver 및 복구 처리를 포함하고, global/local 경로 계획은 제외한다. 초기화·코드 생성·컴파일의 영향을 줄이기 위해 **각 trial의 첫 호출만 제외**하고 나머지 step을 합쳐 집계했다. 따라서 주요 표의 solver 시간과 표본 범위도 다르다.

### Rectangle width 0.10 m

| Controller | Mean controller time (ms/step) | P95 controller time (ms/step) | Steps |
|---|---:|---:|---:|
| MPCC | 36.442 | 54.744 | 1303 |
| CC-SF | 41.118 | 59.433 | 1442 |
| CC-ECS | 41.832 | 55.677 | 1433 |
| CC-MF-IMP | 84.504 | 121.337 | 1552 |

### Rectangle width 0.08 m

| Controller | Mean controller time (ms/step) | P95 controller time (ms/step) | Steps |
|---|---:|---:|---:|
| MPCC | 40.269 | 67.250 | 1204 |
| CC-SF | 42.051 | 61.390 | 1279 |
| CC-ECS | 40.534 | 55.914 | 1268 |
| CC-MF-IMP | 81.748 | 117.042 | 1350 |

## 원본 자료

- 폭 0.10 m — MPCC / CC-SF / CC-ECS: [clearance·traversal·시간 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/maze_rectangle_dd_perturb_four_methods_5trials_20261026_165452/benchmark_summary.csv), [solver 시간 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/maze_rectangle_dd_perturb_four_methods_5trials_20261026_165452/summary.csv), [trial 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/maze_rectangle_dd_perturb_four_methods_5trials_20261026_165452/trial_results.csv). 2026-10-26 실행 자료를 사용했다.
- 폭 0.10 m — CC-MF-IMP: [clearance 비교 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_imp_maze_rectangle_dd_5trials_20261027_180434/clearance_comparison.csv), [solver 시간 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_imp_maze_rectangle_dd_5trials_20261027_180434/baseline/summary.csv), [trial 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_imp_maze_rectangle_dd_5trials_20261027_180434/baseline/trial_results.csv). 2026-10-27 실행 자료를 사용했다.
- 폭 0.08 m — 네 controller: [전체 지표 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/maze_rectangle_w008_dd_four_methods_5trials_20261027_181526/benchmark_summary.csv), [solver 시간 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/maze_rectangle_w008_dd_four_methods_5trials_20261027_181526/summary.csv), [trial 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/maze_rectangle_w008_dd_four_methods_5trials_20261027_181526/trial_results.csv), [검증 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/maze_rectangle_w008_dd_four_methods_5trials_20261027_181526/validation.json). 2026-10-27 실행 자료를 사용했다.
- 폭 0.10 m CC-SF는 MPCC·CC-ECS와 함께 실행한 four-method batch의 시간 통계를 사용했다. 별도로 수행한 CC-SF safe-stop batch와 clearance/주행 시간은 같지만, 실행 시간 측정값은 다르므로 혼합하지 않았다.
- Controller 시간은 위 trial 결과가 가리키는 각 `runs/.../data/trial_history_*.csv`의 `controller_computation_time`에서 재계산했다. 모든 solver mean/p95는 해당 실험의 `computation_times_*.csv`와 원본 `summary.csv`에 일치함을 확인했다.
