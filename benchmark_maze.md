# Maze benchmark

2026-10-26, `test_nmpc.py`로 방법별 5회씩 총 20회 순차 평가했다.

- 환경: maze, rectangle **0.15 × 0.10 m**, DD, A*. 방법별 현재 기본 파라미터 사용.
- 시작 perturbation: 위치 표준편차 20 mm, heading 0.05 rad, ±3σ 제한, 초기 clearance ≥ 10 mm. 방법 간 동일한 seed 0–4 사용.
- Localization error 비활성, trial당 최대 60초. 4초간 이동 ≤ 3 cm 또는 20회 연속 제어 실패 시 종료.

| 방법 | 성공 | Clearance mean / p5 [mm] | 최소 [mm] | 제어기 [ms/step] | Solver [ms/step] | Navigation time [s] | Safe stop 합계 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPCC | 5/5 | 7.87 / 1.34 | 1.04 | 35.13 | 0.573 | 26.16 ± 0.15 | 0 |
| CC-SF | 5/5 | 11.82 / 5.41 | 4.95 | 45.74 | 0.669 | 38.34 ± 0.35 | 0 |
| CC-MF | 3/5 | 8.42 / 1.86 | **−105.44** | 35.76 | 0.933 | 41.63 ± 0.12 | **40** |
| CC-ECS | 5/5 | 8.47 / 2.05 | 1.66 | 50.60 | 1.166 | 28.76 ± 0.11 | 0 |

- Clearance는 실패 trial을 포함해 초기 및 0.1초 간격의 실제 footprint–장애물·맵 경계 간격을 합산했다. p5는 하위 5백분위수이며, 음수는 맵 경계 이탈을 뜻한다.
- 제어기 시간은 PSDF 갱신·복구 등을 포함하며 trial별 최초 코드 생성·컴파일 step을 제외했다. Solver 시간은 성공한 step의 `solve()` 호출만 측정했다.
- Navigation time은 성공 trial의 도달시간 평균 ± 표본 표준편차다. Safe stop은 5회 실행의 발생 제어 주기 합계이며, MPCC에는 해당 fallback이 없다.

CC-MF는 seed 1·4에서 `safe_stop_stage0`가 각각 20회 연속 발생해 실패했다. Seed 1에서는 장애물 겹침이 21개 저장 자세에서 확인됐고, footprint가 맵 경계 밖으로 최대 105.44 mm 벗어났다.

다른 방법은 저장 자세에서 충돌이 없었다. 다만 CC-ECS의 예측 ECS 제약 재평가 위반은 최대 23.19 mm였다. 이번 결과는 방법별 5회 및 이산 자세 검사에 한정된다.

원시 자료: [실행 설정](benchmark_runs/maze_rectangle_width010_perturb_four_methods_5trials_20261026_105337/config.yaml) · [집계 CSV](benchmark_runs/maze_rectangle_width010_perturb_four_methods_5trials_20261026_105337/analysis/comparison.csv) · [Trial별 결과](benchmark_runs/maze_rectangle_width010_perturb_four_methods_5trials_20261026_105337/analysis/per_trial.csv)
