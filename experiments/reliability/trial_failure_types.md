# DoN 0.75–0.87 trial별 결과와 그림 재현

[LaTeX 삽입용 PNG](failure_breakdown_latex.png) · 2460 × 1272 px, 600 dpi

## 그림에 사용한 표본

그림은 **DoN 0.83–0.87**을 표시한다. DoN 0.83의 CC-MF는 새 무작위 seed 재실행 5회 전체를 사용한다. 같은 DoN의 CC-SF와 DoN 0.84–0.87의 두 방법은 기존 sweep에서 방법·DoN당 50회 전체를 사용한다. DoN 0.83에서 두 방법의 seed와 실행 환경은 서로 다르다.

각 막대는 해당 표본의 완료 trial 수를 분모로 정규화한다. 아래부터 주황색 Feature-switching, 사선이 있는 파란색 Feature-persistent, 초록색 Success 순으로 쌓는다. 그림에 선택된 표본에서는 Progress stagnation과 Other failures가 각각 0건이다.

| DoN | Controller | Cohort | 전체 | Success | Feature-switching | Feature-persistent |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0.83 | CC-SF | Sweep | 50 | 47 | 3 | 0 |
| 0.83 | CC-MF | Rerun | 5 | 5 | 0 | 0 |
| 0.84 | CC-SF | Sweep | 50 | 34 | 16 | 0 |
| 0.84 | CC-MF | Sweep | 50 | 49 | 0 | 1 |
| 0.85 | CC-SF | Sweep | 50 | 32 | 18 | 0 |
| 0.85 | CC-MF | Sweep | 50 | 45 | 5 | 0 |
| 0.86 | CC-SF | Sweep | 50 | 4 | 45 | 1 |
| 0.86 | CC-MF | Sweep | 50 | 20 | 26 | 4 |
| 0.87 | CC-SF | Sweep | 50 | 0 | 50 | 0 |
| 0.87 | CC-MF | Sweep | 50 | 0 | 46 | 4 |

## Trial별 결과

DoN 0.75–0.87의 원래 sweep 1,300건과 DoN 0.80–0.83의 CC-MF 재실행 20건을 수록했다. Sweep의 Trial은 원본 trial 번호다. Rerun의 Trial은 해당 DoN에서의 실행 순서다. 두 cohort의 기록은 독립적으로 유지했다.

- **Success**는 목표 도달, **Progress stagnation**은 진행 정체 조건에 따른 종료다.
- **Feature-switching**과 **Feature-persistent**는 첫 stage-0 guard 진입에서 적용 입력 직전의 최근접 기하 feature가 각각 바뀌었거나 유지된 경우다. Guard 거리는 0.05 mm다.
- **Other failures**는 위 유형에 해당하지 않는 실패다. 수록 표본에는 0건이다.

[원래 sweep](../../benchmark_runs/rectangle_constraint_reliability_20261107/rectangle_don_sweep_50_20261105_current_sf/) · [실패 분류 근거](../../benchmark_runs/rectangle_constraint_reliability_20261107/failure_modes_20261110/failure_trial_classification.csv) · [무작위 seed 재실행](../../benchmark_runs/rectangle_constraint_reliability_20261107/cc_mf_don080_083_random5_20261110_161545/trial_results.csv)

### DoN 0.75

| Cohort | Controller | Trial | Seed | Outcome |
| --- | --- | ---: | ---: | --- |
| Sweep | CC-SF | 1 | 0 | Success |
| Sweep | CC-SF | 2 | 1 | Success |
| Sweep | CC-SF | 3 | 2 | Success |
| Sweep | CC-SF | 4 | 3 | Success |
| Sweep | CC-SF | 5 | 4 | Success |
| Sweep | CC-SF | 6 | 5 | Success |
| Sweep | CC-SF | 7 | 6 | Success |
| Sweep | CC-SF | 8 | 7 | Success |
| Sweep | CC-SF | 9 | 8 | Success |
| Sweep | CC-SF | 10 | 9 | Success |
| Sweep | CC-SF | 11 | 10 | Success |
| Sweep | CC-SF | 12 | 11 | Success |
| Sweep | CC-SF | 13 | 12 | Success |
| Sweep | CC-SF | 14 | 13 | Success |
| Sweep | CC-SF | 15 | 14 | Success |
| Sweep | CC-SF | 16 | 15 | Success |
| Sweep | CC-SF | 17 | 16 | Success |
| Sweep | CC-SF | 18 | 17 | Success |
| Sweep | CC-SF | 19 | 18 | Success |
| Sweep | CC-SF | 20 | 19 | Success |
| Sweep | CC-SF | 21 | 20 | Success |
| Sweep | CC-SF | 22 | 21 | Success |
| Sweep | CC-SF | 23 | 22 | Success |
| Sweep | CC-SF | 24 | 23 | Success |
| Sweep | CC-SF | 25 | 24 | Success |
| Sweep | CC-SF | 26 | 25 | Success |
| Sweep | CC-SF | 27 | 26 | Success |
| Sweep | CC-SF | 28 | 27 | Success |
| Sweep | CC-SF | 29 | 28 | Success |
| Sweep | CC-SF | 30 | 29 | Success |
| Sweep | CC-SF | 31 | 30 | Success |
| Sweep | CC-SF | 32 | 31 | Success |
| Sweep | CC-SF | 33 | 32 | Success |
| Sweep | CC-SF | 34 | 33 | Success |
| Sweep | CC-SF | 35 | 34 | Success |
| Sweep | CC-SF | 36 | 35 | Success |
| Sweep | CC-SF | 37 | 36 | Success |
| Sweep | CC-SF | 38 | 37 | Success |
| Sweep | CC-SF | 39 | 38 | Success |
| Sweep | CC-SF | 40 | 39 | Success |
| Sweep | CC-SF | 41 | 40 | Success |
| Sweep | CC-SF | 42 | 41 | Success |
| Sweep | CC-SF | 43 | 42 | Success |
| Sweep | CC-SF | 44 | 43 | Success |
| Sweep | CC-SF | 45 | 44 | Success |
| Sweep | CC-SF | 46 | 45 | Success |
| Sweep | CC-SF | 47 | 46 | Success |
| Sweep | CC-SF | 48 | 47 | Success |
| Sweep | CC-SF | 49 | 48 | Success |
| Sweep | CC-SF | 50 | 49 | Success |
| Sweep | CC-MF | 1 | 0 | Success |
| Sweep | CC-MF | 2 | 1 | Success |
| Sweep | CC-MF | 3 | 2 | Success |
| Sweep | CC-MF | 4 | 3 | Success |
| Sweep | CC-MF | 5 | 4 | Success |
| Sweep | CC-MF | 6 | 5 | Success |
| Sweep | CC-MF | 7 | 6 | Success |
| Sweep | CC-MF | 8 | 7 | Success |
| Sweep | CC-MF | 9 | 8 | Success |
| Sweep | CC-MF | 10 | 9 | Success |
| Sweep | CC-MF | 11 | 10 | Success |
| Sweep | CC-MF | 12 | 11 | Success |
| Sweep | CC-MF | 13 | 12 | Success |
| Sweep | CC-MF | 14 | 13 | Success |
| Sweep | CC-MF | 15 | 14 | Success |
| Sweep | CC-MF | 16 | 15 | Success |
| Sweep | CC-MF | 17 | 16 | Success |
| Sweep | CC-MF | 18 | 17 | Success |
| Sweep | CC-MF | 19 | 18 | Success |
| Sweep | CC-MF | 20 | 19 | Success |
| Sweep | CC-MF | 21 | 20 | Success |
| Sweep | CC-MF | 22 | 21 | Success |
| Sweep | CC-MF | 23 | 22 | Success |
| Sweep | CC-MF | 24 | 23 | Progress stagnation |
| Sweep | CC-MF | 25 | 24 | Success |
| Sweep | CC-MF | 26 | 25 | Success |
| Sweep | CC-MF | 27 | 26 | Success |
| Sweep | CC-MF | 28 | 27 | Success |
| Sweep | CC-MF | 29 | 28 | Success |
| Sweep | CC-MF | 30 | 29 | Success |
| Sweep | CC-MF | 31 | 30 | Success |
| Sweep | CC-MF | 32 | 31 | Success |
| Sweep | CC-MF | 33 | 32 | Success |
| Sweep | CC-MF | 34 | 33 | Success |
| Sweep | CC-MF | 35 | 34 | Success |
| Sweep | CC-MF | 36 | 35 | Success |
| Sweep | CC-MF | 37 | 36 | Success |
| Sweep | CC-MF | 38 | 37 | Success |
| Sweep | CC-MF | 39 | 38 | Success |
| Sweep | CC-MF | 40 | 39 | Success |
| Sweep | CC-MF | 41 | 40 | Success |
| Sweep | CC-MF | 42 | 41 | Success |
| Sweep | CC-MF | 43 | 42 | Success |
| Sweep | CC-MF | 44 | 43 | Success |
| Sweep | CC-MF | 45 | 44 | Success |
| Sweep | CC-MF | 46 | 45 | Success |
| Sweep | CC-MF | 47 | 46 | Success |
| Sweep | CC-MF | 48 | 47 | Success |
| Sweep | CC-MF | 49 | 48 | Success |
| Sweep | CC-MF | 50 | 49 | Success |

### DoN 0.76

| Cohort | Controller | Trial | Seed | Outcome |
| --- | --- | ---: | ---: | --- |
| Sweep | CC-SF | 1 | 0 | Success |
| Sweep | CC-SF | 2 | 1 | Success |
| Sweep | CC-SF | 3 | 2 | Success |
| Sweep | CC-SF | 4 | 3 | Success |
| Sweep | CC-SF | 5 | 4 | Success |
| Sweep | CC-SF | 6 | 5 | Feature-switching |
| Sweep | CC-SF | 7 | 6 | Success |
| Sweep | CC-SF | 8 | 7 | Success |
| Sweep | CC-SF | 9 | 8 | Success |
| Sweep | CC-SF | 10 | 9 | Success |
| Sweep | CC-SF | 11 | 10 | Success |
| Sweep | CC-SF | 12 | 11 | Success |
| Sweep | CC-SF | 13 | 12 | Success |
| Sweep | CC-SF | 14 | 13 | Success |
| Sweep | CC-SF | 15 | 14 | Success |
| Sweep | CC-SF | 16 | 15 | Success |
| Sweep | CC-SF | 17 | 16 | Success |
| Sweep | CC-SF | 18 | 17 | Success |
| Sweep | CC-SF | 19 | 18 | Success |
| Sweep | CC-SF | 20 | 19 | Success |
| Sweep | CC-SF | 21 | 20 | Success |
| Sweep | CC-SF | 22 | 21 | Feature-switching |
| Sweep | CC-SF | 23 | 22 | Success |
| Sweep | CC-SF | 24 | 23 | Success |
| Sweep | CC-SF | 25 | 24 | Success |
| Sweep | CC-SF | 26 | 25 | Success |
| Sweep | CC-SF | 27 | 26 | Success |
| Sweep | CC-SF | 28 | 27 | Success |
| Sweep | CC-SF | 29 | 28 | Success |
| Sweep | CC-SF | 30 | 29 | Success |
| Sweep | CC-SF | 31 | 30 | Success |
| Sweep | CC-SF | 32 | 31 | Success |
| Sweep | CC-SF | 33 | 32 | Success |
| Sweep | CC-SF | 34 | 33 | Success |
| Sweep | CC-SF | 35 | 34 | Success |
| Sweep | CC-SF | 36 | 35 | Success |
| Sweep | CC-SF | 37 | 36 | Success |
| Sweep | CC-SF | 38 | 37 | Success |
| Sweep | CC-SF | 39 | 38 | Success |
| Sweep | CC-SF | 40 | 39 | Success |
| Sweep | CC-SF | 41 | 40 | Success |
| Sweep | CC-SF | 42 | 41 | Success |
| Sweep | CC-SF | 43 | 42 | Success |
| Sweep | CC-SF | 44 | 43 | Success |
| Sweep | CC-SF | 45 | 44 | Success |
| Sweep | CC-SF | 46 | 45 | Success |
| Sweep | CC-SF | 47 | 46 | Success |
| Sweep | CC-SF | 48 | 47 | Success |
| Sweep | CC-SF | 49 | 48 | Success |
| Sweep | CC-SF | 50 | 49 | Success |
| Sweep | CC-MF | 1 | 0 | Success |
| Sweep | CC-MF | 2 | 1 | Success |
| Sweep | CC-MF | 3 | 2 | Success |
| Sweep | CC-MF | 4 | 3 | Success |
| Sweep | CC-MF | 5 | 4 | Success |
| Sweep | CC-MF | 6 | 5 | Feature-switching |
| Sweep | CC-MF | 7 | 6 | Success |
| Sweep | CC-MF | 8 | 7 | Success |
| Sweep | CC-MF | 9 | 8 | Success |
| Sweep | CC-MF | 10 | 9 | Success |
| Sweep | CC-MF | 11 | 10 | Success |
| Sweep | CC-MF | 12 | 11 | Success |
| Sweep | CC-MF | 13 | 12 | Success |
| Sweep | CC-MF | 14 | 13 | Success |
| Sweep | CC-MF | 15 | 14 | Success |
| Sweep | CC-MF | 16 | 15 | Success |
| Sweep | CC-MF | 17 | 16 | Success |
| Sweep | CC-MF | 18 | 17 | Success |
| Sweep | CC-MF | 19 | 18 | Success |
| Sweep | CC-MF | 20 | 19 | Success |
| Sweep | CC-MF | 21 | 20 | Success |
| Sweep | CC-MF | 22 | 21 | Success |
| Sweep | CC-MF | 23 | 22 | Success |
| Sweep | CC-MF | 24 | 23 | Success |
| Sweep | CC-MF | 25 | 24 | Success |
| Sweep | CC-MF | 26 | 25 | Success |
| Sweep | CC-MF | 27 | 26 | Success |
| Sweep | CC-MF | 28 | 27 | Success |
| Sweep | CC-MF | 29 | 28 | Success |
| Sweep | CC-MF | 30 | 29 | Success |
| Sweep | CC-MF | 31 | 30 | Success |
| Sweep | CC-MF | 32 | 31 | Success |
| Sweep | CC-MF | 33 | 32 | Success |
| Sweep | CC-MF | 34 | 33 | Success |
| Sweep | CC-MF | 35 | 34 | Success |
| Sweep | CC-MF | 36 | 35 | Success |
| Sweep | CC-MF | 37 | 36 | Success |
| Sweep | CC-MF | 38 | 37 | Success |
| Sweep | CC-MF | 39 | 38 | Success |
| Sweep | CC-MF | 40 | 39 | Success |
| Sweep | CC-MF | 41 | 40 | Success |
| Sweep | CC-MF | 42 | 41 | Success |
| Sweep | CC-MF | 43 | 42 | Success |
| Sweep | CC-MF | 44 | 43 | Success |
| Sweep | CC-MF | 45 | 44 | Success |
| Sweep | CC-MF | 46 | 45 | Success |
| Sweep | CC-MF | 47 | 46 | Success |
| Sweep | CC-MF | 48 | 47 | Success |
| Sweep | CC-MF | 49 | 48 | Success |
| Sweep | CC-MF | 50 | 49 | Success |

### DoN 0.77

| Cohort | Controller | Trial | Seed | Outcome |
| --- | --- | ---: | ---: | --- |
| Sweep | CC-SF | 1 | 0 | Success |
| Sweep | CC-SF | 2 | 1 | Success |
| Sweep | CC-SF | 3 | 2 | Success |
| Sweep | CC-SF | 4 | 3 | Success |
| Sweep | CC-SF | 5 | 4 | Success |
| Sweep | CC-SF | 6 | 5 | Success |
| Sweep | CC-SF | 7 | 6 | Success |
| Sweep | CC-SF | 8 | 7 | Success |
| Sweep | CC-SF | 9 | 8 | Success |
| Sweep | CC-SF | 10 | 9 | Success |
| Sweep | CC-SF | 11 | 10 | Success |
| Sweep | CC-SF | 12 | 11 | Success |
| Sweep | CC-SF | 13 | 12 | Success |
| Sweep | CC-SF | 14 | 13 | Success |
| Sweep | CC-SF | 15 | 14 | Success |
| Sweep | CC-SF | 16 | 15 | Success |
| Sweep | CC-SF | 17 | 16 | Success |
| Sweep | CC-SF | 18 | 17 | Success |
| Sweep | CC-SF | 19 | 18 | Success |
| Sweep | CC-SF | 20 | 19 | Success |
| Sweep | CC-SF | 21 | 20 | Success |
| Sweep | CC-SF | 22 | 21 | Success |
| Sweep | CC-SF | 23 | 22 | Success |
| Sweep | CC-SF | 24 | 23 | Success |
| Sweep | CC-SF | 25 | 24 | Success |
| Sweep | CC-SF | 26 | 25 | Success |
| Sweep | CC-SF | 27 | 26 | Success |
| Sweep | CC-SF | 28 | 27 | Success |
| Sweep | CC-SF | 29 | 28 | Success |
| Sweep | CC-SF | 30 | 29 | Success |
| Sweep | CC-SF | 31 | 30 | Success |
| Sweep | CC-SF | 32 | 31 | Success |
| Sweep | CC-SF | 33 | 32 | Success |
| Sweep | CC-SF | 34 | 33 | Success |
| Sweep | CC-SF | 35 | 34 | Success |
| Sweep | CC-SF | 36 | 35 | Success |
| Sweep | CC-SF | 37 | 36 | Success |
| Sweep | CC-SF | 38 | 37 | Success |
| Sweep | CC-SF | 39 | 38 | Success |
| Sweep | CC-SF | 40 | 39 | Success |
| Sweep | CC-SF | 41 | 40 | Success |
| Sweep | CC-SF | 42 | 41 | Success |
| Sweep | CC-SF | 43 | 42 | Success |
| Sweep | CC-SF | 44 | 43 | Success |
| Sweep | CC-SF | 45 | 44 | Success |
| Sweep | CC-SF | 46 | 45 | Success |
| Sweep | CC-SF | 47 | 46 | Success |
| Sweep | CC-SF | 48 | 47 | Success |
| Sweep | CC-SF | 49 | 48 | Success |
| Sweep | CC-SF | 50 | 49 | Success |
| Sweep | CC-MF | 1 | 0 | Success |
| Sweep | CC-MF | 2 | 1 | Success |
| Sweep | CC-MF | 3 | 2 | Success |
| Sweep | CC-MF | 4 | 3 | Success |
| Sweep | CC-MF | 5 | 4 | Success |
| Sweep | CC-MF | 6 | 5 | Success |
| Sweep | CC-MF | 7 | 6 | Success |
| Sweep | CC-MF | 8 | 7 | Success |
| Sweep | CC-MF | 9 | 8 | Success |
| Sweep | CC-MF | 10 | 9 | Success |
| Sweep | CC-MF | 11 | 10 | Success |
| Sweep | CC-MF | 12 | 11 | Success |
| Sweep | CC-MF | 13 | 12 | Success |
| Sweep | CC-MF | 14 | 13 | Success |
| Sweep | CC-MF | 15 | 14 | Success |
| Sweep | CC-MF | 16 | 15 | Success |
| Sweep | CC-MF | 17 | 16 | Success |
| Sweep | CC-MF | 18 | 17 | Success |
| Sweep | CC-MF | 19 | 18 | Success |
| Sweep | CC-MF | 20 | 19 | Success |
| Sweep | CC-MF | 21 | 20 | Success |
| Sweep | CC-MF | 22 | 21 | Success |
| Sweep | CC-MF | 23 | 22 | Success |
| Sweep | CC-MF | 24 | 23 | Success |
| Sweep | CC-MF | 25 | 24 | Success |
| Sweep | CC-MF | 26 | 25 | Success |
| Sweep | CC-MF | 27 | 26 | Success |
| Sweep | CC-MF | 28 | 27 | Success |
| Sweep | CC-MF | 29 | 28 | Success |
| Sweep | CC-MF | 30 | 29 | Success |
| Sweep | CC-MF | 31 | 30 | Success |
| Sweep | CC-MF | 32 | 31 | Success |
| Sweep | CC-MF | 33 | 32 | Success |
| Sweep | CC-MF | 34 | 33 | Success |
| Sweep | CC-MF | 35 | 34 | Success |
| Sweep | CC-MF | 36 | 35 | Success |
| Sweep | CC-MF | 37 | 36 | Success |
| Sweep | CC-MF | 38 | 37 | Success |
| Sweep | CC-MF | 39 | 38 | Success |
| Sweep | CC-MF | 40 | 39 | Success |
| Sweep | CC-MF | 41 | 40 | Success |
| Sweep | CC-MF | 42 | 41 | Success |
| Sweep | CC-MF | 43 | 42 | Success |
| Sweep | CC-MF | 44 | 43 | Success |
| Sweep | CC-MF | 45 | 44 | Success |
| Sweep | CC-MF | 46 | 45 | Success |
| Sweep | CC-MF | 47 | 46 | Success |
| Sweep | CC-MF | 48 | 47 | Feature-switching |
| Sweep | CC-MF | 49 | 48 | Success |
| Sweep | CC-MF | 50 | 49 | Success |

### DoN 0.78

| Cohort | Controller | Trial | Seed | Outcome |
| --- | --- | ---: | ---: | --- |
| Sweep | CC-SF | 1 | 0 | Success |
| Sweep | CC-SF | 2 | 1 | Success |
| Sweep | CC-SF | 3 | 2 | Success |
| Sweep | CC-SF | 4 | 3 | Success |
| Sweep | CC-SF | 5 | 4 | Success |
| Sweep | CC-SF | 6 | 5 | Success |
| Sweep | CC-SF | 7 | 6 | Success |
| Sweep | CC-SF | 8 | 7 | Success |
| Sweep | CC-SF | 9 | 8 | Success |
| Sweep | CC-SF | 10 | 9 | Success |
| Sweep | CC-SF | 11 | 10 | Success |
| Sweep | CC-SF | 12 | 11 | Success |
| Sweep | CC-SF | 13 | 12 | Success |
| Sweep | CC-SF | 14 | 13 | Success |
| Sweep | CC-SF | 15 | 14 | Success |
| Sweep | CC-SF | 16 | 15 | Success |
| Sweep | CC-SF | 17 | 16 | Success |
| Sweep | CC-SF | 18 | 17 | Success |
| Sweep | CC-SF | 19 | 18 | Success |
| Sweep | CC-SF | 20 | 19 | Success |
| Sweep | CC-SF | 21 | 20 | Success |
| Sweep | CC-SF | 22 | 21 | Success |
| Sweep | CC-SF | 23 | 22 | Success |
| Sweep | CC-SF | 24 | 23 | Success |
| Sweep | CC-SF | 25 | 24 | Success |
| Sweep | CC-SF | 26 | 25 | Success |
| Sweep | CC-SF | 27 | 26 | Success |
| Sweep | CC-SF | 28 | 27 | Success |
| Sweep | CC-SF | 29 | 28 | Success |
| Sweep | CC-SF | 30 | 29 | Success |
| Sweep | CC-SF | 31 | 30 | Success |
| Sweep | CC-SF | 32 | 31 | Success |
| Sweep | CC-SF | 33 | 32 | Success |
| Sweep | CC-SF | 34 | 33 | Success |
| Sweep | CC-SF | 35 | 34 | Success |
| Sweep | CC-SF | 36 | 35 | Success |
| Sweep | CC-SF | 37 | 36 | Success |
| Sweep | CC-SF | 38 | 37 | Success |
| Sweep | CC-SF | 39 | 38 | Success |
| Sweep | CC-SF | 40 | 39 | Success |
| Sweep | CC-SF | 41 | 40 | Success |
| Sweep | CC-SF | 42 | 41 | Success |
| Sweep | CC-SF | 43 | 42 | Success |
| Sweep | CC-SF | 44 | 43 | Success |
| Sweep | CC-SF | 45 | 44 | Success |
| Sweep | CC-SF | 46 | 45 | Success |
| Sweep | CC-SF | 47 | 46 | Success |
| Sweep | CC-SF | 48 | 47 | Feature-switching |
| Sweep | CC-SF | 49 | 48 | Success |
| Sweep | CC-SF | 50 | 49 | Success |
| Sweep | CC-MF | 1 | 0 | Success |
| Sweep | CC-MF | 2 | 1 | Success |
| Sweep | CC-MF | 3 | 2 | Success |
| Sweep | CC-MF | 4 | 3 | Success |
| Sweep | CC-MF | 5 | 4 | Success |
| Sweep | CC-MF | 6 | 5 | Success |
| Sweep | CC-MF | 7 | 6 | Success |
| Sweep | CC-MF | 8 | 7 | Success |
| Sweep | CC-MF | 9 | 8 | Success |
| Sweep | CC-MF | 10 | 9 | Success |
| Sweep | CC-MF | 11 | 10 | Success |
| Sweep | CC-MF | 12 | 11 | Success |
| Sweep | CC-MF | 13 | 12 | Success |
| Sweep | CC-MF | 14 | 13 | Success |
| Sweep | CC-MF | 15 | 14 | Success |
| Sweep | CC-MF | 16 | 15 | Success |
| Sweep | CC-MF | 17 | 16 | Success |
| Sweep | CC-MF | 18 | 17 | Success |
| Sweep | CC-MF | 19 | 18 | Success |
| Sweep | CC-MF | 20 | 19 | Success |
| Sweep | CC-MF | 21 | 20 | Success |
| Sweep | CC-MF | 22 | 21 | Success |
| Sweep | CC-MF | 23 | 22 | Success |
| Sweep | CC-MF | 24 | 23 | Success |
| Sweep | CC-MF | 25 | 24 | Success |
| Sweep | CC-MF | 26 | 25 | Success |
| Sweep | CC-MF | 27 | 26 | Success |
| Sweep | CC-MF | 28 | 27 | Success |
| Sweep | CC-MF | 29 | 28 | Success |
| Sweep | CC-MF | 30 | 29 | Success |
| Sweep | CC-MF | 31 | 30 | Success |
| Sweep | CC-MF | 32 | 31 | Success |
| Sweep | CC-MF | 33 | 32 | Success |
| Sweep | CC-MF | 34 | 33 | Success |
| Sweep | CC-MF | 35 | 34 | Success |
| Sweep | CC-MF | 36 | 35 | Feature-persistent |
| Sweep | CC-MF | 37 | 36 | Success |
| Sweep | CC-MF | 38 | 37 | Success |
| Sweep | CC-MF | 39 | 38 | Success |
| Sweep | CC-MF | 40 | 39 | Success |
| Sweep | CC-MF | 41 | 40 | Success |
| Sweep | CC-MF | 42 | 41 | Success |
| Sweep | CC-MF | 43 | 42 | Success |
| Sweep | CC-MF | 44 | 43 | Success |
| Sweep | CC-MF | 45 | 44 | Success |
| Sweep | CC-MF | 46 | 45 | Success |
| Sweep | CC-MF | 47 | 46 | Success |
| Sweep | CC-MF | 48 | 47 | Feature-persistent |
| Sweep | CC-MF | 49 | 48 | Success |
| Sweep | CC-MF | 50 | 49 | Success |

### DoN 0.79

| Cohort | Controller | Trial | Seed | Outcome |
| --- | --- | ---: | ---: | --- |
| Sweep | CC-SF | 1 | 0 | Success |
| Sweep | CC-SF | 2 | 1 | Success |
| Sweep | CC-SF | 3 | 2 | Success |
| Sweep | CC-SF | 4 | 3 | Success |
| Sweep | CC-SF | 5 | 4 | Success |
| Sweep | CC-SF | 6 | 5 | Success |
| Sweep | CC-SF | 7 | 6 | Success |
| Sweep | CC-SF | 8 | 7 | Success |
| Sweep | CC-SF | 9 | 8 | Success |
| Sweep | CC-SF | 10 | 9 | Success |
| Sweep | CC-SF | 11 | 10 | Success |
| Sweep | CC-SF | 12 | 11 | Success |
| Sweep | CC-SF | 13 | 12 | Success |
| Sweep | CC-SF | 14 | 13 | Success |
| Sweep | CC-SF | 15 | 14 | Success |
| Sweep | CC-SF | 16 | 15 | Success |
| Sweep | CC-SF | 17 | 16 | Success |
| Sweep | CC-SF | 18 | 17 | Success |
| Sweep | CC-SF | 19 | 18 | Success |
| Sweep | CC-SF | 20 | 19 | Success |
| Sweep | CC-SF | 21 | 20 | Success |
| Sweep | CC-SF | 22 | 21 | Success |
| Sweep | CC-SF | 23 | 22 | Success |
| Sweep | CC-SF | 24 | 23 | Success |
| Sweep | CC-SF | 25 | 24 | Success |
| Sweep | CC-SF | 26 | 25 | Success |
| Sweep | CC-SF | 27 | 26 | Success |
| Sweep | CC-SF | 28 | 27 | Success |
| Sweep | CC-SF | 29 | 28 | Success |
| Sweep | CC-SF | 30 | 29 | Success |
| Sweep | CC-SF | 31 | 30 | Success |
| Sweep | CC-SF | 32 | 31 | Success |
| Sweep | CC-SF | 33 | 32 | Success |
| Sweep | CC-SF | 34 | 33 | Success |
| Sweep | CC-SF | 35 | 34 | Success |
| Sweep | CC-SF | 36 | 35 | Success |
| Sweep | CC-SF | 37 | 36 | Success |
| Sweep | CC-SF | 38 | 37 | Success |
| Sweep | CC-SF | 39 | 38 | Success |
| Sweep | CC-SF | 40 | 39 | Success |
| Sweep | CC-SF | 41 | 40 | Success |
| Sweep | CC-SF | 42 | 41 | Success |
| Sweep | CC-SF | 43 | 42 | Success |
| Sweep | CC-SF | 44 | 43 | Success |
| Sweep | CC-SF | 45 | 44 | Success |
| Sweep | CC-SF | 46 | 45 | Success |
| Sweep | CC-SF | 47 | 46 | Success |
| Sweep | CC-SF | 48 | 47 | Success |
| Sweep | CC-SF | 49 | 48 | Success |
| Sweep | CC-SF | 50 | 49 | Success |
| Sweep | CC-MF | 1 | 0 | Success |
| Sweep | CC-MF | 2 | 1 | Success |
| Sweep | CC-MF | 3 | 2 | Success |
| Sweep | CC-MF | 4 | 3 | Success |
| Sweep | CC-MF | 5 | 4 | Success |
| Sweep | CC-MF | 6 | 5 | Success |
| Sweep | CC-MF | 7 | 6 | Success |
| Sweep | CC-MF | 8 | 7 | Success |
| Sweep | CC-MF | 9 | 8 | Success |
| Sweep | CC-MF | 10 | 9 | Success |
| Sweep | CC-MF | 11 | 10 | Success |
| Sweep | CC-MF | 12 | 11 | Success |
| Sweep | CC-MF | 13 | 12 | Success |
| Sweep | CC-MF | 14 | 13 | Success |
| Sweep | CC-MF | 15 | 14 | Success |
| Sweep | CC-MF | 16 | 15 | Success |
| Sweep | CC-MF | 17 | 16 | Success |
| Sweep | CC-MF | 18 | 17 | Success |
| Sweep | CC-MF | 19 | 18 | Success |
| Sweep | CC-MF | 20 | 19 | Success |
| Sweep | CC-MF | 21 | 20 | Success |
| Sweep | CC-MF | 22 | 21 | Success |
| Sweep | CC-MF | 23 | 22 | Success |
| Sweep | CC-MF | 24 | 23 | Success |
| Sweep | CC-MF | 25 | 24 | Success |
| Sweep | CC-MF | 26 | 25 | Success |
| Sweep | CC-MF | 27 | 26 | Success |
| Sweep | CC-MF | 28 | 27 | Success |
| Sweep | CC-MF | 29 | 28 | Success |
| Sweep | CC-MF | 30 | 29 | Success |
| Sweep | CC-MF | 31 | 30 | Success |
| Sweep | CC-MF | 32 | 31 | Success |
| Sweep | CC-MF | 33 | 32 | Success |
| Sweep | CC-MF | 34 | 33 | Success |
| Sweep | CC-MF | 35 | 34 | Success |
| Sweep | CC-MF | 36 | 35 | Success |
| Sweep | CC-MF | 37 | 36 | Success |
| Sweep | CC-MF | 38 | 37 | Success |
| Sweep | CC-MF | 39 | 38 | Success |
| Sweep | CC-MF | 40 | 39 | Success |
| Sweep | CC-MF | 41 | 40 | Success |
| Sweep | CC-MF | 42 | 41 | Success |
| Sweep | CC-MF | 43 | 42 | Success |
| Sweep | CC-MF | 44 | 43 | Success |
| Sweep | CC-MF | 45 | 44 | Success |
| Sweep | CC-MF | 46 | 45 | Success |
| Sweep | CC-MF | 47 | 46 | Success |
| Sweep | CC-MF | 48 | 47 | Success |
| Sweep | CC-MF | 49 | 48 | Success |
| Sweep | CC-MF | 50 | 49 | Success |

### DoN 0.80

| Cohort | Controller | Trial | Seed | Outcome |
| --- | --- | ---: | ---: | --- |
| Sweep | CC-SF | 1 | 0 | Success |
| Sweep | CC-SF | 2 | 1 | Success |
| Sweep | CC-SF | 3 | 2 | Success |
| Sweep | CC-SF | 4 | 3 | Success |
| Sweep | CC-SF | 5 | 4 | Success |
| Sweep | CC-SF | 6 | 5 | Success |
| Sweep | CC-SF | 7 | 6 | Success |
| Sweep | CC-SF | 8 | 7 | Success |
| Sweep | CC-SF | 9 | 8 | Success |
| Sweep | CC-SF | 10 | 9 | Success |
| Sweep | CC-SF | 11 | 10 | Success |
| Sweep | CC-SF | 12 | 11 | Success |
| Sweep | CC-SF | 13 | 12 | Success |
| Sweep | CC-SF | 14 | 13 | Success |
| Sweep | CC-SF | 15 | 14 | Success |
| Sweep | CC-SF | 16 | 15 | Success |
| Sweep | CC-SF | 17 | 16 | Success |
| Sweep | CC-SF | 18 | 17 | Success |
| Sweep | CC-SF | 19 | 18 | Success |
| Sweep | CC-SF | 20 | 19 | Success |
| Sweep | CC-SF | 21 | 20 | Success |
| Sweep | CC-SF | 22 | 21 | Success |
| Sweep | CC-SF | 23 | 22 | Success |
| Sweep | CC-SF | 24 | 23 | Success |
| Sweep | CC-SF | 25 | 24 | Success |
| Sweep | CC-SF | 26 | 25 | Success |
| Sweep | CC-SF | 27 | 26 | Success |
| Sweep | CC-SF | 28 | 27 | Success |
| Sweep | CC-SF | 29 | 28 | Success |
| Sweep | CC-SF | 30 | 29 | Success |
| Sweep | CC-SF | 31 | 30 | Success |
| Sweep | CC-SF | 32 | 31 | Success |
| Sweep | CC-SF | 33 | 32 | Success |
| Sweep | CC-SF | 34 | 33 | Success |
| Sweep | CC-SF | 35 | 34 | Success |
| Sweep | CC-SF | 36 | 35 | Success |
| Sweep | CC-SF | 37 | 36 | Success |
| Sweep | CC-SF | 38 | 37 | Success |
| Sweep | CC-SF | 39 | 38 | Success |
| Sweep | CC-SF | 40 | 39 | Success |
| Sweep | CC-SF | 41 | 40 | Success |
| Sweep | CC-SF | 42 | 41 | Success |
| Sweep | CC-SF | 43 | 42 | Success |
| Sweep | CC-SF | 44 | 43 | Success |
| Sweep | CC-SF | 45 | 44 | Success |
| Sweep | CC-SF | 46 | 45 | Success |
| Sweep | CC-SF | 47 | 46 | Success |
| Sweep | CC-SF | 48 | 47 | Success |
| Sweep | CC-SF | 49 | 48 | Success |
| Sweep | CC-SF | 50 | 49 | Success |
| Sweep | CC-MF | 1 | 0 | Success |
| Sweep | CC-MF | 2 | 1 | Success |
| Sweep | CC-MF | 3 | 2 | Success |
| Sweep | CC-MF | 4 | 3 | Success |
| Sweep | CC-MF | 5 | 4 | Success |
| Sweep | CC-MF | 6 | 5 | Success |
| Sweep | CC-MF | 7 | 6 | Success |
| Sweep | CC-MF | 8 | 7 | Success |
| Sweep | CC-MF | 9 | 8 | Success |
| Sweep | CC-MF | 10 | 9 | Success |
| Sweep | CC-MF | 11 | 10 | Success |
| Sweep | CC-MF | 12 | 11 | Success |
| Sweep | CC-MF | 13 | 12 | Success |
| Sweep | CC-MF | 14 | 13 | Success |
| Sweep | CC-MF | 15 | 14 | Success |
| Sweep | CC-MF | 16 | 15 | Success |
| Sweep | CC-MF | 17 | 16 | Success |
| Sweep | CC-MF | 18 | 17 | Success |
| Sweep | CC-MF | 19 | 18 | Success |
| Sweep | CC-MF | 20 | 19 | Success |
| Sweep | CC-MF | 21 | 20 | Success |
| Sweep | CC-MF | 22 | 21 | Success |
| Sweep | CC-MF | 23 | 22 | Success |
| Sweep | CC-MF | 24 | 23 | Success |
| Sweep | CC-MF | 25 | 24 | Success |
| Sweep | CC-MF | 26 | 25 | Success |
| Sweep | CC-MF | 27 | 26 | Success |
| Sweep | CC-MF | 28 | 27 | Success |
| Sweep | CC-MF | 29 | 28 | Success |
| Sweep | CC-MF | 30 | 29 | Success |
| Sweep | CC-MF | 31 | 30 | Success |
| Sweep | CC-MF | 32 | 31 | Success |
| Sweep | CC-MF | 33 | 32 | Success |
| Sweep | CC-MF | 34 | 33 | Success |
| Sweep | CC-MF | 35 | 34 | Success |
| Sweep | CC-MF | 36 | 35 | Success |
| Sweep | CC-MF | 37 | 36 | Success |
| Sweep | CC-MF | 38 | 37 | Success |
| Sweep | CC-MF | 39 | 38 | Feature-switching |
| Sweep | CC-MF | 40 | 39 | Success |
| Sweep | CC-MF | 41 | 40 | Success |
| Sweep | CC-MF | 42 | 41 | Success |
| Sweep | CC-MF | 43 | 42 | Success |
| Sweep | CC-MF | 44 | 43 | Success |
| Sweep | CC-MF | 45 | 44 | Success |
| Sweep | CC-MF | 46 | 45 | Success |
| Sweep | CC-MF | 47 | 46 | Success |
| Sweep | CC-MF | 48 | 47 | Success |
| Sweep | CC-MF | 49 | 48 | Success |
| Sweep | CC-MF | 50 | 49 | Success |
| Rerun | CC-MF | 1 | 1037497346 | Success |
| Rerun | CC-MF | 2 | 3325414130 | Success |
| Rerun | CC-MF | 3 | 1972572940 | Success |
| Rerun | CC-MF | 4 | 2991169048 | Success |
| Rerun | CC-MF | 5 | 711002714 | Success |

### DoN 0.81

| Cohort | Controller | Trial | Seed | Outcome |
| --- | --- | ---: | ---: | --- |
| Sweep | CC-SF | 1 | 0 | Success |
| Sweep | CC-SF | 2 | 1 | Success |
| Sweep | CC-SF | 3 | 2 | Success |
| Sweep | CC-SF | 4 | 3 | Success |
| Sweep | CC-SF | 5 | 4 | Success |
| Sweep | CC-SF | 6 | 5 | Success |
| Sweep | CC-SF | 7 | 6 | Success |
| Sweep | CC-SF | 8 | 7 | Success |
| Sweep | CC-SF | 9 | 8 | Success |
| Sweep | CC-SF | 10 | 9 | Success |
| Sweep | CC-SF | 11 | 10 | Success |
| Sweep | CC-SF | 12 | 11 | Success |
| Sweep | CC-SF | 13 | 12 | Success |
| Sweep | CC-SF | 14 | 13 | Success |
| Sweep | CC-SF | 15 | 14 | Success |
| Sweep | CC-SF | 16 | 15 | Success |
| Sweep | CC-SF | 17 | 16 | Success |
| Sweep | CC-SF | 18 | 17 | Success |
| Sweep | CC-SF | 19 | 18 | Success |
| Sweep | CC-SF | 20 | 19 | Success |
| Sweep | CC-SF | 21 | 20 | Success |
| Sweep | CC-SF | 22 | 21 | Success |
| Sweep | CC-SF | 23 | 22 | Success |
| Sweep | CC-SF | 24 | 23 | Success |
| Sweep | CC-SF | 25 | 24 | Success |
| Sweep | CC-SF | 26 | 25 | Success |
| Sweep | CC-SF | 27 | 26 | Success |
| Sweep | CC-SF | 28 | 27 | Success |
| Sweep | CC-SF | 29 | 28 | Success |
| Sweep | CC-SF | 30 | 29 | Success |
| Sweep | CC-SF | 31 | 30 | Success |
| Sweep | CC-SF | 32 | 31 | Success |
| Sweep | CC-SF | 33 | 32 | Success |
| Sweep | CC-SF | 34 | 33 | Success |
| Sweep | CC-SF | 35 | 34 | Success |
| Sweep | CC-SF | 36 | 35 | Success |
| Sweep | CC-SF | 37 | 36 | Success |
| Sweep | CC-SF | 38 | 37 | Success |
| Sweep | CC-SF | 39 | 38 | Success |
| Sweep | CC-SF | 40 | 39 | Success |
| Sweep | CC-SF | 41 | 40 | Success |
| Sweep | CC-SF | 42 | 41 | Success |
| Sweep | CC-SF | 43 | 42 | Success |
| Sweep | CC-SF | 44 | 43 | Success |
| Sweep | CC-SF | 45 | 44 | Success |
| Sweep | CC-SF | 46 | 45 | Success |
| Sweep | CC-SF | 47 | 46 | Success |
| Sweep | CC-SF | 48 | 47 | Success |
| Sweep | CC-SF | 49 | 48 | Success |
| Sweep | CC-SF | 50 | 49 | Success |
| Sweep | CC-MF | 1 | 0 | Success |
| Sweep | CC-MF | 2 | 1 | Success |
| Sweep | CC-MF | 3 | 2 | Success |
| Sweep | CC-MF | 4 | 3 | Success |
| Sweep | CC-MF | 5 | 4 | Feature-persistent |
| Sweep | CC-MF | 6 | 5 | Success |
| Sweep | CC-MF | 7 | 6 | Success |
| Sweep | CC-MF | 8 | 7 | Success |
| Sweep | CC-MF | 9 | 8 | Success |
| Sweep | CC-MF | 10 | 9 | Success |
| Sweep | CC-MF | 11 | 10 | Success |
| Sweep | CC-MF | 12 | 11 | Success |
| Sweep | CC-MF | 13 | 12 | Success |
| Sweep | CC-MF | 14 | 13 | Success |
| Sweep | CC-MF | 15 | 14 | Success |
| Sweep | CC-MF | 16 | 15 | Success |
| Sweep | CC-MF | 17 | 16 | Success |
| Sweep | CC-MF | 18 | 17 | Success |
| Sweep | CC-MF | 19 | 18 | Success |
| Sweep | CC-MF | 20 | 19 | Success |
| Sweep | CC-MF | 21 | 20 | Success |
| Sweep | CC-MF | 22 | 21 | Success |
| Sweep | CC-MF | 23 | 22 | Success |
| Sweep | CC-MF | 24 | 23 | Success |
| Sweep | CC-MF | 25 | 24 | Success |
| Sweep | CC-MF | 26 | 25 | Success |
| Sweep | CC-MF | 27 | 26 | Success |
| Sweep | CC-MF | 28 | 27 | Success |
| Sweep | CC-MF | 29 | 28 | Feature-switching |
| Sweep | CC-MF | 30 | 29 | Success |
| Sweep | CC-MF | 31 | 30 | Success |
| Sweep | CC-MF | 32 | 31 | Success |
| Sweep | CC-MF | 33 | 32 | Success |
| Sweep | CC-MF | 34 | 33 | Success |
| Sweep | CC-MF | 35 | 34 | Success |
| Sweep | CC-MF | 36 | 35 | Success |
| Sweep | CC-MF | 37 | 36 | Success |
| Sweep | CC-MF | 38 | 37 | Success |
| Sweep | CC-MF | 39 | 38 | Success |
| Sweep | CC-MF | 40 | 39 | Success |
| Sweep | CC-MF | 41 | 40 | Success |
| Sweep | CC-MF | 42 | 41 | Success |
| Sweep | CC-MF | 43 | 42 | Success |
| Sweep | CC-MF | 44 | 43 | Success |
| Sweep | CC-MF | 45 | 44 | Success |
| Sweep | CC-MF | 46 | 45 | Success |
| Sweep | CC-MF | 47 | 46 | Success |
| Sweep | CC-MF | 48 | 47 | Success |
| Sweep | CC-MF | 49 | 48 | Success |
| Sweep | CC-MF | 50 | 49 | Success |
| Rerun | CC-MF | 1 | 3421124405 | Success |
| Rerun | CC-MF | 2 | 2977814964 | Success |
| Rerun | CC-MF | 3 | 18080092 | Success |
| Rerun | CC-MF | 4 | 918373203 | Success |
| Rerun | CC-MF | 5 | 2756024056 | Success |

### DoN 0.82

| Cohort | Controller | Trial | Seed | Outcome |
| --- | --- | ---: | ---: | --- |
| Sweep | CC-SF | 1 | 0 | Success |
| Sweep | CC-SF | 2 | 1 | Success |
| Sweep | CC-SF | 3 | 2 | Success |
| Sweep | CC-SF | 4 | 3 | Success |
| Sweep | CC-SF | 5 | 4 | Success |
| Sweep | CC-SF | 6 | 5 | Success |
| Sweep | CC-SF | 7 | 6 | Success |
| Sweep | CC-SF | 8 | 7 | Success |
| Sweep | CC-SF | 9 | 8 | Success |
| Sweep | CC-SF | 10 | 9 | Success |
| Sweep | CC-SF | 11 | 10 | Success |
| Sweep | CC-SF | 12 | 11 | Success |
| Sweep | CC-SF | 13 | 12 | Success |
| Sweep | CC-SF | 14 | 13 | Success |
| Sweep | CC-SF | 15 | 14 | Success |
| Sweep | CC-SF | 16 | 15 | Feature-switching |
| Sweep | CC-SF | 17 | 16 | Success |
| Sweep | CC-SF | 18 | 17 | Success |
| Sweep | CC-SF | 19 | 18 | Success |
| Sweep | CC-SF | 20 | 19 | Success |
| Sweep | CC-SF | 21 | 20 | Success |
| Sweep | CC-SF | 22 | 21 | Feature-switching |
| Sweep | CC-SF | 23 | 22 | Success |
| Sweep | CC-SF | 24 | 23 | Success |
| Sweep | CC-SF | 25 | 24 | Success |
| Sweep | CC-SF | 26 | 25 | Success |
| Sweep | CC-SF | 27 | 26 | Success |
| Sweep | CC-SF | 28 | 27 | Feature-switching |
| Sweep | CC-SF | 29 | 28 | Success |
| Sweep | CC-SF | 30 | 29 | Success |
| Sweep | CC-SF | 31 | 30 | Success |
| Sweep | CC-SF | 32 | 31 | Success |
| Sweep | CC-SF | 33 | 32 | Success |
| Sweep | CC-SF | 34 | 33 | Success |
| Sweep | CC-SF | 35 | 34 | Success |
| Sweep | CC-SF | 36 | 35 | Success |
| Sweep | CC-SF | 37 | 36 | Success |
| Sweep | CC-SF | 38 | 37 | Success |
| Sweep | CC-SF | 39 | 38 | Success |
| Sweep | CC-SF | 40 | 39 | Success |
| Sweep | CC-SF | 41 | 40 | Success |
| Sweep | CC-SF | 42 | 41 | Success |
| Sweep | CC-SF | 43 | 42 | Success |
| Sweep | CC-SF | 44 | 43 | Success |
| Sweep | CC-SF | 45 | 44 | Success |
| Sweep | CC-SF | 46 | 45 | Success |
| Sweep | CC-SF | 47 | 46 | Success |
| Sweep | CC-SF | 48 | 47 | Success |
| Sweep | CC-SF | 49 | 48 | Success |
| Sweep | CC-SF | 50 | 49 | Success |
| Sweep | CC-MF | 1 | 0 | Success |
| Sweep | CC-MF | 2 | 1 | Success |
| Sweep | CC-MF | 3 | 2 | Feature-switching |
| Sweep | CC-MF | 4 | 3 | Success |
| Sweep | CC-MF | 5 | 4 | Success |
| Sweep | CC-MF | 6 | 5 | Success |
| Sweep | CC-MF | 7 | 6 | Success |
| Sweep | CC-MF | 8 | 7 | Success |
| Sweep | CC-MF | 9 | 8 | Feature-switching |
| Sweep | CC-MF | 10 | 9 | Success |
| Sweep | CC-MF | 11 | 10 | Success |
| Sweep | CC-MF | 12 | 11 | Success |
| Sweep | CC-MF | 13 | 12 | Success |
| Sweep | CC-MF | 14 | 13 | Success |
| Sweep | CC-MF | 15 | 14 | Success |
| Sweep | CC-MF | 16 | 15 | Success |
| Sweep | CC-MF | 17 | 16 | Success |
| Sweep | CC-MF | 18 | 17 | Success |
| Sweep | CC-MF | 19 | 18 | Success |
| Sweep | CC-MF | 20 | 19 | Success |
| Sweep | CC-MF | 21 | 20 | Success |
| Sweep | CC-MF | 22 | 21 | Success |
| Sweep | CC-MF | 23 | 22 | Success |
| Sweep | CC-MF | 24 | 23 | Success |
| Sweep | CC-MF | 25 | 24 | Success |
| Sweep | CC-MF | 26 | 25 | Success |
| Sweep | CC-MF | 27 | 26 | Success |
| Sweep | CC-MF | 28 | 27 | Success |
| Sweep | CC-MF | 29 | 28 | Success |
| Sweep | CC-MF | 30 | 29 | Success |
| Sweep | CC-MF | 31 | 30 | Success |
| Sweep | CC-MF | 32 | 31 | Success |
| Sweep | CC-MF | 33 | 32 | Success |
| Sweep | CC-MF | 34 | 33 | Feature-switching |
| Sweep | CC-MF | 35 | 34 | Success |
| Sweep | CC-MF | 36 | 35 | Success |
| Sweep | CC-MF | 37 | 36 | Feature-switching |
| Sweep | CC-MF | 38 | 37 | Feature-switching |
| Sweep | CC-MF | 39 | 38 | Success |
| Sweep | CC-MF | 40 | 39 | Success |
| Sweep | CC-MF | 41 | 40 | Success |
| Sweep | CC-MF | 42 | 41 | Feature-switching |
| Sweep | CC-MF | 43 | 42 | Success |
| Sweep | CC-MF | 44 | 43 | Feature-switching |
| Sweep | CC-MF | 45 | 44 | Success |
| Sweep | CC-MF | 46 | 45 | Feature-switching |
| Sweep | CC-MF | 47 | 46 | Success |
| Sweep | CC-MF | 48 | 47 | Success |
| Sweep | CC-MF | 49 | 48 | Success |
| Sweep | CC-MF | 50 | 49 | Success |
| Rerun | CC-MF | 1 | 2063807083 | Success |
| Rerun | CC-MF | 2 | 1176644122 | Success |
| Rerun | CC-MF | 3 | 1697645524 | Success |
| Rerun | CC-MF | 4 | 1301404181 | Success |
| Rerun | CC-MF | 5 | 3864669725 | Success |

### DoN 0.83

| Cohort | Controller | Trial | Seed | Outcome |
| --- | --- | ---: | ---: | --- |
| Sweep | CC-SF | 1 | 0 | Success |
| Sweep | CC-SF | 2 | 1 | Success |
| Sweep | CC-SF | 3 | 2 | Feature-switching |
| Sweep | CC-SF | 4 | 3 | Success |
| Sweep | CC-SF | 5 | 4 | Success |
| Sweep | CC-SF | 6 | 5 | Feature-switching |
| Sweep | CC-SF | 7 | 6 | Success |
| Sweep | CC-SF | 8 | 7 | Success |
| Sweep | CC-SF | 9 | 8 | Success |
| Sweep | CC-SF | 10 | 9 | Success |
| Sweep | CC-SF | 11 | 10 | Success |
| Sweep | CC-SF | 12 | 11 | Success |
| Sweep | CC-SF | 13 | 12 | Success |
| Sweep | CC-SF | 14 | 13 | Success |
| Sweep | CC-SF | 15 | 14 | Success |
| Sweep | CC-SF | 16 | 15 | Success |
| Sweep | CC-SF | 17 | 16 | Success |
| Sweep | CC-SF | 18 | 17 | Success |
| Sweep | CC-SF | 19 | 18 | Success |
| Sweep | CC-SF | 20 | 19 | Success |
| Sweep | CC-SF | 21 | 20 | Success |
| Sweep | CC-SF | 22 | 21 | Success |
| Sweep | CC-SF | 23 | 22 | Success |
| Sweep | CC-SF | 24 | 23 | Success |
| Sweep | CC-SF | 25 | 24 | Success |
| Sweep | CC-SF | 26 | 25 | Success |
| Sweep | CC-SF | 27 | 26 | Success |
| Sweep | CC-SF | 28 | 27 | Success |
| Sweep | CC-SF | 29 | 28 | Success |
| Sweep | CC-SF | 30 | 29 | Success |
| Sweep | CC-SF | 31 | 30 | Success |
| Sweep | CC-SF | 32 | 31 | Success |
| Sweep | CC-SF | 33 | 32 | Success |
| Sweep | CC-SF | 34 | 33 | Success |
| Sweep | CC-SF | 35 | 34 | Success |
| Sweep | CC-SF | 36 | 35 | Success |
| Sweep | CC-SF | 37 | 36 | Success |
| Sweep | CC-SF | 38 | 37 | Success |
| Sweep | CC-SF | 39 | 38 | Success |
| Sweep | CC-SF | 40 | 39 | Success |
| Sweep | CC-SF | 41 | 40 | Success |
| Sweep | CC-SF | 42 | 41 | Success |
| Sweep | CC-SF | 43 | 42 | Success |
| Sweep | CC-SF | 44 | 43 | Success |
| Sweep | CC-SF | 45 | 44 | Success |
| Sweep | CC-SF | 46 | 45 | Success |
| Sweep | CC-SF | 47 | 46 | Success |
| Sweep | CC-SF | 48 | 47 | Success |
| Sweep | CC-SF | 49 | 48 | Success |
| Sweep | CC-SF | 50 | 49 | Feature-switching |
| Sweep | CC-MF | 1 | 0 | Success |
| Sweep | CC-MF | 2 | 1 | Success |
| Sweep | CC-MF | 3 | 2 | Success |
| Sweep | CC-MF | 4 | 3 | Success |
| Sweep | CC-MF | 5 | 4 | Success |
| Sweep | CC-MF | 6 | 5 | Success |
| Sweep | CC-MF | 7 | 6 | Success |
| Sweep | CC-MF | 8 | 7 | Feature-switching |
| Sweep | CC-MF | 9 | 8 | Success |
| Sweep | CC-MF | 10 | 9 | Success |
| Sweep | CC-MF | 11 | 10 | Success |
| Sweep | CC-MF | 12 | 11 | Success |
| Sweep | CC-MF | 13 | 12 | Success |
| Sweep | CC-MF | 14 | 13 | Success |
| Sweep | CC-MF | 15 | 14 | Success |
| Sweep | CC-MF | 16 | 15 | Success |
| Sweep | CC-MF | 17 | 16 | Success |
| Sweep | CC-MF | 18 | 17 | Success |
| Sweep | CC-MF | 19 | 18 | Success |
| Sweep | CC-MF | 20 | 19 | Success |
| Sweep | CC-MF | 21 | 20 | Success |
| Sweep | CC-MF | 22 | 21 | Feature-switching |
| Sweep | CC-MF | 23 | 22 | Success |
| Sweep | CC-MF | 24 | 23 | Success |
| Sweep | CC-MF | 25 | 24 | Success |
| Sweep | CC-MF | 26 | 25 | Success |
| Sweep | CC-MF | 27 | 26 | Success |
| Sweep | CC-MF | 28 | 27 | Success |
| Sweep | CC-MF | 29 | 28 | Success |
| Sweep | CC-MF | 30 | 29 | Success |
| Sweep | CC-MF | 31 | 30 | Success |
| Sweep | CC-MF | 32 | 31 | Success |
| Sweep | CC-MF | 33 | 32 | Success |
| Sweep | CC-MF | 34 | 33 | Success |
| Sweep | CC-MF | 35 | 34 | Success |
| Sweep | CC-MF | 36 | 35 | Success |
| Sweep | CC-MF | 37 | 36 | Feature-switching |
| Sweep | CC-MF | 38 | 37 | Success |
| Sweep | CC-MF | 39 | 38 | Success |
| Sweep | CC-MF | 40 | 39 | Success |
| Sweep | CC-MF | 41 | 40 | Success |
| Sweep | CC-MF | 42 | 41 | Success |
| Sweep | CC-MF | 43 | 42 | Success |
| Sweep | CC-MF | 44 | 43 | Success |
| Sweep | CC-MF | 45 | 44 | Success |
| Sweep | CC-MF | 46 | 45 | Success |
| Sweep | CC-MF | 47 | 46 | Success |
| Sweep | CC-MF | 48 | 47 | Feature-switching |
| Sweep | CC-MF | 49 | 48 | Success |
| Sweep | CC-MF | 50 | 49 | Success |
| Rerun | CC-MF | 1 | 1689483296 | Success |
| Rerun | CC-MF | 2 | 2404568397 | Success |
| Rerun | CC-MF | 3 | 814651749 | Success |
| Rerun | CC-MF | 4 | 346266070 | Success |
| Rerun | CC-MF | 5 | 2904948481 | Success |

### DoN 0.84

| Cohort | Controller | Trial | Seed | Outcome |
| --- | --- | ---: | ---: | --- |
| Sweep | CC-SF | 1 | 0 | Feature-switching |
| Sweep | CC-SF | 2 | 1 | Feature-switching |
| Sweep | CC-SF | 3 | 2 | Feature-switching |
| Sweep | CC-SF | 4 | 3 | Success |
| Sweep | CC-SF | 5 | 4 | Success |
| Sweep | CC-SF | 6 | 5 | Feature-switching |
| Sweep | CC-SF | 7 | 6 | Success |
| Sweep | CC-SF | 8 | 7 | Success |
| Sweep | CC-SF | 9 | 8 | Success |
| Sweep | CC-SF | 10 | 9 | Feature-switching |
| Sweep | CC-SF | 11 | 10 | Success |
| Sweep | CC-SF | 12 | 11 | Feature-switching |
| Sweep | CC-SF | 13 | 12 | Success |
| Sweep | CC-SF | 14 | 13 | Feature-switching |
| Sweep | CC-SF | 15 | 14 | Feature-switching |
| Sweep | CC-SF | 16 | 15 | Feature-switching |
| Sweep | CC-SF | 17 | 16 | Success |
| Sweep | CC-SF | 18 | 17 | Feature-switching |
| Sweep | CC-SF | 19 | 18 | Success |
| Sweep | CC-SF | 20 | 19 | Success |
| Sweep | CC-SF | 21 | 20 | Success |
| Sweep | CC-SF | 22 | 21 | Feature-switching |
| Sweep | CC-SF | 23 | 22 | Feature-switching |
| Sweep | CC-SF | 24 | 23 | Success |
| Sweep | CC-SF | 25 | 24 | Feature-switching |
| Sweep | CC-SF | 26 | 25 | Success |
| Sweep | CC-SF | 27 | 26 | Success |
| Sweep | CC-SF | 28 | 27 | Success |
| Sweep | CC-SF | 29 | 28 | Success |
| Sweep | CC-SF | 30 | 29 | Success |
| Sweep | CC-SF | 31 | 30 | Success |
| Sweep | CC-SF | 32 | 31 | Success |
| Sweep | CC-SF | 33 | 32 | Feature-switching |
| Sweep | CC-SF | 34 | 33 | Success |
| Sweep | CC-SF | 35 | 34 | Success |
| Sweep | CC-SF | 36 | 35 | Success |
| Sweep | CC-SF | 37 | 36 | Success |
| Sweep | CC-SF | 38 | 37 | Success |
| Sweep | CC-SF | 39 | 38 | Feature-switching |
| Sweep | CC-SF | 40 | 39 | Success |
| Sweep | CC-SF | 41 | 40 | Success |
| Sweep | CC-SF | 42 | 41 | Success |
| Sweep | CC-SF | 43 | 42 | Feature-switching |
| Sweep | CC-SF | 44 | 43 | Success |
| Sweep | CC-SF | 45 | 44 | Success |
| Sweep | CC-SF | 46 | 45 | Success |
| Sweep | CC-SF | 47 | 46 | Success |
| Sweep | CC-SF | 48 | 47 | Success |
| Sweep | CC-SF | 49 | 48 | Success |
| Sweep | CC-SF | 50 | 49 | Success |
| Sweep | CC-MF | 1 | 0 | Success |
| Sweep | CC-MF | 2 | 1 | Success |
| Sweep | CC-MF | 3 | 2 | Success |
| Sweep | CC-MF | 4 | 3 | Success |
| Sweep | CC-MF | 5 | 4 | Success |
| Sweep | CC-MF | 6 | 5 | Success |
| Sweep | CC-MF | 7 | 6 | Success |
| Sweep | CC-MF | 8 | 7 | Success |
| Sweep | CC-MF | 9 | 8 | Success |
| Sweep | CC-MF | 10 | 9 | Success |
| Sweep | CC-MF | 11 | 10 | Success |
| Sweep | CC-MF | 12 | 11 | Success |
| Sweep | CC-MF | 13 | 12 | Success |
| Sweep | CC-MF | 14 | 13 | Success |
| Sweep | CC-MF | 15 | 14 | Success |
| Sweep | CC-MF | 16 | 15 | Success |
| Sweep | CC-MF | 17 | 16 | Success |
| Sweep | CC-MF | 18 | 17 | Success |
| Sweep | CC-MF | 19 | 18 | Success |
| Sweep | CC-MF | 20 | 19 | Success |
| Sweep | CC-MF | 21 | 20 | Success |
| Sweep | CC-MF | 22 | 21 | Success |
| Sweep | CC-MF | 23 | 22 | Success |
| Sweep | CC-MF | 24 | 23 | Success |
| Sweep | CC-MF | 25 | 24 | Success |
| Sweep | CC-MF | 26 | 25 | Success |
| Sweep | CC-MF | 27 | 26 | Success |
| Sweep | CC-MF | 28 | 27 | Success |
| Sweep | CC-MF | 29 | 28 | Success |
| Sweep | CC-MF | 30 | 29 | Success |
| Sweep | CC-MF | 31 | 30 | Success |
| Sweep | CC-MF | 32 | 31 | Success |
| Sweep | CC-MF | 33 | 32 | Success |
| Sweep | CC-MF | 34 | 33 | Success |
| Sweep | CC-MF | 35 | 34 | Success |
| Sweep | CC-MF | 36 | 35 | Success |
| Sweep | CC-MF | 37 | 36 | Success |
| Sweep | CC-MF | 38 | 37 | Feature-persistent |
| Sweep | CC-MF | 39 | 38 | Success |
| Sweep | CC-MF | 40 | 39 | Success |
| Sweep | CC-MF | 41 | 40 | Success |
| Sweep | CC-MF | 42 | 41 | Success |
| Sweep | CC-MF | 43 | 42 | Success |
| Sweep | CC-MF | 44 | 43 | Success |
| Sweep | CC-MF | 45 | 44 | Success |
| Sweep | CC-MF | 46 | 45 | Success |
| Sweep | CC-MF | 47 | 46 | Success |
| Sweep | CC-MF | 48 | 47 | Success |
| Sweep | CC-MF | 49 | 48 | Success |
| Sweep | CC-MF | 50 | 49 | Success |

### DoN 0.85

| Cohort | Controller | Trial | Seed | Outcome |
| --- | --- | ---: | ---: | --- |
| Sweep | CC-SF | 1 | 0 | Feature-switching |
| Sweep | CC-SF | 2 | 1 | Success |
| Sweep | CC-SF | 3 | 2 | Feature-switching |
| Sweep | CC-SF | 4 | 3 | Success |
| Sweep | CC-SF | 5 | 4 | Success |
| Sweep | CC-SF | 6 | 5 | Feature-switching |
| Sweep | CC-SF | 7 | 6 | Feature-switching |
| Sweep | CC-SF | 8 | 7 | Success |
| Sweep | CC-SF | 9 | 8 | Success |
| Sweep | CC-SF | 10 | 9 | Success |
| Sweep | CC-SF | 11 | 10 | Feature-switching |
| Sweep | CC-SF | 12 | 11 | Feature-switching |
| Sweep | CC-SF | 13 | 12 | Success |
| Sweep | CC-SF | 14 | 13 | Feature-switching |
| Sweep | CC-SF | 15 | 14 | Success |
| Sweep | CC-SF | 16 | 15 | Success |
| Sweep | CC-SF | 17 | 16 | Feature-switching |
| Sweep | CC-SF | 18 | 17 | Success |
| Sweep | CC-SF | 19 | 18 | Feature-switching |
| Sweep | CC-SF | 20 | 19 | Success |
| Sweep | CC-SF | 21 | 20 | Feature-switching |
| Sweep | CC-SF | 22 | 21 | Feature-switching |
| Sweep | CC-SF | 23 | 22 | Feature-switching |
| Sweep | CC-SF | 24 | 23 | Success |
| Sweep | CC-SF | 25 | 24 | Success |
| Sweep | CC-SF | 26 | 25 | Success |
| Sweep | CC-SF | 27 | 26 | Feature-switching |
| Sweep | CC-SF | 28 | 27 | Success |
| Sweep | CC-SF | 29 | 28 | Success |
| Sweep | CC-SF | 30 | 29 | Success |
| Sweep | CC-SF | 31 | 30 | Success |
| Sweep | CC-SF | 32 | 31 | Feature-switching |
| Sweep | CC-SF | 33 | 32 | Success |
| Sweep | CC-SF | 34 | 33 | Success |
| Sweep | CC-SF | 35 | 34 | Feature-switching |
| Sweep | CC-SF | 36 | 35 | Success |
| Sweep | CC-SF | 37 | 36 | Success |
| Sweep | CC-SF | 38 | 37 | Success |
| Sweep | CC-SF | 39 | 38 | Feature-switching |
| Sweep | CC-SF | 40 | 39 | Success |
| Sweep | CC-SF | 41 | 40 | Success |
| Sweep | CC-SF | 42 | 41 | Success |
| Sweep | CC-SF | 43 | 42 | Feature-switching |
| Sweep | CC-SF | 44 | 43 | Success |
| Sweep | CC-SF | 45 | 44 | Success |
| Sweep | CC-SF | 46 | 45 | Success |
| Sweep | CC-SF | 47 | 46 | Success |
| Sweep | CC-SF | 48 | 47 | Success |
| Sweep | CC-SF | 49 | 48 | Feature-switching |
| Sweep | CC-SF | 50 | 49 | Success |
| Sweep | CC-MF | 1 | 0 | Success |
| Sweep | CC-MF | 2 | 1 | Success |
| Sweep | CC-MF | 3 | 2 | Success |
| Sweep | CC-MF | 4 | 3 | Success |
| Sweep | CC-MF | 5 | 4 | Success |
| Sweep | CC-MF | 6 | 5 | Feature-switching |
| Sweep | CC-MF | 7 | 6 | Success |
| Sweep | CC-MF | 8 | 7 | Success |
| Sweep | CC-MF | 9 | 8 | Success |
| Sweep | CC-MF | 10 | 9 | Success |
| Sweep | CC-MF | 11 | 10 | Success |
| Sweep | CC-MF | 12 | 11 | Success |
| Sweep | CC-MF | 13 | 12 | Success |
| Sweep | CC-MF | 14 | 13 | Success |
| Sweep | CC-MF | 15 | 14 | Success |
| Sweep | CC-MF | 16 | 15 | Success |
| Sweep | CC-MF | 17 | 16 | Success |
| Sweep | CC-MF | 18 | 17 | Success |
| Sweep | CC-MF | 19 | 18 | Success |
| Sweep | CC-MF | 20 | 19 | Success |
| Sweep | CC-MF | 21 | 20 | Success |
| Sweep | CC-MF | 22 | 21 | Success |
| Sweep | CC-MF | 23 | 22 | Success |
| Sweep | CC-MF | 24 | 23 | Feature-switching |
| Sweep | CC-MF | 25 | 24 | Feature-switching |
| Sweep | CC-MF | 26 | 25 | Success |
| Sweep | CC-MF | 27 | 26 | Success |
| Sweep | CC-MF | 28 | 27 | Success |
| Sweep | CC-MF | 29 | 28 | Success |
| Sweep | CC-MF | 30 | 29 | Success |
| Sweep | CC-MF | 31 | 30 | Success |
| Sweep | CC-MF | 32 | 31 | Success |
| Sweep | CC-MF | 33 | 32 | Success |
| Sweep | CC-MF | 34 | 33 | Success |
| Sweep | CC-MF | 35 | 34 | Success |
| Sweep | CC-MF | 36 | 35 | Success |
| Sweep | CC-MF | 37 | 36 | Success |
| Sweep | CC-MF | 38 | 37 | Feature-switching |
| Sweep | CC-MF | 39 | 38 | Success |
| Sweep | CC-MF | 40 | 39 | Feature-switching |
| Sweep | CC-MF | 41 | 40 | Success |
| Sweep | CC-MF | 42 | 41 | Success |
| Sweep | CC-MF | 43 | 42 | Success |
| Sweep | CC-MF | 44 | 43 | Success |
| Sweep | CC-MF | 45 | 44 | Success |
| Sweep | CC-MF | 46 | 45 | Success |
| Sweep | CC-MF | 47 | 46 | Success |
| Sweep | CC-MF | 48 | 47 | Success |
| Sweep | CC-MF | 49 | 48 | Success |
| Sweep | CC-MF | 50 | 49 | Success |

### DoN 0.86

| Cohort | Controller | Trial | Seed | Outcome |
| --- | --- | ---: | ---: | --- |
| Sweep | CC-SF | 1 | 0 | Feature-switching |
| Sweep | CC-SF | 2 | 1 | Feature-switching |
| Sweep | CC-SF | 3 | 2 | Success |
| Sweep | CC-SF | 4 | 3 | Feature-switching |
| Sweep | CC-SF | 5 | 4 | Success |
| Sweep | CC-SF | 6 | 5 | Feature-switching |
| Sweep | CC-SF | 7 | 6 | Feature-switching |
| Sweep | CC-SF | 8 | 7 | Feature-switching |
| Sweep | CC-SF | 9 | 8 | Feature-switching |
| Sweep | CC-SF | 10 | 9 | Feature-switching |
| Sweep | CC-SF | 11 | 10 | Feature-switching |
| Sweep | CC-SF | 12 | 11 | Feature-switching |
| Sweep | CC-SF | 13 | 12 | Feature-switching |
| Sweep | CC-SF | 14 | 13 | Feature-switching |
| Sweep | CC-SF | 15 | 14 | Feature-switching |
| Sweep | CC-SF | 16 | 15 | Feature-switching |
| Sweep | CC-SF | 17 | 16 | Feature-switching |
| Sweep | CC-SF | 18 | 17 | Feature-switching |
| Sweep | CC-SF | 19 | 18 | Feature-switching |
| Sweep | CC-SF | 20 | 19 | Feature-switching |
| Sweep | CC-SF | 21 | 20 | Feature-switching |
| Sweep | CC-SF | 22 | 21 | Feature-switching |
| Sweep | CC-SF | 23 | 22 | Feature-switching |
| Sweep | CC-SF | 24 | 23 | Feature-switching |
| Sweep | CC-SF | 25 | 24 | Feature-switching |
| Sweep | CC-SF | 26 | 25 | Feature-switching |
| Sweep | CC-SF | 27 | 26 | Feature-switching |
| Sweep | CC-SF | 28 | 27 | Feature-switching |
| Sweep | CC-SF | 29 | 28 | Success |
| Sweep | CC-SF | 30 | 29 | Feature-switching |
| Sweep | CC-SF | 31 | 30 | Feature-switching |
| Sweep | CC-SF | 32 | 31 | Feature-switching |
| Sweep | CC-SF | 33 | 32 | Feature-switching |
| Sweep | CC-SF | 34 | 33 | Feature-switching |
| Sweep | CC-SF | 35 | 34 | Feature-switching |
| Sweep | CC-SF | 36 | 35 | Feature-switching |
| Sweep | CC-SF | 37 | 36 | Feature-switching |
| Sweep | CC-SF | 38 | 37 | Feature-switching |
| Sweep | CC-SF | 39 | 38 | Feature-switching |
| Sweep | CC-SF | 40 | 39 | Feature-switching |
| Sweep | CC-SF | 41 | 40 | Feature-switching |
| Sweep | CC-SF | 42 | 41 | Feature-switching |
| Sweep | CC-SF | 43 | 42 | Feature-switching |
| Sweep | CC-SF | 44 | 43 | Feature-switching |
| Sweep | CC-SF | 45 | 44 | Feature-persistent |
| Sweep | CC-SF | 46 | 45 | Feature-switching |
| Sweep | CC-SF | 47 | 46 | Success |
| Sweep | CC-SF | 48 | 47 | Feature-switching |
| Sweep | CC-SF | 49 | 48 | Feature-switching |
| Sweep | CC-SF | 50 | 49 | Feature-switching |
| Sweep | CC-MF | 1 | 0 | Feature-switching |
| Sweep | CC-MF | 2 | 1 | Feature-switching |
| Sweep | CC-MF | 3 | 2 | Success |
| Sweep | CC-MF | 4 | 3 | Feature-switching |
| Sweep | CC-MF | 5 | 4 | Success |
| Sweep | CC-MF | 6 | 5 | Feature-persistent |
| Sweep | CC-MF | 7 | 6 | Success |
| Sweep | CC-MF | 8 | 7 | Feature-switching |
| Sweep | CC-MF | 9 | 8 | Feature-switching |
| Sweep | CC-MF | 10 | 9 | Success |
| Sweep | CC-MF | 11 | 10 | Success |
| Sweep | CC-MF | 12 | 11 | Success |
| Sweep | CC-MF | 13 | 12 | Feature-switching |
| Sweep | CC-MF | 14 | 13 | Success |
| Sweep | CC-MF | 15 | 14 | Feature-switching |
| Sweep | CC-MF | 16 | 15 | Success |
| Sweep | CC-MF | 17 | 16 | Feature-switching |
| Sweep | CC-MF | 18 | 17 | Feature-switching |
| Sweep | CC-MF | 19 | 18 | Feature-switching |
| Sweep | CC-MF | 20 | 19 | Feature-switching |
| Sweep | CC-MF | 21 | 20 | Success |
| Sweep | CC-MF | 22 | 21 | Feature-switching |
| Sweep | CC-MF | 23 | 22 | Success |
| Sweep | CC-MF | 24 | 23 | Feature-switching |
| Sweep | CC-MF | 25 | 24 | Success |
| Sweep | CC-MF | 26 | 25 | Success |
| Sweep | CC-MF | 27 | 26 | Feature-switching |
| Sweep | CC-MF | 28 | 27 | Feature-persistent |
| Sweep | CC-MF | 29 | 28 | Success |
| Sweep | CC-MF | 30 | 29 | Success |
| Sweep | CC-MF | 31 | 30 | Feature-switching |
| Sweep | CC-MF | 32 | 31 | Feature-switching |
| Sweep | CC-MF | 33 | 32 | Feature-switching |
| Sweep | CC-MF | 34 | 33 | Feature-switching |
| Sweep | CC-MF | 35 | 34 | Feature-switching |
| Sweep | CC-MF | 36 | 35 | Feature-switching |
| Sweep | CC-MF | 37 | 36 | Feature-switching |
| Sweep | CC-MF | 38 | 37 | Success |
| Sweep | CC-MF | 39 | 38 | Feature-switching |
| Sweep | CC-MF | 40 | 39 | Feature-switching |
| Sweep | CC-MF | 41 | 40 | Feature-switching |
| Sweep | CC-MF | 42 | 41 | Feature-persistent |
| Sweep | CC-MF | 43 | 42 | Success |
| Sweep | CC-MF | 44 | 43 | Feature-switching |
| Sweep | CC-MF | 45 | 44 | Success |
| Sweep | CC-MF | 46 | 45 | Feature-switching |
| Sweep | CC-MF | 47 | 46 | Feature-persistent |
| Sweep | CC-MF | 48 | 47 | Success |
| Sweep | CC-MF | 49 | 48 | Success |
| Sweep | CC-MF | 50 | 49 | Success |

### DoN 0.87

| Cohort | Controller | Trial | Seed | Outcome |
| --- | --- | ---: | ---: | --- |
| Sweep | CC-SF | 1 | 0 | Feature-switching |
| Sweep | CC-SF | 2 | 1 | Feature-switching |
| Sweep | CC-SF | 3 | 2 | Feature-switching |
| Sweep | CC-SF | 4 | 3 | Feature-switching |
| Sweep | CC-SF | 5 | 4 | Feature-switching |
| Sweep | CC-SF | 6 | 5 | Feature-switching |
| Sweep | CC-SF | 7 | 6 | Feature-switching |
| Sweep | CC-SF | 8 | 7 | Feature-switching |
| Sweep | CC-SF | 9 | 8 | Feature-switching |
| Sweep | CC-SF | 10 | 9 | Feature-switching |
| Sweep | CC-SF | 11 | 10 | Feature-switching |
| Sweep | CC-SF | 12 | 11 | Feature-switching |
| Sweep | CC-SF | 13 | 12 | Feature-switching |
| Sweep | CC-SF | 14 | 13 | Feature-switching |
| Sweep | CC-SF | 15 | 14 | Feature-switching |
| Sweep | CC-SF | 16 | 15 | Feature-switching |
| Sweep | CC-SF | 17 | 16 | Feature-switching |
| Sweep | CC-SF | 18 | 17 | Feature-switching |
| Sweep | CC-SF | 19 | 18 | Feature-switching |
| Sweep | CC-SF | 20 | 19 | Feature-switching |
| Sweep | CC-SF | 21 | 20 | Feature-switching |
| Sweep | CC-SF | 22 | 21 | Feature-switching |
| Sweep | CC-SF | 23 | 22 | Feature-switching |
| Sweep | CC-SF | 24 | 23 | Feature-switching |
| Sweep | CC-SF | 25 | 24 | Feature-switching |
| Sweep | CC-SF | 26 | 25 | Feature-switching |
| Sweep | CC-SF | 27 | 26 | Feature-switching |
| Sweep | CC-SF | 28 | 27 | Feature-switching |
| Sweep | CC-SF | 29 | 28 | Feature-switching |
| Sweep | CC-SF | 30 | 29 | Feature-switching |
| Sweep | CC-SF | 31 | 30 | Feature-switching |
| Sweep | CC-SF | 32 | 31 | Feature-switching |
| Sweep | CC-SF | 33 | 32 | Feature-switching |
| Sweep | CC-SF | 34 | 33 | Feature-switching |
| Sweep | CC-SF | 35 | 34 | Feature-switching |
| Sweep | CC-SF | 36 | 35 | Feature-switching |
| Sweep | CC-SF | 37 | 36 | Feature-switching |
| Sweep | CC-SF | 38 | 37 | Feature-switching |
| Sweep | CC-SF | 39 | 38 | Feature-switching |
| Sweep | CC-SF | 40 | 39 | Feature-switching |
| Sweep | CC-SF | 41 | 40 | Feature-switching |
| Sweep | CC-SF | 42 | 41 | Feature-switching |
| Sweep | CC-SF | 43 | 42 | Feature-switching |
| Sweep | CC-SF | 44 | 43 | Feature-switching |
| Sweep | CC-SF | 45 | 44 | Feature-switching |
| Sweep | CC-SF | 46 | 45 | Feature-switching |
| Sweep | CC-SF | 47 | 46 | Feature-switching |
| Sweep | CC-SF | 48 | 47 | Feature-switching |
| Sweep | CC-SF | 49 | 48 | Feature-switching |
| Sweep | CC-SF | 50 | 49 | Feature-switching |
| Sweep | CC-MF | 1 | 0 | Feature-switching |
| Sweep | CC-MF | 2 | 1 | Feature-switching |
| Sweep | CC-MF | 3 | 2 | Feature-switching |
| Sweep | CC-MF | 4 | 3 | Feature-switching |
| Sweep | CC-MF | 5 | 4 | Feature-switching |
| Sweep | CC-MF | 6 | 5 | Feature-switching |
| Sweep | CC-MF | 7 | 6 | Feature-switching |
| Sweep | CC-MF | 8 | 7 | Feature-switching |
| Sweep | CC-MF | 9 | 8 | Feature-switching |
| Sweep | CC-MF | 10 | 9 | Feature-switching |
| Sweep | CC-MF | 11 | 10 | Feature-switching |
| Sweep | CC-MF | 12 | 11 | Feature-switching |
| Sweep | CC-MF | 13 | 12 | Feature-switching |
| Sweep | CC-MF | 14 | 13 | Feature-switching |
| Sweep | CC-MF | 15 | 14 | Feature-switching |
| Sweep | CC-MF | 16 | 15 | Feature-switching |
| Sweep | CC-MF | 17 | 16 | Feature-switching |
| Sweep | CC-MF | 18 | 17 | Feature-persistent |
| Sweep | CC-MF | 19 | 18 | Feature-switching |
| Sweep | CC-MF | 20 | 19 | Feature-switching |
| Sweep | CC-MF | 21 | 20 | Feature-switching |
| Sweep | CC-MF | 22 | 21 | Feature-switching |
| Sweep | CC-MF | 23 | 22 | Feature-switching |
| Sweep | CC-MF | 24 | 23 | Feature-switching |
| Sweep | CC-MF | 25 | 24 | Feature-switching |
| Sweep | CC-MF | 26 | 25 | Feature-switching |
| Sweep | CC-MF | 27 | 26 | Feature-switching |
| Sweep | CC-MF | 28 | 27 | Feature-switching |
| Sweep | CC-MF | 29 | 28 | Feature-persistent |
| Sweep | CC-MF | 30 | 29 | Feature-switching |
| Sweep | CC-MF | 31 | 30 | Feature-switching |
| Sweep | CC-MF | 32 | 31 | Feature-switching |
| Sweep | CC-MF | 33 | 32 | Feature-switching |
| Sweep | CC-MF | 34 | 33 | Feature-switching |
| Sweep | CC-MF | 35 | 34 | Feature-switching |
| Sweep | CC-MF | 36 | 35 | Feature-switching |
| Sweep | CC-MF | 37 | 36 | Feature-switching |
| Sweep | CC-MF | 38 | 37 | Feature-switching |
| Sweep | CC-MF | 39 | 38 | Feature-switching |
| Sweep | CC-MF | 40 | 39 | Feature-switching |
| Sweep | CC-MF | 41 | 40 | Feature-switching |
| Sweep | CC-MF | 42 | 41 | Feature-switching |
| Sweep | CC-MF | 43 | 42 | Feature-switching |
| Sweep | CC-MF | 44 | 43 | Feature-persistent |
| Sweep | CC-MF | 45 | 44 | Feature-switching |
| Sweep | CC-MF | 46 | 45 | Feature-switching |
| Sweep | CC-MF | 47 | 46 | Feature-switching |
| Sweep | CC-MF | 48 | 47 | Feature-switching |
| Sweep | CC-MF | 49 | 48 | Feature-persistent |
| Sweep | CC-MF | 50 | 49 | Feature-switching |

## PNG 재생성

이 파일이 있는 디렉토리에서 아래 Python 코드를 실행한다. NumPy와 Matplotlib이 필요하다. 코드는 위 trial 표를 직접 읽어 같은 이름의 PNG를 생성한다. 저장된 그림에는 Matplotlib 3.7.5와 DejaVu Sans를 사용했다.

```python
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

report = Path("trial_failure_types.md")
counts = defaultdict(Counter)
don = None
for line in report.read_text(encoding="utf-8").splitlines():
    if line.startswith("### DoN "):
        don = line.split()[-1]
    if line.startswith(("| Sweep |", "| Rerun |")):
        cohort, method, trial, seed, outcome = (value.strip() for value in line[1:-1].split("|"))
        counts[(don, cohort, method)][outcome] += 1

dons = [f"0.{value}" for value in range(83, 88)]
outcomes = (
    ("Feature-switching", "#C46945", ""),
    ("Feature-persistent", "#35688E", "///"),
    ("Success", "#8EBC91", ""),
)
rows = [
    counts[don, "Rerun" if (don, method) == ("0.83", "CC-MF") else "Sweep", method]
    for don in dons for method in ("CC-SF", "CC-MF")
]
sample_sizes = np.array([sum(row.values()) for row in rows])
data = np.array([[row[label] for label, _, _ in outcomes] for row in rows])
assert sample_sizes.tolist() == [50, 5] + [50] * 8
np.testing.assert_equal(data.sum(axis=1), sample_sizes)
percentages = 100 * data / sample_sizes[:, None]

ink, edge = "#20242B", "#2C333B"
centers = np.arange(len(dons))
positions = (centers[:, None] + [-0.178, 0.178]).ravel()
with plt.rc_context({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": ink,
    "axes.labelcolor": ink, "axes.edgecolor": edge, "axes.linewidth": 0.65,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": ink, "ytick.color": ink, "hatch.linewidth": 0.46,
}):
    fig, ax = plt.subplots(figsize=(4.1, 2.12), facecolor="white")
    fig.subplots_adjust(left=0.137, right=0.987, bottom=0.255, top=0.847)
    bottom = np.zeros(len(positions))
    legend = []
    for index, (label, fill, hatch) in enumerate(outcomes):
        height = percentages[:, index]
        visible = height > 0
        ax.bar(positions[visible], height[visible], width=0.321, bottom=bottom[visible],
               color=fill, edgecolor=edge, linewidth=0.55, hatch=hatch, zorder=3)
        bottom += height
        legend.append(Patch(facecolor=fill, edgecolor=edge, linewidth=0.55,
                            hatch=hatch, label=label))

    ax.set_xlim(-0.44, len(dons) - 0.56)
    ax.set_ylim(0, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Trials (%)", labelpad=4)
    ax.set_xticks(positions, ["SF", "MF"] * len(dons))
    ax.tick_params(axis="x", labelsize=8.3, length=0, pad=2)
    ax.tick_params(axis="y", labelsize=8.5, width=0.6, length=2.8, pad=3)
    ax.grid(axis="y", color="#DFE3E8", linewidth=0.55)
    ax.set_axisbelow(True)
    for location, don in zip(centers, dons):
        ax.text(location, -0.162, don, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontweight="bold", fontsize=8.6)
    ax.set_xlabel("DoN")
    ax.xaxis.set_label_coords(0.5, -0.305)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.554, 0.977),
               ncol=3, frameon=False, fontsize=8.3, handlelength=1.32,
               handletextpad=0.43, columnspacing=1.17, borderaxespad=0)
    fig.savefig(report.with_name("failure_breakdown_latex.png"), dpi=600, facecolor="white")
    plt.close(fig)
```
