# Navigation trial CSV 기록

WSL의 기존 MPC 실행 환경에서 다음 명령을 실행한다.

```bash
python test_nmpc.py -c config/config_nmpc_trials_maze_oblique.yaml --results-dir benchmark_runs/maze_oblique_100
```

`maze`, `oblique_maze` × `psdf`, `dcbf`, `obca`를 각각 100회 실행한다(총 600회).
동역학은 `differential_drive`(DD), footprint는 `rectangle`, planner는 A*이다.
시뮬레이션 제한은 trial당 60초이며 애니메이션/플롯 생성은 꺼져 있다.
시작 위치는 기존 Gaussian perturbation을 사용한다: x/y 표준편차 5 mm,
heading 변동 없음, ±3σ 제한, 초기 clearance 최소 10 mm.
trial 1–100의 seed는 0–99이며 같은 trial 번호는 방법 간 동일한 seed를 사용한다.
localization error는 꺼져 있다. 관련 설정은 YAML에서 수정할 수 있다.

`--trials N`으로 반복 횟수를 덮어쓴다. 새 디렉터리로 짧게 확인하려면:

```bash
python test_nmpc.py -c config/config_nmpc_trials_maze_oblique.yaml --trials 1 --results-dir benchmark_runs/maze_oblique_check
```

한 조합만 실행할 때도 CSV 모드를 사용할 수 있다.

```bash
python test_nmpc.py --single --maze-type maze --robot-shape rectangle --dynamics-type differential_drive --optimizer-type psdf --trials 100 --perturb-start --start-seed 0 --no-animation --no-plots
```

`--trials`, `--results-dir`, 또는 YAML의 `trials`/`results_dir`로 CSV 모드를 활성화한다.
기존 명령은 이 옵션을 주지 않으면 기존 실행 경로를 사용한다.
`--results-dir`가 없으면 `data/trials_<timestamp>/`에 저장한다
(YAML의 `output_root_dir`가 있으면 그 아래 `data/`).
상대 결과 경로는 프로젝트 루트 기준이다.
이미 trial CSV가 있는 디렉터리는 오류로 중단하므로 재실행에는 새 경로를 사용한다.
중단한 batch를 이어 실행하는 resume 기능은 제공하지 않는다.

| 파일 | 내용 |
| --- | --- |
| `trial_results.csv` | trial당 한 행: 조합, trial 번호, seed, 시작 pose, 성공 1/실패 0, 상태/사유, 종료 시간, 성공한 solver 스텝 수 및 trial 내 mean/p95 |
| `computation_times_config01_maze_psdf_rectangle_differential_drive_astar.csv` 등 | YAML 설정마다 한 파일(예제는 6개): 성공한 solver 스텝당 한 행. trial 번호, seed, trial 성공 여부, 0부터 시작하는 timestep, solver 시간 |
| `summary.csv` | 설정마다 한 행: 완료/성공/실패/중단 수, success rate, 성공한 solver 스텝 전체의 mean/p95 |
| `run_manifest.yaml` | 실행 설정, CLI 옵션, 설정별 기본 seed |
| `runs/config01/trial_0001/` 등 | 각 실행의 기존 trajectory/start-pose/figure 출력. 반복 실행끼리 덮어쓰지 않음 |

세 CSV는 **각 trial 종료 후** 파일을 닫아 기록을 보존하며, summary는 원자적으로 교체한다.
solver 예외나 초기화 오류는 `status=error`, `success=0`과 오류 내용을 저장하고
다음 trial을 계속 실행한다. 이미 계산에 성공한 앞선 스텝은 오류가 나도 남는다.
Ctrl+C는 현재 trial을 `interrupted`로 저장하고 전체 batch를 멈춘다.

성공은 기존 시뮬레이터의 `status=success`(제한 시간 내 목표 도달) 기준이다.
목표 위치/각도 tolerance, 실패 조건은 기존 `success_criteria`,
`trial_failure_criteria` 설정을 따른다. 새로운 충돌 판정은 추가하지 않는다.
`success_rate`는 **성공 trial / 완료 trial**의 0–1 비율이다.
timeout, failure, error는 분모에 포함하며 사용자 중단은 제외한다.
목표 도달 뒤의 출력/플롯 오류는 `error` 열에 남기고 이미 정해진 주행 결과를 보존한다.

시간 단위는 **초(s)** 이다. PSDF/acados의 `solver.solve()`와
DCBF/OBCA의 `opti.solve()` 호출 전후를 `time.perf_counter()`로 측정한다.
명시적인 optimizer setup, PSDF parameter 갱신, 경로 계획, 그림/CSV 저장 시간은 제외한다.
각 solver 호출 내부에서 수행하는 초기화는 포함한다.
이는 전체 controller 계산 시간과 다르며, 기존 PSDF `solver_times`에 포함되던
parameter 갱신 시간도 이제 제외된다.

계산 성공은 solver 상태 기준(acados status 0, CasADi의 success 보고)이다.
실패/불명 상태, NaN/무한대/음수 시간은 timing CSV와 시간 통계에서 제외한다.
**trial의 최종 성공 여부와 관계없이** 성공한 solver 스텝을 포함한다.
성공 trial의 스텝만 분석하려면 timing CSV의 `trial_success == 1`로 필터링한다.
첫 스텝도 포함하고 warm-up 제외는 하지 않는다.
`summary.csv`의 mean/p95는 개별 성공 스텝을 모두 모아 계산한다
(trial별 mean/p95의 평균이 아님). p95는 NumPy 기본 선형 보간 percentile을 사용한다.
유효한 스텝이 없으면 시간 통계는 빈 칸으로 저장한다.

검증:

```bash
python -m unittest tests.test_trial_metrics tests.test_start_perturbation tests.test_optimizer_benchmark
```
