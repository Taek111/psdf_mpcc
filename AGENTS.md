# Repository Guidelines

## Project Structure & Module Organization
- `control/`: MPC controllers/optimizers (`psdf_optimizer*.py`, `nmpc_optimizer*.py`, `dcbf_optimizer*.py`).
- `models/`: Dynamics + geometry (`dd.py`, `geometry_utils.py`, `psdf_wrapper.py`).
- `planning/`, `sim/`: Planning utilities and simulation loop.
- `config.yaml`: Scenario configs; outputs go to `animations/`, `figures/`, `data/`.
- Tests: top-level `test_*.py` scripts.

## Build, Test, and Development Commands
- Python 3.10+; minimal deps: `pip install numpy matplotlib pandas pyyaml torch` (+ optional `casadi`, `acados`).
- NMPC: `python test_nmpc.py -c config.yaml` or quick `python test_nmpc.py --single`.
- Other runs: `python test_hybrid_astar.py`, `python test_psdf.py`.

## NMPC 반복 실험 (`test_nmpc.py`)
- 설정: `config/config_nmpc_trials_maze_oblique.yaml`. `maze`/`oblique_maze` × `psdf`/`dcbf`/`obca`를 조합별 100 trial씩, 총 600회 실행한다. 동역학은 DD(`differential_drive`), footprint는 `rectangle`, planner는 A*이다.
- Trial당 시뮬레이션 제한은 60초. 시작 pose에 Gaussian perturbation(x/y 표준편차 각각 20 mm, heading 표준편차 0.05 rad ≈ 2.864789°, ±3σ 제한, 초기 clearance ≥ 10 mm)을 적용한다. Seed는 0–99이며 같은 환경·trial 번호에서 방법 간 동일한 시작 pose를 사용한다. Localization error와 animation/plot은 끈다.
- `trial_failure_criteria`는 활성화하며 `movement_window_sec=4.0`, `movement_threshold=0.03`, `max_consecutive_solver_failures=20`을 사용한다. PSDF의 acados status가 0이 아니면 공통 checker의 연속 실패 횟수에 반영한다. 성공 시 횟수를 초기화하며 연속 20회 실패하면 해당 trial을 `failure`로 저장하고 다음 실행으로 진행한다. Solver 예외는 별도로 `error` 처리한다.
- 같은 PC·실행 환경에서 순차 실행한다. 프로젝트 루트에서 아래 명령을 사용하며, 재실행에는 새 `--results-dir`를 지정한다(기존 CSV 덮어쓰기 및 resume 미지원).

```bash
python test_nmpc.py -c config/config_nmpc_trials_maze_oblique.yaml --trials 100 --results-dir benchmark_runs/maze_oblique_100
```

- `trial_results.csv`: trial별 성공(1)/실패(0), 상태·실패 사유, seed, 시작 pose를 매 trial 종료 후 기록한다. 성공은 기존 목표 도달 판정이며 success rate는 성공/완료 trial 비율이다. Timeout·failure·error는 분모에 포함하고 사용자 중단은 제외한다.
- `computation_times_*.csv`: 설정별 한 파일에 **solver가 성공한 스텝만** 기록한다. `solve()` 호출을 `time.perf_counter()`로 측정하며 단위는 초(s)이다. 외부 setup·PSDF parameter 갱신·경로 계획·출력 시간은 제외한다.
- `summary.csv`: 설정별 success rate와 solver 시간 mean/p95. 최종적으로 실패한 trial의 성공 스텝도 포함하여 전체 유효 스텝을 모아 집계한다(trial별 통계의 평균이 아님). 첫 스텝을 포함하고 유효 스텝이 없으면 시간 통계는 빈 칸이다.
- Solver 예외는 실패 사유를 저장하고 다음 trial로 진행한다. Ctrl+C는 현재 trial을 `interrupted`로 저장하고 batch를 중단한다. 상세 기준과 출력 구조는 [TRIAL_METRICS.md](TRIAL_METRICS.md)를 참고한다.
- 이 실험은 사용자가 다른 PC에서 실행할 예정이므로, 별도 실행 요청 전에는 테스트·benchmark를 실행하지 않고 로깅 코드를 검토한다.

## Coding Style & Naming Conventions
- PEP 8, 4-space indents, type hints where practical.
- `snake_case` files/functions, `CamelCase` classes, `UPPER_SNAKE_CASE` constants.
