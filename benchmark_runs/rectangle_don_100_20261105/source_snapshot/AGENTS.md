# Repository Guidelines

## Project Structure & Module Organization
- `control/`: Controllers/optimizers for `obca`, `psdf`, `cc_mf`, `cc_sf`, `dcbf`, `cc_ecs`, `sipoc`, `uobca`, plus shared helpers. SIPOC uses `dilated_polygon_robust_optimizer.py`.
- `models/`: Dynamics + geometry (`dd.py`, `geometry_utils.py`, `psdf_wrapper.py`).
- `planning/`, `sim/`: Planning utilities and simulation loop.
- `config/config.yaml`: Scenario configs; outputs go to `animations/`, `figures/`, `data/`.
- Tests: top-level `test_*.py` scripts.

## Build, Test, and Development Commands
- Python 3.10+; minimal deps: `pip install numpy matplotlib pandas pyyaml torch` (+ optional `casadi`, `acados`).
- NMPC: `python test_nmpc.py -c config.yaml` or quick `python test_nmpc.py --single`.
- Other runs: `python test_hybrid_astar.py`, `python test_psdf.py`.

## 주행 실험 기본 규칙
- 주행 실험 실행 요청에 별도 지정이 없으면 footprint는 `rectangle`, kinematics는 DD(`differential_drive`), environment는 `maze`를 사용한다.
- Animation 추가 요청 시 기존 기본 animation 생성 코드를 사용하고, 별도의 인코딩 작업 없이 `animations/<controller>/` 디렉토리에 저장한다.

## NMPC 반복 실험 (`test_nmpc.py`)
- 설정: `config/config_nmpc_trials_maze_oblique.yaml`. `maze`/`oblique_maze` × `psdf`/`dcbf`/`obca`를 조합별 100 trial씩, 총 600회 실행한다. 동역학은 DD(`differential_drive`), footprint는 `rectangle`, planner는 A*이다.
- Trial당 시뮬레이션 제한은 60초. 시작 pose에 Gaussian perturbation(x/y 표준편차 각각 20 mm, heading 표준편차 0.05 rad ≈ 2.864789°, ±3σ 제한, 초기 clearance ≥ 10 mm)을 적용한다. Seed는 0–99이며 같은 환경·trial 번호에서 방법 간 동일한 시작 pose를 사용한다. Localization error와 animation/plot은 끈다.
- `trial_failure_criteria`는 활성화하며 `movement_window_sec=4.0`, `movement_threshold=0.03`, `max_consecutive_solver_failures=20`을 사용한다. PSDF의 acados status가 0이 아니면 공통 checker의 연속 실패 횟수에 반영한다. 성공 시 횟수를 초기화하며 연속 20회 실패하면 해당 trial을 `failure`로 저장하고 다음 실행으로 진행한다. Solver 예외는 별도로 `error` 처리한다.
- 같은 PC·실행 환경에서 순차 실행한다. 프로젝트 루트에서 아래 명령을 사용하며, 재실행에는 새 `--results-dir`를 지정한다(기존 CSV 덮어쓰기 및 resume 미지원).
- acados 생성 파일(JSON, C 코드, 공유 라이브러리, 백업 solver, PSDF 캐시)은 optimizer 인스턴스별 고유 임시 폴더에 저장하고 `cleanup()`/`reset()`에서 삭제한다. PSDF 빌드 캐시는 실험 간 공유하지 않는다. 별도로 병렬 실행할 때는 프로세스마다 다른 `--results-dir`를 지정한다. 정상 종료·예외·Ctrl+C는 기존 정리 경로를 사용하며, 강제 종료(`kill -9`) 시 임시 폴더가 남을 수 있다.

```bash
python test_nmpc.py -c config/config_nmpc_trials_maze_oblique.yaml --trials 100 --results-dir benchmark_runs/maze_oblique_100
```

- `trial_results.csv`: trial별 성공(1)/실패(0), 상태·실패 사유, seed, 시작 pose를 매 trial 종료 후 기록한다. 성공은 기존 목표 도달 판정이며 success rate는 성공/완료 trial 비율이다. Timeout·failure·error는 분모에 포함하고 사용자 중단은 제외한다.
- `computation_times_*.csv`: 설정별 한 파일에 **solver가 성공한 스텝만** 기록한다. MPCC·CC-SF·CC-MF·CC-ECS는 `solve_nlp()` 전체 호출을 `time.perf_counter()`로 측정하며 단위는 초(s)이다. 함수 내부의 parameter 갱신·주 solver·backup·safe-stop 생성은 포함하고 외부 `setup()`·경로 계획·controller 로깅·파일 출력은 제외한다. 다른 optimizer는 기존 측정을 유지한다. 네 방법의 공정한 비교에는 공통 `mpcc.use_debug: false`(기본값)를 사용한다. `--use-debug` / `--no-use-debug`로 덮어쓸 수 있으며 debug가 켜지면 함수 내부 진단 비용도 포함한다. 상세 pose-SDF·제약·공분산·cycle CSV와 예측 궤적 기록은 debug에서만 생성하고 trial 결과·성공 스텝 시간·summary는 항상 기록한다. 기존 내부 `solve()` 시간 결과와 직접 비교하지 않으며 `run_manifest.yaml`에 측정 범위를 기록한다.
- `summary.csv`: 설정별 success rate와 solver 시간 mean/p95. 최종적으로 실패한 trial의 성공 스텝도 포함하여 전체 유효 스텝을 모아 집계한다(trial별 통계의 평균이 아님). 첫 스텝을 포함하고 유효 스텝이 없으면 시간 통계는 빈 칸이다.
- Solver 예외는 실패 사유를 저장하고 다음 trial로 진행한다. Ctrl+C는 현재 trial을 `interrupted`로 저장하고 batch를 중단한다. 상세 기준과 출력 구조는 [TRIAL_METRICS.md](TRIAL_METRICS.md)를 참고한다.
- 이 실험은 사용자가 다른 PC에서 실행할 예정이므로, 별도 실행 요청 전에는 테스트·benchmark를 실행하지 않고 로깅 코드를 검토한다.

## Rectangle DoN 실험 준비
- `scripts/run_rectangle_don_benchmark.sh`는 DoN 0.5/0.75(너비 0.075/0.1125 m)에서 네 controller를 각 100회(seed 0–99) 실행한다. `scripts/run_rectangle_don_sweep_benchmark.sh`는 DoN 0.70–0.87을 0.01 간격으로 네 controller 각 50회(seed 0–49) 실행한다. DoN은 robot 너비 / 통로 너비 0.150 m이며 후자는 너비 0.105–0.1305 m, 간격 0.0015 m에 해당한다. `--reuse-results`로 전 실험 DoN 0.75의 동일 조건 seed 0–49를 재사용할 수 있다. 실행·분석 조건은 `docs/rectangle_benchmark_analysis_context.md`를 따른다.
- `benchmark_logging.enabled`는 debug와 독립적인 실험 원본 저장 옵션이다. 기존 메모리의 실제 초기/적분 후 자세·적용 입력·solver 시간/상태와 고정 형상을 trial 종료 후, optimizer 정리 전에 `raw_trace.npz`/`raw_trace_metadata.json`으로 저장한다. 주행 중 clearance 계산이나 파일 기록을 추가하지 않는다. 초기 자세 표본의 기존 clearance 수용 검사는 유지한다.
- Mean/P05 clearance는 주행 완료 후 `scripts/analyze_rectangle_clearance.py`로 계산한다. 성공/실패/timeout/error trial의 초기 및 적분 후 전체 표본을 합치고, interrupted는 주 집계에서 제외한다. 접촉·겹침은 0, 지도 경계 침범은 음수다. 누락된 trace는 0으로 채우지 않고 coverage에 표시한다.

## Coding Style & Naming Conventions
- PEP 8, 4-space indents, type hints where practical.
- `snake_case` files/functions, `CamelCase` classes, `UPPER_SNAKE_CASE` constants.
