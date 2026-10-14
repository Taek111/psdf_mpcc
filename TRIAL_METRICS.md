# Navigation trial CSV 기록

요청 조합을 **각각 100회, 총 600회** 실행하려면 프로젝트의 Python 환경을 활성화한 뒤:

```bash
./run_nmpc_trials_25.sh
```

`maze`, `oblique_maze` × `psdf`, `dcbf`, `obca`, DD, rectangle, A* 설정을
순차 실행한다. 스크립트 파일명은 `run_nmpc_trials_25.sh`이지만 실행 횟수는
`--trials 100`으로 지정하며, seed는 0–99이다. 아래의 60초 제한, perturbation,
실패 판정 및 시간 집계 기준을 그대로 사용한다.
실행할 때마다 새 `benchmark_runs/maze_oblique_100_<timestamp>_<pid>/`
폴더에 저장하며, `summary.csv`의 `success_rate`, `solver_time_mean_s`,
`solver_time_p95_s` 열에서 요청한 지표를 확인한다.

결과 경로와 Python 인터프리터를 직접 지정할 수도 있다(상대 결과 경로는 프로젝트 루트 기준).

```bash
PYTHON_BIN=python3 ./run_nmpc_trials_25.sh benchmark_runs/maze_oblique_100_run01
```

WSL의 기존 MPC 실행 환경에서 다음 명령을 실행한다.

```bash
python test_nmpc.py -c config/config_nmpc_trials_maze_oblique.yaml --results-dir benchmark_runs/maze_oblique_100
```

`maze`, `oblique_maze` × `psdf`, `dcbf`, `obca`를 각각 100회 실행한다(총 600회).
동역학은 `differential_drive`(DD), footprint는 `rectangle`, planner는 A*이다.
시뮬레이션 제한은 trial당 60초이며 애니메이션/플롯 생성은 꺼져 있다.
시작 pose는 Gaussian perturbation을 사용한다: x/y 표준편차 각각 20 mm,
heading 표준편차 0.05 rad(약 2.864789°), ±3σ 제한, 초기 clearance 최소 10 mm.
trial 1–100의 seed는 0–99이며 같은 trial 번호는 방법 간 동일한 seed를 사용한다.
localization error는 꺼져 있다. 관련 설정은 YAML에서 수정할 수 있다.

OBCA와 DCBF는 기본적으로 현재 로봇 footprint와 장애물의 거리가
`safe_dist=0.5` m를 초과하는 쌍을 최적화에서 제외한다. 경계값 0.5 m는
포함하며, `use_obstacle_cutoff: false`이면 거리와 무관하게 모두 포함한다.
`safe_dist`는 장애물 선택 거리이고, 충돌 제약의 `margin_dist` 및 시작 pose의
최소 clearance 설정과는 별개이다. 현재 거리 기준 cutoff이며 예측 전체의
도달 가능 영역을 검사하는 방식은 아니다.

OBCA는 선택한 장애물에 대해 `x[1]`부터 `x[horizon]`까지 duality 거리 하한
`D >= margin_dist`를 적용한다. DCBF는 기존 exponential DCBF 제약을
`x[1]`부터 `x[horizon_dcbf]`까지 적용한다. 다음 블록으로 각각 설정할 수 있다.

```yaml
obca:
  safe_dist: 0.5
  use_obstacle_cutoff: true
dcbf:
  safe_dist: 0.5
  use_obstacle_cutoff: true
```

두 optimizer는 선택된 장애물·로봇 geometry와 최적화 설정이 같으면 메인
`Opti` 및 solver를 재사용한다. 매 주기에는 초기 상태, 이전 입력, 참조와
초기 추정값을 갱신하며, DCBF의 현재 거리도 parameter로 갱신한다. 선택된
쌍이나 최적화 설정이 바뀌면 문제를 다시 구성한다. 현재 거리와 초기 dual을
구하는 보조 최적화 문제도 geometry 행렬의 크기가 같으면 재사용한다.
`problem_build_count`, `active_obstacle_count`, `active_constraint_pair_count`로
메인 문제 구성 횟수와 선택 규모를 확인할 수 있다.

`trial_failure_criteria`는 다음과 같이 활성화되어 있다.

```yaml
trial_failure_criteria:
  enabled: true
  movement_window_sec: 4.0
  movement_threshold: 0.03
  max_consecutive_solver_failures: 20
```

최근 4초의 시작·끝 위치 간 변위가 3 cm 이하이면 `status=failure`로 종료한다.
PSDF의 acados status가 0이 아니면 예외를 발생시키지 않고 공통 failure checker에
실패 상태를 전달한다. 성공하면 연속 실패 횟수가 0으로 초기화되며, 연속 20회 실패하면
`status=failure`, `success=0`, `consecutive_solver_failures(20)` 사유를 저장하고
다음 실행으로 진행한다. 한도 도달 전에는 solver가 반환한 제어 입력으로 주행을 계속한다.
실제 solver 예외나 초기화 오류는 별도로 `status=error` 처리한다.

CC-MF는 배치 PSDF 평가의 stage 0 값이 `d_col`보다 작으면 solver와 backup
호출을 건너뛰고 기존 감속 정지 계획을 적용한다. 이 주기는
`solve_mode=safe_stop_stage0`, `solver_called=false`, `success=false`로 기록하고,
실행하지 않은 solver의 상태 코드는 빈 값으로 남긴다. 실제 solver 실패 횟수와
별도로 `stage0_guard_stop_count`를 센다. 이 안전 정지가 포함된 연속 실패는 같은
한도를 적용하되 종료 사유를 `consecutive_unsuccessful_control_steps(N)`로 기록한다.
정지 계획 생성·적용에 실패하면 이전 입력을 반환하지 않고 예외로 종료한다.
이는 현재 상태의 안전 기준 위반을 감지하는 기능이며, 다음 입력의 충돌이나
감속 중 추가 이동까지 검증하는 기능은 아니다.

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

성공은 제한 시간 내 목표 위치와의 **x/y L1 거리**가 0.02 m 이하인 경우이다:
`abs(x - goal_x) + abs(y - goal_y) <= 0.02`. 각도(rad)는 거리(m)에 합산하지 않는다.
100회 설정은 `success_criteria.position_tolerance: 0.02`,
`angle_tolerance: null`로 위치만 판정하며, 기본값도 동일하다.
다른 설정에서 각도 tolerance를 명시하면 목표 자세가 제공되는 경우에만 추가 적용한다.
목표 도달은 `trial_failure_criteria`의 정체·solver 실패 검사보다 먼저 판정한다.
새로운 `trial_results.csv`의 `distance_to_goal`도 같은 L1 거리(m)를 기록한다.
변경 전 결과 CSV의 거리 값은 기존 L2 거리이며, 기존 파일과 성공/실패 판정은 소급 변경하지 않는다.
새로운 충돌 판정은 추가하지 않는다.
`success_rate`는 **성공 trial / 완료 trial**의 0–1 비율이다.
timeout, failure, error는 분모에 포함하며 사용자 중단은 제외한다.
목표 도달 뒤의 출력/플롯 오류는 `error` 열에 남기고 이미 정해진 주행 결과를 보존한다.

시간 단위는 **초(s)** 이다. PSDF/acados 및 CC-MF/CC-ECS의 `solver.solve()`와
DCBF/OBCA의 `opti.solve()` 호출 전후를 `time.perf_counter()`로 측정한다.
명시적인 optimizer setup, PSDF parameter 갱신, 경로 계획, 그림/CSV 저장 시간은 제외한다.
각 solver 호출 내부에서 수행하는 초기화는 포함한다.
이는 전체 controller 계산 시간과 다르며, 기존 PSDF `solver_times`에 포함되던
parameter 갱신 시간도 이제 제외된다.
CC-MF/CC-ECS는 한 제어 주기에 메인 solver와 backup solver를 모두 호출하면
두 `solve()` 호출 시간을 합쳐 `solver_times`에 한 번 기록한다. 최종 solver가
성공한 주기만 timing CSV에 포함하므로 backup 성공 시 앞선 실패 호출 시간도
포함하고, safe-stop 주기는 solver 성공으로 취급하지 않는다. 이 변경 전
CC-MF 결과는 준비·진단 시간을 포함했으므로 새 결과와 직접 합쳐 비교하지 않는다.

CC-MF stage 0 안전 정지처럼 solver를 호출하지 않은 주기는 로그의 스텝 정렬을
위해 내부 `solver_times`에 0을 넣지만, 성공 스텝 timing CSV와 통계에서는 제외한다.
현재 자세 PSDF는 제약 생성 배치의 `phi[0]`를 진단과 CSV가 공유한다. 장애물 제약이
활성화된 정상 경로는 제어 주기당 PSDF 배치 평가 1회이며, 별도 현재 자세 평가는 없다.
상세 예측·MF·구간 진단과 주기별 콘솔 출력은 `debug_mf` / `debug_infeasibility`
설정에 따라 실행한다. 해당 디버그 검사나 MF 예외 복구에는 추가 평가가 있을 수 있다.
장애물 제약을 끈 경우 현재 PSDF는 미측정 값으로 기록하며, 대체 상수 1000을 실제
거리로 기록하거나 로깅을 위해 PSDF를 별도로 호출하지 않는다.

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
python -m unittest tests.test_obca_constraints
python -m unittest tests.test_duality_optimizer_reuse
```

`test_obca_constraints`는 solver나 주행 시뮬레이션을 실행하지 않고,
해석적으로 구한 dual 변수와 pose에서 CasADi 제약을 평가한다.
회전된 rectangle, 먼 초기 장애물, horizon 변경 및 마지막 예측 상태의
최소 거리 위반을 확인한다.

`test_duality_optimizer_reuse`는 거리 query를 mock으로 대체해 cutoff 경계,
빈 장애물 집합, geometry/설정 변경 시 재구성, 상태·입력·참조·DCBF 거리의
parameter 갱신을 확인한다. 보조 거리 문제의 캐시 검사도 `solve()`를 mock으로
대체하며 실제 solver나 주행 시뮬레이션은 실행하지 않는다.

## CC-ECS baseline

`cc_ecs`는 `references/cc-scp.pdf`의 Ellipsoidal Confidence Set(ECS)을
로봇 pose `[x, y, theta]`의 Gaussian 불확실성과 장애물별 signed distance의
국소 선형 근사에 적용한다. CC-MF와 같은 DD/MPCC 상태
`[x, y, theta, s, xi_f, xi_l, xi_theta, xi_c]`, 입력 `[v, omega, v_s]`,
경로 비용과 입력 한계를 사용한다. 원 논문의 SCP 전체 알고리즘을 재현하는
방법은 아니며, acados SQP-RTI에서 ECS 기반 제약을 비교하는 baseline이다.

`chance_epsilon`은 **한 예측 horizon의 총 예산**이며 기본값은 0.20이다.
미래 상태 `k=1,...,N`에 `epsilon_k=chance_epsilon/N`을 균등 배분하고,
terminal `k=N`도 포함한다. 현재 상태 `k=0`에는 ECS 제약을 적용하지 않는다.
pose 차원이 3이므로 `beta=sqrt(chi2.ppf(1-epsilon_k, df=3))`를 사용한다.
각 stage의 같은 pose confidence ellipsoid를 모든 장애물 제약에 사용하므로
장애물 수 `K`로 예산을 다시 나누지 않는다. 모든 장애물을 포함하며 거리 cutoff를
사용하지 않는다. 장애물별 제약은 slack 없는 hard constraint이다.

각 stage/장애물의 nominal pose를 `pose_bar`, 그곳의 signed distance와
world-frame pose gradient를 `d_bar`, `g_bar`라고 할 때 다음 행을 구성한다.

```text
q_bar = T(theta_bar)^T g_bar,  T(theta) = diag(R(theta), 1)
d_bar + g_bar^T (pose - pose_bar) - d_min
      - beta * ||L(xi)^T q_bar||_2 >= 0
Sigma_local = L(xi) L(xi)^T
```

`d_bar`, `g_bar`, `q_bar`는 해당 주기의 nominal linearization에서 고정하고,
`L(xi)`는 최적화 상태의 함수로 남긴다. risk gradient의 frame 회전에는
candidate heading이 아닌 nominal heading을 사용한다. 서로 다른 장애물의
distance를 하나의 최소값으로 먼저 합치지 않는다. 이 구성의 확률 해석은
Gaussian 모델과 **선형화된 안전 영역**에 대한 것이며, 비선형 footprint의
전역 충돌 보장이나 연속 시간·전체 closed-loop rollout의 충돌 확률 보장이 아니다.
진단 로그의 고정된 제약 행 residual과 최종 해에서 distance/gradient를 다시
평가한 residual도 구분해야 한다. Solver 성공은 acados status 기준이며,
SQP-RTI의 한 번의 성공만으로 비선형 ECS 잔차의 만족을 보장하지 않으므로
진단 잔차를 함께 확인해야 한다. Safe-stop fallback 자체도 ECS 확률 보장을
제공하지 않는다.

공분산은 기존 `cc_covariance.py`의 body-frame reduced log-Cholesky 모델을
사용한다. 선택적인 `initial_pose_covariance`는 world-frame 3×3 SPD 행렬이며
현재 heading으로 회전한 뒤 forward–lateral/forward–heading 교차항을 제외한다.
미지정 시 `sigma_f0=0.0002`, `sigma_l0=0.00025`, `sigma_psi0=0.00025`를 사용한다.
CC-ECS의 process-noise 계수는 모두 0(`Q=0`)으로 고정하며 0이 아닌 설정은
오류로 거부한다. 입력에 의존하는 `A Sigma A^T` 전파는 유지한다. `robust_horizon: null`은
`min(10,N)`까지 전파한 뒤 공분산을 동결한다. 이후 stage에서도 동결된
공분산으로 ECS 제약을 적용한다. 초기 공분산을 유지하는 것과 `Q=0`은 다르다.
`localization_error`는 simulator의 관측 잡음 설정이며 optimizer 공분산을
자동으로 설정하지 않는다.

CC-ECS 설정은 독립된 `cc_ecs:` 블록에서 읽고 `cc_mf:` 설정을 병합하지 않는다.
따라서 CC-MF와 비교할 때 동일한 숫자의 `chance_epsilon`이 같은 위험 예산을
뜻한다고 가정하면 안 된다. CC-MF의 stage별 위험 예산, process noise,
Row M slack/활성화 설정은 별도로 맞추고 보고해야 한다.

실행 예제는 rectangle/DD/A*에 대해 `s_path`, `straight_corridor`, `maze`,
`oblique_maze`를 각 한 번, 최대 60초로 실행하며 animation/plot은 끈다.
테스트나 benchmark는 별도 실행 요청 전에는 수행하지 않는다.

```bash
python test_nmpc.py -c config/config_nmpc_cc_ecs_rectangle_all4.yaml --trials 1 --results-dir benchmark_runs/cc_ecs_rectangle_run01
```

기존 `trial_results.csv`, `summary.csv`, solver timing CSV 외에 각 실행의
`data/` 아래 다음 로그를 저장한다. `NAME`은 `MPC_cc_ecs_rectangle_maze` 등의
실행 이름이며, perturbation/output suffix가 있으면 함께 포함한다.

| 파일 | 내용 |
| --- | --- |
| `cc_ecs_constraints_NAME.csv` | stage·장애물별 nominal 정보, ECS margin, 고정된 행 및 재평가된 residual |
| `cc_ecs_covariance_NAME.csv` | 예측 stage별 공분산과 입력 |
| `cc_ecs_cycle_NAME.csv` | 주기별 solver 상태와 복구 방식 |
| `pose_sdf_NAME.csv` | 기존 공통 pose, PSDF, 진행도 기록 |

ECS log의 열 이름은 optimizer가 제공하는 `get_*_log_fields()`를 그대로 사용한다.
모든 제약/공분산 로그는 진단용이며 trial 성공 판정이나 solver 시간 통계에
추가로 포함하지 않는다.
