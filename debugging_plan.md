# RMPCC Row MF 충돌·정지 디버깅 및 파라미터 튜닝 계획

## 0. 목표와 완료 조건

이 계획의 목표는 rectangle 로봇과 differential-drive 동역학에서 `rmpcc_optimizer.py`의 hard deterministic PSDF Row G와 soft multi-feature chance-constraint Row MF를 동시에 사용할 때 발생하는 정지, solver failure, 장애물 침투의 원인을 분리하고, `s_path`와 `maze`를 모두 안전하게 완주하는 파라미터 조합을 찾는 것이다.

- [ ] Row G와 `d_col=0.001`은 모든 실험에서 고정한다.
- [ ] 비교 환경은 `rectangle / differential_drive / astar / {s_path, maze}`로 고정한다.
- [ ] `rmpcc_pv` 완주 결과에서 환경별 진행도 기준선 `S_ref`를 새로 측정한다.
- [ ] 각 파라미터는 OFAT(one factor at a time) 방식으로 2~5개 값만 독립 시험한다.
- [ ] 한 파라미터 시험이 끝날 때마다 다른 모든 파라미터를 기준값으로 원복한다.
- [ ] 독립 시험에서 확인된 변수만 후속 복합 시험에 포함한다.
- [ ] 두 환경 모두에서 기준 진행도 이상, 무충돌, Row G 유지, main/final-active solver failure 및 safe-stop 0회를 만족하는 최종 설정을 정한다.

최종 후보는 단순히 가장 느슨한 설정이 아니라, 위 조건을 만족하는 후보 중 MF를 가장 강하게 유지하는 설정으로 정한다.

---

## 1. 시작 전 체크포인트와 재현 환경

### 1.1 Git 체크포인트

- [x] 기존 작업 상태를 작업 전 커밋으로 보존했다.
  - commit: `8b72288 checkpoint: preserve RMPCC debugging state`
  - 당시 branch: `main`
- [ ] 실제 계측 코드 구현 직전에 `git status --short --branch`와 `git rev-parse HEAD`를 다시 기록한다.
- [ ] 계측·실험 자동화 코드와 단위 테스트가 준비되면 별도 커밋을 만든 후 sweep을 시작한다.
- [ ] sweep 중에는 optimizer 기본값을 직접 고쳐 가며 실행하지 않는다. 불변 baseline config에서 실행별 config를 생성한다.
- [ ] 각 시행 전후 `git diff --exit-code`로 소스가 바뀌지 않았음을 확인한다. 실험 산출물은 별도 output root에 저장한다.

### 1.2 실행 환경 고정

- [ ] Python 3.10+ 가상환경 경로와 `python --version`을 기록한다.
  - 현재 시스템의 `/usr/bin/python3`은 3.8 계열이므로 실제 튜닝에는 사용하지 않는다.
- [ ] `numpy`, `torch`, `casadi`, `acados_template`, HPIPM/acados, `pytest` 버전과 CPU/GPU 장치를 기록한다.
- [ ] Python 3.10+와 호환되는 최신 pytest를 설치하고 collection/capture가 정상인지 확인한다.
- [ ] 모든 비교 시행에서 같은 device를 사용한다.
- [ ] localization error/noise는 끄고, 난수가 사용된다면 seed를 고정한다.
- [ ] 아래 회귀 테스트를 통과시킨다.

```bash
python -m pytest -q tests/test_rmpcc_mf.py tests/test_augmented_psdf.py
python -m pytest -q tests/test_acados_diagnostics.py tests/test_spline_path.py
```

### 1.3 현재 소스 기준값 기록

`debugging_issue.md`의 논의 값과 실제 checkpoint 소스가 다르므로, 아래 값을 source of truth로 사용한다.

| 구분 | checkpoint `8b72288`의 실제 값 |
|---|---:|
| `horizon`, `tf`, `dt` | `20`, `2.0`, `0.1 s` |
| `mf_slack_linear` | `100` |
| `mf_slack_quadratic` | `1` |
| `chance_epsilon` | `0.20` |
| `d_min`, `d_col`, `d_mf_mask` | `0.001`, `0.001`, `0.001` |
| `q_s`, `v_s_target`, effective `v_s_max` | `1`, `0.8`, `0.7` |
| `sigma_f0`, `sigma_l0`, `sigma_psi0` | `0.0002`, `0.00025`, `0.00025` |
| `q_f0`, `q_l0`, `q_psi0` | 모두 `0` |
| `alpha_f`, `alpha_v`, `alpha_kappa` | `0.002`, `0.0004`, `0.01` |
| `beta_v`, `beta_kappa`, `beta_omega` | `0.02`, `0.008`, `0.008` |
| `enable_backup_solver` | `False` |
| MF 활성 구간 | 현재는 stage 1부터 terminal까지 사실상 전 구간 |

현재 MF row에는 별도의 row normalization이 없으므로 slack과 비용은 raw probability residual 단위로 해석한다. `v_s=0.7`의 한 stage progress 이득은 약 `0.394`이고, 현재 선형 가중치 `100`에서 MF slack `0.01`의 비용은 약 `1.0`이다. 따라서 문서의 `1e3` 가정만큼은 아니지만, 현재 `100`도 진행 이득보다 약 2.5배 크다.

### 1.4 기존 산출물은 기준선이 아닌 재현 참고값으로만 사용

| 환경 | 기존 RMPCC `max(path_s)` | 기존 `min(sdf)` | solver failure 수 | 현재 판정 |
|---|---:|---:|---:|---|
| `s_path` | `0.980310` | `-0.0017066` | `1` | 충돌 FAIL |
| `maze` | `0.364182` | `-0.210954` | `182` | 진행 부족 + 충돌 + solver FAIL |

- [ ] 위 결과를 현재 checkpoint와 60초 제한에서 한 번 재현하고 `R0`으로 보관한다.
- [ ] 기존 CSV를 PV 기준선이나 최종 합격 근거로 재사용하지 않는다.

---

## 2. 튜닝 전에 필요한 계측과 자동화

파라미터 sweep 전에 원인이 비용 선택인지, 수치적 solver failure인지, recovery 입력인지 구분할 수 있어야 한다.

### 2.1 MF 활성 구간 파라미터 구현

현재 `mf_active_start_step`, `mf_active_end_step`, `mf_active_terminal`은 구현되어 있지 않으며 YAML에 넣으면 무시된다.

- [ ] `RMPCCOptimizerParam`에 다음 값을 추가한다.

```python
mf_active_start_step = 1
mf_active_end_step = None
mf_active_terminal = True
```

- [ ] MF mask에 stage-window mask를 추가한다.
- [ ] stage 0은 기존처럼 제약 없음으로 유지한다.
- [ ] intermediate window는 stage `1..N-1`에만 `start/end`를 적용한다.
- [ ] terminal stage `N`은 window와 독립적으로 `mf_active_terminal`만으로 켜고 끈다.
- [ ] 위 호환 기본값은 기존의 stage `1..N-1` 및 terminal-on 동작을 그대로 재현해야 한다.
- [ ] `B0`에서만 `end_step=5`, `terminal=False`를 override한다.
- [ ] 중간 stage와 terminal의 mask가 각각 기대대로 적용되는 단위 테스트를 추가한다.
- [ ] 실행 시작 시 실제 active stage 목록을 출력하고 결과 CSV에 저장한다.

### 2.2 cycle별 solver/recovery 경로 기록

MF slack은 상한이 없는 soft slack이므로 MF를 켠 것만으로 feasible set이 직접 줄어드는 구조는 아니다. nonzero status가 발생하면 큰 penalty, row scaling/conditioning, QP linearization, covariance 이상, hard row·dynamics·box bound 충돌을 구분해야 한다.

- [ ] 매 cycle에 아래 항목을 같은 row로 기록한다.
  - 실제 `param.nlp_solver_type`에서 유도한 main solver algorithm: `SQP_RTI`, `SQP` 등
  - recovery/result mode: `main`, `backup_feasible_qp`, `safe_stop`
  - main/backup solver status, QP status, QP iteration, NLP residual
  - 실제 plant에 적용한 `u0=[v0, omega0, v_s0]`
  - solver failure 직전/직후 pose와 exact PSDF
- [ ] `safe_stop` 직후 발생한 침투는 `COLLISION_AFTER_SAFE_STOP`으로 별도 태깅한다.
- [ ] status 0인 정상 solve의 제어 입력에서 발생한 침투는 `COLLISION_AFTER_NORMAL_SOLVE`로 태깅한다.
- [ ] backup 성공 시 main solver 객체의 status와 반환된 active solver의 status를 혼동하지 않도록 active solver 기준으로 기록한다.
- [ ] `main_solver_failure_count`와 recovery 후 `final_active_solver_failure_count`를 별도 누적한다.
- [ ] main solver를 `SQP`로 바꾼 경우에도 mode를 고정 문자열 `sqp_rti`로 기록하지 않도록 계측을 수정한다.

현재 safe-stop은 장애물을 다시 최적화하지 않고 이전 속도를 최대 `1 m/s²`로 감속하며 `omega=0`을 적용한다. 따라서 solver failure 뒤의 충돌은 MF의 최적해가 hard Row G를 무시한 현상과 구분해야 한다.

### 2.3 충돌 계측 보강

현재 `pose_sdf`는 제어 적용 전 pose를 기록하고, 마지막 post-step pose와 0.1초 사이 collision을 놓칠 수 있다.

- [ ] 모든 실제 post-actuation pose에서 exact PSDF를 계산해 저장한다.
- [ ] 같은 post-actuation pose를 동일한 canonical global path에 투영한 `executed_projected_s`를 저장한다.
- [ ] 이미 계산 중인 first-interval substep PSDF를 재사용하고 cycle CSV에 영속화한다.
- [ ] 각 control interval을 `first_interval_substeps=10` 이상으로 나누어 substep-sampled minimum PSDF를 저장한다.
- [ ] `collision = any(executed_or_substep_psdf < -tau_phi)`로 자동 판정한다.
- [ ] 접촉과 수치 오차를 구분하기 위해 `tau_phi=1e-6 m`를 실험 전 고정한다.
- [ ] `min_psdf < d_col - constraint_debug_tolerance`도 `HARD_CLEARANCE_BREACH`로 별도 기록한다.
- [ ] navigation loop에서 collision 판정을 goal 판정보다 먼저 수행하거나, 저장 전에 collision metric으로 outcome을 반드시 재분류한다.
- [ ] collision 조기 종료는 exception이 아니라 정상 `status=failure` outcome을 반환해 CSV saver까지 실행되게 한다.
- [ ] collision 발생 시 현재 시행을 즉시 종료하고 산출물을 정상 저장한다.

substep 값은 연속 구간 전체의 수학적 최솟값이 아니라 이산 표본의 최솟값이다. 결과 이름에 `exact_swept`를 사용하지 않고 `sampled_interval`임을 명시한다.

### 2.4 MF 원인 진단값 추가

- [ ] shifted nominal `z_bar`, `u_bar`를 저장한다.
- [ ] stage별 nominal MF residual과 probability sum을 계산한다.

```text
h_mf_nominal
probability_sum_nominal = chance_epsilon - h_mf_nominal
```

- [ ] solved trajectory에서 아래 값을 기록한다.
  - `mf_mask`, `mf_valid`
  - affine/fresh raw/fresh effective MF residual
  - raw MF slack, maximum/mean MF slack
  - `v`, `omega`, `v_s` horizon trajectory
  - `P_f`, `P_l`, `P_psi`, `P_lpsi`와 대응 표준편차
- [ ] MF gradient를 다음처럼 분리해 기록한다.

```text
||A_pose|| = ||A_mf[:, 0:3]||
A_progress = A_mf[:, 3]       # 기대값 0
||A_cov|| = ||A_mf[:, 4:8]||
```

- [ ] 비용을 같은 cycle/horizon 기준으로 분해한다.

```text
J_MF_slack_per_cycle_horizon
J_progress_per_cycle_horizon
J_contour
J_lag
J_input
```

- [ ] cycle별 horizon cost를 원본으로 저장하고 summary에는 전체 cycle의 median/max 및 마지막 4초 window의 median을 각각 저장한다.
- [ ] 마지막 4초의 median `J_MF_slack/max(abs(J_progress),1e-9) >= 10`이고 status 0, `|v|,|v_s| <= 0.01 m/s`이면 비용에 의한 의도적 정지로 분류한다.
- [ ] 마지막 4초의 median `||A_cov||/max(||A_pose||,1e-12) >= 10`이고 속도가 0으로 가면 MF가 steering보다 covariance slowdown으로 작동하는 것으로 분류한다.

### 2.5 필요할 때만 per-feature 진단 추가

alpha/beta noise-growth를 제거하고 필요하면 true covariance-freeze까지 적용했는데도 probability sum이 큰 경우에만 아래 상세 로그를 켠다.

- [ ] stage/feature별 `family(R→O/O→R)`, obstacle index, distance, projected sigma, zeta, probability, gradient를 기록한다.
- [ ] stage별 `valid_feature_count`, `sum_probability`, `max_probability`, top-5, `count(p>1e-3)`, `count(p>1e-2)`를 기록한다.
- [ ] R→O/O→R family별 합과 obstacle별 합을 기록한다.
- [ ] 소수 feature 지배, 많은 작은 확률의 누적, 양방향 repeated-event counting을 분리한다.

### 2.6 실험 runner와 summary

- [ ] immutable base YAML에서 target factor 하나만 바꾼 실행별 YAML을 생성한다.
- [ ] 실행 전에 resolved parameter 전체를 `resolved_params.yaml`로 저장한다.
- [ ] baseline과 diff하여 allowlist에 target factor만 있는지 검사한다.
- [ ] 다음 필드의 `debug_summary.csv`를 자동 생성한다.

```text
commit, run_id, repeat, scenario, optimizer, factor, value,
status, failure_reason, final_time, distance_to_goal,
s_internal_max, s_physical_max, progress_ratio_internal, progress_ratio_physical,
min_executed_psdf, min_sampled_interval_psdf, collision_count,
main_solver_failure_count, final_active_solver_failure_count,
max_consecutive_solver_failures, safe_stop_count,
min_row_g_residual, mf_slack_max, mf_slack_mean,
J_MF_slack_cycle_median, J_MF_slack_cycle_max,
J_progress_cycle_median, J_progress_cycle_max,
cost_ratio_last_4s_median, gradient_ratio_last_4s_median,
verdict, cause_tag
```

- [ ] process exit code 0이나 config batch의 “completed successfully” 문자열을 합격 근거로 사용하지 않는다. 저장된 outcome과 CSV metric으로 판정한다.
- [ ] 기존 safe-stop batch summary에 `goal_reached`, 두 progress metric, collision metric을 추가하거나, runner가 `run_single_test()` 반환값과 metric을 별도 JSON/CSV로 저장한다.
- [ ] 아래 output 격리 규칙을 실행별 config에 적용해 tracked `data/`, `figures/`, `animations/`를 덮어쓰지 않는다.

```yaml
output_root_dir: "debug_runs/<commit>/<experiment_id>"
output_suffix: "<experiment_id>"
```

`output_root_dir`은 data/figure/animation만 격리하며 저장소 루트의 Acados JSON/code-export 이름은 고정되어 있다.

- [ ] 실행별 Acados artifact 이름/작업 디렉터리까지 격리하기 전에는 같은 worktree의 sweep을 반드시 직렬 실행한다.
- [ ] 병렬 실행이 필요하면 JSON과 code-export directory도 `run_id`별로 분리한 뒤에만 허용한다.

---

## 3. 고정 실험 계약

### 3.1 모든 시행에서 고정할 항목

| 항목 | 고정값 |
|---|---|
| robot / dynamics / planner | `rectangle / differential_drive / astar` |
| 실제 rectangle 크기 | 코드 기준 `0.15 x 0.09 m` |
| scenarios | `s_path`, `maze` |
| simulation limit | `60 s` |
| control interval | `0.1 s` |
| Row G | enabled, hard |
| `d_col` | `0.001 m` |
| `use_obstacle_constraint` | `True` |
| horizon / tf | `20 / 2.0 s` |
| input bounds | `v∈[-0.7,0.7]`, `omega∈[-1.2,1.2]`, `v_s∈[0,0.7]` |
| path, initial pose, obstacles | 동일 환경이면 완전히 동일 |
| success tolerance | baseline과 candidate에 동일 |
| noise/seed/device | localization error off, 동일 seed, CPU |

`use_obstacle_constraint=False`는 Row G와 Row MF를 함께 끄므로 비교군으로 사용하지 않는다.

### 3.2 OFAT 원상복구 절차

모든 값과 두 환경에 대해 아래 절차를 반복한다.

- [ ] `B0` immutable config를 복사해 새 `run_id`를 만든다.
- [ ] target factor 한 개만 override한다.
- [ ] resolved parameter diff가 target allowlist와 일치하는지 확인한다.
- [ ] 새 Python process와 새 optimizer instance로 `s_path`를 먼저 실행한다.
- [ ] `s_path` 결과가 FAIL이어도 별도 process로 `maze`를 반드시 실행한다.
- [ ] 한 환경이라도 FAIL이면 해당 값의 전체 판정은 FAIL로 한다.
- [ ] 각 시행 결과를 고유 output directory에 보존한다.
- [ ] 다음 값으로 넘어가기 전에 target override를 폐기하고 다시 `B0`에서 생성한다.
- [ ] 한 파라미터 sweep이 끝나면 baseline hash와 source diff를 다시 확인한다.

빠른 screening은 값당 각 환경 1회 실행하고, 최종 복합 후보만 각 환경 3회 반복한다. 난수가 완전히 제거된 경우에도 최종 후보는 3회 반복해 solver 재현성을 확인한다. 한 trial은 충돌 시 즉시 종료하지만 다른 환경 trial은 생략하지 않는다.

---

## 4. RMPCC-PV 기준선 측정

PV는 config의 `rmpcc` 블록을 먼저 읽고 `rmpcc_pv` 블록으로 덮어쓴다. 따라서 RMPCC 후보용 `rmpcc:` 설정이 들어 있는 config로 PV 기준선을 실행하지 않는다.

### 4.1 기준선 실행

- [ ] `rmpcc:` 후보 override가 없는 PV 전용 config를 만든다.
- [ ] PV config에도 candidate와 같은 success criteria, trial failure criteria, localization-error off 조건을 명시한다.
- [ ] `s_path`와 `maze`를 각각 3회, 60초 제한으로 실행한다.
- [ ] 3회 모두 `status=success`, `goal_reached=true`, 무충돌인지 확인한다.
- [ ] 각 환경에서 세 run의 `max(path_s)`와 `max(executed_projected_s)` 범위가 각각 `<=1e-3 m`인지 확인한다. 넘으면 `S_ref`를 정하지 않고 재현성 문제를 먼저 해결한다.
- [ ] solver failure가 있는 PV run은 기준선 표본에서 제외하지 말고 기준선 자체의 문제로 처리하고 원인을 먼저 해결한다.

```bash
python test_nmpc.py --single \
  -c <PV_BASELINE_CONFIG> \
  --maze-type s_path \
  --robot-shape rectangle \
  --optimizer-type rmpcc_pv \
  --dynamics-type differential_drive \
  --path-planner astar \
  --simulation-time 60 \
  --no-animation --no-plots

python test_nmpc.py --single \
  -c <PV_BASELINE_CONFIG> \
  --maze-type maze \
  --robot-shape rectangle \
  --optimizer-type rmpcc_pv \
  --dynamics-type differential_drive \
  --path-planner astar \
  --simulation-time 60 \
  --no-animation --no-plots
```

두 환경 outcome을 한 번에 저장할 때는 다음 batch 형식을 사용할 수 있다. 단, 현재 batch summary만으로는 `goal_reached`와 progress/collision gate를 판정할 수 없으므로 Section 2.6의 확장 summary 또는 runner를 함께 사용한다.

```bash
python test_nmpc.py \
  -c <PV_BASELINE_CONFIG> \
  --safe-stop-batch \
  --safe-stop-robot-shapes rectangle \
  --safe-stop-maze-types s_path maze \
  --safe-stop-optimizer-types rmpcc_pv \
  --simulation-time 60 \
  --no-animation --no-plots \
  --safe-stop-summary <RUN_DIR>/pv_outcomes.csv
```

### 4.2 진행도 기준 정의

optimizer 내부 진행도가 물리적 이동보다 앞서는 경우를 방지하기 위해 두 종류의 `s`를 함께 사용한다.

```text
S_ref_internal(env) = max over 3 stable successful PV runs of max(path_s)
S_ref_physical(env) = max over 3 stable successful PV runs of max(executed_projected_s)
```

- [ ] 사용자 요구의 주 비교값으로 `path_s` 기준을 기록한다.
- [ ] 실제 pose가 이동하지 않고 `v_s`만 전진하는 false progress를 막기 위해 physical projection도 필수 gate로 사용한다.
- [ ] 기존 `planner_projected_s_raw`는 pre-actuation 보조값으로 보존하고 post-actuation `executed_projected_s`와 일치 추세를 확인한다.
- [ ] `predicted_s`는 미래 예측값이므로 완주 판정에 사용하지 않는다.
- [ ] 경로 총 길이 sanity check는 `s_path≈0.987914 m`, `maze≈2.891614 m`로 한다.
- [ ] goal tolerance 때문에 완주 시 `s`가 이론적 총 길이보다 조금 작을 수 있으므로 이론값이 아니라 성공한 PV 측정값을 최종 기준으로 사용한다.

---

## 5. PASS/FAIL 및 조기 중단 규칙

실험 전에 아래 허용오차를 고정한다.

```text
tau_s = 0 m
tau_phi = 1e-6 m
constraint_tolerance = 1e-6
stop_speed_threshold = 0.01 m/s
cost_dominance_ratio = 10
gradient_dominance_ratio = 10
```

`tau_s=0`을 기본으로 하여 PV 기준보다 조금이라도 덜 진행하면 FAIL로 판정한다. 별도 참고 열에는 `1 mm` 이내 차이를 `NUMERICALLY_CLOSE`로 표시할 수 있지만 PASS로 승격하지 않는다.

### 5.1 Functional PASS

각 환경에서 아래를 모두 만족해야 한다.

- [ ] `status == success` 및 `goal_reached == true`
- [ ] `max(path_s) + tau_s >= S_ref_internal`
- [ ] `max(executed_projected_s) + tau_s >= S_ref_physical`
- [ ] 모든 executed/substep PSDF가 `>= -tau_phi`

사용자 정의에 따라 기준 PV보다 덜 진행하거나 한 번이라도 충돌하면 해당 시행은 FAIL이다.

### 5.2 Clean PASS

최종 후보가 되려면 Functional PASS와 아래를 모두 만족해야 한다.

- [ ] `main_solver_failure_count == 0`
- [ ] `final_active_solver_failure_count == 0`
- [ ] `safe_stop_count == 0`
- [ ] 모든 실제 pose에서 `PSDF >= d_col - constraint_tolerance`
- [ ] `min_row_g_residual >= -constraint_tolerance`
- [ ] NaN/Inf 및 process exception 없음

### 5.3 FAIL 원인 태그

| 태그 | 조건 |
|---|---|
| `PROGRESS_SHORT` | PV 기준 `s` 미달 또는 timeout/stuck |
| `COLLISION_AFTER_NORMAL_SOLVE` | status 0 입력 뒤 PSDF 침투 |
| `COLLISION_AFTER_BACKUP_RECOVERY` | backup solver가 반환한 입력 뒤 PSDF 침투 |
| `COLLISION_AFTER_SAFE_STOP` | safe-stop 감속 입력 뒤 PSDF 침투 |
| `HARD_CLEARANCE_BREACH` | collision 전이라도 `PSDF < d_col-tol` |
| `SOLVER_NUMERIC_FAIL` | nonzero status, QP/NLP failure, non-finite |
| `MF_COST_STOP` | status 0, 큰 MF cost/slack과 함께 `v,v_s≈0` |
| `COVARIANCE_STOP` | `A_cov` 지배 및 covariance 증가와 함께 정지 |
| `FEATURE_ACCUMULATION` | true covariance-freeze에서도 probability sum 초과 |

### 5.4 조기 중단

- [ ] collision, NaN/Inf, process exception이면 현재 환경의 시행을 즉시 종료한다.
- [ ] 4초 동안 실제 이동량이 `<=0.03 m`이면 `stuck`으로 종료한다.
- [ ] 연속 solver failure가 20회면 종료한다.
- [ ] weak MF anchor부터 nonzero status가 반복되면 cost sweep을 무작정 계속하지 않고 solver/conditioning branch로 이동한다.
- [ ] alpha/beta=0에서도 정지하면 covariance trajectory를 확인하고 Section 8.2의 true freeze로 kinematic propagation을 분리한다.
- [ ] true covariance-freeze에서도 정지 상태의 `probability_sum > epsilon`이면 growth sweep을 중단하고 per-feature branch로 이동한다.

---

## 6. 대조군과 weak-MF anchor

### 6.1 대조군

- [ ] `G0`: `use_row_mf=False`로 두 환경을 실행해 Row G-only 완주를 현재 harness에서 재확인한다.
- [ ] `R0`: 아래 checkpoint 기본값으로 Row MF-on 실패를 재현한다.

```text
linear=100, quadratic=1, epsilon=0.20,
d_min=0.001, d_mf_mask=0.001,
mf_active_start_step=1, mf_active_end_step=20,
mf_active_terminal=True
```

- [ ] `G0`와 `R0`에서 경로, 초기 상태, 나머지 파라미터가 동일함을 resolved config diff로 확인한다.

### 6.2 독립 sweep의 immutable anchor `B0`

```yaml
success_criteria:
  position_tolerance: 0.01
  angle_tolerance: 0.01

localization_error:
  enabled: false

trial_failure_criteria:
  enabled: true
  movement_window_sec: 4.0
  movement_threshold: 0.03
  max_consecutive_solver_failures: 20

rmpcc:
  use_obstacle_constraint: true
  use_row_mf: true
  augmented_psdf_device: cpu
  d_col: 0.001
  d_min: 0.001
  d_mf_mask: 0.0
  mf_slack_linear: 1.0
  mf_slack_quadratic: 0.1
  chance_epsilon: 0.20
  mf_active_start_step: 1
  mf_active_end_step: 5
  mf_active_terminal: false
  q_s: 1.0
  sigma_f0: 0.0002
  sigma_l0: 0.00025
  sigma_psi0: 0.00025
  q_f0: 0.0
  q_l0: 0.0
  q_psi0: 0.0
  alpha_f: 0.002
  alpha_v: 0.0004
  alpha_kappa: 0.01
  beta_v: 0.02
  beta_kappa: 0.008
  beta_omega: 0.008
  enable_backup_solver: false
```

- [ ] `B0`를 두 환경에 먼저 실행한다.
- [ ] `B0`가 status 0으로 진행하면 primary OFAT sweep을 시작한다.
- [ ] `B0`가 status 0이지만 정지하면 MF cost/covariance 진단값을 확인한다.
- [ ] `B0`부터 solver failure/safe-stop이면 primary cost sweep 전에 Section 9 solver branch를 실행한다.

---

## 7. Primary OFAT sweep

각 행은 별도 실험이다. target 이외의 모든 값은 매번 `B0`로 원복한다. 각 값은 `s_path`와 `maze`에 각각 적용한다.

| ID | 단일 factor | 시험값 | 선택 규칙 |
|---|---|---|---|
| `P1` | `mf_slack_linear` | `1, 3, 10, 30, 100` | 두 환경 Clean PASS 중 가장 큰 값 |
| `P2` | `mf_slack_quadratic` | `0.1, 1, 10` | 큰 violation을 억제하면서 Clean PASS인 가장 큰 값 |
| `P3` | `mf_active_end_step` | `3, 5, 10, 20` | terminal off 상태에서 Clean PASS인 가장 먼 구간 |
| `P4` | `chance_epsilon` | `0.20, 0.30, 0.50` | Clean PASS인 가장 작은 epsilon |
| `P5` | `covariance_growth_scale` | `0, 0.01, 0.1, 0.3, 1.0` | Clean PASS인 가장 큰 original scale |

`P3`에서 `N=20`, terminal off, `end_step=20`의 실제 active intermediate set은 stage `1..19`다.

`P5`는 optimizer에 존재하지 않는 YAML 키를 추가하는 방식으로 실행하지 않는다. 실험 runner가 아래 여섯 control-dependent 계수에 같은 배율을 곱한 실제 override 여섯 개로 확장하는 derived screening factor로 구현한다. 현재 `q_f0/q_l0/q_psi0=0`은 그대로 둔다.

```text
alpha_f, alpha_v, alpha_kappa,
beta_v, beta_kappa, beta_omega
```

- [ ] `P5`의 resolved parameter diff allowlist에는 위 여섯 derived key만 허용한다.
- [ ] 모델 동역학과 shifted nominal covariance 재전파가 동일한 여섯 resolved 값을 사용하는지 smoke test로 확인한다.

### 7.1 각 primary sweep 체크리스트

- [ ] 가장 약한/완화된 값에서 시작해 강한 값 방향으로 진행한다.
- [ ] 값마다 resolved config가 target factor만 바뀌었는지 확인한다.
- [ ] `s_path` 후 `maze`를 실행한다.
- [ ] `S_ref` 대비 두 진행도, min PSDF, solver/safe-stop, MF slack, cost ratio를 기록한다.
- [ ] PASS→FAIL 경계의 양쪽 값을 보관한다.
- [ ] expected monotonicity와 반대 결과가 나오면 반복 실행하고 mask/solver mode를 확인한다.
- [ ] `chance_epsilon=0.50`은 원인 확인용이며 근거 없이 최종 안전 설정으로 선택하지 않는다.
- [ ] `covariance_growth_scale=0`에서만 진행하면 alpha/beta noise-growth 항을 원인으로 확정한다.

---

## 8. Conditional/secondary OFAT sweep

Primary 결과로 필요한 branch만 실행한다. 각 표 안에서도 한 번에 한 factor만 바꾸고 나머지는 해당 branch anchor로 원복한다.

### 8.1 MF domain/threshold 진단

| ID | factor | 시험값 | 비고 |
|---|---|---|---|
| `D1` | `d_min` | `0, 0.001` | `d_col=0.001`은 계속 고정 |
| `D2` | `d_mf_mask` | `0, 0.001, 0.005` | domain 진단용; 먼 horizon 해결용 아님 |
| `D3` | `mf_active_terminal` | `False, True` | nonterminal horizon이 통과한 뒤에만 시행 |

- [ ] `d_mf_mask`를 최종 보수성 knob처럼 사용하지 않는다. 값이 커질수록 벽 가까이에서 MF가 꺼진다는 점을 결과에 명시한다.
- [ ] `D3`에서는 intermediate window를 그대로 유지하고 terminal stage `N`만 독립적으로 변경한다.
- [ ] terminal on에서만 실패하면 먼 terminal covariance/cost 효과로 분류한다.

### 8.2 true covariance-freeze 진단

`P5=0`은 alpha/beta noise-growth 항만 제거한다. covariance recursion의 `omega²P`, `vP_lpsi`, `v²P_psi`, `vP_psi` kinematic propagation은 남으므로 이를 fixed covariance라고 부르지 않는다.

| ID | debug factor | 시험값 | 용도 |
|---|---|---|---|
| `C0` | `freeze_covariance_state` | `False, True` | kinematic propagation과 초기 covariance 효과 분리 |

- [ ] `freeze_covariance_state`는 debug-only factor로 구현하고 model covariance dynamics와 shifted nominal 재전파 양쪽에서 동일하게 초기 covariance를 유지한다.
- [ ] `False`는 `P5=0` anchor, `True`는 모든 horizon stage에서 초기 covariance 고정으로 정의한다.
- [ ] 두 경로가 같은 covariance trajectory를 만드는 단위 테스트를 추가한다.
- [ ] `True`에서만 진행하면 남아 있던 kinematic covariance propagation을 원인으로 분류한다.
- [ ] 이 factor를 근거 없이 최종 production 설정으로 채택하지 않는다.

### 8.3 초기 covariance 민감도

초기 covariance 자체를 분리할 때만 각 sigma를 독립 시험한다. branch anchor는 `freeze_covariance_state=True`, `q_*=0`, `alpha_*=0`, `beta_*=0` 및 나머지 `B0` 값으로 고정한다.

| ID | factor | 시험값 |
|---|---|---|
| `C1a` | `sigma_f0` | `0.0001, 0.0002, 0.0004` |
| `C1b` | `sigma_l0` | `0.000125, 0.00025, 0.0005` |
| `C1c` | `sigma_psi0` | `0.000125, 0.00025, 0.0005` |

- [ ] 한 sigma를 바꿀 때 다른 두 sigma와 모든 covariance-growth coefficient를 위 branch anchor로 원복한다.
- [ ] 초기 sigma를 0으로 만들면 variance-validity mask와 원인 혼동이 생길 수 있으므로 이 sweep에서는 0을 쓰지 않는다.

### 8.4 covariance-growth 계수 위치 특정

`P5`에서 growth scale이 원인으로 확인된 경우에만 아래 여섯 factor를 각각 독립 시험한다. 각 factor 외 나머지 다섯 계수는 원래 값으로 원복한다.

| factor | 시험값 |
|---|---|
| `alpha_f` | `0, 0.0002, 0.0006, 0.002` |
| `alpha_v` | `0, 0.00004, 0.00012, 0.0004` |
| `alpha_kappa` | `0, 0.001, 0.003, 0.01` |
| `beta_v` | `0, 0.002, 0.006, 0.02` |
| `beta_kappa` | `0, 0.0008, 0.0024, 0.008` |
| `beta_omega` | `0, 0.0008, 0.0024, 0.008` |

- [ ] stage 1에서 covariance가 초기값보다 몇 배 커지는지 계수별로 기록한다.
- [ ] `A_cov/A_pose`, `v/omega`, probability sum의 변화를 함께 비교한다.

### 8.5 progress cost 후순위 시험

MF weight/horizon/covariance를 합리적 범위로 만든 뒤에도 status 0의 소극적 주행이 남을 때만 시행한다.

| ID | factor | 시험값 |
|---|---|---|
| `M1` | `q_s` | `1, 2, 5` |
| `M2` | `v_s_target` | `0.4, 0.6, 0.8` |

- [ ] 먼저 `q_s`만 시험한다. 큰 MF penalty를 progress cost로 억지로 덮지 않는다.
- [ ] `v_s_target`은 느리지만 지속적인 진행 여부를 확인할 때만 시험한다.
- [ ] `mat_Qe`, `mat_Re`, `q_s_ref`, horizon은 위 branch로 원인이 설명되지 않을 때 별도 후속 계획을 세우고 본 sweep에 섞지 않는다.

---

## 9. Solver/conditioning branch

`B0`처럼 매우 약한 MF에서도 nonzero status가 반복되거나 collision이 safe-stop 뒤에 발생할 때 실행한다. 각 변수는 독립 시험한다.

| ID | factor | 시험값 | 해석 |
|---|---|---|---|
| `S1` | `qp_solver_iter_max` | `50, 100, 200` | QP iteration 부족 확인 |
| `S2` | `tol` | `1e-3, 1e-4, 1e-5` | conditioning/수렴 민감도 확인 |
| `S3` | `nlp_solver_type` | `SQP_RTI, SQP` | RTI linearization 한계 확인 |
| `S4` | `enable_backup_solver` | `False, True` | recovery 효과 확인용 |

- [ ] failed iterate에서 hard Row G, dynamics/stage-0 equality, state/input box, soft MF-after-slack residual을 분리한다.
- [ ] row coefficient norm, QP status/iteration, NLP residual, covariance finite/PSD 여부를 기록한다.
- [ ] `SQP_RTI`에서 `nlp_solver_max_iter`를 올리는 시험은 반복 SQP와 같은 의미가 아니므로 우선 시행하지 않는다.
- [ ] backup을 켜서 충돌만 사라지고 main solver failure가 계속되면 근본 해결로 간주하지 않는다.
- [ ] main solver status 0인데 침투하면 solver branch가 아니라 hard linearization/execution-interval 진단으로 분류한다.
- [ ] solver failure 후 장애물 비인식 safe-stop에서 침투하면 recovery policy 문제를 별도 issue로 분리한다.

---

## 10. Per-feature/formulation branch

true covariance-freeze와 약한 weight에서도 정지 solution의 `probability_sum > epsilon`이 유지될 때 실행한다.

- [ ] stage별 probability sum이 초기 stage부터 큰지, 후반 stage에서만 커지는지 구분한다.
- [ ] top-5 feature가 합의 대부분을 차지하는지 확인한다.
- [ ] 한두 feature가 지배하면 해당 obstacle geometry, distance, projected sigma를 확인한다.
- [ ] 많은 작은 확률이 누적되면 Boole-sum conservatism과 valid feature 수를 확인한다.
- [ ] 같은 obstacle event가 R→O와 O→R 양쪽에서 중복 합산되는지 확인한다.
- [ ] `h_mf_nominal<0`이고 `A_pose≈0`이면 좌우 feature gradient 상쇄를 확인한다.
- [ ] 이 branch에서 formulation 문제가 확인되면 weight를 더 낮추는 것으로 종료하지 않고 별도 모델 수정 계획을 만든다.

---

## 11. 독립 결과 선택과 복합 검증

### 11.1 독립 변수 후보 선정

- [ ] 각 factor에서 두 환경 모두 Clean PASS한 값만 후보로 남긴다.
- [ ] `mf_slack_*`은 통과한 값 중 더 강한 penalty를 우선한다.
- [ ] active horizon은 통과한 값 중 더 긴 구간을 우선한다.
- [ ] epsilon은 통과한 값 중 더 작은 값을 우선한다.
- [ ] covariance scale은 통과한 값 중 original `1.0`에 가까운 값을 우선한다.
- [ ] 통과 값이 없으면 해당 factor를 복합 시험에 넣지 않고 원인 branch를 먼저 해결한다.

### 11.2 누적 조합 순서

확인된 값을 한꺼번에 모두 넣지 않고 아래 순서로 하나씩 누적한다. 매 추가 단계마다 두 환경을 다시 실행한다.

1. [ ] 선택한 `mf_slack_linear`
2. [ ] 선택한 `mf_slack_quadratic`
3. [ ] 선택한 MF active horizon/terminal
4. [ ] 선택한 covariance growth scale 또는 특정 계수
5. [ ] 선택한 `chance_epsilon`
6. [ ] 필요할 때만 `q_s` 또는 `v_s_target`

- [ ] 새 변수를 누적했을 때 회귀가 생기면 바로 이전 조합으로 돌아가 interaction을 기록한다.
- [ ] interaction 확인 시 두 값씩의 작은 조합만 추가 시험하며 전체 Cartesian product를 만들지 않는다.
- [ ] 각 조합은 `s_path`, `maze`를 모두 통과해야 다음 단계로 간다.

### 11.3 최종 후보 재현성 시험

- [ ] 최종 후보 1~3개를 선정한다.
- [ ] 각 후보를 각 환경에서 3회 실행한다.
- [ ] 총 6회 모두 Functional PASS 및 Clean PASS인지 확인한다.
- [ ] `S_ref` 대비 진행도, completion time, min PSDF, MF slack, probability sum, solver time 분포를 PV와 비교한다.
- [ ] 최종 resolved config, commit, 환경 정보, raw CSV, summary CSV를 보존한다.
- [ ] 가장 강한 MF 설정과 여유가 있는 fallback 설정을 각각 기록한다.

---

## 12. 실행 결과 표 템플릿

### 12.1 시행별 표

| run_id | env | factor | value | status | `s_int/S_ref` | `s_phys/S_ref` | min PSDF | main fail | final fail | safe-stop | min G | max MF slack | verdict/cause |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### 12.2 factor 요약 표

| factor | PASS 범위 | FAIL 경계 | `s_path` 최선 | `maze` 최선 | 원인 해석 | 복합 시험 채택값 |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |

### 12.3 최종 비교 표

| method/config | env | completion | `S_max` | min PSDF | main/final fail | safe-stop | MF slack max | median solve time | 최종 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `rmpcc_pv` baseline | `s_path` |  |  |  |  |  | N/A |  |  |
| `rmpcc_pv` baseline | `maze` |  |  |  |  |  | N/A |  |  |
| final RMPCC | `s_path` |  |  |  |  |  |  |  |  |
| final RMPCC | `maze` |  |  |  |  |  |  |  |  |

---

## 13. 종료 체크리스트

- [ ] PV 환경별 `S_ref_internal`, `S_ref_physical`을 확정했다.
- [ ] G-only, current-default MF, weak-MF anchor 결과를 보존했다.
- [ ] 모든 primary factor를 독립적으로 시험하고 매번 비대상 변수를 원복했다.
- [ ] 필요한 conditional branch만 실행했다.
- [ ] 충돌이 정상 solve 뒤인지 safe-stop 뒤인지 전부 분류했다.
- [ ] 독립 시험에서 확인된 변수만 복합 검증했다.
- [ ] 최종 후보가 두 환경의 PV 기준 진행도 이상을 달성했다.
- [ ] 최종 후보가 실제 pose 및 substep-sampled interval에서 무충돌이었다.
- [ ] 최종 후보가 Row G clearance/residual을 유지했다.
- [ ] 최종 후보의 main/final-active solver failure와 safe-stop이 모두 0회였다.
- [ ] 최종 설정, 코드 commit, resolved config, raw data, summary를 재현 가능하게 보존했다.
- [ ] 원인과 최종 선택 근거를 별도 `debugging_report.md`에 요약했다.
