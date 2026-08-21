# RMPCC Row MF infeasibility·충돌 해결 계획

## 0. 목표와 완료 기준

목표는 `control/rmpcc_optimizer.py`의 hard Row G와 soft Row MF를 함께 사용할 때 발생하는 정지, solver failure, 충돌을 해결하고, `s_path`와 `maze`에서 기준 진행도에 도달하는 설정을 찾는 것이다.

- Row G는 hard constraint로 유지하고 `d_col=0.001`은 바꾸지 않는다.
- 환경은 `rectangle / differential_drive / astar / {s_path, maze}`로 고정한다.
- 모든 Primary OFAT는 `R0`에서 factor 하나만 바꿔 직렬 실행한다.
- 실험 checkbox 하나는 같은 설정으로 `s_path`와 `maze`를 차례로 실행하는 작업 하나를 뜻한다.

진행 완료는 `S_ref - s_max <= 0.1`, 즉 `s_max >= S_ref - 0.1`로 판정한다.

여기서 `s`는 기존 결과 CSV의 `predicted_s`이며, `s_max=max(predicted_s)`로 정의한다.

| scenario | `S_ref` | 완료 허용차 | 최소 완료 `s_max` |
|---|---:|---:|---:|
| `s_path` | `0.9803319765` | `0.1` | `0.8803319765` |
| `maze` | `2.8829378751` | `0.1` | `2.7829378751` |

각 환경의 PASS 조건은 다음과 같다.

- 위 진행 완료 기준을 만족한다.
- `min_psdf >= d_col - 1e-6`이다.
- solver failure와 NaN/Inf가 0회다.

---

## 1. 기준 설정

| ID | 용도 | 설정 |
|---|---|---|
| `G0` | Row G-only 대조군 | `use_row_mf=False` |
| `R0` | 현재 실패 재현 및 OFAT 기준 | 현재 코드값, `use_row_mf=True` |

`R0`의 주요 값은 다음과 같다.

- `mf_slack_linear=100`, `mf_slack_quadratic=1`, `chance_epsilon=0.20`
- `d_min=0.001`, `d_mf_mask=0.001`
- 별도 active-stage window 없이 stage `1..N`에서 domain/valid mask 적용
- `sigma_f0=0.0002`, `sigma_l0=0.00025`, `sigma_psi0=0.00025`
- `q_f0=q_l0=q_psi0=0`
- `alpha_f=0.002`, `alpha_v=0.0004`, `alpha_kappa=0.01`
- `beta_v=0.02`, `beta_kappa=0.008`, `beta_omega=0.008`

모든 실행은 localization error를 끄고 동일 seed/device, horizon/tf `20 / 2.0 s`, simulation limit `60 s`를 사용한다.

---

## 2. 튜닝 전 필수 구현

### 2.1 MF 활성 구간

- [x] `mf_active_start_step`, `mf_active_end_step`, `mf_active_terminal`을 파라미터와 config에 추가한다.
- [x] intermediate와 terminal mask를 분리하고 기존 동작과 호환되는 기본값을 둔다.

`start/end`는 inclusive이며 intermediate에는 stage `1..N-1`만 적용한다. stage `N`은 `mf_active_terminal`만 따른다.

### 2.2 MF 및 covariance trajectory 진단

- [x] solver가 유지한 nominal trajectory의 stage별 `mf_probability_sum`과 solved MF slack trajectory를 `solve_mode`, solver status와 함께 기록한다.
- [x] 매 solve cycle의 horizon에서 covariance trajectory를 별도 파일로 저장한다.

`covariance_trajectory.csv`는 아래 8개 필드만 사용한다.

```text
time, stage, v, omega,
sigma_f, sigma_l, sigma_psi, P_lpsi
```

- `sigma_f=sqrt(max(P_f,0))`, `sigma_l=sqrt(max(P_l,0))`, `sigma_psi=sqrt(max(P_psi,0))`로 계산한다.
- velocity가 커지는 stage부터 covariance와 MF violation이 커지면 covariance-growth 계수가 원인이다.
- 정지 해에서 covariance가 거의 증가하지 않으면 optimizer가 covariance 증가를 피하려고 정지를 선택한 것이다.
- stage 1부터 MF violation이 크면 초기 covariance 또는 feature probability sum을 확인한다.

### 2.3 실험 runner와 summary

- [x] immutable `R0` config에서 target factor 하나만 override하고 실행별 config/output directory를 만드는 직렬 runner를 구현한다.

cycle raw record는 아래 8개 필드만 저장한다. `mf_probability_sum`과 `mf_slack`은 horizon 배열이다.

```text
time, solve_mode, solver_status, u0,
s, psdf, mf_probability_sum, mf_slack
```

`debug_summary.csv`도 아래 8개 필드만 저장한다.

```text
run_id, scenario, setting, status,
s_max, min_psdf, solver_failure_count, mf_slack_max
```

- `setting`에는 `R0`, `linear=3`, `horizon=1-10`처럼 factor와 값을 함께 기록한다.
- `status`는 `success`, `stuck`, `collision`, `clearance_fail`, `solver_fail`, `error` 중 하나다.
- repeat와 전체 파라미터는 `run_id`와 실행별 resolved config로 추적한다.
- `solver_failure_count`는 main/backup의 모든 nonzero status와 safe-stop 발생을 센다.

---

## 3. 기준 실행

- [x] `G0`: 두 환경에서 Row G-only 진행도, clearance, solver status를 측정한다.
- [x] `R0`: 두 환경에서 현재 Row MF의 정지, solver failure, 충돌을 재현한다.

`G0`가 FAIL이면 Row MF 튜닝을 시작하지 않고 Row G 또는 simulation 문제를 먼저 해결한다. 이후 모든 OFAT 결과는 `R0`와 비교한다.

---

## 4. Primary OFAT

각 설정은 `R0`에서 target factor만 변경한다. 각 실행 후 `s_max`, `min_psdf`, solver failure, MF slack, covariance trajectory를 비교한다.

### 4.1 `mf_slack_linear`

- [x] `P1-L1`: `mf_slack_linear=1`로 두 환경을 실행한다.
- [x] `P1-L3`: `mf_slack_linear=3`으로 두 환경을 실행한다.
- [x] `P1-L10`: `mf_slack_linear=10`으로 두 환경을 실행한다.
- [x] `P1-L30`: `mf_slack_linear=30`으로 두 환경을 실행한다.

`linear=100`은 `R0` 결과를 재사용한다. 두 환경이 PASS한 값 중 가장 큰 값을 우선한다.

### 4.2 `mf_slack_quadratic`

- [x] `P2-Q0.1`: `mf_slack_quadratic=0.1`로 두 환경을 실행한다.
- [x] `P2-Q10`: `mf_slack_quadratic=10`으로 두 환경을 실행한다.

`quadratic=1`은 `R0` 결과를 재사용한다. 두 환경이 PASS한 값 중 가장 큰 값을 우선한다.

### 4.3 `chance_epsilon`

- [x] `P3-E0.3`: `chance_epsilon=0.30`으로 두 환경을 실행한다.
- [x] `P3-E0.5`: `chance_epsilon=0.50`으로 두 환경을 실행한다.

`epsilon=0.20`은 `R0` 결과를 재사용한다. 두 환경이 PASS한 값 중 가장 작은 epsilon을 우선하며, `0.50`은 원인 확인용으로만 사용한다.

### 4.4 MF active horizon

terminal MF를 끄고 아래 순서로 horizon을 늘린다.

- [x] `P4-H5`: active horizon `1-5`로 두 환경을 실행한다.
- [x] `P4-H10`: active horizon `1-10`으로 두 환경을 실행한다.
- [x] `P4-H20`: active horizon `1-20`으로 두 환경을 실행한다.

두 환경이 PASS한 가장 긴 horizon을 우선한다. `N=20`, terminal off에서 `1-20`의 실제 intermediate stage는 `1..19`다.

### 4.5 Covariance growth

`covariance_growth_scale`은 runner의 derived factor다. 원래의 `alpha_f`, `alpha_v`, `alpha_kappa`, `beta_v`, `beta_kappa`, `beta_omega`에 같은 배율을 적용한다. 현재 0인 `q_f0/q_l0/q_psi0`과 초기 `sigma_*`는 유지한다.

- [x] `P5-C0`: covariance growth scale `0`으로 두 환경을 실행한다.
- [x] `P5-C0.01`: covariance growth scale `0.01`로 두 환경을 실행한다.
- [x] `P5-C0.03`: covariance growth scale `0.03`으로 두 환경을 실행한다.
- [x] `P5-C0.1`: covariance growth scale `0.1`로 두 환경을 실행한다.
- [x] `P5-C0.3`: covariance growth scale `0.3`으로 두 환경을 실행한다.

`scale=1.0`은 `R0` 결과를 재사용한다. 두 환경이 PASS한 값 중 `1.0`에 가장 가까운 값을 우선한다.

---

## 5. 실패 원인 분기

### 5.1 Solver failure

- [x] 최초 failed cycle에서 hard Row G residual, dynamics/bound violation, MF-after-slack residual, solver status를 확인한다.
- [ ] 확인된 원인 하나만 수정하고 같은 설정을 다시 실행한다.

- `R0/maze` 첫 failure: status `4`, Row G `-0.0109`, Row M-after-slack `-0.6912`, dynamics equality `0.0584`; box bound 위반은 없었다.

weak MF 또는 짧은 horizon에서도 nonzero status가 반복될 때만 solver 설정을 조정한다.

### 5.2 충돌

- [x] 충돌 직전 `solve_mode`, solver status, `psdf`를 확인해 normal solve 충돌인지 recovery 이후 충돌인지 분류한다.
- [ ] normal solve면 Row G 경로를, recovery 이후면 safe-stop/recovery 입력을 수정하고 같은 설정을 다시 실행한다.

- `R0/maze`는 normal solve가 시작한 충돌이다. 직전 solve는 exact predicted PSDF `-0.0099`에서 `v=0.7`을 적용했고, 다음 cycle current PSDF가 `-0.0130`이 된 뒤 safe-stop이 `-0.1221`까지 악화시켰다.

### 5.3 Covariance scale 0에서도 정지

- [ ] covariance trajectory에서 stage 1부터 큰지, velocity에 따라 성장하는지 확인한다.
- [ ] 초기 stage부터 probability sum이 크면 per-feature probability/family/obstacle 진단을 추가해 top feature와 R→O/O→R 중복을 확인한다.

---

## 6. 최종 조합과 검증

단독 PASS 값이 있으면 아래 우선순위로 선택한다.

- linear/quadratic penalty: 가장 큰 값
- `chance_epsilon`: 가장 작은 값
- active horizon: 가장 긴 값
- covariance scale: `1.0`에 가장 가까운 값

효과가 확인된 factor만 하나씩 누적한다.

- [ ] 선택한 linear를 `R0`에 적용해 두 환경을 실행한다.
- [ ] 선택한 quadratic을 추가해 두 환경을 실행한다.
- [ ] 선택한 epsilon을 추가해 두 환경을 실행한다.
- [ ] 선택한 active horizon을 추가해 두 환경을 실행한다.
- [ ] 선택한 covariance scale을 추가해 두 환경을 실행한다.
- [ ] 회귀가 생기면 마지막 factor를 되돌리고 인접 값 하나를 실행한다.

최종 조합 screening은 아래 반복 횟수에 포함하지 않는다.

- [ ] 최종 후보를 두 환경에서 반복 1회차로 실행한다.
- [ ] 최종 후보를 두 환경에서 반복 2회차로 실행한다.
- [ ] 최종 후보를 두 환경에서 반복 3회차로 실행한다.
- [ ] source commit, 최종 resolved config, raw record, covariance trajectory, 8-field summary를 보존한다.

최종 완료 조건은 3회 모두 환경별 `s_max` 기준, clearance, solver failure 조건을 PASS하는 것이다.
