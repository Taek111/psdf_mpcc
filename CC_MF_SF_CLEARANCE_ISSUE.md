# CC-MF와 CC-SF의 clearance 차이: 검토 요청

작성일: 2026-10-19. 분석 기준: `Taek111/psdf_mpcc`, commit `071cd802024cc88c09898ab8425673152ed82555`.

## 1. 현재 상황과 질문

좁은 구역에서 CC-MF의 안전성을 clearance로 평가하고자 한다. 그러나 동일한 maze에서 실행한 nominal 주행 1회씩의 결과는 **CC-SF가 CC-MF보다 최소·평균·P5 clearance 모두 높았다.** 이 차이가 확률 제약의 수치적 근사, 제어에 따른 공분산/경로 선택, 파라미터 또는 구현 문제에서 비롯되는지 판단하고 싶다.

- `CCMFOptimizer`와 `CCSFOptimizer`는 공통 `MPCCOptimizer`를 상속하며 tracking cost와 공통 설정을 공유한다. 각 controller의 safety constraint는 별도로 유지했다.
- 두 CC 모두 **Log-Cholesky 상태**를 유지한다. 공분산 원소를 직접 최적화하는 상태 표현으로 변경하지 않았다.
- `alpha_*`, `beta_*`는 MF 기준으로 통일했고 PSDF 계산은 CUDA를 사용한다.
- 이번 실행에서 세 controller 모두 목표 도달, 모든 solver status `0`, 접촉·겹침 샘플 `0`이었다. MF/SF의 backup 및 safe-stop 사용도 `0`회였다.
- 동작 실패는 재현되지 않았으며, 과거 controller 버전을 별도로 주행시킨 비교는 하지 않았다. 현재 자료만으로 리팩터링 회귀나 일반적인 안전성 우열을 확정할 수 없다.

## 2. 실행 조건과 clearance 결과

DD kinematics, `maze`, rectangle **0.15 × 0.07 m**, A* + Line of Sight, 동일 시작 pose `[0.075, 0.795, -pi/2]`. 시작 perturbation과 localization error는 비활성화했다. 각 방법 1회, 시간 제한 60 s, controller별 YAML override 없이 Python 기본값으로 실행했다.

Clearance는 실제 rectangle footprint와 모든 장애물 사이의 최소 Euclidean 거리 및 map boundary 간격의 최솟값이다. 저장된 **plant update 후 pose**를 사용하고 초기 pose는 아래 통계에서 제외했다. 단위는 mm다.

| Controller | 샘플 수 | 도달시간 (s) | 최소 clearance | 평균 clearance | P5 clearance |
|---|---:|---:|---:|---:|---:|
| Deterministic MPCC | 233 | 23.3 | 1.063 | 8.698 | 1.599 |
| CC-MF | 310 | 31.0 | 7.861 | 18.322 | 8.677 |
| CC-SF | 277 | 27.7 | 8.554 | 19.330 | 8.980 |
| SF − MF | — | −3.3 | +0.693 | +1.007 | +0.303 |

- P5는 선형 보간한 **5th percentile**이며, 가장 작은 5% 샘플들의 평균이 아니다.
- 0.1 s 간격의 시간 기준 통계다. 주행시간과 구간별 체류시간이 달라 평균/P5의 가중치도 달라진다. 동일 경로 진행률로 맞춘 비교는 아직 하지 않았다.
- MF 최소는 9.9 s, SF 최소는 8.9 s에 발생했다. 두 최소 지점은 같은 장애물 모서리 부근이지만 위치·heading은 다르다.
- 초기 pose까지 포함해 검사해도 접촉·겹침은 없었다. 샘플 사이의 연속 시간 안전성이나 localization noise가 있는 반복 실험의 충돌확률을 검증한 결과는 아니다.

## 3. 공통 파라미터와 의미

아래는 **CC-MF/SF의 이번 실행 값**이다. Deterministic MPCC에는 chance/covariance 항이 없다.

| 항목 | 공통 값 | 해석 |
|---|---|---|
| `use_obstacle_constraint` | `True` | 장애물 제약 활성화 |
| `d_min` | `0.001 m` | SF에서는 margin을 뺀 뒤의 하한, MF에서는 feature 위험확률의 거리 임계값 |
| `chance_epsilon` | `0.20` | stage별 모델 위험 예산; 전체 horizon/mission의 충돌확률이 아님 |
| `horizon`, `tf`, `dt` | `20`, `2.0 s`, `0.1 s` | 예측 구간 |
| `robust_horizon` | `None` → `10` | 첫 10 transition에서 공분산 전파 후 body-frame Log-Cholesky 상태 고정; 이후에도 safety constraint 적용 |
| `initial_pose_covariance` | `None` | 아래 표준편차로 매 제어 주기 초기 공분산 설정 |
| `sigma_f0`, `sigma_l0`, `sigma_psi0` | `0.0002 m`, `0.00025 m`, `0.00025 rad` | 초기 표준편차; 분산 값이 아님 |
| `q_f0`, `q_l0`, `q_psi0` | 모두 `0` | 기본 process-noise 항 |
| `alpha_f`, `alpha_v`, `alpha_kappa` | `0.0012`, `0.00024`, `0.006` | 아래 전진/측방 noise 증가식 계수 |
| `beta_v`, `beta_kappa`, `beta_omega` | `0.012`, `0.0048`, `0.0048` | 아래 heading noise 증가식 계수 |
| `vmin`, `vmax` | `−0.7`, `0.7 m/s` | 선속도 범위 |
| `omegamin`, `omegamax` | `−1.2`, `1.2 rad/s` | 각속도 범위 |
| `mat_Qe`, `mat_Re` | `diag(1, 10)`, `diag(1, 0.01)` | contour/lag 및 v/omega 비용 |
| `q_s`, `q_s_ref`, `terminal_weight` | `1`, `2`, `5` | progress/terminal 비용 |
| `v_s_target`, `v_s_max` | `0.8`, `2.0` | `enforce_v_s_not_faster_than_vmax=True`로 실제 progress 속도 상한은 `0.7` |
| `nlp_solver_type`, `integrator_type` | `SQP_RTI`, `DISCRETE` | 주기당 한 번의 RTI 수행 |
| `qp_solver`, `hessian_approx` | `PARTIAL_CONDENSING_HPIPM`, `GAUSS_NEWTON` | 공통 solver 설정 |
| `qp_solver_iter_max`, `nlp_solver_max_iter`, `tol` | `50`, `50`, `1e-4` | `nlp_solver_max_iter=50`이 주기당 50회 재선형화를 뜻하지 않음; `tol`은 clearance buffer가 아님 |
| PSDF device | CUDA | acados solver 자체는 CPU |

Log-Cholesky 상태는 `xi = [log(L_ff), log(L_ll), log(L_theta,theta), L_theta,l]`이며 공분산은 `L L^T`로 복원한다. 초기 공분산은 `diag(4e-8, 6.25e-8, 6.25e-8)`이다. 전진-측방 및 전진-heading 교차항은 생략하고 측방-heading 결합은 유지한다.

Process-noise 증분은 다음과 같다. `kappa`는 참조 경로 곡률이다.

$$
\begin{aligned}
Q_f &= \Delta t\,(q_{f0}+\alpha_f v^2),\\
Q_l &= \Delta t\,(q_{l0}+\alpha_v v^2+\alpha_\kappa v^2|\kappa|),\\
Q_\theta &= \Delta t\,(q_{\psi0}+\beta_v v^2+\beta_\kappa v^2|\kappa|+\beta_\omega\omega^2).
\end{aligned}
$$

**계수가 같아도 선택하는 입력과 경로가 다르면 예측 공분산과 feature 방향의 표준편차는 달라진다.**

## 4. 다른 safety constraint와 파라미터

### 제약식

**CC-SF:** nominal pose에서 PSDF 거리와 gradient를 고정하여 거리의 affine 모델 `phi_hat`과 body-frame 방향 `q_bar`를 만들고 다음 hard constraint를 사용한다.

$$
\hat\phi(x)-d_{\min}-\operatorname{clip}\!\left(\gamma\|L(\xi)^T\bar q\|,\;\mathrm{floor},\;\mathrm{cap}\right)\ge0.
$$

이 식도 acados의 RTI 과정에서 선형화된다. 후보 pose마다 PSDF geometry를 새로 평가하는 제약은 아니다.

**CC-MF:** 선택된 여러 feature의 affine Gaussian 거리 모델에 대해 비선형 위험 합을 정의한다.

$$
h_{\mathrm{MF}}(z)=\epsilon-\sum_\ell\Phi\!\left(\frac{d_{\min}-d_\ell}{\|L(\xi)^Tq_\ell\|}\right).
$$

실제로 solver에 전달하는 것은 nominal `z_bar`에서의 **확률합 전체의 1차 근사** `A_MF z + c_MF + slack >= 0`이다. 별도로 hard PSDF guard `phi_hat - d_col >= 0`을 둔다. MF row만 soft constraint다.

| 항목 | CC-MF | CC-SF |
|---|---|---|
| 위험 반영 방식 | 여러 feature의 Gaussian 위험 합 → affine row | PSDF 최소 feature 방향의 `gamma × sigma` margin |
| `d_col` | `0.001 m`; hard guard 및 현재 pose 사전 검사 | 별도 필드 없음 |
| `d_mf_mask` | `0.001 m`; nominal PSDF가 이 값보다 **클 때만** MF row 활성 | 해당 조건 없음 |
| 위험 항 switch | `use_row_mf=True` | `use_risk_margin=True` |
| `chance_epsilon` 입력 형태 | scalar 또는 길이 `N+1` 배열; 이번 값은 scalar `0.20` | scalar; 이번 값 `0.20` |
| `chance_gamma` | 없음 | `Phi^-1(0.8) = 0.8416212336` |
| `risk_margin_floor`, `risk_margin_cap` | 없음 | `0`, `2.0 m`; 이번 실행에서 cap 비활성 |
| stage 범위 | `mf_active_start_step=1`, `mf_active_end_step=None` → `N−1`, `mf_active_terminal=True` | 모든 제약 stage에 margin 적용, scale `1`; `risk_margin_active_start_step/end_step` 제거됨 |
| Slack 비용 | `mf_slack_linear=100`, `mf_slack_quadratic=1` | safety slack 없음 |
| 초기 상태 검사 | 실제 현재 PSDF `< d_col`이면 solve 진행 대신 safe-stop 경로 | MF와 같은 별도 현재-pose veto 없음 |
| Backup solver | `enable_backup_solver=False` | backup 생성 시도; 별도 enable 필드 없음 |
| 위험 진단 | `debug_mf=False`, `debug_infeasibility=False` | `debug_risk=True`, nominal head 출력 `False` |

실제 OCP의 safety row는 양쪽 모두 `1…N`에 적용된다. Stage 0은 초기 상태 등식으로 고정되며 별도 nonlinear safety row가 설치되지 않는다. MF의 현재-pose 사전 검사는 이와 별개다. MF의 mask는 **미래 nominal pose**에도 적용되므로 실행 pose의 최소 clearance만 보고 활성 여부를 판단할 수 없다.

설정 해석 시 주의점:

- 양쪽 `risk_cov_jitter=1e-9`, MF의 `covariance_growth_scale=0.6`, SF의 `use_projected_variance_margin=True`는 현재 생산 경로에서 수식을 바꾸지 않는 호환 필드다. 특히 `covariance_growth_scale`을 바꿔도 alpha/beta가 재계산되지 않는다.
- MF의 `constraint_debug_tolerance=1e-6`, `first_interval_substeps=10`은 진단용이다. Substep sampling이 OCP constraint나 실제 충돌 veto를 추가하지 않는다.
- MF는 기본값 → 최상위 `cc_mf` override 순서다. SF는 기본값 → 최상위 `cc_mf`(`use_risk_visualization` 제외) → 최상위 `cc_sf` 순서다. 이번 실험 설정에는 두 override 블록이 없다.
- SF의 유효 `chance_gamma`가 `chance_epsilon`보다 우선한다. YAML helper는 epsilon만 override하면 gamma를 재계산하지만, Python에서 epsilon만 변경하면 기존 gamma가 남는다.
- MF의 `d_mf_mask`는 생성 시 `d_col`을 한 번 복사한다. 이후 `d_col` 변경과 자동 연동되지 않는다. 설정 비교는 새 optimizer를 생성하여 수행해야 한다.

## 5. 확인된 사실, 가설 및 agent에게 요청할 검토

**확인된 사실:** MF 최대 slack은 약 `1.596e-10`으로 사실상 0이다. 따라서 큰 slack 완화 때문에 clearance가 줄었다는 설명은 지지되지 않는다. 현재 로그의 `mf_probability_sum`은 **solve 전 nominal 상태의 값**이다. 이를 post-solve constraint 위반으로 해석하면 안 된다. 최적화 후 새 geometry에서 평가한 비선형 MF 위험 합과 stage별 mask는 이번 결과에 저장되지 않았다.

**우선 검토할 가설:** MF의 확률합 선형화가 원래 위험 제약을 느슨하게 만들었을 가능성이 있다. 예를 들어 feature 하나, 고정 표준편차, affine 거리에서 `epsilon=0.2`이면 원래 조건은 `(d-d_min)/sigma >= 0.842`다. Nominal 값 `1.5`에서 Gaussian CDF를 선형화하면 하한은 약 `0.472`까지 낮아진다. 이는 가능성을 설명하는 예시이며 이번 주행에서의 오차를 측정한 값은 아니다. SF에도 geometry/Log-Cholesky 관련 RTI 근사가 존재한다.

동일한 separated pose·공분산·`d_min`·`epsilon`, 일치하는 gamma, cap/floor 비활성, SF의 active feature가 MF 집합에 포함된 조건에서는 **선형화 전 MF 위험 합 조건이 SF 조건을 함의한다.** 이 관계가 서로 다른 폐루프 경로의 최소 clearance나 실제 affine RTI 제약의 포함관계를 보장하지는 않는다.

다음 항목을 검토해 달라:

1. 현재 구현에서 MF/SF의 선형화 방식, feature 선택, mask 및 공분산 전파 차이가 clearance 역전을 설명할 수 있는가? 구현 오류가 의심되면 구체적인 코드 근거를 제시해 달라.
2. 동일 pose·Log-Cholesky 상태에서 두 제약을 교차 평가하고, MF 해의 affine residual / 고정 geometry에서의 비선형 위험 합 / 새 geometry에서의 위험 합을 비교하는 진단으로 원인을 구분할 수 있는가?
3. 원인을 가려내기 위한 최소한의 추가 로그와 통제 실험을 제안해 달라. 공분산 전파의 RTI dynamics defect와 미래 nominal mask 비활성 여부도 고려해 달라.
4. Nominal clearance뿐 아니라 동일 진행률의 좁은 구간 clearance, 불확실성 대비 여유, 동일 noise realization의 반복 주행을 통해 안전성을 어떻게 공정하게 비교할 것인가? MF 우세를 전제로 결론을 내리기보다 관측 결과와 불확실성을 구분해 달라.

검토·개선안에서도 **현재 Log-Cholesky 표현을 유지**해야 한다. 이 문서 작성 과정에서는 새 simulation/benchmark나 controller 수정 없이 저장 결과와 소스만 확인했다.

## 근거 파일

- [실험 설정](config/config_mpcc_refactor_validation.yaml), [실행 보고서](benchmark_runs/mpcc_refactor_validation_20261019/README.md), [사후 분석](benchmark_runs/mpcc_refactor_validation_20261019/posthoc/analysis.json)
- Clearance 원시 자료: [MPCC](benchmark_runs/mpcc_refactor_validation_20261019/posthoc/config01_trial_0001_clearances.csv), [MF](benchmark_runs/mpcc_refactor_validation_20261019/posthoc/config02_trial_0001_clearances.csv), [SF](benchmark_runs/mpcc_refactor_validation_20261019/posthoc/config03_trial_0001_clearances.csv). 표의 통계는 `sample_index > 0`으로 계산했다.
- [CC-MF](control/cc_mf_optimizer.py), [CC-SF](control/cc_sf_optimizer.py), [공통 covariance](control/cc_covariance.py), [공통 MPCC](control/mpcc_optimizer.py)
- [Feature 및 MF 계수 계산](models/augmented_psdf.py), [설정 적용 및 factory](sim/simulation_mpc.py), [방법 설명](method_description.md)

`method_description.md`는 수학적 배경 자료다. Stage 0 적용 여부와 초기 공분산 설정 등 실제 실행 동작은 위 소스 기준으로 확인해야 한다.
