# MPCC 복원 및 CC-MF / CC-SF 상속 계획

작성일: 2026-10-17 (Asia/Seoul)

수정일: 2026-10-19 (Asia/Seoul)

분석 기준: `70ffe5b106fe8f743150ac7a2435b63fbf64bfee`의 구현. 현재 코드에 대한 설명과 아래의 변경 후 목표를 구분한다.

## 1. 이번 작업의 결과와 변경 범위

`control/mpcc_optimizer.py`의 과거 버전을 찾았으며, 마지막으로 존재했던 버전을 **원문 그대로 복원**했다. 현재 잘 동작하는 CC-MF / CC-SF의 실행 코드는 수정하지 않았다. 이 문서는 이후 상속 리팩터링을 위한 설계와 검증 계획이다.

**사용자 요구에 따라 현재 CC-MF / CC-SF의 Log-Cholesky Parameterization을 그대로 유지한다.** 공분산 행렬 원소를 직접 최적화 상태로 사용하는 방식으로 바꾸지 않는다. 과거 MPCC에서는 공통 부모와 override 구조를 만드는 데 필요한 부분만 참고한다. 과거의 4차원 모델, ERK 설정, warm start, solve 절차로 현재 CC 구현을 되돌리지 않는다.

이번 변경 파일:

- `control/mpcc_optimizer.py`: 과거 원본 복원. 현재 factory/CLI에 `mpcc`를 다시 등록하지 않았다.
- `MPCC_OVERRIDE_PLAN.md`: 구현 분석, 설정 방식, 상속 경계, 단계별 작업 및 검증 계획.

이번에 수행한 확인은 Git 이력 조회, 소스·AST 정적 비교, 복원 파일의 원본 바이트 비교다. 저장소 지침에 따라 테스트·시뮬레이션·benchmark 및 optimizer import/solver 생성은 실행하지 않았다. 따라서 복원본의 현 환경 실행 성공이나 리팩터링 후 수치 동등성을 검증했다고 주장하지 않는다.

### 1.1 2026-10-19 확정 변경 사항

이번 문서 수정은 아래 세 항목을 후속 구현의 목표로 반영한다. 실행 코드에는 아직 적용하지 않았다.

| 항목 | 변경 후 목표 |
|---|---|
| SF risk margin 적용 범위 | `risk_margin_active_start_step`, `risk_margin_active_end_step` 설정과 단계 구간 판정을 제거한다. `use_risk_margin=True`이면 기존 안전 제약을 가진 모든 stage에 terminal까지 margin을 적용한다. |
| MF/SF alpha 계수 | MF의 현재 실제 값으로 통일: `alpha_f=0.0012`, `alpha_v=0.00024`, `alpha_kappa=0.006` |
| MF/SF beta 계수 | MF의 현재 실제 값으로 통일: `beta_v=0.012`, `beta_kappa=0.0048`, `beta_omega=0.0048` |
| device | MPCC 공통 부모와 MF/SF의 PSDF/PyTorch 경로는 모두 `cuda`를 사용한다. CPU 자동 fallback과 SF의 CPU 고정을 제거한다. |

이 세 항목은 기존의 동작·설정 보존 원칙에 대한 명시적 변경이다. Log-Cholesky 표현과 전파식의 형태, 나머지 제약·복구·시간·로그 계약은 유지한다. SF의 계수 변경과 CUDA 전환 후 결과가 과거 설정과 완전히 같아야 한다고 요구하지 않는다. MF의 별도 `mf_active_*` 설정과 geometry mask는 이번 제거 대상이 아니다.

## 2. 복원 출처

| 항목 | 확인 결과 |
|---|---|
| 최초 추가 | `8cad032cf6d416276b866846e7f1e9259042bd48`, 2026-08-09 |
| 중간 수정 | `8b72288316fbc1a269ba82a66c182b05008f104f`, 2026-08-19 |
| 복원 원본 / 마지막 수정 | `b642916ea91160b58ba19c3f8f28ae887f3f1b78`, 2026-10-12, `add cc-ecs` |
| 삭제 커밋 | `70ffe5b106fe8f743150ac7a2435b63fbf64bfee`, 2026-10-14, `update 1014` |
| 원래 경로 / 복원 경로 | `control/mpcc_optimizer.py` |
| Git blob | `1ae6fee465f7fabd6c84cfbcb26b483cfd8d2d5e` |
| SHA-256 | `f4e037e7e81757b46f69b29075ff59cbb42dd8d2f00accc11cf84115f22fc62e` |
| 파일 크기 | 27,493 bytes, 672 lines |

재확인 명령:

```bash
git log --all --name-status -- control/mpcc_optimizer.py
git show b642916ea91160b58ba19c3f8f28ae887f3f1b78:control/mpcc_optimizer.py
git hash-object control/mpcc_optimizer.py
```

복원 파일은 과거 구현의 출처가 명확한 참고 원본이다. 후속 작업에서 부모 클래스용으로 수정하더라도 위 커밋에서 원본을 다시 조회할 수 있다. 기존 `psdf` MPC와 CC-MF / CC-SF 선택 경로에는 연결하지 않았다.

## 3. 반드시 보존할 Log-Cholesky 표현

현재 두 CC 방법의 최적화 벡터는 다음과 같다.

```text
z = [x, y, theta, s, xi_f, xi_l, xi_theta, xi_c]  # nx = 8
u = [v, omega, v_s]                              # nu = 3

xi_f     = log(L_ff)
xi_l     = log(L_ll)
xi_theta = log(L_thetatheta)
xi_c     = L_thetal

L = [[exp(xi_f), 0,          0            ],
     [0,         exp(xi_l), 0            ],
     [0,         xi_c,      exp(xi_theta)]]

Sigma = L @ L.T
```

특히 `xi_c`는 `Sigma_ltheta` 자체가 아니다. `Sigma_ltheta = exp(xi_l) * xi_c`, `Sigma_thetatheta = xi_c**2 + exp(2*xi_theta)`다. `Sigma`는 초기 입력의 변환·해석·진단에 사용할 수 있지만 최적화 변수의 마지막 네 자리는 위 Log-Cholesky 좌표를 유지한다.

- [control/cc_covariance.py](control/cc_covariance.py), 27–75행: factor와 축약 공분산 간 변환.
- 같은 파일 78–123행: 초기값 설정. `initial_pose_covariance`가 없으면 `sigma_*0`를 표준편차로 보고 제곱한 뒤 Log-Cholesky로 변환한다.
- world-frame `initial_pose_covariance`가 있으면 현재 heading으로 body frame에 회전하고 forward–lateral / forward–heading 교차항을 제거한다. 이것은 현재의 모델 축약이며 임의 PSD repair가 아니다.
- 같은 파일 161–250행: 현재 NumPy / CasADi Log-Cholesky 전파식의 형태를 유지하고, MF/SF의 alpha/beta 기본값은 1.1절의 MF 기준으로 통일한다. 원소별 covariance 적분, clipping, 새 jitter 주입으로 대체하지 않는다.
- `robust_horizon=None`이면 `N_R=min(10,N)`. `i<N_R`인 transition에서만 Log-Cholesky를 전파하고 이후에는 그대로 유지한다. 이 구간 이후에도 SF margin은 전역 switch가 켜져 있으면 terminal까지 적용하며, MF는 기존 MF 활성 조건을 유지한다.
- 매 solve의 초기 Log-Cholesky 값은 현재 초기 불확실성 설정에서 다시 구성하고 horizon guess를 재생성한다. 이전 solver의 마지막 네 값을 그대로 shift하는 방식으로 바꾸지 않는다.
- 현재 optimizer는 진행도 `s`만 box bound로 제한한다. Log-Cholesky 변수에 공분산 원소용 bound를 추가하지 않는다.

여기서 Log-Cholesky 네 값이 현행 8차원 최적화 벡터에 포함된다는 점도 유지한다. 사용자의 “공분산 상태를 사용하지 말라”는 조건은 행렬 원소를 직접 상태로 쓰는 표현을 도입하지 않는 것으로 반영하며, 현행 Log-Cholesky 변수를 제거하거나 외부 고정값으로 바꾸는 변경은 계획에 포함하지 않는다.

## 4. 현재 구조와 과거 MPCC의 차이

| 항목 | PSDF MPC | 복원 MPCC | 현재 CC-SF | 현재 CC-MF |
|---|---|---|---|---|
| 추종 목적 | pose reference MPC | contour / lag / progress MPCC | 같은 MPCC 비용 | 같은 MPCC 비용 |
| 상태 차원 | 3 | 4 | 8, 마지막 4개 Log-Cholesky | 8, 마지막 4개 Log-Cholesky |
| 입력 차원 | 2 | 3 | 3 | 3 |
| 경로 parameter | pose reference | path 5개 | path 6개, curvature 포함 | path 6개, curvature 포함 |
| 동역학 | 연속 DD, ERK | 연속 DD, ERK | DISCRETE DD step + Log-Cholesky 전파 | 동일한 전파 구조 |
| 장애물 제약 | deterministic PSDF | deterministic Taylor PSDF | hard PSDF − projected margin | hard guard G + soft MF row M |
| terminal 안전 row | 이 계획의 통합 대상 아님 | 명시적 terminal row 없음 | hard row 1개 | G/M row 2개 |
| backup | 이 계획의 통합 대상 아님 | 없음 | 항상 생성 시도 | 기본 비활성, 설정 시 생성 |

PSDF MPC와 MPCC는 목적함수·상태·입력이 다르므로 `PSDFOptimizer`를 이번 공통 부모로 사용하지 않는다. 복원 MPCC의 deterministic 제약은 일반적인 설명이며 실제 OCP에는 nominal pose에서 만든 1차 Taylor PSDF가 들어간다. 현재 SF도 같은 analytic bridge를 사용한다.

근거: [control/psdf_optimizer.py](control/psdf_optimizer.py) 145–250행, [control/mpcc_optimizer.py](control/mpcc_optimizer.py) 175–448행, [control/cc_sf_optimizer.py](control/cc_sf_optimizer.py) 308–436 / 787–953행, [control/cc_mf_optimizer.py](control/cc_mf_optimizer.py) 410–519 / 857–1025행.

### 4.1 solver에 전달하는 stage parameter 배치

장애물 제약이 활성화되고 SF가 통상적인 7개 Taylor parameter를 사용할 때:

| 방법 | 배치 | 합계 |
|---|---|---|
| 복원 MPCC | path `[0:5]`, PSDF `[5:12]` | 12 |
| SF | path `[0:6]`, risk terms `[6:10]`, margin scale `[10:11]`, PSDF `[11:18]`, propagation flag `[18:19]` | 19 |
| MF | path `[0:6]`, guard affine `[6:10]`, MF affine `[10:19]`, propagation flag `[19:20]` | 20 |

`path6 = [p_ref_x,p_ref_y,t_ref_x,t_ref_y,s_ref,kappa_ref]`.
PSDF7은 `[pose_bar(3),phi_bar(1),gradient_world(3)]`다.
SF risk4는 `[q_f²,q_l²,q_theta²,q_l*q_theta]`다.
MF guard4는 `[gradient_world(3),intercept]`, MF9는 `[A_mf(8),c_mf]`다.

SF에서 PSDF model을 생략하면 PSDF block이 0개이므로 전체 크기도 줄어든다. 또한 SF의 자식인 CC-ECS는 risk/PSDF block 크기를 확장한다. 따라서 공통 부모에서 `np=19`, `risk_dim=4`, `psdf_dim=7`을 고정하면 안 된다. 각 방법이 layout을 정의하고 `model.p` 크기로 검증해야 한다.

SF의 단계별 활성 범위 옵션을 제거해도 margin scale slot과 `_build_stage_param(..., risk_margin_scale, ...)` 시그니처는 유지한다. SF는 `0..N`에서 scale을 모두 `1`로 채우고 margin의 전역 on/off는 `use_risk_margin`으로 처리한다. 이 slot을 공유하는 CC-ECS는 binary mask가 아니라 연속값인 ECS beta를 전달하므로 부모에서 삭제하거나 `1`로 고정하면 안 된다. propagation flag는 기존대로 `i<N_R`을 사용한다.

### 4.2 공통으로 추출할 MPCC 비용

현재 세 MPCC 계열에 공통인 식:

```text
t = t_raw / (norm(t_raw) + tangent_reg_delta)
p_ref_lin = p_ref + (s - s_ref) * t
e_l = -dot(position - p_ref_lin, t)
e_c = (x - p_ref_lin_x)*t_y - (y - p_ref_lin_y)*t_x
e_s = s - s_ref
r_vs = 0.5 * max(q_s, 0) / max(v_s_target, 1e-6)

l = q_c*e_c² + q_l*e_l² + q_s_ref*e_s²
    + r_v*v² + r_w*omega² + r_vs*v_s² - q_s*v_s
l_terminal = terminal_weight * (q_c*e_c² + q_l*e_l² + q_s_ref*e_s²)
```

부모의 `_build_mpcc_cost(param, x, u, path_p)` 같은 helper로 이 식만 추출한다. 현행 contour/lag 부호, tangent 정규화, terminal 배율, 연산 순서를 보존한다. 이 helper는 전달받은 앞 4개 상태와 path 앞 5개를 사용하고, Log-Cholesky 변수나 위험 모델을 생성하지 않는다. 각 자식은 지금의 `create_model()`에서 이를 호출한다.

## 5. 목표 제약과 유지할 실행 동작

### 5.1 CC-SF

- 변경 후 제약(`use_risk_margin=True`): `phi_taylor(pose) - d_min - clip(gamma*||L(xi).T q||, floor, cap) >= 0`. stage별 구간 mask는 사용하지 않는다.
- `q`는 nominal pose의 world gradient를 body frame으로 변환한 값이다. geometry gradient는 stage parameter이고 `L(xi)`는 현행 Log-Cholesky 변수에 의존한다.
- path와 terminal에 hard row를 유지한다. zero-gradient에서 margin과 미분이 유한하도록 한 분기까지 그대로 보존한다.
- 단계별 시작/종료 설정과 `_is_risk_margin_active_stage()`의 구간 판정을 제거한다. margin은 `robust_horizon` 이후 frozen tail을 포함해 terminal까지 적용한다. stage-0 equality 및 안전 row 설치 정책은 기존대로 두며 새 row를 추가하지 않는다.
- `use_risk_margin=False`는 margin만 제거한다. Log-Cholesky 모델이나 deterministic PSDF 제약을 없애는 옵션이 아니다.
- `use_obstacle_constraint=False`는 PSDF 초기화와 제약 추가를 생략한다.
- prepare 순서: 진행도 갱신 → 전체 stage-0 equality → bounds → finite 검사/clip → stage 진행도 예측 → Log-Cholesky guess 재생성 → PSDF batch → risk terms 및 전구간 scale=1 → terminal 포함 `N+1`개 parameter → solve.
- `_build_risk_margin_scale_batch()`는 SF에서 전구간 `1`을 반환하도록 단순화한다. `get_risk_margin_active_stage_mask()`와 risk 로그/그림도 전역 switch와 전구간 적용 기준으로 맞춘다. 삭제된 설정이 남은 YAML은 기존 unknown-key 경고 정책으로 처리하며, 활성 옵션으로 다시 읽지 않는다.
- main 실패 시 backup `SQP_WITH_FEASIBLE_QP`를 시도하고, 실패하면 nominal safe-stop plan을 사용한다. backup 성공 시 실제 active solver에서 결과·진행도를 읽고 primal guess를 main에 복사한다.
- safe stop에서 `v_s=0`, 진행도 고정 및 Log-Cholesky guess 재생성을 유지한다.

근거: [control/cc_sf_optimizer.py](control/cc_sf_optimizer.py) 437–561, 651–733, 897–953, 1032–1360행.

### 5.2 CC-MF

- Row G: `g_phi·pose + phi_bar - g_phi·pose_bar - d_col >= 0`. 항상 hard guard다.
- Row M: `A_mf·z+c_mf >= 0`에만 slack을 둔다. `h_bar=epsilon-sum(feature CDF)`, `c_mf=h_bar-A_mf·z_bar`이며 `A_mf`의 마지막 네 계수는 현행 Log-Cholesky 좌표에 대한 것이다.
- row 순서 G=0, M=1과 `idxsh=[1]`을 유지한다. intermediate/terminal에 각각 두 row, slack penalty는 1차원 vector다.
- MF 활성 조건은 `phi > d_mf_mask` AND 유효한 MF data AND stage window다. 경계의 strict inequality를 바꾸지 않는다. 비활성 M은 `A=0,c=epsilon`으로 설정하며 hard G는 유지한다.
- `d_col`은 guard/현재 pose의 충돌 기준, `d_min`은 MF 내부 확률 계산의 clearance, `d_mf_mask`는 MF 적용 domain 기준이다. 세 설정을 하나로 합치지 않는다.
- stage 0에는 OCP 안전 row를 넣지 않는다. 대신 측정 pose를 포함한 batch의 exact phi가 `d_col`보다 작으면 solve를 건너뛰고 safe stop한다. `phi == d_col`은 허용한다.
- `use_row_mf=False`이면 MF 평가만 생략하고 pose-only PSDF guard는 유지한다. 현행 8차원 모델과 epsilon 검증은 그대로 남는다.
- `forward_mf` batch 실패 시 pose-only guard 복구와 stage별 MF 재시도를 유지한다. 실패한 M만 비활성화하며 G까지 제거하지 않는다.
- `set_state()`에서 `_current_psdf=None` 캐시 무효화, 정상 cycle의 한 번의 batch 결과 재사용, safe-stop 시 spatial risk overlay 제거를 유지한다.
- 기존 retained warm-start trajectory를 수동으로 한 칸 shift하지 않는다.

근거: [control/cc_mf_optimizer.py](control/cc_mf_optimizer.py) 294–304, 556–802, 954–1025, 1163–1336, 1861–1939행; [models/augmented_psdf.py](models/augmented_psdf.py) 80–177행.

### 5.3 반환값·시간·진단을 성급하게 합치지 않는다

| 항목 | 현재 SF | 현재 MF |
|---|---|---|
| 시간 측정 | `time.time()`, prepare/recovery/debug 포함 | `perf_counter()`, 실제 `solver.solve()`만; main+backup 합 |
| solution 생성자 | `(solver,N,variables)` | `(solver,horizon,*,status_info=None)` |
| status | mutable solver의 status 조회 | cycle의 명시적 snapshot |
| stage-0 veto | MF의 전용 veto 흐름 없음 | `status_code=None`, `success=False`, `solver_called=False` |
| 로그/visualization | projected margin | MF residual/slack/Boole risk, stage0 exact PSDF |

현재 시간 정의의 차이는 별도 개선 대상으로 기록하고 이번 구조 리팩터링에서 바꾸지 않는다. MF의 skipped solve를 이전 `solver.status=0` 때문에 성공으로 처리하면 trial 결과가 바뀐다. safe-stop 계획 설치 실패 때의 예외와 backup 실패 때 cache 복원도 유지한다.

MF cycle CSV의 `solver_status`는 main status를 보존하지만 solution stats는 recovery 이후 effective status를 나타낼 수 있다. 이 차이도 의도적으로 유지한다. [control/controller.py](control/controller.py) 28–86행의 status 해석과 93–148행의 반환·로깅 호출을 계약으로 취급한다.

## 6. 현재 parameter 설정 방식과 변경 후 기본값

### 6.1 입력 경로와 우선순위

```text
load_config() / YAML 읽기
  -> mpc_test()에서 방식별 Param 객체 생성
  -> 알려진 속성만 override
  -> Param으로 SplineReferenceGenerator 구성
  -> optimizer 생성
  -> BaseController(optimizer, opt_param)
  -> 매 cycle setup(param,system,reference,obstacles) -> solve_nlp()

MF: CCMFOptimizerParam 기본값 -> 최상위 cc_mf 블록
SF: CCSFOptimizerParam 기본값 -> 최상위 cc_mf 블록 -> 최상위 cc_sf 블록
```

SF 병합에서는 `cc_mf.use_risk_visualization`만 먼저 제거한다. SF가 갖고 있지 않은 다른 MF 전용 키는 override 단계에서 경고 후 무시된다. `test_configs` 안에 넣은 개별 `cc_mf:` / `cc_sf:` 블록을 최상위에 병합하는 기능은 현재 없다.

근거: [sim/simulation_mpc.py](sim/simulation_mpc.py) 253–275, 1821–1907, 2354–2360, 2457–2465행; [test_nmpc.py](test_nmpc.py) 540–550행.

현재 [config/config.yaml](config/config.yaml)은 `psdf` 시나리오만 활성화되어 있고 활성 `cc_mf:` / `cc_sf:` override 블록이 없다. 따라서 CC를 선택할 때의 설정 출발점은 각 Python Param 기본값이다.

### 6.2 변경 후 공통 기본값

별도 표시한 변경 항목 외에는 현재 값을 유지한다. alpha/beta와 device는 이번 수정의 목표값이다.

| 그룹 | 기본값 |
|---|---|
| horizon | `horizon=20`, `tf=2.0`, `dt=tf/N=0.1` |
| MPCC 비용 | `mat_Qe=diag(1,10)`, `mat_Re=diag(1,.01)`, `q_s=1`, `q_s_ref=2`, `terminal_weight=5`, `v_s_target=.8` |
| 진행 속도 | `v_s_max=2`, `enforce_v_s_not_faster_than_vmax=True` |
| 물리 입력 | `vmin/vmax=-.7/.7`, `omegamin/omegamax=-1.2/1.2` |
| 진행도 bounds | `s_upper_guard=100`, `use_s_state_bounds=True`, `s_backtrack_tolerance=.3` |
| projection | `line_search_window=.5`, `line_search_samples=41` |
| 경로 | `tangent_reg_delta=1e-6`, `fd_eps=1e-3`, `max_segments=20` |
| 기존 호환 필드 | `path_gate_epsilon=.05`, `reference_weight_reg=1e-8` |
| 초기 불확실성 | `initial_pose_covariance=None`, `sigma_f0=.0002`, `sigma_l0=.00025`, `sigma_psi0=.00025` |
| 전파 | `robust_horizon=None` → `min(10,N)`, `q_f0=q_l0=q_psi0=0`, `risk_cov_jitter=1e-9` |
| alpha 계수 — MF/SF 통일 | `alpha_f=0.0012`, `alpha_v=0.00024`, `alpha_kappa=0.006` |
| beta 계수 — MF/SF 통일 | `beta_v=0.012`, `beta_kappa=0.0048`, `beta_omega=0.0048` |
| PSDF/PyTorch device — 통일 | `cuda`, CPU 자동 fallback 없음 |
| 기본 solver | `PARTIAL_CONDENSING_HPIPM`, `GAUSS_NEWTON`, `SQP_RTI`, QP/NLP 반복 최대 50/50, `tol=1e-4` |
| integrator | 실제 OCP는 두 CC 모두 `DISCRETE` 강제 |

solver의 유효 `v_s` 상한은 `min(v_s_max,max(0,vmax))=.7`이다. 그러나 planner는 원래 `v_s_max=2`로 `reference_speed=max(.25,.7*v_s_max)=1.4`, `window_length=max(v_s_max*tf*1.25,.8)=5.0`을 계산한다. 공통화하면서 planner도 `.7`로 계산하게 바꾸지 않는다.

planner는 추가로 `num_horizon=N`, `local_path_timestep=tf/N`, `sample_count=N`, `max_segments`, `proj_dist_buffer=.03`, `min_point_distance=1e-4`, natural spline, projected backtrack 기본 `.15`를 사용한다. 현재 두 CC는 `psdf_detection` 블록을 명시적으로 무시한다.

### 6.3 변경 후 각 방법 고유 기본값

| 항목 | MF | SF |
|---|---|---|
| clearance | `d_min=.001,d_col=.001,d_mf_mask=.001` | `d_min=.001` |
| 기능 | `use_row_mf=True`, `use_obstacle_constraint=True` | `use_risk_margin=True`, `use_obstacle_constraint=True` |
| epsilon | `.20`, scalar 또는 `N+1` stage 값 | `.20`, scalar |
| gamma | 없음 | `Phi^-1(1-.20)≈.8416212336` |
| slack | `mf_slack_linear=100`, `mf_slack_quadratic=1` | hard row, MF slack 없음 |
| 활성 구간 | 기존 MF start=1, end=None → N−1, terminal=True 별도 | 단계 구간 옵션 없음. 전역 switch가 켜져 있으면 terminal까지 전구간 적용 |
| margin | 해당 없음 | floor=0, cap=2 |
| growth scale | `covariance_growth_scale=.6` | scale 필드 없음 |
| backup | `enable_backup_solver=False` | 설정 필드 없이 항상 생성 시도 |
| diagnostics | `debug_mf=False`, `debug_infeasibility=False`, `use_risk_visualization=True` | `debug_risk=True`, nominal heads 출력=False, only_active=True |
| 기타 | `constraint_debug_tolerance=1e-6`, `first_interval_substeps=10` | `use_projected_variance_margin=True` 호환 필드 |

기존 값의 근거: [control/cc_mf_optimizer.py](control/cc_mf_optimizer.py) 33–113행, [control/cc_sf_optimizer.py](control/cc_sf_optimizer.py) 32–112행. 삭제할 SF 구간 옵션, 공통 alpha/beta 및 CUDA 전환은 1.1절의 변경 후 목표를 따른다.

MF의 여섯 실제 계수를 공통 초기화 함수에 한 번만 정의하여 두 CC Param에 적용한다. 위 값에는 MF의 `covariance_growth_scale=0.6` 효과가 이미 반영되어 있으므로 다시 `0.6`을 곱하지 않는다. MF의 실제 기본값은 유지하고 SF가 그 값에 맞춰진다. 기존 명시적 계수 override 경로는 유지하되 비교 실험에서는 두 방법에 동일한 설정을 전달한다.

### 6.4 override와 파생 값의 주의점

현재 `_apply_param_overrides()`는 `hasattr` 후 `setattr`만 한다. 이 기본 방식은 유지하되, 삭제한 SF 구간 옵션과 CUDA 사용 조건은 명시적 변경으로 반영한다. 아래는 현재 경로에서 주의해야 할 동작이다.

1. `horizon`만 바꿔도 `tf`는 자동 갱신되지 않는다. simulation은 `tf/N`과 plant timestep이 다르면 실패시킨다(1978–1983행).
2. `d_mf_mask=self.d_col`은 생성 시 값 복사다. `d_col`만 override해도 mask threshold는 따라 바뀌지 않는다.
3. `covariance_growth_scale`만 override해도 MF alpha/beta는 재계산되지 않는다. scale과 계수의 관계는 생성 시에만 반영된다. `run_optimizer_benchmark.py` 84–97행과 `run_cc_mf_debug_experiments.py` 139–149행은 별도로 6개 계수까지 계산하는 경로다.
4. SF의 최종 병합 블록에 epsilon이 있고 gamma가 없을 때만 helper가 gamma를 재계산한다. 명시된 유효 gamma가 우선한다. Python에서 epsilon만 직접 바꾸면 이미 생성된 gamma는 유지된다. MF의 `N+1`개 epsilon 배열이 SF에 그대로 병합되면 gamma 계산의 `float(epsilon)`에서 실패할 수 있다. 같은 YAML에서 MF에 stage별 배열을 쓰면 `cc_sf.chance_epsilon`은 scalar로 명시적으로 덮어써야 한다.
5. `mat_Qe` / `mat_Re`의 YAML list는 ndarray로 변환되지 않는다. 모델은 `[0,0]` 인덱싱하므로 현재 loader에서 행렬 YAML override가 정상 지원된다고 가정하면 안 된다. 직접 Python 설정 시에는 ndarray를 사용한다.
6. `planner_projected_s_backtrack_tolerance`, `hpipm_mode`, `regularize_method`는 코드가 읽을 수 있지만 기본 Param에는 없어 YAML helper에서는 unknown으로 무시된다. backup은 Python에서 속성을 직접 추가한다.
7. `integrator_type` YAML 키가 받아들여져도 현재 CC OCP는 `DISCRETE`를 강제한다. 부모의 ERK 설정을 상속하지 않는다. Hessian 관련 주석과 무관하게 현재 기본값은 `GAUSS_NEWTON`이다.
8. 현재 `path_gate_epsilon`, `reference_weight_reg`, `risk_cov_jitter`와 SF의 `use_projected_variance_margin`은 해당 CC 수식에서 동작을 바꾸는 분기로 사용되지 않는다. 리팩터링 중 새 의미를 부여하지 않는다.
9. `initial_pose_covariance`는 명시적 설정이다. localization-error 설정에서 자동으로 estimator covariance를 연결하는 경로는 없다.
10. Param의 모든 필드를 solver 생성 후 자유롭게 바꿀 수 있는 구조가 아니다. 비용·동역학 계수·차원·제약 구조는 solver 생성 시 수식에 고정된다. stage parameters, 현재 pose, reference, 장애물, 초기 Log-Cholesky 구성 등은 cycle마다 갱신되는 별도 경로다. 생성 후 alpha/beta 또는 dt 관련 값을 바꾸면 Python의 guess 전파는 새 값을 읽고 이미 생성된 CasADi 동역학은 이전 값을 유지하는 불일치도 생길 수 있다. 이런 변경은 재생성이 필요하며, 이번 계획에서 새 live-update 보장을 추가하지 않는다.

변경 후에도 사용할 설정 예시(이번 작업에서 YAML에 적용하지 않음):

```yaml
# 최상위 블록. 아래 cc_mf 값은 SF에도 먼저 적용된다.
cc_mf:
  horizon: 20
  tf: 2.0
  d_col: 0.001
  d_mf_mask: 0.001
  chance_epsilon: 0.20
  enable_backup_solver: false
  # MF의 실제 값이며 SF에도 먼저 적용된다. scale을 다시 곱하지 않는다.
  alpha_f: 0.0012
  alpha_v: 0.00024
  alpha_kappa: 0.006
  beta_v: 0.012
  beta_kappa: 0.0048
  beta_omega: 0.0048

cc_sf:
  chance_epsilon: 0.20   # gamma가 명시되지 않으면 helper에서 재계산
  use_risk_margin: true
```

이 예시의 MF 전용 키는 SF에서 unknown 경고를 낼 수 있다. MF와 SF의 완전히 독립적인 설정이 필요하면 별도 YAML을 사용하거나 각 방법의 공통 속성을 SF 블록에서 명시적으로 덮어쓴다. SF가 MF 설정 블록을 먼저 적용하는 fallback의 제거, 새 `mpcc:` 블록 도입, 타입 변환/파생값 자동화는 상속 리팩터링과 별도 변경으로 다룬다.

### 6.5 CUDA 통일 범위

- MPCC 공통 부모 및 MF/SF의 `initialize_ped_model()` / `initialize_augmented_psdf()`, `AugmentedPSDFWrapper`, `AnalyticPSDFCasADi`에 동일한 `cuda` device를 명시적으로 전달한다. 생성자 기본값만 바꾸지 말고 setup의 `device="cpu"` 전달과 CUDA 가용성에 따른 CPU 선택도 제거한다.
- geometry vertices/edges, pose, Log-Cholesky 입력 tensor, MF epsilon, PSDF/risk 계산 tensor와 wrapper/bridge의 device를 맞춘다. main/backup과 진단 경로도 같은 CUDA 정책을 사용한다.
- 기존 device 설정 속성을 호환성 때문에 남길 경우에도 CUDA만 허용한다. CUDA를 사용할 수 없으면 PSDF 초기화 시 명확한 오류로 종료하고 CPU로 자동 전환하지 않는다.
- acados/CasADi solver와 기존 NumPy 전파까지 GPU solver로 교체하는 작업은 포함하지 않는다. PSDF 결과를 acados stage parameter로 넘기는 경계의 `.detach().cpu().numpy()` 변환은 유지한다. CUDA 전환에 따른 수치 허용오차와 전송 비용은 검증에서 별도로 확인한다.

## 7. 권장 상속 구조

```text
MPCCOptimizer                  # 공통 진행도/경로/MPCC 비용 + deterministic 기본 구현
├── CCSFOptimizer              # 현재 Log-Cholesky 모델 + SF 제약/solve/recovery override
│   └── CCECSOptimizer         # 기존 상속 체인 유지
└── CCMFOptimizer              # 현재 Log-Cholesky 모델 + MF 제약/solve/recovery override
    + CCMFDiagnosticsMixin    # 현재 진단 역할 유지
```

목표 선언은 `CCSFOptimizer(MPCCOptimizer)`, `CCMFOptimizer(CCMFDiagnosticsMixin, MPCCOptimizer)`다. MF를 SF의 자식으로 만들거나 두 클래스를 alias로 합치지 않는다. 공통 부모는 MF/SF 방식 이름으로 분기하는 거대한 클래스가 되지 않도록 하고, 기존 Log-Cholesky 수식의 공유 위치인 `control/cc_covariance.py`를 계속 사용한다.

### 7.1 바로 상속 가능한 메서드

정적 AST 비교상 세 클래스에서 동일한 메서드:

- `get_progress_info`
- `_clamp_s`
- `_get_current_s_bounds`
- `set_reference_trajectory`
- `update_obstacles`
- `reset`, `__del__` — 단, 각 자식의 `cleanup()`을 그대로 dispatch하도록 유지해야 한다.

SF는 추가로 `set_state`, `_get_effective_v_s_max`, `_get_progress_speed_penalty`, `_project_s_with_line_search`, `update_reference_path_params`도 복원 MPCC와 동일하다.

MF는 이 추가 그룹에 작은 차이가 있다. `set_state`는 PSDF cache를 지우고, projection은 `np.asarray`로 입력을 변환하며, 일부 parameter는 `getattr` 기본값 대신 직접 읽는다. 이 차이는 override/adapter로 보존한다. 정상 입력의 계산이 같다는 이유만으로 누락 속성의 예외 동작까지 같다고 가정하지 않는다.

### 7.2 작은 helper 추출 후 공유할 부분

| 대상 | 부모에 둘 공통 부분 | 자식이 유지할 내용 |
|---|---|---|
| MPCC 비용 | `_build_mpcc_cost(param,x,u,path_p)` | 현행 8차원 모델 및 Log-Cholesky 전파 |
| 경로 parameter | 기존 `stage_parameter()` 호출, curvature 포함 여부 hook | MPCC=5, CC=6과 curvature 사용 |
| stage 진행도 예측 | `_predict_stage_s_values(s0,lower,upper,solver=None)` | backup용 solver 전달 및 기존 blending/clipping |
| OCP 기본 설정 | 비용 종류, 입력 bounds, 차원 설정의 공통 helper | full x0, state bounds, parameter layout, DISCRETE, safety rows |
| 생성 자원 | JSON/code-export/model 이름을 명시적으로 받는 작은 helper | 방법별 main/backup 이름과 생성·해제 정책 |
| Log-Cholesky 변환/전파 | 기존 `cc_covariance.py` 함수 재사용, MF 기준 공통 alpha/beta 초기화 | 현재 4개 factor 변수, 전파식 형태, 초기화, horizon flag |
| PSDF device | 공통 CUDA 초기화·검증 | 방법별 wrapper/bridge 종류와 solver 전달 경계 |

`_predict_stage_s_values`는 현재 SF/MF가 동일하고 `solver=None`을 받는다. 과거 부모는 `self.solver`만 읽는다. 부모에 optional 인자를 추가해 예전 호출과 backup 호출을 모두 보존한 다음 중복을 제거한다.

경로 helper에는 예를 들어 `_path_includes_curvature=False`를 기본으로 두고 두 CC에서 `True`로 지정할 수 있다. 실제 layout과 반환 길이를 함께 검사한다. 과거의 `include_curvature=False`를 CC에 그대로 상속하면 전파에 필요한 `kappa_ref`가 사라진다.

첫 단계부터 `setup_ocp`, `setup`, `solve_nlp` 전체를 부모에 몰아넣지 않는다. 같은 수식·작은 설정 블록부터 공유하고, 수치·예외·로그 계약이 다른 부분은 명시적인 override로 남기는 편이 유지보수에 유리하다.

### 7.3 첫 리팩터링에서 유지할 override

| 메서드/영역 | SF | MF |
|---|---|---|
| `__init__`, `cleanup` | main/backup·risk cache 초기화/정리 | diagnostics·status·PSDF cache·자원 초기화/정리 |
| `set_state` | 부모와 동일하므로 상속 가능 | 부모 호출 + `_current_psdf=None` |
| `create_model` | Log-Cholesky 8-state, risk/PSDF layout | Log-Cholesky 8-state, guard/MF layout |
| `_build_stage_param` | SF block pack | MF block pack |
| `setup_ocp` | `(param, reference_path_data)` 시그니처 유지 | `(param)` 시그니처 유지 |
| `create_solver` | `(json_filename=None,code_export_directory=None)`, solver 반환 | 같은 공개 시그니처와 solver 반환 유지 |
| `add_obstacle_avoidance_constraint` | `(param,system,obstacles_geo)`, hard SF | `(param,obstacles_geo)`, hard G / soft M |
| `add_warm_start`, `_prepare_and_solve` | 현재 guess/finite 검사/parameter 갱신 순서 | 현재 cache·batch·stage0 veto 순서 |
| `setup`, `solve_nlp`, recovery | 현재 backup 정책·timing·stats | 현재 선택적 backup·safe stop·timing snapshot |
| PSDF 초기화 | `initialize_ped_model`, analytic bridge, CUDA | `initialize_augmented_psdf`, MF wrapper, CUDA |
| solution/diagnostics | 기존 `AcadosSolution` 및 risk getter | 기존 `AcadosSolution` 및 mixin/log getter |

단순히 `super().create_model()`을 부른 뒤 4차원 상태에 네 값을 덧붙이는 방식은 사용하지 않는다. 자식에서 현재 심볼·동역학을 만들고 공통 cost helper만 호출한다. 과거 `create_solver()`는 `self.solver`를 대입하고 반환하지 않으므로 현재 자식의 solver-return 계약으로 그대로 대체할 수 없다.

### 7.4 Param 구조

초기 단계에서는 `CCMFOptimizerParam`, `CCSFOptimizerParam` 이름을 보존한다. 허용 속성 집합은 SF의 두 구간 옵션 삭제와 CUDA 설정 검증을 제외하고 유지한다. 공통 비용/진행도/물리 입력 기본값은 작은 initializer로 추출하고, MF/SF의 여섯 alpha/beta 값은 별도 공통 initializer에서 동일하게 설정한다. 더 나아간다면 **공통 추종 부분만 담은** `MPCCTrackingParam`을 두고 세 Param이 각각 이를 사용하게 할 수 있다. 불확실성이 없는 deterministic 부모에 alpha/beta를 강제로 추가할 필요는 없다.

`CCMFOptimizerParam(MPCCOptimizerParam)`처럼 복원 Param 전체를 바로 상속하지 않는다. 과거 Param에는 `use_soft_constraint`, `slack_weight`, `detection_*` 등 현재 MF/SF에 없는 속성이 있다. 이 속성이 생기면 `hasattr` 기반 loader에서 예전에는 무시하던 YAML이 갑자기 허용되어 설정 계약이 달라진다.

SF/MF 고유 safety/risk/backup 기본값은 삭제 대상 SF 구간 옵션을 제외하고 각 클래스에 남긴다. device는 공통 CUDA 정책을 따른다. matrix 객체는 인스턴스마다 새로 만들고 mutable default를 공유하지 않는다. 필드 집합·타입·수치·파생값 계산 시점은 1.1절의 의도된 변경과 나머지 호환성 항목을 구분해서 비교한다.

### 7.5 기존 CC-ECS 및 자원 관리

[control/cc_ecs_optimizer.py](control/cc_ecs_optimizer.py)는 이미 SF를 상속하고 있다(104행). 아래 계약도 유지한다.

- `CCSFOptimizer.__init__(variables=None,costs=None,dynamics_opt=None)`
- `create_model(param,p_psdf_sym=None)`
- `setup_ocp(param,reference_path_data)`
- `create_solver(json_filename=None,code_export_directory=None)`
- `from control.cc_sf_optimizer import AcadosSolution`
- CC-ECS가 바꿀 수 있는 risk/PSDF block 크기와 slices.

SF의 margin scale slot은 CC-ECS에서 ECS beta 값에 사용되므로 유지하고, SF의 전구간 scale=1 로직을 ECS에 강제하지 않는다. MF/SF의 alpha/beta 통일은 CC-ECS의 process noise=0 규약을 변경하지 않는다. 공통 경로를 상속하는 CC-ECS의 PSDF device도 CUDA 정책에 맞추되, 자체 Param의 CPU fallback과 초기화 호출을 함께 확인한다. ECS의 위험 모델과 별도 계수는 그대로 둔다.

복원 MPCC의 cleanup은 공용 `acados_ocp.json` / `c_generated_code`를 직접 정리한다(115–124행). CC 자식 cleanup에서 이 구현으로 무조건 `super()` 위임하지 않는다. SF/MF의 현재 main/backup 경로를 유지하고, 자원 공통화를 할 때는 실제 해당 인스턴스가 생성한 파일/디렉터리만 추적하는 별도 helper를 사용한다.

## 8. 후속 구현 순서와 완료 조건

1. **기준 고정:** 현재 CC 파일과 Param/solver 계약을 기준으로 둔다. 복원 원본의 commit/blob를 보존한다. 이번 작업에서 완료.
2. **최소 상속:** 위 클래스 선언만 연결하고 현재 생성자·모델·제약·solve/recovery·cleanup은 override로 유지한다. 완전히 동일한 메서드만 부모로 올린다.
3. **MPCC 비용 추출:** 세 방법의 비용식을 helper 하나로 공유한다. Log-Cholesky 표현과 전파식 형태를 유지한다.
4. **경로/진행도 정리:** curvature hook, optional solver, MF cache/입력 변환 adapter를 도입하고 동일성이 확인된 중복만 제거한다.
5. **OCP/Param 반복 정리:** 작은 공통 설정 helper와 공통 기본값 initializer를 추출한다. 각 방법의 차원·DISCRETE·backup·공개 시그니처를 유지한다.
6. **확정한 세 변경 적용:** SF 구간 옵션과 구간 판정을 제거하고 전구간 margin을 적용한다. MF 기준의 여섯 alpha/beta 실제 값을 공유하며 중복 scale 적용을 막는다. PSDF 초기화·wrapper·bridge·진단을 CUDA로 통일하고 CPU fallback을 제거한다. 기존 SF 구간 제한 테스트의 기대값도 전구간 정책으로 갱신한다.
7. **회귀 검증 후 확대 판단:** 아래 검증이 통과한 범위만 채택한다. 의도된 세 변경과 구조 리팩터링의 영향은 나누어 비교한다. solve/recovery/diagnostics 통합은 동작 보존 이득이 확인될 때 별도로 진행하며 첫 완료 조건으로 삼지 않는다.

완료 기준은 “공통 MPCC 계산을 한 곳에서 유지하고, SF 전구간 margin·MF 기준 공통 계수·CUDA 통일을 적용하며, 그 밖의 기존 Log-Cholesky 표현과 제약·실행·설정·로그 계약을 보존하는 것”이다.

## 9. 검증 계획 — 현재 실행하지 않음

저장소의 테스트·benchmark 미실행 지침을 유지한다. 아래는 이후 사용자가 실행을 요청한 환경에서 수행할 계획이다.

| 검증 | 확인 내용 / 기존 근거 |
|---|---|
| 설정 호환성 | 삭제된 SF 구간 키가 더 이상 영향을 주지 않고 unknown 경고 대상인지 확인. 그 외 필드·타입·값, SF 병합 우선순위, epsilon/gamma, scale/threshold 파생값, planner 인자 보존 |
| 공통 계수 | MF/SF의 여섯 alpha/beta 기본값이 1.1절과 동일한지, 이미 반영된 `.6` 배율을 중복 적용하지 않는지 확인 |
| Log-Cholesky | `tests/test_cc_covariance.py`: factor 변환, 기존 전파, 양수 초기 표준편차, frozen tail |
| 모델 동등성 | `tests/test_cc_mf.py`의 SF/MF dynamics·cost 비교, frozen tail 입력 민감도 0, bound 검사. 같은 MF 기준 공통 계수에서 비교 |
| 물리 step | `tests/test_dd_dynamics.py`: DD pose map과 plant RK4 일치 |
| 경로 | `tests/test_spline_path.py`: curvature 포함, fallback, projection; main과 backup solver의 진행도 예측 |
| SF 안전 제약 | `tests/test_cc_sf.py`: 기존 구간 제한 기대값을 전구간 적용으로 갱신. signed gradient factor norm, zero-gradient/terminal 유한 미분, global off, frozen tail에서도 margin 유지 |
| MF 안전 제약 | `tests/test_cc_mf.py`: p slices, G hard/M soft, strict mask, window/terminal, MF 오류 시 G 유지 |
| MF stage 0 | `tests/test_cc_mf_stage0.py`: exact phi 재사용, threshold equality, solver skip, stale status 차단, cache reset, stop 설치 실패 |
| timing/status/log | `tests/test_cc_mf.py`, `tests/test_cc_mf_psdf_logging.py`: main+backup solver-only 시간, 예외, 성공 통계, 중복 PSDF 호출 방지 |
| 간접 상속 | `tests/test_cc_ecs.py`, 필요 시 `tests/test_cc_ecs_integration.py`: SF 시그니처·가변 parameter block·ECS beta slot·process noise=0 보존 |
| CUDA | 초기화 호출·wrapper·bridge·PSDF/risk/진단 tensor의 device 일치, CPU 자동 fallback 부재, CUDA 불가 시 명확한 오류, solver 경계의 CPU 변환 및 수치 허용오차 |
| 자원 관리 | main/backup 이름 분리, 다른 optimizer의 생성 파일 보존, reset/cleanup/부분 초기화 |

추가로 기존 구현과 리팩터링 구현에 동일한 입력을 넣어 stage/terminal 비용과 미분, discrete map, constraint value/Jacobian, `p` 배열, bounds/slack 설정, prepare 시 solver에 전달한 값을 비교한다. 구조 리팩터링의 비교 기준은 기존 SF에도 MF 기준 계수·전구간 margin·CUDA를 적용한 구현으로 맞춘다. 기존 MF도 CUDA에서 비교한다. 이 기준에서는 shape/index/flag가 같아야 하며, 원래 SF 설정과의 차이는 의도된 변경 영향으로 따로 기록한다. 부동소수 계산은 기존 허용오차를 출발점으로 CUDA의 정밀도 차이를 확인하고 차이가 크면 원인을 조사한다.

실제 acados 회귀 실행은 같은 환경·설정·seed·시작 pose에서 순차 수행한다. 첫 제어 입력, 경로 진행도, solver status, 복구 mode/count, terminal 제약, 성공/실패 이유를 비교한다. SF/MF의 기존 시간 측정 의미는 따로 확인한다. 실행할 때마다 새 결과 디렉터리를 사용하고 기존 CSV를 덮어쓰지 않는다.

## 10. 집에서 이어서 작업할 때

먼저 이 문서 3절의 Log-Cholesky 보존 조건과 7절의 override 경계를 기준으로 시작한다. 복원된 파일이 있다는 이유만으로 factory에 `mpcc`를 등록하거나 CC의 `create_model/setup/solve_nlp`를 부모 구현으로 교체하지 않는다.

다음 구현 단위는 **“최소 상속 연결 + 완전히 동일한 helper 제거 + 공통 MPCC 비용 추출 + 확정한 세 변경 적용”**이다. SF 구간 옵션 삭제, MF 기준 alpha/beta, CUDA 통일은 이번 계획에 포함한다. 그 밖의 파라미터 스키마 개선, 추가 튜닝, 위험 모델 변경, 시간 정의 통일은 분리한다. 원본 복원과 계획만 저장된 상태이며, 이번 수정도 문서에만 반영했다. 실제 상속과 새 설정 적용은 후속 구현 작업이다.
