# RMPCC-MF 고정 2행 안전 제약 구조 변경 내역

## 1. 변경 목적

`RMPCCOptimizer`의 장애물 안전 제약을 horizon 전체에서 의미와 순서가 변하지 않는 두 개의 scalar affine row로 구성했다.

- Row 0: signed-PSDF 기반 Guard/Recovery
- Row 1: Multi-Feature(MF) probability margin

중간 stage `i = 1, ..., N-1`과 terminal stage `N`에는 항상 두 행이 존재한다. Stage 0은 측정 상태가 고정되는 지점이므로 `con_h_expr_0`을 정의하지 않는다.

## 2. 상태 및 stage parameter 구성

상태 순서는 다음과 같다.

```text
[x, y, theta, s, P_f, P_l, P_theta, P_ltheta]
```

입력 순서는 기존과 동일하다.

```text
[v, omega, v_s]
```

각 stage의 parameter는 총 19개 scalar로 구성된다.

| 구간 | 차원 | 내용 |
|---|---:|---|
| Path | 6 | `[p_ref_x, p_ref_y, t_ref_x, t_ref_y, s_ref, kappa_ref]` |
| Guard | 4 | `[g_x, g_y, g_theta, c_guard]` |
| MF | 9 | `[A_mf(8), c_mf]` |

```text
p_stage = [path parameters, guard parameters, MF parameters]
```

코드의 parameter slice도 위 순서에 맞게 고정했다.

## 3. Row 0: signed-PSDF Guard/Recovery

Nominal pose를 `x_bar_i = [x_bar, y_bar, theta_bar]`, 해당 위치의 signed PSDF와 world-frame gradient를 각각 `phi_bar_i`, `g_bar_i`라고 하면 다음 affine row를 사용한다.

```text
h_guard(z_i) = A_guard_i^T z_i + c_guard_i
```

```text
A_guard_i = [g_x, g_y, g_theta, 0, 0, 0, 0, 0]
c_guard_i = phi_bar_i - g_bar_i^T x_bar_i - d_col
```

따라서 nominal pose 주변에서 다음 signed-PSDF 선형화와 동일하다.

```text
h_guard(z_i) = phi_bar_i + g_bar_i^T (x_i - x_bar_i) - d_col
```

주요 동작은 다음과 같다.

- 모든 stage에서 동일한 `d_col`을 사용한다.
- 별도의 recovery mode, progressive target, `rho_recovery`를 사용하지 않는다.
- 접촉 또는 penetration 상태에서도 Row 0의 의미나 계수를 바꾸지 않는다.
- penetration에서는 signed PSDF gradient와 높은 우선순위의 lower slack이 같은 Row 0을 recovery 제약으로 동작시킨다.
- gradient는 `forward_mf()`가 반환하는 world-frame `[d/dx, d/dy, d/dtheta]`를 그대로 사용한다.

## 4. Row 1: MF probability-margin

MF row는 기존의 affine 계수를 유지한다.

```text
h_mf(z_i) = A_mf_i^T z_i + c_mf_i
```

다만 각 stage가 separated domain에 있고 MF 데이터가 유효한 경우에만 원래 계수를 활성화한다.

```text
mf_mask_i = (phi_bar_i > d_mf_mask) and mf_data_valid_i
```

마스크가 꺼진 stage에는 다음 값을 설정해 해당 stage의 MF row만 비활성화한다.

```text
A_mf_i = zeros(8)
c_mf_i = epsilon_i
```

이 경우 Row 1은 항상 만족되는 `epsilon_i >= 0` 형태가 된다.

구현상 보장되는 사항은 다음과 같다.

- 현재 측정 pose의 exact signed PSDF를 근거로 미래 MF row 전체를 끄지 않는다.
- 각 stage의 `phi_bar_i`, affine coefficient, 유효성만으로 독립적으로 mask를 결정한다.
- batched `forward_mf()`가 실패하면 stage별 호출로 fallback한다.
- stage별 fallback 중 하나가 실패해도 실패한 stage만 mask하고 나머지 stage의 MF row는 보존한다.
- covariance와 무관한 exact PSDF가 필요한 경우 pose-only wrapper 평가를 사용한다.

## 5. 고정된 두 행과 soft constraint

중간 및 terminal stage의 constraint 순서는 항상 다음과 같다.

```text
index 0 = Guard/Recovery
index 1 = MF probability margin
```

두 행 모두 lower-bound soft constraint이다.

```python
idxsh   = [0, 1]
idxsh_e = [0, 1]
lh      = [0, 0]
lh_e    = [0, 0]
```

`lsh`, `ush`, `lsh_e`, `ush_e`는 acados에서 slack variable의 하한을 의미하므로 모두 0으로 설정했다. Runtime에는 두 행의 개수나 의미를 바꾸지 않고 stage parameter와 MF mask만 변경한다.

기본 slack penalty는 다음과 같다.

| Row | Linear | Quadratic |
|---|---:|---:|
| Guard | `1e2` | `1e1` |
| MF | `1e1` | `1e0` |

Guard linear penalty는 MF보다 한 자리 크게 유지하고, quadratic penalty는 상대적으로 작게 유지한다.

Affine 식의 `A`, `c`와 slack penalty는 모든 stage에서 raw 단위를 그대로 사용한다. Row norm에 따른 coefficient 또는 penalty 정규화는 수행하지 않으며, intermediate/terminal stage 모두 동일한 고정 penalty를 사용한다.

## 6. Stage 0 처리

Stage 0에서는 다음 원칙을 적용한다.

- `con_h_expr_0`을 정의하지 않는다.
- `nh_0 = 0`, `ns_0 = 0`을 유지한다.
- 19차원 parameter layout은 유지하되, 안전 constraint에 사용되지 않는 trivial feasible Guard/MF 값을 전달한다.
- nominal `x_bar_0`을 측정 pose에 정확히 re-anchor하고 이를 assert한다.
- `exact_current_phi`는 측정 pose에서 pose-only PSDF 평가로 직접 계산한다.

## 7. Solver 준비 및 recovery 경로

각 solve 전에 다음 순서로 stage 데이터를 구성한다.

1. Shifted nominal trajectory를 만들고 stage 0을 측정 상태에 re-anchor한다.
2. Nominal node에서 signed PSDF, world gradient, MF affine 데이터를 계산한다.
3. Guard coefficient와 stage별 MF mask를 만든다.
4. Stage `1, ..., N`에 고정 순서의 parameter를 설정한다. Raw slack cost는 OCP 생성 시 한 번 고정되며 solve마다 재설정하지 않는다.
5. Fast SQP-RTI solver를 실행하고 필요하면 설정된 fallback 경로를 수행한다.

Backup solver 실패 후 safe-stop으로 전환되는 경우에도 다음 정합성을 유지하도록 보완했다.

- Main/backup constraint cache가 서로 섞이지 않도록 snapshot과 restore를 수행한다.
- Fast status와 backup status를 별도로 보존한다.
- Safe-stop predicted covariance를 covariance dynamics로 다시 전파한다.
- 실제 최적화로 얻지 않은 lower slack은 0으로 오인하지 않고 unavailable(`NaN`)로 기록한다.
- 대신 각 row의 affine residual로부터 필요한 lower slack을 별도로 계산한다.

## 8. 진단 및 로깅

Constraint ordering은 모든 진단에서 `0 = Guard/Recovery`, `1 = MF`로 고정된다.

추가한 주요 진단 항목은 다음과 같다.

- 측정 pose의 exact current signed PSDF
- Guard affine residual
- predicted node에서 새로 평가한 exact-PSDF Guard residual
- MF affine residual
- 새 MF 데이터로 재계산한 effective residual
- separated domain 밖의 분석을 위한 raw MF residual
- Guard lower slack 및 required lower slack
- MF lower slack 및 required lower slack
- stage별 MF validity/mask
- predicted nodes의 최소 exact PSDF
- 첫 shooting interval substep의 최소 exact PSDF
- Guard/MF raw row coefficient norm(진단 전용이며 cost에는 미사용)
- NLP residuals
- QP status 및 iteration 수
- fixed-row post-slack primal residual
- Fast/backup/fallback solver status와 solve mode

마스킹된 MF stage의 effective fresh residual은 `epsilon_i`로 기록하며, domain 밖의 raw MF 값과 구분한다.

## 9. acados QP residual 관련 제한

현재 설치된 acados에서 `nlp_solver_ext_qp_res = 1`을 활성화하면 stage 0의 slack 차원이 0이고 이후 stage의 slack 차원이 2인 이 OCP 구조에서 native crash가 발생한다.

따라서 해당 옵션은 활성화하지 않았으며, 안전하게 제공 가능한 다음 정보를 기록한다.

- NLP KKT residuals
- QP status
- QP iteration 수
- 직접 계산한 두 안전 행의 post-slack primal residual

Exact acados QP KKT residual vector는 현재 환경에서는 `None`으로 명시한다. 이를 지원하려면 acados upstream 수정 또는 별도 C-level 진단 구현이 필요하다.

## 10. 변경 파일

- `control/rmpcc_optimizer.py`
  - 19차원 stage parameter
  - 고정 Guard/MF 2행
  - stage별 MF mask 및 fallback
  - 계층 slack penalty
  - exact PSDF 및 solver diagnostics
  - backup/safe-stop 정합성 개선
- `test_rmpcc_mf.py`
  - parameter layout과 affine row 순서 검증
  - stage 0 constraint 부재 검증
  - soft lower slack과 penalty 순서 검증
  - Guard 부호와 상수항 검증
  - penetration 중 미래 MF stage 독립 활성화 검증
  - invalid MF stage 단독 마스킹 검증
  - batch 실패 후 stagewise fallback 검증
- `config/config.yaml`
  - `rmpcc` optimizer 설명을 고정 Guard/MF 2행 구조에 맞게 갱신

## 11. 검증 결과

다음 검사를 완료했다.

```text
python3 -m py_compile control/rmpcc_optimizer.py test_rmpcc_mf.py
python3 -m unittest discover -v -p 'test_*.py'
git diff --check
```

결과는 다음과 같다.

- 전체 단위 테스트 17개 통과
- Python 문법 검사 통과
- Git whitespace 검사 통과
- 실제 acados OCP 생성 및 solve 성공
- Penetration stage에서 MF row가 stage별로 마스킹됨을 확인
- Guard lower slack이 실제 Guard violation과 일치함을 확인
- Fixed-row post-slack violation이 수치 오차 수준임을 확인
