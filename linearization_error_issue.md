# 독립 검토 요청: CC-MF의 음수 확률 근사와 hard guard 통과 후 충돌

아래 데이터와 수식만으로 원인을 비판적으로 검토해 주세요. 원본 저장소나 다른 파일을 열 수 없다는 전제이며, 필요한 관측값을 본문에 발췌했습니다. 기존 해석에 동의하는 것보다, 데이터로 확인되는 사실과 추가 검증이 필요한 가설을 구분하는 것이 목적입니다.

질문은 다음과 같습니다. CC-MF, CC-SF, CC-ECS 모두 기하·동역학 선형화를 사용합니다. 그런데 관찰된 MF 실패에서는 hard guard를 만족한 QP 후보가 실제로 충돌했습니다. 왜 공통 선형화 취약점이 이번 MF의 입력 선택에서 드러났는지, 특히 CDF 합의 음수 affine probability가 어떤 역할을 했는지 분석해 주세요. SF/ECS가 동일 조건에서 항상 안전하다고 가정하지는 말아 주세요.

표기: PSDF `phi > 0`은 분리, `phi < 0`은 침투입니다. `nominal`은 현재 QP를 선형화한 기준, `candidate`는 QP의 해, `plant next`는 해당 입력을 비선형 dynamics에 적용한 다음 상태입니다. 제약 residual은 0 이상일 때 만족합니다. `stage 1`은 현재 제어 주기에서 한 스텝 뒤의 예측 노드이며, trial 번호·제어 시간과 구분합니다.

상태는 `z=[x,y,theta,s,xi_f,xi_l,xi_theta,xi_c]`, 입력은 `u=[v,omega,v_s]`입니다. `s`는 경로 진행도입니다. 공분산은 body frame에서 `P=L L^T`로 복원하며,

```text
L = [[exp(xi_f), 0,           0            ],
     [0,         exp(xi_l),   0            ],
     [0,         xi_c,        exp(xi_theta)]]
```

입니다. `xi_c`는 로그가 아닌 Cholesky 비대각 원소입니다. 아래 공분산은 물리적 오차 분포를 측정한 값이 아니라 제어기가 사용하는 예측 모델의 값입니다. localization noise를 끈 결정론적 실행입니다.

## 1. 실험 조건

- 환경: maze, rectangle `0.15 × 0.10 m`, differential drive, A*.
- 제어 간격: `0.1 s`, horizon `20`, trial 최대 navigation time `60 s`.
- 시작 perturbation: xy 표준편차 `0.02 m`, heading 표준편차 `0.05 rad`, ±3σ 제한. localization noise off.
- 파라미터: `epsilon=0.2`, `d_col=d_min=d_mf_mask=0.001 m`, `v ∈ [-0.7,0.7] m/s`, `omega ∈ [-1.2,1.2] rad/s`.
- 결과: Trial 1·2 실패, Trial 3·4·5 성공. 실패 seed는 각각 `0`, `1`이다.



최적화는 매 제어 주기 한 번의 SQP-RTI를 사용합니다. feature 대응, projection branch, body-frame gradient와 mask는 그 QP 안에서 고정합니다. 초기 sigma는 `(0.0002 m, 0.00025 m, 0.00025 rad)`이며, 초기 covariance는 매 주기 재설정됩니다. MF slack 비용은 stage별 `100*delta + 0.5*delta^2`이며, guard에는 slack이 없습니다. `delta`는 거리 단위가 아닌 확률합 residual의 무차원 slack입니다.

## 2. 세 방법의 제약 구조와 비교 범위

선형화를 사용한다는 사실은 공통 취약점을 뜻한다. 충돌하려면 실제 optimizer가 그 근사가 부정확해지는 방향과 크기의 후보를 선택해야 한다. nominal 궤적, 활성 feature, 공분산, 비용, slack, 입력 변화가 다르면 선택하는 후보와 오차도 달라진다.

MF의 hard guard는 다음 집합을 강제한다.

$$
\widehat{\mathcal S}(\bar x)
=\{x:\phi(\bar x)+\nabla\phi(\bar x)^\top(x-\bar x)\ge d_{\rm col}\}.
$$

이는 실제 안전 집합 $\mathcal S=\{x:\phi(x)\ge d_{\rm col}\}$와 같지 않을 수 있다. 이번 후보는 $\widehat{\mathcal S}\setminus\mathcal S$에 들어갔다. **hard는 해당 선형식을 slack 없이 만족시킨다는 뜻이지, 정확한 PSDF까지 만족한다는 뜻이 아니다.** MF slack이 hard guard를 직접 완화한 것도 아니다.

또 세 방법이 동일한 hard guard를 갖는다는 전제는 구현과 다르다.

| 방법 | 실제 obstacle 제약 구조 | 이번 비교에서 중요한 차이 |
|---|---|---|
| CC-MF | 전체 환경 PSDF의 pose-only affine hard guard + CDF 확률합의 affine soft 행 | 확률합 위반을 줄이는 입력 선택 압력, feature 간 음수 근사 상쇄, slack |
| CC-SF | 선택된 PSDF feature의 Gaussian margin을 포함한 단일 hard 행 | CDF 합을 직접 선형화하지 않으며 MF와 다른 QP 행을 만듦 |
| CC-ECS | 각 장애물 polygon에 대한 PSDF와 ellipsoidal margin을 포함한 hard 행들 | 전역 최소 feature 하나만 유지하는 구조와 다름. 현재 구현의 process noise 계수도 모두 0 |

같은 MF nominal·candidate에 SF의 첫 단계 행을 교차 평가하면 SF도 이 후보를 통과시킨다. 따라서 **“SF가 같은 위험 입력을 거부해서 성공했다”는 설명은 이 자료와 맞지 않는다.** 확인한 차이는 MF 행이 큰 입력 보정을 요구하는 정도다. SF가 실제 폐루프에서 어느 단계에서 다른 후보를 선택했는지까지 확정하려면 별도의 동일 조건 비교가 필요하다.

## 3. 원본 로그에 직접 기록된 실패 전환

아래는 원본 필드의 발췌·열 선택이다. `nominal_p_sum_stage1`과 `candidate_slack_stage1`은 각각 JSON 배열 필드 `mf_probability_sum[1]`, `mf_slack[1]`을 펼친 값이다. **둘은 같은 평가 지점의 값이 아니다.**

```text
trial,time_s,mode,status,v,omega,current_psdf_m,nominal_p_sum_stage1,candidate_slack_stage1
1,3.5,sqp_rti,0,0.700000000000,1.200000000000,0.001459699124,0.488381228568,0.076454915457
1,3.6,safe_stop_stage0,,0,0,-0.014844730496,0.762986103545,null
2,2.9,sqp_rti,0,-0.633693204065,-1.200000000000,0.001427367330,0.511060098053,0.017509422864
2,3.0,safe_stop_stage0,,0,0,-0.011011376977,1.431146244718,null
```

정지 주기의 nominal probability는 정지 시점의 참조 평가이며, 새로 채택한 최적해의 확률이 아니다. MF 확률합은 개별 feature 확률의 합이므로 1을 넘을 수 있다.

마지막 성공 주기의 stage 1 후보 공분산 로그는 다음과 같다.

```text
trial,control_time_s,stage,sigma_f_m,sigma_l_m,sigma_psi_rad
1,3.5,1,3.1512450245938054e-09,5.352059927545706e-09,1.004342929967221e-06
2,2.9,1,2.3912686957595887e-05,1.485951356717542e-04,1.430268543471083e-04
```

이 값은 실제 위치 오차를 측정한 값이 아니라 solver 후보의 공분산을 복원해 저장한 값이다. 첫 정지 주기에는 solver status가 비어 있고 입력이 0이며, MF slack은 `null`이다. 충돌을 일으킨 마지막 명령 자체는 solver status `0`이었다.

원본 CSV에는 **feature별 확률, 후보의 feature별 affine probability, nominal/candidate pose horizon, MF Jacobian·상수·mask가 저장되지 않았다.** 마지막 성공 주기의 `mf_probability_sum` 배열에는 음수가 없다. 첫 정지 주기의 stage 0에 나타나는 `-2.980232227667301e-9`는 이번에 분석한 `-0.1738` 등의 큰 음수 affine probability와 구분해야 한다. 충돌 geometry에서 MF 평가값이 0에 가깝게 나오는 것을 안전 증거로 해석해서도 안 된다.

## 4. 후보와 MF 행의 오프라인 복원

오프라인 복원에서는 새로운 QP를 풀지 않았다. stage 0 현재 상태가 고정된다는 점을 이용해, 저장된 이전 입력 $\bar u_0$와 실제 선택 입력 $u_0$로 다음 후보를 계산한다.

$$
x_1^{\rm QP}=f(x_0,\bar u_0)+f_u(x_0,\bar u_0)(u_0-\bar u_0).
$$

실행 당시 retained/unshifted warm start에 따라 이전 주기의 stage 1 후보를 다음 주기의 nominal pose로 사용한다. stage 0 공분산은 매 주기 초기값으로 재설정된다. 로그의 공분산에서 후보 log-Cholesky 좌표를 복원하고, covariance 접선식에 맞는 $|\kappa|$를 추정한 뒤 나머지 좌표와 nominal MF 확률합으로 교차 검증한다.

| 검증 항목 | Trial 1 | Trial 2 |
|---|---:|---:|
| 원본 nominal MF 합, stage 1 | 0.488381228568 | 0.511060098053 |
| 복원 nominal MF 합, CPU float32 geometry | 0.488387676540 | 0.511059147253 |
| 원본과의 차이 | +6.448e-6 | −9.508e-7 |
| log-Cholesky 4좌표 접선 복원 오차 | 약 1e-15 | 약 1e-15 |
| 추정된 절대 곡률 | 7.68807813 | 8.45347964 |

아래 주요 표는 CPU float32 geometry로 복원한 수치다. geometry 계산과 확률 합산의 정밀도 차이 및 QP 허용오차 때문에 마지막 자릿수와 residual에 작은 차이가 있다. 원래 QP 메모리 덤프를 bit-for-bit 재생한 것은 아니다. CPU float64 교차 계산에서도 같은 현상이 확인됐다.

## 5. 음수 확률 현상의 진단 로그

MF는 각 feature의 Gaussian CDF를 nominal에서 계산하고, 그 합을 한 행으로 선형화한다. 아래 $p_j$는 nominal에서 feature 대응·gradient·body frame을 고정한 국소 Gaussian 거리 모델의 위반 확률이며, 전체 비선형 충돌 사건의 정확한 확률이 아니다. $d_j(x)$도 이 국소 모델의 거리 근사를 뜻한다.

$$
p_j(z)=\Phi\!\left(\frac{d_{\min}-d_j(x)}{\sigma_j(\xi)}\right),\quad
\widehat p_j=p_j(\bar z)+\nabla p_j(\bar z)^\top(z-\bar z),\quad
\widehat h_{\rm MF}=\varepsilon-\sum_j\widehat p_j.
$$

CDF 자체는 `[0,1]` 범위지만 그 접선 $\widehat p_j$에는 같은 범위 보장이 없다. 실제 solver에는 feature별 행이 아니라 합산한 MF 행이 들어간다. 아래는 **그 합산 행을 오프라인에서 feature별로 분해한 진단값**이다.

`h_pose_gain`, `h_cov_gain`을 각 feature가 MF residual을 증가시키는 기여로 정의하면 `p_affine = p_nominal - h_pose_gain - h_cov_gain`이다. 아래 발췌는 원본 runtime 로그가 아닌 검증된 오프라인 재구성이다.

```text
OFFLINE_RECONSTRUCTED, trial=2, seed=1, control_time=2.9 s, horizon_stage=1
slot,p_nominal,h_pose_gain,h_cov_gain,p_affine
14,0.499040294923, 0.051355612441,0.004292860283, 0.443391822200
112,0.011229946717,-0.115956684506,0.301003579855,-0.173816948632
1,0.000789405610, 0.014499461955,0.038281290126,-0.051991346471
11,0.000000545861,0.000008665039,0.000060282956,-0.000068402134

valid_features                         = 136
features_with_p_affine_below_-1e-6     = 3
sum_negative_p_affine                  = -0.225876705286
sum_all_p_affine                       =  0.217515116914
sum_from_aggregate_MF_row              =  0.217513784042
decomposition_difference               =  0.000001332872
epsilon                               =  0.200000000000
logged_slack_stage1                    =  0.017509422864
MF_row_residual_after_logged_slack     = -0.000004361179
zero_clipped_sum_DIAGNOSTIC_ONLY       =  0.443391822200
```

slot 번호는 해당 nominal에서 사용된 feature 배열 인덱스이며 장애물 번호와 다르다. 가장 가까운 slot 14는 약 `0.4434`의 양의 위험 근사를 유지한다. 다른 feature들의 음수 값이 이를 상쇄해 합이 `0.2175` 수준까지 내려간다. slot 112에서는 pose 변화가 위험을 증가시키는데도, 큰 covariance 감소의 선형 기여가 이를 넘어 음수 근사를 만든다.

`zero_clipped_sum`은 음수 상쇄의 크기를 보기 위해 사후에 `max(0,p_affine)`를 합한 진단값이다. 실제 확률, 안전 보장, 다시 최적화한 해가 아니다. 출력만 clamp하는 것으로 원래 solver 문제가 해결되는 것도 아니다.

Trial 1에서는 이 메커니즘이 유의미하게 나타나지 않았다.

```text
OFFLINE_RECONSTRUCTED, trial=1, seed=0, control_time=3.5 s, horizon_stage=1
dominant_slot                          = 14
p_nominal                             =  0.488388033421
h_pose_gain                           =  0.055470305759
h_cov_gain                            =  0.156437853802
p_affine                              =  0.276479873861
features_with_p_affine_below_-1e-6     = 0
sum_negative_p_affine                  = -1.167364831815e-11
logged_slack_stage1                    =  0.076454915457
```

따라서 **음수 확률 상쇄를 Trial 1·2의 공통 단독 원인으로 쓰면 안 된다.**

## 6. MF와 SF가 다른 입력 보정을 요구한 수치적 근거

단일 feature에서는 원래 MF 확률 조건과 SF Gaussian margin이 같은 안전 경계를 표현한다. 하지만 서로 다른 residual을 별도로 선형화하면, 경계에서 떨어진 nominal에서는 다른 QP 행이 된다.

$a=(d-d_{\min})/\sigma$, $\ell=\log\sigma$, $\gamma=\Phi^{-1}(0.8)=0.841621$로 놓으면 단일 feature의 두 선형 조건은 다음과 같다. $\varphi$는 표준정규 PDF다.

$$
\text{MF: }\frac{\Delta d}{\sigma}-a\Delta\ell
\ge\frac{\Phi(-a)-\varepsilon}{\varphi(a)},
\qquad
\text{SF: }\frac{\Delta d}{\sigma}-\gamma\Delta\ell\ge\gamma-a.
$$

여기서 실제 4차원 log-Cholesky 상태 $\xi$와 연결되는 $\Delta\ell$은 $\nabla_\xi\log\sigma(\bar\xi)^\top\Delta\xi$라는 1차 증분이다. 후보와 nominal의 `log σ` 유한 차이를 그대로 대입하는 식이 아니다. 아래 입력 조건 표는 실제 8차원 상태의 Jacobian과 첫 단계 동역학으로 계산했다.

Trial 1의 지배 feature는 `a≈0.02911`, `p≈0.48839`다. MF의 공분산 방향 기울기가 작아, 큰 확률 위반을 접선으로 해소하는 데 매우 큰 log-scale 감소가 필요하다. 단순히 “MF가 공분산에 더 민감하다”라고 표현하면 이 현상을 잘못 설명한다.

첫 단계 동역학을 소거하고 회전 입력을 실제 후보값으로 고정해 비교한 결과는 다음과 같다.

| 항목 | Trial 1 | Trial 2 |
|---|---:|---:|
| nominal 첫 속도 [m/s] | −0.055847 | +0.182370 |
| 실제 후보 첫 속도 [m/s] | +0.700000 | −0.633693 |
| 고정한 omega [rad/s] | +1.2 | −1.2 |
| MF 첫 행의 무슬랙 조건 | `v ≥ +1.174362` | `v ≤ −0.702038` |
| SF 첫 행의 조건 | `v ≥ −0.016445` | `v ≤ +0.027431` |
| v·omega 입력 범위 전체에서 MF residual 최대값 | −0.076480 | −0.000522 |
| 그 최대점을 만드는 `[v,omega]` | `[+0.7,+1.2]` | `[−0.7,−1.2]` |

Trial 1의 실제 입력은 MF 첫 행 위반을 가장 줄이는 입력과 같다. 이는 MF 행이 큰 입력 보정을 요구했다는 근거다. 하지만 모든 horizon 비용·제약의 KKT 조건을 재현한 결과는 아니므로, 해당 행 하나가 최종 입력을 단독 결정했다고 단정하지 않는다.

동일한 위험 후보를 SF 행에 대입하면 residual은 각각 **+40.658 mm, +39.769 mm**로 통과한다. 같은 MF 후보에 SF 행을 교차 평가해 얻은 표준편차의 접선 값은 **−47.547 mm, −45.224 mm**로, 근사의 유효 범위를 벗어났다. 실제 표준편차가 음수라는 뜻은 아니다. 따라서 SF도 공통 근사 취약점이 있다. SF의 성공을 동일 후보에 대한 안전 판별 능력으로 설명해서는 안 된다.

## 7. hard guard가 충돌을 막지 못한 경로

| 복원한 stage 1 항목 | Trial 1 | Trial 2 |
|---|---:|---:|
| nominal PSDF [mm] | +1.110945 | +1.031324 |
| 후보의 affine guard residual, `phi_hat−d_col` [mm] | **+0.641078** | **+1.707547** |
| 후보 자세의 정확한 PSDF [mm] | **−18.171028** | **−14.886104** |
| 실제 plant 다음 자세의 PSDF, 원본 로그 [mm] | −14.844730 | −11.011377 |
| nominal에서 가장 가까운 장애물 index | 1 | 1 |
| 후보에서 가장 가까운 장애물 index | 0, 아래쪽 벽 | 14, 왼쪽 경계 |
| QP 후보와 plant 다음 위치 차이 [mm] | 4.201 | 4.965 |

전체 환경 PSDF의 nominal 최소 branch로 만든 하나의 guard는 큰 이동 후 바뀐 최소 장애물을 반영하지 못했다. 선형화 오차에는 같은 branch 내부의 곡률 오차뿐 아니라 **활성 장애물·feature 변경을 고정한 채 진행한 오차**도 포함된다.

공분산에서도 큰 입력 부호 반전을 접선으로 외삽했다. 후보의 log-Cholesky 좌표에서 복원한 공분산은 양의 정부호를 유지하지만, 동일 입력을 원래 covariance 전파식에 넣은 값과 크게 다르다. PSD 위반이 아니라 비선형 전파식과 QP 접선 사이의 불일치다. 예를 들어 forward log-factor 전파에는 다음 형태가 있다.

$$
\xi_{f,1}=\tfrac12\log\!\left(\sigma_{f,0}^2+(\Delta t\omega)^2\sigma_{l,0}^2
+\Delta t\alpha_f v^2\right).
$$

이 식의 `alpha_f=0.0012`다. nominal 입력 `[v,omega]`는 Trial 1에서 `[-0.055847247716,0.087972594216]`, Trial 2에서 `[0.182370263770,0.017329559821]`였다.

nominal 속도의 부호를 따라가는 접선을 반대 부호의 큰 입력까지 연장하면, 모델의 원래 `v²` noise가 커져도 QP에서는 log-factor가 줄어들 수 있다. Trial 1·2의 후보 lateral sigma는 각각 약 `0.00000535 mm`, `0.14860 mm`였지만, 동일 입력을 원래 비선형 covariance 식으로 다시 계산하면 약 `47.67 mm`, `45.24 mm`였다. 후자는 측정한 실제 공분산이 아니라 설정된 모델의 비선형 재평가값이다. 이 전파식은 MF와 SF가 공유하는 취약점이며, MF만의 전파식 오류로 분류하면 안 된다.

Trial 1은 추가로 3.3·3.4초의 stage 1 nominal PSDF가 `0.953/0.626 mm`로 mask threshold보다 작아 MF 행이 꺼졌다가 3.5초에 다시 켜졌다. Trial 2의 마지막 7개 분석 주기에는 계속 켜져 있었다. mask 전환 역시 두 실패의 공통 원인은 아니다.

## 8. ECS 비교와 아직 확인하지 않은 인과관계

현재 CC-ECS 구현은 obstacle polygon마다 별도의 hard 행을 둔다. MF의 전역 최소 branch 한 개를 사용하는 guard와 달리, 현재 최소 거리가 아닌 벽의 제약도 함께 유지한다. 따라서 “동일한 hard guard인데 결과만 다르다”는 비교가 아니다.

이를 확인하기 위해 **동일한 MF nominal·candidate 자세**에 ECS 방식의 장애물별 기하 평면을 대입했다. 새로운 QP를 풀지 않고 `phi_hat_j−d_min`만 비교한 결과다.

| 같은 MF 후보에 대한 기하 residual [mm] | Trial 1 | Trial 2 |
|---|---:|---:|
| MF의 전역 최소 branch guard | +0.641 | +1.708 |
| 충돌한 장애물의 개별 affine plane | **−19.786** | **−16.503** |
| 개별 plane의 장애물 | 아래벽 index 0 | 왼쪽 경계 index 14 |

즉, 이번 두 후보의 충돌은 **다른 장애물의 평면도 유지하는 기하 제약만으로 이미 드러난다.** MF의 guard는 그 평면들을 유지하지 않았다. 이 차이는 “선형화는 공통인데 왜 ECS와 같은 결과가 아닌가”에 대한 구체적인 구조적 근거다. 다만 이 계산에는 ECS covariance margin 및 그 SQP 접선을 넣지 않았으므로 **ECS 전체 QP가 같은 후보를 반드시 거부한다는 증명은 아니다.** 장애물별 평면도 각 polygon 내부에서는 국소 근사다.

MF가 여러 feature를 합한다고 해서 이 개별 기하 guard와 같아지는 것도 아니다. Trial 1에서 나중에 충돌한 아래벽의 feature(slot 1)는 nominal 확률이 약 `1.73e-14`이고, 위치 방향 확률 기울기도 약 `4.52e-11 /m`로 매우 작았다. 큰 이동으로 그 벽과 겹쳐도 고정한 확률 접선은 그 위험 증가를 충분히 반영하지 못했다. Trial 2에서는 앞서 기록한 음수 확률 상쇄가 추가로 나타났다.

또 현재 ECS의 `q_f0`, `q_l0`, `q_psi0`, `alpha_f`, `alpha_v`, `alpha_kappa`, `beta_v`, `beta_kappa`, `beta_omega`는 모두 0이다. 초기 공분산의 동역학 전파는 남지만, 이번 MF/SF의 속도 의존 process-noise 항을 동일하게 갖지 않는다. 현재 구현을 과거 비교 실행의 정확한 설정으로 소급해 단정하지 않는다.

과거 SF/ECS 비교의 상세 실행 설정과 원시 로그는 확보하지 못했다. 따라서 이번 MF 5회 데이터와 과거 방법별 성공률을 결합해 동일 조건 비교라고 가정할 수 없다.

현재 증거로 확인한 것은 다음 세 수준이다.

1. **직접 기록:** 큰 입력 명령, 후보 공분산, slack, 다음 실제 PSDF 음수, 이후 safe stop.
2. **검증된 오프라인 복원:** 후보 자체의 충돌, guard 통과, MF 첫 행의 입력 범위 내 무슬랙 불가능, Trial 2의 음수 feature 확률 상쇄, 같은 후보에 대한 SF 첫 행 통과.
3. **아직 미확정:** SF/ECS 전체 optimizer가 같은 초기 조건에서 언제 다른 경로를 택했는지, 각 수정의 단독 효과, 수정 후 성공률.

Trial 2의 affine 행을 독립적으로 검산할 수 있는 수치도 아래에 포함한다. `w=[x,y,theta,xi_f,xi_l,xi_theta,xi_c]`로 놓으면 `h_MF=A_without_s @ w+c`다. 진행도 `s`의 MF 계수는 0이므로 생략했다. guard는 `nominal_phi + gradient @ (candidate_pose-nominal_pose) - d_col`로 계산한다.

```json
{
  "provenance": "offline_reconstructed_trial2_stage1",
  "units": "pose=[m,m,rad]; xi_c is not a log coordinate",
  "nominal_pose": [
    0.11052981300460255,
    0.5676388594356221,
    -0.8650045695542883
  ],
  "candidate_pose": [
    0.06824210126976854,
    0.6148519667153137,
    -0.9850045695542397
  ],
  "nominal_xi": [
    -6.210739383977745,
    -4.341172905738754,
    -4.325159339636714,
    8.75349486949253e-08
  ],
  "candidate_xi": [
    -10.641101404824038,
    -8.81428516059221,
    -8.852478339363559,
    8.739056637491198e-08
  ],
  "nominal_phi_m": 0.0010313242673873901,
  "guard_gradient_world": [
    -0.7610830068588257,
    -0.6486545205116272,
    -0.0009729978628456593
  ],
  "d_col_m": 0.001,
  "MF_A_without_s": [
    -20.369773864746094,
    -19.55767250061035,
    -0.09911998361349106,
    -0.0014127115719020367,
    -0.07534893602132797,
    -7.405199721688405e-05,
    0.15403714776039124
  ],
  "MF_c": 12.620166778564453,
  "logged_slack": 0.017509422863836558
}
```

## 9. 검토 요청

1. 위 데이터만으로 확정할 수 있는 실패 메커니즘은 어디까지인가요? 공통된 선형화 취약점과 MF의 후보 선택을 악화시키는 요인을 구분하고, 아직 증명되지 않은 인과관계를 지적해 주세요.
2. 단일 feature에서는 확률 수준에서 동치인 MF와 SF가 왜 다른 QP 보정량을 요구하나요? CDF residual의 재매개화, 작은 공분산 방향 기울기, log-Cholesky dynamics의 입력 부호 반전 외삽을 수식으로 검토해 주세요. Trial 1과 Trial 2를 같은 원인으로 단정하지 말아 주세요.
3. Trial 2의 음수 affine probability 상쇄를 막는 방법을 제안해 주세요. 단순 출력 clipping, 제약 내부의 max(0,p_affine), feature별 위험예산, 보수적 상한, trust region의 차이를 설명하고 안전성·계산량·보수성 관점에서 비교해 주세요. MF는 현재 stage당 합산 행 하나를 사용하는 구조여서 feature 수에 따라 QP 크기가 증가하는 변경은 tradeoff가 있습니다.
4. SF도 동일한 위험 후보를 통과시키는 결과와, ECS의 장애물별 기하 항은 해당 후보의 위반을 드러내는 결과를 함께 설명해 주세요. 어떤 추가 데이터나 최소 ablation이 있어야 실제 방법 간 실패 원인을 구분할 수 있나요?
5. 구현 수정의 우선순위를 제안해 주세요. 입력·log-factor trust region, 재선형화, 복수 장애물 guard, 비선형 첫 스텝 검증, slack 비용 변경을 구분하고 각각 막을 수 있는 실패와 남는 한계를 명시해 주세요. 수치 튜닝을 제안한다면 검증된 값이 아닌 시작 후보임을 밝혀 주세요.

새로운 주행 실험으로 수정 효과를 검증한 결과는 없습니다. `zero_clipped_sum`은 진단량이며 실제 충돌확률 또는 안전 보장이 아닙니다. 전체 horizon의 KKT 조건, 비용·제약별 최적 입력 기여는 복원하지 않았습니다.
