# PSDF-MPCC-MF 코드 수정 문제

**User:** Anonymous  
**Created:** 8/11/2026 20:44:01  
**Updated:** 8/19/2026 20:04:24  
**Exported:** 8/19/2026 23:12:57  
**Link:** [https://chatgpt.com/g/g-p-6a7037b645148191a091878da988c7b9/c/6a7b0b01-43c8-83e8-8705-ebcbae131a0a](https://chatgpt.com/g/g-p-6a7037b645148191a091878da988c7b9/c/6a7b0b01-43c8-83e8-8705-ebcbae131a0a)  

## Response:
2026. 8. 19. 오후 7:50:24

> **검토 범위 정리**
>
> 현재 제공된 코드에서 하드 G 이후 소프트 M이 정지하는 경로를 추적하려고 합니다. 먼저 소프트 가중치, 정지 조건, 잔차·상태 진단 변수처럼 직접 조정 가능한 항목을 중심으로 확인하겠습니다.
>
> **검토 범위 정리 완료**
>
> 
>
> **조정 변수 우선순위**
>
> 소프트 M 정지가 패널티 과중 때문인지 확인하려고, 먼저 선형·이차 완화 가중치를 크게 낮춰 실험하겠습니다. 이어 허용확률, 반복 한도, 종료 허용오차를 바꾸며 잔차와 완화량 변화를 함께 기록하겠습니다.
>
> **조정 변수 우선순위를 정리함**
>
> 
>
> **진단 지점 좁히기**
>
> 소프트 M 정지가 완화비용보다 공분산 축소와 정지 보상을 선호하는지 확인하겠습니다. 특히 `mf_slack_linear`, `mf_slack_quadratic`, `chance_epsilon`, 활성 구간, 초기 불확실성을 단계적으로 기록하겠습니다. PSDF 하드 제약은 기준선으로 고정하겠습니다.
>
> **진단 지점을 좁혔습니다**
>
> 
>
> **공분산 성장 확인**
>
> 초기 불확실성보다 이동 중 공분산 증가항이 훨씬 클 수 있습니다. 저는 속도·곡률·각속도 관련 잡음 계수와 전파 간격을 낮춰, 시간에 따라 확률 잔차가 악화되는지 먼저 확인하겠습니다.
>
> **공분산 성장을 확인함**
>
> 
>
> **RMPCC 튜닝·디버그 변수와 MF 제약 정지 원인 점검**
>
> 
>
> 11m 55s 동안 처리함

지금 실험 결과로 원인 범위가 상당히 좁혀졌습니다.

$$
\text{hard Row G + Row M off}
\rightarrow \text{정상 주행}
$$

$$
\text{hard Row G + soft Row M on}
\rightarrow \text{정지}
$$

이 경우에는 path tracking이나 PSDF guard보다, **soft MF row가 motion을 억제하는 비용으로 작동하는 것**이 핵심입니다. MF는 physical velocity $v,\omega$가 커지면 covariance가 증가하고 feature violation probability가 커지므로, optimizer 입장에서는 **움직이지 않는 것이 MF penalty를 줄이는 가장 쉬운 방법**일 수 있습니다.

다만 파라미터를 튜닝하기 전에 반드시 확인해야 할 코드 문제가 하나 있습니다.

---

# 1. 먼저 확인할 것: hard G로 바꾸면서 slack index도 완전히 수정했는가

첨부한 원래 코드는 두 row가 모두 soft라는 전제로 작성되어 있습니다.

```python
idxsh = [0, 1]

slack slot 0 = Row G
slack slot 1 = Row M
```

이에 맞춰 slack cost도

```python
[guard_slack_linear, mf_slack_linear]
```

순서이고, diagnostic도 `sl[0]`과 `sl[1]` 두 개가 존재한다고 가정합니다. `rmpcc_optimizer.py` `rmpcc_optimizer.py`

Row G를 hard로 바꾸면 구성은 다음과 같아야 합니다.

```python
soft_indices = np.array([1], dtype=np.int64)

self.ocp.dims.nsh = 1
self.ocp.dims.ns = 1
self.ocp.dims.nsh_e = 1
self.ocp.dims.ns_e = 1

self.ocp.constraints.idxsh = soft_indices
self.ocp.constraints.idxsh_e = soft_indices.copy()

self.ocp.constraints.lsh = np.zeros(1)
self.ocp.constraints.ush = np.zeros(1)
self.ocp.constraints.lsh_e = np.zeros(1)
self.ocp.constraints.ush_e = np.zeros(1)

self.ocp.cost.zl = np.array([param.mf_slack_linear])
self.ocp.cost.Zl = np.array([param.mf_slack_quadratic])
self.ocp.cost.zu = np.array([param.mf_slack_linear])
self.ocp.cost.Zu = np.array([param.mf_slack_quadratic])

self.ocp.cost.zl_e = np.array([param.mf_slack_linear])
self.ocp.cost.Zl_e = np.array([param.mf_slack_quadratic])
self.ocp.cost.zu_e = np.array([param.mf_slack_linear])
self.ocp.cost.Zu_e = np.array([param.mf_slack_quadratic])
```

이제 유일한 slack은

```python
solver.get(i, "sl")[0]
```

이고, 이것이 **MF slack**입니다.

따라서 다음 함수도 한 개의 slack 기준으로 바꿔야 합니다.

- `_set_stage_slack_penalties()`
- `_read_lower_slack_trajectory()`
- `_build_constraint_diagnostics()`

이 부분이 완전히 수정되지 않았다면 다음 중 하나가 발생할 수 있습니다.

- Row M에 의도한 $10^3$이 아니라 기존 guard weight $10^5$가 적용됨
- MF slack을 잘못된 index에서 읽음
- diagnostic에는 MF slack이 계속 `NaN`으로 출력됨
- row normalization cost가 잘못된 slack slot에 들어감

즉, **튜닝 전에 실제 Row M weight가 무엇인지부터 확인해야 합니다.**

PSDF-MPC가 기본적으로 obstacle constraint를 hard로 사용하고 정상 동작하므로, 현재 비교에서는 Row G와 `d_col`은 그대로 유지하는 것이 맞습니다. `psdf_optimizer.py`

---

# 2. 현재 MF weight는 progress cost보다 지나치게 큼

현재 주요 값은 다음과 같습니다.

```python
q_s = 1.0
v_s_target = 0.8

mf_slack_linear = 1e3
mf_slack_quadratic = 1e0

horizon = 20
```

`rmpcc_optimizer.py` `rmpcc_optimizer.py`

현재 progress cost는

$$
r_{v_s}v_s^2-q_sv_s,
\qquad
r_{v_s}
=
\frac{q_s}{2v_{s,\mathrm{target}}}
=
0.625
$$

입니다.

현재 effective $v_s$ upper bound가 $0.7$이면, $v_s=0.7$일 때 한 stage의 최대 progress 이득은 대략

$$
0.625(0.7)^2-1.0(0.7)
=
-0.394
$$

입니다.

반면 normalized MF violation이 단지

$$
\xi_i^M=0.01
$$

이어도 현재 linear slack cost는

$$
10^3\times0.01=10
$$

입니다.

즉, 한 stage에서만 MF residual을 $0.01$ 위반해도, **한 stage의 progress reward보다 약 25배 큰 비용**이 발생합니다. 이것이 여러 stage에서 누적되면 optimizer가 진행하지 않는 것은 자연스럽습니다.

따라서 지금의 $10^3$은 “soft MF”라기보다 실질적으로 **거의 hard MF**에 가깝습니다.

---

# 3. 튜닝할 파라미터

## 3.1 가장 먼저 튜닝할 파라미터

| 파라미터 | 증가시키면 | 감소시키면 | 권장 초기 sweep |
|---|---|---|---|
| `mf_slack_linear` | MF 위반을 피하기 위해 더 많이 감속 또는 정지 | MF violation을 허용하고 진행 | `1, 3, 10, 30` |
| `mf_slack_quadratic` | 큰 MF violation을 강하게 억제 | 큰 violation도 비교적 쉽게 허용 | `0.1, 1, 10` |
| MF active horizon | 먼 미래까지 covariance와 probability를 고려하여 더 보수적 | 가까운 미래만 고려하여 진행성 증가 | stage `1-3`, `1-5`, `1-10`, `1-20` |
| `chance_epsilon` | 증가하면 MF 조건 완화 | 감소하면 MF 조건 강화 | `0.2, 0.3, 0.5` |

### 추천 시작값

```python
mf_slack_linear = 1.0
mf_slack_quadratic = 0.1

mf_active_start_step = 1
mf_active_end_step = 5
```

그 상태에서 로봇이 진행하면 다음 순서로 올리는 것이 좋습니다.

```text
mf_slack_linear:
1 → 3 → 10 → 30

mf_active_end_step:
5 → 10 → 20
```

**가장 큰 weight부터 낮추는 방식보다, 가장 작은 weight에서 시작해 안전 행동이 나타날 때까지 올리는 방식이 훨씬 명확합니다.**

---

## 3.2 MF active horizon을 별도로 추가하는 것이 좋음

현재 코드는

```python
mf_mask = (phi > d_mf_mask) & mf_valid
```

이므로 separated 상태라면 stage $1,\ldots,N$ 전체에서 MF가 활성화됩니다. `rmpcc_optimizer.py`

다음 파라미터를 추가하는 것이 좋습니다.

```python
self.mf_active_start_step = 1
self.mf_active_end_step = 5
self.mf_active_terminal = False
```

그리고 mask를 다음처럼 구성합니다.

```python
stage_idx = np.arange(num_stages)

mf_stage_mask = (
    (stage_idx >= self.param.mf_active_start_step)
    & (stage_idx <= self.param.mf_active_end_step)
)

mf_mask = (
    (phi > d_mf_mask)
    & mf_valid
    & mf_stage_mask
)
```

working PSDF-MPCC-SF 코드도 uncertainty margin을 기본적으로 stage 0부터 10까지만 활성화하고 있습니다. 따라서 MF도 처음부터 20개 stage 전체에 넣기보다, 우선 5개 또는 10개 stage에서 검증하는 것이 비교상 자연스럽습니다. `rmpcc_pv_optimizer.py`

특히 terminal MF row는 먼 horizon의 큰 covariance 때문에 controller 전체를 멈추게 할 수 있으므로, 초기 튜닝에서는 끄는 것이 좋습니다.

---

## 3.3 Chance-constraint 파라미터

### `chance_epsilon`

MF residual은

$$
h_i^{\mathrm{MF}}
=
\varepsilon_i-\sum_\ell p_{\ell,i}
$$

입니다.

현재

```python
chance_epsilon = 0.20
```

이므로 여러 feature probability의 합이 $0.2$보다 크면 violation입니다.

튜닝 시 다음 순서로 확인하십시오.

```python
chance_epsilon = 0.20
chance_epsilon = 0.30
chance_epsilon = 0.50
```

- $0.2$에서 정지하고 $0.5$에서 진행하면 MF probability sum 자체가 지나치게 큽니다.
- 이 경우 slack weight만의 문제가 아니라 feature 중복, covariance 또는 여러 feature의 probability accumulation을 봐야 합니다.
- $0.5$는 최종 안전 설정이라기보다 원인 확인용입니다.

코드는 scalar뿐 아니라 stage별 $\varepsilon_i$ 배열도 받을 수 있으므로, 필요하면 가까운 stage에는 작게, 먼 stage에는 크게 줄 수도 있습니다.

---

### `d_min`

현재

```python
d_min = 0.001
d_col = 0.001
```

입니다.

Row G가 hard라면 두 값을 반드시 같게 유지할 필요는 없습니다.

- `d_col`은 deterministic collision guard threshold
- `d_min`은 MF Gaussian feature-distance event의 threshold

예를 들어 debugging에서는

```python
d_col = 0.001
d_min = 0.0
```

으로 두어 MF가 penetration probability를 규제하고, hard G가 $1\text{ mm}$ geometric clearance를 담당하게 할 수 있습니다.

다만 현재 $d_{\min}=1\text{ mm}$ 자체가 아주 큰 값은 아니므로, 첫 번째 튜닝 대상은 아닙니다.

---

### `d_mf_mask`

현재 mask는

$$
\bar\phi_i>d_{\mathrm{mf,mask}}
$$

일 때 MF를 활성화합니다.

따라서 `d_mf_mask`를 증가시키면 **벽과 가까운 stage에서 MF가 꺼집니다.**

```python
d_mf_mask = 0.0
d_mf_mask = d_col
d_mf_mask = 0.005
```

정도를 확인할 수 있습니다.

다만 이 파라미터는 MF의 유효 domain을 제한하는 값이지, 전체 conservatism을 조절하는 주된 파라미터로 사용하는 것은 권장하지 않습니다.

또한 먼 horizon의 MF를 끄는 용도로는 쓸 수 없습니다. 먼 stage가 벽에서 떨어져 있으면 오히려 계속 활성화되기 때문입니다. 먼 horizon 문제는 `mf_active_end_step`으로 해결해야 합니다.

---

# 4. Covariance 관련 튜닝 파라미터

현재 MF는 physical input으로 인해 covariance가 증가하도록 되어 있습니다.

```python
sigma_f0
sigma_l0
sigma_psi0

q_f0
q_l0
q_psi0

alpha_f
alpha_v
alpha_kappa

beta_v
beta_kappa
beta_omega
```

`rmpcc_optimizer.py`

이 값들이 커질수록

$$
v,\omega
\rightarrow
\Sigma_i\uparrow
\rightarrow
p_{\ell,i}\uparrow
\rightarrow
\text{MF penalty}\uparrow
$$

가 되므로 optimizer는 $v,\omega$를 0으로 만들려 합니다.

## 가장 유용한 covariance ablation

먼저 다음처럼 control-dependent covariance 증가를 완전히 제거하십시오.

```python
q_f0 = 0.0
q_l0 = 0.0
q_psi0 = 0.0

alpha_f = 0.0
alpha_v = 0.0
alpha_kappa = 0.0

beta_v = 0.0
beta_kappa = 0.0
beta_omega = 0.0
```

초기 covariance인

```python
sigma_f0
sigma_l0
sigma_psi0
```

만 유지합니다.

### 결과 해석

- 이 상태에서는 진행한다  
  → 정지 원인은 **control-dependent covariance growth**
- 이 상태에서도 멈춘다  
  → 초기 covariance, feature probability sum 또는 MF slack weight 문제
- fixed covariance에서는 진행하지만 원래 값에서는 정지한다  
  → 계수를 다음처럼 scale sweep

```text
0.01 × original
0.03 × original
0.1  × original
0.3  × original
1.0  × original
```

특히 MF에서는 여러 feature probability가 합산되므로, SF에서 사용하던 covariance coefficient를 그대로 사용해도 훨씬 강한 감속이 나타날 수 있습니다.

---

# 5. MPCC 쪽에서 튜닝할 수 있는 파라미터

MF 자체를 먼저 정상화한 뒤에만 조정하는 것이 좋습니다.

| 파라미터 | 역할 | 조정 방향 |
|---|---|---|
| `q_s` | progress objective의 전체 강도 | 증가하면 MF penalty 대비 진행 의지가 커짐 |
| `v_s_target` | unconstrained progress target | 감소하면 느리지만 지속적인 진행을 유도 |
| `q_s_ref` | progress state와 path anchor의 일치 | 너무 크면 physical stop 시 $v_s$도 함께 멈춤 |
| `mat_Re[0,0]` | physical velocity $v$ penalty | 감소하면 움직이기 쉬움 |
| `mat_Re[1,1]` | angular velocity penalty | 증가하면 회전 억제 |
| `mat_Qe[0,0]` | contour error | 증가하면 lateral deviation 억제 |
| `mat_Qe[1,1]` | lag error | 증가하면 physical motion과 progress 불일치 억제 |

현재 `q_s`가 증가하면 코드가 $r_{v_s}=q_s/(2v_{s,\mathrm{target}})$도 같이 증가시키므로, unconstrained optimum $v_s=v_{s,\mathrm{target}}$는 유지되고 **progress objective 전체의 상대적 중요도만 커집니다.**

따라서 MF weight를 정상 범위로 줄인 후에도 지나치게 소극적이면

```python
q_s = 1.0
q_s = 2.0
q_s = 5.0
```

순으로 조정할 수 있습니다.

하지만 `q_s`를 크게 만들어 지나치게 큰 MF penalty를 억지로 이기게 하는 방식보다, 먼저 `mf_slack_linear`를 낮추는 것이 맞습니다.

---

# 6. 이미 코드에서 확인 가능한 debug 변수

현재 diagnostic에는 다음 값들이 들어 있습니다. `rmpcc_optimizer.py` `rmpcc_optimizer.py`

## 6.1 가장 먼저 볼 변수

| 변수 | 의미 | 정지 상황 해석 |
|---|---|---|
| `solve_mode` | 정상 QP인지 fallback인지 | `safe_stop`이면 MF cost가 아니라 solver failure |
| `mf_mask[i]` | 해당 stage MF 활성 여부 | 대부분 1이면 전체 horizon MF 활성 |
| `mf_affine_residual[i]` | 실제 QP에 들어간 MF row residual | 음수이면 slack 필요 |
| `mf_fresh_raw_residual[i]` | solved trajectory에서 재계산한 raw MF residual | 실제 MF model 위반 정도 |
| `mf_fresh_effective_residual[i]` | mask까지 반영한 fresh residual | 현재 mask 정책하의 위반 |
| `mf_lower_slack[i]` | solver가 선택한 MF violation | 양수이면 MF를 포기하고 진행 또는 정지 |
| `mf_lower_slack_normalized[i]` | row scale을 제거한 비교 가능한 slack | slack cost 계산에 사용 |
| `mf_probability_sum[i]` | $\varepsilon-h_i^{MF}$ | $\varepsilon$보다 크면 MF 위반 |
| `row_normalization_scales[:,1]` | MF row scale | 지나치게 큰 coefficient 확인 |
| `minimum_exact_psdf_predicted_nodes` | hard G safety 확인 | G가 정상 작동하는지 확인 |

단, hard G + soft M 한 개로 변경한 뒤에는 기존 diagnostic의 `mf_lower_slack = lower_slacks[:,1]`을 그대로 사용할 수 없습니다. 이제 유일한 MF slack은 `sl[0]`입니다.

---

## 6.2 반드시 함께 출력할 control 변수

각 cycle에서 최소한 다음을 출력하십시오.

```python
u0 = solver.get(0, "u")

v0 = u0[0]
omega0 = u0[1]
v_s0 = u0[2]
```

그리고 horizon 전체도 확인합니다.

```python
u_sol = np.stack([
    solver.get(i, "u")
    for i in range(self.N)
])

v_traj = u_sol[:, 0]
omega_traj = u_sol[:, 1]
v_s_traj = u_sol[:, 2]
```

### 해석

- $v_0\approx0$, $v_{s,0}\approx0$  
  → controller가 완전히 정지를 선택
- $v_0\approx0$, $v_{s,0}>0$  
  → physical motion과 progress가 분리됨
- 가까운 stage는 이동하지만 먼 stage의 $v_i$가 0  
  → far-horizon MF 또는 covariance accumulation 문제
- solver status가 실패하고 `safe_stop`  
  → tuning 이전에 infeasibility 확인

---

# 7. 추가해야 하는 가장 중요한 debug 변수

현재 optimizer는 aggregate $A_i^{MF},c_i^{MF}$만 저장합니다. 정지 원인을 정확히 보려면 아래를 추가하는 것이 좋습니다.

## 7.1 Nominal MF residual

QP solve 전 shifted nominal에서

$$
h_{i,\mathrm{nom}}^{MF}
=
(A_{i,\mathrm{raw}}^{MF})^\top\bar z_i
+
c_{i,\mathrm{raw}}^{MF}
$$

를 계산하십시오.

```python
h_mf_nom = (
    np.einsum(
        "ij,ij->i",
        self._last_mf_A_raw,
        z_bar,
    )
    + self._last_mf_c_raw
)

p_sum_nom = epsilon - h_mf_nom
```

반드시 `z_bar`도 저장하는 것이 좋습니다.

```python
self._last_z_bar = z_bar.copy()
self._last_u_bar = u_bar.copy()
```

### 해석

- solve 전부터 `h_mf_nom < 0`  
  → nominal trajectory 자체가 MF 조건을 위반
- 초기 stage부터 매우 음수  
  → feature sum, epsilon 또는 초기 covariance 문제
- 후반 stage만 음수  
  → covariance accumulation 또는 active horizon 문제

---

## 7.2 MF gradient를 pose와 covariance로 분리

State ordering이

$$
[x,y,\theta,s,P_f,P_l,P_\psi,P_{l\psi}]
$$

이므로 다음을 출력하십시오.

```python
A_pose = A_mf_raw[:, 0:3]
A_progress = A_mf_raw[:, 3]
A_cov = A_mf_raw[:, 4:8]

A_pose_norm = np.linalg.norm(A_pose, axis=1)
A_cov_norm = np.linalg.norm(A_cov, axis=1)
```

### 해석

#### 경우 1

```text
A_pose_norm 작음
A_cov_norm 큼
```

MF를 개선하는 주된 방법이 pose 이동이 아니라 covariance 감소입니다. 따라서 optimizer는 $v,\omega$를 줄이고 정지합니다.

#### 경우 2

```text
A_pose_norm 큼
A_cov_norm 작음
```

MF가 주로 geometric steering을 유도합니다.

#### 경우 3

```text
h_mf_nom < 0
A_pose_norm ≈ 0
```

좌우 feature gradient가 상쇄되었을 가능성이 큽니다.

그리고

```python
A_progress
```

는 이론적으로 항상 0이어야 합니다. 0이 아니면 state ordering이나 coefficient assembly 문제입니다.

---

## 7.3 Covariance trajectory

다음 값을 stage별로 출력하십시오.

```python
P_f = x_sol[:, 4]
P_l = x_sol[:, 5]
P_psi = x_sol[:, 6]
P_lpsi = x_sol[:, 7]

sigma_f = np.sqrt(np.maximum(P_f, 0.0))
sigma_l = np.sqrt(np.maximum(P_l, 0.0))
sigma_psi = np.sqrt(np.maximum(P_psi, 0.0))
```

추천 출력 형식은 다음과 같습니다.

```text
stage
v
omega
sigma_f
sigma_l
sigma_psi
h_mf_nom
h_mf_sol
mf_slack
```

### 해석

- velocity가 커지는 stage부터 covariance와 MF violation이 급격히 증가  
  → covariance coefficient 문제
- 정지 solution에서는 covariance가 거의 증가하지 않음  
  → optimizer가 의도적으로 covariance를 억제하기 위해 정지
- stage 1부터 이미 violation  
  → covariance propagation보다 초기 feature sum 문제

---

# 8. augmented PSDF 내부에서 추가해야 할 per-feature debug

Aggregate probability sum만 보면 왜 큰지 알 수 없습니다. `forward_mf()` 내부에서 debug mode일 때 아래 값들을 반환하거나 저장하는 것이 가장 좋습니다.

각 stage와 feature에 대해:

```text
valid mask m_l
feature family: R→O or O→R
obstacle index
feature distance d_l
projected standard deviation sigma_l
standardized value zeta_l
violation probability p_l
feature-distance gradient q_l
```

특히 다음 summary가 중요합니다.

```python
valid_feature_count
probability_sum
max_probability
top_5_probabilities
count_p_gt_1e_3
count_p_gt_1e_2
sum_probability_R_to_O
sum_probability_O_to_R
sum_probability_per_obstacle
```

가장 유용한 출력은 다음 형태입니다.

```text
stage 3:
epsilon = 0.20
sum_p = 0.47
valid_count = 24

top features:
family  obs  d       sigma   zeta    p
R->O    2    0.018   0.012   -1.42   0.078
O->R    2    0.018   0.012   -1.42   0.078
R->O    3    0.022   0.015   -1.40   0.081
...
```

이 결과로 다음을 구분할 수 있습니다.

- 한 개 feature의 $p_\ell$이 큼
- 좌우 벽 feature가 동시에 큼
- R→O와 O→R에서 같은 event가 반복됨
- 많은 작은 probability가 누적되어 sum이 커짐
- 먼 obstacle feature까지 비정상적으로 기여함

---

# 9. MF slack cost와 progress cost를 직접 비교

이 값 하나만 계산해도 왜 정지하는지 거의 바로 알 수 있습니다.

Normalized MF slack을

$$
\tilde\xi_i^M
=
\frac{\xi_i^M}{\alpha_i^M}
$$

라고 하면,

```python
xi_mf_norm = mf_slack / mf_scale

J_mf = np.sum(
    mf_slack_linear * xi_mf_norm
    + 0.5 * mf_slack_quadratic * xi_mf_norm**2
)
```

progress 부분은

```python
r_vs = 0.5 * q_s / v_s_target

J_progress = np.sum(
    r_vs * v_s_traj**2
    - q_s * v_s_traj
)
```

입니다.

다음처럼 출력하십시오.

```text
J_MF_slack
J_progress
J_contour
J_lag
J_input
```

### 해석

```text
J_MF_slack >> abs(J_progress)
```

이면 정지는 정상적인 optimizer 선택입니다.

이 경우 solver가 잘못된 것이 아니라 weight scale이 잘못된 것입니다.

---

# 10. 권장 debug 순서

## Step 1. Slack index 확인

hard G + soft M 한 개로 정확히 구성했는지 확인합니다.

```text
idxsh = [1]
ns = 1
sl[0] = MF slack
```

## Step 2. 아주 약한 MF로 시작

```python
mf_slack_linear = 1.0
mf_slack_quadratic = 0.1
mf_active_end_step = 5
chance_epsilon = 0.20
```

이 상태에서 진행 여부를 확인합니다.

## Step 3. Weight sweep

진행한다면

```text
1 → 3 → 10 → 30
```

순서로 MF linear weight를 증가시킵니다.

로봇이 다시 멈추기 시작하는 지점 바로 아래 값을 사용합니다.

## Step 4. Active horizon 확장

```text
1-5 → 1-10 → 1-20
```

순서로 늘립니다.

## Step 5. Fixed covariance ablation

여전히 멈춘다면 모든 `alpha`, `beta`, `q`를 0으로 둡니다.

- 진행함 → covariance growth coefficient 문제
- 계속 멈춤 → nominal probability sum 또는 feature 중복 문제

## Step 6. Per-feature probability 확인

각 stage의 top probability feature를 확인합니다.

- 몇 개 feature가 지배 → geometry 및 covariance 확인
- 많은 작은 값이 누적 → Boole conservatism 또는 feature count 문제
- R→O/O→R 값이 반복 → repeated-event counting 문제

---

# 최종 추천 설정

첫 번째 정상동작 확인용 설정은 다음이 적절합니다.

```python
# Row G
# hard constraint
d_col = 0.001

# Row M
mf_slack_linear = 1.0
mf_slack_quadratic = 0.1

chance_epsilon = 0.20
d_min = 0.001
d_mf_mask = 0.0

mf_active_start_step = 1
mf_active_end_step = 5
mf_active_terminal = False
```

그리고 아래 여섯 항목을 매 cycle 출력하면 됩니다.

```text
1. solve_mode
2. u0 = [v0, omega0, v_s0]
3. nominal probability sum
4. mf_affine_residual
5. normalized MF slack
6. ||A_pose|| and ||A_cov||
```

판단 기준은 간단합니다.

$$
\boxed{
\|A_{\mathrm{cov}}^{MF}\|
\gg
\|A_{\mathrm{pose}}^{MF}\|
}
$$

이면서 physical velocity가 0으로 떨어지면, MF가 obstacle avoidance 방향을 만드는 것보다 **covariance를 줄이는 slowdown constraint로 작동해서 정지하는 것**입니다.

반대로

$$
\boxed{
\sum_\ell p_{\ell,i}>\varepsilon_i
}
$$

가 정지 상태에서도 계속 유지되면, weight tuning만으로 해결할 문제가 아니라 probability sum을 구성하는 feature들을 확인해야 합니다.


