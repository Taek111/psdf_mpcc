
# Signed PSDF Guard/Recovery + Soft MF Row 최종 변경안

## 1. 최종 설계 원칙

acados에는 prediction stage마다 항상 다음 두 개의 scalar constraint row를 유지한다.

1. **Row G:** signed PSDF를 이용한 geometric guard/recovery
    
2. **Row M:** Boole-based multi-feature chance constraint
    

두 row의 종류와 인덱스는 runtime에 변경하지 않는다. Constraint-level `NOMINAL/RECOVERY` mode, progressive recovery target, row switching은 사용하지 않는다.

최종 구조는 다음과 같다.

$$
\boxed{  
\text{constant soft Row G}  
+  
\text{stage-wise masked soft Row M}  
}  
$$

- Row G는 $i=1,\ldots,N$에서 항상 활성화한다.
    
- Row M은 feature-distance model이 유효한 separated domain에서만 활성화한다.
    
- 두 row 모두 lower-bound soft constraint로 구현한다.
    
- Row G slack linear penalty를 Row M보다 한 자리 크게 설정한다.
    
- obstacle 또는 feature 수와 관계없이 acados constraint 수는 stage당 2개로 고정한다.
    

이 구조는 기존 프롬프트의 progressive recovery target과 global recovery mode를 제거하고, 고정된 Row G가 normal configuration에서는 guard로, penetration에서는 recovery row로 작동하도록 한 최종안이다.

---

## 2. State ordering

Augmented state는 다음 순서를 사용한다.

$$
z_i
=
\begin{bmatrix}  
q_i^\top &  
s_i &  
\eta_i^\top  
\end{bmatrix}^{\top}
=
\begin{bmatrix}  
x_i&  
y_i&  
\theta_i&  
s_i&  
P_{f,i}&  
P_{l,i}&  
P_{\theta,i}&  
P_{l\theta,i}  
\end{bmatrix}^{\top}  
\in\mathbb R^8,  
$$

여기서 pose state는

$$
q_i
=
\begin{bmatrix}  
x_i&y_i&\theta_i  
\end{bmatrix}^{\top}.  
$$

---

## 3. Row G: Signed PSDF guard/recovery

Shifted nominal pose $\bar q_i$에서 signed PSDF를 선형화한다.

$$
\bar\phi_i
=
\phi(\bar q_i,\mathcal E),  
\qquad  
\bar g_i
=
\nabla_q\phi(\bar q_i,\mathcal E).  
$$

Affine PSDF model은

$$
\hat\phi_i(q_i)
=
\bar\phi_i  
+  
\bar g_i^\top(q_i-\bar q_i)  
$$

이다.

최종 Row G는 모든 future stage에서 동일한 constant target $d_{\mathrm{col}}$을 사용한다.

$$
\boxed{  
h_i^G(z_i)
=
\bar\phi_i  
+  
\bar g_i^\top(q_i-\bar q_i)
-
d_{\mathrm{col}}  
}  
$$

Soft lower-bound constraint는

$$
\boxed{  
h_i^G(z_i)+\xi_i^G\ge0,  
\qquad  
\xi_i^G\ge0  
}  
$$

또는 동등하게

$$
h_i^G(z_i)\ge-\xi_i^G  
$$

로 쓴다.

Affine coefficient는 다음과 같다.

$$
\boxed{  
A_i^G
=
\begin{bmatrix}  
\bar g_{x,i}&  
\bar g_{y,i}&  
\bar g_{\theta,i}&  
0&0&0&0&0  
\end{bmatrix}^{\top}  
}  
$$

$$
\boxed{  
c_i^G
=
\bar\phi_i
-
\bar g_i^\top\bar q_i
-
d_{\mathrm{col}}  
}  
$$

따라서

$$
h_i^G(z_i)
=
(A_i^G)^\top z_i+c_i^G.  
$$

### 3.1 Guard와 recovery의 동작

Row G의 식과 target은 normal state와 penetration state에서 동일하다.

$$
\boxed{  
d_i^{\mathrm{target}}=d_{\mathrm{col}},  
\qquad i=1,\ldots,N  
}  
$$

별도의

$$
\tau_i
=
\min(d_{\mathrm{col}},\phi_0+i\rho_{\mathrm{rec}})  
$$

와 같은 progressive target은 사용하지 않는다. `rho_recovery`, `target_i`, constraint-level `recovery_mode`도 제거한다.

Normal configuration에서는

$$
\hat\phi_i(q_i)\ge d_{\mathrm{col}}  
$$

이 no-penetration guard로 작동한다.

Penetration에서는 $h_i^G<0$이므로 최적 lower slack은 개념적으로

$$
\xi_i^{G\star}
=
\left[  
d_{\mathrm{col}}-\hat\phi_i(q_i)  
\right]_+  
$$

가 된다. Linear slack cost가

$$
w_G\xi_i^G  
$$

이면 violated region에서 상태에 대한 cost gradient는

$$
\nabla_{q_i}  
\left(  
w_G\xi_i^{G\star}  
\right)
=
-w_G\bar g_i.  
$$

따라서 minimization direction은

$$
+\bar g_i  
$$

이며, 이는 signed PSDF를 증가시키는 방향이다. 즉, 같은 Row G가 penetration에서 별도 mode switching 없이 recovery direction을 제공한다.

---

## 4. Row M: Soft MF probability-margin row

기존 augmented PSDF가 반환하는 MF affine coefficient를 유지한다.

$$
\boxed{  
h_i^M(z_i)
=
(A_i^{\mathrm{MF}})^\top z_i  
+  
c_i^{\mathrm{MF}}  
}  
$$

Soft lower-bound constraint는

$$
\boxed{  
h_i^M(z_i)+\xi_i^M\ge0,  
\qquad  
\xi_i^M\ge0  
}  
$$

이다.

Row M은 geometric collision guard나 penetration recovery를 담당하지 않는다. Row M의 역할은 separated configuration에서 Boole-based probability margin을 추가하는 것이다.

---


## 5. MF row의 stage-wise domain mask

MF row는 unsigned Euclidean feature distance와 feature-distance gradient가 유효한 separated domain에서만 사용한다.

Stage별 domain indicator를

$$
\chi_i^M
=
\begin{cases}  
1,  
&  
\bar\phi_i>d_{\mathrm{mask}}  
\text{ and MF coefficient data are valid},  
\\[1mm]  
0,  
&  
\text{otherwise}  
\end{cases}  
$$

로 정의한다.

### 5.1 MF 활성 stage

$$
\chi_i^M=1  
$$

이면 기존 MF affine coefficient를 그대로 사용한다.

$$
\boxed{  
A_i^M=A_i^{\mathrm{MF}},  
\qquad  
c_i^M=c_i^{\mathrm{MF}}  
}  
$$

### 5.2 MF 비활성 stage

$$
\chi_i^M=0  
$$

이면 Row M을 trivially feasible한 constant row로 만든다.

$$
\boxed{  
A_i^M=0,  
\qquad  
c_i^M=\varepsilon_i  
}  
$$

따라서

$$
h_i^M(z_i)
=
\varepsilon_i>0  
$$

가 되어 해당 stage의 MF row가 실질적으로 비활성화된다.

기존 프롬프트의

$$
c_i^M=1  
$$

도 수학적으로는 row를 비활성화하지만, $\varepsilon_i$를 사용하는 편이 기존 MF residual

$$
h_i^{\mathrm{MF}}
=
\varepsilon_i-\sum_\ell p_{\ell,i}  
$$

의 scale과 일관된다.

### 5.3 Mask의 의미

$d_{\mathrm{mask}}$는 Row G와 Row M 사이의 safety responsibility를 전환하는 threshold가 아니다.

- Row G는 $i=1,\ldots,N$에서 항상 활성화된다.
    
- $d_{\mathrm{mask}}$는 MF feature-distance model의 적용 domain만 제한한다.
    
- 따라서 MF masking이 발생해도 geometric guard가 사라지지 않는다.
    
- 현재 measured pose가 penetration인지 여부를 이용해 horizon 전체 MF row를 일괄적으로 끄지 않는다.
    
- 각 stage의 $\bar\phi_i$를 이용해 해당 stage만 개별적으로 mask한다.
    

변수 이름도 d_mf_switch보다 d_mf_mask가  적절하다.

---

## 6. Slack objective와 priority

두 row를 모두 soft constraint로 설정한다.

전체 objective는 다음과 같다.

$$
\boxed{  
J
=
J_{\mathrm{MPCC}}  
+  
\sum_{i=1}^{N}  
\left[  
w_G\xi_i^G  
+  
\frac12W_G(\xi_i^G)^2  
+  
w_M\xi_i^M  
+  
\frac12W_M(\xi_i^M)^2  
\right]  
}  
$$

Slack penalty는 raw residual 단위에서

$$
\boxed{w_G > w_M}
$$

로 설정한다. 구현값은 Guard linear/quadratic $=(10^2,10^1)$, MF
linear/quadratic $=(10^1,10^0)$이다. 서로 단위가 다른 tracking 항과의 단순
스칼라 크기 비교는 hierarchy의 보장으로 해석하지 않는다.

Linear penalty를 주 penalty로 사용하고 quadratic penalty는 numerical
regularization 수준으로 둔다. Row coefficient와 penalty는 정규화하거나
stage별로 다시 매핑하지 않으며, intermediate/terminal stage에서 같은 고정값을
사용한다.

- $\xi_i^G>0$: geometric guard/recovery violation
    
- $\xi_i^M>0$: modeled MF probability-sum condition violation
    
- Row G linear slack은 safety-critical 우선순위를 위해 Row M보다 한 자리 크게 둔다.
    
- 모든 obstacle row를 soft하게 하면 obstacle constraint 자체가 만드는 infeasibility는 제거되지만, dynamics equality, state/input bounds, NaN 또는 QP numerical failure까지 자동으로 제거되는 것은 아니다.
    

최종안은 hard obstacle row를 남기지 않는다.


---

## 7. Stage 0 처리

현재 state는

$$
z_0=\hat z_k  
$$

로 고정되어 있으므로 stage 0에서 violated safety constraint를 부과해도 optimizer가 상태를 수정할 수 없다.

따라서 다음을 적용한다.

- `con_h_expr_0`을 정의하지 않는다.
    
- `lh_0`, `uh_0`, `idxsh_0`도 설정하지 않는다.
    
- Row G와 Row M은 $i=1,\ldots,N-1$ 및 terminal stage $N$에만 적용한다.
    
- Current exact PSDF와 MF residual은 diagnostic 및 supervisor에만 사용한다.
    

Code structure상 stage 0에도 parameter vector를 설정해야 한다면 safety parameter는 trivially feasible하게 채운다.

```python
guard_affine_0 = np.array([
    0.0, 0.0, 0.0, 1.0
])

mf_affine_0 = np.zeros(9)
mf_affine_0[-1] = float(epsilon_0)
```

그러나 `con_h_expr_0`이 없으므로 이 값은 stage-0 nonlinear constraint로 사용되지 않는다.


# Prompt for modifying


```
Modify the RMPCC-MF implementation to use exactly two fixed scalar affine
constraint rows at every intermediate stage i=1,...,N-1 and at terminal
stage N. Do not define con_h_expr_0.

State ordering:
[x, y, theta, s, P_f, P_l, P_theta, P_ltheta].

Constraint index 0: signed-PSDF guard/recovery row

h_guard(z_i)
= phi_bar_i
+  g_bar_i^T (x_i - x_bar_i)
- d_col
= A_guard_i^T z_i + c_guard_i.

Use:
A_guard_i =
[g_bar_i[0], g_bar_i[1], g_bar_i[2], 0, 0, 0, 0, 0]

c_guard_i =
phi_bar_i - g_bar_i^T x_bar_i - d_col.

Use the same constant target d_col at every stage. Do not introduce a
recovery mode, a progressive recovery target, rho_recovery, or
mode-dependent Row-G semantics. When the current state is penetrating,
the high-priority lower slack and the signed PSDF gradient must make the
same Row G act as the recovery row.

Constraint index 1: MF probability-margin row

h_mf(z_i) = A_mf_i^T z_i + c_mf_i.

Retain the existing MF affine coefficients only when the stage is in the
valid separated domain. If phi_bar_i <= d_mf_mask, or the MF stage data
are invalid, disable only that stage's MF row using:

A_mf_i = zeros(8)
c_mf_i = epsilon_i.

Do not globally disable all future MF rows based on the exact current
signed PSDF.

Make both rows lower-bound soft constraints:

idxsh   = [0, 1]
idxsh_e = [0, 1]

Use fixed raw-row penalties with the Guard linear penalty one order larger
than the MF linear penalty:

Use relatively small quadratic slack penalties. The selected values are:

guard_slack_linear    = 1e2
guard_slack_quadratic = 1e1
mf_slack_linear       = 1e1
mf_slack_quadratic    = 1e0

Do not normalize the row coefficients or remap slack penalties by row norm.
Use these same raw-unit penalties at every intermediate and terminal stage.

Add a 4-scalar guard parameter
[g_x, g_y, g_theta, c_guard]
and retain the existing 9-scalar MF parameter [A_mf, c_mf].

The complete stage parameter remains:
[path parameters, guard parameters, MF parameters]
with dimension 6 + 4 + 9 = 19.

Use the world-frame PSDF gradient returned by forward_mf().
Contact and penetration are handled by Row G.

Do not change the number or meaning of the two constraint rows at runtime.
Only their stage parameters and the domain mask of Row M may change.

Compute exact_current_phi directly at the measured current pose, or assert
that x_bar_0 is exactly re-anchored to the measured pose before using
phi_bar[0].

Log, with fixed constraint ordering:
0 = guard/recovery
1 = MF

- exact current PSDF
- guard affine residual and fresh exact-PSDF residual
- MF affine residual and fresh recomputed MF residual
- guard lower slack
- MF lower slack
- per-stage MF mask
- minimum exact PSDF at predicted nodes
- minimum exact PSDF over first-interval substeps
- raw row coefficient norms (diagnostics only; never used for scaling)
- solver and QP residual diagnostics.

```
