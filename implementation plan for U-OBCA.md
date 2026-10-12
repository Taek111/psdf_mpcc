
추천하는 방향은 **기존 `OBCAOptimizer`를 상속하고, collision constraint와 uncertainty parameter 처리만 교체하는 것**이다. 현재 코드의 dynamics, tracking cost, input bounds, IPOPT 설정, problem 재사용 구조는 그대로 활용하면 된다. 이미 `_add_convex_pair_constraint()`가 분리되어 있어서 이 방식에 잘 맞는다. `obca_optimizer.py` `obca_optimizer.py`

다만 **robot footprint가 rectangle인지 먼저 확인해야 한다.** 현재 코드는 일반 `ConvexRegion2D`를 받지만, 논문의 polygon-polygon U-OBCA는 rectangular robot을 가정하여 dual equality를 제거한다. 첨부된 optimizer만으로 실제 footprint가 rectangle인지는 확인할 수 없다. 임의의 polygon을 bounding box로 바꾸는 것은 같은 geometry를 사용하는 구현이 아니다. `obca_optimizer.py` `u-obca.pdf`

아래에서는 첫 구현 범위를 **rectangular robot + 정확히 알려진 static polygon obstacles + robot pose uncertainty**로 제안한다. 이 조건이면 논문의 일반식을 상당히 짧게 구현할 수 있다.

## 1. U-OBCA에서 실제로 바뀌는 부분

현재 OBCA는 각 obstacle-stage pair에 대해 다음 certificate를 사용한다.

$$
\begin{aligned}
\lambda,\mu&\ge0,\\
\|A^\top\lambda\|_2&\le1,\\
G^\top\mu+R(\theta)^\top A^\top\lambda&=0,\\
-g^\top\mu+(At-b)^\top\lambda&\ge d_{\min}.
\end{aligned}
$$

여기서 `(A,b)`는 obstacle의 world-frame half-space representation, `(G,g)`는 robot의 body-frame representation이다. 현재 코드와 그대로 대응한다. `obca_optimizer.py`

U-OBCA는 단순히 마지막 식에 uncertainty margin 하나를 붙이는 방법이 아니다. **Heading uncertainty 때문에 dual equality 자체가 stochastic해지는 문제를 먼저 처리한다.** Rectangle 구조를 이용해 $\mu$를 제거한 뒤, 필요한 조건을 **dual nonnegativity에 대한 두 chance constraints와 separation에 대한 하나의 chance constraint**로 바꾼다. 이 세 조건을 Wasserstein distributionally robust tightening으로 deterministic constraints로 변환한다. `u-obca.pdf` `u-obca.pdf` `u-obca.pdf`

따라서 구현 관점의 핵심은 다음과 같다.

> **`x`, `u`, obstacle dual $\lambda$는 유지하고, robot dual $\mu\in\mathbb R^4$를 auxiliary variable $\xi\in\mathbb R^2$로 교체한다. 기존 dual equality는 제거하고, 세 개의 uncertainty-aware inequalities를 넣는다.**

여기서 $\xi$는 네 method의 log-Cholesky state와 무관하다. 코드에서는 `xi_dual`처럼 이름을 구분하는 편이 좋다.

## 2. 네 코드에 맞춘 짧은 constraint formulation

아래 식은 **논문 Eq. (19), (22), (29)를 deterministic static obstacle에 맞춰 직접 정리한 특수형**이다. 원문의 $q,r,b^v$ 및 큰 joint covariance를 전부 구현하는 대신, 같은 조건을 작은 행렬로 표현한다. `u-obca.pdf` `u-obca.pdf` `u-obca.pdf`

### Geometry와 auxiliary variables

Rectangle representation을 다음 순서로 맞춘다.

$$
G=
\begin{bmatrix}I_2\\-I_2\end{bmatrix},
\qquad
g=\begin{bmatrix}g_+\\g_-\end{bmatrix},
\qquad
g_+,g_-\in\mathbb R^2.
$$

$g_+$와 $g_-$를 별도로 유지하므로 robot reference point가 rectangle 중심일 필요는 없다.

각 obstacle-stage pair에서

$$
n=A^\top\lambda,
\qquad
\lambda\ge0,\qquad \xi\in\mathbb R_{\ge0}^{2}
$$

를 사용한다. 논문의 elimination은 이 표기에서

$$
\mu(\Theta)=
\begin{bmatrix}
\xi-R(\Theta)^\top n\\
\xi
\end{bmatrix}
$$

가 된다. $\Theta$는 uncertain heading이다.

### 추가할 세 개의 inequalities

Robot의 random position을 $T$, mean pose를 $(t,\theta)$라 하고,

$$
M=\mathbb E[R(\Theta)],
\qquad
C_1=\operatorname{Cov}(R(\Theta)e_1),
\qquad
C_2=\operatorname{Cov}(R(\Theta)e_2),
$$

$$
C_3=\operatorname{Cov}\!\left(T+R(\Theta)g_+\right)
$$

로 정의한다. $e_1,e_2$는 2D standard basis이다.

그러면 구현할 조건은 다음뿐이다.

$$
\boxed{
\begin{aligned}
&\lambda\ge0,\qquad \xi\ge0,\qquad n^\top n\le1,\\[1mm]
&\xi_1-n^\top Me_1
\ge \eta_1\sqrt{n^\top C_1n},\\
&\xi_2-n^\top Me_2
\ge \eta_2\sqrt{n^\top C_2n},\\
&n^\top(t+Mg_+)-b^\top\lambda
-(g_++g_-)^\top\xi\\
&\hspace{18mm}\ge
d_{\min}+\eta_3\sqrt{n^\top C_3n}.
\end{aligned}}
$$

첫 두 식은 uncertain $\mu$의 nonnegativity를, 마지막 식은 uncertain separation을 다룬다. **Static obstacle을 정확히 알고 있다면 obstacle pose, obstacle noise, moving-obstacle prediction을 새로 만들 필요가 없다.**

### Moment 계산도 작은 helper로 끝낼 수 있다

논문은 position과 heading noise의 independence, zero-mean noise, symmetric heading noise를 가정한다. 이 가정 아래

$$
C_3=P_t+\operatorname{Cov}(R(\Theta)g_+),
$$

이며 $P_t=\operatorname{Cov}(T)$이다. `u-obca.pdf`

구현용으로 다음 closed form을 사용할 수 있다. 이는 위 가정에서 직접 계산한 rotation moments이다.

$$
c_1=\mathbb E[\cos w_\theta],
\qquad
c_2=\mathbb E[\cos 2w_\theta],
\qquad
M=c_1R(\theta),
$$

$$
\operatorname{Cov}(R(\Theta)a)
=
R(\theta)
\left[
\left(\frac{1+c_2}{2}-c_1^2\right)aa^\top
+
\frac{1-c_2}{2}(Ja)(Ja)^\top
\right]
R(\theta)^\top,
\quad
J=\begin{bmatrix}0&-1\\1&0\end{bmatrix}.
$$

이 helper에 $a=e_1,e_2,g_+$를 넣으면 필요한 covariance가 모두 나온다. Gaussian heading noise $w_\theta\sim\mathcal N(0,s_\theta^2)$를 채택한다면 $c_1=e^{-s_\theta^2/2}$, $c_2=e^{-2s_\theta^2}$로 계산하면 된다.

**주의:** 네 method의 covariance에는 lateral-heading cross-covariance가 포함된다. 이를 위 independence 기반 helper에 그대로 넣거나, 해당 항을 설명 없이 삭제하면 안 된다. 첫 구현에서는 지원하는 uncertainty assumption을 명시하고, nonzero position-heading cross-covariance 입력은 검출하도록 하는 편이 안전하다. `method_description.md` `u-obca.pdf`

## 3. Covariance와 Wasserstein coefficient 처리

### Covariance는 optimization state로 추가하지 않는다

논문의 practical implementation인 Remark 11처럼 **initial-guess trajectory에서 covariance matrices를 계산하고, 해당 NLP solve 동안 고정**하는 것으로 시작하면 된다. 따라서 네 log-Cholesky dynamics나 control-dependent covariance propagation을 이번 baseline에 이식할 필요는 없다. `u-obca.pdf`

위 특수형에서는 다음 구분이 중요하다.

- $C_1,C_2,C_3$: initial guess에서 계산하여 parameter로 고정한다.
- Mean 항의 $t$, $M=c_1R(\theta)$: 현재 optimization variables로 계산한다.
- $\sqrt{n^\top C_an}$: covariance matrix는 고정되어도 $\lambda$, 즉 $n$에 따라 solve 안에서 변한다.

따라서 **covariance를 freeze하는 것과 uncertainty margin 전체를 상수로 만드는 것은 다르다.**

### $\eta_1,\eta_2,\eta_3$는 시작 시 한 번 계산한다

Risk allocation은 우선 논문의 설정처럼

$$
(\alpha_1,\alpha_2,\alpha_3)
=(0.2\alpha,\;0.2\alpha,\;0.6\alpha)
$$

로 고정하면 충분하다. 이를 optimization variables로 만들 이유는 없다. 논문의 corridor simulation도 이 비율을 사용한다. `u-obca.pdf`

각 $\eta_a$는 Eq. (12)의 scalar problem으로 계산한다. Gaussian reference에 대해 식을 정리하면, $z_a=\Phi^{-1}(1-\alpha_a)$일 때

$$
\eta_a\big[\Phi(\eta_a)-\Phi(z_a)\big]
+\varphi(\eta_a)-\varphi(z_a)
=\theta_{w,a},
\qquad \eta_a\ge z_a
$$

의 root를 찾으면 된다. 작은 bisection helper면 충분하며, $\theta_{w,a}=0$이면 $\eta_a=z_a$이다. 이는 원문의 Eq. (12)를 Gaussian reference에 맞춰 정리한 식이다. `u-obca.pdf`

$\theta_w$는 geometric clearance가 아니라 **distributional ambiguity radius**이다. 또한 $\alpha$는 원문에서 obstacle-stage별 risk bound이므로, 그대로 trajectory 전체의 collision probability라고 해석하면 안 된다. `u-obca.pdf` `u-obca.pdf`

## 4. 코드 변경의 큰 그림

파일은 우선 다음 정도로 제한하는 것을 권한다.

```text
control/u_obca_optimizer.py
    UOBCAOptimizerParam(OBCAOptimizerParam)
    UOBCAOptimizer(OBCAOptimizer)
    rotation_moments(...)
    wasserstein_eta(...)

tests/test_u_obca.py
```

**별도 planner framework, uncertainty model hierarchy, generic distribution interface는 만들지 않는다.**

| 수정 지점 | 필요한 변경 |
|---|---|
| `_make_constraint_pair()` | Robot rectangle의 half-space row를 canonical order로 정리한다. 정리된 geometry로 기존 distance query와 dual initialization을 사용한다. |
| `initialize_variables()` | `super()`로 `x`, `u`를 만든 뒤, uncertainty moments용 `Opti.parameter`만 추가한다. |
| `_add_convex_pair_constraint()` | 기존 $\mu$와 dual equality를 제거한다. $\lambda,\xi$ 및 위 세 inequalities를 생성한다. |
| `_set_dual_initial()` | Canonical robot representation의 deterministic dual에서 `xi_dual` 초기값을 만든다. 위 표기에서는 `mu_curr[2:4]`가 대응한다. 이는 initial guess이지 chance-feasible certificate는 아니다. |
| `setup()` | Initial guess와 noise statistics로 covariance parameters를 갱신한다. 기존 geometry selection과 graph reuse는 유지한다. |
| `add_warm_start()` / `solve_nlp()` | 이전 `x,u` solution을 저장·shift하는 정도만 추가한다. 같은 guess를 covariance 평가에도 사용한다. |

현재 `add_warm_start()`는 하나의 `x_ws`를 모든 prediction stage에 반복해서 넣는다. Covariance를 nominal trajectory에서 평가하려면, 이 부분은 **shifted previous solution 또는 dynamics rollout으로 만든 horizon trajectory**로 바꾸는 것이 좋다. `obca_optimizer.py`

구조상 특별한 문제가 없다면 **`_build_problem()` 전체를 복사할 필요는 없다.** 기존 build 과정이 `initialize_variables()`와 obstacle constraint method를 호출하므로 해당 hook을 활용하면 된다. 다만 base method가 `_parameters`를 초기화하므로, 새 parameter는 그 이후 실행되는 hook에서 생성해야 한다. `obca_optimizer.py`

또한 매 cycle 바뀌는 covariance arrays를 `param`의 속성에 계속 넣는 방식은 피하는 것이 좋다. 현재 `_graph_signature()`가 `vars(param)` 전체를 포함하기 때문에, 그렇게 하면 covariance 변경만으로도 NLP graph가 재생성된다. **Static configuration은 `param`, runtime moments는 `Opti.parameter`로 분리**하면 된다. `obca_optimizer.py` `obca_optimizer.py`

온라인 흐름은 다음 정도면 충분하다.

```text
현재 state 및 uncertainty 입력
→ nominal trajectory 준비
→ 기존 obstacle selection / 필요 시 graph build
→ covariance parameters와 warm start 갱신
→ IPOPT 한 번 solve
→ 기존 방식으로 첫 input 적용
```

## 5. Codex에는 이 순서로 구현시키면 된다

**첫째, uncertainty가 없는 경우부터 연결한다.**  
Rectangle canonicalization과 $\mu\rightarrow\xi$ elimination을 먼저 구현하고, 고정 pose에서 기존 OBCA와 certificate feasibility가 일치하는지 검사한다. 서로 다른 NLP parameterization이므로 최종 trajectory의 bitwise 일치까지 요구할 필요는 없다.

**둘째, rotation moments와 세 tightening을 추가한다.**  
Moment helper를 Monte Carlo mean/covariance와 비교하고, zero-noise 입력에서 deterministic constraints로 돌아가는지 확인한다. Zero variance에서 square-root derivative가 문제가 되지 않도록 deterministic 경로도 검사한다.

**셋째, single-obstacle scenario 이후 maze에 연결한다.**  
처음에는 cutoff를 끄고 constraint를 검증한 뒤 기존 selection을 연결한다. Covariance 값만 바꿨을 때 graph가 불필요하게 rebuild되지 않는지도 확인한다. 현재 obstacle selection은 current-pose distance를 사용하므로, 그 자체를 horizon 전체의 uncertainty-aware screening으로 간주하면 안 된다. `obca_optimizer.py`

마지막으로 **현재 collision threshold는 `safe_dist`가 아니라 `margin_dist`**이다. 실제 inequality에 들어가는 값이 `param.margin_dist`이므로 이를 $d_{\min}$에 연결하고, uncertainty margin은 위 tightening으로 추가해야 한다. `obca_optimizer.py`

이 범위라면 첫 버전은 **기존 MPC 구조를 유지한 U-OBCA static-obstacle specialization**으로 정리할 수 있다. Moving-obstacle uncertainty, arbitrary robot polygon 지원, OCP 내부 covariance propagation은 첫 구현에서 제외하고, 필요한 경우에만 별도 확장하는 것이 유지보수에 유리하다.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)