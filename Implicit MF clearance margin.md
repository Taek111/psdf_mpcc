

## 1. 무슨 뜻인가: 확률의 여유 대신, clearance threshold의 여유를 계산합니다

현재 constraint는 다음 질문을 합니다.

> 현재 요구 clearance $d_{\min}$에서, feature violation probability의 합이 $\varepsilon$ 이하인가?

제안한 방식은 같은 조건을 다음 질문으로 바꿉니다.

> **확률합이 정확히 $\varepsilon$이 될 때까지, 요구 clearance $d_{\min}$을 얼마나 높일 수 있는가?**

그 높일 수 있는 양을 $r$이라고 정의합니다.

예를 들어 $r=0.02\,\mathrm m$이면, 현재 distance와 covariance를 그대로 두고 요구 clearance를 $d_{\min}+0.02\,\mathrm m$까지 높여도 probability-sum 조건을 만족한다는 뜻입니다. 반대로 $r=-0.02\,\mathrm m$이면, 요구 clearance를 $0.02\,\mathrm m$ 낮춰야 경계에 도달합니다.

**이 $r$은 robot이 어느 방향으로든 이동해도 되는 거리가 아닙니다.** 어디까지나 *공통 clearance threshold에 대한 여유*입니다.

---

## 2. 최종 권장 formulation

표기를 줄이기 위해 stage index $i$는 생략하고, valid feature만 합산하겠습니다. 아래 $d_\ell(z),\sigma_\ell(z)$는 문서의 nominal-geometry model인 $\hat d_{\ell,i}(x_i),\hat\sigma_{\ell,i}(\xi_i)$를 간단히 쓴 것입니다. 즉, distance는 pose에 affine이고, standard deviation은 covariance state에 비선형적으로 의존합니다. `paper_outline.md`

### 2.1. 확률합을 만족하는 clearance margin을 정의합니다

먼저 threshold shift $r$에 대한 확률합을

$$
F(z,r)
:=
\sum_\ell
\Phi\!\left(
\frac{d_{\min}+r-d_\ell(z)}
{\sigma_\ell(z)}
\right)
$$

로 둡니다. 그리고 **$r(z)$를 다음 scalar equation의 해로 정의**합니다.

$$
\boxed{
F(z,r(z))=\varepsilon.
}
\tag{1}
$$

각 $\sigma_\ell>0$이므로

$$
\frac{\partial F}{\partial r}
=
\sum_\ell
\frac{1}{\sigma_\ell}
\varphi_{\mathcal N}\!\left(
\frac{d_{\min}+r-d_\ell}{\sigma_\ell}
\right)>0.
$$

따라서 $F$는 $r$에 대해 strictly increasing입니다. Valid feature가 하나 이상이면 $0<\varepsilon<1$에서 해가 유일하게 존재하며,

$$
\boxed{
\sum_\ell
\Phi\!\left(
\frac{d_{\min}-d_\ell(z)}{\sigma_\ell(z)}
\right)
\le\varepsilon
\quad\Longleftrightarrow\quad
r(z)\ge0.
}
\tag{2}
$$

**즉, 기존 MF constraint의 feasible set은 바뀌지 않습니다.** 기존의 probability residual 대신, 단위가 meter인 clearance residual을 사용하는 것입니다. 단, 여기서 보존되는 것은 현재의 **Boole-based affine Gaussian feature model**이지, exact nonlinear collision probability는 아닙니다. `paper_outline.md`

### 2.2. Nominal에서 gradient를 계산합니다

Nominal $\bar z$에서 식 (1)을 풀어 $\bar r=r(\bar z)$를 얻고,

$$
\bar\zeta_\ell
:=
\frac{d_\ell(\bar z)-d_{\min}-\bar r}
{\sigma_\ell(\bar z)}
$$

를 정의합니다. 가중치는

$$
\bar\varpi_\ell
:=
\frac{
\varphi_{\mathcal N}(\bar\zeta_\ell)/\sigma_\ell(\bar z)
}{
\displaystyle\sum_j
\varphi_{\mathcal N}(\bar\zeta_j)/\sigma_j(\bar z)
},
\qquad
\sum_\ell\bar\varpi_\ell=1
$$

입니다.

Implicit differentiation을 적용하면

$$
\boxed{
\nabla_z r(\bar z)
=
\sum_\ell\bar\varpi_\ell
\left[
\nabla_z d_\ell(\bar z)
-
\bar\zeta_\ell\nabla_z\sigma_\ell(\bar z)
\right].
}
\tag{3}
$$

해석은 단순합니다.

> **각 feature의 Gaussian clearance-margin gradient를 정규화된 가중치로 합합니다.**

인용문의 $u_\ell$는 여기서 $-\bar\zeta_\ell$이므로, 인용문에 있는 $+\;u_\ell\nabla\sigma_\ell$와 같은 식입니다.

### 2.3. QP에는 기존처럼 affine row 하나만 전달합니다

$$
\boxed{
\widehat r(z)
=
\bar r+\nabla_z r(\bar z)^\top(z-\bar z).
}
\tag{4}
$$

따라서 기존 coefficient 이름을 유지하면서

$$
\boxed{
\begin{aligned}
A^{\mathrm{MF}}&=\nabla_z r(\bar z),\\
c^{\mathrm{MF}}&=\bar r-(A^{\mathrm{MF}})^\top\bar z,\\
(A^{\mathrm{MF}})^\top z+c^{\mathrm{MF}}+\delta&\ge0,
\qquad \delta\ge0
\end{aligned}
}
\tag{5}
$$

으로 구성하면 됩니다.

**중요하게도, $r$을 QP의 새로운 decision variable로 추가하는 것이 아닙니다.** QP 밖에서 scalar root와 gradient를 계산하고, QP에는 $A^{\mathrm{MF}},c^{\mathrm{MF}}$만 전달합니다. 현재의 8-dimensional state와 stage당 9-scalar MF interface를 유지할 수 있습니다. `method_description.md` `method_description.md`

---

## 3. 이 방식을 추천하는 가장 명확한 수학적 이유

### Single-feature에서 현재 CC-SF와 정확히 일치합니다

Feature가 하나이면 식 (1)에서 바로

$$
\boxed{
r(z)
=
d(z)-d_{\min}
-\Phi^{-1}(1-\varepsilon)\sigma(z)
}
\tag{6}
$$

를 얻습니다.

이것은 현재 CC-SF가 선형화하는 Gaussian-margin residual과 같은 형태입니다. 동일한 minimizing feature와 nominal-geometry model을 사용하면, **covariance dependence까지 포함하여 residual과 RTI row가 일치**합니다. `method_description.md`

Probit residual은 single-feature에서

$$
\psi(z)=\frac{d(z)-d_{\min}}{\sigma(z)}-\gamma
$$

이므로, $\sigma$가 변하면 현재 SF residual과 따로 선형화한 row가 일반적으로 달라집니다. 이 차이가 $r$에서는 없습니다.

### 모든 distance가 함께 변하는 방향에서는 정확합니다

Covariance를 고정하고 모든 feature distance에 동일한 $c$를 더하면,

$$
\boxed{
r(d_1+c,\ldots,d_K+c;\sigma)
=
r(d_1,\ldots,d_K;\sigma)+c.
}
\tag{7}
$$

따라서 이 공통 변화 방향에서는 residual이 정확히 affine입니다.

이것이 인용문에서 말한 “threshold 쪽 역변환”의 실질적인 장점입니다. **작은 probability 변화량을 외삽하는 대신, clearance threshold의 변화량을 직접 표현**합니다.

단, 모든 distance가 같은 양만큼 변하는 상황과 실제 robot motion은 다릅니다. 마주 보는 두 벽 사이에서 옆으로 움직이면 한쪽 distance는 증가하고 다른 쪽은 감소하므로 식 (7)의 상황이 아닙니다.

### 최종 정리

권장 방식은 다음 세 식입니다.

$$
\boxed{
\begin{gathered}
\sum_\ell
\Phi\!\left(
\frac{d_{\min}+r(z)-d_\ell(z)}{\sigma_\ell(z)}
\right)
=\varepsilon,
\\[1mm]
\nabla r(\bar z)
=
\sum_\ell\bar\varpi_\ell
\left(
\nabla d_\ell-\bar\zeta_\ell\nabla\sigma_\ell
\right)_{\bar z},
\\[1mm]
r(\bar z)+\nabla r(\bar z)^\top(z-\bar z)+\delta\ge0.
\end{gathered}
}
$$

**“기존 MF feasible set을 유지하면서, SF와 일관된 distance-unit residual을 선형화한다”**가 이 방법의 핵심입니다.

논문에서는 이 정도로 설명하면 충분합니다. RIA/PMA 비유나 Mills-ratio 증명은 핵심 설명에 필요하지 않습니다. 다만 **one-row 선형화의 상쇄 문제는 남으므로, hard PSDF guard와 returned trajectory의 nonlinear constraint 검증은 별도로 유지해야 합니다.**
