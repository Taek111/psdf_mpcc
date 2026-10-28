# Inverse-CDF MF Residual

## 1. Residual definition

기존 feature violation probabilities의 합을

$$
R_i(z_i)\coloneqq
\sum_{\ell=1}^{n_F}m_\ell(x_i)p_{\ell,i}(x_i,\xi_i),
\qquad
p_{\ell,i}(x_i,\xi_i)
=\Phi\!\left(\frac{d_{\min}-d_\ell(x_i)}{\sigma_{\ell,i}(x_i,\xi_i)}\right),
$$

$$
\sigma_{\ell,i}(x_i,\xi_i)
=\left\|L(\xi_i)^\top q_\ell(x_i)\right\|
$$

로 정의한다. 확률합을 Gaussian quantile scale로 재표현하기 위해, $0<R_i(z_i)<1$에서 **inverse-CDF residual**을 정의한다.

$$
\boxed{
\psi_i^{\mathrm{MF}}(z_i)
\coloneqq
\Phi^{-1}(\varepsilon_i)-\Phi^{-1}\!\bigl(R_i(z_i)\bigr)
=-\Phi^{-1}\!\bigl(R_i(z_i)\bigr)-\gamma_i,
\qquad
\gamma_i=\Phi^{-1}(1-\varepsilon_i).
}
$$

$0<\varepsilon_i<1$에서 $\Phi^{-1}$의 단조성에 의해, 위 정의역에서

$$
\boxed{
\psi_i^{\mathrm{MF}}(z_i)\ge0
\iff
R_i(z_i)\le\varepsilon_i.
}
$$

따라서 이 변환은 정의역 내에서 기존 **unrelaxed Boole-based MF condition의 feasible set**을 보존한다. $R_i$는 feature violation probabilities의 합이지 실제 union probability 자체가 아니며, 변환은 기존 Gaussian feature model의 정확도나 Boole bound의 보수성을 바꾸지 않는다.

## 2. Analytic gradient and affine RTI row

Feature correspondence와 validity mask가 국소적으로 일정한 differentiable branch에서 일반 residual의 exact gradient는

$$
\nabla_{z_i}\psi_i^{\mathrm{MF}}(z_i)
=
-\frac{\nabla_{z_i}R_i(z_i)}
{\varphi_{\mathcal N}\!\left(\Phi^{-1}(R_i(z_i))\right)}.
$$

RTI에서는 기존 nominal-geometry approximation을 유지한다. Nominal $\bar z_i$에서 feature correspondence, projection case, feature-distance gradient, validity mask와 body-fixed frame을 고정하고

$$
\begin{aligned}
\hat d_{\ell,i}(x_i)
&=d_\ell(\bar x_i)
+q_\ell(\bar x_i)^\top T(\bar\theta_i)^\top(x_i-\bar x_i),\\
\hat\sigma_{\ell,i}(\xi_i)
&=\left\|L(\xi_i)^\top q_\ell(\bar x_i)\right\|,\\
\hat p_{\ell,i}(x_i,\xi_i)
&=\Phi\!\left(
\frac{d_{\min}-\hat d_{\ell,i}(x_i)}{\hat\sigma_{\ell,i}(\xi_i)}
\right),\\
\hat R_i(z_i;\bar z_i)
&=\sum_{\ell=1}^{n_F}
 m_\ell(\bar x_i)\hat p_{\ell,i}(x_i,\xi_i).
\end{aligned}
$$

여기서 $\hat R_i(\bar z_i;\bar z_i)=R_i(\bar z_i)$이며, $\bar R_i\coloneqq R_i(\bar z_i)$로 쓴다. 이 nominal-geometry model에 inverse-CDF 변환을 적용한 뒤 선형화하면

$$
\boxed{
\begin{aligned}
\hat h_i^{\mathrm{MF}}(z_i)
&=\psi_i^{\mathrm{MF}}(\bar z_i)
 +(A_i^{\mathrm{MF}})^\top(z_i-\bar z_i)\\
&=(A_i^{\mathrm{MF}})^\top z_i+c_i^{\mathrm{MF}},\\[1mm]
A_i^{\mathrm{MF}}
&=-\frac{
\displaystyle\sum_{\ell=1}^{n_F}m_\ell(\bar x_i)
\left.\nabla_{z_i}\hat p_{\ell,i}(x_i,\xi_i)\right|_{\bar z_i}
}{\varphi_{\mathcal N}\!\left(\Phi^{-1}(\bar R_i)\right)},\\
c_i^{\mathrm{MF}}
&=\psi_i^{\mathrm{MF}}(\bar z_i)
 -(A_i^{\mathrm{MF}})^\top\bar z_i.
\end{aligned}
}
$$

Standard normal density의 대칭성에 의해 분모는 $\varphi_{\mathcal N}(\psi_i^{\mathrm{MF}}(\bar z_i)+\gamma_i)$와 같다. 기존 feature-probability derivatives를 재사용할 수 있고, direct progress coefficient는 0이다.

**위 $A_i^{\mathrm{MF}}$는 nominal-geometry model의 analytic gradient이다.** 일반 함수의 projected standard deviation은 $q_\ell(x_i)$를 통해 pose에 의존하지만, nominal-geometry approximation은 그 dependence를 미분하지 않는다. 따라서 위 계수를 일반 residual의 exact gradient $\nabla_{z_i}\psi_i^{\mathrm{MF}}(\bar z_i)$와 무조건 동일시하지 않는다.

기존 hard PSDF guard를 유지하고 MF row는

$$
\hat h_i^{\mathrm{MF}}(z_i)+\delta_i\ge0,
\qquad \delta_i\ge0
$$

으로 적용한다. MF-specific root solve나 feature별 solver variable은 추가하지 않으며, stage당 하나의 affine MF row와 하나의 scalar slack을 유지한다.

## 3. Main properties and scope

### Single-feature reduction

유효 feature가 하나이면 CDF와 inverse CDF가 정확히 상쇄되어

$$
\psi_i^{\mathrm{MF}}(z_i)
=\frac{d_\ell(x_i)-d_{\min}}{\sigma_{\ell,i}(x_i,\xi_i)}-\gamma_i.
$$

Nominal-geometry model에서는 $d_\ell,\sigma_{\ell,i}$를 $\hat d_{\ell,i},\hat\sigma_{\ell,i}$로 대체한 식이 성립한다. 따라서 standard deviation이 고정되고 distance가 affine이면 residual 자체가 affine이다. Covariance가 variable이면 ratio의 비선형성은 남으며, standardized residual과 distance-unit SF margin의 RTI rows가 일반적으로 일치하지는 않는다.

### Comparison with the probability-residual row

동일한 nominal point와 동일한 nominal-geometry derivatives를 사용하고, $0<\bar R_i,\varepsilon_i<1/2$이면, **slack 없는 inverse-CDF affine row의 feasible half-space는 기존 probability-residual affine row의 feasible half-space에 포함된다.** 이는 $f(r)=-\Phi^{-1}(r)$가 $(0,1/2)$에서 convex이기 때문이다.

다만 두 row의 normal vector는 동일한 방향이다. 기존 probability-residual coefficient를 $A_{i,\mathrm{prob}}=-\nabla_{z_i}\hat R_i|_{\bar z_i}$로 쓰면

$$
A_i^{\mathrm{MF}}
=\frac{A_{i,\mathrm{prob}}}
{\varphi_{\mathcal N}(\Phi^{-1}(\bar R_i))}.
$$

특히 $\bar R_i=\varepsilon_i$에서는 두 unrelaxed affine rows가 양의 상수배 관계로 같은 half-space를 나타낸다. 따라서 inverse-CDF 변환은 feature 간 상대적인 gradient weighting이나 gradient cancellation을 제거하지 않으며, multi-feature residual의 curvature나 RTI error가 모든 방향에서 감소함을 보장하지 않는다.

### Domain and numerical evaluation

$R_i$는 probability sum이므로 1 이상이 될 수 있으며, 이때 원래 inverse-CDF residual은 finite real value로 정의되지 않는다. 이를 처리하기 위해 **$R_i\ge1/2$에서는 $R_i=1/2$에서 접하는 선형 branch를 이용한 smooth extension을 적용한다.** Stage index $i$를 생략하면 다음과 같다.

$$
\boxed{
\psi_{\mathrm{ext}}(z)
=
\begin{cases}
-\Phi^{-1}(R(z))-\gamma,
&0<R(z)\le \dfrac12,\\[2mm]
-\sqrt{2\pi}\left(R(z)-\dfrac12\right)-\gamma,
&R(z)>\dfrac12.
\end{cases}
}
\tag{4}
$$

두 branch는 $R=1/2$에서 함숫값 $-\gamma$와 $R$에 대한 1차 미분 $-\sqrt{2\pi}$가 일치한다. 따라서 $R>1/2$ 및 $R\ge1$에서도 residual과 gradient를 유한하게 평가할 수 있으며, $R>1/2$에서는 $\nabla_z\psi_{\mathrm{ext}}(z)=-\sqrt{2\pi}\nabla_zR(z)$를 사용한다. RTI row도 해당 branch의 residual 값과 nominal-geometry gradient로 구성한다.

이 확장은 $0<\varepsilon_i<1/2$에서 기존 unrelaxed MF condition의 feasible set을 보존한다. 앞의 single-feature reduction은 확장 residual에 대해서는 $0<R_i\le1/2$인 inverse-CDF branch에서 성립한다.

$R_i=0$에서는 이 확장으로도 finite residual과 gradient를 얻을 수 없으므로, empty feature set과 numerical underflow를 구분하여 별도로 처리해야 한다. 추가적인 clipping을 적용한다면, 그 미분도 clipping된 함수와 일치하도록 처리해야 한다.
