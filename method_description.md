# Method Description

## Augmented PSDF Predictive Control with a Boole-Based Multi-Feature Chance Constraint

## 1. Method Scope and Final Architecture

This work develops a real-time predictive control architecture for planar polygonal robots navigating narrow and cluttered polygonal environments. The method combines four components. The inherited Polygonal Signed Distance Function (PSDF) provides the PSDF signed distance between convex polygons and the corresponding PSDF gradient. A reduced body-fixed covariance state predicts pose uncertainty that depends on the control. At each stage, an indexed family of nearby point–segment geometric features defines a Boole-based multi-feature chance constraint. A Model Predictive Contouring Control (MPCC) layer based on local tangents optimizes path progress independently of fixed traversal timing.

The **augmented PSDF** is the GPU module that combines the inherited PSDF computation, geometric-feature construction and its evaluation at each stage, feature violation probability evaluation, and analytic generation of the RTI linearization of the MF chance constraint. The name does not modify the mathematical definition of the PSDF signed distance. It identifies the extended batched tensor graph used by the final controller.

The environment and generic geometric functions carry no prediction-stage index. Stage dependence is represented by evaluation at $x_i$, whereas shifted-nominal stage values use barred symbols with subscript $i$.

The complete computational chain is

$$
\text{shifted nominal trajectory}
\rightarrow
\text{augmented PSDF}
\rightarrow
\{A_i^{\mathrm{MF}},c_i^{\mathrm{MF}}\}_{i=0}^{N}
\rightarrow
\text{SQP-RTI MPCC}
$$

The multi-feature probabilistic statement applies to the Gaussian feature distances induced by first-order affine approximations about the predicted mean. The affine RTI row is a separate implementation approximation constructed about the shifted nominal trajectory. Neither statement is a claim about the exact collision probability.

---

## 2. Problem Setting

The predicted mean robot pose is

$$
x_i=
\begin{bmatrix}
p_i^\top & \theta_i
\end{bmatrix}^{\!\top},
$$

and the physical dynamics are

$$
x_{i+1}=f_x^d(x_i,u_i),
\qquad
x_i\in\mathcal X,
\qquad
u_i\in\mathcal U.
$$

The robot footprint is a convex polygon $P_0(x_i)$. A nonconvex footprint is represented by a convex decomposition consistent with the implementation. The environment geometry uses indexed objects $P_\kappa$, each with edge set $\mathcal E_\kappa$. The source files do not establish whether one $P_\kappa$ is an obstacle polygon or a convex obstacle component after decomposition; this object type must be verified before it is named more specifically. The horizon-static environment geometry is

$$
\mathcal E
=
\bigsqcup_{\kappa=1}^{K}\mathcal E_\kappa.
$$

At each control cycle, the environment remains unchanged over the local prediction horizon, so $\mathcal E_i=\mathcal E$ for $i=0,\ldots,N$ and the stage index is otherwise omitted. The environment may be refreshed between control cycles. This convention does not fix feature correspondence: $\mathcal F(x_i)$ may vary with the predicted pose.

The method assumes planar motion, polygonal geometry, static obstacle geometry during one local horizon, Gaussian pose perturbations in the stated neighborhood, a positive-definite initial pose covariance, positive baseline process-noise intensities, and one SQP-RTI step per control cycle.

---

## 3. Base PSDF and Deterministic PSDF Control

### 3.1 PSDF Signed Distance

For the robot footprint $P_0(x)$ and $P_\kappa$, the polygonal signed distance is

$$
\operatorname{sd}\!\left(P_0(x),P_\kappa\right)
=
\operatorname{dist}\!\left(P_0(x),P_\kappa\right)
-
\operatorname{pen}\!\left(P_0(x),P_\kappa\right).
$$

The PSDF signed distance for the environment is

$$
\phi(x,\mathcal E)
=
\min_{\kappa=1,\ldots,K}
\operatorname{sd}\!\left(P_0(x),P_\kappa\right).
$$

Its sign convention is

$$
\phi>0 \text{ in separation},
\qquad
\phi=0 \text{ at contact},
\qquad
\phi<0 \text{ in penetration}.
$$

The associated PSDF gradient is

$$
g(x,\mathcal E)=\nabla_x\phi(x,\mathcal E).
$$

The inherited PSDF evaluates obstacle geometry in the body-fixed frame. It computes point–segment distance candidates in both directions, applies the Separating Axis Theorem to handle overlap, and reduces the resulting geometric quantities to $\phi$ and $g$. The runtime realization is described in Section 9.

### 3.2 Deterministic Baseline

The deterministic PSDF-based safety condition is

$$
\phi(x_i,\mathcal E)-d_{\min}\ge0.
$$

At the shifted nominal pose $\bar x_i$, SQP-RTI uses

$$
\bar\phi_i
=
\phi(\bar x_i,\mathcal E),
\qquad
\bar g_i
=
\left.
\nabla_x\phi(x,\mathcal E)
\right|_{x=\bar x_i},
$$

and the affine model

$$
\hat\phi_i(x_i)
=
\bar\phi_i
+
\bar g_i^\top(x_i-\bar x_i).
$$

This row defines the deterministic PSDF-MPC-Det and PSDF-MPCC-Det variants.

---

## 4. Geometric Feature Extension in the Augmented PSDF

### 4.1 Bidirectionally Pooled Geometric Features

At a generic robot pose $x$, the augmented PSDF constructs two indexed feature families from its bidirectional point–segment candidates. For each fixed index $\kappa$ and robot vertex, the robot-to-obstacle family $\mathcal F^{\mathrm{R\to O}}(x)$ minimizes over the obstacle edges of $P_\kappa$. For each fixed index $\kappa$ and robot edge, the obstacle-to-robot family $\mathcal F^{\mathrm{O\to R}}(x)$ minimizes over the obstacle vertices of $P_\kappa$.

The pose-dependent indexed feature family at $x$ is the disjoint union of the two families,

$$
\boxed{
\mathcal F(x)
=
\mathcal F^{\mathrm{R\to O}}(x)
\sqcup
\mathcal F^{\mathrm{O\to R}}(x)
}.
$$

The disjoint union preserves the labels of entries from both families. Each feature index carries its family label and a validity mask $m_\ell(x)\in\{0,1\}$. The following geometric definitions apply to valid entries.

For each labeled feature $\ell\in\mathcal F(x)$, the selected ordered closest-point pair is

$$
\ell
\equiv
\left(
c_\ell^{\mathrm R}(x),
c_\ell^{\mathrm O}(x)
\right),
$$

where $c_\ell^{\mathrm R}(x)$ and $c_\ell^{\mathrm O}(x)$ denote the robot-side and obstacle-side points, respectively. For a valid feature in a separated configuration, its feature distance and unit separation direction are

$$
d_\ell(x)
=
\left\|
c_\ell^{\mathrm R}(x)
-
c_\ell^{\mathrm O}(x)
\right\|,
$$

$$
n_\ell(x)
=
\frac{
c_\ell^{\mathrm R}(x)
-
c_\ell^{\mathrm O}(x)
}{
d_\ell(x)
},
\qquad
d_\ell(x)>0.
$$

### 4.2 Feature-Distance Gradient

Let an infinitesimal body-fixed pose perturbation around $x$ be

$$
\delta\tilde x
=
\begin{bmatrix}
\delta p_f & \delta p_l & \delta\theta
\end{bmatrix}^{\!\top},
\qquad
J
=
\begin{bmatrix}
0&-1\\
1&0
\end{bmatrix}.
$$

For either feature direction, the first-order change in the feature distance is determined by the rigid-body motion of the robot-side closest point,

$$
\delta d_\ell(x)
\approx
n_\ell(x)^{\top}
\left(
\delta\tilde p
+
\delta\theta Jc_\ell^{\mathrm R}(x)
\right)
=
q_\ell(x)^{\top}\delta\tilde x.
$$

The feature-distance gradient is therefore

$$
\boxed{
q_\ell(x)
=
\nabla_{\tilde x}d_\ell(x)
=
\begin{bmatrix}
(n_\ell(x))_f\\
(n_\ell(x))_l\\
n_\ell(x)^{\top}Jc_\ell^{\mathrm R}(x)
\end{bmatrix}
}.
$$

For a robot-to-obstacle feature, $c_\ell^{\mathrm R}(x)$ is a robot vertex. For an obstacle-to-robot feature, it is the point on a robot edge obtained by projecting the selected obstacle vertex onto that edge. The same gradient expression therefore applies to both feature families.

The feature-distance derivative is defined in a neighborhood where the feature identity, the selected robot vertex and obstacle edge or obstacle vertex and robot edge, the point-to-segment projection case (hereafter, projection case), and the closest-point correspondence remain fixed during differentiation. Changes in feature correspondence and the projection case are not represented by this derivative. The geometric correspondences are evaluated again before the next QP subproblem, as described in Section 9.

---

## 5. Single-Feature Chance-Constraint Baseline

Let the random robot pose be

$$
X_i\sim\mathcal N(x_i,\Sigma_{x,i}),
$$

and define the random PSDF signed distance

$$
S_i=\phi(X_i,\mathcal E).
$$

The chance target at stage $i$ is

$$
\mathbb P(S_i\ge d_{\min})\ge1-\varepsilon_i.
$$

With $g_i\coloneqq g(x_i,\mathcal E)=\left.\nabla_x\phi(x,\mathcal E)\right|_{x=x_i}$, a first-order PSDF model gives

$$
S_i
\approx
\phi(x_i,\mathcal E)
+
g_i^\top(X_i-x_i),
$$

and therefore

$$
S_i
\approx
\mathcal N\!\left(
\phi(x_i,\mathcal E),
g_i^\top\Sigma_{x,i}g_i
\right).
$$

With

$$
\gamma_i=\Phi^{-1}(1-\varepsilon_i),
$$

the deterministic condition obtained from the affine approximation is

$$
h_i^{\mathrm{SF}}
=
\phi(x_i,\mathcal E)
-
\gamma_i
\sqrt{g_i^\top\Sigma_{x,i}g_i}
-
d_{\min}
\ge0.
$$

This SF baseline uses only the point–segment pair that attains the PSDF minimum at $x_i$ and the corresponding PSDF gradient. Its violation event concerns the affine PSDF signed-distance model, whereas the MF formulation in Section 7 concerns the union of several feature-distance violation events. Both formulations project pose covariance through first-order geometric gradients, but neither is treated as a special case of the other.

---

## 6. Body-Fixed Covariance with Control Dependence

### 6.1 Reduced Covariance State

The reduced covariance state is

$$
\eta_i
=
\begin{bmatrix}
\tilde\Sigma_{ff,i}&
\tilde\Sigma_{ll,i}&
\tilde\Sigma_{\theta\theta,i}&
\tilde\Sigma_{l\theta,i}
\end{bmatrix}^{\!\top}.
$$

It reconstructs the body-fixed pose covariance

$$
\tilde\Sigma_{x,i}(\eta_i)
=
\begin{bmatrix}
\tilde\Sigma_{ff,i}&0&0\\
0&\tilde\Sigma_{ll,i}&\tilde\Sigma_{l\theta,i}\\
0&\tilde\Sigma_{l\theta,i}&\tilde\Sigma_{\theta\theta,i}
\end{bmatrix}.
$$

Define the pose-frame transformation

$$
T(\theta)
=
\begin{bmatrix}
C(\theta)&0\\
0&1
\end{bmatrix},
$$

where $C(\theta)$ is the planar rotation matrix.

The world-frame and body-fixed covariance matrices satisfy

$$
\Sigma_{x,i}
=
T(\theta_i)
\tilde\Sigma_{x,i}(\eta_i)
T(\theta_i)^\top.
$$

For the SF baseline,

$$
\tilde g_i
=
T(\theta_i)^\top g_i,
\qquad
g_i^\top\Sigma_{x,i}g_i
=
\tilde g_i^\top
\tilde\Sigma_{x,i}(\eta_i)
\tilde g_i.
$$

The same predicted body-fixed covariance is projected through the feature-distance gradients evaluated at each stage in Section 7.

### 6.2 Covariance Propagation

Body-fixed perturbations follow

$$
\delta\tilde x_{i+1}
=
\tilde A_i(x_i,u_i)\delta\tilde x_i+w_i,
\qquad
w_i\sim
\mathcal N\!\left(
0,
\tilde W_i(x_i,u_i)
\right).
$$

For the physical input $u_i=[v_i,\omega_i]^\top$, the body-fixed process-noise covariance is modeled as

$$
\begin{aligned}
\tilde W_i(x_i,u_i)
&=
\operatorname{diag}\!\left(
q^{w}_{f,i},
q^{w}_{l,i},
q^{w}_{\theta,i}
\right),
\\
q^{w}_{f,i}
&=
q^{w}_{f,0}
+
\alpha_f v_i^2,
\\
q^{w}_{l,i}
&=
q^{w}_{l,0}
+
\alpha_{l,v}v_i^2
+
\alpha_{l,\omega}|v_i\omega_i|,
\\
q^{w}_{\theta,i}
&=
q^{w}_{\theta,0}
+
\alpha_{\theta,v}v_i^2
+
\alpha_{\theta,\omega}\omega_i^2.
\end{aligned}
$$

The constant terms represent baseline process uncertainty and satisfy

$$
q^{w}_{f,0}>0,
\qquad
q^{w}_{l,0}>0,
\qquad
q^{w}_{\theta,0}>0.
$$

The control-dependent coefficients are nonnegative. Translational motion increases the forward, lateral, and heading components through the $v_i^2$ terms. The coupled term $|v_i\omega_i|$ models additional lateral uncertainty during simultaneous translation and turning, while the $\omega_i^2$ term models turning-dependent heading uncertainty.

The full covariance update is

$$
\tilde\Sigma_{x,i+1}^{\mathrm{full}}
=
\tilde A_i(x_i,u_i)
\tilde\Sigma_{x,i}(\eta_i)
\tilde A_i(x_i,u_i)^\top
+
\tilde W_i(x_i,u_i).
$$

The reduced covariance state is initialized from the pose covariance supplied by the state estimator so that

$$
\tilde\Sigma_{x,0}(\eta_0)\succ0.
$$

Define

$$
\underline w
=
\min\left\{
q^{w}_{f,0},
q^{w}_{l,0},
q^{w}_{\theta,0}
\right\}
>0.
$$

Because the control-dependent process-noise terms are nonnegative, the full covariance update satisfies

$$
\tilde\Sigma_{x,i+1}^{\mathrm{full}}
\succeq
\tilde W_i(x_i,u_i)
\succeq
\underline w I_3.
$$

The reduced update discards only the forward--lateral and forward--heading cross-covariance entries. Splitting an arbitrary vector into its forward component and its lateral--heading component shows that its quadratic form under the reduced matrix is the sum of the corresponding quadratic forms of two principal blocks of $\tilde\Sigma_{x,i+1}^{\mathrm{full}}$. The same lower bound therefore survives the reduction. Consequently, every predicted covariance satisfies

$$
\boxed{
\tilde\Sigma_{x,i}(\eta_i)
\succeq
\underline\lambda I_3,
\qquad
\underline\lambda
=
\min\left\{
\lambda_{\min}\!\left(\tilde\Sigma_{x,0}(\eta_0)\right),
\underline w
\right\}
>0
}.
$$

The reduced dynamics extract the four modeled entries,

$$
\eta_{i+1}
=
F_\Sigma(\eta_i,x_i,u_i).
$$

This design-level model makes $F_\Sigma$ depend on the planned physical control through both the perturbation dynamics and the directional process-noise terms. The virtual progress input $v_{s,i}$ does not enter the physical covariance dynamics and therefore does not directly alter the predicted pose covariance.


---

## 7. Boole-Based Multi-Feature Chance Constraint

### 7.1 First-Order Approximation of Random Feature Distances

At stage $i$, the indexed feature family is $\mathcal F(x_i)$, obtained by evaluating the feature construction at the predicted mean pose $x_i$. The random pose perturbation expressed in the body-fixed frame of $x_i$ is

$$
\delta\tilde x_i
=
T(\theta_i)^{\top}(X_i-x_i),
\qquad
\delta\tilde x_i
\sim
\mathcal N\!\left(
0,
\tilde\Sigma_{x,i}(\eta_i)
\right).
$$

For each valid feature $\ell\in\mathcal F(x_i)$, the nonlinear random feature distance is $d_\ell(X_i)$. Using the feature correspondence selected at $x_i$, define its first-order affine approximation about $x_i$ by

$$
D_{\ell,i}
\coloneqq
d_\ell(x_i)
+
q_\ell(x_i)^{\top}\delta\tilde x_i,
$$

so that

$$
d_\ell(X_i)
\approx
D_{\ell,i}.
$$

This approximation assumes that the feature identity; the selected robot vertex and obstacle edge or obstacle vertex and robot edge; the projection case; and the closest-point correspondence remain unchanged over the perturbation neighborhood. Because the pose perturbation is Gaussian, the affine distance has the induced distribution

$$
D_{\ell,i}
\sim
\mathcal N\!\left(
\mu_{\ell,i},
\sigma_{\ell,i}^{2}
\right),
$$

where

$$
\boxed{
\mu_{\ell,i}
=
d_\ell(x_i)
},
$$

and

$$
\boxed{
\sigma_{\ell,i}^{2}
=
q_\ell(x_i)^{\top}
\tilde\Sigma_{x,i}(\eta_i)
q_\ell(x_i)
}.
$$

**Lemma 1.** For every valid feature $\ell$ at stage $i$, $\sigma_{\ell,i}^2\ge\underline\lambda>0$.

*Proof.* The translational block of $q_\ell(x_i)$ is the unit separation direction $n_\ell(x_i)$, and therefore

$$
\left\|q_\ell(x_i)\right\|^2
=
1
+
\left(
n_\ell(x_i)^\top Jc_\ell^{\mathrm R}(x_i)
\right)^2
\ge1.
$$

Combining this inequality with the uniform covariance bound gives

$$
\sigma_{\ell,i}^2
=
q_\ell(x_i)^\top
\tilde\Sigma_{x,i}(\eta_i)
q_\ell(x_i)
\ge
\underline\lambda
\left\|q_\ell(x_i)\right\|^2
\ge
\underline\lambda.
\qquad\square
$$

No numerical variance or additional uncertainty term is added to $\sigma_{\ell,i}^{2}$. Lemma 1 ensures that the feature violation probability is well defined for every valid feature without an imposed variance floor.

The feature violation probability is

$$
p_{\ell,i}(x_i,\eta_i)
=
\mathbb P\!\left(D_{\ell,i}<d_{\min}\right)
=
\Phi\!\left(
\frac{
d_{\min}-\mu_{\ell,i}
}{
\sigma_{\ell,i}
}
\right).
$$

No separate proximity weight is used. The mean feature distance and projected covariance enter directly through the standardized distance.

### 7.2 Multi-Feature Condition

The multi-feature chance target directly requires all valid Gaussian variables induced by the affine feature-distance models to satisfy the threshold,

$$
\boxed{
\mathbb P\!\left(
\bigcap_{\ell\in\mathcal F(x_i)}
\left\{
D_{\ell,i}\ge d_{\min}
\right\}
\right)
\ge
1-\varepsilon_i
}.
$$

By taking the complementary event, this target is equivalent to

$$
\boxed{
\mathbb P\!\left(
\bigcup_{\ell\in\mathcal F(x_i)}
\left\{
D_{\ell,i}<d_{\min}
\right\}
\right)
\le
\varepsilon_i
}.
$$

Boole's inequality gives

$$
\mathbb P\!\left(
\bigcup_{\ell\in\mathcal F(x_i)}
\left\{
D_{\ell,i}<d_{\min}
\right\}
\right)
\le
\sum_{\ell\in\mathcal F(x_i)}
\mathbb P\!\left(D_{\ell,i}<d_{\min}\right)
=
\sum_{\ell=1}^{L}
m_\ell(x_i)p_{\ell,i}(x_i,\eta_i).
$$

The Boole-based sufficient condition is therefore

$$
\boxed{
\sum_{\ell=1}^{L}
m_\ell(x_i)p_{\ell,i}(x_i,\eta_i)
\le
\varepsilon_i
}.
$$

The corresponding general nonlinear residual is

$$
\boxed{
h_i^{\mathrm{MF}}(x_i,\eta_i)
=
\varepsilon_i
-
\sum_{\ell=1}^{L}
m_\ell(x_i)p_{\ell,i}(x_i,\eta_i)
},
$$

and the stage condition is

$$
h_i^{\mathrm{MF}}(x_i,\eta_i)\ge0.
$$

This condition does not require independence among the Gaussian variables $D_{\ell,i}$. The shared pose perturbation generally makes them correlated. Because the disjoint union preserves labels from the robot-to-obstacle and obstacle-to-robot families, identical or overlapping violation events may contribute more than one summand to the Boole bound. This repeated counting can increase conservatism without invalidating the sufficient condition. No omission probability term is added. The probabilistic statement is defined only for the indexed feature family $\mathcal F(x_i)$ at stage $i$.

The indexed feature family is associated with the predicted mean pose, and each affine distance model assumes that the feature correspondence and projection case remain unchanged in the stated neighborhood. Feature switching, correspondence changes, and feature-distance curvature are outside this first-order model. The condition is therefore not a direct probability certificate for the exact nonlinear PSDF. Section 8 introduces a separate approximation about the shifted nominal trajectory and its affine RTI row.

### 7.3 Motion Regulation through Control Dependence

When the mean feature distance exceeds the safety threshold, $\mu_{\ell,i}>d_{\min}$, increasing $\sigma_{\ell,i}$ increases the corresponding feature violation probability. The planned physical control therefore affects the MF condition through the covariance dynamics,

$$
\text{more aggressive physical motion}
\rightarrow
\text{larger predicted covariance}
\rightarrow
\text{larger feature violation probabilities}
\rightarrow
\text{tighter MF condition}
\rightarrow
\text{slower or smoother optimized motion}.
$$

No separate slowdown heuristic is introduced.

---

## 8. MF Chance-Constraint Linearization about the Shifted Nominal Trajectory

For the RTI construction, define the augmented state and its shifted nominal value as

$$
z_i
=
\begin{bmatrix}
x_i^{\top}&s_i&\eta_i^{\top}
\end{bmatrix}^{\!\top},
\qquad
\bar z_i
=
\begin{bmatrix}
\bar x_i^{\top}&\bar s_i&\bar\eta_i^{\top}
\end{bmatrix}^{\!\top}.
$$

The geometric data evaluated at the shifted nominal pose for stage $i$ are

$$
\bar{\mathcal F}_i
\coloneqq
\mathcal F(\bar x_i),
$$

$$
\bar d_{\ell,i}
=
d_\ell(\bar x_i),
\qquad
\bar q_{\ell,i}
=
q_\ell(\bar x_i).
$$

The nominal validity mask is

$$
\bar m_{\ell,i}
\coloneqq
m_\ell(\bar x_i).
$$

The shifted covariance trajectory is propagated by $F_\Sigma$ along the shifted state and input trajectories. Lemma 1 therefore applies at $\bar\eta_i$, so the nominal feature-distance variances used below are strictly positive.

The mean-pose increment expressed in the nominal body-fixed frame is

$$
\Delta\tilde x_i
=
T(\bar\theta_i)^{\top}(x_i-\bar x_i).
$$

Around the shifted nominal trajectory, the generic feature quantities from Section 7 are approximated by

$$
\boxed{
d_\ell(x_i)
\approx
\bar d_{\ell,i}
+
\bar q_{\ell,i}^{\top}\Delta\tilde x_i
},
$$

$$
\boxed{
q_\ell(x_i)
\approx
\bar q_{\ell,i}
}.
$$

The approximation $q_\ell(x_i)\approx\bar q_{\ell,i}$ does not introduce a separate linearization of the gradient. It states that the feature-distance gradient evaluated at $\bar x_i$ is treated as a fixed parameter for stage $i$ in the current QP subproblem. The covariance in the following affine model is represented in the same nominal body-fixed frame defined by $\bar\theta_i$.

The mean, variance, and feature violation probability obtained from the affine approximation about $\bar x_i$ are

$$
\mu_{\ell,i}^{\mathrm{loc}}
=
\bar d_{\ell,i}
+
\bar q_{\ell,i}^{\top}\Delta\tilde x_i,
$$

$$
\left(\sigma_{\ell,i}^{\mathrm{loc}}\right)^2
=
\bar q_{\ell,i}^{\top}
\tilde\Sigma_{x,i}(\eta_i)
\bar q_{\ell,i},
$$

$$
p_{\ell,i}^{\mathrm{loc}}(z_i)
=
\Phi\!\left(
\frac{
d_{\min}-\mu_{\ell,i}^{\mathrm{loc}}
}{
\sigma_{\ell,i}^{\mathrm{loc}}
}
\right).
$$

The corresponding MF residual constructed from the quantities evaluated at $\bar x_i$ is

$$
\boxed{
h_i^{\mathrm{MF}}(\bar z_i)
=
\varepsilon_i
-
\sum_{\ell=1}^{L}
\bar m_{\ell,i}p_{\ell,i}^{\mathrm{loc}}(\bar z_i)
}.
$$

The corresponding nonlinear implementation condition is

$$
h_i^{\mathrm{MF}}(\bar z_i)
\ge0.
$$

This residual is the implementation model of the general condition $h_i^{\mathrm{MF}}(x_i,\eta_i)$ in Section 7, constructed from geometric quantities evaluated at $\bar x_i$. At $\bar z_i$, define

$$
\bar\sigma_{\ell,i}^{2}
=
\bar q_{\ell,i}^{\top}
\tilde\Sigma_{x,i}(\bar\eta_i)
\bar q_{\ell,i},
\qquad
\bar\zeta_{\ell,i}
=
\frac{d_{\min}-\bar d_{\ell,i}}
{\bar\sigma_{\ell,i}}.
$$

### 8.1 Pose Jacobian

Let $\varphi_{\mathcal N}$ denote the standard normal probability density function. The nominal body-fixed pose gradient is

$$
\nabla_{\Delta\tilde x_i}
h_i^{\mathrm{MF}}(\bar z_i)
=
\sum_{\ell=1}^{L}
\bar m_{\ell,i}
\frac{
\varphi_{\mathcal N}(\bar\zeta_{\ell,i})
}{
\bar\sigma_{\ell,i}
}
\bar q_{\ell,i}.
$$

The pose gradient in the world frame is

$$
\nabla_{x_i}
h_i^{\mathrm{MF}}(\bar z_i)
=
T(\bar\theta_i)
\nabla_{\Delta\tilde x_i}
h_i^{\mathrm{MF}}(\bar z_i).
$$

If $x_i$ contains components unrelated to pose, their coefficients are zero.

### 8.2 Covariance State Jacobian

For

$$
\bar q_{\ell,i}
=
\begin{bmatrix}
\bar q_{f,\ell,i}&
\bar q_{l,\ell,i}&
\bar q_{\theta,\ell,i}
\end{bmatrix}^{\!\top},
$$

define

$$
r(\bar q_{\ell,i})
=
\begin{bmatrix}
\bar q_{f,\ell,i}^{2}\\
\bar q_{l,\ell,i}^{2}\\
\bar q_{\theta,\ell,i}^{2}\\
2\bar q_{l,\ell,i}\bar q_{\theta,\ell,i}
\end{bmatrix}.
$$

The covariance state gradient is

$$
\nabla_{\eta_i}
h_i^{\mathrm{MF}}(\bar z_i)
=
\sum_{\ell=1}^{L}
\bar m_{\ell,i}
\varphi_{\mathcal N}(\bar\zeta_{\ell,i})
\frac{
\bar\zeta_{\ell,i}
}{
2\bar\sigma_{\ell,i}^{2}
}
r(\bar q_{\ell,i}).
$$

### 8.3 Affine QP Row

The augmented PSDF assembles the full gradient in the augmented-state order,

$$
\boxed{
A_i^{\mathrm{MF}}
=
\left.
\nabla_{z_i}
h_i^{\mathrm{MF}}(z_i)
\right|_{\bar z_i}
},
$$

with zero coefficients for progress and all states that do not directly enter the MF residual. The affine constant is

$$
c_i^{\mathrm{MF}}
=
h_i^{\mathrm{MF}}(\bar z_i)
-
(A_i^{\mathrm{MF}})^{\top}\bar z_i.
$$

The RTI linearization of the MF chance constraint is

$$
\boxed{
\hat h_i^{\mathrm{MF}}(z_i)
=
(A_i^{\mathrm{MF}})^{\top}z_i
+
c_i^{\mathrm{MF}}
\ge0
}.
$$

This affine row is a first-order approximation of the MF implementation model about $\bar z_i$. It is not an independent certificate for the general nonlinear MF condition or the exact nonlinear PSDF.

---

## 9. The Augmented PSDF Runtime Interface

The augmented PSDF receives the shifted nominal pose and covariance trajectories, the geometry and violation-probability bound for each stage, and a fixed-size feature configuration. Its single tensor graph evaluates the base PSDF and the robot-to-obstacle and obstacle-to-robot features over the prediction horizon before each QP subproblem.

At each stage, the module evaluates $\bar{\mathcal F}_i=\mathcal F(\bar x_i)$ together with the selected robot vertices, robot edges, obstacle vertices, obstacle edges, projection cases, $\bar d_{\ell,i}$, $\bar q_{\ell,i}$, family labels, and nominal validity masks $\bar m_{\ell,i}$. These geometric quantities are treated as fixed parameters for stage $i$ in the current QP subproblem. They are evaluated again from the updated shifted nominal trajectory before the next QP subproblem.

The PSDF and geometric-feature computations reuse the same bidirectional point-to-segment candidate tensors. For each robot vertex, the feature computation reduces along the tensor dimension indexing obstacle edges. For each robot edge, the reciprocal computation reduces along the tensor dimension indexing obstacle vertices. The arrays for the two families are concatenated directly. The implementation does not merge features that induce identical or overlapping violation events. The MF residual and its Jacobians from Section 8 are then evaluated with analytic batched formulas. Runtime autograd is not used to generate the MF Jacobian.

The per-stage output of the augmented PSDF is $\{\bar\phi_i,\bar g_i,A_i^{\mathrm{MF}},c_i^{\mathrm{MF}}\}$.

The feature correspondences, nominal feature distances, feature-distance gradients, masks, direction labels, and individual feature violation probabilities remain inside the GPU module. For the PSDF-MPCC-MF controller, the MF part of the acados stage parameter is

$$
p_i^{\mathrm{MF}}
=
\begin{bmatrix}
A_i^{\mathrm{MF}}\\
c_i^{\mathrm{MF}}
\end{bmatrix},
$$

which contains $n_z+1$ scalars. Its size does not depend on the number of geometric features, obstacles, obstacle vertices, or obstacle edges. The QP contains one MF safety row at each initial, intermediate, and terminal stage.

---

## 10. MPCC with Local Tangents

### 10.1 Progress Dynamics and Path Anchors

The progress state satisfies

$$
s_{i+1}
=
s_i+\Delta t\,v_{s,i},
$$

where $v_{s,i}$ is the virtual progress input. The path anchor at stage $i$ is

$$
\pi_i
=
\left(
p_i^{\mathrm{ref}},
t_i^{\mathrm{ref}},
n_i^{\mathrm{ref}},
s_i^{\mathrm{ref}}
\right),
$$

with

$$
p_i^{\mathrm{ref}}
=
p_r(s_i^{\mathrm{ref}}),
$$

$$
t_i^{\mathrm{ref}}
=
\frac{p_r'(s_i^{\mathrm{ref}})}
{\|p_r'(s_i^{\mathrm{ref}})\|},
\qquad
n_i^{\mathrm{ref}}
=
Jt_i^{\mathrm{ref}}.
$$

The local tangent approximation is

$$
\hat p_i(s_i)
=
p_i^{\mathrm{ref}}
+
(s_i-s_i^{\mathrm{ref}})t_i^{\mathrm{ref}}.
$$

### 10.2 Path Errors and Cost

The contouring, lag, and anchor deviation errors are

$$
e_{c,i}
=
(p_i-p_i^{\mathrm{ref}})^\top n_i^{\mathrm{ref}},
$$

$$
e_{l,i}
=
-(p_i-p_i^{\mathrm{ref}})^\top t_i^{\mathrm{ref}}
+
(s_i-s_i^{\mathrm{ref}})
\|t_i^{\mathrm{ref}}\|^2,
$$

$$
e_{s,i}
=
s_i-s_i^{\mathrm{ref}}.
$$

A representative stage cost is

$$
\ell_i
=
\lambda_c e_{c,i}^2
+
\lambda_l e_{l,i}^2
+
\lambda_s e_{s,i}^2
+
\|u_i\|_{R_u}^2
+
\rho_{v_s}v_{s,i}^2
-
\lambda_{\mathrm{prog}}v_{s,i}.
$$

The path anchors are fixed during one RTI solve. The path errors are therefore affine in $(x_i,s_i)$, and the path cost is quadratic. Optimized progress allows the controller to reduce physical speed when the MF condition tightens without incurring a penalty associated with fixed timing.

---

## 11. Unified PSDF-MPCC-MF Problem

The augmented state and input are

$$
z_i
=
\begin{bmatrix}
x_i^\top&s_i&\eta_i^\top
\end{bmatrix}^{\!\top},
\qquad
\nu_i
=
\begin{bmatrix}
u_i^\top&v_{s,i}
\end{bmatrix}^{\!\top}.
$$

The augmented dynamics are

$$
z_{i+1}
=
f_d(z_i,\nu_i)
=
\begin{bmatrix}
f_x^d(x_i,u_i)\\
s_i+\Delta t\,v_{s,i}\\
F_\Sigma(\eta_i,x_i,u_i)
\end{bmatrix}.
$$

Define

$$
e_i(z_i;\pi_i)
=
\begin{bmatrix}
e_{c,i}&e_{l,i}&e_{s,i}
\end{bmatrix}^{\!\top}.
$$

Section 7 defines the general probabilistic condition $h_i^{\mathrm{MF}}(x_i,\eta_i)\ge0$. Section 8 evaluates the MF implementation model constructed from geometric quantities at $\bar x_i$ as $h_i^{\mathrm{MF}}(\bar z_i)$ and constructs its affine RTI approximation $\hat h_i^{\mathrm{MF}}(z_i)$. Only the affine RTI row enters the implemented SQP-RTI subproblem.

The implemented SQP-RTI subproblem uses

$$
\begin{aligned}
\min_{\{z_i,\nu_i\}}
\quad &
\sum_{i=0}^{N-1}
\left(
\|e_i(z_i;\pi_i)\|_{\Lambda}^{2}
+
\|\nu_i\|_{R_\nu}^{2}
-
\lambda_{\mathrm{prog}}v_{s,i}
\right)
+
\|e_N(z_N;\pi_N)\|_{\Lambda_N}^{2}
\\
\text{s.t.}\quad
&z_0=\hat z_k,
\\
&z_{i+1}=f_d(z_i,\nu_i),
\qquad i=0,\ldots,N-1,
\\
&\hat h_i^{\mathrm{MF}}(z_i)
=
(A_i^{\mathrm{MF}})^\top z_i+c_i^{\mathrm{MF}}
\ge0,
\qquad i=0,\ldots,N,
\\
&x_i\in\mathcal X,
\qquad
\nu_i\in\mathcal U_\nu.
\end{aligned}
$$

Because $\eta_0$ is fixed by the initial-state constraint and the covariance dynamics enter as equality constraints, the predicted covariance states are determined by the planned physical inputs and require no separate admissibility constraint.

The final OCP jointly predicts the mean motion, path progress, and reduced covariance. Computations that depend on obstacles and features remain outside the inner QP.

---

## 12. Online Control Procedure

At each control cycle, the controller executes the following fixed sequence.

1. Shift the previous state, physical input, progress, and covariance trajectories to obtain $\{\bar z_i,\bar\nu_i\}_{i=0}^{N}$.
2. Refresh the local path anchors $\{\pi_i\}_{i=0}^{N}$ from the shifted progress trajectory.
3. Execute the augmented PSDF over the horizon. For each stage, evaluate $\bar{\mathcal F}_i=\mathcal F(\bar x_i)$, the nominal feature distances, feature-distance gradients, and nominal validity masks $\bar m_{\ell,i}$, and then construct the MF residual from those geometric quantities and generate $\{A_i^{\mathrm{MF}},c_i^{\mathrm{MF}}\}$.
4. Transfer only the MF affine coefficients and MPCC parameters to acados.
5. Update the horizon parameter vector and perform one SQP-RTI preparation and feedback step.
6. Apply the first physical control input and repeat at the next control cycle.

The analytic Jacobian used by the augmented PSDF is validated against finite differences and PyTorch autograd. This validation is performed offline and is not part of the online control path.

---

## 13. Controller Variants

| Controller        | Safety formulation                                                     | Behavior formulation     |
| ----------------- | ---------------------------------------------------------------------- | ------------------------ |
| **PSDF-MPC-Det**  | Deterministic affine PSDF signed distance row                          | Fixed-time tracking MPC  |
| **PSDF-MPCC-Det** | Deterministic affine PSDF signed distance row                          | MPCC with local tangents |
| **PSDF-MPCC-SF**  | Gaussian chance constraint using the PSDF gradient for the pair that attains the PSDF minimum | MPCC with local tangents |
| **PSDF-MPCC-MF**  | Boole-based multi-feature chance constraint with one RTI row per stage | MPCC with local tangents |

The trace norm or trace matrix MF formulation is not part of the proposed method. It may appear only as the explicitly labeled **MF-Trace surrogate** ablation and has no chance constraint interpretation.

---

## 14. Fixed Method Decisions

The final method applies the following decisions globally.

- The runtime module is named **augmented PSDF**.
- The final PSDF-MPCC-MF controller uses one aggregate MF safety row at each prediction stage.
- The final PSDF-MPCC-MF controller has no separate deterministic PSDF guard row.
- Each stage uses a bidirectionally pooled feature set consisting of robot-vertex-to-obstacle-edge and obstacle-vertex-to-robot-edge families.
- The arrays for the robot-to-obstacle and obstacle-to-robot feature families are concatenated without merging entries with equal geometry or entries that induce identical or overlapping violation events.
- At a nominal pose in a separated configuration, the point–segment pair that attains the PSDF minimum is represented in one of the two feature families and is not inserted as a separate reserved feature.
- The MF condition uses feature violation probabilities and Boole's inequality.
- Feature independence is not assumed.
- The violation-probability bound at stage $i$ is represented only by $\varepsilon_i$.
- No Gaussian margin scaling factor is used.
- No omission probability term is used.
- No numerical variance $\sigma_{\mathrm{num}}^2$ is added.
- No distance weights, risk matrix, trace expression, or trace norm is used in the proposed formulation.
- Raw feature quantities and Gaussian CDF evaluations for individual features remain outside acados.
- The inner QP contains no constraint rows for individual features.
- Feature correspondences, nominal feature distances, feature-distance gradients, and validity masks are evaluated along the shifted nominal trajectory and treated as fixed parameters for each stage in the current QP subproblem. These quantities are evaluated again from the updated trajectory before the next QP subproblem.


---

## 15. Probabilistic and Implementation Claim Boundaries

The general nonlinear condition

$$
h_i^{\mathrm{MF}}(x_i,\eta_i)\ge0
$$

is a Boole-based sufficient condition for the joint chance constraint on the Gaussian variables $D_{\ell,i}$ induced by the affine distance models for $\mathcal F(x_i)$. The feature correspondences are evaluated at the predicted mean pose $x_i$. The condition does not require feature independence.

The condition constructed from the shifted nominal trajectory,

$$
h_i^{\mathrm{MF}}(\bar z_i)
\ge0
$$

is the implementation model obtained by evaluating the feature correspondences, feature distances, and feature-distance gradients at each pose on the shifted nominal trajectory. These geometric quantities are treated as fixed parameters for the corresponding stage in the current QP subproblem. The model uses a first-order affine approximation of the general feature functions about $\bar x_i$.

The affine row

$$
(A_i^{\mathrm{MF}})^{\top}z_i+c_i^{\mathrm{MF}}\ge0
$$

is the first-order RTI linearization of the MF implementation condition about $\bar z_i$. It is not an exact or independently certified chance constraint away from the current linearization point.

The method does not claim a direct bound on the collision probability of the exact nonlinear PSDF, geometric correspondences outside $\mathcal F(x_i)$, or the collision probability over a full mission under repeated replanning.

The number of obstacle edges and geometric features affects augmented PSDF evaluation time. It does not increase the decision dimension of the inner OCP, the MF parameter dimension per stage, or the number of MF QP rows.

---

## 16. Main Technical Contributions

1. **Augmented PSDF.** The inherited PSDF computation is extended into a single GPU module that reuses shared geometry tensors to evaluate geometric features and feature violation probabilities at each stage and to generate analytic MF RTI coefficients.
2. **Control dependence of body-fixed covariance.** A reduced body-fixed covariance state is propagated inside the predictive controller so the planned physical motion determines future variances of feature distances.
3. **Boole-based multi-feature chance constraint.** The Gaussian pose perturbation and first-order affine distance approximations induce correlated Gaussian feature-distance variables. Their violation probabilities are summed to obtain one sufficient condition at each stage.
4. **Unified real-time PSDF-MPCC-MF.** Mean motion, progress, covariance, and multi-feature probabilistic safety are combined while the inner QP retains one aggregate MF row per stage.

---

## 17. Method in One Sentence

At each stage, the method evaluates an indexed family of point–segment geometric features along the shifted nominal trajectory and forms first-order affine approximations of their distances. The Gaussian pose perturbation and control-dependent body-fixed covariance induce the corresponding Gaussian distance distributions. Boole's inequality bounds the probability of their violation union, and the local-tangent PSDF-MPCC controller receives one affine RTI safety row per stage.
