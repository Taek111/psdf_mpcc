# Method Description

**PSDF-Based Predictive Control with a Boole-Based Multi-Feature Chance Constraint**

## 1. Overview, Problem Setting, and Notation

The method combines the inherited Polygonal Signed Distance Function (PSDF), control-dependent covariance, a Boole-based multi-feature (MF) chance constraint, and local-tangent Model Predictive Contouring Control (MPCC). PSDF denotes the inherited geometric function. The **extended PSDF computation** augments the GPU module to evaluate features, violation probabilities, and analytic MF coefficients while preserving that function and its signed-distance output, the PSDF value. A four-dimensional log-Cholesky state is propagated to a robust horizon and held constant afterward. Mean motion and progress continue over the full horizon, with one affine MF row and an independent hard PSDF-based safety guard per stage in the Sequential Quadratic Programming with Real-Time Iteration (SQP-RTI) controller.

### 1.1 System and Environment

At control cycle $k$, the predicted mean pose and physical input are

$$
x_i=[p_i^\top,\theta_i]^\top\in\mathbb R^3,
\qquad
u_i=[v_i,\omega_i]^\top\in\mathbb R^2,
\qquad
x_{i+1}=f_x^d(x_i,u_i),
$$

Here, $f_x^d$ is the discrete planar motion model, and $\mathcal X$ and $\mathcal U$ are the admissible pose and physical-input sets. There are $N$ transitions of sampling interval $\Delta t$, with states at prediction stages $i=0,\ldots,N$ and inputs at $i=0,\ldots,N-1$. The cycle index $k$ is omitted from prediction variables.

The footprint is a convex polygon $P_0(x)$. Nonconvex footprints use a consistent convex decomposition. At cycle $k$, define the environment boundary edge set by

$$
\mathcal E_k:=\bigsqcup_{i=1}^{M_k}\partial P_k^i.
$$

Here, $M_k$ is the number of environment polygons, $P_k^i$ is the $i$th polygon, and $\partial P_k^i$ is its boundary edge collection. The disjoint union preserves polygon identity, boundary connectivity, and orientation; it does not assume that the polygons are geometrically disjoint. The index $i$ in this definition numbers polygons only. Subsequently, polygon indices are written as $j=1,\ldots,M_k$ to distinguish them from prediction stages $i$. The polygon family $\{P_k^j\}_{j=1}^{M_k}$ and its boundary edge set $\mathcal E_k$ are different objects.

Geometry is fixed within each horizon and may be refreshed between cycles, while feature correspondences can vary with pose. For a fixed cycle, write $\mathcal E:=\mathcal E_k$ throughout the following equations. This abbreviation always denotes the same boundary edge set. Whether $P_k^j$ denotes an original obstacle polygon or a convex decomposition component remains an implementation-verification item.

### 1.2 Coordinate and Notation Conventions

The planar rotation matrix and the corresponding pose-frame transformation are

$$
\mathcal R(\theta)=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix},
\qquad
T(\theta)=\operatorname{diag}(\mathcal R(\theta),1),
\qquad
J=\begin{bmatrix}0&-1\\1&0\end{bmatrix}.
$$

Here, $\mathcal R(\theta)$ maps body-fixed translation coordinates to the world frame. For the local Gaussian planning model,

$$
\delta\tilde x_i=T(\theta_i)^\top(X_i-x_i),
\qquad
\delta\tilde x_i\sim\mathcal N(0,\tilde\Sigma_{x,i}),
\qquad
\Sigma_{x,i}=T(\theta_i)\tilde\Sigma_{x,i}T(\theta_i)^\top.
$$

The model assumes planar motion, a positive-definite initial covariance, positive baseline process noise, and one SQP-RTI step per cycle. Prediction includes no future measurement update.

| Symbol or convention | Meaning |
| --- | --- |
| PSDF | Inherited geometric function |
| $\phi(x,\mathcal E)$ | **PSDF value**: signed distance returned by the PSDF |
| $\nabla_x\phi(x,\mathcal E)$ | **PSDF gradient**: pose gradient of the PSDF, expressed in the world frame |
| $d_\ell(x)$ | **Feature distance**: unsigned Euclidean distance used in the MF formulation |
| PSDF-based safety guard | Hard constraint using the PSDF to maintain geometric separation; implemented through the affine approximation of the PSDF value in Section 5.4 |
| $X_i$, $x_i$, $\bar x_i$ | Random pose, predicted mean pose, and nominal pose used for RTI linearization |
| $\delta\tilde x_i$ | Random pose perturbation in the local body-fixed frame |
| $k$, $i$, $j$, $\mathcal E_k$ | Control cycle, prediction stage, subsequent polygon index, and environment boundary edge set |
| $f,l,\theta$ | Forward, lateral, and heading component labels |
| $\ell=1,\ldots,n_F$ | Feature index and feature-index capacity. $L$ is reserved for the Cholesky factor |
| $\xi_i$, $L(\xi_i)$, $\tilde\Sigma_{x,i}$ | Log-Cholesky optimization state, reconstructed factor, and reconstructed body-fixed covariance |
| $\bar\xi_i$ | Nominal log-Cholesky coordinates regenerated from the current initial covariance for RTI linearization |
| $\tilde A_i$, $\tilde W_i$ | Body-fixed perturbation transition matrix and process-noise covariance |
| $q_\ell(x)$, $q^w_{a,i}$ | Feature-distance gradient and process-noise intensity in direction $a\in\{f,l,\theta\}$ |
| $\varepsilon_i$ | Stage violation-probability bound, with $\varepsilon_i\in(0,1)$ |
| $d_{\mathrm{col}}$, $d_{\min}$ | Required geometric clearance for the hard PSDF-based safety guard and clearance threshold in the chance formulation, respectively |
| $\delta_i$, $\mathcal P_{\mathrm{MF}}$ | Nonnegative stage-local MF slack and its objective penalty; neither changes the state or input definition |
| $\Phi$, $\varphi_{\mathcal N}$ | Standard normal cumulative distribution function and probability density function |

Generic geometric functions have no stage index. Bars denote nominal states, inputs, or explicitly defined nominal evaluations; feature quantities are evaluated directly as $d_\ell(\bar x_i)$, $q_\ell(\bar x_i)$, and $m_\ell(\bar x_i)$. Hats on $\hat d_{\ell,i}$, $\hat\sigma_{\ell,i}$, and $\hat p_{\ell,i}$ identify nominal-geometry approximations, which need not be affine. The solver row $\hat h_i^{\mathrm{MF}}$ and local path model $\hat p_i^{\mathrm{ref}}$ are affine in their stated arguments. Hats on $\hat x_k$ and $\hat\Sigma_{x,k}$ denote current estimates. Mathematical objects use the names feature violation probability and violation-probability bound; general titles and discussion may use risk or risk-aware without renaming those objects.

## 2. PSDF Geometry and Feature Construction

### 2.1 PSDF Definition and Sign Convention

The signed distance returned by the inherited geometric function is the **PSDF value**, and its derivative with respect to pose is the **PSDF gradient**. In world-frame coordinates, they are

$$
\phi(x,\mathcal E)
=\min_{j=1,\ldots,M_k}\left[
\operatorname{dist}(P_0(x),P_k^j)
-\operatorname{pen}(P_0(x),P_k^j)
\right],
\qquad
g(x,\mathcal E)=\nabla_x\phi(x,\mathcal E).
$$

The PSDF includes sign-determination logic and returns a signed distance that is positive in separation, zero at contact, and negative in penetration. The inherited PSDF evaluates bidirectional point–segment candidates in the body-fixed frame and handles overlap using the Separating Axis Theorem. Polygon membership and boundary topology are retained for this signed overlap handling; the edge-set notation does not replace the PSDF with an unsigned edge distance. [Citation needed]

### 2.2 Bidirectional Features and Their Gradients

For each polygon $P_k^j$ and robot vertex, the robot-to-obstacle family minimizes over edges in $\partial P_k^j$. For each polygon $P_k^j$ and robot edge, the obstacle-to-robot family minimizes over that polygon's boundary vertices. Their indexed union is

$$
\mathcal F(x)=\mathcal F^{\mathrm{R\to O}}(x)
\sqcup\mathcal F^{\mathrm{O\to R}}(x).
$$

The union preserves labels without deduplication. Masks $m_\ell(x)\in\{0,1\}$ identify valid entries among $\ell=1,\ldots,n_F$. The PSDF-minimizing pair is already represented in separated configurations and is not inserted again.

For a valid separated feature, the body-fixed closest points $c_\ell^{\mathrm R}(x)$ and $c_\ell^{\mathrm O}(x)$ lie on the robot and obstacle, respectively. Define the **feature distance** $d_\ell(x)$ as their unsigned Euclidean distance used in the MF formulation, and $n_\ell(x)$ as the unit separation direction:

$$
d_\ell(x)=\|c_\ell^{\mathrm R}(x)-c_\ell^{\mathrm O}(x)\|,
\qquad
n_\ell(x)=\frac{c_\ell^{\mathrm R}(x)-c_\ell^{\mathrm O}(x)}{d_\ell(x)},
\qquad d_\ell(x)>0.
$$

With $\delta\tilde x=[\delta p_f,\delta p_l,\delta\theta]^\top$ and $\delta\tilde p=[\delta p_f,\delta p_l]^\top$, rigid-body motion gives

$$
\delta d_\ell
\approx n_\ell(x)^\top
\left(\delta\tilde p+\delta\theta Jc_\ell^{\mathrm R}(x)\right)
=q_\ell(x)^\top\delta\tilde x,
\qquad
q_\ell(x)=
\begin{bmatrix}
(n_\ell(x))_f\\
(n_\ell(x))_l\\
n_\ell(x)^\top Jc_\ell^{\mathrm R}(x)
\end{bmatrix}.
$$

The robot-side point is a vertex or a projected point on a robot edge, so the expression applies to both families. Here, $\nabla_{\tilde x}$ denotes differentiation with respect to the local body-fixed pose perturbation used above, and $q_\ell(x)=\nabla_{\tilde x}d_\ell(x)$ on the selected differentiable branch. Differentiation holds the feature correspondence and point-to-segment projection case fixed. Unsigned Euclidean feature distances do not replace the PSDF value at contact or in penetration. In particular, when $\phi(x,\mathcal E)<0$, these unsigned distances do not encode penetration and their gradients need not point toward depenetration. The PSDF-based safety guard in Section 5.4 therefore remains independent of the MF condition.

## 3. Control-Dependent Covariance Prediction

### 3.1 Reduced Covariance via Log-Cholesky Parameterization

The reduced covariance retains three variances and the lateral–heading cross-covariance. A local heading perturbation changes the direction of translation and therefore contributes to lateral position uncertainty during motion. Retaining this coupling while limiting the covariance state to four coordinates is a modeling and computational trade-off. It does not establish that forward–lateral or forward–heading correlations vanish or remain small, and the reduced covariance is not automatically a conservative upper bound on the full covariance. Its independent optimization state and factor are[^log-cholesky]

$$
\xi_i=
\begin{bmatrix}
\xi_{f,i}&\xi_{l,i}&\xi_{\theta,i}&\xi_{c,i}
\end{bmatrix}^{\!\top}\in\mathbb R^4,
\qquad
L(\xi_i)=
\begin{bmatrix}
e^{\xi_{f,i}}&0&0\\
0&e^{\xi_{l,i}}&0\\
0&\xi_{c,i}&e^{\xi_{\theta,i}}
\end{bmatrix}.
$$

The first three coordinates are log-diagonal factor entries, and the fourth is the signed off-diagonal entry. Reconstruction gives

$$
\boxed{
\tilde\Sigma_{x,i}=L(\xi_i)L(\xi_i)^\top
=
\begin{bmatrix}
e^{2\xi_{f,i}}&0&0\\
0&e^{2\xi_{l,i}}&\xi_{c,i}e^{\xi_{l,i}}\\
0&\xi_{c,i}e^{\xi_{l,i}}&\xi_{c,i}^2+e^{2\xi_{\theta,i}}
\end{bmatrix}.
}
$$

Write $\tilde\Sigma_{ab,i}$ for entry $(a,b)$ of $\tilde\Sigma_{x,i}$. These entries are derived, not independent states. In particular, $\xi_{c,i}$ is not a covariance entry, and $\xi_{\theta,i}$ is not the log of the marginal heading standard deviation.

Define the covariance-to-coordinate map $\mathcal C:\mathbb S_{++}^3\rightarrow\mathbb R^4$, where $\mathbb S_{++}^3$ is the set of symmetric positive-definite $3\times3$ matrices. For a body-fixed covariance $\tilde\Sigma_x$, the vector $\xi=\mathcal C(\tilde\Sigma_x)$ is defined by the retained-block reconstruction

$$
L(\xi)L(\xi)^\top=
\begin{bmatrix}
\tilde\Sigma_{ff}&0&0\\
0&\tilde\Sigma_{ll}&\tilde\Sigma_{l\theta}\\
0&\tilde\Sigma_{l\theta}&\tilde\Sigma_{\theta\theta}
\end{bmatrix}.
$$

The positive diagonal of $L(\xi)$ makes these coordinates unique. The following component formulas give the explicit inverse on the retained blocks:

$$
\xi_i=\mathcal C(\tilde\Sigma_{x,i})=
\begin{bmatrix}
\tfrac12\log\tilde\Sigma_{ff,i}\\[1mm]
\tfrac12\log\tilde\Sigma_{ll,i}\\[1mm]
\tfrac12\log\!\left(
\tilde\Sigma_{\theta\theta,i}
-\frac{\tilde\Sigma_{l\theta,i}^2}{\tilde\Sigma_{ll,i}}
\right)\\[1mm]
\tilde\Sigma_{l\theta,i}/\sqrt{\tilde\Sigma_{ll,i}}
\end{bmatrix}.
$$

These formulas invert the reconstruction on the retained forward and lateral–heading blocks. The same map $\mathcal C$ accepts a full positive-definite body-fixed covariance and discards its forward–lateral and forward–heading cross-covariances. Reconstruction then sets those entries to zero. This block reduction is distinct from the world-to-body coordinate transformation and is not an eigenvalue-based PSD repair.

At every control cycle $k$, use the current estimator covariance $\hat\Sigma_{x,k}\succ0$ to initialize

$$
\xi_0=\mathcal C\!\left(T(\hat\theta_k)^\top\hat\Sigma_{x,k}T(\hat\theta_k)\right).
$$

This initial condition is refreshed each cycle rather than set only once at controller startup.

For finite $\xi_i$, the factor is invertible in exact arithmetic, so $\tilde\Sigma_{x,i}\succ0$ at any optimizer iterate. This guarantee applies to factor reconstruction, not an affine approximation of covariance entries. Nonlinear dynamics satisfaction is discussed with the solver in Section 5.4.

### 3.2 Directional Process Noise and Covariance Propagation

The local open-loop perturbation model is

$$
\delta\tilde x_{i+1}=\tilde A_i(x_i,u_i)\delta\tilde x_i+w_i,
\qquad
w_i\sim\mathcal N(0,\tilde W_i),
$$

where $\tilde A_i$ is the first-order body-fixed perturbation transition matrix. The directional process-noise model is

$$
\begin{aligned}
\tilde W_i
&=\operatorname{diag}(q^w_{f,i},q^w_{l,i},q^w_{\theta,i}),\\
q^w_{f,i}
&=q^w_{f,\mathrm{base}}+\alpha_f v_i^2,\\
q^w_{l,i}
&=q^w_{l,\mathrm{base}}+\alpha_{l,v}v_i^2
+\alpha_{l,\omega}|v_i\omega_i|,\\
q^w_{\theta,i}
&=q^w_{\theta,\mathrm{base}}+\alpha_{\theta,v}v_i^2
+\alpha_{\theta,\omega}\omega_i^2.
\end{aligned}
$$

Baseline intensities are positive and all $\alpha$ coefficients are nonnegative. The subscript $\mathrm{base}$ denotes a constant parameter, not stage zero. The model describes design-level uncertainty associated with translation, turning, and their coupling.

The unreduced one-step covariance prediction is

$$
\boxed{
\tilde\Sigma_{x,i+1}^{\mathrm{full}}
=\tilde A_i\tilde\Sigma_{x,i}\tilde A_i^\top+\tilde W_i.
}
$$

Within the propagation interval, the next log-Cholesky state is $\mathcal C(\tilde\Sigma_{x,i+1}^{\mathrm{full}})$ using Section 3.1. Reconstruction and propagation therefore use the same four coordinates without an intermediate covariance-entry state.

Since $\tilde W_i\succ0$, the predicted matrix and retained blocks are positive definite, making the logarithms and square roots in the coordinate formulas well defined.

### 3.3 Robust Horizon and Rollout Properties

Following Bradford and Imsland,[^robust-horizon] choose $N_R\in\{1,\ldots,N\}$ and $T_R=N_R\Delta t$. Covariance propagates to stage $N_R$ and is held constant afterward. Using the prediction from Section 3.2, the controller transition is

$$
\boxed{
\xi_{i+1}=F_{\xi,i}(\xi_i,x_i,u_i)=
\begin{cases}
\begin{bmatrix}
\tfrac12\log\tilde\Sigma_{ff,i+1}^{\mathrm{full}}\\[1mm]
\tfrac12\log\tilde\Sigma_{ll,i+1}^{\mathrm{full}}\\[1mm]
\tfrac12\log\!\left(
\tilde\Sigma_{\theta\theta,i+1}^{\mathrm{full}}
-\dfrac{(\tilde\Sigma_{l\theta,i+1}^{\mathrm{full}})^2}
{\tilde\Sigma_{ll,i+1}^{\mathrm{full}}}
\right)\\[1mm]
\dfrac{\tilde\Sigma_{l\theta,i+1}^{\mathrm{full}}}
{\sqrt{\tilde\Sigma_{ll,i+1}^{\mathrm{full}}}}
\end{bmatrix},&0\le i<N_R,\\[4mm]
\xi_i,&N_R\le i<N.
\end{cases}
}
$$

Thus $\xi_i=\xi_{N_R}$ and $\tilde\Sigma_{x,i}=\tilde\Sigma_{x,N_R}$ for $i\ge N_R$. Setting $N_R=N$ recovers full-horizon propagation. The fixed schedule adds no optimization states or constraint rows.

Freezing is body-fixed. The world-frame covariance $T(\theta_i)\tilde\Sigma_{x,N_R}T(\theta_i)^\top$ can rotate, and feature gradients can change with pose. Projected variances and violation probabilities are therefore not fixed.

Finite-precision evaluation still requires scaling and finite-value checks.

Beyond $N_R$, covariance is a planning approximation, not the continued open-loop covariance. Freezing introduces neither fictitious measurements nor feedback gains and does not establish a closed-loop covariance bound.

## 4. Chance-Constraint Formulation

### 4.1 Deterministic and Single-Feature Baselines

The deterministic safety condition is $\phi(x_i,\mathcal E)-d_{\min}\ge0$. Under the Gaussian planning model, the single-feature (SF) baseline approximates the target $\mathbb P(\phi(X_i,\mathcal E)\ge d_{\min})\ge1-\varepsilon_i$ by the first-order affine condition

$$
\mathbb P\!\left(
\phi(x_i,\mathcal E)+g_i^\top(X_i-x_i)\ge d_{\min}
\right)\ge1-\varepsilon_i,
\qquad g_i=g(x_i,\mathcal E).
$$

The affine expression is Gaussian with mean $\phi(x_i,\mathcal E)$ and variance $g_i^\top\Sigma_{x,i}g_i$. Its probability condition is equivalent to

$$
h_i^{\mathrm{SF}}
=\phi(x_i,\mathcal E)-d_{\min}
-\Phi^{-1}(1-\varepsilon_i)
\|L(\xi_i)^\top T(\theta_i)^\top g_i\|\ge0.
$$

SF uses the PSDF value and PSDF gradient on the branch selected by the PSDF-minimizing pair. In a separated configuration on a locally differentiable branch, retaining only that feature in MF and using the same affine Gaussian distance model and planning covariance makes the single-feature probability condition equivalent to this SF Gaussian margin. A multi-feature union event is generally different from the SF single event. This probability-level equivalence does not imply that separate RTI linearizations of different residual functions yield identical QP rows. 

### 4.2 Multi-Feature Gaussian Distance Model

For each valid feature selected at the predicted mean pose $x_i$, replace the nonlinear random feature distance by its first-order affine approximation,

$$
d_\ell(X_i)\approx D_{\ell,i}
\coloneqq d_\ell(x_i)+q_\ell(x_i)^\top\delta\tilde x_i.
$$

The Gaussian pose perturbation induces

$$
D_{\ell,i}\sim\mathcal N(d_\ell(x_i),\sigma_{\ell,i}^2),
\qquad
\sigma_{\ell,i}^2
=q_\ell(x_i)^\top\tilde\Sigma_{x,i}q_\ell(x_i)
=\|L(\xi_i)^\top q_\ell(x_i)\|^2.
$$

Writing $q_\ell(x_i)=[q_{f,\ell,i},q_{l,\ell,i},q_{\theta,\ell,i}]^\top$ gives the direct evaluation

$$
\sigma_{\ell,i}^2
=e^{2\xi_{f,i}}q_{f,\ell,i}^2
+\left(e^{\xi_{l,i}}q_{l,\ell,i}+\xi_{c,i}q_{\theta,\ell,i}\right)^2
+e^{2\xi_{\theta,i}}q_{\theta,\ell,i}^2.
$$

Since $q_\ell$ has a unit translational block, variance is positive at every finite $\xi_i$. No numerical variance is added. The violation probability is

$$
p_{\ell,i}(x_i,\xi_i)
=\mathbb P(D_{\ell,i}<d_{\min})
=\Phi\!\left(\frac{d_{\min}-d_\ell(x_i)}{\sigma_{\ell,i}}\right).
$$

### 4.3 Boole-Based MF Condition and Control Dependence

The stage target requires joint satisfaction of all valid affine feature-distance constraints,

$$
\mathbb P\!\left(
\bigcap_{\ell\in\mathcal F(x_i)}\{D_{\ell,i}\ge d_{\min}\}
\right)\ge1-\varepsilon_i.
$$

Boole's inequality bounds the complementary violation event,

$$
\mathbb P\!\left(
\bigcup_{\ell\in\mathcal F(x_i)}\{D_{\ell,i}<d_{\min}\}
\right)
\le\sum_{\ell=1}^{n_F}m_\ell(x_i)p_{\ell,i}(x_i,\xi_i).
$$

The sufficient scalar condition is therefore

$$
\boxed{
h_i^{\mathrm{MF}}(x_i,\xi_i)
=\varepsilon_i-
\sum_{\ell=1}^{n_F}m_\ell(x_i)p_{\ell,i}(x_i,\xi_i)\ge0.
}
$$

Independence is unnecessary. Shared perturbations correlate the distances, while duplicate or overlapping events can increase conservatism. No proximity weights, feature-count normalization, Gaussian margin multiplier, omission term, or trace surrogate is used.

For $d_\ell(x_i)>d_{\min}$, larger $\sigma_{\ell,i}$ increases violation probability. Inputs before $N_R$ affect covariance through $\tilde A_i$ and $\tilde W_i$. Later, $\partial F_{\xi,i}/\partial u_i=0$, but earlier inputs still affect the frozen covariance and later inputs still affect feature geometry. Virtual progress does not directly affect covariance. This coupling can favor slower or smoother motion without guaranteeing slowdown or adding a separate heuristic.

The unrelaxed condition $h_i^{\mathrm{MF}}\ge0$ is the probabilistic reference. Section 5.4 implements its affine RTI approximation with a penalized nonnegative slack while enforcing an independent hard PSDF-based safety guard; satisfaction of the relaxed row does not by itself establish the original bound $\varepsilon_i$.

## 5. Local-Tangent MPCC and SQP-RTI Implementation

### 5.1 Local-Tangent Path Model

The progress state satisfies

$$
s_{i+1}=s_i+\Delta t\,v_{s,i},
$$

where $v_{s,i}$ is the virtual progress input. For the reference path $p_r(s)$, the stage anchor is

$$
\begin{gathered}
\pi_i=(p_i^{\mathrm{ref}},t_i^{\mathrm{ref}},n_i^{\mathrm{ref}},s_i^{\mathrm{ref}}),\\
p_i^{\mathrm{ref}}=p_r(s_i^{\mathrm{ref}}),
\qquad
t_i^{\mathrm{ref}}=
\frac{p_r'(s_i^{\mathrm{ref}})}{\|p_r'(s_i^{\mathrm{ref}})\|},
\qquad
n_i^{\mathrm{ref}}=Jt_i^{\mathrm{ref}}.
\end{gathered}
$$

The local path approximation is

$$
\hat p_i^{\mathrm{ref}}(s_i)=p_i^{\mathrm{ref}}
+(s_i-s_i^{\mathrm{ref}})t_i^{\mathrm{ref}}.
$$

Using the unit tangent, define the contouring, lag, and anchor-deviation errors,

$$
\begin{aligned}
e_{\mathrm{con},i}&=(p_i-p_i^{\mathrm{ref}})^\top n_i^{\mathrm{ref}},\\
e_{\mathrm{lag},i}&=-(p_i-p_i^{\mathrm{ref}})^\top t_i^{\mathrm{ref}}
+(s_i-s_i^{\mathrm{ref}}),\\
e_{s,i}&=s_i-s_i^{\mathrm{ref}},
\qquad e_i=[e_{\mathrm{con},i},e_{\mathrm{lag},i},e_{s,i}]^\top.
\end{aligned}
$$

Fixed anchors make the errors affine in position and progress. Their quadratic penalties regulate path fidelity, progress consistency, and anchor deviation without fixed traversal timing. The penalty $\lambda_s(s_i-s_i^{\mathrm{ref}})^2$ is a soft regularization against large anchor deviations, not a hard trust region or an accuracy guarantee. Curvature, spacing between stage anchors, and mismatch between actual motion and the nominal trajectory limit this local-tangent approximation. Section 5.2 specifies anchor selection and refresh.

### 5.2 Augmented Dynamics and Nominal Evaluation

The augmented state and input are

$$
z_i=[x_i^\top,s_i,\xi_i^\top]^\top\in\mathbb R^8,
\qquad
\nu_i=[u_i^\top,v_{s,i}]^\top\in\mathbb R^3.
$$

The stage-dependent dynamics combine physical motion, progress, and the covariance transition from Section 3.3,

$$
z_{i+1}=f_{d,i}(z_i,\nu_i)=
\begin{bmatrix}
f_x^d(x_i,u_i)\\
s_i+\Delta t\,v_{s,i}\\
F_{\xi,i}(\xi_i,x_i,u_i)
\end{bmatrix}.
$$

At every cycle $k$, prepare nominal pose and physical-input trajectories $\{\bar x_i\}$ and $\{\bar u_i\}$ from the previous optimization, with the initial pose reset to the current estimate. Separately, shift the previous solution's progress trajectory to form $\{\bar s_i\}_{i=0}^{N}$ and prepare the nominal virtual inputs. This progress shift is used for path anchors; it does not shift the covariance trajectory. Initialize $\xi_0$ from the current covariance as in Section 3.1, and sequentially regenerate the nominal covariance coordinates:

$$
\bar\xi_0=\xi_0,
\qquad
\bar\xi_{i+1}=F_{\xi,i}(\bar\xi_i,\bar x_i,\bar u_i),
\qquad i=0,\ldots,N-1.
$$

For $0\le i<N_R$, this evaluates $\tilde A_i$ and $\tilde W_i$ at the current nominal pose/input and applies $\mathcal C$ to the resulting one-step covariance prediction. For $N_R\le i<N$, it uses $\bar\xi_{i+1}=\bar\xi_i$. Thus the propagated covariance reflects the current initial covariance and preceding nominal inputs. The previous cycle's covariance trajectory is not shifted to initialize this nominal rollout. Only after regeneration are $\bar z_i=[\bar x_i^\top,\bar s_i,\bar\xi_i^\top]^\top$ assembled for dynamics linearization and MF coefficient evaluation. Here, nominal denotes the linearization reference; the OCP coordinates $\xi_i$ remain variable and coupled to inputs through covariance dynamics.

Before cost and QP preparation, set

$$
s_i^{\mathrm{ref}}=\bar s_i,
\qquad i=0,\ldots,N,
$$

and evaluate the global spline outside the solver to construct each complete tuple $\pi_i$ from Section 5.1, including the terminal tuple. Hold all tuple components fixed throughout this cycle's single RTI step and its QP. Changes to the optimization variables $s_i$ and $v_{s,i}$ do not trigger spline reevaluation or force $s_i=s_i^{\mathrm{ref}}$ across the horizon. At the next cycle, shift the new solution's progress and refresh the tuples. [The earlier thesis draft, Section 4.2.3](draft_ref/draft_thesis/thesis_draft_4-2-3.md), establishes this selection rule; initialization without a previous solution, terminal progress extension, alignment between $\hat s_k$ and $\bar s_0$, and path-boundary or unusable-solution handling remain implementation-verification items.

At each nominal pose, the extended PSDF computation evaluates $\mathcal F(\bar x_i)$, $d_\ell(\bar x_i)$, $q_\ell(\bar x_i)$, $m_\ell(\bar x_i)$, $\phi(\bar x_i,\mathcal E)$, and $\bar g_i=\nabla_x\phi(\bar x_i,\mathcal E)$. Correspondences, projection cases, gradients, masks, and the nominal body frame are fixed QP parameters and refreshed at the next control cycle. Geometry is evaluated over the full horizon, including the covariance tail.

### 5.3 Nominal-Geometry MF Approximation and Analytic Linearization

The nominal-geometry approximation uses

$$
\begin{aligned}
\hat d_{\ell,i}(x_i)
&=d_\ell(\bar x_i)+q_\ell(\bar x_i)^\top T(\bar\theta_i)^\top(x_i-\bar x_i),\\
\hat\sigma_{\ell,i}(\xi_i)
&=\|L(\xi_i)^\top q_\ell(\bar x_i)\|,\\
\hat p_{\ell,i}(x_i,\xi_i)
&=\Phi\!\left(\frac{d_{\min}-\hat d_{\ell,i}(x_i)}{\hat\sigma_{\ell,i}(\xi_i)}\right).
\end{aligned}
$$

The feature-distance approximation is affine in pose, whereas the standard deviation and feature probability remain nonlinear in $\xi_i$. This approximation does not differentiate pose dependence of nominal gradients or frame rotation. At the nominal point, it reproduces the general feature probabilities and residual:

$$
h_i^{\mathrm{MF}}(\bar x_i,\bar\xi_i)
=\varepsilon_i-\sum_{\ell=1}^{n_F}m_\ell(\bar x_i)
\hat p_{\ell,i}(\bar x_i,\bar\xi_i).
$$

The world-frame pose derivative of the MF probability residual is

$$
\left.\nabla_{x_i}\!\left[\varepsilon_i-\sum_{\ell=1}^{n_F}m_\ell(\bar x_i)\hat p_{\ell,i}(x_i,\xi_i)\right]\right|_{\bar z_i}
=T(\bar\theta_i)\sum_{\ell=1}^{n_F}m_\ell(\bar x_i)
\frac{\varphi_{\mathcal N}\!\left(\dfrac{d_{\min}-d_\ell(\bar x_i)}{\hat\sigma_{\ell,i}(\bar\xi_i)}\right)}{\hat\sigma_{\ell,i}(\bar\xi_i)}
q_\ell(\bar x_i).
$$

The local variance is differentiated directly with respect to the same log-Cholesky state used in Section 3,

$$
\nabla_{\xi_i}\hat\sigma_{\ell,i}(\xi_i)^2
=
\begin{bmatrix}
2e^{2\xi_{f,i}}[q_\ell(\bar x_i)]_f^2\\
2e^{\xi_{l,i}}[q_\ell(\bar x_i)]_l
\left(e^{\xi_{l,i}}[q_\ell(\bar x_i)]_l+\xi_{c,i}[q_\ell(\bar x_i)]_\theta\right)\\
2e^{2\xi_{\theta,i}}[q_\ell(\bar x_i)]_\theta^2\\
2[q_\ell(\bar x_i)]_\theta
\left(e^{\xi_{l,i}}[q_\ell(\bar x_i)]_l+\xi_{c,i}[q_\ell(\bar x_i)]_\theta\right)
\end{bmatrix}.
$$

Thus,

$$
\left.\nabla_{\xi_i}\!\left[\varepsilon_i-\sum_{\ell=1}^{n_F}m_\ell(\bar x_i)\hat p_{\ell,i}(x_i,\xi_i)\right]\right|_{\bar z_i}
=\sum_{\ell=1}^{n_F}m_\ell(\bar x_i)
\varphi_{\mathcal N}\!\left(\frac{d_{\min}-d_\ell(\bar x_i)}{\hat\sigma_{\ell,i}(\bar\xi_i)}\right)
\frac{d_{\min}-d_\ell(\bar x_i)}{2\hat\sigma_{\ell,i}(\bar\xi_i)^3}
\left.\nabla_{\xi_i}\hat\sigma_{\ell,i}(\xi_i)^2\right|_{\bar\xi_i}.
$$

The extended PSDF computation directly assembles

$$
A_i^{\mathrm{MF}}
=\left.\nabla_{z_i}\!\left[\varepsilon_i-\sum_{\ell=1}^{n_F}m_\ell(\bar x_i)\hat p_{\ell,i}(x_i,\xi_i)\right]\right|_{\bar z_i}
\in\mathbb R^{n_z},
\qquad
c_i^{\mathrm{MF}}
=h_i^{\mathrm{MF}}(\bar x_i,\bar\xi_i)-(A_i^{\mathrm{MF}})^\top\bar z_i,
\qquad n_z=8,
$$

with a zero coefficient for progress. The affine MF residual is defined by

$$
\boxed{
\hat h_i^{\mathrm{MF}}(z_i)
=(A_i^{\mathrm{MF}})^\top z_i+c_i^{\mathrm{MF}}.
}
$$

The general condition $h_i^{\mathrm{MF}}$, nominal-geometry probability approximation $\hat p_{\ell,i}$, and affine row $\hat h_i^{\mathrm{MF}}$ are distinct. The robust horizon changes covariance prediction, not these stage-local derivatives.

### 5.4 OCP and Solver Interface

Let $\Lambda=\operatorname{diag}(\lambda_{\mathrm{con}},\lambda_{\mathrm{lag}},\lambda_s)$ weight contouring, lag, and anchor-deviation errors, let $\Lambda_N$ be the terminal path-error weight, and let $R_\nu=\operatorname{diag}(R_u,\rho_{v_s})$ weight physical and virtual inputs. Define the stage cost by

$$
\mathcal L_i=\|e_i\|_\Lambda^2+\|\nu_i\|_{R_\nu}^2-\lambda_{\mathrm{prog}}v_{s,i}.
$$

At each control cycle, two complementary obstacle-avoidance conditions are linearized about the same nominal pose trajectory: a PSDF-based safety guard for geometric separation and the MF condition for uncertainty-aware avoidance. The PSDF-based safety guard is a hard constraint using an affine approximation of the PSDF value, formed with the nominal PSDF gradient:

$$
\begin{aligned}
\bar g_i&=\nabla_x\phi(\bar x_i,\mathcal E),\\
\hat\phi_i(x_i)&=\phi(\bar x_i,\mathcal E)
+\bar g_i^\top(x_i-\bar x_i),\\
\hat\phi_i(x_i)-d_{\mathrm{col}}&\ge0,\qquad i=0,\ldots,N.
\end{aligned}
$$

Here $d_{\mathrm{col}}$ is the required geometric clearance, distinct in role from the chance-formulation threshold $d_{\min}$. Their numerical values and any equality between them must be specified from the implementation. The guard depends only on pose, admits no slack relaxation, and remains active independently of the MF row at every stage, including the initial and terminal stages.

The MF row is enforced as a soft constraint, $\hat h_i^{\mathrm{MF}}(z_i)+\delta_i\ge0$, with one scalar slack $\delta_i\ge0$ per stage. Let $\mathcal P_{\mathrm{MF}}(\delta_i)$ denote a nonnegative, increasing, QP-compatible slack penalty with $\mathcal P_{\mathrm{MF}}(0)=0$, applied at all stages including $i=N$. The supplied modification does not specify the penalty form or weights; these must match the implemented solver. The OCP is

$$
\begin{aligned}
\min_{\{z_i,\delta_i\}_{i=0}^{N},\{\nu_i\}_{i=0}^{N-1}}\quad&
\sum_{i=0}^{N-1}\mathcal L_i+\|e_N\|_{\Lambda_N}^2
+\sum_{i=0}^{N}\mathcal P_{\mathrm{MF}}(\delta_i)\\
\mathrm{s.t.}\quad
&z_0=\hat z_k,\\
&z_{i+1}=f_{d,i}(z_i,\nu_i),&&i=0,\ldots,N-1,\\
&\hat\phi_i(x_i)-d_{\mathrm{col}}\ge0,&&i=0,\ldots,N,\\
&\hat h_i^{\mathrm{MF}}(z_i)+\delta_i\ge0,\quad\delta_i\ge0,&&i=0,\ldots,N,\\
&x_i\in\mathcal X,&&i=0,\ldots,N,\\
&\nu_i\in\mathcal U_\nu,&&i=0,\ldots,N-1.
\end{aligned}
$$

The initial state is $\hat z_k=[\hat x_k^\top,\hat s_k,\xi_0^\top]^\top$, with current progress initialization $\hat s_k$. The sets $\mathcal X$ and $\mathcal U_\nu$ specify admissible poses and augmented inputs. Anchors, PSDF-based safety guard data, and MF coefficients are fixed during the RTI step. The objective balances path errors, inputs, and MF relaxation against progress. The slacks are stage-local auxiliary variables, not components of the eight-dimensional state $z_i$ or three-dimensional input $\nu_i$.

Positive slack permits violation of the unrelaxed affine MF row; it does not relax the PSDF-based safety guard and is not a clearance-distance slack. Zero slack recovers the unrelaxed affine row, but this alone does not certify the general MF probability condition because the row is an RTI linearization. Hard enforcement refers to the affine PSDF-based safety guard without slack, not an exact nonlinear separation guarantee; the latter would require additional justification or nonlinear validation. In particular, MF relaxation cannot restore feasibility when the fixed initial pose violates the hard guard.

SQP-RTI linearizes the dynamics about $(\bar z_i,\bar\nu_i)$ and need not satisfy the nonlinear recursion exactly in one step.[^rti] Returned covariances are reconstructed from $L(\xi_i)L(\xi_i)^\top$. The four coordinates replace the four covariance-entry states without adding a PSD constraint.

The GPU evaluates $\{\phi(\bar x_i,\mathcal E),\bar g_i,A_i^{\mathrm{MF}},c_i^{\mathrm{MF}}\}$ per stage. Only $(A_i^{\mathrm{MF}},c_i^{\mathrm{MF}})$, containing $n_z+1$ scalars, is transferred for the MF row. The pose-only guard additionally uses the PSDF value and three-component PSDF gradient; its data are not included in the MF transfer count. The slack coefficient is the fixed scalar $+1$ and requires no feature-dependent derivative or GPU parameter. Features, masks, Gaussian evaluations, and analytic Jacobian assembly remain on the GPU. The feature arrays are concatenated without deduplication, and masking precedes standardization to avoid undefined padded values. MF Jacobians do not use runtime automatic differentiation.

The deterministic and SF baseline residuals reuse the affine approximation of the PSDF value defined above. The deterministic baseline retains its threshold $d_{\min}$ and enforces

$$
\hat h_i^{\mathrm{Det}}(x_i)=\hat\phi_i(x_i)-d_{\min}\ge0.
$$

For SF, let $\gamma_i=\Phi^{-1}(1-\varepsilon_i)$ and use the fixed-gradient, nominal-frame Gaussian-margin residual

$$
\begin{aligned}
\hat\sigma_i^{\mathrm{SF}}(\xi_i)
&=\|L(\xi_i)^\top T(\bar\theta_i)^\top\bar g_i\|,\\
r_i^{\mathrm{SF}}(z_i)
&=\hat\phi_i(x_i)-\gamma_i\hat\sigma_i^{\mathrm{SF}}(\xi_i)-d_{\min}.
\end{aligned}
$$

The PSDF gradient and body frame are fixed while covariance coordinates remain variables. The QP uses

$$
\hat h_i^{\mathrm{SF}}(z_i)
=r_i^{\mathrm{SF}}(\bar z_i)
+\nabla_{z_i}r_i^{\mathrm{SF}}(\bar z_i)^\top(z_i-\bar z_i)\ge0.
$$

These expressions retain the deterministic and SF baseline residuals. The independent hard PSDF-based safety guard remains active in CC-MF; guard and relaxation policies for the other variants must be stated separately rather than inferred from the MF modification. This Gaussian-margin choice follows the formulation in [the earlier Sections 5.2–5.3 draft](draft_prev/5.2-5.3.md); agreement with the actual controller code remains to be verified. Its probability-level relationship to MF is stated in Section 4.1 and does not establish equality of their affine QP rows. [Citation needed for inherited PSDF implementation]

### 5.5 Online Control Procedure

At each control cycle, the controller performs the following operations.

1. Receive the current pose and covariance. Transform the covariance to the body frame and apply $\mathcal C$ to initialize $\xi_0$. Prepare nominal pose/input trajectories and shift the previous progress solution. Regenerate $\bar\xi_i$ from this current initial covariance with $F_{\xi,i}$, including the frozen tail; do not shift the previous covariance trajectory.
2. Set $s_i^{\mathrm{ref}}=\bar s_i$ for $i=0,\ldots,N$ and evaluate the global spline outside the solver to refresh every component of $\pi_i$. Use the extended PSDF computation at all stages to assemble the PSDF-based safety guard data together with the MF probabilities, analytic derivatives, and affine coefficients at the regenerated nominal state.
3. Transfer the PSDF-based safety guard data, MF coefficients, path parameters, and fixed robust-horizon schedule to the solver. In the implemented CC-MF family, stage 0 safety rows are inactive; its exact signed PSDF instead triggers `safe_stop_stage0` before solving when below `d_col`. Future stages $1,\ldots,N$ retain the hard guard. The soft MF row uses its configured stage window, terminal switch, and strict nominal domain gate $\bar\phi_i>d_{\mathrm{mf\_mask}}$ (by default stages $1,\ldots,N-1$ and terminal $N$). Linearize the dynamics and perform one SQP-RTI preparation and feedback step, holding the complete anchor tuples and nominal geometry fixed.
4. Apply the first physical input. Reconstruct returned covariances from their log-Cholesky states and monitor nonlinear dynamics defects separately. Log MF slack activation and magnitude alongside unrelaxed MF residuals and PSDF values. Reinitialize and regenerate the nominal covariance at the next cycle.

Offline validation should compare analytic feature and MF derivatives with finite differences and automatic differentiation. These checks are separate from the online computation; the document revisions do not establish that code-level validation has been completed.

### 5.6 Optional CC-MF-IMP: Implicit Clearance Margin

`cc_mf_imp` (input alias `cc-mf-imp`) retains the CC-MF feature model, eight-dimensional state, covariance dynamics, hard guard, and stage policy. It replaces the probability residual by a common clearance-threshold margin $r(z)$ defined outside the OCP:

$$
F(z,r)=\sum_{\ell\in\bar{\mathcal F}}
\Phi\!\left(\frac{d_{\min}+r-\widehat d_\ell(x_p)}{\sigma_\ell(\xi)}\right),
\qquad F(z,r(z))=\varepsilon.
$$

With at least one valid feature and positive feature standard deviations, $F$ is strictly increasing in $r$. Thus $F(z,0)\le\varepsilon$ is equivalent to $r(z)\ge0$ for this same frozen-geometry Gaussian feature model before slack relaxation. The margin is measured in meters of common threshold shift; it is not a robot-motion radius or an exact nonlinear collision certificate.

At the nominal root $\bar r$, define $\zeta_\ell=(\bar d_\ell-d_{\min}-\bar r)/\bar\sigma_\ell$ and normalized weights $w_\ell\propto\varphi_{\mathcal N}(\zeta_\ell)/\bar\sigma_\ell$. Implicit differentiation gives

$$
\nabla_z r(\bar z)=\sum_\ell w_\ell
\left(\nabla_z\widehat d_\ell-\zeta_\ell\nabla_z\sigma_\ell\right),
\qquad
A_{\mathrm{imp}}=\nabla_z r(\bar z),\quad
c_{\mathrm{imp}}=\bar r-A_{\mathrm{imp}}^\top\bar z.
$$

The solver receives the same nine MF parameters and enforces $A_{\mathrm{imp}}^\top z+c_{\mathrm{imp}}+\delta\ge0$, with only Row M softened. The root adds no OCP state or feature-specific decision variable. Its derivative is evaluated at $\bar r$, including the Log-Cholesky covariance derivatives; normalizing the old probability gradient at $r=0$ would not give this row. A single feature yields $r=d-d_{\min}-\Phi^{-1}(1-\varepsilon)\sigma$, matching the corresponding CC-SF residual under the same feature/covariance model. Opposing feature gradients can still cancel, and satisfying the affine row does not certify its nonlinear counterpart.

The implementation uses stage-batched PyTorch eager FP64 risk calculations over the existing FP32 geometry. For $b_\ell=\bar d_\ell-d_{\min}$ and $M=|\bar{\mathcal F}|$, initialize upper-bound Newton at $r_{\mathrm{hi}}=\min_\ell\{b_\ell+\bar\sigma_\ell\Phi^{-1}(\varepsilon)\}$, with lower bracket $r_{\mathrm{lo}}=\min_\ell\{b_\ell+\bar\sigma_\ell\Phi^{-1}(\varepsilon/M)\}$. This numerical method supports $0<\varepsilon<0.5$; $M=1$ uses the closed form. Acceptance requires both the probability residual tolerance and a signed local bracket around the final candidate. No bisection retry, sigma floor, probability clipping, CUDA Graph, or `torch.compile` is used.

Required-stage root or coefficient failures raise `MFImplicitRootError`, preserve failure diagnostics, and stop the current run and batch before another control cycle or trial. They do not disable the row or invoke guard-only recovery. Ordinary QP recovery remains available after valid coefficients are obtained. Normally inactive rows use $A=0,c=1\,\mathrm m$; their undefined margin is logged as missing. Negative valid margins remain soft-constraint deficits, not numerical errors.

Lower slack is now in meters. `mf_slack_linear` and `mf_slack_quadratic` therefore have cost/m and cost/m² units, and their provisional defaults $100$ and $1$ require retuning. The `cc_mf_imp` YAML block is independent of `cc_mf`. Logs distinguish nominal $F(z,0)$, nominal implicit margin, affine residual, distance slack, root status/iterations/error bound, preprocessing time, and total cycle time; existing solver time continues to cover only solver calls. In debug mode, returned-state evaluation distinguishes frozen nominal features from fresh geometry. See [config/config_cc_mf_imp.yaml](config/config_cc_mf_imp.yaml) for a separate example; numerical tests, simulations, and benchmark validation require an explicit execution request.

In the simulation, `control_cycle_time_s` starts on entry to the enhanced control callback, before obstacle updates, and ends after setup, solver processing, input selection, and pose logging. It excludes path planning, plant integration (`run_system`), and failure snapshot export. On an implicit evaluation error it ends before the exception reaches the failure exporter. Direct `solve_nlp()` callers retain the narrower optimizer-only elapsed time as a fallback for this field.

## 6. Controller Variants and Scope

The four baseline controllers in the evaluation plan follow Section 6-1 of [cc_paper_outline.md](cc_paper_outline.md). The implementation also provides the optional CC-MF-IMP variant for comparison. All use the same local-tangent MPCC formulation.

| Short label | Controller    | Safety formulation                                                                               |
| ----------- | ------------- | ------------------------------------------------------------------------------------------------ |
| **Det**     | PSDF-MPCC-Det | Affine deterministic constraint on the PSDF value                                                |
| **CC-SF**   | PSDF-MPCC-SF  | RTI-linearized Gaussian margin using the PSDF value and PSDF gradient for the minimizing feature |
| **CC-MF**   | PSDF-MPCC-MF  | Hard PSDF-based safety guard plus one slack-relaxed affine MF row per stage                      |
| **CC-MF-IMP** | PSDF-MPCC-MF-IMP | Same guard and stage policy as CC-MF, with a slack-relaxed affine implicit clearance margin |
| **CC-ECS**  | PSDF-MPCC-ECS | Per-polygon PSDF adaptation of an ellipsoidal-confidence-set constraint                          |

Covariance-dependent variants use the same initialization, log-Cholesky formulation, selected robust horizon, and stage-wise violation-probability bounds. Setting $N_R=N$ provides the untruncated configuration. Fixed-time MPC and MF-Trace are not part of the current evaluation plan.

For CC-ECS, define the per-polygon PSDF value at pose $x$ as $\phi_{j,k}(x):=\phi(x,\partial P_k^j)$. The planned PSDF adaptation of the ECS condition in Lew, Bonalli, and Pavone, *Chance-Constrained Sequential Convex Programming for Robust Trajectory Optimization*, Eq. (27), uses

$$
\phi_{j,k}(x_i)-d_{\min}
-\sqrt{\chi_3^2(1-\varepsilon_i)}
\sqrt{
\bigl(\nabla_{\tilde x}\phi_{j,k}(x_i)\bigr)^\top
\tilde\Sigma_{x,i}
\nabla_{\tilde x}\phi_{j,k}(x_i)
}\ge0,
\qquad j=1,\ldots,M_k.
$$

Here, $\chi_3^2(p)$ is the chi-square quantile for the three-dimensional Gaussian pose perturbation, and $\nabla_{\tilde x}\phi_{j,k}(x_i)=T(\theta_i)^\top\nabla_x\phi_{j,k}(x_i)$ includes heading sensitivity. Use one common confidence set per stage without dividing $\varepsilon_i$ by the polygon count. The ellipsoid describes pose uncertainty while polygonal geometry is retained. Apply the common nominal-geometry RTI treatment and retain covariance sensitivity through $\xi_i$. This local-affine PSDF adaptation does not reproduce CC-SCP or inherit its geometric or convergence guarantees. Its per-polygon row structure is distinct from the aggregate MF row.

The Boole result applies only to the selected affine Gaussian distances. It does not certify nonlinear distances, the exact collision probability defined by the PSDF value, feature switching, unrepresented correspondences, or full-mission safety. Tail probabilities use the frozen planning covariance, not the continued open-loop covariance.

The implementation fixes nominal geometry and the body frame and enforces a hard PSDF-based safety guard together with a slack-relaxed affine MF row. Neither affine enforcement nor a zero MF slack provides an independent nonlinear chance certificate. SF/MF infeasibility comparisons isolate feature aggregation only when guard and relaxation policies are matched; otherwise they compare the integrated controllers. Log-Cholesky reconstruction ensures positive definiteness in exact arithmetic, not exact dynamics satisfaction, uniform conditioning, or recursive feasibility.

The CC-MF-family OCP retains one hard PSDF-based safety guard, one slack-relaxed MF row, and one nonnegative scalar MF slack slot at each stage $i=0,\ldots,N$. Stage 0 rows are inactive and future-row activation follows Section 5.5. The allocated slacks add $N+1$ auxiliary decision variables, but no feature-specific variables. Obstacle and feature counts affect GPU cost, not the state/input dimensions, stage-local slack count, MF parameter dimension, or number of avoidance rows per stage.

## Document Alignment

This document retains its independent detailed structure, including valid Sections 5.3, 5.4, and 5.5. The current manuscript uses Sections 5.1 and 5.2 only (written 5-1 and 5-2 in the outlines). Content transfers follow the mapping below and ISSUE-15 of [the consistency checklist](<문서 간 정합성 수정 체크리스트.md>); section-number differences alone do not indicate an erroneous reference.

| This document | Current manuscript location | Role |
| --- | --- | --- |
| 1.1 System and stage definitions | 3.1 and 5.2 | Introduce geometry, prediction indices, and dynamics at first use |
| 1.2 Coordinates and notation | 4.1 and C Appendix A | Synchronize frames, nominal quantities, and approximation notation |
| 2.1 PSDF Definition and Sign Convention | 3.1 | Preserve inherited sign determination and boundary topology |
| 2.2 Features and gradients | 3.2 and 4.1 | Separate feature construction from uncertainty projection |
| 3.1–3.3 Covariance | 4.2 and initialization/rollout in 5.2 | Distinguish covariance model from per-cycle nominal regeneration |
| 4.1–4.3 SF/MF conditions | 4.1 and 4.3 | Match the common Gaussian model and conditional SF/MF equivalence |
| 5.1 Local-tangent path | 5.1 | Match path and error notation |
| 5.2 Dynamics and nominal evaluation | 5.2 | Define current covariance rollout, anchor selection, and nominal geometry |
| 5.3 MF approximation and existing derivative detail | 5.2 | Match current notation while retaining this document's detailed derivation |
| 5.4 OCP and solver interface | 5.2 | Match cost, baseline rows, and solver interface |
| 5.5 Online procedure | 5.2 | Connect initialization, regeneration, anchors/geometry, coefficients, and RTI |
| 6 Controller variants and scope | C 6-1 and manuscript scope statements | Match Det, CC-SF, CC-MF, and CC-ECS evaluation plans |

## References

[^log-cholesky]: J. C. Pinheiro and D. M. Bates, "Unconstrained parametrizations for variance-covariance matrices," *Statistics and Computing*, vol. 6, pp. 289–296, 1996. DOI: 10.1007/BF00140873.

[^rti]: J. Frey, A. Nurkanovic, and M. Diehl, "Advanced-Step Real-time Iterations with Four Levels – New Error Bounds and Fast Implementation in acados," arXiv:2403.07101, 2024.

[^robust-horizon]: E. Bradford and L. Imsland, "Economic Stochastic Model Predictive Control Using the Unscented Kalman Filter," *IFAC-PapersOnLine*, vol. 51, no. 18, pp. 417–422, 2018. DOI: 10.1016/j.ifacol.2018.09.336. Section 3.2 and Algorithm 1, Eq. (10e), hold the covariance square root constant beyond the robust horizon. Eq. (15) states the covariance freeze for untransformed variables. The present method adapts this strategy to its reduced body-fixed covariance without adopting the paper's UKF propagation or state-variable log transformation.
