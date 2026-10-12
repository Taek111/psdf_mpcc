# Method Description

**Augmented PSDF Predictive Control with a Boole-Based Multi-Feature Chance Constraint**

## 1. Overview, Problem Setting, and Notation

The method combines the inherited Polygonal Signed Distance Function (PSDF), control-dependent covariance, a Boole-based multi-feature (MF) chance constraint, and local-tangent Model Predictive Contouring Control (MPCC). The **augmented PSDF** extends the GPU geometry computation to evaluate features, violation probabilities, and analytic MF coefficients without redefining the signed distance. A four-dimensional log-Cholesky state is propagated to a robust horizon and held constant afterward. Mean motion and progress continue over the full horizon, with one affine MF row per stage in the Sequential Quadratic Programming with Real-Time Iteration (SQP-RTI) controller.

### 1.1 System and Environment

At control cycle $k$, the predicted mean pose and physical input are

$$
x_i=[p_i^\top,\theta_i]^\top\in\mathbb R^3,
\qquad
u_i=[v_i,\omega_i]^\top\in\mathbb R^2,
\qquad
x_{i+1}=f_x^d(x_i,u_i),
$$

with $x_i\in\mathcal X$ and $u_i\in\mathcal U$. There are $N$ transitions of duration $\Delta t$, with states at $i=0,\ldots,N$ and inputs at $i=0,\ldots,N-1$. The cycle index $k$ is omitted from prediction variables.

The footprint is a convex polygon $P_0(x)$. Nonconvex footprints use a consistent convex decomposition. Environment objects $P_\kappa$ have edge sets $\mathcal E_\kappa$, with $\mathcal E=\bigsqcup_{\kappa=1}^{K}\mathcal E_\kappa$. Geometry is fixed within each horizon and may be refreshed between cycles, while feature correspondences can vary with pose. Whether $P_\kappa$ denotes an obstacle polygon or a decomposed component remains unspecified in the source and requires implementation verification.

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
| $X_i$, $x_i$, $\bar x_i$ | Random pose, predicted mean pose, and shifted nominal pose used for RTI linearization |
| $\delta\tilde x_i$, $\Delta\tilde x_i$ | Random pose perturbation and deterministic optimizer increment, respectively |
| $f,l,\theta$ | Forward, lateral, and heading component labels |
| $\ell=1,\ldots,n_F$ | Feature index and feature-index capacity. $L$ is reserved for the Cholesky factor |
| $\xi_i$, $L(\xi_i)$, $\tilde\Sigma_{x,i}$ | Log-Cholesky optimization state, reconstructed factor, and reconstructed body-fixed covariance |
| $\tilde A_i$, $\tilde W_i$ | Body-fixed perturbation transition matrix and process-noise covariance |
| $q_\ell(x)$, $q^w_{a,i}$ | Feature-distance gradient and process-noise intensity in direction $a\in\{f,l,\theta\}$ |
| $\varepsilon_i$ | Stage violation-probability bound, with $\varepsilon_i\in(0,1)$ |
| $\Phi$, $\varphi_{\mathcal N}$ | Standard normal cumulative distribution function and probability density function |

Generic geometric functions have no stage index. Bars denote nominal evaluations, $\mathrm{loc}$ denotes the local implementation model, and hats on functions denote affine approximations. Hats on $\hat x_k$ and $\hat\Sigma_{x,k}$ denote current estimates.

## 2. PSDF Geometry and Feature Construction

### 2.1 PSDF Signed Distance

The inherited PSDF signed distance and its world-frame pose gradient are

$$
\phi(x,\mathcal E)
=\min_{\kappa}\left[
\operatorname{dist}(P_0(x),P_\kappa)
-\operatorname{pen}(P_0(x),P_\kappa)
\right],
\qquad
g(x,\mathcal E)=\nabla_x\phi(x,\mathcal E).
$$

The signed distance is positive in separation, zero at contact, and negative in penetration. The inherited PSDF evaluates bidirectional point–segment candidates in the body-fixed frame and handles overlap using the Separating Axis Theorem. [Citation needed]

### 2.2 Bidirectional Features and Their Gradients

For each fixed $\kappa$ and robot vertex, the robot-to-obstacle family minimizes over obstacle edges. For each fixed $\kappa$ and robot edge, the obstacle-to-robot family minimizes over obstacle vertices. Their indexed union is

$$
\mathcal F(x)=\mathcal F^{\mathrm{R\to O}}(x)
\sqcup\mathcal F^{\mathrm{O\to R}}(x).
$$

The union preserves labels without deduplication. Masks $m_\ell(x)\in\{0,1\}$ identify valid entries among $\ell=1,\ldots,n_F$. The PSDF-minimizing pair is already represented in separated configurations and is not inserted again.

For a valid separated feature, the body-fixed closest points $c_\ell^{\mathrm R}(x)$ and $c_\ell^{\mathrm O}(x)$ lie on the robot and obstacle, respectively. Define

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

The robot-side point is a vertex or a projected point on a robot edge, so the expression applies to both families. Differentiation holds the feature correspondence and point-to-segment projection case fixed. Euclidean feature distances do not replace the signed PSDF at contact or in penetration.

## 3. Control-Dependent Covariance Prediction

### 3.1 Reduced Covariance via Log-Cholesky Parameterization

The reduced covariance retains three variances and the lateral–heading cross-covariance. Its independent optimization state and factor are[^log-cholesky]

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

Conversely, the log-Cholesky coordinates are recovered directly from the retained covariance entries,

$$
\xi_i=
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

These formulas invert the reconstruction on the retained forward and lateral–heading blocks. The reduction sets forward–lateral and forward–heading cross-covariances to zero and is not an eigenvalue-based PSD repair.

For initialization, set the retained entries of $\tilde\Sigma_{x,0}$ to those of the transformed estimator covariance $T(\hat\theta_k)^\top\hat\Sigma_{x,k}T(\hat\theta_k)$, where $\hat\Sigma_{x,k}\succ0$, and apply the formulas above at $i=0$ to obtain $\xi_0$.

For finite $\xi_i$, the factor is invertible in exact arithmetic, so $\tilde\Sigma_{x,i}\succ0$ independently of nonlinear dynamics defects at an RTI iterate. This guarantee applies to factor reconstruction, not an affine approximation of covariance entries.

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

Within the propagation interval, the next log-Cholesky state is obtained from the retained entries of this prediction using the formulas in Section 3.1. Reconstruction and propagation therefore use the same four coordinates without an intermediate covariance-entry state.

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

SF uses the PSDF-minimizing pair and its gradient. Its affine violation event differs from the MF union event, and neither formulation is treated as a special case of the other.

### 4.2 Multi-Feature Gaussian Distance Model

For each valid feature selected at the predicted mean pose $x_i$, replace the nonlinear random distance by its first-order affine approximation,

$$
d_\ell(X_i)\approx D_{\ell,i}
\coloneqq d_\ell(x_i)+q_\ell(x_i)^\top\delta\tilde x_i.
$$

The Gaussian pose perturbation induces

$$
D_{\ell,i}\sim\mathcal N(\mu_{\ell,i},\sigma_{\ell,i}^2),
\qquad
\mu_{\ell,i}=d_\ell(x_i),
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
=\Phi\!\left(\frac{d_{\min}-\mu_{\ell,i}}{\sigma_{\ell,i}}\right).
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

For $\mu_{\ell,i}>d_{\min}$, larger $\sigma_{\ell,i}$ increases violation probability. Inputs before $N_R$ affect covariance through $\tilde A_i$ and $\tilde W_i$. Later, $\partial F_{\xi,i}/\partial u_i=0$, but earlier inputs still affect the frozen covariance and later inputs still affect feature geometry. Virtual progress does not directly affect covariance. This coupling can favor slower or smoother motion without guaranteeing slowdown or adding a separate heuristic.

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
\hat p_i(s_i)=p_i^{\mathrm{ref}}
+(s_i-s_i^{\mathrm{ref}})t_i^{\mathrm{ref}}.
$$

Using the unit tangent, define the contouring, lag, and anchor-deviation errors,

$$
\begin{aligned}
e_{c,i}&=(p_i-p_i^{\mathrm{ref}})^\top n_i^{\mathrm{ref}},\\
e_{l,i}&=-(p_i-p_i^{\mathrm{ref}})^\top t_i^{\mathrm{ref}}
+(s_i-s_i^{\mathrm{ref}}),\\
e_{s,i}&=s_i-s_i^{\mathrm{ref}},
\qquad e_i=[e_{c,i},e_{l,i},e_{s,i}]^\top.
\end{aligned}
$$

Fixed anchors make the errors affine in position and progress. Their quadratic penalties regulate path fidelity, progress consistency, and anchor deviation without fixed traversal timing. The path approximation remains local.

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

Shift the previous mean, input, and progress trajectories to form $\{\bar z_i\}_{i=0}^{N}$ and $\{\bar\nu_i\}_{i=0}^{N-1}$. Reset the initial pose and covariance to the current estimates and regenerate

$$
\bar\xi_0=\xi_0,
\qquad
\bar\xi_{i+1}=F_{\xi,i}(\bar\xi_i,\bar x_i,\bar u_i).
$$

At each nominal pose, the augmented PSDF evaluates

$$
\begin{gathered}
\bar{\mathcal F}_i=\mathcal F(\bar x_i),
\qquad
\bar d_{\ell,i}=d_\ell(\bar x_i),
\qquad
\bar q_{\ell,i}=q_\ell(\bar x_i),
\qquad
\bar m_{\ell,i}=m_\ell(\bar x_i),\\
\bar\phi_i=\phi(\bar x_i,\mathcal E),
\qquad
\bar g_i=g(\bar x_i,\mathcal E).
\end{gathered}
$$

Correspondences, projection cases, gradients, masks, and the nominal body frame are fixed QP parameters and refreshed before the next QP. Covariance coordinates remain optimization variables. Geometry is evaluated over the full horizon, including the covariance tail.

### 5.3 Local MF Model and Analytic Linearization

The deterministic mean-pose increment in the nominal body-fixed frame and the local feature statistics are

$$
\begin{aligned}
\Delta\tilde x_i
&=T(\bar\theta_i)^\top(x_i-\bar x_i),\\
\mu_{\ell,i}^{\mathrm{loc}}
&=\bar d_{\ell,i}+\bar q_{\ell,i}^\top\Delta\tilde x_i,\\
(\sigma_{\ell,i}^{\mathrm{loc}})^2
&=\|L(\xi_i)^\top\bar q_{\ell,i}\|^2.
\end{aligned}
$$

The nonlinear implementation residual is

$$
h_i^{\mathrm{MF,loc}}(z_i)
=\varepsilon_i-\sum_{\ell=1}^{n_F}\bar m_{\ell,i}
\Phi\!\left(
\frac{d_{\min}-\mu_{\ell,i}^{\mathrm{loc}}}
{\sigma_{\ell,i}^{\mathrm{loc}}}\right).
$$

The local model does not differentiate pose dependence of nominal gradients or frame rotation. At $\bar z_i$, define

$$
\bar h_i^{\mathrm{MF}}=h_i^{\mathrm{MF,loc}}(\bar z_i),
\qquad
\bar\sigma_{\ell,i}^2=\|L(\bar\xi_i)^\top\bar q_{\ell,i}\|^2,
\qquad
\bar\zeta_{\ell,i}=\frac{d_{\min}-\bar d_{\ell,i}}{\bar\sigma_{\ell,i}}.
$$

The world-frame pose derivative is

$$
\left.\nabla_{x_i}h_i^{\mathrm{MF,loc}}\right|_{\bar z_i}
=T(\bar\theta_i)\sum_{\ell=1}^{n_F}\bar m_{\ell,i}
\frac{\varphi_{\mathcal N}(\bar\zeta_{\ell,i})}{\bar\sigma_{\ell,i}}
\bar q_{\ell,i}.
$$

The local variance is differentiated directly with respect to the same log-Cholesky state used in Section 3,

$$
\nabla_{\xi_i}(\sigma_{\ell,i}^{\mathrm{loc}})^2
=
\begin{bmatrix}
2e^{2\xi_{f,i}}\bar q_{f,\ell,i}^2\\
2e^{\xi_{l,i}}\bar q_{l,\ell,i}
\left(e^{\xi_{l,i}}\bar q_{l,\ell,i}+\xi_{c,i}\bar q_{\theta,\ell,i}\right)\\
2e^{2\xi_{\theta,i}}\bar q_{\theta,\ell,i}^2\\
2\bar q_{\theta,\ell,i}
\left(e^{\xi_{l,i}}\bar q_{l,\ell,i}+\xi_{c,i}\bar q_{\theta,\ell,i}\right)
\end{bmatrix}.
$$

Thus,

$$
\left.\nabla_{\xi_i}h_i^{\mathrm{MF,loc}}\right|_{\bar z_i}
=\sum_{\ell=1}^{n_F}\bar m_{\ell,i}
\varphi_{\mathcal N}(\bar\zeta_{\ell,i})
\frac{\bar\zeta_{\ell,i}}{2\bar\sigma_{\ell,i}^2}
\left.\nabla_{\xi_i}(\sigma_{\ell,i}^{\mathrm{loc}})^2\right|_{\bar\xi_i}.
$$

The augmented PSDF directly assembles

$$
A_i^{\mathrm{MF}}
=\left.\nabla_{z_i}h_i^{\mathrm{MF,loc}}\right|_{\bar z_i}
\in\mathbb R^{n_z},
\qquad
c_i^{\mathrm{MF}}
=\bar h_i^{\mathrm{MF}}-(A_i^{\mathrm{MF}})^\top\bar z_i,
\qquad n_z=8,
$$

with a zero coefficient for progress. The affine RTI row is

$$
\boxed{
\hat h_i^{\mathrm{MF}}(z_i)
=(A_i^{\mathrm{MF}})^\top z_i+c_i^{\mathrm{MF}}\ge0.
}
$$

The general condition $h_i^{\mathrm{MF}}$, local model $h_i^{\mathrm{MF,loc}}$, and affine row $\hat h_i^{\mathrm{MF}}$ are distinct. The robust horizon changes covariance prediction, not these stage-local derivatives.

### 5.4 OCP and Solver Interface

Let $\Lambda=\operatorname{diag}(\lambda_c,\lambda_l,\lambda_s)$, let $\Lambda_N$ be the terminal path-error weight, and let $R_\nu=\operatorname{diag}(R_u,\rho_{v_s})$ weight physical and virtual inputs. The OCP with externally assembled MF rows is

$$
\begin{aligned}
\min_{\{z_i\}_{i=0}^{N},\{\nu_i\}_{i=0}^{N-1}}\quad&
\sum_{i=0}^{N-1}
\left(\|e_i\|_\Lambda^2+\|\nu_i\|_{R_\nu}^2
-\lambda_{\mathrm{prog}}v_{s,i}\right)
+\|e_N\|_{\Lambda_N}^2\\
\text{s.t.}\quad
&z_0=\hat z_k,\\
&z_{i+1}=f_{d,i}(z_i,\nu_i),&&i=0,\ldots,N-1,\\
&\hat h_i^{\mathrm{MF}}(z_i)\ge0,&&i=0,\ldots,N,\\
&x_i\in\mathcal X,&&i=0,\ldots,N,\\
&\nu_i\in\mathcal U_\nu,&&i=0,\ldots,N-1.
\end{aligned}
$$

The initial state is $\hat z_k=[\hat x_k^\top,\hat s_k,\xi_0^\top]^\top$, with current progress initialization $\hat s_k$. Anchors and MF coefficients are fixed during the RTI step. The objective balances path errors and inputs against progress.

SQP-RTI linearizes the dynamics about $(\bar z_i,\bar\nu_i)$ and need not satisfy the nonlinear recursion exactly in one step.[^rti] Returned covariances are reconstructed from $L(\xi_i)L(\xi_i)^\top$. The four coordinates replace the four covariance-entry states without adding a PSD constraint.

The GPU returns $\{\bar\phi_i,\bar g_i,A_i^{\mathrm{MF}},c_i^{\mathrm{MF}}\}$ per stage. Only $(A_i^{\mathrm{MF}},c_i^{\mathrm{MF}})$, containing $n_z+1$ scalars, is transferred for the MF row. Features, masks, Gaussian evaluations, and analytic Jacobian assembly remain on the GPU. The feature arrays are concatenated without deduplication, and masking precedes standardization to avoid undefined padded values. MF Jacobians do not use runtime automatic differentiation.

For the deterministic and SF baseline implementations, the inherited affine PSDF mean model is

$$
\hat\phi_i(x_i)=\bar\phi_i+\bar g_i^\top(x_i-\bar x_i).
$$

The deterministic baseline enforces $\hat\phi_i-d_{\min}\ge0$. SF adds its projected-covariance term from Section 4.1. These inherited baseline rows are not added to the MF OCP. [Citation needed]

### 5.5 Online Control Procedure

At each control cycle, the controller performs the following operations.

1. Receive the current pose and covariance, initialize $\xi_0$, shift the mean, input, and progress trajectories, and reset the initial nominal values. Regenerate $\bar\xi_i$ with $F_{\xi,i}$, including the frozen tail.
2. Refresh the local path anchors. Evaluate the augmented PSDF at all stages and assemble the MF probabilities, analytic derivatives, and affine coefficients.
3. Transfer the MF coefficients, path parameters, and fixed robust-horizon schedule to the solver. Perform one SQP-RTI preparation and feedback step.
4. Apply the first physical input. Reconstruct returned covariances from their log-Cholesky states and monitor nonlinear dynamics defects separately. Reinitialize and regenerate the nominal covariance at the next cycle.

Offline validation compares analytic feature and MF derivatives with finite differences and automatic differentiation. These checks are separate from the online computation.

## 6. Controller Variants and Scope

| Controller | Safety formulation | Behavior formulation |
| --- | --- | --- |
| **PSDF-MPC-Det** | Affine deterministic PSDF row | Fixed-time tracking MPC |
| **PSDF-MPCC-Det** | Affine deterministic PSDF row | Local-tangent MPCC |
| **PSDF-MPCC-SF** | Gaussian affine PSDF chance constraint | Local-tangent MPCC |
| **PSDF-MPCC-MF** | Boole-based MF condition with one affine RTI row per stage | Local-tangent MPCC |

Covariance-dependent variants use the same log-Cholesky formulation and selected robust horizon. Setting $N_R=N$ provides the untruncated configuration. Any **MF-Trace surrogate** remains a separate ablation without a chance-constraint interpretation.

The Boole result applies only to the selected affine Gaussian distances. It does not certify nonlinear distances, exact PSDF collision probability, feature switching, unrepresented correspondences, or full-mission safety. Tail probabilities use the frozen planning covariance, not the continued open-loop covariance.

The implementation fixes nominal geometry and the body frame and enforces an affine RTI row, not an independent nonlinear chance certificate. Log-Cholesky reconstruction ensures positive definiteness in exact arithmetic, not exact dynamics satisfaction, uniform conditioning, or recursive feasibility.

The OCP retains one MF row at each stage $i=0,\ldots,N$. Obstacle and feature counts affect GPU cost, but not the OCP decision dimension, MF parameter dimension, or MF row count.

## References

[^log-cholesky]: J. C. Pinheiro and D. M. Bates, "Unconstrained parametrizations for variance-covariance matrices," *Statistics and Computing*, vol. 6, pp. 289–296, 1996. DOI: 10.1007/BF00140873.

[^rti]: J. Frey, A. Nurkanovic, and M. Diehl, "Advanced-Step Real-time Iterations with Four Levels – New Error Bounds and Fast Implementation in acados," arXiv:2403.07101, 2024.

[^robust-horizon]: E. Bradford and L. Imsland, "Economic Stochastic Model Predictive Control Using the Unscented Kalman Filter," *IFAC-PapersOnLine*, vol. 51, no. 18, pp. 417–422, 2018. DOI: 10.1016/j.ifacol.2018.09.336. Section 3.2 and Algorithm 1, Eq. (10e), hold the covariance square root constant beyond the robust horizon. Eq. (15) states the covariance freeze for untransformed variables. The present method adapts this strategy to its reduced body-fixed covariance without adopting the paper's UKF propagation or state-variable log transformation.
