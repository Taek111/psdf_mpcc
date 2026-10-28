# CC-MF-IMP: implicit clearance margin 기반 컨트롤러 구현 계획

작성일: 2026-10-26 · 상태: 수학 및 코드 검토를 마친 구현 계획. 컨트롤러 구현·테스트·시뮬레이션·benchmark는 수행하지 않았다.

## 1. 구현 방향과 범위

기존 CC-MF의 확률 residual을 직접 선형화하는 대신, **동일한 확률합이 위험 예산에 도달하는 공통 clearance threshold의 이동량**을 계산하고 이 거리 residual을 선형화한다. 새 컨트롤러의 표시 이름은 **CC-MF-IMP**, Python/config의 정식 식별자는 **`cc_mf_imp`**, 사용자 입력 별칭은 **`cc-mf-imp`**로 정한다.

핵심 변경은 `models/augmented_psdf.py`에서 생성하는 MF 계수의 의미다. `control/cc_mf_optimizer.py`가 이미 `A @ z + c`를 받아 제약으로 사용하므로, acados에 root equation을 넣거나 상태를 추가할 필요는 없다. 다만 wrapper, 확률 로그, 비활성 행 상수, slack 단위, controller 등록도 함께 확장해야 한다. 두 파일의 출력만 바꾸면 기존 확률 로그가 잘못된 값을 기록한다.

첫 구현은 기존 CC-MF와 독립적으로 선택 가능한 subclass로 추가한다. DD dynamics, Log-Cholesky 상태, 공분산 전파, robust horizon, MPCC cost, 경로 parameter, hard PSDF guard, stage 활성 정책 및 일반적인 solver recovery 흐름을 재사용한다. 단, **필요한 implicit root 또는 계수 계산이 실패하면 guard-only 제어로 전환하지 않고 해당 제어 실행을 중단한다.** 기존 `cc_mf`와 `forward_mf()`의 동작·출력 계약은 유지한다.

Root 계산은 **GPU stage-batched upper-bound Newton을 PyTorch eager로 구현**한다. 이번 구현에는 CUDA Graph capture/replay, `torch.compile`, custom CUDA/Triton kernel을 추가하지 않는다. acados의 기존 model/codegen 경로는 그대로 유지한다.

검토 근거:

- [Implicit MF clearance margin.md](</home/taek111/projects/psdf_mpcc/Implicit MF clearance margin.md>): implicit margin의 정의와 미분.
- [method_description.md](/home/taek111/projects/psdf_mpcc/method_description.md), §4.2–4.3, §5.3–5.4: Gaussian feature 모델, nominal geometry, Log-Cholesky 미분, OCP 구조.
- [augmented_psdf.py](/home/taek111/projects/psdf_mpcc/models/augmented_psdf.py), [augmented_psdf_wrapper.py](/home/taek111/projects/psdf_mpcc/models/augmented_psdf_wrapper.py): 실제 geometry 및 MF 계수 생성.
- [cc_mf_optimizer.py](/home/taek111/projects/psdf_mpcc/control/cc_mf_optimizer.py), [cc_mf_diagnostics.py](/home/taek111/projects/psdf_mpcc/utils/cc_mf_diagnostics.py): stage 정책, solver 주입, slack, 로그·복구.
- [CC_MF_SF_CLEARANCE_ISSUE.md](/home/taek111/projects/psdf_mpcc/CC_MF_SF_CLEARANCE_ISSUE.md), [linearization_error_issue.md](/home/taek111/projects/psdf_mpcc/linearization_error_issue.md): 기존 선형화·진단상의 한계.

## 2. 현재 구현에서 유지할 조건

| 항목 | 현재 코드와 새 구현의 기준 |
|---|---|
| 상태 | `z=[x,y,theta,s,xi_f,xi_l,xi_theta,xi_c]`, 8차원 |
| 입력 | `[v,omega,v_s]`, 3차원 |
| Feature | 장애물별 양방향 vertex/edge 후보를 현재 방식으로 pooling·concatenation. 중복 제거, 거리 기반 pruning, 확률합 정규화 없음 |
| 공분산 | `L(xi)L(xi)^T`. 현재 MF projected variance에는 jitter/floor를 추가하지 않음 |
| Geometry 미분 | nominal feature, `q`, frame rotation, mask를 고정. 거리의 pose 1차 미분과 표준편차의 `xi` 미분만 사용 |
| Solver 행 | stage별 hard Row G + soft Row M, MF parameter는 8개 계수 + 상수 1개 |
| Stage 0 | OCP safety row 비활성. 현재 exact PSDF가 `d_col` 미만이면 solve를 건너뛰는 기존 `safe_stop_stage0` 경로 사용 |
| Future Row G | 장애물 제약이 켜져 있으면 stage `1…N`에서 hard affine PSDF guard |
| Future Row M | 기존 obstacle/MF switch, `phi > d_mf_mask`, 설정된 stage window에 따라 필요한 stage를 먼저 결정. 기본 `1…N-1` 및 terminal `N`. 필요한 stage의 MF 데이터·root·계수 실패는 행 비활성화가 아니라 제어 중단 |
| Timing | 기존 solver 시간은 `solver.solve()` 호출 구간만 기록. 새 root 계산은 preprocessing |

`method_description.md`의 모든 stage 적용 서술과 실제 stage-0 정책은 다르다. 이번 설계는 현재 코드를 기준으로 한다. 동일 feasible set이라는 수학적 주장은 아래의 **유효 feature 모델에서 slack을 넣기 전의 비선형 조건**에 관한 것이며, mask 또는 soft constraint로 완화한 실행 전체에 대한 보장이 아니다.

## 3. 수학적 정의

### 3.1 고정된 nominal geometry와 covariance

Stage index를 생략한다. `x_p=[x,y,theta]^T`를 pose, `s`를 progress, `xi`를 covariance state라 두고, nominal을 `z_bar`라 한다. 유효 feature 집합은 `F_bar`, 그 크기는 `M>=1`이다. `T(theta_bar)=diag(R(theta_bar),1)`은 body perturbation을 world pose perturbation으로 옮긴다.

각 feature에 대해 현재 코드가 반환하는 거리와 body gradient를 `d_bar_l`, `q_bar_l`라 하면,

$$
g_\ell=T(\bar\theta)\bar q_\ell,\qquad
\widehat d_\ell(x_p)=\bar d_\ell+g_\ell^\top(x_p-\bar x_p).
$$

$$
L(\xi)=\begin{bmatrix}
e^{\xi_f}&0&0\\
0&e^{\xi_l}&0\\
0&\xi_c&e^{\xi_\theta}
\end{bmatrix},\qquad
\sigma_\ell(\xi)=\|L(\xi)^\top\bar q_\ell\|>0.
$$

`q_bar`, `g`, feature 선택과 mask는 한 RTI 계산 동안 고정한다. 특히 `sigma`의 pose 의존성을 다시 미분하지 않는다. `xi`는 OCP 상태로 남고 입력과 covariance dynamics를 통해 연결된다. Covariance를 고정 상수로 바꾸는 설계가 아니다.

### 3.2 Implicit margin과 feasible set

$$
F(z,r)=\sum_{\ell\in\overline{\mathcal F}}
\Phi\!\left(\frac{d_{\min}+r-\widehat d_\ell(x_p)}{\sigma_\ell(\xi)}\right),
\qquad F(z,r(z))=\varepsilon,\quad 0<\varepsilon<1.
$$

`r [m]`는 공통 요구 clearance를 얼마나 높일 수 있는지 나타낸다. 양수는 threshold 여유, 음수는 부족량이다. 실제 로봇 이동거리, 방향별 이동 가능 반경 또는 Euclidean signed distance는 아니다.

`sigma_l>0`이면

$$
F_r(z,r)=\sum_\ell\frac{\varphi_{\mathcal N}((d_{\min}+r-\widehat d_\ell)/\sigma_\ell)}{\sigma_\ell}>0.
$$

또한 `r→−∞`에서 `F→0`, `r→+∞`에서 `F→M`이다. 따라서 `M>=1`, `0<epsilon<1`이면 유일한 유한 root가 존재한다. 단조성으로

$$
\boxed{F(z,0)\le\varepsilon\iff r(z)\ge0.}
$$

보존되는 것은 동일 feature/mask/covariance에 대한 Boole 기반 Gaussian 근사 조건이다. `F` 자체는 확률들의 합으로 1보다 클 수 있으며, exact collision probability가 아니다. Feature 독립성은 필요하지 않는다. Stage별 epsilon을 mission 전체의 위험 예산으로 해석하지 않는다.

## 4. Analytic gradient와 RTI row

### 4.1 Root에서 평가할 미분

Nominal에서 root `r_bar`를 구한 후 다음 값을 **다시 계산**한다.

$$
\zeta_\ell=\frac{\bar d_\ell-d_{\min}-\bar r}{\bar\sigma_\ell},\qquad
a_\ell=\frac{\varphi_{\mathcal N}(\zeta_\ell)}{\bar\sigma_\ell},\qquad
w_\ell=\frac{a_\ell}{\sum_j a_j}.
$$

Implicit differentiation `F_z+F_r grad(r)=0`을 적용하면

$$
\boxed{\nabla_z r=\sum_\ell w_\ell
\left(\nabla_z\widehat d_\ell-\zeta_\ell\nabla_z\sigma_\ell\right).}
$$

기존 `_compute_A()`가 `r=0`에서 구한 확률 gradient를 단순히 normalize하면 이 식이 되지 않는다. 기존 계수를 그대로 나누는 shortcut은 사용하지 않는다.

### 4.2 현재 Log-Cholesky 상태에 대한 명시적 계산

Feature마다 다음 중간량을 사용한다.

$$
u_f=e^{\xi_f}q_f,\quad u_l=e^{\xi_l}q_l,\quad
u_\theta=e^{\xi_\theta}q_\theta,\quad b=u_l+\xi_cq_\theta,
\qquad v=\sigma^2=u_f^2+b^2+u_\theta^2.
$$

$$
\nabla_\xi v=
\begin{bmatrix}2u_f^2\\2u_lb\\2u_\theta^2\\2q_\theta b\end{bmatrix},
\qquad \nabla_\xi\sigma=\frac{\nabla_\xi v}{2\sigma}.
$$

이는 현재 `_compute_A()`의 `variance_gradient`와 같다. 따라서 새 계수는

$$
\boxed{
\begin{aligned}
A^{\rm imp}_{0:3}&=T(\bar\theta)\sum_\ell w_\ell\bar q_\ell,\\
A^{\rm imp}_3&=0,\\
A^{\rm imp}_{4:8}&=-\sum_\ell w_\ell\zeta_\ell
\frac{\nabla_\xi v_\ell}{2\bar\sigma_\ell},\\
c^{\rm imp}&=\bar r-(A^{\rm imp})^\top\bar z.
\end{aligned}}
$$

모든 항은 nominal에서 평가한다. 이 미분은 runtime autograd나 root iteration의 역전파 없이 계산한다.

### 4.3 Solver에 넣는 제약

$$
\widehat r(z)=\bar r+(A^{\rm imp})^\top(z-\bar z)
=(A^{\rm imp})^\top z+c^{\rm imp},\qquad
\widehat r(z)+\delta\ge0,\quad\delta\ge0.
$$

`r`은 QP decision variable이 아니다. Geometry·root·gradient는 solver 밖에서 계산하고 acados에는 9개 계수를 전달한다. 이미 affine인 행은 RTI에서 그대로 사용한다. Dynamics는 현재 SQP-RTI 처리 경로를 따른다.

별도의 hard guard는 유지한다.

$$
\widehat\phi(x_p)-d_{\rm col}
=\bar\phi+\bar g_\phi^\top(x_p-\bar x_p)-d_{\rm col}\ge0.
$$

`d_min`은 확률 모델 threshold, `d_col`은 guard threshold, `d_mf_mask`는 nominal domain gate다. 서로 다른 역할이며 현재처럼 같은 수치를 사용해도 혼동하지 않는다.

### 4.4 동등성과 개선 범위

- 단일 feature에서는 `r=d-d_min-gamma*sigma`, `gamma=Phi^-1(1-epsilon)`이다. 동일 feature, covariance, gamma, 비활성 SF floor/cap, 동일 nominal model이면 **CC-SF의 residual 및 1차 row와 일치**한다. 기존 CC-SF는 hard constraint이고 CC-MF 계열은 soft row이므로 컨트롤러 전체가 같아지는 것은 아니다.
- 모든 distance가 동일한 `c`만큼 변하고 sigma가 고정되면 `r(d+c,sigma)=r(d,sigma)+c`다.
- 경계 `r_bar=0`에서는 `A_imp=A_MF/F_r(z_bar,0)`이므로 unrelaxed tangent halfspace가 같다. 경계 밖에서는 일반적으로 다르다.
- 새로운 row도 여러 feature gradient를 한 행에 합한다. 마주 보는 벽의 gradient 상쇄, geometry 전환, covariance dynamics의 RTI 오차, 큰 상태 변화의 외삽 문제는 남는다.
- `delta=0`이고 affine row가 만족되어도 nonlinear `r(z)>=0` 또는 실제 충돌 회피가 증명되지는 않는다. 검증 항목에서 affine/frozen/fresh residual을 분리한다.

## 5. GPU scalar root 계산

### 5.1 해를 포함하는 analytic bracket

`b_l=d_bar_l-d_min`, `Q(p)=Phi^-1(p)`라 두고 stage별 valid feature 수 `M`을 사용한다.

$$
r_{\rm lo}=\min_\ell\{b_\ell+\sigma_\ell Q(\varepsilon/M)\},\qquad
r_{\rm hi}=\min_\ell\{b_\ell+\sigma_\ell Q(\varepsilon)\}.
$$

`r_lo`에서는 각 feature 확률이 `epsilon/M` 이하이므로 `F(r_lo)<=epsilon`이다. `r_hi`를 결정한 feature는 확률이 `epsilon`이고 나머지는 비음수이므로 `F(r_hi)>=epsilon`이다. 이 bracket은 **Newton 초기값과 수치 검증에 사용**하며, 초기값은 `r^(0)=r_hi`로 둔다. `epsilon/M`은 **root bracket 계산용**이며, feature별 위험 예산 제약으로 모델을 변경하는 것이 아니다. `M=1`이면 두 끝점이 같고 닫힌형 root를 바로 사용한다.

### 5.2 첫 구현의 수치 알고리즘

**Newton 적용 범위와 update.** §3.2의 수학적 root 정의는 `0<epsilon<1`에서 유지한다. 다만 이번 upper-bound Newton 구현의 지원 설정은 **`0<epsilon<0.5`**로 제한한다. 이 범위에서는 `r<=r_hi`일 때 모든 `zeta_l=(b_l-r)/sigma_l`가 양수이므로

$$
\begin{aligned}
F'(r)&=\sum_\ell\frac{\varphi_{\mathcal N}(\zeta_\ell)}{\sigma_\ell}>0,\\
F''(r)&=\sum_\ell\frac{\zeta_\ell\varphi_{\mathcal N}(\zeta_\ell)}{\sigma_\ell^2}>0.
\end{aligned}
$$

따라서

$$
\boxed{
r^{(0)}=r_{\rm hi},\qquad
r^{(k+1)}=r^{(k)}-\frac{F(r^{(k)})-\varepsilon}{F'(r^{(k)})}
}
$$

는 exact arithmetic에서 `r_star<=r^(k+1)<=r^(k)`를 만족한다. 이 성질은 floating-point 검사를 생략할 근거가 아니다. `epsilon>=0.5`를 임의로 clipping하거나 다른 solver로 조용히 전환하지 않고, 지원 범위 밖의 configuration으로 명시적인 오류를 낸다. Config initialization에서 epsilon, 유한한 threshold, 양의 iteration limit 및 유효한 tolerance를 검증한다.

**Eager batch 실행.**

1. 한 geometry batch에서 모든 stage의 `phi,g,d,q,mask`를 구한다. Root iteration마다 geometry를 재계산하지 않는다. 기존 switch·stage·domain 정책으로 root 계산이 필요한 stage를 결정하고, 정상 비활성 stage는 계산 대상에서 제외한다. 이 판정에 `root_valid`나 feature 수를 넣어 계산 실패를 비활성화로 바꾸지 않는다.
2. `d,q,z_bar,epsilon`의 risk 연산을 GPU `float64`로 승격해 `sigma`, variance gradient, bracket을 계산한다. Geometry는 기존 `float32`를 유지한다. Covariance 계산에 쓸 원래 nominal은 CPU의 double에서 직접 전달하여 불필요한 float32 왕복을 피한다.
3. `F(r)=sum(0.5*erfc((d-d_min-r)/(sqrt(2)*sigma)))`를 사용한다. 작은 tail에 대해 기존 `0.5*(1+erf(...))`의 소거를 피하려는 선택이다. 기존 CC-MF 경로까지 함께 바꾸지는 않는다. Newton derivative는 동일한 `zeta`에서 `F_prime=sum(exp(-0.5*zeta**2)/(sqrt(2*pi)*sigma))`로 계산한다. Geometry, `sigma`, variance gradient는 iteration 동안 재계산하지 않는다.
4. `r_lo,r_hi`에서 유효 bracket을 확인한다. 반올림에 따른 작은 경계 오차는 outward rounding으로 보정하고 재확인한다. 허용 오차를 넘는 bracket 실패는 root 실패로 기록한다. `M=1`은 닫힌형 candidate를 사용하고 Newton update를 생략하되 최종 수치 검사는 수행한다.
5. `M>1`인 stage는 `r=r_hi`에서 시작하여 설정한 최대 횟수의 **batched Newton update**를 수행한다. 매 update 전에 `F,F_prime`의 유한성 및 `F_prime>0`을 확인한다. `F(r)-epsilon`의 부호에 따라 검증용 bracket endpoint를 갱신하되, midpoint를 다음 iterate로 사용하지 않는다. `F(r)<epsilon`이면 lower endpoint, `F(r)>epsilon`이면 upper endpoint를 갱신하며, `r` 자체의 부호로 판단하지 않는다. 비유한 step, 반올림 수준을 넘는 bracket 이탈, probability residual 기준도 만족하지 못한 stagnation은 실패 상태로 기록한다. Roundoff 수준의 endpoint 이탈만 해당 endpoint로 보정할 수 있으며, 일반적인 step clipping이나 derivative floor로 실패를 숨기지 않는다.
6. Loop 안에서 `F(r)==epsilon`이거나, floating-point에서 다음 iterate가 현재값과 같고 probability residual이 아래의 `tau_p` 이하이면 candidate를 고정하고 최종 검사에 맡긴다. 그 밖의 유효 stage는 최대 횟수까지 Newton update를 수행한다. 고정된 candidate도 최종 local bracket 검사를 통과하기 전에는 `root_valid=True`로 설정하지 않는다. 반복 안에서 `.item()`, `.cpu()`, Python의 tensor-dependent 종료 판정을 하지 않고, GPU mask와 유한 dummy 입력으로 고정 길이 eager loop를 실행한다. 수렴·실패 stage를 mask해도 전체 tensor 연산 비용이 반드시 생략되는 것은 아니다.
7. 마지막 update 이후의 candidate도 반드시 다시 평가한다. 아래의 **probability residual + local bracket 검사**를 모두 통과한 동일 candidate만 `r_bar`로 반환한다. 초기/global bracket의 폭이 작아졌는지는 종료 조건으로 사용하지 않는다. Root에서 `zeta`, weights, analytic gradient, `A,c`를 계산하고 계수의 유한성도 확인한다. Root 음수는 정상적인 constraint 위반값이므로 0으로 clipping하지 않으며, 음수라는 이유만으로 controller를 중단하지 않는다.
8. 필요한 stage 단위 출력만 CPU로 전달한다. Feature 배열은 GPU에 남긴다. Controller는 solver parameter 설치 및 `solver.solve()` **이전**에 필요한 모든 stage의 유효성을 확인한다. 하나라도 실패하면 §5.3의 fail-stop으로 연결한다. Bisection 재시도, stagewise 자동 재시도, 이전 root/계수 재사용 또는 guard-only continuation은 하지 않는다.

**최종 수렴 판정.** 반환 candidate를 `r_tilde`라 하고

$$
\tau_r=\texttt{atol\_m}+\texttt{rtol}\,|\widetilde r|,\qquad
\tau_p=\texttt{prob\_atol}
$$

로 둔다. 다음 두 조건을 모두 검사한다.

$$
\boxed{
\begin{aligned}
|F(\widetilde r)-\varepsilon|&\le\tau_p,\\
F(\widetilde r-\tau_r)&\le\varepsilon\le F(\widetilde r+\tau_r).
\end{aligned}}
$$

두 번째 조건은 정확한 함수 평가에서 단조성에 의해 `abs(r_tilde-r_star)<=tau_r`를 뜻한다. 첫 번째 조건이나 작은 Newton correction만으로 성공을 판정하지 않는다. Local bracket의 부호 검사는 probability tolerance만큼 느슨하게 하지 않는다. FP64에서 실제 구성한 양 끝점이 유한하고 candidate의 양쪽에 있는지도 확인한다. Outward rounding을 사용했다면 실제 endpoint와 candidate의 최대 거리를 `root_error_bound`에 기록한다. 이 값은 floating-point 함수 평가에 따른 numerical check이며 interval arithmetic 기반의 엄밀한 인증을 뜻하지 않는다.

Upper-bound Newton은 정확한 root에 접근해도 초기 lower endpoint가 거의 움직이지 않을 수 있다. 따라서 기존 `root_bracket_width`는 추적 중인 global bracket의 진단값으로만 유지한다. 최종 정확도는 local bracket으로 확인하며, 성공한 stage의 `root_error_bound [m]`와 실제 Newton update 수 `root_iterations`를 추가로 기록한다. 미평가 stage와 최종 검사를 통과하지 못한 stage의 `root_error_bound`는 결측으로 남긴다.

초기 설정 후보는 `mf_imp_root_max_iter=12`, `mf_imp_root_atol_m=1e-8`, `mf_imp_root_rtol=1e-10`, `mf_imp_root_prob_atol=1e-10`이다. `max_iter`는 Newton update 횟수이며 마지막 candidate 검사는 별도다. 12회 수렴을 보장하는 설정이 아니라 검증 전 출발값이다. 미수렴하면 반복 수를 자동 증가시키거나 다른 solver로 전환하지 않고 실패 정보를 남겨 중단한다. Geometry 자체의 float32 오차가 root 오차보다 클 수 있으므로 이 tolerance를 실제 clearance 정확도라고 해석하지 않는다. 매우 작은 sigma나 극단적인 epsilon에서 두 종료 조건을 만족하지 못하면 실패 상태를 반환하며 성공을 가정하지 않는다.

Eager 실행만 구현한다. CUDA Graph 및 `torch.compile` 적용, custom kernel fusion, warm-start root 정책은 이번 범위에서 제외한다. 정상 경로의 CPU 전송은 stage 단위 결과를 회수할 때로 모으며, 실패가 확인된 뒤 debugging을 위한 추가 동기화·snapshot 저장은 허용한다.

Weights는 직접 `pdf/sigma`를 합한 뒤 작은 상수로 나누기보다

$$
\log a_\ell=-\tfrac12\zeta_\ell^2-\log\sigma_\ell-\tfrac12\log(2\pi),
\qquad w=\operatorname{softmax}_{\rm valid}(\log a)
$$

로 계산한다. Invalid slot은 연산 전 유한 dummy 입력으로 치환하고 reduction에서 제외한다. 모든 slot이 invalid인 경우는 softmax 전에 별도 분기한다. Sigma floor, 확률 clipping, feature pruning, denominator epsilon 추가로 원래 모델을 바꾸지 않는다.

Inverse CDF와 complementary error function에는 PyTorch의 `torch.special.ndtri`, `torch.special.erfc`를 사용할 수 있다. 실제 설치 버전과 GPU dtype 지원은 향후 구현 환경에서 확인한다. [PyTorch 공식 문서](https://docs.pytorch.org/docs/2.14/special.html)

### 5.3 Empty/invalid 처리와 fail-stop

**정상적인 비활성화와 계산 실패를 구분한다.** `mf_required`는 기존 obstacle/MF switch, obstacle 존재 여부, stage window·terminal switch 및 `phi>d_mf_mask`로 결정한다. Row G 평가가 필요한데 PSDF 값이나 gradient가 비유한 경우에는 기존 오류 처리로 넘기며, NaN 비교를 domain 비활성으로 해석하지 않는다. `mf_required` 판정 후 root와 계수를 검사한다.

| 상황 | 처리 |
|---|---|
| 활성 장애물 없음 | `F0=0`, root는 정의하지 않음. `margin=NaN`, `root_status=no_obstacles`, Row M 비활성. 정상적인 미평가이며 root 실패가 아님 |
| Stage 0, stage window 밖, terminal 제외, obstacle/MF switch off | 기존 정책대로 Row M 비활성. `root_status=not_requested`, root 평가 생략. 정상적인 미평가 |
| `phi<=d_mf_mask` | 기존 정책대로 Row M 비활성. `root_status=domain_inactive`. Unsigned feature margin으로 충돌/침투 구간의 안전성을 주장하지 않음 |
| Root가 필요한 stage에서 valid feature가 0개 | `root_status=no_valid_features`, `root_valid=False`. **제어 중단** |
| 필요한 stage의 sigma 0/비유한, valid feature/risk 중간값·bracket·계수 비유한 | 구체적 실패 사유와 stage를 기록하고 **제어 중단**. Invalid padding은 기존 mask 처리 대상이며 이에 포함하지 않음 |
| Newton derivative/step 오류, bracket 이탈, stagnation, 최대 반복 후 최종 검사 실패 | `root_valid=False`, 구체적 `root_status`와 마지막 candidate 정보를 기록하고 **제어 중단** |
| 유효하게 계산된 `r_bar<0` | 정상적인 constraint 부족량. Root failure가 아니며 기존 soft Row M을 그대로 구성 |
| 정상 비활성 row의 solver 값 | `A=0,c=1.0 m`의 유한 양수 상수. `epsilon`을 거리 값으로 재사용하지 않음. **계산 실패를 이 상수로 대체하지 않음** |

OCP nominal `z_bar` 또는 dynamics 입력 자체가 비유한 경우는 기존 전역 validation 오류 처리를 유지한다. 유한한 nominal에서 발생한 feature/risk 계산 실패도 더 이상 stage mask로 숨기지 않는다. 유효 모델에서는 수학적으로 root가 존재하므로 여기서의 실패는 모델 입력 또는 수치 계산 실패이며, `r<0`인 chance constraint 위반과 구분한다.

**Fail-stop 계약.** GPU evaluator는 stage별 status와 진단값을 반환하고, controller 연결부는 필요한 stage 중 실패가 하나라도 있으면 `MFImplicitRootError`와 같은 전용 예외로 전파한다. Batch 호출 자체의 오류도 implicit 경로에서는 자동 stagewise retry나 pose-only guard 복구로 숨기지 않고 원인을 보존해 중단한다. 실패한 평가의 solver parameter를 설치하거나 QP를 실행하지 않는다. 일반 solver recovery를 호출하거나 이전 cycle의 입력·root·계수를 재사용하여 계속 주행하지 않는다.

최상위 제어 실행부는 오류를 일반 `safe_stop` 후 다음 cycle 재시도로 바꾸지 않는다. **현재 run/제어 loop를 중단하고, 실패 로그를 보존한 뒤 명시적인 reset 또는 재시작 전에는 정상 제어를 재개하지 않는다.** 시뮬레이션에서는 run을 중단한다. 실제 actuator에 연결된 실행부에서는 진단 dump보다 기존 정지 명령 전달을 우선하여 이전 명령이 계속 유지되지 않도록 한 뒤 예외를 전파한다. Python exception 발생만으로 물리적 정지가 보장된다고 가정하지 않는다.

기존 CC-MF의 batch/stagewise/guard fallback은 변경하지 않는다. CC-MF-IMP에서도 **root와 계수가 유효한 상태에서 발생한 일반 QP 실패**에 대한 기존 recovery는 유지하되, recovery 중 implicit 평가가 실패하면 동일하게 중단한다. 새로운 nonlinear chance-condition acceptance veto는 추가하지 않는다. Debug 재평가에서도 평가가 요청된 유효 domain의 root 계산 실패는 숨기지 않고 구체적 평가 context와 함께 중단하며, 단순 nonlinear constraint 위반과는 구분한다.

## 6. Slack의 단위와 비용

Row M의 lower slack `delta`는 이제 **미터**다. Lower slack penalty는

$$
P(\delta)=w_1\delta+\tfrac12w_2\delta^2
$$

이며 `w1`은 cost/m, `w2`는 cost/m² 단위다. 기존 probability-unit slack과 같은 수치를 써도 같은 최적화 문제가 아니다. 비선형 margin에 slack을 넣었다면 `r(z)+delta>=0`은 threshold를 `d_min-delta`로 낮춘 확률 조건과 동치이지만, 실제 제약은 affine `r_hat`이므로 그 해석도 근사에 한정된다.

첫 구현은 subclass의 독립 설정으로 기존 필드명 `mf_slack_linear`, `mf_slack_quadratic`를 사용하고, 기본 숫자 `100`, `1`은 **재튜닝이 필요한 임시 시작값**으로 명시한다. 기존 `cc_mf` YAML의 probability slack 값을 새 컨트롤러에 자동 상속하지 않는다. 최종 실험 전에는 원하는 거리 위반 비용을 기준으로 결정해야 한다.

확률 비용과의 국소 비교가 필요하면 기준 경계에서 `kappa_ref=F_r(z_ref,0)`를 구해 `w1_imp≈kappa_ref*w1_MF`, `w2_imp≈kappa_ref^2*w2_MF`로 초기 보정할 수 있다. 이는 기준점 부근의 보정이며 전역 동등 변환이 아니다. 매 stage마다 자동 재가중하는 정책은 이번 구현에 넣지 않는다.

`idxsh=[1]`을 유지해 Row G에 slack이 붙지 않게 한다. acados는 lower/upper slack을 별도로 다루며, `lsh/ush`는 slack의 하한이다. 실제 위반 분석은 lower slack을 읽는다. `zl/Zl` 및 terminal 항에 거리 단위 penalty를 적용하고 upper-bound 측 설정도 기존 구조와 일관되게 유지한다. [acados Python interface](https://docs.acados.org/python_interface/)

## 7. API와 최소 코드 변경

### 7.1 새 출력 계약

기존 `forward_mf(...) -> (phi,g,A_mf,c_mf)`는 유지한다. 새 `forward_mf_imp(...)`는 이름 있는 결과 구조를 반환한다.

```text
MFImplicitResult
  phi:                 (H,)       signed PSDF [m]
  gradient:            (H,3)      world-frame PSDF gradient
  A:                   (H,8)      implicit residual coefficients
  c:                   (H,)       implicit residual constants [m]
  margin:              (H,)       nominal r_bar [m]
  probability_sum:     (H,)       F(z_bar,0), dimensionless
  root_residual:       (H,)       F(z_bar,r_bar)-epsilon
  root_bracket_width:  (H,)       tracked global bracket width [m]; diagnostic only
  root_error_bound:    (H,)       accepted local-bracket radius [m]; NaN if unavailable
  root_iterations:     (H,)       actual Newton update count; 0 if skipped/closed-form
  feature_count:       (H,)       valid count
  root_valid:          (H,)       bool
  root_status:         (H,)       status codes
  frozen_features:     optional   GPU-only snapshot for diagnostics
```

MF solver parameter 수는 여전히 9개다. 나머지는 stage 단위 진단 metadata이며 OCP parameter에 넣지 않는다. API 내부에서는 geometry dtype와 risk dtype를 분리하고, `A,c,margin,probability_sum`은 float64로 반환한다. `phi,g`는 기존 geometry dtype을 허용한다.

Root 평가 필요 여부를 전달할 stage mask를 새 API에 명시하여 정상 비활성 stage를 구분한다. `root_valid=False`만으로 실패 여부를 판단하지 않고 요청 여부와 `root_status`를 함께 사용한다. 필요한 stage가 실패하면 `A,c`를 inactive sentinel로 바꾸어 solver에 넘기지 않는다. 해당 stage의 유효하지 않은 `margin,A,c`는 결측으로 남기고 마지막 Newton candidate는 실패 진단에만 보존한다.

### 7.2 파일별 작업

| 파일 | 작업 |
|---|---|
| `models/augmented_psdf.py` | 현재 `_forward_impl(...,return_features=True)` 재사용. Variance 및 variance-gradient 공통 helper 추출. Eager batched upper-bound Newton 기반 `_solve_mf_margin`, `_compute_mf_imp_affine`, `forward_mf_imp` 추가. 기존 `_compute_A`·`forward_mf` 결과 보존 |
| `models/augmented_psdf_wrapper.py` | 새 API forwarding, obstacle-empty 처리, double risk 입력과 geometry 입력 정규화 분리. `forward_mf` tuple 계약 유지 |
| 새 `control/cc_mf_imp_optimizer.py` | `CCMFImpOptimizerParam(CCMFOptimizerParam)`, `CCMFImpOptimizer(CCMFOptimizer)` 추가. `_codegen_name="cc_mf_imp"`, implicit kernel, Newton 설정, 거리 slack 및 진단 label 설정. 전용 implicit 평가 예외와 fail-stop 연결 |
| `control/cc_mf_optimizer.py` | Kernel 호출, 정상 비활성 row 상수, nominal metric 수집, 진단 label 및 평가 실패 정책을 작은 overridable hook으로 분리. 기존 CC-MF fallback 유지. Implicit 전용 예외는 generic fallback에서 재전파하여 stagewise/guard-only 복구를 건너뜀. OCP의 `A·z+c` 구조 재사용 |
| `utils/cc_mf_diagnostics.py` | Residual 종류/단위 metadata, Newton iteration 및 local-bracket 진단, root 실패 상태, nominal/affine/nonlinear 값 분리. 중단 시 최소 재현 snapshot 저장. 필요하면 implicit 전용 mixin/helper 추가 |
| `control/__init__.py` | `SUPPORTED_OPTIMIZER_TYPES`에 canonical `cc_mf_imp` 등록 |
| `sim/simulation_mpc.py` | Factory/config, output prefix, PSDF obstacle update 대상, CSV 저장 분기에 variant 추가. Implicit 평가 예외는 실패 로그를 보존하고 현재 run을 중단하며 일반 safe-stop/retry로 변환하지 않음 |
| `test_nmpc.py` 및 공통 variant resolver | CLI choices 이전에 `cc-mf-imp`를 `cc_mf_imp`로 정규화. YAML/factory도 같은 별칭 처리. 모든 method 이름을 무차별 변환하지 않음 |
| 새 example config | `cc_mf_imp`만의 override 예제. 현재 사용자가 선택한 `config/config.yaml` 실행 설정은 바꾸지 않음 |
| `method_description.md` | 구현 완료 단계에서 variant 수식 및 실제 stage 정책 반영. 이번 계획 작성에서는 수정하지 않음 |

Factory는 `CCMFImpOptimizerParam()` 기본값 → `config['cc_mf_imp']` 순서로 적용한다. `cc_mf` 블록의 묵시적 상속은 하지 않는다. 공통 설정은 필요할 때 YAML에서 명시한다. `control/controller.py`는 현재 duck typing 흐름으로 충분한지 확인하고 불필요한 분기는 추가하지 않는다. 단, generic exception 처리에서 implicit 평가 오류가 정상 입력 반환이나 다음 cycle 자동 재시도로 흡수되지 않도록 전용 예외 전파 및 기존 정지 인터페이스 연결을 확인한다.

### 7.3 반드시 수정할 내부 연결

1. `_compute_mf_affine_data`: probability/implicit kernel 선택을 hook으로 분리한다. 기존 CC-MF의 batch fallback과 stagewise retry는 유지하되, CC-MF-IMP의 필요한 stage 계산 실패 또는 batch exception은 전용 예외로 전파한다. Implicit 경로의 자동 stagewise retry와 pose-only guard 복구는 실행하지 않는다. Root metadata는 명시적인 반환 데이터로 전달한다.
2. `_compute_constraint_affine_data`: 기존 domain/stage 정책으로 `mf_required`를 먼저 계산한다. 필요한 stage에서 `root_valid=False`이거나 계수가 유효하지 않으면 중단하며, `mf_valid=False`로 바꾸어 행을 제거하지 않는다. 정상 비활성 stage에만 mode별 inactive 상수를 적용한다.
3. `_set_constraint_parameters`: stage-0 inactive 상수도 같은 hook을 사용한다. Terminal까지 기존 parameter 크기를 유지한다.
4. `_prepare_and_solve`: 기존 `probability_sum = epsilon - (A_raw @ z_bar + c_raw)`는 CC-MF에서만 유효하다. 새 mode에서는 결과의 `probability_sum=F(z_bar,0)`를 사용하고 `margin`은 별도 보관한다. 필요한 모든 stage의 root 및 계수 검증이 끝나기 전에 solver parameter를 설치하거나 `solver.solve()`를 호출하지 않는다.
5. `_try_backup_recovery`, `cleanup/reset`: nominal root/확률 metadata를 기존 PSDF·constraint cache와 같은 수명으로 저장·복원·초기화한다. Backup이 enabled라면 implicit 행을 사용해야 한다. Root failure를 이유로 backup을 호출하지 않으며 backup 내부의 implicit 평가 실패도 재전파한다. 실패 snapshot은 cleanup 전에 보존하고, 중단한 실행은 명시적 reset 또는 재시작으로만 재개한다.
6. `_record_experiment_logs` 및 CSV export: 확률과 거리 residual/slack 열을 분리한다. Safe-stop에서 최적화된 slack이 없는 경우 기존처럼 결측으로 표시한다. Implicit 계산 실패로 중단한 cycle은 별도 원인으로 기록하고, 새 QP가 실행되지 않았다면 solver 시간·상태 및 slack을 이전 cycle 값으로 채우지 않는다.
7. `_codegen_name` 기반 임시 생성 폴더를 재사용하되 controller 인스턴스마다 분리하고 기존 cleanup 경로를 따른다. 현재 `create_model()`의 `model.name="differential_drive_cc_mf"`도 hook으로 분리해 새 model 이름을 `differential_drive_cc_mf_imp`로 지정한다. `_codegen_name` override만으로 model 이름까지 바뀐다고 가정하지 않는다. 기존 CC-MF solver 파일을 덮어쓰지 않는다.

## 8. 진단과 시간 측정

정상 주기의 최소 로그는 실제 `mf_probability_sum_nominal`, `mf_imp_margin_nominal_m`, `mf_imp_slack_m`, `mf_active`, `root_valid`, `root_status`, `root_residual`, `root_bracket_width_m`, `root_error_bound_m`, `root_iterations`이다. `root_bracket_width_m`는 global bracket의 진단값이며 Newton 수렴 판정값이 아니다. Margin이 정의되지 않은 stage는 결측으로 남긴다. 기존 `mf_probability_sum` 열을 유지한다면 값은 반드시 nominal `F0`이며 그 의미를 문서화한다.

중단 시에는 cycle/stage, 평가 context(nominal/backup/debug), 실패 사유, nominal state, epsilon, 유효 feature 수, bracket, 마지막 candidate·`F,F_prime`·iteration 수와 tolerance를 기록한다. 가능한 경우 실패 stage의 `d,q,mask,sigma`를 별도 snapshot으로 저장하여 재현할 수 있게 한다. 실패 경로에서만 해당 feature 배열의 CPU 전송을 허용한다. Snapshot 저장 실패가 원래 계산 예외를 덮어쓰지 않게 한다. Root가 필요 없는 정상 비활성 stage는 root 실패 횟수에 포함하지 않는다.

Debug 모드에서는 returned state `z_star`에 대해 다음을 구분한다.

| 진단 | 의미 |
|---|---|
| `r_bar` / `F_bar(0)` | solve 전 nominal |
| `A_bar @ z_star+c_bar` 및 lower slack | 실제 설치한 QP 행의 반환점 residual |
| `r_frozen(z_star)` / `F_frozen(z_star,0)` | QP를 만들 때의 `d_bar,q_bar,g_bar,mask,theta_bar`를 유지하고 distance affine model 및 새 `xi_star`로 재평가 |
| `r_fresh(z_star)` / `F_fresh(z_star,0)` | 반환 pose에서 geometry·feature를 다시 추출한 근사 chance model |
| Exact signed PSDF | 별도의 geometric separation 검사 |

현재 diagnostics의 `_compute_constraint_affine_data(states)` 호출은 fresh geometry 재평가다. Frozen-model 검사에는 nominal GPU feature snapshot을 별도 보존해야 한다. 이 snapshot은 debug에서만 유지하고 CPU로 feature 배열 전체를 옮기지 않는다. 진단 호출은 순수 반환값을 사용하여 nominal 로그 cache를 덮어쓰지 않는다.

공분산 dynamics defect를 분리하려면 별도 검증에서 동일 반환 입력의 nonlinear rollout covariance로도 재평가한다. 이 결과와 OCP의 `xi_star`로 평가한 결과를 구별한다. Nonlinear 검증은 진단이며 첫 구현에서 새로운 제어 acceptance 정책을 추가하지 않는다. 다만 요청된 debug root 평가 자체가 실패한 경우의 중단은 §5.3의 수치 오류 정책을 따른다. 유효한 음수 margin 또는 nonlinear chance-model 위반을 root failure로 처리하지 않는다.

기존 `computation_times_*.csv` 정의는 유지한다. 추가 비용 비교에는 geometry/root/계수 생성의 preprocessing 시간과 전체 제어 주기 시간도 별도로 기록한다. GPU wall time을 잴 때는 완료까지 포함하도록 timing 경계를 정의하고, device 실행 시간은 CUDA event 등으로 별도 측정한다. Root를 solver 밖으로 옮겼다는 이유로 solver 시간만으로 속도 향상을 결론내리지 않는다. 이번 측정은 PyTorch eager 경로를 대상으로 하며 CUDA Graph replay나 compilation 초기 비용은 포함되지 않는다. Failure snapshot 저장 시간은 정상 cycle latency와 분리한다.

## 9. 구현 순서와 향후 검증

다음은 **향후 구현·실행 단계의 계획**이다. 저장소 지침에 따라 이번 작업에서는 테스트 또는 benchmark를 실행하지 않는다.

### 단계 A — 수학 kernel

공통 feature/variance helper를 추출하고 eager upper-bound Newton root 및 analytic coefficient 함수를 추가한다. Controller와 무관한 frozen feature 입력으로 정확성을 먼저 확인할 수 있게 구성한다.

검증할 성질:

- 단일 feature의 닫힌형 `d-d_min-gamma*sigma` 및 pose/covariance gradient와 일치.
- 여러 feature의 `F(r_lo)<=epsilon<=F(r_hi)`, root 유일성, root residual.
- `0<epsilon<0.5`에서 upper-bound Newton의 감소 방향과 reference root를 넘지 않는 성질을 수치 허용 오차 내 확인. `M=1`은 Newton update 수 0.
- 마지막 update 이후 candidate의 재평가, probability residual 및 local bracket의 동시 판정. 초기/global bracket이 넓게 남아도 정확한 Newton candidate는 통과.
- 잘못된 epsilon/tolerance 설정, derivative 오류, bracket 이탈, stagnation, 반복 한도와 최종 검사 실패가 명시적인 실패로 연결됨. 미수렴 upper iterate를 성공 margin으로 반환하거나 bisection으로 자동 전환하지 않음.
- 허용 오차 밖에서 `F(z,0)<=epsilon`과 `r>=0`의 부호 일치.
- 모든 distance의 공통 shift에 대한 정확한 이동 성질.
- 동일 feature `M`개 중복 시 `r=d-d_min+sigma*Phi^-1(epsilon/M)`와 일치. 중복이 위험 합에 반영되는지 확인.
- `r_bar=0`에서 기존 MF gradient와 `A_imp=A_MF/F_r` 관계.
- Frozen `q,g,mask,frame` 상태에서 pose 3개·covariance 4개 변수의 중앙차분과 analytic gradient 비교. Geometry를 재선택하는 finite difference와 혼동하지 않음.
- 비영 `xi_c`, 여러 heading, stage별 epsilon, 서로 다른 sigma, 작은 tail, 극단 입력의 실패 처리.
- Padding NaN/inf 제외, no-obstacle, no-valid-feature, nonpositive variance 및 root 미수렴 처리.

### 단계 B — wrapper와 controller

새 API와 subclass, 공통 hook을 연결한다. Mock solver로 아래 계약을 확인하고, 실행 환경에서 acados codegen 검증을 이어간다.

- 상태 8, 입력 3, Row G/M 두 행, MF parameter 9개 유지.
- Row G hard, Row M만 lower slack으로 완화. 중간/terminal 비용 모두 거리 단위.
- Stage 0 비활성 및 exact-PSDF veto, strict `phi>d_mf_mask`, stage window, terminal switch 유지.
- Master obstacle switch와 `use_row_mf=False`, no-obstacle에서 불필요한 root 평가 생략.
- 필요한 stage 하나에 root/계수 실패를 주입했을 때 solver parameter 설치·QP 실행·guard-only continuation·stagewise retry·backup이 수행되지 않고 run이 중단됨. Failure status·snapshot이 보존됨.
- 정상 비활성 stage는 fail-stop을 유발하지 않으며, 유효한 음수 root는 soft Row M으로 전달됨.
- Generic exception/recovery 경로가 implicit 전용 예외를 삼키지 않음. 자동 다음 cycle 재개나 이전 입력 재사용이 없고 명시적 reset/재시작이 필요함.
- Eager root loop 내부에 CPU tensor-dependent 종료 또는 CUDA Graph/`torch.compile` 경로가 없음.
- `F0`가 `epsilon-r`로 계산되지 않음. Inactive 상수와 결측 margin이 구분됨.
- Debug fresh/frozen 재평가 및 backup 실패 복원이 nominal cache를 손상하지 않음.
- 기존 `test_cc_mf`, stage0, PSDF logging 및 augmented PSDF 검증의 회귀 없음.

### 단계 C — 선택·출력 통합

Canonical 이름과 별칭으로 동일 variant가 생성되는지 확인한다. 설정 override 격리, 출력 파일 prefix, obstacle update, CSV 생성 및 solver timing 수집까지 연결한다. 별도 model/codegen 이름과 cleanup도 확인한다. 최상위 실행부까지 implicit 평가 예외가 전파되고 실패 로그를 보존한 채 run이 종료되는지 확인한다.

### 단계 D — 사용자가 실행을 요청한 후 비교

동일 geometry, 공분산, nominal state에서 CC-MF와 CC-MF-IMP의 raw `F0`가 수치 허용 오차 내 일치하고 row만 달라지는지 확인한다. 단일 feature 조건에서는 CC-SF와 residual/Jacobian을 비교하되 guard·slack 차이는 별도로 기록한다.

이후 동일 seed·시작 pose·환경·동역학 설정으로 실험한다. Success rate, solver 실패, safe-stop, clearance, 거리 slack, nonlinear chance-model 위반, root 실패, preprocessing 및 전체 주기 시간을 함께 비교한다. Root 실패로 중단된 run은 정상 완료나 기존 safe-stop 복구로 집계하지 않고 별도 중단 원인으로 기록한다. 실행은 기존 순차 정책과 새로운 results directory를 사용한다. Slack penalty가 다른 단위라는 점을 명시하고, 검증 전에는 안전성·성공률·속도 개선을 확정하지 않는다.

## 10. 완료 기준

- 기존 `cc_mf`를 유지하면서 `cc_mf_imp`를 독립 선택할 수 있다.
- 새 계수는 root에서 계산한 analytic implicit derivative이며 frozen nominal geometry 정의와 일치한다.
- Solver 내부에 root finder나 feature별 추가 decision variable을 넣지 않는다.
- Probability, margin, slack, inactive sentinel의 의미와 단위가 구분된다.
- Stage/mask/guard 및 일반 QP recovery 정책은 유지하되, 필요한 implicit root/계수 실패는 guard-only continuation 없이 제어 중단으로 연결되고 재현 가능한 로그가 남는다.
- Root는 `r_hi` 초기값의 eager batched Newton으로 계산하며, 최종 probability residual과 local bracket 검사를 통과한 candidate만 사용한다. Bisection 재시도, CUDA Graph 및 `torch.compile` 경로를 추가하지 않는다.
- 향후 허가된 검증에서 수학 성질·회귀·통합 검사가 확인된 뒤 실험으로 넘어간다.

현재 산출물은 이 계획 문서이며 위 구현 및 실행 항목은 아직 수행하지 않았다.
