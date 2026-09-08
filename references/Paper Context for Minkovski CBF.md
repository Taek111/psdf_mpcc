# CBF via Minkowski Operations — 구현 컨텍스트 문서

**대상 논문**: Y.-H. Chen, S. Liu, W. Xiao, C. Belta, M. Otte, *"Control Barrier Functions via Minkowski Operations for Safe Navigation among Polytopic Sets"*, arXiv:2504.00364v3

**원 논문 구현**: MATLAB, `quadprog`(SDF + CLF-CBF-QP), `ode45`, 100 Hz

**목표 스택**: Python + CasADi + acados 이 문서는 **논문의 수식을 그대로** 구현하기 위한 컨텍스트다. 폐형식 재정식화나 근사 대체를 하지 않고, 논문과 동일하게 매 제어 주기마다

**CO 구성 → QP 또는 LP 풀이 → KKT 미분** 파이프라인을 돈다. 논문이 명시하지 않아 구현자가 판단해야 하는 지점은 ⚠️ 로 표시했다.

---

## 0. 요약

다각형 로봇 $\mathcal{R}$과 다각형 장애물 $\mathcal{O}$ 사이의 **정확한 signed distance (SDF)** 를 CBF로 쓴다. 핵심은 "두 볼록집합 사이의 거리" 문제를 Minkowski 차 $\mathcal{O}^\mathcal{C}=\mathcal{O}\oplus(-\mathcal{R})$ 가 사는

**MD-space**에서 "원점 $0_c$ 와 볼록집합 하나" 사이의 문제로 바꾸는 것이다.

- $0_c \notin \mathcal{O}^\mathcal{C}$ (분리) → **QP** 로 최소거리
- $0_c \in \mathcal{O}^\mathcal{C}$ (충돌) → **LP** 로 침투깊이 두 경우 모두 최적해 $z^*$ 의 $x$ 에 대한 야코비안을 구해 $\dot h$ 를 만들고, CLF-CBF-QP에 넣는다. 근사(구/타원/다항식/하한)를 쓰지 않는 것이 논문의 핵심 주장이므로, 구현에서도 근사를 넣지 않는다.

---

## 1. 표기법 및 문제 설정

| 기호 | 의미 |
|---|---|
| $\mathcal{W}$ / $\mathcal{M_D}$ | workspace / Minkowski-difference space. 본 구현은 **$d=2$** |
| $\mathcal{R}(x)$, $\mathcal{O}_i$ | 로봇 다각형(상태 의존), $i$번째 정적 장애물 다각형 |
| $\mathcal{O}^\mathcal{C}_i(x)$ | Configuration Obstacle $=\mathcal{O}_i\oplus(-\mathcal{R}(x))$ |
| $A^\mathcal{C}\in\mathbb{R}^{\ell_\mathcal{C}\times d}$, $b^\mathcal{C}\in\mathbb{R}^{\ell_\mathcal{C}}$ | CO의 H-rep: $\{y: A^\mathcal{C}y\le b^\mathcal{C}\}$ |
| $\ell_r,\ell_o,\ell_\mathcal{C}$ | 로봇/장애물/CO의 변 개수. a.e. $\ell_\mathcal{C}=\ell_r+\ell_o$ |
| $0_c$ | MD-space 원점 |
| $z^*$ | critical point ($\partial\mathcal{O}^\mathcal{C}$ 위, 원점에 가장 가까운 점) |
| $\lambda^*$ | QP/LP의 dual 변수 |
| $k_{act}$ | active constraint 인덱스 집합 |
| $\xi=(z,\lambda)$ | KKT 변수 묶음 |

로봇/장애물은 H-rep으로 주어진다.

$$\mathcal{R}(x)=\{y: A_r(x)y\le b_r(x)\},\qquad \mathcal{O}_i=\{y: A_{o_i}y\le b_{o_i}\}$$

SE(2) 강체 로봇의 H-rep 파라미터화 (논문 IV-D):

$$A(x)=A\,R(\theta-\theta_0)^\top,\qquad b(x)=b+A(x)p-Ap_0,$$

$p=[x,y]^\top$, $p_0=[x_0,y_0]^\top$, $R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}$, $A(x_0),b(x_0)$ 는 초기 자세에서의 로봇 형상.

**논문의 가정**

1. $\mathcal{R}$ 은 강체, uniformly bounded, 내부 nonempty ($\forall x\in\mathcal{X}$).
2. $A^\mathcal{C}(x), b^\mathcal{C}(x)$ 는 a.e. 미분가능하고 **minimal representation**. 거의 모든 $x$ 에서 $\ell_\mathcal{C}=\ell_r+\ell_{o_i}$. 로봇과 장애물에 평행한 변이 있으면 행이 하나 이상 줄어드는데, 이는 measure-zero 집합에서만 발생.
3. 모든 장애물은 정적, uniformly bounded, 내부 nonempty, 정보는 로봇이 접근 가능. **Problem 1**: 아핀 동역학 $\dot x=f(x)+g(x)u$ 의 로봇에 대해 $J=\int_0^T u^\top u\,dt$ 를 최소화하면서 목표 $(x_d,y_d)$ 에 도달하고, $\text{sd}(\mathcal{R},\mathcal{O})>d_{\text{safe}}$ 와 $u_{\min}\le u\le u_{\max}$ 를 만족하는 $u^*(x)$ 를 구한다.

---

## 2. 이론 (논문 그대로)

### 2.1 CBF / CLF

$\dot x=f(x)+g(x)u$, $x\in\mathcal{X}\subset\mathbb{R}^n$, $u\in\mathcal{U}\subset\mathbb{R}^q$. 안전집합 $\mathcal{C}=\{x\in\mathcal{X}: h(x)\ge0\}$ 에 대해 $h$ 가 CBF이려면 extended class-$\mathcal{K}_\infty$ 함수 $\alpha$ 가 존재해

$$\sup_{u\in\mathcal{U}}\big[L_fh(x)+L_gh(x)u+\alpha(h(x))\big]\ \ge\ 0 .$$

본 구현은 $\alpha(h)=\gamma h$. 정리 1에 의해 이를 만족하는 Lipschitz 제어기는 $\mathcal{C}$ 를 전방불변으로 만든다. CLF: $c_1\|x\|^2\le V(x)\le c_2\|x\|^2$ 이고 $\inf_u[L_fV+L_gVu+c_3V]\le0$.

**Remark 1 (논문)**: 이 SDF는 공간변수에 대해 $C^1$ 이지만 **방향($\theta$)에 대해서는 a.e.만 미분가능**하고 (active-set 전환), 그것도 로봇과 장애물이 충돌하지 않은 경우에 한한다. 접촉/침투 시에는 locally Lipschitz이므로 역시 a.e. 미분가능하며, 비미분점은 measure-zero (꼭짓점, 내부 medial axis, 회전에 의한 active-set 전환).

**정의 2의 고전적 CBF는 그 점들에서 적용되지 않는다.** 논문의 증명은 비충돌 + 단일적분기 케이스에만 엄밀히 적용되며, 충돌/회전 케이스는 conjecture로 남아 있다 (엄밀한 nonsmooth 해석은 future work).

### 2.2 Minkowski 연산과 CO

$$\mathcal{A}\oplus\mathcal{B}=\{a+b\mid a\in\mathcal{A}, b\in\mathcal{B}\},\qquad
\mathcal{C_R}(\mathcal{O})=\mathcal{O}\oplus(-\mathcal{R}),\quad -\mathcal{R}=\{-p\mid p\in\mathcal{R}\}$$

- **Lemma 1**: $\mathcal{R}\cap\mathcal{O}\ne\emptyset \iff 0_c\in\mathcal{C_R}(\mathcal{O})$. **볼록집합에서만 성립.**
- **Lemma 2**: 볼록 다각형 $\mathcal{R},\mathcal{O}$ (변 $\ell_r,\ell_o$)의 CO는 변이 $\le\ell_r+\ell_o$ 인 볼록 다각형이고 $O(\ell_r+\ell_o)$ 시간에 계산된다.
- 비볼록 장애물은 볼록 분해 후 (볼록로봇 × 볼록장애물) 쌍마다 CO를 만들고 Lemma 1을 반복 적용.

### 2.3 CO의 실제 계산 — 매 제어 주기의 첫 단계

논문은 Lemma 2를 인용할 뿐 알고리즘을 적지 않는다. 회전이 있는 경우 논문은 "각 시간 스텝마다 현재 자세로부터 CO를 선형시간에 계산한다"고만 밝힌다. 구현 절차:

1. 현재 상태 $x$ 에서 로봇 H-rep $A_r(x),b_r(x)$ → V-rep(꼭짓점) 변환, 반사하여 $-\mathcal{R}(x)$ 획득.
2. $\mathcal{O}\oplus(-\mathcal{R}(x))$ 를 **볼록 다각형 Minkowski 합** 알고리즘으로 계산. 두 다각형을 CCW로 정렬하고 최하단-최좌측 꼭짓점에서 시작해, 두 변 벡터열을 **극각 기준으로 머지**하며 누적하면 $\ell_r+\ell_o$ 개 변을 갖는 합 다각형이 $O(\ell_r+\ell_o)$ 에 나온다 (논문 참고문헌 [37], De Berg).
3. V-rep → H-rep: 연속한 꼭짓점 $w_k,w_{k+1}$ 에 대해 변 $e=w_{k+1}-w_k$, 외향법선 $a_k=[e_y,-e_x]/\|e\|$, $b_k=a_k^\top w_k$. ⚠️ **논문 미명시 — provenance 태깅 (구현 필수)** 회전 케이스에서 $d_xA^\mathcal{C},d_xb^\mathcal{C}$ 를 유한차분으로 구하려면 (§2.8), 섭동 전후 CO의 **행이 서로 대응**되어야 한다. 그런데 회전하면 초평면의 기울기와 순서가 모두 바뀐다(논문 IV-D). 따라서 CO의 각 facet이 **로봇의 몇 번째 변에서 왔는지 / 장애물의 몇 번째 변에서 왔는지** 를 태그로 들고 다녀야 한다. 위 머지 알고리즘은 각 출력 변이 어느 입력 변에서 왔는지 자연스럽게 알려주므로 태깅이 공짜다. 인덱스나 법선 각도로 행을 매칭하려 하면 실패한다. ⚠️ **평행 변 축퇴**: 평행한 변이 머지되면 $\ell_\mathcal{C}<\ell_r+\ell_o$ 가 되고 태그가 사라진다. 가정 2에 따라 measure-zero이지만, 유한차분 중에 발생하면 해당 방향은 단측차분으로 대체한다.

### 2.4 Signed distance — QP와 LP

$$\text{sd}(\mathcal{R},\mathcal{O})=\text{dist}(\mathcal{R},\mathcal{O})-\text{pd}(\mathcal{R},\mathcal{O})
=\text{sd}(0_c,\mathcal{O}^\mathcal{C})$$

$$\text{dist}(\mathcal{R},\mathcal{O})=\min\{\|t\|\mid (0_c+t)\cap\mathcal{C_R}(\mathcal{O})\ne\emptyset\},\qquad
\text{pd}(\mathcal{R},\mathcal{O})=\min\{\|t\|\mid (0_c+t)\cap\mathcal{C_R}(\mathcal{O})=\emptyset\}$$

**분기 판정 (Lemma 1)**: $0_c\in\mathcal{O}^\mathcal{C} \iff A^\mathcal{C}\cdot 0\le b^\mathcal{C} \iff b^\mathcal{C}\ge 0$ (성분별).

#### (A) 분리 → QP (논문 식 10)

$$\min_z\ \|z\|_2^2 \quad\text{s.t.}\quad A^\mathcal{C}z\le b^\mathcal{C},
\qquad \text{dist}(0_c,\mathcal{O}^\mathcal{C})=\|z^*\|_2$$

$z^*$ 는 원점의 $\mathcal{O}^\mathcal{C}$ 위로의 사영이다. 참고로 W-space에서의 동등한 문제는 식 (9): $\min\|x-y\|_2^2$ s.t. $A_rx\le b_r,\ A_oy\le b_o$ — **검증용으로만 쓴다.**

- Hessian이 $2I_d$ 인 **strictly convex QP** → 해 유일 (Lemma 3).
- ⚠️ **dual 스케일 주의**: 대부분의 QP 솔버는 $\tfrac12 z^\top Pz+q^\top z$ 형태를 받는다. §2.6의 KKT 식 $2I_dz+(A^\mathcal{C})^\top\lambda=0$ 과 $\lambda$ 스케일을 맞추려면 **$P=2I_d$** 로 넘겨야 한다. $P=I_d$ 로 넘기면 $\lambda$ 가 2배 어긋나 야코비안이 틀린다. **Lemma 3 (논문)**: $\mathcal{X}_f=\{x: \mathcal{R}(x)\cap\mathcal{O}=\emptyset\}$ 에서 $z^*(x)$ 는 유일하고, 연속이며 a.e. 미분가능. **Corollary 1**: $z^*(x)$ 의 Clarke generalized subdifferential은 $\mathcal{X}_f$ 에서 nonempty이고, measure-zero 집합을 제외하면 유일한 야코비안을 갖는다.

#### (B) 침투 → LP (논문 식 12)

침투깊이는 두 교차하는 볼록집합을 분리하는 **최소 이동 벡터(MTV)** 문제다. W-space에서 식 (11) $\min\|z\|_2$ s.t. $\mathcal{R}'\cap\mathcal{O}=\emptyset$ 는 분리 제약이 볼록이 아니라 못 푼다. MD-space에서 **점의 depth** 로 바꾸면 LP가 된다:

$$\text{depth}(0_c,\mathcal{O}^\mathcal{C})=\max_s\ s
\quad\text{s.t.}\quad s\ge0,\quad s\|A^\mathcal{C}_k\|_2\le b^\mathcal{C}_k,\quad k=1,\dots,\ell_\mathcal{C}$$

그리고 $\text{pd}(\mathcal{R},\mathcal{O})=\text{depth}(0_c,\mathcal{O}^\mathcal{C})=s^*$ (논문 참고문헌 [25],[34]). 결정변수는 스칼라 $s$ 하나뿐이다.

**분리 방향 복원.** LP는 크기 $s^*$ 만 주고 방향을 주지 않으므로 dual에서 뽑는다. LP의 강쌍대성 → complementary slackness → **nonzero dual $\lambda_{k_{act}}$ 가 active constraint에 대응**.

- 대부분의 경우 active constraint가 하나이고, 그 행 $a^\mathcal{C}_{k_{act}}$ 가 분리 방향을 유일하게 결정한다.
- 여러 개가 active면 $A^\mathcal{C}_{k_{act}}$ 가 행렬이 되고, 원점이 여러 초평면에서 등거리이므로 분리 방향은 그 법선들의 convex cone 안에 있다. 이는 measure-zero 상황이며, 실제로 발생하면 **아무 active constraint나 골라도 유효한 분리 방향**이다 (논문 IV-A). critical point는 그 초평면 $H=\{y: a^\mathcal{C}_{k_{act}}y=b^\mathcal{C}_{k_{act}}\}$ 위로 원점을 사영해 얻는다 (논문 식 13):

$$z^*=\text{proj}_H(0_c)=b^\mathcal{C}_{k_{act}}\big(a^\mathcal{C}_{k_{act}}\big)^\top\big/\big\|a^\mathcal{C}_{k_{act}}\big\|_2^2$$

LP 솔버는 dual/marginal을 반환해야 한다 (예: HiGHS는 부등식 제약 marginal 제공). $s^*>0$ 이면 stationarity에서 $\sum_k\lambda_k\|A^\mathcal{C}_k\|_2=1$ 이므로, $\lambda$ 가 가장 큰 인덱스를 $k_{act}$ 로 잡으면 된다.

### 2.5 CBF 정의 (논문 식 14)

$$h(x)=\text{sd}(\mathcal{R}(x),\mathcal{O})-d_{\text{safe}}=\text{sd}(0_c,\mathcal{O}^\mathcal{C}(x))-d_{\text{safe}},\qquad
\text{sd}=\begin{cases}+\|z^*(x)\|_2 & 0_c\notin\mathcal{O}^\mathcal{C}\\[2pt] -\|z^*(x)\|_2 & 0_c\in\mathcal{O}^\mathcal{C}\end{cases}$$

안전집합 $\mathcal{C}=\{x: \text{sd}(0_c,\mathcal{O}^\mathcal{C}(x))\ge d_{\text{safe}}\}$. $h$ 는 **암시적(implicit)** 이다 — 상태 $x$ 에 의존하는 최적화의 해로만 정의된다. 이 점이 §3의 acados 설계를 규정한다.

### 2.6 도함수 — 분리(QP) 경로

체인룰 (논문 식 15):

$$\frac{\partial h}{\partial x}=\frac{\partial h}{\partial z^*}\frac{\partial z^*}{\partial x},
\qquad \frac{\partial h}{\partial z^*}=\frac{{z^*}^\top}{h+d_{\text{safe}}}\in\mathbb{R}^{1\times d}$$

QP (10)의 KKT (stationarity + complementary slackness):

$$G(z,\lambda)=\begin{bmatrix}2I_dz+(A^\mathcal{C})^\top\lambda\\ D(\lambda)(A^\mathcal{C}z-b^\mathcal{C})\end{bmatrix}=0$$

$D(\cdot)$ 는 벡터를 대각행렬로 만드는 연산자. 솔버는 $G(\xi^*)=0$ 을 푸는 root-finder로 볼 수 있다. 음함수정리 (논문 식 16–17):

$$\partial_x\xi^*=-\big[\partial_\xi G(\xi^*)\big]^{-1}\partial_xG(\xi^*)$$

$$\partial_\xi G(\xi^*)=\begin{bmatrix}2I_d & (A^\mathcal{C})^\top\\ D(\lambda^*)A^\mathcal{C} & D(A^\mathcal{C}z^*-b^\mathcal{C})\end{bmatrix},\qquad
\partial_xG(\xi^*)=\begin{bmatrix}d_x(A^\mathcal{C})^\top\lambda^*\\ D(\lambda^*)\big(d_xA^\mathcal{C}\!\cdot\!z^*-d_xb^\mathcal{C}\big)\end{bmatrix}$$

- $d_xA^\mathcal{C}\in\mathbb{R}^{\ell_\mathcal{C}\times d\times n}$ 는 3-tensor, $d_xb^\mathcal{C}\in\mathbb{R}^{\ell_\mathcal{C}\times n}$.
- $d_x(A^\mathcal{C})^\top\lambda^*$ 는 $\ell$ 인덱스로 축약 → $\mathbb{R}^{d\times n}$.
- $d_xA^\mathcal{C}\!\cdot\!z^*$ 는 $d$ 인덱스로 축약 → $\mathbb{R}^{\ell_\mathcal{C}\times n}$.
- $\partial_x\xi^*$ 의 위쪽 $d$ 행이 $\partial z^*/\partial x$ 이고, 아래쪽은 dual 변수의 도함수다.
- 이 야코비안은 논문 참고문헌 [42] (Amos & Kolter, *OptNet*) 의 프레임워크 그대로다. **구현 전 이 논문을 참조할 것.** **Remark 5 (논문, 구현상 중요)**: (17)은 KKT를 미분해 얻었고 complementary slackness가 성립하므로 **도함수 계산에는 active constraint만 고려하면 충분하다.** 즉 $A^\mathcal{C},b^\mathcal{C}$ 를 $A^\mathcal{C}_{k_{act}},b^\mathcal{C}_{k_{act}}$ 로 대체할 수 있다. 이때 $D(A_{k_{act}}z^*-b_{k_{act}})=0$ 이므로 $\partial_\xi G$ 가 $\begin{bmatrix}2I_d & A_{k_{act}}^\top\\ D(\lambda_{k_{act}})A_{k_{act}} & 0\end{bmatrix}$ 로 줄어들고, 2D에서 $|k_{act}|\in\{1,2\}$ 이므로 $3\times3$ 또는 $4\times4$ 선형계가 된다. 전체 $\ell_\mathcal{C}$ 행을 그대로 쓰면 inactive 행의 $\lambda_k=0$ 때문에 $\partial_\xi G$ 가 특이해지므로, **축약형을 쓰는 편이 수치적으로 안전하다.** ⚠️ **특이점**: $\partial h/\partial z^*$ 의 분모 $h+d_{\text{safe}}=\text{sd}$ 는 접촉 순간 0이 된다. Remark 1이 말하는 measure-zero 비적용 지점이며 논문은 대응책을 주지 않는다. $d_{\text{safe}}>0$ 이면 안전집합 경계($h=0$)에서 $\text{sd}=d_{\text{safe}}>0$ 이라 실제 운용 구간에서는 피할 수 있다. $\|z^*\|$ 하한 tolerance는 별도로 두는 것이 안전하다.

### 2.7 도함수 — 침투(LP) 경로

**Remark 4 (논문)**: Lemma 3의 침투 케이스 확장은 future work다. 최소거리는 strictly convex QP지만 침투깊이는 **LP**여서 [42]의 미분가능성 결과가 직접 적용되지 않는다. 논문은 "LP 정식화가 경험적으로 잘 동작하고 동일한 MD-space 프레임워크 하에서 안정적인 도함수를 준다"고만 밝히고, 엄밀한 미분가능성 해석은 future work로 남긴다. ⚠️ **논문 미명시 — 침투 케이스에서 $\partial z^*/\partial x$ 를 실제로 어떻게 얻는가.** 본문은 "동일한 differentiable optimization 프레임워크"라고 하지만 IV-C는 QP의 KKT만 제시한다. LP의 Hessian은 0이라 (17)을 그대로 쓸 수 없다. 구현 가능한 해석은 하나뿐이다:

**$k_{act}$ 를 LP dual에서 확정한 뒤, 식 (13)을 $x$ 로 직접 미분한다.** $a:=a^\mathcal{C}_{k_{act}}$, $b:=b^\mathcal{C}_{k_{act}}$, $\hat n:=a^\top/\|a\|_2$ 라 두면 $z^*=b\,a^\top/\|a\|_2^2$ 이고

$$\frac{\partial z^*}{\partial x}
=\frac{a^\top}{\|a\|_2^2}\frac{\partial b}{\partial x}
+\frac{b}{\|a\|_2^2}\big(I_d-2\hat n\hat n^\top\big)\frac{\partial a^\top}{\partial x}$$

$\partial a^\top/\partial x$ 와 $\partial b/\partial x$ 는 $d_xA^\mathcal{C},d_xb^\mathcal{C}$ 의 $k_{act}$ 행이다 (§2.8). 이후 $\partial h/\partial x$ 는 §2.6과 동일한 체인룰로 얻되, 침투 시 $\text{sd}=-\|z^*\|$ 이므로 부호가 반영된다. 이는 논문이 제시한 식 (13)을 미분한 것이므로 새로운 근사가 아니다. active set이 바뀌는 순간에는 유효하지 않으며(measure-zero), Remark 4가 말하는 "엄밀성 미확보" 구간이 정확히 여기다.

### 2.8 $d_xA^\mathcal{C}$, $d_xb^\mathcal{C}$ 계산 (논문 IV-D)

CO의 경계 제약이 상태에 따라 어떻게 변하는지가 위 모든 야코비안의 입력이다.

**(a) 병진 성분 — 폐형식.** 정의 (6)에 의해 로봇이 $\delta p$ 만큼 평행이동하면 $-\mathcal{R}$ 은 $-\delta p$, 따라서 CO 전체가 $-\delta p$ 만큼 평행이동한다. 모양이 변하지 않으므로

$$\frac{\partial A^\mathcal{C}}{\partial p}=0_{\ell_\mathcal{C}\times d\times 2},\qquad
\frac{\partial b^\mathcal{C}}{\partial p}=-A^\mathcal{C}$$

논문은 이를 "순수 병진" 특수 케이스로 서술하지만, **회전이 있어도 위치 성분(열)에 대해서는 그대로 성립**한다 ($\theta$ 를 고정한 채 $p$ 만 미분하는 것이므로). Case I(단일적분기)은 이것만으로 끝난다.

**(b) 회전 성분 — 수치미분.** 회전하면 경계 초평면의 기울기와 순서가 모두 바뀌어 CO가 비자명하게 변형된다. Minkowski 합을 선형 사영으로 계산할 수는 있으나(논문 참고문헌 [45]) CO의 해석적 표현은 나오지 않으며,

**논문은 이 경우 각 시간 스텝마다 CO를 새로 구성하고 경계 제약의 도함수를 수치적으로 근사**한다. CO의 해석적 표현 도출은 논문이 명시한 open problem이다. 구현:

$$\frac{\partial A^\mathcal{C}}{\partial\theta}\approx\frac{A^\mathcal{C}(\theta+\delta)-A^\mathcal{C}(\theta-\delta)}{2\delta},\qquad
\frac{\partial b^\mathcal{C}}{\partial\theta}\approx\frac{b^\mathcal{C}(\theta+\delta)-b^\mathcal{C}(\theta-\delta)}{2\delta}$$

- 섭동마다 **CO를 완전히 재구성**하고, §2.3의 provenance 태그로 행을 매칭한다.
- 유니사이클 $x=[x,y,\theta,v]^\top$ 에서 CO는 $v$ 에 무관하므로 $\partial(\cdot)/\partial v=0$ 이다. 병진 열은 (a)의 폐형식이므로, **실제로 유한차분이 필요한 것은 $\theta$ 열뿐** → 제어 주기당 CO 재구성 2회(중앙차분).
- Remark 5에 따라 $k_{act}$ 행만 있으면 되지만, 매칭을 위해 CO 전체를 구성하는 편이 단순하고 비용도 선형시간이다.
- $\delta$ 는 $10^{-6}\sim10^{-5}$ rad 수준. active-set 전환 근처에서는 중앙차분이 무효화되므로 태그 불일치를 감지해 단측차분으로 대체.

### 2.9 CLF-CBF-QP (논문 식 18)

호라이즌 $[0,T]$ 를 $\Delta t$ 간격으로 이산화하고, 각 스텝 시작에서 상태를 고정한 채 $t=k\Delta t$ 에서 제어를 $[t_k,t_{k+1})$ 동안 상수로 유지한다 → 제약이 $u$ 에 대해 아핀이 된다. $x(t_k)$ 로 표준 CLF-CBF-QP를 풀어 $u_k$ 를 얻고, ZOH로 동역학에 적용한다.

$$\min_{u(t),\delta(t)}\ \int_0^T u^\top u+p\,\delta^2\ dt$$

$$\text{s.t.}\quad
\begin{aligned}
&L_fh(x)+L_gh(x)u+\gamma h(x)\ \ge\ \epsilon\\
&L_fV(x)+L_gV(x)u+cV(x)\ \le\ \delta\\
&u_{\min}\le u\le u_{\max}
\end{aligned}$$

- $\delta$: CLF 완화 슬랙, $p>0$: 페널티. 실현가능성 확보용.
- $\epsilon>0$: unsafe set에서 시작했을 때 **유한시간 내에 $h$ 의 부호가 바뀌도록** 보장하는 작은 상수 (논문 참고문헌 [6]).
- 다중 장애물: 각 쌍 $(\mathcal{R},\mathcal{O}_i)$ 에 대해 CBF 제약을 **독립적으로** 추가한다 (Remark 3). $h_i(x)=\text{sd}(0_c,\mathcal{O}^\mathcal{C}_i(x))-d_{\text{safe}}\ge0$, $L_fh_i+L_gh_iu+\gamma h_i\ge\epsilon$. **상대차수.** 유니사이클 $\dot x=v\cos\theta,\ \dot y=v\sin\theta,\ \dot\theta=u_1,\ \dot v=u_2$ 에서 $f=[v\cos\theta,v\sin\theta,0,0]^\top$, $g=[0_{2\times2};I_2]$ 이므로

$$L_gh=\Big[\frac{\partial h}{\partial\theta},\ 0\Big]$$

CO는 위치뿐 아니라 **방향에도 의존**하므로 상대차수가 1이고, 논문 주장대로 HOCBF나 auxiliary-variable CBF가 불필요하다. 다만 CBF 제약이 실제로 구속하는 입력은 $u_1$(turning rate) 뿐이며 가속도 $u_2$ 는 $\dot h$ 에 나타나지 않는다.

**CLF.** Case I: $V(x)=(x-x_d)^2$. Case II: $V_1(x)=\big(\theta-\arctan\frac{y_d-y}{x_d-x}\big)^2$, $V_2(x)=(v-v_d)^2$. ⚠️ 식 (18)은 슬랙 $\delta$ 를 하나로 쓰지만 유니사이클은 CLF가 둘이다. 슬랙 공유/분리는 구현자가 결정한다. ⚠️ $V_1$ 의 각도 오차는 $[-\pi,\pi]$ 로 wrap해야 한다 (논문 미명시).

---

## 3. acados OCP 설계

### 3.1 근본 제약: $h(x)$ 는 acados 안으로 들어갈 수 없다

acados는 CasADi 심볼릭 식을 **C 코드로 생성**해 푼다. 그런데 $h(x)$ 는 §2.5대로 **암시적**이다 — CO 구성(조합적 알고리즘) + QP 또는 LP 풀이 + dual 분석을 거쳐야만 값이 나온다.

- `casadi.Callback` 으로 파이썬 함수를 감싸는 방법은 **코드 생성이 불가능**하므로 acados에서 쓸 수 없다. 시도하지 말 것.
- 따라서 유일한 경로는: **$h$ 와 $\nabla h$ 를 호스트(Python)에서 계산해 acados의 런타임 파라미터로 주입**하고, OCP 안에서는 그 파라미터를 상수로 취급하는 것이다. 이는 근사가 아니라 아키텍처적 귀결이다. 근사가 개입하는지 여부는 다음 절에서 갈린다.

### 3.2 $N=1$ — 논문의 정확한 재현

논문의 제어기는 pointwise QP다. acados에서 $N=1$ 로 두면:

- 파라미터로 넣은 $\bar h=h(x_k)$, $\nabla\bar h=\partial h/\partial x|_{x_k}$ 는 **현재 상태에서의 정확한 값**이다.
- 제약 $\nabla\bar h^\top\big(f(x_k)+g(x_k)u\big)+\gamma\bar h\ \ge\ \epsilon$ 는 $L_fh+L_ghu+\gamma h\ge\epsilon$ 와 **완전히 동일**하다.
- 즉 **선형화 오차가 0**이다. acados는 논문 식 (18)을 그대로 푼다. 이 경우 acados가 실제로 하는 일은 결정변수 $(u,\delta)$ 3~4개짜리 dense QP 한 번이다. 적분기, SQP 반복, condensing, RTI는 모두 무의미하다. → **논문 재현만이 목적이라면 acados는 순이익이 없다.** 표준 QP 솔버로 먼저 정답을 만들고, acados $N=1$ 결과가 이와 일치하는지 대조하는 용도로 쓰는 것이 합리적이다. acados의 가치는 3.3의 확장, 그리고 생성된 C 코드의 임베디드 배포에서 나온다.

### 3.3 $N>1$ — 여기서 처음으로 근사가 들어온다

호라이즌을 늘리면 스테이지 $k$ 의 상태 $x_k$ 를 미리 알 수 없으므로 $h(x_k)$ 를 파라미터로 고정할 수 없다. 예측 궤적 $\bar x_k$ 에서 선형화해야 한다:

$$\hat h_k(x)=\bar h_k+\nabla\bar h_k^\top(x-\bar x_k)$$

이때 추가로 필요한 것:

- **스테이지별 SDF 재계산**: 각 $\bar x_k$ ($k=0..N$)마다, 각 장애물마다 CO 구성 + QP/LP + 야코비안. 제어 주기당 비용이 **$(N+1)\times n_{obs}$ 배**로 늘어난다. 논문의 4-장애물 케이스가 이미 19.63 ms/iter임을 고려할 것.
- **재선형화 루프**: RTI 1회로는 $\bar x$ 와 실제 해가 어긋난다. 해를 얻은 뒤 $\bar x$ 를 갱신하고 SDF를 다시 풀어 1~3회 반복. active-set 전환 근처에서 진동하므로 $\bar x\leftarrow\bar x+\alpha(x_{new}-\bar x)$ 형태의 damping이 필요하다.
- **CBF 제약 형태**: 연속시간 형태를 스테이지마다 걸거나, 이산시간 CBF $\hat h_{k+1}(x_{k+1})\ge(1-\gamma\Delta t)\hat h_k(x_k)$ 를 쓴다. 전자가 논문과 형태상 일치하고, 후자가 MPC에서는 더 자연스럽다. **이 확장은 논문의 범위 밖이다.** 논문 결과를 검증하기 전에 $N>1$ 로 가면 선형화 오차와 논문 방법의 특성이 뒤섞여 디버깅이 불가능해진다. $N=1$ 재현 → 검증 → 확장 순서를 지킬 것.

### 3.4 OCP 구성 요소

| 항목 | 설계 |
|---|---|
| 상태 / 입력 | Case I: $x\in\mathbb{R}^2$, $u\in\mathbb{R}^2$ / Case II: $x=[x,y,\theta,v]^\top$, $u=[u_1,u_2]^\top$ |
| 동역학 | CasADi 심볼릭으로 명시. $f,g$ 는 닫힌 식이므로 문제없음 |
| 파라미터 | 장애물당 $[\bar h,\ \nabla\bar h\ (n\text{개})]$. $N>1$ 이면 $\bar x_k\ (n\text{개})$ 추가 |
| CBF 제약 | 비선형 제약 슬롯에 $\nabla\bar h^\top(f+gu)+\gamma\hat h$, 하한 $\epsilon$, 상한 $+\infty$ |
| CLF 제약 | 동일 슬롯에 $L_fV+L_gVu+cV$, 상한 0. **소프트 제약으로 두어 acados 슬랙이 논문의 $\delta$ 역할** |
| 입력 제약 | 박스 제약 $u_{\min}\le u\le u_{\max}$ |
| 비용 | $N=1$: $u^\top u$ (+ 슬랙 페널티 $p\delta^2$). $N>1$: 추적 비용으로 CLF를 대체하는 것도 가능 |
| Hessian | 비용이 이차이고 CBF 제약이 $u$ 에 아핀이므로 Gauss-Newton으로 충분 |

**슬랙 설계.** 논문의 $p\delta^2$ 는 acados 소프트 제약의 이차 페널티에 대응하고, 선형 페널티는 논문에 없으므로 0으로 둔다. CBF 제약도 소프트로 두는 것을 권한다 — 충돌 상태에서 시작하는 Case II-1은 $\epsilon>0$ 제약과 입력 박스가 동시에 걸려 hard constraint면 실현불가능해질 수 있다. 단, **CBF 슬랙 페널티는 CLF 슬랙보다 수 자릿수 크게** 잡아야 "목표 도달을 포기하더라도 안전을 지킨다"는 우선순위가 유지된다.

### 3.5 코드 생성 시점에 고정되는 것들

- 파라미터 차원 $n_p$, **장애물 개수**, 호라이즌 $N$, 상태/입력 차원, 제약 개수.
- 장애물 수가 바뀌면 재코드생성이 필요하다. 다중 장애물 케이스는 최대 개수를 고정하고, 비활성 장애물은 $\bar h$ 를 큰 값, $\nabla\bar h=0$ 으로 채워 제약을 무력화하는 방식으로 처리한다.
- CO의 변 개수 $\ell_\mathcal{C}$ 는 acados 쪽에 전혀 노출되지 않는다. acados는 $(\bar h,\nabla\bar h)$ 만 본다. 이것이 이 설계의 장점이다 — CO 구성 / QP / LP / 유한차분의 복잡도가 전부 코드 생성 경계 바깥에 있어서, SDF 백엔드를 바꿔도 생성된 C 코드를 다시 만들 필요가 없다.

### 3.6 한 제어 주기의 데이터 흐름

```
x_k
 └─ 장애물 i마다:
      CO 구성 (Minkowski 합, provenance 태그 포함)         [§2.3]
      원점 포함 판정: b^C >= 0 ?                            [Lemma 1]
        ├─ 분리 → QP (10) → z*, λ*, active set             [§2.4-A]
        └─ 충돌 → LP (12) → s*, dual → k_act → z* (13)      [§2.4-B]
      d_x A^C, d_x b^C
        ├─ 위치 열: 폐형식 (0, -A^C)                        [§2.8-a]
        └─ θ 열: CO 재구성 2회 + 태그 매칭 중앙차분           [§2.8-b]
      ∂z*/∂x
        ├─ 분리: 축약 KKT 선형계 (16)(17) + Remark 5        [§2.6]
        └─ 충돌: 식 (13) 직접 미분                           [§2.7]
      h_i, ∇h_i
 └─ acados 파라미터 주입 → solve → u_k
 └─ ZOH 적분 (ode45 상당) → x_{k+1}
```

---

## 부록: 구현 전 확인할 참고문헌

- **[42] Amos & Kolter, "OptNet" (ICML 2017)** — 식 (16)(17) 야코비안의 출처. **필독**
- **[24] Gilbert–Johnson–Keerthi (1988)** — GJK, MD-space 거리의 원류
- **[25] van den Bergen (2001)**, **[34] Cameron & Culley (1986)** — penetration depth / MTD
- **[37] De Berg, *Computational Geometry*** — 볼록 다각형 Minkowski 합 알고리즘 (§2.3)
- **[40] Boyd & Vandenberghe, Ch. 8** — projection과 depth의 정의
- **[18],[19] Thirugnanam et al.** — 가장 가까운 선행연구 (duality-based MDF, NCBF)
- **[6] Liu, Xiao, Belta (CDC 2024)** — $\epsilon>0$ 의 유한시간 수렴 보장 시연 영상: `https://youtu.be/3Dh0gtDW8bE` (W-space / MD-space 동시 렌더링 — 디버깅 시 참고)
