# DilatedPolygonRobustController 구현

`control/dilated_polygon_robust_optimizer.py`의
`DilatedPolygonRobustController`는 `BaseController`가 사용하는 optimizer
인터페이스를 구현한다. `optimizer_type: sipoc`으로 `test_nmpc.py`에서 선택한다.
`DilatedPolygonRobustOptimizer`는 동일 클래스의 별칭이다.

근거는 [로컬 논문](references/sipoc/sip_paper.pdf)과
[공식 SIPOC 구현](https://github.com/boschresearch/sip_optimal_control)이다.
모바일 로봇 Python 실험은 C++ controller를 pybind로 호출한다. 이 포팅은 ROS,
Nav2, pybind, SDF raster와 MP-500의 식별 모델을 요구하지 않는다. 하위 거리
문제는 Python Clarabel, 상위 OCP는 기존 환경의 CasADi/acados를 사용한다.

## 수식과 모델 대응

상태는 \(x=(p_x,p_y,\theta)\), 입력은 \(u=(v,\omega)\)다. 기존
`DifferentialDriveDynamics.forward_dynamics_opt(dt)`를 상위 OCP의
`disc_dyn_expr`에 직접 사용한다. 따라서 plant와 동일한 Euler map이며
acados의 integrator는 `DISCRETE`다.

\[
F(x,u)=\begin{bmatrix}
p_x+\Delta t\,v\cos\theta\\
p_y+\Delta t\,v\sin\theta\\
\theta+\Delta t\,\omega
\end{bmatrix},\qquad \Delta t=t_f/N.
\]

원본 실험의 9-state MP-500 모델, 가속도 입력, RK4 적분을 현재 로봇에
그대로 적용하지 않는다. 논문 (7)의 일반 discrete map을 현재 DD 모델로
치환했다. plant timestep과 예측 timestep이 다르면 오류를 낸다.

footprint의 convex component를 \(\Gamma_j\)라 하면 dilated robot은
\(p+R(\theta)\Gamma_j\oplus r\mathbb B_2\)다. 여기서
\(r=\texttt{polygon_dilation_radius}\)이며 `d_safe`는 별도 clearance다.
`equiv_rep()`의 각 component를 독립적으로 처리하므로 offset과 union을
유지한다. 지원 geometry는 bounded convex polygon이며 원형 footprint는
polygon으로 표현해야 한다.

### 불확실성 전파: 논문 (13)–(14), zoRO

\[
A_k=\frac{\partial F}{\partial x}(x_k,u_k),\quad
B_k=\frac{\partial F}{\partial u}(x_k,u_k),\quad
P_{k+1}=(A_k+B_kK)P_k(A_k+B_kK)^\top+W.
\]

Jacobian은 OCP와 같은 discrete map을 CasADi로 미분한다. YAML의
`sipoc.initial_covariance`가 \(P_0\), `sipoc.process_noise`가 한 discrete step의
\(Q_d=W\)다. 두 값 모두 상태 순서 `[x, y, theta]`의 3×3 행렬이다. 이 controller는
ancillary feedback을 구현하지 않으므로 `feedback_gain`은 영행렬만 허용한다.
행렬은 world frame이며 PSD·대칭·차원을 검사한다.

기하 검증에는 \(S_k=\alpha P_k\), \(\alpha=\texttt{robust_scale}\)을 쓴다.
반경은 \(\sqrt\alpha\)에 비례한다. 각 상위 SQP subproblem 내에서는 \(P_k\)를
고정하고 새 입력 궤적을 얻은 뒤 다시 전파한다.

### 하위 거리 문제: 논문 (54)의 convex polygon 장애물 확장

\(LL^\top=S_{k,pp}\), obstacle polygon이 \(O_i=\{q:A_iq\le b_i\}\)일 때

\[
\min_{\gamma,z,q}\ \frac12\|p+R(\theta)\gamma+Lz-q\|_2^2,
\qquad \gamma\in\Gamma_j,\quad \|z\|_2\le1,\quad q\in O_i.
\]

SIPOC 원본의 obstacle point를 convex obstacle 전체로 확장했다. 이는
본 저장소의 `RectangleRegion`/`PolytopeRegion` 표현을 이용하며 경계 샘플링
간격을 새로 도입하지 않는다. 작은 장애물이 로봇 내부에 완전히 포함된
경우도 overlap으로 판정한다. 다각형 obstacle 때문에 최근접점이 유일하지
않을 수 있으므로 원본 point-cloud local-reduction의 미분 가능성 가정을
그대로 주장하지 않는다.

Clarabel의 6-variable convex QCQP/SOC 풀이로
\(\gamma^*,q^*,\delta^*=Lz^*\)를 얻는다. obstacle decision은 robot position
주변으로 이동한 좌표계에서 풀어 큰 world coordinate의 상쇄 오차를 줄인다.
고유값 기반 factorization으로 isotropic·rank-one·zero uncertainty를 처리한다.
하위 solver의 실패는 예외로 전달하며 nominal 결과로 대체하지 않는다.

분리된 경우 법선은

\[
n=\frac{p+R\gamma^*+\delta^*-q^*}
        {\|p+R\gamma^*+\delta^*-q^*\|}.
\]

이는 논문의 envelope gradient와 일치한다. 원본
`dilated_polygon_robust_controller.cpp:267`의
\(p+R\gamma^*-q^*\) 법선과 달리 최악 위치 변위를 포함한다. 비등방성
uncertainty에서 이 차이가 중요하며 finite-difference test로 검증했다.

거리 0의 overlap에서는 polygon edge normal과 angular search로 recovery
방향을 선택한다. 최종 판정은 그 방향의 정확한 support gap을 이용한다.
penetration의 전역 signed-distance 최적값이라고 주장하지 않는다.

### 회전 backoff와 상위 제약: 논문 (51), (57)

\[
\beta_{k,j}=r_{\max,j}\sqrt{S_{k,\theta\theta}},\qquad
r_{\max,j}=\max_{\gamma\in\Gamma_j}\|\gamma\|.
\]

각 component의 model origin 기준 반경을 사용한다. uncertainty가
translation과 rotation 사이에 상관관계를 가지더라도 두 projection의
최악값을 합하므로 보수적이다.

nominal iterate \(\bar x\)에서 \(R'=\partial R/\partial\theta\)라 두면

\[
C=[n_x,n_y,n^\top R'\gamma],\qquad
\ell_g=n^\top q-n^\top\delta-
n^\top(R-\bar\theta R')\gamma+r+d_{\rm safe}+\beta,
\qquad Cx\ge\ell_g.
\]

평행한 polygon face의 경우 최소 support를 갖는 꼭짓점 제약을 함께
추가하고, 같은 outer solve에서 발견한 꼭짓점을 유지한다. 한 꼭짓점씩
바꿔 가며 heading이 진동하는 문제를 막는다. `max_constraints`는 한 stage의
실제 row 수 제한이다. 넘으면 `constraint_capacity_exceeded`로 실패하며
필요한 제약을 조용히 버리지 않는다.

상위 비용은 acados `LINEAR_LS` convention으로

\[
J=\frac12\sum_{k=0}^{N-1}\Delta t\,
[(x_k-x_k^{\rm ref})^\top Q(x_k-x_k^{\rm ref})+u_k^\top Ru_k]
+\frac12 w_T(x_N-x_N^{\rm ref})^\top Q(x_N-x_N^{\rm ref})
+J_{\rm slack}.
\]

각 reference는 기존 ConstantSpeedTrajectoryGenerator의 pose reference다.
terminal을 포함해 \(N+1\)개를 사용하고 heading을 현재 yaw 주변으로 unwrap한다.
원본처럼 slack의 L1/L2 penalty를 제공하되 최종 geometry 검증은 slack을
허용하지 않는다. numerical feasibility tolerance만 적용한다.

## 한 controller 호출의 동작과 성공 판정

1. 이전 성공 입력을 한 step 이동해 초기화하거나 reference 기반 seed를 만든다.
2. 현재 입력을 정확히 rollout하고 uncertainty를 전파한다.
3. 미래 각 stage에서 robot component–obstacle pair의 하위 문제를 푼다.
   robot+uncertainty+dilation+angular backoff의 AABB 거리로 명백히 먼 pair만
   제외한다. 현재 pose만으로 장애물을 제외하지 않는다.
4. support 제약을 갱신하고 acados `SQP`를 최대 4회 내부 iteration으로 푼다.
   Levenberg–Marquardt regularization은 `1e-7`, 상위 solver tolerance는 `1e-8`이다.
5. 기본 `outer_step_length=1.0`에서는 상위 후보 입력을 그대로 사용한다.
   더 작은 값을 지정하면 이전 outer 입력과 섞는다. 그 입력을 plant discrete
   map으로 다시 rollout하고 \(P\)도 다시 계산한다.
   초기·첫 예측·terminal을 포함한 모든 node의 support gap을 재검사한다.
6. input bounds, geometry clearance, state/input 변화량 tolerance를 모두 만족하면
   성공이다. 상위 SQP의 status 0과 2(`ACADOS_MAXITER`)는 후보를 제공할 수 있지만,
   어느 경우든 독립적인 최종 검사를 통과해야 한다. 다른 status는 실패이며,
   `SQP_RTI`에서는 status 0만 허용한다. outer iteration은 최대 100회다.

기본 `SQP`, 내부 4회, regularization `1e-7`, outer 최대 100회는 원본 Python
mobile robot 실험의 code generator와 `OcpIterParams` 기본값을 따른다.
원본 robust controller도 `ACADOS_SUCCESS` 또는 `ACADOS_MAXITER`의 후보를
다음 outer iteration에 사용한다. `nlp_solver_type`, `nlp_solver_max_iter`,
`levenberg_marquardt`를 설정에서 변경할 수 있으며 변경 시 solver를 재생성한다.
status 2 허용은 inner solve의 중단 결과를 outer 알고리즘에 전달하는 정책이다.
outer 변화량 기준과 `feasibility_tolerance=1e-5` m는 그대로 유지한다.

원본 code generator에는 명시적인 tolerance 지정이 없으며 현재 설치된
acados C 구현의 기본 SQP tolerance는 `1e-8`이다. 초기 포팅의 `tol=1e-6`은
이보다 느슨했으므로 `tol=1e-8`로 복원했다. 이 값은 상위 풀이의 정확도이며
최종 geometry의 `feasibility_tolerance`와 outer 변화량의 `step_tolerance`와
별개다. 최종 기본 설정은 damping 없는 전체 step이다.

전체 polygon 장애물의 corner에서 frozen-geometry 풀이가 두 입력 사이를
반복할 가능성을 조사하기 위해, 이 포팅은 원본에 없는 선택적 outer damping을
제공한다. 아래 corner 재풀이에서는 damping이 문제를 해결하지 못했으며,
상위 tolerance 복원을 기본 해결책으로 선택했다.
상위 후보 입력을 \(\hat u^{(j)}\), 이전 입력을 \(u^{(j)}\)라 하면

\[
u^{(j+1)}=u^{(j)}+\alpha(\hat u^{(j)}-u^{(j)}),\qquad
\alpha=\texttt{outer_step_length}\in(0,1].
\]

기본값은 1.0이며 입력의 전체 step을 사용한다. 상태를 두 궤적의
평균으로 만들지 않고 새 입력을 DD discrete map으로 정확히 rollout한다.
uncertainty와 최종 geometry도 그 rollout에 맞춰 재계산한다. outer 수렴은
damping 전 후보의 residual

\[
\rho^{(j)}=\max\bigl(\|\hat u^{(j)}-u^{(j)}\|_\infty,
\|\operatorname{rollout}(\hat u^{(j)})-x^{(j)}\|_\infty\bigr)
\]

에 `step_tolerance`를 적용한다. 따라서 \(\alpha\)를 아주 작게 설정해 실제
업데이트만 줄여도 미수렴 후보가 성공으로 바뀌지 않는다. `iteration_history`의
`state_step_inf`/`input_step_inf`는 damping 전 residual이며 `outer_step_length`도
기록한다. 이 relaxation은 polygon 장애물 확장을 위한 추가 선택이고 원본
point-cloud 실험의 동일 알고리즘 또는 동일 runtime을 뜻하지 않는다.

고정 normal의 실제 support gap은

\[
g=n^\top p+\min_{\gamma\in\Gamma_j}n^\top R\gamma
-\max_{q\in O_i}n^\top q-\sqrt{n^\top S_{pp}n}
-r-d_{\rm safe}-\beta.
\]

\(g\ge-\texttt{feasibility_tolerance}\)가 최종 geometry 기준이다.
raw acados state를 그대로 적용하지 않고 정확한 입력 rollout을 반환하므로
상위 solver의 dynamics residual이 최종 clearance 검사에서 숨겨지지 않는다.

upper failure, 초기 robust clearance 위반, capacity 초과, outer 수렴 실패는
`success=False`, status code 1과 영속도 궤적을 반환한다. 영속도는 실패시
명령 정책이며 실제 불확실한 plant에 대한 별도 안전 보장은 아니다.
실패 원인은 solution `stats()`에 남으며 공통 연속 실패 checker에 반영된다.
하위 solver exception은 기존 trial framework에서 `error`로 처리한다.

이 실패시 영입력 정책은 원본보다 엄격한 의도적 adaptation이다. 원본
`base_controller.cpp:301–306`은 마지막 state/input을 저장하고 outer 반복
제한을 `reachMaxIters`로 보고한다. 원본 Nav2 robust plugin은
`nav2_dilated_polygon_robust_controller_plugin.cpp:492–508`에서 이 상태에도
마지막 입력을 읽어 command를 갱신하고, `519–530`에서 `acadosFail`만 별도로
처리하므로 `reachMaxIters`에서는 다음 step의 warm start도 유지한다.
Python 실험도 `main_dilated_polygon_controller.py:187–208`에서 status를 출력한
뒤 결과 배열을 그대로 읽고 시각화한다. 반면 이 포팅은 최종 geometry가
feasible하더라도 outer 변화량 기준을 통과하지 못한 `maximum_iterations`
후보를 적용하지 않는다. 따라서 원본의 입력 적용 정책이나 전체 주행 성능을
동일하게 재현했다고 해석해서는 안 된다. 상위 SQP status 2 후보의 수용과
outer `reachMaxIters` 후보의 plant 적용은 서로 다른 결정이다.

## 설정, 실행 및 시간 통계

예제 [config/config_nmpc_sipoc.yaml](config/config_nmpc_sipoc.yaml)는 maze와
s_path에서 DD rectangle/A*를 사용한다. 각 환경 1 trial이며 animation,
plot, localization error, start perturbation은 꺼져 있다. 기존 600-trial config는
변경하지 않았다.

Padding 비교 설정은 0 mm와 1 mm 두 가지다. `config/config_nmpc_sipoc.yaml`은
1 mm(`polygon_dilation_radius: 0.001`),
`config/config_nmpc_sipoc_padding_0mm.yaml`은 0 mm(`0.0`)이며 나머지 설정과
reference path 생성 조건은 같다. 각 파일을 별도 `--results-dir`로 실행한다.
불확실성은 각 파일의 최상위 `sipoc` 블록에서 설정한다.

예제 \(P_0\)와 \(W\)는 illustrative assumed uncertainty다. plant에 noise를
주입하지 않으며 실측값·확률 calibration·PSDF와 동일한 uncertainty를 뜻하지
않는다. \(P_0\)는 MPC 호출마다 설정값으로 재초기화한다.
\(P_0=Q_d=0\) 비교에는 `initial_covariance`와 `process_noise`를 모두
`[[0, 0, 0], [0, 0, 0], [0, 0, 0]]`으로 설정하고 `robust_scale: 1.0`을 유지한다.
스칼라 `0`이나 `P0`, `Q_d`라는 YAML 키는 지원하지 않는다.
`robust_scale: 0.0`만 설정하면 기하 backoff는 없어지지만 covariance 자체가
0이 되지는 않는다.

```bash
python3 test_nmpc.py -c config/config_nmpc_sipoc.yaml --trials 1 --results-dir benchmark_runs/sipoc_01
```

재실행에는 새 results directory를 지정한다. 의존성은 기존 CasADi/acados와
`numpy scipy clarabel`이며 generated code는 controller별 temporary directory에
생성한다. 같은 instance에서 dimension·weight·solver 설정이 같으면 solver를
재사용한다. `cleanup()`/`reset()`은 temporary files, warm start, counters를 비운다.

`solver_times`는 `solve_nlp()` 전체를 `time.perf_counter()`로 측정한다. 하위
Clarabel, covariance 전파, 제약 갱신, 최종 geometry 검사, 상위 acados 호출의
합계를 포함한다. 외부 `setup()`, code generation, path planning, 출력은 제외한다.
이 알고리즘의 내부 단계가 전체 solver를 이루므로 bare acados 시간과 구별해야 한다.
`stats()['upper_solver_time_s']`는 acados `solve()` 호출만 합산한 별도 값이다.
일반 `computation_times_*.csv`에는 성공한 전체 알고리즘 호출만 들어간다.

## 검증 범위와 결과

2026-10-07 실제 acados code generation/solve를 포함한 focused suite **45개**와
기존 trial metrics/start perturbation 회귀검증 **27개**가 통과했다. 변경 Python
파일의 syntax 검사와 `git diff --check`도 통과했다.

```bash
SIPOC_RUN_ACADOS_SMOKE=1 python3 -m unittest tests.test_dilated_polygon_robust -v
```

수식 검증은 실제 Clarabel geometry, 해석 가능한 rectangle/ellipse 예제,
anisotropic optimum 및 envelope gradient의 finite difference를 사용한다.
인터페이스 검증은 prescribed upper iterates로 initial/terminal collision,
capacity overflow, 상태·입력 전달, SQP status 2 후보, 실패 영입력, timing,
solver 재사용/재생성과 선택적 damping의 정확한 rollout/거짓 수렴 방지를
검사한다. opt-in smoke는 실제 acados를 생성해 nominal/robust wall OCP를
풀고 같은 solver 재사용을 검사한다.

최종 주행 결과는
[benchmark_runs/sipoc_maze_s_path_verified_20261007_163630/README.md](benchmark_runs/sipoc_maze_s_path_verified_20261007_163630/README.md)에 정리했다.
기존 `test_nmpc.main()`을 사용하여 동일 PC에서 **maze → s_path 각 1회**를
순차 실행했다. DD rectangle/A*, dt=0.1 s, horizon=15/tf=1.5 s,
최대 simulation 60 s이며 시작 perturbation·localization noise를 껐다.
controller의 가정된 비영 P0/W는 유지했지만 plant에 noise를 주입하지 않았다.

| 환경 | 판정 | 종료 simulation 시간 | 성공/실패 solver 호출 | 실제 node 최소 clearance | 성공 호출 solver 평균/p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| maze | 정체로 실패 | 9.9 s | 96 / 3 | 25.171 mm | 3.305 / 10.618 s |
| s_path | 기존 목표 판정 성공 | 5.8 s | 58 / 0 | 25.183 mm | 0.554 / 1.605 s |

maze는 마지막 4 s 이동량 28.140 mm가 기준 30 mm 이하라 실패했다.
마지막 pose는 약 `(0.795293, 0.646040, -0.716873)`이며 checked goal까지
L1 거리는 1.575747 m이다. outer 한도에 도달한 3회는 영입력을 적용한 뒤
다음 성공 호출에서 진행을 재개했다. 연속 20회 실패가 종료 원인은 아니며,
마지막 호출도 outer 입력 residual 약 `7.10e-5`로 수렴하고 robust gap
약 `-7.55e-9` m로 최종 geometry 기준을 통과했다.

독립 SAT 검사에서 마지막 maze 참조 pose 16개 중 **12개(node 0–11)**의
nominal rectangle footprint가 obstacle 3 (`x=0.90–1.05, y=0.60–0.90`)
과 엄밀히 겹쳤다. 모든 edge-normal 축에서 겹치며 최소 정규화 투영 overlap은
1.25 mm였다. 따라서 이 참조 구간은 uncertainty를 제외해도 footprint에
feasible하지 않다. 이는 실제 확인한 조건이고, 정체의 전체 원인이나 다른
경로의 존재 여부를 단독으로 증명하지 않는다. 경로 참조와 footprint/회전
제약의 불일치가 local optimum에 영향을 주었을 가능성은 추론이다.
근거: [reference_geometry_maze.json](benchmark_runs/sipoc_maze_s_path_verified_20261007_163630/reference_geometry_maze.json).
maze 목표 도달은 검증을 통과하지 못했다.

기존 목표 checker는 A* grid-center 경로 끝점을 사용한다. s_path checked
point는 `(0.825, 0.825)`이고 최종 L1 오차는 19.530 mm라 기준 20 mm를
통과했다. 환경 설정의 goal `(0.8, 0.8)`까지는 31.907 mm이다. 기존 판정
기준을 변경하거나 후자의 오차를 전자의 성공으로 표현하지 않았다.

두 주행의 기록된 simulation node에서 독립 SAT/vertex–edge geometry
검사로 접촉·충돌이 없음을 확인했다.
실제 Euler rollout 오차는 각각 `1.11e-16 / 5.55e-17`이고,
plant 입력·solver counters·성공 호출 CSV의 시간 통계가 독립적으로 일치했다.
각 환경에서 upper solver는 한 번씩 생성되었다.

최종 accepted robust gap의 최소 진단값은 maze 약 `-5.77e-8` m,
s_path 약 `-1.12e-8` m로 지정 tolerance `1e-5` m 이내였다. 이 값은 controller
자체 진단이고, 위 독립 SAT 검사는 실제/예측 nominal footprint를 검사한다.
두 검사를 동일한 독립 robust uncertainty 인증으로 해석하지 않는다.

전체 157회 풀이가 모두 plant period 0.1 s보다 길었다. 평균/p95는 성공한
전체 알고리즘 호출만 집계했으며 failed maze의 성공 호출도 포함했다.
실패를 포함한 전체 시간·inner 시간·outer iteration 분포는
[verification.json](benchmark_runs/sipoc_maze_s_path_verified_20261007_163630/verification.json)에 별도로 기록했다.
현재 Python 포팅은 이 환경에서 실시간 10 Hz 검증을 통과하지 못했다.
각 환경 1회의 deterministic 결과를 통계적인 navigation success rate나
논문의 성능 재현으로 일반화하지 않는다.

실행기는 설치된 numerical stack의 Python 3.8.10이다. Python 3.8에 없는
`argparse.BooleanOptionalAction`은 저장된 관측 runner에서 runtime backport를
적용했다. controller와 저장소 CLI 코드는 바꾸지 않았다. 같은 보완으로
기존 회귀검증 27개도 완료했으며 정상 재실행에는 저장소의 Python 3.10+
요구사항과 의존성을 권장한다. 실제 config/manifest, source snapshot/hash,
state/input/prediction, geometry, per-call/iteration trace, raw CSV 및 로그를
최종 실행 디렉터리에 보존했다.

초기 `SQP_RTI`/outer20 실행은 두 환경 모두 outer 한도에 따른 연속 실패로
종료했다. 원본의 `SQP` 내부 4회·LM `1e-7`·outer100으로 맞추고 상위
정밀도를 현재 설치된 acados의 원본 기본값 `1e-8`로 복원했다. 중간
`SQP`/`tol=1e-6` maze 진단 실행은 반복 업데이트 문제를 조사하기 위해
`interrupted`로 보존했다. `alpha=0.5`의 corner 재풀이는 `tol=1e-6`과
`1e-8` 모두 100회에서 실패했지만, `alpha=1`/`tol=1e-8` 재풀이는 100회에
수렴했다. 최종 전체 검증은 후자의 full-step 설정이다. clearance·불확실성·
실패 기준·outer 수렴 tolerance를 완화하지 않았다. 이전 실행과 재풀이
기록도 각각 `benchmark_runs/sipoc_maze_s_path_20261007_161918/`,
`benchmark_runs/sipoc_maze_s_path_sqp100_20261007_162714/`에 남겼다.

## 원본 코드 대응

| 구현 기능 | SIPOC 원본 |
|---|---|
| Python 실험 연결 | `sipoc_mobile_robot/sipoc_mr_solver/py_scripts/main_dilated_polygon_controller.py:178` |
| lower/upper 반복 | `sipoc_mobile_robot/sipoc_mr_solver/src/dilated_polygon_robust_controller.cpp:84` |
| uncertainty propagation | `sipoc_mobile_robot/sipoc_mr_utils/src/ellipsoidal_uncertainty_propagation.cpp:82` |
| ellipse QCQP | `sipoc_mobile_robot/sipoc_mr_utils/src/distance_minimizer_ellipsoid_to_polygon.cpp:126` |
| robust plane/backoff | `sipoc_mobile_robot/sipoc_mr_solver/src/dilated_polygon_robust_controller.cpp:275` |
| outer convergence | `sipoc_mobile_robot/sipoc_mr_solver/src/base_controller.cpp:279` |
| SQP 4회 및 LM 설정 | `sipoc_mobile_robot/sipoc_mr_solver/py_scripts/generate_code_for_acados_solver.py:150` |
| SQP MAXITER 후보 전달 | `sipoc_mobile_robot/sipoc_mr_solver/src/dilated_polygon_robust_controller.cpp:142` |
| outer 최대 100회 기본값 | `sipoc_mobile_robot/sipoc_mr_utils/include/sipoc_mr_utils/common_structs.hpp:115` |

경로는 모두 `references/sipoc/` 기준이다.
