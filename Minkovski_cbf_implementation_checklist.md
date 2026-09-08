# Minkowski CBF Implementation Checklist

> 목표: 논문의 Minkowski-operation SDF-CBF를 현재 3-state differential-drive 모델에 맞춘 독립적인 pointwise CLF-CBF-QP optimizer로 구현하고 검증한다.
>
> 진행 규칙: 코드 작성과 해당 acceptance test가 모두 끝난 뒤에만 항목을 [x]로 변경한다.

## 관련 자료

- 논문: references/Minkovski_cbf.pdf
- 구현 메모: references/Paper Context for Minkovski CBF.md
- optimizer 계약 참고: control/psdf_optimizer.py
- controller 계약: control/controller.py
- dynamics/geometry: models/dd.py, models/geometry_utils.py

## 고정된 설계 결정

- optimizer 식별자: minkowski_cbf
- 신규 파일: control/minkowski_cbf_optimizer.py
- 상태/입력: x=[x,y,theta], u=[v,omega]
- 매 control tick에 푸는 pointwise N=1 QP
- 장애물별 hard CBF, CLF에만 slack
- 분리: 논문 식 (10)의 projection QP와 reduced KKT 미분
- 침투: 식 (12)의 exact scalar solution과 식 (13)
- 기존 geometry가 제공하는 convex component만 지원
- acados, PSDF/PED/Torch, dynamic obstacle, 자동 비볼록 분해, benchmark 확장은 1차 범위 밖

## 0. Baseline과 범위 보호

- [x] 현재 git status와 사용자 미커밋 변경을 기록한다.
- [x] control/psdf_optimizer.py의 기존 사용자 변경을 건드리지 않는 것을 확인한다.
- [x] 대상 WSL Python에서 numpy, scipy, osqp import와 버전을 확인한다.
- [x] OSQP dual 부호가 논문의 Az-b<=0, lambda>=0 convention과 일치하는지 작은 QP로 검증한다.
- [x] 지원 geometry를 ConvexRegion2D 계열로 제한한다.
- [x] None, circle 등 미지원 geometry는 근사하지 않고 명시적 예외를 내도록 한다.
- [x] containment, active, dual, contact, facet-match tolerance를 parameter class에 모은다.

Acceptance:

- [x] 구현 착수 시 worktree 상태, dependency, 기존 테스트 상태를 기록한다.
- [x] 이 단계에서 models, planning, 기존 optimizer 파일에 변경이 없다.

## 1. Optimizer 골격과 외부 계약

- [x] control/minkowski_cbf_optimizer.py를 추가한다.
- [x] MinkowskiCBFOptimizerParam에 필요한 값만 정의한다.
  - [x] d_safe, gamma, epsilon
  - [x] vmin, vmax, omegamin, omegamax
  - [x] CLF rate, heading weight, CLF slack weight
  - [x] theta_fd_eps와 모든 numerical tolerance
  - [x] OSQP tolerance와 iteration limit
- [x] MinkowskiCBFOptimizer에 setup, solve_nlp, reset을 구현한다.
- [x] setup은 매 tick 최신 state, system, local trajectory, obstacles를 저장한다.
- [x] 전체 CO-SDF-gradient-control-QP를 solve_nlp timing에 포함한다.
- [x] solver_times에 solve 호출당 정확히 한 항목을 기록한다.
- [x] variables={x: x, u: u} 계약을 제공한다.
- [x] solution wrapper를 구현한다.
  - [x] value(u)는 shape (2,1)을 반환한다.
  - [x] value(x)는 shape (3,2)를 반환한다.
  - [x] get_input_trajectory를 제공한다.
  - [x] get_state_trajectory를 제공한다.
  - [x] stats는 success 또는 failure를 반환한다.
- [x] one-step state는 system._dt와 기존 forward_dynamics로 계산한다.

Acceptance:

- [x] no-obstacle 입력에서 BaseController가 finite shape (2,) 입력을 반환한다.
- [x] controller logger가 state/input shape 오류 없이 기록한다.
- [x] reset 이후 이전 state, obstacles, solution, timing이 남지 않는다.

## 2. Configuration Obstacle

- [x] 로봇 component의 body-frame 꼭짓점을 get_ccw_vertices로 가져온다.
- [x] 현재 pose로 로봇 꼭짓점을 world frame에 회전·이동한다.
- [x] 반사된 -R(x)와 obstacle 꼭짓점의 pairwise sum을 만든다.
- [x] optimizer 내부 SciPy ConvexHull로 O (+) (-R(x))의 일관된 V/H-rep을 구성한다.
- [x] 중복 꼭짓점과 zero-length edge를 제거한다.
- [x] CO를 A_c z <= b_c H-rep으로 변환한다.
- [x] 모든 facet row를 norm(a_k)=1로 정규화한다.
- [x] outward normal 방향을 일관되게 유지한다.
- [x] 평행·중복 facet을 tolerance 기반으로 병합해 minimal H-rep을 만든다.
- [x] CO 결과에 normalized A_c, b_c, vertices를 보관한다.
- [x] 모든 convex robot component-obstacle pair에 독립 CO를 만든다.
- [x] 임의 비볼록 polygon 자동 분해는 구현하지 않는다.

Acceptance:

- [x] axis-aligned rectangle 예제의 CO bounds가 손계산과 일치한다.
- [x] 입력 꼭짓점 시작 인덱스가 달라도 같은 normalized H-rep을 얻는다.
- [x] 모든 CO 꼭짓점에서 A_c @ vertex <= b_c + tol이 성립한다.
- [x] origin 포함 판정이 알려진 분리/접촉/침투 예제와 일치한다.

## 3. Signed distance

- [x] origin이 CO 밖이면 projection QP를 구성한다.
  - [x] OSQP objective를 P=2I, q=0으로 둔다.
  - [x] A_c z <= b_c를 upper-bound constraint로 둔다.
  - [x] z_star, raw dual lambda, solver status를 반환한다.
- [x] primal residual과 positive dual을 함께 사용해 active set을 선택한다.
- [x] active 선택에는 normalized H-rep만 사용한다.
- [x] origin이 CO 안이면 ratio_k=b_k/norm(a_k)의 최솟값으로 penetration depth를 계산한다.
- [x] penetration active facet은 argmin과 tolerance tie set으로 결정한다.
- [x] 식 (13)으로 critical point z_star를 복원한다.
- [x] signed distance와 h를 계산한다.
  - [x] 분리: sd=+norm(z_star)
  - [x] 침투: sd=-norm(z_star)
  - [x] h=sd-d_safe
- [x] multiple active penetration facet은 deterministic index를 선택하고 tie 정보를 남긴다.
- [x] invalid geometry와 solver failure에서 constraint를 조용히 생략하지 않는다.

Acceptance:

- [x] 분리/접촉/침투 사각형 예제의 sd가 +0.5, 0, -0.1과 일치한다.
- [x] projection QP의 primal, dual, stationarity, complementarity residual이 tolerance 이내다.
- [x] penetration depth가 독립 scalar LP 또는 손계산과 일치한다.
- [x] obstacle 순서를 바꿔도 pair별 결과가 변하지 않는다.

## 4. Gradient와 nonsmooth 처리

- [x] 분리 branch에서 active constraint만 사용한 reduced KKT matrix를 구성한다.
- [x] inverse 대신 np.linalg.solve(K, -B)로 dz_star/dx를 계산한다.
- [x] translation derivative에 dA/dx=dA/dy=0을 사용한다.
- [x] translation derivative에 db/dx=-A[:,0], db/dy=-A[:,1]을 사용한다.
- [x] smooth rotation case에서 theta +/- theta_fd_eps의 CO를 재구성한다.
- [x] base active facet과 perturbed facet을 normalized normal angle과 offset으로 매칭한다.
- [x] 매칭된 active A_c, b_c로 theta derivative를 중앙차분한다.
- [x] 분리 branch는 논문 식 (15)의 chain rule로 grad_h를 계산한다.
- [x] 침투 branch는 고정 active facet에서 식 (13)의 미분 또는 동치인 normalized-depth 미분을 사용한다.
- [x] 침투 gradient가 논문 원문에 명시되지 않은 구현 결정임을 주석으로 남긴다.
- [x] 평행 facet, active switch, facet-match ambiguity를 감지한다.
- [x] ambiguity 시 theta 성분은 end-to-end scalar SDF 중앙차분으로 deterministic fallback한다.
- [x] fallback 결과에 nonsmooth=true를 기록한다.
- [x] exact contact와 norm(z_star)<=contact_tol에서 0으로 나누지 않는다.
- [x] 접촉 시 선택한 one-sided/subgradient 정책과 보장 한계를 명시한다.
- [x] 어떤 branch에서도 NaN 또는 Inf를 조용히 반환하지 않는다.

Acceptance:

- [x] smooth pose에서 grad_h가 end-to-end SDF finite difference와 일치한다.
- [x] translation gradient가 여러 orientation에서 finite difference와 일치한다.
- [x] theta=0 aligned rectangle에서 index mismatch, singular solve, NaN/Inf가 없다.
- [x] 같은 nonsmooth 입력을 반복하면 같은 active/fallback 결과를 얻는다.
- [x] contact test가 정의된 diagnostic을 반환하고 crash하지 않는다.

## 5. 현재 3-state 모델용 control QP

- [x] 현재 dynamics의 input matrix B(x)를 사용한다.
  - [x] 첫 번째 열은 [cos(theta), sin(theta), 0]^T이다.
  - [x] 두 번째 열은 [0, 0, 1]^T이다.
- [x] 각 component-obstacle pair에 독립 hard CBF row를 추가한다.
- [x] CBF 식을 grad_h @ B(x) @ [v,omega] + gamma*h >= epsilon으로 구현한다.
- [x] local trajectory 마지막 pose를 pointwise reference로 사용한다.
- [x] angle error를 [-pi,pi]로 wrap한다.
- [x] pose-error CLF V와 grad_V를 구현한다.
- [x] grad_V @ B(x) @ u + c*V <= delta를 추가한다.
- [x] delta>=0과 input bounds를 추가한다.
- [x] objective를 u.T R u + p_clf delta^2로 구성한다.
- [x] CBF slack은 추가하지 않는다.
- [x] no-obstacle case에서도 CLF-QP가 동작하게 한다.
- [x] hard CBF infeasible이면 임의 fallback 입력 없이 failure를 전달한다.
- [x] 마지막 h, grad_h, CBF/CLF residual, active/fallback 상태를 보관한다.

Acceptance:

- [x] 반환 순서가 항상 [v,omega]이고 bounds를 만족한다.
- [x] 성공한 solve의 모든 hard CBF residual이 -tol 이상이다.
- [x] CLF residual이 slack을 포함해 tolerance 이내다.
- [x] safe, near-boundary, penetrating state에서 QP row 부호가 수치 검증과 일치한다.
- [x] epsilon>0 unsafe-start test에서 feasible한 동안 h가 회복 방향으로 변한다.

## 6. Simulation과 config 최소 통합

- [x] sim/simulation_mpc.py에 optimizer_type == minkowski_cbf 분기를 추가한다.
- [x] 해당 분기에서 Param과 Optimizer를 import하고 생성한다.
- [x] config의 minkowski_cbf block이 있으면 기존 parameter override를 사용한다.
- [x] simulation docstring optimizer 목록에 minkowski_cbf를 추가한다.
- [x] simulation_mpc.py와 test_nmpc.py의 output prefix를 추가한다.
- [x] choices 제한이 없는 CLI parser는 불필요하게 수정하지 않는다.
- [x] 새 optimizer를 PSDF/PED obstacle-detection monkey patch 대상에 넣지 않는다.
- [x] benchmark의 hard-coded optimizer 목록은 수정하지 않는다.
- [x] integration test에서 실제 optimizer class를 확인해 unknown fallback을 막는다.

Acceptance:

- [x] CLI의 --optimizer-type minkowski_cbf가 정확한 optimizer class를 선택한다.
- [x] config override가 지정된 값만 변경한다.
- [x] 한 tick의 setup, solve, plant input, logger 흐름이 완료된다.

## 7. 자동 테스트

- [x] tests/test_minkowski_cbf.py를 추가한다.
- [x] CO geometry test를 작성한다.
  - [x] rectangle analytic CO bounds
  - [x] vertex-order invariance
  - [x] normalized/minimal H-rep
  - [x] origin containment branches
- [x] signed-distance test를 작성한다.
  - [x] separated/contact/penetrating values
  - [x] projection-QP KKT residuals
  - [x] penetration depth와 active facet
  - [x] multiple obstacles와 component pairs
- [x] gradient test를 작성한다.
  - [x] translation vs finite difference
  - [x] smooth theta vs finite difference
  - [x] aligned parallel-facet fallback
  - [x] contact diagnostic
- [x] control-QP test를 작성한다.
  - [x] input bounds
  - [x] hard CBF residuals
  - [x] CLF/slack residual
  - [x] no-obstacle solve
  - [x] infeasible status propagation
- [x] controller contract test를 작성한다.
  - [x] value(u).shape==(2,1)
  - [x] value(x).shape==(3,2)
  - [x] logger compatibility
  - [x] solver status와 solver_times
- [x] simulation integration test에서 실제 class와 1-tick 실행을 확인한다.

Acceptance:

- [x] python3 -m unittest tests.test_minkowski_cbf 가 통과한다.
- [x] python3 -m unittest tests.test_psdf 가 통과한다.
- [x] smooth와 nonsmooth test가 모두 포함된다.

## 8. Scenario smoke와 Definition of Done

- [x] 다음 s_path smoke를 실행한다.

      python3 test_nmpc.py --single --optimizer-type minkowski_cbf --maze-type s_path --robot-shape rectangle --simulation-time 2 --no-animation --no-plots

- [x] oblique_maze smoke로 회전과 일반 convex polygon을 검증한다.
- [x] unsafe-start closed-loop test로 penetration branch와 회복을 확인한다.
- [x] multi-obstacle closed-loop test에서 모든 pairwise CBF residual을 확인한다.
- [x] straight_corridor 실행 전 d_safe < 0.009 m인지 확인한다.
- [x] 기존 PSDF 짧은 smoke를 시도하고 환경 차단 사유를 기록한다.
- [x] NaN/Inf, silent constraint drop, input-bound 위반이 없는지 확인한다.
- [x] CO, SDF/gradient, control-QP runtime을 분리 측정한다.
- [x] 병목이 없으면 edge merge, solver cache, warm start를 추가하지 않는다.
- [x] penetration/contact/nonsmooth 정책을 module docstring에 기록한다.
- [x] 최종 diff가 새 optimizer, 최소 simulation 연결, tests에 집중되어 있는지 검토한다.

Definition of Done:

- [x] minkowski_cbf가 현재 3-state model과 BaseController 계약으로 실행된다.
- [x] exact MD-space signed distance의 분리와 침투 branch가 검증된다.
- [x] smooth gradient와 deterministic nonsmooth fallback이 모두 검증된다.
- [x] hard CBF/soft CLF QP residual이 tolerance를 만족한다.
- [x] Minkowski unit/integration, targeted PSDF 회귀, 필수 Minkowski smoke가 통과한다.
- [x] 모든 acceptance checkbox가 [x]이다.

## 검증 기록 (2026-09-01)

- 착수 시 사용자 변경 `control/psdf_optimizer.py`와 untracked `references/`를 확인했고 수정하지 않았다.
- WSL runtime: numpy 1.24.4, scipy 1.10.1, osqp 1.0.4, casadi 3.7.1.
- `python3 -m unittest -q tests.test_minkowski_cbf tests.test_psdf`: 20 tests 통과(Minkowski 19 + PSDF 1).
- 전체 discovery는 83개 중 81개 통과. 남은 2개는 변경하지 않은 RMPCC 기본값/시각화 테스트의 기존 불일치다.
- 실제 CLI `s_path` 2초와 `oblique_maze` 0.3초 smoke가 bounds/residual 오류 없이 완료됐다.
- 기존 PSDF 1-tick smoke는 acados 외부 런타임 `libqpOASES_e.so` 부재로 setup 단계에서 차단됐다. 생성된 codegen 산출물은 정리했다.
- 100회 microbenchmark 평균: CO 0.84 ms, SDF+gradient 6.32 ms, control QP 1.48 ms. 1차 구현에는 cache/warm-start를 추가하지 않았다.
- 기존 `PolytopeRegion`의 H-to-V 재변환은 near-parallel facet에서 불안정했으므로, geometry 모듈은 변경하지 않고 optimizer 내부에서 SciPy hull의 vertices/equations를 함께 사용했다.
