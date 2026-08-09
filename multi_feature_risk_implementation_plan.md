# RMPCC Multi-Feature Risk 구현 계획

## Tracking 규칙

- [x] 구현을 시작할 때 이 계획을 저장소 루트에 저장한다.
- [x] 각 항목은 구현과 해당 검증이 모두 끝난 뒤에만 `[x]`로 변경한다.
- [x] 하위 항목이 모두 완료된 뒤 상위 단계의 완료 여부를 갱신한다.
- [x] `rmpcc_pv` 분리 완료 전에는 기존 `rmpcc_optimizer.py`의 MF 변경을 시작하지 않는다.

## 1. 기존 `rmpcc_pv` 동작 분리

- [x] 현재 projected-variance 구현을 `control/rmpcc_pv_optimizer.py`로 분리하고 `RMPCCPVOptimizerParam`, `RMPCCPVOptimizer`로 명명한다.
- [x] PV optimizer는 독립 구현으로 유지하고 새 MF optimizer를 상속하거나 공유 분기하지 않는다.
- [x] PV 파라미터에서 projected-variance 모드를 고정하고 risk-tensor/MF 분기를 제거한다.
- [x] JSON과 code-generation 이름을 `rmpcc_pv` 전용으로 변경한다.
- [x] `sim/simulation_mpc.py`가 `rmpcc`와 `rmpcc_pv`를 별도 모듈에서 직접 생성하도록 변경한다.
- [x] PV 설정은 기존 `rmpcc` 블록을 기본값으로 읽고 `rmpcc_pv` 블록으로 덮어쓸 수 있게 한다.
- [x] optimizer 이름, CSV, animation, safe-stop batch에서 `rmpcc_pv` 이름이 유지되는지 확인한다.

## 2. Augmented PSDF MF 출력 완성

- [x] 기존 `forward(poses) -> (phi, gradient)` 계약을 보존한다.
- [x] `PSDF.forward_mf(A, B, mask, z_bar, epsilon, d_min)` batched API를 추가한다.
- [x] `z_bar`의 `(H, 8)` 상태 순서와 dtype/device/finite 조건을 검증한다.
- [x] 기존 feature distance, gradient, validity mask를 재사용해 `A_mf`, `c_mf`를 계산한다.
- [x] runtime autograd, top-k, 거리 가중치, Gaussian margin factor를 추가하지 않는다.
- [x] scalar 또는 `(H,)` epsilon을 stage별 tensor로 정규화한다.
- [x] 출력 shape `(H,)`, `(H,3)`, `(H,8)`, `(H,)`를 보장한다.
- [x] empty obstacle에서 `A_mf=0`, `c_mf=epsilon`을 반환한다.
- [x] `AugmentedPSDFWrapper.forward_mf(z_bar, epsilon, d_min)`을 추가한다.

## 3. RMPCC를 MF affine constraint로 교체

- [x] legacy risk tensor, L4CasADi, projected-variance 분기와 관련 파라미터를 제거한다.
- [x] covariance dynamics와 noise/PSD 관련 파라미터는 유지한다.
- [x] `chance_epsilon`, `d_min`을 `i=0,...,N` 전체에 적용하고 risk-active stage window를 제거한다.
- [x] acados parameter를 `[path(6), A_mf(8), c_mf(1)]` 15차원으로 고정한다.
- [x] obstacle constraint를 `A_mf^T z + c_mf >= 0` 한 행으로 교체한다.
- [x] 별도 deterministic PSDF guard와 개별 feature constraint를 추가하지 않는다.
- [x] path와 terminal constraint를 모두 등록한다.
- [x] obstacle constraint가 꺼져도 parameter dimension은 15로 유지한다.
- [x] RMPCC는 `AugmentedPSDFWrapper`만 초기화한다.

## 4. Shifted nominal과 online parameter update

- [x] 첫 solve에서는 warm-start trajectory를 nominal로 사용한다.
- [x] 이후 solve에서는 이전 state/input solution을 한 stage shift한다.
- [x] stage 0을 현재 pose/progress/초기 covariance로 덮어쓴다.
- [x] 중간 stage는 이전 solution의 다음 stage pose/progress를 사용한다.
- [x] terminal pose/progress를 마지막 입력으로 한 step rollout한다.
- [x] shifted physical input으로 horizon covariance를 다시 전파한다.
- [x] shifted progress에서 MPCC path anchor를 생성한다.
- [x] `z_bar` 전체를 한 번에 augmented PSDF에 전달한다.
- [x] path와 terminal에 `[path, A_mf, c_mf]`를 설정한다.
- [x] backup solver도 solve 직전에 MF 파라미터를 다시 생성한다.
- [x] backup/safe-stop trajectory를 다음 cycle shift 원본으로 사용할 수 있게 한다.

## 5. 진단 API와 설정 문서 정리

- [x] `get_last_mf_residual_trajectory()`를 추가한다.
- [x] feature violation probability sum 조회 API를 추가한다.
- [x] 거리 단위 risk-margin getter는 PV optimizer에만 유지한다.
- [x] MF debug 출력에 residual, probability sum, clearance, solve mode를 포함한다.
- [x] `config.yaml`의 optimizer 설명을 갱신한다.
- [x] legacy RMPCC risk 설정을 제거하거나 PV 전용으로 이동한다.

## 6. 테스트와 완료 조건

- [x] 기존 augmented PSDF 테스트가 계속 통과한다.
- [x] nominal affine identity를 검증한다.
- [x] pose/covariance analytic Jacobian을 central finite difference와 비교한다.
- [x] 회전, mask, stage별 epsilon, empty obstacle을 테스트한다.
- [x] MF parameter 15차원과 CasADi affine constraint를 검증한다.
- [x] path/terminal MF constraint 등록을 확인한다.
- [x] shifted nominal과 covariance 재전파를 테스트한다.
- [x] `rmpcc`/`rmpcc_pv` routing을 테스트한다.
- [x] 단위 테스트와 `py_compile`을 통과한다.
- [x] MF와 PV 짧은 smoke simulation을 실행한다.
- [x] MF에 legacy risk tensor/L4CasADi 호출이 없고 PV 출력 명명 회귀가 없음을 확인한다.

## 확정 가정

- MF RMPCC는 기존 risk-tensor RMPCC를 대체한다.
- `rmpcc_pv`는 분리 시점의 projected-variance 동작을 유지한다.
- MF 조건은 초기, 중간, terminal stage에 각각 한 행 적용한다.
- feature 수는 acados parameter dimension이나 constraint row 수를 바꾸지 않는다.
- covariance floor는 covariance 양의 정부호 유지에만 사용하며 feature variance에 별도 수치 분산을 더하지 않는다.
