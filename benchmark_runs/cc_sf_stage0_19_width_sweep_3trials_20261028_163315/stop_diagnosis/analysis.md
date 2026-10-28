## 0.110 m: zero-input 후 정지 지속과 d_min 미만 진입 원인 (2026-10-28 추가 분석)

**0.110 m의 원인은 앞서 분석한 0.120 m의 최소 장애물 전환 사례와 다르다.** Zero input으로 미래 margin은 줄었지만 현재 stage-0 margin은 남아 복구되지 않았다. 그 위치에 진입한 직전 QP에서는 **covariance 동역학을 선형화한 결과 불확실성이 과소평가되고, risk-margin 접선이 음수가 되어 phi<1 mm인 해도 hard row를 만족**했다. 이번 세 입력에서는 phi tangent와 물리 kinematics 오차가 오히려 실제 clearance를 늘리는 방향이었다. 앞 절의 phi·kinematics 설명만으로는 이 현상을 설명할 수 없다.

### 1. Zero input인데 seed 1이 복구되지 않은 이유

입력 의존 noise는 이후 covariance 전파에 반영된다. 매 cycle의 현재 covariance는 설정 초기값(표준편차: forward 0.2 mm, lateral 0.25 mm, heading 0.25 mrad)으로 다시 고정된다. 따라서 현재 $r_0$는 현재 covariance와 gradient로 결정되며, 현재 입력을 0으로 바꿔도 변하지 않는다. 정지한다고 초기 위치·자세 불확실성까지 없어지지는 않는다.

Seed 1의 관측값은 다음과 같다. 단위는 mm이다.

| 값 | 첫 정지 시도 k43, t=4.3 s | Zero input 후 k44, t=4.4 s |
|---|---:|---:|
| phi(x0) | 1.056027 | 1.056027 |
| Stage-0 margin r0 | 0.209900 | 0.209900 |
| Stage-1 nominal margin | 1.560160 | 0.209900 |
| Stage-10 nominal margin | 3.743221 | 0.209900 |
| h0 = phi − 1 mm − r0 | −0.153874 | −0.153874 |

미래 margin은 실제로 크게 감소했다. 하지만 seed 1에서 허용되는 margin은 phi−d_min=**0.056027 mm**뿐이므로, 남은 **0.209900 mm**로도 현재 hard row를 위반한다. 이 잔여 margin은 `risk_margin_floor` 설정 때문이 아니라 비영 초기 covariance 때문이며 해당 파라미터는 0이다. Pose와 covariance를 포함한 x0 전체가 등식으로 고정되어 미래 입력으로 h0<0을 해소할 수 없다. k43–62의 fast/backup 총 40회 시도에서 동일한 phi0·r0·h0가 유지됐고, fast status=4 / backup status=2가 반복됐다.

코드 근거: [초기 covariance 및 x0 고정](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/source_snapshot/control/cc_sf_optimizer.py:812), [covariance 재전파](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/source_snapshot/control/cc_sf_optimizer.py:521), [zero-input plan 설치](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/source_snapshot/control/cc_sf_optimizer.py:943).

### 2. phi가 1 mm보다 작은 곳에 진입한 직접 원인

첫 정지 직전의 성공 입력 3개를 분석했다. 표의 QP phi는 고정 tangent 평가값, 재평가 phi는 QP가 반환한 다음 pose에서 다시 계산한 PSDF, 적용 후 phi는 실제 RK4 적분 후 PSDF이다. 모든 수치는 mm이며 세 입력 모두 solver status=0이었다.

| Seed | 입력 적용 구간 (s) | QP phi | QP risk 접선 | QP hard residual | 다음 pose 재평가 phi | 적용 후 phi |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4.1→4.2 | 0.839771 | -0.774102 | 0.613873 | 0.968312 | 0.983508 |
| 1 | 4.2→4.3 | 0.912194 | -0.779120 | 0.691314 | 1.041426 | 1.056027 |
| 2 | 4.0→4.1 | 0.805809 | -0.706762 | 0.512571 | 0.937196 | 0.952340 |

원래 nonlinear margin $r(\xi)=\gamma\|L(\xi)^Tq\|$는 비음수지만, QP는 그 접선 $r_{lin}=\bar r+\nabla_\xi r(\bar\xi)^T(\xi-\bar\xi)$을 사용한다. 이 접선의 비음수성은 보장되지 않는다.

예를 들어 seed 0은 다음처럼 QP row를 만족했다.

$$
h_{QP}=0.839771-1.000000-(-0.774102)=+0.613873\;\mathrm{mm}.
$$

**QP 내부에서 이미 phi가 1 mm보다 작았고, 음수 risk 접선이 이를 허용했다.** 원래 margin 자체가 음수가 된 것은 아니다. 같은 반환 covariance에서 nonlinear margin을 평가하면 seed별 **+0.114551 / +0.113762 / +0.125595 mm**이며, 재평가한 nonlinear SF residual은 세 seed 모두 음수였다.

### 3. Covariance와 margin 선형화가 크게 어긋난 경로

Seed 0의 nominal 각속도는 **−0.028147 rad/s**, 반환 입력은 **+0.636302 rad/s**로, 한 RTI step에서 0을 가로질러 **+0.664448 rad/s** 바뀌었다. 다른 두 seed도 같은 방향의 부호 전환이 있었다. Covariance의 process noise는 `|v*omega|`와 `omega**2`를 포함한다. 음수 omega에서 만든 국소 선형 모델을 큰 양수 omega까지 외삽하면 실제로 증가하는 회전 noise를 감소하는 방향으로 예측할 수 있다.

| Seed 0의 다음-stage 값 | Nominal anchor | QP 반환 covariance | 적용 입력을 원래 covariance 식으로 전파 |
|---|---:|---:|---:|
| Lateral 표준편차 (mm) | 0.634930 | 0.051732 | 1.548167 |
| 동일 frozen q에서 nonlinear margin (mm) | 0.583912 | 0.114551 | 1.584790 |

QP의 covariance는 같은 입력을 nonlinear map에 넣은 값보다 지나치게 작다. 이 잘못된 covariance 이동을 따라 margin 접선도 **−0.774102 mm**까지 내려갔다. 위 마지막 열은 동일 frozen gradient에서 재전파한 **계획 margin**이며, 다음 cycle에 초기 covariance로 reset하여 계산하는 r0=0.209877 mm와는 다른 값이다.

반면 seed 0의 거리 tangent는 0.839771 mm로 다음 pose의 정확한 PSDF 0.968312 mm보다 **0.128541 mm 작게** 계산했다. Kinematics 차이도 적용 후 거리를 **0.015196 mm 늘렸다**(pose 위치 차이 0.017903 mm). 나머지 두 seed도 같은 부호다. 세 사례 모두 최소 장애물 index는 anchor·반환 pose·적용 후 pose에서 **0**으로 같았다. 따라서 이 사례의 d_min 진입 원인을 최소 장애물 전환이나 거리 과대평가로 설명해서는 안 된다.

코드 근거: [입력 의존 noise](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/source_snapshot/control/cc_covariance.py:180), [Log-Cholesky 전이](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/source_snapshot/control/cc_covariance.py:229), [nonlinear margin 및 hard row](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/source_snapshot/control/cc_sf_optimizer.py:674).

### 검증 범위와 시사점

- 반환 states·inputs를 직접 저장한 로그를 사용했다. Linearized dynamics equality는 최대 3.95e−15 오차로, 실제 RK4 적용 결과는 로그와 정확히 일치했다. 153개 obstacle–pose 거리를 독립 polygon geometry로 대조했고 float32/64 PSDF 차이는 최대 0.000037 mm였다. Production CasADi constraint·dynamics 식으로 margin 접선과 covariance 계산도 독립 확인했다. 새 주행이나 최적화 solver 실행은 하지 않았다.
- Stage 0은 현재 위반을 검출하지만 직전 입력의 진입을 예방하지는 않는다. 수정 방향은 독립적인 deterministic phi≥d_min row로 음수 risk 접선의 안전거리 완화를 막고, covariance·입력 변화 제한 또는 재선형화, 입력 적용 전 nonlinear pose·covariance 안전성 검사를 함께 검토하는 것이다. Deterministic row도 affine phi 모델에 걸리므로 exact geometry 검사를 대체하지 않는다. 이 분석에서 controller 동작은 변경하지 않았다.
- [오차 분해 데이터](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stop_diagnosis/entry_decomposition.csv), [반환 상태·gradient 상세](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stop_diagnosis/entry_details.json), [정지 후 margin 데이터](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stop_margin_audit.json), [production 식 대조](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stop_diagnosis/production_expression_review.json), [검증 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stop_diagnosis/validation.json), [재현 스크립트](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stop_diagnosis/analyze_entry.py).
