## CC-SF 폭 증가 실험 (2026-10-27)

사용자가 최종 지정한 여섯 폭에서 각 3회, 총 **18 trials**를 새로 실행했다. **목표 도달 16/18, safe-stop step 23회, 연속 safe-stop 구간 4개**였다. 0.120 m 이상에서는 장애물 겹침이 관측됐으므로, 목표 도달을 무충돌 성공으로 해석하면 안 된다. 한 폭에서 3회 모두 목표 도달에 실패한 경우는 없어 여섯 폭을 모두 실행했다.

### 실험 조건과 측정 정의

- 환경은 기존 maze이며 직선 corridor 폭 0.15 m, rectangle 길이 0.15 m를 유지했다. Robot width는 0.105 / 0.110 / 0.115 / 0.120 / 0.125 / 0.130 m이다. DD, A*, seed 0–2, dt=0.1 s, trial당 최대 60 s, localization error 비활성화 조건이다.
- 시작 자세 교란은 x/y 표준편차 0.02 m, heading 표준편차 0.05 rad, 각 성분 ±3σ 제한이다. **사용자 선택에 따라 초기 clearance 조건을 여섯 폭 모두 ≥ 0.001 m로 통일했다.** 0.130 m에서는 정렬된 중앙 자세의 양측 여유가 각 10 mm이므로 기존 ≥10 mm 조건과 연속 Gaussian 자세 교란을 함께 만족하는 후보를 생성하지 못한다. 최종 실험에서는 seed별 시작 자세가 여섯 폭에서 정확히 일치함을 검증했다.
- CC-SF는 기존 기본값 d_min=0.001 m, chance_epsilon=0.20, gamma=0.841621, horizon=20 (2 s), robust horizon=10을 유지했다. 초기 covariance는 매 cycle sigma_f0=0.0002 m, sigma_l0=0.00025 m, sigma_psi0=0.00025 rad로 설정된다. 전체 설정은 [optimizer parameters](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/width_105mm/optimizer_parameters.json)에 보관했다.
- 실패 기준은 solver 연속 실패 20회 또는 4 s 동안 이동량 ≤0.03 m이다. 목표 도달은 기존 판정이다. 실행 소스를 사본으로 고정하고 같은 Python 3.10.13 / PyTorch 2.4.1 / CasADi 3.7.0 / RTX 3060 환경, PyTorch CPU thread 6개에서 trial을 순차 실행했다. Animation/plot은 비활성화했다.
- **Clearance**는 기존 표와 동일하게 rectangle–장애물 polygon 거리와 지도 경계 margin의 최솟값이다. 초기 및 모든 적분 후 자세(최종 자세 포함)를 사용하며, 실패 trial도 포함한 전체 표본을 합쳐 mean/p05/min을 계산했다. 장애물 접촉·겹침의 polygon 거리는 0으로 집계되고, **음수는 지도 경계 침범**을 뜻한다. 따라서 음수의 크기를 일반적인 장애물 침투 깊이로 해석하면 안 된다. 장애물 겹침은 Shapely 교차 면적으로 별도 검증했다.
- **Safe stop 횟수는 zero-input을 적용한 control step 수**이다. 연속 20 step은 20회·2.0 s이며 연속 정지 구간은 1개이다. 아래 seed별 횟수는 seed 0 / 1 / 2 순서다. 기록된 자세 사이의 연속시간 충돌은 검사하지 않았다.
- 초기 clearance ≥10 mm로 수행하다 조건 변경 전에 중단한 예비 실험은 [별도 폴더](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/pilot_initial_clearance10mm)에 보존하고 최종 집계에서 제외했다. 비교용 0.100 m 3회는 아래 본 실험 18회에 포함하지 않는다.

### 폭별 결과

| Robot width (m) | 목표 도달 | Mean clearance (mm) | P05 (mm) | Min (mm) | Safe-stop step: seed 0 / 1 / 2 | 장애물 겹침 표본 |
|---|---:|---:|---:|---:|---:|---:|
| 0.105 | 3/3 | 7.425 | 2.240 | 1.863 | 0/0/0 | 0 |
| 0.110 | 2/3 | 6.040 | 1.832 | 0.822 | 20/0/0 | 0 |
| 0.115 | 3/3 | 5.150 | 1.987 | 0.117 | 0/0/0 | 0 |
| 0.120 | 3/3 | 4.334 | 1.680 | -0.251 | 0/0/0 | 13 |
| 0.125 | 3/3 | 3.857 | 0.754 | -3.704 | 0/1/0 | 37 |
| 0.130 | 2/3 | 2.290 | 0.000 | -11.802 | 0/1/1 | 178 |

비교용 **0.100 m**는 목표 도달 3/3, safe stop 0/0/0회, mean/p05/min clearance는 각각 8.571 / 2.362 / 1.961 mm였다. 이 기준 실행의 초기 clearance 조건은 ≥10 mm였으며, 실제 seed별 시작 자세는 최종 여섯 폭과 같았다. 상단의 과거 5-trial 표와는 표본 수가 다르다.

**0.120 / 0.125 / 0.130 m의 모든 trial에서 장애물 겹침이 있었다.** 지도 경계 침범 표본은 각각 1 / 8 / 38개이며 장애물 겹침 표본과 중복될 수 있다. 기록된 모든 자세에서 clearance>0을 유지하면서 목표에 도달한 trial은 폭별로 3 / 2 / 3 / 0 / 0 / 0회였다. 0.115 m도 최소 clearance가 0.117 mm로 d_min=1 mm보다 작아, safe stop이 없다는 사실만으로 실제 geometric margin 준수를 판단할 수 없다.

### Trial별 결과

| Width (m) | Seed | 종료 상태 | 종료 시간 (s) | Mean clearance (mm) | Min (mm) | Safe-stop step | 겹침 표본 |
|---|---:|---|---:|---:|---:|---:|---:|
| 0.105 | 0 | 목표 도달 | 30.0 | 7.483 | 1.866 | 0 | 0 |
| 0.105 | 1 | 목표 도달 | 30.2 | 7.357 | 1.866 | 0 | 0 |
| 0.105 | 2 | 목표 도달 | 30.0 | 7.435 | 1.863 | 0 | 0 |
| 0.110 | 0 | 실패 | 26.0 | 6.215 | 0.822 | 20 | 0 |
| 0.110 | 1 | 목표 도달 | 32.0 | 5.945 | 1.466 | 0 | 0 |
| 0.110 | 2 | 목표 도달 | 31.8 | 5.992 | 1.402 | 0 | 0 |
| 0.115 | 0 | 목표 도달 | 33.2 | 5.225 | 1.406 | 0 | 0 |
| 0.115 | 1 | 목표 도달 | 33.8 | 5.072 | 0.587 | 0 | 0 |
| 0.115 | 2 | 목표 도달 | 33.2 | 5.155 | 0.117 | 0 | 0 |
| 0.120 | 0 | 목표 도달 | 36.2 | 4.329 | 0.000 | 0 | 4 |
| 0.120 | 1 | 목표 도달 | 36.1 | 4.333 | -0.251 | 0 | 6 |
| 0.120 | 2 | 목표 도달 | 36.0 | 4.340 | 0.000 | 0 | 3 |
| 0.125 | 0 | 목표 도달 | 40.7 | 3.891 | -3.704 | 0 | 9 |
| 0.125 | 1 | 목표 도달 | 40.7 | 3.872 | -1.141 | 1 | 15 |
| 0.125 | 2 | 목표 도달 | 40.5 | 3.809 | -3.634 | 0 | 13 |
| 0.130 | 0 | 실패 | 17.0 | 2.080 | -2.881 | 0 | 41 |
| 0.130 | 1 | 목표 도달 | 47.9 | 2.328 | -4.584 | 1 | 72 |
| 0.130 | 2 | 목표 도달 | 44.3 | 2.330 | -11.802 | 1 | 65 |

### Safe stop과 실패 사유

- **0.110 m, seed 0**: simulation time 24.0–26.0 s에 safe stop 20 step이 연속 발생했고 `consecutive_solver_failures(20)`로 종료했다. 나머지 두 seed는 safe stop 없이 목표에 도달했다.
- **0.125 m, seed 1**: 10.6–10.7 s에 1 step 정지 후 재개해 목표에 도달했다.
- **0.130 m, seed 1 / 2**: 각각 6.7–6.8 s / 22.2–22.3 s에 1 step 정지 후 재개해 목표에 도달했다. **seed 0의 실패는 safe stop 없이** 17.0 s에 `stuck_for_4.00s_disp_0.0281m_le_0.0300m` 기준으로 종료된 것이다.
- Safe-stop step 23회 모두 주 SQP-RTI는 acados status=4 (**QP solver failure**), 복구 SQP_WITH_FEASIBLE_QP는 status=2 (**maximum iterations**)를 반환했다. 복구 solver의 성공 횟수는 0회였다. 이 반환 코드는 비선형 문제의 수학적 infeasibility를 증명하지 않는다. [정지 구간](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/safe_stop_events.csv), [상태·margin 전후 기록](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/failure_diagnostics.json)에 근거를 보관했다.

### Minimum-feature margin 가설과의 관계

[method_description.md §4.1 및 §5.5](/home/taek111/projects/psdf_mpcc/method_description.md:290)의 SF margin은 다음과 같다.

$$
r^{\mathrm{SF}}=\Phi^{-1}(1-\varepsilon)\sqrt{q^\top\tilde\Sigma q},\qquad q=T(\theta)^\top g,\qquad h^{\mathrm{SF}}=\phi-d_{\min}-r^{\mathrm{SF}}.
$$

최소 feature 전환으로 gradient가 바뀌면 projected variance와 margin이 불연속일 수 있다. 다만 전환 전후의 projected variance가 같으면 margin도 같으며, 특히 q가 -q로 바뀌는 경우 quadratic form은 변하지 않는다. 따라서 최소 feature 선택만으로 항상 margin 점프나 infeasibility가 생기는 것은 아니다.

이번 실험에서는 0.110 m에서 지속적인 solver 실패가 나타났지만 0.115 / 0.120 m에서는 safe stop이 없었고, 더 큰 폭의 정지 3건은 다음 step에 복구됐다. 실패 빈도는 폭에 대해 단조롭지 않았다. 0.110 m seed 0의 첫 정지 직전/시작 시점에서 현재 자세의 진단용 margin은 **0.210407 → 0.210406 mm**로 급증하지 않았다. 미래 stage의 feature 전환이나 RTI nominal geometry의 영향까지 배제하는 결과는 아니다.

생성된 실제 solver JSON은 **nh_0=0, nh=nh_e=1, slack=0**이었다. 즉 stage 0에는 obstacle row가 없고, 미래 stage 1–20에 hard SF row가 적용된다. 따라서 stage-0 residual이 음수라는 기록만으로 ‘고정 초기 상태 제약 위반 때문에 infeasible’이라고 해석할 수 없다. [주·복구 solver의 stage 정책](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/solver_stage_policy.json)을 확인했다. 저장된 자세에서의 겹침도 존재하므로, SQP-RTI가 반환한 성공 및 affine constraint 처리를 exact nonlinear clearance 보장과 동일시할 수 없다.

이 결과는 폭별 현상 측정이다. Feature ID의 전환 추적이나 risk margin을 끄는 대조 실험은 수행하지 않았으므로 **‘feature switch → margin 불연속 → solver 실패’의 인과관계는 이번 실험만으로 확정하지 않았다.**

### 원본 자료와 검증

- [폭별 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/benchmark_summary.csv), [trial별 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/benchmark_trials.csv), [모든 자세의 clearance](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/clearance_samples.csv).
- [겹침·경계 침범 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/geometry_violations_by_trial.csv), [위반 자세와 교차 면적](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/geometry_violations.csv), [nominal risk 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/nominal_risk_summary.csv). 각 폭 폴더에 `trial_results.csv`, `cycles.csv`, `nominal_risk.jsonl`, runtime JSON, 원본 history 및 실행 로그를 보관했다.
- [검증 결과](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/validation.json): source 사본 불변성, 폭 외 geometry·optimizer parameter 동일성, seed별 시작 자세 동일성, 초기·최종 자세 포함, mean/p05 독립 재계산, safe-stop 누적수·step 로그·zero-input 및 solver CSV 대조를 통과했다. Shapely로 모든 비양수 clearance 표본을 포함한 총 945개 자세의 거리를 독립 검증했다(비교용 기준 실행 포함).
- 재현용 [순차 실행](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/run_sweep.py)·[단일 폭 실행·관측](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/run_benchmark.py)·[clearance 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/analyze.py)·[겹침 분류](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/classify_violations.py)·[정지 시점 분석](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_width_sweep_3trials_20261027_232811/diagnose_stops.py)을 보관했다.
