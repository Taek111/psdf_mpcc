# 이전 125 mm Guard ablation 참고 기록

이 문서는 기존 프로젝트 보고서에서 보존한 배경 자료다. 여기에서 언급한 이전 10회·baseline 실험의 raw data는 이번 120회 패키지 범위에 포함하지 않는다. 본문의 원본 프로젝트 경로는 출처 기록이며 패키지 내부 링크가 아니다. 현재 120회 원본과 분석 결과는 [결과 문서](../results.md)를 참고한다.

# Width 0.125 m — CC-SF Guard / stage-0 ablation

2026-11-04(KST), 새 CC-SF **10/10 trials 완료**. 비교 대상은 2026-11-03의 동일 폭·seed 0–9 CC-SF 및 CC-MF다. 기존 결과는 다시 실행하거나 덮어쓰지 않았다.

## 변경 및 재현 조건

- CC-SF의 미래 stages **1–20(terminal 포함)**에 hard PSDF Guard와 기존 hard SF chance row를 적용했다.
- Stage 0의 OCP safety rows는 비활성화하고, 현재 exact PSDF가 `d_col=0.00005 m`보다 작을 때만 solver 호출 없이 `safe_stop_stage0`를 적용했다.
- SF chance row에는 slack이나 MF의 nominal-domain gate를 추가하지 않았다. 기존 SF는 stages 0–19에 hard chance row를 적용하고 terminal row가 없었다.
- 기존 실행의 source snapshot에서 controller는 `control/cc_sf_optimizer.py`만 교체했다. 실행 harness의 methods-subset 지원도 갱신했으며, simulation/model/planning 소스는 이전 버전 그대로다.
- Rectangle 0.150 × 0.125 m, maze / differential drive / A*, dt=0.1 s, horizon=20, trial 제한 80 s, 정체 판정 10 s / 0.03 m, 연속 실패 20회. Localization error, animation, plot은 비활성화했다.
- 기존 SF의 전체 파라미터를 고정하고 `d_col`만 추가했다. 시작 pose는 같은 seed별로 **10/10 정확히 일치**했다. 각 trial은 별도 프로세스에서 순차 실행했다.

## 결과 비교

Safe stop step은 zero-input fallback이 적용된 제어 cycle 수이고, episode는 연속된 stop 구간 하나다. 주행 시간은 성공 trial만, clearance는 초기 자세와 매 적분 후 모든 표본을 합쳐 집계한다.

| Method | 성공/완료 | Safe-stop steps | Episodes | 최장 연속 stop | 평균 성공 주행 시간 (s) | P05 clearance (mm) | 최소 clearance (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 기존 CC-SF | 2/10 | 178 | 26 | 20 | 45.95 | 0.424 | 0.000 |
| CC-SF + Guard / stage-0 제외 | **10/10** | **4** | **4** | **1** | **47.86** | **0.754** | **0.410** |
| 기존 CC-MF | 10/10 | 1 | 1 | 1 | 49.61 | 0.639 | 0.089 |

기존 SF의 평균 주행 시간은 성공한 2개 trial만의 값이므로 나머지 두 방법의 전체 10개 성공 trial 평균과 직접적인 속도 우열로 비교하지 않는다. 새 SF와 기존 MF 모두 sampled clearance가 0 이하인 표본은 없었다. 이 값은 이산 표본의 기하학적 clearance이며 확률적 안전 보장은 아니다.

새 SF의 seed 0–9 safe-stop step 수는 **[1, 1, 0, 1, 0, 0, 0, 0, 0, 1]**이다. 네 정지는 모두 solver status 4로 발생했고 다음 cycle에 성공적으로 복구됐다. Stage-0 Guard 정지는 0회다. 100 control cycles당 stop은 기존 SF 5.153, 새 SF 0.0836, 기존 MF 0.0202다. 실패로 조기 종료한 trial의 노출 시간 차이는 이 비율에도 남는다.

## Stage-0 사후 진단

기존 SF의 실패 8개 trial은 각각 마지막에 20-step 연속 정지가 있었으며, 이 **160개 cycle 모두** 측정 pose와 재초기화 covariance로 재평가한 stage-0 SF 잔차가 -1 µm보다 작았다. 나머지 18회 stop은 단발성이다. 새 SF에서는 같은 잔차의 음수 표본이 없었다.

이는 고정 초기상태의 hard chance row를 위반하면 zero-input fallback 후에도 같은 위반 상태에 머물 수 있다는 지속적 실패 구조와 일치한다. 다만 CPU float32 geometry 재생이며 GPU feature tie 선택 및 미세한 affine-anchor 보정 차이가 가능하다. -1 µm는 진단용 부호 기준이고 solver tolerance나 solver의 infeasibility certificate가 아니다. 최초 위반 진입을 switching이 유발했을 가능성까지 배제하지 않는다.

## 새 궤적에서의 switching-boundary 평가

새 SF의 **10개 seed 모두**, stage 10, progress 0.6–1.2 m를 고정해 오프라인 분석했다. 성공 cycle 중 직전 cycle도 성공한 표본에서 feature 전환 후보를 찾고, 보간 경계 양쪽에 동일한 왼쪽 covariance를 적용했다. 양쪽 모두 `phi > d_min`이고 MF가 활성·유효한 **376개 경계**를 비교했다. SF는 동일 quantile 단위로 변환했고 MF는 실제 구현의 inverse-CDF/linear-extension 잔차를 사용했다.

| 경계 양측 잔차 차이의 절댓값 | SF quantile | MF quantile |
|---|---:|---:|
| 중앙값 | 0.519299 | 0.000003796 |
| P95 | 1.661033 | 0.000018666 |
| 최대 | 1.809799 | 0.001291090 |

**375/376개 경계에서 MF 변화가 작았다.** 372개는 일반 inverse-CDF 분기, 4개는 양쪽 모두 linear-extension 분기다. SF의 원래 거리 단위 잔차 변화는 중앙값 1.246 mm, P95 5.150 mm였다. Mask 변화는 없었지만 내부 feature 선택 변화는 남아 있다. 이는 선정된 경계에서의 국소적 민감도 차이를 뒷받침하며, float32 해상도보다 작은 보간폭을 사용한 결과를 엄밀한 연속성 증명이나 보편적인 감소 배율로 해석하지 않는다. 경계들은 서로 독립적인 trial도 아니다.

새 SF의 4개 stop 시점은 progress 약 1.306 / 1.311 / 1.766 / 1.312 m로 위 경계 평가 구간 밖이다. 해당 cycle의 재구성 stage-10 nominal geometry는 모두 penetration/SAT branch였으며, 3개는 직전 nominal 대비 selector가 바뀌고 1개는 바뀌지 않았다. 이는 separated-domain boundary 결과를 남은 네 stop의 직접적인 원인 설명으로 사용할 수 없음을 뜻한다. 실패 solver iterate는 fallback plan으로 덮어써져 저장되지 않았다.

## 해석과 추가 지표

- **지지되는 주장:** 활성 MF 잔차는 관측된 최근접-feature 전환 경계에서 SF보다 작은 변화를 보였다.
- **이번 실험에서 지지되지 않는 주장:** 이 차이 때문에 MF의 목표 도달 성공률이 높아졌다는 인과 해석. Stage 정책을 수정한 SF도 10/10 성공하여 관측된 성공률 격차가 사라졌다.
- Safe stop 4 대 1은 차이가 남지만, SF의 hard/ungated row와 MF의 soft/gated row가 아직 다르다. 10개 paired trial만으로 aggregation의 고유 효과를 확정하지 않는다.
- 같은 affine geometry에서 `d_min >= d_col`이고 risk가 비음수이면 hard SF row가 Guard를 이미 함의한다. 따라서 Guard 자체가 feasible set을 넓힌 것은 아니며, stage-0 제거·terminal 추가 및 수치적 영향이 함께 바뀐 비교다.
- 다음에는 모든 stage의 **설치·binding 여부, dual, slack, fresh/affine 잔차 차이, 실제 solver 실패 QP iterate와 row별 잔차**를 저장하는 것이 유용하다. Switch 발생률만 세기보다 동일 상태·covariance에서의 잔차/row-normal 변화와 실패 직전 시간 순서를 확인해야 한다.
- 인과 분리에는 같은 Guard/gate/slack/stage 정책 아래 single-feature quantile row와 multi-feature row를 비교하고, Guard-only를 별도 대조군으로 두는 실험이 필요하다.

초기 trial과 짧은 CPU 오프라인 분석이 겹쳤으므로 이번 batch의 wall-clock 계산 시간은 성능 우월성 주장에 사용하지 않는다. 주행 시간과 성공 판정은 시뮬레이션 시간 기준이다.

자료: 실행 조건 (원본 프로젝트 경로: `benchmark_runs/cc_sf_guard_width125_10trials_20261104/experiment.json`) · 10개 trial (원본 프로젝트 경로: `benchmark_runs/cc_sf_guard_width125_10trials_20261104/trial_results.csv`) · 비교 집계 및 stage-0 진단 (원본 프로젝트 경로: `benchmark_runs/cc_sf_guard_width125_10trials_20261104/sf_guard_comparison/comparison.json`) · Switching-boundary 원자료 (원본 프로젝트 경로: `benchmark_runs/cc_sf_guard_width125_10trials_20261104/sf_guard_comparison/switch_boundaries.json`) · 시작 pose·파라미터 검증 (원본 프로젝트 경로: `benchmark_runs/cc_sf_guard_width125_10trials_20261104/comparison_validation.json`) · 환경·계산시간 주석 (원본 프로젝트 경로: `benchmark_runs/cc_sf_guard_width125_10trials_20261104/execution_environment.json`).
