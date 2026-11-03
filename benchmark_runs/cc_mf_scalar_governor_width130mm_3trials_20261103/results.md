# 임시 scalar governor 실험 — CC-MF, rectangle 0.150 × 0.130 m

2026-11-03, seed 0–2 각 1회. maze / DD / A*, dt=0.1 s, horizon=20, 최대 80 s, 이동 정체 10 s / 0.03 m, 연속 제어 실패 20회. Localization error 없음. 앞선 실험과 동일한 controller 파라미터와 시작 자세 sampling을 사용했다.

임시 실행 파일에서 런타임 hook만 적용했다. 기존 controller/model 소스 파일은 변경하지 않았다. 기존 GPU batch의 현재 stage 중간값을 재사용하며 추가 PSDF 평가, 후보 rollout, NMPC 재solve는 없다. 정적 장애물의 face normal은 supporting plane 후보로 함께 사용한다. Endpoint 여유는 min(5 µm, 현재 잔여 여유의 10%)다.

입력 변화율 허용 λ와 안전 λ가 겹치지 않으면 기존 emergency-stop 예외와 같이 영입력을 적용하고 별도 기록했다. 이 경우 일반 rate bound는 위반할 수 있다. λ<1이면 다음 주기에 실제 위치에서 progress를 다시 투영한다.

## Trial 결과

| Seed | 결과 | 종료 시각 (s) | 최소 signed clearance (mm) | Mean/P05 signed clearance (mm) | Mean/P95 cycle (ms) | Mean/P95 governor (ms) | 입력 축소 step | Governor 긴급정지 step | 기존 safe stop step | Animation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | success | 74.5 | 0.135712251 | 2.093 / 0.498 | 40.083 / 48.769 | 0.933 / 1.153 | 231 | 3 | 0 | [MP4](../../animations/cc_mf/MPC_cc_mf_rectangle_maze_width130mm_seed0_widthbench_20261103_150325_startseed0.mp4) |
| 1 | failure | 7.1 | 0.050016353 | 1.496 / 0.050 | 34.461 / 53.849 | 0.486 / 0.855 | 7 | 0 | 20 | [MP4](../../animations/cc_mf/MPC_cc_mf_rectangle_maze_width130mm_seed1_widthbench_20261103_150325_startseed1.mp4) |
| 2 | failure | 21.7 | 0.050005722 | 2.376 / 0.050 | 34.998 / 49.361 | 0.700 / 1.011 | 67 | 1 | 20 | [MP4](../../animations/cc_mf/MPC_cc_mf_rectangle_maze_width130mm_seed2_widthbench_20261103_150325_startseed2.mp4) |

총 3회 중 1회 성공. 저장된 자세 1036개에서 signed penetration 표본 0개. 최소 signed clearance 0.050005722 mm. 전체 mean/P05 2.111/0.132 mm.

## 정지 원인

| Seed | 첫 stage-0 정지 (s) | 실시간 float32 φ (mm) | 정밀 기하 φ (mm) |
|---:|---:|---:|---:|
| 1 | 5.1 | 0.049998336 | 0.050016353 |
| 2 | 19.7 | 0.049975879 | 0.050005722 |

두 실패 모두 기하 침범 없이 d_col=0.05 mm 부근까지 감속했다. Endpoint reserve가 현재 여유와 함께 작아지면서 실시간 float32 PSDF와 정밀 기하 평가의 차이가 strict stage-0 정지를 유발했다. 이후 20 step 연속 정지로 failure가 됐다.

## 계산 시간과 한계

첫 cycle 제외, 전체 유효 step pooled mean/P95 cycle = 38.635/49.086 ms. Governor 계산 및 GPU 중간값 cache 비용은 0.854/1.108 ms. Governor CSV 쓰기는 governor 전용 timer에서 제외되지만 전체 cycle에는 포함된다.

100 ms 초과 cycle 1/1030개, 최대 100.229 ms. p95가 제어 주기 미만이라는 사실이 hard deadline 보장을 뜻하지 않는다.

Governor 긴급정지 4 step에서는 입력 변화율 bound를 넘었고, 두 trial에는 수치 임계값 정지가 남았다. 전체 주행을 완성하는 해결책이나 원래 nonlinear MF 확률 조건의 보장을 검증한 결과는 아니다. Signed geometry는 기록된 실제 자세들을 float64로 독립 평가했으며, 새로운 trajectory 적분은 하지 않았다.

Animation은 기존 animate_world 코드를 사용해 animations/cc_mf에 frame_skip=5로 저장했다. 별도 재인코딩은 하지 않았다.

[방법과 수식](../../scalar_safety_filter.md) · [Trial CSV](trial_results.csv) · [상세 집계](governor_summary.csv) · [전체 집계](aggregate.json) · [설정](experiment.json) · [Deadline 집계](deadline_metrics.json)

이 방법의 임시 실행 소스와 보존용 사본은 사용자 요청으로 삭제했다. 실험 로그와 Animation은 보존했다.
