## CC-SF 안전 제약 stage 0–19 재실험 (2026-10-28)

**0.105 m는 3/3 목표 도달·safe stop 0회, 0.110 m는 0/3 목표 도달·각 seed에서 safe stop 20 step이었다.** 기존에 허용한 전원 실패 시 생략 규칙에 따라 **0.115 / 0.120 / 0.125 / 0.130 m는 실행하지 않았다.** 총 6 trials를 완료했으며, 기록된 모든 자세에서 장애물 접촉·겹침 및 지도 경계 침범은 없었다. 실행한 6개 trial의 MP4도 저장했다.

### 변경과 실험 조건

- **N=20, dt=0.1 s, tf=2 s를 유지**했다. 상태는 x0–x20, 입력은 u0–u19이며, hard SF 안전 제약 20개의 적용 위치만 기존 stage 1–20에서 **stage 0–19**로 옮겼다. 현재 상태 x0는 고정하고 terminal state x20의 안전 제약은 제외했다. Horizon을 19개 제어 구간으로 줄인 실험이 아니다.
- 주·복구 solver의 실제 생성 JSON 12개 모두 **nh_0=1, nh=1, nh_e=0, slack=0**을 확인했다. Stage 0은 별도의 사전 검사로 처리하지 않고 OCP hard row에 포함했다.
- 이전 width sweep의 고정 소스에서 `control/cc_sf_optimizer.py`의 제약 stage 부분만 변경했다. 폭 이외의 설정은 동일하다: corridor 0.150 m, robot 길이 0.150 m, DD·A*, seed 0–2, 초기 clearance ≥1 mm, localization error 비활성화, d_min=1 mm, epsilon=0.20, robust horizon=10, 최대 60 s. 실행한 각 폭·seed의 시작 자세는 이전 실험과 정확히 같다.
- Clearance는 초기 및 모든 적분 후 자세를 포함한 polygon 거리와 지도 경계 margin의 최솟값이다. 실패 trial 및 정지 중 반복 자세도 포함해 pooled mean/p05/min을 계산했다. Safe-stop 횟수는 zero input을 적용한 **control step 수**이며, 연속 20 step=2 s=정지 구간 1개이다.

### 폭별 결과

| Width (m) | 목표 도달 | Mean clearance (mm) | P05 (mm) | Min (mm) | Safe-stop step: seed 0 / 1 / 2 | 겹침 표본 |
|---|---:|---:|---:|---:|---:|---:|
| 0.105 | 3/3 | 7.012 | 2.219 | 1.870 | 0/0/0 | 0 |
| 0.110 | 0/3 | 5.989 | 0.952 | 0.952 | 20/20/20 | 0 |
| 0.115 | 생략 | — | — | — | — | — |
| 0.120 | 생략 | — | — | — | — | — |
| 0.125 | 생략 | — | — | — | — | — |
| 0.130 | 생략 | — | — | — | — | — |

0.105 m의 seed 0 / 1 / 2 목표 도달 시간은 **29.1 / 29.2 / 28.9 s**였다. 기존 stage 1–20 실험의 0.110 m는 목표 도달 2/3, safe stop 20/0/0 step이었지만, 이번 stage 0–19 설정에서는 세 seed 모두 더 일찍 정지했다. 두 설정은 terminal row 유무도 다르며, 실패로 주행 길이가 달라지므로 pooled 평균 clearance만으로 안전성을 비교하지 않는다.

### 0.110 m의 정지 원인

정지 직전의 현재 상태에서 $h_0=\phi(x_0)-d_{\min}-r_0$가 음수였다. 아래 phi는 로그의 signed PSDF이며, polygon 거리와 최대 0.000031 mm 차이였다. 단위는 mm이다.

| Seed | Safe-stop 구간 (s) | phi(x0) | Risk margin r0 | d_min+r0 | h0 | 종료 사유 |
|---|---:|---:|---:|---:|---:|---|
| 0 | 4.2–6.2 | 0.983508 | 0.209877 | 1.209877 | -0.226369 | solver 연속 실패 20회 |
| 1 | 4.3–6.3 | 1.056027 | 0.209900 | 1.209900 | -0.153874 | solver 연속 실패 20회 |
| 2 | 4.1–6.1 | 0.952340 | 0.209871 | 1.209871 | -0.257531 | solver 연속 실패 20회 |

- 실패 60 step 모두 주 SQP-RTI **status=4**, 복구 SQP_WITH_FEASIBLE_QP **status=2**였고, 복구 성공은 0회였다. Stage-0 residual이 음수인 60 step 모두 safe stop으로 이어졌으며, 음수인데 solver 성공으로 통과한 step은 없었다.
- x0의 pose와 초기 covariance는 등식으로 고정된다. 따라서 현재 h0<0인 상태는 미래 입력으로 해소할 수 없고, zero input으로 멈춘 후 같은 조건이 유지되어 20회 연속 실패로 종료했다. Seed 1은 실제 거리가 1 mm보다 커도 risk margin을 더한 약 1.210 mm보다 작아 정지했다.
- **이번 6 trials에서는 겹치기 전에 멈췄지만, stage 0 추가가 다음 입력 적용 후의 무충돌을 보장하는 것은 아니다.** 첫 정지 바로 전 cycle은 세 seed 모두 solver status=0이었고 h0가 양수였으나, 입력 적용 후 현재 margin 조건을 위반했다. 고정 tangent와 비선형 입력 적용에 따른 차이는 그대로 남아 있다. 기록된 자세 사이의 연속시간 충돌은 검사하지 않았다.

### Animation

기존 `simulation_mpc.animate_world`를 그대로 사용했다(기본 렌더링 설정, 10 fps, frame_skip=5, HTML 변환 생략). 별도 고화질 렌더러는 추가하지 않았다. 각 영상은 해당 trial의 마지막 정지 또는 목표 도달 자세까지 포함한다.

| Width (m) | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 0.105 | [MP4](/home/taek111/projects/psdf_mpcc/animations/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/MPC_cc_sf_rectangle_maze_width105mm_seed0_stage0-19.mp4) | [MP4](/home/taek111/projects/psdf_mpcc/animations/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/MPC_cc_sf_rectangle_maze_width105mm_seed1_stage0-19.mp4) | [MP4](/home/taek111/projects/psdf_mpcc/animations/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/MPC_cc_sf_rectangle_maze_width105mm_seed2_stage0-19.mp4) |
| 0.110 | [MP4](/home/taek111/projects/psdf_mpcc/animations/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/MPC_cc_sf_rectangle_maze_width110mm_seed0_stage0-19.mp4) | [MP4](/home/taek111/projects/psdf_mpcc/animations/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/MPC_cc_sf_rectangle_maze_width110mm_seed1_stage0-19.mp4) | [MP4](/home/taek111/projects/psdf_mpcc/animations/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/MPC_cc_sf_rectangle_maze_width110mm_seed2_stage0-19.mp4) |

### 검증과 원본 자료

- 관련 단위 테스트 **10개 통과**. 소스 사본 불변성, 기존 시작 자세 일치, runtime·cycle·history의 safe-stop 수, zero input 및 정지 중 자세 유지, solver 시간 CSV와 성공 step 수를 대조했다. 전체 **1,064개 자세**의 clearance를 Shapely로 독립 확인했다. MP4 6개의 유효한 영상 stream·프레임 수·파일 크기도 확인했다.
- [폭별 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/benchmark_summary.csv), [Trial별 집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/benchmark_trials.csv), [정지 시점 데이터](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stage0_stop_onsets.csv), [생략 내역](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/sweep_progress.json), [검증](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/stage0_validation.json), [Animation 목록](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/animations.csv), [생성 solver의 stage 정책](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/solver_stage_policy.json).
- 관측용 로깅 함수의 인자 오류로 첫 사전 실행이 주행 전에 종료된 기록은 `setup_observer_error_excluded/`에 분리했다. 실제 적용 입력 0회였으며 위 6 trials와 통계에서 제외했다. 수정 후 새 폭별 출력 폴더에서 재실행했다.
- 재현용 [순차 실행](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/run_sweep.py)·[집계](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/analyze.py)·[stage-0 검증](/home/taek111/projects/psdf_mpcc/benchmark_runs/cc_sf_stage0_19_width_sweep_3trials_20261028_163315/audit_stage0.py) 스크립트를 보관했다. 재실행에는 새 결과 폴더를 사용한다.
