DCBF 단독 재실험 (2026-09-10)

결과: **목표 도착**. 시뮬레이션 37.6초, 376회 제어 입력 적용, A* 종점까지 0.009879 m.

`test_nmpc.run_single_test()`에서 실제 `control.dcbf_optimizer.NmpcDbcfOptimizer`(CasADi/IPOPT)를 실행했다. maze, differential_drive, rectangle 0.15×0.09 m, A*+LoS, 초기 위치 교란 없음, dt=0.1 s, 최대 60초. horizon=20, horizon_dcbf=10, gamma=0.8, margin_dist=0.0001 m, |v|≤0.6 m/s, |ω|≤1 rad/s. 다른 제어기나 진단 solver를 동시에 실행하지 않았다.

현재 작업 파일에 있는 수정된 비용 가중치를 사용했다. 제어기 파일은 이번 재실험에서 수정하지 않았다.

| 항목 | 이전 DCBF 실험 | 이번 단독 재실험 |
|---|---|---|
| Q 대각 성분 | [50, 50, 1] | [10, 10, 0] |
| R 대각 성분 | [5, 0.05] | [20, 0] |
| 주행 결과 | 정체 확인 후 36.9초에 SIGINT 진단 종료 | 목표 도착 |
| 종료 시 A* 종점까지 거리 | 1.398779 m | 0.009879 m |
| 최소 외곽 간격 | 0.168 mm | 0.246 mm |

- 최종 자세: [1.809596333, 0.097464924, -0.376736421].
- 기록된 solver 성공: 376 / 376; 상태 분포: {'None:Solve_Succeeded': 376}.
- 초기 자세 및 매 0.1초 실제 주행 상태에서 접촉/겹침 0회. 최소 간격 시각 33.8초. 샘플 사이의 연속 시간 충돌 보증은 아니다.
- 전체 이동 거리 2.828600 m, 마지막 5초간 변위 0.143657 m.
- 제어 호출 시간 중앙값 1332.1 ms, p95 1744.7 ms; IPOPT 구간 중앙값 757.7 ms. 전체 실제 경과 시간 511.2초. 제어 호출 중 100 ms 초과 376 / 376회.

도착 판정은 현재 코드의 A* 종점 (1.81875, 0.09375)까지 0.01 m 이내인지 사용한다. A*가 x/y만 제공하므로 최종 각도는 판정하지 않는다. 원래 환경 goal (1.8,0.09)까지 최종 거리는 12.158 mm이다.

원본 소스 커밋은 `65b8b6f37408873e2638b3e20ea01c5be5b46bbd`이며, 해당 커밋에 작업 중 수정된 DCBF 가중치가 적용된 상태다. 실행 소스 SHA-256: `b30d71fcdb3d704b54659747a92758581c822d0631c63a4d50192e0659d6a19b`. 실행 도중 소스 해시가 유지됐는지: True.

소스 스냅샷 `dcbf_optimizer_snapshot.py`, 커밋 대비 변경 `source_diff.patch`, 결과 `dcbf/result.json`, 파라미터·solver 상태 `dcbf/diagnostics.json`, 상태·입력·예측 궤적 `dcbf/diagnostics.npz`, 실행 로그 `dcbf/run.log`, 거리 분석 `metrics.json`에 보관했다. 제어기가 정상 종료한 경우 `dcbf/figures/`에 궤적과 속도·외곽 간격 그림을 저장한다.

재현: WSL 저장소 루트에서 `python3 debug_runs/dcbf_retest_20260910/run_experiment.py`. 실행 소스가 스냅샷과 같아야 한다. 기존 의존성이 설치된 Python 3.8.10을 사용하며, 실행기가 Python 3.10 argparse의 BooleanOptionalAction만 가져와 CLI 호환성을 보완한다. OMP/OpenBLAS/MKL 스레드는 각각 1이다.
