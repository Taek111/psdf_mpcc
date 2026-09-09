OBCA 공통 파라미터를 DCBF에 맞춘 실행 (2026-09-10)

결과: **목표 도착**. 시뮬레이션 39.6초, A* 종점까지 거리 9.431 mm, 완료된 solver 호출 396회 중 실패 0회.

수정 사항

`control/obca_optimizer.py`의 기본값을 현재 DCBF와 맞췄다. Q=diag(50,50,1)→diag(10,10,0), R=diag(5,0.05)→diag(20,0), Rold=diag(10,1)×0으로 표기를 맞췄다(Rold의 수치값은 변경 전후 모두 0). 입력 제한·예측 horizon·안전 마진 등 나머지 공통 파라미터는 이미 일치했다. 파라미터 객체에서 공통 13개 항목의 수치 일치를 검사해 `parameter_alignment.json`으로 저장했다.

OBCA의 장애물 제약 구간은 기존 horizon_dcbf=20을 유지했다. DCBF의 horizon_dcbf=10은 이번 공통 파라미터 동기화 대상에서 제외한 알고리즘 설정이다. 두 제어기의 rectangle 제약식은 현재 구현상 같은 DCBF 감쇠식을 사용한다. 이 작업에서는 제약식·초기화·솔버를 변경하지 않았다.

실행 조건

- `test_nmpc.build_parser()` / `run_single_test()`를 통해 `control.obca_optimizer.OBCAOptimizer`(CasADi/IPOPT) 실행.
- maze, differential_drive, rectangle 0.15×0.09 m, A*+LoS, 초기 위치 교란/위치 추정 오차 없음, dt=0.1 s, 최대 60초.
- horizon=20, Q=diag(10,10,0), R=diag(20,0), Rold=dR=0, terminal_weight=1, gamma=0.8, pomega=10, margin_dist=0.0001 m, |v|≤0.6 m/s, |ω|≤1 rad/s.
- 이번 OBCA와 직전 DCBF는 각각 단독 프로세스로 실행했다. 각 한 번의 nominal 실행이며 반복 실험 성공률은 아니다.

| 지표 | 직전 DCBF | 이번 OBCA |
|---|---:|---:|
| 결과 | 도착 | 목표 도착 |
| 기록된 시뮬레이션 시간 | 37.6 s | 39.6 s |
| A* 종점까지 거리 | 9.879 mm | 9.431 mm |
| 최소 외곽 간격 | 0.246 mm | 0.239 mm |
| 완료된 solver 실패 | 0 / 376 | 0 / 396 |
| 제어 호출 시간 중앙값 | 1.332 s | 2.903 s |

OBCA 상세 기록

- 최종 자세: [1.810011075, 0.097296035, -0.376787264].
- 실제 초기 자세 및 0.1초 간격 주행 상태에서 접촉/겹침 0회. 최소 외곽 간격 발생 시각 36.0초. 이는 샘플 사이 연속 시간 무충돌 보증은 아니다.
- 총 이동 거리 3.013959 m, 마지막 5초간 변위 0.141698 m.
- solver 상태 분포: {'None:Solve_Succeeded': 396}.
- 전체 제어 호출 중앙값 2903.5 ms, p95 4178.6 ms. IPOPT 구간 중앙값 1888.6 ms. 실제 실행 경과 시간 1168.2초. 100 ms 초과 제어 호출 396 / 396회.

현재 코드의 도착 판정은 A* 종점 (1.81875,0.09375)까지 거리≤0.01 m이다. A*가 x/y만 반환하므로 최종 yaw는 판정하지 않는다. 원래 환경 goal (1.8,0.09)까지 최종 거리는 12.388 mm다. 60초 제한에서 확인한 결과이며 기본 YAML의 20초 제한과 구분해야 한다.

재현과 증거

커밋 `65b8b6f37408873e2638b3e20ea01c5be5b46bbd` 위에 수정된 OBCA 기본값을 적용했다. 실행 파일 SHA-256: `736a8a0b875bf8dc2a55a7fff31f7fcc0262c25f556990743a7da7ca03251626`, 실행 중 소스 해시 유지: True.

`obca_optimizer_snapshot.py`, `dcbf_optimizer_snapshot.py`, `source_diff.patch`, `parameter_alignment.json`에 실행 소스와 공통 파라미터 검증을 보관했다. `obca/config.yaml`, `obca/result.json`, `obca/run.log`, `obca/diagnostics.json`, `obca/diagnostics.npz`, `metrics.json`, `obca/clearance.csv`에 조건·결과·궤적·입력·solver 상태가 있다. 정상 종료 그림은 `obca/figures/`에 저장된다.

재실행: WSL 저장소 루트에서 `python3 debug_runs/obca_dcbf_params_20260910/run_experiment.py`. 실행 소스가 스냅샷과 같아야 한다. 기존 의존성이 있는 Python 3.8.10을 사용하며 설치된 Python 3.10 argparse의 BooleanOptionalAction만 가져와 CLI 호환성을 보완한다. OMP/OpenBLAS/MKL 스레드는 각각 1이다. 원시 기록 분석과 보고서 생성은 이 폴더의 `analyze.py`, `write_report.py`로 실행한다.
