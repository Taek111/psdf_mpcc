"""Write a concise report from the completed DCBF-only rerun."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
metrics = json.loads((ROOT / "metrics.json").read_text())["dcbf"]
result = json.loads((ROOT / "dcbf/result.json").read_text())
diag = json.loads((ROOT / "dcbf/diagnostics.json").read_text())
data = np.load(ROOT / "dcbf/diagnostics.npz")
old = json.loads((ROOT.parent / "maze_dd_rectangle_20260910/metrics.json").read_text())["dcbf"]

status_text = {"success": "목표 도착", "timeout": "60초 시간 초과", "exception": "실행 예외"}.get(result["status"], result["status"])
text = f"""DCBF 단독 재실험 (2026-09-10)

결과: **{status_text}**. 시뮬레이션 {metrics['final_time']:.1f}초, {metrics['steps']}회 제어 입력 적용, A* 종점까지 {metrics['distance_to_planner_goal']:.6f} m.

`test_nmpc.run_single_test()`에서 실제 `control.dcbf_optimizer.NmpcDbcfOptimizer`(CasADi/IPOPT)를 실행했다. maze, differential_drive, rectangle 0.15×0.09 m, A*+LoS, 초기 위치 교란 없음, dt=0.1 s, 최대 60초. horizon=20, horizon_dcbf=10, gamma=0.8, margin_dist=0.0001 m, |v|≤0.6 m/s, |ω|≤1 rad/s. 다른 제어기나 진단 solver를 동시에 실행하지 않았다.

현재 작업 파일에 있는 수정된 비용 가중치를 사용했다. 제어기 파일은 이번 재실험에서 수정하지 않았다.

| 항목 | 이전 DCBF 실험 | 이번 단독 재실험 |
|---|---|---|
| Q 대각 성분 | [50, 50, 1] | [10, 10, 0] |
| R 대각 성분 | [5, 0.05] | [20, 0] |
| 주행 결과 | 정체 확인 후 36.9초에 SIGINT 진단 종료 | {status_text} |
| 종료 시 A* 종점까지 거리 | {old['distance_to_planner_goal']:.6f} m | {metrics['distance_to_planner_goal']:.6f} m |
| 최소 외곽 간격 | {old['min_clearance'] * 1000:.3f} mm | {metrics['min_clearance'] * 1000:.3f} mm |

- 최종 자세: {np.asarray(metrics['final_pose']).round(9).tolist()}.
- 기록된 solver 성공: {metrics['steps'] - metrics['solver_failure_count']} / {metrics['steps']}; 상태 분포: {metrics['solver_status_counts']}.
- 초기 자세 및 매 0.1초 실제 주행 상태에서 접촉/겹침 {metrics['contact_or_overlap_samples']}회. 최소 간격 시각 {metrics['min_clearance_time']:.1f}초. 샘플 사이의 연속 시간 충돌 보증은 아니다.
- 전체 이동 거리 {metrics['path_length']:.6f} m, 마지막 5초간 변위 {metrics['last_5s_displacement']:.6f} m.
- 제어 호출 시간 중앙값 {metrics['controller_times']['median'] * 1000:.1f} ms, p95 {metrics['controller_times']['p95'] * 1000:.1f} ms; IPOPT 구간 중앙값 {metrics['solver_times']['median'] * 1000:.1f} ms. 전체 실제 경과 시간 {metrics['wall_seconds']:.1f}초. 제어 호출 중 100 ms 초과 {metrics['controller_times']['over_100ms']} / {metrics['controller_times']['count']}회.

도착 판정은 현재 코드의 A* 종점 (1.81875, 0.09375)까지 0.01 m 이내인지 사용한다. A*가 x/y만 제공하므로 최종 각도는 판정하지 않는다. 원래 환경 goal (1.8,0.09)까지 최종 거리는 {metrics['distance_to_requested_goal'] * 1000:.3f} mm이다.

원본 소스 커밋은 `{result['source_commit']}`이며, 해당 커밋에 작업 중 수정된 DCBF 가중치가 적용된 상태다. 실행 소스 SHA-256: `{result['optimizer_source_sha256']}`. 실행 도중 소스 해시가 유지됐는지: {result['optimizer_source_unchanged_during_run']}.

소스 스냅샷 `dcbf_optimizer_snapshot.py`, 커밋 대비 변경 `source_diff.patch`, 결과 `dcbf/result.json`, 파라미터·solver 상태 `dcbf/diagnostics.json`, 상태·입력·예측 궤적 `dcbf/diagnostics.npz`, 실행 로그 `dcbf/run.log`, 거리 분석 `metrics.json`에 보관했다. 제어기가 정상 종료한 경우 `dcbf/figures/`에 궤적과 속도·외곽 간격 그림을 저장한다.

재현: WSL 저장소 루트에서 `python3 debug_runs/dcbf_retest_20260910/run_experiment.py`. 실행 소스가 스냅샷과 같아야 한다. 기존 의존성이 설치된 Python 3.8.10을 사용하며, 실행기가 Python 3.10 argparse의 BooleanOptionalAction만 가져와 CLI 호환성을 보완한다. OMP/OpenBLAS/MKL 스레드는 각각 1이다.
"""
(ROOT / "report.md").write_text(text, encoding="utf-8")
print(text.splitlines()[2])
