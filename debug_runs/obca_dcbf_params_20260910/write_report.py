"""Report the OBCA run with common parameters aligned to current DCBF."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
m = json.loads((ROOT / "metrics.json").read_text())["obca"]
r = json.loads((ROOT / "obca/result.json").read_text())
diag = json.loads((ROOT / "obca/diagnostics.json").read_text())
dcbf = json.loads((ROOT.parent / "dcbf_retest_20260910/metrics.json").read_text())["dcbf"]
alignment = json.loads((ROOT / "parameter_alignment.json").read_text())
dcbf_diag = json.loads((ROOT.parent / "dcbf_retest_20260910/dcbf/diagnostics.json").read_text())
runtime_alignment = {
    key: bool(np.array_equal(diag["parameters"][key], dcbf_diag["parameters"][key]))
    for key in alignment["shared_parameters_equal"]
}
assert all(runtime_alignment.values()), runtime_alignment
(ROOT / "runtime_parameter_alignment.json").write_text(json.dumps(runtime_alignment, indent=2))
assert diag["optimizer_module"] == "control.obca_optimizer"
assert all(alignment["shared_parameters_equal"].values())
assert np.array_equal(diag["parameters"]["mat_Q"], np.diag([10., 10., 0.]))
assert np.array_equal(diag["parameters"]["mat_R"], np.diag([20., 0.]))
assert diag["parameters"]["horizon_dcbf"] == 20
status = {"success": "목표 도착", "timeout": "60초 시간 초과", "exception": "실행 예외"}.get(r["status"], r["status"])
report = f"""OBCA 공통 파라미터를 DCBF에 맞춘 실행 (2026-09-10)

결과: **{status}**. 시뮬레이션 {m['final_time']:.1f}초, A* 종점까지 거리 {m['distance_to_planner_goal'] * 1000:.3f} mm, 완료된 solver 호출 {m['steps']}회 중 실패 {m['solver_failure_count']}회.

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
| 결과 | 도착 | {status} |
| 기록된 시뮬레이션 시간 | {dcbf['final_time']:.1f} s | {m['final_time']:.1f} s |
| A* 종점까지 거리 | {dcbf['distance_to_planner_goal'] * 1000:.3f} mm | {m['distance_to_planner_goal'] * 1000:.3f} mm |
| 최소 외곽 간격 | {dcbf['min_clearance'] * 1000:.3f} mm | {m['min_clearance'] * 1000:.3f} mm |
| 완료된 solver 실패 | {dcbf['solver_failure_count']} / {dcbf['steps']} | {m['solver_failure_count']} / {m['steps']} |
| 제어 호출 시간 중앙값 | {dcbf['controller_times']['median']:.3f} s | {m['controller_times']['median']:.3f} s |

OBCA 상세 기록

- 최종 자세: {np.asarray(m['final_pose']).round(9).tolist()}.
- 실제 초기 자세 및 0.1초 간격 주행 상태에서 접촉/겹침 {m['contact_or_overlap_samples']}회. 최소 외곽 간격 발생 시각 {m['min_clearance_time']:.1f}초. 이는 샘플 사이 연속 시간 무충돌 보증은 아니다.
- 총 이동 거리 {m['path_length']:.6f} m, 마지막 5초간 변위 {m['last_5s_displacement']:.6f} m.
- solver 상태 분포: {m['solver_status_counts']}.
- 전체 제어 호출 중앙값 {m['controller_times']['median'] * 1000:.1f} ms, p95 {m['controller_times']['p95'] * 1000:.1f} ms. IPOPT 구간 중앙값 {m['solver_times']['median'] * 1000:.1f} ms. 실제 실행 경과 시간 {m['wall_seconds']:.1f}초. 100 ms 초과 제어 호출 {m['controller_times']['over_100ms']} / {m['controller_times']['count']}회.

현재 코드의 도착 판정은 A* 종점 (1.81875,0.09375)까지 거리≤0.01 m이다. A*가 x/y만 반환하므로 최종 yaw는 판정하지 않는다. 원래 환경 goal (1.8,0.09)까지 최종 거리는 {m['distance_to_requested_goal'] * 1000:.3f} mm다. 60초 제한에서 확인한 결과이며 기본 YAML의 20초 제한과 구분해야 한다.

재현과 증거

커밋 `{r['source_commit']}` 위에 수정된 OBCA 기본값을 적용했다. 실행 파일 SHA-256: `{r['optimizer_source_sha256']}`, 실행 중 소스 해시 유지: {r['optimizer_source_unchanged_during_run']}.

`obca_optimizer_snapshot.py`, `dcbf_optimizer_snapshot.py`, `source_diff.patch`, `parameter_alignment.json`에 실행 소스와 공통 파라미터 검증을 보관했다. `obca/config.yaml`, `obca/result.json`, `obca/run.log`, `obca/diagnostics.json`, `obca/diagnostics.npz`, `metrics.json`, `obca/clearance.csv`에 조건·결과·궤적·입력·solver 상태가 있다. 정상 종료 그림은 `obca/figures/`에 저장된다.

재실행: WSL 저장소 루트에서 `python3 debug_runs/obca_dcbf_params_20260910/run_experiment.py`. 실행 소스가 스냅샷과 같아야 한다. 기존 의존성이 있는 Python 3.8.10을 사용하며 설치된 Python 3.10 argparse의 BooleanOptionalAction만 가져와 CLI 호환성을 보완한다. OMP/OpenBLAS/MKL 스레드는 각각 1이다. 원시 기록 분석과 보고서 생성은 이 폴더의 `analyze.py`, `write_report.py`로 실행한다.
"""
(ROOT / "report.md").write_text(report, encoding="utf-8")
print(report.splitlines()[2])
