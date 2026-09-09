import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT))
from sim.simulation_mpc import simulation_mpc
from models.dd import DifferentialDriveMultipleGeometry,DifferentialDriveRectangleGeometry
m=json.loads((ROOT/'metrics.json').read_text())['dcbf']
diag=json.loads((ROOT/'dcbf/diagnostics.json').read_text())
result=json.loads((ROOT/'dcbf/result.json').read_text())
assert diag['optimizer_module']=='control.dcbf_optimizer'
assert diag['optimizer_class']=='NmpcDbcfOptimizer'
data=np.load(ROOT/'dcbf/diagnostics.npz')
render=simulation_mpc(); render.output_root_dir=str(ROOT/'dcbf')
start,goal,grid,obstacles=render.create_env('maze')
geometry=DifferentialDriveMultipleGeometry(); geometry.add_geometry(DifferentialDriveRectangleGeometry(.15,.09,0))
robot=SimpleNamespace(_system=SimpleNamespace(_geometry=geometry),_system_logger=SimpleNamespace(_xs=data['states']),
                     _global_planner_logger=SimpleNamespace(_paths=[np.array(diag['global_path'])]),
                     _local_planner_logger=SimpleNamespace(_trajs=data['references']),
                     _controller_logger=SimpleNamespace(_xtrajs=data['predicted_states']))
simulation=SimpleNamespace(_robot=robot,_obstacles=obstacles)
render.plot_world(simulation,[len(data['states'])-1],figure_name='dcbf_stopped_prediction',local_traj_indexes=[len(data['states'])-1],maze_type='maze')

probe=json.loads((ROOT/'stopped_state_probe_exact.json').read_text())
base=next(s for s in probe['solves'] if s['name']=='default')
full=next(s for s in probe['solves'] if s['name']=='horizon11_cbf11')
lines=[
'DCBF CasADi/IPOPT 연결 변경 및 maze 재실험 (2026-09-09)',
'',
'`sim/simulation_mpc.py:1874`에서 dcbf와 dcbf_casadi가 모두 control/dcbf_optimizer.py의 NmpcDbcfOptimizer 및 NmpcDcbfOptimizerParam을 사용하도록 변경했다. config/config.yaml의 옵션 설명도 수정했다. 실험 종료 시 실제 객체의 모듈과 클래스가 각각 control.dcbf_optimizer, NmpcDbcfOptimizer임을 확인했다.',
'',
'조건: maze, differential_drive, rectangle 0.15 × 0.09 m, A* + Line of Sight, 시작점 교란/위치 추정 오차 없음, dt=0.1 s, 최대 시뮬레이션 시간 60 s. 시작 자세는 (0.075,0.795,-π/2). 요청한 dcbf_optimizer.py의 기본값 horizon=11, horizon_dcbf=6, gamma=0.8, margin_dist=0, terminal_weight=10을 유지했다. 제어기 내부 파라미터나 계산식은 수정하지 않았다.',
'',
'| 항목 | 결과 |',
'|---|---|',
f"| 상태 | {result['status']} / {result['failure_reason']} |",
f"| 시뮬레이션 시간 / 적용 입력 수 | {m['final_time']:.1f} s / {m['steps']}회 |",
f"| 최종 자세 | ({m['final_pose'][0]:.6f}, {m['final_pose'][1]:.6f}, {m['final_pose'][2]:.6f}) |",
f"| A* 종점까지 남은 거리 | {m['distance_to_planner_goal']:.6f} m |",
f"| 실제 궤적 최소 외곽 간격 | {m['min_clearance']*1000:.6f} mm |",
f"| 기록된 상태의 접촉/겹침 | {m['contact_or_overlap_samples']}회 |",
f"| 솔버 실패 | {m['solver_failure_count']}회 |",
f"| 마지막 5초 이동량 | {m['last_5s_displacement']:.3e} m |",
f"| 전체 제어 호출 중앙값 / P95 | {m['controller_times']['median']*1000:.1f} / {m['controller_times']['p95']*1000:.1f} ms |",
f"| 최적화 구간 중앙값 / P95 | {m['solver_times']['median']*1000:.1f} / {m['solver_times']['p95']*1000:.1f} ms |",
f"| 실제 실행 시간 | {m['wall_seconds']:.1f} s |",
'',
'이번에는 60초까지 중단 없이 실행한 결과다. 기존 SQP 실행처럼 첫 계산에서 막히는 문제는 없었지만, 장애물 앞에서 입력이 거의 0으로 수렴해 도착에 실패했다. 솔버의 Solve_Succeeded는 각 최적화 문제의 종료 상태이며 주행 성공과 다르다.',
'',
'실패 원인 분석:',
'',
'1. 예측은 11단계인데 충돌 회피 제약은 6단계까지만 적용된다(control/dcbf_optimizer.py:11–12,146–168). 종료 직전의 실제 예측과 정확한 최종 상태 재실행을 확인하면 7~11단계의 로봇 외곽이 장애물과 겹친다. 실제 로봇의 충돌과 예측 궤적의 충돌을 구분해야 한다. 현재 구성에서는 이 후반 겹침이 제약 위반으로 취급되지 않는다.',
f"2. 기본 설정의 최종 상태 재실행은 첫 입력 v={base['first_control'][0]:.3e}, omega={base['first_control'][1]:.3e}로 사실상 정지한다. 다음 단계에는 후진·회전을, 안전 제약이 끝난 뒤에는 장애물 내부를 통과하는 전진을 계획한다. 실제로는 첫 입력만 적용하고 다시 풀기 때문에 동일한 상태와 계획을 반복한다. 서로 다른 후진·회전 초기 추정도 같은 정지 해로 수렴했으므로 초기화만의 문제로 설명하기 어렵다.",
f"3. 원인을 분리하기 위해 단일 최적화 진단에서 horizon=11은 유지하고 horizon_dcbf만 11로 늘렸다. 첫 입력은 v={full['first_control'][0]:.6f} m/s, omega={full['first_control'][1]:.6f} rad/s로 바뀌었고, 예측 궤적의 접촉/겹침 단계는 {full['zero_clearance_stages']}였다. 최대 제약 위반은 {full['max_constraint_violation']:.2e}였으며 솔버는 {full['status']}를 반환했다. 이 설정으로 전체 maze 완주를 확인한 것은 아니다. 서로 제약이 다른 문제의 비용을 우열 비교에 사용하지 않았다.",
'4. 참조 길이도 맞지 않는다. 로컬 참조 생성기는 20개 참조를 만들지만(planning/trajectory_generator/constant_speed_generator.py:14), 최적화는 11단계이고 terminal cost는 항상 reference_trajectory[-1]을 사용한다(control/dcbf_optimizer.py:78). 참조를 11개로 맞추는 진단에서는 첫 입력이 바뀌었으나 7~11단계의 장애물 겹침은 남았다. 안전 제약 범위와 참조 길이를 함께 정렬할 필요가 있다.',
'',
'진단에서는 최종 상태, 이전 입력, 참조 궤적을 저장 데이터에서 읽었다. 실행 중 반올림 로그로 수행한 예비 검사는 *_approx.json에, 종료 후 정확한 값으로 수행한 재검증은 stopped_state_probe_exact.json에 분리했다. 진단의 파라미터 변경은 실험용 스크립트 안에서만 적용했으며 저장소 기본값은 바꾸지 않았다.',
'',
'측정 범위:',
'',
'- 도착 판정은 현재 코드의 A* 마지막 좌표 (1.81875,0.09375) 기준이다. 원래 환경 목표 (1.8,0.09,0)와 다르며 A* 주행에서는 목표 각도 조건이 적용되지 않는다.',
'- 실제 충돌 검사는 초기 자세와 0.1초 간격 저장 상태에서 rectangle 외곽 대 장애물 및 맵 경계를 계산했다. 연속 시간 전체의 안전을 증명하는 검사는 아니다.',
'- 이전 실험과 동일한 WSL2/Python 3.8/스레드 제한 환경이며, Python 3.8의 BooleanOptionalAction 호환성은 별도 실행기에서 보완했다. 제어 시간은 setup 및 solve를 포함하고, 최적화 시간은 solve_nlp 내부 측정이다. 짧은 진단 실행이 기본 실행 일부와 겹쳤으므로 엄밀한 성능 벤치마크로 해석하지 않는다.',
'',
'변경 검증: 실제 600회 제어 루프 실행, 생성 객체의 모듈/클래스 확인, YAML 로딩 및 git diff --check. 변경된 저장소 파일은 sim/simulation_mpc.py와 config/config.yaml이다. source_changes.patch에 변경 내용을 보관했다.',
'',
'재현 파일: run_experiment.py, dcbf/config.yaml, dcbf/run.log, dcbf/result.json, dcbf/diagnostics.npz, dcbf/diagnostics.json, metrics.json, probe_dcbf.py, stopped_state_probe_exact.json.',
'',
'[실제 주행 궤적](dcbf/figures/trajectory/mpc_dcbf_rectangle_maze_20260909.png) · [정체 위치와 예측 궤적](dcbf/figures/trajectory/dcbf_stopped_prediction.png)',
]
(ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(ROOT/'report.md')
