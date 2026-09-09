import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
metrics = json.loads((ROOT/'metrics.json').read_text())
commit = json.loads((ROOT/'psdf/result.json').read_text())['source_commit']
ps = metrics['psdf']
lines = [
    'maze / differential drive / rectangle 실행 결과 (2026-09-09)',
    '',
    f'소스 커밋: `{commit}`. `test_nmpc.py`의 `build_parser()`와 `run_single_test()`를 사용했다. 원본 제어기 및 시뮬레이션 소스는 변경하지 않았다.',
    '',
    '동일 조건: maze, differential_drive, rectangle 0.15 × 0.09 m, A* + Line of Sight, horizon 20, dt=0.1 s, 최대 시뮬레이션 시간 60 s. 시작 자세는 (0.075, 0.795, -π/2), 시작점 교란과 위치 추정 오차는 사용하지 않았다. 각 방법의 기존 비용 가중치·입력 제한·안전 마진은 유지했다. 한 번씩 실행한 결과이며 성공률 통계는 아니다.',
    '',
    '| 방법 | 결과 | 주행 시간 | A* 종점까지 거리 | 외곽 최소 간격 | 제어 계산 중앙값 |',
    '|---|---|---:|---:|---:|---:|',
]
for method in ('psdf','dcbf','obca'):
    if method in metrics:
        m=metrics[method]
        lines.append(f"| {method.upper()} | {('도착' if method=='psdf' else '정체 확인 후 수동 중단')} | {m['final_time']:.1f} s | {m['distance_to_planner_goal']*1000:.3f} mm | {m['min_clearance']*1000:.3f} mm | {m['controller_times']['median']*1000:.1f} ms |")
    elif method=='dcbf':
        lines.append('| DCBF | 첫 제어 계산 중 중단 | 0 s, 입력 적용 0회 | 해당 없음 | 주행 평가 불가 | 첫 계산 미완료 |')
    else:
        lines.append('| OBCA | 실행 중 | — | — | — | — |')
lines += [
    '',
    f"PSDF는 {ps['steps']}회 모두 solver status 0이었으며 도착했다. 기록된 상태에서 외곽과 장애물·맵 경계 사이 접촉/겹침은 {ps['contact_or_overlap_samples']}회였다. 최소 간격은 t={ps['min_clearance_time']:.1f} s에서 {ps['min_clearance']*1000:.3f} mm였다. 최적화 구간 중앙값은 {ps['solver_times']['median']*1000:.1f} ms, 전처리를 포함한 제어 호출 중앙값은 {ps['controller_times']['median']*1000:.1f} ms이다. 제어 호출 {ps['controller_times']['over_100ms']}/{ps['steps']}회가 100 ms를 넘었다. 이 환경에서는 실시간 10 Hz 요건을 충족하지 못했다.",
    '',
    'DCBF는 실행 시작 약 314.73초 뒤에도 첫 solver.solve()가 반환되지 않아 SIGINT를 보냈고, 15초 안에 처리되지 않아 SIGTERM으로 종료했다. 컴파일은 완료된 상태였고 실제 제어 입력은 한 번도 적용되지 않았다. 이는 시뮬레이션 60초 timeout 또는 솔버 infeasible 반환이 아니다. 주행 결과는 미완료이며, 아래의 설정 오류를 확인해 잘못 구성된 계산을 중단한 것이다.',
    '',
    'DCBF 원인 분석:',
    '',
    '1. 장애물 제약이 생성된 솔버에 포함되지 않는다. `control/dcbf_optimizer_sqp.py:474`에서 `create_solver()`를 호출한 뒤 475행에서 제약을 추가한다. 또한 425–427행은 `constraints.expr_h`, `lg`, `ug`에 저장한다. 설치된 acados는 `model.con_h_expr`와 `constraints.lh/uh`를 사용한다. 실제 생성된 JSON의 `nh=nh_0=nh_e=ng=0`을 확인했다. 별도 setup 검사에서는 사용되지 않는 식만 4,199행 존재했다.',
    '2. 목표 경로가 전달되지 않는다. `setup()`은 전달받은 `reference_trajectory`를 멤버에 저장하지 않는다. `self.reference_trajectory`가 계속 None이므로 502–505행에서 모든 참조 파라미터를 0으로 채운다. 최초 제공 참조는 [0.09375, 0.745, -1.5707963]이지만 실제 설정값은 [0, 0, 0, 0, 0]으로 재현됐다.',
    '3. 대수변수 정의에 모순이 있다. 148–149행의 잔차 `[xdot-f, z]=0`은 모든 듀얼 변수와 완화 변수를 z=0으로 강제한다. 의도한 충돌 회피식을 올바르게 연결해도 양수 마진 0.001 m와 양립하지 않는다. z=0으로 식을 직접 평가하면 323개 거리 제약의 잔차가 -0.001이다. 제약 등록 순서만 바꾸는 수정으로는 충분하지 않다.',
    '4. 변수 차원과 생성 비용이 과다하다. 179–190행이 이미 단계별로 반복되는 모델의 대수변수에 horizon을 다시 곱한다. maze의 17개 장애물 × (4+4+1) × horizon 20으로 매 단계 nz=3,060이다. 첫 계산 지연의 구조적 원인으로 판단된다. 또한 controller.py:69가 매 제어 호출마다 setup()을 부르고 DCBF setup()은 매번 솔버를 새로 생성하므로, 첫 계산 이후에도 생성 비용이 반복되는 구조다.',
    '',
    'DCBF 수정 우선순위는 참조 저장 → 단계별 변수 정의와 대수 방정식 재설계 → 올바른 필드로 제약을 등록한 후 솔버 생성 → 생성한 솔버 재사용이다. 이번 작업에서는 제어기를 수정하거나 수정판의 성공을 주장하지 않았다.',
    '',
    '판정과 비교 시 유의할 점:',
    '',
    '- 이 코드의 A* 도착 판정은 환경의 원래 goal pose가 아니라 A* 마지막 좌표까지 0.01 m 이내인지 검사한다. 원래 목표는 (1.8, 0.09, 0), A* 종점은 (1.81875, 0.09375)이다. A*가 x/y만 반환하므로 최종 각도 허용 오차는 적용되지 않는다. PSDF 최종점에서 원래 목표까지 거리는 15.817 mm, 각도 오차는 0.015552 rad이다. 따라서 여기서 success는 현재 코드의 도착 판정 기준을 뜻한다.',
    '- 외곽 간격은 저장된 0.1초 간격 상태와 초기 자세를 대상으로 다각형 거리와 맵 경계를 검사했다. 샘플 사이 연속 시간의 충돌을 증명한 것은 아니다.',
    '- PSDF는 |v|≤0.6 m/s, |ω|≤1 rad/s, 기본 안전 오프셋 0.000537 m이다. OBCA는 |v|≤0.5 m/s, |ω|≤0.5 rad/s, margin_dist=0.01 m이다. 비용 가중치도 달라 주행 시간 차이를 알고리즘만의 차이로 해석할 수 없다.',
    '- 시간 값은 i5-10600 / WSL2 / OMP·OpenBLAS·MKL 스레드 각각 1의 이 실행에서 측정됐다. 제어 계산 시간에는 setup과 solve가 포함되며, 각 optimizer의 solver_time에는 서로 다른 작업 구간이 포함될 수 있다. 첫 호출은 초기화/생성 비용도 포함한다.',
    '',
    '실행 환경 문제와 재현:',
    '',
    '`python3 test_nmpc.py --help`부터 Python 3.8.10의 argparse.BooleanOptionalAction 부재로 실패했다(test_nmpc.py:423). 기존 CasADi/Torch/acados 의존성이 Python 3.8에 설치되어 있어, 실험용 실행기에서 설치된 Python 3.9 표준 라이브러리의 해당 Action만 연결했다. 이 처리는 명령줄 옵션 호환성만 보완한다. ACADOS_SOURCE_DIR와 공유 라이브러리 경로를 지정했다.',
    '',
    '재현 실행기는 이 폴더의 run_experiment.py이며 각 방법의 config.yaml, result.json, run.log와 함께 보관했다. 실행은 저장소의 test_nmpc.run_single_test를 호출하며 제어기 계산을 대체하지 않는다. inspect_dcbf.py는 별도의 설정 검사이고 수치 솔버를 recorder로 바꿔 파라미터 전달과 식을 확인한다. 이 검사에서 출력되는 success/time은 실제 최적화 결과가 아니다.',
    '',
    '원시 결과: metrics.json, environment.json, psdf/{result.json,diagnostics.json,diagnostics.npz,data/,figures/}, dcbf/{run.log,result.json,stop_record.json,generated_ocp_dimensions.json}, dcbf_setup_inspection.json. OBCA 결과도 같은 구조로 기록한다.',
]
if 'obca' in metrics:
    ob=metrics['obca']
    probe=json.loads((ROOT/'obca_stopped_state_probe.json').read_text())
    default, alternative=probe['solves']
    index=next(i for i,line in enumerate(lines) if line.startswith('DCBF는 실행 시작'))
    lines[index:index]=[
        f"OBCA는 시뮬레이션 35.7초에 정체를 확인하고 SIGINT로 중단했다. 실제 실행 시간은 {ob['wall_seconds']:.1f}초(약 {ob['wall_seconds']/60:.1f}분)였다. 최종 자세는 (1.860060, 0.394951, -1.870796), A* 종점까지 {ob['distance_to_planner_goal']:.6f} m였다. 마지막 5초 동안 이동량은 {ob['last_5s_displacement']:.3e} m로 사실상 정지했다. 이는 60초 timeout이나 IPOPT infeasible 종료가 아니며 원시 상태는 interrupted이다.",
        '',
        f"OBCA의 357회 완료된 최적화는 모두 Solve_Succeeded였고 기록된 상태의 접촉/겹침은 0회였다. 외곽 최소 간격은 {ob['min_clearance']*1000:.3f} mm이다. 최적화 구간 중앙값은 {ob['solver_times']['median']:.3f}초, 전체 제어 호출 중앙값은 {ob['controller_times']['median']:.3f}초였으며 완료된 357회 모두 100 ms를 넘었다. 중단된 마지막 호출 1회는 통계에서 제외했다. YAML 기본 20초 시점에는 A* 종점까지 {ob['distance_to_planner_goal_at_20s']:.6f} m 남아 있었다.",
        '',
        'OBCA 원인 분석:',
        '',
        '1. 마지막 위치는 장애물 index 12(0부터 시작), x∈[1.8,1.95], y∈[0.15,0.30] 바로 위이다. 약 10 mm 안전 마진에 걸린 상태에서 기본 초기화가 정지 해를 반복해서 찾는다. control/obca_optimizer.py:196–201은 nominal_safe_controller가 반환한 한 정지 자세/입력을 모든 단계의 초기값에 반복한다. setup()은 매번 Opti를 새로 만들며 이전 최적 궤적을 이동해 재사용하는 처리가 없다.',
        f"2. 반올림 로그를 이용한 예비 검사 후, diagnostics.npz/diagnostics.json의 정확한 마지막 상태와 참조로 재검증했다. 기본 초기화는 비용 {default['cost']:.6f}, 첫 입력 ({default['first_control'][0]:.3e}, {default['first_control'][1]:.3e})의 정지 해를 다시 반환했다. 같은 제약과 목적함수에 후진·회전 초기 추정만 주면 비용 {alternative['cost']:.6f}로 {100*(1-alternative['cost']/default['cost']):.2f}% 낮은 해를 찾았고 첫 입력은 v={alternative['first_control'][0]:.6f} m/s, ω={alternative['first_control'][1]:.6f} rad/s였다.",
        f"3. 두 해 모두 Solve_Succeeded였고 제약 최대 위반량은 각각 {default['max_constraint_violation']:.2e}, {alternative['max_constraint_violation']:.2e}였다. 대안의 예측 궤적 최소 외곽 간격도 {alternative['min_predicted_clearance']*1000:.6f} mm였다. 따라서 이 지점에서 움직이는 해 자체가 없는 것이 아니라, 초기화에 따라 더 나쁜 정지 해에 수렴하는 현상으로 해석할 수 있다. 대안 초기화로 전체 미로를 완주했는지는 시험하지 않았다.",
        '',
        'OBCA 개선 방향은 이전 해를 시간 이동한 초기값 재사용, 정체 시 후진·회전 등의 여러 초기 추정 비교, 성공 상태 외에 이동량/목표 진행량 검사 추가다. 해당 개선을 적용한 제어기는 이번 실행에 포함하지 않았다. probe_obca.py와 obca_stopped_state_probe.json에 동일한 단일 최적화 문제의 재현 결과를 저장했다.',
        '',
    ]
    lines += ['', '궤적 그림: [PSDF](psdf/figures/trajectory/mpc_psdf_rectangle_maze_20260909.png), [OBCA 정체 위치](obca/figures/trajectory/mpc_obca_rectangle_maze_20260909_interrupted.png).']
(ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(ROOT/'report.md')

