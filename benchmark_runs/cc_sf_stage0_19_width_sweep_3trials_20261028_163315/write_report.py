"""Append this experiment without rewriting other benchmark sections."""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def link(label, name):
    return f'[{label}]({ROOT / name})'


summaries = {float(r['width_m']): r for r in rows(ROOT / 'benchmark_summary.csv')}
stops = rows(ROOT / 'stage0_stop_onsets.csv')
trials = rows(ROOT / 'benchmark_trials.csv')
animations = rows(ROOT / 'animations.csv')
validation = json.loads((ROOT / 'stage0_validation.json').read_text())
clearance_validation = json.loads((ROOT / 'validation.json').read_text())
shapely_checks = sum(r['independent_shapely_clearance_samples_checked']
                     for r in clearance_validation['checks'])
assert len(trials) == len(animations) == 6
assert sum(int(r['safe_stop_steps']) for r in trials) == 60
assert sum(r['status'] == 'success' for r in trials) == 3
assert validation['first_overlap_count'] == 0
heading = '## CC-SF 안전 제약 stage 0–19 재실험 (2026-10-28)'
lines = [
    heading, '',
    '**0.105 m는 3/3 목표 도달·safe stop 0회, 0.110 m는 0/3 목표 도달·각 seed에서 safe stop 20 step이었다.** 기존에 허용한 전원 실패 시 생략 규칙에 따라 **0.115 / 0.120 / 0.125 / 0.130 m는 실행하지 않았다.** 총 6 trials를 완료했으며, 기록된 모든 자세에서 장애물 접촉·겹침 및 지도 경계 침범은 없었다. 실행한 6개 trial의 MP4도 저장했다.', '',
    '### 변경과 실험 조건', '',
    '- **N=20, dt=0.1 s, tf=2 s를 유지**했다. 상태는 x0–x20, 입력은 u0–u19이며, hard SF 안전 제약 20개의 적용 위치만 기존 stage 1–20에서 **stage 0–19**로 옮겼다. 현재 상태 x0는 고정하고 terminal state x20의 안전 제약은 제외했다. Horizon을 19개 제어 구간으로 줄인 실험이 아니다.',
    '- 주·복구 solver의 실제 생성 JSON 12개 모두 **nh_0=1, nh=1, nh_e=0, slack=0**을 확인했다. Stage 0은 별도의 사전 검사로 처리하지 않고 OCP hard row에 포함했다.',
    '- 이전 width sweep의 고정 소스에서 `control/cc_sf_optimizer.py`의 제약 stage 부분만 변경했다. 폭 이외의 설정은 동일하다: corridor 0.150 m, robot 길이 0.150 m, DD·A*, seed 0–2, 초기 clearance ≥1 mm, localization error 비활성화, d_min=1 mm, epsilon=0.20, robust horizon=10, 최대 60 s. 실행한 각 폭·seed의 시작 자세는 이전 실험과 정확히 같다.',
    '- Clearance는 초기 및 모든 적분 후 자세를 포함한 polygon 거리와 지도 경계 margin의 최솟값이다. 실패 trial 및 정지 중 반복 자세도 포함해 pooled mean/p05/min을 계산했다. Safe-stop 횟수는 zero input을 적용한 **control step 수**이며, 연속 20 step=2 s=정지 구간 1개이다.', '',
    '### 폭별 결과', '',
    '| Width (m) | 목표 도달 | Mean clearance (mm) | P05 (mm) | Min (mm) | Safe-stop step: seed 0 / 1 / 2 | 겹침 표본 |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
for width in [.105, .110, .115, .120, .125, .130]:
    if width not in summaries:
        lines.append(f'| {width:.3f} | 생략 | — | — | — | — | — |')
    else:
        r = summaries[width]
        lines.append(f"| {width:.3f} | {r['successes']}/3 | {float(r['clearance_mean_mm']):.3f} | "
                     f"{float(r['clearance_p05_mm']):.3f} | {float(r['clearance_min_mm']):.3f} | "
                     f"{r['safe_stop_steps_seed012']} | 0 |")
lines += [
    '',
    '0.105 m의 seed 0 / 1 / 2 목표 도달 시간은 **29.1 / 29.2 / 28.9 s**였다. 기존 stage 1–20 실험의 0.110 m는 목표 도달 2/3, safe stop 20/0/0 step이었지만, 이번 stage 0–19 설정에서는 세 seed 모두 더 일찍 정지했다. 두 설정은 terminal row 유무도 다르며, 실패로 주행 길이가 달라지므로 pooled 평균 clearance만으로 안전성을 비교하지 않는다.', '',
    '### 0.110 m의 정지 원인', '',
    '정지 직전의 현재 상태에서 $h_0=\phi(x_0)-d_{\min}-r_0$가 음수였다. 아래 phi는 로그의 signed PSDF이며, polygon 거리와 최대 0.000031 mm 차이였다. 단위는 mm이다.', '',
    '| Seed | Safe-stop 구간 (s) | phi(x0) | Risk margin r0 | d_min+r0 | h0 | 종료 사유 |',
    '|---|---:|---:|---:|---:|---:|---|',
]
for r in stops:
    seed = int(r['seed'])
    trial = next(t for t in trials if float(t['width_m']) == .110 and int(t['seed']) == seed)
    lines.append(f"| {seed} | {float(r['time_before_s']):.1f}–{float(trial['final_time_s']):.1f} | "
                 f"{float(r['phi_stage0_mm']):.6f} | {float(r['risk_margin_stage0_mm']):.6f} | "
                 f"{1 + float(r['risk_margin_stage0_mm']):.6f} | {float(r['h_stage0_mm']):.6f} | solver 연속 실패 20회 |")
lines += [
    '',
    '- 실패 60 step 모두 주 SQP-RTI **status=4**, 복구 SQP_WITH_FEASIBLE_QP **status=2**였고, 복구 성공은 0회였다. Stage-0 residual이 음수인 60 step 모두 safe stop으로 이어졌으며, 음수인데 solver 성공으로 통과한 step은 없었다.',
    '- x0의 pose와 초기 covariance는 등식으로 고정된다. 따라서 현재 h0<0인 상태는 미래 입력으로 해소할 수 없고, zero input으로 멈춘 후 같은 조건이 유지되어 20회 연속 실패로 종료했다. Seed 1은 실제 거리가 1 mm보다 커도 risk margin을 더한 약 1.210 mm보다 작아 정지했다.',
    '- **이번 6 trials에서는 겹치기 전에 멈췄지만, stage 0 추가가 다음 입력 적용 후의 무충돌을 보장하는 것은 아니다.** 첫 정지 바로 전 cycle은 세 seed 모두 solver status=0이었고 h0가 양수였으나, 입력 적용 후 현재 margin 조건을 위반했다. 고정 tangent와 비선형 입력 적용에 따른 차이는 그대로 남아 있다. 기록된 자세 사이의 연속시간 충돌은 검사하지 않았다.', '',
    '### Animation', '',
    '기존 `simulation_mpc.animate_world`를 그대로 사용했다(기본 렌더링 설정, 10 fps, frame_skip=5, HTML 변환 생략). 별도 고화질 렌더러는 추가하지 않았다. 각 영상은 해당 trial의 마지막 정지 또는 목표 도달 자세까지 포함한다.', '',
    '| Width (m) | Seed 0 | Seed 1 | Seed 2 |',
    '|---|---|---|---|',
]
for width in sorted(summaries):
    cells = []
    for seed in range(3):
        video = next(r for r in animations if float(r['width_m']) == width and int(r['seed']) == seed)
        cells.append(f"[MP4]({video['path']})")
    lines.append(f'| {width:.3f} | ' + ' | '.join(cells) + ' |')
lines += [
    '',
    '### 검증과 원본 자료', '',
    f'- 관련 단위 테스트 **10개 통과**. 소스 사본 불변성, 기존 시작 자세 일치, runtime·cycle·history의 safe-stop 수, zero input 및 정지 중 자세 유지, solver 시간 CSV와 성공 step 수를 대조했다. 전체 **{shapely_checks:,}개 자세**의 clearance를 Shapely로 독립 확인했다. MP4 6개의 유효한 영상 stream·프레임 수·파일 크기도 확인했다.',
    '- ' + ', '.join([
        link('폭별 집계', 'benchmark_summary.csv'), link('Trial별 집계', 'benchmark_trials.csv'),
        link('정지 시점 데이터', 'stage0_stop_onsets.csv'), link('생략 내역', 'sweep_progress.json'),
        link('검증', 'stage0_validation.json'), link('Animation 목록', 'animations.csv'),
        link('생성 solver의 stage 정책', 'solver_stage_policy.json'),
    ]) + '.',
    '- 관측용 로깅 함수의 인자 오류로 첫 사전 실행이 주행 전에 종료된 기록은 `setup_observer_error_excluded/`에 분리했다. 실제 적용 입력 0회였으며 위 6 trials와 통계에서 제외했다. 수정 후 새 폭별 출력 폴더에서 재실행했다.',
    '- 재현용 ' + link('순차 실행', 'run_sweep.py') + '·' + link('집계', 'analyze.py') + '·' + link('stage-0 검증', 'audit_stage0.py') + ' 스크립트를 보관했다. 재실행에는 새 결과 폴더를 사용한다.',
]
section = '\n'.join(lines) + '\n'
(ROOT / 'report_section.md').write_text(section)
report = PROJECT / 'benchmark_results.md'
existing = report.read_text()
assert heading not in existing, 'This section is already present.'
prefix = report.read_bytes()
fd = os.open(str(report), os.O_WRONLY | os.O_APPEND)
try:
    payload = ('\n' + section).encode()
    assert os.write(fd, payload) == len(payload)
finally:
    os.close(fd)
assert report.read_bytes().startswith(prefix)
(ROOT / 'report_validation.json').write_text(json.dumps({
    'previous_report_sha256': hashlib.sha256(prefix).hexdigest(),
    'previous_report_preserved': True, 'section_lines': len(lines),
    'unit_tests': {'command': 'ACADOS_SOURCE_DIR=/home/taek111/projects/acados LD_LIBRARY_PATH=/home/taek111/projects/acados/lib /home/taek111/anaconda3/envs/pp/bin/python -m unittest discover -s tests -p test_cc_sf.py -q',
                   'tests_run': 10, 'result': 'OK'},
}, indent=2) + '\n')
print(report)
print(section)
