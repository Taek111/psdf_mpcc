#!/usr/bin/env python3
"""Build the Korean Markdown report solely from preserved offline outputs."""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parents[1]
METHODS = ['mpcc','cc_ecs','cc_sf','cc_mf']
NAMES = dict(zip(METHODS,['MPCC','CC-ECS','CC-SF','CC-MF']))


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def f(value, digits=3):
    return '—' if pd.isna(value) else f'{value:.{digits}f}'


def main():
    a = pd.read_csv(OUT/'tables/A_quantitative_table.csv')
    b = pd.read_csv(OUT/'tables/B_success_rates.csv')
    s = pd.read_csv(OUT/'tables/success_and_runtime_summary.csv')
    tr = pd.read_csv(OUT/'tables/trials_enriched.csv',keep_default_na=False)
    common = pd.read_csv(OUT/'tables/common_four_success_traversal.csv')
    pairs = pd.read_csv(OUT/'tables/paired_success_comparisons.csv')
    pt = pd.read_csv(OUT/'tables/paired_traversal_comparisons.csv')
    cl = pd.read_csv(OUT/'offline_clearance/clearance_trials.csv')
    cm = json.loads((OUT/'offline_clearance/analysis_manifest.json').read_text())
    prov = json.loads((OUT/'audit/provenance_audit.json').read_text())
    raw = pd.read_csv(OUT/'tables/raw_timing_exclusions.csv')
    assert 'clearance_mean_mm' in a, 'Run aggregate_metrics.py after clearance completes'
    success = tr.groupby('cohort').apply(lambda x:int((x.status=='success').sum()))
    zero = cl[cl.included_in_primary_pool==1].groupby('suite_root').agg(samples=('pose_samples','sum'),zero=('zero_clearance_samples','sum'),negative=('negative_boundary_samples','sum'))
    lines = [
        '# Rectangle benchmark 분석 — 현재 CC-SF snapshot, 2026-11-05 실행',
        '',
        f'A 800/800건과 B 3,600/3,600건이 모두 완료됐으며, 총 {cm["cohort_trial_memberships"]:,}개 cohort 행에서 B DoN 0.75의 200건을 제외한 고유 원자료는 {cm["unique_source_trial_origins"]:,}건이다. 원본 trial·trace는 전부 가용하고 strict clearance audit를 통과했다. A/B의 실행 source·기록 환경·resolved 파라미터는 동일하다. **이는 승인된 현재 snapshot 내부의 일치이며, 2026-11-03 제어 수식과의 동일성을 뜻하지 않는다.**',
        '',
        'A DoN 0.50에서는 네 방법이 모두 100/100 성공했다. DoN 0.75에서는 MPCC 98/100, CC-ECS 100/100, CC-SF 100/100, CC-MF 99/100이었다. B에서는 CC-ECS가 DoN 0.78–0.87, MPCC가 0.79–0.87에서 관측 성공 0/50이었다. CC-SF·CC-MF는 0.86에서도 각각 4/50·20/50 성공했고 0.87에서는 모두 0/50이었다. 이 결과는 측정한 너비·seed 집합의 목표 도달 결과이며, 연속적인 임계값이나 무충돌 보장은 아니다.',
        '',
        '**목표 도달과 접촉 관측은 분리해야 한다.** A DoN 0.75의 MPCC는 75/100 trial에서 clearance ≤0이 관측됐고 그중 73건은 목표 도달 성공이었다. 나머지 두 실패 trial에서는 지도 경계 침범이 관측됐다. A의 다른 일곱 방법·너비 조합에서는 비양수 clearance 표본이 없었다. P05처럼 pooled 요약값이 양수여도 짧은 접촉을 놓칠 수 있다.',
        '',
        '## 데이터 범위와 승인된 기준 변경',
        '',
        '- A: `../rectangle_don_100_20261105_current_sf`, manifest 생성 시각 **2026-11-05 16:05:53.873879 KST**; DoN 0.50·0.75, 각 방법 seed 0–99.',
        '- B: `../rectangle_don_sweep_50_20261105_current_sf`, manifest 생성 시각 **2026-11-05 16:56:52.798695 KST**; DoN 0.70–0.87, 0.01 간격, 각 방법 seed 0–49.',
        '- 기준 정의: DoN = rectangle 너비 / 직선 통로 너비 0.150 m. 길이 0.150 m, rear_dist=0, DD, maze, A*, dt=0.1 s, horizon=20, tf=2 s, 제한 80 s.',
        '- 성공은 목표 위치 L1 오차 ≤0.02 m이며 heading 조건은 없다. 10초 이동량 ≤0.03 m 또는 연속 unsuccessful control step 20회가 실패 기준이다. 초기 교란은 xy 표준편차 0.02 m, heading 0.05 rad, ±3σ 제한, 초기 clearance ≥0.001 m다.',
        '- localization error, animation, plot, master `use_debug`는 모두 꺼져 있다. `benchmark_logging`으로 저장된 최소 실제 trace를 사용했다.',
        '',
        f'실행 snapshot 59개 파일을 각각 재해시한 공통 aggregate SHA-256은 `{prov["source_aggregate_sha256"]}`이다. [실행 소스·설정 audit](audit/provenance_audit.json), [파일별 hash](audit/provenance_source_hashes.csv), [생성 YAML/run manifest 대조](audit/provenance_configs.csv), [trial 조건 전수 대조](audit/provenance_trial_conditions.csv)에 근거를 남겼다.',
        '',
        '최초 `../rectangle_don_100_20261105`는 preflight에서 과거 baseline에 없는 CC-SF `d_col=0.00005 m`가 검출되어 **주행 전에 실패**했다. 이 디렉토리의 trial CSV는 0개다. baseline JSON의 `approved_amendments`는 사용자가 현재 CC-SF 유지, 기준 변경 기록 및 실행을 승인했음을 기록한다. 최초 시도와 현재 A snapshot 사이 변경 파일은 baseline JSON 하나이며, resolved 파라미터는 동일하다. 그 외 비진단 파라미터 검사는 유지됐고 A/B는 수정된 기준에 `matched: true`, `differences: {}`로 일치한다.',
        '',
        '과거와 현재 CC-SF source는 실제로 다르다. 2026-11-03 구현은 stage 0..N−1의 projected-variance 행 한 개와 terminal 제약 없음이었고, 현재는 stage 1..N의 hard PSDF guard 및 projected-variance 두 행, terminal 제약, 측정 pose의 별도 stage-0 guard를 사용한다. 따라서 `d_col`만 새로 기록했다는 사실을 근거로 **과거 제어 수식도 같다고 해석하면 안 된다.** [과거/현재 소스 근거](audit/provenance_notes.md)와 [실제 source diff](audit/provenance_cc_sf_historical_to_current.diff)를 함께 참조한다.',
        '',
        'Run manifest의 `cli.simulation_time=60`, `s_path`, `psdf`는 batch 실행에 쓰이지 않은 parser 기본값이다. 실제 적용값은 생성 config/configurations와 snapshot의 override 경로에서 80 s, maze, 해당 네 controller로 확인했다.',
        '',
        '## 완료성·매칭·trace audit',
        '',
        table(['항목','A','B'],[
            ['계획 / 기록 / 완료','800 / 800 / 800','3,600 / 3,600 / 3,600'],
            ['성공 / failure',f'{success["A"]} / {800-success["A"]}',f'{success["B"]} / {3600-success["B"]}'],
            ['timeout / error / interrupted','0 / 0 / 0','0 / 0 / 0'],
            ['미실행 / suite 내부 중복 / 예상 밖 seed','0 / 0 / 0','0 / 0 / 0'],
            ['가용 완료 trace / 누락·손상','800 / 0','3,600 / 0'],
            ['같은 너비·seed의 네 방법 pose 매칭','200/200 seed 묶음','900/900 seed 묶음'],
        ]),
        '',
        '전 trace의 schema, NPZ SHA-256, shape·finite 값, 배열 개수, 초기 pose, 시간 증가, 실제 적분 수 N과 상태 N+1, 실제 geometry, CSV/metadata outcome, step CSV 및 기존 summary를 대조했다. 성공 시간 mean/P95는 80개 그룹에서 다시 계산하여 기존 summary와 일치했다. RK4를 저장 입력으로 오프라인 재계산한 잔차도 audit에 기록했다. 최종 성공 판정의 실제 A* grid endpoint는 `[1.81875, 0.09375]`로 source에서 확인했으며 nominal 목표와 혼동하지 않았다.',
        '',
        '같은 너비 안에서 seed별 시작 pose는 네 방법이 일치한다(절대 허용오차 1e−12). 초기 clearance 최솟값은 약 1.009015 mm로 1 mm 수락 조건을 만족한다. 너비가 달라지면 rejection sampling으로 수락 pose가 달라질 수 있으므로 너비 간 pose 일치를 가정하지 않는다. [전수 trace audit](audit/trace_audit.json)과 관련 `trace*.csv`에 trial별 근거를 보존했다.',
        '',
        'B DoN 0.75의 200건은 A의 동일 너비·방법·seed 0–49다. 원본 suite/trial CSV hash, NPZ/metadata byte hash, 결과·pose·geometry·parameters·timing·환경이 모두 일치한다. B 내부 canonical 경로의 사본 및 A에서 필터한 성공 step 75,912행을 확인했다. A의 100건과 B의 50건을 실험별로 유지하며 독립 150건으로 합산하지 않는다. 고유 원자료 집계는 origin suite-manifest hash, width_id, controller, seed로 구분한다. [재사용 trial 목록](audit/provenance_reuse_trials.csv), [step 대조](audit/provenance_reuse_steps.csv).',
        '',
        '## A: 주행 정량 비교',
        '',
        '아래 CI는 성공률의 Wilson 95% 구간이다. Clearance는 모든 완료 trial의 초기 및 적분 후 pose를 pooled하여 mean/P05를 계산했다. Solve time은 최종 outcome과 무관하게 성공한 solver step 전체를 pooled한 mean/P95다. Traversal은 성공 trial의 final simulation time 평균이다.',
        '',
    ]
    for don in [.5,.75]:
        g=a[np.isclose(a.nominal_don,don)].set_index('optimizer_type')
        lines += [f'### DoN {don:.2f} · 너비 {don*.15:.4f} m','',table(
            ['방법','성공/완료','성공률 % [95% CI]','Clearance mean/P05 (mm)','Solve mean/P95 (ms/step)','성공 traversal (s; n)','Safe-stop / recovery'],[
                [NAMES[m],f'{int(r.successful_trials)}/{int(r.completed_trials)}',f'{r.success_rate*100:.1f} [{r.wilson95_low*100:.1f}, {r.wilson95_high*100:.1f}]',
                 f'{r.clearance_mean_mm:.3f} / {r.clearance_p05_mm:.3f}',f'{r.solve_mean_s*1000:.3f} / {r.solve_p95_s*1000:.3f}',
                 f'{r.traversal_success_mean_s:.3f}; {int(r.traversal_success_n)}',
                 '— / —¹' if m=='mpcc' else f'{int(r.safe_stop_count)} / {int(r.backup_feasible_qp_success_count)}']
                for m in METHODS for r in [g.loc[m]]]),'']
    lines += [
        '¹ MPCC에는 CC 방식의 zero-input safe-stop/backup 경로가 없으며 runtime 카운터가 저장되지 않았다. 구조적으로 해당 fallback이 없다는 source 근거는 있지만, 관측 카운터를 숫자 0으로 채우지 않았다. CC 세 방법의 A 카운터는 각 조건 100/100 metadata에서 가용하며 모두 관측 0이다.',
        '',
        table(['DoN','방법','Clearance trial / pose 표본','Solve trial / 성공 step','미성공 trial에서 포함한 성공 step','첫 step 포함 수'],[
            [f'{r.nominal_don:.2f}',NAMES[r.optimizer_type],f'{int(r.available_completed_trials)} / {int(r.pooled_pose_samples):,}',
             f'{int(r.included_trials)} / {int(r.included_steps):,}',int(r.included_non_success_trial_steps),int(r.first_steps_included)]
            for _,r in a.iterrows()]),
        '',
        'A DoN 0.75에서 MPCC 실패 trial의 성공 step 341개와 CC-MF 실패 trial의 성공 step 321개도 시간 통계에 포함했다. 모든 A 그룹의 첫 step 100개가 포함됐고, step CSV 안의 invalid/비성공 행 제외는 0개였다. Raw trace에 있는 비성공·solver 미호출 시간은 주 통계에서 제외되며 [제외 내역](tables/raw_timing_exclusions.csv)에 구분했다.',
        '',
        '성공-only traversal은 방법별 성공 seed가 달라 선택 편향이 있다. 네 방법이 모두 성공한 동일 seed 집합을 따로 적용하면 다음과 같다.',
        '',
        table(['DoN','공통 성공 seed 수','MPCC (s)','CC-ECS (s)','CC-SF (s)','CC-MF (s)'],[
            [f'{d:.2f}',int(common[(common.cohort=='A')&np.isclose(common.nominal_don,d)].iloc[0].common_success_n)]+
            [f(common[(common.cohort=='A')&np.isclose(common.nominal_don,d)&(common.optimizer_type==m)].iloc[0].mean_traversal_s) for m in METHODS]
            for d in [.5,.75]]),
        '',
        'DoN 0.50은 seed 0–99 전부, 0.75는 seed 23·30·99를 제외한 97개다. 공통 성공 cohort의 simulation traversal은 DoN 0.50에서 CC-SF, DoN 0.75에서 MPCC가 가장 짧았다. 그러나 목표 미도달 trial을 제거한 조건부 비교이며 전체 시도 성능이나 안전성의 단일 순위로 해석하지 않는다. 실패에 80초를 대입하지 않았다.',
        '',
        'A DoN 0.75의 paired 성공 결과는 다음과 같다. 동일 seed·pose의 완료 trial을 비교했으며 별도 유의성 검정이나 다중비교 p값을 계산하지 않았다.',
        '',
        table(['방법 A / B','공통 완료','둘 다 성공','둘 다 미성공','A만 성공','B만 성공'],[
            [f'{NAMES[r.method_a]} / {NAMES[r.method_b]}',int(r.common_completed_n),int(r.both_success),int(r.both_non_success),int(r.a_only_success),int(r.b_only_success)]
            for _,r in pairs[(pairs.cohort=='A')&np.isclose(pairs.nominal_don,.75)].iterrows()]),
        '',
        '[A 전체 정량 CSV](tables/A_quantitative_table.csv), [모든 그룹 성공·runtime](tables/success_and_runtime_summary.csv), [pooled solve time](tables/pooled_solve_nlp_times.csv), [paired seed 전 목록](tables/paired_seed_membership.csv), [모든 pair traversal 차이](tables/paired_traversal_comparisons.csv), [네 방법 공통 성공 seed와 traversal](tables/common_four_success_traversal.csv).',
        '',
        '## B: DoN sweep 성공률',
        '',
        '모든 지점에서 방법당 완료 n=50이다. 셀은 `성공/완료; 성공률% [Wilson 95% CI%]`다. 0/50의 구간은 [0.0, 7.1]%, 50/50은 [92.9, 100.0]%여서 관측 0·100%를 확정 확률로 해석하지 않는다.',
        '',
        table(['DoN','너비 (m)']+[NAMES[m] for m in METHODS],[
            [f'{d:.2f}',f'{d*.15:.4f}']+[
                f'{int(r.successful_trials)}/{int(r.completed_trials)}; {r.success_rate*100:.0f} [{r.wilson95_low*100:.1f}, {r.wilson95_high*100:.1f}]'
                for m in METHODS for r in [b[np.isclose(b.nominal_don,d)&(b.optimizer_type==m)].iloc[0]]]
            for d in sorted(b.nominal_don.unique())]),
        '',
        '![B DoN별 성공률과 Wilson 95% 구간. 네 panel의 축 범위가 동일하고 각 지점의 실제 n을 표시한다.](figures/B_success_rates.png)',
        '',
        'DoN 0.70–0.74에서는 네 방법 모두 50/50이었다. MPCC는 0.76에서 12/50으로 낮아졌다가 0.77·0.78에서 47/50으로 돌아왔다. 원본 trial/summary를 독립 재검증한 실제 비단조 결과다. 0.76의 실패 38건은 모두 연속 solver/control 실패였고, 0.77의 실패 3건(seed 6·11·20)은 모두 정체였다. 상세 내부 제약 로그가 없으므로 이 차이를 특정 제약·feature 전환의 인과로 단정할 수 없다.',
        '',
        'CC-ECS는 0.77의 50/50에서 0.78의 0/50으로 변했다. CC-SF와 CC-MF는 더 큰 측정 DoN에서도 성공이 관측됐으며, 0.84·0.85·0.86에서 각각 CC-SF 34·32·4건, CC-MF 49·45·20건이었다. 이 범위의 관측 차이를 보여주는 결과이지 미측정 지점의 한계값 추정은 아니다. 각 너비의 seed pose는 방법 간 같지만 너비 간에는 달라질 수 있다.',
        '',
        'B의 성공-only traversal, 네 방법 공통 성공 cohort, 여섯 method pair별 공통 성공 traversal도 각각 CSV에 계산했다. CC-ECS 성공이 0인 DoN 0.78 이상에서는 네 방법 모두 성공한 seed가 없어 four-way traversal은 —이며, 가용한 두 방법 간 pair 결과는 유지했다. 서로 다른 성공 집합의 단순 평균으로 속도 우열을 주장하지 않는다.',
        '',
        '[B 성공률 원표](tables/B_success_rates.csv), [vector 그림](figures/B_success_rates.svg), [B를 포함한 paired 성공 비교](tables/paired_success_comparisons.csv).',
        '',
        '## Safe-stop·recovery 및 실패 분포',
        '',
        'Safe-stop은 metadata `optimizer_runtime_stats.safe_stop_count`에 기록된 zero-input fallback **제어 step 수**다. 연속 20개 stop은 20 step이며 독립적인 20개 위험 사건으로 세지 않는다. 실제 입력 0만으로 stop을 재분류하지 않았다. Recovery는 `backup_feasible_qp_success_count`다.',
        '',
    ]
    rt=s[s.cohort=='B'].groupby('optimizer_type').sum(numeric_only=True,min_count=1)
    lines += [table(['B 방법','가용 runtime trial','Safe-stop step','그중 stage-0 guard','Recovery 성공','주 solver 실패 counter'],[
        [NAMES[m],f'{int(rt.loc[m,"safe_stop_count_available_trials"])}/900',f(rt.loc[m,'safe_stop_count'],0),
         '해당 경로 없음' if m=='cc_ecs' else f(rt.loc[m,'stage0_guard_stop_count'],0),
         f(rt.loc[m,'backup_feasible_qp_success_count'],0),
         '미갱신 (저장값 0)' if m=='cc_ecs' else f(rt.loc[m,'main_solver_failure_count'],0)] for m in METHODS]),
        '',
        'CC-ECS의 주 solver 실패 카운터는 상속된 초기값을 override `solve_nlp()`가 갱신하지 않는다. 저장값 0은 실제 실패 0의 증거가 아니다. ECS safe-stop 44 step은 별도 유효 카운터이며 stage-0 guard 경로는 없다. CC-SF는 safe-stop 2,921 step 중 stage-0 guard 2,780, CC-MF는 2,146 중 2,100이었다. Recovery 성공은 CC 세 방법 모두 관측 0이다. A의 CC safe-stop/recovery는 모두 0이며, 위 B 합계에는 재사용된 A seed 0–49가 포함되어 있다.',
        '',
        '일부 raw status에는 `safe_stop_stage0`와 `solver_called=false`가 있어 stage-0 stop step을 식별할 수 있다. 그러나 일반 fallback/recovery의 정확한 per-call solve_mode가 완전하게 저장되지 않아 전체 safe-stop episode 수·최장 연속 길이·정확한 발생 시점 및 recovery episode는 **미계측**으로 남긴다. Aggregate 카운터나 마지막 모드로 이를 복원하지 않았다.',
        '',
        table(['cohort','방법','정체','연속 solver/control 실패','연속 stage-0 stop','timeout','exception'],[
            [c,NAMES[m]]+[int(((tr.cohort==c)&(tr.optimizer_type==m)&(tr.failure_category==cat)).sum()) for cat in
                         ['stagnation','consecutive_solver_or_control_failure','consecutive_stage0_stop','timeout','solver_or_other_exception']]
            for c in ['A','B'] for m in METHODS]),
        '',
        'A 실패는 DoN 0.75의 MPCC seed 30·99(`consecutive_solver_failures(20): failure`)와 CC-MF seed 23(`stuck_for_10.00s_disp_0.0279m_le_0.0300m`)이다. B의 CC-ECS 실패 503건은 정체, CC-SF 139건은 반복 stage-0 stop, CC-MF 106건은 반복 stage-0 stop 105건과 정체 1건이다. MPCC 실패 495건은 연속 실패 289건과 정체 206건이다. 실패 원문과 seed는 [trial 목록](tables/failure_trials.csv), [너비·방법·원문별 분포](tables/outcome_reason_distribution.csv)에 보존했다. 목표 미도달을 접촉이라고 분류하지 않았다.',
        '',
        '## Clearance의 의미와 관측 범위',
        '',
        f'완료 trace {cm["recorded_trials"]:,}개 cohort 행, 고유 원본 NPZ/metadata 쌍 {cm["unique_available_source_trace_records"]:,}개가 가용했다. 실제 초기 pose와 매 plant 적분 후 pose 총 {int(cl.pose_samples.sum()):,}개 cohort 표본을 사용했다(재사용 표본은 A/B 각 cohort에 포함). Clearance missing/invalid는 0이며 0 대체를 하지 않았다.',
        '',
        '각 pose의 실제 rectangle을 world 좌표로 변환하고 snapshot `sim/start_perturbation.py`의 `pose_clearance()` / `convex_polygon_clearance()`로 모든 obstacle의 Euclidean polygon 거리와 signed map-boundary margin의 최솟값을 계산했다. 예측 궤적, PSDF 로그 또는 chance margin을 사용하지 않았다. 장애물 접촉·겹침은 0, 지도 경계 침범은 음수다. 0은 지도 경계에 정확히 닿은 경우도 포함하므로 음수를 obstacle penetration depth로 해석하지 않는다.',
        '',
        'Mean/P05는 trial 평균들의 평균이 아니라 개별 pose를 모두 합친 **step 가중 통계**다. 오래 주행하거나 정체한 trial이 더 큰 가중치를 가지며, 성공·failure·timeout·error를 포함하고 interrupted는 제외한다. A의 모든 그룹에서 실제 coverage는 계획 trial 수와 같다.',
        '',
        table(['cohort','가용 pose 표본','clearance == 0 표본','clearance < 0 표본','0 이하가 있는 trial','그중 목표 성공 trial'],[
            [c,f'{int(g.pose_samples.sum()):,}',f'{int(g.zero_clearance_samples.sum()):,}',f'{int(g.negative_boundary_samples.sum()):,}',
             int(((g.zero_clearance_samples+g.negative_boundary_samples)>0).sum()),
             int((((g.zero_clearance_samples+g.negative_boundary_samples)>0)&(g.status=='success')).sum())]
            for c in ['A','B'] for g in [cl[cl.suite_root.str.contains('rectangle_don_100_' if c=='A' else 'rectangle_don_sweep_50_')]]]),
        '',
        '위 판정은 추가 허용오차 없이 `==0` 및 `<0`으로 구분했다. 최소 clearance와 표본·trial별 접촉 분포는 [clearance trial CSV](offline_clearance/clearance_trials.csv)와 [clearance summary](offline_clearance/clearance_summary.csv)에 남겼다. 모든 관측은 dt=0.1 s의 이산 pose이며, 표본 사이 연속 시간의 무충돌이나 확률적 안전성을 증명하지 않는다. 양수 clearance의 guard stop도 실제 접촉과 구분한다.',
        '',
        'A DoN 0.75 MPCC의 비양수 표본은 101/30,093개(0.3356%)다. 성공 73건에서 0 표본 79개가 관측됐고, 실패 seed 30·99에서 경계 침범이 각각 11개 관측됐으며 최솟값은 −118.201 mm다. A 전체의 낮은 비양수 표본 비율은 해당 trial 비율 75/100을 대체하지 않는다. B도 모든 방법·너비의 원자료를 유지한 [접촉 요약](offline_clearance/contact_summary.csv)과 [비양수 관측 trial 목록](offline_clearance/nonpositive_clearance_trials.csv)을 제공한다.',
        '',
        '## 지표 계산과 해석 제한',
        '',
        '성공률 분모는 `success`, `failure`, `timeout`, `error` 완료 trial이고 `interrupted`는 제외한다. Wilson 구간은 z=1.959963984540054, d=1+z²/n, 중심=(p+z²/(2n))/d, 반폭=z/d×sqrt(p(1−p)/n+z²/(4n²))를 썼다. n=0은 공란이다. 표시에서만 반올림하고 CSV에는 계산 정밀도를 보존했다. 비교는 seed를 공유하므로 paired이며 CI 겹침 여부만으로 차이의 유의성을 판단하지 않았다.',
        '',
        '`solve_nlp()` 시간은 `time.perf_counter()` 기반 전체 호출 wall time으로 parameter 갱신·warm-start·주 solver·backup·복구·safe-stop 생성을 포함한다. 외부 setup, planning, 적분, 로깅, trace 저장 및 offline clearance는 제외한다. 성공 solver step만, 첫 step 포함, 최종 실패 trial의 성공 step 포함이라는 동일 규칙을 네 방법에 적용했다. 계산은 초 단위, 표시는 ms/step이며 quantile은 NumPy/Pandas 기본 linear 보간이다.',
        '',
        '과거 2026-11-03의 controller 전체 cycle 시간(첫 cycle 제외, fallback 포함, CUDA 동기화) 및 더 이른 내부 `solver.solve()` 시간은 현재 측정량과 다르므로 직접 비교하지 않았다. 현재 wall time을 CUDA-event kernel time 또는 명시 동기화된 GPU 시간으로 재해석하지 않는다.',
        '',
        '기록된 A/B 환경은 Python 3.10.12, NumPy 1.24.4, Torch 2.4.1+cu121, CUDA 12.1, RTX 4090, CasADi 3.7.1, compute thread 6, torch interop thread 24로 일치한다. CPU 모델·GPU driver·acados revision, 온도·전력·다른 프로세스 부하는 기록되지 않았다. 동일 기록 환경과 순차 실행 설계만으로 모든 wall-time 변동을 제거했다고 주장할 수 없다. 방법 고유 수식·noise 차이는 실험 오염으로 간주하지 않았고, 현재 CC-SF의 과거 대비 수식 차이는 별도로 공개했다.',
        '',
        '이번 분석에는 새로운 주행·solver 실행을 추가하지 않았다. 최소 trace에 없는 내부 제약 진단이나 세부 복구 모드는 새 debug 실험으로 보충하거나 기존 timing cohort에 혼합하지 않았다. 관측 결과로 인과적 기전·전 구간 최적성·안전 보장을 주장하지 않는다.',
        '',
        '## 재현·audit 산출물',
        '',
        '분석은 프로젝트 `.venv/bin/python`을 사용했고 산출물은 이 새 분석 디렉토리에만 보존했다. [입력 파일 inventory](audit/input_inventory_before.json), [원본 보존 최종 검사](audit/input_preservation.json), [분석 code/output hash manifest](analysis_manifest.json)를 제공한다. 최초 실패 preflight와 승인 baseline, 현 source snapshot 및 실제 과거 source의 근거는 provenance audit에 포함된다.',
        '',
        'Offline clearance는 원본 helper의 `--strict` 경로를 사용했다. 순차 계산의 약 29분 예상 시간을 줄이기 위해 별도 분석 wrapper가 **원본 `calculate_trial()`와 snapshot의 scalar 기하 함수를 그대로** 12개 offline worker에서 계산한 후 원본 strict audit/집계에 전달했다. 최종 종료 코드는 **0**, 소요 시간은 172.77 s다. geometry 정의와 통계는 변경하지 않았으며, 10개 trial은 직렬 재계산과 array 단위 완전 일치로 확인했다. [최종 strict 실행 기록](audit/clearance_parallel_execution.json), [직렬 일치 검증](audit/clearance_parallel_equivalence.json), [최초 순차 취소 기록](audit/clearance_execution.json)과 관련 attempt 파일에 command·중단 사유·exit code·worker 수·코드 사본을 보존했다.',
        '',
        '초기 import 중 snapshot 아래에 Python bytecode 캐시 2개가 새로 생성된 점을 확인했다. 생성 시각과 본 주행의 bytecode 비활성화 설정을 확인한 뒤 분석 디렉토리에 사본/receipt를 남기고, 분석이 만든 두 캐시와 빈 디렉토리만 제거했다. 이후 `-B`/`PYTHONDONTWRITEBYTECODE=1`로 재실행했다. CSV·NPZ·metadata·source 내용은 수정하지 않았다. [복원 기록](audit/bytecode_cleanup.json). Trace audit 초안의 nominal goal 가정 오류도 실제 A* endpoint로 고친 후 전수 재검증했으며, 원본 오류와 구분해 [분석 수정 기록](audit/trace_audit_revision.json)에 남겼다.',
        '',
        '- [집계 코드](code/aggregate_metrics.py): Wilson, pooled solve time, runtime, 실패 분포, 성공-only/four-way/pairwise traversal, seed 목록, B plot.',
        '- [provenance 코드](code/provenance_audit.py), [trace audit 코드](code/trace_audit.py), [offline helper 사본](code/analyze_rectangle_clearance.py), [병렬 offline wrapper](code/parallel_clearance.py), [보고서 생성 코드](code/write_report.py).',
        '- [독립 지표 검증](audit/independent_metrics_review.json): 별도 구현으로 Wilson·성공 수·시간·paired cohort·비단조 결과를 검토.',
        '- [기존 summary 대조](audit/metrics_summary_reconciliation.csv), [정확한 timing 원본 경로/hash](audit/timing_source_receipts.json), [전체 trial 정보](tables/trials_enriched.csv).',
        '',
        '원본 source snapshot·생성 YAML·resolved parameters가 실행 설정의 권위 있는 근거다. 현재 checkout 기본값이나 과거 10-trial 결과로 당시 설정을 추정하거나 표본을 합산하지 않았다.',
        '',
    ]
    (OUT/'report_ko.md').write_text('\n'.join(lines),encoding='utf-8')
    receipts=[]
    for folder in ['code','tables','figures','audit','offline_clearance']:
        for p in sorted((OUT/folder).rglob('*')):
            if p.is_file() and p.suffix != '.pyc' and '__pycache__' not in p.parts:
                receipts.append({'path':str(p.relative_to(OUT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
    (OUT/'analysis_manifest.json').write_text(json.dumps({'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_aggregate_sha256':prov['source_aggregate_sha256'],'python':'/home/taekwon/projects/psdf_mpcc/.venv/bin/python',
        'report':'report_ko.md','report_sha256':hashlib.sha256((OUT/'report_ko.md').read_bytes()).hexdigest(),
        'raw_sources_modified':False,'new_driving_experiments':0,'files':receipts},indent=2,ensure_ascii=False))
    print(OUT/'report_ko.md')


if __name__=='__main__':
    main()
