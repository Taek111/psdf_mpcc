#!/usr/bin/env python3
"""Offline aggregation; never imports optimizers, simulation, or solver packages."""
from pathlib import Path
import itertools
import json
import math
import hashlib
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parents[1]
ROOTS = {'A': PROJECT/'benchmark_runs/rectangle_don_100_20261105_current_sf',
         'B': PROJECT/'benchmark_runs/rectangle_don_sweep_50_20261105_current_sf'}
METHODS = ['mpcc', 'cc_ecs', 'cc_sf', 'cc_mf']
LABELS = dict(zip(METHODS, ['MPCC', 'CC-ECS', 'CC-SF', 'CC-MF']))
COMPLETE = {'success', 'failure', 'timeout', 'error'}
COUNTERS = ['safe_stop_count', 'backup_feasible_qp_success_count',
            'stage0_guard_stop_count', 'main_solver_failure_count',
            'backup_solver_failure_count', 'plant_input_apply_count']


def load(path):
    return json.loads(path.read_text())


def save(df, name):
    df.to_csv(OUT/'tables'/name, index=False, float_format='%.17g')


def wilson(k, n):
    if not n:
        return np.nan, np.nan
    z = 1.959963984540054
    p = k/n
    d = 1+z*z/n
    center = (p+z*z/(2*n))/d
    radius = z/d*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return max(0., center-radius), min(1., center+radius)


def failure_category(r):
    reason = str(r.failure_reason)
    if r.status == 'success':
        return 'success'
    if r.status == 'timeout':
        return 'timeout'
    if r.status == 'interrupted':
        return 'interrupted'
    if r.status == 'error':
        return 'solver_or_other_exception'
    if 'consecutive' in reason and 'stage0' in reason:
        return 'consecutive_stage0_stop'
    if 'consecutive' in reason:
        return 'consecutive_solver_or_control_failure'
    if 'stuck' in reason:
        return 'stagnation'
    return 'other_failure'


def main():
    trials, summaries, timing, summary_checks, raw_audit, receipts = [], [], [], [], [], []
    for cohort, root in ROOTS.items():
        manifest = load(root/'suite_manifest.json')
        for w in manifest['widths']:
            wr = root/w['results_dir']
            td = pd.read_csv(wr/'trial_results.csv', keep_default_na=False)
            td['cohort'] = cohort
            td['width_m'], td['nominal_don'] = w['width_m'], w['nominal_don']
            td['suite_root'] = str(root)
            td['source_trial_csv'] = str(wr/'trial_results.csv')
            td['failure_category'] = td.apply(failure_category, axis=1)
            for c in COUNTERS:
                td['runtime_'+c] = np.nan
            for idx, row in td.iterrows():
                data = wr/'runs'/row.configuration_id/f'trial_{int(row.trial_number):04d}'/'data'
                mp, npz = data/'raw_trace_metadata.json', data/'raw_trace.npz'
                ar = dict(cohort=cohort, width_m=w['width_m'], nominal_don=w['nominal_don'],
                          optimizer_type=row.optimizer_type, start_seed=int(row.start_seed))
                if not mp.exists() or not npz.exists():
                    ar['unavailable'] = True
                    raw_audit.append(ar)
                    continue
                meta = load(mp)
                stats = meta['outcome'].get('optimizer_runtime_stats') or {}
                for c in COUNTERS:
                    if c in stats and stats[c] is not None:
                        td.loc[idx, 'runtime_'+c] = stats[c]
                with np.load(npz, allow_pickle=False) as a:
                    times = np.array(a['solve_nlp_times_s'])
                    ar['pose_samples'] = len(a['poses'])
                    ar['applied_input_samples'] = len(a['applied_inputs'])
                statuses = meta['solver_status_infos']
                counts = Counter()
                for i, t in enumerate(times):
                    if i >= len(statuses):
                        counts['status_missing'] += 1
                    elif statuses[i].get('solver_called') is False:
                        counts['solver_not_called'] += 1
                    elif statuses[i].get('success') is not True:
                        counts['failed_or_unknown_status'] += 1
                    elif not np.isfinite(t) or t < 0:
                        counts['invalid_time'] += 1
                    else:
                        counts['eligible_success_steps'] += 1
                ar.update({k: counts[k] for k in ['status_missing', 'solver_not_called',
                          'failed_or_unknown_status', 'invalid_time', 'eligible_success_steps']})
                ar.update(raw_time_samples=len(times), raw_status_samples=len(statuses),
                          status_without_time=max(0, len(statuses)-len(times)))
                raw_audit.append(ar)
            trials.append(td)
            sd = pd.read_csv(wr/'summary.csv')
            step_frames = []
            for path in sorted(wr.glob('computation_times_*.csv')):
                steps = pd.read_csv(path)
                steps['step_source'] = str(path)
                step_frames.append(steps)
                receipts.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            steps = pd.concat(step_frames, ignore_index=True)
            for method in METHODS:
                g = td[td.optimizer_type == method]
                done = g[g.status.isin(COMPLETE)]
                success = done[done.status == 'success']
                low, high = wilson(len(success), len(done))
                r = dict(cohort=cohort, width_m=w['width_m'], nominal_don=w['nominal_don'],
                         optimizer_type=method, expected_trials=manifest['trials_per_configuration'],
                         recorded_trials=len(g), completed_trials=len(done), successful_trials=len(success),
                         interrupted_trials=int((g.status == 'interrupted').sum()),
                         success_rate=len(success)/len(done) if len(done) else np.nan,
                         wilson95_low=low, wilson95_high=high,
                         traversal_success_n=len(success), traversal_success_mean_s=pd.to_numeric(success.final_time).mean())
                for c in COUNTERS:
                    vals = done['runtime_'+c]
                    r[c] = vals.sum(min_count=1)
                    r[c+'_available_trials'] = int(vals.notna().sum())
                    r[c+'_positive_trials'] = int((vals>0).sum()) if vals.notna().any() else np.nan
                summaries.append(r)
                st = steps[steps.optimizer_type == method].copy()
                num = pd.to_numeric(st.solver_computation_time_s, errors='coerce')
                valid = np.isfinite(num) & (num >= 0) & (st.solver_status_code == 0)
                eligible_trials = set(zip(done.trial_number.astype(int), done.start_seed.astype(int)))
                valid &= np.array([(int(t),int(s)) in eligible_trials for t,s in zip(st.trial_number, st.start_seed)])
                v = num[valid]
                timing.append(dict(cohort=cohort, width_m=w['width_m'], nominal_don=w['nominal_don'],
                                   optimizer_type=method, step_csv_rows=len(st), included_steps=len(v),
                                   excluded_step_csv_rows=int((~valid).sum()),
                                   included_trials=int(st.loc[valid,'trial_number'].nunique()),
                                   included_non_success_trial_steps=int((st.loc[valid,'trial_status']!='success').sum()),
                                   first_steps_included=int((st.loc[valid,'timestep']==0).sum()),
                                   solve_mean_s=v.mean(), solve_p95_s=v.quantile(.95)))
                old = sd[sd.optimizer_type == method].iloc[0]
                for field, actual in {'completed_trials':len(done), 'successful_trials':len(success),
                                      'successful_solver_steps':len(v), 'success_rate':r['success_rate'],
                                      'solver_time_mean_s':v.mean(),'solver_time_p95_s':v.quantile(.95)}.items():
                    observed = old[field]
                    summary_checks.append(dict(cohort=cohort,width_m=w['width_m'],optimizer_type=method,
                                               field=field,reported=observed,recomputed=actual,
                                               matched=bool(np.isclose(observed, actual, rtol=1e-10,atol=1e-12,equal_nan=True))))
    tr = pd.concat(trials, ignore_index=True)
    su, ti = pd.DataFrame(summaries), pd.DataFrame(timing)
    save(tr, 'trials_enriched.csv')
    save(su, 'success_and_runtime_summary.csv')
    save(ti, 'pooled_solve_nlp_times.csv')
    save(pd.DataFrame(raw_audit), 'raw_timing_exclusions.csv')
    save(tr[tr.status != 'success'], 'failure_trials.csv')
    save(tr.groupby(['cohort','nominal_don','optimizer_type','status','failure_category','failure_reason'],dropna=False).size().rename('trials').reset_index(), 'outcome_reason_distribution.csv')
    pd.DataFrame(summary_checks).to_csv(OUT/'audit'/'metrics_summary_reconciliation.csv',index=False)
    (OUT/'audit'/'timing_source_receipts.json').write_text(json.dumps(receipts,indent=2))
    paired_summary, paired_seeds, pair_traversal, all_common = [], [], [], []
    for (cohort, don), group in tr.groupby(['cohort','nominal_don'],sort=True):
        by = {m:g.set_index('start_seed') for m,g in group.groupby('optimizer_type')}
        common4 = sorted(set.intersection(*(set(g[g.status=='success'].index) for g in by.values())))
        for m in METHODS:
            all_common.append(dict(cohort=cohort,nominal_don=don,optimizer_type=m,
                                   common_success_n=len(common4), mean_traversal_s=pd.to_numeric(by[m].loc[common4,'final_time']).mean(),
                                   seeds=json.dumps([int(x) for x in common4])))
        for a,b in itertools.combinations(METHODS,2):
            shared = sorted(set(by[a][by[a].status.isin(COMPLETE)].index)&set(by[b][by[b].status.isin(COMPLETE)].index))
            buckets = Counter()
            diffs=[]
            for seed in shared:
                ra,rb=by[a].loc[seed],by[b].loc[seed]
                pose_match=np.allclose(ra[['initial_x','initial_y','initial_theta']].to_numpy(dtype=float), rb[['initial_x','initial_y','initial_theta']].to_numpy(dtype=float),rtol=0,atol=1e-12)
                sa,sb=ra.status=='success',rb.status=='success'
                typ='both_success' if sa and sb else 'a_only_success' if sa else 'b_only_success' if sb else 'both_non_success'
                buckets[typ]+=1
                delta=float(ra.final_time)-float(rb.final_time) if sa and sb and pose_match else np.nan
                if np.isfinite(delta): diffs.append(delta)
                paired_seeds.append(dict(cohort=cohort,nominal_don=don,method_a=a,method_b=b,start_seed=int(seed),
                                         pose_matched=pose_match,outcome_pair=typ, traversal_a_s=float(ra.final_time) if sa else np.nan,
                                         traversal_b_s=float(rb.final_time) if sb else np.nan, paired_delta_a_minus_b_s=delta,
                                         common_all_four_success=seed in common4))
            paired_summary.append(dict(cohort=cohort,nominal_don=don,method_a=a,method_b=b,common_completed_n=len(shared),
                                       **{k:buckets[k] for k in ['both_success','both_non_success','a_only_success','b_only_success']}))
            pair_traversal.append(dict(cohort=cohort,nominal_don=don,method_a=a,method_b=b,paired_success_n=len(diffs),
                                       mean_delta_a_minus_b_s=np.mean(diffs) if diffs else np.nan,
                                       median_delta_a_minus_b_s=np.median(diffs) if diffs else np.nan))
    save(pd.DataFrame(paired_summary),'paired_success_comparisons.csv')
    save(pd.DataFrame(paired_seeds),'paired_seed_membership.csv')
    save(pd.DataFrame(pair_traversal),'paired_traversal_comparisons.csv')
    save(pd.DataFrame(all_common),'common_four_success_traversal.csv')
    a = su[su.cohort=='A'].merge(ti,on=['cohort','width_m','nominal_don','optimizer_type'],validate='one_to_one')
    clearpath = OUT/'offline_clearance'/'clearance_summary.csv'
    if clearpath.exists():
        cl = pd.read_csv(clearpath)
        cl['cohort'] = cl.suite_root.map({str(v):k for k,v in ROOTS.items()})
        cols=['cohort','width_m','nominal_don','optimizer_type','available_completed_trials','unavailable_completed_trials','pooled_pose_samples','clearance_mean_mm','clearance_p05_mm','clearance_min_mm','zero_clearance_samples','negative_boundary_samples','coverage_complete']
        a=a.merge(cl[cols],on=['cohort','width_m','nominal_don','optimizer_type'],validate='one_to_one',how='left')
    save(a,'A_quantitative_table.csv')
    b=su[su.cohort=='B']
    save(b[['nominal_don','width_m','optimizer_type','successful_trials','completed_trials','success_rate','wilson95_low','wilson95_high']], 'B_success_rates.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#2563A6','#B77912','#C95645','#657F35']
    markers=['o','s','^','D']
    fig, axes=plt.subplots(2,2,figsize=(15,9),sharex=True,sharey=True,layout='constrained')
    for ax,m,color,marker in zip(axes.flat,METHODS,colors,markers):
        g=b[b.optimizer_type==m].sort_values('nominal_don')
        x=g.nominal_don.to_numpy(); y=g.success_rate.to_numpy()*100
        ax.fill_between(x,g.wilson95_low.to_numpy()*100,g.wilson95_high.to_numpy()*100,color=color,alpha=.17)
        ax.plot(x,y,marker=marker,color=color,lw=1.8,ms=5)
        ax.set_title(LABELS[m],loc='left',weight='bold')
        for xx,n in zip(x,g.completed_trials):
            ax.text(xx,105,str(int(n)),fontsize=8,ha='center',color='#444444')
        ax.text(.696,105,'n',fontsize=8,ha='right')
        ax.axvline(.75,color='#777777',linestyle=':',linewidth=.9)
        ax.set_ylim(-3,112);ax.set_xlim(.697,.873)
        ax.set_xticks(np.arange(.70,.871,.01));ax.tick_params(axis='x',labelsize=9)
        ax.set_yticks([0,20,40,60,80,100]);ax.grid(axis='y',alpha=.2)
        ax.set_xlabel('Nominal DoN = robot width / 0.150 m');ax.set_ylabel('Goal-reaching success (%)')
    fig.suptitle('B: Rectangle DoN sweep | Wilson 95% intervals',fontsize=16,weight='bold')
    fig.supxlabel('Each point: actual completed n shown above. DoN 0.75 reuses A seeds 0–49; A seeds 50–99 are excluded.\nWidth = DoN × 0.150 m (0.1050–0.1305 m). Connecting lines describe observations only.',fontsize=10)
    fig.savefig(OUT/'figures'/'B_success_rates.png',dpi=180)
    fig.savefig(OUT/'figures'/'B_success_rates.svg')
    plt.close(fig)
    print(su[su.cohort=='A'][['nominal_don','optimizer_type','successful_trials','completed_trials','traversal_success_mean_s','safe_stop_count','backup_feasible_qp_success_count']].to_string(index=False))
    print('Summary reconciliation mismatches:',sum(not c['matched'] for c in summary_checks))


if __name__ == '__main__':
    main()
