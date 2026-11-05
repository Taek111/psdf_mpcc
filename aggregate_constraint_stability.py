"""Aggregate archived constraint diagnostics at the trial grain; never run a controller."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def values(d, key):
    return np.asarray(d[key], dtype=float)


def finite_stats(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {k: np.nan for k in ('mean', 'sd', 'p05', 'p95', 'min', 'max')}
    return {'mean': a.mean(), 'sd': a.std(ddof=1) if len(a)>1 else np.nan,
            'p05': np.quantile(a,.05), 'p95': np.quantile(a,.95), 'min':a.min(), 'max':a.max()}


def summarize_cache(d, scope, limit=np.inf):
    """Equal trial weights are applied later; step denominators remain explicit."""
    ident={k: np.asarray(d[k]).item() for k in ('width_mm','method','seed','stage')}
    ident['scope']=scope
    result=[]
    def put(metric, val, n):
        result.append(dict(ident, metric=metric, value=float(val), n_samples=int(n)))
    def describe(name,a):
        a=np.asarray(a,dtype=float)
        a=a[np.isfinite(a)]
        for stat,val in finite_stats(a).items(): put(name+'_'+stat,val,len(a))
    time=values(d,'time')
    within=time<=limit+1e-9
    post=np.asarray(d['post_valid'],bool)&within
    put('successful_normal_cycles',int(post.sum()),int(within.sum()))
    pvalid=np.asarray(d['post_probability_valid'],bool)&post
    phi=values(d,'post_phi_m')
    guard=values(d,'post_guard_margin_mm')
    describe('guard_margin_mm',guard[post])
    if post.any():
        put('guard_violation_pct',100*np.mean(guard[post]<-0.001),post.sum())
        put('nonpositive_phi_pct',100*np.mean(phi[post]<-1e-6),post.sum())
        put('probability_domain_exit_pct',100*np.mean(phi[post]<=0.0001),post.sum())
    for field,name in [('R','R'),('p_selected','p_selected'),('p_max','p_max'),
                       ('sf_quantile','sf_quantile'),('mf_quantile','mf_quantile')]:
        describe(name,values(d,'post_'+field)[pvalid])
    if pvalid.any():
        R=values(d,'post_R')[pvalid]
        ps=values(d,'post_p_selected')[pvalid]
        pm=values(d,'post_p_max')[pvalid]
        put('R_exceedance_pct',100*np.mean(R>0.1+1e-6),pvalid.sum())
        put('pmax_exceedance_pct',100*np.mean(pm>0.1+1e-6),pvalid.sum())
        put('selected_exceedance_pct',100*np.mean(ps>0.1+1e-6),pvalid.sum())
        describe('R_budget_residual',0.1-R)
        describe('R_budget_excess',np.maximum(R-.1,0))
    # A composite condition is a disjunction, never an imputed probability.
    geom=np.isfinite(phi)&post
    covered=(phi<=.0001)|np.asarray(d['post_probability_valid'],bool)
    composite=geom&covered
    if composite.any():
        bad=(phi<=.0001)|(values(d,'post_R')>.1+1e-6)
        put('domain_or_R_exceedance_pct',100*np.mean(bad[composite]),composite.sum())
    fkey='frozenpost_probability_valid'
    if fkey in d:
        fm=np.asarray(d[fkey],bool)&post&np.asarray(d['nominal_valid'],bool)
        if fm.any():
            ps=values(d,'frozenpost_p_selected')
            po=values(d,'frozenpost_p_other_max')
            omitted=(ps<=.1-1e-6)&(po>.1+1e-6)
            put('frozen_omission_pct',100*np.mean(omitted[fm]),fm.sum())
            if 'frozenpost_p_other_gap_at_nominal_mm' in d:
                near=values(d,'frozenpost_p_other_gap_at_nominal_mm')<=.5
                put('frozen_near_omission_pct',100*np.mean((omitted&near)[fm]),fm.sum())
    # SF rows remain defined outside the MF probability domain. In particular,
    # do not discard a guard-failure precursor merely because phi_nom <= d_min.
    if 'frozenpost_sf_native_margin_mm' in d and 'frozenpost_sf_affine_margin_mm' in d:
        frozen=values(d,'frozenpost_sf_native_margin_mm')
        affine=values(d,'frozenpost_sf_affine_margin_mm')
        fresh=values(d,'post_sf_native_margin_mm')
        sf_valid=(post & np.asarray(d['nominal_valid'],bool)
                  & np.isfinite(frozen) & np.isfinite(affine)
                  & np.isfinite(fresh) & np.isfinite(guard))
        describe('sf_geometry_error_abs_mm',np.abs(fresh-frozen)[sf_valid])
        describe('sf_covariance_error_abs_mm',np.abs(frozen-affine)[sf_valid])
        describe('sf_total_error_abs_mm',np.abs(fresh-affine)[sf_valid])
        if sf_valid.any():
            put('sf_affine_pass_fresh_guard_fail_pct',
                100*np.mean(((affine>=-.001)&(guard<-.001))[sf_valid]),sf_valid.sum())
    # Consecutive samples only: never bridge a failed solve or a missing replay.
    # These are common offline models at validated reconstructed states. A
    # CUDA-vs-CPU mismatch of the implemented MF row must not censor either
    # evaluator; replay_valid is reserved for native solver-row comparisons.
    nv=np.asarray(d['nominal_valid'],bool)&within
    prob=np.asarray(d['nominal_probability_valid'],bool)
    mfok=np.asarray(d['nominal_mf_required'],bool)&np.asarray(d['nominal_mf_valid'],bool)
    sf_quantile=values(d,'nominal_sf_quantile')
    mf_quantile=values(d,'nominal_mf_quantile')
    eligible=nv&prob&mfok&np.isfinite(sf_quantile)&np.isfinite(mf_quantile)
    consecutive=(np.diff(values(d,'cycle'))==1)&(np.diff(time)>0)
    adjacent=eligible[1:]&eligible[:-1]&consecutive
    changed=np.asarray(d['nominal_selector_id'])[1:]!=np.asarray(d['nominal_selector_id'])[:-1]
    branch=np.asarray(d['nominal_mf_branch'])
    ordinary=(branch[1:]==1)&(branch[:-1]==1)
    if adjacent.any(): put('selector_switch_pct',100*np.mean(changed[adjacent]),adjacent.sum())
    for group,mask in [('all',adjacent),('switch',adjacent&changed),('non_switch',adjacent&~changed),
                       ('ordinary_switch',adjacent&changed&ordinary)]:
        ds=np.abs(np.diff(sf_quantile))
        dm=np.abs(np.diff(mf_quantile))
        for evaluator,delta in [('SF',ds),('MF',dm)]:
            describe('temporal_'+group+'_'+evaluator+'_abs_delta',delta[mask])
        if mask.any(): put('temporal_'+group+'_MF_smaller_pct',100*np.mean(dm[mask]<ds[mask]),mask.sum())
    return result


def aggregate(frame):
    return frame.groupby(['stage','scope','width_mm','method','metric'],dropna=False).agg(
        mean=('value','mean'),sd=('value','std'),n_trials=('value','count'),
        min=('value','min'),max=('value','max'),total_samples=('n_samples','sum')).reset_index()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--experiment',type=Path,default=Path('benchmark_runs/sf_mf_width125_130mm_step001_10trials_20261104'))
    ap.add_argument('--analysis-dir',type=Path)
    args=ap.parse_args()
    base=args.analysis_dir or args.experiment/'constraint_stability_20261105'
    files=sorted((base/'cycle_cache').glob('*.npz'))
    if len(files)!=240: raise RuntimeError(f'Expected 240 stage/trial caches, got {len(files)}')
    data={}
    for p in files:
        with np.load(p,allow_pickle=False) as f:
            d={k:f[k] for k in f.files}
        params=json.loads(str(np.asarray(d['params_json']).item()))
        for key_expected,val_expected in [('chance_epsilon',.1),('d_min',.0001),('d_col',.00005)]:
            if not np.isclose(params.get(key_expected,val_expected),val_expected,rtol=0,atol=1e-12):
                raise ValueError(f'Non-common diagnostic parameter {key_expected} in {p}')
        key=tuple(np.asarray(d[k]).item() for k in ('width_mm','method','seed','stage'))
        if key in data: raise RuntimeError(f'Duplicate {key}')
        data[key]=d
    expected={(w,m,seed,stage) for w in range(125,131)
              for m in ('cc_sf','cc_mf') for seed in range(10) for stage in (1,10)}
    if set(data)!=expected:
        raise RuntimeError(f'Cache population mismatch: missing={sorted(expected-set(data))}; '
                           f'unexpected={sorted(set(data)-expected)}')
    rows=[]
    coverage=[]
    for (w,m,seed,stage),d in data.items():
        other=data[w,'cc_mf' if m=='cc_sf' else 'cc_sf',seed,stage]
        final=[]
        for x in (d,other):
            mask=np.asarray(x['post_valid'],bool)
            final.append(float(values(x,'time')[mask].max()) if mask.any() else -np.inf)
        limit=min(final)
        rows.extend(summarize_cache(d,'full_observed'))
        rows.extend(summarize_cache(d,'paired_common_time',limit))
        coverage.append({'width_mm':w,'method':m,'seed':seed,'stage':stage,
                         'cycles':len(d['time']),'successful_normal':int(np.sum(d['post_valid'])),
                         'nominal_valid':int(np.sum(d['nominal_valid'])),
                         'post_probability_valid':int(np.sum(np.asarray(d['post_probability_valid'],bool)&np.asarray(d['post_valid'],bool))),
                         'common_time_end_s':limit})
    frame=pd.DataFrame(rows)
    frame.to_csv(base/'trial_metrics.csv',index=False)
    grouped=aggregate(frame)
    grouped.to_csv(base/'width_mean_sd.csv',index=False)
    pd.DataFrame(coverage).to_csv(base/'coverage.csv',index=False)
    pairs=frame.pivot(index=['stage','scope','width_mm','seed','metric'],columns='method',values='value').reset_index()
    pairs['MF_minus_SF']=pairs['cc_mf']-pairs['cc_sf']
    pairs.to_csv(base/'paired_trial_differences.csv',index=False)
    pairs.groupby(['stage','scope','width_mm','metric']).agg(mean=('MF_minus_SF','mean'),
        sd=('MF_minus_SF','std'),n_pairs=('MF_minus_SF','count')).reset_index().to_csv(base/'paired_mean_sd.csv',index=False)
    old=json.loads((args.experiment/'transition_analysis/comparison.json').read_text())
    actual=[]
    for t in old['trials']:
        actual.append({'width_mm':t['width_mm'],'method':t['method'],'seed':t['seed'],
            'success':t['success'],'actual_min_phi_mm':1000*t['current_signed_psdf_m']['min'],
            'actual_p05_phi_mm':1000*t['current_signed_psdf_m']['p05'],
            'qp_fallback_steps':t['qp_failure_stop_steps'],
            'qp_fallback_per_100_attempts':t['qp_failure_steps_per_100_solver_attempts'],
            'guard_entry':t['terminal_safe_stop_episodes'],
            'safe_stop_steps':t['safe_stop_steps'],'safe_stop_episodes':t['safe_stop_episodes'],
            'solver_attempts':t['solver_attempted_cycles'],'final_time_s':t['final_time_s']})
    pd.DataFrame(actual).to_csv(base/'actual_trial_metrics.csv',index=False)
    manifest={'experiment':str(args.experiment.resolve()),'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cache_files':len(files),'trials':120,'stages':[1,10],
              'aggregation':'Compute each metric within each trial first; arithmetic mean and sample SD(ddof=1) across trials at each width/method/stage. No pooled step weighting.',
              'scopes':{'full_observed':'All available successful normal solves, including failed trials before termination.',
                        'paired_common_time':'Each pair truncated at the earlier final successful normal cycle time; descriptive exposure sensitivity, not causal matching.'},
              'probability':'R is the raw implemented multi-feature Boole surrogate sum, not an empirical collision probability; conditional on phi>d_min and valid features.',
              'thresholds':{'epsilon':.1,'probability_tolerance':1e-6,'guard_margin_violation_mm':-.001},
              'temporal_change':'Changes between consecutive nominal stages; pose/covariance both vary. Not a discontinuity estimate.',
              'sf_error_decomposition':{'geometry':'Fresh minus frozen nominal-feature SF margin at the same postsolve state; includes distance and gradient/orientation changes.',
                                        'covariance':'Frozen nonlinear SF margin minus its affine approximation in covariance-state coordinates; not a covariance dynamics defect.',
                                        'domain':'SF error and affine-pass/fresh-guard-fail metrics use all finite validated nominal/postsolve comparisons, including outside the MF probability domain. Common offline SF/MF diagnostics use nominal_valid, not native MF row replay_valid.'},
              'missing':'No valid samples =>NaN, never zero. Native SF and MF residuals have different units and are never directly pooled.',
              'across_widths':'Seeds0..9 reused; no independent n=60 inference.'}
    (base/'aggregation_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Wrote {len(frame)} trial metrics, {len(grouped)} width summaries; 120 trials /240 stage caches.')

if __name__=='__main__': main()
