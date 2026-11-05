"""Paper figures for trial-level archived CC-SF / CC-MF diagnostics."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

COLORS={'cc_sf':'#0072B2','cc_mf':'#D55E00','SF':'#0072B2','MF':'#D55E00'}
MARKERS={'cc_sf':'o','cc_mf':'D','SF':'o','MF':'D'}
NAMES={'cc_sf':'CC-SF','cc_mf':'CC-MF','SF':'SF residual','MF':'MF residual'}
WIDTHS=list(range(125,131))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
 'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
 'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white',
 'xtick.direction':'out','ytick.direction':'out','lines.linewidth':1.1})


def tidy(ax, ylabel, title, zero=False):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#dddddd',lw=.5,zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(WIDTHS)
    ax.set_xlim(124.55,130.45)
    ax.set_xlabel('Robot width (mm)')
    ax.set_ylabel(ylabel)
    ax.set_title(title,loc='left',pad=7)
    if zero: ax.axhline(0,color='#555555',lw=.7,zorder=1)


def legend(fig, labels=('cc_sf','cc_mf')):
    h=[Line2D([0],[0],color=COLORS[m],marker=MARKERS[m],markersize=4,
              markerfacecolor=COLORS[m] if m in ('cc_sf','SF') else 'white',
              label=NAMES[m]) for m in labels]
    fig.legend(handles=h,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.53,1.005))


def trial_series(ax, frame, metric, stage=1, scope='full_observed', key='method', groups=('cc_sf','cc_mf'), scale=1.0):
    if 'metric' in frame:
        part=frame[(frame.metric==metric)&(frame.stage==stage)&(frame.scope==scope)]
        ykey='value'
    else:
        part=frame
        ykey=metric
    for j,m in enumerate(groups):
        means=[]; errs=[]; xs=[]
        for w in WIDTHS:
            s=part[(part.width_mm==w)&(part[key]==m)].sort_values('seed')
            good=np.isfinite(s[ykey].astype(float))
            s=s[good]
            yy=s[ykey].astype(float).to_numpy()*scale
            if not len(yy): continue
            x=w+(-.12 if j==0 else .12)
            jitter=(s.seed.to_numpy(dtype=float)-4.5)*.012
            ax.scatter(x+jitter,yy,s=9,marker=MARKERS[m],facecolors=COLORS[m] if j==0 else 'none',
                       edgecolors=COLORS[m],linewidths=.5,alpha=.45,zorder=2)
            xs.append(x);means.append(yy.mean());errs.append(yy.std(ddof=1) if len(yy)>1 else np.nan)
        ax.errorbar(xs,means,yerr=errs,color=COLORS[m],fmt=MARKERS[m]+'-',
                    markerfacecolor=COLORS[m] if j==0 else 'white',markersize=4,
                    capsize=2,elinewidth=.9,lw=1.05,zorder=4)


def save(fig,folder,name,caption,records):
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(folder/(name+'.png'),dpi=220,bbox_inches='tight')
    (folder/(name+'.caption.txt')).write_text(caption+'\n')
    records.append({'figure':name,'caption':caption,'pdf':name+'.pdf','png':name+'.png'})
    plt.close(fig)


def common_figure(frame,folder,stage,name,records):
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.1))
    specs=[('guard_margin_mm_p05','P05 margin (mm)','(a) Fresh geometric margin'),
           ('guard_violation_pct','Violating candidate steps (%)','(b) Fresh Guard violations'),
           ('R_exceedance_pct','Budget exceedance (%)','(c) Common multi-feature risk'),
           ('R_budget_residual_p05','P05 risk-budget margin (pp)','(d) Common chance-budget margin')]
    for ax,(metric,y,title) in zip(axes.flat,specs):
        trial_series(ax,frame,metric,stage=stage,scale=100 if metric=='R_budget_residual_p05' else 1)
        tidy(ax,y,title,zero=True)
        # Probability-budget residual is epsilon-R; zero is its constraint boundary.
    legend(fig)
    fig.tight_layout(rect=(0,0,1,.96),h_pad=2,w_pad=2)
    caption=(f'Stage {stage} ({stage*.1:g} s) diagnostics for all 120 trials, including observations before failure. '
      'Small symbols: one trial statistic; connected large symbols: arithmetic mean across ten trials per method/width; bars: sample SD (ddof=1), not confidence intervals. '
      'Within each trial, successful normal solves are evaluated using freshly recomputed geometry and their saved planning covariance. '
      '(a) P05 of phi-d_col, d_col=0.05 mm. (b) Fraction with phi-d_col < -1 micrometre. '
      '(c) Fraction with R>0.1+1e-6, conditional on phi>d_min=0.1 mm and a valid separated feature model. '
      '(d) P05 of the common chance-budget residual epsilon-R, epsilon=0.1 and R=sum_i p_i; zero is the constraint boundary. R is a common local Gaussian/Boole surrogate evaluated on both controllers, not a measured collision probability, and may exceed one. '
      'Domain exits are excluded only from (c,d) and are reported separately in the coverage figure/tables. Failure-trial observation lengths differ; see paired common-time sensitivity. '
      'Stage 10 is a predicted horizon state, not an executed next pose.' if stage==10 else
      f'Stage {stage} (0.1 s) diagnostics for all 120 trials, including observations before failure. Small symbols: one trial statistic; connected large symbols: mean across ten trials per method/width; bars: sample SD (ddof=1), not confidence intervals. '
      'Successful normal solves only, using fresh geometry and saved planning covariance. (a) Within-trial P05 of phi-d_col (d_col=0.05 mm). (b) Fraction with phi-d_col < -1 micrometre. '
      '(c) Fraction with R>0.1+1e-6 conditional on phi>d_min=0.1 mm and valid separated features. (d) Within-trial P05 of the same chance-budget residual epsilon-R, with epsilon=0.1 and zero as the constraint boundary. '
      'R is the same local Gaussian/Boole surrogate on both controllers, not measured collision probability, and may exceed one. Domain exits are reported separately. Observation lengths differ; paired common-time sensitivity is supplementary.')
    save(fig,folder,name,caption,records)



def boundary_figures(base, folder, records):
    path=base/'boundaries/per_trial_metrics.csv'
    if not path.exists():
        print('Boundary figures pending full boundary extraction.')
        return
    raw=pd.read_csv(path)
    report=json.loads((base/'boundaries/summary.json').read_text())
    if report['completed_trial_stages']!=240:
        print('Boundary figures pending: incomplete trial-stage scope.')
        return
    raw=raw[(raw.offset_um==10)&(raw.subset=='ordinary_inverse_cdf')]
    for stage,name in [(1,'fig02_switch_boundary'),(10,'figS05_stage10_switch_boundary')]:
        fig,axes=plt.subplots(2,2,figsize=(7.2,5.25),sharey='row')
        for col,source in enumerate(('cc_sf','cc_mf')):
            cohort=raw[(raw.stage==stage)&(raw.method==source)]
            for row,(sfkey,mfkey,ylabel) in enumerate([
                ('delta_sf_quantile_mean_abs','delta_mf_quantile_mean_abs','Mean |residual change| (quantile)'),
                ('delta_p_selected_mean_abs','delta_probability_sum_mean_abs','Mean |risk-budget change| (pp)')]):
                q=[]
                for evaluator,key in [('SF',sfkey),('MF',mfkey)]:
                    d=cohort[['width_mm','seed',key]].copy().rename(columns={key:'diagnostic'})
                    d['evaluation']=evaluator
                    if row==1: d['diagnostic']*=100
                    q.append(d)
                q=pd.concat(q)
                ax=axes[row,col]
                trial_series(ax,q,'diagnostic',key='evaluation',groups=('SF','MF'))
                tidy(ax,ylabel,f'({chr(97+2*row+col)}) {NAMES[source]} source trajectories',zero=True)
                counts=q[q.evaluation=='SF'].groupby('width_mm').diagnostic.count()
                ax.set_xticklabels([f'{w}\nn={int(counts.get(w,0))}' for w in WIDTHS])
        legend(fig,('SF','MF'));fig.tight_layout(rect=(0,0,1,.96),h_pad=2,w_pad=2)
        save(fig,folder,name,
          f'Stage-{stage} switching-boundary sensitivity across all widths and both source controllers, without selecting a route segment or successful trials. '
          'Two endpoints lie 10 micrometres on each side of each localized canonical-feature boundary in the body-motion metric ||delta_xy||+r|delta_theta|. '
          'The original left-cycle body-frame planning covariance is fixed for both endpoints and both evaluator functions. No extrapolation past the observed nominal segment is used. '
          'Both endpoints must be separated, valid and MF-active, with ordinary inverse-CDF branches; clipping and linear-extension events are excluded from this primary figure. '
          'Each small symbol is one trial mean absolute finite-offset change; large symbols and bars are the mean and sample SD across trials with eligible events. n is the number of such trials, not independent boundary events. '
          'Top: standardized SF and MF residual changes in common quantile units. Bottom: changes of epsilon-p_selected and epsilon-R in percentage points, using the same event set. '
          'R is a local Boole surrogate sum, not measured collision probability. Five and fifty micrometre offsets, all active branches, P95/RMS, and all excluded-event counts are retained in the tables. '
          'These are local finite-offset sensitivities, not mathematical discontinuity limits or proof of improved closed-loop completion.',records)


def rollout_figure(base,folder,records):
    path=base/'rollout_covariance/per_trial_metrics.csv'
    if not path.exists(): return
    raw=pd.read_csv(path)
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.15),sharey='row')
    for col,(model,label) in enumerate([('saved','Saved optimizer covariance'),('rollout','Dynamics-consistent covariance')]):
        part=raw[raw.covariance_model==model].copy()
        part['stage']=1;part['scope']='full_observed'
        for row,(metric,ylabel) in enumerate([('R_exceedance_pct','Budget exceedance (%)'),('R_budget_residual_p05','P05 risk-budget margin (pp)')]):
            ax=axes[row,col]
            trial_series(ax,part,metric,scale=100 if row else 1)
            tidy(ax,ylabel,f'({chr(97+row*2+col)}) {label}',zero=True)
    legend(fig);fig.tight_layout(rect=(0,0,1,.96),h_pad=2,w_pad=2)
    save(fig,folder,'figS07_covariance_consistency',
      'Stage-1 common-risk sensitivity at identical saved candidate poses and a common valid separated domain. '
      'Left: covariance stored in the optimizer solution. Right: covariance recomputed by the archived nonlinear one-step dynamics from the reset initial covariance and applied input. '
      'Each small symbol is one trial statistic; large symbols and bars are the mean and sample SD across ten trials. '
      'R is a local Gaussian/Boole surrogate in both cases, not measured collision probability. Candidate poses and actual trial outcomes are not changed or re-simulated. '
      'A documented log-Cholesky outlier strongly affects mean raw R, while its effect on threshold-exceedance frequency is much smaller. '
      'Both versions remain in the data; no outlier is deleted. Trial exposure varies, including only two valid candidates in the shortest 130 mm MF trial.',records)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--analysis-dir',type=Path,default=Path('benchmark_runs/sf_mf_width125_130mm_step001_10trials_20261104/constraint_stability_20261105'))
    args=ap.parse_args();base=args.analysis_dir
    folder=base/'figures';folder.mkdir(exist_ok=True)
    frame=pd.read_csv(base/'trial_metrics.csv');records=[]
    common_figure(frame,folder,1,'fig01_common_margin_risk',records)
    common_figure(frame,folder,10,'figS01_stage10_margin_risk',records)
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.1))
    specs=[('frozen_omission_pct','Candidate steps (%)','(a) Unselected-feature risk'),
           ('frozen_near_omission_pct','Candidate steps (%)','(b) Near-feature risk (gap <= 0.5 mm)'),
           ('sf_geometry_error_abs_mm_p95','P95 absolute error (mm)','(c) Geometry / feature model discrepancy'),
           ('sf_covariance_error_abs_mm_p95','P95 absolute error (mm)','(d) Covariance-margin linearization')]
    for ax,(metric,y,title) in zip(axes.flat,specs):
        trial_series(ax,frame,metric);tidy(ax,y,title,zero=True)
    legend(fig);fig.tight_layout(rect=(0,0,1,.96),h_pad=2,w_pad=2)
    save(fig,folder,'fig03_feature_coverage',
      'Stage-1 trial statistics; mean +/- sample SD over ten trials per width/method, with individual trial symbols. '
      '(a,b) Same frozen nominal features at the solved candidate: p_selected<=0.1-1e-6 and max p_other>0.1+1e-6. Equivalent selected feature aliases are excluded. '
      '(b) Also requires the offending feature to lie within 0.5 mm of the selected feature at the nominal state. Rates use valid nominal separated-domain candidates. '
      '(c) Within-trial P95 of |h_fresh-h_frozen|; (d) P95 of |h_frozen-h_affine|, using the same SF diagnostic row on both trajectories, in mm. '
      'These are local-model discrepancies; (d) is covariance-margin linearization error, not a covariance dynamics defect. Refer to manifest for finite-row eligibility. '
      'The MF path is evaluated counterfactually with the same SF diagnostic; these are not native MF solver residuals.',records)
    # Domain and actual-system guardrails keep conditional-risk plots interpretable.
    actual=pd.read_csv(base/'actual_trial_metrics.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.1))
    trial_series(axes[0,0],frame,'probability_domain_exit_pct');tidy(axes[0,0],'Candidate steps (%)','(a) Outside separated risk domain',zero=True)
    trial_series(axes[0,1],frame,'domain_or_R_exceedance_pct');tidy(axes[0,1],'Candidate steps (%)','(b) Domain exit or risk-budget excess',zero=True)
    trial_series(axes[1,0],actual,'actual_min_phi_mm');tidy(axes[1,0],'Minimum signed PSDF (mm)','(c) Actual geometric minimum',zero=True)
    trial_series(axes[1,1],actual,'qp_fallback_per_100_attempts');tidy(axes[1,1],'QP fallbacks / 100 attempts','(d) Transient solver fallbacks',zero=True)
    legend(fig);fig.tight_layout(rect=(0,0,1,.96),h_pad=2,w_pad=2)
    save(fig,folder,'figS02_domain_and_execution',
      'Trial points and mean +/- sample SD across ten trials. (a) Stage-1 successful normal candidates with phi<=d_min=0.1 mm. '
      '(b) Disjunction: phi<=d_min OR (valid separated model and R>epsilon+1e-6); this is a condition frequency, not an imputed probability. '
      '(c) Minimum actual signed PSDF, including failed trials; repeated terminal stops do not change the minimum. '
      '(d) QP fallback episodes per 100 solver attempts; stage-0 Guard stops are excluded. Rates do not remove trajectory/exposure confounding.',records)
    # Both residual definitions on each identical trajectory: actual temporal movement, not a boundary limit.
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.1))
    for row,stage in enumerate((1,10)):
        for col,path_method in enumerate(('cc_sf','cc_mf')):
            ax=axes[row,col]
            selected=[]
            for evaluation in ('SF','MF'):
                part=frame[(frame.stage==stage)&(frame.scope=='full_observed')&(frame.method==path_method)&
                    (frame.metric==f'temporal_ordinary_switch_{evaluation}_abs_delta_mean')].copy()
                part['evaluation']=evaluation;part['diagnostic']=part['value'];selected.append(part)
            part=pd.concat(selected).drop(columns=['metric'])
            trial_series(ax,part,'diagnostic',key='evaluation',groups=('SF','MF'))
            tidy(ax,'Mean |temporal change| (quantile)',f'({chr(97+2*row+col)}) Stage {stage}, {NAMES[path_method]} paths',zero=True)
            counts=part[part.evaluation=='SF'].groupby('width_mm').diagnostic.count()
            ax.set_xticklabels([f'{w}\nn={int(counts.get(w,0))}' for w in WIDTHS])
    legend(fig,('SF','MF'));fig.tight_layout(rect=(0,0,1,.96),h_pad=2,w_pad=2)
    save(fig,folder,'figS03_temporal_switch_changes',
      'Consecutive-cycle nominal residual changes conditional on a canonical selected-feature change. '
      'Both SF standardized residual and MF quantile residual are evaluated at the same states on each source trajectory; both MF endpoints must use the ordinary inverse-CDF branch (no extension or clipping). '
      'Each dot is the trial mean absolute change; bars are sample SD across trials with eligible events, never zero-imputed trials. Only adjacent valid, active separated-domain cycles are used; missing/fallback gaps are not bridged. '
      'Pose and covariance vary across cycles: these are observed temporal changes, not discontinuity estimates. The controlled finite-offset boundary figure isolates local boundary sensitivity. '
      'All-pair, non-switch, and ordinary inverse-CDF branch results and sample counts are retained in the CSV tables.',records)
    # Matched-prefix sensitivity: a paired effect retains the same seeds.
    paired=pd.read_csv(base/'paired_trial_differences.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8))
    for ax,metric,title,y in zip(axes,['R_exceedance_pct','guard_margin_mm_p05'],
       ['(a) Common risk-budget exceedance','(b) Lower-tail geometric margin'],
       ['MF - SF (percentage points)','MF - SF (mm)']):
        for j,(scope,label,color,marker) in enumerate([
            ('full_observed','Full observed','#0072B2','o'),('paired_common_time','Common time prefix','#D55E00','D')]):
            xs=[];means=[];stds=[]
            for w in WIDTHS:
                q=paired[(paired.stage==1)&(paired.scope==scope)&(paired.width_mm==w)&(paired.metric==metric)].sort_values('seed')
                yy=q.MF_minus_SF.to_numpy();valid=np.isfinite(yy);yy=yy[valid]
                x=w+(-.12 if j==0 else .12)
                ax.scatter(x+(q.seed.to_numpy()[valid]-4.5)*.012,yy,s=9,marker=marker,
                           facecolors=color if j==0 else 'none',edgecolors=color,lw=.5,alpha=.45)
                xs.append(x);means.append(np.mean(yy) if len(yy) else np.nan);stds.append(np.std(yy,ddof=1) if len(yy)>1 else np.nan)
            ax.errorbar(xs,means,yerr=stds,fmt=marker+'-',color=color,mfc=color if j==0 else 'white',ms=4,capsize=2,lw=1,label=label)
        tidy(ax,y,title,zero=True)
    axes[0].legend(frameon=False,loc='best',fontsize=7)
    fig.tight_layout(w_pad=2)
    save(fig,folder,'figS04_paired_exposure_sensitivity',
      'Stage-1 paired differences, MF minus SF, with ten same-seed pairs per width. Each dot is one pair and bars are mean +/- sample SD across pairs. '
      'Full observed uses each complete available trajectory. Common time prefix truncates both members at the earlier last successful normal solve time. '
      'This checks exposure sensitivity; matching elapsed time does not match pose, covariance, controls, or causal treatment.',records)
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.1))
    for row,stage in enumerate((1,10)):
        for col,(metric,ylabel) in enumerate([('R_mean','Within-trial mean R'),('R_sd','Within-trial SD of R')]):
            ax=axes[row,col]
            trial_series(ax,frame,metric,stage=stage)
            tidy(ax,ylabel,f'({chr(97+2*row+col)}) Stage {stage}',zero=True)
    legend(fig);fig.tight_layout(rect=(0,0,1,.96),h_pad=2,w_pad=2)
    save(fig,folder,'figS06_risk_mean_and_dispersion',
      'The raw common multi-feature risk sum R is evaluated at each successful normal candidate with valid separated geometry. '
      'Each point is a within-trial mean (left) or within-trial sample SD (right); large symbols and bars are the across-trial mean and sample SD of those statistics. '
      'All observed trials are included, including pre-termination segments of failed trials. R may exceed one and is not an empirical collision probability. '
      'Extreme saved planning-covariance states remain included; robust P05 chance-budget margin summaries in the main figure and the outlier audit complement these moment statistics. '
      'The two levels of SD describe temporal dispersion within a trial and dispersion between trials, respectively; neither is a confidence interval.',records)
    rollout_figure(base,folder,records)
    boundary_figures(base,folder,records)
    (folder/'figure_manifest.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_trial_metrics_sha256':hashlib.sha256((base/'trial_metrics.csv').read_bytes()).hexdigest(),'figures':records},indent=2))
    print(f'Saved {len(records)} PDF/PNG figure pairs to {folder}')

if __name__=='__main__': main()
