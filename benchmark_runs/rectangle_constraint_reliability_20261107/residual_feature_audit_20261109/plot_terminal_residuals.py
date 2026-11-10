"""Show retained-obstacle versus fresh residuals at actual guard-entry poses."""

import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / 'high_don_mechanism_20261109'
sys.path[:0] = [str(ROOT), str(PRIOR)]
from scripts.analyze_sf_mf_reliability import make_wrapper, evaluate
from utils.offline_cc_margin import _snapshot_imports
from recorded_cluster_risks import evaluate_cluster_risks, track_two_pose_clusters


def save_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(2)
    audit_path = PRIOR / 'terminal_step_tangent_audit.csv'
    trials = list(csv.DictReader(audit_path.open(encoding='utf-8')))
    suite = ROOT / 'benchmark_runs/rectangle_constraint_reliability_20261107/rectangle_don_sweep_50_20261105_current_sf'
    rows, time_rows, hashes = [], [], {}
    with _snapshot_imports(suite / 'source_snapshot') as (module, covariance, _):
        for trial in trials:
            path = ROOT / trial['trace']
            meta_path = path.with_name('raw_trace_metadata.json')
            raw = np.load(path)
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            params = SimpleNamespace(**meta['optimizer_parameters'])
            xi0 = covariance.log_cholesky_from_covariance(covariance.initial_local_covariance(params))
            k = int(trial['applied_step'])
            u = raw['applied_inputs'][k]
            xi1 = covariance.propagate_log_cholesky(xi0, u[0], u[1], 0., meta['dt_s'], params)
            wrapper = make_wrapper(module, meta, torch.float64)
            tracked = track_two_pose_clusters(wrapper, raw['poses'][k], raw['poses'][k + 1], xi1, params.d_min)
            ev = tracked.pop('evaluation')
            fresh = evaluate(wrapper, raw['poses'][k + 1], xi1, .1, params.d_min)
            assert abs(float(fresh['p_sf'][0]) - tracked['new_after_probability']) < 1e-12
            row = dict(source=trial['source'], seed=int(trial['seed']), time_s=float(raw['pose_times_s'][k + 1]),
                phi_after_mm=float(fresh['phi'][0] * 1000), mf_native_domain=bool(fresh['phi'][0] > params.d_min),
                **tracked)
            row['b_keep_pp'] = 100 * (.1 - row['old_after_probability'])
            row['b_fresh_pp'] = 100 * (.1 - row['new_after_probability'])
            row['retained_obstacle_optimism_pp'] = row['b_keep_pp'] - row['b_fresh_pp']
            row['retained_feasible_fresh_violated'] = row['b_keep_pp'] >= 0 and row['b_fresh_pp'] < 0
            rows.append(row)
            if row['source'] == 'cc_sf' and row['seed'] == 10:
                # Exactly the saved 0.1 s records, each with its own preceding-input
                # reset-one-step covariance. Both obstacles share xi at each time.
                idx = np.flatnonzero((raw['pose_times_s'][1:] >= 4.1 - 1e-8) & (raw['pose_times_s'][1:] <= 4.7 + 1e-8))
                inputs = raw['applied_inputs'][idx]
                xis = covariance.propagate_log_cholesky(xi0, inputs[:, 0], inputs[:, 1], 0., meta['dt_s'], params)
                curve = evaluate_cluster_risks(wrapper, raw['poses'][idx + 1], xis, params.d_min)
                for j, step in enumerate(idx):
                    time_rows.append(dict(time_s=float(raw['pose_times_s'][step + 1]),
                        bottom_residual_pp=float(100 * curve['residual'][j, 0]),
                        left_residual_pp=float(100 * curve['residual'][j, 14]),
                        bottom_distance_mm=float(1000 * curve['phi'][j, 0]),
                        left_distance_mm=float(1000 * curve['phi'][j, 14]),
                        active_cluster=int(curve['active_cluster'][j]),
                        mf_native_domain=bool(curve['mf_domain_active'][j])))
            for file in (path, meta_path):
                hashes[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    save_csv('terminal_residual_pairs.csv', rows)
    save_csv('seed10_recorded_obstacle_residuals.csv', time_rows)
    summary = []
    for method in ('cc_sf', 'cc_mf'):
        data = [r for r in rows if r['source'] == method]
        summary.append(dict(source=method, failures=len(data),
            retained_feasible_fresh_violated=sum(r['retained_feasible_fresh_violated'] for r in data),
            b_keep_pp_min=float(min(r['b_keep_pp'] for r in data)),
            b_keep_pp_median=float(np.median([r['b_keep_pp'] for r in data])),
            b_fresh_pp_max=float(max(r['b_fresh_pp'] for r in data)),
            b_fresh_pp_median=float(np.median([r['b_fresh_pp'] for r in data])),
            optimism_pp_min_median_max=[float(f([r['retained_obstacle_optimism_pp'] for r in data])) for f in (min, np.median, max)]))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'normal'})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.4), gridspec_kw={'width_ratios': [1.2, 1]})
    fig.subplots_adjust(left=.075, right=.98, bottom=.27, top=.80, wspace=.30)
    fig.suptitle('Residual differences across obstacles at the actual next pose', fontsize=17, y=.97)
    fig.text(.5, .91, 'DoN 0.85 | signed-distance Gaussian diagnostic, not logged optimizer residuals', ha='center', fontsize=12, color='#555555')
    ax = axes[0]
    t = [r['time_s'] for r in time_rows]
    ax.plot(t, [r['bottom_residual_pp'] for r in time_rows], '-o', color='#716757', lw=2, label='Bottom wall C0')
    ax.plot(t, [r['left_residual_pp'] for r in time_rows], '-o', color='#bd5427', lw=2, label='Left wall C14')
    ax.axhline(0, color='#222222', lw=1)
    ax.axvline(4.7, color='#bd5427', ls=':', lw=1.3)
    ax.set(xlim=(4.06, 4.79), ylim=(-95, 24), xlabel='Recorded time (s)', ylabel='Signed residual 100(epsilon - p), pp',
           title='(a) SF seed 10: two walls at each recorded pose')
    ax.legend(loc='center left', bbox_to_anchor=(.0, .40), frameon=False)
    ax.grid(axis='y', alpha=.20)
    ax.text(4.69, 13, '+9.98 pp', ha='right', color='#716757')
    ax.text(4.69, -87, '-80.12 pp', ha='right', color='#bd5427')
    ax.annotate('4.5 s: -17.83 pp\nphi = 0.281 mm > d_min', xy=(4.5, -17.82793),
                xytext=(4.35, -38), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='#716757'), color='#716757')
    ax = axes[1]
    for method, color, marker, linestyle in [('cc_sf', '#bd5427', 'o', '-'), ('cc_mf', '#26739e', '^', '--')]:
        subset = [r for r in rows if r['source'] == method]
        for i, row in enumerate(subset):
            ax.plot([0, 1], [row['b_keep_pp'], row['b_fresh_pp']], color=color, alpha=.43, lw=1, ls=linestyle)
            ax.scatter([0, 1], [row['b_keep_pp'], row['b_fresh_pp']], color=color, marker=marker,
                       s=22, alpha=.75, label=f'{method.upper().replace("_", "-")} failures (n={len(subset)})' if i == 0 else None)
    ax.axhline(0, color='#222222', lw=1)
    ax.set(xlim=(-.18, 1.18), ylim=(-95, 24), xticks=[0, 1], xticklabels=['Keep previous\nobstacle', 'Fresh active\nobstacle'],
           ylabel='Signed residual, pp', title='(b) All 23 terminal guard-entry poses')
    ax.grid(axis='y', alpha=.20)
    ax.legend(loc='center left', frameon=False, fontsize=10)
    fig.text(.075, .18, 'Each pair in (b) uses one actual next pose and one shared reconstructed covariance. No interpolated poses or local probes.', fontsize=10)
    fig.text(.075, .13, 'Terminal poses are outside the native MF activation domain. Their values are formal signed-distance Gaussian scores, not native MF violations.', fontsize=10)
    fig.text(.075, .08, 'Keeping an obstacle recomputes its closest geometry; it does not recover the primitive or nominal row used by the optimizer.', fontsize=10)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'obstacle_handover_residuals.{ext}', dpi=220, facecolor='white')
    plt.close(fig)
    report = dict(scope='All 23 terminal transitions at DoN .85; every endpoint is produced by a successful solver call before next-cycle stage0 rejection.',
        covariance='Configured initial local covariance reset then propagated for one step with the recorded applied input. The same xi1 is used across obstacles and across both poses in the exact decomposition; the time-series uses the preceding interval xi1 at each sample.',
        residual='b=epsilon-Phi((d_min-phi_cluster)/sqrt(q_cluster Sigma1 q_cluster)); fresh within-obstacle geometry, retained obstacle identity.',
        caveat='This is not native planned-stage feasibility, not a native MF probability sum, and not certified collision probability. Terminal MF domain is inactive. Actual native row, nominal pose, optimized covariance and slack were not saved.',
        groups=summary, source_hashes=hashes,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / 'terminal_residual_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
