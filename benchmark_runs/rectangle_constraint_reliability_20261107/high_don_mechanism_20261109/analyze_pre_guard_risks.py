"""Retrospective risk warnings at recorded SF poses; no probes or simulation."""

import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.analyze_sf_mf_reliability import evaluate, make_wrapper
from utils.offline_cc_margin import _snapshot_imports

SNAPSHOT = ROOT / 'benchmark_runs/rectangle_constraint_reliability_20261107/rectangle_don_sweep_50_20261105_current_sf/source_snapshot'


def save_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(2)
    trials = list(csv.DictReader((OUT / 'trial_terminal_audit.csv').open(encoding='utf-8')))
    trials = [r for r in trials if float(r['don']) == .85 and r['source'] == 'cc_sf']
    samples, summaries, hashes = [], [], {}
    with _snapshot_imports(SNAPSHOT) as (module, covariance, _):
        for trial in trials:
            path = ROOT / trial['trace']
            meta_path = path.with_name('raw_trace_metadata.json')
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            raw = np.load(path)
            poses, times, inputs = raw['poses'], raw['pose_times_s'], raw['applied_inputs']
            guard = float(trial['first_guard_current_pose_time_s']) if trial['first_guard_current_pose_time_s'] else np.nan
            # All first 10 s, plus a predeclared last 2 s before any first guard.
            before_guard = (times[1:] < guard - 1e-8) if np.isfinite(guard) else np.ones(len(inputs), bool)
            window = (times[1:] <= 10 + 1e-8) | (np.isfinite(guard) & (times[1:] >= guard - 2 - 1e-8))
            eligible = np.array([bool(s['success'] and s['solver_called']) for s in meta['solver_status_infos']])
            steps = np.flatnonzero(window & before_guard & eligible)
            p = SimpleNamespace(**meta['optimizer_parameters'])
            xi0 = covariance.log_cholesky_from_covariance(covariance.initial_local_covariance(p))
            xi = covariance.propagate_log_cholesky(xi0, inputs[steps, 0], inputs[steps, 1], 0., meta['dt_s'], p)
            wrapper = make_wrapper(module, meta, torch.float64)
            ev = evaluate(wrapper, poses[steps + 1], xi, .1, p.d_min)
            states = np.column_stack((poses[steps + 1], np.zeros(len(steps)), xi))
            result = wrapper.forward_mf_icdf(torch.tensor(states, dtype=torch.float64), .1, p.d_min,
                                             return_frozen_features=True)
            d, q, mask = [result.frozen_features[k].numpy() for k in ('d', 'q', 'mask')]
            variance = (np.exp(xi[:, None, 0]) * q[:, :, 0]) ** 2
            variance += (np.exp(xi[:, None, 1]) * q[:, :, 1] + xi[:, None, 3] * q[:, :, 2]) ** 2
            variance += (np.exp(xi[:, None, 2]) * q[:, :, 2]) ** 2
            sigma = np.sqrt(np.where(mask, variance, 1))
            pi = (.5 * torch.special.erfc(torch.tensor((d - p.d_min) / sigma) / np.sqrt(2))).numpy()
            pi[~mask] = 0
            if np.max(np.abs(pi.sum(1) - ev['r_mf'])) > 1e-10:
                raise AssertionError('Per-feature probabilities fail MF sum cross-check')
            maxp, argmax = pi.max(1), pi.argmax(1)
            valid = ev['phi'] > p.d_min
            individual = valid & (ev['p_sf'] <= .100001) & (maxp > .100001)
            aggregate = valid & (maxp <= .100001) & (ev['r_mf'] > .100001)
            preguard = np.isfinite(guard) & (times[steps + 1] >= guard - 2 - 1e-8)
            for k, step in enumerate(steps):
                samples.append(dict(seed=int(trial['seed']), paired_group=trial['paired_group'], outcome=trial['status'],
                    step=int(step), time_s=float(times[step + 1]), first_guard_time_s=guard,
                    pre_guard_2s=bool(preguard[k]), phi_mm=float(1000 * ev['phi'][k]), valid_phi=bool(valid[k]),
                    x_m=float(poses[step + 1, 0]), y_m=float(poses[step + 1, 1]),
                    p_sf=float(ev['p_sf'][k]), max_feature_p=float(maxp[k]), r_mf=float(ev['r_mf'][k]),
                    top_feature_slot=int(argmax[k]), extra_individual_warning=bool(individual[k]),
                    aggregation_only_warning=bool(aggregate[k])))
            for label, subset in [('first10s', times[steps + 1] <= 10 + 1e-8), ('pre_guard_2s', preguard)]:
                if not subset.any():
                    continue
                last = np.flatnonzero(subset)[-1]
                summaries.append(dict(seed=int(trial['seed']), paired_group=trial['paired_group'], outcome=trial['status'],
                    window=label, samples=int(subset.sum()), valid_samples=int((subset & valid).sum()),
                    sf_risk_exceedances=int(np.sum(subset & valid & (ev['p_sf'] > .100001))),
                    extra_individual_warnings=int(np.sum(subset & individual)),
                    aggregation_only_warnings=int(np.sum(subset & aggregate)),
                    max_additional_individual_risk=float(np.max((maxp - ev['p_sf'])[subset])),
                    max_sum_increment=float(np.max((ev['r_mf'] - ev['p_sf'])[subset])),
                    last_time_s=float(times[steps[last] + 1]), last_phi_mm=float(1000 * ev['phi'][last]),
                    last_p_sf=float(ev['p_sf'][last]), last_max_feature_p=float(maxp[last]),
                    last_r_mf=float(ev['r_mf'][last])))
            for file in (path, meta_path):
                hashes[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    save_csv('pre_guard_risk_samples.csv', samples)
    save_csv('pre_guard_risk_trial_summary.csv', summaries)
    cohorts = []
    for label, selected in [
        ('MF-only success / SF failed, last 2s before guard', [r for r in summaries if r['paired_group'] == 'mf_only_success' and r['window'] == 'pre_guard_2s']),
        ('All SF failures, last 2s before guard', [r for r in summaries if r['outcome'] != 'success' and r['window'] == 'pre_guard_2s']),
        ('All SF success, first 10s', [r for r in summaries if r['outcome'] == 'success' and r['window'] == 'first10s']),
    ]:
        cohorts.append(dict(cohort=label, n_trials=len(selected), samples=sum(r['samples'] for r in selected),
            valid_samples=sum(r['valid_samples'] for r in selected),
            trials_sf_exceeded=sum(r['sf_risk_exceedances'] > 0 for r in selected),
            trials_extra_individual_warning=sum(r['extra_individual_warnings'] > 0 for r in selected),
            trials_aggregation_only_warning=sum(r['aggregation_only_warnings'] > 0 for r in selected),
            extra_individual_warning_samples=sum(r['extra_individual_warnings'] for r in selected),
            aggregation_only_warning_samples=sum(r['aggregation_only_warnings'] for r in selected),
            trials_last_sample_sf_exceeded=sum(r['last_p_sf'] > .100001 for r in selected)))
    report = dict(scope='Recorded actual post-integration poses only; initial covariance reset then propagated one step with recorded applied input; no native planned covariance/candidate/slack available. No local probes, optimizer or simulations.',
        selection='All 50 DoN .85 SF trials. Only samples produced by successful solver calls, strictly before first guard, in first 10s or final 2s before guard. Warnings require phi>d_min. Successful controls are time-window controls, not spatially matched.',
        epsilon=.1, probability_tolerance=1e-6, cohorts=cohorts, source_hashes=hashes,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / 'pre_guard_risk_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(cohorts, indent=2))


if __name__ == '__main__':
    main()
