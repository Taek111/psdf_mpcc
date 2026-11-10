"""Audit saved SF/MF transitions into the first stage-0 guard; no probes or solves."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.analyze_sf_mf_reliability import make_wrapper
from utils.offline_cc_margin import _snapshot_imports
from utils.offline_feature_selectors import extract_feature_selectors


def main():
    torch.set_num_threads(2)
    audit_path = OUT / 'trial_terminal_audit.csv'
    trials = list(csv.DictReader(audit_path.open(encoding='utf-8')))
    trials = [row for row in trials if np.isclose(float(row['don']), .85)
              and row['source'] in ('cc_sf', 'cc_mf') and row['status'] == 'failure']
    records, endpoint_pairs, metadata_rows, sources = [], [], [], [audit_path, Path(__file__)]
    for trial in trials:
        path = ROOT / trial['trace']
        meta_path = path.with_name('raw_trace_metadata.json')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['raw_trace_sha256']
        trace = np.load(path, allow_pickle=False)
        guard = int(float(trial['first_guard_step']))
        assert guard > 0 and not meta['solver_status_infos'][guard]['solver_called']
        poses = trace['poses'][guard - 1:guard + 1]
        assert poses.shape == (2, 3)
        times = trace['pose_times_s'][guard - 1:guard + 1]
        assert abs(times[-1] - float(trial['first_guard_current_pose_time_s'])) < 1e-8
        row = dict(source=trial['source'], seed=int(trial['seed']), paired_group=trial['paired_group'],
                   applied_step=guard - 1, first_guard_step=guard,
                   time_before_s=float(times[0]), time_after_s=float(times[1]),
                   previous_solver_success=bool(meta['solver_status_infos'][guard - 1]['success']),
                   d_col_mm=1000 * meta['optimizer_parameters']['d_col'],
                   trace=str(path.relative_to(ROOT)))
        delta = poses[1] - poses[0]
        for j, field in enumerate(('x_m', 'y_m', 'theta_rad')):
            row[field + '_before'] = float(poses[0, j])
            row[field + '_after'] = float(poses[1, j])
            row['delta_' + field] = float(delta[j])
        for j, field in enumerate(('v_mps', 'omega_radps', 'virtual_speed')):
            row[field] = float(trace['applied_inputs'][guard - 1, j])
        records.append(row)
        endpoint_pairs.append(poses)
        metadata_rows.append(meta)
        sources.extend([path, meta_path])
    assert sum(row['source'] == 'cc_sf' for row in records) == 18
    assert sum(row['source'] == 'cc_mf' for row in records) == 5
    geometries = [json.dumps(meta['geometry'], sort_keys=True) for meta in metadata_rows]
    assert len(set(geometries)) == 1
    pairs = np.asarray(endpoint_pairs)
    delta = pairs[:, 1] - pairs[:, 0]
    suite = ROOT / 'benchmark_runs/rectangle_constraint_reliability_20261107/rectangle_don_sweep_50_20261105_current_sf'
    with _snapshot_imports(suite / 'source_snapshot') as (module, _, _):
        for dtype, suffix in ((torch.float64, ''), (torch.float32, '_float32')):
            wrapper = make_wrapper(module, metadata_rows[0], dtype)
            poses = pairs.reshape(-1, 3)
            selected = extract_feature_selectors(wrapper, poses, verify=True)
            ph = np.asarray(selected['phi']).reshape(-1, 2)
            grad = np.asarray(selected['world_gradient']).reshape(-1, 2, 3)
            ids = np.asarray(selected['canonical_id']).reshape(-1, 2)
            clusters = np.asarray(selected['active_cluster']).reshape(-1, 2)
            pc, gc = wrapper.forward_clusters(torch.tensor(poses, dtype=dtype))
            pc = pc.numpy().astype(float).reshape(len(pairs), 2, -1)
            gc = gc.numpy().astype(float).reshape(len(pairs), 2, -1, 3)
            lin = ph[:, 0] + np.einsum('ni,ni->n', grad[:, 0], delta)
            all_cluster_lin = pc[:, 0] + np.einsum('nki,ni->nk', gc[:, 0], delta)
            for i, row in enumerate(records):
                threshold = row['d_col_mm'] / 1000
                cluster_change = bool(clusters[i, 0] != clusters[i, 1])
                feature_change = bool(ids[i, 0] != ids[i, 1])
                missed = bool(lin[i] >= threshold and ph[i, 1] < threshold)
                values = dict(
                    phi_before_mm=float(ph[i, 0] * 1000),
                    phi_after_mm=float(ph[i, 1] * 1000),
                    active_tangent_end_mm=float(lin[i] * 1000),
                    actual_minus_tangent_mm=float((ph[i, 1] - lin[i]) * 1000),
                    active_cluster_before=int(clusters[i, 0]),
                    active_cluster_after=int(clusters[i, 1]),
                    feature_id_before=str(ids[i, 0]), feature_id_after=str(ids[i, 1]),
                    cluster_changed=cluster_change, feature_changed=feature_change,
                    current_tangent_safe=bool(lin[i] >= threshold),
                    actual_below_guard=bool(ph[i, 1] < threshold),
                    missed_by_current_active_tangent=missed,
                    min_all_cluster_tangents_mm=float(all_cluster_lin[i].min() * 1000),
                    all_cluster_tangents_flag_guard=bool(all_cluster_lin[i].min() < threshold),
                    after_cluster_before_distance_mm=float(pc[i, 0, clusters[i, 1]] * 1000),
                    after_cluster_before_tangent_mm=float(all_cluster_lin[i, clusters[i, 1]] * 1000),
                )
                for j, name in enumerate(('x', 'y', 'theta')):
                    values['gradient_before_' + name] = float(grad[i, 0, j])
                row.update({key + suffix: value for key, value in values.items()})
    with (OUT / 'terminal_step_tangent_audit.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    def aggregate(group, suffix=''):
        field = lambda row, name: row[name + suffix]
        return dict(
            n_trials=len(group),
            previous_solver_success=sum(row['previous_solver_success'] for row in group),
            actual_below_guard=sum(field(row, 'actual_below_guard') for row in group),
            feature_changed=sum(field(row, 'feature_changed') for row in group),
            cluster_changed=sum(field(row, 'cluster_changed') for row in group),
            missed_by_current_active_tangent=sum(field(row, 'missed_by_current_active_tangent') for row in group),
            missed_with_cluster_change=sum(field(row, 'missed_by_current_active_tangent') and field(row, 'cluster_changed') for row in group),
            missed_same_cluster_feature_change=sum(field(row, 'missed_by_current_active_tangent') and not field(row, 'cluster_changed') and field(row, 'feature_changed') for row in group),
            missed_same_feature=sum(field(row, 'missed_by_current_active_tangent') and not field(row, 'feature_changed') for row in group),
            all_cluster_tangents_flag_guard=sum(field(row, 'all_cluster_tangents_flag_guard') for row in group),
            missed_by_current_but_flagged_by_all_cluster_tangents=sum(field(row, 'missed_by_current_active_tangent') and field(row, 'all_cluster_tangents_flag_guard') for row in group),
            missed_seeds=[row['seed'] for row in group if field(row, 'missed_by_current_active_tangent')],
        )
    groups = {'all_sf_failures': [row for row in records if row['source'] == 'cc_sf'],
              'mf_only_success_sf_failure': [row for row in records if row['source'] == 'cc_sf' and row['paired_group'] == 'mf_only_success'],
              'all_mf_failures': [row for row in records if row['source'] == 'cc_mf']}
    sources += [ROOT / 'scripts/analyze_sf_mf_reliability.py', ROOT / 'utils/offline_feature_selectors.py',
                suite / 'source_snapshot/models/augmented_psdf.py', suite / 'source_snapshot/models/augmented_psdf_wrapper.py']
    report = dict(
        scope='All 18 DoN .85 SF failures, including 17 paired MF-only-success trials, plus all 5 MF failures as a comparator. Only two recorded endpoint poses immediately before first stage0 guard; no pose interpolation, local probes, simulations or solver calls.',
        diagnostic='phi(x_before)+gradient(x_before) dot (actual_x_after-actual_x_before), compared with actual fresh phi(x_after) and d_col=.05mm.',
        caveat='This CURRENT-POSE tangent is not the native future-stage row anchored at retained warm-start nominal state. All-cluster tangents are a geometric counterfactual, not the deployed MF probability-sum row. Neither comparison identifies the causal source of the SF/MF success difference. Both methods share a single signed-PSDF guard.',
        precision_checks={label: {name: aggregate(group, suffix) for name, group in groups.items()}
                          for label, suffix in (('float64', ''), ('float32', '_float32'))},
        classification_precision_disagreements=sum(row['missed_by_current_active_tangent'] != row['missed_by_current_active_tangent_float32'] for row in records),
        sources=[{'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources],
    )
    (OUT / 'terminal_step_tangent_summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report['precision_checks'], indent=2))


if __name__ == '__main__':
    main()
