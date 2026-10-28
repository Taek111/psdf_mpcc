"""Audit fixed-state safety rows, first overlaps and default-rendered videos."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / 'cc_sf_width_sweep_3trials_20261027_232811'


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def save_csv(path, records):
    if not records:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    trials = read_csv(ROOT / 'benchmark_trials.csv')
    violations = read_csv(ROOT / 'geometry_violations.csv')
    all_samples = read_csv(ROOT / 'clearance_samples.csv')
    sample_by_key = {(float(r['width_m']), int(r['seed']), int(r['sample_index'])): r
                     for r in all_samples}
    first_events, stop_events, trial_audits, videos = [], [], [], []
    solver_policies = []
    for trial in trials:
        width, seed = float(trial['width_m']), int(trial['seed'])
        folder = ROOT / f'width_{round(width * 1000):03d}mm'
        cycles = [r for r in read_csv(folder / 'cycles.csv') if int(r['seed']) == seed]
        assert [int(r['cycle']) for r in cycles] == list(range(len(cycles)))
        baseline_starts = json.loads((BASELINE / folder.name / 'expected_start_poses.json').read_text())
        starts = json.loads((folder / 'expected_start_poses.json').read_text())
        assert starts == baseline_starts
        trial_overlaps = [r for r in violations if float(r['width_m']) == width
                          and int(r['seed']) == seed and r['obstacle_overlap'] == 'True']
        overlap_indices = {int(r['sample_index']) for r in trial_overlaps}
        overlap_entries = sum(i - 1 not in overlap_indices for i in overlap_indices)
        negative = [r for r in cycles if float(r['stage0_residual_m']) < -1e-9]
        negatives_success = [r for r in negative if r['safe_stop'] == '0']
        stopped = [r for r in cycles if r['safe_stop'] == '1']
        for row in stopped:
            index = int(row['cycle'])
            before = sample_by_key[(width, seed, index)]
            after = sample_by_key[(width, seed, index + 1)]
            np.testing.assert_array_equal([float(before[k]) for k in ('x', 'y', 'theta')],
                                          [float(after[k]) for k in ('x', 'y', 'theta')])
        start_indices = [int(r['cycle']) for i, r in enumerate(cycles)
                         if r['safe_stop'] == '1' and (i == 0 or cycles[i-1]['safe_stop'] == '0')]
        for index in start_indices:
            row = cycles[index]
            sample = sample_by_key[(width, seed, index)]
            stop_events.append({
                'width_m': width, 'seed': seed, 'cycle': index,
                'time_before_s': index * .1,
                'phi_stage0_mm': float(row['stage0_phi_m']) * 1000,
                'risk_margin_stage0_mm': float(row['stage0_margin_m']) * 1000,
                'h_stage0_mm': float(row['stage0_residual_m']) * 1000,
                'polygon_clearance_before_mm': float(sample['clearance_m']) * 1000,
                'overlap_before_stop': index in overlap_indices,
                'fast_status': row['fast_status'], 'backup_status': row['backup_status'],
            })
        if trial_overlaps:
            first = min(trial_overlaps, key=lambda r: int(r['sample_index']))
            index = int(first['sample_index'])
            assert index > 0
            applied = cycles[index - 1]
            next_cycle = cycles[index] if index < len(cycles) else None
            first_events.append({
                'width_m': width, 'seed': seed, 'time_after_s': float(first['time_s']),
                'input_cycle': index - 1, 'input_mode': applied['mode'],
                'input_fast_status': applied['fast_status'],
                'h_stage0_before_input_mm': float(applied['stage0_residual_m']) * 1000,
                'next_mode': next_cycle['mode'] if next_cycle else '',
                'next_h_stage0_mm': float(next_cycle['stage0_residual_m']) * 1000 if next_cycle else '',
                'next_phi_stage0_mm': float(next_cycle['stage0_phi_m']) * 1000 if next_cycle else '',
                'max_intersection_area_m2': float(first['max_intersection_area_m2']),
                'overlap_obstacle_indices': first['obstacle_indices'],
            })
        trial_audits.append({
            'width_m': width, 'seed': seed, 'status': trial['status'],
            'control_steps': len(cycles), 'safe_stop_steps': len(stopped),
            'safe_stop_episodes': len(start_indices),
            'negative_stage0_rows': len(negative),
            'negative_stage0_rows_with_success': len(negatives_success),
            'negative_stage0_rows_with_safe_stop': sum(r['safe_stop'] == '1' for r in negative),
            'overlap_samples': len(trial_overlaps), 'overlap_entries': overlap_entries,
            'safe_stop_steps_while_overlapping': sum(int(r['cycle']) in overlap_indices for r in stopped),
            'minimum_stage0_h_mm': min(float(r['stage0_residual_m']) for r in cycles) * 1000,
        })
        policies = [r for r in json.loads((folder / 'solver_stage_policy.json').read_text())
                    if r['seed'] == seed]
        assert len(policies) == 2, policies
        assert {p['nlp_solver_type'] for p in policies} == {'SQP_RTI', 'SQP_WITH_FEASIBLE_QP'}
        for policy in policies:
            assert policy['dims'] == dict(N=20, nx=8, nu=3, nh_0=1, nh=1, nh_e=0,
                                          nsh_0=0, nsh=0, nsh_e=0)
            assert policy['tf_s'] == 2.0
            solver_policies.append({'width_m': width, **policy})
        manifest = [v for v in json.loads((folder / 'animations.json').read_text()) if v['seed'] == seed]
        assert len(manifest) == 1
        video = manifest[0]
        path = Path(video['path'])
        assert path.is_file() and path.stat().st_size == video['size_bytes'] > 0
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_frames,r_frame_rate,codec_name:format=duration',
            '-of', 'json', str(path)], text=True))
        expected_frames = list(range(0, len(cycles), video['frame_skip']))
        if expected_frames[-1] != len(cycles) - 1:
            expected_frames.append(len(cycles) - 1)
        assert int(probe['streams'][0]['nb_frames']) == len(expected_frames)
        videos.append({**video, 'width_pixels': probe['streams'][0]['width'],
                       'height_pixels': probe['streams'][0]['height'],
                       'frames': len(expected_frames), 'fps': probe['streams'][0]['r_frame_rate'],
                       'codec': probe['streams'][0]['codec_name'],
                       'duration_s': float(probe['format']['duration'])})
    save_csv(ROOT / 'stage0_trial_audit.csv', trial_audits)
    save_csv(ROOT / 'stage0_stop_onsets.csv', stop_events)
    save_csv(ROOT / 'first_overlaps.csv', first_events)
    save_csv(ROOT / 'animations.csv', videos)
    (ROOT / 'solver_stage_policy.json').write_text(json.dumps(solver_policies, indent=2) + '\n')
    old_hashes = json.loads((BASELINE / 'source_hashes.json').read_text())
    hashes = json.loads((ROOT / 'source_hashes.json').read_text())
    changed = [name for name in hashes if hashes[name] != old_hashes[name]]
    assert changed == ['control/cc_sf_optimizer.py']
    assert all(hashlib.sha256((ROOT / 'source_snapshot' / n).read_bytes()).hexdigest() == h
               for n, h in hashes.items())
    validation = {
        'trials_checked': len(trials), 'videos_checked': len(videos),
        'generated_solver_policies_checked': len(solver_policies),
        'same_initial_poses_as_baseline': True,
        'source_changes_from_baseline': changed,
        'negative_stage0_rows': sum(r['negative_stage0_rows'] for r in trial_audits),
        'negative_stage0_rows_with_success': sum(r['negative_stage0_rows_with_success'] for r in trial_audits),
        'first_overlap_count': len(first_events),
        'preflight_setup_error_excluded': True,
        'all_safe_stop_poses_unchanged': True,
        'scope': 'Recorded control sample endpoints only; no continuous-time swept collision check.',
    }
    (ROOT / 'stage0_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation, indent=2))
    print(json.dumps(stop_events, indent=2))


if __name__ == '__main__':
    main()
