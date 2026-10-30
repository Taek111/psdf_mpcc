"""Read existing trial logs; quantify reversals, stops, and reference continuity."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def series(rows, name):
    return np.asarray([float(row[name]) for row in rows])


def intervals(mask, minimum_samples=1):
    result = []
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    for start, end in zip(starts, ends):
        if end - start + 1 >= minimum_samples:
            result.append((int(start), int(end)))
    return result


def summarize_window(h, p, start, end):
    selected = slice(start, end + 1)
    v, omega = series(h, 'v')[selected], series(h, 'omega')[selected]
    xy = np.column_stack((series(p, 'x')[selected], series(p, 'y')[selected]))
    post_xy = np.column_stack((series(h, 'x')[selected], series(h, 'y')[selected]))
    raw = series(p, 'planner_projected_s_raw')[selected]
    path = series(p, 'path_s')[selected]
    idx = series(p, 'slice_start_idx')[selected]
    flips = ((omega[:-1] * omega[1:] < 0) &
             (np.minimum(abs(omega[:-1]), abs(omega[1:])) >= 0.05))
    return {
        'step_start': start, 'step_end': end,
        'pre_step_time_start': round(start * 0.1, 6),
        'pre_step_time_end': round(end * 0.1, 6),
        'post_step_time_start': float(h[start]['time']),
        'post_step_time_end': float(h[end]['time']),
        'mean_abs_v': float(np.mean(abs(v))),
        'v_range': [float(np.min(v)), float(np.max(v))],
        'omega_range': [float(np.min(omega)), float(np.max(omega))],
        'omega_reversals_both_magnitudes_ge_0_05': int(flips.sum()),
        'net_displacement_between_pre_step_poses': float(np.linalg.norm(xy[-1] - xy[0])),
        'sum_distance_between_pre_step_poses': float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        'net_displacement_between_post_step_poses': float(np.linalg.norm(post_xy[-1] - post_xy[0])),
        'sum_distance_between_post_step_poses': float(np.linalg.norm(np.diff(post_xy, axis=0), axis=1).sum()),
        'raw_projection_range': [float(raw.min()), float(raw.max())],
        'max_raw_projection_step': float(np.abs(np.diff(raw)).max()) if len(raw) > 1 else 0,
        'path_s_range': [float(path.min()), float(path.max())],
        'max_path_s_step': float(np.abs(np.diff(path)).max()) if len(path) > 1 else 0,
        'slice_start_indices': sorted(set(map(int, idx))),
        'slice_change_count': int(np.count_nonzero(np.diff(idx))),
        'solver_status_counts': {str(status): sum(row['solver_status_code'] == status for row in h[selected]) for status in sorted(set(row['solver_status_code'] for row in h[selected]))},
    }


def analyze_trial(directory):
    history = list(directory.glob('trial_history*'))
    pose = list(directory.glob('pose_sdf*'))
    cycle = list(directory.glob('cc_mf_cycle*'))
    if not (history and pose and cycle):
        return None
    h, p, c = csv_rows(history[0]), csv_rows(pose[0]), csv_rows(cycle[0])
    v, omega, path = series(h, 'v'), series(h, 'omega'), series(p, 'path_s')
    raw, idx = series(p, 'planner_projected_s_raw'), series(p, 'slice_start_idx')
    flips = np.flatnonzero((omega[:-1] * omega[1:] < 0) &
        (np.minimum(abs(omega[:-1]), abs(omega[1:])) >= .05)) + 1
    changes = np.flatnonzero(np.diff(idx)) + 1
    flip_near_slice = [int(i) for i in flips if np.any(abs(changes - i) <= 1)]
    low_speed = [summarize_window(h, p, a, b) for a,b in intervals(abs(v) < .01, 5)]
    count_modes = {mode: sum(row['solve_mode'] == mode for row in c) for mode in sorted(set(row['solve_mode'] for row in c))}
    summary = {
        'steps': len(h), 'solver_modes': count_modes,
        'omega_reversal_count_both_magnitudes_ge_0_05': len(flips),
        'omega_reversal_steps': flips.tolist(),
        'slice_change_steps': changes.tolist(),
        'omega_reversals_within_1_step_of_slice_change': flip_near_slice,
        'max_abs_raw_projection_step': float(np.max(abs(np.diff(raw)))),
        'max_abs_path_s_step': float(np.max(abs(np.diff(path)))),
        'low_speed_intervals_abs_v_lt_0_01_at_least_0_5sec': low_speed,
        'last_4_sec': summarize_window(h, p, max(0, len(h)-41), len(h)-1),
    }
    for w in low_speed:
        a,b=w['step_start'],w['step_end']
        summary.setdefault('low_speed_sample_rows',[]).append([
            {key: h[i][key] for key in ('timestep','time','x','y','theta','v','omega','solver_status_code')}
            | {key:p[i][key] for key in ('path_s','predicted_s','planner_projected_s_raw','slice_start_idx','sdf_value')}
            for i in range(a,b+1) if i in (a,b) or i%5==0
        ])
    return summary


def main():
    result = {'notes': [
        'Analysis reads standard CSVs only; no new simulations.',
        'trial_history applies input at timestep k and logs post-step state at (k+1)dt; pose_sdf logs pre-step state at k*dt.',
        'A reversal counts only consecutive omega commands of opposite sign with both magnitudes >=0.05 rad/s.',
        'Low speed means |v|<0.01 m/s for >=5 samples; it does not mean zero angular motion.',
    ], 'trials':{}}
    for directory in sorted((ROOT / 'results' / 'runs' / 'config01').glob('trial_*/data')):
        summary = analyze_trial(directory)
        if summary is not None:
            result['trials'][directory.parent.name] = summary
    output = ROOT / 'reference_analysis.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    for name, summary in result['trials'].items():
        print(name, json.dumps({k:v for k,v in summary.items() if k not in ('low_speed_sample_rows','omega_reversal_steps','low_speed_intervals_abs_v_lt_0_01_at_least_0_5sec')}, separators=(',',':')))
        print('low speed intervals', [(w['pre_step_time_start'],w['pre_step_time_end'],w['mean_abs_v'],w['omega_reversals_both_magnitudes_ge_0_05'],w['slice_change_count']) for w in summary['low_speed_intervals_abs_v_lt_0_01_at_least_0_5sec']])


if __name__ == '__main__':
    main()
