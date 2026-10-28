"""Rebuild benchmark CSVs from retained raw logs and frozen scene geometry.

Run with the experiment's Python interpreter after all 20 trials finish.
Only numpy is required; the clearance helper is loaded from the source snapshot.
"""

import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
METHODS = ('mpcc', 'cc_sf', 'cc_mf', 'cc_ecs')
ID_FIELDS = ('configuration_id', 'optimizer_type', 'trial_number', 'start_seed')


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    assert rows, f'No records for {path}'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def statistics(values, prefix):
    values = np.asarray(values, dtype=float)
    assert np.all(np.isfinite(values)), f'Invalid values for {prefix}'
    return {
        f'{prefix}_count': len(values),
        f'{prefix}_mean': float(np.mean(values)) if len(values) else '',
        f'{prefix}_min': float(np.min(values)) if len(values) else '',
        f'{prefix}_p05': float(np.percentile(values, 5)) if len(values) else '',
        f'{prefix}_p95': float(np.percentile(values, 95)) if len(values) else '',
        f'{prefix}_max': float(np.max(values)) if len(values) else '',
    }


def polygons_overlap(first, second):
    """Strict interior overlap via separating axes; contact alone is false."""
    edges = np.concatenate((np.roll(first, -1, axis=0) - first,
                            np.roll(second, -1, axis=0) - second))
    axes = np.column_stack((-edges[:, 1], edges[:, 0]))
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > 1e-12] / lengths[lengths > 1e-12, None]
    a, b = first @ axes.T, second @ axes.T
    return bool(np.all(np.minimum(a.max(axis=0), b.max(axis=0))
                       - np.maximum(a.min(axis=0), b.min(axis=0)) > 1e-10))


def main():
    helper_path = ROOT / 'source_snapshot/sim/start_perturbation.py'
    spec = importlib.util.spec_from_file_location('frozen_start_perturbation', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    geometry = json.loads((ROOT / 'geometry.json').read_text())
    footprint = np.asarray(geometry['footprint'])
    obstacles = [np.asarray(polygon) for polygon in geometry['obstacles']]
    bounds = np.asarray(geometry['bounds'])
    dt = geometry['dt_s']
    trials = read_csv(ROOT / 'trial_results.csv')
    assert len(trials) == 20, f'Expected 20 finished trials; got {len(trials)}'
    assert {row['optimizer_type'] for row in trials} == set(METHODS)
    for method in METHODS:
        subset = [row for row in trials if row['optimizer_type'] == method]
        assert len(subset) == 5
        assert {int(row['start_seed']) for row in subset} == set(range(5))
    for seed in range(5):
        poses = [[float(row[key]) for key in ('initial_x', 'initial_y', 'initial_theta')]
                 for row in trials if int(row['start_seed']) == seed]
        assert np.allclose(poses, poses[0], rtol=0, atol=1e-14), seed

    per_trial, steps, clearance_samples = [], [], []
    groups = {method: {'clearance': [], 'solver': [], 'controller': [],
                      'controller_steady': [], 'traversal': [], 'rows': [],
                      'overlap_samples': 0, 'boundary_violation_samples': 0}
              for method in METHODS}
    raw_summary = {row['optimizer_type']: row for row in read_csv(ROOT / 'summary.csv')}
    for trial in trials:
        identity = {key: trial[key] for key in ID_FIELDS}
        method = trial['optimizer_type']
        group = groups[method]
        assert trial['status'] in ('success', 'failure', 'timeout'), trial
        run_dir = ROOT / 'runs' / trial['configuration_id'] / f"trial_{int(trial['trial_number']):04d}" / 'data'
        history = read_csv(run_dir / f"trial_history_{trial['current_name']}.csv")
        pose_log = read_csv(run_dir / f"pose_sdf_{trial['current_name']}.csv")
        start_info = json.loads((run_dir / f"start_pose_{trial['current_name']}.json").read_text())
        assert len(history) == len(pose_log) > 0
        assert len(history) == round(float(trial['final_time']) / dt)
        initial_pose = [float(trial[key]) for key in ('initial_x', 'initial_y', 'initial_theta')]
        poses = [initial_pose] + [[float(row[key]) for key in ('x', 'y', 'theta')]
                                  for row in history]
        clearances, overlap_count, boundary_count = [], 0, 0
        for sample_index, pose in enumerate(poses):
            pose = np.asarray(pose)
            clearance = helper.pose_clearance(pose, [footprint], obstacles, bounds)
            cosine, sine = np.cos(pose[2]), np.sin(pose[2])
            world = footprint @ np.array([[cosine, sine], [-sine, cosine]]) + pose[:2]
            boundary_clearance = float(min(np.min(world - bounds[0]), np.min(bounds[1] - world)))
            overlap = any(polygons_overlap(world, obstacle) for obstacle in obstacles) if clearance <= 1e-10 else False
            overlap_count += int(overlap)
            boundary_count += int(boundary_clearance < -1e-10)
            clearances.append(clearance)
            clearance_samples.append({
                **identity, 'sample_index': sample_index,
                'simulation_time_s': sample_index * dt,
                'x_m': pose[0], 'y_m': pose[1], 'theta_rad': pose[2],
                'clearance_m': clearance, 'boundary_clearance_m': boundary_clearance,
                'obstacle_overlap': overlap,
            })
        assert math.isclose(clearances[0], start_info['initial_clearance'], abs_tol=1e-12)
        successful_times, controller_times = [], []
        for index, row in enumerate(history):
            assert int(row['timestep']) == index
            assert math.isclose(float(row['time']), (index + 1) * dt, abs_tol=1e-9)
            assert np.allclose([float(pose_log[index][key]) for key in ('x', 'y', 'theta')],
                               poses[index], rtol=0, atol=1e-12)
            solver_time, controller_time = (float(row[key]) for key in
                                           ('solver_computation_time', 'controller_computation_time'))
            assert math.isfinite(solver_time) and solver_time >= 0
            assert math.isfinite(controller_time) and controller_time >= solver_time
            success = row['solver_success'] == 'True'
            if success:
                successful_times.append(solver_time)
            controller_times.append(controller_time)
            psdf = float(pose_log[index]['sdf_value'])
            steps.append({
                **identity, 'trial_status': trial['status'], 'timestep': index,
                'simulation_time_start_s': index * dt,
                'simulation_time_end_s': (index + 1) * dt,
                'clearance_start_m': clearances[index], 'clearance_end_m': clearances[index + 1],
                'logged_psdf_start_m': psdf if math.isfinite(psdf) else '',
                'solver_computation_time_s': solver_time,
                'controller_computation_time_s': controller_time,
                'first_step_includes_initialization': index == 0,
                'solver_success': success,
                'solver_raw_status': row['solver_raw_status'],
                'solver_status_code': row['solver_status_code'],
            })
        assert len(successful_times) == int(trial['successful_solver_steps'])
        if successful_times:
            assert math.isclose(np.mean(successful_times), float(trial['solver_time_mean_s']), rel_tol=1e-12)
            assert math.isclose(np.percentile(successful_times, 95), float(trial['solver_time_p95_s']), rel_tol=1e-12)
        reached = trial['status'] == 'success'
        traversal = float(trial['final_time']) if reached else ''
        summary = {
            **identity, 'status': trial['status'], 'failure_reason': trial['failure_reason'],
            'success': int(reached), 'traversal_time_s': traversal,
            'elapsed_simulation_time_s': float(trial['final_time']),
            'control_steps': len(history), 'unsuccessful_control_steps': len(history) - len(successful_times),
            **statistics(clearances, 'clearance_m'),
            'obstacle_overlap_samples': overlap_count, 'boundary_violation_samples': boundary_count,
            **statistics(successful_times, 'solver_success_time_s'),
            **statistics(controller_times, 'controller_all_time_s'),
            **statistics(controller_times[1:], 'controller_steady_time_s'),
            'first_controller_time_s': controller_times[0],
        }
        per_trial.append(summary)
        group['rows'].append(summary)
        group['clearance'].extend(clearances)
        group['solver'].extend(successful_times)
        group['controller'].extend(controller_times)
        group['controller_steady'].extend(controller_times[1:])
        group['overlap_samples'] += overlap_count
        group['boundary_violation_samples'] += boundary_count
        if reached:
            group['traversal'].append(traversal)
    summaries = []
    for method, group in groups.items():
        original = raw_summary[method]
        assert int(original['successful_solver_steps']) == len(group['solver'])
        assert math.isclose(float(original['solver_time_mean_s']), np.mean(group['solver']), rel_tol=1e-12)
        assert math.isclose(float(original['solver_time_p95_s']), np.percentile(group['solver'], 95), rel_tol=1e-12)
        timing_path, = ROOT.glob(f'computation_times_*_{method}_*.csv')
        timing_rows = read_csv(timing_path)
        assert len(timing_rows) == len(group['solver'])
        assert np.allclose([float(row['solver_computation_time_s']) for row in timing_rows],
                           group['solver'], rtol=0, atol=1e-15)
        traversal = group['traversal']
        summaries.append({
            'optimizer_type': method, 'trials': len(group['rows']),
            'successful_trials': len(traversal), 'success_rate': len(traversal) / len(group['rows']),
            'traversal_time_mean_s': float(np.mean(traversal)) if traversal else '',
            'traversal_time_std_s': float(np.std(traversal, ddof=1)) if len(traversal) > 1 else '',
            'traversal_time_min_s': min(traversal) if traversal else '',
            'traversal_time_max_s': max(traversal) if traversal else '',
            **statistics(group['clearance'], 'clearance_m'),
            'obstacle_overlap_samples': group['overlap_samples'],
            'boundary_violation_samples': group['boundary_violation_samples'],
            **statistics(group['solver'], 'solver_success_time_s'),
            **statistics(group['controller'], 'controller_all_time_s'),
            **statistics(group['controller_steady'], 'controller_steady_time_s'),
        })
    write_csv(ROOT / 'benchmark_summary.csv', summaries)
    comparison = [{
        'method': row['optimizer_type'],
        'successes': row['successful_trials'], 'trials': row['trials'],
        'clearance_mean_mm': row['clearance_m_mean'] * 1000,
        'clearance_min_mm': row['clearance_m_min'] * 1000,
        'clearance_p05_mm': row['clearance_m_p05'] * 1000,
        'traversal_mean_s': row['traversal_time_mean_s'],
        'traversal_std_s': row['traversal_time_std_s'],
        'solver_mean_ms_per_step': row['solver_success_time_s_mean'] * 1000,
        'solver_p95_ms_per_step': row['solver_success_time_s_p95'] * 1000,
        'controller_steady_mean_ms_per_step': row['controller_steady_time_s_mean'] * 1000,
        'controller_steady_p95_ms_per_step': row['controller_steady_time_s_p95'] * 1000,
        'controller_all_mean_ms_per_step': row['controller_all_time_s_mean'] * 1000,
    } for row in summaries]
    write_csv(ROOT / 'comparison.csv', comparison)
    write_csv(ROOT / 'benchmark_trials.csv', per_trial)
    write_csv(ROOT / 'step_metrics.csv', steps)
    write_csv(ROOT / 'clearance_samples.csv', clearance_samples)
    definitions = {
        'clearance_m': 'Minimum exact Euclidean footprint-to-obstacle distance and signed map-boundary margin, using frozen sim/start_perturbation.py. Obstacle overlap/contact maps to zero; negative values indicate map-boundary violation. Obstacle interior overlap is recorded separately.',
        'clearance_sampling': 'Initial true pose plus every post-integration true pose, dt=0.1 s. Includes failed trials and final poses. Does not certify clearance between samples.',
        'traversal_time_s': 'Simulated time to the existing goal-reaching criterion, only for success trials. Empty for failures/timeouts; elapsed_simulation_time_s remains available.',
        'solver_success_time_s': 'perf_counter wall time inside solve() only, summed over main and recovery calls per step. Successful control cycles only, including first steps and steps from failed trials. Setup, parameter updates and planning excluded.',
        'controller_all_time_s': 'All logged controller calls, including unsuccessful cycles and each trial\'s first initialization/code-generation/compilation step. Local/global planning excluded.',
        'controller_steady_time_s': 'Controller calls with timestep>0 for each trial; includes failed control cycles. Only the first step of each trial is excluded.',
        'aggregation': 'Clearance and computation statistics pool individual samples/steps; traversal statistics average successful trials. std is sample standard deviation (ddof=1). All CSV times are seconds and distances are meters.',
    }
    (ROOT / 'metric_definitions.json').write_text(json.dumps(definitions, indent=2))
    validation = {
        'trial_count': len(trials), 'matched_seeds_and_initial_poses': True,
        'all_histories_complete': True, 'timing_statistics_match_raw_summary': True,
        'clearance_sample_count': len(clearance_samples), 'control_step_count': len(steps),
        'sources_changed_during_run': json.loads((ROOT / 'environment.json').read_text()).get('sources_changed_during_run'),
    }
    assert validation['sources_changed_during_run'] == [], validation
    (ROOT / 'validation.json').write_text(json.dumps(validation, indent=2))
    for row in summaries:
        print(json.dumps(row))


if __name__ == '__main__':
    main()
