"""Reproduce the existing benchmark table's metrics and validate the samples."""

import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
DATA = ROOT / 'frozen'


def rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def save_csv(path, records):
    assert records, path
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def distribution(values):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    result = {'count': len(values), 'mean': float(np.mean(values)),
              'p05': float(np.percentile(values, 5)),
              'p95': float(np.percentile(values, 95)), 'min': float(np.min(values))}
    assert math.isclose(result['mean'], math.fsum(values) / len(values), abs_tol=1e-13)
    ordered = sorted(values)
    for percentile in (5, 95):
        rank = percentile / 100 * (len(values) - 1)
        low, high = math.floor(rank), math.ceil(rank)
        independent = ordered[low] + (rank - low) * (ordered[high] - ordered[low])
        assert math.isclose(result[f'p{percentile:02}'], independent, abs_tol=1e-13)
    return result


def main():
    helper_path = ROOT / 'source_snapshot/sim/start_perturbation.py'
    spec = importlib.util.spec_from_file_location('frozen_clearance', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    archive = PROJECT / 'benchmark_runs/maze_rectangle_w008_dd_four_methods_5trials_20261027_181526'
    assert helper_path.read_bytes() == (archive / 'source_snapshot/sim/start_perturbation.py').read_bytes()
    summaries, trial_metrics, samples, timings, checks, source_hashes = [], [], [], [], [], {}
    for name, width in (('width_010', .10), ('width_008', .08)):
        folder = DATA / name
        geometry = json.loads((folder / 'geometry.json').read_text())
        footprint = np.asarray(geometry['footprint'])
        obstacles = [np.asarray(vertices) for vertices in geometry['obstacles']]
        bounds = geometry['bounds']
        np.testing.assert_allclose(np.ptp(footprint, axis=0), [.15, width], rtol=0, atol=1e-14)
        expected = {entry['seed']: entry for entry in json.loads((folder / 'expected_start_poses.json').read_text())}
        trials = rows(folder / 'trial_results.csv')
        assert len(trials) == 5 and {int(t['start_seed']) for t in trials} == set(range(5))
        group_clearance, group_solver, group_controller, group_traversal = [], [], [], []
        group_modes = Counter()
        for trial in trials:
            assert (trial['maze_type'], trial['optimizer_type'], trial['robot_shape'],
                    trial['dynamics_type'], trial['path_planner']) == (
                        'maze', 'cc_mf_icdf', 'rectangle', 'differential_drive', 'astar')
            assert trial['status'] in ('success', 'failure', 'timeout'), trial
            seed = int(trial['start_seed'])
            pose0 = [float(trial[key]) for key in ('initial_x', 'initial_y', 'initial_theta')]
            np.testing.assert_allclose(pose0, expected[seed]['reference_pose'], rtol=0, atol=1e-14)
            data = folder / 'runs' / trial['configuration_id'] / f"trial_{int(trial['trial_number']):04d}" / 'data'
            history_path = data / f"trial_history_{trial['current_name']}.csv"
            history = rows(history_path)
            psdf = rows(data / f"pose_sdf_{trial['current_name']}.csv")
            cycles = rows(data / f"cc_mf_icdf_cycle_{trial['current_name']}.csv")
            assert len(history) == len(psdf) == len(cycles) == round(float(trial['final_time']) / .1)
            source_hashes[str(history_path.relative_to(ROOT))] = hashlib.sha256(history_path.read_bytes()).hexdigest()
            poses = [pose0] + [[float(row[key]) for key in ('x', 'y', 'theta')] for row in history]
            clearance = []
            for sample_index, pose in enumerate(poses):
                value = helper.pose_clearance(pose, [footprint], obstacles, bounds)
                clearance.append(value)
                samples.append({'width_m': width, 'seed': seed, 'sample_index': sample_index,
                                'time_s': .1 * sample_index, 'x': pose[0], 'y': pose[1],
                                'theta': pose[2], 'clearance_m': value})
            assert math.isclose(clearance[0], expected[seed]['clearance_m'], abs_tol=1e-12)
            solver, controller = [], []
            for step, row in enumerate(history):
                assert int(row['timestep']) == step
                assert math.isclose(float(row['time']), .1 * (step + 1), abs_tol=1e-9)
                np.testing.assert_allclose([float(psdf[step][key]) for key in ('x', 'y', 'theta')],
                                           poses[step], rtol=0, atol=1e-12)
                solve_s = float(row['solver_computation_time'])
                control_s = float(row['controller_computation_time'])
                assert math.isfinite(solve_s) and 0 <= solve_s <= control_s and math.isfinite(control_s)
                successful = row['solver_success'] == 'True'
                if successful:
                    solver.append(solve_s)
                controller.append(control_s)
                timings.append({'width_m': width, 'seed': seed, 'timestep': step,
                                'solver_success': successful, 'solver_time_s': solve_s,
                                'controller_time_s': control_s, 'first_step': step == 0})
            assert len(solver) == int(trial['successful_solver_steps'])
            modes = Counter(row['solve_mode'] for row in cycles)
            group_modes.update(modes)
            cs, ss, ct = distribution(clearance), distribution(solver), distribution(controller[1:])
            assert math.isclose(ss['mean'], float(trial['solver_time_mean_s']), rel_tol=1e-12)
            assert math.isclose(ss['p95'], float(trial['solver_time_p95_s']), rel_tol=1e-12)
            trial_metrics.append({
                'width_m': width, 'seed': seed, 'status': trial['status'],
                'final_time_s': float(trial['final_time']), 'control_steps': len(history),
                'successful_solver_steps': len(solver), 'clearance_samples': len(clearance),
                'clearance_mean_mm': cs['mean'] * 1000, 'clearance_p05_mm': cs['p05'] * 1000,
                'clearance_min_mm': cs['min'] * 1000, 'contact_or_boundary_violation_samples': sum(v <= 0 for v in clearance),
                'solver_mean_ms': ss['mean'] * 1000, 'solver_p95_ms': ss['p95'] * 1000,
                'controller_mean_ms': ct['mean'] * 1000, 'controller_p95_ms': ct['p95'] * 1000,
                'safe_stop_count': sum(count for mode, count in modes.items() if mode.startswith('safe_stop')),
            })
            group_clearance.extend(clearance)
            group_solver.extend(solver)
            group_controller.extend(controller[1:])
            if trial['status'] == 'success':
                group_traversal.append(float(trial['final_time']))

        original = rows(folder / 'summary.csv')[0]
        successful = rows(next(folder.glob('computation_times_*.csv')))
        recorded_solver = [float(row['solver_computation_time_s']) for row in successful]
        np.testing.assert_allclose(group_solver, recorded_solver, rtol=0, atol=0)
        cs, ss, ct = distribution(group_clearance), distribution(group_solver), distribution(group_controller)
        assert int(original['successful_solver_steps']) == ss['count']
        assert math.isclose(ss['mean'], float(original['solver_time_mean_s']), rel_tol=1e-12)
        assert math.isclose(ss['p95'], float(original['solver_time_p95_s']), rel_tol=1e-12)
        assert int(original['successful_trials']) == len(group_traversal)
        summaries.append({
            'width_m': width, 'controller': 'CC-MF-ICDF', 'trials': 5,
            'successes': len(group_traversal), 'success_rate': len(group_traversal) / 5,
            'clearance_mean_mm': cs['mean'] * 1000, 'clearance_p05_mm': cs['p05'] * 1000,
            'clearance_min_mm': cs['min'] * 1000, 'clearance_samples': cs['count'],
            'solver_mean_ms': ss['mean'] * 1000, 'solver_p95_ms': ss['p95'] * 1000,
            'successful_solver_steps': ss['count'],
            'traversal_mean_s': float(np.mean(group_traversal)) if group_traversal else None,
            'controller_mean_ms': ct['mean'] * 1000, 'controller_p95_ms': ct['p95'] * 1000,
            'controller_steps_excluding_first': ct['count'],
            'contact_or_boundary_violation_samples': sum(v <= 0 for v in group_clearance),
            'safe_stop_count': sum(count for mode, count in group_modes.items() if mode.startswith('safe_stop')),
        })
        checks.append({'width_m': width, 'same_geometry_and_seed_poses_as_reference': True,
                       'solver_csv_and_summary_match': True, 'initial_and_final_pose_included': True,
                       'controller_first_step_excluded_per_trial': True,
                       'quantile_and_mean_independently_checked': True, 'solve_modes': dict(group_modes)})
    save_csv(ROOT / 'benchmark_summary.csv', summaries)
    save_csv(ROOT / 'benchmark_trials.csv', trial_metrics)
    save_csv(ROOT / 'clearance_samples.csv', samples)
    save_csv(ROOT / 'step_metrics.csv', timings)
    validation = {'checks': checks, 'source_history_sha256': source_hashes,
                  'metric_definition': 'Same as benchmark_results.md: pooled exact polygon/boundary clearance at initial and every integrated pose, successful solver calls including first, controller calls excluding only each trial first, traversal over successful trials.'}
    (ROOT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
