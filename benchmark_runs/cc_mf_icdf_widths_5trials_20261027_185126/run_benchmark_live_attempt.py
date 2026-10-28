"""Sequential matched-seed width benchmark without editing controller sources."""

import argparse
import csv
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parents[1]
BASE_008 = PROJECT / 'benchmark_runs/maze_rectangle_w008_dd_four_methods_5trials_20261027_181526'
BASE_010 = PROJECT / 'benchmark_runs/cc_mf_imp_maze_rectangle_dd_5trials_20261027_180434'
sys.path.insert(0, str(PROJECT))


def stamp():
    return dt.datetime.now(ZoneInfo('Asia/Seoul')).isoformat()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, default=lambda value: value.tolist()) + '\n')


def child(width):
    import numpy as np
    import torch
    import casadi
    import yaml
    from control.cc_mf_icdf_optimizer import CCMFICDFOptimizerParam
    from models.dd import DifferentialDriveRectangleGeometry
    from sim.start_perturbation import sample_start_pose
    import test_nmpc

    module = importlib.import_module('sim.simulation_mpc')
    output = OUTPUT / ('width_010' if width == .10 else 'width_008')
    assert not (output / 'trial_results.csv').exists(), 'Use a fresh result directory.'
    config = yaml.safe_load((BASE_008 / 'config.yaml').read_text())
    config.pop('cc_sf', None)
    config.pop('cc_ecs', None)
    options = config.pop('cc_mf_imp')
    config['cc_mf_icdf'] = {key: value for key, value in options.items()
                             if not key.startswith('mf_imp_root_')}
    config['test_configs'] = [{
        'maze_type': 'maze', 'robot_shape': 'rectangle',
        'optimizer_type': 'cc_mf_icdf', 'dynamics_type': 'differential_drive',
        'path_planner': 'astar',
    }]
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))

    # Only the simulator's local constructor binding is replaced. Returned
    # objects retain their original type; all controller/model sources stay intact.
    def rectangle_for_run(length, width=None, rear_dist=0.0):
        assert length == .15 and rear_dist == 0.0
        return DifferentialDriveRectangleGeometry(length=length, width=selected_width,
                                                   rear_dist=rear_dist)

    selected_width = width
    module.DifferentialDriveRectangleGeometry = rectangle_for_run
    footprint = rectangle_for_run(.15)
    start, goal, grid, obstacles = module.simulation_mpc().create_env('maze')
    geometry = {
        'bounds': grid[0], 'nominal_start': start, 'goal': goal, 'dt_s': .1,
        'footprint': footprint.equiv_rep()[0].get_ccw_vertices(),
        'obstacles': [obstacle.get_ccw_vertices() for obstacle in obstacles],
    }
    dump(output / 'geometry.json', geometry)
    archive = BASE_010 if width == .10 else BASE_008
    assert json.loads((output / 'geometry.json').read_text()) == json.loads((archive / 'geometry.json').read_text())
    previous_trials = archive / ('baseline/trial_results.csv' if width == .10 else 'trial_results.csv')
    with previous_trials.open(newline='') as stream:
        reference = {int(row['start_seed']): row for row in csv.DictReader(stream)
                     if row['optimizer_type'] == 'cc_mf_imp'}
    starts = []
    for seed in range(5):
        settings = {**config['start_perturbation'], 'seed': seed}
        pose, details = sample_start_pose(start, footprint, obstacles, grid[0], settings)
        expected = [float(reference[seed][key]) for key in ('initial_x', 'initial_y', 'initial_theta')]
        np.testing.assert_allclose(pose, expected, rtol=0, atol=1e-14)
        starts.append({'seed': seed, 'pose': pose, 'reference_pose': expected,
                       'clearance_m': details['initial_clearance']})
    dump(output / 'expected_start_poses.json', starts)

    params = CCMFICDFOptimizerParam()
    module.simulation_mpc._apply_param_overrides(params, config['cc_mf_icdf'], 'CC-MF-ICDF')
    dump(output / 'optimizer_parameters.json', vars(params))
    archived_params = json.loads((BASE_008 / 'optimizer_defaults.json').read_text())['cc_mf_imp']
    current = json.loads((output / 'optimizer_parameters.json').read_text())
    common_differences = {key: [value, current[key]] for key, value in archived_params.items()
                          if key in current and value != current[key]}
    assert not common_differences, common_differences
    assert torch.cuda.is_available() and torch.get_num_threads() == 6
    metadata = {
        'created_at_kst': stamp(), 'width_m': width, 'length_m': .15,
        'python': sys.executable, 'python_version': sys.version,
        'platform': platform.platform(), 'torch': torch.__version__,
        'numpy': np.__version__, 'casadi': casadi.__version__,
        'gpu': torch.cuda.get_device_name(0), 'torch_num_threads': torch.get_num_threads(),
        'thread_environment': {key: os.environ.get(key) for key in
                               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'runtime_geometry_override': 'sim.simulation_mpc.DifferentialDriveRectangleGeometry',
        'reference_geometry_matches': True, 'reference_start_poses_match': True,
        'reference_common_parameter_differences': common_differences,
    }
    dump(output / 'environment.json', metadata)
    args = test_nmpc.build_parser().parse_args([
        '-c', str(output / 'config.yaml'), '--trials', '5',
        '--results-dir', str(output), '--no-animation', '--no-plots',
    ])
    begun = time.perf_counter()
    try:
        test_nmpc.run_trial_batch(args, config=config)
    finally:
        metadata.update(finished_at_kst=stamp(), wall_time_s=time.perf_counter() - begun)
        dump(output / 'environment.json', metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=float, choices=[.10, .08])
    args = parser.parse_args()
    os.chdir(PROJECT)
    if args.width is not None:
        child(args.width)
        return
    environment = dict(os.environ)
    environment.update(ACADOS_SOURCE_DIR='/home/taek111/projects/acados',
                       LD_LIBRARY_PATH='/home/taek111/projects/acados/lib',
                       MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment.pop(key, None)
    hashes = {}
    sources = [PROJECT / 'test_nmpc.py']
    for folder in ('control', 'models', 'sim', 'planning', 'utils'):
        sources.extend(sorted((PROJECT / folder).rglob('*.py')))
    for source in sources:
        relative = source.relative_to(PROJECT)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        target = OUTPUT / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    dump(OUTPUT / 'source_hashes.json', hashes)
    for width, name in ((.10, 'width_010'), (.08, 'width_008')):
        print(f'Starting {name}: 5 trials, seeds 0-4', flush=True)
        with (OUTPUT / name / 'run.log').open('w') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--width', str(width)],
                           cwd=PROJECT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        print(f'Finished {name}', flush=True)
    changed = [name for name, digest in hashes.items()
               if hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() != digest]
    dump(OUTPUT / 'source_integrity.json', {'sources_changed_during_run': changed})
    assert not changed, changed


if __name__ == '__main__':
    main()
