"""Run five CC-SF trials with passive per-cycle solver/stop instrumentation.

Usage: <pp-python> run_experiment.py [fresh-results-directory]
The launch environment must include the acados shared-library directory.
Instrumentation delegates to the original methods without modifying decisions.
"""
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path('/home/taek111/projects/psdf_mpcc')
HERE = Path(__file__).resolve().parent
OUTPUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE
OUTPUT.mkdir(parents=True, exist_ok=True)
if (OUTPUT / 'trial_results.csv').exists() or (OUTPUT / 'solver_cycles.csv').exists():
    raise FileExistsError('Select a fresh results directory for each run.')
if OUTPUT != HERE:
    shutil.copy2(__file__, OUTPUT / 'run_experiment.py')
    shutil.copy2(HERE / 'config.yaml', OUTPUT / 'config.yaml')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['MPLBACKEND'] = 'Agg'
ANIMATIONS = ROOT / 'animation' / 'cc_sf'
ANIMATIONS.mkdir(parents=True, exist_ok=True)
STAMP = dt.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
print(f'RESULTS_DIR={OUTPUT}', flush=True)
log = (OUTPUT / 'run.log').open('w', buffering=1)
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)

import casadi
import numpy as np
import torch
import yaml
import test_nmpc
from control.cc_sf_optimizer import CCSFOptimizer, CCSFOptimizerParam
from sim.simulation_mpc import simulation_mpc


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


sources = [ROOT / 'test_nmpc.py']
for directory in ('control', 'models', 'planning', 'sim', 'utils'):
    sources.extend(sorted((ROOT / directory).rglob('*.py')))
hashes = {}
for source in sources:
    relative = source.relative_to(ROOT)
    hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = OUTPUT / 'source_snapshot' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
metadata = {
    'started_at_kst': dt.datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
    'python': sys.executable, 'python_version': sys.version,
    'platform': platform.platform(), 'numpy': np.__version__,
    'casadi': casadi.__version__, 'torch': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'source_sha256': hashes,
    'animation_directory': str(ANIMATIONS),
    'animation_frame_skip': 1,
    'thread_environment': {key: os.environ.get(key) for key in (
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
}
(OUTPUT / 'environment.json').write_text(json.dumps(metadata, indent=2))
(OUTPUT / 'optimizer_defaults.json').write_text(
    json.dumps(vars(CCSFOptimizerParam()), indent=2, default=serialize))

cycle_fields = [
    'trial_number', 'start_seed', 'timestep', 'time_before_s', 'time_after_s',
    'main_status', 'backup_status', 'final_status', 'main_object_status_after',
    'solve_mode', 'safe_stop', 'safe_stop_counter', 'v', 'omega', 'virtual_input',
    'solver_time_s', 'error',
]
cycle_file = (OUTPUT / 'solver_cycles.csv').open('w', newline='', buffering=1)
cycle_writer = csv.DictWriter(cycle_file, fieldnames=cycle_fields)
cycle_writer.writeheader()
active = {}
summaries = []
original_prepare = CCSFOptimizer._prepare_and_solve
original_solve = CCSFOptimizer.solve_nlp
original_run_single = test_nmpc.run_single_test


def measured_prepare(self, solver):
    role = 'main' if solver is self.solver else 'backup'
    try:
        status = original_prepare(self, solver)
    except Exception as exc:
        active['calls'].append({'role': role, 'status': None, 'error': repr(exc)})
        raise
    active['calls'].append({'role': role, 'status': int(status)})
    return status


def measured_solve(self):
    active['calls'] = []
    step = len(active['cycles'])
    time_before = float(self._system._time)
    row = {
        'trial_number': active['trial_number'], 'start_seed': active['seed'],
        'timestep': step, 'time_before_s': time_before,
        'time_after_s': time_before + float(self._system._dt),
    }
    counter_before = int(self._safe_stop_count)
    solution = None
    try:
        solution = original_solve(self)
        inputs = np.asarray(solution.solver.get(0, 'u')).reshape(-1)
        row.update({
            'final_status': int(solution.solver.status),
            'solve_mode': self._last_solve_mode,
            'v': float(inputs[0]), 'omega': float(inputs[1]),
            'virtual_input': float(inputs[2]) if inputs.size > 2 else None,
        })
        return solution
    except Exception as exc:
        row.update({'solve_mode': 'exception', 'error': repr(exc)})
        raise
    finally:
        for call in active['calls']:
            row[f"{call['role']}_status"] = call['status']
        row.update({
            'main_object_status_after': int(self.solver.status),
            'safe_stop': int(self._safe_stop_count > counter_before),
            'safe_stop_counter': int(self._safe_stop_count),
            'solver_time_s': float(self._cycle_solver_time),
        })
        active['cycles'].append(row)
        cycle_writer.writerow(row)
        cycle_file.flush()


def animation_path(self, animation_name, method_name=None):
    return str(ANIMATIONS / (
        f'{animation_name}_trial{active["trial_number"]:02d}_{STAMP}.mp4'))


def generate_animation(test_sim, args, interrupted=False):
    if not args.generate_animation or not test_nmpc.has_animation_data(test_sim):
        return
    name = test_sim.current_name + ('_interrupted' if interrupted else '')
    print(f'Generating animation for trial {active["trial_number"]}...', flush=True)
    active['animation'] = test_sim.animate_world(
        test_sim.sim, animation_name=name, maze_type=args.maze_type,
        frame_skip=args.frame_skip, method_name=args.optimizer_type,
        return_html=False,
    )


def measured_run_single(args, config=None, *, collect_metrics=False):
    active.clear()
    active.update({
        'trial_number': len(summaries) + 1, 'seed': args.start_seed,
        'cycles': [], 'calls': [], 'animation': None,
    })
    result = original_run_single(args, config, collect_metrics=collect_metrics)
    cycles = active['cycles']
    stops = [row for row in cycles if row['safe_stop']]
    episodes = sum(
        bool(row['safe_stop']) and (i == 0 or not cycles[i - 1]['safe_stop'])
        for i, row in enumerate(cycles)
    )
    runtime_count = test_nmpc.get_safe_stop_count(result)
    summary = {
        'trial_number': active['trial_number'], 'start_seed': active['seed'],
        'status': result.get('status'), 'failure_reason': result.get('failure_reason'),
        'final_time_s': result.get('final_time'), 'total_cycles': len(cycles),
        'main_nonzero_status_count': sum(row.get('main_status') not in (None, 0) for row in cycles),
        'backup_recovery_count': sum(row.get('solve_mode') == 'backup_feasible_qp' for row in cycles),
        'safe_stop_count': len(stops), 'safe_stop_episodes': episodes,
        'safe_stop_duration_s': sum(row['time_after_s'] - row['time_before_s'] for row in stops),
        'safe_stop_status_zero_count': sum(row.get('final_status') == 0 for row in stops),
        'final_status_zero_count': sum(row.get('final_status') == 0 for row in cycles),
        'safe_stop_status_codes': json.dumps(sorted({row.get('final_status') for row in stops})),
        'runtime_safe_stop_count': runtime_count,
        'counter_matches': runtime_count == len(stops),
        'safe_stop_all_inputs_zero': all(
            row.get('v') == 0.0 and row.get('omega') == 0.0
            and row.get('virtual_input') in (None, 0.0) for row in stops),
        'animation_path': active['animation'],
    }
    summaries.append(summary)
    with (OUTPUT / 'safe_stop_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerows(summaries)
    (Path(result['output_root_dir']) / 'result.json').write_text(json.dumps(
        {key: value for key, value in result.items() if key != 'successful_solver_steps'},
        indent=2, default=serialize))
    print('SAFE_STOP_RESULT ' + json.dumps(summary), flush=True)
    return result


CCSFOptimizer._prepare_and_solve = measured_prepare
CCSFOptimizer.solve_nlp = measured_solve
simulation_mpc._get_animation_output_path = animation_path
test_nmpc.generate_animation_if_possible = generate_animation
test_nmpc.run_single_test = measured_run_single
args = test_nmpc.build_parser().parse_args([
    '-c', str(OUTPUT / 'config.yaml'), '--trials', '5',
    '--results-dir', str(OUTPUT), '--no-plots',
])
config = test_nmpc.load_runtime_config(args.config)
started = time.perf_counter()
try:
    test_nmpc.run_trial_batch(args, config=config)
finally:
    cycle_file.close()
    metadata['finished_at_kst'] = dt.datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    metadata['wall_time_s'] = time.perf_counter() - started
    metadata['sources_changed_during_run'] = [
        name for name, digest in hashes.items()
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    (OUTPUT / 'environment.json').write_text(json.dumps(metadata, indent=2))
    print('Batch finished:', OUTPUT, flush=True)
