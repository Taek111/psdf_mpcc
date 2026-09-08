"""Reproduce one controller run, preserving partial results on solver failure.

Run from WSL with the project's ped Python and acados libraries on LD_LIBRARY_PATH:
    python benchmark_runs/navigation_comparison_20260907_seed42/run_one.py psdf
Choices: psdf, obca, dcbf. Each uses a separate directory for acados code generation.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from sim.simulation_mpc import simulation_mpc
from sim.simulation import Robot
from sim.start_perturbation import pose_clearance

parser = argparse.ArgumentParser()
parser.add_argument('method', choices=['dcbf', 'obca', 'psdf'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--duration', type=float, default=60.0)
parser.add_argument('--wall-limit', type=int, default=600)
args = parser.parse_args()
run_dir = Path(__file__).resolve().parent / args.method
run_dir.mkdir(parents=True, exist_ok=True)
os.chdir(run_dir)

def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def save(name, value):
    (run_dir / name).write_text(json.dumps(clean(value), indent=2, allow_nan=False))

sim = simulation_mpc()
config = {
    'output_root_dir': str(run_dir),
    'start_perturbation': {'enabled': True, 'seed': args.seed},
    'localization_error': {'enabled': False},
}
save('config.json', {**config, 'method': args.method, 'duration': args.duration,
                     'robot_shape': 'rectangle', 'maze_type': 'maze',
                     'dynamics_type': 'differential_drive', 'path_planner': 'astar'})
started = time.perf_counter()
original_controller = Robot.run_controller
original_system = Robot.run_system

def checkpoint(robot, phase):
    save('checkpoint.json', {
        'phase': phase, 'wall_seconds': time.perf_counter() - started,
        'simulation_time': robot._system._time,
        'state': robot._system.get_state(),
        'previous_applied_input': robot._system._state._u,
        'solver_status': robot._controller.get_last_solver_status_info(),
    })

def run_controller(robot, obstacles):
    checkpoint(robot, 'solving')
    return original_controller(robot, obstacles)

def run_system(robot):
    result = original_system(robot)
    checkpoint(robot, 'step_complete')
    return result

Robot.run_controller = run_controller
Robot.run_system = run_system

def wall_timeout(signum, frame):
    raise TimeoutError(f'Wall-clock budget of {args.wall_limit} seconds exceeded')

signal.signal(signal.SIGALRM, wall_timeout)
signal.alarm(args.wall_limit)
error = None
try:
    sim.mpc_test('maze', 'rectangle', args.method, 'differential_drive', 'astar',
                 args.duration, config)
except (Exception, KeyboardInterrupt) as exc:
    error = f'{type(exc).__name__}: {exc}'
    traceback.print_exc()
finally:
    signal.alarm(0)
    Robot.run_controller = original_controller
    Robot.run_system = original_system

result = dict(sim.last_run_outcome or {})
result.update(method=args.method, wall_seconds=time.perf_counter() - started,
              error=error, start_perturbation=sim.start_perturbation_info)
robot = sim.robot
if robot is not None:
    result['final_time'] = robot._system._time
    result['final_pose'] = robot._system.get_state().copy()
    goal = sim.sim._goal_position if sim.sim is not None else sim.create_env('maze')[1]
    result['goal_pose'] = goal
    result['distance_to_environment_goal'] = float(np.linalg.norm(result['final_pose'][:2] - goal[:2]))
    result['optimizer_class'] = type(robot._controller._optimizer).__name__
    params = robot._controller._param
    result['parameters'] = {k: v for k, v in vars(params).items()
                            if isinstance(v, (int, float, str, bool, np.ndarray))}
    states = np.vstack([sim.initial_pose, *robot._system_logger._xs])
    inputs = np.asarray(robot._system_logger._us).reshape(-1, 2)
    obstacles = [o.get_ccw_vertices() for o in sim.sim._obstacles]
    footprints = [g.get_ccw_vertices() for g in robot._system._geometry.equiv_rep()]
    bounds = sim.create_env('maze')[2][0]
    clearances = [pose_clearance(x, footprints, obstacles, bounds) for x in states]
    result['completed_steps'] = len(inputs)
    result['minimum_sampled_clearance'] = min(clearances)
    result['minimum_clearance_time'] = int(np.argmin(clearances)) * .1
    result['contact_or_collision_samples'] = sum(c <= 1e-9 for c in clearances)
    result['first_input'] = inputs[0] if len(inputs) else None
    result['max_abs_inputs'] = np.max(np.abs(inputs), axis=0) if len(inputs) else None
    result['solver_statuses'] = robot._controller_logger._solver_status_infos
    result['solver_failure_steps'] = sum(s.get('success') is False for s in result['solver_statuses'])
    np.savez_compressed(run_dir / 'trajectory.npz', states=states, inputs=inputs,
                        clearance=np.asarray(clearances), times=np.arange(len(states)) * .1)
    optimizer = robot._controller._optimizer
    if error and getattr(optimizer, 'opti', None) is not None:
        try:
            result['ipopt_stats'] = optimizer.opti.stats()
            optimizer.opti.debug.show_infeasibilities(1e-5)
        except Exception as diagnostics_error:
            result['diagnostics_error'] = str(diagnostics_error)
    if error:
        result['status'] = 'error'
        result['failure_reason'] = error
    sim.save_trial_history_to_csv(f'trial_history_{sim.current_name}')
    save('result.json', result)
    try:
        result['animation'] = sim.animate_world(
            sim.sim, animation_name=sim.current_name + ('_failed' if error else ''),
            maze_type='maze', frame_skip=1, method_name=args.method,
            return_html=False, include_initial_pose=True,
        )
    except Exception as animation_error:
        result['animation_error'] = f'{type(animation_error).__name__}: {animation_error}'
        traceback.print_exc()
    if hasattr(optimizer, 'cleanup'):
        optimizer.cleanup()
else:
    result.update(status='error', failure_reason=error)
save('result.json', result)
save('checkpoint.json', {'phase': 'finished', 'status': result.get('status'),
                         'animation': result.get('animation'), 'error': error})
print(json.dumps(clean(result), indent=2), flush=True)
