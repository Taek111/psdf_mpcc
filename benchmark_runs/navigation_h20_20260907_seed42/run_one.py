"""Reproduce one controller run, preserving partial results on solver failure.

Run from WSL with the project's pp Python (includes l4casadi) and acados libraries on LD_LIBRARY_PATH:
    python benchmark_runs/navigation_h20_20260907_seed42/run_one.py psdf
Choices: psdf, obca, dcbf. Each uses a separate directory for acados code generation.
"""
import argparse
import json
import os
import re
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
parser.add_argument('--run-label', default=None)
parser.add_argument('--recover-from-budget', action='store_true')
parser.add_argument('--resume-from', type=Path, default=None)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--duration', type=float, default=60.0)
parser.add_argument('--wall-limit', type=int, default=900)
args = parser.parse_args()
if args.resume_from is not None:
    args.resume_from = args.resume_from.resolve()
run_dir = Path(__file__).resolve().parent / (args.run_label or args.method)
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
save('runtime.json', {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__})
save('config.json', {**config, 'method': args.method, 'duration': args.duration,
                     'robot_shape': 'rectangle', 'maze_type': 'maze',
                     'dynamics_type': 'differential_drive', 'path_planner': 'astar'})
started = time.perf_counter()
original_controller = Robot.run_controller
original_system = Robot.run_system
solve_instrumented = False

def checkpoint(robot, phase):
    save('checkpoint.json', {
        'phase': phase, 'wall_seconds': time.perf_counter() - started,
        'simulation_time': robot._system._time,
        'state': robot._system.get_state(),
        'previous_applied_input': robot._system._state._u,
        'solver_status': robot._controller.get_last_solver_status_info(),
    })

def save_live(robot):
    np.savez_compressed(run_dir / 'live_trajectory.npz',
        states=np.vstack([sim.initial_pose, *robot._system_logger._xs]),
        inputs=np.asarray(robot._system_logger._us).reshape(-1,2),
        references=np.asarray(robot._local_planner_logger._trajs),
        predictions=np.asarray(robot._controller_logger._xtrajs),
        global_path=np.asarray(robot._global_path))

def run_controller(robot, obstacles):
    global solve_instrumented
    if not solve_instrumented:
        params = robot._controller._param
        assert params.horizon == 20
        assert getattr(params, 'horizon_dcbf', 20) == 20
        assert abs(getattr(params, 'tf', 2.0) - 2.0) < 1e-12
        save('parameters.json', vars(params))
        optimizer = robot._controller._optimizer
        original_solve = optimizer.solve_nlp
        def measured_solve():
            checkpoint(robot, 'solve_nlp')
            json_path = Path(getattr(optimizer, 'json_filename', 'acados_ocp.json'))
            if json_path.exists():
                compiled = json.loads(json_path.read_text())
                save('compiled_ocp_evidence.json', {k:compiled.get(k) for k in ('dims','constraints','solver_options')})
            return original_solve()
        optimizer.solve_nlp = measured_solve
        solve_instrumented = True
    checkpoint(robot, 'setup')
    save_live(robot)
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
if args.resume_from:
    from sim.simulation import SingleAgentSimulation
    from planning.trajectory_generator.constant_speed_generator import ConstantSpeedTrajectoryGenerator
    prior = args.resume_from.resolve()
    live = np.load(prior / 'trajectory.npz')
    prior_result = json.loads((prior / 'result.json').read_text())
    save('resume_provenance.json', {'source':str(prior), 'completed_steps':len(live['inputs']), 'method':prior_result['method'], 'prior_status':prior_result['status']})
    original_global_planner = Robot.run_global_planner
    def resume_global_planner(robot, system, obstacles, goal):
        original_global_planner(robot, system, obstacles, goal)
        np.testing.assert_allclose(robot._global_path, live['global_path'])
        states, inputs = live['states'], live['inputs']
        # Replay only the inexpensive deterministic reference generator to restore its index.
        for state in states[:-1]:
            robot._system._state._x = state.copy()
            robot._local_planner.generate_trajectory(robot._system, robot._global_path)
        robot._system._state._x = states[-1].copy()
        robot._system._state._u = inputs[-1].copy() if len(inputs) else np.zeros(2)
        robot._system._time = len(inputs)*.1
        if len(live['references']) > len(inputs):
            recovered_ref = robot._local_planner.generate_trajectory(robot._system, robot._global_path)
            np.testing.assert_allclose(recovered_ref, live['references'][len(inputs)], atol=1e-12)
            save('resume_reference_check.json', {'matches_pending_reference':True,'global_path_index':robot._local_planner._global_path_index})
        robot._system_logger._xs = list(states[1:])
        robot._system_logger._us = list(inputs)
        robot._local_planner_logger._trajs = list(live['references'][:len(inputs)])
        robot._controller_logger._xtrajs = list(live['predictions'][:len(inputs)])
        robot._controller_logger._solver_status_infos = list(prior_result.get('solver_statuses', []))
        robot._controller_logger._solver_status_infos += [{'raw_status':None,'status_code':None,'success':None}]*(len(inputs)-len(robot._controller_logger._solver_status_infos))
        robot._controller_logger._computation_times = [float('nan')]*len(inputs)
        prior_log = prior.parent / (prior.name + '.log')
        times = re.findall(r'solver time:\s+([0-9.]+)\s*\nplant_input_stage0:', prior_log.read_text())
        assert len(times) >= len(inputs), 'Missing recorded solve durations for resumption'
        robot._controller._optimizer.solver_times = [float(t) for t in times[:len(inputs)]]
        robot._controller_logger._risk_margin_trajs = [None]*len(inputs)
        robot._controller_logger._boole_risk_visualization_data = [None]*len(inputs)
    Robot.run_global_planner = resume_global_planner
if args.recover_from_budget:
    from sim.simulation import SingleAgentSimulation
    live = np.load(run_dir / 'live_trajectory.npz')
    def recover_navigation(simulation, duration):
        robot = simulation._robot
        states, inputs = live['states'], live['inputs']
        robot._system._state._x = states[-1].copy()
        robot._system._state._u = inputs[-1].copy() if len(inputs) else np.zeros(2)
        robot._system._time = len(inputs) * .1
        robot._system_logger._xs = list(states[1:])
        robot._system_logger._us = list(inputs)
        robot._local_planner_logger._trajs = list(live['references'])
        robot._controller_logger._xtrajs = list(live['predictions'])
        robot._global_path = live['global_path']
        robot._global_planner_logger._paths = [robot._global_path]
        raise TimeoutError('Externally enforced wall-clock budget; see budget_checkpoint.json')
    SingleAgentSimulation.run_navigation = recover_navigation
error = None
postprocessing_error = None
try:
    sim.mpc_test('maze', 'rectangle', args.method, 'differential_drive', 'astar',
                 args.duration, config)
except (Exception, KeyboardInterrupt) as exc:
    error = f'{type(exc).__name__}: {exc}'
    if (sim.last_run_outcome or {}).get('goal_reached'):
        postprocessing_error, error = error, None
    traceback.print_exc()
finally:
    signal.alarm(0)
    Robot.run_controller = original_controller
    Robot.run_system = original_system

result = dict(sim.last_run_outcome or {})
result.update(method=args.method, wall_seconds=time.perf_counter() - started,
              error=error, postprocessing_error=postprocessing_error, start_perturbation=sim.start_perturbation_info)
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
                        clearance=np.asarray(clearances), times=np.arange(len(states)) * .1,
                        references=np.asarray(robot._local_planner_logger._trajs),
                        predictions=np.asarray(robot._controller_logger._xtrajs),
                        global_path=np.asarray(robot._global_path))
    optimizer = robot._controller._optimizer
    if error and getattr(optimizer, 'opti', None) is not None:
        try:
            result['ipopt_stats'] = optimizer.opti.stats()
            optimizer.opti.debug.show_infeasibilities(1e-5)
        except Exception as diagnostics_error:
            result['diagnostics_error'] = str(diagnostics_error)
    if error:
        result['status'] = 'wall_timeout' if args.recover_from_budget else 'error'
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
