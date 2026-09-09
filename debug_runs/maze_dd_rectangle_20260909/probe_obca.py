"""Diagnose OBCA's stopped state, reconstructed from rounded live controls."""
import json, math, os, re, sys, traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from control.obca_optimizer import OBCAOptimizer, OBCAOptimizerParam
from models.dd import DifferentialDriveDynamics, DifferentialDriveSystem, DifferentialDriveStates, DifferentialDriveMultipleGeometry, DifferentialDriveRectangleGeometry
from planning.trajectory_generator.constant_speed_generator import ConstantSpeedTrajectoryGenerator
from sim.simulation_mpc import simulation_mpc
from sim.start_perturbation import convex_polygon_clearance, pose_clearance

start,goal,grid,obstacles=simulation_mpc().create_env('maze')
path=np.array(json.loads((ROOT/'psdf/diagnostics.json').read_text())['global_path'])
inputs=np.array([[float(v) for v in s.split()] for s in re.findall(r'plant_input_stage0: full=\[([^\]]+)\]',(ROOT/'obca/run.log').read_text())])
x=start.copy(); generator=ConstantSpeedTrajectoryGenerator()
generator._num_waypoint=len(path); generator._global_path=path
for u in inputs:
    generator.generate_trajectory_internal(x[:2],path)
    x=DifferentialDriveDynamics.forward_dynamics(x,u,.1)
reference=generator.generate_trajectory_internal(x[:2],path)
diag=json.loads((ROOT/'obca/diagnostics.json').read_text())
data=np.load(ROOT/'obca/diagnostics.npz')
x=np.array(diag['final_state'])
reference=data['references'][-1]
inputs=data['inputs']
geometry=DifferentialDriveMultipleGeometry(); geometry.add_geometry(DifferentialDriveRectangleGeometry(.15,.09,0))
system=DifferentialDriveSystem(state=DifferentialDriveStates(x,inputs[-1]),geometry=geometry,dynamics=DifferentialDriveDynamics())
footprint=np.array(geometry.equiv_rep()[0].get_ccw_vertices())
rotation=system._state.rotation()
world=footprint@rotation.T+x[:2]
clearances=[convex_polygon_clearance(world,np.array(o.get_ccw_vertices())) for o in obstacles]
report=dict(note='Exact replay from captured final state and final reference. Only primal warm-start guesses vary; constraints and objective are identical.', reconstructed_steps=len(inputs), reconstructed_state=x.tolist(),
            reference=reference.tolist(), global_path_index=generator._global_path_index, obstacle_clearances=clearances,
            nearest_obstacle_index=int(np.argmin(clearances)), nearest_obstacle_vertices=obstacles[int(np.argmin(clearances))].get_ccw_vertices().tolist(), solves=[])

for name in ('default','reverse_turn'):
    param=OBCAOptimizerParam()
    optimizer=OBCAOptimizer({}, {}, DifferentialDriveDynamics.forward_dynamics_opt(.1))
    optimizer.setup(param,system,reference,obstacles)
    if name!='default':
        guess=x.copy()
        for k in range(param.horizon):
            if name=='reverse_turn':
                control=np.array([-.06 if k<6 else .10, -.5 if k<15 else 0])
            else:
                control=np.array([-.06 if k<6 else .10, .5 if k<15 else 0])
            guess=DifferentialDriveDynamics.forward_dynamics(guess,control,.1)
            optimizer.opti.set_initial(optimizer.variables['x'][:,k+1],guess)
            optimizer.opti.set_initial(optimizer.variables['u'][:,k],control)
    try:
        solution=optimizer.solve_nlp()
        trajectory=np.array(solution.value(optimizer.variables['x'])).T
        controls=np.array(solution.value(optimizer.variables['u'])).T
        g=np.asarray(solution.value(optimizer.opti.g)).ravel()
        lb=np.asarray(solution.value(optimizer.opti.lbg)).ravel(); ub=np.asarray(solution.value(optimizer.opti.ubg)).ravel()
        violation=float(max(0,np.max(lb-g),np.max(g-ub)))
        clear=float(min(pose_clearance(p,[footprint],[np.array(o.get_ccw_vertices()) for o in obstacles],grid[0]) for p in trajectory))
        costs={key:float(solution.value(value)) for key,value in optimizer.costs.items()}
        solve=dict(name=name,max_constraint_violation=violation,min_predicted_clearance=clear,status=solution.stats()['return_status'],cost=sum(costs.values()),cost_components=costs,first_control=controls[0].tolist(),trajectory=trajectory.tolist(),controls=controls.tolist())
    except Exception as exc:
        solve=dict(name=name,status='exception',error=str(exc))
    report['solves'].append(solve)
    print(json.dumps({k:v for k,v in solve.items() if k not in ('trajectory','controls')},indent=2),flush=True)
(ROOT/'obca_stopped_state_probe.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('solves','reference','obstacle_clearances')},indent=2))


