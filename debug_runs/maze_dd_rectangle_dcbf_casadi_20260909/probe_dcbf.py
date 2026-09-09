"""Bounded one-step diagnostics; the nominal test continues independently."""
import json, math, re, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT))
from control.dcbf_optimizer import NmpcDbcfOptimizer, NmpcDcbfOptimizerParam
from models.dd import DifferentialDriveDynamics, DifferentialDriveSystem, DifferentialDriveStates, DifferentialDriveMultipleGeometry, DifferentialDriveRectangleGeometry
from planning.trajectory_generator.constant_speed_generator import ConstantSpeedTrajectoryGenerator
from sim.simulation_mpc import simulation_mpc
from sim.start_perturbation import convex_polygon_clearance, pose_clearance
start,goal,grid,obstacles=simulation_mpc().create_env('maze')
path=np.array(json.loads((PROJECT/'debug_runs/maze_dd_rectangle_20260909/psdf/diagnostics.json').read_text())['global_path'])
geometry=DifferentialDriveMultipleGeometry(); geometry.add_geometry(DifferentialDriveRectangleGeometry(.15,.09,0))
if (ROOT/'dcbf/diagnostics.npz').exists():
    saved=np.load(ROOT/'dcbf/diagnostics.npz')
    diag=json.loads((ROOT/'dcbf/diagnostics.json').read_text())
    x=np.array(diag['final_state']); last_u=saved['inputs'][-1]
    generator=ConstantSpeedTrajectoryGenerator(); generator._num_waypoint=len(path); generator._global_path=path
    for pose in np.vstack([start,saved['states']]):
        reference=generator.generate_trajectory_internal(pose[:2],path)
    reference=saved['references'][-1]
    assert np.allclose(path,np.asarray(diag['global_path']))
    state_source='exact saved final state'
else:
    inputs=np.array([[float(v) for v in s.split()] for s in re.findall(r'plant_input_stage0: full=\[([^\]]+)\]',(ROOT/'dcbf/run.log').read_text())])
    x=start.copy(); generator=ConstantSpeedTrajectoryGenerator(); generator._num_waypoint=len(path); generator._global_path=path
    for u in inputs:
        generator.generate_trajectory_internal(x[:2],path)
        x=DifferentialDriveDynamics.forward_dynamics(x,u,.1)
    reference=generator.generate_trajectory_internal(x[:2],path)
    last_u=inputs[-1]; state_source='approximate reconstruction from rounded live controls'
system=DifferentialDriveSystem(state=DifferentialDriveStates(x,last_u),geometry=geometry,dynamics=DifferentialDriveDynamics())
footprint=np.array(geometry.equiv_rep()[0].get_ccw_vertices()); polygons=[np.array(o.get_ccw_vertices()) for o in obstacles]
world=footprint@system._state.rotation().T+x[:2]
clearances=[convex_polygon_clearance(world,p) for p in polygons]
report=dict(state_source=state_source,state=x.tolist(),previous_input=last_u.tolist(),reference=reference.tolist(),
            global_path_index=generator._global_path_index,nearest_obstacle_index=int(np.argmin(clearances)),
            current_clearance=float(min(clearances)),solves=[])
names=sys.argv[1:] or ('default','reverse_left','reverse_right','horizon11_cbf11','horizon20_cbf6','horizon20_cbf20','aligned_terminal_reference')
for name in names:
    param=NmpcDcbfOptimizerParam()
    if name.startswith('horizon20'): param.horizon=20
    if name=='horizon20_cbf20': param.horizon_dcbf=20
    if name=='horizon11_cbf11': param.horizon_dcbf=11
    ref=reference[:param.horizon] if name=='aligned_terminal_reference' else reference
    optimizer=NmpcDbcfOptimizer({}, {}, DifferentialDriveDynamics.forward_dynamics_opt(.1))
    optimizer.setup(param,system,ref,obstacles)
    if name.startswith('reverse'):
        guess=x.copy()
        for k in range(param.horizon):
            u_guess=np.array([-.06 if k<6 else .1, .5 if name=='reverse_left' else -.5])
            guess=DifferentialDriveDynamics.forward_dynamics(guess,u_guess,.1)
            optimizer.opti.set_initial(optimizer.variables['x'][:,k+1],guess)
            optimizer.opti.set_initial(optimizer.variables['u'][:,k],u_guess)
    try:
        solution=optimizer.solve_nlp()
        trajectory=np.array(solution.value(optimizer.variables['x'])).T
        controls=np.array(solution.value(optimizer.variables['u'])).T
        g=np.array(solution.value(optimizer.opti.g)).ravel()
        lb=np.array(solution.value(optimizer.opti.lbg)).ravel(); ub=np.array(solution.value(optimizer.opti.ubg)).ravel()
        costs={key:float(solution.value(value)) for key,value in optimizer.costs.items()}
        clearance=np.array([pose_clearance(p,[footprint],polygons,grid[0]) for p in trajectory])
        result=dict(name=name,horizon=param.horizon,horizon_dcbf=param.horizon_dcbf,status=solution.stats()['return_status'],cost=sum(costs.values()),
                    cost_components=costs,first_control=controls[0].tolist(),max_constraint_violation=float(max(0,np.max(lb-g),np.max(g-ub))),
                    min_predicted_clearance=float(clearance.min()),zero_clearance_stages=np.flatnonzero(clearance<=0).tolist(),trajectory=trajectory.tolist(),controls=controls.tolist())
    except Exception as exc:
        result=dict(name=name,status='exception',error=str(exc))
    report['solves'].append(result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('trajectory','controls')},indent=2),flush=True)
output='stopped_state_probe_exact.json' if state_source.startswith('exact') else 'stopped_state_probe_approx.json'
if len(sys.argv)>1: output=output.replace('.json','_'+'_'.join(names)+'.json')
(ROOT/output).write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('solves','reference')},indent=2))
