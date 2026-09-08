import json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/taek111/projects/psdf_mpcc');sys.path.insert(0,str(ROOT))
from sim.simulation_mpc import simulation_mpc
from models.dd import DifferentialDriveDynamics, DifferentialDriveStates, DifferentialDriveSystem, DifferentialDriveMultipleGeometry, DifferentialDriveRectangleGeometry
from control.obca_optimizer import OBCAOptimizer, OBCAOptimizerParam
from sim.start_perturbation import pose_clearance
out=ROOT/'benchmark_runs/navigation_h20_20260907_seed42'
old=ROOT/'benchmark_runs/navigation_comparison_20260907_seed42'
data=json.loads((old/'obca_corner_diagnosis.json').read_text())
check=json.loads((old/'obca/checkpoint.json').read_text())
pose=np.array(data['pose']); reference=np.array(data['reference'])
_,goal,grid,obstacles=simulation_mpc().create_env('maze')
geometry=DifferentialDriveMultipleGeometry();geometry.add_geometry(DifferentialDriveRectangleGeometry(.15,.09,0.))
system=DifferentialDriveSystem(DifferentialDriveStates(pose),geometry,DifferentialDriveDynamics())
system._state._u=np.load(old/'obca/trajectory.npz')['inputs'][-1]
vertices=geometry.equiv_rep()[0].get_ccw_vertices();obs_vertices=[o.get_ccw_vertices() for o in obstacles]
result={'pose':pose.tolist(),'reference':reference.tolist(),'reference_clearances':[pose_clearance(x,[vertices],obs_vertices,grid[0]) for x in reference], 'cases':{}}
for n,h in [(11,6),(11,11),(20,6),(20,20)]:
 param=OBCAOptimizerParam();param.horizon=n;param.horizon_dcbf=h
 opt=OBCAOptimizer({},{},DifferentialDriveDynamics.forward_dynamics_opt(.1))
 opt.setup(param,system,reference,obstacles)
 try:
  sol=opt.solve_nlp(); xs=sol.value(opt.variables['x']).T;us=sol.value(opt.variables['u']).T
  record={'status':sol.stats()['return_status'],'input':us[0].tolist(),'states':xs.tolist(),'inputs':us.tolist(), 'clearances':[pose_clearance(x,[vertices],obs_vertices,grid[0]) for x in xs], 'costs':{k:float(sol.value(v)) for k,v in opt.costs.items()}, 'wall_solve':opt.solver_times[-1]}
 except Exception as exc:record={'error':str(exc)}
 result['cases'][f'N{n}_H{h}']=record
 (out/'obca_horizon_ablation.json').write_text(json.dumps(result,indent=2))
 print(n,h,{k:v for k,v in record.items() if k not in ['states','inputs']},flush=True)
