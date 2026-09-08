import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/taek111/projects/psdf_mpcc')
sys.path.insert(0,str(ROOT))
from sim.simulation_mpc import simulation_mpc
from models.dd import DifferentialDriveDynamics, DifferentialDriveStates, DifferentialDriveSystem, DifferentialDriveMultipleGeometry, DifferentialDriveRectangleGeometry
from planning.path_generator.search_path_generator import AstarLoSPathGenerator
from planning.trajectory_generator.constant_speed_generator import ConstantSpeedTrajectoryGenerator
from control.obca_optimizer import OBCAOptimizer, OBCAOptimizerParam
from sim.start_perturbation import pose_clearance, convex_polygon_clearance
out=ROOT/'benchmark_runs/navigation_comparison_20260907_seed42'
checkpoint=json.loads((out/'obca/checkpoint.json').read_text())
pose=np.array(checkpoint['state'])
start=json.loads(next((out/'obca/data').glob('start_pose_*.json')).read_text())['initial_pose']
_,goal,grid,obstacles=simulation_mpc().create_env('maze')
geometry=DifferentialDriveMultipleGeometry()
geometry.add_geometry(DifferentialDriveRectangleGeometry(.15,.09,0.))
system=DifferentialDriveSystem(DifferentialDriveStates(np.array(start)),geometry,DifferentialDriveDynamics())
planner=AstarLoSPathGenerator(grid,quad=False,margin=.03)
path=planner.generate_path(system,obstacles,goal)
local=ConstantSpeedTrajectoryGenerator()
local.generate_trajectory(system,path)
system._state._x=pose
system._state._u=np.array(checkpoint['previous_applied_input'])
reference=local.generate_trajectory(system,path)
param=OBCAOptimizerParam()
optimizer=OBCAOptimizer({},{},DifferentialDriveDynamics.forward_dynamics_opt(.1))
optimizer.setup(param,system,reference,obstacles)
solution=optimizer.solve_nlp()
xs=solution.value(optimizer.variables['x']).T
us=solution.value(optimizer.variables['u']).T
vertices=geometry.equiv_rep()[0].get_ccw_vertices()
theta=pose[2]
rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
world=vertices@rotation.T+pose[:2]
clearance_by_obstacle=[convex_polygon_clearance(world,o.get_ccw_vertices()) for o in obstacles]
near=int(np.argmin(clearance_by_obstacle))
result={'pose':pose.tolist(),'input':us[0].tolist(),'local_path_index':local._global_path_index,
        'reference':reference.tolist(),'global_path':path.tolist(),
        'predicted_states':xs.tolist(),'predicted_inputs':us.tolist(),
        'predicted_clearance':[pose_clearance(x,[vertices],[o.get_ccw_vertices() for o in obstacles],grid[0]) for x in xs],
        'nearest_obstacle_index':near,'nearest_obstacle_vertices':obstacles[near].get_ccw_vertices().tolist(),
        'footprint_world_vertices':world.tolist(),'clearance_by_obstacle':clearance_by_obstacle,
        'solver_status':solution.stats()['return_status'],
        'costs':{k:float(solution.value(v)) for k,v in optimizer.costs.items()}}
(out/'obca_corner_diagnosis.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ('pose','input','local_path_index','nearest_obstacle_index','nearest_obstacle_vertices','predicted_clearance','costs')},indent=2))
