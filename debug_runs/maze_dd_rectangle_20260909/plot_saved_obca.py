"""Use the repository's native plotting method on the saved OBCA trajectory."""
import json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from sim.simulation_mpc import simulation_mpc
from models.dd import DifferentialDriveMultipleGeometry, DifferentialDriveRectangleGeometry
render=simulation_mpc()
render.output_root_dir=str(ROOT/'obca')
start,goal,grid,obstacles=render.create_env('maze')
diag=json.loads((ROOT/'obca/diagnostics.json').read_text())
data=np.load(ROOT/'obca/diagnostics.npz')
geometry=DifferentialDriveMultipleGeometry(); geometry.add_geometry(DifferentialDriveRectangleGeometry(.15,.09,0))
robot=SimpleNamespace(_system=SimpleNamespace(_geometry=geometry),_system_logger=SimpleNamespace(_xs=data['states']),
                     _global_planner_logger=SimpleNamespace(_paths=[np.array(diag['global_path'])]),
                     _local_planner_logger=SimpleNamespace(_trajs=data['references']),
                     _controller_logger=SimpleNamespace(_xtrajs=data['predicted_states']))
simulation=SimpleNamespace(_robot=robot,_obstacles=obstacles)
render.plot_world(simulation,[],figure_name='mpc_obca_rectangle_maze_20260909_interrupted',maze_type='maze')
print('Saved native OBCA trajectory plot.')
