from pathlib import Path
import sys,json
import numpy as np
ROOT=Path('/home/taek111/projects/psdf_mpcc')
sys.path.insert(0,str(ROOT))
from models.dd import DifferentialDriveRectangleGeometry, DifferentialDriveDynamics
from models.geometry_utils import get_dist_region_to_region
from sim.simulation_mpc import simulation_mpc
out=ROOT/'benchmark_runs/navigation_comparison_20260907_seed42'
result=json.loads((out/'dcbf/result.json').read_text())
nominal,goal,grid,obstacles=simulation_mpc().create_env('maze')
G,g=DifferentialDriveRectangleGeometry(.15,.09,0.)._region.get_convex_rep()
A,b=obstacles[1].get_convex_rep()
evidence={}
for name,pose in [('nominal',nominal),('failure',np.array(result['final_pose']))]:
    t=pose[2]
    R=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    translated=G@R.T@pose[:2]+g
    fixed=G@R.T@pose[:2,None]+g
    item={'translation_term_shape':list((G@R.T@pose[:2]).shape),'robot_g_shape':list(g.shape),
          'actual_rhs_shape':list(translated.shape),'expected_rhs_shape':list(fixed.shape)}
    try:
        get_dist_region_to_region(A,b,G@R.T,translated)
    except Exception as exc:
        item['error']=str(exc)
    evidence[name]=item
v=result['first_input'][0]
evidence['obstacle_filter_radius_initial']=DifferentialDriveDynamics.safe_dist(.1,0.,1.,.001)
evidence['obstacle_filter_radius_after_step1']=DifferentialDriveDynamics.safe_dist(.1,v,1.,.001)
(out/'dcbf_shape_diagnosis.json').write_text(json.dumps(evidence,indent=2))
print(json.dumps(evidence,indent=2))
