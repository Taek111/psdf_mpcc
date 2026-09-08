from pathlib import Path
import sys,os,json
import numpy as np
import casadi as ca
root=Path('/home/taek111/projects/psdf_mpcc');sys.path.insert(0,str(root))
from sim.simulation_mpc import simulation_mpc
from models.dd import DifferentialDriveStates,DifferentialDriveSystem,DifferentialDriveDynamics,DifferentialDriveMultipleGeometry,DifferentialDriveRectangleGeometry
from control.dcbf_optimizer_sqp import NmpcDcbfOptimizerSqp,NmpcDcbfOptimizerSqpParam
from models.geometry_utils import get_dist_region_to_region
out=root/'benchmark_runs/navigation_h20_20260907_seed42'
work=out/'diagnostics/dcbf_structure';work.mkdir(parents=True,exist_ok=True);os.chdir(work)
pose,goal,grid,obstacles=simulation_mpc().create_env('maze')
geometry=DifferentialDriveMultipleGeometry();geometry.add_geometry(DifferentialDriveRectangleGeometry(.15,.09,0.))
system=DifferentialDriveSystem(DifferentialDriveStates(pose),geometry,DifferentialDriveDynamics())
param=NmpcDcbfOptimizerSqpParam();opt=NmpcDcbfOptimizerSqp();opt.set_state(system._state)
# One obstacle is sufficient to test the structural dual-variable contradiction.
opt.setup_ocp(param,np.tile(pose,(20,1)),[obstacles[1]],geometry.equiv_rep())
opt.add_obstacle_avoidance_constraint(param,system,[obstacles[1]])
model=opt.ocp.model;nz=int(model.z.numel())
h=opt.ocp.constraints.expr_h
h_zero=np.asarray(ca.Function('h',[model.x,model.z],[h])(pose,np.zeros(nz))).reshape(-1)
lower=np.asarray(opt.ocp.constraints.lg)
viol=np.where(h_zero<lower-1e-10)[0]
J=np.asarray(ca.DM(ca.jacobian(model.f_impl_expr,model.z)[3:,:]))
G,g=geometry.equiv_rep()[0].get_convex_rep();R=system._state.rotation()
A,b=obstacles[1].get_convex_rep();bad=G@R.T@pose[:2]+g;good=G@R.T@pose[:2,None]+g
try:get_dist_region_to_region(A,b,G@R.T,bad)
except Exception as exc:shape_error=str(exc)
fixed_distance=get_dist_region_to_region(A,b,G@R.T,good)[0]
evidence={'horizon':param.horizon,'safety_horizon':param.horizon_dcbf,'tf':param.tf,
          'one_obstacle_nz':nz,'algebraic_residual_jacobian_is_identity':bool(np.array_equal(J,np.eye(nz))),
          'safety_rows_violated_when_z_zero':len(viol),'violated_h_at_z_zero':h_zero[viol].tolist(),'violated_lower_bounds':lower[viol].tolist(),
          'model_con_h_expr':str(model.con_h_expr),'wrong_rhs_shape':list(bad.shape),'correct_rhs_shape':list(good.shape),
          'shape_error':shape_error,'distance_with_column_translation':float(fixed_distance)}
raw=out/'dcbf_pp/acados_ocp.json'
if raw.exists():
    data=json.loads(raw.read_text());evidence['actual_compiled_dims']=data['dims']
    (out/'dcbf_pp/compiled_ocp_evidence.json').write_text(json.dumps({k:data.get(k) for k in ('dims','constraints','solver_options')},indent=2))
if 'actual_compiled_dims' not in evidence:
    evidence['actual_compiled_dims']=json.loads((out/'dcbf_pp/compiled_ocp_evidence.json').read_text())['dims']
# Capture the parameters actually sent by solve_nlp without an expensive solve.
class CaptureSolver:
    def __init__(self): self.parameters={}; self.status=0
    def set(self, stage, field, value):
        if field=='p': self.parameters[stage]=np.asarray(value).tolist()
    def solve(self): return 0
    def get(self, stage, field): return pose.copy() if field=='x' else np.zeros(2)
probe=NmpcDcbfOptimizerSqp()
capture=CaptureSolver()
def create_capture():
    probe.solver=capture
    probe.variables={'x':'x','u':'u'}
probe.create_solver=create_capture
supplied_ref=np.tile(pose,(20,1))
probe.setup(param,system,supplied_ref,[obstacles[1]])
probe.solve_nlp()
evidence['supplied_reference_first_pose']=supplied_ref[0].tolist()
evidence['reference_attribute_after_setup_is_none']=probe.reference_trajectory is None
evidence['parameters_sent_to_solver']=capture.parameters
evidence['all_reference_parameters_sent_are_zero']=all(not np.any(v) for v in capture.parameters.values())
(out/'dcbf_structure_diagnosis.json').write_text(json.dumps(evidence,indent=2))
print(json.dumps(evidence,indent=2))
