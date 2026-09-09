"""Inspect the original DCBF setup, replacing solver construction only with a recorder."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import casadi as ca
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from control.dcbf_optimizer_sqp import NmpcDcbfOptimizerSqp, NmpcDcbfOptimizerSqpParam
from models.dd import DifferentialDriveDynamics, DifferentialDriveSystem, DifferentialDriveStates, DifferentialDriveMultipleGeometry, DifferentialDriveRectangleGeometry
from sim.simulation_mpc import simulation_mpc

start, goal, grid, obstacles = simulation_mpc().create_env("maze")
geometry = DifferentialDriveMultipleGeometry()
geometry.add_geometry(DifferentialDriveRectangleGeometry(0.15, 0.09, 0.0))
system = DifferentialDriveSystem(state=DifferentialDriveStates(start), geometry=geometry, dynamics=DifferentialDriveDynamics())
reference = np.load(ROOT / "psdf/diagnostics.npz")["references"][0]
(ROOT / 'dcbf_probe').mkdir(exist_ok=True)
os.chdir(ROOT / 'dcbf_probe')
optimizer = NmpcDcbfOptimizerSqp()
param = NmpcDcbfOptimizerSqpParam()
report = {}
class Recorder:
    status = 0
    def __init__(self): self.values = {}
    def set(self, stage, field, value): self.values[(stage, field)] = np.asarray(value)
    def get(self, stage, field): return self.values.get((stage, field), np.zeros(3 if field == "x" else 2))
    def solve(self): return 0

def inspect_create_solver():
    report["before_solver_creation"] = dict(nz=optimizer.ocp.dims.nz, nh=optimizer.ocp.dims.nh, ng=optimizer.ocp.dims.ng,
                                           con_h_expr_empty=optimizer.ocp.model.con_h_expr is None or isinstance(optimizer.ocp.model.con_h_expr, list) and len(optimizer.ocp.model.con_h_expr) == 0,
                                           reference_is_none=optimizer.reference_trajectory is None)
    optimizer.solver = Recorder()
    optimizer.variables.update(x="x", u="u")
optimizer.create_solver = inspect_create_solver
optimizer.setup(param, system, reference, obstacles)
report["after_setup"] = dict(nh=optimizer.ocp.dims.nh, ng=optimizer.ocp.dims.ng,
                            con_h_expr_empty=optimizer.ocp.model.con_h_expr is None or isinstance(optimizer.ocp.model.con_h_expr, list) and len(optimizer.ocp.model.con_h_expr) == 0,
                            unused_expr_h_rows=optimizer.ocp.constraints.expr_h.numel(),
                            reference_is_none=optimizer.reference_trajectory is None)
optimizer.solve_nlp()
report["supplied_reference_stage0"] = reference[0].tolist()
report["actual_reference_parameter_stage0"] = optimizer.solver.values[(0,"p")].tolist()
model = optimizer.ocp.model
residual = ca.Function("residual_probe",[model.x,model.xdot,model.u,model.z],[model.f_impl_expr])
report["algebraic_residual_with_all_z_one"] = np.unique(np.asarray(residual(start,np.zeros(3),np.zeros(2),np.ones(model.z.numel())))[3:]).tolist()
h_probe = ca.Function('h_probe',[model.x,model.z],[optimizer.ocp.constraints.expr_h])
h_values = np.asarray(h_probe(start,np.zeros(model.z.numel()))).ravel()
report['intended_constraint_violations_at_z_zero'] = int((h_values < optimizer.ocp.constraints.lg - 1e-12).sum())
report['minimum_intended_constraint_residual_at_z_zero'] = float((h_values-optimizer.ocp.constraints.lg).min())
report["note"] = "Setup recorder only; no numerical optimization performed. The unmodified baseline run is in dcbf/run.log."
(ROOT / "dcbf_setup_inspection.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
optimizer.solver = None


