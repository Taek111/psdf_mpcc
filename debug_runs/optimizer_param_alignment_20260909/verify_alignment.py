"""Check the constructed costs, bounds, and obstacle-stage dependencies."""
import json
import sys
from pathlib import Path

import casadi as ca
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from control.dcbf_optimizer import NmpcDbcfOptimizer, NmpcDcbfOptimizerParam
from control.obca_optimizer import OBCAOptimizer, OBCAOptimizerParam
from control.psdf_optimizer import PSDFOptimizerParam
from models.dd import DifferentialDriveRectangleGeometry, DifferentialDriveStates
from models.geometry_utils import RectangleRegion

baseline = PSDFOptimizerParam()
report = {}
for name, optimizer_type, param_type in (
    ('dcbf', NmpcDbcfOptimizer, NmpcDcbfOptimizerParam),
    ('obca', OBCAOptimizer, OBCAOptimizerParam),
):
    param = param_type()
    for key in ('horizon', 'mat_Q', 'mat_R', 'terminal_weight', 'vmin', 'vmax', 'omegamin', 'omegamax'):
        np.testing.assert_array_equal(getattr(param, key), getattr(baseline, key))
    assert param.margin_dist == baseline.d_safe
    optimizer = optimizer_type({}, {}, None)
    optimizer.opti = ca.Opti()
    optimizer.initialize_variables(param)
    optimizer.add_reference_trajectory_tracking_cost(param, np.zeros((param.horizon, 3)))
    optimizer.add_input_stage_cost(param)
    state = np.zeros((3, param.horizon + 1))
    control = np.zeros((2, param.horizon))
    cost = ca.Function('cost', [optimizer.variables['x'], optimizer.variables['u']], [
        optimizer.costs['reference_trajectory_tracking'] + optimizer.costs['input_stage']])
    # The formerly omitted stage N-1 must contribute the same position cost as stage 0.
    state[0, 0] = 1.0
    first_cost = float(cost(state, control))
    state[0, 0] = 0.0
    state[0, -2] = 1.0
    assert np.isclose(float(cost(state, control)), first_cost)
    state[0, -2] = 0.0
    state[0, -1] = 1.0
    assert np.isclose(float(cost(state, control)), first_cost * baseline.terminal_weight)
    state[:] = 0.0
    control[:, -1] = [0.2, 0.3]
    assert np.isclose(float(cost(state, control)), control[:, -1] @ baseline.mat_R @ control[:, -1])

    # Override every bound and inspect the actual constructed inequalities.
    param.vmin, param.vmax = -0.21, 0.37
    param.omegamin, param.omegamax = -0.42, 0.68
    optimizer.add_input_constraint(param)
    bounds = ca.Function('bounds', [], [optimizer.opti.lbg, optimizer.opti.ubg])()
    lb = np.array(bounds['o0']).ravel()
    ub = np.array(bounds['o1']).ravel()
    g_eval = ca.Function('g', [optimizer.variables['u']], [optimizer.opti.g])
    for u in ([param.vmin, param.omegamin], [param.vmax, param.omegamax]):
        g = np.array(g_eval(np.tile(u, (param.horizon, 1)).T)).ravel()
        assert np.all(g >= lb) and np.all(g <= ub)
    for u in ([param.vmin - 0.01, 0.0], [param.vmax + 0.01, 0.0],
              [0.0, param.omegamin - 0.01], [0.0, param.omegamax + 0.01]):
        g = np.array(g_eval(np.tile(u, (param.horizon, 1)).T)).ravel()
        assert np.any(g < lb) or np.any(g > ub)

    # Build only the rectangle avoidance constraints and inspect state dependencies.
    avoidance = optimizer_type({}, {'decay_rate_relaxing': 0}, None)
    avoidance.opti = ca.Opti()
    avoidance.initialize_variables(param)
    avoidance.set_state(DifferentialDriveStates(np.zeros(3)))
    avoidance.add_convex_to_convex_constraint(
        param, DifferentialDriveRectangleGeometry(0.15, 0.09, 0.0).equiv_rep()[0],
        RectangleRegion(0.3, 0.4, -0.1, 0.1), float('inf'))
    jac = ca.jacobian(avoidance.opti.g, ca.vec(avoidance.variables['x']))
    _, columns = jac.sparsity().get_triplet()
    stages = sorted(set(column // 3 for column in columns))
    assert stages == list(range(1, param.horizon_dcbf + 1))
    assert param.horizon_dcbf == (10 if name == 'dcbf' else 20)
    report[name] = dict(defaults_match_psdf=True, last_stage_cost_included=True,
                        input_penalty_verified=True, custom_input_bounds_enforced=True,
                        rectangle_obstacle_constraint_stages=stages)

for name in ('psdf', 'dcbf', 'obca'):
    diag = json.loads((ROOT / name / 'diagnostics.json').read_text())
    data = np.load(ROOT / name / 'diagnostics.npz')
    statuses = diag['solver_statuses']
    report.setdefault(name, {})['smoke'] = dict(
        optimizer_module=diag['optimizer_module'], steps=len(data['inputs']),
        final_time=diag['final_time'], statuses=statuses)
    assert len(data['inputs']) == 3
    assert np.all(np.isfinite(data['inputs']))

(ROOT / 'verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))