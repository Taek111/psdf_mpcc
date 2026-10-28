"""Evaluate saved RTI solutions at first stage-0 violations; never runs a solver."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import casadi as ca
import numpy as np
from shapely.geometry import Polygon
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source_snapshot'
sys.path.insert(0, str(SOURCE))
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.dd_dynamics import DifferentialDriveDynamics, symbolic_dd_step
spec = importlib.util.spec_from_file_location('frozen_covariance', SOURCE / 'control/cc_covariance.py')
covariance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(covariance)


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def functions(param):
    state = ca.MX.sym('state', 8)
    control = ca.MX.sym('input', 3)
    terms = ca.MX.sym('terms', 4)
    xi = state[4:]
    nx = ca.vertcat(symbolic_dd_step(state[:3], control[0], control[1], .1),
                    state[3] + .1 * control[2],
                    covariance.symbolic_log_cholesky_transition(
                        xi, control[0], control[1], 0., .1, param))
    dynamics = ca.Function('saved_dynamics', [state, control],
                           [nx, ca.jacobian(nx, state), ca.jacobian(nx, control)])
    q_l = ca.sqrt(ca.fmax(terms[1], 0.))
    q_theta = ca.if_else(terms[3] < 0., -1., 1.) * ca.sqrt(ca.fmax(terms[2], 0.))
    var = (ca.exp(2*xi[0])*terms[0] + (ca.exp(xi[1])*q_l + xi[3]*q_theta)**2
           + ca.exp(2*xi[2])*terms[2])
    positive = var > 0.
    margin = param.chance_gamma * ca.if_else(positive, ca.sqrt(ca.if_else(positive, var, 1.)), 0.)
    if param.risk_margin_floor > 0:
        margin = ca.fmax(margin, param.risk_margin_floor)
    if np.isfinite(param.risk_margin_cap):
        margin = ca.fmin(margin, param.risk_margin_cap)
    risk = ca.Function('saved_risk', [state, terms], [margin, ca.jacobian(margin, state)])
    return dynamics, risk


def transform(vertices, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return vertices @ np.array([[c, s], [-s, c]]) + pose[:2]


folder = ROOT / 'width_110mm'
param = SimpleNamespace(**json.loads((folder / 'optimizer_parameters.json').read_text()))
dynamics, risk = functions(param)
geometry = json.loads((folder / 'geometry.json').read_text())
footprint = np.asarray(geometry['footprint'])
obstacles = [np.asarray(o) for o in geometry['obstacles']]
wrapper32 = AugmentedPSDFWrapper(torch.tensor(footprint, dtype=torch.float32, device='cuda'),
                                 K_max=20, E_max=100, device='cuda')
wrapper64 = AugmentedPSDFWrapper(torch.tensor(footprint, dtype=torch.float64),
                                 K_max=20, E_max=4, device='cpu')
for wrapper in (wrapper32, wrapper64):
    wrapper.update_edge_clusters(obstacles, [np.roll(o, -1, axis=0) for o in obstacles])
stops = rows(ROOT / 'stage0_stop_onsets.csv')
needed = {(int(r['seed']), int(r['cycle']) - j) for r in stops for j in (0, 1)}
records = {}
with (folder / 'nominal_risk.jsonl').open() as stream:
    for line in stream:
        record = json.loads(line)
        if record['kind'] == 'fast' and (record['seed'], record['cycle']) in needed:
            records[(record['seed'], record['cycle'])] = record
results, details = [], []
for stop in stops:
    seed, cycle = int(stop['seed']), int(stop['cycle']) - 1
    saved = records[(seed, cycle)]
    following = records[(seed, cycle+1)]
    assert saved['status'] == 0 and following['status'] != 0
    nominal = np.asarray(saved['states'])
    returned = np.asarray(saved['returned_states'])
    nominal_u = np.asarray(saved['inputs'])
    returned_u = np.asarray(saved['returned_inputs'])
    history = rows(folder / 'runs/config01' / f'trial_{seed+1:04d}' / 'data' /
                   f'trial_history_MPC_cc_sf_rectangle_maze_startseed{seed}.csv')
    applied = np.array([float(history[cycle][key]) for key in ('v', 'omega')])
    np.testing.assert_array_equal(returned_u[0,:2], applied)
    after = np.array([float(history[cycle][key]) for key in ('x','y','theta')])
    physical = DifferentialDriveDynamics.forward_dynamics(nominal[0,:3], applied, .1)
    np.testing.assert_array_equal(physical, after)
    np.testing.assert_array_equal(following['states'][0][:3], after)
    fbar, A, B = dynamics(nominal[0], nominal_u[0])
    qp_next = (np.asarray(fbar).ravel() + np.asarray(A) @ (returned[0] - nominal[0])
               + np.asarray(B) @ (returned_u[0] - nominal_u[0]))
    np.testing.assert_allclose(qp_next, returned[1], rtol=0, atol=1e-9)
    poses = np.stack([nominal[1,:3], returned[1,:3], after])
    d32, g32 = wrapper32.forward_clusters(torch.tensor(poses, dtype=torch.float32, device='cuda'))
    d64, g64 = wrapper64.forward_clusters(torch.tensor(poses))
    d32, g32, d64, g64 = [a.detach().cpu().numpy() for a in (d32,g32,d64,g64)]
    active = d32.argmin(axis=1)
    phi32 = d32.min(axis=1).astype(float)
    np.testing.assert_allclose(phi32[[0,2]], [saved['phi_m'][1], following['phi_m'][0]], rtol=0, atol=1e-10)
    gradient = g32[0,active[0]].astype(float)
    affine_phi = phi32[0] + gradient @ (returned[1,:3] - nominal[1,:3].astype(np.float32).astype(float))
    risk_terms = np.asarray(saved['risk_terms'][1])
    margin_bar, gradient_r = risk(nominal[1], risk_terms)
    np.testing.assert_allclose(float(margin_bar), saved['risk_margin_m'][1], rtol=0, atol=1e-12)
    margin_qp = float(margin_bar) + float(np.asarray(gradient_r).ravel() @ (returned[1] - nominal[1]))
    margin_returned = float(risk(returned[1], risk_terms)[0])
    consistent = np.asarray(dynamics(nominal[0], returned_u[0])[0]).ravel()
    margin_consistent = float(risk(consistent, risk_terms)[0])
    margin_linear_consistent = float(margin_bar) + float(
        np.asarray(gradient_r).ravel() @ (consistent - nominal[1]))
    vbar, wbar = nominal_u[0,:2]
    vnew, wnew = returned_u[0,:2]
    vw_abs_bar = abs(vbar*wbar)
    vw_abs_linear = (vw_abs_bar + np.sign(vbar*wbar) *
                     (wbar*(vnew-vbar) + vbar*(wnew-wbar)))
    qp_residual = affine_phi - param.d_min - margin_qp
    assert qp_residual >= -1e-7, qp_residual
    distances_independent = []
    for pose in poses:
        robot = Polygon(transform(footprint, pose))
        distances_independent.append([robot.distance(Polygon(o)) for o in obstacles])
    distances_independent = np.asarray(distances_independent)
    assert d64.min() > 0, 'This separated-pose check expects no overlap.'
    np.testing.assert_allclose(d64, distances_independent, rtol=0, atol=1e-9)
    fd = []
    for axis, eps in enumerate([1e-7, 1e-7, 1e-6]):
        delta = np.eye(3)[axis] * eps
        plus = Polygon(transform(footprint, poses[0]+delta)).distance(Polygon(obstacles[active[0]]))
        minus = Polygon(transform(footprint, poses[0]-delta)).distance(Polygon(obstacles[active[0]]))
        fd.append((plus-minus)/(2*eps))
    np.testing.assert_allclose(fd, g64[0,active[0]], rtol=0, atol=1e-5)
    event = {
        'width_m': .110, 'seed': seed, 'cycle': cycle,
        'time_before_s': cycle*.1, 'time_after_s': (cycle+1)*.1,
        'solver_status': saved['status'],
        'nominal_active_obstacle': int(active[0]), 'returned_active_obstacle': int(active[1]),
        'actual_active_obstacle': int(active[2]),
        'current_phi_before_mm': saved['phi_m'][0]*1000,
        'anchor_phi_mm': phi32[0]*1000,
        'affine_phi_at_returned_mm': affine_phi*1000,
        'fresh_phi_at_returned_mm': phi32[1]*1000,
        'actual_phi_after_mm': phi32[2]*1000,
        'independent_actual_clearance_mm': distances_independent[2].min()*1000,
        'geometry_taylor_gap_mm': (affine_phi-phi32[1])*1000,
        'kinematics_distance_gap_mm': (phi32[1]-phi32[2])*1000,
        'nominal_margin_mm': float(margin_bar)*1000,
        'qp_linearized_margin_mm': margin_qp*1000,
        'nonlinear_margin_at_returned_xi_mm': margin_returned*1000,
        'nonlinear_margin_at_consistent_xi_mm': margin_consistent*1000,
        'qp_residual_mm': qp_residual*1000,
        'margin_linear_at_consistent_xi_mm': margin_linear_consistent*1000,
        'lateral_std_nominal_mm': np.exp(nominal[1,5])*1000,
        'lateral_std_returned_mm': np.exp(returned[1,5])*1000,
        'lateral_std_consistent_mm': np.exp(consistent[5])*1000,
        'abs_vomega_nominal': vw_abs_bar,
        'abs_vomega_linear_at_new_input': vw_abs_linear,
        'abs_vomega_actual_new_input': abs(vnew*wnew),
        'nonlinear_residual_at_returned_mm': (phi32[1]-param.d_min-margin_returned)*1000,
        'translation_defect_mm': np.linalg.norm(returned[1,:2]-after[:2])*1000,
        'heading_defect_rad': returned[1,2]-after[2],
        'anchor_to_returned_translation_mm': np.linalg.norm(returned[1,:2]-nominal[1,:2])*1000,
        'anchor_to_returned_heading_deg': np.rad2deg(returned[1,2]-nominal[1,2]),
        'nominal_v_mps': nominal_u[0,0], 'nominal_omega_radps': nominal_u[0,1],
        'applied_v_mps': applied[0], 'applied_omega_radps': applied[1],
        'qp_dynamics_replay_max_error': np.max(np.abs(qp_next-returned[1])),
        'physical_rk4_replay_error': np.max(np.abs(physical-after)),
        'independent_geometry_max_error_mm': np.max(np.abs(d64-distances_independent))*1000,
        'float32_vs_float64_phi_max_error_mm': np.max(np.abs(phi32-d64.min(axis=1)))*1000,
    }
    results.append(event)
    branches = []
    for obstacle in sorted(set(active)):
        tangent64 = d64[0,obstacle] + g64[0,obstacle] @ (poses[1]-poses[0])
        branches.append({'obstacle_index':int(obstacle),
                        'anchor_distance_mm':d64[0,obstacle]*1000,
                        'returned_distance_mm':d64[1,obstacle]*1000,
                        'actual_distance_mm':d64[2,obstacle]*1000,
                        'branch_tangent_at_returned_mm':tangent64*1000,
                        'same_branch_taylor_error_mm':(tangent64-d64[1,obstacle])*1000})
    details.append({**event, 'nominal_state0':nominal[0].tolist(), 'nominal_state1':nominal[1].tolist(),
                    'returned_state0':returned[0].tolist(), 'returned_state1':returned[1].tolist(),
                    'actual_after_pose':after.tolist(), 'consistent_state1':consistent.tolist(),
                    'nominal_gradient':gradient.tolist(), 'risk_terms':risk_terms.tolist(),
                    'risk_gradient_at_nominal':np.asarray(gradient_r).ravel().tolist(),
                    'branches':branches})
    print(json.dumps(event), flush=True)
output = Path(__file__).resolve().parent
with (output/'entry_decomposition.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(results[0]))
    writer.writeheader(); writer.writerows(results)
(output/'entry_details.json').write_text(json.dumps(details,indent=2)+'\n')
hashes = json.loads((ROOT/'source_hashes.json').read_text())
assert all(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
validation = {'new_navigation_or_solver_runs':0,'events':len(results),'returned_states_directly_logged':True,
              'frozen_sources_unchanged':True,'independent_shapely_distances':3*3*len(obstacles),
              'max_qp_dynamics_equality_error':max(r['qp_dynamics_replay_max_error'] for r in results),
              'max_rk4_replay_error':max(r['physical_rk4_replay_error'] for r in results),
              'max_independent_geometry_error_mm':max(r['independent_geometry_max_error_mm'] for r in results),
              'max_float32_vs_float64_phi_error_mm':max(r['float32_vs_float64_phi_max_error_mm'] for r in results)}
(output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation,indent=2))
