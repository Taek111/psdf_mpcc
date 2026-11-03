"""Offline geometry audit of recorded poses only; no dynamics rollout or solve."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN / 'source_snapshot'))
import numpy as np
import torch
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.geometry_utils import polygon_to_edges
from models.dd import DifferentialDriveRectangleGeometry
from sim.simulation_mpc import simulation_mpc

torch.set_num_threads(6)
_, _, grid, obstacles = simulation_mpc().create_env('maze')
A, B = zip(*(polygon_to_edges(obstacle) for obstacle in obstacles))
results = []
for width, seed in ((128, 3), (129, 1), (129, 3), (129, 4)):
    run = RUN / f'width_{width}mm/cc_mf/seed_{seed}'
    cycles = list(csv.DictReader((run / 'cycles.csv').open()))
    k = next(i for i, row in enumerate(cycles) if row['solve_mode'] == 'safe_stop_stage0')
    logs = list(csv.DictReader(next((run / 'data').glob('cc_mf_cycle_*.csv')).open()))
    cp = [json.loads(line) for line in (run / 'animation_checkpoint.jsonl').open()]
    # cp[0] is initial/global path. cp[j+1] is after applied cycle j.
    # Cycle k-1 uses retained (unshifted) stage1 from cycle k-2.
    nominal = np.array(cp[k-1]['optimized_traj'][1][:3])
    predicted = np.array(cp[k]['optimized_traj'][1][:3])
    before = np.array(cp[k-1]['pose'])
    actual = np.array(cp[k]['pose'])
    poses = np.stack((before, nominal, predicted, actual))
    geometry = DifferentialDriveRectangleGeometry(.15, width / 1000, 0)
    verts = torch.tensor(geometry._region.get_ccw_vertices(), dtype=torch.float32, device='cuda')
    wrapper = AugmentedPSDFWrapper(verts, K_max=20, E_max=100, device='cuda')
    wrapper.update_edge_clusters(list(A), list(B))
    with torch.no_grad():
        phi_t, gradient_t = wrapper(torch.tensor(poses, dtype=torch.float32, device='cuda'))
        clusters_t, cluster_grad_t = wrapper.forward_clusters(torch.tensor(poses, dtype=torch.float32, device='cuda'))
    phi, gradient = phi_t.cpu().numpy(), gradient_t.cpu().numpy()
    clusters = clusters_t.cpu().numpy()
    cluster_gradients = cluster_grad_t.cpu().numpy()
    affine = float(phi[1]) + gradient[1].astype(float) @ (predicted - nominal)
    nominal_logged = float(logs[k-1]['psdf'])
    actual_logged = float(logs[k]['psdf'])
    assert abs(float(phi[0]) - nominal_logged) < 2e-7
    assert abs(float(phi[3]) - actual_logged) < 2e-7
    nearest = np.argmin(clusters, axis=1)
    new_obstacle = int(nearest[3])
    new_affine = float(clusters[1, new_obstacle]) + cluster_gradients[1, new_obstacle].astype(float) @ (predicted - nominal)
    result = {
        'new_obstacle_nominal_phi_mm': float(clusters[1, new_obstacle]) * 1000,
        'new_obstacle_nominal_rank': int(np.where(np.argsort(clusters[1]) == new_obstacle)[0][0]) + 1,
        'new_obstacle_affine_candidate_phi_mm': new_affine * 1000,
        'width_m': width / 1000, 'seed': seed, 'first_stop_cycle': k,
        'first_stop_time_s': float(logs[k]['time']),
        'before_phi_mm': float(phi[0]) * 1000,
        'nominal_stage1_phi_mm': float(phi[1]) * 1000,
        'affine_stage1_phi_mm': affine * 1000,
        'predicted_stage1_exact_phi_mm': float(phi[2]) * 1000,
        'actual_next_phi_mm': float(phi[3]) * 1000,
        'geometry_overestimate_mm': (affine - float(phi[2])) * 1000,
        'dynamics_phi_error_mm': (float(phi[2]) - float(phi[3])) * 1000,
        'predicted_actual_xy_error_um': float(np.linalg.norm(predicted[:2] - actual[:2])) * 1e6,
        'nominal_to_predicted_delta_theta_rad': float(predicted[2] - nominal[2]),
        'nominal_to_predicted_delta_xy_mm': ((predicted[:2] - nominal[:2]) * 1000).tolist(),
        'nearest_obstacle_indices_before_nominal_predicted_actual': nearest.tolist(),
        'nominal_gradient': gradient[1].tolist(),
        'before_pose': before.tolist(), 'nominal_pose': nominal.tolist(),
        'predicted_pose': predicted.tolist(), 'actual_pose': actual.tolist(),
        'last_control': json.loads(logs[k-1]['u0']),
        'stage1_nominal_mf_residual': json.loads(logs[k-1]['mf_icdf_residual_nominal'])[1],
        'stage1_slack': json.loads(logs[k-1]['mf_slack'])[1],
    }
    results.append(result)
    print(json.dumps(result), flush=True)
(RUN / 'stop_diagnosis/recorded_pose_geometry.json').write_text(json.dumps(results, indent=2) + '\n')
