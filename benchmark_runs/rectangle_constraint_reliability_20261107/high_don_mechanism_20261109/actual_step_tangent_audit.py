"""Current-pose PSDF tangent on two saved endpoints; no simulation or probes."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.analyze_sf_mf_reliability import make_wrapper
from utils.offline_cc_margin import _snapshot_imports
from utils.offline_feature_selectors import extract_feature_selectors


def main():
    torch.set_num_threads(2)
    suite = ROOT / 'benchmark_runs/rectangle_constraint_reliability_20261107/rectangle_don_sweep_50_20261105_current_sf'
    directory = suite / 'widths/width_127p5mm/runs/config03/trial_0011/data'
    trace_path = directory / 'raw_trace.npz'
    metadata_path = directory / 'raw_trace_metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    trace = np.load(trace_path, allow_pickle=False)
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == metadata['raw_trace_sha256']
    step = 46
    poses = trace['poses'][step:step + 2]
    times = trace['pose_times_s'][step:step + 2]
    delta = poses[1] - poses[0]
    np.testing.assert_allclose(times, [4.6, 4.7], atol=1e-12, rtol=0)
    assert metadata['solver_status_infos'][step]['success']
    assert not metadata['solver_status_infos'][step + 1]['solver_called']
    records = []
    with _snapshot_imports(suite / 'source_snapshot') as (module, _, _):
        for dtype in (torch.float64, torch.float32):
            wrapper = make_wrapper(module, metadata, dtype)
            select = extract_feature_selectors(wrapper, poses, verify=True)
            phi, gradient = select['phi'], select['world_gradient']
            phi_clusters, gradient_clusters = wrapper.forward_clusters(torch.tensor(poses, dtype=dtype))
            pc, gc = phi_clusters.numpy().astype(float), gradient_clusters.numpy().astype(float)
            tangent = float(phi[0] + np.dot(gradient[0], delta))
            cluster_tangents = pc[0] + gc[0] @ delta
            active = [int(index) for index in np.argmin(pc, axis=1)]
            detail = []
            for cluster in sorted(set(active)):
                detail.append({
                    'cluster': cluster,
                    'signed_psdf_before_mm': float(pc[0, cluster] * 1000),
                    'signed_psdf_after_mm': float(pc[1, cluster] * 1000),
                    'current_tangent_end_mm': float(cluster_tangents[cluster] * 1000),
                    'world_gradient_before': gc[0, cluster].tolist(),
                    'world_gradient_after': gc[1, cluster].tolist(),
                    'obstacle_polygon_world_m': metadata['geometry']['obstacle_polygons'][cluster],
                })
            records.append({
                'geometry_dtype': str(dtype),
                'canonical_id_before': str(select['canonical_id'][0]),
                'canonical_id_after': str(select['canonical_id'][1]),
                'active_cluster_before': active[0], 'active_cluster_after': active[1],
                'phi_before_mm': float(phi[0] * 1000),
                'phi_after_mm': float(phi[1] * 1000),
                'current_active_tangent_end_mm': tangent * 1000,
                'actual_minus_current_tangent_mm': float((phi[1] - tangent) * 1000),
                'world_gradient_before': gradient[0].tolist(),
                'world_gradient_after': gradient[1].tolist(),
                'critical_cluster_details': detail,
                'closest_three_clusters_before': [
                    {'cluster': int(i), 'signed_psdf_mm': float(pc[0, i] * 1000)}
                    for i in np.argsort(pc[0])[:3]],
                'closest_three_clusters_after': [
                    {'cluster': int(i), 'signed_psdf_mm': float(pc[1, i] * 1000)}
                    for i in np.argsort(pc[1])[:3]],
            })
    sources = [Path(__file__), trace_path, metadata_path,
               ROOT / 'scripts/analyze_sf_mf_reliability.py',
               ROOT / 'utils/offline_feature_selectors.py',
               suite / 'source_snapshot/models/augmented_psdf.py',
               suite / 'source_snapshot/models/augmented_psdf_wrapper.py']
    report = {
        'source': 'cc_sf', 'seed': 10, 'don': .85, 'applied_step': step,
        'recorded_endpoint_times_s': times.tolist(),
        'recorded_poses_before_after': poses.tolist(),
        'actual_pose_delta_x_y_theta': delta.tolist(),
        'applied_input_v_omega_virtual_speed': trace['applied_inputs'][step].tolist(),
        'applied_step_solver_status': metadata['solver_status_infos'][step],
        'next_step_solver_status': metadata['solver_status_infos'][step + 1],
        'd_col_mm': metadata['optimizer_parameters']['d_col'] * 1000,
        'formula': 'phi_current + gradient_current dot (actual_pose_next - actual_pose_current)',
        'interpretation': 'The active obstacle/feature changes between the two recorded endpoints. The current active PSDF tangent can predict increasing clearance while another obstacle becomes the signed minimum. This is a diagnostic of a current-pose tangent only, NOT reconstruction of the native stage-1 optimizer row, whose retained nominal anchor and returned state were not logged. Both deployed methods share a single signed-PSDF geometric guard; this example does not establish the causal source of the SF/MF success difference.',
        'geometry_diagnostic_only': True,
        'planning_covariance_evaluated': False,
        'interpolated_pose_used': False,
        'local_probe_used': False,
        'simulation_executed': False,
        'solver_executed': False,
        'precision_checks': records,
        'sources': [{'path': str(path.relative_to(ROOT)),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in sources],
    }
    path = OUT / 'actual_step_tangent_audit.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
