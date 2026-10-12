"""Reproduce the completed scalar/augmented parity validation (no timing).

Run from the project root with the same Python environment used by measure.py:
    /home/taek111/anaconda3/envs/pp/bin/python \
        benchmark_runs/rollout_check_latency_20261012/validate_phi_only.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_DIR))
from measure import make_obstacles, make_wrapper, EXACT
from phi_only import PhiOnlyPSDFWrapper, compute_exact_phi


def main():
    torch.set_num_threads(1)
    start, obstacles = make_obstacles()
    rng = np.random.default_rng(0)
    poses = np.column_stack((
        rng.uniform(.07, 1.9, 128),
        rng.uniform(.07, .83, 128),
        rng.uniform(-np.pi, np.pi, 128),
    ))
    poses[0] = start
    poses[1] = [1.886143, .778253, -.864363]
    # Include obstacle centers to exercise the removed legacy containment branch.
    poses = np.concatenate((
        poses,
        np.column_stack((obstacles.mean(axis=1), np.zeros(len(obstacles)))),
    ))
    for device in ['cpu', 'cuda']:
        if device == 'cuda' and not torch.cuda.is_available():
            continue
        for width in [.11, .12]:
            vertices = [
                [-.075, -width/2], [.075, -width/2],
                [.075, width/2], [-.075, width/2],
            ]
            for edges in [4, 100]:
                ref = make_wrapper(vertices, obstacles, device, edges)
                phi_only = PhiOnlyPSDFWrapper(
                    torch.tensor(vertices, dtype=torch.float32, device=device),
                    20, edges, device,
                )
                phi_only.update_edge_clusters(ref.A[:17], ref.B[:17], ref.mask[:17])
                max_error = 0.
                for i in range(0, len(poses), 8):
                    batch = poses[i:i+8]
                    wanted, _ = EXACT(SimpleNamespace(psdf_wrapper=ref), batch)
                    actual = compute_exact_phi(phi_only, batch)
                    np.testing.assert_allclose(actual, wanted, atol=1e-6, rtol=0)
                    max_error = max(max_error, float(np.max(np.abs(actual-wanted))))
                print({
                    'device': device, 'width_m': width, 'edges': edges,
                    'poses': len(poses), 'max_phi_error_m': max_error,
                }, flush=True)


if __name__ == '__main__':
    main()
