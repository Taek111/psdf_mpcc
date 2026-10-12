"""Interleaved confirmation of existing checks versus gradient-free prototype."""

import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from measure import EXACT, make_obstacles, make_wrapper, poses_at, summary
from types import SimpleNamespace
from phi_only import PhiOnlyPSDFWrapper, compute_exact_phi


def main():
    out = Path(__file__).resolve().parent
    start, obstacles = make_obstacles()
    vertices = [[-.075, -.06], [.075, -.06], [.075, .06], [-.075, .06]]
    rng = np.random.default_rng(0)
    poses = np.column_stack((rng.uniform(.07, 1.9, 128), rng.uniform(.07, .83, 128), rng.uniform(-np.pi, np.pi, 128)))
    poses[0] = start
    poses[1] = [1.886143, .778253, -.864363]
    controls = np.column_stack((rng.uniform(-.7, .7, 128), rng.uniform(-1.2, 1.2, 128)))
    default_threads = torch.get_num_threads()
    configurations = [("cpu", 1, 4), ("cpu", default_threads, 100)]
    if torch.cuda.is_available():
        configurations.append(("cuda", default_threads, 100))
    samples, results, parity = [], [], []
    for device, threads, existing_edges in configurations:
        torch.set_num_threads(threads)
        existing = make_wrapper(vertices, obstacles, device, existing_edges)
        scalar = PhiOnlyPSDFWrapper(torch.tensor(vertices, dtype=torch.float32, device=device), 20, 4, device)
        starts = [torch.tensor(v, dtype=torch.float32, device=device) for v in obstacles]
        scalar.update_edge_clusters(starts, [torch.roll(a, -1, 0) for a in starts])
        max_error = 0.
        for offset in range(0, len(poses), 8):
            expected, _ = EXACT(SimpleNamespace(psdf_wrapper=existing), poses[offset:offset+8])
            actual = compute_exact_phi(scalar, poses[offset:offset+8])
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
            max_error = max(max_error, float(np.max(np.abs(expected-actual))))
        parity.append({"device": device, "threads": threads, "max_phi_error_m": max_error, "poses": len(poses)})

        def check(variant, points, index):
            rollout = poses_at(poses[index], controls[index], points)
            if variant == "existing_augmented":
                phi, _ = EXACT(SimpleNamespace(psdf_wrapper=existing), rollout)
            else:
                phi = compute_exact_phi(scalar, rollout)
            return bool(np.all(phi >= .001))

        cases = [(v, n) for v in ("existing_augmented", "phi_only_prototype") for n in (1, 6, 11)]
        for v, n in cases:
            for i in range(30):
                check(v, n, i)
        timings = {case: [] for case in cases}
        for i in range(300):
            for case_index in rng.permutation(len(cases)):
                v, n = cases[case_index]
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                check(v, n, i % len(poses))
                elapsed = time.perf_counter_ns() - t0
                timings[(v, n)].append(elapsed)
                samples.append({"variant": v, "device": device, "threads": threads, "edges": existing_edges if v == "existing_augmented" else 4, "points": n, "sample": i, "elapsed_ns": elapsed})
        for (v, n), elapsed in timings.items():
            row = {"variant": v, "device": device, "threads": threads, "edges": existing_edges if v == "existing_augmented" else 4, "points": n, **summary(elapsed)}
            results.append(row)
            print(json.dumps(row), flush=True)
    (out / "phi_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "phi_parity.json").write_text(json.dumps({"comparisons": parity, "prototype_sha256": hashlib.sha256((out / "phi_only.py").read_bytes()).hexdigest(), "notes": "warm 300 calls/case after 30 warmup; case order shuffled per iteration; synthetic poses; no solver; existing augmented includes gradients, prototype only phi; all 17 obstacles; dt=.1; initial pose included for points>1"}, indent=2) + "\n")
    for name, data in (("phi_samples.csv", samples), ("phi_summary.csv", results)):
        with (out / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
