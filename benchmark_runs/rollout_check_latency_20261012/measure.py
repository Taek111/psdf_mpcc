"""Measure additional first-input safety-check latency; never run an MPC solver.

Uses the current repository's RK4, exact CC-MF PSDF adapter (AST extracted to
avoid solver imports), and maze geometry. Compact E=4 is a benchmark candidate,
not a controller change. All 17 obstacles remain present in every evaluation.
"""

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.dd_dynamics import DifferentialDriveDynamics


def extract_method(path, class_name, method_name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[method_name]


class RectangleBounds:
    def __init__(self, left, right, bottom, top):
        self.vertices = [[left, bottom], [right, bottom], [right, top], [left, top]]


def make_obstacles():
    create_env = extract_method(
        "sim/simulation_mpc.py", "simulation_mpc", "create_env",
        {"np": np, "math": math, "RectangleRegion": RectangleBounds},
    )
    start, _, _, obstacles = create_env(None, "maze")
    assert len(obstacles) == 17
    return start, np.asarray([o.vertices for o in obstacles], dtype=np.float32)


EXACT = extract_method(
    "control/cc_mf_optimizer.py", "CCMFOptimizer", "_compute_exact_psdf", {"np": np, "torch": torch},
)


def make_wrapper(vertices, obstacle_vertices, device, edges):
    wrapper = AugmentedPSDFWrapper(
        torch.tensor(vertices, dtype=torch.float32, device=device), 20, edges, device,
    )
    starts = [torch.tensor(v, dtype=torch.float32, device=device) for v in obstacle_vertices]
    wrapper.update_edge_clusters(starts, [torch.roll(a, -1, 0) for a in starts])
    return wrapper


def poses_at(q, u, points):
    if points == 1:
        return DifferentialDriveDynamics.forward_dynamics(q, u, 0.1)[None, :]
    # Independent partial-duration RK4 from q: terminal pose matches plant RK4.
    result = np.empty((points, 3))
    result[0] = q
    for k in range(1, points):
        result[k] = DifferentialDriveDynamics.forward_dynamics(q, u, 0.1 * k / (points - 1))
    return result


def exact_check(wrapper, poses):
    phi, gradient = EXACT(SimpleNamespace(psdf_wrapper=wrapper), poses)
    # Keep the CPU acceptance decision, finite-result validation and transfers
    # inside the timed call. EXACT already checks phi and gradient finiteness.
    return bool(np.all(phi >= 0.001))


def summary(values):
    values = np.asarray(values) / 1e6
    return {
        "count": len(values), "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)), "max_ms": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=25)
    args = parser.parse_args()
    output_dir = Path(__file__).resolve().parent
    start, obstacles = make_obstacles()
    vertices = [[-.075, -.060], [.075, -.060], [.075, .060], [-.075, .060]]
    rng = np.random.default_rng(0)
    # These are synthetic timing inputs, not the missing saved replay.
    poses = np.column_stack((rng.uniform(.07, 1.9, 128), rng.uniform(.07, .83, 128), rng.uniform(-math.pi, math.pi, 128)))
    poses[0] = start
    poses[1] = [1.886143, .778253, -.864363]
    inputs = np.column_stack((rng.uniform(-.7, .7, 128), rng.uniform(-1.2, 1.2, 128)))
    default_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    source_files = [
        "models/augmented_psdf.py", "models/augmented_psdf_wrapper.py",
        "models/dd_dynamics.py", "control/cc_mf_optimizer.py", "sim/simulation_mpc.py",
    ]
    environment = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "cpu": next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")),
        "numpy": np.__version__, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "default_torch_threads": default_threads,
        "thread_environment": {k: os.getenv(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_files},
        "footprint_m": {"length": .15, "width": .12}, "obstacles": 17,
        "dt_s": .1, "seed": 0, "points": [1, 3, 6, 11],
        "sampling": "1: endpoint only; otherwise initial pose plus evenly spaced partial RK4 points through 0.1 s",
        "timing": vars(args), "original_replay_available": False,
        "includes": "NumPy/tensor conversions, exact phi+gradient, finite validation, GPU-to-CPU return, boolean threshold",
        "excludes": "one-time wrapper construction/obstacle upload, MPC solve, fallback construction, logs",
        "cuda_timing": "synchronize before each timed call; CPU result extraction inside exact adapter synchronizes completion",
    }
    cpu100 = make_wrapper(vertices, obstacles, "cpu", 100)
    cpu4 = make_wrapper(vertices, obstacles, "cpu", 4)
    parity = []
    max_phi_error = 0.0
    max_grad_error = 0.0
    for offset in range(0, len(poses), 8):
        batch = poses[offset:offset+8]
        p100, g100 = EXACT(SimpleNamespace(psdf_wrapper=cpu100), batch)
        p4, g4 = EXACT(SimpleNamespace(psdf_wrapper=cpu4), batch)
        np.testing.assert_allclose(p4, p100, rtol=0, atol=1e-6)
        np.testing.assert_allclose(g4, g100, rtol=0, atol=1e-5)
        max_phi_error = max(max_phi_error, float(np.max(np.abs(p100-p4))))
        max_grad_error = max(max_grad_error, float(np.max(np.abs(g100-g4))))
    representative_phi, _ = EXACT(SimpleNamespace(psdf_wrapper=cpu100), poses[1])
    parity.append({"device": "cpu", "poses": len(poses), "max_phi_error_m": max_phi_error, "max_gradient_error": max_grad_error})
    environment["representative_phi_m"] = representative_phi
    environment["padding_parity"] = parity
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    records = []
    aggregated = []

    def measure(label, callback, device, edges, points, threads):
        for i in range(args.warmup):
            callback(i % len(poses))
        samples = []
        for run in range(args.rounds):
            for i in range(args.repeats):
                index = (run * args.repeats + i) % len(poses)
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                callback(index)
                elapsed = time.perf_counter_ns() - t0
                samples.append(elapsed)
                records.append({"operation": label, "device": device, "edges": edges, "points": points, "torch_threads": threads, "round": run, "sample": i, "elapsed_ns": elapsed})
        row = {"operation": label, "device": device, "edges": edges, "points": points, "torch_threads": threads, **summary(samples)}
        aggregated.append(row)
        print(json.dumps(row), flush=True)

    for points in environment["points"]:
        measure("rk4_only", lambda i, n=points: poses_at(poses[i], inputs[i], n), "cpu", 0, points, 1)
    configurations = [("cpu", 1), ("cpu", default_threads)]
    if torch.cuda.is_available():
        configurations.append(("cuda", 1))
    for device, threads in configurations:
        torch.set_num_threads(threads)
        for edges in (100, 4):
            wrapper = make_wrapper(vertices, obstacles, device, edges)
            if device == "cuda":
                candidate, _ = EXACT(SimpleNamespace(psdf_wrapper=wrapper), poses[:16])
                reference, _ = EXACT(SimpleNamespace(psdf_wrapper=cpu100), poses[:16])
                np.testing.assert_allclose(candidate, reference, rtol=0, atol=1e-6)
                parity.append({"device": device, "edges": edges, "poses": 16, "max_phi_error_m": float(np.max(np.abs(candidate-reference)))})
            for points in environment["points"]:
                precomputed = [poses_at(q, u, points) for q, u in zip(poses, inputs)]
                measure("psdf_and_decision", lambda i: exact_check(wrapper, precomputed[i]), device, edges, points, threads)
                measure("rollout_and_check", lambda i: exact_check(wrapper, poses_at(poses[i], inputs[i], points)), device, edges, points, threads)
    environment["padding_parity"] = parity
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(aggregated, indent=2) + "\n")
    for filename, rows in (("samples.csv", records), ("summary.csv", aggregated)):
        with (output_dir / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
