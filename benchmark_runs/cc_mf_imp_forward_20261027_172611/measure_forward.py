"""Measure CC-MF-IMP PSDF preprocessing without running an OCP or plant.

Run with the existing CUDA environment, from any working directory:
/home/taek111/anaconda3/envs/pp/bin/python <this file>
"""

import argparse
import csv
from dataclasses import fields
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

OUTPUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.mf_implicit import root_status_name


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def gpu_snapshot():
    result = subprocess.run(
        ["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,driver_version,"
         "memory.total,memory.used,utilization.gpu", "--format=csv"],
        check=False, capture_output=True, text=True,
    )
    return result.stdout.strip()


def describe_result(result):
    required = result.mf_required.cpu().numpy()
    valid = result.root_valid.cpu().numpy()
    statuses = result.root_status.cpu().numpy()
    iterations = result.root_iterations.cpu().numpy()
    residual = result.root_residual.cpu().numpy()
    return {
        "required_stages": int(required.sum()),
        "valid_required_stages": int((valid & required).sum()),
        "failed_stages": np.flatnonzero(required & ~valid).tolist(),
        "root_status": [root_status_name(value) for value in statuses],
        "root_iterations": iterations.tolist(),
        "max_abs_root_residual": (
            float(np.max(np.abs(residual[required]))) if required.any() else None
        ),
        "feature_count": result.feature_count.cpu().tolist(),
        "minimum_psdf_m": float(result.phi.min()),
        "margin_m": [float(value) if np.isfinite(value) else None
                     for value in result.margin.cpu().numpy()],
    }


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)), "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()), "max_ms": float(values.max()),
        "std_ms": float(values.std()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=320)
    parser.add_argument("--warmup", type=int, default=40)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 1:
        raise ValueError("samples and warmup must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested production-path measurement")
    device = torch.device("cuda:0")
    source = ROOT / "benchmark_runs/maze_rectangle_dd_perturb_four_methods_5trials_20261026_165452"
    data = source / "runs/config03/trial_0001/data"
    geometry_path = source / "geometry.json"
    trial_path = data / "trial_history_MPC_cc_mf_rectangle_maze_startseed0.csv"
    covariance_path = data / "covariance_trajectory.csv"
    start_path = data / "start_pose_MPC_cc_mf_rectangle_maze_startseed0.json"
    config_path = ROOT / "config/config_cc_mf_imp.yaml"
    geometry = json.loads(geometry_path.read_text())
    config = yaml.safe_load(config_path.read_text())["cc_mf_imp"]
    horizon = 20
    H = horizon + 1
    epsilon = np.full(H, config["chance_epsilon"], dtype=np.float64)
    requested = np.ones(H, dtype=bool)
    requested[0] = False
    kwargs = {
        "d_mf_mask": float(config["d_mf_mask"]),
        "root_max_iter": config["mf_imp_root_max_iter"],
        "root_atol_m": config["mf_imp_root_atol_m"],
        "root_rtol": config["mf_imp_root_rtol"],
        "root_prob_atol": config["mf_imp_root_prob_atol"],
        "return_frozen_features": False,
    }
    d_min = float(config["d_min"])
    wrapper = AugmentedPSDFWrapper(
        torch.tensor(geometry["footprint"], dtype=torch.float32, device=device),
        K_max=20, E_max=100, device=str(device),
    )
    polygons = [np.asarray(polygon) for polygon in geometry["obstacles"]]
    wrapper.update_edge_clusters(polygons, [np.roll(polygon, -1, axis=0) for polygon in polygons])
    xi0 = [math.log(.0002), math.log(.00025), math.log(.00025), 0.0]
    nominal = np.tile([*geometry["nominal_start"], 0.0, *xi0], (H, 1))
    cases = [{"label": "nominal_start", "kind": "nominal", "z": nominal}]

    initial = json.loads(start_path.read_text())["initial_pose"]
    trajectory = read_rows(trial_path)
    poses = np.asarray([initial] + [
        [float(row[key]) for key in ("x", "y", "theta")] for row in trajectory
    ])
    covariances = {}
    for row in read_rows(covariance_path):
        cycle = int(round(float(row["time"]) / geometry["dt_s"]))
        stage = int(row["stage"])
        sf, sl, st, cross = [float(row[key]) for key in (
            "sigma_f", "sigma_l", "sigma_psi", "P_lpsi",
        )]
        coupling = cross / sl
        conditional = st * st - coupling * coupling
        if conditional <= 0:
            raise ValueError("recorded covariance is not positive definite")
        covariances.setdefault(cycle, np.empty((H, 4)))[stage] = [
            math.log(sf), math.log(sl), .5 * math.log(conditional), coupling,
        ]
    for cycle in np.linspace(0, len(poses) - H, 16, dtype=int):
        z = np.empty((H, 8), dtype=np.float64)
        z[:, :3] = poses[cycle:cycle + H]
        z[:, 3] = np.r_[0.0, np.linalg.norm(np.diff(z[:, :2], axis=0), axis=1).cumsum()]
        z[:, 4:] = covariances[int(cycle)]
        cases.append({"label": f"replay_cycle_{cycle:03d}", "kind": "replay", "z": z})

    for case in cases:
        case["z_gpu"] = torch.as_tensor(case["z"], device=device)
        case["poses_gpu"] = case["z_gpu"][:, :3].to(torch.float32).contiguous()
    epsilon_gpu = torch.as_tensor(epsilon, device=device)
    requested_gpu = torch.as_tensor(requested, device=device)
    A = wrapper.A[:wrapper.active_clusters]
    B = wrapper.B[:wrapper.active_clusters]
    mask = wrapper.mask[:wrapper.active_clusters]

    def geometry_forward(case):
        return wrapper.psdf._forward_impl(A, B, mask, case["poses_gpu"], return_features=True)

    def implicit_gpu(case):
        return wrapper.forward_mf_imp(
            case["z_gpu"], epsilon_gpu, d_min,
            stage_requested=requested_gpu, **kwargs,
        )

    def implicit_host(case):
        return wrapper.forward_mf_imp(
            case["z"], epsilon, d_min, stage_requested=requested, **kwargs,
        )

    def implicit_host_summary(case):
        result = implicit_host(case)
        arrays = {
            field.name: getattr(result, field.name).detach().cpu().numpy()
            for field in fields(result)
            if isinstance(getattr(result, field.name), torch.Tensor)
        }
        arrays["root_status"] = np.asarray([
            root_status_name(value) for value in arrays["root_status"]
        ], dtype=object)
        return result, arrays

    functions = {
        "geometry_with_features": geometry_forward,
        "implicit_forward_gpu_inputs": implicit_gpu,
        "implicit_forward_cpu_inputs": implicit_host,
        "implicit_forward_cpu_inputs_and_summary": implicit_host_summary,
    }
    torch.cuda.synchronize()
    started = time.perf_counter()
    cold_result = implicit_host(cases[0])
    torch.cuda.synchronize()
    first_call_ms = 1000 * (time.perf_counter() - started)
    cold_description = describe_result(cold_result)
    print(f"First forward call: {first_call_ms:.3f} ms", flush=True)
    del cold_result

    validity = {}
    for case in cases:
        result = implicit_host(case)
        validity[case["label"]] = describe_result(result)
    del result
    (OUTPUT / "input_validity.json").write_text(json.dumps(validity, indent=2, allow_nan=False))
    failed_cases = [name for name, item in validity.items() if item["failed_stages"]]
    print(f"Validated {len(cases)} input batches; failed batches: {failed_cases}", flush=True)

    inputs = {
        case["label"]: {"kind": case["kind"], "z": case["z"].tolist()}
        for case in cases
    }
    (OUTPUT / "inputs.json").write_text(json.dumps(inputs, indent=2))
    shutil.copy2(geometry_path, OUTPUT / "geometry.json")
    shutil.copy2(config_path, OUTPUT / "config.yaml")
    source_paths = [ROOT / path for path in (
        "models/augmented_psdf.py", "models/augmented_psdf_wrapper.py",
        "models/mf_implicit.py", "control/cc_mf_imp_optimizer.py",
    )]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source_paths}
    environment = {
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "python": sys.executable, "python_version": sys.version,
        "platform": platform.platform(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(device), "torch_num_threads": torch.get_num_threads(),
        "nvidia_before": gpu_snapshot(), "source_sha256": hashes,
        "horizon": horizon, "stage_batch_size": H, "active_clusters": wrapper.active_clusters,
        "K_max": wrapper.K_max, "E_max": wrapper.E_max,
        "geometry_dtype": "float32", "risk_dtype": "float64",
        "warmup_per_boundary": args.warmup, "samples_per_boundary": args.samples,
        "input_cases": len(cases), "configuration": config,
        "geometry_source": str(geometry_path), "trajectory_source": str(trial_path),
        "covariance_source": str(covariance_path),
        "input_note": "Replay batches use recorded executed future poses and returned "
                      "covariance stages as realistic timing inputs, not CC-MF-IMP OCP predictions.",
        "measurement_note": "Per-call synchronized perf_counter wall time is primary. "
                            "CUDA-event elapsed intervals also contain eager submission gaps; "
                            "they are not a sum of kernel-only busy times. Setup, obstacle update, "
                            "file I/O, solver and plant integration are outside timed calls.",
        "first_call_ms": first_call_ms, "first_call_validity": cold_description,
        "failed_input_batches": failed_cases,
    }
    (OUTPUT / "environment.json").write_text(json.dumps(environment, indent=2))

    rows = []
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for name, function in functions.items():
        for sample in range(args.warmup):
            result = function(cases[sample % len(cases)])
        torch.cuda.synchronize()
        del result
        print(f"Measuring {name}: {args.samples} calls", flush=True)
        for sample in range(args.samples):
            case = cases[sample % len(cases)]
            torch.cuda.synchronize()
            start_event.record()
            started = time.perf_counter()
            result = function(case)
            end_event.record()
            end_event.synchronize()
            wall_ms = 1000 * (time.perf_counter() - started)
            gpu_ms = start_event.elapsed_time(end_event)
            rows.append({
                "boundary": name, "sample": sample, "input_case": case["label"],
                "input_kind": case["kind"], "wall_ms": wall_ms, "cuda_event_ms": gpu_ms,
                "required_stages": validity[case["label"]]["required_stages"],
                "failed_stages": len(validity[case["label"]]["failed_stages"]),
            })
            del result
        current = [row["wall_ms"] for row in rows if row["boundary"] == name]
        print(json.dumps({"boundary": name, **summarize(current)}), flush=True)
        with (OUTPUT / "samples.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summaries = []
    for name in functions:
        selected = [row for row in rows if row["boundary"] == name]
        for scope in ("all", "nominal", "replay"):
            subset = selected if scope == "all" else [row for row in selected if row["input_kind"] == scope]
            if not subset:
                continue
            wall = summarize([row["wall_ms"] for row in subset])
            gpu = summarize([row["cuda_event_ms"] for row in subset])
            summaries.append({
                "boundary": name, "scope": scope, **wall,
                "cuda_event_mean_ms": gpu["mean_ms"], "cuda_event_p95_ms": gpu["p95_ms"],
                "samples_with_failed_roots": sum(row["failed_stages"] > 0 for row in subset),
            })
    with (OUTPUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (OUTPUT / "summary.json").write_text(json.dumps(summaries, indent=2))
    environment["nvidia_after"] = gpu_snapshot()
    environment["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    environment["sources_changed_during_run"] = [
        name for name, digest in hashes.items()
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
    ]
    (OUTPUT / "environment.json").write_text(json.dumps(environment, indent=2))
    print(f"RESULTS_DIR={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
