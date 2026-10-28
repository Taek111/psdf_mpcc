"""Compare original PSDF/CC-MF with CC-MF-IMP on the same CUDA input batches."""

import csv
from dataclasses import fields
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
sys.path.insert(0, str(ROOT))

from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.mf_implicit import compute_mf_imp, root_status_name
from models.psdf_wrapper import psdfWrapper


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": len(values), "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()), "max_ms": float(values.max()),
    }


@torch.no_grad()
def main():
    prior = ROOT / "benchmark_runs/cc_mf_imp_forward_20261027_172611_interleaved"
    geometry = json.loads((prior / "geometry.json").read_text())
    saved = json.loads((prior / "inputs.json").read_text())
    prior_env = json.loads((prior / "environment.json").read_text())
    config = prior_env["configuration"]
    device = torch.device("cuda:0")
    warmup, samples = 40, 340
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    verts = torch.tensor(geometry["footprint"], dtype=torch.float32, device=device)
    augmented = AugmentedPSDFWrapper(verts, 20, 100, str(device))
    legacy = psdfWrapper(verts, 20, 100, str(device))
    polygons = [np.asarray(polygon) for polygon in geometry["obstacles"]]
    augmented.update_edge_clusters(polygons, [np.roll(p, -1, axis=0) for p in polygons])
    polygon_tensors = [torch.as_tensor(p, dtype=torch.float32, device=device) for p in polygons]
    legacy.update_edge_clusters(polygon_tensors, [p.roll(-1, dims=0) for p in polygon_tensors])
    H = prior_env["stage_batch_size"]
    epsilon_cpu = np.full(H, config["chance_epsilon"], dtype=np.float64)
    epsilon32 = torch.as_tensor(epsilon_cpu, dtype=torch.float32, device=device)
    epsilon64 = torch.as_tensor(epsilon_cpu, dtype=torch.float64, device=device)
    stage_requested_cpu = np.ones(H, dtype=bool)
    stage_requested_cpu[0] = False
    stage_requested_gpu = torch.as_tensor(stage_requested_cpu, device=device)
    kwargs = {
        "d_mf_mask": config["d_mf_mask"], "root_max_iter": config["mf_imp_root_max_iter"],
        "root_atol_m": config["mf_imp_root_atol_m"], "root_rtol": config["mf_imp_root_rtol"],
        "root_prob_atol": config["mf_imp_root_prob_atol"],
    }
    d_min = config["d_min"]
    cases = []
    for name, entry in saved.items():
        z = np.asarray(entry["z"], dtype=np.float64)
        z64 = torch.as_tensor(z, device=device)
        z32 = z64.to(torch.float32)
        pose32 = z32[:, :3].contiguous()
        features = augmented.psdf._forward_impl(
            augmented.A[:augmented.active_clusters], augmented.B[:augmented.active_clusters],
            augmented.mask[:augmented.active_clusters], pose32, return_features=True,
        )
        cases.append({"name": name, "kind": entry["kind"], "z": z,
                      "z32": z32, "z64": z64, "pose32": pose32, "features": features})

    def legacy_distance(case):
        return legacy(case["pose32"])

    def existing_analytic(case):
        return augmented(case["pose32"])

    def existing_mf_gpu(case):
        return augmented.forward_mf(case["z32"], epsilon32, d_min)

    def existing_mf_cpu_summary(case):
        # Match CCMFOptimizer._compute_mf_affine_data, including original FP32
        # upload and its four-tensor summary retrieval. No OCP/path work.
        z_bar = torch.as_tensor(case["z"], dtype=augmented.A.dtype, device=device)
        epsilon = torch.as_tensor(epsilon_cpu, dtype=z_bar.dtype, device=device)
        result = augmented.forward_mf(z_bar, epsilon, d_min)
        return tuple(item.detach().cpu().numpy() for item in result)

    def implicit_gpu(case):
        return augmented.forward_mf_imp(
            case["z64"], epsilon64, d_min, stage_requested=stage_requested_gpu, **kwargs,
        )

    def implicit_cpu_summary(case):
        result = augmented.forward_mf_imp(
            case["z"], epsilon_cpu, d_min, stage_requested=stage_requested_cpu, **kwargs,
        )
        arrays = {
            item.name: getattr(result, item.name).detach().cpu().numpy()
            for item in fields(result) if isinstance(getattr(result, item.name), torch.Tensor)
        }
        arrays["root_status"] = np.asarray([
            root_status_name(value) for value in arrays["root_status"]
        ], dtype=object)
        return result, arrays

    def implicit_risk_only(case):
        # Exactly the current root/analytic-coefficient helper, with geometry
        # already evaluated once outside this boundary; no model changes.
        return compute_mf_imp(
            *case["features"], case["z64"], epsilon64, d_min,
            stage_requested=stage_requested_gpu, **kwargs,
        )

    functions = {
        "legacy_scalar_psdf_gpu": legacy_distance,
        "existing_analytic_psdf_phi_gradient_gpu": existing_analytic,
        "existing_cc_mf_forward_gpu": existing_mf_gpu,
        "existing_cc_mf_cpu_inputs_and_summary": existing_mf_cpu_summary,
        "cc_mf_imp_forward_gpu": implicit_gpu,
        "cc_mf_imp_cpu_inputs_and_summary": implicit_cpu_summary,
        "cc_mf_imp_root_and_coefficients_only_gpu": implicit_risk_only,
    }
    validity = []
    for case in cases:
        scalar = legacy_distance(case).squeeze(-1)
        phi, _ = existing_analytic(case)
        imp, arrays = implicit_cpu_summary(case)
        finite = all(torch.isfinite(value).all().item() for value in existing_mf_gpu(case))
        validity.append({
            "input_case": case["name"], "legacy_vs_analytic_max_abs_phi_diff_m":
                float((scalar - phi).abs().max()), "existing_mf_outputs_finite": bool(finite),
            "required_roots": int(arrays["mf_required"].sum()),
            "valid_required_roots": int((arrays["mf_required"] & arrays["root_valid"]).sum()),
        })
    (OUTPUT / "input_validity.json").write_text(json.dumps(validity, indent=2))
    source_names = ["models/psdf.py", "models/psdf_wrapper.py", "models/augmented_psdf.py",
                    "models/augmented_psdf_wrapper.py", "models/mf_implicit.py"]
    source_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                     for name in source_names}
    environment = {
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "python": sys.executable, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device), "platform": platform.platform(),
        "warmup_per_boundary": warmup, "samples_per_boundary": samples,
        "H": H, "K": augmented.active_clusters, "E_max": augmented.E_max,
        "inputs_source": str(prior / "inputs.json"), "geometry_source": str(prior / "geometry.json"),
        "schedule": "All boundaries shuffled per paired input round with random seed0",
        "note": "Synchronized per-call wall time. GPU scalar legacy PSDF returns distance only, "
                "not its production CPU+Jacobian pipeline. Augmented PSDF returns phi and analytic "
                "gradient. Old MF uses original FP32; IMP uses FP64 risk by design. Replay inputs "
                "are timing proxies from prior CC-MF execution, not new IMP OCP predictions.",
        "source_sha256": source_hashes,
    }
    (OUTPUT / "environment.json").write_text(json.dumps(environment, indent=2))
    for name, function in functions.items():
        for index in range(warmup):
            result = function(cases[index % len(cases)])
        torch.cuda.synchronize()
        del result
        print(f"Warmed {name}", flush=True)
    rows = []
    rng = random.Random(0)
    names = list(functions)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for index in range(samples):
        case = cases[index % len(cases)]
        order = names.copy()
        rng.shuffle(order)
        for name in order:
            torch.cuda.synchronize()
            begin.record()
            started = time.perf_counter()
            result = functions[name](case)
            end.record()
            end.synchronize()
            rows.append({
                "boundary": name, "sample": index, "input_case": case["name"],
                "input_kind": case["kind"], "wall_ms": (time.perf_counter()-started)*1000,
                "cuda_event_ms": begin.elapsed_time(end),
            })
            del result
        if (index + 1) % 50 == 0:
            print(f"Completed {index + 1}/{samples} rounds", flush=True)
    with (OUTPUT / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = [{"boundary": name, **stats([row["wall_ms"] for row in rows
                                            if row["boundary"] == name])} for name in names]
    with (OUTPUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (OUTPUT / "summary.json").write_text(json.dumps(summaries, indent=2))
    for summary in summaries:
        print(json.dumps(summary), flush=True)

    # Profiling is separate from all timing samples. Device-only busy time and
    # eager launch counts help interpret wall time; profiler timings themselves
    # must not be substituted for the unprofiled latency table above.
    try:
        from torch.profiler import ProfilerActivity, profile
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            implicit_risk_only(cases[0])
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(OUTPUT / "root_profile_trace.json"))
        averages = prof.key_averages()
        (OUTPUT / "root_profile_cpu.txt").write_text(
            averages.table(sort_by="self_cpu_time_total", row_limit=35)
        )
        cuda_events = [event for event in prof.events() if "CUDA" in str(event.device_type)]
        cpu_events = [event for event in prof.events() if "CPU" in str(event.device_type)]
        profile_summary = {
            "profiled_boundary": "implicit_risk_only, one nominal input batch",
            "cuda_event_count": len(cuda_events),
            "sum_cuda_event_duration_ms": sum(event.time_range.elapsed_us() for event in cuda_events)/1000,
            "cpu_event_count": len(cpu_events),
            "cuda_runtime_launch_count": sum(event.count for event in averages
                                             if event.key == "cudaLaunchKernel"),
            "caution": "Separate instrumented run; events can include transfers and profiling overhead.",
        }
        (OUTPUT / "root_profile_summary.json").write_text(json.dumps(profile_summary, indent=2))
        print("PROFILE", json.dumps(profile_summary), flush=True)
    except Exception as error:
        print(f"Optional profiling unavailable: {type(error).__name__}: {error}", flush=True)
    environment["sources_changed_during_run"] = [name for name, digest in source_hashes.items()
                                                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest]
    (OUTPUT / "environment.json").write_text(json.dumps(environment, indent=2))
    print(f"RESULTS_DIR={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
