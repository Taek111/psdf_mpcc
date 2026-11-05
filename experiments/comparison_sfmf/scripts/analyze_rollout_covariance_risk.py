"""Stage-1 risk sensitivity to exact one-step planning-covariance propagation.

Reads completed checkpoints and stage-1 caches. No controller or simulator is
instantiated. Saved poses and inputs remain unchanged; original caches are never
modified. The rollout covariance is a model reference, not measured uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml

from utils.offline_cc_margin import _maze_polygons, _snapshot_imports


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def cdf(z):
    return (0.5 * torch.special.erfc(-torch.as_tensor(z, dtype=torch.float64)
                                    / np.sqrt(2.0))).numpy()


def feature_risk(distance, q, mask, xi, d_min):
    variance = ((np.exp(xi[:, 0, None]) * q[..., 0]) ** 2
                + (np.exp(xi[:, 1, None]) * q[..., 1]
                   + xi[:, 3, None] * q[..., 2]) ** 2
                + (np.exp(xi[:, 2, None]) * q[..., 2]) ** 2)
    sigma = np.sqrt(variance)
    good = np.isfinite(distance) & np.isfinite(q).all(-1) & np.isfinite(sigma) & (sigma > 0)
    valid = ((~mask) | good).all(-1) & mask.any(-1)
    probability = np.zeros_like(distance)
    used = mask & good
    probability[used] = cdf((d_min - distance[used]) / sigma[used])
    return probability.sum(-1), probability.max(-1), valid


def checkpoint_arrays(path):
    inputs, post, stage0 = [], [], []
    with path.open() as stream:
        next(stream)  # Initial state/global path; no control call yet.
        for line in stream:
            states = np.asarray(json.loads(line)["optimized_traj"], dtype=float)
            inputs.append(states[1, 8:10])
            post.append(states[1, :8])
            stage0.append(states[0, :8])
    return np.asarray(inputs), np.asarray(post), np.asarray(stage0)


def covariance_reference(covariance, params, current_pose, inputs):
    p = SimpleNamespace(**params)
    if params.get("initial_pose_covariance") is None:
        initial = covariance.log_cholesky_from_covariance(covariance.initial_local_covariance(p))
        xi0 = np.broadcast_to(initial, (len(inputs), 4)).copy()
    else:
        xi0 = np.asarray([covariance.log_cholesky_from_covariance(
            covariance.initial_local_covariance(p, SimpleNamespace(_x=pose)))
            for pose in current_pose])
    dt = params["tf"] / params["horizon"]
    if not np.isclose(dt, 0.1, rtol=0, atol=1e-12):
        raise ValueError("This stage-1 analysis expects the recorded 0.1 s control period")
    robust_horizon = params.get("robust_horizon")
    if robust_horizon is not None and robust_horizon <= 0:
        return xi0, xi0.copy()
    return xi0, covariance.propagate_log_cholesky(
        xi0, inputs[:, 0], inputs[:, 1], np.zeros(len(inputs)), dt, p)


def make_wrapper(module, config, polygons, device):
    length, width = (config["rectangle_geometry"][key] for key in ("length", "width"))
    vertices = torch.tensor([[-length / 2, -width / 2], [length / 2, -width / 2],
                             [length / 2, width / 2], [-length / 2, width / 2]], dtype=torch.float32)
    wrapper = module.AugmentedPSDFWrapper(vertices, K_max=20, E_max=100, device=device)
    wrapper.update_edge_clusters([torch.tensor(p, dtype=torch.float32) for p in polygons],
                                 [torch.tensor(np.roll(p, -1, axis=0), dtype=torch.float32) for p in polygons])
    return wrapper


def trial_metrics(frame, epsilon):
    identity = {key: frame.iloc[0][key] for key in ("width_mm", "method", "seed")}
    normal = frame.post_valid.to_numpy(bool)
    valid = frame.common_probability_valid.to_numpy(bool)
    rows = []
    def put(model, metric, value, n):
        rows.append(dict(identity, covariance_model=model, metric=metric, value=value, n_samples=int(n)))
    for model in ("saved", "rollout"):
        for field in ("R", "pmax", "selected_p", "R_budget_residual"):
            sample = frame.loc[valid, model + "_" + field].to_numpy(float)
            for name, function in (("mean", np.mean), ("p05", lambda x: np.quantile(x, .05)),
                                   ("p95", lambda x: np.quantile(x, .95)), ("max", np.max)):
                put(model, field + "_" + name, function(sample) if len(sample) else np.nan, len(sample))
            if field != "R_budget_residual":
                put(model, field + "_exceedance_pct",
                    100 * np.mean(sample > epsilon + 1e-6) if len(sample) else np.nan, len(sample))
        geometry = frame.loc[normal, "guard_margin_mm"].to_numpy(float)
        geometry = geometry[np.isfinite(geometry)]
        put(model, "guard_margin_mm_mean", np.mean(geometry) if len(geometry) else np.nan, len(geometry))
        put(model, "guard_margin_mm_p05", np.quantile(geometry, .05) if len(geometry) else np.nan, len(geometry))
        put(model, "guard_violation_pct", 100 * np.mean(geometry < -.001) if len(geometry) else np.nan, len(geometry))
    for field in ("xi_defect_max_abs", "covariance_defect_frobenius_raw", "heading_variance_ratio_saved_to_rollout"):
        sample = frame.loc[normal, field].to_numpy(float)
        sample = sample[np.isfinite(sample)]
        put("comparison", field + "_max", np.max(sample) if len(sample) else np.nan, len(sample))
        put("comparison", field + "_p95", np.quantile(sample, .95) if len(sample) else np.nan, len(sample))
    lookup = {(r["covariance_model"], r["metric"]): r for r in rows}
    for metric in ("R_mean", "R_p95", "R_exceedance_pct", "pmax_exceedance_pct", "selected_p_exceedance_pct"):
        put("rollout_minus_saved", metric, lookup["rollout", metric]["value"] - lookup["saved", metric]["value"], valid.sum())
    return rows


@torch.no_grad()
def analyze_trial(root, cache, width, method, seed, module, covariance, polygons, device, batch_size):
    started = time.perf_counter()
    with np.load(cache, allow_pickle=False) as stream:
        d = {key: stream[key] for key in stream.files}
    expected = (width, method, seed, 1)
    actual = tuple(d[key].item() for key in ("width_mm", "method", "seed", "stage"))
    if actual != expected:
        raise ValueError(f"Cache identity mismatch: {actual} != {expected}")
    directory = root / f"width_{width:03d}mm/{method}/seed_{seed}"
    param_path, config_path = directory / "optimizer_parameters.json", directory / "config.yaml"
    params = json.loads(param_path.read_text())
    if params != json.loads(str(d["params_json"])):
        raise ValueError("Cache and recorded parameters differ")
    config = yaml.safe_load(config_path.read_text())
    if config["localization_error"]["enabled"]:
        raise ValueError("This analysis expects the recorded localization-error-disabled setting")
    checkpoint = directory / "animation_checkpoint.jsonl"
    inputs, saved_post, stage0 = checkpoint_arrays(checkpoint)
    history_path = directory / "data/trial_history.csv"
    history = pd.read_csv(history_path)
    applied = history[["v", "omega"]].to_numpy(float)
    n = len(d["time"])
    if len(inputs) != n or len(applied) != n:
        raise ValueError("Input/checkpoint/cache cycle counts differ")
    np.testing.assert_array_equal(d["cycle"], np.arange(n))
    np.testing.assert_allclose(inputs, applied, rtol=0, atol=1e-7)
    post_valid = np.asarray(d["post_valid"], bool)
    np.testing.assert_allclose(saved_post[post_valid], d["post_states"][post_valid], rtol=0, atol=1e-12)
    np.testing.assert_allclose(stage0[:, :3], d["current_pose"], rtol=0, atol=1e-6)
    xi0, rollout_xi = covariance_reference(covariance, params, d["current_pose"], inputs)
    np.testing.assert_allclose(stage0[post_valid, 4:8], xi0[post_valid], rtol=0, atol=1e-7)
    ids = np.flatnonzero(post_valid)
    saved_cov = covariance.covariance_from_log_cholesky(saved_post[ids, 4:8])
    rollout_cov = covariance.covariance_from_log_cholesky(rollout_xi[ids])
    frame = pd.DataFrame({"width_mm": width, "method": method, "seed": seed, "stage": 1,
                          "cycle": d["cycle"], "time_s": d["time"], "post_valid": post_valid,
                          "cached_probability_valid": d["post_probability_valid"],
                          "common_probability_valid": False,
                          "cached_post_phi_m": d["post_phi_m"], "cached_saved_R": d["post_R"]})
    columns = ["phi_m", "guard_margin_mm", "xi_defect_max_abs", "covariance_defect_frobenius_raw",
               "heading_variance_ratio_saved_to_rollout", "saved_heading_variance", "rollout_heading_variance",
               "gpu_rollout_R_check_abs_error", "saved_gpu_R_minus_cached_R"]
    for model in ("saved", "rollout"):
        columns += [model + "_" + key for key in ("R", "pmax", "selected_p", "R_budget_residual", "selected_sigma_m")]
    for column in columns:
        frame[column] = np.nan
    frame.loc[ids, "xi_defect_max_abs"] = np.max(np.abs(saved_post[ids, 4:8] - rollout_xi[ids]), axis=1)
    frame.loc[ids, "covariance_defect_frobenius_raw"] = np.linalg.norm(saved_cov - rollout_cov, axis=(1, 2))
    frame.loc[ids, "saved_heading_variance"] = saved_cov[:, 2, 2]
    frame.loc[ids, "rollout_heading_variance"] = rollout_cov[:, 2, 2]
    frame.loc[ids, "heading_variance_ratio_saved_to_rollout"] = saved_cov[:, 2, 2] / rollout_cov[:, 2, 2]
    q_cached = d["post_local_q"][ids]
    selected_sigmas = {
        "saved": np.sqrt(np.einsum("ni,nij,nj->n", q_cached, saved_cov, q_cached)),
        "rollout": np.sqrt(np.einsum("ni,nij,nj->n", q_cached, rollout_cov, q_cached)),
    }
    for model, sigma in selected_sigmas.items():
        frame.loc[ids, model + "_selected_sigma_m"] = sigma
        frame.loc[ids, model + "_selected_p"] = cdf((params["d_min"] - d["post_phi_m"][ids]) / sigma)
    wrapper = make_wrapper(module, config, polygons, device)
    for start in range(0, len(ids), batch_size):
        index = ids[start:start + batch_size]
        z = saved_post[index].copy()
        z[:, 4:8] = rollout_xi[index]
        evaluated = wrapper.forward_mf_icdf(
            torch.as_tensor(z, dtype=torch.float64, device=wrapper.A.device),
            params["chance_epsilon"], params["d_min"], return_frozen_features=True)
        features = evaluated.frozen_features
        distance, q, mask = [features[key].detach().cpu().numpy() for key in ("d", "q", "mask")]
        phi = evaluated.phi.detach().cpu().numpy().astype(float)
        saved_R, saved_max, valid_saved = feature_risk(distance, q, mask, saved_post[index, 4:8], params["d_min"])
        rolled_R, rolled_max, valid_rolled = feature_risk(distance, q, mask, rollout_xi[index], params["d_min"])
        common = (d["post_probability_valid"][index] & (phi > params["d_min"])
                  & valid_saved & valid_rolled
                  & np.isfinite(frame.loc[index, ["saved_selected_p", "rollout_selected_p"]]).all(axis=1).to_numpy())
        frame.loc[index, "phi_m"] = phi
        frame.loc[index, "guard_margin_mm"] = (phi - params["d_col"]) * 1000
        frame.loc[index, "common_probability_valid"] = common
        for model, R, maximum in (("saved", saved_R, saved_max), ("rollout", rolled_R, rolled_max)):
            frame.loc[index[common], model + "_R"] = R[common]
            frame.loc[index[common], model + "_pmax"] = maximum[common]
            frame.loc[index[common], model + "_R_budget_residual"] = params["chance_epsilon"] - R[common]
        checked = evaluated.probability_sum.detach().cpu().numpy()
        frame.loc[index[common], "gpu_rollout_R_check_abs_error"] = np.abs(rolled_R[common] - checked[common])
        frame.loc[index[common], "saved_gpu_R_minus_cached_R"] = saved_R[common] - d["post_R"][index[common]]
    # Invalid-domain numeric probabilities are not usable collision evidence.
    invalid = ~frame.common_probability_valid.to_numpy(bool)
    frame.loc[invalid, ["saved_selected_p", "rollout_selected_p"]] = np.nan
    valid = frame.common_probability_valid.to_numpy(bool)
    gpu_error = frame.loc[valid, "gpu_rollout_R_check_abs_error"].to_numpy(float)
    if len(gpu_error) and np.max(gpu_error) > 1e-10:
        raise ValueError("CPU feature sum disagrees with the archived GPU risk evaluation")
    metrics = trial_metrics(frame, params["chance_epsilon"])
    sources = {str(path): sha(path) for path in (cache, checkpoint, history_path, param_path, config_path)}
    meta = {"width_mm": width, "method": method, "seed": seed, "stage": 1, "cycles": n,
            "normal_success_cycles": int(post_valid.sum()), "common_probability_cycles": int(valid.sum()),
            "source_probability_valid_cycles": int((post_valid & d["post_probability_valid"]).sum()),
            "input_memory_max_abs_error": float(np.max(np.abs(inputs - applied))),
            "saved_post_checkpoint_max_abs_error": float(np.max(np.abs(saved_post[post_valid] - d["post_states"][post_valid]))),
            "gpu_vs_cached_phi_max_abs_m": float(np.max(np.abs(frame.loc[ids, "phi_m"] - d["post_phi_m"][ids]))),
            "saved_gpu_R_vs_cached_max_abs": float(np.nanmax(np.abs(frame.saved_gpu_R_minus_cached_R))),
            "gpu_rollout_R_check_max_abs": float(np.max(gpu_error)) if len(gpu_error) else None,
            "epsilon": params["chance_epsilon"], "d_min_m": params["d_min"], "d_col_m": params["d_col"],
            "dt_s": params["tf"] / params["horizon"], "robust_horizon": params["robust_horizon"],
            "elapsed_s": time.perf_counter() - started, "sources": sources}
    outlier = None
    if (width, method, seed) == (129, "cc_mf", 1):
        i = 120
        c_saved = covariance.covariance_from_log_cholesky(saved_post[i, 4:8])
        c_rollout = covariance.covariance_from_log_cholesky(rollout_xi[i])
        nominal_replay = covariance.propagate_log_cholesky(
            xi0[i], inputs[i - 1, 0], inputs[i - 1, 1], 0., meta["dt_s"], SimpleNamespace(**params))
        cycle_path = directory / "cycles.csv"
        with cycle_path.open() as stream:
            cycle_record = list(csv.DictReader(stream))[i]
        outlier = {"width_mm": width, "method": method, "seed": seed, "cycle": i, "stage": 1,
                   "checkpoint_line_1based": i + 2, "checkpoint_record_time_s": float(d["time"][i] + meta["dt_s"]),
                   "current_cycle_time_s": float(d["time"][i]), "input_v_omega": inputs[i],
                   "applied_history_input_v_omega": applied[i],
                   "previous_input_v_omega_for_nominal": inputs[i - 1],
                   "nominal_replay_xi1": nominal_replay,
                   "nominal_replay_vs_cache_max_abs": np.max(np.abs(nominal_replay - d["nominal_states"][i, 4:8])),
                   "reset_xi0": xi0[i], "saved_stage0_xi": stage0[i, 4:8],
                   "saved_xi1": saved_post[i, 4:8], "exact_rollout_xi1": rollout_xi[i],
                   "xi_dynamics_defect": saved_post[i, 4:8] - rollout_xi[i],
                   "saved_covariance": c_saved, "exact_rollout_covariance": c_rollout,
                   "saved_heading_std_rad": np.sqrt(c_saved[2, 2]),
                   "rollout_heading_std_rad": np.sqrt(c_rollout[2, 2]),
                   "cache_vs_checkpoint_post_max_abs": np.max(np.abs(saved_post[i] - d["post_states"][i])),
                   "risk_row": frame.iloc[i].to_dict(), "cycle_record": cycle_record,
                   "sources": sources | {str(cycle_path): sha(cycle_path)},
                   "interpretation": "A recorded nonlinear covariance-dynamics inconsistency, not an offline reconstruction artifact. Exact propagation is a planning-model reference, not measured uncertainty."}
    return frame, metrics, meta, outlier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("benchmark_runs/sf_mf_width125_130mm_step001_10trials_20261104"))
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--widths-mm", type=int, nargs="+", default=list(range(125, 131)))
    parser.add_argument("--methods", nargs="+", default=["cc_sf", "cc_mf"], choices=["cc_sf", "cc_mf"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--wait-for-caches-seconds", type=float, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.experiment.resolve()
    base = (args.analysis_dir or root / "constraint_stability_20261105").resolve()
    output = base / "rollout_covariance"
    per_trial = output / "trial_cache"
    per_trial.mkdir(parents=True, exist_ok=True)
    script_hash = sha(__file__)
    all_frames, all_metrics, metas = [], [], []
    with _snapshot_imports(root / "source_snapshot") as (module, covariance, _):
        polygons = _maze_polygons(root / "source_snapshot/sim/simulation_mpc.py")
        for width in args.widths_mm:
            for method in args.methods:
                for seed in args.seeds:
                    stem = f"width_{width:03d}mm_{method}_seed_{seed}_stage_1"
                    cache = base / "cycle_cache" / (stem + ".npz")
                    began = time.monotonic()
                    while not cache.exists() and time.monotonic() - began < args.wait_for_caches_seconds:
                        time.sleep(1)
                    if not cache.exists():
                        raise FileNotFoundError(cache)
                    meta_path, csv_path = per_trial / (stem + ".json"), per_trial / (stem + ".csv")
                    prior = json.loads(meta_path.read_text()) if args.resume and meta_path.exists() and csv_path.exists() else None
                    if (prior and prior.get("script_sha256") == script_hash and prior.get("device") == args.device
                            and all(Path(path).exists() and sha(path) == digest for path, digest in prior["sources"].items())):
                        frame = pd.read_csv(csv_path)
                        metrics, meta = trial_metrics(frame, prior["epsilon"]), prior
                    else:
                        frame, metrics, meta, outlier = analyze_trial(root, cache, width, method, seed, module,
                                                                    covariance, polygons, args.device, args.batch_size)
                        meta.update(script_sha256=script_hash, device=args.device)
                        frame.to_csv(csv_path, index=False)
                        write_json(meta_path, meta)
                        if outlier is not None:
                            outlier["analysis_source_sha256"] = script_hash
                            outlier["covariance_source_sha256"] = sha(root / "source_snapshot/control/cc_covariance.py")
                            write_json(base / "covariance_outlier_audit.json", outlier)
                    all_frames.append(frame); all_metrics.extend(metrics); metas.append(meta)
                    print(json.dumps({"trial": stem, "normal": meta["normal_success_cycles"],
                                      "common_valid": meta["common_probability_cycles"],
                                      "elapsed_s": round(meta["elapsed_s"], 3)}), flush=True)
    pd.concat(all_frames, ignore_index=True).to_csv(output / "per_cycle.csv", index=False)
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(output / "per_trial_metrics.csv", index=False)
    grouped = metrics.groupby(["width_mm", "method", "covariance_model", "metric"]).agg(
        mean=("value", "mean"), sd=("value", "std"), n_trials=("value", "count"),
        min=("value", "min"), max=("value", "max"), total_samples=("n_samples", "sum")).reset_index()
    grouped.to_csv(output / "mean_sd.csv", index=False)
    write_json(output / "manifest.json", {
        "experiment": str(root), "script_sha256": script_hash, "device": args.device,
        "widths_mm": args.widths_mm, "methods": args.methods, "seeds": args.seeds,
        "completed_trials": len(metas), "trial_metadata": metas,
        "covariance_source_sha256": sha(root / "source_snapshot/control/cc_covariance.py"),
        "definitions": [
            "Stage 1 only; reset covariance at the recorded current state, then propagate the archived exact covariance map with the saved/applied first input and dt=0.1 s. Respect robust_horizon=0 freezing.",
            "Saved solution poses and inputs are unchanged. Neither controller nor simulator is instantiated; source caches are not modified.",
            "The rollout covariance is model-consistent planning covariance, not measured or calibrated physical uncertainty.",
            "Saved and rollout raw R/pmax use exactly the same fresh GPU float32 geometry features with float64 risk arithmetic. R=sum p_i is a Boole surrogate, can exceed one, and is not actual collision probability.",
            "Selected probability uses the cached SF-selected full local gradient and signed phi, held identical between the covariance variants; it does not reselect a potentially different GPU float32 tie branch.",
            "Both covariance variants use the same common domain: cached separated phi>d_min, GPU phi>d_min, and valid finite feature variances for both covariances. Domain exclusions are counted; no invalid-domain probability is imputed.",
            "Each trial statistic is computed first; width/method means and sample SD(ddof=1) weight trials equally. Comparison rows retain within-trial rollout-minus-saved differences.",
            "Covariance-defect Frobenius norms combine position/heading coordinates and are raw numerical diagnostics, not a physical distance. Large outliers are retained without clipping or replacement.",
        ],
    })
    print(json.dumps({"output": str(output), "completed_trials": len(metas)}), flush=True)


if __name__ == "__main__":
    main()
