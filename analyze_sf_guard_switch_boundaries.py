"""Offline SF/MF boundary comparison on archived SF+Guard trajectories.

No solver, simulation, or controller is instantiated. The default domain is
predeclared as all ten seeds, stage10, progress0.6--1.2m. Missing, failed, and
post-failure nominal pairs are excluded from boundary statistics and counted.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import sys
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from utils.offline_cc_margin import _maze_polygons, _snapshot_imports
from utils.offline_feature_selectors import _capture_forward, extract_feature_selectors


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def stats(values):
    values = np.abs(np.asarray(values, dtype=float))
    if not len(values):
        return {"n": 0}
    return {"n": len(values), "min_abs": float(values.min()),
            "median_abs": float(np.median(values)),
            "p95_abs": float(np.percentile(values, 95)), "max_abs": float(values.max())}


def q_local(gradient, theta):
    q = np.asarray(gradient, dtype=float).copy()
    c, s = np.cos(theta), np.sin(theta)
    q[:, 0] = c * gradient[:, 0] + s * gradient[:, 1]
    q[:, 1] = -s * gradient[:, 0] + c * gradient[:, 1]
    return q


def sf_values(phi, gradient, z, params):
    q = q_local(gradient, z[:, 2])
    variance = ((np.exp(z[:, 4]) * q[:, 0]) ** 2
                + (np.exp(z[:, 5]) * q[:, 1] + z[:, 7] * q[:, 2]) ** 2
                + (np.exp(z[:, 6]) * q[:, 2]) ** 2)
    sigma = np.sqrt(variance)
    gamma = params.get("chance_gamma")
    if gamma is None:
        gamma = NormalDist().inv_cdf(1 - params["chance_epsilon"])
    raw_risk = gamma * sigma
    risk = np.clip(raw_risk, params.get("risk_margin_floor", 0), params.get("risk_margin_cap", np.inf))
    native = np.asarray(phi, dtype=float).reshape(-1) - params["d_min"] - risk
    quantile = (np.asarray(phi, dtype=float).reshape(-1) - params["d_min"]) / sigma - gamma
    return native, quantile, risk, q, bool(np.array_equal(risk, raw_risk))


def wrapper_for(module, config, polygons):
    length, width = (config["rectangle_geometry"][k] for k in ("length", "width"))
    vertices = torch.tensor([[-length / 2, -width / 2], [length / 2, -width / 2],
                             [length / 2, width / 2], [-length / 2, width / 2]], dtype=torch.float32)
    wrapper = module.AugmentedPSDFWrapper(vertices, K_max=20, E_max=100, device="cpu")
    wrapper.update_edge_clusters([torch.tensor(p, dtype=torch.float32) for p in polygons],
                                 [torch.tensor(np.roll(p, -1, axis=0), dtype=torch.float32)
                                  for p in polygons])
    return wrapper


def nominal_state(covariance, params, states, current_pose, cycle, stage):
    p = SimpleNamespace(**params)
    z = states[cycle - 1, stage, :8].copy()
    xi = covariance.log_cholesky_from_covariance(covariance.initial_local_covariance(
        p, SimpleNamespace(_x=current_pose[cycle])))
    for j in range(min(stage, covariance.resolve_robust_horizon(p))):
        v, omega = states[cycle - 1, j + 1, 8:10]
        xi = covariance.propagate_log_cholesky(xi, v, omega, 0.0,
                                             params["tf"] / params["horizon"], p)
    z[4:8] = xi
    return z


def probe(wrapper, params, z, ids, features, progress, seed):
    pairs = [j for j in range(1, len(ids)) if ids[j] == ids[j - 1] + 1
             and features["canonical_id"][j] != features["canonical_id"][j - 1]
             and features["separated"][j] and features["separated"][j - 1]]
    if not pairs:
        return [], {"eligible_changed_separated_pairs": 0}
    grid = np.linspace(0, 1, 33)
    poses = np.array([z[j - 1, :3] + grid[:, None] * (z[j, :3] - z[j - 1, :3]) for j in pairs])
    selected = extract_feature_selectors(wrapper, torch.tensor(poses.reshape(-1, 3), dtype=torch.float32), verify=False)
    names = selected["canonical_id"].reshape(len(pairs), 33)
    phi = selected["phi"].reshape(len(pairs), 33)
    brackets = [(j, grid[k - 1], grid[k], names[row, k - 1])
                for row, j in enumerate(pairs) for k in range(1, 33)
                if names[row, k] != names[row, k - 1]
                and phi[row, k - 1] > params["d_min"] and phi[row, k] > params["d_min"]]
    if not brackets:
        return [], {"eligible_changed_separated_pairs": len(pairs), "active_brackets": 0}
    js = np.array([r[0] for r in brackets])
    lo, hi = (np.array([r[k] for r in brackets]) for k in (1, 2))
    left_names = np.array([r[3] for r in brackets])
    base, change = z[js - 1, :3], z[js, :3] - z[js - 1, :3]
    for _ in range(18):
        mid = (lo + hi) / 2
        values = extract_feature_selectors(wrapper, torch.tensor(base + mid[:, None] * change,
                                                               dtype=torch.float32), verify=False)
        same = values["canonical_id"] == left_names
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
    left, right = z[js - 1].copy(), z[js - 1].copy()
    left[:, :3], right[:, :3] = base + lo[:, None] * change, base + hi[:, None] * change
    n = len(left)
    both = np.concatenate([left, right])
    ff = extract_feature_selectors(wrapper, torch.tensor(both[:, :3], dtype=torch.float32))
    native, sf, risk, q, unclipped = sf_values(ff["phi"], ff["world_gradient"], both, params)
    mf = wrapper.forward_mf_icdf(torch.tensor(both, dtype=torch.float64), params["chance_epsilon"],
                                 params["d_min"], return_frozen_features=True)
    frozen = mf.frozen_features
    mask_changes = (frozen["mask"][:n] != frozen["mask"][n:]).sum(axis=1).numpy()
    qdiff = frozen["q"][n:].numpy() - frozen["q"][:n].numpy()
    _, _, raw = _capture_forward(wrapper, torch.tensor(both[:, :3], dtype=torch.float32), True)
    internal_changes = np.concatenate([(raw["idx_v"][n:] != raw["idx_v"][:n]).reshape(n, -1),
                                       (raw["idx_e"][n:] != raw["idx_e"][:n]).reshape(n, -1)], axis=1).sum(axis=1)
    records = []
    excluded = 0
    for k, j in enumerate(js):
        active = bool(mf.required[k] and mf.required[n + k] and mf.valid[k] and mf.valid[n + k])
        if not active:
            excluded += 1
            continue
        dsf, dmf = float(sf[n + k] - sf[k]), float(mf.residual[n + k] - mf.residual[k])
        records.append({"seed": seed, "cycle_before": int(ids[j - 1]), "cycle_after": int(ids[j]),
                        "progress_before_m": float(progress[ids[j - 1]]), "progress_after_m": float(progress[ids[j]]),
                        "alpha": float((lo[k] + hi[k]) / 2), "z_left": left[k].tolist(), "z_right": right[k].tolist(),
                        "selector_left": str(ff["canonical_id"][k]), "selector_right": str(ff["canonical_id"][n + k]),
                        "selector_still_differs": bool(ff["canonical_id"][k] != ff["canonical_id"][n + k]),
                        "phi_left_mm": float(ff["phi"][k]) * 1000, "phi_right_mm": float(ff["phi"][n + k]) * 1000,
                        "sf_native_delta_mm": float(native[n + k] - native[k]) * 1000,
                        "sf_quantile_left": float(sf[k]), "sf_quantile_right": float(sf[n + k]),
                        "sf_quantile_delta": dsf, "mf_quantile_left": float(mf.residual[k]),
                        "mf_quantile_right": float(mf.residual[n + k]), "mf_quantile_delta": dmf,
                        "ratio_abs_mf_over_sf": abs(dmf / dsf) if abs(dsf) > 1e-12 else None,
                        "R_left": float(mf.probability_sum[k]), "R_right": float(mf.probability_sum[n + k]),
                        "R_delta": float(mf.probability_sum[n + k] - mf.probability_sum[k]),
                        "mf_required_valid_both": active, "mf_branches": [int(mf.branch[k]), int(mf.branch[n + k])],
                        "sf_risk_unclipped": unclipped, "mask_changes": int(mask_changes[k]),
                        "raw_internal_selector_changes": int(internal_changes[k]),
                        "feature_q_translation_change_max": float(np.linalg.norm(qdiff[k, :, :2], axis=1).max()),
                        "feature_q_heading_change_max_m": float(np.abs(qdiff[k, :, 2]).max())})
    return records, {"eligible_changed_separated_pairs": len(pairs), "active_brackets": len(brackets),
                     "excluded_final_inactive_or_invalid": excluded}


def summarize(records):
    return {"boundaries": len(records), **{key: stats([r[key] for r in records])
            for key in ("sf_native_delta_mm", "sf_quantile_delta", "mf_quantile_delta", "R_delta")},
            "mf_less_than_sf_count": sum(abs(r["mf_quantile_delta"]) < abs(r["sf_quantile_delta"]) for r in records),
            "sf_abs_quantile_jump_gt_1e_3_count": sum(abs(r["sf_quantile_delta"]) > 1e-3 for r in records),
            "mask_change_boundaries": sum(r["mask_changes"] > 0 for r in records),
            "raw_internal_selector_change_boundaries": sum(r["raw_internal_selector_changes"] > 0 for r in records),
            "mf_branch_pair_counts": dict(Counter(",".join(map(str, r["mf_branches"])) for r in records))}


def enrich_report(report, experiment):
    """Summarize saved records only; no geometry or covariance replay."""
    records = report["records"]
    report["summary"] = summarize(records)
    ordinary = [r for r in records if r["mf_branches"] == [1, 1]]
    report["ordinary_inverse_cdf_subset_summary"] = summarize(ordinary)
    report["mf_not_smaller_cases"] = [r for r in records
                                      if abs(r["mf_quantile_delta"]) >= abs(r["sf_quantile_delta"])]
    report["maximum_mf_change_case"] = max(records, key=lambda r: abs(r["mf_quantile_delta"])) if records else None
    report["residual_definitions"] = {
        "sf_native": "phi-d_min-clip(gamma*sigma,floor,cap), in m; recorded delta is converted to mm",
        "sf_quantile": "(phi-d_min)/sigma-gamma; gamma=Phi_inverse(1-epsilon)",
        "mf_quantile": "Implemented quantile-scale residual: inverse CDF for branch1, archived linear extension for branch2",
        "mf_branch_codes": {"1": "inverse_cdf", "2": "linear_extension_for_R_above_0.5", "3": "lower_tail_clipped"},
        "ordinary_subset": "Both endpoints have branch1; excludes no case based on jump size",
        "R": "Boole sum of feature violation probabilities; not itself an exact collision probability",
    }
    for event in report["stop_episode_precursors"]:
        if not event.get("reconstructed"):
            continue
        params_path = experiment / f"width_{report['scope']['width_mm']:03d}mm/cc_sf/seed_{event['seed']}/optimizer_parameters.json"
        params = json.loads(params_path.read_text())
        for side in ("previous", "failure"):
            phi = event[f"nominal_h_{side}_mm"] + params["d_min"] * 1000 + event[f"risk_{side}_mm"]
            event[f"nominal_phi_{side}_mm"] = phi
            event[f"nominal_separated_{side}"] = phi > 0
        event["nominal_phi_derivation"] = "phi_mm=h_mm+d_min_mm+risk_mm, reconstructed from recorded SF residual components"
    report["record_integrity"] = {
        "boundary_records": len(records),
        "per_seed_boundary_total": sum(r["boundaries"] for r in report["per_seed"]),
        "all_endpoints_mf_required_valid": all(r["mf_required_valid_both"] for r in records),
        "all_final_selectors_differ": all(r["selector_still_differs"] for r in records),
        "sf_risk_unclipped_all_pairs": all(r["sf_risk_unclipped"] for r in records),
        "all_branch_endpoints_match": all(r["mf_branches"][0] == r["mf_branches"][1] for r in records),
        "stop_episode_count": len(report["stop_episode_precursors"]),
        "consecutive_pair_total": sum(r["consecutive_pairs"] for r in report["per_seed"]),
        "selector_changed_pair_total": sum(r["selector_changed_pairs"] for r in report["per_seed"]),
        "selector_unchanged_pair_total": sum(r["selector_unchanged_pairs"] for r in report["per_seed"]),
    }
    report["analysis_runtime"] = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                                  "geometry_device": "CPU", "geometry_dtype": "float32", "risk_dtype": "float64"}
    report["summary_postprocessing_source"] = {
        "path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Aggregate subsets and stop-phi derivations use stored records only; no new geometry evaluation",
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=10)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--width-mm", type=int, default=125)
    parser.add_argument("--progress-min", type=float, default=0.6)
    parser.add_argument("--progress-max", type=float, default=1.2)
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()
    experiment = args.results_dir.resolve()
    output = experiment / "sf_guard_comparison/switch_boundaries.json"
    if args.summarize_existing:
        report = enrich_report(json.loads(output.read_text()), experiment)
        with output.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"output": str(output), "summary": report["summary"],
                          "ordinary_inverse_cdf_subset_summary": report["ordinary_inverse_cdf_subset_summary"],
                          "integrity": report["record_integrity"], "events": report["stop_episode_precursors"]}, indent=2))
        return
    torch.set_num_threads(6)
    source_paths = [Path(__file__), Path("utils/offline_cc_margin.py"), Path("utils/offline_feature_selectors.py")]
    records, per_seed, events = [], [], []
    with _snapshot_imports(experiment / "source_snapshot") as (module, covariance, _):
        polygons = _maze_polygons(experiment / "source_snapshot/sim/simulation_mpc.py")
        for seed in range(args.trials):
            directory = experiment / f"width_{args.width_mm:03d}mm/cc_sf/seed_{seed}"
            paths = {"parameters": directory / "optimizer_parameters.json", "config": directory / "config.yaml",
                     "checkpoint": directory / "animation_checkpoint.jsonl", "cycles": directory / "cycles.csv",
                     "poses": next((directory / "data").glob("pose_sdf_*.csv"))}
            source_paths.extend(paths.values())
            params = json.loads(paths["parameters"].read_text())
            config = yaml.safe_load(paths["config"].read_text())
            checkpoints = [json.loads(line) for line in paths["checkpoint"].read_text().splitlines()]
            states = np.asarray([row["optimized_traj"] for row in checkpoints[1:]])
            cycles, poses = read_csv(paths["cycles"]), read_csv(paths["poses"])
            current_pose = np.array([[float(row[k]) for k in ("x", "y", "theta")] for row in poses])
            progress = np.array([float(row["path_s"]) for row in poses])
            success = np.array([row["solver_success"].lower() == "true" for row in cycles])
            ids = np.array([i for i in range(1, len(cycles)) if success[i] and success[i - 1]
                            and args.progress_min <= progress[i] <= args.progress_max])
            z = np.array([nominal_state(covariance, params, states, current_pose, i, args.stage) for i in ids])
            wrapper = wrapper_for(module, config, polygons)
            features = extract_feature_selectors(wrapper, torch.tensor(z[:, :3], dtype=torch.float32))
            consecutive = [j for j in range(1, len(ids)) if ids[j] == ids[j - 1] + 1]
            changed = [j for j in consecutive if features["canonical_id"][j] != features["canonical_id"][j - 1]]
            found, counts = probe(wrapper, params, z, ids, features, progress, seed)
            records.extend(found)
            per_seed.append({"seed": seed, "cycles": len(cycles), "selected_nominal_cycles": len(ids),
                             "consecutive_pairs": len(consecutive), "selector_changed_pairs": len(changed),
                             "selector_unchanged_pairs": len(consecutive) - len(changed),
                             **counts, **summarize(found)})
            stop_starts = [i for i, row in enumerate(cycles) if row["solve_mode"].startswith("safe_stop")
                           and (i == 0 or not cycles[i - 1]["solve_mode"].startswith("safe_stop"))]
            for i in stop_starts:
                event = {"seed": seed, "cycle": i, "time_s": i * params["tf"] / params["horizon"],
                         "progress_m": float(progress[i]), "solve_mode": cycles[i]["solve_mode"],
                         "solver_status_code": cycles[i]["solver_status_code"], "reconstructed": False}
                if i >= 2 and success[i - 1] and success[i - 2] and cycles[i]["solve_mode"] == "safe_stop":
                    pair = np.array([nominal_state(covariance, params, states, current_pose, j, args.stage) for j in (i - 1, i)])
                    feature = extract_feature_selectors(wrapper, torch.tensor(pair[:, :3], dtype=torch.float32))
                    h, psi, risk, q, _ = sf_values(feature["phi"], feature["world_gradient"], pair, params)
                    event.update(reconstructed=True, selector_previous=str(feature["canonical_id"][0]),
                                 selector_failure_nominal=str(feature["canonical_id"][1]),
                                 selector_changed=bool(feature["canonical_id"][0] != feature["canonical_id"][1]),
                                 nominal_h_previous_mm=float(h[0] * 1000), nominal_h_failure_mm=float(h[1] * 1000),
                                 risk_previous_mm=float(risk[0] * 1000), risk_failure_mm=float(risk[1] * 1000),
                                 current_psdf_mm=float(poses[i]["sdf_value"]) * 1000)
                events.append(event)
            print(json.dumps({"seed": seed, **summarize(found)}), flush=True)
    source_paths += [experiment / "source_snapshot" / p for p in ("models/augmented_psdf.py",
                     "models/augmented_psdf_wrapper.py", "models/mf_inverse_cdf.py", "models/mf_implicit.py",
                     "control/cc_covariance.py", "control/cc_sf_optimizer.py", "sim/simulation_mpc.py")]
    report = {"scope": {"stage": args.stage, "seeds": list(range(args.trials)),
                         "progress_m": [args.progress_min, args.progress_max], "width_mm": args.width_mm,
                         "domain": "Both boundary endpoints satisfy phi>d_min and MF required/valid"},
              "summary": summarize(records), "per_seed": per_seed, "records": records,
              "stop_episode_precursors": events,
              "limitations": ["Selected geometrical interpolation boundaries, not observed continuous-time switching or independent samples.",
                              "CPU float32 geometry; original trajectory was generated with CUDA float32. Tiny differences are not a mathematical continuity proof.",
                              "Pairs are selected by global SF selector changes, not by jump magnitude; rare negligible jumps remain in the records.",
                              "Main boundary statistics use successful cycles preceded by a successful cycle. Failures and immediate post-failure cycles are excluded.",
                              "At stop onsets only retained pre-solve nominal geometry is reconstructed; failed solver iterates were overwritten by the fallback plan.",
                              "A changed selector at a stop onset is an association, not proof of causation. Geometry, covariance, path references and all other horizon constraints also change.",
                              "MF is evaluated counterfactually at SF states without slack or solving; stable residuals can still be negative."],
              "sources": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in dict.fromkeys(source_paths)]}
    report = enrich_report(report, experiment)
    output.parent.mkdir(exist_ok=True, parents=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "summary": report["summary"], "events": events}, indent=2))


if __name__ == "__main__":
    main()
