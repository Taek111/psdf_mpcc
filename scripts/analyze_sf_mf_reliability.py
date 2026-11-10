"""Offline SF/MF risk sensitivity from completed minimum benchmark traces.

No optimizer, solver, simulation, or benchmark is executed. The saved traces do
not contain returned stage-1 candidates/covariance. Rates here are explicitly a
surrogate using actual post-integration poses and one-step reconstructed model
covariance, and must not be reported as the requested stage-1 violation rate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.offline_cc_margin import _snapshot_imports
from utils.offline_feature_selectors import extract_feature_selectors


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def make_wrapper(module, metadata, dtype):
    geometry = metadata["geometry"]
    vertices = torch.tensor(geometry["footprint_polygons"][0], dtype=dtype)
    polygons = [np.asarray(polygon) for polygon in geometry["obstacle_polygons"]]
    # Compact padding changes no valid geometry. It avoids evaluating 96
    # masked-out edge slots per four-edge rectangle in every offline probe.
    wrapper = module.AugmentedPSDFWrapper(
        vertices, K_max=len(polygons), E_max=max(map(len, polygons)), device="cpu",
    )
    wrapper.update_edge_clusters(
        [torch.tensor(polygon, dtype=dtype) for polygon in polygons],
        [torch.tensor(np.roll(polygon, -1, axis=0), dtype=dtype) for polygon in polygons],
    )
    return wrapper


def selector(wrapper, poses):
    return extract_feature_selectors(wrapper, poses, batch_size=256, verify=False)


def evaluate(wrapper, poses, xi, epsilon, d_min, features=False):
    """Fresh geometry, same body covariance; no slack or disabled-row filter."""
    poses = np.atleast_2d(np.asarray(poses, dtype=float))
    xi = np.broadcast_to(np.asarray(xi, dtype=float), (len(poses), 4))
    chunks = []
    for first in range(0, len(poses), 256):
        pp, xx = poses[first:first + 256], xi[first:first + 256]
        tensor = torch.tensor(pp, dtype=wrapper.A.dtype)
        phi, gradient = wrapper(tensor)
        q = gradient.numpy().astype(float)
        c, s = np.cos(pp[:, 2]), np.sin(pp[:, 2])
        qf = c * q[:, 0] + s * q[:, 1]
        ql = -s * q[:, 0] + c * q[:, 1]
        qt = q[:, 2]
        variance = (np.exp(xx[:, 0]) * qf) ** 2
        variance += (np.exp(xx[:, 1]) * ql + xx[:, 3] * qt) ** 2
        variance += (np.exp(xx[:, 2]) * qt) ** 2
        sigma = np.sqrt(variance)
        ph = phi.numpy().reshape(-1)
        zeta = np.divide(d_min - ph, sigma, out=np.zeros_like(sigma), where=sigma > 0)
        p = (0.5 * torch.special.erfc(-torch.tensor(zeta) / np.sqrt(2.0))).numpy()
        p[sigma == 0] = (ph[sigma == 0] < d_min).astype(float)
        z = np.column_stack((pp, np.zeros(len(pp)), xx))
        mf = wrapper.forward_mf_icdf(
            torch.tensor(z, dtype=torch.float64), epsilon, d_min,
            collect_diagnostics=True, return_frozen_features=features,
        )
        result = {"phi": ph, "p_sf": p, "r_mf": mf.probability_sum.numpy(),
                  "q": np.column_stack((qf, ql, qt))}
        if features:
            result["mask"] = mf.frozen_features["mask"].numpy()
        chunks.append(result)
    return {key: np.concatenate([item[key] for item in chunks]) for key in chunks[0]}


def interpolate(poses, steps, alpha):
    # Heading is stored continuously by the plant; no circular interpolation is
    # needed for these adjacent samples (the maximum increment is audited).
    return poses[steps] + alpha[:, None] * (poses[steps + 1] - poses[steps])


def event_rows(wrapper, wrapper32, poses, xi, steps, metadata, epsilon, d_min):
    """Locate transitions on each sampled switch segment with a nine-point grid."""
    if not len(steps):
        return []
    common = {key: metadata[key] for key in ("don", "source", "seed")}
    rows = []
    # Adjacent pairs: covariance held fixed even though actual pose advances dt.
    pair_left, pair_right = poses[steps], poses[steps + 1]
    left = evaluate(wrapper, pair_left, xi[steps], epsilon, d_min, True)
    right = evaluate(wrapper, pair_right, xi[steps], epsilon, d_min, True)
    for i, step in enumerate(steps):
        rows.append(pack_event(common, int(step), i, "adjacent", np.nan, np.nan,
                               pair_left[i], pair_right[i], xi[step], left, right, i,
                               metadata["dt"], True, d_min))
    grid = np.linspace(0, 1, 9)
    grid_steps = np.repeat(steps, len(grid))
    grid_alpha = np.tile(grid, len(steps))
    ids = selector(wrapper, interpolate(poses, grid_steps, grid_alpha))["canonical_id"]
    ids = np.asarray(ids).reshape(len(steps), len(grid))
    rr, cc = np.where(ids[:, :-1] != ids[:, 1:])
    if not len(rr):
        return rows
    brackets = steps[rr]
    lower, upper = grid[cc].copy(), grid[cc + 1].copy()
    lower_ids, upper_ids = ids[rr, cc].copy(), ids[rr, cc + 1].copy()
    for _ in range(26):
        middle = (lower + upper) / 2
        mid_ids = np.asarray(selector(wrapper, interpolate(poses, brackets, middle))["canonical_id"])
        same = mid_ids == lower_ids
        lower = np.where(same, middle, lower)
        upper = np.where(same, upper, middle)
        upper_ids = np.where(same, upper_ids, mid_ids)
    centers = (lower + upper) / 2
    # The grid can contain several different transitions in the same segment.
    # Preserve every identified boundary, not only the largest jump.
    for radius in (1e-3, 1e-4):
        aa, bb = np.maximum(centers - radius, 0), np.minimum(centers + radius, 1)
        pl, pr = interpolate(poses, brackets, aa), interpolate(poses, brackets, bb)
        si_left, si_right = selector(wrapper, pl), selector(wrapper, pr)
        stable = (np.asarray(si_left["canonical_id"]) == lower_ids)
        stable &= np.asarray(si_right["canonical_id"]) == upper_ids
        stable &= np.asarray(si_left["canonical_id"]) != np.asarray(si_right["canonical_id"])
        dl = evaluate(wrapper, pl, xi[brackets], epsilon, d_min, True)
        dr = evaluate(wrapper, pr, xi[brackets], epsilon, d_min, True)
        fl = evaluate(wrapper32, pl, xi[brackets], epsilon, d_min)
        fr = evaluate(wrapper32, pr, xi[brackets], epsilon, d_min)
        for i, step in enumerate(brackets):
            row = pack_event(common, int(step), i, "boundary", radius, centers[i],
                             pl[i], pr[i], xi[step], dl, dr, i, metadata["dt"],
                             bool(stable[i]), d_min)
            row.update({"canonical_left": si_left["canonical_id"][i],
                        "canonical_right": si_right["canonical_id"][i],
                        "raw_left": si_left["raw_id"][i],
                        "raw_right": si_right["raw_id"][i],
                        "j_sf_pp_float32": 100 * abs(fr["p_sf"][i] - fl["p_sf"][i]),
                        "j_mf_pp_float32": 100 * abs(fr["r_mf"][i] - fl["r_mf"][i])})
            rows.append(row)
    return rows


def pack_event(common, step, event, kind, radius, alpha, pl, pr, xi, left, right,
               index, dt, stable, d_min):
    i = index
    row = {**common, "step": step, "event_id": event, "kind": kind,
           "probe_radius": radius, "alpha": alpha, "stable_ids": stable,
           "domain_active": bool(left["phi"][i] > d_min and right["phi"][i] > d_min),
           "mask_changes": int(np.count_nonzero(left["mask"][i] != right["mask"][i])),
           "q_change_norm": float(np.linalg.norm(right["q"][i] - left["q"][i])),
           "pose_delta_m": float(np.linalg.norm(pr[:2] - pl[:2])),
           "heading_delta_rad": float(abs(pr[2] - pl[2])),
           "phi_left_m": left["phi"][i], "phi_right_m": right["phi"][i],
           "b_sf_left": 0.1 - left["p_sf"][i], "b_sf_right": 0.1 - right["p_sf"][i],
           "b_mf_left": 0.1 - left["r_mf"][i], "b_mf_right": 0.1 - right["r_mf"][i],
           "j_sf_pp": 100 * abs(right["p_sf"][i] - left["p_sf"][i]),
           "j_mf_pp": 100 * abs(right["r_mf"][i] - left["r_mf"][i]),
           "time_s": (step + (alpha if np.isfinite(alpha) else 0.5)) * dt}
    for j, name in enumerate(("x", "y", "theta")):
        row[name + "_left"] = pl[j]
        row[name + "_right"] = pr[j]
    for j in range(4):
        row[f"xi_{j}"] = xi[j]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "benchmark_runs/rectangle_constraint_reliability_20261107/rectangle_don_sweep_50_20261105_current_sf")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_runs/rectangle_constraint_reliability_20261107")
    parser.add_argument("--dons", type=float, nargs="+", default=[.83, .84, .85, .86])
    parser.add_argument("--switch-don", type=float, default=.85)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for sub in ("tables", "cache", "audit"):
        (args.output / sub).mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.suite / "suite_manifest.json").read_text(encoding="utf-8"))
    snapshot = args.suite / "source_snapshot"
    metrics, events, cohorts, sources = [], [], [], []
    audit = {"exact_stage1_chance_constraint_violation_rate": "unavailable: returned stage-1 pose and planning covariance not logged",
             "surrogate": "actual post-integration pose + one-step covariance propagated from configured initial covariance and applied input",
             "geometry": "snapshot implementation, CPU float64 primary; float32 same-probe robustness",
             "geometry_padding": "K_max=number of obstacles, E_max=max polygon edge count; masked padding removed. Raw IDs refer to this compact representation, canonical geometry IDs are preserved.",
             "probability_tolerance": 1e-6, "epsilon": .1,
             "boundary_selection": "all canonical endpoint switches; nine-point grid, 26 bisections; all finite probes retained",
             "simulation_executed": False, "solver_executed": False,
             "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
             "platform": platform.platform(), "max_pairs": args.max_pairs,
             "source_snapshot": str(snapshot.resolve()), "sources": sources}
    with _snapshot_imports(snapshot) as (module, covariance, _):
        for don in args.dons:
            width = next(w for w in manifest["widths"] if abs(float(w["nominal_don"]) - don) < 1e-8)
            directory = args.suite / "widths" / width["width_id"]
            trials_path = directory / "trial_results.csv"
            trials = [r for r in read_csv(trials_path) if r["optimizer_type"] in ("cc_sf", "cc_mf")]
            success = {method: {int(r["start_seed"]) for r in trials if r["optimizer_type"] == method and r["success"] == "1"} for method in ("cc_sf", "cc_mf")}
            paired = sorted(success["cc_sf"] & success["cc_mf"])
            cohorts.append({"don": don, "sf_success": len(success["cc_sf"]), "mf_success": len(success["cc_mf"]),
                            "paired_success": len(paired), "paired_seeds": " ".join(map(str, paired)),
                            "sf_only_seeds": " ".join(map(str, sorted(success["cc_sf"] - success["cc_mf"]))),
                            "mf_only_seeds": " ".join(map(str, sorted(success["cc_mf"] - success["cc_sf"])))})
            if args.max_pairs is not None:
                paired = paired[:args.max_pairs]
            sources.append({"path": str(trials_path.resolve()), "sha256": sha(trials_path)})
            for method in ("cc_sf", "cc_mf"):
                for seed in paired:
                    row = next(r for r in trials if r["optimizer_type"] == method and int(r["start_seed"]) == seed)
                    data = directory / "runs" / row["configuration_id"] / f"trial_{int(row['trial_number']):04d}" / "data"
                    mp, npz = data / "raw_trace_metadata.json", data / "raw_trace.npz"
                    meta = json.loads(mp.read_text(encoding="utf-8"))
                    trace = np.load(npz, allow_pickle=False)
                    poses, inputs = trace["poses"], trace["applied_inputs"]
                    if set(trace.files) != {"poses", "pose_times_s", "applied_inputs", "solve_nlp_times_s"}:
                        raise ValueError(f"Unexpected minimum trace schema: {npz}")
                    if sha(npz) != meta["raw_trace_sha256"]:
                        raise ValueError(f"Trace hash mismatch: {npz}")
                    if len(poses) != len(inputs) + 1:
                        raise ValueError("Pose/input alignment failed")
                    params = meta["optimizer_parameters"]
                    if abs(params["chance_epsilon"] - .1) > 1e-12:
                        raise ValueError("This analysis is specified for epsilon=.1")
                    if params.get("enable_backup_solver", False):
                        raise ValueError("Normal-solve filtering needs per-call mode with backup enabled")
                    p = SimpleNamespace(**params)
                    xi0 = covariance.log_cholesky_from_covariance(covariance.initial_local_covariance(p))
                    xi = covariance.propagate_log_cholesky(xi0, inputs[:, 0], inputs[:, 1], 0., meta["dt_s"], p)
                    wrapper = make_wrapper(module, meta, torch.float64)
                    result = evaluate(wrapper, poses[1:], xi, .1, params["d_min"])
                    statuses = meta["solver_status_infos"]
                    if len(statuses) != len(inputs):
                        raise ValueError("Status/input alignment failed in a successful trial")
                    eligible = np.array([s["success"] and s.get("solver_called", True) is not False for s in statuses], dtype=bool)
                    risk = result["p_sf"] if method == "cc_sf" else result["r_mf"]
                    valid = eligible & np.isfinite(risk)
                    violation = valid & (risk > .1 + 1e-6)
                    select = selector(wrapper, poses)
                    identifiers = np.asarray(select["canonical_id"], dtype=str)
                    switch_steps = np.flatnonzero((identifiers[:-1] != identifiers[1:]) & eligible)
                    trial_events = []
                    if abs(don - args.switch_don) < 1e-8:
                        trial_events = event_rows(wrapper, make_wrapper(module, meta, torch.float32), poses, xi,
                                                  switch_steps, {"don": don, "source": method, "seed": seed, "dt": meta["dt_s"]},
                                                  .1, params["d_min"])
                        events.extend(trial_events)
                    record = {"don": don, "source": method, "seed": seed, "n_steps": len(inputs),
                              "n_solver_success": int(eligible.sum()), "n_excluded_solver_failure": int((~eligible).sum()),
                              "n_valid": int(valid.sum()), "n_invalid": int((eligible & ~np.isfinite(risk)).sum()),
                              "n_violation": int(violation.sum()), "violation_rate_pct": 100 * violation.sum() / valid.sum(),
                              "n_switch_pairs": len(switch_steps),
                              "n_boundary_events": sum(r["kind"] == "boundary" and r["probe_radius"] == 1e-4 for r in trial_events),
                              "trace": str(npz.relative_to(ROOT)), "metadata": str(mp.relative_to(ROOT))}
                    metrics.append(record)
                    cache = args.output / "cache" / f"don{don:.2f}_{method}_seed{seed}.npz"
                    np.savez_compressed(cache, poses=poses, times=trace["pose_times_s"], applied_inputs=inputs,
                                        xi1=xi, p_sf=result["p_sf"], r_mf=result["r_mf"], phi=result["phi"],
                                        eligible=eligible, valid=valid, canonical_id=identifiers)
                    sources.extend([{"path": str(npz.resolve()), "sha256": sha(npz)}, {"path": str(mp.resolve()), "sha256": sha(mp)}])
                    print(f"DoN={don:.2f} {method} seed={seed}: {violation.sum()}/{valid.sum()} ({record['violation_rate_pct']:.3f}%), switches={len(switch_steps)} boundaries={record['n_boundary_events']}", flush=True)
                    write_csv(args.output / "tables/per_trial_metrics.csv", metrics)
                    write_csv(args.output / "tables/switch_events.csv", events)
    summary = []
    for don in args.dons:
        for method in ("cc_sf", "cc_mf"):
            group = [r for r in metrics if r["don"] == don and r["source"] == method]
            ee = [r for r in events if r["don"] == don and r["source"] == method and r["kind"] == "boundary" and r["probe_radius"] == 1e-4]
            stable = [r for r in ee if r["stable_ids"] and r["mask_changes"] == 0 and r["domain_active"]]
            count, vv = sum(r["n_valid"] for r in group), sum(r["n_violation"] for r in group)
            item = {"don": don, "source": method, "n_trials": len(group), "n_valid": count, "n_violation": vv,
                    "pooled_rate_pct": 100 * vv / count if count else np.nan,
                    "mean_trial_rate_pct": float(np.mean([r["violation_rate_pct"] for r in group])),
                    "median_trial_rate_pct": float(np.median([r["violation_rate_pct"] for r in group])),
                    "n_boundary_events": len(ee), "n_stable_mask_domain_events": len(stable)}
            for label, selected in (("", ee), ("stable_", stable)):
                for method_short in ("sf", "mf"):
                    values = [r[f"j_{method_short}_pp"] for r in selected]
                    item[f"{label}mean_j_{method_short}_pp"] = float(np.mean(values)) if values else np.nan
                    item[f"{label}median_j_{method_short}_pp"] = float(np.median(values)) if values else np.nan
            summary.append(item)
    write_csv(args.output / "tables/paired_summary.csv", summary)
    write_csv(args.output / "tables/paired_cohort.csv", cohorts)
    for path in [Path(__file__), ROOT / "utils/offline_cc_margin.py", ROOT / "utils/offline_feature_selectors.py"]:
        sources.append({"path": str(path.resolve()), "sha256": sha(path)})
    for path in sorted(snapshot.rglob("*.py")):
        sources.append({"path": str(path.resolve()), "sha256": sha(path)})
    audit["n_trials_analyzed"] = len(metrics)
    audit["n_event_rows"] = len(events)
    (args.output / "audit/analysis_manifest.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
