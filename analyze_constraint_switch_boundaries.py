"""Compare SF/MF at finite offsets around saved nominal selector switches.

Read-only with respect to the source experiment. The primary displacement is
10 micrometres on EACH side in ||delta_xy||+radius*abs(delta_theta); 5 and50
micrometres are sensitivity analyses. No optimizer or simulation is run.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import inspect
import json
from pathlib import Path
from statistics import NormalDist
import sys
import time

import numpy as np
import torch
import yaml

from utils.offline_cc_margin import _maze_polygons, _snapshot_imports
from utils.offline_feature_selectors import _endpoint_aliases, extract_feature_selectors


def compact_forward(wrapper, poses, keep_internal):
    """Capture only selector tensors; avoid transferring full distance arrays."""
    target = inspect.unwrap(wrapper.psdf._forward_impl).__code__
    segment = inspect.unwrap(wrapper.psdf._p2seg).__code__
    captured, projections = [], []
    previous = sys.getprofile()
    def observe(frame, event, result):
        if event == "return" and frame.f_code is target:
            captured.append({key: frame.f_locals[key] for key in
                             ("active_cluster", "separated", "idx_sep", "idx_v", "idx_e")})
        elif event == "return" and frame.f_code is segment:
            projections.append(frame.f_locals["t"])
    try:
        sys.setprofile(observe)
        with torch.no_grad():
            phi, gradient = wrapper(poses)
    finally:
        sys.setprofile(previous)
    if len(captured) != 1 or len(projections) != 2:
        raise RuntimeError("Unexpected archived geometry call structure")
    raw = captured[0]
    h, m = len(poses), wrapper.psdf.V.shape[0]
    rows = torch.arange(h, device=poses.device)
    cluster = raw["active_cluster"]
    feature = raw["idx_sep"][rows, cluster]
    robot_vertex = feature.clamp(0, m - 1)
    robot_edge = (feature - m).clamp(0, m - 1)
    obstacle_edge = raw["idx_v"][rows, cluster, robot_vertex]
    obstacle_endpoint = raw["idx_e"][rows, cluster, robot_edge]
    t_vertex = projections[0][rows, cluster, obstacle_edge, robot_vertex, 0]
    t_edge = projections[1][rows, cluster, obstacle_endpoint, robot_edge, 0]
    selected = torch.stack((cluster, raw["separated"][rows, cluster], feature,
                            obstacle_edge, obstacle_endpoint, t_vertex, t_edge), dim=1).detach().cpu().numpy()
    internal = None
    if keep_internal:
        internal = torch.cat((raw["idx_v"].reshape(h, -1), raw["idx_e"].reshape(h, -1)), dim=1).detach().cpu().numpy()
    return phi.detach().cpu().numpy(), gradient.detach().cpu().numpy(), selected, internal


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(wrapper, poses, aliases, *, keep_internal=False, batch_size=128):
    """The archived forward's selectors, canonicalized exactly as the helper.

Only separated identities are needed; SAT entries receive a sentinel because
all admitted boundaries must have separated endpoints.
"""
    output = {key: [] for key in ("phi", "gradient", "id", "separated")}
    if keep_internal:
        output["internal"] = []
    m, edges = wrapper.psdf.V.shape[0], wrapper.A.shape[1]
    for start in range(0, len(poses), batch_size):
        phi, gradient, selected, internal = compact_forward(
            wrapper, torch.tensor(poses[start:start + batch_size], dtype=torch.float32, device=wrapper.A.device), keep_internal)
        names, separation = [], []
        for i in range(len(phi)):
            cluster = int(selected[i, 0])
            separated = bool(selected[i, 1])
            separation.append(separated)
            if not separated:
                names.append(f"C{cluster}|sat")
                continue
            feature = int(selected[i, 2])
            if feature < m:
                edge = int(selected[i, 3])
                t = selected[i, 5]
                region = 0 if t == 0 else 2 if t == 1 else 1
                robot = f"V{feature}"
                obstacle = f"V{aliases[cluster, edge if region == 0 else edges + edge]}" if region != 1 else f"E{edge}"
            else:
                edge = feature - m
                endpoint = int(selected[i, 4])
                t = selected[i, 6]
                region = 0 if t == 0 else 2 if t == 1 else 1
                robot = f"V{edge if region == 0 else (edge + 1) % m}" if region != 1 else f"E{edge}"
                obstacle = f"V{aliases[cluster, endpoint]}"
            names.append(f"C{cluster}|sep|R{robot}|O{obstacle}")
        output["phi"].append(phi.reshape(-1).astype(float))
        output["gradient"].append(gradient.astype(float))
        output["id"].append(np.asarray(names))
        output["separated"].append(np.asarray(separation, dtype=bool))
        if keep_internal:
            output["internal"].append(internal)
    return {key: np.concatenate(value) for key, value in output.items()}


def sf_values(data, z, params):
    gradient = data["gradient"]
    q = gradient.copy()
    c, s = np.cos(z[:, 2]), np.sin(z[:, 2])
    q[:, 0] = c * gradient[:, 0] + s * gradient[:, 1]
    q[:, 1] = -s * gradient[:, 0] + c * gradient[:, 1]
    var = ((np.exp(z[:, 4]) * q[:, 0]) ** 2
           + (np.exp(z[:, 5]) * q[:, 1] + z[:, 7] * q[:, 2]) ** 2
           + (np.exp(z[:, 6]) * q[:, 2]) ** 2)
    sigma = np.sqrt(var)
    gamma = params.get("chance_gamma")
    if gamma is None:
        gamma = NormalDist().inv_cdf(1 - params["chance_epsilon"])
    standard = (data["phi"] - params["d_min"]) / sigma
    p = (0.5 * torch.erfc(torch.tensor(standard, dtype=torch.float64) / np.sqrt(2))).numpy()
    risk = np.clip(gamma * sigma, params.get("risk_margin_floor", 0), params.get("risk_margin_cap", np.inf))
    return {"quantile": standard - gamma, "p_selected": p, "sigma": sigma,
            "risk": risk, "native": data["phi"] - params["d_min"] - risk}


METRICS = ("delta_sf_quantile", "delta_mf_quantile", "delta_p_selected", "delta_probability_sum",
           "delta_sf_probability_margin", "delta_mf_probability_margin", "delta_sf_native_mm")


def statistics(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "mean_abs": None, "p95_abs": None, "rms": None, "mean_signed": None, "max_abs": None}
    return {"n": len(a), "mean_abs": float(np.mean(np.abs(a))), "p95_abs": float(np.percentile(np.abs(a), 95)),
            "rms": float(np.sqrt(np.mean(a * a))), "mean_signed": float(a.mean()), "max_abs": float(np.abs(a).max())}


def summary(rows):
    accepted = [r for r in rows if r["accepted"]]
    return {"candidate_events": len(rows), "accepted_events": len(accepted),
            "exclusion_counts": dict(Counter(r["exclusion_reason"] for r in rows if not r["accepted"])),
            "ordinary_inverse_cdf_events": sum(r.get("mf_branch_left") == 1 and r.get("mf_branch_right") == 1 for r in accepted),
            "within_initial_grid_bracket_events": sum(r["within_initial_grid_bracket"] for r in accepted),
            "mask_changed_events": sum(r["mask_changes"] > 0 for r in accepted),
            "internal_selector_changed_events": sum(r["internal_selector_changes"] > 0 for r in accepted),
            "mf_branch_pair_counts": dict(Counter(f"{r['mf_branch_left']}:{r['mf_branch_right']}" for r in accepted)),
            "mf_abs_quantile_change_smaller_count": sum(abs(r["delta_mf_quantile"]) < abs(r["delta_sf_quantile"]) for r in accepted),
            "metrics": {key: statistics([r[key] for r in accepted]) for key in METRICS}}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["width_mm", "method", "seed", "stage", "offset_um", "accepted"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_trial(root, cache_path, module, polygons, width, method, seed, stage, offsets, device, center_tolerance_um):
    started = time.perf_counter()
    with np.load(cache_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in ("nominal_states", "replay_valid", "nominal_valid",
                                            "nominal_selector_id", "nominal_separated", "cycle", "s", "time")}
    directory = root / f"width_{width:03d}mm/{method}/seed_{seed}"
    param_path, config_path = directory / "optimizer_parameters.json", directory / "config.yaml"
    params = json.loads(param_path.read_text())
    sf_param_path = root / f"width_{width:03d}mm/cc_sf/seed_{seed}/optimizer_parameters.json"
    sf_params = json.loads(sf_param_path.read_text())
    if any(params[key] != sf_params[key] for key in ("d_min", "chance_epsilon")):
        raise ValueError("Source and reference SF chance thresholds differ")
    config = yaml.safe_load(config_path.read_text())
    length, footprint_width = (config["rectangle_geometry"][key] for key in ("length", "width"))
    radius = float(np.hypot(length / 2, footprint_width / 2))
    vertices = torch.tensor([[-length / 2, -footprint_width / 2], [length / 2, -footprint_width / 2],
                             [length / 2, footprint_width / 2], [-length / 2, footprint_width / 2]], dtype=torch.float32)
    wrapper = module.AugmentedPSDFWrapper(vertices, K_max=20, E_max=100, device=device)
    wrapper.update_edge_clusters([torch.tensor(p, dtype=torch.float32) for p in polygons],
                                 [torch.tensor(np.roll(p, -1, axis=0), dtype=torch.float32) for p in polygons])
    aliases = _endpoint_aliases(wrapper)
    z = data["nominal_states"]
    valid = ((data["nominal_valid"] == 1)
             & np.isfinite(z).all(axis=1) & (data["nominal_selector_id"] != ""))
    adjacent = [i for i in range(1, len(z)) if valid[i] and valid[i - 1]
                and data["cycle"][i] == data["cycle"][i - 1] + 1]
    changed = [i for i in adjacent if data["nominal_selector_id"][i] != data["nominal_selector_id"][i - 1]]
    pairs = [i for i in changed if data["nominal_separated"][i] == 1 and data["nominal_separated"][i - 1] == 1]
    selection = {"cycles": len(z), "valid_nominal_cycles": int(valid.sum()), "valid_adjacent_pairs": len(adjacent),
                 "valid_nominal_native_row_replay_invalid_cycles": int(np.sum(valid & (data["replay_valid"] != 1))),
                 "selector_changed_pairs": len(changed), "selector_unchanged_pairs": len(adjacent) - len(changed),
                 "changed_both_separated_pairs": len(pairs), "cache_selector_disagreement_pairs": 0,
                 "zero_motion_pairs": 0}
    compared = np.flatnonzero(valid & (data["nominal_separated"] == 1))
    disagreements = []
    if len(compared):
        fresh = geometry(wrapper, z[compared, :3], aliases)
        different = compared[fresh["id"] != data["nominal_selector_id"][compared]]
        disagreements = [{"cycle": int(data["cycle"][i]), "cached_canonical": str(data["nominal_selector_id"][i]),
                          "device_canonical": str(fresh["id"][j])}
                         for j, i in enumerate(compared) if fresh["id"][j] != data["nominal_selector_id"][i]]
    else:
        different = np.array([], dtype=int)
    different_set = set(different)
    selection.update(cache_separated_selector_comparison_cycles=len(compared),
                     cache_separated_selector_disagreement_cycles=len(different),
                     cache_selector_disagreement_changed_pairs=sum(i in different_set or i - 1 in different_set for i in changed),
                     cache_selector_disagreement_unchanged_pairs=sum(i in different_set or i - 1 in different_set for i in adjacent if i not in changed))
    verify_ids = np.flatnonzero(valid & (data["nominal_separated"] == 1))[:16]
    if len(verify_ids):
        compact = geometry(wrapper, z[verify_ids, :3], aliases)
        reference = extract_feature_selectors(wrapper, torch.tensor(z[verify_ids, :3], dtype=torch.float32, device=wrapper.A.device))
        if not np.array_equal(compact["id"], reference["canonical_id"]) or not np.array_equal(compact["phi"], reference["phi"].reshape(-1)):
            raise ValueError("Compact selector extraction disagrees with full archived extraction")
    selection["compact_extractor_verified_points"] = len(verify_ids)
    brackets = []
    if pairs:
        grid = np.linspace(0, 1, 33)
        poses = np.array([z[i - 1, :3] + grid[:, None] * (z[i, :3] - z[i - 1, :3]) for i in pairs])
        geom = geometry(wrapper, poses.reshape(-1, 3), aliases)
        names = geom["id"].reshape(len(pairs), 33)
        separation = geom["separated"].reshape(len(pairs), 33)
        for row, i in enumerate(pairs):
            if names[row, 0] != data["nominal_selector_id"][i - 1] or names[row, -1] != data["nominal_selector_id"][i]:
                selection["cache_selector_disagreement_pairs"] += 1
                continue
            change = z[i, :3] - z[i - 1, :3]
            motion = np.linalg.norm(change[:2]) + radius * abs(change[2])
            if not np.isfinite(motion) or motion <= 0:
                selection["zero_motion_pairs"] += 1
                continue
            for k in range(1, 33):
                if names[row, k - 1] != names[row, k] and separation[row, k - 1] and separation[row, k]:
                    brackets.append((i, grid[k - 1], grid[k], names[row, k - 1], motion))
    selection["initial_separated_switch_brackets"] = len(brackets)
    records = []
    if brackets:
        ids = np.array([r[0] for r in brackets])
        original_lo = np.array([r[1] for r in brackets]); original_hi = np.array([r[2] for r in brackets])
        lo, hi = original_lo.copy(), original_hi.copy()
        left_names = np.array([r[3] for r in brackets])
        motion = np.array([r[4] for r in brackets])
        base, change = z[ids - 1, :3], z[ids, :3] - z[ids - 1, :3]
        iterations = np.zeros(len(brackets), dtype=int)
        for _ in range(32):
            active = np.flatnonzero((hi - lo) * motion > center_tolerance_um * 1e-6)
            if not len(active):
                break
            mid = (lo[active] + hi[active]) / 2
            names = geometry(wrapper, base[active] + mid[:, None] * change[active], aliases)["id"]
            same = names == left_names[active]
            lo[active], hi[active] = np.where(same, mid, lo[active]), np.where(same, hi[active], mid)
            iterations[active] += 1
        if np.any((hi - lo) * motion > center_tolerance_um * 1e-6):
            raise ValueError("Boundary localization tolerance was not reached")
        center = (lo + hi) / 2
        for offset in offsets:
            half_alpha = float(offset) * 1e-6 / motion
            lower, upper = center - half_alpha, center + half_alpha
            event_rows = []
            for j, i in enumerate(ids):
                row = {"width_mm": width, "method": method, "seed": seed, "stage": stage,
                       "event_index": j, "offset_um": float(offset), "primary_offset": offset == 10,
                       "cycle_before": int(data["cycle"][i - 1]), "cycle_after": int(data["cycle"][i]),
                       "time_before_s": float(data["time"][i - 1]), "time_after_s": float(data["time"][i]),
                       "progress_before_m": float(data["s"][i - 1]), "progress_after_m": float(data["s"][i]),
                       "boundary_alpha": float(center[j]), "alpha_left": float(lower[j]), "alpha_right": float(upper[j]),
                       "segment_body_motion_m": float(motion[j]), "footprint_radius_m": radius,
                       "bisected_body_bracket_width_m": float((hi[j] - lo[j]) * motion[j]),
                       "boundary_center_uncertainty_upper_bound_m": float((hi[j] - lo[j]) * motion[j] / 2),
                       "bisection_iterations": int(iterations[j]),
                       "within_initial_grid_bracket": bool(lower[j] >= original_lo[j] and upper[j] <= original_hi[j]),
                       "accepted": False, "exclusion_reason": None}
                if lower[j] < 0 or upper[j] > 1:
                    row["exclusion_reason"] = "insufficient_original_segment_support"
                event_rows.append(row)
            eligible = np.array([j for j, row in enumerate(event_rows) if row["exclusion_reason"] is None], dtype=int)
            for chunk_start in range(0, len(eligible), 64):
                idx = eligible[chunk_start:chunk_start + 64]
                left, right = z[ids[idx] - 1].copy(), z[ids[idx] - 1].copy()
                left[:, :3] = base[idx] + lower[idx, None] * change[idx]
                right[:, :3] = base[idx] + upper[idx, None] * change[idx]
                both = np.concatenate([left, right]); size = len(idx)
                geom = geometry(wrapper, both[:, :3], aliases, keep_internal=True)
                sf = sf_values(geom, both, sf_params)
                mf = wrapper.forward_mf_icdf(torch.tensor(both, dtype=torch.float64, device=wrapper.A.device), params["chance_epsilon"],
                                             params["d_min"], return_frozen_features=True)
                frozen = mf.frozen_features
                mf_array = {key: getattr(mf, key).detach().cpu().numpy() for key in
                            ("required", "valid", "branch", "residual", "probability_sum")}
                mask_change_count = (frozen["mask"][:size] != frozen["mask"][size:]).sum(dim=1).detach().cpu().numpy()
                for local, j in enumerate(idx):
                    row = event_rows[j]; l, r = local, local + size
                    row.update(selector_left=str(geom["id"][l]), selector_right=str(geom["id"][r]),
                               separated_left=bool(geom["separated"][l]), separated_right=bool(geom["separated"][r]),
                               phi_left_m=float(geom["phi"][l]), phi_right_m=float(geom["phi"][r]),
                               mf_required_left=bool(mf_array["required"][l]), mf_required_right=bool(mf_array["required"][r]),
                               mf_valid_left=bool(mf_array["valid"][l]), mf_valid_right=bool(mf_array["valid"][r]),
                               mf_branch_left=int(mf_array["branch"][l]), mf_branch_right=int(mf_array["branch"][r]),
                               internal_selector_changes=int(np.sum(geom["internal"][r] != geom["internal"][l])),
                               mask_changes=int(mask_change_count[local]),
                               finite_offset_body_motion_left_m=float(np.linalg.norm(left[local, :2] - (base[j] + center[j] * change[j])[:2])
                                                                     + radius * abs(left[local, 2] - (base[j] + center[j] * change[j])[2])),
                               state_left_json=json.dumps(left[local].tolist()), state_right_json=json.dumps(right[local].tolist()))
                    if geom["id"][l] == geom["id"][r]:
                        row["exclusion_reason"] = "same_canonical_at_finite_offsets"
                    elif not geom["separated"][l] or not geom["separated"][r]:
                        row["exclusion_reason"] = "finite_endpoint_not_separated"
                    elif not bool(mf_array["required"][l] and mf_array["required"][r]):
                        row["exclusion_reason"] = "finite_endpoint_mf_domain_inactive"
                    elif not bool(mf_array["valid"][l] and mf_array["valid"][r]):
                        row["exclusion_reason"] = "finite_endpoint_mf_invalid"
                    elif not np.isfinite([sf["quantile"][l], sf["quantile"][r]]).all():
                        row["exclusion_reason"] = "sf_quantile_nonfinite"
                    else:
                        row["accepted"] = True
                        row.update(sf_quantile_left=float(sf["quantile"][l]), sf_quantile_right=float(sf["quantile"][r]),
                                   mf_quantile_left=float(mf_array["residual"][l]), mf_quantile_right=float(mf_array["residual"][r]),
                                   p_selected_left=float(sf["p_selected"][l]), p_selected_right=float(sf["p_selected"][r]),
                                   probability_sum_left=float(mf_array["probability_sum"][l]), probability_sum_right=float(mf_array["probability_sum"][r]),
                                   delta_sf_quantile=float(sf["quantile"][r] - sf["quantile"][l]),
                                   delta_mf_quantile=float(mf_array["residual"][r] - mf_array["residual"][l]),
                                   delta_p_selected=float(sf["p_selected"][r] - sf["p_selected"][l]),
                                   delta_probability_sum=float(mf_array["probability_sum"][r] - mf_array["probability_sum"][l]),
                                   delta_sf_probability_margin=float(sf["p_selected"][l] - sf["p_selected"][r]),
                                   delta_mf_probability_margin=float(mf_array["probability_sum"][l] - mf_array["probability_sum"][r]),
                                   delta_sf_native_mm=float(sf["native"][r] - sf["native"][l]) * 1000,
                                   delta_phi_mm=float(geom["phi"][r] - geom["phi"][l]) * 1000,
                                   delta_sf_risk_mm=float(sf["risk"][r] - sf["risk"][l]) * 1000)
            records.extend(event_rows)
    offset_summaries = []
    for offset in offsets:
        selected = [r for r in records if r["offset_um"] == offset]
        for subset in ("all_valid", "ordinary_inverse_cdf", "within_initial_grid_bracket"):
            subset_rows = selected if subset == "all_valid" else [r for r in selected if (
                r.get("mf_branch_left") == 1 and r.get("mf_branch_right") == 1 if subset == "ordinary_inverse_cdf"
                else r.get("within_initial_grid_bracket"))]
            offset_summaries.append({"width_mm": width, "method": method, "seed": seed, "stage": stage,
                                     "offset_um": offset, "subset": subset, **summary(subset_rows)})
    metadata = {"width_mm": width, "method": method, "seed": seed, "stage": stage,
                "selection_counts": selection, "offset_summaries": offset_summaries,
                "elapsed_s": time.perf_counter() - started,
                "sources": {str(p): digest(p) for p in (cache_path, param_path, config_path, sf_param_path)},
                "chance_epsilon": params["chance_epsilon"], "d_min_m": params["d_min"],
                "offsets_um": offsets, "analysis_source_sha256": digest(Path(__file__)),
                "reference_sf_risk_margin_floor_m": sf_params.get("risk_margin_floor", 0),
                "reference_sf_risk_margin_cap_m": sf_params.get("risk_margin_cap", None)}
    metadata["cache_selector_disagreement_records"] = disagreements
    metadata.update(geometry_device=device, center_tolerance_um=center_tolerance_um)
    return records, metadata


def aggregate_trials(metas):
    rows = [row for meta in metas for row in meta["offset_summaries"]]
    groups = {}
    for row in rows:
        key = (row["width_mm"], row["method"], row["stage"], row["offset_um"], row["subset"])
        groups.setdefault(key, []).append(row)
    out = []
    for key, trials in groups.items():
        base = dict(zip(("width_mm", "method", "stage", "offset_um", "subset"), key))
        base.update(trials=len(trials), trials_with_events=sum(t["accepted_events"] > 0 for t in trials),
                    accepted_events=sum(t["accepted_events"] for t in trials))
        for metric in METRICS:
            for statistic in ("mean_abs", "p95_abs", "rms", "mean_signed"):
                values = [t["metrics"][metric][statistic] for t in trials if t["metrics"][metric][statistic] is not None]
                base[f"{metric}_{statistic}_trial_mean"] = float(np.mean(values)) if values else None
                base[f"{metric}_{statistic}_trial_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                base[f"{metric}_{statistic}_trial_n"] = len(values)
        out.append(base)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--widths-mm", nargs="+", type=int, default=list(range(125, 131)))
    parser.add_argument("--methods", nargs="+", choices=("cc_sf", "cc_mf"), default=["cc_sf", "cc_mf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 10])
    parser.add_argument("--offsets-um", nargs="+", type=float, default=[5, 10, 50])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--center-tolerance-um", type=float, default=0.25)
    parser.add_argument("--wait-for-caches-seconds", type=float, default=0,
                        help="Wait for each missing atomic cache; never reads partial files")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.results_dir.resolve()
    analysis = (args.analysis_dir or root / "constraint_stability_20261105").resolve()
    output = analysis / "boundaries"; output.mkdir(parents=True, exist_ok=True)
    script_hash = digest(Path(__file__))
    all_records, metas = [], []
    with _snapshot_imports(root / "source_snapshot") as (module, _, _):
        polygons = _maze_polygons(root / "source_snapshot/sim/simulation_mpc.py")
        for width in args.widths_mm:
            for method in args.methods:
                for seed in args.seeds:
                    for stage in args.stages:
                        stem = f"width_{width:03d}mm_{method}_seed_{seed}_stage_{stage}"
                        cache = analysis / "cycle_cache" / f"{stem}.npz"
                        wait_started = time.monotonic()
                        while not cache.exists() and time.monotonic() - wait_started < args.wait_for_caches_seconds:
                            time.sleep(1)
                        if not cache.exists():
                            raise FileNotFoundError(f"Required atomic cache is not ready: {cache}")
                        meta_path, records_path = output / f"{stem}.json", output / f"{stem}_events.json"
                        old = json.loads(meta_path.read_text()) if args.resume and meta_path.exists() and records_path.exists() else None
                        if old and old.get("analysis_source_sha256") == script_hash and old.get("offsets_um") == args.offsets_um \
                                and old.get("geometry_device") == args.device and old.get("center_tolerance_um") == args.center_tolerance_um \
                                and old.get("sources", {}).get(str(cache)) == digest(cache):
                            meta, records = old, json.loads(records_path.read_text())
                        else:
                            records, meta = evaluate_trial(root, cache, module, polygons, width, method, seed, stage, args.offsets_um, args.device, args.center_tolerance_um)
                            meta_path.write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
                            records_path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
                            write_csv(output / f"{stem}_events.csv", records)
                        all_records.extend(records); metas.append(meta)
                        primary = next(r for r in meta["offset_summaries"] if r["offset_um"] == 10 and r["subset"] == "all_valid")
                        print(json.dumps({"trial": stem, "elapsed_s": round(meta["elapsed_s"], 3),
                                          "primary_accepted": primary["accepted_events"],
                                          "ordinary": primary["ordinary_inverse_cdf_events"],
                                          "cache_selector_disagreement_pairs": meta["selection_counts"]["cache_selector_disagreement_pairs"]}), flush=True)
    group_rows = aggregate_trials(metas)
    trial_rows = []
    for meta in metas:
        for r in meta["offset_summaries"]:
            row = {k: v for k, v in r.items() if k not in ("metrics", "exclusion_counts", "mf_branch_pair_counts")}
            row["exclusion_counts_json"] = json.dumps(r["exclusion_counts"])
            row["mf_branch_pair_counts_json"] = json.dumps(r["mf_branch_pair_counts"])
            for metric, values in r["metrics"].items():
                row.update({f"{metric}_{k}": v for k, v in values.items()})
            trial_rows.append(row)
    write_csv(output / "all_events.csv", all_records)
    write_csv(output / "per_trial_metrics.csv", trial_rows)
    write_csv(output / "trial_mean_sd.csv", group_rows)
    report = {"requested_scope": vars(args) | {"results_dir": str(root), "analysis_dir": str(analysis)},
              "completed_trial_stages": len(metas), "trial_stages": metas, "trial_mean_sd": group_rows,
              "total_selection_counts": {key: sum(meta["selection_counts"].get(key, 0) for meta in metas)
                                         for key in set().union(*(meta["selection_counts"] for meta in metas))},
              "methodology": ["Primary endpoint displacement is10um on each side;5/50um are prespecified sensitivity levels.",
                              "Scalar interpolation uses raw pose coordinates. Body-motion length is norm(delta_xy)+circumscribed_footprint_radius*abs(delta_heading). Covariance is frozen to the original left cycle's nominal covariance.",
                              "Candidates are adjacent valid normal nominal states with changed separated canonical SF selectors; native MF row replay tolerance is not a gate for this common same-state comparison.33 grid points bracket changes and adaptive bisection reduces body-motion bracket width below0.25um(default). Per-event remaining bracket and center-error bound are recorded.",
                              "No extrapolation beyond the original nominal-pose segment. Same canonical IDs at finite offsets, nonseparated endpoints, inactive/invalid MF domains and nonfinite SF residuals are excluded and counted.",
                              "Finite offsets may extend outside the initial33-point grid bracket; this is flagged and a within-grid-bracket subset is also summarized. The final bisected bracket is only a center locator.",
                              "The all-valid comparison includes the implemented MF linear extension and clipping branches. The ordinary inverse-CDF subset is reported separately.",
                              "Per-trial event mean absolute change, p95 and RMS are aggregated equally across trials as mean±sampleSD. Trials without events are missing, not zero. Events within trials are not independent replicates.",
                              "Finite-offset changes include ordinary smooth variation and possible additional nearby switches; they are not mathematical jump limits or proof of solver-failure causality.",
                              "Geometry uses the recorded device with archived float32 code; risk arithmetic float64. CPU-cache selector disagreement pairs are excluded and counted. Compare source hashes and nominal-cache validation before interpreting results."],
              "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "torch_threads": 2, "geometry_device": args.device},
              "analysis_source": {"path": str(Path(__file__).resolve()), "sha256": script_hash}}
    archive_paths = [root / "source_snapshot" / name for name in (
        "models/psdf_wrapper.py", "models/psdf.py", "models/augmented_psdf_wrapper.py", "models/augmented_psdf.py", "models/mf_implicit.py", "control/cc_covariance.py",
        "control/cc_sf_optimizer.py", "control/cc_mf_optimizer.py", "sim/simulation_mpc.py")]
    report["archived_source_hashes"] = {str(path): digest(path) for path in archive_paths if path.exists()}
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"output": str(output), "completed_trial_stages": len(metas), "events_including_exclusions": len(all_records)}))


if __name__ == "__main__":
    main()
