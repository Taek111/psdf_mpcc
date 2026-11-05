"""Extract all-cycle, same-state CC diagnostics from completed archived trials.

No optimizer or simulation is instantiated. Fresh probabilities are retained
only in the separated MF domain (phi > d_min). Frozen probabilities retain the
nominal domain and refer to affine Gaussian features, not actual collision
probabilities. NPZ files contain numeric/string arrays, never pickled objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

from analyze_omitted_feature_events import wrapper_for
from utils.offline_cc_margin import _maze_polygons, _snapshot_imports, load_trial
from utils.offline_feature_selectors import extract_feature_selectors


VERSION = 2
FLOAT_FIELDS = (
    "phi_m", "guard_margin_mm", "sf_native_margin_mm", "sf_quantile",
    "p_selected", "R", "p_max", "p_other_max", "sigma_selected_m",
    "mf_quantile", "mf_required", "mf_valid", "mf_branch", "separated",
    "feature_count", "selected_slot", "selected_equivalent_slots",
    "selected_distance_match_error_m", "selected_q_match_max_abs_error",
    "near_min_nonselected_slots_0p1mm", "near_min_nonselected_slots_0p5mm",
    "near_min_nonselected_slots_1mm", "near_min_nonselected_slots_2mm",
)
FROZEN_FIELDS = (
    "p_selected", "p_other_max", "R", "p_max", "sf_native_margin_mm",
    "sf_affine_margin_mm", "mf_quantile", "mf_affine_quantile",
    "p_other_gap_at_nominal_mm", "p_other_at_nominal", "worst_other_slot",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_clean(value):
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def variance(q, xi):
    return ((np.exp(xi[:, 0, None]) * q[..., 0]) ** 2
            + (np.exp(xi[:, 1, None]) * q[..., 1] + xi[:, 3, None] * q[..., 2]) ** 2
            + (np.exp(xi[:, 2, None]) * q[..., 2]) ** 2)


def cdf(value):
    return (.5 * torch.special.erfc(-torch.as_tensor(value, dtype=torch.float64)
                                   / math.sqrt(2))).numpy()


def probability_rows(d, q, mask, states, selected):
    sigma = np.sqrt(variance(q, states[:, 4:8]))
    good = np.isfinite(d) & np.isfinite(q).all(-1) & np.isfinite(sigma) & (sigma > 0)
    row_valid = ((~mask) | good).all(-1) & mask.any(-1)
    probabilities = np.zeros_like(d)
    return sigma, good, row_valid, probabilities


def selected_q(gradient, theta):
    q = gradient.astype(float).copy()
    cosine, sine = np.cos(theta), np.sin(theta)
    q[:, 0] = cosine * gradient[:, 0] + sine * gradient[:, 1]
    q[:, 1] = -sine * gradient[:, 0] + cosine * gradient[:, 1]
    return q


def gamma(params):
    value = params.get("chance_gamma")
    return float(value) if value is not None else NormalDist().inv_cdf(1 - params["chance_epsilon"])


def quantile_from_R(R, epsilon):
    value = torch.as_tensor(R, dtype=torch.float64)
    lower = torch.finfo(torch.float64).tiny
    g = NormalDist().inv_cdf(1 - epsilon)
    quantile = torch.special.ndtri(value.clamp(min=lower, max=.5)).numpy()
    return np.where(R > .5, -math.sqrt(2 * math.pi) * (R - .5) - g, -quantile - g)


def apply_sf_reference(arrays, sf_params):
    """Use the paired SF configuration on BOTH controllers' saved states.

    This also repairs version-1 shadow SF rows without repeating geometry.
    In particular, a large saved postsolve planning covariance can activate
    SF's 2m risk cap even when the reconstructed nominal cap is inactive.
    """
    g = gamma(sf_params)
    d_min = sf_params["d_min"]
    floor, cap = sf_params.get("risk_margin_floor", 0.), sf_params.get("risk_margin_cap", math.inf)
    stats = {}
    for prefix in ("nominal", "post"):
        sigma = arrays[f"{prefix}_sigma_selected_m"]
        phi = arrays[f"{prefix}_phi_m"]
        raw = g * sigma
        arrays[f"{prefix}_sf_native_margin_mm"] = (phi - d_min - np.clip(raw, floor, cap)) * 1000
        arrays[f"{prefix}_sf_quantile"] = (phi - d_min) / sigma - g
        arrays[f"{prefix}_sf_cap_active"] = np.where(np.isfinite(raw), (raw >= cap).astype(float), np.nan)
        arrays[f"{prefix}_sf_floor_active"] = np.where(np.isfinite(raw), (raw <= floor).astype(float), np.nan)
        finite = np.isfinite(sigma)
        index = int(np.nanargmax(sigma)) if finite.any() else None
        stats[prefix] = dict(
            finite_cycles=int(finite.sum()), cap_active_cycles=int(np.sum(raw >= cap)),
            floor_active_cycles=int(np.sum(raw <= floor)),
            max_projected_sigma_m=float(sigma[index]) if index is not None else None,
            max_raw_risk_margin_m=float(raw[index]) if index is not None else None,
            max_sigma_cycle=index,
            max_sigma_time_s=float(arrays["time"][index]) if index is not None else None,
        )
    nominal, post = arrays["nominal_states"], arrays["post_states"]
    q = arrays["nominal_local_q"]
    gradient = arrays["nominal_world_gradient"]
    phi = arrays["nominal_phi_m"]
    anchor = nominal[:, :3].astype(np.float32).astype(float)
    distance = phi + (gradient * (post[:, :3] - anchor)).sum(-1)
    sigma_post = np.sqrt(variance(q[:, None, :], post[:, 4:8])[:, 0])
    sigma_nominal = arrays["nominal_sigma_selected_m"]
    raw_nominal = g * sigma_nominal
    xi = nominal[:, 4:8]
    forward = np.exp(xi[:, 0]) * q[:, 0]
    lateral = np.exp(xi[:, 1]) * q[:, 1]
    heading = np.exp(xi[:, 2]) * q[:, 2]
    coupled = lateral + xi[:, 3] * q[:, 2]
    variance_gradient = np.column_stack((2 * forward ** 2, 2 * lateral * coupled,
                                          2 * heading ** 2, 2 * q[:, 2] * coupled))
    margin_gradient = (g / (2 * sigma_nominal))[:, None] * variance_gradient
    margin_gradient[(raw_nominal <= floor) | (raw_nominal >= cap)] = 0.
    arrays["frozenpost_sf_native_margin_mm"] = (distance - d_min - np.clip(g * sigma_post, floor, cap)) * 1000
    arrays["frozenpost_sf_affine_margin_mm"] = (
        distance - d_min - np.clip(raw_nominal, floor, cap)
        - (margin_gradient * (post[:, 4:8] - nominal[:, 4:8])).sum(-1)) * 1000
    arrays["frozenpost_sigma_selected_m"] = sigma_post
    arrays["frozenpost_sf_cap_active"] = np.where(
        np.isfinite(sigma_post), (g * sigma_post >= cap).astype(float), np.nan)
    arrays["sf_affine_pass_fresh_guard_fail"] = (
        arrays["sf_affine_guard_comparison_valid"]
        & (arrays["frozenpost_sf_affine_margin_mm"] >= -.001)
        & (arrays["post_guard_margin_mm"] < -.001))
    arrays["sf_reference_params_json"] = np.array(json.dumps(sf_params))
    arrays["sf_reference_cap_m"] = np.array(cap)
    arrays["sf_reference_floor_m"] = np.array(floor)
    arrays["sf_reference_gamma"] = np.array(g)
    stats["frozenpost"] = dict(
        cap_active_cycles=int(np.sum(g * sigma_post >= cap)),
        max_projected_sigma_m=float(np.nanmax(sigma_post)))
    return stats


def atomic_npz(target, arrays):
    temporary = target.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, target)


def repair_existing_sf_reference(cache, experiment):
    records = []
    for target in sorted(cache.glob("*.npz")):
        with np.load(target, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        if int(arrays["schema_version"]) == VERSION:
            continue
        if int(arrays["schema_version"]) != 1:
            raise ValueError(f"Unsupported old cache version: {target}")
        width, seed = int(arrays["width_mm"]), int(arrays["seed"])
        sf_path = experiment / f"width_{width:03d}mm/cc_sf/seed_{seed}/optimizer_parameters.json"
        sf_params = json.loads(sf_path.read_text())
        stats = apply_sf_reference(arrays, sf_params)
        meta = json.loads(str(arrays["metadata_json"]))
        meta.update(schema_version=VERSION, sf_reference_statistics=stats,
                    sf_reference_parameters_source=dict(path=str(sf_path), sha256=sha(sf_path)),
                    extraction_pipeline_sha256=sha(__file__),
                    sf_reference_correction=dict(
                        reason="Version1 MF shadow SF used an unlimited cap; paired SF has risk_margin_cap=2m. Saved optimized planning covariance sometimes activates that cap.",
                        prior_extractor_sha256=meta["extractor_sha256"], correction_source_sha256=sha(__file__),
                        geometry_recomputed=False))
        arrays["schema_version"] = np.array(VERSION)
        arrays["metadata_json"] = np.array(json.dumps(json_clean(meta), ensure_ascii=False))
        atomic_npz(target, arrays)
        records.append(dict(cache=target.name, method=str(arrays["method"]), width_mm=width,
                            seed=seed, stage=int(arrays["stage"]), statistics=stats,
                            original_extractor_sha256=meta["extractor_sha256"]))
    return records


def fresh_batch(wrapper, states, params):
    selectors = extract_feature_selectors(
        wrapper, torch.as_tensor(states[:, :3], dtype=torch.float32), verify=False)
    result = wrapper.forward_mf_icdf(
        torch.as_tensor(states, dtype=torch.float64), params["chance_epsilon"],
        params["d_min"], return_frozen_features=True)
    feature = {key: value.numpy() if isinstance(value, torch.Tensor) else value
               for key, value in result.frozen_features.items()}
    phi = result.phi.numpy().astype(float)
    np.testing.assert_allclose(phi, selectors["phi"], rtol=0, atol=1e-8)
    gradient = feature["gradient"].astype(float)
    q_sf = selected_q(gradient, states[:, 2])
    sigma_sf = np.sqrt(variance(q_sf[:, None, :], states[:, 4:8])[:, 0])
    sigma, good, valid, probabilities = probability_rows(
        feature["d"], feature["q"], feature["mask"], states, None)
    mask = feature["mask"]
    use = mask & good
    probabilities[use] = cdf((params["d_min"] - feature["d"][use]) / sigma[use])
    R = probabilities.sum(-1)
    np.testing.assert_allclose(R[valid], result.probability_sum.numpy()[valid], atol=1e-12, rtol=1e-12)
    separated = selectors["separated"].astype(bool)
    slot = selectors["active_cluster"].astype(int) * 8 + selectors["feature_index"].astype(int)
    slot[~separated] = -1
    safe_slot = np.maximum(slot, 0)
    rows = np.arange(len(states))
    equivalent = (mask & separated[:, None]
                  & (np.abs(feature["d"] - feature["d"][rows, safe_slot, None]) <= 1e-7)
                  & (np.max(np.abs(feature["q"] - feature["q"][rows, safe_slot, None, :]), axis=-1) <= 1e-6))
    other = mask & ~equivalent
    other_probability = np.where(other, probabilities, -np.inf).max(-1)
    other_probability[~other.any(-1)] = np.nan
    probability_valid = valid & separated & (phi > params["d_min"])
    distance_error = np.full(len(states), np.nan)
    q_error = np.full(len(states), np.nan)
    distance_error[separated] = feature["d"][rows[separated], slot[separated]] - phi[separated]
    q_error[separated] = np.max(np.abs(feature["q"][rows[separated], slot[separated]]
                                              - q_sf[separated]), axis=-1)
    if probability_valid.any():
        if np.max(np.abs(distance_error[probability_valid])) > 1e-7:
            raise ValueError("Selected feature does not reproduce actual SF distance")
        if np.max(q_error[probability_valid]) > 1e-6:
            raise ValueError("Selected feature does not reproduce actual SF full q")
    risk = np.clip(gamma(params) * sigma_sf, params.get("risk_margin_floor", 0.),
                   params.get("risk_margin_cap", math.inf))
    out = {
        "phi_m": phi, "guard_margin_mm": (phi - params["d_col"]) * 1000,
        "sf_native_margin_mm": (phi - params["d_min"] - risk) * 1000,
        "sf_quantile": (phi - params["d_min"]) / sigma_sf - gamma(params),
        "sigma_selected_m": sigma_sf,
        "p_selected": cdf((params["d_min"] - phi) / sigma_sf),
        "R": R, "p_max": probabilities.max(-1), "p_other_max": other_probability,
        "mf_quantile": result.residual.numpy(), "mf_required": result.required.numpy().astype(float),
        "mf_valid": result.valid.numpy().astype(float), "mf_branch": result.branch.numpy().astype(float),
        "separated": separated.astype(float), "feature_count": mask.sum(-1).astype(float),
        "selected_slot": slot.astype(float), "selected_equivalent_slots": equivalent.sum(-1).astype(float),
        "selected_distance_match_error_m": distance_error, "selected_q_match_max_abs_error": q_error,
        "selector_id": selectors["canonical_id"].astype("U128"),
        "world_gradient": gradient, "local_q": q_sf, "probability_valid": probability_valid,
    }
    minimum = np.where(mask, feature["d"], np.inf).min(-1)
    for distance, name in ((.0001, "0p1"), (.0005, "0p5"), (.001, "1"), (.002, "2")):
        out[f"near_min_nonselected_slots_{name}mm"] = (
            other & (feature["d"] <= minimum[:, None] + distance)).sum(-1).astype(float)
    # The represented distances are unsigned outside the separated domain.
    # Never expose their numeric probability sums as valid collision evidence.
    for key in ("p_selected", "R", "p_max", "p_other_max"):
        out[key][~probability_valid] = np.nan
    feature.update(sf_q=q_sf, sf_sigma=sigma_sf, selected=slot, selected_equivalent=equivalent,
                   probabilities=probabilities, probability_valid=probability_valid,
                   mf_A=result.A.numpy(), mf_c=result.c.numpy())
    return out, feature


def frozen_batch(feature, nominal, post, params, implicit):
    displacement = post[:, :3] - nominal[:, :3]
    d = feature["d"] + (feature["pose_feature_gradient"] * displacement[:, None, :]).sum(-1)
    sigma, good, valid, p = probability_rows(d, feature["q"], feature["mask"], post, feature["selected"])
    use = feature["mask"] & good
    p[use] = cdf((params["d_min"] - d[use]) / sigma[use])
    valid &= feature["probability_valid"]
    other = feature["mask"] & ~feature["selected_equivalent"]
    masked_p = np.where(other, p, -np.inf)
    worst = np.argmax(masked_p, axis=1)
    rows = np.arange(len(post))
    has_other = other.any(-1)
    anchor = nominal[:, :3].astype(np.float32).astype(float)
    sf_distance = feature["phi"] + (feature["gradient"] * (post[:, :3] - anchor)).sum(-1)
    sf_sigma = np.sqrt(variance(feature["sf_q"][:, None, :], post[:, 4:8])[:, 0])
    g = gamma(params)
    floor, cap = params.get("risk_margin_floor", 0.), params.get("risk_margin_cap", math.inf)
    raw_margin = g * feature["sf_sigma"]
    variance_gradient = implicit.projected_variance_and_gradient(
        torch.as_tensor(feature["sf_q"][:, None, :], dtype=torch.float64),
        torch.as_tensor(nominal, dtype=torch.float64))[1].numpy()[:, 0]
    margin_gradient = (g / (2 * feature["sf_sigma"]))[:, None] * variance_gradient
    margin_gradient[(raw_margin <= floor) | (raw_margin >= cap)] = 0.
    sf_native = sf_distance - params["d_min"] - np.clip(g * sf_sigma, floor, cap)
    sf_affine = sf_distance - params["d_min"] - np.clip(raw_margin, floor, cap) \
        - (margin_gradient * (post[:, 4:8] - nominal[:, 4:8])).sum(-1)
    selected_slot = np.maximum(feature["selected"], 0)
    R = p.sum(-1)
    out = {
        "p_selected": cdf((params["d_min"] - sf_distance) / sf_sigma),
        "p_other_max": masked_p[rows, worst], "R": R, "p_max": p.max(-1),
        "sf_native_margin_mm": sf_native * 1000, "sf_affine_margin_mm": sf_affine * 1000,
        "mf_quantile": quantile_from_R(R, params["chance_epsilon"]),
        "mf_affine_quantile": (feature["mf_A"] * post).sum(-1) + feature["mf_c"],
        "p_other_gap_at_nominal_mm": (feature["d"][rows, worst]
                                      - feature["d"][rows, selected_slot]) * 1000,
        "p_other_at_nominal": feature["probabilities"][rows, worst],
        "worst_other_slot": worst.astype(float), "valid": valid,
    }
    for key in ("p_other_max", "p_other_gap_at_nominal_mm", "p_other_at_nominal", "worst_other_slot"):
        out[key][~has_other] = np.nan
    for key in FROZEN_FIELDS:
        if key not in ("sf_native_margin_mm", "sf_affine_margin_mm"):
            out[key][~valid] = np.nan
    return out


@torch.no_grad()
def extract(experiment, method, width, seed, stage, output, batch_size):
    started = time.perf_counter()
    data = load_trial(experiment, method, width, seed, stage=stage)
    params = dict(data["params"])
    params.setdefault("d_col", .00005)
    sf_path = experiment / f"width_{width:03d}mm/cc_sf/seed_{seed}/optimizer_parameters.json"
    sf_params = json.loads(sf_path.read_text())
    for key in ("d_min", "chance_epsilon"):
        if params[key] != sf_params[key]:
            raise ValueError(f"The common SF/MF risk reference requires matched {key}")
    for key in ("chance_gamma", "risk_margin_floor", "risk_margin_cap"):
        if key in sf_params:
            params[key] = sf_params[key]
    n = len(data["time"])
    normal = data["solver_success"] & (data["solve_mode"] == ("fast" if method == "cc_sf" else "sqp_rti"))
    nominal_valid = normal & np.isfinite(data["nominal_states"]).all(-1)
    arrays = {
        "schema_version": np.array(VERSION), "method": np.array(method), "width_mm": np.array(width),
        "seed": np.array(seed), "stage": np.array(stage), "cycle": np.arange(n),
        "time": data["time"], "s": data["s"], "solver_success": data["solver_success"],
        "solve_mode": data["solve_mode"].astype("U32"), "replay_valid": data["replay_valid"].copy(),
        "nominal_states": data["nominal_states"], "post_states": data["post_states"],
        "current_pose": data["current_pose"], "nominal_valid": nominal_valid, "post_valid": normal,
        "safe_stop": data["safe_stop"], "native_affine": data["postsolve_affine"].copy(),
        "native_fresh": data["postsolve_fresh"], "slack": data["slack"],
        "native_nominal": data["nominal"], "native_nominal_active": data["nominal_active"],
        "native_fresh_active": data["fresh_active"],
        "sf_affine_pass_fresh_guard_fail": np.zeros(n, dtype=bool),
        "sf_affine_guard_comparison_valid": nominal_valid & normal,
    }
    for prefix in ("nominal", "post"):
        for field in FLOAT_FIELDS:
            arrays[f"{prefix}_{field}"] = np.full(n, np.nan)
        arrays[f"{prefix}_selector_id"] = np.full(n, "", dtype="U128")
        arrays[f"{prefix}_world_gradient"] = np.full((n, 3), np.nan)
        arrays[f"{prefix}_local_q"] = np.full((n, 3), np.nan)
        arrays[f"{prefix}_probability_valid"] = np.zeros(n, dtype=bool)
    for field in FROZEN_FIELDS:
        arrays[f"frozenpost_{field}"] = np.full(n, np.nan)
    arrays["frozenpost_valid"] = np.zeros(n, dtype=bool)
    with _snapshot_imports(experiment / "source_snapshot") as (module, _, implicit):
        polygons = _maze_polygons(experiment / "source_snapshot/sim/simulation_mpc.py")
        wrapper = wrapper_for(module, data["config"], polygons)
        for prefix, mask, states in (("nominal", nominal_valid, data["nominal_states"]),
                                     ("post", normal, data["post_states"])):
            indices = np.flatnonzero(mask)
            for start in range(0, len(indices), batch_size):
                ids = indices[start:start + batch_size]
                fresh, features = fresh_batch(wrapper, states[ids], params)
                for key, value in fresh.items():
                    arrays[f"{prefix}_{key}"][ids] = value
                if prefix == "nominal":
                    fixed = frozen_batch(features, states[ids], data["post_states"][ids], params, implicit)
                    for key, value in fixed.items():
                        arrays[f"frozenpost_{key}"][ids] = value
    arrays["sf_affine_pass_fresh_guard_fail"] = (
        nominal_valid & (arrays["frozenpost_sf_affine_margin_mm"] >= -.001)
        & (arrays["post_guard_margin_mm"] < -.001))
    arrays["frozenpost_probability_valid"] = arrays["frozenpost_valid"].copy()
    # load_trial reports this mismatch but does not mask the original affine
    # array. The cache applies a conservative explicit exclusion additively.
    mask_disagreement = np.zeros(n, dtype=bool)
    if method == "cc_mf":
        mask_disagreement = nominal_valid & (
            (arrays["nominal_mf_required"] == 1) != data["nominal_active"])
        arrays["replay_valid"][mask_disagreement] = False
        arrays["native_affine"][mask_disagreement] = np.nan
    arrays["native_replay_mask_disagreement"] = mask_disagreement
    sf_reference_stats = apply_sf_reference(arrays, sf_params)
    meta = dict(data["metadata"], schema_version=VERSION, extraction_seconds=time.perf_counter() - started,
                epsilon=params["chance_epsilon"], d_min_m=params["d_min"], d_col_m=params["d_col"],
                normal_success_cycles=int(normal.sum()), nominal_available_cycles=int(nominal_valid.sum()),
                post_probability_valid_cycles=int(arrays["post_probability_valid"].sum()),
                nominal_probability_valid_cycles=int(arrays["nominal_probability_valid"].sum()),
                frozenpost_valid_cycles=int(arrays["frozenpost_valid"].sum()),
                native_replay_mask_disagreement_cycles=int(mask_disagreement.sum()),
                selected_distance_match_max_abs_m=float(np.nanmax(np.abs(arrays["post_selected_distance_match_error_m"]))),
                selected_q_match_max_abs=float(np.nanmax(arrays["post_selected_q_match_max_abs_error"])),
                extractor_sha256=sha(__file__), offline_margin_sha256=sha(Path(__file__).parent / "utils/offline_cc_margin.py"),
                selector_sha256=sha(Path(__file__).parent / "utils/offline_feature_selectors.py"))
    meta.update(sf_reference_statistics=sf_reference_stats,
                sf_reference_parameters_source=dict(path=str(sf_path), sha256=sha(sf_path)),
                extraction_pipeline_sha256=sha(__file__))
    arrays["metadata_json"] = np.array(json.dumps(json_clean(meta), ensure_ascii=False))
    arrays["params_json"] = np.array(json.dumps(data["params"]))
    target = output / f"width_{width:03d}mm_{method}_seed_{seed}_stage_{stage}.npz"
    atomic_npz(target, arrays)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--widths-mm", nargs="+", type=int)
    parser.add_argument("--methods", nargs="+", choices=("cc_sf", "cc_mf"), default=["cc_sf", "cc_mf"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 10])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--manifest-name", default="extraction_manifest.json")
    parser.add_argument("--repair-existing-only", action="store_true")
    args = parser.parse_args()
    experiment = args.results_dir.resolve()
    manifest = json.loads((experiment / "experiment.json").read_text())
    if not manifest.get("finished_at_kst"):
        parser.error("The archived driving batch must be complete")
    widths = args.widths_mm or [round(value * 1000) for value in manifest["widths_m"]]
    seeds = args.seeds or manifest["seeds"]
    output_root = args.output_dir or experiment / "constraint_stability_20261105"
    cache = output_root / "cycle_cache"
    cache.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    metadata, errors = [], []
    schema = {
        "schema_version": VERSION, "stage_dt_s": .1,
        "risk_domain": "Fresh p_selected/R/p_max/p_other_max are NaN outside phi>d_min, separated, finite represented features; use prefix_probability_valid.",
        "frozen_domain": "frozenpost_valid uses the nominal separated MF domain; affine feature distances may become negative at the saved postsolve state.",
        "units": {"sf_native_margin_mm": "millimetres; paired SF gamma/floor/cap on BOTH controllers' trajectories", "sf_quantile": "(phi-d_min)/sigma_selected - gamma; dimensionless, UNCLIPPED, not equivalent to native capped SF feasibility when cap/floor active", "mf_quantile": "implemented inverse-CDF with linear extension for R>.5 and lower-tail clipping", "native_affine": "m for SF; standard-normal quantile for MF"},
        "masks": {"post_valid": "successful normal solve only", "nominal_valid": "successful normal solve with finite reconstructed pre-solve nominal", "replay_valid": "native replay validation, including MF activation-mask consistency", "frozenpost_valid": "nominal model probability comparison is valid"},
        "selector_id": "canonical branch identity with endpoint aliases merged; a change alone does not establish discontinuity",
        "other_feature": "Exclude nominal selected-equivalent d/full-q events (tolerances1e-7m/1e-6); other slots are not mutually deduplicated.",
        "probability": "Affine Gaussian marginal probabilities and raw Boole sum with duplicates/correlation; R can exceed1 and is not empirical or exact collision probability.",
        "native_affine": "Before MF lower slack. Source helper native residual arrays preserve its original semantics; use explicit masks.",
        "SF_affine_pass_fresh_guard_fail": "SF affine >= -1um while fresh phi-d_col < -1um; can also be evaluated as a shadow SF row on MF states.",
        "source_hashes_sha256": sha(experiment / "source_hashes.json"),
        "experiment_sha256": sha(experiment / "experiment.json"),
    }
    schema_path = output_root / "cache_schema.json"
    if not schema_path.exists() or json.loads(schema_path.read_text()).get("schema_version") != VERSION:
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    if args.repair_existing_only:
        repaired = repair_existing_sf_reference(cache, experiment)
        report = dict(reason="Enforce the actual paired SF risk cap in common SF rows on MF trajectories", records=repaired,
                      correction_source_sha256=sha(__file__), schema_version=VERSION)
        (output_root / "sf_reference_correction_manifest.json").write_text(json.dumps(json_clean(report), indent=2), encoding="utf-8")
        print(f"Repaired {len(repaired)} existing caches without geometry replay", flush=True)
        return
    for width in widths:
        for method in args.methods:
            for seed in seeds:
                for stage in args.stages:
                    target = cache / f"width_{width:03d}mm_{method}_seed_{seed}_stage_{stage}.npz"
                    if args.skip_existing and target.exists():
                        with np.load(target, allow_pickle=False) as prior:
                            if int(prior["schema_version"]) != VERSION:
                                raise ValueError(f"Schema mismatch in {target}")
                            metadata.append(json.loads(str(prior["metadata_json"])))
                        continue
                    try:
                        meta = extract(experiment, method, width, seed, stage, cache, args.batch_size)
                        metadata.append(meta)
                        print(f"OK width={width} method={method} seed={seed} stage={stage} "
                              f"cycles={meta['cycles']} elapsed={meta['extraction_seconds']:.2f}s", flush=True)
                    except Exception as error:
                        failure = dict(width_mm=width, method=method, seed=seed, stage=stage,
                                       error=f"{type(error).__name__}: {error}")
                        errors.append(failure)
                        print("ERROR " + json.dumps(failure), flush=True)
    result = dict(schema_version=VERSION, requested_widths_mm=widths, requested_methods=args.methods,
                  requested_seeds=seeds, stages=args.stages, caches=metadata, errors=errors,
                  extractor_sha256=sha(__file__), sources=schema,
                  complete=len(metadata) == len(widths) * len(args.methods) * len(seeds) * len(args.stages))
    (output_root / args.manifest_name).write_text(
        json.dumps(json_clean(result), indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
