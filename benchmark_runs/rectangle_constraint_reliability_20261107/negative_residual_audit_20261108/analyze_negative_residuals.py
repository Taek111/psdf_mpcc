"""Explain seed-9 executed-pose residuals using archived geometry only."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import NormalDist
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.analyze_sf_mf_reliability import make_wrapper
from utils.offline_cc_margin import _snapshot_imports
from utils.offline_feature_selectors import _capture_forward, extract_feature_selectors

OUTPUT = Path(__file__).resolve().parent
ANALYSIS = OUTPUT.parent
SUITE = ROOT / "benchmark_runs/rectangle_don_sweep_50_20261105_current_sf"
SNAPSHOT = SUITE / "source_snapshot"


def write_csv(name, rows):
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    clear_path = SNAPSHOT / "sim/start_perturbation.py"
    spec = importlib.util.spec_from_file_location("archived_clearance", clear_path)
    clearance_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clearance_module)
    caches, metadata, selections, selected_deep = {}, {}, {}, {}
    source_hashes = {str(clear_path.relative_to(ROOT)): digest(clear_path)}
    for path in (
        ROOT / "scripts/analyze_sf_mf_reliability.py",
        ROOT / "utils/offline_feature_selectors.py",
        ROOT / "utils/offline_cc_margin.py",
        SNAPSHOT / "models/augmented_psdf.py",
        SNAPSHOT / "models/augmented_psdf_wrapper.py",
        SNAPSHOT / "models/mf_inverse_cdf.py",
        SNAPSHOT / "models/mf_implicit.py",
        SNAPSHOT / "control/cc_covariance.py",
    ):
        source_hashes[str(path.relative_to(ROOT))] = digest(path)
    for method, config in (("cc_sf", "config03"), ("cc_mf", "config04")):
        cache_path = ANALYSIS / "cache" / f"don0.85_{method}_seed9.npz"
        meta_path = SUITE / "widths/width_127p5mm/runs" / config / "trial_0010/data/raw_trace_metadata.json"
        caches[method] = np.load(cache_path)
        metadata[method] = json.loads(meta_path.read_text(encoding="utf-8"))
        for path in (cache_path, meta_path, meta_path.with_name("raw_trace.npz")):
            source_hashes[str(path.relative_to(ROOT))] = digest(path)
        cache = caches[method]
        risk = cache["p_sf" if method == "cc_sf" else "r_mf"]
        valid = np.flatnonzero(cache["eligible"])
        top = valid[np.argsort(risk[valid])[-(3 if method == "cc_sf" else 8):][::-1]]
        selected_deep[method] = set(map(int, top))
        selections[method] = sorted({int(j) for k in top for j in range(max(0, k - 2), min(len(risk), k + 3))})
    # Add the nearest-position comparator itself for the same complete audit.
    for method, other_method in (("cc_sf", "cc_mf"), ("cc_mf", "cc_sf")):
        other = caches[other_method]
        candidates = np.flatnonzero(other["eligible"])
        additions = set(selections[other_method])
        for step in selected_deep[method]:
            pose = caches[method]["poses"][step + 1]
            distances = np.linalg.norm(other["poses"][candidates + 1, :2] - pose[:2], axis=1)
            additions.add(int(candidates[np.argmin(distances)]))
        selections[other_method] = sorted(additions)
    samples, features, comparisons, summaries = [], [], [], []
    with _snapshot_imports(SNAPSHOT) as (module, covariance, _):
        for method in ("cc_sf", "cc_mf"):
            cache, meta = caches[method], metadata[method]
            geometry = meta["geometry"]
            wrapper = make_wrapper(module, meta, torch.float64)
            index = np.array(selections[method])
            poses = cache["poses"][index + 1]
            xi = cache["xi1"][index]
            states = np.column_stack((poses, np.zeros(len(index)), xi))
            result = wrapper.forward_mf_icdf(torch.tensor(states, dtype=torch.float64), .1, .0001,
                                            return_frozen_features=True)
            frozen = result.frozen_features
            d, q, mask = [frozen[name].numpy() for name in ("d", "q", "mask")]
            forward = (np.exp(xi[:, None, 0]) * q[:, :, 0]) ** 2
            coupled = (np.exp(xi[:, None, 1]) * q[:, :, 1] + xi[:, None, 3] * q[:, :, 2]) ** 2
            heading = (np.exp(xi[:, None, 2]) * q[:, :, 2]) ** 2
            variance = forward + coupled + heading
            sigma = np.sqrt(np.where(mask, variance, 1))
            probability = (0.5 * torch.special.erfc(torch.tensor((d - .0001) / sigma) / np.sqrt(2))).numpy()
            probability[~mask] = 0
            _, _, capture = _capture_forward(wrapper, torch.tensor(poses, dtype=torch.float64), True)
            cr = capture["c_R"].reshape(len(index), -1, 2)
            co = capture["c_O"].reshape(len(index), -1, 2)
            selectors = extract_feature_selectors(wrapper, poses, verify=True)
            # Explicitly counterfactual: configured initial covariance with
            # no one-step process-noise propagation, at the same actual pose.
            covariance0 = covariance.initial_local_covariance(SimpleNamespace(**meta["optimizer_parameters"]))
            sigma0 = np.sqrt(np.where(mask, np.einsum("nmi,ij,nmj->nm", q, covariance0, q), 1))
            probability0 = (0.5 * torch.special.erfc(torch.tensor((d - .0001) / sigma0) / np.sqrt(2))).numpy()
            probability0[~mask] = 0
            qsf = selectors["local_q"]
            sigma0_sf = np.sqrt(np.einsum("ni,ij,nj->n", qsf, covariance0, qsf))
            psf0 = (0.5 * torch.special.erfc(torch.tensor((cache["phi"][index] - .0001) / sigma0_sf) / np.sqrt(2))).numpy()
            footprint = [np.asarray(x) for x in geometry["footprint_polygons"]]
            obstacles = [np.asarray(x) for x in geometry["obstacle_polygons"]]
            for row_index, step in enumerate(index):
                active = np.flatnonzero(mask[row_index])
                order = active[np.argsort(probability[row_index, active])[::-1]]
                dominant = int(order[0])
                duplicates = [int(k) for k in active if abs(d[row_index, k] - d[row_index, dominant]) <= 1e-12
                              and np.max(np.abs(q[row_index, k] - q[row_index, dominant])) <= 1e-10
                              and np.max(np.abs(cr[row_index, k] - cr[row_index, dominant])) <= 1e-12
                              and np.max(np.abs(co[row_index, k] - co[row_index, dominant])) <= 1e-12]
                local_covariance = covariance.covariance_from_log_cholesky(xi[row_index])
                dominant_q = q[row_index, dominant]
                explicit_terms = (
                    dominant_q[0] ** 2 * local_covariance[0, 0],
                    dominant_q[1] ** 2 * local_covariance[1, 1],
                    dominant_q[2] ** 2 * local_covariance[2, 2],
                    2 * dominant_q[1] * dominant_q[2] * local_covariance[1, 2],
                )
                risk = cache["p_sf" if method == "cc_sf" else "r_mf"][step]
                sample = {
                    "source": method, "seed": 9, "step": int(step), "time_s": float(cache["times"][step + 1]),
                    "selected_deep": int(step) in selected_deep[method], "eligible": bool(cache["eligible"][step]),
                    "x_m": poses[row_index, 0], "y_m": poses[row_index, 1], "heading_rad": poses[row_index, 2],
                    "p_sf": cache["p_sf"][step], "R_mf": cache["r_mf"][step], "own_residual_pp": 100 * (.1 - risk),
                    "sum_minus_single_pp": 100 * (cache["r_mf"][step] - cache["p_sf"][step]),
                    "true_clearance_mm": 1000 * clearance_module.pose_clearance(poses[row_index], footprint, obstacles, geometry["bounds"]),
                    "psdf_mm": 1000 * cache["phi"][step],
                    "v_m_s": cache["applied_inputs"][step, 0], "omega_rad_s": cache["applied_inputs"][step, 1],
                    "sigma_f_mm": 1000 * np.sqrt(local_covariance[0, 0]),
                    "sigma_l_mm": 1000 * np.sqrt(local_covariance[1, 1]),
                    "sigma_heading_rad": np.sqrt(local_covariance[2, 2]),
                    "dominant_slot": dominant, "dominant_cluster": dominant // 8, "dominant_local_feature": dominant % 8,
                    "dominant_p": probability[row_index, dominant], "dominant_share_R": probability[row_index, dominant] / probability[row_index].sum(),
                    "dominant_d_mm": 1000 * d[row_index, dominant], "dominant_sigma_mm": 1000 * sigma[row_index, dominant],
                    "dominant_required_d90_mm": .1 + 1000 * NormalDist().inv_cdf(.9) * sigma[row_index, dominant],
                    "dominant_zeta": (.0001 - d[row_index, dominant]) / sigma[row_index, dominant],
                    "counterfactual_Sigma0_dominant_sigma_mm": 1000 * sigma0[row_index, dominant],
                    "counterfactual_Sigma0_p_sf": psf0[row_index],
                    "counterfactual_Sigma0_R_mf": probability0[row_index].sum(),
                    "dominant_q_f": q[row_index, dominant, 0], "dominant_q_l": q[row_index, dominant, 1], "dominant_q_theta": q[row_index, dominant, 2],
                    "dominant_variance_forward_fraction": forward[row_index, dominant] / variance[row_index, dominant],
                    "dominant_variance_coupled_fraction": coupled[row_index, dominant] / variance[row_index, dominant],
                    "dominant_variance_heading_fraction": heading[row_index, dominant] / variance[row_index, dominant],
                    "qf2_Sigmaff_mm2": 1e6 * explicit_terms[0],
                    "ql2_Sigmall_mm2": 1e6 * explicit_terms[1],
                    "qtheta2_Sigmatheta_mm2": 1e6 * explicit_terms[2],
                    "cross_2qlqtheta_Sigmalt_mm2": 1e6 * explicit_terms[3],
                    "qf2_Sigmaff_fraction": explicit_terms[0] / variance[row_index, dominant],
                    "ql2_Sigmall_fraction": explicit_terms[1] / variance[row_index, dominant],
                    "qtheta2_Sigmatheta_fraction": explicit_terms[2] / variance[row_index, dominant],
                    "cross_2qlqtheta_Sigmalt_fraction": explicit_terms[3] / variance[row_index, dominant],
                    "dominant_same_witness_slots": " ".join(map(str, duplicates)),
                    "terms_p_gt_0p001": int(np.sum(probability[row_index] > .001)),
                    "second_term_p": probability[row_index, order[1]],
                    "feature_previous": str(cache["canonical_id"][step]), "feature_current": str(cache["canonical_id"][step + 1]),
                    "feature_changed": bool(cache["canonical_id"][step] != cache["canonical_id"][step + 1]),
                    "feature_next_changed": bool(step + 2 < len(cache["canonical_id"]) and cache["canonical_id"][step + 1] != cache["canonical_id"][step + 2]),
                }
                samples.append(sample)
                if int(step) in selected_deep[method]:
                    for feature in order[:8]:
                        features.append({
                            "source": method, "seed": 9, "step": int(step), "time_s": sample["time_s"],
                            "slot": int(feature), "cluster": int(feature) // 8, "local_feature": int(feature) % 8,
                            "d_mm": d[row_index, feature] * 1000, "sigma_mm": sigma[row_index, feature] * 1000,
                            "p": probability[row_index, feature], "q_f": q[row_index, feature, 0], "q_l": q[row_index, feature, 1],
                            "q_heading": q[row_index, feature, 2], "robot_witness_x": cr[row_index, feature, 0], "robot_witness_y": cr[row_index, feature, 1],
                            "obstacle_witness_x": co[row_index, feature, 0], "obstacle_witness_y": co[row_index, feature, 1],
                        })
                    other_method = "cc_sf" if method == "cc_mf" else "cc_mf"
                    other = caches[other_method]
                    candidates = np.flatnonzero(other["eligible"])
                    distances = np.linalg.norm(other["poses"][candidates + 1, :2] - poses[row_index, :2], axis=1)
                    nearest = int(candidates[np.argmin(distances)])
                    theta_delta = other["poses"][nearest + 1, 2] - poses[row_index, 2]
                    wrapped_theta_delta = np.arctan2(np.sin(theta_delta), np.cos(theta_delta))
                    comparisons.append({
                        "source": method, "step": int(step), "time_s": sample["time_s"], "own_residual_pp": sample["own_residual_pp"],
                        "other_source": other_method, "other_step": nearest, "other_time_s": float(other["times"][nearest + 1]),
                        "position_distance_mm": 1000 * distances.min(), "heading_difference_rad": wrapped_theta_delta,
                        "other_x_m": other["poses"][nearest + 1, 0], "other_y_m": other["poses"][nearest + 1, 1],
                        "other_p_sf": other["p_sf"][nearest], "other_R_mf": other["r_mf"][nearest],
                        "other_own_residual_pp": 100 * (.1 - other["p_sf" if other_method == "cc_sf" else "r_mf"][nearest]),
                        "other_v_m_s": other["applied_inputs"][nearest, 0], "other_omega_rad_s": other["applied_inputs"][nearest, 1],
                        "comparison_rule": "nearest XY position among eligible other-trajectory samples; not pose/covariance matched",
                    })
            sf, mf, eligible = cache["p_sf"], cache["r_mf"], cache["eligible"]
            own = sf if method == "cc_sf" else mf
            severity_sf, severity_mf = np.maximum(sf - .1, 0), np.maximum(mf - .1, 0)
            summaries.append({
                "source": method, "eligible_samples": int(eligible.sum()),
                "own_negative_samples": int(np.sum(eligible & (own > .100001))),
                "same_pose_sf_negative_samples": int(np.sum(eligible & (sf > .100001))),
                "same_pose_mf_negative_samples": int(np.sum(eligible & (mf > .100001))),
                "mf_only_exceedance_samples": int(np.sum(eligible & (mf > .100001) & (sf <= .100001))),
                "mean_same_pose_sf_negative_part_pp": 100 * severity_sf[eligible].mean(),
                "mean_same_pose_mf_negative_part_pp": 100 * severity_mf[eligible].mean(),
                "aggregation_increment_in_mean_negative_part_pp": 100 * (severity_mf[eligible] - severity_sf[eligible]).mean(),
                "cache_mf_probability_max_abs_recalculation_error": float(np.max(np.abs(probability.sum(1) - cache["r_mf"][index]))),
            })
    write_csv("sample_metrics.csv", samples)
    write_csv("feature_terms.csv", features)
    write_csv("nearest_other_trajectory_samples.csv", comparisons)
    write_csv("same_pose_cross_evaluation.csv", summaries)
    source_hashes[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    summary = {
        "seed": 9, "don": .85,
        "selection": "Top eight eligible own-risk samples on MF trajectory and top three on SF trajectory; adjacent +/-2 recorded steps and nearest-position other-trajectory comparators.",
        "duplicate_rule": "Same distance within 1e-12 m, q within 1e-10, both body-frame witnesses within 1e-12 m; no deduplication changes to risk score.",
        "scope": "Executed-pose risk with reset-one-step model covariance; never native planning candidate, no optimizer or simulation invoked.",
        "Sigma0_counterfactual": "Risk at the same actual pose using configured initial body covariance only; no one-step process-noise propagation. This is a sensitivity calculation, not an observed planning covariance.",
        "nearest_pose_caveat": "Other trajectory comparisons minimize XY distance only; heading and covariance differ and are not controlled.",
        "source_hashes": source_hashes, "cross_evaluation": summaries,
        "deep_samples": [row for row in samples if row["selected_deep"]],
    }
    (OUTPUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    for row in samples:
        if row["selected_deep"]:
            print({k: row[k] for k in ("source", "time_s", "p_sf", "R_mf", "true_clearance_mm", "dominant_d_mm", "dominant_sigma_mm", "dominant_share_R", "v_m_s", "omega_rad_s", "dominant_same_witness_slots")})


if __name__ == "__main__":
    main()
