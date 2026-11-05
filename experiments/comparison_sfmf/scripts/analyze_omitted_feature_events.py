"""Offline, same-state audit of omitted SF features before geometric incidents.

Run only after a width sweep has completed. No optimizer, solver, or simulation
is instantiated. Events and deterministic reference cycles are selected from
saved telemetry before geometry is evaluated. Stage 1 is reconstructed from
the preceding retained plan and the current cycle's reset covariance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from utils.offline_cc_margin import _maze_polygons, _snapshot_imports
from utils.offline_feature_selectors import extract_feature_selectors


GAPS_M = (0.0001, 0.0005, 0.001, 0.002)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(clean(value)) if isinstance(value, (dict, list, tuple))
                             else clean(value) for key, value in row.items()})


def wrapper_for(module, config, polygons):
    length, width = (config["rectangle_geometry"][key] for key in ("length", "width"))
    vertices = torch.tensor([[-length / 2, -width / 2], [length / 2, -width / 2],
                             [length / 2, width / 2], [-length / 2, width / 2]],
                            dtype=torch.float32)
    wrapper = module.AugmentedPSDFWrapper(vertices, K_max=20, E_max=100, device="cpu")
    wrapper.update_edge_clusters([torch.tensor(p, dtype=torch.float32) for p in polygons],
                                 [torch.tensor(np.roll(p, -1, axis=0), dtype=torch.float32)
                                  for p in polygons])
    return wrapper


def gamma_for(params):
    gamma = params.get("chance_gamma")
    if gamma is not None and math.isfinite(gamma) and gamma >= 0:
        return float(gamma)
    epsilon = float(np.clip(params["chance_epsilon"], 1e-6, .499999))
    return NormalDist().inv_cdf(1 - epsilon)


def q_local(gradient, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * gradient[0] + s * gradient[1],
                     -s * gradient[0] + c * gradient[1], gradient[2]])


def variances(q, xi):
    return ((np.exp(xi[0]) * q[..., 0]) ** 2
            + (np.exp(xi[1]) * q[..., 1] + xi[3] * q[..., 2]) ** 2
            + (np.exp(xi[2]) * q[..., 2]) ** 2)


def sf_frozen(phi, gradient, q, anchor_pose, z, params):
    distance = phi + np.dot(gradient, z[:3] - anchor_pose)
    sigma = math.sqrt(max(float(variances(q, z[4:8])), 0.))
    margin = np.clip(gamma_for(params) * sigma, params.get("risk_margin_floor", 0.),
                     params.get("risk_margin_cap", math.inf))
    return {"distance_m": distance, "sigma_m": sigma,
            "h_m": distance - params["d_min"] - margin,
            "p_selected": NormalDist().cdf((params["d_min"] - distance) / sigma)
            if sigma > 0 else None}


def feature_summary(d, q, mask, z, params, selected=None):
    """Keep the implemented sum, but exclude equivalent selected events separately."""
    sigma = np.sqrt(variances(q, z[4:8]))
    finite_valid = np.isfinite(d) & np.isfinite(q).all(axis=1) & np.isfinite(sigma) & (sigma > 0)
    if np.any(mask & ~finite_valid):
        raise ValueError("A represented feature has invalid distance/gradient/projected variance; partial risk sums are not used")
    valid = mask & finite_valid
    zeta = np.zeros_like(d)
    zeta[valid] = (params["d_min"] - d[valid]) / sigma[valid]
    p = np.zeros_like(d)
    p[valid] = (.5 * torch.special.erfc(
        -torch.tensor(zeta[valid], dtype=torch.float64) / math.sqrt(2))).numpy()
    R = float(p.sum())
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        return {"valid_features": 0}, [], p
    minimum = float(d[candidates].min())
    selected = int(candidates[np.argmin(d[candidates])]) if selected is None else int(selected)
    # Equal distances AND full gradients represent the same affine Gaussian
    # event. Do not merge normals alone: robot lever arms change heading risk.
    equivalent = valid & (np.abs(d - d[selected]) <= 1e-7) \
        & (np.max(np.abs(q - q[selected]), axis=1) <= 1e-6)
    omitted = candidates[~equivalent[candidates]]
    worst = int(omitted[np.argmax(p[omitted])]) if len(omitted) else None
    most_risky = int(candidates[np.argmax(p[candidates])])
    eps = params["chance_epsilon"]
    density_weights = np.zeros_like(d)
    if R > np.finfo(np.float64).tiny:
        quantile = NormalDist().inv_cdf(min(R, .5))
        exponent = -.5 * zeta[valid] ** 2 if R > .5 else .5 * (quantile ** 2 - zeta[valid] ** 2)
        density_weights[valid] = np.exp(np.minimum(exponent, 0.)) / sigma[valid]
    summary = {
        "valid_features": int(valid.sum()), "selected_slot": selected,
        "selected_probability": p[selected], "selected_sigma_m": sigma[selected],
        "R_raw": R, "raw_omitted_sum_including_selected_duplicates": R - p[selected],
        "selected_equivalent_slots": int(equivalent.sum()),
        "non_equivalent_omitted_sum": float(p[omitted].sum()),
        "worst_omitted_slot": worst,
        "worst_omitted_probability": p[worst] if worst is not None else None,
        "worst_omitted_distance_gap_m": d[worst] - d[selected] if worst is not None else None,
        "highest_probability_slot": most_risky,
        "highest_probability_is_selected_equivalent": bool(equivalent[most_risky]),
        "selected_passes_but_distinct_omitted_fails": bool(
            p[selected] <= eps and worst is not None and p[worst] > eps),
        "selected_passes_but_Boole_sum_fails": bool(p[selected] <= eps and R > eps),
        "near_min_counts": {f"{gap * 1000:g}mm": int(np.sum(valid & (d <= minimum + gap)))
                            for gap in GAPS_M},
        "near_min_non_equivalent_counts": {
            f"{gap * 1000:g}mm": int(np.sum(valid & ~equivalent & (d <= minimum + gap)))
            for gap in GAPS_M},
    }
    keep = set(candidates[np.argsort(-p[candidates])[:8]].tolist())
    keep.update(candidates[np.argsort(d[candidates])[:8]].tolist())
    keep.add(selected)
    if worst is not None:
        keep.add(worst)
    ranks = np.empty_like(d, dtype=int)
    ranks[candidates[np.argsort(d[candidates])]] = np.arange(1, len(candidates) + 1)
    rows = [{"slot": i, "cluster": i // 8, "family": "R_to_O" if i % 8 < 4 else "O_to_R",
             "robot_feature": i % 4, "distance_rank": int(ranks[i]),
             "distance_m": d[i], "gap_from_min_m": d[i] - minimum, "sigma_m": sigma[i],
             "probability": p[i], "q_forward": q[i, 0], "q_lateral": q[i, 1],
             "q_heading_m": q[i, 2], "selected_equivalent": bool(equivalent[i]),
             "icdf_pose_weight_per_m": density_weights[i]}
            for i in sorted(keep)]
    return summary, rows, p


def evaluate_stage(wrapper, implicit, nominal, solution, actual, params):
    """Separate nominal selection, affine RTI, frozen geometry, and fresh geometry."""
    all_states = np.stack([nominal, solution, actual])
    selectors = extract_feature_selectors(
        wrapper, torch.tensor(all_states[:, :3], dtype=torch.float32), verify=True)
    with torch.no_grad():
        evaluated = wrapper.forward_mf_icdf(
            torch.tensor(all_states, dtype=torch.float64), params["chance_epsilon"],
            params["d_min"], stage_requested=np.ones(3, dtype=bool), return_frozen_features=True)
    frozen = evaluated.frozen_features
    d = frozen["d"].numpy()
    q = frozen["q"].numpy()
    mask = frozen["mask"].numpy()
    world_q = frozen["pose_feature_gradient"].numpy()
    phi = evaluated.phi.numpy()
    gradient = frozen["gradient"].numpy()
    np.testing.assert_allclose(phi, selectors["phi"], rtol=0, atol=1e-8)
    selected_slots = [int(selectors["active_cluster"][i]) * 8 + int(selectors["feature_index"][i])
                      if selectors["separated"][i] else None for i in range(3)]
    for i, slot in enumerate(selected_slots):
        if slot is not None and phi[i] > params["d_min"]:
            np.testing.assert_allclose(d[i, slot], phi[i], rtol=0, atol=1e-7)
            np.testing.assert_allclose(q[i, slot], q_local(gradient[i], all_states[i, 2]),
                                       rtol=0, atol=1e-6)
    nominal_phi = float(phi[0])
    nominal_q = q_local(gradient[0], nominal[2])
    sigma_nominal = math.sqrt(float(variances(nominal_q, nominal[4:8])))
    gamma = gamma_for(params)
    raw_margin = gamma * sigma_nominal
    variance_gradient = implicit.projected_variance_and_gradient(
        torch.tensor(nominal_q[None, None, :], dtype=torch.float64),
        torch.tensor(nominal[None], dtype=torch.float64))[1].numpy()[0, 0]
    margin_gradient = gamma * variance_gradient / (2 * sigma_nominal)
    if not params.get("risk_margin_floor", 0.) < raw_margin < params.get("risk_margin_cap", math.inf):
        margin_gradient[:] = 0.
    anchor = nominal[:3].astype(np.float32).astype(float)
    nominal_sf = sf_frozen(nominal_phi, gradient[0], nominal_q, anchor, nominal, params)
    result = {"nominal": {}, "solution": {}, "actual_next": {},
              "nominal_mf_domain_active": bool(evaluated.required[0]),
              "nominal_mf_evaluation_valid": bool(evaluated.valid[0]),
              "nominal_mf_quantile": float(evaluated.residual[0]),
              "nominal_sf_h_m": nominal_sf["h_m"],
              "nominal_sf_probability": nominal_sf["p_selected"],
              "actual_covariance_source": "Saved solution stage1 body-local planning covariance held numerically fixed at actual heading; world covariance is not held fixed and this is not measured covariance",
              "actual_minus_solution_translation_m": float(np.linalg.norm(actual[:2] - solution[:2])),
              "actual_minus_solution_heading_rad": float(math.atan2(
                  math.sin(actual[2] - solution[2]), math.cos(actual[2] - solution[2])))}
    feature_rows = []
    nominal_selected = selected_slots[0]
    nominal_probability = None
    for index, (name, z) in enumerate(zip(("nominal", "solution", "actual_next"), all_states)):
        fresh, features, probs = feature_summary(d[index], q[index], mask[index], z, params,
                                                selected=selected_slots[index])
        fresh["signed_psdf_m"] = float(phi[index])
        fresh["psdf_selector_id"] = str(selectors["canonical_id"][index])
        fresh["selected_slot_is_actual_separated_sf_branch"] = selected_slots[index] is not None
        fresh["probability_interpretation_valid_separated_domain"] = bool(phi[index] > params["d_min"])
        if not fresh["probability_interpretation_valid_separated_domain"]:
            fresh["selected_passes_but_distinct_omitted_fails"] = None
            fresh["selected_passes_but_Boole_sum_fails"] = None
        fresh["fresh_mf_domain_active"] = bool(evaluated.required[index])
        fresh["fresh_mf_quantile"] = float(evaluated.residual[index])
        fresh_sf_q = q_local(gradient[index], z[2])
        fresh["fresh_sf"] = sf_frozen(float(phi[index]), gradient[index], fresh_sf_q,
                                      z[:3].astype(np.float32).astype(float), z, params)
        result[name]["fresh"] = fresh
        feature_rows.extend(dict(row, evaluation=name + "_fresh") for row in features)
        if index == 0:
            nominal_probability = probs
            continue
        distance_frozen = d[0] + world_q[0] @ (z[:3] - nominal[:3])
        fixed, features, _ = feature_summary(distance_frozen, q[0], mask[0], z, params,
                                             selected=nominal_selected)
        fixed["nominal_domain_active"] = bool(evaluated.required[0])
        fixed["valid_as_probability_reference"] = bool(phi[0] > params["d_min"])
        fixed["frozen_sf"] = sf_frozen(nominal_phi, gradient[0], nominal_q, anchor, z, params)
        fixed["sf_rti_affine_h_m"] = (
            nominal_phi + np.dot(gradient[0], z[:3] - anchor) - params["d_min"]
            - np.clip(raw_margin, params.get("risk_margin_floor", 0.), params.get("risk_margin_cap", math.inf))
            - np.dot(margin_gradient, z[4:8] - nominal[4:8]))
        fixed["mf_rti_affine_quantile_without_slack"] = (
            float(evaluated.A[0].numpy() @ z + float(evaluated.c[0]))
            if bool(evaluated.valid[0]) else None)
        worst = fixed["worst_omitted_slot"]
        fixed["worst_omitted_probability_at_nominal"] = (
            nominal_probability[worst] if worst is not None else None)
        result[name]["frozen_nominal_geometry"] = fixed
        feature_rows.extend(dict(row, evaluation=name + "_frozen",
                                 probability_at_nominal=nominal_probability[row["slot"]])
                            for row in features)
    result["nominal_selected_matches_psdf_distance_m"] = (
        float(d[0, nominal_selected] - phi[0]) if nominal_selected is not None else None)
    result["nominal_selected_matches_psdf_q_max_abs"] = (
        float(np.max(np.abs(q[0, nominal_selected] - nominal_q))) if nominal_selected is not None else None)
    omitted_probability = result["nominal"]["fresh"]["worst_omitted_probability"]
    result["nominal_model_distinct_omitted_risk_miss"] = bool(
        nominal_sf["p_selected"] <= params["chance_epsilon"]
        and omitted_probability is not None and omitted_probability > params["chance_epsilon"]
    ) if phi[0] > params["d_min"] and nominal_selected is not None else None
    result["sf_affine_passes_but_solution_fresh_guard_fails"] = bool(
        result["solution"]["frozen_nominal_geometry"]["sf_rti_affine_h_m"] >= -1e-6
        and phi[1] < params["d_col"] - 1e-6)
    return result, feature_rows


def select_samples(cycles, poses, clearance, params, stride):
    """Select incident onsets before looking at reconstructed feature metrics."""
    event_indices = {}
    for label, threshold in (("first_current_phi_below_guard", params["d_col"]),
                             ("first_current_phi_below_0.5mm", .0005),
                             ("first_current_phi_below_zero", 0.)):
        indices = [i for i, row in enumerate(poses) if float(row["sdf_value"]) < threshold]
        if indices:
            event_indices.setdefault(indices[0], []).append(label)
    for label, predicate in (("first_exact_clearance_zero_contact_or_overlap", lambda v: v == 0),
                             ("first_exact_clearance_negative_boundary_violation", lambda v: v < 0)):
        indices = [int(row["pose_index"]) for row in clearance if predicate(float(row["clearance_m"]))]
        if indices:
            event_indices.setdefault(indices[0], []).append(label)
    samples = [{"kind": "first_geometric_event", "pose_index": index, "cycle": index - 1,
                "event_types": labels} for index, labels in sorted(event_indices.items())]
    successful = [i for i, row in enumerate(cycles) if row["solver_success"].lower() == "true"]
    event_cycles = [index - 1 for index in event_indices]
    for i in successful[::stride]:
        if not any(abs(i - event) <= 5 for event in event_cycles):
            samples.append({"kind": "deterministic_reference", "pose_index": i + 1,
                            "cycle": i, "event_types": []})
    return samples


def cross_width_references(experiment, manifest, reference_width):
    """Find same-seed SF reference cycles at similar progress, without geometry replay."""
    references = {}
    for seed in manifest["seeds"]:
        reference = experiment / f"width_{reference_width:03d}mm/cc_sf/seed_{seed}"
        files = list((reference / "data").glob("pose_sdf_*.csv"))
        if len(files) != 1 or not (reference / "cycles.csv").exists():
            continue
        poses, cycles = read_csv(files[0]), read_csv(reference / "cycles.csv")
        if not poses or len(poses) != len(cycles):
            continue
        successful = [i for i, row in enumerate(cycles) if row["solver_success"].lower() == "true"]
        if not successful:
            continue
        chosen = {}
        for width_m in manifest["widths_m"]:
            width = round(width_m * 1000)
            if width == reference_width:
                continue
            target = experiment / f"width_{width:03d}mm/cc_sf/seed_{seed}"
            files = list((target / "data").glob("pose_sdf_*.csv"))
            if len(files) != 1:
                continue
            target_poses = read_csv(files[0])
            params = json.loads((target / "optimizer_parameters.json").read_text())
            incidents = [i for i, row in enumerate(target_poses)
                         if float(row["sdf_value"]) < params.get("d_col", .00005)]
            if not incidents or incidents[0] == 0:
                continue
            event_cycle = incidents[0] - 1
            progress = float(target_poses[event_cycle]["path_s"])
            cycle = min(successful, key=lambda i: abs(float(poses[i]["path_s"]) - progress))
            gap = float(poses[cycle]["path_s"]) - progress
            chosen.setdefault(cycle, []).append({"event_width_mm": width,
                "event_cycle": event_cycle, "event_progress_m": progress,
                "reference_progress_gap_m": gap, "within_50mm_progress": abs(gap) <= .05})
        references[reference_width, "cc_sf", seed] = [
            {"kind": "cross_width_progress_reference", "cycle": cycle,
             "pose_index": cycle + 1, "event_types": [], "cross_width_targets": targets}
            for cycle, targets in chosen.items()]
    return references


def paired_mf_progress_references(experiment, manifest):
    """Select actual MF telemetry near SF incident progress, never identical states."""
    references = {}
    for width_m in manifest["widths_m"]:
        width = round(width_m * 1000)
        for seed in manifest["seeds"]:
            sf = experiment / f"width_{width:03d}mm/cc_sf/seed_{seed}"
            mf = experiment / f"width_{width:03d}mm/cc_mf/seed_{seed}"
            sf_files = list((sf / "data").glob("pose_sdf_*.csv"))
            mf_files = list((mf / "data").glob("pose_sdf_*.csv"))
            if len(sf_files) != 1 or len(mf_files) != 1 or not (mf / "cycles.csv").exists():
                continue
            sf_poses, mf_poses = read_csv(sf_files[0]), read_csv(mf_files[0])
            params = json.loads((sf / "optimizer_parameters.json").read_text())
            incidents = [i for i, row in enumerate(sf_poses)
                         if float(row["sdf_value"]) < params.get("d_col", .00005)]
            if not incidents or incidents[0] == 0:
                continue
            event_cycle = incidents[0] - 1
            progress = float(sf_poses[event_cycle]["path_s"])
            cycles = read_csv(mf / "cycles.csv")
            if not mf_poses or len(mf_poses) != len(cycles):
                continue
            successful = [i for i, row in enumerate(cycles) if row["solver_success"].lower() == "true"]
            if not successful:
                continue
            cycle = min(successful, key=lambda i: abs(float(mf_poses[i]["path_s"]) - progress))
            gap = float(mf_poses[cycle]["path_s"]) - progress
            references[width, "cc_mf", seed] = [{
                "kind": "paired_mf_progress_reference", "cycle": cycle, "pose_index": cycle + 1,
                "event_types": [], "sf_event_cycle": event_cycle,
                "sf_event_progress_m": progress, "mf_reference_progress_gap_m": gap,
                "within_50mm_progress": abs(gap) <= .05,
                "comparison_scope": "Same width/seed and similar progress; pose, covariance, input, and nominal trajectory differ",
            }]
    return references


def audit_trial(experiment, method, width, seed, module, covariance, implicit, polygons, stride,
                additional_samples=()):
    directory = experiment / f"width_{width:03d}mm" / method / f"seed_{seed}"
    params = json.loads((directory / "optimizer_parameters.json").read_text())
    params.setdefault("d_col", .00005)
    config = yaml.safe_load((directory / "config.yaml").read_text())
    if config["localization_error"]["enabled"] or params.get("enable_backup_solver"):
        raise ValueError("This audit requires localization error and backup solver disabled")
    cycles = read_csv(directory / "cycles.csv")
    pose_files = list((directory / "data").glob("pose_sdf_*.csv"))
    if len(pose_files) != 1:
        raise ValueError("Expected one measured pose log")
    poses, clearance = read_csv(pose_files[0]), read_csv(directory / "clearance.csv")
    checkpoint = [json.loads(line) for line in (directory / "animation_checkpoint.jsonl").read_text().splitlines()]
    states = np.asarray([row["optimized_traj"] for row in checkpoint[1:]], dtype=float)
    if len(states) != len(cycles) or len(poses) != len(cycles) or len(clearance) != len(cycles) + 1:
        raise ValueError("Cycle/pose/checkpoint/clearance alignment mismatch")
    np.testing.assert_array_equal([int(row["step"]) for row in cycles], np.arange(len(cycles)))
    np.testing.assert_array_equal([int(row["timestep"]) for row in poses], np.arange(len(cycles)))
    np.testing.assert_array_equal([int(row["pose_index"]) for row in clearance], np.arange(len(clearance)))
    np.testing.assert_allclose([float(row["simulation_time_s"]) for row in clearance],
                               np.arange(len(clearance)) * params["tf"] / params["horizon"],
                               rtol=0, atol=1e-8)
    if states.ndim != 3 or states.shape[1] != params["horizon"] + 1 or states.shape[2] != 10:
        raise ValueError("Expected saved augmented trajectories including physical-input memory")
    history = read_csv(directory / "data/trial_history.csv")
    if len(history) != len(cycles):
        raise ValueError("Applied-input history and cycles disagree")
    applied_inputs = np.array([[float(row[key]) for key in ("v", "omega")] for row in history])
    np.testing.assert_allclose(states[:, 1, 8:10], applied_inputs, rtol=0, atol=1e-7)
    current = np.array([[float(row[key]) for key in ("x", "y", "theta")] for row in poses])
    np.testing.assert_allclose(states[:, 0, :3], current, rtol=0, atol=1e-6)
    wrapper = wrapper_for(module, config, polygons)
    p = SimpleNamespace(**params)
    mf_files = list((directory / "data").glob("cc_mf_cycle_*.csv"))
    mf_telemetry = read_csv(mf_files[0]) if len(mf_files) == 1 else None
    if mf_telemetry is not None and len(mf_telemetry) != len(cycles):
        raise ValueError("MF telemetry and cycle counts disagree")
    records, features = [], []
    samples = select_samples(cycles, poses, clearance, params, stride) + list(additional_samples)
    for sample in samples:
        cycle = sample["cycle"]
        record = dict(sample, method=method, width_mm=width, seed=seed,
                      sample_id=f"{width}_{method}_{seed}_{cycle}_{sample['kind']}")
        records.append(record)
        if cycle < 1:
            record["unavailable"] = "No preceding retained-plan snapshot; initial event or first solve"
            continue
        record["preceding_solve_mode"] = cycles[cycle]["solve_mode"]
        if cycles[cycle]["solver_success"].lower() != "true":
            record["unavailable"] = "Incident follows unsuccessful control; no successful event-producing solve"
            continue
        previous_mode = cycles[cycle - 1]["solve_mode"]
        if cycles[cycle - 1]["solver_success"].lower() != "true" and not previous_mode.startswith("safe_stop"):
            record["unavailable"] = "Retained solver trajectory unknown"
            continue
        if previous_mode.startswith("safe_stop") and not (
            np.allclose(states[cycle - 1, :, :3], current[cycle - 1], rtol=0, atol=1e-12)
            and np.allclose(states[cycle - 1, 1:, 8:10], 0., rtol=0, atol=1e-12)):
            record["unavailable"] = "Previous safe-stop retained plan was not validated"
            continue
        nominal = states[cycle - 1, 1, :8].copy()
        xi0 = covariance.log_cholesky_from_covariance(covariance.initial_local_covariance(
            p, SimpleNamespace(_x=current[cycle])))
        v, omega = states[cycle - 1, 1, 8:10]
        nominal[4:8] = covariance.propagate_log_cholesky(
            xi0, v, omega, 0., params["tf"] / params["horizon"], p)
        solution = states[cycle, 1, :8].copy()
        actual = solution.copy()
        actual[:3] = [float(clearance[cycle + 1][key]) for key in ("x", "y", "theta")]
        applied_v, applied_omega = states[cycle, 1, 8:10]
        expected_xi1 = covariance.propagate_log_cholesky(
            xi0, applied_v, applied_omega, 0., params["tf"] / params["horizon"], p)
        expected_covariance = covariance.covariance_from_log_cholesky(expected_xi1)
        solved_covariance = covariance.covariance_from_log_cholesky(solution[4:8])
        record.update(current_progress_m=float(poses[cycle]["path_s"]),
                      current_psdf_m=float(poses[cycle]["sdf_value"]),
                      actual_next_clearance_m=float(clearance[cycle + 1]["clearance_m"]),
                      nominal_state=nominal.tolist(), solution_state=solution.tolist(),
                      actual_next_state_with_predicted_covariance=actual.tolist(),
                      applied_input_from_checkpoint=[float(applied_v), float(applied_omega)],
                      xi1_exact_covariance_rollout=expected_xi1.tolist(),
                      xi1_covariance_dynamics_defect=(solution[4:8] - expected_xi1).tolist(),
                      covariance_dynamics_defect_entries=(solved_covariance - expected_covariance).tolist(),
                      covariance_dynamics_defect_frobenius=float(np.linalg.norm(
                          solved_covariance - expected_covariance)),
                      initial_pose=[float(clearance[0][key]) for key in ("x", "y", "theta")])
        evaluated, rows = evaluate_stage(wrapper, implicit, nominal, solution, actual, params)
        record.update(evaluated)
        if cycle + 1 < len(poses):
            record["actual_next_signed_psdf_replay_error_m"] = (
                evaluated["actual_next"]["fresh"]["signed_psdf_m"] - float(poses[cycle + 1]["sdf_value"]))
        features.extend(dict(row, sample_id=record["sample_id"], kind=sample["kind"],
                             method=method, width_mm=width, seed=seed, cycle=cycle) for row in rows)
        if mf_telemetry is not None:
            logged = mf_telemetry[cycle]
            record["actual_mf_nominal_stage1_active"] = json.loads(logged["mf_active"])[1]
            record["actual_mf_stage1_slack_quantile"] = json.loads(logged["mf_slack"])[1]
            record["actual_mf_logged_nominal_R"] = json.loads(logged["mf_probability_sum"])[1]
            if record["actual_mf_logged_nominal_R"] is not None:
                record["mf_nominal_R_replay_error"] = (
                    record["nominal"]["fresh"]["R_raw"] - record["actual_mf_logged_nominal_R"])
                record["mf_nominal_R_replay_within_1e_5"] = abs(record["mf_nominal_R_replay_error"]) <= 1e-5
    return records, features


def sample_summary(record):
    row = {key: record.get(key) for key in (
        "sample_id", "kind", "method", "width_mm", "seed", "cycle", "event_types",
        "unavailable", "current_progress_m", "current_psdf_m", "actual_next_clearance_m",
        "actual_minus_solution_translation_m", "actual_minus_solution_heading_rad",
        "nominal_model_distinct_omitted_risk_miss", "nominal_mf_domain_active",
        "nominal_sf_h_m", "sf_affine_passes_but_solution_fresh_guard_fails",
        "actual_mf_nominal_stage1_active", "actual_mf_stage1_slack_quantile",
        "mf_nominal_R_replay_error", "covariance_dynamics_defect_frobenius")}
    for name in ("nominal", "solution", "actual_next"):
        fresh = record.get(name, {}).get("fresh", {})
        for key in ("signed_psdf_m", "selected_probability", "worst_omitted_probability", "R_raw",
                    "selected_passes_but_distinct_omitted_fails", "selected_passes_but_Boole_sum_fails"):
            row[name + "_" + key] = fresh.get(key)
        row[name + "_near_min_nonselected_equivalent_slots_0.5mm"] = fresh.get("near_min_non_equivalent_counts", {}).get("0.5mm")
        fixed = record.get(name, {}).get("frozen_nominal_geometry", {})
        for key in ("sf_rti_affine_h_m", "mf_rti_affine_quantile_without_slack",
                    "worst_omitted_probability", "worst_omitted_probability_at_nominal", "R_raw"):
            row[name + "_frozen_" + key] = fixed.get(key)
        row[name + "_frozen_sf_h_m"] = fixed.get("frozen_sf", {}).get("h_m")
        row[name + "_fresh_sf_h_m"] = fresh.get("fresh_sf", {}).get("h_m")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-stride", type=int, default=25)
    parser.add_argument("--methods", nargs="+", default=["cc_sf", "cc_mf"])
    parser.add_argument("--reference-width-mm", type=int, default=125)
    args = parser.parse_args()
    experiment = args.results_dir.resolve()
    manifest = json.loads((experiment / "experiment.json").read_text())
    if not manifest.get("finished_at_kst"):
        parser.error("Wait until the driving batch is complete before running this offline audit")
    if args.reference_stride < 1:
        parser.error("reference-stride must be positive")
    output = args.output_dir or experiment / "omitted_feature_events"
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    records, features, exclusions = [], [], []
    references = cross_width_references(experiment, manifest, args.reference_width_mm)
    references.update(paired_mf_progress_references(experiment, manifest))
    polygons = _maze_polygons(experiment / "source_snapshot/sim/simulation_mpc.py")
    with _snapshot_imports(experiment / "source_snapshot") as (module, covariance, implicit):
        for width_m in manifest["widths_m"]:
            width = round(width_m * 1000)
            for method in args.methods:
                for seed in manifest["seeds"]:
                    try:
                        rows, feature_rows = audit_trial(experiment, method, width, seed, module,
                                                        covariance, implicit, polygons, args.reference_stride,
                                                        references.get((width, method, seed), ()))
                        records.extend(rows)
                        features.extend(feature_rows)
                    except (FileNotFoundError, ValueError, AssertionError) as error:
                        exclusions.append({"width_mm": width, "method": method, "seed": seed,
                                           "reason": str(error)})
    report = {
        "scope": {"experiment": str(experiment), "stage": 1,
                  "reference_stride": args.reference_stride, "near_min_gaps_m": GAPS_M,
                  "event_rule": "First threshold crossing per type per trial; shared indices merged",
                  "reference_rule": f"Every {args.reference_stride}th successful cycle, excluding +/-5 cycles of selected incidents",
                  "cross_width_reference_mm": args.reference_width_mm,
                  "paired_mf_reference_rule": "Same width/seed MF successful cycle nearest in progress to first SF guard incident"},
        "records": records, "excluded_trials": exclusions,
        "counts": dict(Counter((row["kind"] + ("_unavailable" if "unavailable" in row else "_evaluated"))
                               for row in records)),
        "definitions_and_limits": [
            "Probabilities refer to represented affine Gaussian feature distances, not empirical or exact nonlinear collision probabilities.",
            "Raw MF probability sum includes duplicate and correlated events; sum>epsilon alone does not prove true risk>epsilon.",
            "A distinct omitted p_l>epsilon while selected p<=epsilon is stronger local-model coverage evidence, conditional on separated nominal domain.",
            "MF features still contain internal nearest-edge/vertex selection; they do not enumerate every primitive correspondence.",
            "Full gradient including heading lever arm and distance define selected-equivalent events for the separate omitted-feature metric; the implemented raw sum is not deduplicated.",
            "Near-min non-equivalent counts exclude copies of the selected event but do not deduplicate alternative slots against one another.",
            "Fresh postsolve geometry is distinct from nominal/frozen geometry. An alternative becoming risky only after a large update may be an RTI linearization problem.",
            "MF counterfactual rows are evaluated without slack and are not solved; they do not prove the implemented soft controller would prevent the incident.",
            "Actual-next risk uses solved stage1 covariance; position discrepancy and covariance-dynamics defects remain separate explanations.",
            "A clearance value of zero conflates contact and overlap; negative clearance reports a map-boundary violation. Predicted penetration is not an actual contact event.",
            "CPU float32 reconstruction can differ from CUDA feature selection at ties. Saved source and model formulas are retained.",
            "Multiple incident types from one trial and reference cycles are correlated; they are not independent trials.",
            "No same-progress MF trajectory is treated as the same pose/covariance counterfactual.",
            "Cross-width SF references match seed and nearest progress only; footprint geometry and accepted initial poses may differ. Progress gaps are reported, including gaps above 50mm.",
        ],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "snapshot_hashes_file_sha256": hashlib.sha256((experiment / "source_hashes.json").read_bytes()).hexdigest(),
    }
    (output / "analysis.json").write_text(json.dumps(clean(report), indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output / "features.csv", features)
    write_csv(output / "samples.csv", [sample_summary(record) for record in records])
    print(json.dumps({"output": str(output), "records": len(records), "excluded_trials": len(exclusions)}))


if __name__ == "__main__":
    main()
