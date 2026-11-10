"""All recorded DoN .85 SF/MF normal steps: retained-obstacle diagnostics.

No solver/simulation is executed. The obstacle nearest the current actual pose
is retained as an identity; its distance/gradient are freshly evaluated at the
next actual pose. This is neither a retained vertex/edge primitive nor the
unlogged nominal feature used by the real optimizer.
"""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
SUITE = ROOT / "benchmark_runs/rectangle_constraint_reliability_20261107/rectangle_don_sweep_50_20261105_current_sf"
sys.path.insert(0, str(ROOT))
from scripts.analyze_sf_mf_reliability import make_wrapper
from utils.offline_cc_margin import _snapshot_imports

HELPER = OUTPUT.parent / "high_don_mechanism_20261109/recorded_cluster_risks.py"
spec = importlib.util.spec_from_file_location("recorded_cluster_risks", HELPER)
risk_helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(risk_helper)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(name, rows):
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def counts(rows):
    n = len(rows)
    switched = [r for r in rows if r["cluster_changed"]]
    output = {"normal_steps": n, "contributing_trials": len({(r["source"], r["seed"]) for r in rows}),
              "cluster_switch_steps": len(switched), "terminal_guard_entry_steps": sum(r["terminal_guard_entry"] for r in rows)}
    for key in ("hidden_dcol_unsafe", "hidden_dmin_unsafe", "probability_false_safe", "probability_false_safe_strict_tol"):
        selected = [r for r in rows if r[key]]
        output[key + "_steps"] = len(selected)
        output[key + "_rate_per_normal_step"] = len(selected) / n if n else None
        output[key + "_rate_per_switched_step"] = len(selected) / len(switched) if switched else None
        output[key + "_any_trials"] = len({(r["source"], r["seed"]) for r in selected})
        output[key + "_terminal_guard_entry_steps"] = sum(r["terminal_guard_entry"] for r in selected)
    output["geometric_ignore_gap_mean_mm"] = float(np.mean([r["geometric_ignore_gap_mm"] for r in rows])) if n else None
    output["geometric_ignore_gap_max_mm"] = max((r["geometric_ignore_gap_mm"] for r in rows), default=None)
    output["signed_residual_optimism_mean_pp"] = float(np.mean([r["signed_residual_optimism_pp"] for r in rows])) if n else None
    output["positive_residual_optimism_mean_pp"] = float(np.mean([max(0, r["signed_residual_optimism_pp"]) for r in rows])) if n else None
    return output


def main():
    torch.set_num_threads(3)
    folder = SUITE / "widths/width_127p5mm"
    trial_path = folder / "trial_results.csv"
    trials = [r for r in csv.DictReader(trial_path.open(encoding="utf-8-sig")) if r["optimizer_type"] in ("cc_sf", "cc_mf")]
    if len(trials) != 100 or any(len({int(r["start_seed"]) for r in trials if r["optimizer_type"] == m}) != 50 for m in ("cc_sf", "cc_mf")):
        raise ValueError("Expected all 50 trials of each controller")
    step_rows, trial_rows, sources = [], [], []
    sources.append({"path": str(trial_path.relative_to(ROOT)), "sha256": sha(trial_path)})
    with _snapshot_imports(SUITE / "source_snapshot") as (module, covariance, _):
        for trial in trials:
            source, seed = trial["optimizer_type"], int(trial["start_seed"])
            directory = folder / "runs" / trial["configuration_id"] / f"trial_{int(trial['trial_number']):04d}" / "data"
            npz_path, meta_path = directory / "raw_trace.npz", directory / "raw_trace_metadata.json"
            raw = np.load(npz_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            poses, inputs, times = raw["poses"], raw["applied_inputs"], raw["pose_times_s"]
            statuses = meta["solver_status_infos"]
            if sha(npz_path) != meta["raw_trace_sha256"] or len(poses) != len(inputs) + 1 or len(statuses) != len(inputs):
                raise ValueError(f"Raw alignment/hash mismatch: {directory}")
            if not all(np.isfinite(raw[key]).all() for key in raw.files):
                raise ValueError(f"Non-finite raw arrays: {directory}")
            if meta["outcome"]["status"] != trial["status"]:
                raise ValueError(f"Outcome mismatch: {directory}")
            params = SimpleNamespace(**meta["optimizer_parameters"])
            if params.initial_pose_covariance is not None:
                raise ValueError("This initial-covariance reconstruction expects configured body sigmas")
            if params.enable_backup_solver:
                raise ValueError("Normal-solve interpretation requires this archived no-backup configuration")
            xi0 = covariance.log_cholesky_from_covariance(covariance.initial_local_covariance(params))
            xi1 = covariance.propagate_log_cholesky(xi0, inputs[:, 0], inputs[:, 1], 0, meta["dt_s"], params)
            states_xi = np.vstack((xi0, xi1))
            wrapper = make_wrapper(module, meta, torch.float64)
            collected = {name: [] for name in ("phi", "probability_surrogate", "sigma", "active_cluster", "mf_domain_active")}
            for start in range(0, len(poses), 256):
                result = risk_helper.evaluate_cluster_risks(wrapper, poses[start:start + 256], states_xi[start:start + 256], params.d_min, params.chance_epsilon)
                for name in collected:
                    collected[name].append(result[name])
            evaluated = {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}
            clusters = evaluated["active_cluster"]
            eligible = np.array([s["success"] and s.get("solver_called") is True for s in statuses], dtype=bool)
            guard = np.array([s["raw_status"] == "safe_stop_stage0" for s in statuses])
            normal_indices = np.flatnonzero(eligible)
            local_rows = []
            for step in normal_indices:
                old, fresh = int(clusters[step]), int(clusters[step + 1])
                d_old, d_fresh = evaluated["phi"][step + 1, [old, fresh]]
                p_old, p_fresh = evaluated["probability_surrogate"][step + 1, [old, fresh]]
                b_old, b_fresh = params.chance_epsilon - p_old, params.chance_epsilon - p_fresh
                next_guard = step + 1 < len(guard) and bool(guard[step + 1])
                terminal_entry = bool(next_guard and guard[step + 1:].all() and trial["status"] == "failure")
                next_pose = poses[step + 1]
                record = {
                    "source": source, "seed": seed, "trial_outcome": trial["status"], "step": int(step),
                    "current_time_s": float(times[step]), "next_recorded_time_s": float(times[step + 1]),
                    "current_x_m": poses[step, 0], "current_y_m": poses[step, 1], "current_heading_rad": poses[step, 2],
                    "next_x_m": next_pose[0], "next_y_m": next_pose[1], "next_heading_rad": next_pose[2],
                    "applied_v_m_s": inputs[step, 0], "applied_omega_rad_s": inputs[step, 1],
                    "old_cluster": old, "fresh_cluster": fresh, "cluster_changed": old != fresh,
                    "next_center_first_corner_roi": bool(0 <= next_pose[0] <= .2 and .45 <= next_pose[1] <= .65),
                    "next_status_stage0_guard": bool(next_guard), "terminal_guard_entry": terminal_entry,
                    "d_old_cluster_next_mm": 1000 * float(d_old), "d_fresh_cluster_next_mm": 1000 * float(d_fresh),
                    "geometric_ignore_gap_mm": 1000 * float(max(0, d_old - d_fresh)),
                    "d_col_mm": 1000 * params.d_col, "d_min_mm": 1000 * params.d_min,
                    "endpoint_phi_gt_dmin": bool(d_fresh > params.d_min),
                    "hidden_dcol_unsafe": bool(d_old >= params.d_col and d_fresh < params.d_col),
                    "hidden_dmin_unsafe": bool(d_old >= params.d_min and d_fresh < params.d_min),
                    "p_keep_cluster_surrogate": float(p_old), "p_fresh_cluster_surrogate": float(p_fresh),
                    "b_keep_cluster_pp": 100 * float(b_old), "b_fresh_cluster_pp": 100 * float(b_fresh),
                    "signed_residual_optimism_pp": 100 * float(b_old - b_fresh),
                    "probability_false_safe": bool(b_old >= 0 and b_fresh < 0),
                    "probability_false_safe_strict_tol": bool(b_old >= 1e-6 and b_fresh < -1e-6),
                    "sigma_keep_cluster_mm": 1000 * float(evaluated["sigma"][step + 1, old]),
                    "sigma_fresh_cluster_mm": 1000 * float(evaluated["sigma"][step + 1, fresh]),
                    **{f"xi1_{j}": float(xi1[step, j]) for j in range(4)},
                }
                if d_old + 1e-12 < d_fresh:
                    raise ValueError("Fresh geometric selector is not the minimum")
                local_rows.append(record)
            step_rows.extend(local_rows)
            trial_rows.append({
                "source": source, "seed": seed, "trial_outcome": trial["status"], "total_applied_steps": len(inputs),
                "excluded_unsuccessful_or_skipped_steps": int((~eligible).sum()),
                **counts(local_rows),
                "probability_false_safe_phi_gt_dmin_steps": sum(r["probability_false_safe"] and r["endpoint_phi_gt_dmin"] for r in local_rows),
                "probability_false_safe_phi_le_dmin_steps": sum(r["probability_false_safe"] and not r["endpoint_phi_gt_dmin"] for r in local_rows),
                "first_corner_normal_steps": sum(r["next_center_first_corner_roi"] for r in local_rows),
                "first_corner_probability_false_safe_steps": sum(r["next_center_first_corner_roi"] and r["probability_false_safe"] for r in local_rows),
                "first_corner_hidden_dcol_unsafe_steps": sum(r["next_center_first_corner_roi"] and r["hidden_dcol_unsafe"] for r in local_rows),
            })
            sources.extend({"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in (npz_path, meta_path))
            print(f"{source} seed {seed}: {len(local_rows)} normal steps; "
                  f"hidden dcol {trial_rows[-1]['hidden_dcol_unsafe_steps']}; "
                  f"false safe probability {trial_rows[-1]['probability_false_safe_steps']}", flush=True)
    summaries = []
    for source in ("cc_sf", "cc_mf"):
        for region in ("all", "first_corner_roi", "outside_first_corner_roi"):
            for domain in ("all", "phi_gt_dmin", "phi_le_dmin"):
                for termination in ("all", "terminal_guard_entry", "nonterminal"):
                    rows = [r for r in step_rows if r["source"] == source
                            and (region == "all" or r["next_center_first_corner_roi"] == (region == "first_corner_roi"))
                            and (domain == "all" or r["endpoint_phi_gt_dmin"] == (domain == "phi_gt_dmin"))
                            and (termination == "all" or r["terminal_guard_entry"] == (termination == "terminal_guard_entry"))]
                    summaries.append({"source": source, "region": region, "endpoint_domain": domain,
                                      "termination_group": termination, "attempted_trials": 50, **counts(rows)})
    save_csv("normal_step_cluster_audit.csv", step_rows)
    save_csv("all_trial_cluster_counts.csv", trial_rows)
    save_csv("cluster_diagnostic_summary.csv", summaries)
    events = [r for r in step_rows if r["hidden_dcol_unsafe"] or r["hidden_dmin_unsafe"] or r["probability_false_safe"] or r["terminal_guard_entry"]]
    save_csv("cluster_diagnostic_events.csv", events)
    for path in (Path(__file__), HELPER, ROOT / "scripts/analyze_sf_mf_reliability.py", ROOT / "utils/offline_cc_margin.py",
                 SUITE / "source_snapshot/models/augmented_psdf.py", SUITE / "source_snapshot/models/augmented_psdf_wrapper.py",
                 SUITE / "source_snapshot/control/cc_covariance.py"):
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path)})
    manifest = {
        "scope": "All 50 CC-SF and 50 CC-MF DoN .85 trials, including successful and failed outcomes; normal solver-success called actual steps only.",
        "simulation_executed": False, "optimizer_executed": False, "geometry_precision": "archived implementation, CPU float64",
        "retained_identity": "Closest obstacle cluster at actual current pose. Its closest primitive and signed distance/gradient are recomputed at next actual pose. This is not a retained primitive or native nominal solver row.",
        "covariance": "Configured Sigma0 reset each cycle, one-step propagation with actual applied v/omega; shared by all next-pose obstacle scores. No native planning covariance was saved.",
        "geometric_diagnostic": "d_keep(next)>=threshold while min_cluster_d(next)<threshold, threshold d_col=.00005m or d_min=.0001m; no tolerance added. Ignore gap=max(0,d_keep-min_d).",
        "probability_diagnostic": "p_i=Phi((d_min-d_i)/sqrt(q_i Sigma1 q_i)); b_i=epsilon-p_i. False-safe: b_keep>=0 and b_fresh<0, plus strict +/-1e-6 probability tolerance sensitivity column.",
        "signed_optimism": "b_keep-b_fresh; may be negative because geometric nearest need not have greatest standardized Gaussian surrogate probability.",
        "endpoint_domain": "phi_next>d_min versus phi_next<=d_min. Latter is only a formal signed-distance Gaussian diagnostic outside native MF required-row domain, never MF probability. No MF feature sum is evaluated anywhere here.",
        "first_corner_roi": "Inclusive next recorded center x in [0,.20]m and y in [.45,.65]m.",
        "terminal_guard_entry": "Normal successful step whose following raw status starts the final all-stage0-guard suffix ending in trial failure; risk evaluated at its actual post-integration pose.",
        "trial_count": len(trial_rows), "normal_step_count": len(step_rows), "event_rows": len(events),
        "sources": sources,
    }
    (OUTPUT / "retained_cluster_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps([r for r in summaries if r["region"] == "all" and r["termination_group"] == "all"], indent=2))


if __name__ == "__main__":
    main()
