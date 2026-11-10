"""Aggregate recorded first-guard transitions across all five high-DoN cohorts.

Only existing actual endpoint poses and archived signed geometry are evaluated.
No optimizer, simulation, nominal rollout, interpolation, or probability model.
"""

import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SUITE = ROOT / "benchmark_runs/rectangle_don_sweep_50_20261105_current_sf"
AUDIT = OUT.parent / "high_don_mechanism_20261109/trial_terminal_audit.csv"
sys.path.insert(0, str(ROOT))
from scripts.analyze_sf_mf_reliability import make_wrapper
from utils.offline_cc_margin import _snapshot_imports


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(count, total):
    z = NormalDist().inv_cdf(0.975)
    p = count / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def main():
    torch.set_num_threads(3)
    audit = pd.read_csv(AUDIT)
    assert len(audit) == 500
    assert sorted(audit.don.unique()) == [.83, .84, .85, .86, .87]
    records, pending, geometry_by_don, initial_poses = [], {}, {}, {}
    for row in audit.itertuples(index=False):
        npz_path = ROOT / row.trace
        meta_path = npz_path.parent / "raw_trace_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        npz_hash, meta_hash = sha(npz_path), sha(meta_path)
        assert npz_hash == row.trace_sha256 == meta["raw_trace_sha256"]
        assert meta_hash == row.metadata_sha256
        raw = np.load(npz_path)
        poses, inputs, times = raw["poses"], raw["applied_inputs"], raw["pose_times_s"]
        statuses = meta["solver_status_infos"]
        assert len(poses) == len(inputs) + 1 == len(times)
        assert len(statuses) == len(inputs)
        assert meta["outcome"]["status"] == row.status
        assert all(np.isfinite(raw[key]).all() for key in raw.files)
        assert row.status in ("success", "failure")
        initial_poses[(row.don, row.source, row.seed)] = poses[0].copy()
        geometry_text = json.dumps(meta["geometry"], sort_keys=True)
        if row.don in geometry_by_don:
            assert geometry_text == geometry_by_don[row.don][0]
        else:
            geometry_by_don[row.don] = (geometry_text, meta)
        guard_indices = [k for k, s in enumerate(statuses)
                         if s["raw_status"] == "safe_stop_stage0"]
        rec = dict(don=row.don, source=row.source, seed=row.seed, outcome=row.status,
                   successful_trial=row.status == "success", guard_failure=bool(guard_indices),
                   specified_tradeoff_event=False, other_guard_failure=False,
                   old_safe_global_unsafe_guard=False,
                   trial_final_time_s=float(times[-1]), normal_step_count=sum(
                       bool(s["success"] and s.get("solver_called") is True) for s in statuses),
                   guard_call_count=len(guard_indices), trace=row.trace,
                   raw_trace_sha256=npz_hash, metadata_sha256=meta_hash)
        if guard_indices:
            g = guard_indices[0]
            assert row.status == "failure" and g > 0 and g == int(row.first_guard_step)
            assert statuses[g - 1]["success"] and statuses[g - 1].get("solver_called") is True
            assert guard_indices == list(range(g, len(statuses))) and len(guard_indices) == 20
            assert np.array_equal(poses[g:], np.broadcast_to(poses[g], poses[g:].shape))
            assert np.count_nonzero(inputs[g:]) == 0
            rec.update(first_guard_step=g, previous_solve_success=True,
                       current_time_s=float(times[g - 1]), next_recorded_time_s=float(times[g]),
                       applied_v_m_s=float(inputs[g - 1, 0]),
                       applied_omega_rad_s=float(inputs[g - 1, 1]),
                       current_x_m=float(poses[g - 1, 0]), current_y_m=float(poses[g - 1, 1]),
                       current_heading_rad=float(poses[g - 1, 2]),
                       next_x_m=float(poses[g, 0]), next_y_m=float(poses[g, 1]),
                       next_heading_rad=float(poses[g, 2]),
                       d_col_mm=1000 * meta["optimizer_parameters"]["d_col"])
            pending.setdefault(row.don, []).append((rec, poses[g - 1:g + 1].copy()))
        else:
            assert row.status == "success" and row.actual_guard_status_count == 0
        records.append(rec)
    for don in sorted(geometry_by_don):
        for seed in range(50):
            assert np.array_equal(initial_poses[(don, "cc_sf", seed)],
                                  initial_poses[(don, "cc_mf", seed)])

    with _snapshot_imports(SUITE / "source_snapshot") as (module, _, __):
        for don, items in sorted(pending.items()):
            wrapper = make_wrapper(module, geometry_by_don[don][1], torch.float64)
            poses = np.concatenate([pose for _, pose in items])
            with torch.no_grad():
                distances, _ = wrapper.forward_clusters(torch.as_tensor(poses, dtype=torch.float64))
            phi = distances.cpu().numpy()
            assert np.isfinite(phi).all()
            for index, (rec, _) in enumerate(items):
                before, after = phi[2 * index:2 * index + 2]
                old, fresh = int(before.argmin()), int(after.argmin())
                old_before, old_after, global_after = 1000 * np.array(
                    [before[old], after[old], after[fresh]])
                gain = old_after - old_before
                guard_margin = global_after - rec["d_col_mm"]
                event = bool(gain > 0 and old_after >= rec["d_col_mm"] and guard_margin < 0)
                assert guard_margin < 0
                rec.update(old_cluster=old, fresh_cluster=fresh,
                           cluster_changed=old != fresh,
                           d_old_cluster_before_mm=float(old_before),
                           d_old_cluster_after_mm=float(old_after),
                           d_new_cluster_before_mm=float(1000 * before[fresh]),
                           d_global_after_mm=float(global_after),
                           old_clearance_change_mm=float(gain),
                           old_before_guard_margin_mm=float(old_before - rec["d_col_mm"]),
                           old_after_guard_margin_mm=float(old_after - rec["d_col_mm"]),
                           global_after_guard_margin_mm=float(guard_margin),
                           geometric_omission_gap_mm=float(old_after - global_after),
                           specified_tradeoff_event=event, other_guard_failure=not event,
                           old_safe_global_unsafe_guard=bool(
                               old_after >= rec["d_col_mm"] and guard_margin < 0),
                           event_with_gain_tolerance_1e_minus6_mm=bool(
                               gain > 1e-6 and old_after >= rec["d_col_mm"] and guard_margin < 0),
                           old_clearance_nonincreasing=bool(gain <= 0),
                           old_cluster_also_below_guard=bool(old_after < rec["d_col_mm"]))
            print(f"DoN {don:.2f}: evaluated {len(items)} first-guard transitions", flush=True)

    table = pd.DataFrame(records).sort_values(["don", "source", "seed"])
    details = table[table.guard_failure].copy()
    assert len(details) == 223
    summaries, pairs = [], []
    for (don, source), group in table.groupby(["don", "source"], sort=True):
        assert len(group) == 50 and set(group.seed) == set(range(50))
        event = int(group.specified_tradeoff_event.sum())
        guard = int(group.guard_failure.sum())
        other = int(group.other_guard_failure.sum())
        success = int(group.successful_trial.sum())
        lo, hi = wilson(event, len(group))
        assert success + event + other == 50
        assert event == int(group.event_with_gain_tolerance_1e_minus6_mm.eq(True).sum())
        summaries.append(dict(don=don, source=source, trial_count=len(group),
                              successful_trials=success, guard_failed_trials=guard,
                              specified_event_trials=event, other_guard_failed_trials=other,
                              old_safe_global_unsafe_guard_trials=int(
                                  group.old_safe_global_unsafe_guard.sum()),
                              specified_event_fraction=event / len(group),
                              specified_event_percent=100 * event / len(group),
                              wilson95_lower_fraction=lo, wilson95_upper_fraction=hi,
                              wilson95_lower_percent=100 * lo, wilson95_upper_percent=100 * hi,
                              specified_event_fraction_of_guard_failures=event / guard,
                              normal_steps=int(group.normal_step_count.sum())))
    for don, group in table.groupby("don", sort=True):
        sf = set(group.loc[group.source.eq("cc_sf") & group.specified_tradeoff_event, "seed"])
        mf = set(group.loc[group.source.eq("cc_mf") & group.specified_tradeoff_event, "seed"])
        pairs.append(dict(don=don, paired_trials=50, both_events=len(sf & mf),
                          sf_event_only=len(sf - mf), mf_event_only=len(mf - sf),
                          neither_event=len(set(range(50)) - sf - mf),
                          sf_event_seeds=" ".join(map(str, sorted(sf))),
                          mf_event_seeds=" ".join(map(str, sorted(mf)))))
    summary = pd.DataFrame(summaries)
    old = pd.read_csv(OUT.parent / "residual_feature_audit_20261109/terminal_distance_reversal.csv")
    current85 = details[details.don.eq(.85)]
    joined = current85.merge(old, on=["source", "seed"], validate="one_to_one")
    assert len(joined) == 23
    assert (joined.specified_tradeoff_event == joined.tradeoff_guard_entry).all()
    for current, previous in (("d_old_cluster_after_mm", "d_old_cluster_next_mm"),
                              ("d_global_after_mm", "d_fresh_cluster_next_mm"),
                              ("old_clearance_change_mm", "old_distance_change_mm")):
        assert np.allclose(joined[current], joined[previous], atol=1e-6, rtol=0)
    check85 = summary[summary.don.eq(.85)].set_index("source")
    assert check85.loc["cc_sf", "specified_event_trials"] == 17
    assert check85.loc["cc_mf", "specified_event_trials"] == 2
    table.to_csv(OUT / "all_trials.csv", index=False)
    details.to_csv(OUT / "guard_transition_details.csv", index=False)
    summary.to_csv(OUT / "cohort_summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUT / "paired_event_counts.csv", index=False)
    paths = [AUDIT, Path(__file__), ROOT / "scripts/analyze_sf_mf_reliability.py",
             ROOT / "utils/offline_cc_margin.py"]
    paths.extend(sorted((SUITE / "source_snapshot/models").rglob("*.py")))
    manifest = dict(
        scope="All 500 saved DoN .83-.87 SF/MF trials, 50 per method/DoN; all 223 first-guard transitions have previous successful solver calls and exactly 20 subsequent stationary zero-input guard calls.",
        metric_name="Trial fraction with increasing previous-obstacle clearance followed by another obstacle's guard entry",
        event_definition="For nearest obstacle j at actual x[k], d_j(x[k+1])-d_j(x[k])>0 AND d_j(x[k+1])>=d_col AND min_i d_i(x[k+1])<d_col; k+1 is the first recorded guard pose.",
        d_col_mm=0.05, geometry="Archived signed PSDF, CPU float64, exact recorded endpoint poses. Retain obstacle identity while freshly recomputing its own closest primitive.",
        confidence_interval="Two-sided Wilson 95% interval for trial-level event fraction; z=NormalDist().inv_cdf(.975). Descriptive across 50 perturbation seeds, not independent time-step inference; SF/MF share seeds.",
        denominator="All 50 trials per method/DoN including successful trials and all failure types; no success conditioning or favorable DoN filtering.",
        not_measured="Not native future-stage feature choice, optimized-stage constraint satisfaction, chance probability, or proof of controller causation. No simulation, solver, interpolation, or local pose probes.",
        checks={"trials": len(table), "guard_transitions": len(details),
                "matched_initial_poses": 250, "excluded_trials": 0,
                "all_guards_absorbing_20_steps": True,
                "gain_tolerance_1e_minus6_mm_changes_event_counts": False,
                "don085_matches_prior_all_normal_step_scan": True},
        source_aggregate_sha256=json.loads((SUITE / "source_hashes.json").read_text())["aggregate_sha256"],
        input_hashes=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in paths],
        raw_hashes="Per-trial raw and metadata SHA256 columns are recorded in all_trials.csv.",
        output_hashes={p.name: sha(p) for p in [OUT / "all_trials.csv", OUT / "guard_transition_details.csv", OUT / "cohort_summary.csv", OUT / "paired_event_counts.csv"]})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
