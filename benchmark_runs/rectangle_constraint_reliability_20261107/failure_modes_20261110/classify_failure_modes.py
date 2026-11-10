"""Classify stored CC-SF/CC-MF maze failures at the first stage-0 guard pose."""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SUITE = OUT.parent / "rectangle_don_sweep_50_20261105_current_sf"
COMPARISON = (ROOT / "benchmark_runs" / "rectangle_analysis_20261105_current_sf"
              / "rectangle_don_100_20261105_current_sf")
CONTROLLERS = ("cc_sf", "cc_mf")
CATEGORIES = ("progress_stagnation", "feature_persistent", "feature_switching",
              "other_failures")

sys.path.insert(0, str(ROOT))
from scripts.analyze_rectangle_clearance import load_clearance_function
from scripts.analyze_sf_mf_reliability import make_wrapper, read_csv, sha, write_csv
from utils.offline_cc_margin import _snapshot_imports
from utils.offline_feature_selectors import extract_feature_selectors


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_suite(suite):
    manifest = read_json(suite / "suite_manifest.json")
    size = manifest["trials_per_configuration"]
    cohorts = {}
    for width in manifest["widths"]:
        don = round(width["nominal_don"], 2)
        rows = read_csv(suite / "widths" / width["width_id"] / "trial_results.csv")
        for method in CONTROLLERS:
            cohort = [row for row in rows if row["optimizer_type"] == method]
            assert len(cohort) == size
            assert {int(row["start_seed"]) for row in cohort} == set(range(size))
            assert all(row["status"] in {"success", "failure", "timeout", "error"}
                       for row in cohort)
            cohorts[(don, method)] = sorted(cohort, key=lambda row: int(row["start_seed"]))
    for don, method in cohorts:
        if method == CONTROLLERS[0]:
            for left, right in zip(cohorts[(don, method)], cohorts[(don, CONTROLLERS[1])]):
                for key in ("initial_x", "initial_y", "initial_theta"):
                    assert float(left[key]) == float(right[key])
    return cohorts


def read_failure(don, method, trial):
    traces = {
        key: SUITE / "widths" / trial[key].replace("\\", "/").rsplit("/widths/", 1)[1]
        for key in ("raw_trace_path", "raw_trace_metadata_path")
    }
    trace = traces["raw_trace_path"]
    metadata_path = traces["raw_trace_metadata_path"]
    meta = read_json(metadata_path)
    assert sha(trace) == meta["raw_trace_sha256"]
    assert meta["scenario"]["optimizer_type"] == method
    assert np.isclose(meta["robot"]["width"], don * .150, rtol=0, atol=1e-12)
    for field in ("status", "failure_reason"):
        assert trial[field] == meta["outcome"][field]
    with np.load(trace, allow_pickle=False) as raw:
        assert all(np.isfinite(raw[key]).all() for key in raw.files)
        poses, inputs, times = raw["poses"], raw["applied_inputs"], raw["pose_times_s"]
    statuses = meta["solver_status_infos"]
    assert len(poses) == len(times) == len(inputs) + 1 == len(statuses) + 1
    assert poses.shape[1] == 3 and inputs.shape[1] >= 2
    assert np.allclose(np.diff(times), meta["dt_s"], rtol=0, atol=1e-10)
    initial = [float(trial[key]) for key in ("initial_x", "initial_y", "initial_theta")]
    assert np.allclose(poses[0], initial, rtol=0, atol=1e-12)
    assert np.isclose(times[-1], float(trial["final_time"]), rtol=0, atol=1e-9)
    record = dict(don=don, controller=method, seed=int(trial["start_seed"]),
                  trial_number=int(trial["trial_number"]), status=trial["status"],
                  failure_reason=trial["failure_reason"], final_time_s=float(times[-1]),
                  category="other_failures", other_reason="Other termination",
                  directed_switching=False, trace=str(trace.relative_to(ROOT)),
                  trace_sha256=meta["raw_trace_sha256"], metadata_sha256=sha(metadata_path))
    guards = [i for i, step in enumerate(statuses) if step["raw_status"] == "safe_stop_stage0"]
    if trial["failure_reason"].startswith("stuck_for_") and not guards:
        record.update(category="progress_stagnation", other_reason="")
        return record, None
    if not trial["failure_reason"].endswith(": safe_stop_stage0"):
        return record, None
    g = guards[0]
    assert g > 0 and guards == list(range(g, len(statuses))) and len(guards) == 20
    assert statuses[g - 1]["success"] and statuses[g - 1]["solver_called"]
    assert all(not step["solver_called"] for step in statuses[g:])
    assert np.all(poses[g:] == poses[g]) and np.count_nonzero(inputs[g:]) == 0
    record.update(other_reason="Guard geometry ambiguous", first_guard_step=g,
                  prior_time_s=float(times[g - 1]), first_guard_time_s=float(times[g]),
                  prior_v_m_s=float(inputs[g - 1, 0]), prior_omega_rad_s=float(inputs[g - 1, 1]),
                  guard_mm=1000 * float(meta["optimizer_parameters"]["d_col"]))
    return record, (poses[g - 1:g + 1], inputs[g - 1, :2], meta)


def check_same_obstacle(record, pair, control, meta, wrapper, wrapper32, step):
    """Compare separated primitive witnesses without conflating SAT and separation."""
    dt = float(meta["dt_s"])
    threshold = record["guard_mm"] / 1000
    start, recorded_end = pair

    def poses_at(fractions):
        return np.asarray([step(start, control, fraction * dt) for fraction in fractions])

    def phi(samples):
        with torch.no_grad():
            return wrapper(torch.as_tensor(samples))[0].numpy().reshape(-1)

    assert np.allclose(poses_at([1])[0], recorded_end, rtol=0, atol=2e-10)
    fractions = np.linspace(0, 1, 129)
    first_unsafe = np.flatnonzero(phi(poses_at(fractions)) < threshold)[0]
    assert first_unsafe > 0
    low, high = fractions[first_unsafe - 1:first_unsafe + 1]
    for _ in range(24):
        mid = (low + high) / 2
        if phi(poses_at([mid]))[0] < threshold:
            high = mid
        else:
            low = mid
    samples = np.vstack((start, poses_at([low])[0]))
    witnesses = extract_feature_selectors(wrapper, samples, verify=False)
    witnesses32 = extract_feature_selectors(wrapper32, samples, verify=False)
    assert np.all(witnesses["separated"]) and np.all(witnesses32["separated"])
    assert witnesses["active_cluster"][0] == record["old_obstacle_id"]
    assert witnesses["active_cluster"][1] == record["new_obstacle_id"]
    start_id, entry_id = witnesses["canonical_id"]
    start32, entry32 = witnesses32["canonical_id"]
    assert (start_id != entry_id) == (start32 != entry32)
    record.update(previous_primitive=start_id, guard_entry_primitive=entry_id,
                  reconstructed_crossing_time_s=record["prior_time_s"] + high * dt)
    return start_id != entry_id


def classify_guards(pending):
    clearance, _ = load_clearance_function(SUITE)
    with _snapshot_imports(SUITE / "source_snapshot") as (module, _, __):
        from models.dd_dynamics import DifferentialDriveDynamics

        step = DifferentialDriveDynamics.forward_dynamics
        for items in pending.values():
            geometry = items[0][3]["geometry"]
            assert all(meta["geometry"] == geometry for _, _, _, meta in items)
            footprints = [np.asarray(polygon) for polygon in geometry["footprint_polygons"]]
            obstacles = [np.asarray(polygon) for polygon in geometry["obstacle_polygons"]]
            bounds = np.asarray(geometry["bounds"])
            wrapper = make_wrapper(module, items[0][3], torch.float64)
            wrapper32 = make_wrapper(module, items[0][3], torch.float32)
            batch = torch.as_tensor(np.concatenate([pair for _, pair, _, _ in items]))
            with torch.no_grad():
                distances = 1000 * wrapper.forward_clusters(batch)[0].numpy()
            assert np.isfinite(distances).all()
            for index, (record, pair, control, meta) in enumerate(items):
                before, after = distances[2 * index:2 * index + 2]
                old, new = int(before.argmin()), int(after.argmin())
                old_before, old_after, new_before, new_after = (
                    before[old], after[old], before[new], after[new]
                )
                threshold = record["guard_mm"]
                tie_gap_before = np.diff(np.sort(before)[:2]).item()
                tie_gap_after = np.diff(np.sort(after)[:2]).item()
                exact_after = 1000 * clearance(pair[1], footprints, obstacles, bounds)
                record.update(old_obstacle_id=old, new_obstacle_id=new,
                              obstacle_changed=old != new,
                              previous_nearest_gap_mm=tie_gap_before,
                              guard_nearest_gap_mm=tie_gap_after,
                              old_before_mm=old_before, old_after_mm=old_after,
                              new_before_mm=new_before, new_after_mm=new_after,
                              old_gain_mm=old_after - old_before,
                              guard_signed_psdf_mm=new_after,
                              exact_footprint_clearance_mm=exact_after,
                              footprint_contact_or_overlap=exact_after == 0,
                              footprint_outside_map=exact_after < 0)
                if old_before < threshold or new_after >= threshold:
                    record["other_reason"] = "Offline guard threshold mismatch"
                elif min(tie_gap_before, tie_gap_after) <= 0.0001:
                    record["other_reason"] = "Nearest obstacles tied within 0.1 micrometre"
                else:
                    switched = old != new or check_same_obstacle(
                        record, pair, control, meta, wrapper, wrapper32, step
                    )
                    record.update(category=("feature_switching" if switched
                                            else "feature_persistent"), other_reason="")
                    record["directed_switching"] = bool(
                        old != new and old_after - old_before > 1e-6 and old_after >= threshold
                    )


def verify_published_overlap(failures):
    previous = read_csv(OUT.parent / "aggregate_guard_tradeoff_20261109"
                        / "guard_transition_details.csv")
    current = {(row["don"], row["controller"], row["seed"]): row
               for row in failures if row["don"] >= .83}
    assert len(previous) == len(current)
    for old in previous:
        row = current[(float(old["don"]), old["source"], int(old["seed"]))]
        assert row["trace_sha256"] == old["raw_trace_sha256"]
        metadata = (ROOT / row["trace"]).with_name("raw_trace_metadata.json").read_bytes()
        newline_variants = (metadata, metadata.replace(b"\r\n", b"\n"),
                            metadata.replace(b"\n", b"\r\n"))
        assert old["metadata_sha256"] in {hashlib.sha256(data).hexdigest()
                                          for data in newline_variants}
        assert row["old_obstacle_id"] == int(float(old["old_cluster"]))
        assert row["new_obstacle_id"] == int(float(old["fresh_cluster"]))
        assert np.isclose(row["guard_signed_psdf_mm"], float(old["d_global_after_mm"]),
                          rtol=0, atol=1e-8)
        assert row["directed_switching"] == (old["specified_tradeoff_event"] == "True")


def main():
    torch.set_num_threads(3)
    sweep = collect_suite(SUITE)
    comparison = collect_suite(COMPARISON)
    sweep_reuse = read_json(SUITE / "suite_manifest.json")["widths"]
    assert next(width for width in sweep_reuse if width["nominal_don"] == .75)["reuse"]["seeds"] == list(range(50))
    comparison_failures = {(don, method, int(row["start_seed"])): row["failure_reason"]
                           for (don, method), cohort in comparison.items()
                           for row in cohort if row["status"] != "success"}
    pending, failures = defaultdict(list), []
    for (don, method), cohort in sorted(sweep.items()):
        for trial in cohort:
            if trial["status"] == "success":
                assert trial["success"] == "1"
                continue
            assert trial["success"] == "0"
            record, geometry = read_failure(don, method, trial)
            if geometry is not None:
                pending[don].append((record, *geometry))
            failures.append(record)
    classify_guards(pending)
    verify_published_overlap(failures)
    assert all(any((row["don"], row["controller"], row["seed"]) == key
                   and row["failure_reason"] == reason for row in failures)
               for key, reason in comparison_failures.items())
    by_cohort = Counter((row["don"], row["controller"], row["category"]) for row in failures)
    directed = Counter((row["don"], row["controller"]) for row in failures
                       if row["directed_switching"])
    groups = []
    for (don, method), cohort in sorted(sweep.items()):
        success = sum(row["status"] == "success" for row in cohort)
        counts = {category: by_cohort[don, method, category] for category in CATEGORIES}
        assert sum(counts.values()) + success == len(cohort)
        groups.append(dict(don=don, controller=method, completed_trials=len(cohort),
                           successful_trials=success, **counts,
                           directed_switching=directed[don, method]))
    write_csv(OUT / "failure_trial_classification.csv", failures)
    write_csv(OUT / "failure_modes_by_don.csv", groups)
    print(len(failures), Counter((row["controller"], row["category"]) for row in failures))
    print("Guard/contact:", Counter((row["controller"], row["footprint_contact_or_overlap"],
                                    row["footprint_outside_map"])
                                   for row in failures if "guard_mm" in row))
    print("A100 failures also present in sweep:", comparison_failures.keys())
    print("Other:", [(row["don"], row["controller"], row["seed"], row["other_reason"])
                     for row in failures if row["category"] == "other_failures"])


if __name__ == "__main__":
    main()
