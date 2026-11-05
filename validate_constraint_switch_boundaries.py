"""Independently validate saved finite-offset boundary analysis artifacts."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    args = parser.parse_args()
    base = args.analysis_dir.resolve()
    folder = base / "boundaries"
    report = json.loads((folder / "summary.json").read_text())
    rows = list(csv.DictReader((folder / "all_events.csv").open()))
    metas = report["trial_stages"]
    assert len(metas) == report["completed_trial_stages"] == 240
    keys = [(r["width_mm"], r["method"], r["seed"], r["stage"], r["event_index"], r["offset_um"]) for r in rows]
    assert len(set(keys)) == len(rows)
    manifest = json.loads((base / "cache_sha256.json").read_text())
    checked = 0
    for meta in metas:
        assert meta["analysis_source_sha256"] == report["analysis_source"]["sha256"]
        for source, digest in meta["sources"].items():
            path = Path(source)
            assert sha(path) == digest, source
            if path.suffix == ".npz":
                assert manifest[path.name] == digest
                checked += 1
    assert sha(Path(report["analysis_source"]["path"])) == report["analysis_source"]["sha256"]
    by_trial = collections.defaultdict(list)
    for row in rows:
        by_trial[(int(row["width_mm"]), row["method"], int(row["seed"]), int(row["stage"]), float(row["offset_um"]))].append(row)
    for meta in metas:
        for summary in meta["offset_summaries"]:
            selected = by_trial[(meta["width_mm"], meta["method"], meta["seed"], meta["stage"], summary["offset_um"])]
            if summary["subset"] == "ordinary_inverse_cdf":
                selected = [r for r in selected if r["mf_branch_left"] == r["mf_branch_right"] == "1"]
            elif summary["subset"] == "within_initial_grid_bracket":
                selected = [r for r in selected if r["within_initial_grid_bracket"] == "True"]
            accepted = [r for r in selected if r["accepted"] == "True"]
            assert len(selected) == summary["candidate_events"]
            assert len(accepted) == summary["accepted_events"]
            for metric, result in summary["metrics"].items():
                values = [float(r[metric]) for r in accepted]
                assert len(values) == result["n"]
                if values:
                    assert math.isclose(statistics.mean(map(abs, values)), result["mean_abs"], rel_tol=1e-12, abs_tol=1e-15)
    for group in report["trial_mean_sd"]:
        summaries = [s for m in metas for s in m["offset_summaries"]
                     if all(s[k] == group[k] for k in ("width_mm", "method", "stage", "offset_um", "subset"))]
        assert len(summaries) == group["trials"]
        for metric in summaries[0]["metrics"]:
            for statistic in ("mean_abs", "p95_abs", "rms", "mean_signed"):
                values = [s["metrics"][metric][statistic] for s in summaries if s["metrics"][metric][statistic] is not None]
                name = f"{metric}_{statistic}_trial_"
                assert len(values) == group[name + "n"]
                if values:
                    assert math.isclose(statistics.mean(values), group[name + "mean"], rel_tol=1e-12, abs_tol=1e-15)
                if len(values) > 1:
                    assert math.isclose(statistics.stdev(values), group[name + "sd"], rel_tol=1e-12, abs_tol=1e-15)
    accepted = [r for r in rows if r["accepted"] == "True"]
    for row in accepted:
        assert all(row[k] == "True" for k in ("mf_required_left", "mf_required_right", "mf_valid_left", "mf_valid_right", "separated_left", "separated_right"))
        assert row["selector_left"] != row["selector_right"]
        assert 0 <= float(row["alpha_left"]) < float(row["alpha_right"]) <= 1
        assert float(row["bisected_body_bracket_width_m"]) <= .25e-6
        assert json.loads(row["state_left_json"])[4:] == json.loads(row["state_right_json"])[4:]
        assert abs(float(row["finite_offset_body_motion_left_m"]) - float(row["offset_um"]) * 1e-6) < 1e-12
    subsets = []
    for stage in (1, 10):
        for offset in (5, 10, 50):
            for ordinary in (False, True):
                selected = [r for r in rows if int(r["stage"]) == stage and float(r["offset_um"]) == offset]
                admitted = [r for r in selected if r["accepted"] == "True" and (not ordinary or r["mf_branch_left"] == r["mf_branch_right"] == "1")]
                subsets.append({"stage": stage, "offset_um": offset, "subset": "ordinary_inverse_cdf" if ordinary else "all_valid",
                    "all_candidates": len(selected), "accepted": len(admitted),
                    "all_exclusion_counts": dict(collections.Counter(r["exclusion_reason"] for r in selected if r["accepted"] != "True")),
                    "branch_pairs": dict(collections.Counter(r["mf_branch_left"] + ":" + r["mf_branch_right"] for r in admitted)),
                    "mask_changed": sum(int(r["mask_changes"]) > 0 for r in admitted),
                    "internal_changed": sum(int(r["internal_selector_changes"]) > 0 for r in admitted),
                    "MF_quantile_change_smaller": sum(abs(float(r["delta_mf_quantile"])) < abs(float(r["delta_sf_quantile"])) for r in admitted),
                    "MF_probability_change_smaller": sum(abs(float(r["delta_probability_sum"])) < abs(float(r["delta_p_selected"])) for r in admitted)})
    result = {"completed_trial_stages": 240, "unique_event_offset_rows": len(rows), "cache_hashes_verified": checked,
        "all_source_hashes_verified": True, "accepted_domain_selector_and_frozen_covariance_checks_passed": True,
        "per_trial_metric_count_and_mean_verified": True, "group_trial_mean_and_sample_sd_verified": True,
        "selection_counts": report["total_selection_counts"], "event_subsets": subsets,
        "source": {"summary_sha256": sha(folder / "summary.json"), "all_events_csv_sha256": sha(folder / "all_events.csv"),
                   "validator_path": str(Path(__file__).resolve()), "validator_sha256": sha(Path(__file__))},
        "limitations": ["Body-local covariance factors are fixed; world covariance rotates with heading.",
                        "Changed endpoint canonical identities define candidates; transient switches inside unchanged-endpoint pairs are not enumerated.",
                        "Finite-offset geometry is float32; bisection bounds are interpolation localization bounds, not rigorous exact-arithmetic jump bounds.",
                        "Probability sum R is the implemented multi-feature surrogate, not empirical collision probability."]}
    (folder / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"validated_trial_stages": len(metas), "event_offset_rows": len(rows), "cache_hashes": checked}))


if __name__ == "__main__":
    main()
