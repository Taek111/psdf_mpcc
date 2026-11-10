"""Prepare matched-seed plotting data; no geometry or solver is rerun."""

import hashlib
import json
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent
AUDIT = OUT.parent
MECHANISM = AUDIT.parent / "high_don_mechanism_20261109"


def main():
    event_path = AUDIT / "terminal_distance_reversal.csv"
    trial_path = MECHANISM / "trial_terminal_audit.csv"
    events = pd.read_csv(event_path)
    trials = pd.read_csv(trial_path)
    trials = trials[trials.don.eq(0.85)].copy()
    rows = []
    for seed in range(50):
        row = {"seed": seed, "grid_row_0based": seed // 10,
               "grid_col_0based": seed % 10}
        for source, prefix in (("cc_sf", "sf"), ("cc_mf", "mf")):
            trial = trials[trials.source.eq(source) & trials.seed.eq(seed)]
            assert len(trial) == 1
            trial = trial.iloc[0]
            event = events[events.source.eq(source) & events.seed.eq(seed)]
            assert len(event) <= 1
            row[f"{prefix}_specified_event"] = (
                bool(event.iloc[0].tradeoff_guard_entry) if len(event) else False)
            row[f"{prefix}_outcome"] = trial.status
            row[f"{prefix}_any_terminal_guard"] = bool(len(event))
            row[f"{prefix}_first_guard_time_s"] = trial.first_guard_current_pose_time_s
            for field in ("old_distance_change_mm", "d_old_cluster_next_mm",
                          "d_fresh_cluster_next_mm", "geometric_ignore_gap_mm"):
                row[f"{prefix}_{field}"] = event.iloc[0][field] if len(event) else None
        row["paired_event_class"] = (
            "both" if row["sf_specified_event"] and row["mf_specified_event"] else
            "sf_only" if row["sf_specified_event"] else
            "mf_only" if row["mf_specified_event"] else "neither")
        rows.append(row)
    matrix = pd.DataFrame(rows)
    matrix.to_csv(OUT / "paired_event_seed_matrix.csv", index=False)

    sf = set(matrix.loc[matrix.sf_specified_event, "seed"])
    mf = set(matrix.loc[matrix.mf_specified_event, "seed"])
    candidate_trials = trials[
        trials.source.eq("cc_sf") & trials.paired_group.eq("mf_only_success")]
    median = float(candidate_trials.final_time_s.median())
    selected = candidate_trials.assign(
        median_time_distance=(candidate_trials.final_time_s - median).abs().round(10)
    ).sort_values(["median_time_distance", "seed"]).iloc[0]
    sf_events = events[events.source.eq("cc_sf") & events.tradeoff_guard_entry]
    seed10 = sf_events[sf_events.seed.eq(10)].iloc[0]
    tie = candidate_trials[
        (candidate_trials.final_time_s - median).abs().round(10).eq(
            float(selected.median_time_distance))]
    review = {
        "don": 0.85,
        "denominator_per_method": 50,
        "paired_event_counts": matrix.paired_event_class.value_counts().to_dict(),
        "event_intersection_seeds": sorted(sf & mf),
        "sf_event_seeds": sorted(sf), "mf_event_seeds": sorted(mf),
        "sf_non_event_outcomes": matrix.loc[~matrix.sf_specified_event,
                                              "sf_outcome"].value_counts().to_dict(),
        "mf_non_event_outcomes": matrix.loc[~matrix.mf_specified_event,
                                              "mf_outcome"].value_counts().to_dict(),
        "seed10_selection_independent_of_clearance_gap": {
            "cohort": "MF-only successful pairs at DoN .85",
            "cohort_size": len(candidate_trials),
            "criterion": "Minimum absolute SF failure-time difference from cohort median; smallest seed resolves ties",
            "median_sf_failure_time_s": median,
            "selected_seed": int(selected.seed),
            "tied_seed_candidates": tie.seed.astype(int).tolist(),
            "selected_sf_failure_time_s": float(selected.final_time_s),
            "selected_sf_first_guard_time_s": float(selected.first_guard_current_pose_time_s),
            "selected_old_distance_gain_mm": float(seed10.old_distance_change_mm),
            "old_distance_gain_ascending_rank_among_sf_events": int(
                (sf_events.old_distance_change_mm < seed10.old_distance_change_mm).sum() + 1),
            "selected_geometric_omission_gap_mm": float(seed10.geometric_ignore_gap_mm),
            "omission_gap_ascending_rank_among_sf_events": int(
                (sf_events.geometric_ignore_gap_mm < seed10.geometric_ignore_gap_mm).sum() + 1),
            "sf_event_count": len(sf_events)
        },
        "visual_encoding": {
            "layout": "Two aligned 5-row by 10-column grids in natural seed order; row=seed//10, col=seed%10",
            "filled_marker": "The precisely specified geometric event occurred at least once in the trial",
            "empty_marker": "The specified event was not observed; this does not mean the trial reached the goal",
            "event_definition": "At actual pose k choose nearest obstacle j. Over its recorded solver-success step, d_j increases and d_j(next)>=0.05 mm, while min_i d_i(next)<0.05 mm",
            "avoid_label": "No chance probability, primitive feature, planner infeasibility, or causal claim is encoded"
        },
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (event_path, trial_path)},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }
    (OUT / "paired_event_design_review.json").write_text(
        json.dumps(review, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
