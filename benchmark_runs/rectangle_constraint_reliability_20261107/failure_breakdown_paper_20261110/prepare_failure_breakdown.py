"""Prepare the paper panel from complete benchmark or rerun cohorts."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
BASE = OUTPUT.parent
ORIGINAL = BASE / "rectangle_don_sweep_50_20261105_current_sf"
CLASSIFIED = BASE / "failure_modes_20261110" / "failure_modes_by_don.csv"
RERUN = BASE / "cc_mf_don080_083_random5_20261110_161545" / "trial_results.csv"
WIDTH_IDS = {
    80: "width_120mm",
    81: "width_121p5mm",
    82: "width_123mm",
    83: "width_124p5mm",
    84: "width_126mm",
    85: "width_127p5mm",
    86: "width_129mm",
    87: "width_130p5mm",
}
FAILURE_FIELDS = (
    "progress_stagnation",
    "feature_persistent",
    "feature_switching",
    "other_failures",
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def group_don(row):
    return round(100 * float(row["don"]))


def build_rows():
    classified = {
        (group_don(row), row["controller"]): row
        for row in read_csv(CLASSIFIED)
    }
    rerun = defaultdict(list)
    for row in read_csv(RERUN):
        assert row["optimizer_type"] == "cc_mf"
        rerun[group_don(row)].append(row)
    assert set(rerun) == set(range(80, 84))
    assert len({row["start_seed"] for rows in rerun.values() for row in rows}) == 20

    rows = []
    for don, width_id in WIDTH_IDS.items():
        archived = read_csv(ORIGINAL / "widths" / width_id / "trial_results.csv")
        for controller in ("cc_sf", "cc_mf"):
            if controller == "cc_mf" and don in rerun:
                trials = rerun[don]
                assert len(trials) == 5
                assert all(row["status"] == "success" and row["goal_reached"] == "True"
                           for row in trials)
                failures = dict.fromkeys(FAILURE_FIELDS, 0)
                cohort = "random_seed_rerun"
            else:
                trials = [row for row in archived if row["optimizer_type"] == controller]
                assert len(trials) == 50
                assert {int(row["start_seed"]) for row in trials} == set(range(50))
                summary = classified[don, controller]
                failures = {field: int(summary[field]) for field in FAILURE_FIELDS}
                cohort = "original_sweep"

            outcomes = Counter(trial["status"] for trial in trials)
            assert set(outcomes) <= {"success", "failure", "timeout", "error"}
            assert len(trials) == outcomes["success"] + sum(failures.values())
            rows.append(dict(
                don=f"{don / 100:.2f}",
                controller=controller,
                completed_trials=len(trials),
                successful_trials=outcomes["success"],
                **failures,
                cohort=cohort,
            ))
    return rows


def main():
    rows = build_rows()
    with (OUTPUT / "failure_breakdown.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
