"""Independent scalar aggregation and MP4 inspection for the completed sweep."""

import csv
import json
import math
from pathlib import Path
import re
import subprocess


OUTPUT = Path(__file__).resolve().parent


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    return math.fsum(values) / len(values) if values else None


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


trials = read(OUTPUT / "trial_results.csv")
summaries = read(OUTPUT / "benchmark_summary.csv")
assert len(trials) == 280 and len(summaries) == 28
assert len({(row["width_m"], row["method"], row["seed"]) for row in trials}) == 280
checks = []
video_checks = []
for summary in summaries:
    rows = [row for row in trials if row["width_m"] == summary["width_m"]
            and row["method"] == summary["method"]]
    assert len(rows) == 10 and {int(row["seed"]) for row in rows} == set(range(10))
    clearances, times, traversal = [], [], []
    for row in rows:
        run = OUTPUT / row["run_dir"]
        if not (run / "outcome.json").exists():
            partial = json.loads((run / "partial_animation.json").read_text())
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=codec_name,width,height,nb_frames,r_frame_rate", "-of", "json",
                partial["animation_path"]], text=True))["streams"][0]
            poses = partial["saved_poses"]
            frames = (poses - 1) // 5 + 1 + int((poses - 1) % 5 != 0)
            assert probe["codec_name"] == "h264" and probe["r_frame_rate"] == "10/1"
            assert int(probe["nb_frames"]) == frames
            video_checks.append({"run": row["run_dir"], "frames": frames,
                                 "partial": True, "source": partial["source"]})
            continue
        outcome = json.loads((run / "outcome.json").read_text())
        samples, cycles = read(run / "clearance.csv"), read(run / "cycles.csv")
        clearances.extend(float(item["clearance_m"]) * 1000 for item in samples)
        times.extend(float(item["cycle_time_s"]) * 1000 for item in cycles if int(item["step"]) > 0)
        if row["success"] == "1":
            traversal.append(float(row["final_time_s"]))
        path = Path(outcome["animation_path"])
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=codec_name,width,height,nb_frames,r_frame_rate,duration", "-of", "json", str(path)
        ], text=True))["streams"][0]
        text = (run / "run.log").read_text()
        frames = int(re.search(r"Saving animation: .* \((\d+) frames", text)[1])
        assert probe["codec_name"] == "h264" and probe["r_frame_rate"] == "10/1"
        assert int(probe["nb_frames"]) == frames and frames > 0
        video_checks.append({"run": row["run_dir"], "frames": frames,
                             "width": probe["width"], "height": probe["height"]})
    expected = {"success_rate": len(traversal) / 10,
        "mean_clearance_mm": mean(clearances), "p05_clearance_mm": percentile(clearances, 0.05),
        "mean_computation_time_ms": mean(times), "p95_computation_time_ms": percentile(times, 0.95),
        "mean_traversal_time_s": mean(traversal),
        "safe_stop_count": sum(int(row["safe_stop_count"] or 0) for row in rows)
            + sum((OUTPUT / row["run_dir"] / "run.log").read_text().count(
                "Applied zero-input safe-stop plan after infeasible CC-SF solve.")
                for row in rows if not (OUTPUT / row["run_dir"] / "outcome.json").exists())}
    for field, value in expected.items():
        actual = summary[field]
        assert (actual == "") if value is None else math.isclose(float(actual), value,
            rel_tol=1e-12, abs_tol=1e-10), (summary["width_m"], summary["method"], field, actual, value)
    checks.append({"width_m": summary["width_m"], "method": summary["method"],
                   "independent_metrics_match": True, "metrics": expected})
(OUTPUT / "report_validation.json").write_text(json.dumps({
    "trials": 280, "configurations": 28, "metrics": checks,
    "animations_probed": len(video_checks), "animations": video_checks,
    "method": "stdlib math.fsum and sorted linear quantiles; ffprobe codec/fps/frame counts"
}, indent=2) + "\n")
print("Verified 28 aggregate rows independently and probed all 280 MP4 files.")
