"""Check completed trial records, safe-stop events, and existing MP4 metadata."""

from collections import Counter
import csv
import json
from pathlib import Path
import subprocess

OUTPUT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


rows = read_csv(OUTPUT / "safe_stop_summary.csv")
trials = read_csv(OUTPUT / "results/trial_results.csv")
events = read_csv(OUTPUT / "safe_stop_events.csv")
event_counts = Counter(int(event["trial_number"]) for event in events)
assert len(rows) == len(trials) == 10, (len(rows), len(trials))
assert [int(row["start_seed"]) for row in rows] == list(range(10))
source_changes = json.loads((OUTPUT / "source_changed_during_batch.json").read_text())
# The source benchmark YAML may be edited separately after our saved config is loaded.
assert all(path == "config/config_benchmark.yaml" for path in source_changes), source_changes
manifest = __import__("yaml").safe_load((OUTPUT / "results/run_manifest.yaml").read_text())
saved_config = __import__("yaml").safe_load((OUTPUT / "experiment_config.yaml").read_text())
assert manifest["config"] == saved_config
videos = []
for row, trial in zip(rows, trials):
    number = int(row["trial_number"])
    assert row["status"] == trial["status"]
    assert row["success"] == trial["success"]
    assert int(row["safe_stop_count"]) == int(row["safe_stop_event_rows"]) == event_counts[number]
    history_path, = (Path(row["output_root_dir"]) / "data").glob("trial_history_*.csv")
    history = read_csv(history_path)
    for event in (item for item in events if int(item["trial_number"]) == number):
        step = history[int(event["timestep"])]
        assert abs(float(step["v"])) < 1e-12 and abs(float(step["omega"])) < 1e-12
        assert step["solver_success"] == "False"
    video = Path(row["animation_path"])
    assert video.is_file() and video.stat().st_size > 0
    metadata = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration",
        "-of", "json", str(video),
    ], text=True))
    stream, = metadata["streams"]
    assert stream["codec_name"] == "h264" and int(stream["nb_frames"]) > 0
    videos.append({"trial_number": number, "path": str(video),
                   "size_bytes": video.stat().st_size, **metadata})

successes = sum(int(row["success"]) for row in rows)
safe_stops = sum(int(row["safe_stop_count"]) for row in rows)
summary = {
    "controller": "cc_sf", "environment": "maze", "footprint": "rectangle",
    "rectangle_length_m": 0.15, "rectangle_width_m": 0.12,
    "dynamics": "differential_drive", "planner": "astar",
    "baseline_config": "config/config_benchmark.yaml",
    "simulation_limit_s": 60, "seeds": list(range(10)),
    "completed_trials": len(rows), "successful_trials": successes,
    "success_rate": successes / len(rows),
    "status_counts": dict(Counter(row["status"] for row in rows)),
    "safe_stop_count": safe_stops,
    "trials_with_safe_stop": sum(int(row["safe_stop_count"]) > 0 for row in rows),
    "safe_stop_definition": "Zero-input fallback control steps; consecutive steps each count once.",
    "safe_stop_events_verified_against_zero_control_history": True,
    "animation_count": len(videos), "animation_metadata_verified": True,
    "source_changes_during_batch": source_changes,
    "control_code_unchanged_during_batch": True,
    "saved_config_matches_run_manifest": True,
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(OUTPUT / "animation_manifest.json").write_text(json.dumps(videos, indent=2) + "\n")
report = [
    "CC-SF / rectangle (0.15 x 0.12 m) / differential drive / maze / A*",
    "Baseline: config/config_benchmark.yaml; current controller parameters unchanged.",
    "10 sequential trials, perturbed initial poses with seeds 0-9; limit 60 s/trial.",
    "Failure checker: movement <= 0.03 m over 20 s, or 20 consecutive solver failures.",
    f"Success rate: {successes}/10 = {successes * 10}%",
    f"Safe stop: {safe_stops} control steps; {summary['trials_with_safe_stop']}/10 trials.",
    "Safe stop counts each zero-input fallback control step, including consecutive steps.",
    "Safe-stop events verified against applied v=omega=0 and unsuccessful solver status.",
    "10 animations saved using existing animate_world (H.264 ultrafast); no additional encoding.",
    "Original benchmark YAML changed during the batch; the run used the saved experiment config throughout.",
    "", "trial,seed,status,final_time_s,safe_stop_count,failure_reason",
]
report.extend(
    f"{row['trial_number']},{row['start_seed']},{row['status']},{float(row['final_time']):.1f},"
    f"{row['safe_stop_count']},{row['failure_reason']}" for row in rows)
(OUTPUT / "results.txt").write_text("\n".join(report) + "\n")
print(json.dumps(summary, indent=2))
print("\n".join(report[-11:]))
