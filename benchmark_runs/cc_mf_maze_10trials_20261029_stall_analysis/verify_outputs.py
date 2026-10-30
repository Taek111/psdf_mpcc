"""Verify outcome/history coverage and read standard animation metadata."""

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
rows = list(csv.DictReader((BASE / "results/trial_results.csv").open()))
log = (BASE / "run.log").read_text()
video_paths = [Path(value) for value in re.findall(r"^Animation saved: (.+)$", log, re.M)]
saving = re.findall(r"^Saving animation: (.+) \((\d+) frames, 10 fps, 100 DPI\)$", log, re.M)
expected_frames = {value: int(count) for value, count in saving}
assert len(rows) == 10, f"Only {len(rows)} outcomes"
assert len(video_paths) == 10, f"Only {len(video_paths)} animations"
assert [int(row["start_seed"]) for row in rows] == list(range(10))
manifest = []
stall_checks = []
for row in rows:
    trial = int(row["trial_number"])
    matches = [path for path in video_paths if f"trial{trial:04d}" in path.name]
    assert len(matches) == 1, matches
    path = matches[0]
    assert path.parent == ROOT / "animations/cc_mf"
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    stream, = json.loads(result.stdout)["streams"]
    assert stream["codec_name"] == "h264"
    assert stream["r_frame_rate"] == "10/1"
    assert int(stream["nb_frames"]) == expected_frames[str(path)]
    data_dir = Path(row["output_root_dir"]) / "data"
    history, = data_dir.glob("trial_history_*.csv")
    pose, = data_dir.glob("pose_sdf_*.csv")
    history_rows = list(csv.DictReader(history.open()))
    pose_rows = list(csv.DictReader(pose.open()))
    telemetry_count = sum(1 for _ in (BASE / f"telemetry_trial{trial:02d}.jsonl").open())
    assert len(history_rows) == len(pose_rows) == telemetry_count
    assert abs(float(history_rows[-1]["time"]) - float(row["final_time"])) < 1e-6
    if "stuck_for_" in row["failure_reason"]:
        import math
        begin, end = history_rows[-201], history_rows[-1]
        displacement = math.hypot(float(end["x"]) - float(begin["x"]),
                                  float(end["y"]) - float(begin["y"]))
        assert displacement <= 0.03
        assert f"disp_{displacement:.4f}m" in row["failure_reason"]
        assert all(item["solver_success"] == "True" for item in history_rows[-40:])
        stall_checks.append({"trial": trial, "seed": int(row["start_seed"]),
                             "last_20s_displacement_m": displacement,
                             "last_4s_solver_successes": 40})
    manifest.append({
        "trial_number": trial, "seed": int(row["start_seed"]),
        "status": row["status"], "failure_reason": row["failure_reason"],
        "simulation_duration_s": float(row["final_time"]),
        "control_steps": telemetry_count, "animation_frames": int(stream["nb_frames"]),
        "video_duration_s": float(stream["duration"]),
        "codec": stream["codec_name"], "fps": stream["r_frame_rate"],
        "width": stream["width"], "height": stream["height"],
        "animation_path": str(path), "size_bytes": path.stat().st_size,
    })
hashes = json.loads((BASE / "source_hashes.json").read_text())
changed = [key for key, digest in hashes.items()
           if not (ROOT / key).exists()
           or hashlib.sha256((ROOT / key).read_bytes()).hexdigest() != digest]
assert not changed, changed
with (BASE / "animations.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
    writer.writeheader()
    writer.writerows(manifest)
(BASE / "verification.json").write_text(json.dumps({
    "recorded_trials": len(rows), "animations_verified": len(manifest),
    "seeds": [int(row["start_seed"]) for row in rows],
    "source_changes": changed, "history_pose_telemetry_counts_equal": True,
    "final_history_times_match_outcomes": True,
    "video_metadata_matches_standard_writer": True,
    "stalled_trial_checks": stall_checks,
    "frame_skip": 5, "fps": 10,
    "verification": "ffprobe metadata read only; videos not re-encoded",
}, indent=2) + "\n")
print(json.dumps({"outcomes": len(rows), "animations": len(manifest),
                  "source_changes": changed, "statuses": [r["status"] for r in rows]}))
