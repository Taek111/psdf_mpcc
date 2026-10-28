"""Validate the five recorded trials and export a compact result table."""

import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np


RUN = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    trials = read_csv(RUN / "trial_results.csv")
    summary = read_csv(RUN / "summary.csv")[0]
    times = read_csv(next(RUN.glob("computation_times_*.csv")))
    assert len(trials) == 5
    assert [int(row["start_seed"]) for row in trials] == list(range(5))
    assert all(row["start_perturbation_enabled"] == "True" for row in trials)
    assert all(row["error"] == "" for row in trials)
    table = []
    videos = []
    for row in trials:
        directory = Path(row["output_root_dir"])
        history = read_csv(next((directory / "data").glob("trial_history_*.csv")))
        start = json.loads(next((directory / "data").glob("start_pose_*.json")).read_text())
        assert start["seed"] == int(row["start_seed"])
        assert start["enabled"] is True
        assert start["initial_clearance"] >= 0.01
        assert math.isclose(start["position_std"], 0.02)
        assert math.isclose(math.radians(start["heading_std_deg"]), 0.05)
        navigation_time = float(row["final_time"])
        assert 0 < navigation_time <= 60.1
        assert math.isclose(float(history[-1]["time"]), navigation_time, abs_tol=1e-8)
        trial_times = np.array([
            float(step["solver_computation_time_s"]) for step in times
            if step["trial_number"] == row["trial_number"]
        ])
        assert len(trial_times) == int(row["successful_solver_steps"])
        assert len(trial_times) == sum(step["solver_success"] == "True" for step in history)
        if len(trial_times):
            assert np.isclose(trial_times.mean(), float(row["solver_time_mean_s"]))
            assert np.isclose(np.percentile(trial_times, 95), float(row["solver_time_p95_s"]))
        video = next((directory / "animations" / "cc_mf").glob("*.mp4"))
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration,size",
            "-of", "json", str(video),
        ], text=True))
        frames = list(range(0, len(history), 2))
        if frames[-1] != len(history) - 1:
            frames.append(len(history) - 1)
        assert int(probe["streams"][0]["nb_frames"]) == len(frames)
        assert probe["streams"][0]["codec_name"] == "h264"
        subprocess.run([
            "ffmpeg", "-v", "error", "-xerror", "-threads", "1", "-i", str(video),
            "-map", "0:v:0", "-f", "null", "-",
        ], check=True)
        safe_stops = sum("safe_stop" in step["solver_raw_status"] for step in history)
        table.append({
            "trial": int(row["trial_number"]),
            "seed": int(row["start_seed"]),
            "status": row["status"],
            "failure_reason": row["failure_reason"],
            "navigation_time_s": round(navigation_time, 4),
            "solver_mean_ms": float(row["solver_time_mean_s"]) * 1000,
            "solver_p95_ms": float(row["solver_time_p95_s"]) * 1000,
            "successful_solver_steps": len(trial_times),
            "safe_stop_count": safe_stops,
            "animation": str(video.relative_to(RUN)),
        })
        videos.append({"trial": int(row["trial_number"]), "path": str(video), **probe})

    pooled_times = np.array([float(step["solver_computation_time_s"]) for step in times])
    assert int(summary["completed_trials"]) == 5
    assert int(summary["successful_trials"]) == sum(row["status"] == "success" for row in table)
    assert int(summary["successful_solver_steps"]) == len(pooled_times)
    assert np.isclose(float(summary["solver_time_mean_s"]), pooled_times.mean())
    assert np.isclose(float(summary["solver_time_p95_s"]), np.percentile(pooled_times, 95))
    with (RUN / "results_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    successful_navigation_times = [row["navigation_time_s"] for row in table if row["status"] == "success"]
    result = {
        "verified_trials": 5,
        "verified_videos": len(videos),
        "success_rate": float(summary["success_rate"]),
        "successful_navigation_time_mean_s": float(np.mean(successful_navigation_times)) if successful_navigation_times else None,
        "successful_navigation_time_sample_std_s": float(np.std(successful_navigation_times, ddof=1)) if len(successful_navigation_times) > 1 else None,
        "pooled_solver_mean_ms": float(pooled_times.mean()) * 1000,
        "pooled_solver_p95_ms": float(np.percentile(pooled_times, 95)) * 1000,
        "notes": [
            "Navigation time is simulated time to success or early failure, with a 60 s upper limit.",
            "Solver timings include only successful control steps, including steps from failed trials; first steps are included.",
            "Videos use every second 0.1 s simulation step at 10 fps (approximately 2x speed), plus the final frame.",
            "All five videos passed complete ffmpeg decoding; frame counts match the saved histories.",
        ],
        "trials": table,
        "videos": videos,
    }
    (RUN / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "videos"}, indent=2))


if __name__ == "__main__":
    main()
