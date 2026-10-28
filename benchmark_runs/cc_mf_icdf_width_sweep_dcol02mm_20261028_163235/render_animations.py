"""Render recorded trial poses with the existing animate_world implementation."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = ROOT / "source_snapshot"
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(SOURCE))

import numpy as np
from matplotlib import animation
from models.dd import (
    DifferentialDriveDynamics, DifferentialDriveMultipleGeometry,
    DifferentialDriveRectangleGeometry, DifferentialDriveStates,
    DifferentialDriveSystem,
)
from planning.path_generator.search_path_generator import AstarLoSPathGenerator
from sim.simulation import Robot, SingleAgentSimulation
from sim.simulation_mpc import simulation_mpc


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


# The existing renderer fixes save DPI to 300. Override only that export
# argument in this process; animation geometry, drawing and timing are native.
original_save = animation.Animation.save


def save_standard_resolution(self, *args, **kwargs):
    kwargs["dpi"] = 100
    return original_save(self, *args, **kwargs)


animation.Animation.save = save_standard_resolution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    progress = json.loads((ROOT / "sweep_progress.json").read_text())
    if not args.partial:
        assert all(row["status"] in ("completed", "skipped") for row in progress)
    manifest_path = ROOT / "animations.json"
    records = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    known = {(row["width_m"], row["seed"]) for row in records}
    for record in progress:
        if record["status"] != "completed":
            continue
        width = record["width_m"]
        folder = ROOT / f"width_{round(width * 1000):03d}mm"
        geometry_data = json.loads((folder / "geometry.json").read_text())
        for trial in rows(folder / "trial_results.csv"):
            seed = int(trial["start_seed"])
            if (width, seed) in known:
                continue
            data = folder / "runs" / trial["configuration_id"] / \
                f"trial_{int(trial['trial_number']):04d}" / "data"
            history_path = data / f"trial_history_{trial['current_name']}.csv"
            history = rows(history_path)
            poses = np.array([[float(row[key]) for key in ("x", "y", "theta")]
                              for row in history])
            pose0 = np.array([float(trial[key]) for key in
                              ("initial_x", "initial_y", "initial_theta")])
            animator = simulation_mpc()
            animator.output_root_dir = str(PROJECT)
            animator.initial_pose = pose0
            _, goal, grid, obstacles = animator.create_env("maze")
            np.testing.assert_allclose([obs.get_ccw_vertices() for obs in obstacles],
                                       geometry_data["obstacles"], atol=0, rtol=0)
            geometry = DifferentialDriveMultipleGeometry()
            geometry.add_geometry(DifferentialDriveRectangleGeometry(.15, width, 0.))
            np.testing.assert_allclose(geometry.equiv_rep()[0].get_ccw_vertices(),
                                       geometry_data["footprint"], atol=0, rtol=0)
            system = DifferentialDriveSystem(DifferentialDriveStates(pose0), geometry,
                                             DifferentialDriveDynamics())
            robot = Robot(system)
            robot._system_logger._xs = list(poses)
            # Recreate the same deterministic A* context. No controller or
            # solver runs; future local and optimized trajectories are absent
            # from the saved CSV and are deliberately not reconstructed.
            robot.set_global_planner(AstarLoSPathGenerator(grid, quad=False, margin=.03))
            robot.run_global_planner(system, obstacles, goal)
            robot._local_planner_logger = SimpleNamespace(_trajs=[])
            robot._controller_logger = SimpleNamespace(_xtrajs=[])
            simulation = SingleAgentSimulation(robot, obstacles, goal)
            name = (f"MPC_cc_mf_icdf_rectangle_maze_width{round(width * 1000):03d}mm_"
                    f"seed{seed}_dcol02mm_20261028_163235")
            target = Path(animator._get_animation_output_path(name, "cc_mf_icdf"))
            assert not target.exists(), target
            print(f"Rendering width={width:.3f} seed={seed} status={trial['status']}", flush=True)
            path = animator.animate_world(simulation, animation_name=name, maze_type="maze",
                                          frame_skip=1, method_name="cc_mf_icdf",
                                          return_html=False, include_initial_pose=True)
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                    "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration",
                                    "-of", "json", path], capture_output=True, text=True, check=True)
            metadata = json.loads(probe.stdout)["streams"][0]
            assert int(metadata["nb_frames"]) == len(poses) + 1
            assert metadata["r_frame_rate"] == "10/1"
            assert int(metadata["width"]) <= 830 and int(metadata["height"]) <= 500
            records.append({"width_m": width, "seed": seed, "trial_status": trial["status"],
                "path": str(Path(path).resolve()), "size_bytes": Path(path).stat().st_size,
                "source_history": str(history_path),
                "source_history_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
                "initial_and_final_poses_included": True, "dpi": 100, **metadata,
                "renderer": "frozen sim.simulation_mpc.simulation_mpc.animate_world",
                "local_and_predicted_trajectories": "not recorded; omitted"})
            manifest_path.write_text(json.dumps(records, indent=2) + "\n")
            print("ANIMATION_SAVED " + str(path), flush=True)
    print(f"Animations available: {len(records)}", flush=True)


if __name__ == "__main__":
    main()
