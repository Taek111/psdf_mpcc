#!/usr/bin/env python3
"""Render a labeled partial animation from a native-crash checkpoint or replay."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent


def recover(output, run, checkpoint, source):
    sys.path.insert(0, str(output / "source_snapshot"))
    from models.dd import DifferentialDriveMultipleGeometry, DifferentialDriveRectangleGeometry
    from sim.simulation_mpc import simulation_mpc
    import yaml

    config = yaml.safe_load((run / "config.yaml").read_text())
    records = [json.loads(line) for line in checkpoint.read_text().splitlines() if line.strip()]
    initial = records[0]
    steps = records[1:]
    poses = [np.asarray(row["pose"], dtype=float) for row in steps]
    if not poses:
        poses = [np.asarray(initial["initial_pose"], dtype=float)]
    geometry = DifferentialDriveMultipleGeometry()
    geometry.add_geometry(DifferentialDriveRectangleGeometry(
        length=config["rectangle_geometry"]["length"],
        width=config["rectangle_geometry"]["width"], rear_dist=0.0))
    robot = SimpleNamespace(
        _system=SimpleNamespace(_geometry=geometry, get_state=lambda: poses[-1]),
        _system_logger=SimpleNamespace(_xs=poses),
        _local_planner_logger=SimpleNamespace(_trajs=[np.asarray(row["local_path"])
            for row in steps if row.get("local_path") is not None]),
        _controller_logger=SimpleNamespace(_xtrajs=[np.asarray(row["optimized_traj"])
            for row in steps if row.get("optimized_traj") is not None]),
        _global_planner_logger=SimpleNamespace(_paths=[np.asarray(initial["global_path"])]))
    renderer = simulation_mpc()
    _, _, _, obstacles = renderer.create_env("maze")
    simulation = SimpleNamespace(_robot=robot, _obstacles=obstacles)
    renderer.initial_pose = np.asarray(initial["initial_pose"])
    method = run.parent.name
    directory = ROOT / "animations" / method
    directory.mkdir(exist_ok=True, parents=True)
    name = f"MPC_{method}_rectangle_maze_{config['output_suffix']}_partial_{source}"
    path = directory / f"{name}.mp4"
    renderer._get_animation_output_path = lambda *_: str(path)
    renderer.animate_world(simulation, animation_name=name, maze_type="maze",
                           method_name=method, frame_skip=5, return_html=False)
    metadata = {"animation_path": str(path), "partial": True, "source": source,
        "checkpoint": str(checkpoint), "saved_poses": len(poses),
        "last_recorded_time_s": steps[-1]["time_s"] if steps else 0.0,
        "note": "Original trial remains an error with missing aggregate telemetry. "
                "A replay animation is a separate run with identical settings/seed; "
                "it is not evidence of the original trial's unrecorded trajectory. "
                "No detected-obstacle overlay was checkpointed."}
    (run / "partial_animation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source", choices=("checkpoint", "replay"), required=True)
    args = parser.parse_args()
    recover(args.results_dir.resolve(), args.run_dir.resolve(),
            args.checkpoint.resolve(), args.source)
