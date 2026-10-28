"""Run current CC-SF sources with a 0.15 x 0.125 m rectangle and save MP4."""

from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

RUN_DIR = Path(__file__).resolve().parent
PROJECT = RUN_DIR.parents[1]
LENGTH, WIDTH = 0.15, 0.125
os.environ["MPLBACKEND"] = "Agg"
os.environ["ACADOS_SOURCE_DIR"] = "/home/taek111/projects/acados"
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

import numpy as np
import torch
import yaml
from models.dd import DifferentialDriveRectangleGeometry
from test_nmpc import cleanup_simulation

module = importlib.import_module("sim.simulation_mpc")


def dump(path, value):
    def convert(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert) + "\n")


def rectangle_for_run(length, width, rear_dist=0.0):
    assert length == LENGTH and rear_dist == 0.0
    geometry = DifferentialDriveRectangleGeometry(length, WIDTH, rear_dist)
    np.testing.assert_allclose(
        np.ptp(geometry.equiv_rep()[0].get_ccw_vertices(), axis=0),
        [LENGTH, WIDTH], rtol=0, atol=1e-14,
    )
    return geometry


module.DifferentialDriveRectangleGeometry = rectangle_for_run
config = yaml.safe_load((PROJECT / "config/config.yaml").read_text())
config.pop("trials", None)
config.pop("results_dir", None)
config.update(
    output_root_dir="", output_suffix="width125mm_20261028",
    start_perturbation={"enabled": False}, localization_error={"enabled": False},
    cc_sf={"debug_mode": False},
    trial_failure_criteria={"enabled": True, "movement_window_sec": 4.0,
                            "movement_threshold": 0.03,
                            "max_consecutive_solver_failures": 20},
    defaults={"simulation_time": 60.0, "generate_animation": True,
              "generate_plots": False, "frame_skip": 1},
    test_configs=[{"maze_type": "maze", "robot_shape": "rectangle",
                   "optimizer_type": "cc_sf", "dynamics_type": "differential_drive",
                   "path_planner": "astar"}],
)
(RUN_DIR / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
metadata = {
    "started_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
    "length_m": LENGTH, "width_m": WIDTH, "python": sys.executable,
    "source_root": str(PROJECT), "torch_version": torch.__version__,
    "torch_num_threads": torch.get_num_threads(),
    "cuda_available": torch.cuda.is_available(),
    "source_hashes": {
        name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
        for name in ("sim/simulation_mpc.py", "sim/simulation.py", "models/dd.py",
                     "control/cc_sf_optimizer.py", "control/cc_covariance.py",
                     "control/controller.py", "config/config.yaml")
    },
    "geometry_override": "Runner replaces only the simulator rectangle constructor binding.",
}
dump(RUN_DIR / "experiment.json", metadata)
assert torch.cuda.is_available(), "CC-SF requires CUDA."
simulation = module.simulation_mpc()
started = time.perf_counter()
try:
    simulation.mpc_test(**config["test_configs"][0], simulation_time=60.0, config=config)
    footprint = simulation.robot._system._geometry.equiv_rep()[0].get_ccw_vertices()
    np.testing.assert_allclose(np.ptp(footprint, axis=0), [LENGTH, WIDTH], atol=1e-14)
    dump(RUN_DIR / "geometry.json", {"footprint": footprint,
         "length_m": LENGTH, "width_m": WIDTH, "initial_pose": simulation.initial_pose})
    result = dict(simulation.last_run_outcome or {})
    result.update(length_m=LENGTH, width_m=WIDTH, current_name=simulation.current_name,
                  simulation_wall_time_s=time.perf_counter() - started)
    dump(RUN_DIR / "result.json", result)
    print("EXPERIMENT_RESULT " + json.dumps(result), flush=True)
    expected = Path(simulation._get_animation_output_path(simulation.current_name, "cc_sf"))
    if expected.exists():
        raise FileExistsError(expected)
    print("Rendering animation: " + str(expected), flush=True)
    video_path = simulation.animate_world(
        simulation.sim, animation_name=simulation.current_name, maze_type="maze",
        frame_skip=1, method_name="cc_sf", return_html=False, include_initial_pose=True,
    )
    result.update(animation_path=str(Path(video_path).resolve()),
                  animation_size_bytes=Path(video_path).stat().st_size)
    dump(RUN_DIR / "result.json", result)
    print("ANIMATION_SAVED " + str(Path(video_path).resolve()), flush=True)
finally:
    cleanup_simulation(simulation)
    metadata.update(finished_at_kst=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                    wall_time_s=time.perf_counter() - started)
    dump(RUN_DIR / "experiment.json", metadata)
