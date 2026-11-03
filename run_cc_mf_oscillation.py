"""Run matched CC-MF experiments with standard logging and source snapshots."""

import argparse
import json
from pathlib import Path
import sys

from run_width_benchmark import runtime_environment, snapshot_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    runtime_environment()

    import numpy as np
    import torch
    import yaml

    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    from control.cc_mf_optimizer import CCMFOptimizerParam
    from test_nmpc import main as run_trials
    from sim.simulation_mpc import simulation_mpc, DifferentialDriveRectangleGeometry

    output = args.results_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(args.config.read_text())
    snapshot_sources(output)
    parameters = CCMFOptimizerParam()
    for key, value in config.get("cc_mf", {}).items():
        if not hasattr(parameters, key):
            raise ValueError(f"Unknown CC-MF parameter: {key}")
        setattr(parameters, key, value)
    (output / "resolved_cc_mf_parameters.json").write_text(json.dumps(
        vars(parameters), indent=2,
        default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value,
    ) + "\n")
    (output / "environment.json").write_text(json.dumps({
        "python": sys.executable,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "torch_threads": torch.get_num_threads(),
        "note": "Standard CSV logging; no extra online covariance rollout or risk evaluation.",
    }, indent=2) + "\n")
    start, goal, grid, obstacles = simulation_mpc().create_env("maze")
    length = config["rectangle_geometry"]["length"]
    width = config["rectangle_geometry"]["width"]
    geometry = DifferentialDriveRectangleGeometry(length, width, 0.0)
    (output / "environment_geometry.json").write_text(json.dumps({
        "environment": "maze", "bounds": grid[0], "start": start,
        "goal": goal, "footprint_length_m": length, "footprint_width_m": width,
        "footprint_vertices_local": geometry._region.get_ccw_vertices(),
        "obstacles": [obstacle.get_ccw_vertices() for obstacle in obstacles],
    }, indent=2, default=lambda value: np.asarray(value).tolist()) + "\n")
    sys.argv = ["test_nmpc.py", "-c", str(args.config.resolve()),
                "--trials", str(args.trials), "--results-dir", str(output / "results"),
                "--no-animation", "--no-plots"]
    run_trials()


if __name__ == "__main__":
    main()
