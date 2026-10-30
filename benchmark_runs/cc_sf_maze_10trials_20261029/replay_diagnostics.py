"""Replay saved-source trials with passive post-solve observations only."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
SNAPSHOT = BASE / "source_snapshot"
OUT = BASE / "diagnostics"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(SNAPSHOT))

import numpy as np
import torch
import yaml

torch.set_num_threads(6)
torch.set_num_interop_threads(1)

from control.cc_sf_optimizer import CCSFOptimizer
import test_nmpc


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


original_regenerate = CCSFOptimizer._regenerate_covariance_guess
original_build = CCSFOptimizer._build_stage_param
original_prepare = CCSFOptimizer._prepare_and_solve
original_solve = CCSFOptimizer.solve_nlp
cache = {}
active = {}


def regenerate(self, solver, stage_s_values=None):
    result = original_regenerate(self, solver, stage_s_values)
    if active.get("preparing"):
        active["nominal_states"] = result.copy()
        active["nominal_inputs"] = np.stack([solver.get(i, "u") for i in range(self.N)])
    return result


def build(self, *args, **kwargs):
    result = original_build(self, *args, **kwargs)
    if active.get("preparing"):
        active["params"].append(result.copy())
    return result


def prepare(self, solver):
    active.update(preparing=True, params=[])
    try:
        result = original_prepare(self, solver)
        cache[id(solver)] = {
            "params": np.stack(active["params"]),
            "nominal_states": active["nominal_states"],
            "nominal_inputs": active["nominal_inputs"],
        }
        return result
    finally:
        active["preparing"] = False


def solve(self):
    result = original_solve(self)
    solver = result.solver
    states = np.stack([solver.get(i, "x") for i in range(self.N + 1)])
    inputs = np.stack([solver.get(i, "u") for i in range(self.N)])
    frozen = cache[id(solver)]
    fresh = self.ped_model.get_params(states[:, :3])
    if "path" not in active:
        active["path"] = True
        (OUT / f"path_seed{active['seed']}.json").write_text(
            json.dumps(self.reference_path_data, default=plain) + "\n")
    data = {
        "seed": active["seed"], "time": float(self._system._time),
        "mode": self._last_solve_mode, "status": int(solver.status),
        "current_s0": float(self._current_s0),
        "states": states, "inputs": inputs, **frozen,
        "fresh_psdf_params": fresh,
        "path_param_slice": [self.path_param_slice.start, self.path_param_slice.stop],
        "psdf_param_slice": [self.psdf_param_slice.start, self.psdf_param_slice.stop],
        "risk_tensor_param_slice": [self.risk_tensor_param_slice.start, self.risk_tensor_param_slice.stop],
    }
    active["file"].write(json.dumps(data, default=plain) + "\n")
    active["file"].flush()
    return result


CCSFOptimizer._regenerate_covariance_guess = regenerate
CCSFOptimizer._build_stage_param = build
CCSFOptimizer._prepare_and_solve = prepare
CCSFOptimizer.solve_nlp = solve

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=[2, 5, 6, 0])
options = parser.parse_args()
config = yaml.safe_load((BASE / "experiment_config.yaml").read_text())
for seed in options.seeds:
    active.clear()
    active["seed"] = seed
    run_config = copy.deepcopy(config)
    run_config["start_perturbation"]["seed"] = seed
    run_config["output_root_dir"] = str(OUT / f"seed{seed}")
    run_config["output_suffix"] = f"diagnostic_seed{seed}"
    args = test_nmpc.build_parser().parse_args([
        "--single", "--optimizer-type", "cc_sf", "--maze-type", "maze",
        "--robot-shape", "rectangle", "--simulation-time", "60",
        "--start-seed", str(seed), "--no-animation", "--no-plots",
    ])
    print(f"[Replay] seed={seed}, source={test_nmpc.__file__}", flush=True)
    with (OUT / f"telemetry_seed{seed}.jsonl").open("w") as handle:
        active["file"] = handle
        result = test_nmpc.run_single_test(args, run_config, collect_metrics=True)
    (OUT / f"outcome_seed{seed}.json").write_text(json.dumps(result, default=plain, indent=2) + "\n")
    print(f"[Replay] seed={seed}, status={result['status']}, final_time={result['final_time']}", flush=True)
