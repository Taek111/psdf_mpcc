"""Sequential matched-seed CC-MF batch with the previous diagnostic telemetry."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
BASELINE = ROOT / "benchmark_runs/cc_mf_maze_5trials_20261029_diagnostics"
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--weight", type=float, required=True)
parser.add_argument("--label", required=True)
args = parser.parse_args()
OUTPUT = BASE / "batches" / args.label
OUTPUT.mkdir(parents=True, exist_ok=False)

import numpy as np
import torch
import yaml

torch.set_num_threads(6)
torch.set_num_interop_threads(1)

from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
from test_nmpc import main


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


config = yaml.safe_load((BASELINE / "diagnostic_config.yaml").read_text())
config["cc_mf"]["r_delta_omega"] = args.weight
(OUTPUT / "diagnostic_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
sources = [ROOT / "test_nmpc.py", ROOT / "AGENTS.md"]
for directory in ("control", "models", "planning", "sim", "utils", "config"):
    sources.extend(path for path in (ROOT / directory).rglob("*")
                   if path.is_file() and path.suffix in (".py", ".yaml")
                   and "__pycache__" not in path.parts)
hashes = {}
for source in sources:
    relative = source.relative_to(ROOT)
    target = OUTPUT / "source_snapshot" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
(OUTPUT / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
baseline_hashes = json.loads((BASELINE / "source_hashes.json").read_text())
(OUTPUT / "source_changes_from_baseline.json").write_text(json.dumps({
    "changed": [key for key, value in baseline_hashes.items() if hashes.get(key) != value],
    "added": [key for key in hashes if key not in baseline_hashes],
}, indent=2) + "\n")
resolved = CCMFOptimizerParam()
for key, value in config["cc_mf"].items():
    setattr(resolved, key, value)
(OUTPUT / "resolved_cc_mf_parameters.json").write_text(json.dumps(plain(vars(resolved)), indent=2) + "\n")
(OUTPUT / "environment.json").write_text(json.dumps({
    "python": sys.executable, "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(), "torch_threads": torch.get_num_threads(),
    "ACADOS_SOURCE_DIR": os.environ.get("ACADOS_SOURCE_DIR"),
    "r_delta_omega": args.weight, "trials": 3, "seeds": [0, 1, 2],
    "note": "Same baseline config; extra telemetry is outside solve() timing.",
}, indent=2) + "\n")

original_finish = CCMFOptimizer._finish_solve
instance_ids = {}


def finish_with_telemetry(optimizer, mode, status, solver, diagnostics):
    original_finish(optimizer, mode, status, solver, diagnostics)
    if optimizer not in instance_ids:
        instance_ids[optimizer] = len(instance_ids) + 1
    states, inputs = optimizer._read_solver_trajectory(solver)
    data = optimizer._constraint_data or {}
    record = {
        "time": optimizer._get_cycle_log_time(),
        "solve_step": optimizer._solve_count,
        "mode": mode, "status": status,
        "main_status": optimizer._last_main_status,
        "pose": optimizer.state._x,
        "current_s0": optimizer._current_s0,
        "states": states, "inputs": inputs,
        "nominal": {key: data.get(key) for key in (
            "phi", "guard_A", "guard_c", "mf_A", "mf_c", "mf_mask",
            "residual", "probability_sum", "branch", "status", "tail_clipped",
        )},
        "diagnostics": diagnostics,
        "failed_diagnostics": optimizer._last_failed_constraint_diagnostics,
    }
    path = OUTPUT / f"telemetry_trial{instance_ids[optimizer]:02d}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(plain(record), allow_nan=False) + "\n")


CCMFOptimizer._finish_solve = finish_with_telemetry
sys.argv = [
    "test_nmpc.py", "-c", str(OUTPUT / "diagnostic_config.yaml"),
    "--trials", "3", "--results-dir", str(OUTPUT / "results"),
    "--no-animation", "--no-plots",
]
try:
    main()
finally:
    (OUTPUT / "source_changed_during_batch.json").write_text(json.dumps([
        key for key, value in hashes.items()
        if not (ROOT / key).exists()
        or hashlib.sha256((ROOT / key).read_bytes()).hexdigest() != value
    ], indent=2) + "\n")
