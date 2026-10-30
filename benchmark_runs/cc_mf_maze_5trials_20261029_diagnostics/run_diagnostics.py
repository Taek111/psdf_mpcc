"""Run the standard five-trial batch with read-only optimizer telemetry."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

torch.set_num_threads(6)
torch.set_num_interop_threads(1)

from control.cc_mf_optimizer import CCMFOptimizer
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
(OUTPUT / "environment.json").write_text(json.dumps({
    "python": sys.executable, "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(), "torch_threads": torch.get_num_threads(),
    "ACADOS_SOURCE_DIR": os.environ.get("ACADOS_SOURCE_DIR"),
    "note": "Diagnostic timing includes extra evaluations outside solver.solve().",
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
    "--trials", "5", "--results-dir", str(OUTPUT / "results"),
    "--no-animation", "--no-plots",
]
main()
