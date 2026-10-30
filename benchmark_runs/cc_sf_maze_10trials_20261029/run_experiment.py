"""Run the current CC-SF benchmark and persist native safe-stop counters."""

import csv
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
import yaml

torch.set_num_threads(6)
torch.set_num_interop_threads(1)

import test_nmpc
from control.cc_sf_optimizer import CCSFOptimizer, CCSFOptimizerParam


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


config = yaml.safe_load((ROOT / "config/config_benchmark.yaml").read_text())
config["trials"] = 10
config["test_configs"] = [{
    "maze_type": "maze", "robot_shape": "rectangle", "optimizer_type": "cc_sf",
    "dynamics_type": "differential_drive", "path_planner": "astar",
}]
config["defaults"].update(generate_animation=True, generate_plots=False)
config["output_suffix"] = "current10_20261029"
config_path = OUTPUT / "experiment_config.yaml"
config_path.write_text(yaml.safe_dump(config, sort_keys=False))

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
param = CCSFOptimizerParam()
for key, value in config.get("cc_sf", {}).items():
    setattr(param, key, value)
(OUTPUT / "resolved_cc_sf_parameters.json").write_text(
    json.dumps(vars(param), default=plain, indent=2) + "\n")
(OUTPUT / "environment.json").write_text(json.dumps({
    "python": sys.executable, "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    "torch_threads": torch.get_num_threads(),
    "ACADOS_SOURCE_DIR": os.environ.get("ACADOS_SOURCE_DIR"),
    "baseline_config": "config/config_benchmark.yaml",
    "seeds": list(range(10)),
    "safe_stop_definition": "Number of zero-input safe-stop control steps, including consecutive steps.",
    "animation": "Existing animate_world writer; no additional encoding.",
}, indent=2) + "\n")

original_solve = CCSFOptimizer.solve_nlp
original_recorder = test_nmpc.TrialMetricsRecorder
active_trial = {"number": 1, "events": 0}
event_fields = ["trial_number", "start_seed", "time_before_control_s", "timestep",
                "x", "y", "theta", "solver_status", "safe_stop_count"]
event_path = OUTPUT / "safe_stop_events.csv"
original_recorder._append(event_path, event_fields, [])


def solve_with_event_log(optimizer):
    previous = optimizer.get_runtime_stats()["safe_stop_count"]
    result = original_solve(optimizer)
    stats = optimizer.get_runtime_stats()
    if stats["safe_stop_count"] > previous:
        system = optimizer._system
        pose = np.asarray(optimizer.state._x).reshape(-1)
        elapsed = float(system._time)
        original_recorder._append(event_path, event_fields, [{
            "trial_number": active_trial["number"],
            "start_seed": active_trial["number"] - 1,
            "time_before_control_s": elapsed,
            "timestep": round(elapsed / float(system._dt)),
            "x": float(pose[0]), "y": float(pose[1]), "theta": float(pose[2]),
            "solver_status": optimizer.solver.status,
            "safe_stop_count": stats["safe_stop_count"],
        }])
        active_trial["events"] += 1
    return result


class RecorderWithSafeStops(original_recorder):
    def record(self, configuration, trial_number, start_settings, result):
        row = super().record(configuration, trial_number, start_settings, result)
        stats = result.get("optimizer_runtime_stats", {})
        safe_stops = test_nmpc.get_safe_stop_count(result)
        animations = sorted((ROOT / "animations/cc_sf").glob(
            f"{result.get('current_name', '')}_*_config01_trial{trial_number:04d}.mp4"))
        extra = {
            **row, "safe_stop_count": safe_stops,
            "safe_stop_event_rows": active_trial["events"],
            "backup_feasible_qp_success_count": stats.get("backup_feasible_qp_success_count", 0),
            "plant_input_apply_count": stats.get("plant_input_apply_count", 0),
            "animation_path": str(animations[-1]) if animations else "",
        }
        self._append(OUTPUT / "safe_stop_summary.csv", list(extra), [extra])
        (OUTPUT / f"outcome_trial{trial_number:02d}.json").write_text(
            json.dumps(result, default=plain, indent=2) + "\n")
        print(f"[Safe stop] trial={trial_number}, count={safe_stops}, events={active_trial['events']}")
        active_trial.update(number=trial_number + 1, events=0)
        return row


CCSFOptimizer.solve_nlp = solve_with_event_log
test_nmpc.TrialMetricsRecorder = RecorderWithSafeStops
sys.argv = [
    "test_nmpc.py", "-c", str(config_path), "--trials", "10",
    "--results-dir", str(OUTPUT / "results"), "--no-plots",
]
try:
    test_nmpc.main()
finally:
    (OUTPUT / "source_changed_during_batch.json").write_text(json.dumps([
        key for key, value in hashes.items()
        if not (ROOT / key).exists()
        or hashlib.sha256((ROOT / key).read_bytes()).hexdigest() != value
    ], indent=2) + "\n")
