"""Incremental navigation outcomes and successful solver-step timing CSVs."""

import csv
import math
import re
from pathlib import Path

import numpy as np


CONFIG_FIELDS = (
    "configuration_id", "maze_type", "optimizer_type", "robot_shape",
    "dynamics_type", "path_planner",
)
TRIAL_FIELDS = CONFIG_FIELDS + (
    "trial_number", "start_seed", "start_perturbation_enabled",
    "initial_x", "initial_y", "initial_theta", "success", "status",
    "failure_reason", "goal_reached", "final_time", "distance_to_goal",
    "successful_solver_steps", "solver_time_mean_s", "solver_time_p95_s",
    "current_name", "output_root_dir", "error",
)
TIME_FIELDS = CONFIG_FIELDS + (
    "trial_number", "start_seed", "trial_success", "trial_status",
    "timestep", "simulation_time_s", "solver_computation_time_s",
    "solver_raw_status", "solver_status_code",
)
SUMMARY_FIELDS = CONFIG_FIELDS + (
    "recorded_trials", "completed_trials", "successful_trials", "failed_trials",
    "interrupted_trials", "success_rate", "successful_solver_steps",
    "solver_time_mean_s", "solver_time_p95_s",
)


def timing_statistics(times):
    """Pool individual successful steps, rather than averaging trial percentiles."""
    if not times:
        return {"solver_time_mean_s": "", "solver_time_p95_s": ""}
    return {
        "solver_time_mean_s": float(np.mean(times)),
        "solver_time_p95_s": float(np.percentile(times, 95)),
    }


def collect_successful_solver_steps(test_sim):
    """Pair solver timings with their controller status before optimizer cleanup."""
    robot = getattr(test_sim, "robot", None)
    controller = getattr(robot, "_controller", None)
    optimizer = getattr(controller, "_optimizer", None)
    logger = getattr(robot, "_controller_logger", None)
    times = getattr(optimizer, "solver_times", [])
    statuses = getattr(logger, "_solver_status_infos", [])
    dt = float(getattr(getattr(robot, "_system", None), "_dt", 0.1))
    steps = []
    for index, (elapsed, status) in enumerate(zip(times, statuses)):
        # Unknown statuses, failed solves and invalid measurements are excluded.
        if status.get("success") is not True:
            continue
        try:
            elapsed = float(elapsed)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(elapsed) or elapsed < 0.0:
            continue
        steps.append({
            "timestep": index,
            "simulation_time_s": float((index + 1) * dt),
            "solver_computation_time_s": elapsed,
            "solver_raw_status": status.get("raw_status"),
            "solver_status_code": status.get("status_code"),
        })
    return steps


class TrialMetricsRecorder:
    """Close CSV files after every trial; keep a separate timing file per config."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trial_path = self.output_dir / "trial_results.csv"
        self.summary_path = self.output_dir / "summary.csv"
        if (self.trial_path.exists() or self.summary_path.exists()
                or any(self.output_dir.glob("computation_times_*.csv"))):
            raise FileExistsError(
                f"Trial CSVs already exist in {self.output_dir}; choose a new --results-dir."
            )
        self._groups = {}

    @staticmethod
    def _append(path, fieldnames, rows):
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def record(self, configuration, trial_number, start_settings, result):
        identity = {key: configuration[key] for key in CONFIG_FIELDS}
        group_key = identity["configuration_id"]
        group = self._groups.setdefault(group_key, {
            "identity": identity, "trials": [], "times": [],
        })
        status = result.get("status", "error")
        success = int(status == "success")
        start_info = result.get("start_perturbation") or {}
        initial_pose = start_info.get("initial_pose", result.get("initial_pose"))
        pose = list(initial_pose) if initial_pose is not None else []
        steps = result.get("successful_solver_steps", [])
        times = [step["solver_computation_time_s"] for step in steps]
        start_seed = start_info.get("seed", start_settings.get("seed"))
        row = {
            **identity,
            "trial_number": trial_number,
            "start_seed": start_seed,
            "start_perturbation_enabled": start_settings.get("enabled", False),
            "initial_x": pose[0] if len(pose) > 0 else "",
            "initial_y": pose[1] if len(pose) > 1 else "",
            "initial_theta": pose[2] if len(pose) > 2 else "",
            "success": success,
            "status": status,
            "failure_reason": result.get("failure_reason"),
            "goal_reached": result.get("goal_reached", False),
            "final_time": result.get("final_time"),
            "distance_to_goal": result.get("distance_to_goal"),
            "successful_solver_steps": len(times),
            **timing_statistics(times),
            "current_name": result.get("current_name"),
            "output_root_dir": result.get("output_root_dir", ""),
            "error": result.get("error", ""),
        }
        # The numeric config ID keeps separately configured instances apart.
        tokens = (str(identity[key]) for key in CONFIG_FIELDS)
        filename = "computation_times_" + "_".join(
            re.sub(r"[^A-Za-z0-9_.-]", "-", token) for token in tokens
        ) + ".csv"
        time_rows = [{
            **identity, "trial_number": trial_number, "start_seed": start_seed,
            "trial_success": success, "trial_status": status, **step,
        } for step in steps]
        self._append(self.output_dir / filename, TIME_FIELDS, time_rows)
        self._append(self.trial_path, TRIAL_FIELDS, [row])
        group["trials"].append(row)
        group["times"].extend(times)
        self._save_summary()
        return row

    def _save_summary(self):
        rows = []
        for group in self._groups.values():
            trials = group["trials"]
            interrupted = sum(row["status"] == "interrupted" for row in trials)
            completed = len(trials) - interrupted
            successful = sum(row["success"] for row in trials)
            rows.append({
                **group["identity"],
                "recorded_trials": len(trials),
                "completed_trials": completed,
                "successful_trials": successful,
                "failed_trials": completed - successful,
                "interrupted_trials": interrupted,
                "success_rate": successful / completed if completed else "",
                "successful_solver_steps": len(group["times"]),
                **timing_statistics(group["times"]),
            })
        temporary_path = self.summary_path.with_suffix(".csv.tmp")
        with temporary_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(self.summary_path)

