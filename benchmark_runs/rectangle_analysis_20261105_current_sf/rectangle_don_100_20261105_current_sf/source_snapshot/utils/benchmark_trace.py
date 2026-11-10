"""Export existing in-memory trial records after navigation, without geometry work.

No clearance evaluation, extra solver reads, CUDA synchronization or per-step
file writes are performed here. The saved true poses and frozen geometry let
an offline process compute exact sampled footprint clearance later.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported trace metadata value: {type(value).__name__}")


def save_raw_trial_trace(test_sim, outcome):
    """Save t=0 and integrated true poses before the normal optimizer cleanup."""
    robot = getattr(test_sim, "robot", None)
    context = getattr(test_sim, "benchmark_trace_context", None)
    initial_pose = getattr(test_sim, "initial_pose", None)
    if robot is None or context is None or initial_pose is None:
        raise ValueError("Trial ended before its initial pose and geometry were available")

    controller = robot._controller
    optimizer = controller._optimizer
    system_logger = robot._system_logger
    states = np.asarray(system_logger._xs, dtype=float)
    if states.size == 0:
        states = np.empty((0, 3), dtype=float)
    else:
        states = states.reshape(len(system_logger._xs), -1)[:, :3]
    poses = np.vstack((np.asarray(initial_pose, dtype=float)[:3], states))
    inputs = np.asarray(system_logger._us, dtype=float)
    if inputs.size == 0:
        inputs = np.empty((0, int(getattr(optimizer, "nu", None) or 3)), dtype=float)
    if len(inputs) != len(states):
        raise ValueError("Integrated pose and applied-input histories are misaligned")

    dt = float(robot._system._dt)
    times = np.asarray(optimizer.solver_times, dtype=float)
    statuses = list(robot._controller_logger._solver_status_infos)
    if len(times) == len(statuses) + 1:
        # An exception can occur after solve_nlp starts but before Robot calls
        # controller.logging(). Preserve that attempted call's failure status.
        if outcome.get("status") == "interrupted":
            # KeyboardInterrupt bypasses BaseController's Exception handler;
            # its cached status can still describe the preceding success.
            statuses.append({
                "raw_status": "interrupted", "status_code": None,
                "success": False, "solver_called": None,
            })
        else:
            statuses.append(controller.get_last_solver_status_info())
    if len(times) != len(statuses):
        raise ValueError("solve_nlp timing and solver status histories are misaligned")

    output_dir = Path(test_sim.output_root_dir or ".") / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "raw_trace.npz"
    metadata_path = output_dir / "raw_trace_metadata.json"
    if trace_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Raw trial trace already exists in {output_dir}")
    # Compression and serialization happen only after navigation has ended.
    with trace_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            poses=poses,
            pose_times_s=np.arange(len(poses), dtype=float) * dt,
            applied_inputs=inputs,
            solve_nlp_times_s=times,
        )
    outcome_fields = (
        "status", "failure_reason", "goal_reached", "final_time",
        "distance_to_goal", "error", "safe_stop_count",
        "backup_feasible_qp_success_count", "plant_input_apply_count",
        "last_solve_mode", "optimizer_runtime_stats",
    )
    metadata = {
        "schema_version": 1,
        **context,
        "initial_pose": np.asarray(initial_pose, dtype=float)[:3],
        "start_perturbation": getattr(test_sim, "start_perturbation_info", None),
        "dt_s": dt,
        "current_name": getattr(test_sim, "current_name", None),
        "outcome": {key: outcome.get(key) for key in outcome_fields},
        "optimizer_parameters": vars(controller._param),
        "solver_status_infos": statuses,
        "array_counts": {
            "poses": len(poses), "applied_inputs": len(inputs),
            "solve_nlp_times_s": len(times), "solver_status_infos": len(statuses),
        },
        "computation_time": {
            "scope": "solve_nlp", "clock": "time.perf_counter", "unit": "s",
            "raw_samples": "all_attempted_calls_including_failed_or_skipped_solves",
            "csv_summary_samples": "successful_solver_steps_including_first_step",
        },
        "clearance": {
            "computed_online": False,
            "sampling": "initial_true_pose_and_each_completed_plant_integration",
        },
        "raw_trace_file": trace_path.name,
        "raw_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    }
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2,
                  default=_json_value, allow_nan=False)
        stream.write("\n")
    return {
        "raw_trace_path": str(trace_path.resolve()),
        "raw_trace_metadata_path": str(metadata_path.resolve()),
        "raw_trace_error": "",
    }
