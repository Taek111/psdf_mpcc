#!/usr/bin/env python3
"""Run the sequential RMPCC infeasibility/collision tuning experiments."""

import argparse
import copy
import csv
import json
import math
import subprocess
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
import yaml

from sim.simulation_mpc import simulation_mpc


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "debug_runs" / "rmpcc_tuning"
SUMMARY_FIELDS = (
    "run_id",
    "scenario",
    "setting",
    "status",
    "s_max",
    "min_psdf",
    "solver_failure_count",
    "mf_slack_max",
)
SCENARIO_THRESHOLDS = MappingProxyType(
    {
        "s_path": 0.8803319765,
        "maze": 2.7829378751,
    }
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

R0_CONFIG = MappingProxyType(
    {
        "horizon": 20,
        "tf": 2.0,
        "use_obstacle_constraint": True,
        "use_row_mf": True,
        "d_col": 0.001,
        "d_min": 0.001,
        "d_mf_mask": 0.001,
        "mf_slack_linear": 100.0,
        "mf_slack_quadratic": 1.0,
        "chance_epsilon": 0.20,
        "mf_active_start_step": 1,
        "mf_active_end_step": None,
        "mf_active_terminal": True,
        "sigma_f0": 0.0002,
        "sigma_l0": 0.00025,
        "sigma_psi0": 0.00025,
        "q_f0": 0.0,
        "q_l0": 0.0,
        "q_psi0": 0.0,
        "alpha_f": 0.002,
        "alpha_v": 0.0004,
        "alpha_kappa": 0.01,
        "beta_v": 0.02,
        "beta_kappa": 0.008,
        "beta_omega": 0.008,
        "enable_backup_solver": False,
        "debug_mf": True,
        "debug_infeasibility": True,
        "augmented_psdf_device": DEVICE,
    }
)
COVARIANCE_GROWTH_KEYS = (
    "alpha_f",
    "alpha_v",
    "alpha_kappa",
    "beta_v",
    "beta_kappa",
    "beta_omega",
)

SETTING_SPECS = (
    ("G0", "G0", {"use_row_mf": False}),
    ("R0", "R0", {}),
    ("P1-L1", "linear=1", {"mf_slack_linear": 1.0}),
    ("P1-L3", "linear=3", {"mf_slack_linear": 3.0}),
    ("P1-L10", "linear=10", {"mf_slack_linear": 10.0}),
    ("P1-L30", "linear=30", {"mf_slack_linear": 30.0}),
    ("P2-Q0.1", "quadratic=0.1", {"mf_slack_quadratic": 0.1}),
    ("P2-Q10", "quadratic=10", {"mf_slack_quadratic": 10.0}),
    ("P3-E0.3", "epsilon=0.3", {"chance_epsilon": 0.30}),
    ("P3-E0.5", "epsilon=0.5", {"chance_epsilon": 0.50}),
    (
        "P4-H5",
        "horizon=1-5",
        {
            "mf_active_start_step": 1,
            "mf_active_end_step": 5,
            "mf_active_terminal": False,
        },
    ),
    (
        "P4-H10",
        "horizon=1-10",
        {
            "mf_active_start_step": 1,
            "mf_active_end_step": 10,
            "mf_active_terminal": False,
        },
    ),
    (
        "P4-H20",
        "horizon=1-20",
        {
            "mf_active_start_step": 1,
            "mf_active_end_step": 20,
            "mf_active_terminal": False,
        },
    ),
    ("P5-C0", "covariance_scale=0", {"covariance_growth_scale": 0.0}),
    (
        "P5-C0.01",
        "covariance_scale=0.01",
        {"covariance_growth_scale": 0.01},
    ),
    (
        "P5-C0.03",
        "covariance_scale=0.03",
        {"covariance_growth_scale": 0.03},
    ),
    ("P5-C0.1", "covariance_scale=0.1", {"covariance_growth_scale": 0.1}),
    ("P5-C0.3", "covariance_scale=0.3", {"covariance_growth_scale": 0.3}),
)
SETTING_BY_ID = {setting_id: (label, overrides) for setting_id, label, overrides in SETTING_SPECS}


def resolve_rmpcc_config(overrides):
    """Build one setting from the immutable R0 values."""
    resolved = dict(R0_CONFIG)
    setting_overrides = copy.deepcopy(overrides)
    scale = setting_overrides.pop("covariance_growth_scale", None)
    if scale is not None:
        scale = float(scale)
        for key in COVARIANCE_GROWTH_KEYS:
            resolved[key] = float(R0_CONFIG[key]) * scale
    resolved.update(setting_overrides)
    return resolved


def build_run_config(run_dir, run_id, rmpcc_config, simulation_time):
    return {
        "test_configs": [
            {
                "maze_type": None,
                "robot_shape": "rectangle",
                "optimizer_type": "rmpcc",
                "dynamics_type": "differential_drive",
                "path_planner": "astar",
            }
        ],
        "defaults": {
            "simulation_time": float(simulation_time),
            "generate_animation": False,
            "generate_plots": False,
        },
        "output_root_dir": str(run_dir),
        "output_suffix": run_id,
        "localization_error": {"enabled": False},
        "trial_failure_criteria": {
            "enabled": True,
            "movement_window_sec": 5.0,
            "movement_threshold": 0.005,
            "max_consecutive_solver_failures": 10,
        },
        "rmpcc": copy.deepcopy(rmpcc_config),
    }


def source_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def finite_values(values):
    array = np.asarray(list(values), dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def json_values(value):
    if value in (None, ""):
        return np.array([], dtype=float)
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return np.array([], dtype=float)
    return finite_values(item for item in decoded if item is not None)


def is_finite_scalar(value):
    if value in (None, ""):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def has_nonfinite_run_data(optimizer, cycle_rows, pose_sdf_data):
    if cycle_rows:
        for row in cycle_rows:
            if not all(
                is_finite_scalar(row.get(field))
                for field in ("time", "s", "psdf")
            ):
                return True
            try:
                u0 = json.loads(row.get("u0", ""))
            except (TypeError, ValueError, json.JSONDecodeError):
                return True
            if (
                not isinstance(u0, list)
                or not u0
                or any(not is_finite_scalar(value) for value in u0)
            ):
                return True

        covariance_rows = (
            optimizer.get_covariance_log_rows()
            if optimizer is not None
            else []
        )
        if not covariance_rows:
            return True
        required = (
            "time",
            "stage",
            "sigma_f",
            "sigma_l",
            "sigma_psi",
            "P_lpsi",
        )
        for row in covariance_rows:
            if not all(is_finite_scalar(row.get(field)) for field in required):
                return True
            for field in ("v", "omega"):
                value = row.get(field)
                if value not in (None, "") and not is_finite_scalar(value):
                    return True
        return False

    for row in pose_sdf_data:
        for field in ("predicted_s", "sdf_value"):
            value = row.get(field)
            if value not in (None, "") and not is_finite_scalar(value):
                return True
    return False


def classify_status(
    error,
    s_max,
    min_psdf,
    solver_failure_count,
    has_nonfinite=False,
):
    if (
        error is not None
        or has_nonfinite
        or not math.isfinite(s_max)
        or not math.isfinite(min_psdf)
    ):
        return "error"
    if min_psdf < -1e-6:
        return "collision"
    if min_psdf < float(R0_CONFIG["d_col"]) - 1e-6:
        return "clearance_fail"
    if solver_failure_count:
        return "solver_fail"
    return "success" if s_max >= 0.0 else "stuck"


def evaluate_run(scenario, optimizer, pose_sdf_data, error):
    cycle_rows = optimizer.get_cycle_log_rows() if optimizer is not None else []
    if cycle_rows:
        s_values = finite_values(row["s"] for row in cycle_rows)
        psdf_values = finite_values(row["psdf"] for row in cycle_rows)
    else:
        s_values = finite_values(
            row.get("predicted_s") for row in pose_sdf_data
            if row.get("predicted_s") is not None
        )
        psdf_values = finite_values(
            row.get("sdf_value") for row in pose_sdf_data
            if row.get("sdf_value") is not None
        )

    s_max = float(np.max(s_values)) if s_values.size else float("nan")
    min_psdf = float(np.min(psdf_values)) if psdf_values.size else float("nan")

    runtime_stats = optimizer.get_runtime_stats() if optimizer is not None else {}
    solver_failure_count = (
        int(runtime_stats.get("main_solver_failure_count", 0))
        + int(runtime_stats.get("backup_solver_failure_count", 0))
        + int(runtime_stats.get("safe_stop_count", 0))
    )

    slack_values = []
    for row in cycle_rows:
        slack_values.extend(json_values(row.get("mf_slack")).tolist())
    mf_slack_max = max(slack_values) if slack_values else 0.0

    nonfinite_detected = has_nonfinite_run_data(
        optimizer,
        cycle_rows,
        pose_sdf_data,
    )

    status = classify_status(
        error,
        s_max,
        min_psdf,
        solver_failure_count,
        has_nonfinite=nonfinite_detected,
    )
    if (
        status == "success"
        and s_max < float(SCENARIO_THRESHOLDS[scenario])
    ):
        status = "stuck"

    return {
        "status": status,
        "s_max": s_max,
        "min_psdf": min_psdf,
        "solver_failure_count": solver_failure_count,
        "mf_slack_max": float(mf_slack_max),
    }


def load_summary(summary_path):
    if not summary_path.exists():
        return []
    with summary_path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def save_summary(summary_path, rows):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value):
    return "" if not math.isfinite(float(value)) else f"{float(value):.12g}"


def update_plan_checkbox(setting_id):
    plan_path = PROJECT_ROOT / "debugging_plan.md"
    text = plan_path.read_text(encoding="utf-8")
    marker = f"- [ ] `{setting_id}`:"
    if marker not in text:
        return False
    plan_path.write_text(
        text.replace(marker, f"- [x] `{setting_id}`:", 1),
        encoding="utf-8",
    )
    return True


def required_artifacts_exist(run_dir):
    data_dir = run_dir / "data"
    return all(
        path.exists()
        for path in (
            run_dir / "resolved_config.yaml",
            data_dir / "cycle_raw.csv",
            data_dir / "covariance_trajectory.csv",
        )
    )


def cleanup_simulation(test_sim):
    robot = getattr(test_sim, "robot", None)
    controller = getattr(robot, "_controller", None)
    optimizer = getattr(controller, "_optimizer", None)
    if optimizer is not None and hasattr(optimizer, "cleanup"):
        optimizer.cleanup()
    test_sim.sim = None
    test_sim.robot = None


def run_one(setting_id, setting, overrides, scenario, repeat, args):
    token = setting_id.lower().replace(".", "p")
    run_id = f"{token}_{scenario}_r{repeat}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rmpcc_config = resolve_rmpcc_config(overrides)
    config = build_run_config(
        run_dir,
        run_id,
        rmpcc_config,
        args.simulation_time,
    )
    config["test_configs"][0]["maze_type"] = scenario
    config["metadata"] = {
        "run_id": run_id,
        "setting": setting,
        "scenario": scenario,
        "repeat": repeat,
        "source_commit": source_commit(),
        "device": DEVICE,
    }
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    print(
        f"[RUN] {run_id}: setting={setting}, scenario={scenario}",
        flush=True,
    )
    test_sim = simulation_mpc()
    optimizer = None
    error = None
    with (run_dir / "run.log").open("w", encoding="utf-8") as log_stream:
        try:
            with redirect_stdout(log_stream), redirect_stderr(log_stream):
                test_sim.mpc_test(
                    maze_type=scenario,
                    robot_shape="rectangle",
                    optimizer_type="rmpcc",
                    dynamics_type="differential_drive",
                    path_planner="astar",
                    simulation_time=args.simulation_time,
                    config=config,
                )
        except Exception as exc:
            error = exc
            traceback.print_exc(file=log_stream)
            robot = getattr(test_sim, "robot", None)
            if robot is not None:
                try:
                    test_sim.save_rmpcc_debug_history_to_csv()
                    test_sim.save_pose_sdf_to_csv(
                        f"pose_sdf_{getattr(test_sim, 'current_name', run_id)}"
                    )
                except Exception:
                    traceback.print_exc(file=log_stream)
        finally:
            robot = getattr(test_sim, "robot", None)
            controller = getattr(robot, "_controller", None)
            optimizer = getattr(controller, "_optimizer", None)

    metrics = evaluate_run(
        scenario,
        optimizer,
        list(test_sim.pose_sdf_data),
        error,
    )
    cleanup_simulation(test_sim)
    row = {
        "run_id": run_id,
        "scenario": scenario,
        "setting": setting,
        "status": metrics["status"],
        "s_max": format_metric(metrics["s_max"]),
        "min_psdf": format_metric(metrics["min_psdf"]),
        "solver_failure_count": metrics["solver_failure_count"],
        "mf_slack_max": format_metric(metrics["mf_slack_max"]),
    }
    print(
        f"[DONE] {run_id}: status={row['status']}, "
        f"s_max={row['s_max']}, min_psdf={row['min_psdf']}, "
        f"solver_failures={row['solver_failure_count']}",
        flush=True,
    )
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=[item[0] for item in SETTING_SPECS],
        default=[item[0] for item in SETTING_SPECS],
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIO_THRESHOLDS),
        default=list(SCENARIO_THRESHOLDS),
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--simulation-time", type=float, default=60.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="rerun even when a completed summary row and artifacts exist",
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if not args.output_root.is_absolute():
        args.output_root = PROJECT_ROOT / args.output_root
    return args


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "debug_summary.csv"
    summary_rows = load_summary(summary_path)
    rows_by_id = {row["run_id"]: row for row in summary_rows}

    for setting_id in args.settings:
        setting, overrides = SETTING_BY_ID[setting_id]
        for scenario in args.scenarios:
            token = setting_id.lower().replace(".", "p")
            run_id = f"{token}_{scenario}_r{args.repeat}"
            run_dir = args.output_root / run_id
            prior = rows_by_id.get(run_id)
            if (
                args.resume
                and prior is not None
                and prior.get("status") != "error"
                and required_artifacts_exist(run_dir)
            ):
                print(f"[SKIP] {run_id}: completed artifacts found", flush=True)
                continue

            row = run_one(
                setting_id,
                setting,
                overrides,
                scenario,
                args.repeat,
                args,
            )
            rows_by_id[run_id] = row
            summary_rows = [
                existing
                for existing in summary_rows
                if existing["run_id"] != run_id
            ]
            summary_rows.append(row)
            save_summary(summary_path, summary_rows)

        expected_ids = {
            f"{setting_id.lower().replace('.', 'p')}_{scenario}_r{args.repeat}"
            for scenario in SCENARIO_THRESHOLDS
        }
        setting_complete = expected_ids.issubset(rows_by_id) and all(
            rows_by_id[run_id].get("status") != "error"
            and required_artifacts_exist(args.output_root / run_id)
            for run_id in expected_ids
        )
        if setting_complete:
            update_plan_checkbox(setting_id)

    print(f"Summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
