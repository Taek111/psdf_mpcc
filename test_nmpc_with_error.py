import argparse
from collections import Counter
import copy
import csv
import os

from sim.simulation_mpc import simulation_mpc
from test_nmpc import build_parser, load_runtime_config, run_single_test


DEFAULT_LOCALIZATION_ERROR_CONFIG = {
    "enabled": True,
    "position_noise_frame": "local",
    "position_bias": [0.0, 0.0],
    "heading_bias_deg": 0.0,
    "front_lateral_noise_std": [0.01, 0.005],
    "heading_noise_std_deg": 1.0,
    "seed": 7,
}

DEFAULT_SUCCESS_CRITERIA = {
    "position_tolerance": 0.1,
    "angle_tolerance": 0.2,
}

DEFAULT_TRIAL_FAILURE_CRITERIA = {
    "enabled": True,
    "movement_window_sec": 4.0,
    "movement_threshold": 0.03,
    "max_consecutive_solver_failures": 20,
}


def _normalize_pair(values, field_name: str):
    if values is None:
        return None
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    if len(values) == 2:
        return [float(values[0]), float(values[1])]
    raise ValueError(f"{field_name} expects one value or two values.")


def build_error_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Run NMPC tests with localization error"
    parser.set_defaults(simulation_time=60.0)
    parser.add_argument("--trials", type=int, default=5, help="Number of independent trials to run in --single mode")
    parser.add_argument(
        "--loc-front-lateral-std",
        nargs=2,
        type=float,
        metavar=("SIGMA_FRONT", "SIGMA_LATERAL"),
        help="Gaussian localization noise std on front/lateral axes [m]",
    )
    parser.add_argument(
        "--loc-front-std",
        type=float,
        help="Gaussian localization noise std on front axis [m]",
    )
    parser.add_argument(
        "--loc-lateral-std",
        type=float,
        help="Gaussian localization noise std on lateral axis [m]",
    )
    parser.add_argument("--loc-theta-std-deg", type=float, help="Gaussian localization noise std on heading [deg]")
    parser.add_argument(
        "--loc-seed",
        type=int,
        help="Random seed for localization noise sampling",
    )
    parser.add_argument(
        "--failure-window-sec",
        type=float,
        help="Fail a trial if it moves less than threshold over this time window [s]",
    )
    parser.add_argument(
        "--failure-move-threshold",
        type=float,
        help="Minimum displacement over the failure window [m]",
    )
    parser.add_argument(
        "--failure-max-consecutive-infeasible",
        type=int,
        help="Fail a trial after this many consecutive non-success solver statuses",
    )
    return parser


def build_localization_error_config(args: argparse.Namespace, base_config: dict) -> dict:
    config = copy.deepcopy(base_config or {})
    localization_error_config = copy.deepcopy(DEFAULT_LOCALIZATION_ERROR_CONFIG)
    localization_error_config.update(config.get("localization_error", {}))
    localization_error_config["enabled"] = True

    localization_error_config["position_noise_frame"] = "local"
    localization_error_config["position_bias"] = [0.0, 0.0]
    localization_error_config["heading_bias_deg"] = 0.0

    local_noise_std = _normalize_pair(args.loc_front_lateral_std, "loc-front-lateral-std")
    if local_noise_std is None:
        default_noise_std = localization_error_config.get("front_lateral_noise_std", [0.0, 0.0])
        local_noise_std = _normalize_pair(default_noise_std, "front_lateral_noise_std")

    if args.loc_front_std is not None:
        local_noise_std[0] = float(args.loc_front_std)
    if args.loc_lateral_std is not None:
        local_noise_std[1] = float(args.loc_lateral_std)

    localization_error_config["front_lateral_noise_std"] = local_noise_std
    localization_error_config.pop("position_noise_std", None)
    localization_error_config.pop("front_noise_std", None)
    localization_error_config.pop("lateral_noise_std", None)

    if args.loc_theta_std_deg is not None:
        localization_error_config["heading_noise_std_deg"] = float(args.loc_theta_std_deg)
    if args.loc_seed is not None:
        localization_error_config["seed"] = int(args.loc_seed)

    config["localization_error"] = localization_error_config
    return config


def build_success_criteria_config(config: dict) -> dict:
    success_criteria = copy.deepcopy(DEFAULT_SUCCESS_CRITERIA)
    success_criteria.update(config.get("success_criteria", {}))
    config["success_criteria"] = success_criteria
    return config


def build_trial_failure_criteria_config(args: argparse.Namespace, config: dict) -> dict:
    trial_failure_criteria = copy.deepcopy(DEFAULT_TRIAL_FAILURE_CRITERIA)
    trial_failure_criteria.update(config.get("trial_failure_criteria", {}))
    trial_failure_criteria["enabled"] = True

    if args.failure_window_sec is not None:
        trial_failure_criteria["movement_window_sec"] = float(args.failure_window_sec)
    if args.failure_move_threshold is not None:
        trial_failure_criteria["movement_threshold"] = float(args.failure_move_threshold)
    if args.failure_max_consecutive_infeasible is not None:
        trial_failure_criteria["max_consecutive_solver_failures"] = int(args.failure_max_consecutive_infeasible)

    config["trial_failure_criteria"] = trial_failure_criteria
    return config


def build_trial_config(args: argparse.Namespace, base_config: dict, trial_index: int, total_trials: int) -> dict:
    config = build_localization_error_config(args, base_config)
    config = build_success_criteria_config(config)
    config = build_trial_failure_criteria_config(args, config)

    localization_error_config = config["localization_error"]
    base_seed = localization_error_config.get("seed")
    if base_seed is not None:
        localization_error_config["seed"] = int(base_seed) + trial_index - 1

    if total_trials > 1:
        config["output_root_dir"] = os.path.join("w_error", "trials", f"trial_{trial_index:03d}")
        config["output_suffix"] = f"trial{trial_index:03d}"
    else:
        config["output_root_dir"] = "w_error"
        config.pop("output_suffix", None)

    return config


def build_experiment_name(args: argparse.Namespace) -> str:
    experiment_name = simulation_mpc.build_output_name(
        args.robot_shape,
        args.maze_type,
        args.optimizer_type,
        extra_suffix="locerr_trials",
    )
    return experiment_name.lower()


def save_trial_summary(args: argparse.Namespace, trial_results, success_count: int, success_rate: float):
    output_dir = os.path.join("w_error", "data")
    os.makedirs(output_dir, exist_ok=True)
    experiment_name = build_experiment_name(args)
    summary_path = os.path.join(output_dir, f"{experiment_name}_summary.csv")
    aggregate_path = os.path.join(output_dir, f"{experiment_name}_aggregate.csv")
    failure_summary_path = os.path.join(output_dir, f"{experiment_name}_failure_reasons.csv")

    fieldnames = [
        "trial",
        "seed",
        "status",
        "failure_reason",
        "goal_reached",
        "final_time",
        "distance_to_goal",
        "current_name",
        "output_root_dir",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trial_results)

    aggregate_fieldnames = [
        "trial_count",
        "success_count",
        "failure_count",
        "timeout_count",
        "interrupted_count",
        "success_rate",
    ]
    with open(aggregate_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "trial_count": len(trial_results),
                "success_count": success_count,
                "failure_count": sum(1 for result in trial_results if result.get("status") == "failure"),
                "timeout_count": sum(1 for result in trial_results if result.get("status") == "timeout"),
                "interrupted_count": sum(1 for result in trial_results if result.get("status") == "interrupted"),
                "success_rate": success_rate,
            }
        )

    failure_reason_counter = Counter()
    for result in trial_results:
        status = result.get("status", "unknown")
        failure_reason = result.get("failure_reason") or ("success" if status == "success" else "unknown")
        failure_reason_counter[(status, failure_reason)] += 1

    failure_fieldnames = [
        "status",
        "failure_reason",
        "count",
    ]
    with open(failure_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=failure_fieldnames)
        writer.writeheader()
        for (status, failure_reason), count in sorted(failure_reason_counter.items()):
            writer.writerow(
                {
                    "status": status,
                    "failure_reason": failure_reason,
                    "count": count,
                }
            )

    return summary_path, aggregate_path, failure_summary_path


def run_trials(args: argparse.Namespace, base_config: dict) -> None:
    if not args.single:
        raise ValueError("Multi-trial execution requires --single so success rate is computed for one scenario.")

    trial_results = []
    for trial_index in range(1, args.trials + 1):
        print(f"\n{'=' * 60}")
        print(f"Running trial {trial_index}/{args.trials}")
        print(f"{'=' * 60}")

        trial_args = copy.copy(args)
        if args.trials > 1:
            trial_args.output_suffix = f"locerr_trial{trial_index:03d}"

        trial_config = build_trial_config(args, base_config, trial_index, args.trials)
        trial_seed = trial_config["localization_error"].get("seed")

        try:
            trial_result = run_single_test(trial_args, config=trial_config) or {}
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            trial_result = {
                "status": "failure",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "goal_reached": False,
                "final_time": None,
                "distance_to_goal": None,
                "current_name": None,
                "output_root_dir": trial_config.get("output_root_dir", "w_error"),
            }

        trial_result["trial"] = trial_index
        trial_result["seed"] = trial_seed
        trial_results.append(trial_result)

    success_count = sum(1 for result in trial_results if result.get("status") == "success")
    success_rate = float(success_count) / float(len(trial_results)) if trial_results else 0.0
    summary_path, aggregate_path, failure_summary_path = save_trial_summary(
        args, trial_results, success_count, success_rate
    )

    print(f"\nTrials completed: {len(trial_results)}")
    print(f"Successes: {success_count}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Trial summary saved to {summary_path}")
    print(f"Aggregate summary saved to {aggregate_path}")
    print(f"Failure-reason summary saved to {failure_summary_path}")


def main() -> None:
    args = build_error_parser().parse_args()
    args.output_suffix = "locerr"

    base_config = load_runtime_config(args.config)
    if args.single:
        if args.trials < 1:
            raise ValueError("--trials must be at least 1 in --single mode.")
        run_trials(args, base_config)
        return

    config = build_trial_config(args, base_config, trial_index=1, total_trials=1)

    if args.single:
        run_single_test(args, config=config)
        return

    simulation_mpc().run_tests_from_config(config)


if __name__ == "__main__":
    main()
