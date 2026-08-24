import argparse
import csv
import signal
from pathlib import Path
from typing import Dict, List, Optional

from sim.simulation_mpc import simulation_mpc


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = Path("config") / "config.yaml"


def resolve_config_path(config_arg: str) -> str:
    config_path = Path(config_arg)
    if config_path.is_absolute() and config_path.exists():
        return str(config_path)

    candidates = [PROJECT_ROOT / config_path]
    if len(config_path.parts) == 1:
        candidates.append(PROJECT_ROOT / "config" / config_path.name)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(config_path)


def build_output_name(args: argparse.Namespace, interrupted: bool = False) -> str:
    prefix_map = {
        "acados": "MPC_SQP_",
        "psdf": "MPC_psdf_",
        "mpcc": "MPC_mpcc_",
        "rmpcc_pv": "MPC_rmpcc_pv_",
        "rmpcc": "MPC_rmpcc_",
        "dcbf": "MPC_DCBF_",
        "dcbf_casadi": "MPC_DCBF_CASADI_",
    }
    name = prefix_map.get(args.optimizer_type, "MPC_") + f"{args.robot_shape}_{args.maze_type}"
    output_suffix = getattr(args, "output_suffix", None)
    if output_suffix:
        name += f"_{output_suffix}"
    if interrupted:
        name += "_interrupted"
    return name


def load_runtime_config(config_arg: str) -> dict:
    resolved_config_path = resolve_config_path(config_arg)
    return simulation_mpc().load_config(resolved_config_path) or {}


def has_animation_data(test_sim: simulation_mpc) -> bool:
    if not hasattr(test_sim, "sim") or test_sim.sim is None:
        return False
    if not hasattr(test_sim.sim, "_robot") or test_sim.sim._robot is None:
        return False

    robot = test_sim.sim._robot
    if hasattr(robot, "_system_logger") and len(robot._system_logger._xs) > 0:
        return True

    return hasattr(robot, "_system") and robot._system is not None and hasattr(robot._system, "get_state")


def generate_animation_if_possible(
    test_sim: simulation_mpc,
    args: argparse.Namespace,
    interrupted: bool = False,
) -> None:
    if not args.generate_animation:
        return

    if not has_animation_data(test_sim):
        if interrupted:
            print("Interrupted before any trajectory state was available; skipping animation generation.")
        return

    animation_name = getattr(test_sim, "current_name", build_output_name(args, interrupted=interrupted))
    if interrupted and animation_name == getattr(test_sim, "current_name", None):
        animation_name = build_output_name(args, interrupted=True)

    print("Generating animation..." if not interrupted else "Generating interrupted-run animation...")
    test_sim.animate_world(
        test_sim.sim,
        animation_name=animation_name,
        maze_type=args.maze_type,
        frame_skip=args.frame_skip,
        method_name=args.optimizer_type,
        use_risk_visualization=getattr(args, "use_risk_visualization", None),
    )


def is_interrupt_runtime_error(exc: RuntimeError, interrupt_requested: bool) -> bool:
    if not interrupt_requested:
        return False

    message = str(exc)
    return (
        "KeyboardInterrupt" in message
        or "NonIpopt_Exception_Thrown" in message
        or "KeyboardInterruptException" in message
    )


def cleanup_simulation(test_sim: simulation_mpc) -> None:
    if hasattr(test_sim, "robot") and test_sim.robot is not None:
        if hasattr(test_sim.robot, "_controller") and hasattr(test_sim.robot._controller, "_optimizer"):
            optimizer = test_sim.robot._controller._optimizer
            if hasattr(optimizer, "cleanup"):
                optimizer.cleanup()

    test_sim.sim = None
    test_sim.robot = None


def collect_optimizer_runtime_stats(test_sim: simulation_mpc) -> dict:
    if not hasattr(test_sim, "robot") or test_sim.robot is None:
        return {}

    controller = getattr(test_sim.robot, "_controller", None)
    optimizer = getattr(controller, "_optimizer", None)
    if optimizer is None or not hasattr(optimizer, "get_runtime_stats"):
        return {}

    try:
        runtime_stats = optimizer.get_runtime_stats()
    except Exception:
        return {}

    if not isinstance(runtime_stats, dict):
        return {}

    return dict(runtime_stats)


def attach_runtime_stats(result: Optional[dict], test_sim: simulation_mpc) -> Optional[dict]:
    if result is None:
        return None

    runtime_stats = collect_optimizer_runtime_stats(test_sim)
    if not runtime_stats:
        return result

    enriched_result = dict(result)
    enriched_result["optimizer_runtime_stats"] = runtime_stats
    if "safe_stop_count" in runtime_stats:
        enriched_result["safe_stop_count"] = int(runtime_stats["safe_stop_count"])
    if "backup_feasible_qp_success_count" in runtime_stats:
        enriched_result["backup_feasible_qp_success_count"] = int(
            runtime_stats["backup_feasible_qp_success_count"]
        )
    if "plant_input_apply_count" in runtime_stats:
        enriched_result["plant_input_apply_count"] = int(runtime_stats["plant_input_apply_count"])
    if "last_solve_mode" in runtime_stats:
        enriched_result["last_solve_mode"] = runtime_stats["last_solve_mode"]
    return enriched_result


def get_safe_stop_count(result: Optional[dict]) -> int:
    if not result:
        return 0

    if "safe_stop_count" in result and result["safe_stop_count"] is not None:
        return int(result["safe_stop_count"])

    runtime_stats = result.get("optimizer_runtime_stats", {})
    if isinstance(runtime_stats, dict):
        return int(runtime_stats.get("safe_stop_count", 0))

    return 0


def build_safe_stop_summary_path(args: argparse.Namespace, rows: List[Dict]) -> Path:
    if args.safe_stop_summary:
        summary_path = Path(args.safe_stop_summary)
        if not summary_path.is_absolute():
            summary_path = PROJECT_ROOT / summary_path
        return summary_path

    output_root_dir = ""
    for row in rows:
        if row.get("output_root_dir"):
            output_root_dir = row["output_root_dir"]
            break

    base_dir = PROJECT_ROOT / output_root_dir if output_root_dir else PROJECT_ROOT
    data_dir = base_dir / "data"
    robot_token = "-".join(dict.fromkeys(row["robot_shape"] for row in rows))
    maze_token = "-".join(dict.fromkeys(row["maze_type"] for row in rows))
    optimizer_token = "-".join(dict.fromkeys(row["optimizer_type"] for row in rows))
    return data_dir / f"safe_stop_summary_{robot_token}_{maze_token}_{optimizer_token}.csv"


def save_safe_stop_summary(summary_path: Path, rows: List[Dict]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "robot_shape",
        "maze_type",
        "optimizer_type",
        "status",
        "failure_reason",
        "safe_stop_count",
        "backup_feasible_qp_success_count",
        "plant_input_apply_count",
        "last_solve_mode",
        "final_time",
        "distance_to_goal",
        "current_name",
        "output_root_dir",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Safe-stop summary saved to {summary_path}")


def run_safe_stop_batch(args: argparse.Namespace, config: Optional[dict] = None) -> List[Dict]:
    robot_shapes = args.safe_stop_robot_shapes or ["rectangle"]
    maze_types = args.safe_stop_maze_types or ["maze"]
    optimizer_types = args.safe_stop_optimizer_types or ["rmpcc_pv", "rmpcc"]

    rows = []
    total_runs = len(robot_shapes) * len(maze_types) * len(optimizer_types)
    run_index = 0

    for robot_shape in robot_shapes:
        for maze_type in maze_types:
            for optimizer_type in optimizer_types:
                run_index += 1
                print(
                    f"\n[Safe-stop batch] Run {run_index}/{total_runs}: "
                    f"robot_shape={robot_shape}, maze_type={maze_type}, optimizer_type={optimizer_type}"
                )
                run_args = argparse.Namespace(**vars(args))
                run_args.single = True
                run_args.robot_shape = robot_shape
                run_args.maze_type = maze_type
                run_args.optimizer_type = optimizer_type

                result = run_single_test(run_args, config=config) or {}
                row = {
                    "robot_shape": robot_shape,
                    "maze_type": maze_type,
                    "optimizer_type": optimizer_type,
                    "status": result.get("status"),
                    "failure_reason": result.get("failure_reason"),
                    "safe_stop_count": get_safe_stop_count(result),
                    "backup_feasible_qp_success_count": result.get("backup_feasible_qp_success_count", 0),
                    "plant_input_apply_count": result.get("plant_input_apply_count", 0),
                    "last_solve_mode": result.get("last_solve_mode"),
                    "final_time": result.get("final_time"),
                    "distance_to_goal": result.get("distance_to_goal"),
                    "current_name": result.get("current_name"),
                    "output_root_dir": result.get("output_root_dir", ""),
                }
                rows.append(row)
                print(
                    "[Safe-stop batch] Result: "
                    f"safe_stop_count={row['safe_stop_count']}, "
                    f"status={row['status']}, "
                    f"failure_reason={row['failure_reason']}"
                )

    if rows:
        summary_path = build_safe_stop_summary_path(args, rows)
        save_safe_stop_summary(summary_path, rows)
        print("\nSafe-stop counts:")
        for row in rows:
            print(
                f"- {row['optimizer_type']} | {row['robot_shape']} | {row['maze_type']}: "
                f"{row['safe_stop_count']}"
            )

    return rows


def run_single_test(args: argparse.Namespace, config: Optional[dict] = None) -> Optional[dict]:
    print("Running single test...")
    test_sim = simulation_mpc()
    test_sim.profile_heatmap_scales = test_sim._resolve_profile_heatmap_scales(
        (config or {}).get("defaults", {}),
        config or {},
    )
    interrupt_requested = False
    result = None
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum, frame):
        nonlocal interrupt_requested
        interrupt_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        test_sim.mpc_test(
            maze_type=args.maze_type,
            robot_shape=args.robot_shape,
            optimizer_type=args.optimizer_type,
            dynamics_type=args.dynamics_type,
            path_planner=args.path_planner,
            simulation_time=args.simulation_time,
            config=config,
        )

        generate_animation_if_possible(test_sim, args)

        if args.generate_plots:
            print("Generating plots...")
            test_sim.plot_world(
                test_sim.sim,
                snapshot_indexes=[],
                figure_name=test_sim.current_name.lower(),
                local_traj_indexes=[],
                maze_type=args.maze_type,
            )
            test_sim.plot_profiles(
                test_sim.sim,
                figure_name=test_sim.current_name.lower(),
                maze_type=args.maze_type,
                include_risk_profile=args.generate_risk_profile,
            )

        print("Single test completed.")
        result = dict(test_sim.last_run_outcome or {})
        result["current_name"] = getattr(test_sim, "current_name", None)
        result["output_root_dir"] = getattr(test_sim, "output_root_dir", "")
        result = attach_runtime_stats(result, test_sim)
    except KeyboardInterrupt:
        interrupt_requested = True
        print("\nKeyboard interrupt received. Finalizing outputs from the partial run...")
        generate_animation_if_possible(test_sim, args, interrupted=True)
        result = {
            "status": "interrupted",
            "failure_reason": "keyboard_interrupt",
            "current_name": getattr(test_sim, "current_name", None),
            "output_root_dir": getattr(test_sim, "output_root_dir", ""),
        }
        result = attach_runtime_stats(result, test_sim)
    except RuntimeError as exc:
        if not is_interrupt_runtime_error(exc, interrupt_requested):
            raise
        print("\nSIGINT interrupted the solver. Finalizing outputs from the partial run...")
        generate_animation_if_possible(test_sim, args, interrupted=True)
        result = {
            "status": "interrupted",
            "failure_reason": str(exc),
            "current_name": getattr(test_sim, "current_name", None),
            "output_root_dir": getattr(test_sim, "output_root_dir", ""),
        }
        result = attach_runtime_stats(result, test_sim)
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        cleanup_simulation(test_sim)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NMPC tests")
    parser.add_argument(
        "--config",
        "-c",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--single",
        "-s",
        action="store_true",
        help="Run one test from CLI options instead of config file",
    )
    parser.add_argument("--maze-type", default="s_path", help="Environment type")
    parser.add_argument("--robot-shape", default="rectangle", help="Robot shape")
    parser.add_argument("--optimizer-type", default="casadi", help="Optimizer type")
    parser.add_argument("--dynamics-type", default="differential_drive", help="Dynamics type")
    parser.add_argument("--path-planner", default="astar", help="Path planner type")
    parser.add_argument("--simulation-time", type=float, default=60.0, help="Simulation duration [s]")
    parser.add_argument("--frame-skip", type=int, default=5, help="Animation frame skip")
    parser.add_argument(
        "--no-animation",
        dest="generate_animation",
        action="store_false",
        help="Disable animation generation in --single mode",
    )
    parser.add_argument(
        "--no-plots",
        dest="generate_plots",
        action="store_false",
        help="Disable plot generation in --single mode",
    )
    parser.add_argument(
        "--risk-profile",
        dest="generate_risk_profile",
        action="store_true",
        help="Generate an additional risk heatmap profile when risk_margin_max is available",
    )
    parser.add_argument(
        "--risk-visualization",
        dest="use_risk_visualization",
        action="store_true",
        help="Show active stage-wise Boole risk circles in RMPCC-MF animations",
    )
    parser.add_argument(
        "--safe-stop-batch",
        action="store_true",
        help="Run a safe-stop summary batch; defaults to rectangle x maze for rmpcc_pv and rmpcc",
    )
    parser.add_argument(
        "--safe-stop-robot-shapes",
        nargs="+",
        default=None,
        help="Robot shapes to use with --safe-stop-batch (default: rectangle)",
    )
    parser.add_argument(
        "--safe-stop-maze-types",
        nargs="+",
        default=None,
        help="Maze types to use with --safe-stop-batch (default: maze)",
    )
    parser.add_argument(
        "--safe-stop-optimizer-types",
        nargs="+",
        default=None,
        help="Optimizer types to use with --safe-stop-batch (default: rmpcc_pv rmpcc)",
    )
    parser.add_argument(
        "--safe-stop-summary",
        default=None,
        help="Optional CSV path for the safe-stop batch summary",
    )
    parser.set_defaults(
        generate_animation=True,
        generate_plots=True,
        generate_risk_profile=True,
        use_risk_visualization=None,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_runtime_config(args.config)

    if args.safe_stop_batch:
        run_safe_stop_batch(args, config=config)
        return

    if args.single:
        run_single_test(args, config=config)
        return

    simulation_mpc().run_tests_from_config(config)


if __name__ == "__main__":
    main()
