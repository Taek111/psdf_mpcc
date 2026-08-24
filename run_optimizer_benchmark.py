#!/usr/bin/env python3
"""Run the nominal maze optimizer benchmark and RMPCC-MF scale sweep."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import yaml

from sim.simulation_mpc import simulation_mpc


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark_runs" / "maze_rectangle_nominal"
DEFAULT_SCALES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DEADLINE_SECONDS = 0.1

COVARIANCE_GROWTH_BASE = {
    "alpha_f": 0.002,
    "alpha_v": 0.0004,
    "alpha_kappa": 0.01,
    "beta_v": 0.02,
    "beta_kappa": 0.008,
    "beta_omega": 0.008,
}

METHOD_LABELS = {
    "psdf": "PSDF-MPC",
    "mpcc": "PSDF-MPCC",
    "rmpcc_pv": "PSDF-MPCC-CC-SF",
    "rmpcc": "PSDF-MPCC-CC-MF",
}

SUMMARY_FIELDS = (
    "run_id",
    "trial_number",
    "method",
    "optimizer_type",
    "covariance_growth_scale",
    "status",
    "failure_reason",
    "goal_reached",
    "final_time",
    "driving_time",
    "distance_to_goal",
    "step_count",
    "controller_time_median",
    "controller_time_p95",
    "controller_time_max",
    "deadline_miss_count",
    "deadline_miss_rate",
    "solver_time_median",
    "solver_time_p95",
    "solver_time_max",
    "clearance_q05",
    "clearance_min",
    "collision",
    "rms_delta_v",
    "rms_delta_omega",
    "solver_failure_count",
    "solver_failure_rate",
    "safe_stop_count",
    "raw_logs_complete",
    "trial_history_file",
    "pose_sdf_file",
    "error",
)


def scale_token(scale: float) -> str:
    return f"{float(scale):g}".replace("-", "m").replace(".", "p")


def build_rmpcc_scale_config(scale: float) -> Dict[str, float]:
    """Resolve scale-dependent coefficients without changing RMPCCOptimizerParam."""
    scale = float(scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("covariance growth scale must be a finite non-negative value")

    resolved = {"covariance_growth_scale": scale}
    resolved.update(
        {
            key: float(base_value) * scale
            for key, base_value in COVARIANCE_GROWTH_BASE.items()
        }
    )
    return resolved


def source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def ensure_acados_runtime() -> None:
    """Re-exec once with the local acados shared libraries on the loader path."""
    configured_source = os.environ.get("ACADOS_SOURCE_DIR")
    candidates = []
    if configured_source:
        candidates.append(Path(configured_source))
    candidates.append(PROJECT_ROOT.parent / "acados")

    acados_source = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "lib" / "libacados.so").exists()
        ),
        None,
    )
    if acados_source is None:
        return

    lib_dir = str(acados_source / "lib")
    current_paths = [
        path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path
    ]
    os.environ.setdefault("ACADOS_SOURCE_DIR", str(acados_source))
    if lib_dir in current_paths:
        return
    if os.environ.get("PSDF_BENCHMARK_ACADOS_REEXEC") == "1":
        return

    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = ":".join([lib_dir, *current_paths])
    environment["PSDF_BENCHMARK_ACADOS_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)


def build_run_config(
    run_dir: Path,
    run_id: str,
    optimizer_type: str,
    simulation_time: float,
    scale: Optional[float] = None,
) -> dict:
    config = {
        "test_configs": [
            {
                "maze_type": "maze",
                "robot_shape": "rectangle",
                "optimizer_type": optimizer_type,
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
        "metadata": {
            "run_id": run_id,
            "source_commit": source_commit(),
            "maze_type": "maze",
            "robot_shape": "rectangle",
            "dynamics_type": "differential_drive",
            "path_planner": "astar",
            "simulation_time": float(simulation_time),
        },
    }
    if scale is not None:
        config["rmpcc"] = build_rmpcc_scale_config(scale)
        config["metadata"]["covariance_growth_scale"] = float(scale)
    return config


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _finite_values(rows: Iterable[dict], field: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(field)
        if value in (None, ""):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            values.append(parsed)
    return np.asarray(values, dtype=float)


def _parse_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    return None


def _distribution(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _rms_delta(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(np.diff(values)))))


def _csv_value(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    return value


def compute_metrics(
    trial_rows: Sequence[dict],
    pose_rows: Sequence[dict],
    outcome: dict,
    runtime_stats: dict,
) -> dict:
    controller_times = _finite_values(trial_rows, "controller_computation_time")
    solver_times = _finite_values(trial_rows, "solver_computation_time")
    clearances = _finite_values(pose_rows, "sdf_value")
    v_values = _finite_values(trial_rows, "v")
    omega_values = _finite_values(trial_rows, "omega")

    controller_dist = _distribution(controller_times)
    solver_dist = _distribution(solver_times)
    deadline_miss_count = int(np.count_nonzero(controller_times > DEADLINE_SECONDS))
    deadline_miss_rate = (
        float(deadline_miss_count) / float(controller_times.size)
        if controller_times.size
        else float("nan")
    )

    solver_success_values = [
        parsed
        for parsed in (_parse_bool(row.get("solver_success")) for row in trial_rows)
        if parsed is not None
    ]
    solver_failure_count = sum(value is False for value in solver_success_values)
    solver_failure_rate = (
        float(solver_failure_count) / float(len(solver_success_values))
        if solver_success_values
        else float("nan")
    )

    clearance_q05 = (
        float(np.percentile(clearances, 5)) if clearances.size else float("nan")
    )
    clearance_min = float(np.min(clearances)) if clearances.size else float("nan")
    status = outcome.get("status", "error")
    final_time = outcome.get("final_time")

    return {
        "status": status,
        "failure_reason": outcome.get("failure_reason"),
        "goal_reached": bool(outcome.get("goal_reached", False)),
        "final_time": final_time,
        "driving_time": final_time if status == "success" else "",
        "distance_to_goal": outcome.get("distance_to_goal"),
        "step_count": int(len(trial_rows)),
        "controller_time_median": controller_dist["median"],
        "controller_time_p95": controller_dist["p95"],
        "controller_time_max": controller_dist["max"],
        "deadline_miss_count": deadline_miss_count,
        "deadline_miss_rate": deadline_miss_rate,
        "solver_time_median": solver_dist["median"],
        "solver_time_p95": solver_dist["p95"],
        "solver_time_max": solver_dist["max"],
        "clearance_q05": clearance_q05,
        "clearance_min": clearance_min,
        "collision": bool(math.isfinite(clearance_min) and clearance_min < 0.0),
        "rms_delta_v": _rms_delta(v_values),
        "rms_delta_omega": _rms_delta(omega_values),
        "solver_failure_count": int(solver_failure_count),
        "solver_failure_rate": solver_failure_rate,
        "safe_stop_count": int(runtime_stats.get("safe_stop_count", 0)),
    }


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


def load_summary(path: Path) -> List[dict]:
    return _read_csv(path)


def upsert_summary(rows: List[dict], new_row: dict) -> List[dict]:
    updated = [row for row in rows if row.get("run_id") != new_row.get("run_id")]
    updated.append(new_row)
    return updated


def normalize_trial_numbers(rows: Sequence[dict]) -> List[dict]:
    scale_trials = {
        f"mf_scale_{scale_token(scale)}": index
        for index, scale in enumerate(DEFAULT_SCALES, start=1)
    }
    comparison_trials = {
        f"compare_{optimizer_type}": index
        for index, optimizer_type in enumerate(
            ("psdf", "mpcc", "rmpcc_pv"), start=len(DEFAULT_SCALES) + 1
        )
    }
    trial_numbers = {**scale_trials, **comparison_trials}
    normalized = []
    for row in rows:
        normalized_row = dict(row)
        run_id = normalized_row.get("run_id")
        if run_id in trial_numbers:
            normalized_row["trial_number"] = trial_numbers[run_id]
        normalized.append(normalized_row)
    return normalized


def _optimizer_from_sim(test_sim: simulation_mpc):
    robot = getattr(test_sim, "robot", None)
    controller = getattr(robot, "_controller", None)
    return getattr(controller, "_optimizer", None)


def _cleanup_simulation(test_sim: simulation_mpc) -> None:
    optimizer = _optimizer_from_sim(test_sim)
    if optimizer is not None and hasattr(optimizer, "cleanup"):
        optimizer.cleanup()
    test_sim.sim = None
    test_sim.robot = None


def _raw_files(run_dir: Path, current_name: str) -> Dict[str, Path]:
    data_dir = run_dir / "data"
    return {
        "trial": data_dir / f"trial_history_{current_name}.csv",
        "pose": data_dir / f"pose_sdf_{current_name}.csv",
    }


def required_artifacts_exist(output_root: Path, row: dict) -> bool:
    run_id = row.get("run_id", "")
    if not run_id:
        return False
    run_dir = output_root / run_id
    trial_path = run_dir / row.get("trial_history_file", "")
    pose_path = run_dir / row.get("pose_sdf_file", "")
    return (
        (run_dir / "resolved_config.yaml").exists()
        and (run_dir / "run_metrics.csv").exists()
        and trial_path.exists()
        and pose_path.exists()
        and bool(_read_csv(trial_path))
        and bool(_read_csv(pose_path))
    )


def run_one(
    output_root: Path,
    run_id: str,
    trial_number: int,
    optimizer_type: str,
    simulation_time: float,
    scale: Optional[float] = None,
) -> dict:
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = build_run_config(run_dir, run_id, optimizer_type, simulation_time, scale)
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    print(
        f"[RUN {trial_number}] {run_id}: optimizer={optimizer_type}, "
        f"scale={scale if scale is not None else 'default'}",
        flush=True,
    )

    test_sim = simulation_mpc()
    error_text = ""
    with (run_dir / "run.log").open("w", encoding="utf-8") as log_stream:
        try:
            with redirect_stdout(log_stream), redirect_stderr(log_stream):
                test_sim.mpc_test(
                    maze_type="maze",
                    robot_shape="rectangle",
                    optimizer_type=optimizer_type,
                    dynamics_type="differential_drive",
                    path_planner="astar",
                    simulation_time=float(simulation_time),
                    config=config,
                )
        except Exception as exc:  # Preserve partial evidence for failed runs.
            error_text = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=log_stream)
            with redirect_stdout(log_stream), redirect_stderr(log_stream):
                try:
                    if getattr(test_sim, "sim", None) is not None:
                        test_sim.save_trial_history_to_csv(
                            f"trial_history_{getattr(test_sim, 'current_name', run_id)}"
                        )
                    if getattr(test_sim, "pose_sdf_data", None):
                        test_sim.save_pose_sdf_to_csv(
                            f"pose_sdf_{getattr(test_sim, 'current_name', run_id)}"
                        )
                except Exception:
                    traceback.print_exc(file=log_stream)

    current_name = getattr(test_sim, "current_name", run_id)
    raw_files = _raw_files(run_dir, current_name)
    trial_rows = _read_csv(raw_files["trial"])
    pose_rows = _read_csv(raw_files["pose"])
    outcome = dict(getattr(test_sim, "last_run_outcome", None) or {})
    if error_text:
        outcome.update(
            {
                "status": "error",
                "failure_reason": error_text,
                "goal_reached": False,
            }
        )

    optimizer = _optimizer_from_sim(test_sim)
    runtime_stats = {}
    if optimizer is not None and hasattr(optimizer, "get_runtime_stats"):
        try:
            runtime_stats = dict(optimizer.get_runtime_stats())
        except Exception:
            runtime_stats = {}

    metrics = compute_metrics(trial_rows, pose_rows, outcome, runtime_stats)
    raw_logs_complete = bool(trial_rows) and bool(pose_rows)
    row = {
        "run_id": run_id,
        "trial_number": int(trial_number),
        "method": METHOD_LABELS[optimizer_type],
        "optimizer_type": optimizer_type,
        "covariance_growth_scale": "" if scale is None else float(scale),
        **metrics,
        "raw_logs_complete": raw_logs_complete,
        "trial_history_file": str(raw_files["trial"].relative_to(run_dir)),
        "pose_sdf_file": str(raw_files["pose"].relative_to(run_dir)),
        "error": error_text,
    }
    _write_csv(run_dir / "run_metrics.csv", [row], SUMMARY_FIELDS)
    _cleanup_simulation(test_sim)

    print(
        f"[DONE {trial_number}] {run_id}: status={row['status']}, "
        f"raw_logs_complete={raw_logs_complete}, q05={_csv_value(row['clearance_q05'])}",
        flush=True,
    )
    return row


def run_one_isolated(
    output_root: Path,
    run_id: str,
    trial_number: int,
    optimizer_type: str,
    simulation_time: float,
    scale: Optional[float] = None,
) -> dict:
    """Run one trial in a fresh process so regenerated acados libraries cannot leak."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-root",
        str(output_root),
        "--simulation-time",
        str(float(simulation_time)),
        "--worker-run-id",
        run_id,
        "--worker-trial-number",
        str(int(trial_number)),
        "--worker-optimizer-type",
        optimizer_type,
    ]
    if scale is not None:
        command.extend(["--worker-scale", str(float(scale))])

    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    metrics_path = output_root / run_id / "run_metrics.csv"
    metrics_rows = _read_csv(metrics_path)
    if result.returncode == 0 and metrics_rows:
        return metrics_rows[0]

    return {
        "run_id": run_id,
        "trial_number": int(trial_number),
        "method": METHOD_LABELS[optimizer_type],
        "optimizer_type": optimizer_type,
        "covariance_growth_scale": "" if scale is None else float(scale),
        "status": "error",
        "failure_reason": f"worker_exit_{result.returncode}",
        "goal_reached": False,
        "raw_logs_complete": False,
        "error": f"worker exited with code {result.returncode}",
    }


def _float_from_row(row: dict, field: str, default: float) -> float:
    value = row.get(field)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def select_best_scale(rows: Sequence[dict]) -> Optional[dict]:
    candidates = []
    for row in rows:
        if row.get("optimizer_type") != "rmpcc":
            continue
        if row.get("status") != "success" or _parse_bool(row.get("collision")) is True:
            continue
        if _parse_bool(row.get("raw_logs_complete")) is not True:
            continue
        if not math.isfinite(_float_from_row(row, "clearance_q05", float("nan"))):
            continue
        candidates.append(row)

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda row: (
            -_float_from_row(row, "clearance_q05", -float("inf")),
            int(_float_from_row(row, "solver_failure_count", float("inf"))),
            int(_float_from_row(row, "safe_stop_count", float("inf"))),
            _float_from_row(row, "deadline_miss_rate", float("inf")),
            _float_from_row(row, "controller_time_p95", float("inf")),
            _float_from_row(row, "driving_time", float("inf")),
        ),
    )


def save_derived_outputs(output_root: Path, rows: Sequence[dict]) -> Optional[dict]:
    scale_rows = sorted(
        (row for row in rows if row.get("optimizer_type") == "rmpcc"),
        key=lambda row: _float_from_row(row, "covariance_growth_scale", float("inf")),
    )
    _write_csv(output_root / "scale_sweep.csv", scale_rows, SUMMARY_FIELDS)

    selected = select_best_scale(scale_rows)
    comparison_rows = [
        row
        for optimizer_type in ("psdf", "mpcc", "rmpcc_pv")
        for row in rows
        if row.get("optimizer_type") == optimizer_type
    ]
    if selected is not None:
        comparison_rows.append(selected)
    comparison_rows.sort(
        key=lambda row: ("psdf", "mpcc", "rmpcc_pv", "rmpcc").index(
            row["optimizer_type"]
        )
    )
    _write_csv(output_root / "optimizer_comparison.csv", comparison_rows, SUMMARY_FIELDS)

    selection_payload = {
        "selected": selected is not None,
        "covariance_growth_scale": (
            None if selected is None else float(selected["covariance_growth_scale"])
        ),
        "run_id": None if selected is None else selected["run_id"],
        "selection_order": [
            "success_and_no_collision",
            "clearance_q05_desc",
            "solver_failure_count_asc",
            "safe_stop_count_asc",
            "deadline_miss_rate_asc",
            "controller_time_p95_asc",
            "driving_time_asc",
        ],
    }
    with (output_root / "selected_scale.json").open("w", encoding="utf-8") as stream:
        json.dump(selection_payload, stream, ensure_ascii=False, indent=2)
    return selected


def _row_completed(output_root: Path, rows_by_id: Dict[str, dict], run_id: str) -> bool:
    row = rows_by_id.get(run_id)
    return row is not None and required_artifacts_exist(output_root, row)


def write_progress(output_root: Path, rows: Sequence[dict], selected: Optional[dict]) -> None:
    rows_by_id = {row.get("run_id"): row for row in rows}
    lines = [
        "# Goal 실행 체크리스트",
        "",
        "## 준비",
        "",
        "- [x] 벤치마크 실행기와 controller 전체 계산시간 계측 추가",
        "- [x] scale별 alpha/beta 실행 설정 계산 (`RMPCCOptimizerParam` 미수정)",
        "- [x] 계산시간·deadline·주행시간·clearance·입력 변화·실패 지표 집계",
        "- [x] 짧은 smoke test로 raw 로그와 요약 CSV 생성 확인",
        "",
        "## MF Scale Sweep",
        "",
    ]
    for trial_number, scale in enumerate(DEFAULT_SCALES, start=1):
        run_id = f"mf_scale_{scale_token(scale)}"
        checked = "x" if _row_completed(output_root, rows_by_id, run_id) else " "
        row = rows_by_id.get(run_id, {})
        status = f" — {row.get('status')}" if row else ""
        lines.append(
            f"- [{checked}] 시행 {trial_number} — PSDF-MPCC-CC-MF, "
            f"`covariance_growth_scale={scale:g}`{status}"
        )

    selected_text = ""
    if selected is not None:
        selected_text = f" — selected={float(selected['covariance_growth_scale']):g}"
    lines.extend(
        [
            f"- [{'x' if selected is not None else ' '}] 안전 우선 기준으로 최적 scale 선정{selected_text}",
            "",
            "## Optimizer 비교",
            "",
        ]
    )
    for trial_number, optimizer_type in enumerate(
        ("psdf", "mpcc", "rmpcc_pv"), start=len(DEFAULT_SCALES) + 1
    ):
        run_id = f"compare_{optimizer_type}"
        checked = "x" if _row_completed(output_root, rows_by_id, run_id) else " "
        row = rows_by_id.get(run_id, {})
        status = f" — {row.get('status')}" if row else ""
        lines.append(
            f"- [{checked}] 시행 {trial_number} — {METHOD_LABELS[optimizer_type]} "
            f"(`{optimizer_type}`){status}"
        )

    comparison_path = output_root / "optimizer_comparison.csv"
    comparison_complete = len(_read_csv(comparison_path)) == 4
    report_dir = output_root / "report"
    report_complete = (
        (report_dir / "artifact.json").exists()
        and (report_dir / "report.html").exists()
        and (report_dir / "source_notes.md").exists()
        and (report_dir / "validation_receipt.json").exists()
    )
    lines.extend(
        [
            f"- [{'x' if selected is not None else ' '}] 선정된 MF sweep 결과 재사용",
            f"- [{'x' if comparison_complete else ' '}] 4개 optimizer 비교표 생성",
            f"- [{'x' if report_complete else ' '}] 최적 scale 결론과 결과 요약 보고서 작성",
            "",
            "완료 표시는 raw trial/clearance 로그와 run summary가 모두 존재할 때만 갱신됩니다.",
        ]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_or_resume(
    output_root: Path,
    summary_rows: List[dict],
    run_id: str,
    trial_number: int,
    optimizer_type: str,
    simulation_time: float,
    resume: bool,
    scale: Optional[float] = None,
) -> List[dict]:
    rows_by_id = {row.get("run_id"): row for row in summary_rows}
    prior = rows_by_id.get(run_id)
    if resume and prior is not None and required_artifacts_exist(output_root, prior):
        print(f"[SKIP {trial_number}] {run_id}: complete artifacts found", flush=True)
        return summary_rows

    row = run_one_isolated(
        output_root=output_root,
        run_id=run_id,
        trial_number=trial_number,
        optimizer_type=optimizer_type,
        simulation_time=simulation_time,
        scale=scale,
    )
    summary_rows = upsert_summary(summary_rows, row)
    _write_csv(output_root / "all_runs.csv", summary_rows, SUMMARY_FIELDS)
    selected = save_derived_outputs(output_root, summary_rows)
    write_progress(output_root, summary_rows, selected)
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("sweep", "compare", "all"), default="all")
    parser.add_argument("--scales", nargs="+", type=float, default=list(DEFAULT_SCALES))
    parser.add_argument("--simulation-time", type=float, default=60.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one 0.3-scale MF trial for at most 0.3 simulated seconds",
    )
    parser.add_argument("--worker-run-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-trial-number", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-optimizer-type",
        choices=tuple(METHOD_LABELS),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-scale", type=float, help=argparse.SUPPRESS)
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if not args.output_root.is_absolute():
        args.output_root = PROJECT_ROOT / args.output_root
    if args.simulation_time <= 0.0:
        raise ValueError("--simulation-time must be positive")
    if args.smoke and args.worker_run_id is None:
        args.phase = "sweep"
        args.scales = [0.3]
        args.simulation_time = min(float(args.simulation_time), 0.3)
        args.output_root = args.output_root / "smoke"
        args.resume = False
    return args


def main() -> None:
    ensure_acados_runtime()
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.worker_run_id is not None:
        if args.worker_trial_number is None or args.worker_optimizer_type is None:
            raise ValueError("worker run requires trial number and optimizer type")
        run_one(
            output_root=args.output_root,
            run_id=args.worker_run_id,
            trial_number=args.worker_trial_number,
            optimizer_type=args.worker_optimizer_type,
            simulation_time=args.simulation_time,
            scale=args.worker_scale,
        )
        return

    summary_path = args.output_root / "all_runs.csv"
    summary_rows = normalize_trial_numbers(load_summary(summary_path))

    if args.phase in ("sweep", "all"):
        for scale in args.scales:
            if float(scale) not in DEFAULT_SCALES and not args.smoke:
                print(f"Warning: non-plan scale requested: {scale:g}")
            trial_number = (
                DEFAULT_SCALES.index(float(scale)) + 1
                if float(scale) in DEFAULT_SCALES
                else 0
            )
            summary_rows = _run_or_resume(
                output_root=args.output_root,
                summary_rows=summary_rows,
                run_id=f"mf_scale_{scale_token(scale)}",
                trial_number=trial_number,
                optimizer_type="rmpcc",
                simulation_time=args.simulation_time,
                resume=args.resume,
                scale=float(scale),
            )

    if args.phase in ("compare", "all"):
        for trial_number, optimizer_type in enumerate(
            ("psdf", "mpcc", "rmpcc_pv"), start=len(DEFAULT_SCALES) + 1
        ):
            summary_rows = _run_or_resume(
                output_root=args.output_root,
                summary_rows=summary_rows,
                run_id=f"compare_{optimizer_type}",
                trial_number=trial_number,
                optimizer_type=optimizer_type,
                simulation_time=args.simulation_time,
                resume=args.resume,
            )

    _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    selected = save_derived_outputs(args.output_root, summary_rows)
    write_progress(args.output_root, summary_rows, selected)
    if selected is None:
        print("No successful collision-free MF scale is available for selection.")
    else:
        print(
            "Selected covariance_growth_scale="
            f"{float(selected['covariance_growth_scale']):g} ({selected['run_id']})"
        )
    print(f"Progress: {args.output_root / 'progress.md'}")
    print(f"Scale summary: {args.output_root / 'scale_sweep.csv'}")
    print(f"Optimizer comparison: {args.output_root / 'optimizer_comparison.csv'}")


if __name__ == "__main__":
    main()
