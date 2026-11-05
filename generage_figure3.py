#!/usr/bin/env python3
"""Run and redraw the straight-corridor comparison used for Figure 3.

Redraw the archived 0.120 m corridor run (no simulation or acados required)::

    python generage_figure3.py --plot-only

Run a new experiment in the project's CUDA/acados Python environment::

    python generage_figure3.py --results-dir experiments/straight_corridor/new_run

Override the gap without changing the default map or covariance parameters::

    python generage_figure3.py --corridor-width 0.110 --results-dir experiments/straight_corridor/new_gap110

Redraw saved evidence without acados, torch, or a new simulation::

    python generage_figure3.py --plot-only --results-dir experiments/straight_corridor/width_130mm

The spelling of this filename follows the requested figure-generation entrypoint.
Edit STYLES, SPEED_X_LIMITS, BOTTLENECK_DISPLAY_S, plot_speed_profile(), and
plot_snapshots() to change presentation. The x limits crop the original s [m];
they do not normalize or otherwise transform the raw progress values. The
bottleneck display band is user-selected; map geometry and statistical
intervals continue to come from the archived map.json.
The single-column panels figure_3_a (speed) and figure_3_b (CC-MF snapshots)
share fixed canvas and plotting-area widths. The snapshot uses symmetric
horizontal margins to center the corridor within the panel. Edit the
PANEL_* constants for a different publication column width. Snapshot defaults
are the actual initial/final states plus five interior progress samples.
Simulation settings belong in config/config_figure3.yaml. Every run saves its
configuration, resolved parameters, map, raw logs, source snapshot, and outcomes.
No smoothing, resampling, success-step filtering, or artificial final stop is
applied to the plotted profiles.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ROOT / "experiments" / "straight_corridor"
DEFAULT_RESULTS_DIR = EXPERIMENT_ROOT / "width_120mm"
METHODS = ("mpcc", "cc_sf", "cc_mf")
SPEED_X_LIMITS = (0.0, 1.0)
BOTTLENECK_DISPLAY_S = (0.4, 0.6)
PANEL_WIDTH_IN = 3.5
SPEED_PANEL_HEIGHT_IN = 2.15
SNAPSHOT_VERTICAL_SCALE = 1.3
SNAPSHOT_PANEL_HEIGHT_IN = 0.82 * SNAPSHOT_VERTICAL_SCALE
SNAPSHOT_VIEW_HALF_HEIGHT_M = 0.20 * SNAPSHOT_VERTICAL_SCALE
PANEL_LEFT = 0.17
PANEL_RIGHT = 0.965
SNAPSHOT_LEFT = (1.0 - (PANEL_RIGHT - PANEL_LEFT)) / 2.0
SNAPSHOT_RIGHT = 1.0 - SNAPSHOT_LEFT
INTERIOR_SNAPSHOT_COUNT = 5
SNAPSHOT_METHOD = "cc_mf"
# Report anticipatory braking in the final 0.25 m before the front enters.
# This is distinct from the interval where the robot center is in the gap.
APPROACH_LENGTH = 0.25
STYLES = {
    "mpcc": {"label": "Det", "color": "#4C4C4C", "linestyle": "-"},
    "cc_sf": {"label": "CC-SF", "color": "#0072B2", "linestyle": "--"},
    "cc_mf": {"label": "CC-MF", "color": "#D55E00", "linestyle": "-."},
}


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


@contextmanager
def capture_log(path):
    """Capture both native acados output and Python prints."""
    sys.stdout.flush()
    sys.stderr.flush()
    descriptors = [os.dup(1), os.dup(2)]
    with path.open("w", encoding="utf-8", buffering=1) as log:
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            with redirect_stdout(log), redirect_stderr(log):
                yield
        finally:
            log.flush()
            for original, backup in zip((1, 2), descriptors):
                os.dup2(backup, original)
                os.close(backup)


def prepare_acados():
    """Set the Linux dynamic-loader path before importing acados."""
    source = Path(os.environ.get("ACADOS_SOURCE_DIR", ROOT.parent / "acados"))
    library = str(source / "lib")
    paths = list(filter(None, os.environ.get("LD_LIBRARY_PATH", "").split(":")))
    if sys.platform.startswith("linux") and library not in paths:
        environment = dict(os.environ)
        environment["ACADOS_SOURCE_DIR"] = str(source)
        environment["LD_LIBRARY_PATH"] = ":".join([library, *paths])
        os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)


def snapshot_sources(output):
    sources = [Path(__file__), ROOT / "test_nmpc.py", ROOT / "AGENTS.md"]
    for directory in ("control", "models", "planning", "sim", "utils", "config"):
        sources.extend(path for path in (ROOT / directory).rglob("*")
                       if path.is_file() and path.suffix in (".py", ".yaml"))
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(output / "source_hashes.json", hashes)


def run_experiment(output, config_path, corridor_width=None):
    import torch
    import yaml
    from sim.simulation_mpc import simulation_mpc
    from test_nmpc import cleanup_simulation

    if not torch.cuda.is_available():
        raise RuntimeError("The current controllers require the CUDA/acados environment.")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if corridor_width is not None:
        if not np.isfinite(corridor_width) or corridor_width <= 0:
            raise ValueError("Corridor width must be finite and positive.")
        config.setdefault("environment", {}).setdefault("straight_corridor", {})["width"] = corridor_width
    if [item["optimizer_type"] for item in config["test_configs"]] != list(METHODS):
        raise ValueError("Figure 3 requires mpcc, cc_sf, cc_mf in that order.")
    expected = {"maze_type": "straight_corridor", "robot_shape": "rectangle",
                "dynamics_type": "differential_drive", "path_planner": "astar"}
    for item in config["test_configs"]:
        if any(item.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Figure 3 requires {expected}; got {item}")
        if set(item) - {*expected, "optimizer_type"}:
            raise ValueError("Put overrides in the top-level mpcc / method blocks.")
    if config.get("trials", 1) != 1:
        raise ValueError("This illustrative comparison uses one run per method.")
    if any(config.get(key, {}).get("enabled", False)
           for key in ("start_perturbation", "localization_error")):
        raise ValueError("This nominal straight-line comparison requires noise/perturbation disabled.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    snapshot_sources(output)
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    scene = simulation_mpc()
    start, goal, grid, obstacles = scene.create_env("straight_corridor", config=config)
    map_data = {
        "start": start, "goal": goal, "bounds": grid[0], "cell_size": grid[1],
        "obstacles": [obstacle.get_ccw_vertices() for obstacle in obstacles],
        "bottleneck_x": [obstacles[-2].left, obstacles[-2].right],
        "bottleneck_y": [obstacles[-2].up, obstacles[-1].down],
        "robot_length": config["rectangle_geometry"]["length"],
        "robot_width": config["rectangle_geometry"]["width"],
    }
    write_json(output / "map.json", map_data)
    manifest = {
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "methods": METHODS, "runs_per_method": 1,
        "progress": "path_s: optimizer current projected progress, before applying u_k",
        "speed": "v: actual applied forward command u_k[0], paired by timestep",
        "noise": "No sampled plant/localization noise; CC prediction uses configured covariance.",
        "outcomes": {},
    }
    write_json(output / "manifest.json", manifest)
    for method in METHODS:
        run_dir = output / method
        run_dir.mkdir()
        run_config = copy.deepcopy(config)
        run_config["output_root_dir"] = str(run_dir)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
        np.random.seed(0)
        torch.manual_seed(0)
        simulation = simulation_mpc()
        outcome = {}
        interrupted = False
        print(f"Running {STYLES[method]['label']} ... ({run_dir / 'run.log'})", flush=True)
        with capture_log(run_dir / "run.log"):
            try:
                simulation.mpc_test(
                    "straight_corridor", "rectangle", method,
                    "differential_drive", "astar",
                    float(config["defaults"]["simulation_time"]), run_config,
                )
                outcome = dict(simulation.last_run_outcome or {})
            except KeyboardInterrupt:
                interrupted = True
                outcome = {"status": "interrupted", "failure_reason": "keyboard_interrupt"}
            except Exception as exc:
                traceback.print_exc()
                outcome = {"status": "error", "failure_reason": f"{type(exc).__name__}: {exc}"}
            finally:
                try:
                    if simulation.robot is not None:
                        controller = getattr(simulation.robot, "_controller", None)
                        params = getattr(controller, "_param", None)
                        if params is not None:
                            write_json(run_dir / "parameters.json", vars(params))
                        path = getattr(simulation.robot, "_global_path", None)
                        if path is not None:
                            write_json(run_dir / "global_path.json", path)
                    if simulation.sim is not None:
                        simulation.save_trial_history_to_csv("trial_history")
                        simulation.save_pose_sdf_to_csv("pose_sdf")
                    write_json(run_dir / "outcome.json", outcome)
                finally:
                    cleanup_simulation(simulation)
        manifest["outcomes"][method] = outcome
        write_json(output / "manifest.json", manifest)
        print(f"{STYLES[method]['label']}: {outcome.get('status')} "
              f"({outcome.get('final_time', '?')} simulation seconds)", flush=True)
        if interrupted:
            raise KeyboardInterrupt


def load_profiles(output, require_success=True, export_path=None):
    import pandas as pd

    profiles = {}
    for method in METHODS:
        outcome_file = output / method / "outcome.json"
        if not outcome_file.is_file():
            raise RuntimeError(f"{method}: run has not completed; inspect {output / method / 'run.log'}")
        outcome = json.loads(outcome_file.read_text())
        if require_success and outcome.get("status") != "success":
            raise RuntimeError(
                f"Cannot produce a completed-run comparison: {method} "
                f"status={outcome.get('status')}, reason={outcome.get('failure_reason')}. "
                f"Raw evidence is preserved in {output / method}."
            )
        folder = output / method / "data"
        history = pd.read_csv(folder / "trial_history.csv")
        pose = pd.read_csv(folder / "pose_sdf.csv")
        if history.empty or len(history) != len(pose) or not np.array_equal(history.timestep, pose.timestep):
            raise ValueError(f"{method}: state/control and progress logs are not aligned")
        history = history.rename(columns={"x": "post_x", "y": "post_y", "theta": "post_theta"})
        frame = pose.merge(history, on="timestep", validate="one_to_one", sort=False)
        if not np.isfinite(frame[["path_s", "v", "x", "y", "theta"]].to_numpy()).all():
            raise ValueError(f"{method}: non-finite progress, speed, or pose")
        frame.insert(0, "method", method)
        profiles[method] = frame
    if export_path is not None:
        pd.concat(profiles.values(), ignore_index=True).to_csv(export_path, index=False)
    return profiles


def rectangle_vertices(pose, length, width):
    x, y, theta = pose
    local = np.array([[-length, -width], [length, -width],
                      [length, width], [-length, width]]) / 2
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    return local @ rotation.T + [x, y]


def summarize(output, profiles, scene, export_path=None):
    """Keep solver failures and geometric safety separate from the speed effect."""
    from shapely.geometry import Polygon

    start_x = scene["start"][0]
    low, high = np.array(scene["bottleneck_x"]) - start_x
    front_entry_s = low - scene["robot_length"] / 2
    approach_start_s = max(0.0, front_entry_s - APPROACH_LENGTH)
    obstacles = [Polygon(vertices) for vertices in scene["obstacles"]]
    rows = []
    for method, frame in profiles.items():
        inside = frame.path_s.between(low, high)
        approach = frame.path_s.between(approach_start_s, front_entry_s)
        overlap = frame.path_s.between(low - scene["robot_length"] / 2,
                                       high + scene["robot_length"] / 2)
        poses = np.vstack([frame[["x", "y", "theta"]].to_numpy(),
                           frame[["post_x", "post_y", "post_theta"]].iloc[-1].to_numpy()])
        clearances = []
        collision_samples = 0
        for pose in poses:
            robot = Polygon(rectangle_vertices(pose, scene["robot_length"], scene["robot_width"]))
            clearances.append(min(robot.distance(obstacle) for obstacle in obstacles))
            collision_samples += any(robot.intersection(obstacle).area > 1e-12
                                     for obstacle in obstacles)
        outcome = json.loads((output / method / "outcome.json").read_text())
        rows.append({
            "method": method, "status": outcome.get("status"),
            "failure_reason": outcome.get("failure_reason"),
            "final_time_s": outcome.get("final_time"),
            "control_steps": len(frame),
            "unsuccessful_solver_steps": int((frame.solver_success.astype(str).str.lower() != "true").sum()),
            "safe_stop_count": outcome.get("safe_stop_count", 0),
            "peak_speed_mps": float(frame.v.max()),
            "minimum_applied_speed_mps": float(frame.v.min()),
            "reverse_command_steps": int((frame.v < -1e-6).sum()),
            "bottleneck_mean_speed_mps": float(frame.loc[inside, "v"].mean()),
            "bottleneck_min_speed_mps": float(frame.loc[inside, "v"].min()),
            "approach_start_s_m": approach_start_s,
            "approach_end_s_m": front_entry_s,
            "approach_mean_speed_mps": float(frame.loc[approach, "v"].mean()),
            "approach_min_speed_mps": float(frame.loc[approach, "v"].min()),
            "footprint_overlap_mean_speed_mps": float(frame.loc[overlap, "v"].mean()),
            "footprint_overlap_min_speed_mps": float(frame.loc[overlap, "v"].min()),
            "minimum_sampled_clearance_m": min(clearances),
            "collision_samples": collision_samples,
            "max_centerline_deviation_m": float(np.max(np.abs(poses[:, 1] - scene["start"][1]))),
            "mf_max_nominal_predicted_probability_sum": None,
            "mf_max_chance_slack_quantile": None,
        })
        if method == "cc_mf":
            cycle_files = list((output / method / "data").glob("cc_mf_cycle_*.csv"))
            if len(cycle_files) == 1:
                with cycle_files[0].open(newline="", encoding="utf-8") as handle:
                    cycles = list(csv.DictReader(handle))
                for field, target in (
                    ("mf_probability_sum", "mf_max_nominal_predicted_probability_sum"),
                    ("mf_slack", "mf_max_chance_slack_quantile"),
                ):
                    values = [value for cycle in cycles for value in json.loads(cycle[field])
                              if value is not None and np.isfinite(value)]
                    rows[-1][target] = max(values) if values else None
    baseline = rows[0]["bottleneck_mean_speed_mps"]
    for row in rows:
        comparable = row["status"] == rows[0]["status"] == "success"
        row["bottleneck_speed_reduction_vs_mpcc_pct"] = (
            100 * (1 - row["bottleneck_mean_speed_mps"] / baseline)
            if comparable else None)
        row["approach_speed_reduction_vs_mpcc_pct"] = (
            100 * (1 - row["approach_mean_speed_mps"] / rows[0]["approach_mean_speed_mps"])
            if comparable else None)
    rows = json_value(rows)  # A failed run may never reach a reporting interval.
    if export_path is not None:
        with Path(export_path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def compare_saved_runs(result_directories, destination):
    """Export all tested widths, including failures, without redrawing figures.

    The complete-run figures still require all three methods to succeed.
    Failed runs remain labeled in this table; unvisited intervals are blank.
    """
    rows = []
    for directory in result_directories:
        directory = Path(directory).resolve()
        scene = json.loads((directory / "map.json").read_text())
        profiles = load_profiles(directory, require_success=False)
        for row in summarize(directory, profiles, scene):
            rows.append({
                "corridor_width_m": scene["bottleneck_y"][1] - scene["bottleneck_y"][0],
                "robot_width_m": scene["robot_width"],
                "results_directory": str(directory),
                **row,
            })
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def save_figure(fig, folder, name, formats, dpi):
    for extension in formats:
        # Tight cropping changes the two panels' physical and pixel widths.
        fig.savefig(folder / f"{name}.{extension}", dpi=dpi, bbox_inches=None)


def plot_speed_profile(profiles, scene, folder, formats, dpi):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(PANEL_WIDTH_IN, SPEED_PANEL_HEIGHT_IN))
    ax = fig.add_axes([PANEL_LEFT, 0.24, PANEL_RIGHT - PANEL_LEFT, 0.715])
    low, high = BOTTLENECK_DISPLAY_S
    ax.axvspan(low, high, color="0.90", zorder=0)
    ax.text((low + high) / 2, 0.96, "Bottleneck", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=8)
    for method, frame in profiles.items():
        ax.plot(frame.path_s, frame.v, linewidth=1.15, **STYLES[method])
        failed = frame.solver_success.astype(str).str.lower() != "true"
        if failed.any():
            ax.scatter(frame.loc[failed, "path_s"], frame.loc[failed, "v"],
                       marker="x", s=12, color=STYLES[method]["color"], zorder=5)
    ax.set_xlabel(r"Progress variable $s$ [m]")
    ax.set_ylabel(r"Speed $v$ [m/s]")
    ax.set_xlim(*SPEED_X_LIMITS)
    minimum = min(frame.v.min() for frame in profiles.values())
    maximum = max(frame.v.max() for frame in profiles.values())
    ax.set_ylim(min(0, minimum * 1.1), maximum * 1.15)
    ax.legend(loc="lower right", frameon=False, ncol=1, fontsize=8)
    ax.grid(axis="both", color="0.84", linestyle=":", linewidth=0.5)
    ax.tick_params(width=0.6, length=3, pad=3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, folder, "figure_3_a", formats, dpi)
    plt.close(fig)


def select_snapshot_poses(frame, snapshot_progress):
    """Select actual start/end states and distinct interior progress samples."""
    first_s, last_s = float(frame.path_s.iloc[0]), float(frame.path_s.iloc[-1])
    if snapshot_progress is None:
        targets = np.linspace(first_s, last_s, INTERIOR_SNAPSHOT_COUNT + 2)[1:-1]
    else:
        targets = np.asarray(snapshot_progress, dtype=float)
    if any(not np.isfinite(value) or not first_s < value < last_s for value in targets):
        raise ValueError(f"Interior snapshots must lie strictly between {first_s} and {last_s} m.")
    dt = float(frame.time.iloc[0])
    selected = []

    def pre_state(index, phase, requested_s):
        row = frame.iloc[index]
        return {"method": SNAPSHOT_METHOD, "phase": phase,
                "state_source": "pose_sdf", "requested_s": requested_s,
                "timestep": int(row.timestep), "path_s": float(row.path_s),
                "time_s": float(row.time - dt), "x": float(row.x),
                "y": float(row.y), "theta": float(row.theta)}

    selected.append(pre_state(0, "start", first_s))
    indices = [int(np.argmin(np.abs(frame.path_s.to_numpy() - target))) for target in targets]
    if len(set(indices)) != len(indices) or any(index == 0 for index in indices):
        raise ValueError("Requested snapshots do not select distinct interior states.")
    selected.extend(pre_state(index, "intermediate", float(target))
                    for target, index in zip(targets, indices))
    last = frame.iloc[-1]
    # The final plant state is after the last control interval; no optimizer s
    # was logged for this state, so its path_s is deliberately left blank.
    selected.append({"method": SNAPSHOT_METHOD, "phase": "end",
                     "state_source": "trial_history_post_state", "requested_s": None,
                     "timestep": int(last.timestep) + 1, "path_s": None,
                     "time_s": float(last.time), "x": float(last.post_x),
                     "y": float(last.post_y), "theta": float(last.post_theta)})
    return selected


def plot_snapshots(profiles, scene, folder, formats, dpi, snapshot_progress):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(PANEL_WIDTH_IN, SNAPSHOT_PANEL_HEIGHT_IN))
    ax = fig.add_axes([SNAPSHOT_LEFT, 0.04, SNAPSHOT_RIGHT - SNAPSHOT_LEFT, 0.92])
    center_x = float(np.mean(scene["bottleneck_x"]))
    center_y = float(np.mean(scene["bottleneck_y"]))
    # Equal-scale central map strip; true obstacle coordinates are clipped by axes.
    bounds = np.asarray(scene["bounds"], dtype=float)
    for vertices in scene["obstacles"]:
        polygon = np.asarray(vertices, dtype=float)
        lower, upper = polygon.min(axis=0), polygon.max(axis=0)
        spans_map_height = np.isclose(lower[1], bounds[0, 1]) and np.isclose(upper[1], bounds[1, 1])
        touches_side = np.isclose(lower[0], bounds[0, 0]) or np.isclose(upper[0], bounds[1, 0])
        if spans_map_height and touches_side:
            continue  # Hide the left/right boundary walls in this panel only.
        ax.add_patch(Polygon(polygon, facecolor="0.78", edgecolor="0.35",
                             linewidth=0.6, zorder=1, clip_on=True))
    frame = profiles[SNAPSHOT_METHOD]
    snapshot_rows = select_snapshot_poses(frame, snapshot_progress)
    for row in snapshot_rows:
        vertices = rectangle_vertices([row["x"], row["y"], row["theta"]],
                                      scene["robot_length"], scene["robot_width"])
        outline_style = "-" if row["phase"] in ("start", "end") else ":"
        ax.add_patch(Polygon(vertices, facecolor="none", edgecolor="black",
                             linewidth=0.8, linestyle=outline_style,
                             capstyle="round", zorder=3))
    x = np.r_[frame.x.to_numpy(), frame.post_x.iloc[-1]]
    y = np.r_[frame.y.to_numpy(), frame.post_y.iloc[-1]]
    ax.plot(x, y, color="#0066CC", linestyle=":", linewidth=1.2,
            dash_capstyle="round", zorder=4)
    ax.scatter(snapshot_rows[0]["x"], snapshot_rows[0]["y"], marker="o",
               s=13, facecolors="white", edgecolors="black", linewidths=0.7, zorder=5)
    ax.scatter(*scene["goal"][:2], marker="*", s=27, color="black", zorder=5)
    handles = [
        Line2D([], [], color="black", marker="o", markersize=3.3,
               markerfacecolor="white", markeredgewidth=0.7, linestyle="none", label="Start"),
        Line2D([], [], color="black", marker="*", markersize=4.6,
               linestyle="none", label="Goal"),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=2, frameon=False,
              fontsize=6.5, handlelength=0.9, handletextpad=0.3,
              columnspacing=0.8, borderaxespad=0.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylim(center_y - SNAPSHOT_VIEW_HALF_HEIGHT_M,
                center_y + SNAPSHOT_VIEW_HALF_HEIGHT_M)
    half_width = (bounds[1, 0] - bounds[0, 0]) / 2.0
    ax.set_xlim(center_x - half_width, center_x + half_width)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.65)
    save_figure(fig, folder, "figure_3_b", formats, dpi)
    plt.close(fig)
    with (folder / "snapshot_poses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "method", "phase", "state_source", "requested_s", "timestep",
            "path_s", "time_s", "x", "y", "theta",
        ])
        writer.writeheader()
        writer.writerows(snapshot_rows)
    return snapshot_rows


def generate_figures(output, folder, formats, dpi, snapshot_progress):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "font.size": 8.5, "mathtext.fontset": "dejavuserif",
                         "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.bbox": None, "figure.autolayout": False,
                         "figure.constrained_layout.use": False})
    folder.mkdir(parents=True, exist_ok=True)
    scene = json.loads((output / "map.json").read_text())
    # Redrawing is read-only with respect to the archived experiment directory.
    profiles = load_profiles(output, export_path=folder / "profiles.csv")
    rows = summarize(output, profiles, scene, export_path=folder / "summary.csv")
    plot_speed_profile(profiles, scene, folder, formats, dpi)
    snapshot_rows = plot_snapshots(profiles, scene, folder, formats, dpi, snapshot_progress)
    write_json(folder / "figure3_sources.json", {
        "results_directory": output, "map": scene, "summary": rows,
        "snapshot_method": SNAPSHOT_METHOD,
        "snapshot_requested_interior_s_m": snapshot_progress,
        "snapshot_poses": snapshot_rows,
        "panels": {
            "figure_3_a": {"width_in": PANEL_WIDTH_IN, "height_in": SPEED_PANEL_HEIGHT_IN},
            "figure_3_b": {"width_in": PANEL_WIDTH_IN, "height_in": SNAPSHOT_PANEL_HEIGHT_IN},
            "snapshot_vertical_scale": SNAPSHOT_VERTICAL_SCALE,
            "speed_axes_left_fraction": PANEL_LEFT, "speed_axes_right_fraction": PANEL_RIGHT,
            "snapshot_axes_left_fraction": SNAPSHOT_LEFT, "snapshot_axes_right_fraction": SNAPSHOT_RIGHT,
            "snapshot_center_m": [float(np.mean(scene["bottleneck_x"])), float(np.mean(scene["bottleneck_y"]))],
            "snapshot_style": "centered rectangular map frame; black solid start/end robots and five black dotted interior robots; blue dotted CC-MF trajectory; Start circle and Goal star; no ticks, axis/controller labels, or filled side walls",
            "goal_marker_definition": "configured goal position; the end robot remains at the actual last post-state",
        },
        "method_display_labels": {method: style["label"] for method, style in STYLES.items()},
        "speed_plot_view": {
            "x_limits_m": SPEED_X_LIMITS,
            "bottleneck_display_s_m": BOTTLENECK_DISPLAY_S,
            "physical_bottleneck_s_m": (
                np.array(scene["bottleneck_x"]) - scene["start"][0]
            ),
            "progress_transform": "none; display limits crop the original progress in meters",
            "band_definition": "User-selected display annotation; does not change physical geometry or statistics.",
        },
        "speed_alignment": "pose_sdf.path_s[k] vs trial_history.v[k]; all applied steps",
        "bottleneck_statistics": "Arithmetic time-sample mean with robot center s in the bottleneck; dt=0.1s",
        "approach_statistics": (
            f"Arithmetic time-sample mean over final {APPROACH_LENGTH} m before "
            "the robot front enters the bottleneck; see summary for exact s bounds."
        ),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "caveats": [
            "One deterministic nominal run per method; no empirical probability-of-collision claim.",
            "CC-SF uses a hard scalar chance row; CC-MF uses a hard geometric guard and a soft chance row.",
            "Three methods illustrate formulation behavior; causal isolation of control dependence requires an ablation.",
            "Snapshot axes show a cropped central map strip at equal spatial scale.",
            "Collision check is sampled, not a continuous swept-volume certificate.",
            "Terminal slowing is part of the raw run; zero-speed endpoint is not appended.",
        ],
    })
    print(f"Figures saved to {folder}")
    for row in rows:
        print(f"{STYLES[row['method']]['label']}: approach mean "
              f"{row['approach_mean_speed_mps']:.4f} m/s, bottleneck mean "
              f"{row['bottleneck_mean_speed_mps']:.4f} m/s, "
              f"solver failures {row['unsuccessful_solver_steps']}, "
              f"minimum sampled clearance {1000 * row['minimum_sampled_clearance_m']:.2f} mm")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config/config_figure3.yaml")
    parser.add_argument("--results-dir", type=Path,
                        help="Fresh directory for a run; --plot-only defaults to experiments/straight_corridor/width_120mm")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures/figure3")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--corridor-width", type=float,
                        help="Override the bottleneck gap in meters for a new run")
    parser.add_argument("--compare-runs", nargs="+", type=Path,
                        help="Only summarize archived runs, including failures")
    parser.add_argument("--comparison-csv", type=Path,
                        default=ROOT / "figures/figure3/width_comparison.csv")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg", "eps"), default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--snapshot-progress", nargs="+", type=float, default=None,
                        help="Interior snapshot progress values in meters; default: five equally spaced values. Actual start/end states are always included.")
    args = parser.parse_args()
    if args.compare_runs:
        if args.plot_only or args.results_dir is not None or args.corridor_width is not None:
            parser.error("--compare-runs cannot be combined with run/plot options")
        compare_saved_runs(args.compare_runs, args.comparison_csv)
        print(f"Width comparison saved to {args.comparison_csv.resolve()}")
        return
    if args.plot_only and args.corridor_width is not None:
        parser.error("--corridor-width changes experiments; --plot-only uses the archived map")
    output = args.results_dir
    if output is None:
        output = DEFAULT_RESULTS_DIR if args.plot_only else EXPERIMENT_ROOT / (
            "run_" + datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S"))
    output = output.resolve()
    if not args.plot_only:
        prepare_acados()
        run_experiment(output, args.config.resolve(), args.corridor_width)
    generate_figures(output, args.figure_dir.resolve(), args.formats,
                     args.dpi, args.snapshot_progress)


if __name__ == "__main__":
    main()
