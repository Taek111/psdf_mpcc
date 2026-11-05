#!/usr/bin/env python3
"""Run and redraw the straight-corridor comparison used for Figure 2.

Run in the project's CUDA/acados Python environment::

    python generage_figure2.py --results-dir benchmark_runs/figure2_run1

Override the gap without changing the default map or covariance parameters::

    python generage_figure2.py --corridor-width 0.110 --results-dir benchmark_runs/figure2_gap110

Redraw saved evidence without acados, torch, or a new simulation::

    python generage_figure2.py --plot-only --results-dir benchmark_runs/figure2_run1

The spelling of this filename follows the requested figure-generation entrypoint.
Edit STYLES, plot_speed_profile(), and plot_snapshots() to change presentation.
Simulation settings belong in config/config_figure2.yaml. Every run saves its
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
METHODS = ("mpcc", "cc_sf", "cc_mf")
STYLES = {
    "mpcc": {"label": "MPCC", "color": "#4C4C4C", "linestyle": "-"},
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
        raise ValueError("Figure 2 requires mpcc, cc_sf, cc_mf in that order.")
    expected = {"maze_type": "straight_corridor", "robot_shape": "rectangle",
                "dynamics_type": "differential_drive", "path_planner": "astar"}
    for item in config["test_configs"]:
        if any(item.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Figure 2 requires {expected}; got {item}")
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


def load_profiles(output):
    import pandas as pd

    profiles = {}
    for method in METHODS:
        outcome_file = output / method / "outcome.json"
        if not outcome_file.is_file():
            raise RuntimeError(f"{method}: run has not completed; inspect {output / method / 'run.log'}")
        outcome = json.loads(outcome_file.read_text())
        if outcome.get("status") != "success":
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
    pd.concat(profiles.values(), ignore_index=True).to_csv(output / "profiles.csv", index=False)
    return profiles


def rectangle_vertices(pose, length, width):
    x, y, theta = pose
    local = np.array([[-length, -width], [length, -width],
                      [length, width], [-length, width]]) / 2
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    return local @ rotation.T + [x, y]


def summarize(output, profiles, scene):
    """Keep solver failures and geometric safety separate from the speed effect."""
    from shapely.geometry import Polygon

    start_x = scene["start"][0]
    low, high = np.array(scene["bottleneck_x"]) - start_x
    obstacles = [Polygon(vertices) for vertices in scene["obstacles"]]
    rows = []
    for method, frame in profiles.items():
        inside = frame.path_s.between(low, high)
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
            "final_time_s": outcome.get("final_time"),
            "control_steps": len(frame),
            "unsuccessful_solver_steps": int((frame.solver_success.astype(str).str.lower() != "true").sum()),
            "safe_stop_count": outcome.get("safe_stop_count", 0),
            "peak_speed_mps": float(frame.v.max()),
            "bottleneck_mean_speed_mps": float(frame.loc[inside, "v"].mean()),
            "bottleneck_min_speed_mps": float(frame.loc[inside, "v"].min()),
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
        row["bottleneck_speed_reduction_vs_mpcc_pct"] = 100 * (
            1 - row["bottleneck_mean_speed_mps"] / baseline)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def save_figure(fig, folder, name, formats, dpi):
    for extension in formats:
        fig.savefig(folder / f"{name}.{extension}", dpi=dpi, bbox_inches="tight", pad_inches=0.035)


def plot_speed_profile(profiles, scene, folder, formats, dpi):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 2.8), layout="constrained")
    low, high = np.array(scene["bottleneck_x"]) - scene["start"][0]
    ax.axvspan(low, high, color="0.90", zorder=0)
    ax.text((low + high) / 2, 0.96, "Bottleneck", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=9)
    for method, frame in profiles.items():
        ax.plot(frame.path_s, frame.v, linewidth=1.65, **STYLES[method])
        failed = frame.solver_success.astype(str).str.lower() != "true"
        if failed.any():
            ax.scatter(frame.loc[failed, "path_s"], frame.loc[failed, "v"],
                       marker="x", s=18, color=STYLES[method]["color"], zorder=5)
    ax.set_xlabel(r"Progress variable $s$ [m]")
    ax.set_ylabel(r"Speed $v$ [m/s]")
    ax.set_xlim(0, scene["goal"][0] - scene["start"][0])
    minimum = min(frame.v.min() for frame in profiles.values())
    maximum = max(frame.v.max() for frame in profiles.values())
    ax.set_ylim(min(0, minimum * 1.1), maximum * 1.15)
    ax.legend(loc="upper right", frameon=False, ncol=1)
    ax.grid(axis="both", color="0.84", linestyle=":", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, folder, "figure2_speed_profile", formats, dpi)
    plt.close(fig)


def plot_snapshots(profiles, scene, folder, formats, dpi, snapshot_progress):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 5.4), sharex=True, sharey=True,
                             layout="constrained")
    center_y = scene["start"][1]
    # Equal-scale central map strip; true obstacle coordinates are clipped by axes.
    y_limits = (center_y - 0.20, center_y + 0.20)
    snapshot_rows = []
    for ax, (method, frame) in zip(axes, profiles.items()):
        style = STYLES[method]
        for vertices in scene["obstacles"]:
            ax.add_patch(Polygon(vertices, facecolor="0.78", edgecolor="0.35",
                                 linewidth=0.7, zorder=1, clip_on=True))
        ax.plot([scene["start"][0], scene["goal"][0]], [center_y, center_y],
                color="0.6", linestyle=":", linewidth=0.8, zorder=2)
        for target in snapshot_progress:
            if target > frame.path_s.max() + 0.02:
                continue
            index = int(np.argmin(np.abs(frame.path_s.to_numpy() - target)))
            row = frame.iloc[index]
            vertices = rectangle_vertices(row[["x", "y", "theta"]].to_numpy(dtype=float),
                                          scene["robot_length"], scene["robot_width"])
            ax.add_patch(Polygon(vertices, facecolor="none", edgecolor=style["color"],
                                 linewidth=0.9, zorder=3))
            snapshot_rows.append({"method": method, "requested_s": target,
                                  "timestep": int(row.timestep), "path_s": row.path_s,
                                  "time_s": row.time - (frame.time.iloc[0]),
                                  "x": row.x, "y": row.y, "theta": row.theta})
        x = np.r_[frame.x.to_numpy(), frame.post_x.iloc[-1]]
        y = np.r_[frame.y.to_numpy(), frame.post_y.iloc[-1]]
        ax.plot(x, y, linewidth=1.5, **style, zorder=4)
        ax.scatter(*scene["start"][:2], marker="o", s=17, facecolors="white",
                   edgecolors="0.2", linewidths=0.8, zorder=5)
        ax.scatter(*scene["goal"][:2], marker="*", s=42, color="0.2", zorder=5)
        ax.text(0.015, 0.88, style["label"], transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=style["color"])
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylim(*y_limits)
        ax.set_xlim(scene["bounds"][0][0], scene["bounds"][1][0])
        ax.set_yticks([0.35, 0.50, 0.65])
        ax.set_ylabel(r"$y$ [m]")
    low_y, high_y = scene["bottleneck_y"]
    arrow_x = scene["bottleneck_x"][1] + 0.045
    axes[0].annotate("", xy=(arrow_x, low_y), xytext=(arrow_x, high_y),
                     arrowprops={"arrowstyle": "<->", "linewidth": 0.8, "shrinkA": 0, "shrinkB": 0})
    axes[0].text(arrow_x + 0.025, center_y + 0.035,
                 f"{high_y - low_y:.3f} m", fontsize=9, va="center")
    axes[-1].set_xlabel(r"$x$ [m]")
    handles = [Line2D([], [], color="0.25", marker="o", markerfacecolor="white", linestyle="none", label="Start"),
               Line2D([], [], color="0.25", marker="*", linestyle="none", label="Goal")]
    axes[0].legend(handles=handles, loc="upper right", ncol=2, frameon=False,
                   fontsize=8, handletextpad=0.3, columnspacing=0.9)
    save_figure(fig, folder, "figure2_map_snapshots", formats, dpi)
    plt.close(fig)
    with (folder / "snapshot_poses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "method", "requested_s", "timestep", "path_s", "time_s", "x", "y", "theta",
        ])
        writer.writeheader()
        writer.writerows(snapshot_rows)


def generate_figures(output, folder, formats, dpi, snapshot_progress):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "font.size": 10, "mathtext.fontset": "dejavuserif",
                         "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42})
    folder.mkdir(parents=True, exist_ok=True)
    scene = json.loads((output / "map.json").read_text())
    length = scene["goal"][0] - scene["start"][0]
    if any(not np.isfinite(value) or not 0 <= value <= length for value in snapshot_progress):
        raise ValueError(f"Snapshot progress must lie within [0, {length}] m.")
    profiles = load_profiles(output)
    rows = summarize(output, profiles, scene)
    plot_speed_profile(profiles, scene, folder, formats, dpi)
    plot_snapshots(profiles, scene, folder, formats, dpi, snapshot_progress)
    write_json(folder / "figure2_sources.json", {
        "results_directory": output, "map": scene, "summary": rows,
        "snapshot_progress_m": snapshot_progress,
        "speed_alignment": "pose_sdf.path_s[k] vs trial_history.v[k]; all applied steps",
        "bottleneck_statistics": "Arithmetic time-sample mean with robot center s in the bottleneck; dt=0.1s",
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
        print(f"{STYLES[row['method']]['label']}: bottleneck mean "
              f"{row['bottleneck_mean_speed_mps']:.4f} m/s, "
              f"solver failures {row['unsuccessful_solver_steps']}, "
              f"minimum sampled clearance {1000 * row['minimum_sampled_clearance_m']:.2f} mm")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config/config_figure2.yaml")
    parser.add_argument("--results-dir", type=Path, help="Fresh directory for a run; existing directory for --plot-only")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures/figure2")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--corridor-width", type=float,
                        help="Override the bottleneck gap in meters for a new run")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg", "eps"), default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--snapshot-progress", nargs="+", type=float, default=[0.0, 0.30, 0.65, 1.0, 1.26])
    args = parser.parse_args()
    if args.plot_only and args.results_dir is None:
        parser.error("--plot-only requires --results-dir")
    if args.plot_only and args.corridor_width is not None:
        parser.error("--corridor-width changes experiments; --plot-only uses the archived map")
    output = args.results_dir or ROOT / "benchmark_runs" / (
        "figure2_" + datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S"))
    output = output.resolve()
    if not args.plot_only:
        prepare_acados()
        run_experiment(output, args.config.resolve(), args.corridor_width)
    generate_figures(output, args.figure_dir.resolve(), args.formats,
                     args.dpi, args.snapshot_progress)


if __name__ == "__main__":
    main()
