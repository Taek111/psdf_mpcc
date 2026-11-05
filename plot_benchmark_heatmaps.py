"""Render paper figures from saved trials without importing/running a controller.

Example (from the project root):
    python plot_benchmark_heatmaps.py --width-mm 70 --seed 0

Uses the experiment's frozen maze geometry, exact footprint clearance, and
applied linear velocity. Outputs vector PDF/SVG, 600-dpi PNG, and provenance.
The default input is the complete archived benchmark under experiments/maze/.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = ROOT / "experiments" / "maze" / (
    "rectangle_width070_130mm_step005_10trials_20261103"
)
METHODS = ("mpcc", "cc_ecs", "cc_sf", "cc_mf")
LABELS = ("MPCC", "CC-ECS", "CC-SF", "CC-MF")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record(path: Path) -> dict:
    try:
        name = path.relative_to(ROOT).as_posix()
    except ValueError:
        name = path.as_posix()
    return {"path": name, "sha256": sha256(path)}


def numeric_csv(path: Path, fields: tuple[str, ...]) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [[float(row[field]) for field in fields]
                for row in csv.DictReader(handle)]
    result = np.asarray(rows, dtype=float)
    if result.ndim != 2 or len(result) < 2 or not np.isfinite(result).all():
        raise ValueError(f"Missing/nonfinite numeric data: {path}")
    return result


def literal_geometry(node: ast.AST, scale: float):
    """Read only numeric literals, tuples, and multiplication from source AST."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id == "s":
        return scale
    if isinstance(node, (ast.Tuple, ast.List)):
        return [literal_geometry(item, scale) for item in node.elts]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return literal_geometry(node.left, scale) * literal_geometry(node.right, scale)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -literal_geometry(node.operand, scale)
    raise ValueError(f"Unsupported maze literal: {ast.dump(node)}")


def load_maze(snapshot: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Extract archived maze bounds and rectangles without executing source."""
    tree = ast.parse(snapshot.read_text(encoding="utf-8"))
    branches = [node for node in ast.walk(tree)
                if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "env_type"
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "maze"]
    if len(branches) != 1:
        raise ValueError("Expected one archived maze definition")
    scale = None
    bounds = None
    polygons = []
    for statement in branches[0].body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id == "s":
                scale = float(ast.literal_eval(statement.value))
            if isinstance(target, ast.Name) and target.id == "bounds":
                bounds = np.asarray(literal_geometry(statement.value, scale))
        if isinstance(statement, ast.Expr):
            for call in ast.walk(statement):
                if (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "RectangleRegion"):
                    left, right, bottom, top = [
                        literal_geometry(arg, scale) for arg in call.args
                    ]
                    polygons.append(np.array([
                        [left, bottom], [right, bottom], [right, top], [left, top]
                    ]))
    if bounds is None or not polygons:
        raise ValueError("Archived maze geometry is incomplete")
    return bounds, polygons


def load_trial(experiment: Path, method: str, width_mm: int, seed: int) -> dict:
    folder = experiment / f"width_{width_mm:03d}mm" / method / f"seed_{seed}"
    paths = {"clearance": folder / "clearance.csv",
             "history": folder / "data" / "trial_history.csv",
             "outcome": folder / "outcome.json",
             "config": folder / "config.yaml"}
    clearance = numeric_csv(paths["clearance"], (
        "pose_index", "simulation_time_s", "x", "y", "theta", "clearance_m"
    ))
    history = numeric_csv(paths["history"], (
        "timestep", "time", "x", "y", "theta", "v", "omega"
    ))
    outcome = json.loads(paths["outcome"].read_text(encoding="utf-8"))
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    if not np.isclose(config["rectangle_geometry"]["width"] * 1000, width_mm):
        raise ValueError(f"Incorrect rectangle width: {folder}")
    if config["localization_error"]["enabled"]:
        raise ValueError("These figures require the no-localization-error comparison")
    if config["start_perturbation"]["seed"] != seed:
        raise ValueError(f"Incorrect seed: {folder}")
    if outcome["status"] != "success":
        raise ValueError(f"Trial is {outcome['status']}; do not present it as a full traversal")
    if len(clearance) != len(history) + 1:
        raise ValueError(f"Pose/control count mismatch: {folder}")
    np.testing.assert_array_equal(clearance[:, 0], np.arange(len(clearance)))
    np.testing.assert_array_equal(history[:, 0], np.arange(len(history)))
    np.testing.assert_allclose(clearance[1:, 1:5], history[:, 1:5], atol=1e-10, rtol=0)
    np.testing.assert_allclose(clearance[0, 2:5], outcome["initial_pose"], atol=1e-12)
    dt = np.diff(clearance[:, 1])
    if np.any(dt <= 0):
        raise ValueError("Non-increasing timestamps")
    np.testing.assert_allclose(dt, 0.1, atol=1e-10, rtol=0)
    # Validate the saved pose/input pairing with the archived DD RK4 formula.
    theta = clearance[:-1, 4]
    v, omega = history[:, 5], history[:, 6]
    half_theta = theta + dt * omega / 2
    end_theta = theta + dt * omega
    predicted = clearance[:-1, 2:5].copy()
    predicted[:, 0] += dt * v * (np.cos(theta) + 4*np.cos(half_theta) + np.cos(end_theta)) / 6
    predicted[:, 1] += dt * v * (np.sin(theta) + 4*np.sin(half_theta) + np.sin(end_theta)) / 6
    predicted[:, 2] = end_theta
    residual = float(np.max(np.abs(predicted - clearance[1:, 2:5])))
    if residual > 1e-9:
        raise ValueError(f"Applied input does not match recorded DD motion: {residual}")
    return {"method": method, "poses": clearance[:, 2:5],
            "clearance_mm": clearance[:, 5] * 1000,
            "velocity": v, "time": clearance[:, 1], "config": config,
            "sources": {key: source_record(path) for key, path in paths.items()},
            "rk4_max_residual": residual}


def portrait(points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.column_stack((bounds[1, 1] - points[:, 1], points[:, 0] - bounds[0, 0]))


def draw_panel(ax, trial: dict, bounds: np.ndarray, polygons: list[np.ndarray],
               metric: str, norm, cmap) -> None:
    """Draw a map with the shared metric normalization and no text labels."""
    for polygon in polygons:
        ax.add_patch(Polygon(portrait(polygon, bounds), closed=True,
                             facecolor="#dededb", edgecolor="#777772",
                             linewidth=0.45, zorder=1))
    xy = portrait(trial["poses"], bounds)
    segments = np.stack((xy[:-1], xy[1:]), axis=1)
    if metric == "clearance":
        # Each recorded pose owns its adjacent half-segments, preserving extrema.
        mid = (xy[:-1] + xy[1:]) / 2
        segments = np.stack((np.stack((xy[:-1], mid), axis=1),
                             np.stack((mid, xy[1:]), axis=1)), axis=1).reshape(-1, 2, 2)
        values = trial["clearance_mm"]
        segment_values = np.column_stack((values[:-1], values[1:])).ravel()
    else:
        segment_values = trial["velocity"]
    ax.add_collection(LineCollection(segments, colors="#484848", linewidths=2.15,
                                     capstyle="round", zorder=2))
    colored = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.65,
                             capstyle="round", zorder=3)
    colored.set_array(segment_values)
    ax.add_collection(colored)
    for point, marker, size, face, edge in (
        (xy[0], "o", 3.5, "white", "#222222"),
        (xy[-1], "*", 6.0, "black", "black"),
    ):
        ax.plot(*point, marker=marker, markersize=size, markerfacecolor=face,
                markeredgecolor=edge, markeredgewidth=0.7, zorder=4)
    map_width, map_height = (bounds[1] - bounds[0])[::-1]
    ax.set(xlim=(0, map_width), ylim=(0, map_height), aspect="equal", xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set(color="#666666", linewidth=0.55)


def draw_colorbar(fig, cax, metric: str, norm, cmap, ticks) -> None:
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                           cax=cax, orientation="horizontal")
    # Document the power normalization in the manuscript caption.
    colorbar.set_label("Clearance [mm]" if metric == "clearance"
                       else r"Linear velocity $v$ [m/s]", fontsize=9, labelpad=5)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_ticks(ticks)
    colorbar.ax.xaxis.set_major_formatter(
        FormatStrFormatter("%g" if metric == "clearance" else "%.2f"))
    colorbar.ax.tick_params(labelsize=8, length=2.5, width=0.5, pad=2)
    colorbar.outline.set_linewidth(0.45)


def export_individual(trials: list[dict], bounds: np.ndarray,
                      polygons: list[np.ndarray], metric: str, norm, cmap,
                      ticks, output: Path, dpi: int) -> dict:
    """Export map-only panels and one matching colorbar for LaTeX assembly."""
    folder = output / "individual"
    folder.mkdir(exist_ok=True)
    panel_width_mm, pad_mm = 178 * 0.21, 0.4
    map_width, map_height = (bounds[1] - bounds[0])[::-1]
    panel_height_mm = (panel_width_mm - 2*pad_mm) * map_height/map_width + 2*pad_mm
    panel_names = []
    for trial in trials:
        fig = plt.figure(figsize=(panel_width_mm/25.4, panel_height_mm/25.4),
                         facecolor="white")
        ax = fig.add_axes([pad_mm/panel_width_mm, pad_mm/panel_height_mm,
                           1 - 2*pad_mm/panel_width_mm, 1 - 2*pad_mm/panel_height_mm])
        draw_panel(ax, trial, bounds, polygons, metric, norm, cmap)
        name = f"{metric}_{trial['method']}"
        for extension in ("pdf", "svg", "png"):
            fig.savefig(folder / f"{name}.{extension}", dpi=dpi, facecolor="white")
        plt.close(fig)
        panel_names.append(name)
    colorbar_size_mm = [178 * 0.60, 14]
    fig = plt.figure(figsize=tuple(size/25.4 for size in colorbar_size_mm),
                     facecolor="white")
    cax = fig.add_axes([0.05, 0.40, 0.90, 0.16])
    draw_colorbar(fig, cax, metric, norm, cmap, ticks)
    colorbar_name = f"{metric}_colorbar"
    for extension in ("pdf", "svg", "png"):
        fig.savefig(folder / f"{colorbar_name}.{extension}", dpi=dpi, facecolor="white")
    plt.close(fig)
    return {"directory": "individual", "panels": panel_names,
            "colorbar": colorbar_name, "formats": ["pdf", "svg", "png"],
            "panel_size_mm": [panel_width_mm, panel_height_mm],
            "colorbar_size_mm": colorbar_size_mm,
            "panel_labels": "Provided by LaTeX subcaptions",
            "normalization": "Shared with combined figure and all individual panels"}


def make_figure(trials: list[dict], bounds: np.ndarray, polygons: list[np.ndarray],
                metric: str, output: Path, dpi: int, clearance_gamma: float,
                clearance_cmap: str) -> dict:
    is_clearance = metric == "clearance"
    key = "clearance_mm" if is_clearance else "velocity"
    values = np.concatenate([trial[key] for trial in trials])
    quantum = 10 if is_clearance else 0.03
    vmin = min(0.0, np.floor(values.min() / quantum) * quantum)
    vmax = max(quantum, np.ceil(values.max() / quantum) * quantum)
    norm = (PowerNorm(gamma=clearance_gamma, vmin=vmin, vmax=vmax, clip=False)
            if is_clearance else Normalize(vmin=vmin, vmax=vmax))
    cmap = matplotlib.colormaps[clearance_cmap if is_clearance else "inferno"]
    # 178 mm wide: a typical two-column paper figure; labels remain 9 pt.
    fig = plt.figure(figsize=(178 / 25.4, 101 / 25.4), facecolor="white")
    left, right, gap = 0.035, 0.965, 0.072
    panel_width = (right - left - 3 * gap) / 4
    map_width, map_height = (bounds[1] - bounds[0])[::-1]
    panel_height = panel_width * fig.get_figwidth() / fig.get_figheight() * map_height / map_width
    bottom = 0.145
    for index, (trial, label) in enumerate(zip(trials, LABELS)):
        ax = fig.add_axes([left + index*(panel_width + gap), bottom, panel_width, panel_height])
        draw_panel(ax, trial, bounds, polygons, metric, norm, cmap)
        ax.set_xlabel(f"({chr(97 + index)}) {label}", fontsize=9, labelpad=8)

    cax = fig.add_axes([0.25, 0.900, 0.50, 0.023])
    if is_clearance and clearance_gamma != 1:
        candidates = np.array([0, 0.5, 1, 2, 5, 10, 20, 40], dtype=float)
        ticks = np.unique(np.concatenate(([vmin], candidates[
            (candidates > vmin) & (candidates < vmax)
        ], [vmax])))
    else:
        ticks = np.linspace(vmin, vmax, 5 if is_clearance else 7)
    draw_colorbar(fig, cax, metric, norm, cmap, ticks)
    legend_handles = [Line2D([], [], color="none", marker=marker, markersize=size,
                            markerfacecolor=face, markeredgecolor=edge,
                            markeredgewidth=0.7, label=label)
                      for marker, size, face, edge, label in (
                          ("o", 3.5, "white", "#222222", "Start"),
                          ("*", 6.0, "black", "black", "End"),
                      )]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.006),
               frameon=False, ncol=2, fontsize=8, handletextpad=0.2, columnspacing=1.4)
    # Spatial scale in physical figure coordinates, identical to every panel.
    scale_length = panel_width * 0.3 / map_width
    x0, y0 = 0.83, 0.042
    fig.add_artist(Line2D([x0, x0+scale_length], [y0, y0], color="#333333",
                          linewidth=0.8, transform=fig.transFigure))
    for x in (x0, x0+scale_length):
        fig.add_artist(Line2D([x, x], [y0-0.005, y0+0.005], color="#333333",
                              linewidth=0.6, transform=fig.transFigure))
    fig.text(x0+scale_length/2, y0+0.012, "0.3 m", ha="center", fontsize=7.5)

    stem = output / f"{metric}_heatmap"
    for extension in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=dpi, facecolor="white")
    fig.savefig(output / f"{metric}_preview.png", dpi=180, facecolor="white")
    plt.close(fig)
    individual = export_individual(trials, bounds, polygons, metric, norm, cmap,
                                   ticks, output, dpi)
    return {"range": [float(vmin), float(vmax)], "colormap": cmap.name,
            "individual_exports": individual,
            "normalization": "power" if is_clearance else "linear",
            "gamma": clearance_gamma if is_clearance else 1.0,
            "normalized_mapping": "((value - vmin) / (vmax - vmin)) ** gamma",
            "colorbar_ticks": [float(tick) for tick in ticks],
            "unit": "mm" if is_clearance else "m/s",
            "sample_min": float(values.min()), "sample_max": float(values.max()),
            "clipped_samples": int(np.count_nonzero((values < vmin) | (values > vmax)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--width-mm", type=int, default=70)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--clearance-gamma", type=float, default=0.35,
                        help="Power scale exponent; <1 expands low clearance, 1 restores linear")
    parser.add_argument("--clearance-cmap", default="magma",
                        choices=sorted(matplotlib.colormaps),
                        help="Clearance colormap (default: magma)")
    args = parser.parse_args()
    if not np.isfinite(args.clearance_gamma) or args.clearance_gamma <= 0:
        parser.error("--clearance-gamma must be finite and positive")
    experiment = args.experiment.resolve()
    output = args.output_dir or ROOT / "figures" / "paper_heatmaps" / (
        f"maze_width{args.width_mm:03d}mm_seed{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    snapshot = experiment / "source_snapshot" / "sim" / "simulation_mpc.py"
    bounds, polygons = load_maze(snapshot)
    trials = [load_trial(experiment, method, args.width_mm, args.seed) for method in METHODS]
    for trial in trials[1:]:
        np.testing.assert_allclose(trial["poses"][0], trials[0]["poses"][0], atol=1e-12, rtol=0)
        for key in ("rectangle_geometry", "localization_error", "start_perturbation",
                    "trial_failure_criteria", "defaults"):
            if trial["config"][key] != trials[0]["config"][key]:
                raise ValueError(f"Inconsistent comparison setting: {key}")
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                         "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.unicode_minus": True, "path.simplify": False})
    metrics = {metric: make_figure(trials, bounds, polygons, metric, output, args.dpi,
                                 args.clearance_gamma, args.clearance_cmap)
               for metric in ("clearance", "velocity")}
    stats = []
    for trial, label in zip(trials, LABELS):
        row = {"method": label, "status": "success", "steps": len(trial["velocity"]),
               "duration_s": float(trial["time"][-1]),
               "rk4_max_abs_residual": trial["rk4_max_residual"]}
        for key in ("clearance_mm", "velocity"):
            for name, value in (("min", np.min(trial[key])), ("mean", np.mean(trial[key])),
                                ("max", np.max(trial[key]))):
                row[f"{key}_{name}"] = float(value)
        stats.append(row)
    manifest = {
        "experiment": source_record(experiment / "experiment.json"),
        "geometry_source": source_record(snapshot), "generator": source_record(Path(__file__)),
        "width_mm": args.width_mm, "seed": args.seed,
        "rectangle_geometry_m": trials[0]["config"]["rectangle_geometry"],
        "methods": list(METHODS), "initial_pose": trials[0]["poses"][0].tolist(),
        "world_bounds_m": bounds.tolist(), "display_transform": "(x,y) -> (y_max-y,x-x_min)",
        "endpoint_markers": {"start": "open circle", "end": "filled black star"},
        "selection": "User-selected width 70 mm, seed 0" if (args.width_mm, args.seed) == (70, 0)
                     else "Explicit width and seed passed to renderer",
        "population": "One paired trial per method; no trial averaging or success-rate inference",
        "clearance_definition": "Exact rectangle-obstacle polygon distance, min with signed map-boundary margin; obstacle overlap/contact = 0",
        "clearance_encoding": "Observed pose values in mm; each pose colors its adjacent half-segments; no smoothing or clipping",
        "velocity_definition": "Signed applied DD linear input v in m/s; history row i colors pose i to pose i+1",
        "validation": "All values finite; complete contiguous indices; all pose/time joins checked; identical starts/settings; all motion steps verified by RK4",
        "figure_size_mm": [178, 101], "png_dpi": args.dpi,
        "metrics": metrics, "trials": stats,
        "raw_sources": {trial["method"]: trial["sources"] for trial in trials},
    }
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (output / "selected_trial_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    print(json.dumps({"output": str(output), "metrics": metrics, "trials": stats}, indent=2))


if __name__ == "__main__":
    main()
