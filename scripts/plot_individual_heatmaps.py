#!/usr/bin/env python3
"""Re-render the recorded maze heatmaps as individual, matched power-scale maps.

Run from the repository root (no simulation, acados or torch required)::

    python scripts/plot_individual_heatmaps.py
    python scripts/plot_individual_heatmaps.py --metric speed --methods cc_sf cc_mf
    python scripts/plot_individual_heatmaps.py --gamma 0.5 --formats png pdf

The default input is figures/paper_heatmaps/maze_width070mm_seed0/provenance.json.
It identifies the 2026-11-03 maze experiment, rectangle 150 x 70 mm, seed 0.
Clearance comes from clearance.csv (m -> mm). Speed is the signed applied DD
input v from data/trial_history.csv, NOT a finite-difference estimate. All v in
these selected trials are positive. A history row contains the POST-step pose;
its input colors the interval from the preceding pose to that post-step pose.

Both metrics use the SAME gamma (default .35) and colormap (default magma), with
separate physical ranges: clearance 0..40 mm, speed 0...18 m/s. Each metric's
range is fixed across controllers, including when exporting a subset. Gamma < 1
allocates more color resolution to small values and less to large values.
Colorbar labels remain physical values; samples are never smoothed or clipped.

Outputs default to the source figure directory's individual_power/ subdirectory:
clearance_<method>, velocity_<method>, and separate *_colorbar files. The
velocity filename prefix is retained for compatibility with the existing LaTeX
layout. Panels have no embedded title/colorbar, and all have identical size.
The output provenance.json records inputs, hashes, scales and data alignment.
Re-running replaces files with the same names in the chosen output directory.
When changing gamma/cmap, regenerate the entire set or use a new --output-dir
so that partial exports cannot leave differently scaled panels side by side.
Dependencies: numpy, matplotlib. Paths in the input provenance are repo-relative.
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
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROVENANCE = (
    ROOT / "figures/paper_heatmaps/maze_width070mm_seed0/provenance.json"
)
METHODS = ("mpcc", "cc_ecs", "cc_sf", "cc_mf")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def receipt(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path.read_bytes())}


def verified_source(source: dict) -> tuple[Path, dict]:
    """Allow Git's CRLF checkout conversion, but no other source changes."""
    path = ROOT / source["path"]
    data = path.read_bytes()
    actual = sha256(data)
    if actual == source["sha256"]:
        match = "exact"
    elif sha256(data.replace(b"\r\n", b"\n")) == source["sha256"]:
        match = "after_CRLF_to_LF"
    else:
        raise ValueError(f"Source differs from the original figure: {path}")
    return path, {
        "path": str(path.resolve()), "sha256": actual,
        "original_sha256": source["sha256"], "hash_match": match,
    }


def numeric_csv(path: Path, columns: tuple[str, ...]) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    values = np.array([[float(row[key]) for key in columns] for row in rows])
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Empty or non-finite data: {path}")
    return values


def load_trial(method: str, provenance: dict) -> tuple[dict, dict]:
    sources = {}
    paths = {}
    for key in ("clearance", "history", "outcome", "config"):
        paths[key], sources[key] = verified_source(provenance["raw_sources"][method][key])
    clearance = numeric_csv(
        paths["clearance"],
        ("pose_index", "simulation_time_s", "x", "y", "theta", "clearance_m"),
    )
    history = numeric_csv(
        paths["history"], ("timestep", "time", "x", "y", "theta", "v"),
    )
    if len(clearance) != len(history) + 1:
        raise ValueError(f"{method}: expected initial pose plus one pose per input")
    for name, values in (("clearance", clearance), ("history", history)):
        if not np.array_equal(values[:, 0], np.arange(len(values))):
            raise ValueError(f"{method}: missing or repeated {name} indices")
    if not np.all(np.diff(clearance[:, 1]) > 0):
        raise ValueError(f"{method}: pose times must be strictly increasing")
    if not np.allclose(clearance[1:, 1:5], history[:, 1:5], rtol=0, atol=1e-10):
        raise ValueError(f"{method}: post-step pose/time alignment failed")
    if not np.allclose(clearance[0, 2:5], provenance["initial_pose"], rtol=0, atol=1e-10):
        raise ValueError(f"{method}: initial pose differs from the selected paired trial")
    outcome = json.loads(paths["outcome"].read_text(encoding="utf-8"))
    trial = {
        "xy": clearance[:, 2:4],
        "clearance": clearance[:, 5] * 1000.0,
        "velocity": history[:, 5],
    }
    metadata = {
        "sources": sources, "status": outcome["status"],
        "pose_samples": len(clearance), "applied_input_samples": len(history),
        "duration_s": float(clearance[-1, 1] - clearance[0, 1]),
        "post_step_pose_time_max_abs_difference": float(
            np.max(np.abs(clearance[1:, 1:5] - history[:, 1:5]))
        ),
    }
    return trial, metadata


def numeric_expression(node: ast.AST, scale: float):
    """Read only the archived maze's numeric literals, tuples and products."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id == "s":
        return scale
    if isinstance(node, (ast.Tuple, ast.List)):
        return [numeric_expression(item, scale) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -numeric_expression(node.operand, scale)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return numeric_expression(node.left, scale) * numeric_expression(node.right, scale)
    raise ValueError(f"Unsupported archived geometry expression: {ast.dump(node)}")


def load_geometry(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Extract geometry from the archived maze branch without executing it."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    create_env = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "create_env"
    )
    branch = next(
        node for node in ast.walk(create_env)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "env_type"
        and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "maze"
    )
    assignments = {
        node.targets[0].id: node.value for node in branch.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    scale = numeric_expression(assignments["s"], 1.0)
    bounds = np.asarray(numeric_expression(assignments["bounds"], scale))
    polygons = []
    for statement in branch.body:
        for node in ast.walk(statement):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "RectangleRegion"):
                left, right, bottom, top = [
                    numeric_expression(arg, scale) for arg in node.args
                ]
                polygons.append(np.array([
                    [left, bottom], [right, bottom], [right, top], [left, top],
                ]))
    if not polygons:
        raise ValueError("No rectangles found in the archived maze geometry")
    return bounds, polygons


def display_xy(xy: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    # Same rigid rotation/translation as the original figures.
    return np.column_stack((bounds[1, 1] - xy[:, 1], xy[:, 0] - bounds[0, 0]))


def trajectory_segments(xy: np.ndarray, metric: str, values: np.ndarray):
    if metric == "velocity":
        return np.stack((xy[:-1], xy[1:]), axis=1), values
    # A pose colors its two neighboring half-segments. Keep the original
    # per-pose samples (including both endpoints); do not average clearance.
    midpoints = (xy[:-1] + xy[1:]) / 2
    segments = np.empty((2 * (len(xy) - 1), 2, 2))
    segments[0::2] = np.stack((xy[:-1], midpoints), axis=1)
    segments[1::2] = np.stack((midpoints, xy[1:]), axis=1)
    colors = np.column_stack((values[:-1], values[1:])).ravel()
    return segments, colors


def save_figure(fig, stem: Path, formats: list[str], dpi: int) -> None:
    # Fixed canvas: no tight cropping, so individual panels align in LaTeX.
    for extension in formats:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=dpi, facecolor="white")
    plt.close(fig)


def draw_panel(trial, metric, norm, cmap, bounds, polygons, size_mm):
    fig = plt.figure(figsize=np.asarray(size_mm) / 25.4)
    padding = 0.4  # mm; preserves the original panel's equal-aspect map area
    ax = fig.add_axes([
        padding / size_mm[0], padding / size_mm[1],
        1 - 2 * padding / size_mm[0], 1 - 2 * padding / size_mm[1],
    ])
    for polygon in polygons:
        ax.add_patch(Polygon(
            display_xy(polygon, bounds), facecolor="#dededb",
            edgecolor="#777772", linewidth=0.45, zorder=1,
        ))
    xy = display_xy(trial["xy"], bounds)
    segments, colors = trajectory_segments(xy, metric, trial[metric])
    # A quiet outline keeps bright values visible on the white free space.
    ax.add_collection(LineCollection(
        segments, colors="#555550", linewidths=1.95, capstyle="round", zorder=2,
    ))
    colored = LineCollection(
        segments, cmap=cmap, norm=norm, linewidths=1.65, capstyle="round", zorder=3,
    )
    colored.set_array(colors)
    ax.add_collection(colored)
    ax.plot(*xy[0], marker="o", markersize=3.5, markerfacecolor="white",
            markeredgecolor="#222222", markeredgewidth=0.7, linestyle="none", zorder=4)
    ax.plot(*xy[-1], marker="*", markersize=5.0, color="black",
            markeredgewidth=0.7, linestyle="none", zorder=4)
    ax.set(xlim=(0, bounds[1, 1] - bounds[0, 1]),
           ylim=(0, bounds[1, 0] - bounds[0, 0]), xticks=[], yticks=[], aspect="equal")
    for spine in ax.spines.values():
        spine.set_color("#777772")
        spine.set_linewidth(0.45)
    return fig


def draw_colorbar(norm, cmap, ticks, label, gamma, size_mm):
    fig = plt.figure(figsize=np.asarray(size_mm) / 25.4)
    ax = fig.add_axes([0.05, 0.40, 0.90, 0.16])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax,
        orientation="horizontal", ticks=ticks,
        format=FuncFormatter(lambda value, _: f"{value:g}"),
    )
    colorbar.ax.minorticks_off()
    colorbar.ax.tick_params(labelsize=8, length=2.5, width=0.5, pad=2)
    colorbar.outline.set_linewidth(0.5)
    ax.set_title(f"{label}  (power, " + rf"$\gamma={gamma:g}$" + ")", fontsize=9, pad=5)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Default: <provenance directory>/individual_power",
    )
    parser.add_argument(
        "--metric", choices=("both", "clearance", "speed", "velocity"), default="both",
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--gamma", type=float, default=0.35,
        help="Shared PowerNorm exponent (default: .35)",
    )
    parser.add_argument("--cmap", default="magma", help="Shared colormap (default: magma)")
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf", "svg"),
        default=["png", "pdf", "svg"],
    )
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if not np.isfinite(args.gamma) or args.gamma <= 0:
        parser.error("--gamma must be finite and positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.cmap not in matplotlib.colormaps:
        parser.error(f"Unknown colormap: {args.cmap}")
    args.methods = list(dict.fromkeys(args.methods))
    args.formats = list(dict.fromkeys(args.formats))
    return args


def main() -> None:
    args = parse_args()
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    output = args.output_dir or args.provenance.parent / "individual_power"
    if (output / "provenance.json").resolve() == args.provenance.resolve():
        raise ValueError("Choose an output directory separate from the input provenance")
    _, experiment_receipt = verified_source(provenance["experiment"])
    geometry_path, geometry_receipt = verified_source(provenance["geometry_source"])
    bounds, polygons = load_geometry(geometry_path)
    if not np.allclose(bounds, provenance["world_bounds_m"], rtol=0, atol=1e-12):
        raise ValueError("Archived map bounds differ from figure provenance")
    trials, trial_metadata = {}, {}
    for method in METHODS:
        trials[method], trial_metadata[method] = load_trial(method, provenance)
    metrics = ["clearance", "velocity"] if args.metric == "both" else [
        "velocity" if args.metric == "speed" else args.metric
    ]
    settings = {}
    for metric in metrics:
        spec = provenance["metrics"][metric]
        low, high = spec["range"]
        values = np.concatenate([trials[method][metric] for method in METHODS])
        if low < 0 or high <= low or values.min() < low or values.max() > high:
            raise ValueError(
                f"{metric}: data outside the fixed nonnegative range "
                f"{spec['range']}; no clipping allowed"
            )
        norm = PowerNorm(gamma=args.gamma, vmin=low, vmax=high, clip=False)
        # Use readable physical ticks, positioned by the same PowerNorm.
        # Proportional copies of the clearance ticks create long, overlapping
        # speed labels (e.g. .00225); equal gamma does not require equal ticks.
        ticks = (spec["colorbar_ticks"] if metric == "clearance"
                 else [0.0, 0.01, 0.03, 0.06, 0.1, 0.18])
        settings[metric] = {
            "normalization": "power", "gamma": args.gamma, "colormap": args.cmap,
            "range": [low, high], "unit": spec["unit"],
            "normalized_mapping": "((value - vmin) / (vmax - vmin)) ** gamma",
            "colorbar_ticks": list(ticks), "clipped_samples": 0,
            "sample_min_all_methods": float(values.min()),
            "sample_max_all_methods": float(values.max()),
        }
        # Validate everything before creating any output images.
        settings[metric]["_norm"] = norm
    existing_manifest = output / "provenance.json"
    if existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
        changed = any(
            spec["gamma"] != args.gamma or spec["colormap"] != args.cmap
            for spec in previous["metrics"].values()
        )
        planned_files = {
            f"{metric}_{name}.{extension}"
            for metric in metrics for name in [*args.methods, "colorbar"]
            for extension in args.formats
        }
        # Inspect files too: the last manifest may describe only a partial run.
        existing_files = {
            f"{metric}_{name}.{extension}"
            for metric in ("clearance", "velocity")
            for name in [*METHODS, "colorbar"]
            for extension in ("png", "pdf", "svg")
            if (output / f"{metric}_{name}.{extension}").exists()
        }
        complete_replacement = existing_files.issubset(planned_files)
        if changed and not complete_replacement:
            raise ValueError(
                "Partial export would mix color scales in the output directory. "
                "Use a new --output-dir or regenerate all metrics/methods/formats."
            )
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 9,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    output.mkdir(parents=True, exist_ok=True)
    layout = provenance["metrics"]["clearance"]["individual_exports"]
    for metric in metrics:
        norm = settings[metric].pop("_norm")
        for method in args.methods:
            fig = draw_panel(trials[method], metric, norm, args.cmap, bounds, polygons,
                             layout["panel_size_mm"])
            save_figure(fig, output / f"{metric}_{method}", args.formats, args.dpi)
        label = "Clearance [mm]" if metric == "clearance" else "Applied speed [m/s]"
        fig = draw_colorbar(norm, args.cmap, settings[metric]["colorbar_ticks"],
                            label, args.gamma, layout["colorbar_size_mm"])
        save_figure(fig, output / f"{metric}_colorbar", args.formats, args.dpi)
    metadata = {
        "original_figure_provenance": receipt(args.provenance),
        "generator": receipt(Path(__file__)),
        "experiment": experiment_receipt, "geometry_source": geometry_receipt,
        "width_mm": provenance["width_mm"], "seed": provenance["seed"],
        "rectangle_geometry_m": provenance["rectangle_geometry_m"],
        "population": provenance["population"],
        "clearance_definition": provenance["clearance_definition"],
        "velocity_definition": provenance["velocity_definition"],
        "display_transform": provenance["display_transform"],
        "clearance_encoding": provenance["clearance_encoding"],
        "world_bounds_m": bounds.tolist(),
        "obstacle_vertices_m": [polygon.tolist() for polygon in polygons],
        "methods_exported": args.methods, "metrics": settings,
        "formats": args.formats, "png_dpi": args.dpi,
        "files_generated_this_run": [
            f"{metric}_{name}.{extension}"
            for metric in metrics
            for name in [*args.methods, "colorbar"]
            for extension in args.formats
        ],
        "panel_size_mm": layout["panel_size_mm"],
        "colorbar_size_mm": layout["colorbar_size_mm"],
        "trials": trial_metadata,
        "validation": "Source hashes checked; finite values; contiguous indices; "
                      "all post-step pose/time joins checked; paired starts checked; "
                      "all methods fit fixed color limits. No simulation run.",
    }
    (output / "provenance.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(f"Source: {args.provenance.resolve()}")
    print(f"Shared PowerNorm gamma={args.gamma:g}, cmap={args.cmap}")
    print(f"Saved {len(metrics) * (len(args.methods) + 1)} figures to {output.resolve()}")


if __name__ == "__main__":
    main()
