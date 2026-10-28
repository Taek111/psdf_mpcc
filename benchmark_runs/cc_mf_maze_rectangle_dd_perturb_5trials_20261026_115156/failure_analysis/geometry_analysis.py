"""Offline geometry checks of saved failures; no controller or solver is run."""

import ast
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


OUT = Path(__file__).resolve().parent
RUN = OUT.parent
ROOT = RUN.parent.parent
BEFORE = "#276892"
AFTER = "#c26026"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def rectangle(xmin, xmax, ymin, ymax):
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


def load_environment():
    source = ROOT / "sim/simulation_mpc.py"
    expected = json.loads((RUN / "environment.json").read_text())["source_sha256"]["sim/simulation_mpc.py"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "simulation_mpc")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "create_env")
    namespace = {"np": np, "math": math, "RectangleRegion": rectangle}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["create_env"](None, "maze")


def load_rk4():
    source = ROOT / "models/dd_dynamics.py"
    tree = ast.parse(source.read_text())
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_rk4_pose_components")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return lambda pose, u, dt: np.array(namespace["_rk4_pose_components"](pose, u[0], u[1], dt, np.cos, np.sin))


def footprint(pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return rectangle(-0.075, 0.075, -0.05, 0.05) @ np.array([[c, s], [-s, c]]) + pose[:2]


def strict_overlap(first, second):
    edges = np.vstack((np.roll(first, -1, axis=0)-first, np.roll(second, -1, axis=0)-second))
    axes = np.column_stack((-edges[:, 1], edges[:, 0]))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    pa, pb = first @ axes.T, second @ axes.T
    overlap = np.minimum(pa.max(axis=0), pb.max(axis=0))-np.maximum(pa.min(axis=0), pb.min(axis=0))
    return bool(np.all(overlap > 1e-12))


def main():
    _, _, grid, obstacles = load_environment()
    advance = load_rk4()
    source = OUT / "source_snapshot/sim/start_perturbation.py"
    spec = importlib.util.spec_from_file_location("saved_start_perturbation", source)
    geom = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geom)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "DejaVu Sans", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.2), gridspec_kw={"width_ratios": [1, 1.45]})
    evidence = []
    for row_index, (trial, seed, k) in enumerate([(1, 0, 35), (2, 1, 29)]):
        directory = RUN / f"runs/config01/trial_{trial:04d}/data"
        pose_rows = read_csv(next(directory.glob("pose_sdf_*.csv")))
        history = read_csv(next(directory.glob("trial_history_*.csv")))
        before = np.array([float(pose_rows[k][key]) for key in ("x", "y", "theta")])
        after = np.array([float(history[k][key]) for key in ("x", "y", "theta")])
        assert np.allclose(after, [float(pose_rows[k+1][key]) for key in ("x", "y", "theta")], atol=1e-12)
        u = np.array([float(history[k]["v"]), float(history[k]["omega"])])
        endpoint = advance(before, u, 0.1)
        error = float(np.max(np.abs(endpoint-after)))
        assert error < 1e-12
        samples = np.array([advance(before, u, dt) for dt in np.linspace(0, 0.1, 101)])
        overlap_samples = []
        for index, pose in enumerate(samples):
            hits = [i for i, obs in enumerate(obstacles) if strict_overlap(footprint(pose), obs)]
            if hits:
                overlap_samples.append((index * 0.001, hits))
        before_poly, after_poly = footprint(before), footprint(after)
        bounds = np.asarray(grid[0])
        record = {
            "trial": trial, "seed": seed, "control_timestep": k,
            "time_before_s": k * 0.1, "time_after_s": (k+1)*0.1,
            "v_m_s": float(u[0]), "omega_rad_s": float(u[1]),
            "pre_psdf_mm": float(pose_rows[k]["sdf_value"])*1000,
            "post_psdf_mm": float(pose_rows[k+1]["sdf_value"])*1000,
            "strict_overlap_before_obstacle_indices": [i for i, obs in enumerate(obstacles) if strict_overlap(before_poly, obs)],
            "strict_overlap_after_obstacle_indices": [i for i, obs in enumerate(obstacles) if strict_overlap(after_poly, obs)],
            "pre_boundary_clearance_mm": float(min(np.min(before_poly-bounds[0]), np.min(bounds[1]-before_poly)))*1000,
            "post_boundary_clearance_mm": float(min(np.min(after_poly-bounds[0]), np.min(bounds[1]-after_poly)))*1000,
            "first_sampled_overlap_offset_s": overlap_samples[0][0] if overlap_samples else None,
            "rk4_endpoint_max_abs_error": error,
            "before_pose": before.tolist(), "after_pose": after.tolist(),
        }
        before_distances = [geom.convex_polygon_clearance(before_poly, obs)*1000 for obs in obstacles]
        record["pre_geometric_clearance_mm"] = min(before_distances)
        record["pre_nearest_obstacle_index"] = int(np.argmin(before_distances))
        record["pre_clearance_to_subsequently_hit_obstacle_mm"] = {
            str(i): before_distances[i] for i in record["strict_overlap_after_obstacle_indices"]
        }
        record["post_geometric_clearance_mm"] = min(geom.convex_polygon_clearance(after_poly, obs) for obs in obstacles)*1000
        record["overlapped_obstacle_vertices"] = {str(i): obstacles[i].tolist() for i in record["strict_overlap_after_obstacle_indices"]}
        evidence.append(record)

        ax = axes[row_index, 0]
        for obs in obstacles:
            ax.add_patch(Polygon(obs, facecolor="#e1e4e6", edgecolor="#7f878c", linewidth=0.8))
        ax.plot(samples[:, 0], samples[:, 1], color="#333333", linewidth=1.4, linestyle=":", zorder=2)
        ax.add_patch(Polygon(before_poly, fill=False, edgecolor=BEFORE, linewidth=2.1, linestyle="--", label="Before input"))
        ax.add_patch(Polygon(after_poly, facecolor=AFTER, edgecolor=AFTER, alpha=0.30, linewidth=2.1, label="After 0.1 s"))
        ax.plot(before[0], before[1], "o", color=BEFORE, markersize=4)
        ax.plot(after[0], after[1], "s", color=AFTER, markersize=4)
        ax.set(xlim=(-0.045, 0.305), ylim=(0.405, 0.755), xlabel="x [m]", ylabel="y [m]")
        ax.set_aspect("equal")
        ax.set_title(f"Trial {trial} (seed {seed}): recorded transition", loc="left", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.94)
        ax.text(0.02, 0.02, f"v = {u[0]:+.3f} m/s, omega = {u[1]:+.2f} rad/s", transform=ax.transAxes,
                fontsize=8.4, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .93})

        ax = axes[row_index, 1]
        times = np.arange(len(pose_rows))*0.1
        phi = np.array([float(item["sdf_value"])*1000 for item in pose_rows])
        event_time = (k+1)*0.1
        visible = (times >= event_time-.8-1e-9) & (times <= event_time+.6+1e-9)
        ax.axhspan(-18, 0, color=AFTER, alpha=.08)
        ax.axhline(0, color="#333333", linewidth=.9)
        ax.axhline(1, color="#777777", linewidth=1.1, linestyle="--", label="d_col = 1 mm")
        ax.plot(times[visible], phi[visible], color=BEFORE, linewidth=2, marker="o", markersize=3.8)
        ax.axvline(event_time, color=AFTER, linewidth=1.5, linestyle=":")
        ax.plot(event_time, phi[k+1], "s", color=AFTER, markersize=6)
        ax.annotate(f"{phi[k]:+.2f} mm", (k*.1, phi[k]), xytext=(-37, 24), textcoords="offset points",
                    fontsize=10, color=BEFORE, arrowprops={"arrowstyle": "-", "color": BEFORE})
        ax.annotate(f"{phi[k+1]:+.2f} mm", (event_time, phi[k+1]), xytext=(14, 16), textcoords="offset points",
                    fontsize=10, color=AFTER, arrowprops={"arrowstyle": "-", "color": AFTER})
        ax.text(event_time+.025, 17.5, f"First safe_stop\nt = {event_time:.1f} s", fontsize=9.2, color=AFTER, va="top")
        ax.set(xlim=(event_time-.8, event_time+.6), ylim=(-18, 20), xlabel="Simulation time [s]", ylabel="Logged exact PSDF [mm]")
        ax.set_title("PSDF before each control update", loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#dddddd", linewidth=.6)
        ax.legend(loc="lower left", fontsize=8, framealpha=.94)
    fig.suptitle("A successful solver step precedes both geometric failures", x=.06, ha="left", fontsize=15, fontweight="bold")
    fig.text(.06, .935, "Saved states and inputs; rectangle 0.15 x 0.10 m; dt = 0.1 s; no controller rerun", fontsize=10, color="#555555")
    fig.text(.06, .018, "Geometry: dashed outline = before input; filled outline = after input. Negative PSDF is confirmed by strict polygon overlap.\n"
             "Right panels show pre-control samples: a newly unsafe state is visible at the next cycle. Dotted path replays the saved input only.",
             fontsize=9, color="#555555")
    fig.subplots_adjust(top=.87, bottom=.12, left=.07, right=.97, wspace=.26, hspace=.36)
    fig.savefig(OUT / "failure_transition.png", dpi=180)
    fig.savefig(OUT / "failure_transition.pdf")
    plt.close(fig)
    (OUT / "geometry_evidence.json").write_text(json.dumps({
        "source": "Saved trial histories and exact maze definition verified against recorded SHA-256.",
        "method": "Strict convex polygon SAT overlap; map-boundary vertex clearance. First-interval samples use original RK4 function at 1 ms offsets under saved constant input.",
        "limitations": "Substep sampling is not a proof of continuous-time clearance; no retuning or new controller trial was performed. Obstacle indices are zero-based create_env order.",
        "trials": evidence,
    }, indent=2))
    fields = [key for key, value in evidence[0].items() if not isinstance(value, (dict, list))]
    with (OUT / "geometry_evidence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
