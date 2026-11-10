"""Explain finite-step versus boundary residual changes using a saved trace."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_sf_mf_reliability import make_wrapper, evaluate
from utils.offline_cc_margin import _snapshot_imports


def main():
    torch.set_num_threads(6)
    analysis = HERE.parent
    suite = ROOT / "benchmark_runs/rectangle_constraint_reliability_20261107/rectangle_don_sweep_50_20261105_current_sf"
    cache_path = analysis / "cache/don0.85_cc_mf_seed35.npz"
    event_path = analysis / "tables/switch_events.csv"
    metadata_path = suite / "widths/width_127p5mm/runs/config04/trial_0036/data/raw_trace_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache = np.load(cache_path)
    event_data = pd.read_csv(event_path)
    events = event_data[(event_data.source == "cc_mf") & (event_data.seed == 35) & (event_data.step == 99)]
    boundary = events[(events.kind == "boundary") & np.isclose(events.probe_radius, 1e-4)].iloc[0]
    adjacent = events[events.kind == "adjacent"].iloc[0]
    center = float(boundary.alpha)
    xi = np.array([boundary[f"xi_{j}"] for j in range(4)])
    full_alpha = np.unique(np.r_[np.linspace(0, 1, 801), center + np.array([-1e-4, -1e-6, 1e-6, 1e-4])])
    near_alpha = center + np.r_[np.linspace(-.001, -1e-8, 401), np.linspace(1e-8, .001, 401)]
    curves = []
    with _snapshot_imports(suite / "source_snapshot") as (module, _, implicit):
        wrapper = make_wrapper(module, metadata, torch.float64)
        for panel, alpha in (("full", full_alpha), ("boundary", near_alpha)):
            poses = cache["poses"][99] + alpha[:, None] * (cache["poses"][100] - cache["poses"][99])
            result = evaluate(wrapper, poses, xi, .1, .0001)
            frame = pd.DataFrame({"panel": panel, "alpha": alpha, "delta_alpha": alpha - center,
                                  "b_sf_pp": 100 * (.1 - result["p_sf"]),
                                  "b_mf_pp": 100 * (.1 - result["r_mf"]),
                                  "p_sf_pct": 100 * result["p_sf"], "R_pct": 100 * result["r_mf"]})
            for j, column in enumerate(("x_m", "y_m", "heading_rad")):
                frame[column] = poses[:, j]
            curves.append(frame)
        # Retain an independently inspectable decomposition at the endpoints.
        poses = cache["poses"][[99, 100]]
        z = np.column_stack((poses, np.zeros(2), np.tile(xi, (2, 1))))
        risk = wrapper.forward_mf_icdf(torch.tensor(z), .1, .0001, return_frozen_features=True)
        frozen = risk.frozen_features
        variance, _ = implicit.projected_variance_and_gradient(frozen["q"], torch.tensor(z))
        p = .5 * torch.special.erfc(-(0.0001 - frozen["d"]) / variance.sqrt() / np.sqrt(2))
        p = torch.where(frozen["mask"], p, 0).numpy()
    frame = pd.concat(curves, ignore_index=True)
    frame.to_csv(HERE / "residual_curves.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.7), layout="constrained")
    colors = {"sf": "#2864A5", "mf": "#C27620"}
    for ax, name in zip(axs, ("full", "boundary")):
        data = frame[frame.panel == name]
        xx = data.alpha if name == "full" else data.delta_alpha * 1000
        for formulation, line in (("sf", "-"), ("mf", "--")):
            ax.plot(xx, data[f"b_{formulation}_pp"], color=colors[formulation], ls=line,
                    lw=2.2, label=f"{formulation.upper()} residual")
        ax.axvline(center if name == "full" else 0, ls=":", color="#777777", lw=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#DCE1E5", lw=.6)
        ax.set_axisbelow(True)
        ax.set_ylabel("Probability-scale residual b (pp)")
    axs[0].set_title("(a) Entire recorded 0.1 s pose interval")
    axs[0].set_xlabel("Pose interpolation fraction alpha (0 to 1)")
    axs[0].set_xlim(0, 1)
    axs[0].legend(frameon=False, loc="upper left")
    axs[0].text(.97, .13, f"Endpoint change |delta b|\nSF: {adjacent.j_sf_pp:.3f} pp\nMF: {adjacent.j_mf_pp:.3f} pp",
                transform=axs[0].transAxes, ha="right", va="bottom", fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88})
    for formulation in ("sf", "mf"):
        full = curves[0]
        axs[0].scatter([0, 1], full.iloc[[0, -1]][f"b_{formulation}_pp"],
                       color=colors[formulation], s=25, zorder=4)
    axs[1].set_title("(b) Same segment, close to the feature switch")
    axs[1].set_xlabel("Offset from switch, delta alpha (x 0.001)")
    axs[1].set_xlim(-1, 1)
    axs[1].text(.97, .46, f"At delta alpha = +/- 0.0001\nSF: {boundary.j_sf_pp:.4f} pp\nMF: {boundary.j_mf_pp:.4f} pp",
                transform=axs[1].transAxes, ha="right", va="center", fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .92})
    axs[1].text(.03, .17, "Local y-axis range differs from panel (a).", transform=axs[1].transAxes,
                fontsize=8.5, color="#555555")
    fig.suptitle("Smooth finite-step change can exceed a switching residual's endpoint change", fontsize=13)
    fig.supxlabel("DoN 0.85 | CC-MF source, seed 35, 9.9 to 10.0 s | Identical poses and fixed reconstructed covariance\n"
                  "CPU float64 offline model evaluation; interpolated poses are not additional recorded states.", fontsize=9)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"finite_step_vs_boundary.{extension}", dpi=210, bbox_inches="tight")
    plt.close(fig)
    sources = [cache_path, event_path, metadata_path, Path(__file__), ROOT / "scripts/analyze_sf_mf_reliability.py"]
    sources += [suite / "source_snapshot/models" / name for name in ("augmented_psdf.py", "augmented_psdf_wrapper.py", "mf_inverse_cdf.py", "mf_implicit.py")]
    summary = {"source": "cc_mf", "seed": 35, "step": 99, "time_interval_s": [9.9, 10.0],
               "boundary_alpha": center, "fixed_log_cholesky_state": xi.tolist(),
               "full_interval_j_sf_pp": float(adjacent.j_sf_pp), "full_interval_j_mf_pp": float(adjacent.j_mf_pp),
               "boundary_probe_radius": 1e-4, "boundary_j_sf_pp": float(boundary.j_sf_pp),
               "boundary_j_mf_pp": float(boundary.j_mf_pp),
               "endpoint_contributions_pct": {"feature_C4_slot4_flat36": (100 * p[:, 36]).tolist(),
                                                "feature_C1_slot6_flat14": (100 * p[:, 14]).tolist(),
                                                "sum_all_features": (100 * p.sum(axis=1)).tolist()},
               "interpretation": "Illustrative saved segment chosen to explain the two metrics, not a representative population estimate. Same fixed one-step reconstructed covariance in both panels; native planning covariance unavailable.",
               "simulation_executed": False, "solver_executed": False,
               "sources": [{"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources]}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(HERE / "finite_step_vs_boundary.png")


if __name__ == "__main__":
    main()
