"""Render the offline CC-SF/CC-MF sensitivity evidence, without running a solver."""

from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_runs/rectangle_constraint_reliability_20261107"
COLORS = {"cc_sf": "#2864A5", "cc_mf": "#C27620"}
LABELS = {"cc_sf": "CC-SF", "cc_mf": "CC-MF"}


def save(fig, name):
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(folder / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E0E3E6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def boundary_examples(boundary):
    """Show a disclosed conditional median, with all-event statistics elsewhere."""
    import torch
    torch.set_num_threads(6)
    sys.path.insert(0, str(ROOT / "scripts"))
    from analyze_sf_mf_reliability import make_wrapper, evaluate
    from utils.offline_cc_margin import _snapshot_imports

    suite = ROOT / "benchmark_runs/rectangle_don_sweep_50_20261105_current_sf"
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.6), layout="constrained", sharey=True)
    selected = []
    with _snapshot_imports(suite / "source_snapshot") as (module, _, _):
        for ax, source, config in zip(axs, ["cc_sf", "cc_mf"], ["config03", "config04"]):
            data = boundary[(boundary.source == source) & boundary.stable_ids &
                            boundary.domain_active & (boundary.mask_changes == 0) &
                            ((.1 - boundary.b_sf_left >= .05) |
                             (.1 - boundary.b_sf_right >= .05))].copy()
            first_seed = int(data.seed.iloc[0])
            metadata = suite / "widths/width_127p5mm/runs" / config / f"trial_{first_seed + 1:04d}/data/raw_trace_metadata.json"
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            wrapper = make_wrapper(module, meta, torch.float64)
            xi_batch = data[[f"xi_{j}" for j in range(4)]].to_numpy()
            left_q = evaluate(wrapper, data[[f"{k}_left" for k in ("x", "y", "theta")]].to_numpy(),
                              xi_batch, .1, meta["optimizer_parameters"]["d_min"])["q"]
            right_q = evaluate(wrapper, data[[f"{k}_right" for k in ("x", "y", "theta")]].to_numpy(),
                               xi_batch, .1, meta["optimizer_parameters"]["d_min"])["q"]
            data["normal_delta"] = np.linalg.norm(right_q[:, :2] - left_q[:, :2], axis=1)
            data["heading_gradient_delta_m"] = abs(right_q[:, 2] - left_q[:, 2])
            data = data[(data.normal_delta > .1) | (data.heading_gradient_delta_m > .001)]
            # A non-negligible-risk, normal-changing subset, defined before
            # ranking jump values. It is not the median of all switch events.
            target = data.j_sf_pp.median()
            event = data.loc[(data.j_sf_pp - target).abs().idxmin()]
            selected.append({**event.to_dict(), "selection_subset_n": len(data),
                             "selection_median_j_sf_pp": target})
            seed, step = int(event.seed), int(event.step)
            cache = np.load(OUT / "cache" / f"don0.85_{source}_seed{seed}.npz")
            metadata = suite / "widths/width_127p5mm/runs" / config / f"trial_{seed + 1:04d}/data/raw_trace_metadata.json"
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            wrapper = make_wrapper(module, meta, torch.float64)
            offsets = np.r_[np.linspace(-.01, -.00001, 300), np.linspace(.00001, .01, 300)]
            alpha = event.alpha + offsets
            in_range = (alpha >= 0) & (alpha <= 1)
            offsets, alpha = offsets[in_range], alpha[in_range]
            poses = cache["poses"][step] + alpha[:, None] * (
                cache["poses"][step + 1] - cache["poses"][step])
            xi = np.array([event[f"xi_{j}"] for j in range(4)])
            result = evaluate(wrapper, poses, xi, .1, meta["optimizer_parameters"]["d_min"])
            for formulation, key, linestyle in [("cc_sf", "p_sf", "-"), ("cc_mf", "r_mf", "--")]:
                ax.plot(offsets * 1000, 100 * (.1 - result[key]), color=COLORS[formulation],
                        label=f"{LABELS[formulation]} evaluation", lw=2, ls=linestyle)
            ax.axvline(0, color="#555555", lw=.8, ls=":")
            ax.axhline(0, color="#222222", lw=.8)
            ax.set_title(f"{LABELS[source]} source | seed {seed}, t={event.time_s:.3f} s\n"
                         f"boundary J: SF {event.j_sf_pp:.3g}, MF {event.j_mf_pp:.3g} pp | subset n={len(data)}")
            ax.set_xlabel("Interpolation offset from switch, delta alpha (x 0.001)")
            ax.set_ylabel("Probability-scale residual b (pp)")
            style(ax)
    axs[0].legend(frameon=False, loc="best")
    fig.suptitle("Feature-switch neighborhoods | DoN 0.85 | fixed covariance within each panel", fontsize=13)
    fig.supxlabel("Conditional median SF jump: stable mask/IDs, changed normal or rotational gradient, SF risk >= 5%.\n"
                  "CPU float64 offline reevaluation; negative residual means risk exceeds epsilon.", fontsize=9)
    save(fig, "feature_switch_boundary_examples_don085")
    pd.DataFrame(selected).to_csv(OUT / "tables/selected_boundary_examples.csv", index=False)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    trials = pd.read_csv(OUT / "tables/per_trial_metrics.csv")
    events = pd.read_csv(OUT / "tables/switch_events.csv")
    summary = pd.read_csv(OUT / "tables/paired_summary.csv")
    e85 = events[np.isclose(events.don, .85)]
    t85 = trials[np.isclose(trials.don, .85)]
    boundary = e85[(e85.kind == "boundary") & np.isclose(e85.probe_radius, 1e-4)]
    adjacent = e85[e85.kind == "adjacent"]

    fig, axs = plt.subplots(1, 2, figsize=(10.3, 4.3), layout="constrained")
    for ax, data, title in zip(axs, [adjacent, boundary],
                              ["(a) Consecutive recorded poses (0.1 s)",
                               "(b) Localized feature-switch boundaries"]):
        for offset, formulation in [(-.18, "sf"), (.18, "mf")]:
            vals = [data.loc[data.source == src, f"j_{formulation}_pp"].mean()
                    for src in ("cc_sf", "cc_mf")]
            bars = ax.bar(np.arange(2) + offset, vals, width=.32,
                          label=f"{formulation.upper()} evaluation",
                          color=COLORS[f"cc_{formulation}"],
                          hatch="" if formulation == "sf" else "///", zorder=3)
            ax.bar_label(bars, labels=[f"{v:.4g}" for v in vals], padding=5, fontsize=10)
        ax.set_xticks([0, 1], [f"{LABELS[s]} source\n"
                               f"{len(data[data.source == s]):,} events"
                               for s in ("cc_sf", "cc_mf")])
        ax.set_ylabel("Mean absolute residual change (pp)")
        ax.set_title(title)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.27)
        style(ax)
    axs[0].legend(loc="upper left", frameon=False)
    fig.suptitle("DoN 0.85 | 28 paired successful seeds | same pose pair and covariance", fontsize=13)
    fig.supxlabel("Fixed one-step reconstructed covariance; fresh offline geometry.\n"
                  "Boundary probe: original segment fraction alpha +/- 0.0001; all finite detected events.",
                  fontsize=9)
    save(fig, "feature_switch_sensitivity_don085")
    boundary_examples(boundary)

    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.7), layout="constrained",
                            gridspec_kw={"width_ratios": [1.15, 1]})
    pivot = t85.pivot(index="seed", columns="source", values="violation_rate_pct")
    for seed, row in pivot.iterrows():
        axs[0].plot([0, 1], [row.cc_sf, row.cc_mf], color="#B6BCC1", lw=.75, zorder=1)
    for x, src in enumerate(("cc_sf", "cc_mf")):
        axs[0].scatter(np.full(len(pivot), x), pivot[src], s=23,
                       color=COLORS[src], edgecolor="white", lw=.4, zorder=3)
        mean = pivot[src].mean()
        axs[0].scatter(x, mean, s=100, marker="D", color="black", zorder=4)
        axs[0].annotate(f"mean {mean:.2f}%", (x, mean), xytext=(9 if x == 0 else -9, 10),
                        textcoords="offset points", ha="left" if x == 0 else "right", fontsize=9)
    axs[0].set_xticks([0, 1], ["CC-SF own condition", "CC-MF own condition"])
    axs[0].set_xlim(-.25, 1.25)
    axs[0].set_title("(a) DoN 0.85: all 28 paired successful seeds")
    axs[0].set_ylabel("Offline risk-exceedance rate (%)")
    axs[0].set_ylim(bottom=0)
    for src, marker in [("cc_sf", "o"), ("cc_mf", "s")]:
        group = summary[summary.source == src].sort_values("don")
        axs[1].plot(group.don, group.mean_trial_rate_pct, color=COLORS[src],
                    marker=marker, label=LABELS[src], lw=1.8)
    group = summary[summary.source == "cc_sf"].sort_values("don")
    axs[1].set_xticks(group.don, [f"{r.don:.2f}\nn={int(r.n_trials)}"
                                  for r in group.itertuples()])
    axs[1].set_title("(b) Paired-success cohort at each DoN")
    axs[1].set_ylabel("Mean trial risk-exceedance rate (%)")
    axs[1].set_xlabel("DoN (n = paired successful seeds)")
    axs[1].set_ylim(bottom=0)
    axs[1].legend(frameon=False)
    for ax in axs:
        style(ax)
    fig.suptitle("Executed-pose surrogate | not the stage-1 planning-constraint violation rate", fontsize=13)
    fig.supxlabel("Each step uses its applied input to reconstruct one-step covariance from Sigma0.\n"
                  "Only solver-success steps; threshold epsilon + 1e-6. Lines pair identical initial seeds.",
                  fontsize=9)
    save(fig, "offline_risk_exceedance_paired")

    convergence = []
    for keys, group in e85[e85.kind == "boundary"].groupby(["source", "probe_radius"]):
        src, radius = keys
        for formulation in ("sf", "mf"):
            convergence.append({"source": src, "probe_radius": radius,
                                "formulation": formulation, "n_events": len(group),
                                "mean_j_pp": group[f"j_{formulation}_pp"].mean(),
                                "median_j_pp": group[f"j_{formulation}_pp"].median(),
                                "p95_j_pp": group[f"j_{formulation}_pp"].quantile(.95),
                                "float32_mean_j_pp": group[f"j_{formulation}_pp_float32"].mean()})
    pd.DataFrame(convergence).to_csv(OUT / "tables/probe_convergence.csv", index=False)
    print(json.dumps({"figure_directory": str(OUT / "figures"),
                      "paired_085": len(pivot),
                      "mean_trial_rate_pct": pivot.mean().to_dict()}, indent=2))


if __name__ == "__main__":
    main()
