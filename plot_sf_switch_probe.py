"""Plot a labelled offline geometry probe, not a new controller experiment."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import torch

from analyze_sf_feature_switches import (
    EXPERIMENT, OUTPUT, make_wrapper, read_trial, replay_nominal,
    risk_from_q, local_q, _snapshot_imports,
)
from utils.offline_feature_selectors import extract_feature_selectors


def main():
    torch.set_num_threads(2)
    stem = OUTPUT / "sf_feature_switch_audit_width125_stage10"
    with stem.with_name(stem.name + "_boundary_probe.csv").open() as stream:
        events = list(csv.DictReader(stream))
    peak = max(events, key=lambda r: abs(float(r["delta_margin_fixed_cov_mm"])))
    seed, cycle = int(peak["seed"]), int(peak["cycle_after"])
    params, config, states, data = read_trial(seed, 10)
    with _snapshot_imports(EXPERIMENT / "source_snapshot") as (module, covariance, _):
        wrapper = make_wrapper(module, config)
        ids, z = replay_nominal(covariance, params, states, data, 10)
        j = int(np.flatnonzero(ids == cycle)[0])
        a, dz = z[j - 1], z[j] - z[j - 1]
        center = float(peak["interpolation_alpha"])
        # A narrow sweep isolates the selected boundary from nearby switches.
        relative_heading = np.linspace(-.05, .05, 401)
        alpha = center + np.deg2rad(relative_heading) / dz[2]
        scan = np.repeat(a[None, :], len(alpha), axis=0)
        scan[:, :3] = a[:3] + alpha[:, None] * dz[:3]
        f = extract_feature_selectors(wrapper, torch.tensor(scan[:, :3], dtype=torch.float32))
        risk = risk_from_q(local_q(f["world_gradient"], scan[:, 2]), scan, params)
        margin = f["phi"] - risk - params["d_min"]
        mf = wrapper.forward_mf_icdf(torch.tensor(scan, dtype=torch.float64),
                                     params["chance_epsilon"], params["d_min"])
        mf_margin = mf.residual.numpy()
        # Repeating the same probe with right-cycle covariance checks sensitivity
        # to the arbitrary fixed-covariance endpoint choice.
        right_cov = scan.copy()
        right_cov[:, 4:8] = z[j, 4:8]
        right_risk = risk_from_q(local_q(f["world_gradient"], scan[:, 2]), right_cov, params)
        right_margin = f["phi"] - right_risk - params["d_min"]
    switches = np.flatnonzero(f["canonical_id"][1:] != f["canonical_id"][:-1]) + 1
    if len(switches) != 1:
        raise ValueError(f"Selected narrow sweep contains {len(switches)} switches, expected one")
    k = int(switches[0])
    meta = {"selected_event": peak,
            "selection_rule": "Largest absolute fixed-left-covariance SF margin change among 33 detected separated boundaries in the audited intervals",
            "scope": "Width125, successful SF seeds3/6, stage10, s0.6..1.2; linear interpolation of nominal poses, frozen local covariance. MF evaluated on identical SF-generated probe states, not MF controller output.",
            "heading_sweep_deg": [-.05, .05],
            "sf_adjacent_boundary_delta_mm": float((margin[k] - margin[k - 1]) * 1000),
            "sf_boundary_delta_right_covariance_mm": float((right_margin[k] - right_margin[k - 1]) * 1000),
            "mf_adjacent_boundary_delta": float(mf_margin[k] - mf_margin[k - 1]),
            "mf_max_adjacent_delta": float(np.max(np.abs(np.diff(mf_margin)))),
            "sf_phi_boundary_delta_mm": float((f["phi"][k] - f["phi"][k - 1]) * 1000),
            "sf_q_heading_before_m": float(f["local_q"][k - 1, 2]),
            "sf_q_heading_after_m": float(f["local_q"][k, 2]),
            "all_mf_active": bool(mf.required.numpy().all()),
            "all_mf_finite": bool(np.isfinite(mf_margin).all()),
            "limitations": [
                "Detected brackets are not an exhaustive count: same-ID endpoint intervals and narrow multiple transitions can be missed.",
                "33 brackets come from 28 adjacent-cycle intervals in two trials, not 33 independent trials.",
                "This local comparison does not establish global MF continuity, solver feasibility, or a cause of safe stops.",
                "Float32 boundary localization does not justify nanometre precision; focus on millimetre-scale residual changes.",
            ]}
    out = OUTPUT / "sf_feature_boundary_probe_width125"
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    with out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_heading_deg", "interpolation_alpha", "phi_mm", "risk_mm",
                         "sf_margin_mm", "mf_quantile_residual", "sf_margin_right_cov_mm", "feature_id"])
        writer.writerows(zip(relative_heading, alpha, f["phi"] * 1000, risk * 1000,
                             margin * 1000, mf_margin, right_margin * 1000, f["canonical_id"]))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw={"width_ratios": [.85, 1.1, 1.1]})
    colors = {3: "#2C699A", 6: "#C27A2C"}
    for i, trial_seed in enumerate((3, 6)):
        vals = np.sort([abs(float(r["delta_margin_fixed_cov_mm"])) for r in events if int(r["seed"]) == trial_seed])
        offsets = np.linspace(-.14, .14, len(vals))
        axs[0].scatter(i + offsets, vals, s=24, color=colors[trial_seed], alpha=.8)
        axs[0].plot([i - .22, i + .22], [np.median(vals)] * 2, color="#333333", lw=1.5)
        axs[0].text(i, 6.4, f"n = {len(vals)}", ha="center", fontsize=9)
    axs[0].set(xticks=[0, 1], xticklabels=["Seed 3", "Seed 6"], ylim=(-.15, 6.9),
               ylabel="Absolute SF boundary change [mm]", title="(a) Detected separated boundaries")
    axs[1].plot(relative_heading, margin * 1000, color="#2C699A", lw=1.8)
    axs[1].set(ylabel="SF constraint margin [mm]", title="(b) Largest detected SF change", ylim=(-.5, 7.5))
    axs[1].annotate("5.93 mm", xy=(0, 3.6), xytext=(.019, 4.3), fontsize=10,
                    arrowprops={"arrowstyle": "->", "color": "#444444"})
    axs[2].plot(relative_heading, mf_margin, color="#C27A2C", lw=1.8)
    axs[2].set(ylabel="MF inverse-CDF residual [–]", title="(c) MF on the same probe states")
    spread = float(np.ptp(mf_margin))
    axs[2].set_ylim(float(mf_margin.min() - .15 * spread), float(mf_margin.max() + .15 * spread))
    axs[2].yaxis.set_major_locator(MaxNLocator(4))
    axs[2].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    for ax in axs[1:]:
        ax.set_xlabel("Heading offset from boundary [deg]")
        ax.set_xlim(-.05, .05)
        ax.set_xticks([-.05, 0, .05])
        ax.axvline(0, color="#999999", ls=":", lw=.8)
        if ax.get_ylim()[0] <= 0 <= ax.get_ylim()[1]:
            ax.axhline(0, color="#777777", lw=.7)
    for ax in axs:
        ax.grid(axis="y", color="#DDDDDD", lw=.5)
    fig.suptitle("Feature-boundary probe | width 125 mm, stage 10", y=.98, fontsize=12)
    fig.text(.5, .025, "Offline pose interpolation with fixed covariance; not a solver run. Panels (b,c): seed 6, cycles 94–95.\n"
             "Native SF and MF units differ. Panel (a) covers detected boundaries in s = 0.6–1.2 m, successful SF trials only.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, .12, 1, .93), w_pad=2)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=220)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
