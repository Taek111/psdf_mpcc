"""Plot actual recorded-time residuals; no pose interpolation or solver run.

The probability values were evaluated offline at recorded post-integration
poses using one-step reconstructed covariance. They are not native optimizer
stage-1 residual logs. This script reads no local-probe event table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "benchmark_runs/rectangle_constraint_reliability_20261107"
OUT = ANALYSIS / "recorded_signed_residuals_20261108"
COLORS = {"cc_sf": "#2864A5", "cc_mf": "#C27620"}
NAMES = {"cc_sf": "CC-SF", "cc_mf": "CC-MF"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_trace(row):
    cache_path = ANALYSIS / "cache" / f"don0.85_{row.source}_seed{row.seed}.npz"
    raw_path, meta_path = ROOT / row.trace, ROOT / row.metadata
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(raw_path, allow_pickle=False) as raw, np.load(cache_path, allow_pickle=False) as cached:
        c = {key: cached[key].copy() for key in cached.files}
        np.testing.assert_array_equal(c["poses"], raw["poses"])
        np.testing.assert_array_equal(c["times"], raw["pose_times_s"])
        np.testing.assert_array_equal(c["applied_inputs"], raw["applied_inputs"])
    if digest(raw_path) != meta["raw_trace_sha256"]:
        raise ValueError(f"Raw trace hash mismatch: {raw_path}")
    time = c["times"][1:]
    epsilon = float(meta["optimizer_parameters"]["chance_epsilon"])
    risk = c["p_sf"] if row.source == "cc_sf" else c["r_mf"]
    raw_residual = 100 * (epsilon - risk)
    eligible = np.array([s["success"] and s.get("solver_called", True) is not False
                         for s in meta["solver_status_infos"]], dtype=bool)
    np.testing.assert_array_equal(eligible, c["eligible"])
    valid = eligible & np.isfinite(raw_residual)
    changed = c["canonical_id"][1:] != c["canonical_id"][:-1]
    if not (len(time) == len(risk) == len(eligible) == len(changed)):
        raise ValueError("Recorded pose, probability, status and switch counts differ")
    if not np.allclose(np.diff(c["times"]), .1, atol=1e-10, rtol=0):
        raise ValueError("Unexpected sample interval")
    plotted = np.where(valid, raw_residual, np.nan)
    table = pd.DataFrame({
        "source": row.source, "seed": row.seed,
        "step": np.arange(len(time)), "time_s": time,
        "previous_recorded_time_s": c["times"][:-1],
        "x_m": c["poses"][1:, 0], "y_m": c["poses"][1:, 1],
        "heading_rad": c["poses"][1:, 2],
        "epsilon": epsilon, "own_risk": risk,
        "own_residual_pp": raw_residual, "plotted_residual_pp": plotted,
        "solver_called": [s.get("solver_called", True) for s in meta["solver_status_infos"]],
        "solver_success": eligible, "plotted": valid,
        "feature_id_previous": c["canonical_id"][:-1],
        "feature_id_current": c["canonical_id"][1:],
        "feature_id_changed": changed,
    })
    for j in range(4):
        table[f"reconstructed_xi1_{j}"] = c["xi1"][:, j]
    return table, c, meta, [cache_path, raw_path, meta_path]


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_figure(traces, seed, window, name, title, detail=False):
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.145, top=.86, hspace=.34)
    visible = []
    for frame in traces.values():
        subset = frame[(frame.time_s >= window[0] - 1e-9) & (frame.time_s <= window[1] + 1e-9)]
        visible.extend(subset.loc[subset.plotted, "own_residual_pp"].tolist())
    low = np.floor(min(0, min(visible)) / 2) * 2 - 1
    high = 11
    for ax, source in zip(axes, ("cc_sf", "cc_mf")):
        frame = traces[source]
        selected = frame[(frame.time_s >= window[0] - 1e-9) & (frame.time_s <= window[1] + 1e-9)]
        ax.set_ylim(low, high)
        ax.set_xlim(*window)
        ax.axhspan(low, 0, color="#F2F3F5", zorder=0)
        ax.axhline(0, color="#222222", lw=1.15, zorder=2)
        ax.plot(selected.time_s, selected.plotted_residual_pp,
                color=COLORS[source], lw=1.1 if not detail else 1.5,
                marker="o", markersize=1.7 if not detail else 3.4,
                markeredgewidth=0, zorder=3)
        switches = selected[selected.feature_id_changed & selected.plotted]
        # A hollow diamond marks the recorded endpoint at which the feature ID
        # differs from the prior sample. It does not identify the sub-step time.
        ax.scatter(switches.time_s, switches.plotted_residual_pp, marker="D",
                   s=15 if not detail else 31, facecolor="white", edgecolor="#333333",
                   linewidth=.65, zorder=4)
        excluded = selected[~selected.plotted]
        if len(excluded):
            for value in excluded.time_s:
                ax.axvspan(value - .05, value + .05, color="#D8DCE0", alpha=.65, zorder=1)
            ax.plot(excluded.time_s, np.full(len(excluded), .92), "x", color="#555555",
                    transform=ax.get_xaxis_transform(), markersize=5)
            if not detail:
                ax.annotate(f"Excluded solve at {excluded.time_s.iloc[0]:.1f} s",
                            (excluded.time_s.iloc[0], .92), xycoords=("data", "axes fraction"),
                            xytext=(12, -8), textcoords="offset points", fontsize=8,
                            ha="left", va="top", color="#555555")
        formula = r"$b_{\mathrm{SF}}=\epsilon-p_\star$" if source == "cc_sf" else r"$b_{\mathrm{MF}}=\epsilon-R$"
        panel = "(a)" if source == "cc_sf" else "(b)"
        ax.set_title(f"{panel} {NAMES[source]} trajectory / own condition    {formula}",
                     loc="left", fontsize=10, pad=7)
        ax.set_ylabel("Signed residual b (pp)")
        ax.grid(axis="y", color="#E3E6E8", lw=.55, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(.015, .03, "b < 0: condition exceeded", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8, color="#555555")
        if detail:
            ax.set_xticks(np.arange(window[0], window[1] + .5, 1))
        else:
            ax.set_xticks(np.r_[np.arange(0, 51, 10), 54])
    axes[-1].set_xlabel("Recorded simulation time (s)", labelpad=8)
    fig.suptitle(f"{title} | DoN 0.85 | paired successful seed {seed}", y=.985, fontsize=13)
    legend = [Line2D([], [], marker="o", linestyle="-", color="#555555", markersize=3,
                     linewidth=1, label="Recorded 0.1 s sample"),
              Line2D([], [], marker="D", linestyle="none", markerfacecolor="white",
                     markeredgecolor="#333333", markersize=5,
                     label="Feature ID changed since preceding sample"),
              Line2D([], [], color="#222222", lw=1.15, label="b = 0")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.53, .95), ncol=3,
               frameon=False, fontsize=8.5)
    fig.text(.09, .025,
             "Offline reevaluation at recorded poses with one-step reconstructed covariance; not native solver residuals.\n"
             "Open diamonds indicate a change within the preceding 0.1 s; lines only connect recorded samples.",
             fontsize=8.5, ha="left", va="bottom")
    save(fig, name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    metrics_path = ANALYSIS / "tables/per_trial_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[np.isclose(metrics.don, .85)]
    candidates, selection_paths = [], []
    for seed, group in metrics.groupby("seed"):
        duration = {}
        for row in group.itertuples():
            path = ANALYSIS / "cache" / f"don0.85_{row.source}_seed{seed}.npz"
            selection_paths.append(path)
            with np.load(path, allow_pickle=False) as z:
                duration[row.source] = float(z["times"][-1])
        if set(duration) != {"cc_sf", "cc_mf"}:
            raise ValueError("Expected common-success paired seed")
        candidates.append({"seed": seed, "sf_duration_s": duration["cc_sf"],
                           "mf_duration_s": duration["cc_mf"],
                           "pair_mean_duration_s": np.mean(list(duration.values())),
                           "between_method_duration_gap_s": abs(duration["cc_sf"] - duration["cc_mf"])})
    selection = pd.DataFrame(candidates)
    median = selection.pair_mean_duration_s.median()
    selection["distance_from_median_s"] = abs(selection.pair_mean_duration_s - median)
    # Round numerical clock accumulation only for ranking equivalent durations.
    selection["ranking_distance_s"] = selection.distance_from_median_s.round(8)
    selection = selection.sort_values(["ranking_distance_s", "between_method_duration_gap_s", "seed"])
    selection.to_csv(OUT / "seed_selection.csv", index=False)
    seed = int(selection.seed.iloc[0])
    traces, caches, metas, sources = {}, {}, {}, [metrics_path, Path(__file__), *selection_paths]
    for row in metrics[metrics.seed == seed].itertuples():
        frame, cache, meta, paths = load_trace(row)
        traces[row.source], caches[row.source], metas[row.source] = frame, cache, meta
        sources.extend(paths)
    np.testing.assert_allclose(caches["cc_sf"]["poses"][0], caches["cc_mf"]["poses"][0], atol=1e-12, rtol=0)
    full = pd.concat(traces.values(), ignore_index=True)
    full.to_csv(OUT / "recorded_samples.csv", index=False)
    full[full.feature_id_changed].to_csv(OUT / "detected_switch_intervals.csv", index=False)
    draw_figure(traces, seed, (0, 54), "signed_residual_recorded_full",
                "Signed residuals at recorded driving states")
    draw_figure(traces, seed, (30, 36), "signed_residual_recorded_30_36s",
                "Signed residuals: recorded 30-36 s interval", detail=True)
    summary = {
        "selected_seed": seed, "don": .85, "paired_success_candidates": len(selection),
        "selection_rule": "Nearest paired mean traversal to the 28-pair median; ties: smaller inter-method traversal gap, then seed",
        "median_pair_mean_traversal_s": median,
        "detail_window_s": [30, 36],
        "detail_selection": "Illustrative readable interval with zero crossings in both traces; full trace supplied and not clipped",
        "scope": "Own formulation on own recorded post-integration trajectory; not isolated formulation comparison",
        "covariance": "Each step resets configured Sigma0 and propagates the stored applied input once; varies across time, not stored planning Sigma1",
        "switch_timing": "ID difference over [previous_recorded_time_s,time_s]; mark at endpoint, exact sub-step time unknown",
        "eligibility": "Finite own risk and solver_called=true, success=true; excluded steps are NaN gaps",
        "no_initial_risk": "Initial t=0 pose retained only for first switch comparison; risk samples start at t=.1",
        "local_probe_used": False, "interpolated_pose_used": False,
        "simulation_executed": False, "solver_executed": False,
        "metrics_not_in_main_figure": ["P05", "violation rate", "absolute residual change", "local probe"],
        "traces": {},
        "caption_en": "Signed probability-scale residuals along the recorded CC-SF and CC-MF trajectories at DoN=0.85, seed 9. Residuals are reevaluated at 0.1 s recorded poses using one-step reconstructed covariance. Open diamonds mark samples whose canonical nearest-feature ID differs from the preceding sample; the zero line marks the respective unrelaxed condition boundary. Each controller is evaluated with its own formulation. These offline residuals are not logged stage-1 optimizer residuals.",
        "caption_ko": "DoN 0.85, 동일 시작 seed 9의 CC-SF/CC-MF 실제 기록 자세에서 자체 formulation의 signed residual을 재평가했다. 0.1초 표본과 한 step 재구성 공분산을 사용했다. 빈 마름모는 직전 표본과 최근접 feature ID가 다른 시점이며 정확한 전환 시각을 뜻하지 않는다. 검은 0선 아래는 해당 조건 초과다. 저장된 stage-1 solver residual은 아니다.",
        "sources": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in dict.fromkeys(sources)],
    }
    for method, frame in traces.items():
        summary["traces"][method] = {
            "recorded_postintegration_samples": len(frame), "plotted_samples": int(frame.plotted.sum()),
            "omitted_times_s": frame.loc[~frame.plotted, "time_s"].tolist(),
            "detected_switches_total": int(frame.feature_id_changed.sum()),
            "detected_switches_plotted": int((frame.feature_id_changed & frame.plotted).sum()),
            "last_time_s": float(frame.time_s.iloc[-1]),
            "minimum_plotted_b_pp": float(frame.loc[frame.plotted, "own_residual_pp"].min()),
        }
    summary["outputs"] = [{"path": str(p.relative_to(OUT)), "sha256": digest(p)}
                          for p in sorted(OUT.iterdir()) if p.is_file() and p.suffix in (".csv", ".png", ".svg", ".pdf")]
    (OUT / "figure_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "selected_seed": seed, "traces": summary["traces"]}, indent=2))


if __name__ == "__main__":
    main()
