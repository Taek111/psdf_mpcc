"""Plot recorded geometry and first-guard timing; no synthetic pose probes."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from analyze_high_don import ROOT, OUTPUT, SUITE, save_csv


def main():
    records = list(csv.DictReader((OUTPUT / "trial_terminal_audit.csv").open()))
    failed = [r for r in records if r["status"] != "success"]
    save_csv("first_guard_times.csv", [{k: r[k] for k in (
        "don", "source", "seed", "paired_group", "first_guard_current_pose_time_s",
        "final_time_s", "terminal_true_clearance_mm", "pre_guard_previous_solver_success",
        "main_solver_failure_count", "guard_to_end_max_pose_change", "guard_to_end_max_applied_input",
    )} for r in failed])
    timing = []
    for don in (.83, .84, .85, .86, .87, "pooled_0.84_to_0.86"):
        for method in ("cc_sf", "cc_mf"):
            rows = [r for r in failed if r["source"] == method and
                    ((.84 <= float(r["don"]) <= .86) if isinstance(don, str) else float(r["don"]) == don)]
            t = np.array([float(r["first_guard_current_pose_time_s"]) for r in rows])
            timing.append({"don": don, "source": method, "failed_trials": len(rows),
                           "first_guard_before_10s": int(np.sum(t < 10)),
                           "first_guard_10_to_20s": int(np.sum((t >= 10) & (t < 20))),
                           "first_guard_20_to_40s": int(np.sum((t >= 20) & (t < 40))),
                           "first_guard_at_least_40s": int(np.sum(t >= 40)),
                           "median_first_guard_s": float(np.median(t)), "min_first_guard_s": float(t.min()),
                           "max_first_guard_s": float(t.max()),
                           "denominator": "failed trials only; successful trials are not inserted as censored observations"})
    save_csv("first_guard_time_summary.csv", timing)
    caches = {method: np.load(OUTPUT / f"selected_don0.85_{method}_seed10.npz") for method in ("cc_sf", "cc_mf")}
    meta_path = SUITE / "widths/width_127p5mm/runs/config03/trial_0011/data/raw_trace_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    geometry = meta["geometry"]
    colors = {"cc_sf": "#BA5A2A", "cc_mf": "#246B9D"}
    labels = {"cc_sf": "CC-SF", "cc_mf": "CC-MF"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12,
                         "axes.labelsize": 11, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, (ax_map, ax_gap) = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw={"width_ratios": [1, 1.35]})
    fig.subplots_adjust(left=.055, right=.985, top=.84, bottom=.22, wspace=.25)
    gap_position = ax_gap.get_position()
    ax_gap.set_position([gap_position.x0, .48, gap_position.width, .36])
    fig.suptitle("DoN 0.85 · matched seed 10: recorded SF stop and MF passage", y=.975, fontsize=15)
    fig.text(.5, .915, "Median SF failure time among the 17 MF-only successful seed pairs; tie broken by seed ID.",
             ha="center", color="#505050", fontsize=10)
    for polygon in geometry["obstacle_polygons"]:
        ax_map.add_patch(Polygon(polygon, closed=True, facecolor="#E1E3E5", edgecolor="#ADB0B5", linewidth=.7))
    ax_map.axvline(0, color="#272727", linewidth=1.2)
    for method, cache in caches.items():
        keep = cache["times"] <= 10
        pose = cache["poses"][keep]
        ax_map.plot(pose[:, 0], pose[:, 1], color=colors[method], lw=1.8,
                    marker=".", markersize=2.6, label=labels[method])
    footprint = np.array(geometry["footprint_polygons"][0])
    for method, t, alpha in (("cc_sf", 4.7, .18), ("cc_mf", 3.1, .08), ("cc_mf", 4.5, .08)):
        cache = caches[method]
        index = int(np.flatnonzero(np.isclose(cache["times"], t, rtol=0, atol=1e-9))[0])
        pose = cache["poses"][index]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        world = footprint @ np.array([[c, -s], [s, c]]).T + pose[:2]
        ax_map.add_patch(Polygon(world, closed=True, facecolor=colors[method], alpha=alpha, edgecolor="none"))
        ax_map.add_patch(Polygon(world, closed=True, fill=False, edgecolor=colors[method],
                                linewidth=1.5, linestyle="-" if method == "cc_sf" else "--"))
        ax_map.plot(*pose[:2], marker="s" if method == "cc_sf" else "o", color=colors[method], markersize=5)
    stop = caches["cc_sf"]["poses"][47]
    ax_map.annotate("SF: first guard at 4.7 s\nBoundary intrusion: 0.368 mm", xy=stop[:2], xytext=(.018, .686),
                    fontsize=9.5, color=colors["cc_sf"], ha="left",
                    arrowprops={"arrowstyle": "->", "color": colors["cc_sf"], "lw": 1})
    ax_map.annotate("MF: 4.5 s", xy=caches["cc_mf"]["poses"][45, :2], xytext=(.19, .46),
                    color=colors["cc_mf"], fontsize=9.5,
                    arrowprops={"arrowstyle": "->", "color": colors["cc_mf"], "lw": 1})
    ax_map.text(.01, .71, "x = 0 map boundary", fontsize=9, color="#444444")
    ax_map.text(.015, .437, "MF footprint outlines: 3.1 and 4.5 s", fontsize=8.5, color=colors["cc_mf"])
    ax_map.set(xlim=(-.02, .30), ylim=(.43, .735), xlabel="World x (m)", ylabel="World y (m)")
    ax_map.set_aspect("equal")
    ax_map.set_title("(a) First-corner geometry and actual footprints", loc="left", pad=14)
    ax_map.legend(loc="upper right", frameon=True, framealpha=.95, fontsize=9)
    ax_map.grid(color="#EDEDED", lw=.5)
    for method, cache in caches.items():
        keep = cache["times"] <= 10
        ax_gap.plot(cache["times"][keep], 1000 * cache["true_clearances_m"][keep],
                    color=colors[method], lw=1.6, marker="o", markersize=2.4,
                    label=labels[method])
    ax_gap.axhline(0, color="#242424", lw=1.0, label="Contact / map-boundary limit: 0")
    ax_gap.axhline(.05, color="#7B5C83", lw=1.0, linestyle=(0, (4, 3)), label=r"Guard threshold $d_{col}$: 0.05 mm")
    ax_gap.axvline(4.7, color=colors["cc_sf"], lw=1, linestyle=(0, (2, 3)), alpha=.8)
    ax_gap.annotate("4.6 → 4.7 s\n0.326 → −0.368 mm", xy=(4.7, -.368358), xytext=(5.05, 5.2),
                    fontsize=10, color=colors["cc_sf"],
                    arrowprops={"arrowstyle": "->", "color": colors["cc_sf"], "lw": 1})
    ax_gap.scatter([6.7], [-.368358], marker="x", s=45, color=colors["cc_sf"], zorder=5)
    ax_gap.text(6.85, .45, "SF ends: 6.7 s", fontsize=9, color=colors["cc_sf"])
    ax_gap.text(7.2, 10.1, "MF continues; goal at 55.8 s", fontsize=9.5, color=colors["cc_mf"])
    ax_gap.set(xlim=(0, 10), ylim=(-.85, 11.1), ylabel="Clearance (mm)")
    ax_gap.set_title("(b) Actual clearance over the first 10 seconds", loc="left", pad=14)
    ax_gap.grid(color="#E8E8E8", linewidth=.6)
    # A separate actual-time detail keeps every main-plot sample visible.
    detail = fig.add_axes([gap_position.x0, .22, gap_position.width, .15])
    sf = caches["cc_sf"]
    k = (sf["times"] >= 4.3) & (sf["times"] <= 6.7)
    detail.plot(sf["times"][k], sf["true_clearances_m"][k] * 1000, color=colors["cc_sf"], lw=1.2, marker="o", ms=2.5)
    detail.axhline(0, color="#242424", lw=.9)
    detail.axhline(.05, color="#7B5C83", lw=1, linestyle=(0, (4, 3)))
    detail.axvline(4.7, color=colors["cc_sf"], lw=.8, linestyle=(0, (2, 3)))
    detail.set(xlim=(4.3, 6.7), ylim=(-.45, .40), xticks=[4.5, 5, 5.5, 6, 6.5], yticks=[-.4, 0, .3],
               xlabel="Recorded simulation time (s)", ylabel="SF detail (mm)")
    detail.tick_params(labelsize=9)
    detail.set_title("Recorded SF stopping detail: threshold and zero are distinct", fontsize=9, pad=8, loc="left")
    detail.annotate(r"$d_{col}=0.05$ mm (dashed)", xy=(6.15, .05), xytext=(5.45, .25),
                    fontsize=8.5, color="#7B5C83", arrowprops={"arrowstyle": "->", "color": "#7B5C83", "lw": .7})
    detail.text(5.08, -.15, "0: contact / map-boundary limit", color="#242424", fontsize=8.5)
    detail.grid(color="#EAEAEA", lw=.5)
    fig.text(.055, .095,
             "SF: previous solve reported success; plant integration crosses the boundary at 4.7 s. "
             "The next 20 guard steps apply zero input, then the trial fails.", fontsize=10)
    fig.text(.055, .055,
             "Points are actual 0.1 s records; lines only connect samples. Negative clearance means map-boundary intrusion; "
             "zero means obstacle contact/overlap. No continuous-time guarantee.", fontsize=9, color="#505050")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"selected_pair_don085_seed10.{extension}", dpi=240, facecolor="white")
    plt.close(fig)
    print(OUTPUT / "selected_pair_don085_seed10.png")


if __name__ == "__main__":
    main()
