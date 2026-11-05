"""Plot only affine QP margins, aggregated across all successful trials.

python plot_constraint_margin_statistics.py --width-mm 125 --stage 10 \
    --s-range 0.6 1.2 --bin-width 0.02

No solver or simulation runs. Each native-progress bin first averages valid
samples within a trial, then equally weights the available trial means.
The band is between-trial sample SD (ddof=1), not a confidence interval.
Missing/inactive QP rows are never interpolated or replaced by a constant.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = ROOT / "benchmark_runs/rectangle_width070_130mm_step005_10trials_20261103"
METHODS = ("cc_sf", "cc_mf")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nullable(value):
    return float(value) if np.isfinite(value) else None


def successful_seeds(experiment, width_mm, method):
    seeds, population = [], []
    for folder in sorted((experiment / f"width_{width_mm:03d}mm" / method).glob("seed_*")):
        outcome = json.loads((folder / "outcome.json").read_text())
        seed = int(folder.name.split("_")[1])
        population.append({"seed": seed, "status": outcome["status"]})
        if outcome["status"] == "success":
            seeds.append(seed)
    if not seeds:
        raise ValueError(f"No successful trials for {method}")
    return sorted(seeds), population


def load_cached_trial(experiment, cache, width_mm, stage, method, seed):
    stem = cache / f"{method}_seed{seed}"
    if not stem.with_suffix(".json").exists() or not stem.with_suffix(".npz").exists():
        from utils.offline_cc_margin import load_trial
        import torch

        torch.set_num_threads(6)
        trial = load_trial(experiment, method, width_mm, seed, stage)
        arrays = {key: value.astype(str) if value.dtype == object else value
                  for key, value in trial.items() if isinstance(value, np.ndarray)}
        np.savez_compressed(stem.with_suffix(".npz"), **arrays)
        details = {key: trial[key] for key in ("method", "stage", "outcome", "metadata")}
        details["cache_source_helper"] = {
            "path": str(ROOT / "utils/offline_cc_margin.py"),
            "sha256": sha256(ROOT / "utils/offline_cc_margin.py"),
        }
        stem.with_suffix(".json").write_text(json.dumps(details, indent=2, allow_nan=False)+"\n")
    details = json.loads(stem.with_suffix(".json").read_text())
    allowed_helpers = {details.get(key, {}).get("sha256")
                       for key in ("cache_source_helper", "cache_compatible_helper")}
    if sha256(ROOT / "utils/offline_cc_margin.py") not in allowed_helpers:
        raise ValueError(f"Extractor changed without validated cache compatibility: {stem}")
    metadata = details["metadata"]
    if (metadata["method"], metadata["seed"], metadata["width_mm"], metadata["stage"]) \
            != (method, seed, width_mm, stage) or details["outcome"]["status"] != "success":
        raise ValueError(f"Incorrect cache cohort: {stem}")
    # Cache provenance is checked against the original input bytes, not just
    # the human-readable directory name. Shared snapshot hashes are small.
    for record in metadata["sources"].values():
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Archived source changed; refresh cache: {record['path']}")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return arrays, details, {"npz": str(stem.with_suffix(".npz")),
                            "npz_sha256": sha256(stem.with_suffix(".npz")),
                            "json": str(stem.with_suffix(".json")),
                            "json_sha256": sha256(stem.with_suffix(".json"))}


def aggregate(experiment, cache, width_mm, stage, edges):
    bin_rows, trial_rows, groups, provenance = [], [], [], []
    centers = (edges[:-1] + edges[1:]) / 2
    n_bins = len(centers)
    for method in METHODS:
        seeds, population = successful_seeds(experiment, width_mm, method)
        trial_means = np.full((len(seeds), n_bins), np.nan)
        records = []
        reference_lengths = []
        for trial_index, seed in enumerate(seeds):
            arrays, details, record = load_cached_trial(
                experiment, cache, width_mm, stage, method, seed)
            progress, values = arrays["s"], arrays["postsolve_affine"]
            success, active = arrays["solver_success"], arrays["nominal_active"]
            valid = success & active & np.isfinite(values)
            in_view = (progress >= edges[0]) & (progress <= edges[-1])
            assignments = np.searchsorted(edges, progress, side="right") - 1
            assignments[progress == edges[-1]] = n_bins - 1
            if np.any(np.isfinite(values) & ~success) or np.any(np.isfinite(values) & ~active):
                raise ValueError("An inactive/failed cycle has a finite QP residual")
            for j in range(n_bins):
                in_bin = in_view & (assignments == j)
                used = in_bin & valid
                if used.any():
                    trial_means[trial_index, j] = float(np.mean(values[used]))
                trial_rows.append({"method": method, "seed": seed, "bin": j,
                                   "s_left_m": float(edges[j]), "s_right_m": float(edges[j+1]),
                                   "s_center_m": float(centers[j]),
                                   "total_cycles": int(in_bin.sum()),
                                   "valid_qp_cycles": int(used.sum()),
                                   "inactive_cycles": int((in_bin & success & ~active).sum()),
                                   "failed_cycles": int((in_bin & ~success).sum()),
                                   "unavailable_active_cycles": int((in_bin & success & active & ~np.isfinite(values)).sum()),
                                   "trial_bin_mean_native": nullable(trial_means[trial_index, j]),
                                   "native_unit": details["metadata"]["units"]})
            checkpoint = Path(details["metadata"]["sources"]["checkpoint"]["path"])
            with checkpoint.open() as stream:
                reference = np.asarray(json.loads(next(stream))["global_path"], float)
            reference_lengths.append(float(np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()))
            records.append({"seed": seed, "cache": record, "reconstruction": details["metadata"]})
        means, stds = np.full(n_bins, np.nan), np.full(n_bins, np.nan)
        counts = np.isfinite(trial_means).sum(axis=0)
        for j in range(n_bins):
            available = trial_means[:, j][np.isfinite(trial_means[:, j])]
            if len(available):
                means[j] = np.mean(available)
            if len(available) >= 2:
                stds[j] = np.std(available, ddof=1)
            bin_rows.append({"method": method, "bin": j, "s_left_m": float(edges[j]),
                             "s_right_m": float(edges[j+1]), "s_center_m": float(centers[j]),
                             "successful_trials_total": len(seeds), "contributing_trials": int(counts[j]),
                             "mean_native": nullable(means[j]), "sample_std_native": nullable(stds[j]),
                             "mean_minus_std_native": nullable(means[j]-stds[j]),
                             "mean_plus_std_native": nullable(means[j]+stds[j]),
                             "native_unit": "m" if method == "cc_sf" else "standard_normal_quantile"})
        groups.append({"method": method, "seeds": seeds, "s": centers,
                       "mean": means, "std": stds, "n": counts})
        provenance.append({"method": method, "successful_seeds": seeds, "population": population,
                           "contributing_trials_min": int(counts.min()), "contributing_trials_max": int(counts.max()),
                           "empty_bins": int(np.sum(counts == 0)),
                           "bins_with_fewer_than_two_trials": int(np.sum(counts < 2)),
                           "reference_path_lengths_m": dict(zip(map(str, seeds), reference_lengths)),
                           "trials": records})
    return groups, bin_rows, trial_rows, provenance


def render(groups, output, width_mm, stage, edges, dpi):
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                         "axes.titlesize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    color = "#ce4a35"
    fig, axes = plt.subplots(2, 1, figsize=(7.05, 4.8), sharex=True)
    fig.subplots_adjust(left=.115, right=.97, top=.80, bottom=.22, hspace=.43)
    for index, (ax, group) in enumerate(zip(axes, groups)):
        scale = 1000 if group["method"] == "cc_sf" else 1
        mean, std = group["mean"] * scale, group["std"] * scale
        ax.fill_between(group["s"], mean-std, mean+std, color=color, alpha=.22, linewidth=0)
        ax.plot(group["s"], mean, color=color, linewidth=1.5)
        ax.axhline(0, color="#777777", linewidth=.7, linestyle=":")
        ax.grid(axis="y", color="#e8e8e8", linewidth=.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(.6)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        n_min, n_max = int(group["n"].min()), int(group["n"].max())
        coverage = str(n_min) if n_min == n_max else f"{n_min}–{n_max}"
        label = "CC-SF" if index == 0 else "CC-MF"
        ax.set_title(f"({chr(97+index)}) {label} · {len(group['seeds'])} successful trials · n(s) = {coverage}",
                     loc="left", pad=7)
        ax.set_ylabel("QP margin [mm]" if index == 0 else "QP margin [–]")
    axes[-1].set_xlim(edges[0], edges[-1])
    axes[-1].set_xlabel(r"Native progress $s$ [m]")
    fig.suptitle(f"Affine QP constraint margin · robot width {width_mm} mm · k = {stage}",
                 fontsize=10.5, y=.975)
    fig.legend([Line2D([], [], color=color, lw=1.5), Patch(color=color, alpha=.22)],
               ["Mean across trial-bin means", "±1 sample SD across trial-bin means"],
               loc="upper center", bbox_to_anchor=(.55, .93), frameon=False, ncol=2, fontsize=8)
    width_mm_bin = (edges[1]-edges[0]) * 1000
    fig.text(.115, .028,
             f"{width_mm_bin:g} mm bins; equal trial weights. n(s): contributing trials. No interpolation of missing rows.\n"
             "Only active, available QP rows from solver-success cycles; MF margin is before slack. Units differ.\n"
             "SD describes spread, not a confidence interval or min–max; a negative band is not a violation count.\n"
             "Success-conditioned summary on each trial's native reference path; spatial alignment is approximate.",
             fontsize=6.9, color="#444444", va="bottom")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix("."+suffix), dpi=dpi, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--width-mm", type=int, default=125)
    parser.add_argument("--stage", type=int, default=10)
    parser.add_argument("--s-range", type=float, nargs=2, default=[.6, 1.2])
    parser.add_argument("--bin-width", type=float, default=.02)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/paper_constraint_margins")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    low, high = args.s_range
    if high <= low or args.bin_width <= 0:
        parser.error("Require increasing s range and positive bin width")
    n_bins = int(round((high-low) / args.bin_width))
    if n_bins < 1 or not np.isclose(n_bins*args.bin_width, high-low, atol=1e-10, rtol=0):
        parser.error("s range must be an integer number of bins")
    edges = np.linspace(low, high, n_bins+1)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or output_dir / f"aggregate_cache_width{args.width_mm}_stage{args.stage}"
    cache.mkdir(parents=True, exist_ok=True)
    groups, bin_rows, trial_rows, provenance = aggregate(
        args.experiment.resolve(), cache, args.width_mm, args.stage, edges)
    name = (f"qp_margin_mean_std_width{args.width_mm}_stage{args.stage}"
            f"_s{low:g}_{high:g}_bin{args.bin_width*1000:g}mm").replace(".", "p")
    output = output_dir / name
    render(groups, output, args.width_mm, args.stage, edges, args.dpi)
    write_csv(output.with_suffix(".csv"), bin_rows)
    write_csv(output_dir / (name+"_trial_bins.csv"), trial_rows)
    details = {"width_mm": args.width_mm, "stage": args.stage,
               "s_range": args.s_range, "bin_width_m": args.bin_width,
               "definition": "Per-trial mean of eligible cycles within each native-s bin; equal-weight mean and sample SD (ddof=1) across available trial means.",
               "eligibility": "Final trial success; cycle solver_success; nominal row active; finite reconstructed affine residual.",
               "missing_policy": "No interpolation or constant substitution; empty bins remain missing; SD unavailable for n<2.",
               "sd_band": "mean +/- 1 sample SD, untruncated; not a CI or min-max envelope.",
               "conditioning": "Success-conditioned cohorts differ: all successful trials separately for each method; not matched-seed or population-level inference.",
               "alignment": "Native progress; repeated/backtracking samples averaged within bins; seed-dependent reference paths only approximately align spatial positions.",
               "groups": provenance, "script_sha256": sha256(Path(__file__))}
    output.with_suffix(".json").write_text(json.dumps(details, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"figure": str(output.with_suffix(".png")),
                      "coverage": [{k: v for k, v in item.items() if k != "trials"}
                                   for item in provenance]}, indent=2))


if __name__ == "__main__":
    main()
