"""Offline audit of CC-SF feature selection in saved successful trials.

No optimizer or simulation is instantiated. The archived geometry and covariance
propagation are replayed. Adjacent cycle differences are explicitly not identified
as mathematical discontinuities: the nominal pose and covariance both change.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from utils.offline_cc_margin import _maze_polygons, _snapshot_imports
from utils.offline_feature_selectors import extract_feature_selectors


ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "benchmark_runs/rectangle_width070_130mm_step005_10trials_20261103"
OUTPUT = ROOT / "figures/paper_constraint_margins"


def risk_from_q(q, z, params):
    """Archived Log-Cholesky projected risk, evaluated in double precision."""
    variance = (np.exp(z[:, 4]) * q[:, 0]) ** 2 \
        + (np.exp(z[:, 5]) * q[:, 1] + z[:, 7] * q[:, 2]) ** 2 \
        + (np.exp(z[:, 6]) * q[:, 2]) ** 2
    gamma = params.get("chance_gamma")
    if gamma is None:
        gamma = NormalDist().inv_cdf(1 - params["chance_epsilon"])
    return np.clip(gamma * np.sqrt(np.maximum(variance, 0)),
                   params.get("risk_margin_floor", 0), params.get("risk_margin_cap", np.inf))


def local_q(gradient, theta):
    q = gradient.astype(float).copy()
    c, s = np.cos(theta), np.sin(theta)
    q[:, 0] = c * gradient[:, 0] + s * gradient[:, 1]
    q[:, 1] = -s * gradient[:, 0] + c * gradient[:, 1]
    return q


def make_wrapper(wrapper_module, config):
    length, width = (config["rectangle_geometry"][k] for k in ("length", "width"))
    vertices = torch.tensor([[-length / 2, -width / 2], [length / 2, -width / 2],
                             [length / 2, width / 2], [-length / 2, width / 2]], dtype=torch.float32)
    wrapper = wrapper_module.AugmentedPSDFWrapper(vertices, K_max=20, E_max=100, device="cpu")
    polygons = _maze_polygons(EXPERIMENT / "source_snapshot/sim/simulation_mpc.py")
    wrapper.update_edge_clusters([torch.tensor(p, dtype=torch.float32) for p in polygons],
                                 [torch.tensor(np.roll(p, -1, axis=0), dtype=torch.float32)
                                  for p in polygons])
    return wrapper


def read_trial(seed, stage):
    directory = EXPERIMENT / f"width_125mm/cc_sf/seed_{seed}"
    params = json.loads((directory / "optimizer_parameters.json").read_text())
    config = yaml.safe_load((directory / "config.yaml").read_text())
    checkpoint = [json.loads(s) for s in (directory / "animation_checkpoint.jsonl").read_text().splitlines()]
    states = np.array([r["optimized_traj"] for r in checkpoint[1:]])
    cache = OUTPUT / f"aggregate_cache_width125_stage{stage}/cc_sf_seed{seed}.npz"
    data = dict(np.load(cache, allow_pickle=False))
    return params, config, states, data


def replay_nominal(covariance, params, states, data, stage):
    p = SimpleNamespace(**params)
    robust = covariance.resolve_robust_horizon(p)
    dt = params["tf"] / params["horizon"]
    ids = np.flatnonzero(data["replay_valid"])
    z = states[ids - 1, stage, :8].copy()
    for row, i in enumerate(ids):
        xi = covariance.log_cholesky_from_covariance(covariance.initial_local_covariance(
            p, SimpleNamespace(_x=states[i, 0, :3])))
        for j in range(min(stage, robust)):
            v, omega = states[i - 1, j + 1, 8:10]
            xi = covariance.propagate_log_cholesky(xi, v, omega, 0.0, dt, p)
        z[row, 4:8] = xi
    return ids, z


def stats(values):
    v = np.abs(np.asarray(values))
    v = v[np.isfinite(v)]
    return {"n": len(v), "median_abs_mm": float(np.median(v) * 1000) if len(v) else None,
            "p95_abs_mm": float(np.percentile(v, 95) * 1000) if len(v) else None,
            "max_abs_mm": float(v.max() * 1000) if len(v) else None}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def probe_boundaries(wrapper, params, z, ids, features, data, seed):
    """Probe interpolated, separated boundaries, holding left-cycle xi fixed.

    This is a geometric diagnostic, not an observed robot trajectory. Sampling
    33 poses brackets selector changes; bisection narrows each bracket. Float32
    ties can change selectors, so near-zero risk changes remain in the output.
    Only intervals whose recorded endpoint canonical selectors differ are used.
    """
    pairs = [j for j in range(1, len(ids)) if ids[j] == ids[j - 1] + 1
             and .6 <= data["s"][ids[j - 1]] <= 1.2 and .6 <= data["s"][ids[j]] <= 1.2
             and features["canonical_id"][j] != features["canonical_id"][j - 1]
             and features["separated"][j] and features["separated"][j - 1]]
    if not pairs:
        return []
    grid = np.linspace(0, 1, 33)
    poses = np.array([z[j - 1, :3] + grid[:, None] * (z[j, :3] - z[j - 1, :3]) for j in pairs])
    selected = extract_feature_selectors(wrapper, torch.tensor(poses.reshape(-1, 3), dtype=torch.float32), verify=False)
    names = selected["canonical_id"].reshape(len(pairs), 33)
    phi = selected["phi"].reshape(len(pairs), 33)
    brackets = [(j, grid[k - 1], grid[k], names[row, k - 1])
                for row, j in enumerate(pairs) for k in range(1, 33)
                if names[row, k] != names[row, k - 1]
                and phi[row, k - 1] > 1e-5 and phi[row, k] > 1e-5]
    if not brackets:
        return []
    js = np.array([r[0] for r in brackets])
    lo, hi = (np.array([r[k] for r in brackets]) for k in (1, 2))
    left_names = np.array([r[3] for r in brackets])
    a, change = z[js - 1, :3], z[js, :3] - z[js - 1, :3]
    for _ in range(18):
        middle = (lo + hi) / 2
        f = extract_feature_selectors(wrapper, torch.tensor(a + middle[:, None] * change,
                                                           dtype=torch.float32), verify=False)
        same = f["canonical_id"] == left_names
        lo = np.where(same, middle, lo)
        hi = np.where(same, hi, middle)
    p_left, p_right = a + lo[:, None] * change, a + hi[:, None] * change
    f_left = extract_feature_selectors(wrapper, torch.tensor(p_left, dtype=torch.float32), verify=False)
    f_right = extract_feature_selectors(wrapper, torch.tensor(p_right, dtype=torch.float32), verify=False)
    q_left = local_q(f_left["world_gradient"], p_left[:, 2])
    q_right = local_q(f_right["world_gradient"], p_right[:, 2])
    rho_left = risk_from_q(q_left, z[js - 1], params)
    rho_right = risk_from_q(q_right, z[js - 1], params)
    result = []
    for k, j in enumerate(js):
        delta_phi = float(f_right["phi"][k] - f_left["phi"][k])
        result.append({"seed": seed, "cycle_before": int(ids[j - 1]), "cycle_after": int(ids[j]),
                       "s_before_m": data["s"][ids[j - 1]], "s_after_m": data["s"][ids[j]],
                       "interpolation_alpha": (lo[k] + hi[k]) / 2,
                       "left_id": str(f_left["canonical_id"][k]), "right_id": str(f_right["canonical_id"][k]),
                       "phi_boundary_mm": float(f_left["phi"][k]) * 1000,
                       "delta_phi_mm": delta_phi * 1000,
                       "delta_risk_fixed_cov_mm": (rho_right[k] - rho_left[k]) * 1000,
                       "delta_margin_fixed_cov_mm": (delta_phi - rho_right[k] + rho_left[k]) * 1000,
                       "q_translation_change_norm": np.linalg.norm(q_right[k, :2] - q_left[k, :2]),
                       "pose_bracket_width_m": np.linalg.norm(p_right[k, :2] - p_left[k, :2]),
                       "heading_bracket_width_rad": abs(p_right[k, 2] - p_left[k, 2]),
                       "left_risk_mm": rho_left[k] * 1000, "right_risk_mm": rho_right[k] * 1000})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, default=10)
    parser.add_argument("--probe", action="store_true", help="Fixed covariance boundary interpolation diagnostic")
    args = parser.parse_args()
    torch.set_num_threads(2)
    all_rows, summaries, trials, probes = [], [], [], []
    with _snapshot_imports(EXPERIMENT / "source_snapshot") as (module, covariance, _):
        for seed in (3, 6):
            params, config, states, data = read_trial(seed, args.stage)
            wrapper = make_wrapper(module, config)
            ids, z = replay_nominal(covariance, params, states, data, args.stage)
            features = extract_feature_selectors(wrapper, torch.tensor(z[:, :3], dtype=torch.float32))
            q = local_q(features["world_gradient"], z[:, 2])
            phi = features["phi"].reshape(-1)
            rho = risk_from_q(q, z, params)
            h = phi - rho - params["d_min"]
            error = float(np.max(np.abs(h - data["nominal"][ids])))
            if error > 1e-8:
                raise ValueError(f"Nominal replay differs from validated cache: {error}")
            n = len(data["s"])
            full = {k: np.full(n, np.nan) for k in ("phi", "risk", "h")}
            for key, value in (("phi", phi), ("risk", rho), ("h", h)):
                full[key][ids] = value
            switch = np.zeros(n, dtype=bool)
            pair_rows = []
            for j in range(1, len(ids)):
                prev, i = ids[j - 1], ids[j]
                if i != prev + 1:
                    continue
                changed = features["canonical_id"][j] != features["canonical_id"][j - 1]
                switch[i] = changed
                row = {"seed": seed, "cycle_before": int(prev), "cycle_after": int(i),
                       "s_before_m": data["s"][prev], "s_after_m": data["s"][i],
                       "switch": bool(changed),
                       "id_before": str(features["canonical_id"][j - 1]),
                       "id_after": str(features["canonical_id"][j]),
                       "separated_both": bool(features["separated"][j - 1] and features["separated"][j]),
                       "nominal_phi_before_mm": phi[j - 1] * 1000,
                       "nominal_phi_after_mm": phi[j] * 1000,
                       "nominal_risk_before_mm": rho[j - 1] * 1000,
                       "nominal_risk_after_mm": rho[j] * 1000,
                       "delta_nominal_margin_mm": (h[j] - h[j - 1]) * 1000,
                       "delta_phi_mm": (phi[j] - phi[j - 1]) * 1000,
                       "delta_risk_mm": (rho[j] - rho[j - 1]) * 1000,
                       "delta_qp_margin_mm": (data["postsolve_affine"][i] - data["postsolve_affine"][prev]) * 1000,
                       "predicted_pose_displacement_mm": np.linalg.norm(z[j, :2] - z[j - 1, :2]) * 1000,
                       "predicted_heading_change_deg": np.rad2deg(z[j, 2] - z[j - 1, 2]),
                       "midprogress": bool(.6 <= data["s"][prev] <= 1.2 and .6 <= data["s"][i] <= 1.2)}
                pair_rows.append(row)
            all_rows.extend(pair_rows)
            if args.probe:
                probes.extend(probe_boundaries(wrapper, params, z, ids, features, data, seed))
            for scope in ("full", "midprogress"):
                for category in ("all", "switch", "no_switch"):
                    rows = [r for r in pair_rows if (scope == "full" or r["midprogress"])
                            and (category == "all" or r["switch"] == (category == "switch"))]
                    summaries.append({"seed": seed, "scope": scope, "category": category,
                                      "nominal_delta": stats([r["delta_nominal_margin_mm"] / 1000 for r in rows]),
                                      "distance_delta": stats([r["delta_phi_mm"] / 1000 for r in rows]),
                                      "risk_delta": stats([r["delta_risk_mm"] / 1000 for r in rows])})
            trials.append((seed, data, full, switch))
            print(json.dumps({"seed": seed, "cache_replay_error_m": error, "switches": int(switch.sum())}), flush=True)
    output = OUTPUT / f"sf_feature_switch_audit_width125_stage{args.stage}"
    write_csv(output.with_suffix(".csv"), all_rows)
    if probes:
        write_csv(output.with_name(output.name + "_boundary_probe.csv"), probes)
    metadata = {"scope": "Both successful CC-SF width125 trials (seeds3,6)", "stage": args.stage,
                "interpretation": "Adjacent cycle differences mix feature selection, pose, covariance and motion; not isolated discontinuities.",
                "geometry": "Archived CPU float32, original CUDA float32; near-tie identities can differ.",
                "source_experiment": str(EXPERIMENT), "summary": summaries,
                "boundary_probe": {"definition": "Linear pose interpolation between adjacent nominal states with changed endpoint selectors, 33-point bracketing then 18 bisections; left-cycle covariance frozen; separated boundaries only, phi>10um. Not observed continuous-time motion.",
                                   "boundary_count": len(probes),
                                   "margin_jump": stats([r["delta_margin_fixed_cov_mm"] / 1000 for r in probes]),
                                   "risk_jump": stats([r["delta_risk_fixed_cov_mm"] / 1000 for r in probes])},
                "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), ROOT / "utils/offline_feature_selectors.py")}}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 2, figsize=(10.4, 7.0), sharex=True, sharey="row")
    for col, (seed, data, full, switch) in enumerate(trials):
        s = data["s"]
        axes[0, col].set_title(f"CC-SF, seed {seed}")
        axes[0, col].plot(s, full["h"] * 1000, color="#2C699A", lw=1.3, label="Nominal margin")
        axes[0, col].plot(s, data["postsolve_affine"] * 1000, color="#C27A2C", lw=1, ls="--", label="QP affine margin")
        axes[1, col].plot(s, full["phi"] * 1000, color="#2C699A", lw=1.3, label=r"Distance $\phi$")
        axes[1, col].plot(s, full["risk"] * 1000, color="#C27A2C", lw=1.3, ls="--", label=r"Projected risk $\rho$")
        axes[2, col].plot(s, np.r_[np.nan, np.diff(full["risk"])] * 1000, color="#4B5563", lw=.8)
        axes[2, col].scatter(s[switch], (np.r_[np.nan, np.diff(full["risk"])] * 1000)[switch],
                             color="#C27A2C", s=18, zorder=3, label="Feature selector changed")
        for ax in axes[:, col]:
            ax.axhline(0, color="#777777", lw=.6)
            ax.set_xlim(.6, 1.2)
            ax.grid(axis="y", color="#DEDEDE", lw=.5)
            for value in s[switch & (s >= .6) & (s <= 1.2)]:
                ax.axvline(value, color="#9AA0A6", lw=.5, alpha=.4, zorder=0)
            ax.legend(loc="upper right", frameon=True, framealpha=.9, fontsize=8)
        axes[2, col].set_xlabel(r"Current progress $s$ [m]")
    axes[0, 0].set_ylabel("Margin [mm]")
    axes[1, 0].set_ylabel("Distance / risk [mm]")
    axes[2, 0].set_ylabel(r"Cycle change $\Delta\rho$ [mm]")
    fig.suptitle("CC-SF feature-selection diagnostic | width 125 mm, prediction stage 10", y=.98, fontsize=12)
    fig.text(.5, .015, "Raw 0.1 s samples; vertical lines mark reconstructed selector changes.\n"
             "Pose and covariance vary between samples: these are not isolated boundary jumps.", ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, .055, 1, .96))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{ext}"), dpi=200)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
