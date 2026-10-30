"""Compare matched CC-MF navigation logs; never invokes an optimizer solve."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
BASELINE = PROJECT / "benchmark_runs/cc_mf_maze_5trials_20261029_diagnostics"
DT = 0.1
# A* reaches the center of the goal grid cell (cell width .0375 m).
# This matches every trial_results.distance_to_goal from the final logged pose.
GOAL = np.array([1.81875, 0.09375])
sys.path.insert(0, str(PROJECT))


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plain(value):
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def intervals(mask, minimum_samples=5):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [(int(a), int(b)) for a, b in zip(starts, ends)
            if b - a + 1 >= minimum_samples]


def stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return {"median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "max": float(values.max())} if len(values) else None


def load_source_covariance():
    spec = importlib.util.spec_from_file_location(
        "comparison_covariance", BASELINE / "source_snapshot/control/cc_covariance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from control.cc_mf_optimizer import CCMFOptimizerParam
    return module, CCMFOptimizerParam()


COVARIANCE, PARAM = load_source_covariance()


def summarize_controls(history, poses, outcome, limit=None):
    n = len(history) if limit is None else min(limit, len(history))
    h, p = history[:n], poses[:n]
    v = np.array([float(row["v"]) for row in h])
    omega = np.array([float(row["omega"]) for row in h])
    xy = np.array([[float(row["x"]), float(row["y"])] for row in p])
    post_xy = np.array([[float(row["x"]), float(row["y"])] for row in h])
    # Low-speed time near the goal is reported separately from intermediate stalls.
    goal_l1 = np.abs(xy - GOAL).sum(axis=1)
    far_low = (abs(v) < .01) & (goal_l1 > .10)
    near_low = (abs(v) < .01) & (goal_l1 <= .10)
    far_intervals = intervals(far_low)
    du = np.diff(omega)
    reversals = (omega[1:] * omega[:-1] < 0) & (abs(omega[1:]) >= .05) & (abs(omega[:-1]) >= .05)
    small_reversals = (omega[1:] * omega[:-1] < 0) & (abs(omega[1:]) >= .005) & (abs(omega[:-1]) >= .005)
    v_reversals = (v[1:] * v[:-1] < 0) & (abs(v[1:]) >= .001) & (abs(v[:-1]) >= .001)
    durations = [DT * (b - a + 1) for a, b in far_intervals]
    low_cycle_mask = np.zeros(n, bool)
    for a, b in far_intervals:
        low_cycle_mask[a:b + 1] = True
    status_ok = np.array([row["solver_success"].lower() == "true" for row in h])
    clearance = np.array([float(row["sdf_value"]) for row in p])
    return {
        "cycles": n,
        "duration_s": n * DT,
        "omega_reversal_count": int(reversals.sum()),
        "omega_reversals_per_s": float(reversals.sum() / (n * DT)),
        "omega_small_reversal_count_ge_0_005": int(small_reversals.sum()),
        "omega_small_reversals_per_s_ge_0_005": float(small_reversals.sum() / (n * DT)),
        "v_reversal_count_ge_0_001": int(v_reversals.sum()),
        "delta_omega_rms_rad_s": float(np.sqrt(np.mean(du**2))) if len(du) else 0.,
        "delta_omega_abs_mean_rad_s": float(np.mean(abs(du))) if len(du) else 0.,
        "delta_omega_saturation_count": int(np.isclose(abs(du), .35, atol=1e-5).sum()),
        "delta_omega_saturation_fraction": float(np.isclose(abs(du), .35, atol=1e-5).mean()) if len(du) else 0.,
        "mean_abs_v_m_s": float(np.mean(abs(v))),
        "far_goal_low_speed_s": float(far_low.sum() * DT),
        "far_goal_low_speed_fraction": float(far_low.mean()),
        "far_goal_low_speed_episode_count_ge_0_5s": len(far_intervals),
        "far_goal_low_speed_episode_time_s": float(sum(durations)),
        "longest_far_goal_low_speed_episode_s": float(max(durations, default=0.)),
        "near_goal_low_speed_s": float(near_low.sum() * DT),
        "min_psdf_clearance_mm": float(clearance.min() * 1000),
        "solver_failed_cycles": int((~status_ok).sum()),
        "last4s_net_displacement_m": float(np.linalg.norm(post_xy[-1] - post_xy[max(0, n - 41)])),
        "last4s_mean_abs_v_m_s": float(np.mean(abs(v[max(0, n - 40):]))),
        "last4s_omega_reversals": int(reversals[max(0, n - 40):].sum()),
        "last4s_omega_small_reversals_ge_0_005": int(small_reversals[max(0, n - 40):].sum()),
        "last4s_v_reversals_ge_0_001": int(v_reversals[max(0, n - 40):].sum()),
        "last4s_goal_l1_m": float(np.abs(post_xy[-1] - GOAL).sum()),
        "low_speed_episodes": [{"start_s": a * DT, "end_s": (b + 1) * DT,
            "duration_s": (b - a + 1) * DT,
            "net_displacement_m": float(np.linalg.norm(post_xy[b] - xy[a]))}
            for a, b in far_intervals],
    }


def telemetry_metrics(path):
    if not path.exists():
        return None
    sigma_factors, stage1_log_defects, eq, predicted_collision = [], [], [], []
    modes, times = {}, []
    with path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            modes[record["mode"]] = modes.get(record["mode"], 0) + 1
            d = record.get("diagnostics") or {}
            states = np.array(record["states"], float)
            u = np.array(record["inputs"][0], float)
            exact = COVARIANCE.propagate_log_cholesky(states[0, 4:8], u[0], u[1], 0., DT, PARAM)
            predicted_sigma = np.sqrt(COVARIANCE.covariance_from_log_cholesky(states[1, 4:8])[2, 2])
            exact_sigma = np.sqrt(COVARIANCE.covariance_from_log_cholesky(exact)[2, 2])
            times.append(record["time"])
            sigma_factors.append(float(exact_sigma / predicted_sigma))
            stage1_log_defects.append(float(np.max(abs(states[1, 4:7] - exact[:3]))))
            eq.append((d.get("solver") or {}).get("nlp_residuals", [None, None])[1])
            predicted_collision.append(d.get("minimum_exact_psdf_predicted_nodes", 0.) < 0.)
    return {"modes": modes, "cycles": len(times),
            "stage1_sigma_heading_underestimate_factor": stats(sigma_factors),
            "stage1_log_covariance_defect": stats(stage1_log_defects),
            "nlp_equality_residual": stats(eq),
            "last4s_stage1_sigma_heading_underestimate_factor": stats(sigma_factors[-40:]),
            "last4s_nlp_equality_residual": stats(eq[-40:]),
            "predicted_horizon_psdf_negative_cycles": int(sum(predicted_collision))}


def analyze_batch(label, folder, weight, baseline_trials=None):
    outcomes_path = folder / "results/trial_results.csv"
    if not outcomes_path.exists():
        return []
    records = []
    for outcome in read_csv(outcomes_path):
        seed = int(outcome["start_seed"])
        if seed not in range(3):
            continue
        trial = int(outcome["trial_number"])
        directory = folder / "results/runs/config01" / f"trial_{trial:04d}/data"
        hs, ps = list(directory.glob("trial_history*.csv")), list(directory.glob("pose_sdf*.csv"))
        if not hs or not ps:
            continue
        history, poses = read_csv(hs[0]), read_csv(ps[0])
        record = {"label": label, "weight": weight, "seed": seed,
                  "status": outcome["status"], "failure_reason": outcome["failure_reason"],
                  "final_time_s": float(outcome["final_time"]),
                  "distance_to_goal_m": float(outcome["distance_to_goal"]),
                  "initial_pose": [float(outcome[key]) for key in ("initial_x", "initial_y", "initial_theta")],
                  "full": summarize_controls(history, poses, outcome),
                  "telemetry": telemetry_metrics(folder / f"telemetry_trial{trial:02d}.jsonl"),
                  "history_path": str(hs[0]), "pose_path": str(ps[0])}
        if baseline_trials is not None:
            baseline = next(row for row in baseline_trials if row["seed"] == seed)
            common_n = min(len(history), baseline["full"]["cycles"])
            baseline_h = read_csv(Path(baseline["history_path"]))
            baseline_p = read_csv(Path(baseline["pose_path"]))
            record["same_initial_pose"] = bool(np.allclose(record["initial_pose"], baseline["initial_pose"], atol=1e-14, rtol=0))
            record["common_window_s"] = common_n * DT
            record["common_window_new"] = summarize_controls(history, poses, outcome, common_n)
            record["common_window_baseline"] = summarize_controls(baseline_h, baseline_p, outcome, common_n)
        records.append(record)
    return records


def row_flat(record, scope="full"):
    metrics = record[scope]
    return {key: record[key] for key in ("label", "weight", "seed", "status", "final_time_s", "distance_to_goal_m")}


def main():
    baseline = analyze_batch("baseline", BASELINE, 0.)
    batches = []
    for folder in sorted((ROOT / "batches").glob("*")):
        if not folder.is_dir():
            continue
        try:
            weight = float(folder.name.lstrip("r").replace("p", "."))
        except ValueError:
            continue
        records = analyze_batch(folder.name, folder, weight, baseline)
        if records:
            batches.append((weight, records))
    batches.sort(key=lambda item: item[0])
    records = baseline + [record for _, batch in batches for record in batch]
    flattened = []
    paired = []
    for record in records:
        row = row_flat(record)
        row.update({key: value for key, value in record["full"].items() if key != "low_speed_episodes"})
        telemetry = record["telemetry"]
        if telemetry:
            row["stage1_sigma_underestimate_median"] = telemetry["stage1_sigma_heading_underestimate_factor"]["median"]
            row["last4s_stage1_sigma_underestimate_median"] = telemetry["last4s_stage1_sigma_heading_underestimate_factor"]["median"]
            row["nlp_eq_residual_median"] = telemetry["nlp_equality_residual"]["median"]
        flattened.append(row)
        if "common_window_new" in record:
            b, n = record["common_window_baseline"], record["common_window_new"]
            paired.append({**row_flat(record), "same_initial_pose": record["same_initial_pose"],
                "common_window_s": record["common_window_s"],
                **{f"{prefix}_{metric}": values[metric]
                   for prefix, values in (("baseline", b), ("new", n))
                   for metric in ("omega_reversal_count", "omega_reversals_per_s", "delta_omega_rms_rad_s",
                                  "omega_small_reversals_per_s_ge_0_005",
                                  "delta_omega_saturation_fraction", "far_goal_low_speed_s")}})
    write_csv(ROOT / "comparison_metrics.csv", flattened)
    write_csv(ROOT / "paired_common_window_metrics.csv", paired)
    grouped = [baseline] + [batch for _, batch in batches]
    summary = []
    for group in grouped:
        if not group:
            continue
        total_time = sum(record["full"]["duration_s"] for record in group)
        total_reversals = sum(record["full"]["omega_reversal_count"] for record in group)
        transitions = sum(max(0, record["full"]["cycles"] - 1) for record in group)
        batch_summary = {"label": group[0]["label"], "weight": group[0]["weight"],
            "completed_trials": len(group), "goal_arrivals": sum(record["status"] == "success" for record in group),
            "stuck_failures": sum(record["failure_reason"].startswith("stuck_") for record in group),
            "total_simulation_s": total_time,
            "omega_reversal_count": total_reversals, "omega_reversals_per_s": total_reversals / total_time,
            "omega_small_reversal_count_ge_0_005": sum(record["full"]["omega_small_reversal_count_ge_0_005"] for record in group),
            "omega_small_reversals_per_s_ge_0_005": sum(record["full"]["omega_small_reversal_count_ge_0_005"] for record in group) / total_time,
            "delta_omega_rms_rad_s": float(np.sqrt(sum(record["full"]["delta_omega_rms_rad_s"]**2 * max(0, record["full"]["cycles"] - 1) for record in group) / transitions)),
            "far_goal_low_speed_s": sum(record["full"]["far_goal_low_speed_s"] for record in group),
            "far_goal_low_speed_fraction": sum(record["full"]["far_goal_low_speed_s"] for record in group) / total_time,
            "far_goal_low_speed_episode_count_ge_0_5s": sum(record["full"]["far_goal_low_speed_episode_count_ge_0_5s"] for record in group),
            "far_goal_low_speed_episode_time_s": sum(record["full"]["far_goal_low_speed_episode_time_s"] for record in group),
            "solver_failed_cycles": sum(record["full"]["solver_failed_cycles"] for record in group),
            "min_psdf_clearance_mm": min(record["full"]["min_psdf_clearance_mm"] for record in group)}
        common_records = [record for record in group if "common_window_new" in record]
        for prefix in ("common_window_baseline", "common_window_new"):
            if common_records:
                seconds = sum(record[prefix]["duration_s"] for record in common_records)
                count = sum(max(0, record[prefix]["cycles"] - 1) for record in common_records)
                batch_summary[f"{prefix}_s"] = seconds
                batch_summary[f"{prefix}_omega_reversals_per_s"] = sum(record[prefix]["omega_reversal_count"] for record in common_records) / seconds
                batch_summary[f"{prefix}_delta_omega_rms_rad_s"] = float(np.sqrt(sum(record[prefix]["delta_omega_rms_rad_s"]**2 * max(0, record[prefix]["cycles"] - 1) for record in common_records) / count))
            else:
                for key in ("s", "omega_reversals_per_s", "delta_omega_rms_rad_s"):
                    batch_summary[f"{prefix}_{key}"] = None
        summary.append(batch_summary)
    write_csv(ROOT / "batch_summary.csv", summary)
    result = {"definitions": {
        "baseline": str(BASELINE), "matched_seeds": [0, 1, 2], "dt_s": DT,
        "meaningful_reversal": "Adjacent omega signs opposite and each |omega| >=0.05 rad/s.",
        "small_reversal": "Adjacent omega signs opposite and each |omega| >=0.005 rad/s; catches residual chatter below the main threshold.",
        "deltaomega_rms": "RMS of changes between consecutive applied angular commands; excludes initial zero-to-command.",
        "low_speed": "|v|<0.01 m/s; episode needs >=0.5s. Total low-speed seconds counts individual steps too.",
        "far_goal": "L1 distance to A* goal cell center [1.81875,0.09375] >0.10m, using precontrol pose.",
        "common_window": "From t=0 through min(baseline steps,new steps) for each matched seed.",
        "covariance": "Exact same-source nonlinear step evaluated with optimized first input; no new solves.",
        "success_caveat": "Stuck criterion stops after <=3cm net displacement over4s; does not establish permanent deadlock.",
    }, "summary": summary, "trials": records}
    (ROOT / "comparison.json").write_text(json.dumps(plain(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    for record in records:
        print(record["label"], "seed", record["seed"], record["status"], record["final_time_s"],
              "flip/s", round(record["full"]["omega_reversals_per_s"], 3),
              "domegaRMS", round(record["full"]["delta_omega_rms_rad_s"], 3),
              "farLowSec", round(record["full"]["far_goal_low_speed_s"], 2))
    plot_comparison(grouped)
    plot_terminal_controls(grouped)


def plot_comparison(grouped):
    grouped = [group for group in grouped if group]
    labels = ["Before\nr=0" if group[0]["weight"] == 0 else f"After\nr={group[0]['weight']:g}" for group in grouped]
    x = np.arange(len(grouped))
    colors = ["#0072B2", "#D55E00", "#009E73"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metric_panels = [("omega_reversals_per_s", "Angular reversals / second (each |omega| >=0.05)", 1.),
                     ("delta_omega_rms_rad_s", "RMS angular command change [rad/s]", 1.),
                     ("far_goal_low_speed_fraction", "Low-speed time far from goal [%]", 100.)]
    ax = axes.flat[0]
    arrival = [sum(r["status"] == "success" for r in group) for group in grouped]
    totals = [len(group) for group in grouped]
    ax.bar(x, arrival, color="#56B4E9", width=.6)
    for i, (a, n) in enumerate(zip(arrival, totals)):
        ax.text(i, a + .08, f"{a}/{n}", ha="center")
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 3.7), yticks=[0, 1, 2, 3], title="Goal arrivals: matched seeds 0-2")
    ax.grid(axis="y", alpha=.2)
    for ax, (metric, title, scale) in zip(list(axes.flat)[1:], metric_panels):
        for seed, color in enumerate(colors):
            values = [next((r["full"][metric] * scale for r in group if r["seed"] == seed), np.nan)
                      for group in grouped]
            ax.plot(x, values, "o-", color=color, lw=1.3, label=f"Seed {seed}")
        ax.set(xticks=x, xticklabels=labels, title=title)
        ax.grid(axis="y", alpha=.2)
        ax.set_ylim(bottom=0)
    axes.flat[1].legend()
    fig.suptitle("CC-MF angular-change cost | rectangle / DD / maze | full logged trajectories", fontsize=14)
    fig.savefig(ROOT / "comparison.png", dpi=170)


def plot_terminal_controls(grouped):
    grouped = [group for group in grouped if group]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    colors = plt.cm.tab10(np.arange(len(grouped)))
    for index, group in enumerate(grouped):
        for record in group:
            seed = record["seed"]
            history = read_csv(Path(record["history_path"]))[-30:]
            times = np.array([float(row["time"]) for row in history])
            times -= times[-1]
            label = f"r={record['weight']:g} ({record['status']})"
            for row_index, (metric, scale) in enumerate((("omega", 1.), ("v", 1000.))):
                values = np.array([float(row[metric]) * scale for row in history])
                axes[row_index, seed].step(times, values, where="post", color=colors[index],
                                            lw=1.2, alpha=.85, label=label)
    for seed in range(3):
        axes[0, seed].set(title=f"Seed {seed}", ylabel="omega [rad/s]")
        axes[1, seed].set(xlabel="Time relative to this run's endpoint [s]", ylabel="v [mm/s]")
        for ax in axes[:, seed]:
            ax.axhline(0, color="black", lw=.5)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Terminal commands: reduced amplitude can retain alternating signs", fontsize=14)
    fig.savefig(ROOT / "terminal_controls.png", dpi=170)


if __name__ == "__main__":
    main()
