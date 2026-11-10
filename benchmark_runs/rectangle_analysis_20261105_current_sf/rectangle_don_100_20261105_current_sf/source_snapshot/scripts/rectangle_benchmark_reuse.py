"""Validate and copy an explicitly selected subset of a completed suite.

No controller imports or simulation calls occur here. Source CSVs and traces
are never edited; the destination owns portable copies of selected traces.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

import yaml


PROVENANCE_FIELDS = (
    "reused_from_suite", "reused_from_trial_results", "reused_from_trial_number",
)
CONFIG_FIELDS = (
    "configuration_id", "maze_type", "optimizer_type", "robot_shape",
    "dynamics_type", "path_planner",
)
SUMMARY_FIELDS = CONFIG_FIELDS + (
    "recorded_trials", "completed_trials", "successful_trials", "failed_trials",
    "interrupted_trials", "success_rate", "successful_solver_steps",
    "solver_time_mean_s", "solver_time_p95_s",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list | tuple, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def describe_reuse(source_root: Path, widths: list[dict], trials: int, methods: tuple) -> dict:
    source_root = source_root.resolve()
    manifest = read_json(source_root / "suite_manifest.json")
    state = read_json(source_root / "suite_state.json")
    if manifest.get("experiment") != "don" or state.get("status") != "complete":
        raise ValueError("--reuse-results requires a completed first (don) experiment")
    if manifest.get("plan_only") or manifest.get("methods") != list(methods):
        raise ValueError("Reuse source is a prepared-only suite or uses different controllers/order")
    if manifest.get("use_debug") is not False or manifest.get("computation_time_scope") != "solve_nlp":
        raise ValueError("Reuse source must use debug-off solve_nlp timing")
    if not set(range(trials)).issubset(manifest.get("seeds", [])):
        raise ValueError("Reuse source lacks the requested seed range")
    overlap = [
        (target, origin) for target in widths for origin in manifest["widths"]
        if target["width_m"] == origin["width_m"]
    ]
    if len(overlap) != 1 or overlap[0][0]["nominal_don"] != 0.75:
        raise ValueError("Expected exactly the first experiment's DoN=0.75 overlap")
    target, origin = overlap[0]
    source_results = source_root / origin["results_dir"]
    if not source_results.resolve().is_relative_to(source_root):
        raise ValueError("Reuse source results directory escapes its suite root")
    source_csv = source_results / "trial_results.csv"
    reuse = {
        "source_suite_root": str(source_root),
        "source_width_id": origin["width_id"],
        "source_results_dir": origin["results_dir"],
        "seeds": list(range(trials)),
        "source_aggregate_sha256": manifest["source_aggregate_sha256"],
        "source_suite_manifest_sha256": sha256(source_root / "suite_manifest.json"),
        "source_trial_results_sha256": sha256(source_csv),
    }
    target["reuse"] = reuse
    return reuse


def validate_reuse_environment(root: Path, reuse: dict) -> None:
    source = Path(reuse["source_suite_root"])
    hashes = read_json(root / "source_hashes.json")
    if hashes["aggregate_sha256"] != reuse["source_aggregate_sha256"]:
        raise ValueError("Reuse requires exactly the same frozen source hashes")
    for filename in ("optimizer_parameters.json", "environment.json"):
        if read_json(root / filename) != read_json(source / filename):
            raise ValueError(f"Reuse requires exactly matching {filename}")


def percentile95(values: list[float]):
    if not values:
        return ""
    ordered = sorted(values)
    index = (len(ordered) - 1) * 0.95
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def materialize_reused_width(root: Path, width: dict, methods: tuple) -> None:
    reuse = width["reuse"]
    source_root = Path(reuse["source_suite_root"])
    source_results = source_root / reuse["source_results_dir"]
    source_csv = source_results / "trial_results.csv"
    if sha256(source_csv) != reuse["source_trial_results_sha256"]:
        raise ValueError("Reuse source trial_results.csv changed after planning")
    fields, all_rows = read_csv(source_csv)
    seeds = set(reuse["seeds"])
    rows = [row for row in all_rows if int(row["start_seed"]) in seeds]
    expected = {(method, seed) for method in methods for seed in seeds}
    actual = [(row["optimizer_type"], int(row["start_seed"])) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Reuse source must contain exactly one row per selected method and seed")
    source_config = yaml.safe_load((source_results / "run_manifest.yaml").read_text(encoding="utf-8"))["config"]
    target_config = yaml.safe_load((root / width["config_path"]).read_text(encoding="utf-8"))
    # The repetition count changes from 100 to 50; all experimental settings match.
    source_config = {key: value for key, value in source_config.items() if key != "trials"}
    target_protocol = {key: value for key, value in target_config.items() if key != "trials"}
    if source_config != target_protocol:
        raise ValueError("Reuse source and destination width protocols differ")

    traces = []
    selected_by_key = {}
    for row in rows:
        if row["status"] not in ("success", "failure", "timeout", "error"):
            raise ValueError("Interrupted or unknown outcomes cannot be reused")
        method, seed = row["optimizer_type"], int(row["start_seed"])
        config_id = f"config{methods.index(method) + 1:02d}"
        trial_number = int(row["trial_number"])
        if row["configuration_id"] != config_id or trial_number != seed + 1:
            raise ValueError("Reuse source trial/configuration IDs do not match the seed protocol")
        if row.get("raw_trace_error") or not row.get("raw_trace_path") or not row.get("raw_trace_metadata_path"):
            raise ValueError("Selected reuse trial has a raw-trace export error or missing trace reference")
        relative_run = Path("runs") / config_id / f"trial_{trial_number:04d}"
        trace = source_results / relative_run / "data" / "raw_trace.npz"
        metadata = source_results / relative_run / "data" / "raw_trace_metadata.json"
        meta = read_json(metadata)
        trace_hash = sha256(trace)
        if meta.get("raw_trace_sha256") != trace_hash:
            raise ValueError(f"Reuse trace SHA256 does not match metadata: {trace}")
        if meta.get("computation_time", {}).get("scope") != "solve_nlp":
            raise ValueError("Reuse trace uses a different timing boundary")
        if float(meta.get("robot", {}).get("width", -1)) != width["width_m"]:
            raise ValueError("Reuse trace robot width does not match target width")
        if meta.get("start_perturbation", {}).get("seed") != seed:
            raise ValueError("Reuse trace seed does not match trial CSV")
        traces.append((row, relative_run, trace, metadata, trace_hash))
        selected_by_key[(method, seed)] = row

    timing_files = []
    times_by_config = {f"config{index + 1:02d}": [] for index in range(len(methods))}
    counts = {key: 0 for key in expected}
    seen_steps = set()
    for path in sorted(source_results.glob("computation_times_*.csv")):
        time_fields, source_times = read_csv(path)
        selected_times = [row for row in source_times if int(row["start_seed"]) in seeds]
        for row in selected_times:
            key = (row["optimizer_type"], int(row["start_seed"]))
            if key not in selected_by_key:
                raise ValueError("Reuse timing CSV has an unexpected controller/seed")
            step_key = (*key, int(row["timestep"]))
            if step_key in seen_steps:
                raise ValueError("Reuse timing CSV has duplicate step samples")
            seen_steps.add(step_key)
            elapsed = float(row["solver_computation_time_s"])
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Reuse timing CSV has invalid elapsed time")
            counts[key] += 1
            times_by_config[row["configuration_id"]].append(elapsed)
        timing_files.append((path, time_fields, selected_times))
    if len(timing_files) != len(methods):
        raise ValueError("Expected one timing CSV per reused controller")
    for key, row in selected_by_key.items():
        if counts[key] != int(row["successful_solver_steps"]):
            raise ValueError("Reuse timing sample counts do not match selected trial outcomes")

    destination = root / width["results_dir"]
    destination.mkdir()
    trace_manifest = []
    for row, relative_run, trace, metadata, trace_hash in traces:
        output_run = destination / relative_run
        output_data = output_run / "data"
        output_data.mkdir(parents=True)
        shutil.copy2(trace, output_data / trace.name)
        shutil.copy2(metadata, output_data / metadata.name)
        if sha256(output_data / trace.name) != trace_hash:
            raise ValueError("Copied reuse trace failed SHA256 verification")
        trace_manifest.append({
            "configuration_id": row["configuration_id"], "seed": int(row["start_seed"]),
            "source_trace": str(trace), "copied_trace": str(relative_run / "data" / trace.name),
            "raw_trace_sha256": trace_hash, "metadata_sha256": sha256(metadata),
        })
        row.update(
            output_root_dir=str(output_run.resolve()),
            raw_trace_path=str((output_data / trace.name).resolve()),
            raw_trace_metadata_path=str((output_data / metadata.name).resolve()),
            reused_from_suite=str(source_root), reused_from_trial_results=str(source_csv),
            reused_from_trial_number=row["trial_number"],
        )
    output_fields = fields + [name for name in PROVENANCE_FIELDS if name not in fields]
    write_csv(destination / "trial_results.csv", output_fields, rows)
    for path, time_fields, selected_times in timing_files:
        for row in selected_times:
            row.update(
                reused_from_suite=str(source_root), reused_from_trial_results=str(source_csv),
                reused_from_trial_number=row["trial_number"],
            )
        write_csv(destination / path.name, time_fields + list(PROVENANCE_FIELDS), selected_times)

    summaries = []
    for index, method in enumerate(methods, 1):
        selected = [row for row in rows if row["optimizer_type"] == method]
        identity = {name: selected[0][name] for name in CONFIG_FIELDS}
        successes = sum(row["status"] == "success" for row in selected)
        times = times_by_config[f"config{index:02d}"]
        summaries.append({
            **identity, "recorded_trials": len(selected), "completed_trials": len(selected),
            "successful_trials": successes, "failed_trials": len(selected) - successes,
            "interrupted_trials": 0, "success_rate": successes / len(selected),
            "successful_solver_steps": len(times),
            "solver_time_mean_s": statistics.fmean(times) if times else "",
            "solver_time_p95_s": percentile95(times),
        })
    write_csv(destination / "summary.csv", SUMMARY_FIELDS, summaries)
    run_manifest = yaml.safe_load((source_results / "run_manifest.yaml").read_text(encoding="utf-8"))
    run_manifest["source_cli"] = run_manifest.get("cli")
    run_manifest.update(
        trials_per_configuration=len(seeds), config=target_config,
        cli={"execution": "reused_subset_no_simulation"}, reused_from=reuse,
    )
    (destination / "run_manifest.yaml").write_text(yaml.safe_dump(run_manifest, sort_keys=False), encoding="utf-8")
    provenance = {
        "schema_version": 1, **reuse, "selected_trial_rows": len(rows),
        "raw_traces": "copied_byte_identically_into_destination", "traces": trace_manifest,
        "source_timing_csv_sha256": {path.name: sha256(path) for path, _, _ in timing_files},
    }
    (destination / "reuse_manifest.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
