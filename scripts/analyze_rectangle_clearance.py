#!/usr/bin/env python3
"""Calculate exact rectangle clearance from saved trials, entirely offline.

No controller or simulator is imported or instantiated. The saved standalone
``sim/start_perturbation.py`` supplies the same polygon/map-boundary distance
used to accept initial poses. Every initial and post-integration true pose is
included. Completed failed trials are included; interrupted trials are not.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


COMPLETED_STATUSES = {"success", "failure", "timeout", "error"}
TRIAL_FIELDS = (
    "suite_root", "experiment", "width_m", "nominal_don", "optimizer_type",
    "trial_number", "start_seed", "status", "included_in_primary_pool",
    "available", "unavailable_reason", "pose_samples", "clearance_mean_mm",
    "clearance_p05_mm", "clearance_min_mm", "zero_clearance_samples",
    "negative_boundary_samples", "raw_trace_path", "raw_trace_metadata_path",
    "reused", "reused_from_suite", "reused_from_trial_number",
    "origin_suite_manifest_sha256",
)
SUMMARY_FIELDS = (
    "suite_root", "experiment", "width_m", "nominal_don", "optimizer_type",
    "expected_trials", "recorded_trials", "unrecorded_trials",
    "completed_trials", "interrupted_trials", "available_completed_trials",
    "unavailable_completed_trials", "pooled_pose_samples",
    "clearance_mean_mm", "clearance_p05_mm", "clearance_min_mm",
    "zero_clearance_samples", "negative_boundary_samples", "coverage_complete",
)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def load_clearance_function(suite_root: Path):
    """Load one standalone archived helper, bypassing package __init__ files."""
    helper = suite_root / "source_snapshot" / "sim" / "start_perturbation.py"
    hashes = read_json(suite_root / "source_hashes.json")
    if hashes.get("schema_version") != 1 or hashes.get("algorithm") != "sha256":
        raise ValueError("Unsupported source_hashes.json")
    expected_hash = hashes["files"]["sim/start_perturbation.py"]
    if sha256(helper) != expected_hash:
        raise ValueError(f"Archived clearance helper hash mismatch: {helper}")
    source = helper.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(helper))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [item.name.split(".")[0] for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported = [(node.module or "").split(".")[0]]
        else:
            continue
        if any(name not in {"numpy", "secrets", "__future__"} for name in imported):
            raise ValueError(f"Archived clearance helper has unexpected imports: {helper}")
    module_name = "rectangle_clearance_" + sha256(helper)[:16]
    spec = importlib.util.spec_from_file_location(module_name, helper)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load archived clearance helper: {helper}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pose_clearance, receipt(helper)


def polygon_arrays(value, label: str, *, allow_empty=False) -> list[np.ndarray]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{label} must be a list of polygons")
    result = []
    for vertices in value:
        polygon = np.asarray(vertices, dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
            raise ValueError(f"{label} contains an invalid polygon shape")
        if not np.isfinite(polygon).all():
            raise ValueError(f"{label} contains non-finite vertices")
        edges = np.roll(polygon, -1, axis=0) - polygon
        cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
        if np.any(cross > 1e-12) and np.any(cross < -1e-12):
            raise ValueError(f"{label} contains a non-convex polygon")
        if not np.any(np.abs(cross) > 1e-12):
            raise ValueError(f"{label} contains a degenerate polygon")
        result.append(polygon)
    return result


def trace_paths(width_root: Path, trial: dict) -> tuple[Path, Path]:
    """Prefer local explicit paths, with a deterministic relocation fallback."""
    configuration = trial["configuration_id"]
    if not configuration or Path(configuration).name != configuration:
        raise ValueError("Invalid trial configuration_id")
    trial_number = int(trial["trial_number"])
    if trial_number < 1:
        raise ValueError("Invalid trial_number")
    local_data = width_root / "runs" / configuration / f"trial_{trial_number:04d}" / "data"

    def resolve(field: str, filename: str) -> Path:
        raw = trial.get(field, "").strip()
        if raw:
            explicit = Path(raw)
            if not explicit.is_absolute():
                explicit = width_root / explicit
            explicit = explicit.resolve()
            if explicit.is_relative_to(width_root) and explicit.exists():
                return explicit
        # The CSV may contain absolute paths from the original execution PC.
        relocated = local_data / filename
        if relocated.exists() or not raw:
            return relocated
        return explicit

    return resolve("raw_trace_path", "raw_trace.npz"), resolve(
        "raw_trace_metadata_path", "raw_trace_metadata.json"
    )


def calculate_trial(npz_path: Path, metadata_path: Path, trial: dict,
                    width_m: float, pose_clearance):
    metadata = read_json(metadata_path)
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported raw-trace metadata schema_version")
    trace_hash = sha256(npz_path)
    if metadata.get("raw_trace_sha256") != trace_hash:
        raise ValueError("Raw trace hash differs from raw_trace_metadata.json")
    if metadata["scenario"].get("optimizer_type") != trial["optimizer_type"]:
        raise ValueError("Trace optimizer differs from trial_results.csv")
    if metadata["start_perturbation"].get("seed") != int(trial["start_seed"]):
        raise ValueError("Trace start seed differs from trial_results.csv")
    robot = metadata["robot"]
    if robot.get("shape") != "rectangle":
        raise ValueError("Trace footprint is not rectangle")
    if not np.isclose(float(robot["width"]), width_m, rtol=0, atol=1e-12):
        raise ValueError("Trace width differs from suite manifest")
    if not np.isclose(float(robot["length"]), 0.150, rtol=0, atol=1e-12):
        raise ValueError("Trace rectangle length differs from 0.150 m")
    if metadata["outcome"].get("status") != trial["status"]:
        raise ValueError("Trace outcome differs from trial_results.csv")
    geometry = metadata["geometry"]
    bounds = np.asarray(geometry["bounds"], dtype=float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
        raise ValueError("Invalid map bounds")
    footprints = polygon_arrays(geometry["footprint_polygons"], "footprint_polygons")
    obstacles = polygon_arrays(geometry["obstacle_polygons"], "obstacle_polygons", allow_empty=True)
    with np.load(npz_path, allow_pickle=False) as archive:
        poses = np.asarray(archive["poses"], dtype=float)
        pose_times = np.asarray(archive["pose_times_s"], dtype=float)
        inputs = np.asarray(archive["applied_inputs"], dtype=float)
        solve_times = np.asarray(archive["solve_nlp_times_s"], dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) == 0 or not np.isfinite(poses).all():
        raise ValueError("poses must be a nonempty finite (n+1,3) array")
    if pose_times.shape != (len(poses),) or not np.isfinite(pose_times).all():
        raise ValueError("pose_times_s does not match poses")
    if not np.isclose(pose_times[0], 0.0, rtol=0, atol=1e-12) or np.any(np.diff(pose_times) <= 0):
        raise ValueError("pose_times_s must start at zero and strictly increase")
    if inputs.ndim != 2 or inputs.shape[0] != len(poses) - 1 or inputs.shape[1] < 2:
        raise ValueError("applied_inputs must match integrated poses")
    if not np.isfinite(inputs).all():
        raise ValueError("applied_inputs contains non-finite values")
    if solve_times.ndim != 1 or not np.isfinite(solve_times).all() or np.any(solve_times < 0):
        raise ValueError("Invalid solve_nlp_times_s")
    if len(solve_times) < len(inputs):
        raise ValueError("Fewer optimization cycles than applied inputs")
    initial = np.array([float(trial[key]) for key in ("initial_x", "initial_y", "initial_theta")])
    if not np.allclose(poses[0], initial, rtol=0, atol=1e-12):
        raise ValueError("Initial saved pose differs from trial_results.csv")
    clearances = np.fromiter(
        (pose_clearance(pose, footprints, obstacles, bounds) for pose in poses),
        dtype=float, count=len(poses),
    )
    if not np.isfinite(clearances).all():
        raise ValueError("Non-finite offline clearance")
    return clearances, {
        "trace": {"path": str(npz_path.resolve()), "sha256": trace_hash},
        "metadata": receipt(metadata_path),
    }


def clearance_statistics(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"clearance_mean_mm": "", "clearance_p05_mm": "", "clearance_min_mm": "",
                "zero_clearance_samples": "", "negative_boundary_samples": ""}
    return {
        "clearance_mean_mm": float(np.mean(values) * 1000),
        "clearance_p05_mm": float(np.percentile(values, 5) * 1000),
        "clearance_min_mm": float(np.min(values) * 1000),
        "zero_clearance_samples": int(np.count_nonzero(values == 0)),
        "negative_boundary_samples": int(np.count_nonzero(values < 0)),
    }


def write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(suite_roots: list[Path], output_dir: Path, *, strict=False) -> int:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a fresh --output-dir: {output_dir}")
    trials = []
    groups = {}
    expected_by_group = {}
    experiment_by_suite = {}
    sources = []
    seen_paths = set()
    seen_trials = set()
    trial_origins = set()
    trace_origins = set()
    expected_trials = 0
    for suite_root in suite_roots:
        manifest_path = suite_root / "suite_manifest.json"
        manifest = read_json(manifest_path)
        manifest_hash = sha256(manifest_path)
        if manifest.get("schema_version") != 1:
            raise ValueError(f"Unsupported suite manifest: {manifest_path}")
        widths = manifest["widths"]
        experiment_by_suite[str(suite_root)] = manifest["experiment"]
        expected_trials += len(widths) * len(manifest["methods"]) * int(manifest["trials_per_configuration"])
        pose_clearance, helper_source = load_clearance_function(suite_root)
        sources.extend([receipt(manifest_path), receipt(suite_root / "source_hashes.json"), helper_source])
        for width in widths:
            width_m = float(width["width_m"])
            nominal_don = width_m / 0.150
            reuse = width.get("reuse") or {}
            if reuse:
                hashes = read_json(suite_root / "source_hashes.json")
                if reuse["source_aggregate_sha256"] != hashes["aggregate_sha256"]:
                    raise ValueError("Reused trials and destination suite have different source snapshots")
                source_csv = Path(reuse["source_trial_results_path"]) if "source_trial_results_path" in reuse else (
                    Path(reuse["source_suite_root"]) / reuse["source_results_dir"] / "trial_results.csv"
                )
                if source_csv.exists() and sha256(source_csv) != reuse["source_trial_results_sha256"]:
                    raise ValueError("Reused source trial_results.csv hash mismatch")
                source_manifest = Path(reuse["source_suite_root"]) / "suite_manifest.json"
                if source_manifest.exists() and sha256(source_manifest) != reuse["source_suite_manifest_sha256"]:
                    raise ValueError("Reused source suite_manifest.json hash mismatch")
                sources.append({"reused_source_trial_results": str(source_csv),
                                "sha256": reuse["source_trial_results_sha256"],
                                "verified_locally": source_csv.exists()})
                sources.append({"reused_source_suite_manifest": str(source_manifest),
                                "sha256": reuse["source_suite_manifest_sha256"],
                                "verified_locally": source_manifest.exists()})
            width_root = (suite_root / width["results_dir"]).resolve()
            if not width_root.is_relative_to(suite_root):
                raise ValueError("Width results_dir must be inside the suite root")
            result_path = width_root / "trial_results.csv"
            for method in manifest["methods"]:
                group_key = (str(suite_root), width_m, method)
                groups.setdefault(group_key, [])
                if group_key in expected_by_group:
                    raise ValueError(f"Duplicate planned width/controller within suite: {group_key}")
                expected_by_group[group_key] = int(manifest["trials_per_configuration"])
            if not result_path.exists():
                sources.append({"path": str(result_path), "unavailable": "missing trial_results.csv"})
                continue
            sources.append(receipt(result_path))
            with result_path.open(newline="", encoding="utf-8-sig") as stream:
                width_trials = list(csv.DictReader(stream))
            print(f"Calculating {manifest['experiment']} width={width_m:.4f} m: "
                  f"{len(width_trials)} recorded trials", flush=True)
            for trial in width_trials:
                status = trial["status"]
                method = trial["optimizer_type"]
                if status not in COMPLETED_STATUSES | {"interrupted"}:
                    raise ValueError(f"Unexpected trial status {status!r}: {result_path}")
                if method not in manifest["methods"]:
                    raise ValueError(f"Unexpected optimizer {method!r}: {result_path}")
                key = (str(suite_root), width_m, method, int(trial["start_seed"]))
                if key[3] not in manifest["seeds"]:
                    raise ValueError(f"Unplanned start_seed: {key}")
                if key in seen_trials:
                    raise ValueError(f"Duplicate width/controller/seed within suite: {key}")
                seen_trials.add(key)
                if reuse:
                    if key[3] not in reuse["seeds"]:
                        raise ValueError(f"Reused trial has undeclared seed: {key}")
                    if trial.get("reused_from_suite") and trial["reused_from_suite"] != reuse["source_suite_root"]:
                        raise ValueError("CSV reuse source differs from width reuse declaration")
                    origin_suite = reuse["source_suite_manifest_sha256"]
                    origin_width = reuse["source_width_id"]
                else:
                    if trial.get("reused_from_suite"):
                        raise ValueError("CSV trial declares reuse absent from suite width manifest")
                    origin_suite = manifest_hash
                    origin_width = width["width_id"]
                trial_origins.add((origin_suite, origin_width, method, key[3]))
                trace_path, metadata_path = trace_paths(width_root, trial)
                physical_key = (trace_path.resolve(), metadata_path.resolve())
                if physical_key in seen_paths and not reuse:
                    raise ValueError(f"Duplicate underlying trace: {trace_path}")
                seen_paths.add(physical_key)
                row = {
                    "suite_root": str(suite_root), "experiment": manifest["experiment"],
                    "width_m": width_m, "nominal_don": nominal_don,
                    "optimizer_type": method, "trial_number": trial["trial_number"],
                    "start_seed": trial["start_seed"], "status": status,
                    "included_in_primary_pool": int(status in COMPLETED_STATUSES),
                    "available": 0, "unavailable_reason": "", "pose_samples": 0,
                    "raw_trace_path": str(trace_path), "raw_trace_metadata_path": str(metadata_path),
                    "reused": int(bool(reuse)),
                    "reused_from_suite": reuse.get("source_suite_root", ""),
                    "reused_from_trial_number": trial.get("reused_from_trial_number", ""),
                    "origin_suite_manifest_sha256": origin_suite,
                    **clearance_statistics(np.empty(0)),
                }
                values = None
                try:
                    values, input_sources = calculate_trial(
                        trace_path, metadata_path, trial, width_m, pose_clearance
                    )
                    row.update(available=1, pose_samples=len(values), **clearance_statistics(values))
                    sources.append(input_sources)
                    trace_origins.add((input_sources["trace"]["sha256"], input_sources["metadata"]["sha256"]))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    row["unavailable_reason"] = f"{type(error).__name__}: {error}"
                    if trial.get("raw_trace_error"):
                        row["unavailable_reason"] += f"; export: {trial['raw_trace_error']}"
                trials.append(row)
                groups.setdefault((str(suite_root), width_m, method), []).append((row, values))
    summaries = []
    for (suite_name, width_m, method), group in sorted(groups.items()):
        completed = [(row, values) for row, values in group if row["included_in_primary_pool"]]
        samples = [values for row, values in completed if row["available"]]
        pooled = np.concatenate(samples) if samples else np.empty(0)
        available = len(samples)
        expected = expected_by_group[(suite_name, width_m, method)]
        summaries.append({
            "suite_root": suite_name, "experiment": experiment_by_suite[suite_name],
            "width_m": width_m, "nominal_don": width_m / 0.150,
            "optimizer_type": method, "expected_trials": expected,
            "recorded_trials": len(group), "unrecorded_trials": expected - len(group),
            "completed_trials": len(completed), "interrupted_trials": len(group) - len(completed),
            "available_completed_trials": available,
            "unavailable_completed_trials": len(completed) - available,
            "pooled_pose_samples": len(pooled),
            "coverage_complete": int(available == expected),
            **clearance_statistics(pooled),
        })
    unavailable = sum(row["included_in_primary_pool"] and not row["available"] for row in trials)
    completed_count = sum(row["included_in_primary_pool"] for row in trials)
    output_dir.mkdir(parents=True)
    write_csv(output_dir / "clearance_trials.csv", TRIAL_FIELDS, trials)
    write_csv(output_dir / "clearance_summary.csv", SUMMARY_FIELDS, summaries)
    analysis_manifest = {
        "schema_version": 1,
        "analyzer": receipt(Path(__file__)),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "suite_roots": [str(path) for path in suite_roots],
        "expected_trials": expected_trials, "recorded_trials": len(trials),
        "completed_trials": completed_count,
        "cohort_trial_memberships": len(trials),
        "unique_source_trial_origins": len(trial_origins),
        "trial_origin_identity": "origin suite-manifest SHA256, width_id, controller and start seed; independent of file location",
        "unique_available_source_trace_records": len(trace_origins),
        "reused_trial_memberships": sum(row["reused"] for row in trials),
        "completed_trials_with_unavailable_clearance": unavailable,
        "expected_trial_count_reached": completed_count == expected_trials,
        "primary_statuses": sorted(COMPLETED_STATUSES),
        "sampling": "initial true pose plus every post-integration true pose",
        "aggregation": "pool individual samples across completed trials; do not average trial percentiles",
        "cohort_grouping": "each suite, width and controller separately; overlapping A/B cohorts are not merged",
        "clearance": "minimum exact convex polygon distance and signed map-boundary margin",
        "units": "input geometry m; CSV clearance mm",
        "zero_meaning": "obstacle contact or overlap; not a signed PSDF",
        "negative_meaning": "footprint crosses map boundary",
        "sources": sources,
    }
    with (output_dir / "analysis_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(analysis_manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Offline clearance saved to {output_dir}")
    print(f"Recorded {len(trials)}/{expected_trials} expected trials; "
          f"completed trials without valid clearance: {unavailable}")
    return 2 if strict and (unavailable or completed_count != expected_trials) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Fresh output directory (default: first suite/offline_clearance)")
    parser.add_argument("--strict", action="store_true",
                        help="Write reports, then exit 2 if expected trials or completed traces are missing")
    arguments = parser.parse_args()
    roots = [path.expanduser().resolve() for path in arguments.suite_roots]
    if len(set(roots)) != len(roots):
        parser.error("Duplicate suite roots")
    output = (arguments.output_dir or roots[0] / "offline_clearance").expanduser().resolve()
    try:
        return analyze(roots, output, strict=arguments.strict)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Offline clearance error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
