"""Run the frozen CC-MF benchmark at DoN .80–.83 with five random seeds per width."""

import copy
import csv
import hashlib
import json
import os
import platform
import secrets
import subprocess
import sys
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

RESULTS = Path(__file__).resolve().parent
ORIGINAL = RESULTS.parent / "rectangle_don_sweep_50_20261105_current_sf"
SNAPSHOT = RESULTS / "source_snapshot"
WIDTHS = {0.80: "width_120mm", 0.81: "width_121p5mm", 0.82: "width_123mm",
          0.83: "width_124p5mm"}
TIMEZONE = ZoneInfo("Asia/Seoul")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    pending = path.with_suffix(".csv.tmp")
    with pending.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_snapshot():
    archived = read_json(ORIGINAL / "source_hashes.json")
    for name, expected in archived["files"].items():
        source = ORIGINAL / "source_snapshot" / name
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Archived source checksum differs: {source}")
        target = SNAPSHOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    write_json(RESULTS / "source_hashes.json", archived)
    return archived["aggregate_sha256"]


def prepare():
    source_hash = freeze_snapshot()
    seeds = secrets.SystemRandom().sample(range(100, 2**32), 20)
    configurations = []
    (RESULTS / "configs").mkdir()
    (RESULTS / "logs").mkdir()
    (RESULTS / "widths").mkdir()
    for i, (don, width) in enumerate(WIDTHS.items()):
        selected = seeds[5 * i:5 * (i + 1)]
        config_name = f"configs/{width}.yaml"
        archive_config = ORIGINAL / config_name
        config = yaml.safe_load(archive_config.read_text(encoding="utf-8"))
        [scenario] = [row for row in config["test_configs"] if row["optimizer_type"] == "cc_mf"]
        config["test_configs"] = [
            {**copy.deepcopy(scenario), "start_perturbation": {"seed": seed}}
            for seed in selected
        ]
        config["trials"] = 1
        (RESULTS / config_name).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        configurations.append(dict(don=don, width_m=config["rectangle_geometry"]["width"],
                                   width_id=width, seeds=selected, config=config_name,
                                   archived_config_sha256=digest(archive_config)))
    write_json(RESULTS / "run_manifest.json",
               dict(created_kst=datetime.now(TIMEZONE).isoformat(), planned_trials=20,
                    trials_per_don=5, optimizer_type="cc_mf",
                    source_original=str(ORIGINAL), source_aggregate_sha256=source_hash,
                    seed_selection="20 unique OS-random seeds drawn from [100, 2**32)",
                    debug=False, animation=False, configurations=configurations))
    acados = Path(os.environ["ACADOS_SOURCE_DIR"])
    write_json(RESULTS / "environment.json",
               dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                    packages={name: version(name) for name in
                              ("numpy", "torch", "casadi", "scipy", "PyYAML", "matplotlib")},
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                    acados_directory=str(acados),
                    acados_commit=subprocess.check_output(
                        ["git", "-C", str(acados), "rev-parse", "HEAD"], text=True).strip(),
                    acados_library_sha256=digest(acados / "lib/libacados.so"),
                    thread_count=torch.get_num_threads(),
                    archived_environment=read_json(ORIGINAL / "environment.json")))
    return configurations


def verify_trace(row, config, expected_parameters):
    path = Path(row["raw_trace_path"])
    meta = read_json(Path(row["raw_trace_metadata_path"]))
    if row["raw_trace_error"] or meta["raw_trace_sha256"] != digest(path):
        raise ValueError(f"Trace export invalid for seed {row['start_seed']}")
    if meta["outcome"]["status"] != row["status"] or meta["robot"]["width"] != config["width_m"]:
        raise ValueError(f"Trace outcome or width differs for seed {row['start_seed']}")
    if meta["optimizer_parameters"] != expected_parameters:
        changed = [key for key in set(meta["optimizer_parameters"]) | set(expected_parameters)
                   if meta["optimizer_parameters"].get(key) != expected_parameters.get(key)]
        raise ValueError(f"Optimizer parameters differ from the archive: {sorted(changed)}")
    with np.load(path, allow_pickle=False) as raw:
        if len(raw["poses"]) != len(raw["applied_inputs"]) + 1:
            raise ValueError(f"Incomplete pose trace for seed {row['start_seed']}")


def main():
    configurations = prepare()
    expected_parameters = read_json(ORIGINAL / "optimizer_parameters.json")["cc_mf"]
    all_trials, summaries = [], []
    for config in configurations:
        width, don = config["width_id"], config["don"]
        width_results = RESULTS / "widths" / width
        print(f"Running DoN {don:.2f} with random seeds {config['seeds']}", flush=True)
        write_json(RESULTS / "progress.json",
                   dict(status="running", completed=len(all_trials), current_don=don))
        command = [
            sys.executable, "-u", str(SNAPSHOT / "test_nmpc.py"), "-c",
            str(RESULTS / config["config"]), "--trials", "1",
            "--results-dir", str(width_results), "--no-use-debug", "--no-animation", "--no-plots",
        ]
        with (RESULTS / "logs" / f"{width}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=SNAPSHOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = read_csv(width_results / "trial_results.csv")
        if [int(row["start_seed"]) for row in rows] != config["seeds"]:
            raise ValueError(f"Recorded seeds differ from the random plan at DoN {don:.2f}")
        counts = Counter(row["status"] for row in rows)
        all_trials.extend({"don": don, "width_m": config["width_m"], "run_index": i, **row}
                          for i, row in enumerate(rows, 1))
        write_csv(RESULTS / "trial_results.csv", all_trials)
        if counts["error"] == len(rows):
            raise RuntimeError(f"All trials hit execution errors at DoN {don:.2f}; inspect the log.")
        for row in rows:
            if row["optimizer_type"] != "cc_mf" or row["status"] not in {
                "success", "failure", "timeout", "error"
            }:
                raise ValueError(f"Unexpected trial at DoN {don:.2f}: {row}")
            if row["status"] != "error":
                verify_trace(row, config, expected_parameters)
        original = [row for row in read_csv(ORIGINAL / "widths" / width / "trial_results.csv")
                    if row["optimizer_type"] == "cc_mf"]
        summaries.append(dict(don=don, width_mm=1000 * config["width_m"],
                              trial_count=len(rows), success=counts["success"],
                              failure=counts["failure"], timeout=counts["timeout"],
                              error=counts["error"], success_rate=counts["success"] / len(rows),
                              archived_success=sum(row["status"] == "success" for row in original),
                              archived_trial_count=len(original)))
        write_csv(RESULTS / "summary.csv", summaries)
        print(f"Completed DoN {don:.2f}: {dict(counts)}", flush=True)
    write_json(RESULTS / "progress.json", dict(status="complete", completed=len(all_trials)))
    print(f"Completed {len(all_trials)} trials: {RESULTS}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        progress_path = RESULTS / "progress.json"
        progress = read_json(progress_path) if progress_path.is_file() else {}
        progress.update(status="error", error=f"{type(error).__name__}: {error}")
        write_json(progress_path, progress)
        raise
