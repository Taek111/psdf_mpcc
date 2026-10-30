"""Run four matched CC-MF cases and use the standard animation writer once."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
BASELINE = ROOT / "benchmark_runs/cc_mf_maze_5trials_20261029_diagnostics"
CASES = ((0.0, "r0"), (0.03, "r0p03"), (0.1, "r0p1"), (1.0, "r1p0"))
sys.path.insert(0, str(ROOT))


def worker(config_path, results_path):
    import torch
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    from test_nmpc import main
    sys.argv = ["test_nmpc.py", "-c", str(config_path), "--trials", "1",
                "--results-dir", str(results_path), "--no-plots"]
    main()


def main():
    import yaml

    (BASE / "source_snapshot").mkdir(exist_ok=False)
    (BASE / "configs").mkdir(exist_ok=False)
    sources = [ROOT / "test_nmpc.py", ROOT / "AGENTS.md"]
    for directory in ("control", "models", "planning", "sim", "utils", "config"):
        sources.extend(path for path in (ROOT / directory).rglob("*")
                       if path.is_file() and path.suffix in (".py", ".yaml")
                       and "__pycache__" not in path.parts)
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = BASE / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (BASE / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    summary = []
    for weight, label in CASES:
        config = yaml.safe_load((BASELINE / "diagnostic_config.yaml").read_text())
        config["cc_mf"]["r_delta_omega"] = weight
        config["output_suffix"] = f"rdelta_{label.removeprefix('r')}"
        # dt=0.1s and the default writer's 10fps: preserve every control step.
        config["defaults"].update(generate_animation=True, generate_plots=False, frame_skip=1)
        config_path = BASE / "configs" / f"{label}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        results_path = BASE / "results" / label
        log_path = BASE / f"{label}.log"
        print(f"Starting r_delta_omega={weight}, seed=0, one trial", flush=True)
        with log_path.open("w") as handle:
            process = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                      "--worker", str(config_path), str(results_path)],
                                     cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        outcome_path = results_path / "trial_results.csv"
        if outcome_path.exists():
            with outcome_path.open(newline="") as handle:
                outcome, = list(csv.DictReader(handle))
        else:
            outcome = {"status": "process_error", "failure_reason": str(process.returncode)}
        animations = [line.removeprefix("Animation saved: ").strip()
                      for line in log_path.read_text().splitlines()
                      if line.startswith("Animation saved: ")]
        record = {
            "r_delta_omega": weight, "seed": 0, "status": outcome["status"],
            "final_time_s": outcome.get("final_time"),
            "failure_reason": outcome.get("failure_reason"),
            "initial_pose": [outcome.get(key) for key in ("initial_x", "initial_y", "initial_theta")],
            "animation_path": animations[-1] if animations else None,
            "process_return_code": process.returncode,
            "frame_skip": 1, "writer_fps": 10,
            "results_path": str(results_path), "config_path": str(config_path),
        }
        summary.append(record)
        (BASE / "animation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    (BASE / "source_changed_during_runs.json").write_text(json.dumps([
        key for key, value in hashes.items()
        if not (ROOT / key).exists()
        or hashlib.sha256((ROOT / key).read_bytes()).hexdigest() != value
    ], indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, metavar=("CONFIG", "RESULTS"))
    args = parser.parse_args()
    if args.worker:
        worker(*(Path(path) for path in args.worker))
    else:
        main()
