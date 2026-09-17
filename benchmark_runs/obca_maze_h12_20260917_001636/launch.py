"""Run one corrected OBCA trial at horizon 12 using the installed Python 3.8 solvers."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time


RESULTS = Path(__file__).resolve().parent
PROJECT = RESULTS.parents[1]
TEMP_ROOT = Path("/dev/shm/psdf_trials_5_20260916_2320")
sys.path.insert(0, str(TEMP_ROOT / "runtime"))
sys.path.insert(0, str(PROJECT))
os.environ.update(
    MPLBACKEND="Agg",
    MPLCONFIGDIR=str(TEMP_ROOT / "matplotlib"),
    TMPDIR=str(TEMP_ROOT / "work"),
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    ACADOS_SOURCE_DIR="/home/taekwon/projects/acados",
)
os.environ["LD_LIBRARY_PATH"] = (
    "/home/taekwon/projects/acados/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
)
os.chdir(TEMP_ROOT / "work")

# Match the CLI action provided by Python 3.9+, without changing project code.
if not hasattr(argparse, "BooleanOptionalAction"):
    spec = importlib.util.spec_from_file_location(
        "argparse_compat310", "/usr/lib/python3.10/argparse.py",
    )
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    argparse.BooleanOptionalAction = compat.BooleanOptionalAction

import casadi
import numpy
import torch
import test_nmpc

torch.set_num_threads(1)
sources = (
    "test_nmpc.py", "sim/simulation_mpc.py", "sim/simulation.py",
    "sim/start_perturbation.py", "sim/trial_metrics.py", "control/controller.py",
    "control/psdf_optimizer.py", "control/dcbf_optimizer.py",
    "control/obca_optimizer.py", "models/dd.py", "models/psdf.py",
    "models/psdf_wrapper.py", "config/config_nmpc_trials_maze_oblique.yaml",
)
metadata = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "numpy": numpy.__version__,
    "casadi": casadi.__version__,
    "threads": 1,
    "obca_horizon": 12,
    "obca_formulation": "hard dual minimum distance, all obstacles",
    "config_sha256": hashlib.sha256((RESULTS / "config.yaml").read_bytes()).hexdigest(),
    "temporary_runtime_packages": {"cycler": "0.12.1", "kiwisolver": "1.4.7"},
    "argparse_compatibility_source": "/usr/lib/python3.10/argparse.py",
    "build_directory": str(TEMP_ROOT / "work"),
    "source_sha256": {
        name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
        for name in sources
    },
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
metadata_path = RESULTS / "runtime_environment.json"
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
sys.argv = [
    "test_nmpc.py", "-c", str(RESULTS / "config.yaml"),
    "--trials", "1", "--results-dir", str(RESULTS),
]
started = time.perf_counter()
try:
    test_nmpc.main()
finally:
    metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    metadata["elapsed_wall_seconds"] = time.perf_counter() - started
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
