"""Compare RTI and existing full-SQP backup at one fixed stalled pose."""

import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/"source_snapshot"))

import numpy as np
import torch
import yaml

torch.set_num_threads(6)
torch.set_num_interop_threads(1)

from control.cc_sf_optimizer import CCSFOptimizer
from control.cc_covariance import covariance_from_log_cholesky, propagate_log_cholesky
import test_nmpc


class ProbeFinished(BaseException):
    pass


def capture(self, solver):
    x = np.stack([solver.get(i, "x") for i in range(self.N+1)])
    u = np.stack([solver.get(i, "u") for i in range(self.N)])
    local = np.stack([propagate_log_cholesky(x[i, 4:8], *u[i, :2], 0., .1, self.param)
                      if i < 10 else x[i, 4:8] for i in range(self.N)])
    sigma = np.sqrt(covariance_from_log_cholesky(x[1, 4:8])[2, 2])
    exact_sigma = np.sqrt(covariance_from_log_cholesky(local[0])[2, 2])
    return {"status": int(solver.status), "u0": u[0].tolist(),
            "stage1_sigma_psi": float(sigma), "stage1_exact_sigma_psi": float(exact_sigma),
            "max_covariance_log_defect": float(abs(x[1:, 4:7]-local[:, :3]).max()),
            "predicted_progress_m": float(x[-1, 3]-x[0, 3])}


original = CCSFOptimizer.solve_nlp


def solve(self):
    result = original(self)
    if self._system._time >= 40.-1e-8:
        probe = {"seed": 2, "time": float(self._system._time),
                 "pose": np.asarray(self.state._x).tolist(), "rti": capture(self, result.solver)}
        self._copy_primal_guess(result.solver, self.backup_solver)
        status = self._prepare_and_solve(self.backup_solver)
        probe["full_sqp_backup"] = capture(self, self.backup_solver)
        (BASE/"diagnostics/frozen_state_probe.json").write_text(json.dumps(probe, indent=2)+"\n")
        print("[Frozen probe]", json.dumps(probe), flush=True)
        raise ProbeFinished()
    return result


CCSFOptimizer.solve_nlp = solve
config = yaml.safe_load((BASE/"experiment_config.yaml").read_text())
config["start_perturbation"]["seed"] = 2
config["output_root_dir"] = str(BASE/"diagnostics/frozen_probe")
args = test_nmpc.build_parser().parse_args([
    "--single", "--optimizer-type", "cc_sf", "--maze-type", "maze",
    "--robot-shape", "rectangle", "--simulation-time", "60", "--start-seed", "2",
    "--no-animation", "--no-plots",
])
try:
    test_nmpc.run_single_test(args, config, collect_metrics=True)
except ProbeFinished:
    pass
