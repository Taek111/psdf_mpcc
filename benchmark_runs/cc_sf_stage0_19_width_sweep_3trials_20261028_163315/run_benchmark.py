"""Run one CC-SF width with frozen sources and observation-only diagnostics.

Each invocation runs three sequential trials (seeds 0, 1, 2). The caller checks
the completed width before starting another, so all-failure early stopping does
not require modifying the production simulation or controller.
"""

import argparse
import csv
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from zoneinfo import ZoneInfo


OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parents[1]
SOURCE = OUTPUT / "source_snapshot"


def stamp():
    return dt.datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def serialize(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, default=serialize) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=float, required=True)
    args = parser.parse_args()
    width = args.width
    assert 0 < width <= .30
    folder = OUTPUT / f"width_{round(width * 1000):03d}mm"
    folder.mkdir(exist_ok=False)
    hashes = json.loads((OUTPUT / "source_hashes.json").read_text())
    assert all(hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
               for name, digest in hashes.items()), "Frozen sources changed."

    os.environ["ACADOS_SOURCE_DIR"] = "/home/taek111/projects/acados"
    previous_libs = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = "/home/taek111/projects/acados/lib" + (
        ":" + previous_libs if previous_libs else "")
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(SOURCE))
    os.chdir(SOURCE)

    import casadi
    import numpy as np
    import torch
    import yaml
    import test_nmpc
    from control.cc_sf_optimizer import CCSFOptimizer, CCSFOptimizerParam
    from models.dd import DifferentialDriveRectangleGeometry
    from sim.start_perturbation import sample_start_pose

    module = importlib.import_module("sim.simulation_mpc")
    assert Path(module.__file__).is_relative_to(SOURCE)
    assert Path(test_nmpc.__file__).is_relative_to(SOURCE)
    assert torch.cuda.is_available() and torch.get_num_threads() == 6

    config = yaml.safe_load((OUTPUT / "base_config.yaml").read_text())
    config["trials"] = 3
    config.pop("cc_ecs", None)
    config["defaults"].update(generate_animation=True, generate_plots=False, frame_skip=5)
    config["test_configs"] = [{
        "maze_type": "maze", "robot_shape": "rectangle",
        "optimizer_type": "cc_sf", "dynamics_type": "differential_drive",
        "path_planner": "astar",
    }]
    (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    # Change only the simulator's constructor binding. The geometry class and
    # all optimizer/dynamics/planner source files remain the frozen originals.
    def rectangle_for_run(length, width=None, rear_dist=0.0):
        assert length == .15 and rear_dist == 0.0
        return DifferentialDriveRectangleGeometry(length, selected_width, rear_dist)

    selected_width = width
    module.DifferentialDriveRectangleGeometry = rectangle_for_run
    footprint = rectangle_for_run(.15)
    start, goal, grid, obstacles = module.simulation_mpc().create_env("maze")
    geometry = {
        "bounds": grid[0], "nominal_start": start, "goal": goal, "dt_s": .1,
        "footprint": footprint.equiv_rep()[0].get_ccw_vertices(),
        "obstacles": [obstacle.get_ccw_vertices() for obstacle in obstacles],
    }
    np.testing.assert_allclose(np.ptp(geometry["footprint"], axis=0), [.15, width],
                               rtol=0, atol=1e-14)
    dump(folder / "geometry.json", geometry)
    starts = []
    for seed in range(3):
        pose, details = sample_start_pose(start, footprint, obstacles, grid[0],
                                         {**config["start_perturbation"], "seed": seed})
        starts.append({"seed": seed, "pose": pose, "details": details})
    dump(folder / "expected_start_poses.json", starts)
    params = CCSFOptimizerParam()
    module.simulation_mpc._apply_param_overrides(params, config["cc_sf"], "CC-SF")
    dump(folder / "optimizer_parameters.json", vars(params))

    context = {"seed": None, "cycle": 0}
    original_regenerate = CCSFOptimizer._regenerate_covariance_guess
    original_risk = CCSFOptimizer._compute_stage_risk_data
    original_prepare = CCSFOptimizer._prepare_and_solve
    original_solve = CCSFOptimizer.solve_nlp
    original_trial = test_nmpc.run_single_test
    original_create_solver = CCSFOptimizer.create_solver
    stage_policies = []
    animation_manifest = []
    animation_dir = PROJECT / "animations" / OUTPUT.name
    animation_dir.mkdir(parents=True, exist_ok=True)

    def observe_create_solver(self, *call_args, **kwargs):
        solver = original_create_solver(self, *call_args, **kwargs)
        json_path = Path(kwargs.get("json_filename") or self.json_filename)
        generated = json.loads(json_path.read_text())
        dims = {name: generated["dims"][name] for name in
                ("N", "nx", "nu", "nh_0", "nh", "nh_e", "nsh_0", "nsh", "nsh_e")}
        assert dims == dict(N=20, nx=8, nu=3, nh_0=1, nh=1, nh_e=0,
                            nsh_0=0, nsh=0, nsh_e=0), dims
        assert generated["solver_options"]["tf"] == 2.0
        record = {"seed": context["seed"], "solver_json_name": json_path.name,
                  "dims": dims, "tf_s": generated["solver_options"]["tf"],
                  "nlp_solver_type": generated["solver_options"]["nlp_solver_type"],
                  "lh_0": generated["constraints"]["lh_0"],
                  "lh": generated["constraints"]["lh"],
                  "lh_e": generated["constraints"]["lh_e"]}
        stage_policies.append(record)
        dump(folder / "solver_stage_policy.json", stage_policies)
        return solver

    def animation_path(self, animation_name, method_name=None):
        return str(animation_dir / (
            f"MPC_cc_sf_rectangle_maze_width{round(width * 1000):03d}mm_"
            f"seed{context['seed']}_stage0-19.mp4"))

    def render_animation(simulation, trial_args, interrupted=False):
        if not trial_args.generate_animation or not test_nmpc.has_animation_data(simulation):
            return
        video = simulation.animate_world(
            simulation.sim, animation_name=simulation.current_name,
            maze_type=trial_args.maze_type, frame_skip=trial_args.frame_skip,
            method_name=trial_args.optimizer_type, return_html=False,
        )
        record = {"width_m": width, "seed": context["seed"], "path": video,
                  "size_bytes": Path(video).stat().st_size, "frame_skip": trial_args.frame_skip,
                  "renderer": "existing simulation_mpc.animate_world defaults"}
        assert record["size_bytes"] > 0
        animation_manifest.append(record)
        dump(folder / "animations.json", animation_manifest)
        print("ANIMATION_SAVED " + str(video), flush=True)

    def observe_regenerate(self, solver, *call_args, **kwargs):
        result = original_regenerate(self, solver, *call_args, **kwargs)
        self._benchmark_nominal_states = np.array(result, copy=True)
        self._benchmark_nominal_inputs = np.stack([solver.get(i, "u") for i in range(self.N)])
        return result

    def observe_risk(self, poses, *call_args, **kwargs):
        result = original_risk(self, poses, *call_args, **kwargs)
        phi, heads, terms = result
        states = self._benchmark_nominal_states
        margins = self._compute_margin_from_risk_terms(states[:, 4:8], terms, self.param)
        self._benchmark_nominal = {
            "states": states,
            "inputs": self._benchmark_nominal_inputs,
            "phi_m": np.asarray(phi), "risk_terms": terms,
            "risk_margin_m": margins,
            "nominal_residual_m": np.asarray(phi) - self.param.d_min - margins,
        }
        return result

    with (folder / "cycles.csv").open("w", newline="", buffering=1) as cycles, \
         (folder / "nominal_risk.jsonl").open("w", buffering=1) as risk_log:
        writer = csv.DictWriter(cycles, fieldnames=[
            "seed", "cycle", "time_before_s", "mode", "fast_status",
            "backup_status", "safe_stop", "cumulative_safe_stops",
            "fast_nominal_margin_max_m", "fast_nominal_residual_min_m",
            "stage0_phi_m", "stage0_margin_m", "stage0_residual_m",
        ])
        writer.writeheader()

        def observe_prepare(self, solver):
            kind = "fast" if solver is self.solver else "backup"
            self._benchmark_solver = solver
            status = original_prepare(self, solver)
            diagnostic = {**self._benchmark_nominal,
                "returned_states": np.stack([solver.get(i, "x") for i in range(self.N + 1)]),
                "returned_inputs": np.stack([solver.get(i, "u") for i in range(self.N)])}
            self._benchmark_attempts.append((kind, int(status), diagnostic))
            risk_log.write(json.dumps({"seed": context["seed"],
                "cycle": context["cycle"], "kind": kind, "status": int(status),
                **diagnostic}, default=serialize) + "\n")
            return status

        def observe_solve(self):
            self._benchmark_attempts = []
            solution = original_solve(self)
            attempts = {kind: (status, diagnostic)
                        for kind, status, diagnostic in self._benchmark_attempts}
            fast = attempts["fast"][1]
            writer.writerow({
                "seed": context["seed"], "cycle": context["cycle"],
                "time_before_s": context["cycle"] * .1,
                "mode": self._last_solve_mode, "fast_status": attempts["fast"][0],
                "backup_status": attempts.get("backup", ("",))[0],
                "safe_stop": int(self._last_solve_mode == "safe_stop"),
                "cumulative_safe_stops": self._safe_stop_count,
                "fast_nominal_margin_max_m": float(np.max(fast["risk_margin_m"])),
                "fast_nominal_residual_min_m": float(np.min(fast["nominal_residual_m"][:self.N])),
                "stage0_phi_m": float(fast["phi_m"][0]),
                "stage0_margin_m": float(fast["risk_margin_m"][0]),
                "stage0_residual_m": float(fast["nominal_residual_m"][0]),
            })
            context["cycle"] += 1
            return solution

        def observe_trial(trial_args, trial_config=None, **kwargs):
            context.update(seed=trial_args.start_seed, cycle=0)
            result = original_trial(trial_args, trial_config, **kwargs)
            dump(folder / f"runtime_seed{context['seed']}.json", {
                key: value for key, value in result.items()
                if key != "successful_solver_steps"
            })
            print(f"WIDTH_RESULT width={width:.3f} seed={context['seed']} "
                  f"status={result.get('status')} safe_stop={result.get('safe_stop_count')}",
                  flush=True)
            return result

        CCSFOptimizer.create_solver = observe_create_solver
        module.simulation_mpc._get_animation_output_path = animation_path
        test_nmpc.generate_animation_if_possible = render_animation
        CCSFOptimizer._regenerate_covariance_guess = observe_regenerate
        CCSFOptimizer._compute_stage_risk_data = observe_risk
        CCSFOptimizer._prepare_and_solve = observe_prepare
        CCSFOptimizer.solve_nlp = observe_solve
        test_nmpc.run_single_test = observe_trial
        metadata = {
            "created_at_kst": stamp(), "width_m": width, "length_m": .15,
            "python": sys.executable, "python_version": sys.version,
            "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "casadi": casadi.__version__,
            "gpu": torch.cuda.get_device_name(0), "torch_num_threads": torch.get_num_threads(),
            "thread_environment": {name: os.environ.get(name) for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "execution_source_root": str(SOURCE), "observation_only_hooks": True,
        }
        dump(folder / "environment.json", metadata)
        run_args = test_nmpc.build_parser().parse_args([
            "-c", str(folder / "config.yaml"), "--trials", "3",
            "--results-dir", str(folder), "--no-plots",
        ])
        begun = time.perf_counter()
        try:
            test_nmpc.run_trial_batch(run_args, config=config)
        finally:
            metadata.update(finished_at_kst=stamp(), wall_time_s=time.perf_counter() - begun)
            metadata["snapshot_changed"] = [name for name, digest in hashes.items()
                if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != digest]
            dump(folder / "environment.json", metadata)


if __name__ == "__main__":
    main()
