"""Reconstruct CC constraint residuals from archived, already completed trials.

This module never instantiates an optimizer or runs a simulation.  Geometry and
covariance functions are imported from the experiment's source snapshot.  A
nominal replay is possible because these controllers retain (do not shift) the
previous solver trajectory and regenerate covariance from retained controls.
The last two checkpoint state coordinates are physical-input memory, so
``states[1:, 8:10]`` supplies those controls. The first cycle, failed/recovery
solves, and successors of unknown retained solver states remain unavailable.
With backup disabled, the archived safe-stop code installs its complete zero-
input plan in the main solver; after checking that saved plan, its successor
can be replayed normally.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import torch
import yaml


def _csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _source(path):
    return {"path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@contextmanager
def _snapshot_imports(snapshot):
    """Keep archived imports independent of any current checkout modules."""
    prefixes = ("models", "control")
    saved = {key: value for key, value in sys.modules.items()
             if key.split(".")[0] in prefixes}
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, str(snapshot))
    try:
        wrapper = importlib.import_module("models.augmented_psdf_wrapper")
        covariance = importlib.import_module("control.cc_covariance")
        implicit = importlib.import_module("models.mf_implicit")
        yield wrapper, covariance, implicit
    finally:
        sys.path.remove(str(snapshot))
        for key in list(sys.modules):
            if key.split(".")[0] in prefixes:
                del sys.modules[key]
        sys.modules.update(saved)


def _maze_polygons(path):
    """Read archived rectangular maze literals without importing simulation."""
    def literal(node, scale):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id == "s":
            return scale
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return literal(node.left, scale) * literal(node.right, scale)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -literal(node.operand, scale)
        raise ValueError(f"Unsupported archived geometry literal: {ast.dump(node)}")

    tree = ast.parse(path.read_text(encoding="utf-8"))
    branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "env_type"
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "maze"]
    if len(branches) != 1:
        raise ValueError("Expected one archived maze definition")
    scale, polygons = None, []
    for statement in branches[0].body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id == "s":
                scale = float(ast.literal_eval(statement.value))
        if isinstance(statement, ast.Expr):
            for call in ast.walk(statement):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
                        and call.func.id == "RectangleRegion":
                    left, right, bottom, top = [literal(arg, scale) for arg in call.args]
                    polygons.append(np.array([[left, bottom], [right, bottom],
                                              [right, top], [left, top]]))
    if not polygons:
        raise ValueError("Missing archived maze polygons")
    return polygons


def _single(directory, pattern):
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def _float_array(rows, key):
    return np.array([float(row[key]) if row[key] else np.nan for row in rows])


def _native_sf(wrapper, implicit, states, params):
    """SF residual and exact gradient of the frozen-geometry OCP row."""
    pose = torch.tensor(states[:, :3], dtype=torch.float32)
    phi_t, grad_t = wrapper(pose)
    phi, grad = phi_t.numpy().reshape(-1), grad_t.numpy()
    cosine, sine = np.cos(states[:, 2]), np.sin(states[:, 2])
    q = grad.copy().astype(float)
    q[:, 0] = cosine * grad[:, 0] + sine * grad[:, 1]
    q[:, 1] = -sine * grad[:, 0] + cosine * grad[:, 1]
    variance, variance_gradient = implicit.projected_variance_and_gradient(
        torch.tensor(q[:, None, :], dtype=torch.float64),
        torch.tensor(states[:, :8], dtype=torch.float64))
    sigma = np.sqrt(np.maximum(variance.numpy()[:, 0], 0.0))
    gamma = params.get("chance_gamma")
    if gamma is None:
        gamma = NormalDist().inv_cdf(1.0 - float(params["chance_epsilon"]))
    risk_unclipped = gamma * sigma
    floor, cap = params.get("risk_margin_floor", 0.0), params.get("risk_margin_cap", np.inf)
    risk = np.clip(risk_unclipped, floor, cap)
    jacobian = np.zeros((len(states), 8))
    jacobian[:, :3] = grad
    active = (sigma > 0) & (risk_unclipped > floor) & (risk_unclipped < cap)
    jacobian[active, 4:8] = -(gamma / (2 * sigma[active]))[:, None] \
        * variance_gradient.numpy()[active, 0]
    return phi - risk - float(params["d_min"]), jacobian, phi


def load_trial(experiment: Path, method: str, width_mm: int, seed: int,
               stage: int = 1) -> dict:
    """Return cycle-aligned residual arrays and reproducible validation metadata.

    Native units: CC-SF metres, CC-MF standard-normal quantile (dimensionless).
    ``postsolve_affine`` is the reconstructed QP row BEFORE MF lower slack.
    ``postsolve_fresh`` reevaluates nonlinear geometry/covariance at saved x*;
    it is not a solver-reported QP residual. ``nominal`` is logged for MF and
    reconstructed for SF. Missing/invalid replay stays NaN, never zero.
    ``s`` is the current cycle progress s0, not predicted x1.s.
    ``safe_stop`` and ``solve_mode`` preserve fallback events even though
    unsuccessful post-solve residuals are missing. Keep these rows as NaN gaps
    when drawing lines and mark the events separately. ``outcome`` is the final
    trial outcome, distinct from per-cycle solver success.
    """
    if method not in ("cc_sf", "cc_mf"):
        raise ValueError("method must be cc_sf or cc_mf")
    experiment = Path(experiment).resolve()
    folder = experiment / f"width_{width_mm:03d}mm" / method / f"seed_{seed}"
    snapshot = experiment / "source_snapshot"
    paths = {"checkpoint": folder / "animation_checkpoint.jsonl",
             "parameters": folder / "optimizer_parameters.json",
             "config": folder / "config.yaml",
             "pose": _single(folder / "data", "pose_sdf_*.csv"),
             "history": folder / "data" / "trial_history.csv",
             "cycles": folder / "cycles.csv",
             "outcome": folder / "outcome.json",
             "maze": snapshot / "sim" / "simulation_mpc.py"}
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    params = json.loads(paths["parameters"].read_text(encoding="utf-8"))
    outcome = json.loads(paths["outcome"].read_text(encoding="utf-8"))
    if params.get("nlp_solver_type") != "SQP_RTI":
        raise ValueError("Nominal replay supports the archived SQP_RTI lifecycle only")
    if not params.get("use_obstacle_constraint", True):
        raise ValueError("The archived trial has no active obstacle constraint")
    if not np.isscalar(params["chance_epsilon"]):
        raise ValueError("This extractor currently requires a scalar chance_epsilon")
    if not np.isclose(config["rectangle_geometry"]["width"] * 1000, width_mm):
        raise ValueError("Requested and archived widths differ")
    if config["localization_error"]["enabled"]:
        raise ValueError("Offline pose replay currently requires localization_error disabled")
    horizon = int(params["horizon"])
    if not 1 <= stage < horizon:
        raise ValueError("Use a common active SF/MF stage in 1..N-1")
    if method == "cc_mf":
        last = params.get("mf_active_end_step")
        last = horizon - 1 if last is None else int(last)
        if not params.get("use_row_mf", True) or not \
                int(params.get("mf_active_start_step", 1)) <= stage <= last:
            raise ValueError("Selected stage is outside the configured active MF window")
    pose_rows, history_rows = _csv(paths["pose"]), _csv(paths["history"])
    cycle_rows = _csv(paths["cycles"])
    with paths["checkpoint"].open(encoding="utf-8") as stream:
        checkpoint = [json.loads(line) for line in stream if line.strip()]
    entries = checkpoint[1:]
    states = np.asarray([entry["optimized_traj"] for entry in entries], dtype=float)
    n = len(pose_rows)
    if len(entries) != n or len(history_rows) != n or len(cycle_rows) != n \
            or states.shape != (n, horizon + 1, 10):
        raise ValueError("Checkpoint/pose/history counts or state dimensions disagree")
    if not np.isfinite(states).all():
        raise ValueError("Nonfinite checkpoint states")
    s = _float_array(pose_rows, "path_s")
    predicted_s = _float_array(pose_rows, "predicted_s")
    current_pose = np.column_stack([_float_array(pose_rows, k) for k in ("x", "y", "theta")])
    np.testing.assert_allclose(states[:, 0, :3], current_pose, atol=1e-6, rtol=0)
    np.testing.assert_allclose(states[:, 0, 3], s, atol=1e-8, rtol=0)
    np.testing.assert_allclose(states[:, 1, 3], predicted_s, atol=1e-8, rtol=0)
    applied = np.column_stack([_float_array(history_rows, key) for key in ("v", "omega")])
    memory_error = float(np.max(np.abs(states[:, 1, 8:10] - applied)))
    if memory_error > 1e-7:
        raise ValueError("Checkpoint physical-input memory does not match applied controls")
    success = np.array([row["solver_success"].lower() == "true" for row in history_rows])
    np.testing.assert_array_equal(_float_array(cycle_rows, "step"), np.arange(n))
    np.testing.assert_array_equal(success, [row["solver_success"].lower() == "true"
                                          for row in cycle_rows])
    solve_mode = np.array([row["solve_mode"] for row in cycle_rows], dtype=object)
    solver_status = _float_array(cycle_rows, "solver_status_code")
    safe_stop = np.array([str(mode).startswith("safe_stop") for mode in solve_mode])
    normal_mode = "fast" if method == "cc_sf" else "sqp_rti"
    recovery = solve_mode != normal_mode
    backup_enabled = bool(params.get("enable_backup_solver", False))
    installed_safe_stop = np.zeros(n, dtype=bool)
    if not backup_enabled:
        for i in np.flatnonzero(safe_stop):
            # Archived SF945–960 / MF1203–1210 install this exact plan into
            # the retained main solver. Check physical inputs and frozen pose
            # here; the covariance coordinates are checked below as well.
            installed_safe_stop[i] = (
                np.allclose(states[i, :, :3], current_pose[i], atol=1e-12, rtol=0)
                and np.allclose(states[i, :, 3], s[i], atol=1e-12, rtol=0)
                and np.allclose(states[i, 1:, 8:10], 0.0, atol=1e-12, rtol=0)
            )
    retained_known = (success & ~recovery) | installed_safe_stop
    replay_valid = success & ~recovery
    replay_valid[0] = False
    replay_valid[1:] &= retained_known[:-1]
    if backup_enabled:
        # Checkpoints do not identify retained main/backup solver state for SF.
        replay_valid[:] = False
    dt = float(params["tf"]) / horizon
    time = np.array([float(row["time_s"]) for row in entries]) - dt
    nominal = np.full(n, np.nan)
    affine, fresh, slack = (np.full(n, np.nan) for _ in range(3))
    clipped = np.zeros(n, dtype=bool)
    nominal_active = np.full(n, method == "cc_sf", dtype=bool)
    fresh_active = np.full(n, method == "cc_sf", dtype=bool)
    mf_rows = None
    if method == "cc_mf":
        paths["cycle"] = _single(folder / "data", "cc_mf_cycle_*.csv")
        mf_rows = _csv(paths["cycle"])
        if len(mf_rows) != n:
            raise ValueError("MF cycle count disagrees with checkpoints")
        np.testing.assert_allclose(_float_array(mf_rows, "time"), time, atol=1e-8, rtol=0)
        np.testing.assert_array_equal(solve_mode, [row["solve_mode"] for row in mf_rows])
        for i, row in enumerate(mf_rows):
            item = json.loads(row["mf_icdf_residual_nominal"])[stage]
            nominal[i] = np.nan if item is None else item
            item = json.loads(row["mf_slack"])[stage]
            slack[i] = np.nan if item is None else item
            clipped[i] = bool(json.loads(row["mf_tail_clipped"])[stage])
            nominal_active[i] = bool(json.loads(row["mf_active"])[stage])
    p = SimpleNamespace(**params)
    metadata = {"method": method, "width_mm": width_mm, "seed": seed, "stage": stage,
                "units": "m" if method == "cc_sf" else "standard_normal_quantile",
                "current_s_definition": "pose_sdf.path_s = cycle current x0.s",
                "affine_definition": "reconstructed final SQP_RTI QP row before slack",
                "fresh_definition": "archived nonlinear chance model reevaluated at postsolve stage state",
                "feature_switching_available": False,
                "replay_exclusions": (
                    "first cycle, failed/recovery cycles, and successors of unknown retained states; "
                    "successors of verified installed safe-stop plans are replayed when backup is disabled"),
                "verified_installed_safe_stop_cycles": np.flatnonzero(installed_safe_stop).tolist(),
                "geometry_device": "cpu float32 (original benchmark CUDA float32)",
                "input_memory_applied_control_max_abs_error": memory_error,
                "nominal_replay_validation_tolerance": 2e-4,
                "trial_status": outcome.get("status"),
                "goal_reached": outcome.get("goal_reached"),
                "safe_stop_cycles": int(safe_stop.sum()),
                "recovery_events": [
                    {"cycle": int(i), "time_s": float(time[i]), "s_m": float(s[i]),
                     "solve_mode": str(solve_mode[i]), "solver_success": bool(success[i]),
                     "solver_status_code": float(solver_status[i])
                     if np.isfinite(solver_status[i]) else None}
                    for i in np.flatnonzero(recovery | ~success)
                ]}
    if "safe_stop_count" in outcome and int(outcome["safe_stop_count"]) != int(safe_stop.sum()):
        raise ValueError("Cycle safe-stop count disagrees with final trial outcome")
    if method == "cc_sf":
        metadata["nominal_definition"] = "geometric phi(nominal pose) minus projected risk and d_min"
        metadata["sf_affine_anchor"] = "float32 nominal pose, as stored by archived AnalyticPSDFCasADi"
    else:
        metadata["nominal_definition"] = "directly logged pre-solve inverse-CDF residual"
    with _snapshot_imports(snapshot) as (wrapper_module, covariance, implicit):
        robust_horizon = covariance.resolve_robust_horizon(p)
        stop_covariance_error = 0.0
        for i in np.flatnonzero(installed_safe_stop):
            xi = covariance.log_cholesky_from_covariance(covariance.initial_local_covariance(
                p, SimpleNamespace(_x=current_pose[i])))
            for j in range(horizon + 1):
                stop_covariance_error = max(
                    stop_covariance_error, float(np.max(np.abs(xi - states[i, j, 4:8]))))
                if j < robust_horizon:
                    xi = covariance.propagate_log_cholesky(xi, 0.0, 0.0, 0.0, dt, p)
        if stop_covariance_error > 1e-10:
            raise ValueError("Saved safe-stop covariance differs from the archived installed plan")
        metadata["installed_safe_stop_covariance_max_abs_error"] = stop_covariance_error
        length = config["rectangle_geometry"]["length"]
        width = config["rectangle_geometry"]["width"]
        vertices = torch.tensor([[-length / 2, -width / 2], [length / 2, -width / 2],
                                 [length / 2, width / 2], [-length / 2, width / 2]], dtype=torch.float32)
        wrapper = wrapper_module.AugmentedPSDFWrapper(vertices, K_max=20, E_max=100, device="cpu")
        polygons = _maze_polygons(paths["maze"])
        wrapper.update_edge_clusters([torch.tensor(poly, dtype=torch.float32) for poly in polygons],
                                     [torch.tensor(np.roll(poly, -1, axis=0), dtype=torch.float32)
                                      for poly in polygons])
        with torch.no_grad():
            phi_now = wrapper(torch.tensor(current_pose, dtype=torch.float32))[0].numpy().reshape(-1)
        phi_error = float(np.max(np.abs(phi_now - _float_array(pose_rows, "sdf_value"))))
        metadata["current_psdf_max_abs_error_m"] = phi_error
        if phi_error > 2e-6:
            raise ValueError(f"Archived geometry does not match logged PSDF: {phi_error:g} m")
        post = states[:, stage, :8]
        if method == "cc_sf":
            fresh[:] = _native_sf(wrapper, implicit, post, params)[0]
        else:
            post_result = wrapper.forward_mf_icdf(torch.tensor(post, dtype=torch.float64),
                                                  params["chance_epsilon"], params["d_min"])
            fresh[:] = post_result.residual.numpy()
            fresh_active[:] = post_result.required.numpy()
        indices = np.flatnonzero(replay_valid)
        before = np.zeros((len(indices), 8))
        for row, i in enumerate(indices):
            before[row] = states[i - 1, stage, :8]
            # Stage-zero covariance resets at every cycle (including world->body conversion).
            xi = covariance.log_cholesky_from_covariance(covariance.initial_local_covariance(
                p, SimpleNamespace(_x=current_pose[i])))
            for j in range(min(stage, robust_horizon)):
                v, omega = states[i - 1, j + 1, 8:10]
                xi = covariance.propagate_log_cholesky(xi, v, omega, 0.0, dt, p)
            before[row, 4:8] = xi
        if len(indices) and method == "cc_sf":
            pre, jacobian, _ = _native_sf(wrapper, implicit, before, params)
            nominal[indices] = pre
            # The archived CasADi bridge stores its Taylor pose anchor in
            # float32, while solver states remain float64.
            anchor_offset = before[:, :3] - before[:, :3].astype(np.float32).astype(float)
            affine[indices] = pre + np.sum(jacobian * (post[indices] - before), axis=1) \
                + np.sum(jacobian[:, :3] * anchor_offset, axis=1)
            nominal_active[indices] = True
        elif len(indices):
            result = wrapper.forward_mf_icdf(torch.tensor(before, dtype=torch.float64),
                                             params["chance_epsilon"], params["d_min"])
            replay = result.residual.numpy()
            finite = np.isfinite(replay) & np.isfinite(nominal[indices])
            errors = np.abs(replay[finite] - nominal[indices][finite])
            metadata["mf_nominal_max_abs_error"] = float(errors.max()) if len(errors) else None
            metadata["mf_nominal_compared_cycles"] = int(finite.sum())
            comparable = finite & (np.abs(replay - nominal[indices]) <= 2e-4)
            valid_indices = indices[comparable]
            affine[valid_indices] = np.sum(result.A.numpy()[comparable] * post[valid_indices], axis=1) \
                + result.c.numpy()[comparable]
            replay_valid[indices[~comparable]] = False
            mask_disagreement = result.required.numpy() != nominal_active[indices]
            metadata["mf_nominal_replay_rejected_cycles"] = int(
                (finite & ~comparable).sum() + mask_disagreement.sum())
            metadata["mf_replay_mask_disagreement_cycles"] = int(mask_disagreement.sum())
            metadata["mf_nominal_inactive_replay_candidates"] = int(
                (~nominal_active[indices]).sum())
    # A failed solver's state is not a successful QP solution.
    affine[~success] = np.nan
    fresh[~success] = np.nan
    metadata["cycles"] = n
    after_stop = np.r_[False, installed_safe_stop[:-1]] & np.isfinite(affine)
    metadata["replayed_safe_stop_successor_cycles"] = np.flatnonzero(after_stop).tolist()
    metadata["replayed_cycles"] = int(np.isfinite(affine).sum())
    metadata["solver_success_cycles"] = int(success.sum())
    metadata["tail_clipped_nominal_cycles"] = int(clipped.sum())
    if method == "cc_mf":
        metadata["nominal_inactive_cycles"] = int((~nominal_active).sum())
        metadata["fresh_domain_inactive_cycles"] = int((~fresh_active).sum())
        metadata["row_m_activation_definition"] = (
            "Row M is installed only when nominal signed PSDF phi > d_min; "
            "a missing inactive margin is not evidence of chance feasibility")
    for file in ("models/augmented_psdf.py", "models/augmented_psdf_wrapper.py",
                 "models/mf_inverse_cdf.py", "models/mf_implicit.py", "control/cc_covariance.py",
                 "control/cc_sf_optimizer.py", "control/cc_mf_optimizer.py", "control/mpcc_optimizer.py",
                 "control/analytic_psdf_casadi.py"):
        paths[file] = snapshot / file
    metadata["sources"] = {key: _source(path) for key, path in paths.items()}
    return {"method": method, "stage": stage, "s": s, "predicted_s": predicted_s,
            "selected_predicted_s": states[:, stage, 3].copy(),
            "time": time, "solver_success": success, "nominal": nominal,
            "solve_mode": solve_mode, "solver_status": solver_status,
            "safe_stop": safe_stop, "outcome": outcome,
            "postsolve_affine": affine, "postsolve_fresh": fresh, "slack": slack,
            "nominal_valid": np.isfinite(nominal), "nominal_active": nominal_active,
            "fresh_active": fresh_active,
            "nominal_tail_clipped": clipped, "replay_valid": replay_valid,
            "metadata": metadata}
