"""Replay an exact pose/reference snapshot with alternative initial guesses."""
import json
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from control.dcbf_optimizer import NmpcDbcfOptimizer, NmpcDcbfOptimizerParam
from control.obca_optimizer import OBCAOptimizer, OBCAOptimizerParam
from models.dd import (DifferentialDriveDynamics, DifferentialDriveSystem,
                       DifferentialDriveStates, DifferentialDriveMultipleGeometry,
                       DifferentialDriveRectangleGeometry)
from sim.simulation_mpc import simulation_mpc
from sim.start_perturbation import convex_polygon_clearance, pose_clearance

method = sys.argv[1]
names = sys.argv[2:] or ["default", "forward_left", "forward_right",
                        "reverse_left", "reverse_right", "full_constraint_horizon"]
folder = ROOT / method
saved = folder / "diagnostics.json"
if saved.exists():
    diag = json.loads(saved.read_text())
    data = np.load(folder / "diagnostics.npz")
    state = np.asarray(diag["final_state"])
    # References at a stationary final pose differ only at floating precision.
    reference = data["references"][-1]
    previous_input = data["inputs"][-1]
    state_source = "completed baseline final state; last logged reference and input"
else:
    snapshot = json.loads((folder / "progress.json").read_text())
    state = np.asarray(snapshot["state"])
    reference = np.asarray(snapshot["reference"])
    previous_input = np.zeros(2)
    state_source = ("exact live pre-solve state/reference; previous input set to zero "
                    "for a stationary replay (previous-input cost is zero)")

start, goal, grid, obstacles = simulation_mpc().create_env("maze")
geometry = DifferentialDriveMultipleGeometry()
geometry.add_geometry(DifferentialDriveRectangleGeometry(.15, .09, 0))
system = DifferentialDriveSystem(
    state=DifferentialDriveStates(state, previous_input), geometry=geometry,
    dynamics=DifferentialDriveDynamics())
footprint = np.asarray(geometry.equiv_rep()[0].get_ccw_vertices())
polygons = [np.asarray(o.get_ccw_vertices()) for o in obstacles]
world = footprint @ system._state.rotation().T + state[:2]
clearances = [convex_polygon_clearance(world, polygon) for polygon in polygons]
nearest = int(np.argmin(clearances))
report = dict(state_source=state_source, state=state.tolist(),
              reference=reference.tolist(), nearest_obstacle=nearest,
              nearest_obstacle_vertices=polygons[nearest].tolist(),
              current_clearance=min(clearances), solves=[])

for name in names:
    cls, param_cls = ((NmpcDbcfOptimizer, NmpcDcbfOptimizerParam) if method == "dcbf"
                      else (OBCAOptimizer, OBCAOptimizerParam))
    param = param_cls()
    if name in ("full_constraint_horizon", "full_with_safe_seed"):
        param.horizon_dcbf = param.horizon
    if name == "distance_only":
        param.gamma = 0.0
    optimizer = cls({}, {}, DifferentialDriveDynamics.forward_dynamics_opt(.1))
    optimizer.setup(param, system, reference, obstacles)
    if name in ("safe_seed", "full_with_safe_seed", "own_safe_seed", "own_feasible_safe_seed"):
        seed_file = (ROOT / "dcbf/probe_safe_seed_full_with_safe_seed_distance_only.json"
                     if name in ("own_safe_seed", "own_feasible_safe_seed") else
                     ROOT / "obca/probe_default_forward_left_forward_right_reverse_left_reverse_right.json")
        seed = min(json.loads(seed_file.read_text())["solves"], key=lambda row: row["cost"])
        optimizer.opti.set_initial(optimizer.variables["x"], np.asarray(seed["trajectory"]).T)
        optimizer.opti.set_initial(optimizer.variables["u"], np.asarray(seed["inputs"]).T)
        if name == "own_feasible_safe_seed":
            # Rectangle runs allocate x, u, then lambda/mu/omega for each
            # active rectangle obstacle. Retain the first ten stages of
            # each twenty-stage dual block, preserving a feasible seed.
            primal = np.asarray(seed["all_variables"])
            first_dual = 3 * (param.horizon + 1) + 2 * param.horizon
            source_h = seed["horizon_dcbf"]
            target_h = param.horizon_dcbf
            chunks = [primal[:first_dual]]
            for offset in range(first_dual, len(primal), 9 * source_h):
                for nrows, start_offset in ((4, 0), (4, 4 * source_h), (1, 8 * source_h)):
                    start_idx = offset + start_offset
                    values = primal[start_idx:start_idx + nrows * source_h]
                    chunks.append(values.reshape((nrows, source_h), order="F")[:, :target_h].ravel(order="F"))
            optimizer.opti.set_initial(optimizer.opti.x, np.concatenate(chunks))
    if name.startswith(("forward_", "reverse_")):
        guess = state.copy()
        for k in range(param.horizon):
            speed = -.06 if name.startswith("reverse_") and k < 6 else .1
            angular_speed = .6 if name.endswith("left") else -.6
            control = np.array([speed, angular_speed if k < 15 else 0.0])
            guess = DifferentialDriveDynamics.forward_dynamics(guess, control, .1)
            optimizer.opti.set_initial(optimizer.variables["x"][:, k + 1], guess)
            optimizer.opti.set_initial(optimizer.variables["u"][:, k], control)
    try:
        initial_cost = float(optimizer.opti.debug.value(sum(optimizer.costs.values()), optimizer.opti.initial()))
        initial_g = np.asarray(optimizer.opti.debug.value(optimizer.opti.g, optimizer.opti.initial())).ravel()
        initial_lb = np.asarray(optimizer.opti.debug.value(optimizer.opti.lbg, optimizer.opti.initial())).ravel()
        initial_ub = np.asarray(optimizer.opti.debug.value(optimizer.opti.ubg, optimizer.opti.initial())).ravel()
        initial_violation = float(max(0, np.max(initial_lb - initial_g), np.max(initial_g - initial_ub)))
        solution = optimizer.solve_nlp()
        trajectory = np.asarray(solution.value(optimizer.variables["x"])).T
        inputs = np.asarray(solution.value(optimizer.variables["u"])).T
        g = np.asarray(solution.value(optimizer.opti.g)).ravel()
        lower = np.asarray(solution.value(optimizer.opti.lbg)).ravel()
        upper = np.asarray(solution.value(optimizer.opti.ubg)).ravel()
        costs = {key: float(solution.value(value)) for key, value in optimizer.costs.items()}
        clearance = [pose_clearance(p, [footprint], polygons, grid[0]) for p in trajectory]
        result = dict(name=name, status=solution.stats()["return_status"],
                      cost=sum(costs.values()), costs=costs,
                      first_input=inputs[0].tolist(),
                      max_constraint_violation=float(max(0, np.max(lower - g), np.max(g - upper))),
                      min_predicted_clearance=min(clearance),
                      contact_stages=np.flatnonzero(np.asarray(clearance) <= 0).tolist(),
                      trajectory=trajectory.tolist(), inputs=inputs.tolist(),
                      clearance=clearance, horizon_dcbf=param.horizon_dcbf,
                      initial_cost=initial_cost, initial_violation=initial_violation)
        if name == "full_with_safe_seed":
            result["all_variables"] = np.asarray(solution.value(optimizer.opti.x)).ravel().tolist()
    except Exception as exc:
        result = dict(name=name, status="exception", error=str(exc))
    report["solves"].append(result)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("trajectory", "inputs", "clearance", "all_variables")}), flush=True)
    suffix = "_final" if saved.exists() else ""
    output = folder / ("probe_" + "_".join(names) + suffix + ".json")
    output.write_text(json.dumps(report, indent=2))
print("STATE", json.dumps({k: v for k, v in report.items()
                          if k not in ("solves", "reference")}), flush=True)
