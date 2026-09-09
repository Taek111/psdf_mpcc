import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT))
from sim.simulation_mpc import simulation_mpc
from sim.start_perturbation import pose_clearance
from models.dd import DifferentialDriveRectangleGeometry

start, goal, grid, obstacles = simulation_mpc().create_env("maze")
footprints = [DifferentialDriveRectangleGeometry(length=0.15, width=0.09, rear_dist=0.0).equiv_rep()[0].get_ccw_vertices()]
polygons = [np.asarray(obstacle.get_ccw_vertices()) for obstacle in obstacles]
metrics = {}
for method in ("dcbf",):
    folder = ROOT / method
    if not (folder / "diagnostics.npz").exists():
        continue
    data = np.load(folder / "diagnostics.npz", allow_pickle=True)
    diag = json.loads((folder / "diagnostics.json").read_text())
    result = json.loads((folder / "result.json").read_text())
    states = data["states"]
    poses = np.vstack([start, states]) if len(states) else np.array([start])
    clearances = np.array([pose_clearance(pose, footprints, polygons, grid[0]) for pose in poses])
    row = dict(status=result["status"], failure_reason=result.get("failure_reason"), final_time=diag["final_time"], steps=len(states),
               final_pose=diag["final_state"], initial_clearance=float(clearances[0]), min_clearance=float(clearances.min()),
               min_clearance_time=float(np.argmin(clearances)*0.1), contact_or_overlap_samples=int((clearances<=1e-10).sum()),
               distance_to_requested_goal=float(np.linalg.norm(poses[-1,:2]-goal[:2])),
               distance_to_planner_goal=float(np.linalg.norm(poses[-1,:2]-np.array(diag["global_path"])[-1,:2])),
               planner_goal=diag["global_path"][-1], requested_goal=goal.tolist(), wall_seconds=result["wall_seconds"],
               solver_status_counts={})
    for status in diag["solver_statuses"]:
        key = str(status["status_code"]) + ":" + str(status["raw_status"])
        row["solver_status_counts"][key] = row["solver_status_counts"].get(key,0)+1
    for key in ("controller_times", "solver_times"):
        values = np.asarray(data[key], dtype=float)[:len(states)]
        if len(values):
            row[key] = dict(median=float(np.median(values)), p95=float(np.quantile(values,0.95)), maximum=float(values.max()), first=float(values[0]),
                            over_100ms=int((values>0.1).sum()), count=len(values))
    if len(states):
        row["input_abs_max"] = np.max(np.abs(data["inputs"]),axis=0).tolist()
        row["path_length"] = float(np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1).sum())
        row["last_5s_displacement"] = float(np.linalg.norm(poses[-1,:2]-poses[max(0,len(poses)-51),:2]))
    if len(states) >= 200:
        row['pose_at_20s'] = states[199].tolist()
        row['distance_to_planner_goal_at_20s'] = float(np.linalg.norm(states[199,:2]-np.array(diag['global_path'])[-1,:2]))
    row['solver_failure_count'] = sum(status['success'] is False for status in diag['solver_statuses'])
    with (folder / "clearance.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "footprint_clearance_m"])
        writer.writerows(zip(np.arange(len(poses))*0.1,clearances))
    metrics[method] = row
(ROOT / "metrics.json").write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))



