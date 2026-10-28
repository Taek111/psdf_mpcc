"""Draw the measured 0.120 m, seed-0 first-overlap event."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "clearance_diagnosis"
record = json.loads((OUTPUT / "first_overlap_details.json").read_text())[0]
geometry = json.loads((ROOT / "width_120mm/geometry.json").read_text())
footprint = np.asarray(geometry["footprint"])
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (scene, bars) = plt.subplots(1, 2, figsize=(11.8, 4.8),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
for index, vertices in enumerate(geometry["obstacles"]):
    vertices = np.asarray(vertices)
    if index not in (1, 4):
        continue
    color = "#d9dde4" if index == 1 else "#f4cfc6"
    scene.add_patch(Polygon(vertices, facecolor=color, edgecolor="#536072", linewidth=1))
scene.text(.407, .688, "Obstacle 1\nactive at anchor", ha="center", color="#45556c")
scene.text(.646, .485, "Obstacle 4\ncollision", ha="center", color="#994330")
for key, label, color, style in (
        ("nominal_stage1_pose", "Taylor anchor", "#286fba", "--"),
        ("returned_stage1_pose", "Solver prediction", "#d68a12", "-"),
        ("actual_after_pose", "Applied-input RK4", "#b82e37", "-")):
    pose = np.asarray(record[key])
    c, s = np.cos(pose[2]), np.sin(pose[2])
    vertices = footprint @ np.array([[c, s], [-s, c]]) + pose[:2]
    scene.add_patch(Polygon(vertices, fill=False, edgecolor=color,
                            linewidth=1.7, linestyle=style, label=label))
    scene.scatter(*pose[:2], s=16, color=color)
scene.annotate("Overlap", xy=(.602, .57), xytext=(.645, .625),
               arrowprops={"arrowstyle": "->", "color": "#994330"}, color="#994330")
scene.set(xlim=(.382, .692), ylim=(.455, .705), xlabel="World x (m)", ylabel="World y (m)",
          title="0.120 m robot, seed 0: t = 6.3 to 6.4 s")
scene.set_aspect("equal")
scene.legend(loc="lower left", framealpha=.95, fontsize=8)
values = [record["affine_phi_at_predicted_mm"], record["actual_psdf_at_predicted_mm"],
          record["actual_psdf_at_actual_mm"]]
colors = ["#286fba", "#d68a12", "#b82e37"]
bars.bar(np.arange(3), values, color=colors, width=.62)
bars.axhline(0, color="#7c8799", linewidth=.9)
threshold = 1 + record["linearized_qp_margin_mm"]
bars.axhline(threshold, color="#227052", linestyle="--", linewidth=1.2,
             label=f"QP requirement = {threshold:.3f} mm")
for index, value in enumerate(values):
    bars.text(index, value + (.2 if value > 0 else -.2), f"{value:+.3f}",
              ha="center", va="bottom" if value > 0 else "top", weight="bold")
bars.set_xticks(np.arange(3), ["Taylor PSDF\nat prediction", "Fresh PSDF\nat prediction", "Fresh PSDF\nafter input"])
bars.set(ylabel="Signed PSDF (mm)", ylim=(-5.6, 5.1),
         title="The hard QP row passes; the real geometry fails")
bars.legend(loc="upper right", fontsize=8)
fig.tight_layout(pad=1.8)
fig.savefig(OUTPUT / "first_overlap_geometry.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(OUTPUT / "first_overlap_geometry.png")
