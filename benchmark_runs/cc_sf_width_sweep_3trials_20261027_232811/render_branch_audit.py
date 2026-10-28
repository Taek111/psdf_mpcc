"""Render the branch switch behind the large apparent Taylor error."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "clearance_diagnosis"
audit = json.loads((OUTPUT / "branch_audit.json").read_text())
primary = audit["primary_case"]
with (OUTPUT / "primary_branch_path.csv").open() as stream:
    records = list(csv.DictReader(stream))
data = {key: np.array([float(row[key]) for row in records]) for key in records[0]}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (full, zoom) = plt.subplots(1, 2, figsize=(11.4, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
for axis in (full, zoom):
    axis.plot(data["translation_from_anchor_mm"], data["obstacle1_phi_mm"],
              color="#2676b9", linewidth=2, label="Obstacle 1: exact")
    axis.plot(data["translation_from_anchor_mm"], data["obstacle4_phi_mm"],
              color="#d17d16", linewidth=2, label="Obstacle 4: exact")
    axis.plot(data["translation_from_anchor_mm"], data["frozen_tangent_mm"],
              color="#153f69", linewidth=1.6, linestyle="--", label="Frozen tangent of obstacle 1")
    axis.axvline(primary["translation_until_minimum_switch_mm"],
                 color="#994258", linewidth=1, linestyle=":")
    axis.set(xlabel="Translation along anchor-to-QP-pose segment (mm)", ylabel="Signed distance (mm)")
    axis.grid(alpha=.15)
full.axhline(0, color="#7f8793", linewidth=.8)
full.set(xlim=(0, 13.25), ylim=(-5, 5.1), title="The other distance branch becomes the minimum")
full.legend(loc="lower left", fontsize=8)
full.text(12.85, 4.25, "Tangent: +3.766 mm", ha="right", color="#153f69")
full.text(12.85, -4.55, "True minimum: -4.091 mm", ha="right", color="#a35e10")
zoom.set(xlim=(0, .12), ylim=(3.935, 4.025), title="Switch after only 0.052 mm")
zoom.text(.065, 4.012, "Anchor gap:\n0.031 mm", fontsize=9,
          bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
fig.tight_layout(pad=1.8)
fig.savefig(OUTPUT / "distance_branch_switch.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(OUTPUT / "distance_branch_switch.png")
