"""Keep the measured margin/status context around each safe-stop onset."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
diagnostics = []
for width in [.105, .110, .115, .120, .125, .130]:
    folder = ROOT / f"width_{round(width * 1000):03d}mm"
    if not (folder / "cycles.csv").exists():
        continue
    with (folder / "cycles.csv").open() as stream:
        cycles = list(csv.DictReader(stream))
    nominal = {}
    with (folder / "nominal_risk.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            nominal[(row["seed"], row["cycle"], row["kind"])] = row
    for seed in range(3):
        selected = [row for row in cycles if int(row["seed"]) == seed]
        previous_stopped = False
        for index, cycle in enumerate(selected):
            stopped = cycle["mode"] == "safe_stop"
            if stopped and not previous_stopped:
                end = index + 1
                while end < len(selected) and selected[end]["mode"] == "safe_stop":
                    end += 1
                context = []
                for adjacent in (index - 1, index, end):
                    for kind in ("fast", "backup"):
                        row = nominal.get((seed, adjacent, kind))
                        if row is None:
                            continue
                        context.append({"cycle": adjacent, "time_before_s": adjacent * .1,
                            "kind": kind, "status": row["status"],
                            "stage0_phi_mm": row["phi_m"][0] * 1000,
                            "stage0_margin_mm": row["risk_margin_m"][0] * 1000,
                            "stage0_residual_mm": row["nominal_residual_m"][0] * 1000,
                            "nominal_margin_max_mm": max(row["risk_margin_m"]) * 1000,
                            "nominal_residual_min_mm": min(row["nominal_residual_m"]) * 1000,
                            "stage0_risk_terms": row["risk_terms"][0]})
                diagnostics.append({"width_m": width, "seed": seed,
                    "start_time_s": index * .1, "end_time_s": end * .1,
                    "safe_stop_steps": end - index,
                    "resumed_with_successful_solve": end < len(selected),
                    "context": context})
            previous_stopped = stopped

result = {
    "status_code_source": "/home/taek111/projects/acados/include/acados/utils/types.h",
    "status_codes": {"0": "success", "2": "maximum iterations", "4": "QP failure"},
    "safe_stop_onsets": diagnostics,
    "interpretation_limits": [
        "A nonzero solver status is not a proof of nonlinear problem infeasibility.",
        "Stage 0 risk values are diagnostic only: generated solver JSON has nh_0=0, nh=nh_e=1.",
        "Feature identities and controlled ablations were not measured; margin discontinuity is not established as the cause.",
    ],
}
(ROOT / "failure_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(diagnostics, indent=2))
