"""Independent saved-data arithmetic audit; never invokes an optimizer."""

from pathlib import Path
import csv
import hashlib
import json
import math
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from utils.offline_cc_margin import _snapshot_imports

SUITE = ROOT / "benchmark_runs/rectangle_don_sweep_50_20261105_current_sf"
OUTPUT = Path(__file__).resolve().parent
PILOT = ROOT / "benchmark_runs/rectangle_constraint_reliability_20261107_pilot"


def normal_cdf(array):
    array = np.asarray(array)
    return np.fromiter(
        (0.5 * math.erfc(-float(x) / math.sqrt(2)) for x in array.ravel()),
        float, count=array.size,
    ).reshape(array.shape)


def independent_covariance(inputs, dt, params):
    v, omega = inputs[:, 0], inputs[:, 1]
    rotation, translation = dt * omega, dt * v
    cosine, sine = np.cos(rotation), np.sin(rotation)
    hf = translation * (sine + 4 * np.sin(0.5 * rotation)) / 6
    hl = translation * (1 + 4 * np.cos(0.5 * rotation) + cosine) / 6
    transition = np.zeros((len(inputs), 3, 3))
    transition[:, 0, 0], transition[:, 0, 1], transition[:, 0, 2] = cosine, sine, hf
    transition[:, 1, 0], transition[:, 1, 1], transition[:, 1, 2] = -sine, cosine, hl
    transition[:, 2, 2] = 1
    initial = np.diag([params[k] ** 2 for k in ("sigma_f0", "sigma_l0", "sigma_psi0")])
    covariance = transition @ initial @ transition.transpose(0, 2, 1)
    covariance[:, 0, 0] += dt * (params["q_f0"] + params["alpha_f"] * v * v)
    covariance[:, 1, 1] += dt * (
        params["q_l0"] + params["alpha_v"] * v * v
        + params["alpha_kappa"] * np.abs(v * omega)
    )
    covariance[:, 2, 2] += dt * (
        params["q_psi0"] + params["beta_v"] * v * v
        + params["beta_kappa"] * np.abs(v * omega)
        + params["beta_omega"] * omega * omega
    )
    covariance[:, 0, 1] = covariance[:, 1, 0] = 0
    covariance[:, 0, 2] = covariance[:, 2, 0] = 0
    return covariance


def wrapper_for(module, geometry, dtype):
    wrapper = module.AugmentedPSDFWrapper(
        torch.tensor(geometry["footprint_polygons"][0], dtype=dtype), 20, 100, "cpu"
    )
    polygons = [np.array(x) for x in geometry["obstacle_polygons"]]
    wrapper.update_edge_clusters(
        [torch.tensor(x, dtype=dtype) for x in polygons],
        [torch.tensor(np.roll(x, -1, axis=0), dtype=dtype) for x in polygons],
    )
    return wrapper


def independent_probabilities(wrapper, poses, covariances, xi, params):
    sf, mf, phis = [], [], []
    for first in range(0, len(poses), 128):
        pose, matrix = poses[first:first + 128], covariances[first:first + 128]
        phi, gradient = wrapper(torch.tensor(pose, dtype=wrapper.A.dtype))
        gradient = gradient.numpy().astype(float)
        q = gradient.copy()
        c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
        q[:, 0] = c * gradient[:, 0] + s * gradient[:, 1]
        q[:, 1] = -s * gradient[:, 0] + c * gradient[:, 1]
        sigma = np.sqrt(np.einsum("ni,nij,nj->n", q, matrix, q))
        sf.extend(normal_cdf((params["d_min"] - phi.numpy()) / sigma))
        phis.extend(phi.numpy())
        state = np.column_stack((pose, np.zeros(len(pose)), xi[first:first + 128]))
        result = wrapper.forward_mf_icdf(
            torch.tensor(state, dtype=torch.float64), params["chance_epsilon"],
            params["d_min"], return_frozen_features=True,
        )
        frozen = result.frozen_features
        qf, distance = frozen["q"].numpy(), frozen["d"].numpy()
        mask = frozen["mask"].numpy()
        sigma_f = np.sqrt(np.maximum(np.einsum("nmi,nij,nmj->nm", qf, matrix, qf), 0))
        sigma_f = np.where(mask, sigma_f, 1)
        probabilities = normal_cdf((params["d_min"] - distance) / sigma_f)
        mf.extend(np.where(mask, probabilities, 0).sum(axis=1))
    return np.array(sf), np.array(mf), np.array(phis)


def main():
    torch.set_num_threads(2)
    rows = list(csv.DictReader((PILOT / "tables/per_trial_metrics.csv").open(encoding="utf-8-sig")))
    review = {
        "review_type": "Independent static and saved-data arithmetic review; no optimizer/simulation/test suite",
        "engine_path": "scripts/analyze_sf_mf_reliability.py",
        "engine_sha256": hashlib.sha256((ROOT / "scripts/analyze_sf_mf_reliability.py").read_bytes()).hexdigest(),
        "reviewed_artifact": str(PILOT.relative_to(ROOT)),
        "checks": [],
        "scope_limitations": [
            "No returned stage-1 candidates or planning covariance exist in source traces; these remain offline surrogate diagnostics.",
            "Independent probability arithmetic covers pilot DoN .85 seed 1 from each controller, not all trial traces.",
            "Geometry is archived implementation; this review checks covariance/probability arithmetic, not physical model calibration.",
            "CPU float32 sensitivity is not bitwise reproduction of archived CUDA float32 execution.",
        ],
        "static_findings": {
            "alignment": "xi1[k] resets Sigma0 and propagates applied_inputs[k], then evaluates poses[k+1]; no cumulative run-wide covariance.",
            "eligibility": "Successful solver-called steps supply both numerator and denominator, fallback excluded, first successful step retained.",
            "cohort": "Common-success seeds at each width, with matching initial poses independently audited.",
            "mf_rows": "Evaluable probability_sum includes domain-inactive rows, with no slack and no required-row eligibility filter.",
            "switch_comparison": "Both formulations use identical pose pairs and fixed body covariance, reported separately by source trajectory.",
            "aggregation": "Pooled counts and mean trial rates are separate; boundary-jump averages are event-weighted.",
            "switch_sampling": "Only endpoint-ID-changing intervals inspected; nine-point grid/bisection does not guarantee every continuous crossing.",
            "probability": "MF R is an untruncated probability sum/union bound and can exceed 1, not an empirical collision probability.",
        },
    }
    with _snapshot_imports(SUITE / "source_snapshot") as (module, covariance_module, _):
        for row in rows:
            meta = json.loads((ROOT / row["metadata"]).read_text(encoding="utf-8"))
            raw = np.load(ROOT / row["trace"])
            name = f"don{float(row['don']):.2f}_{row['source']}_seed{row['seed']}.npz"
            cache = np.load(PILOT / "cache" / name)
            poses, inputs, params = raw["poses"][1:], raw["applied_inputs"], meta["optimizer_parameters"]
            cov = independent_covariance(inputs, meta["dt_s"], params)
            cache_cov = covariance_module.covariance_from_log_cholesky(cache["xi1"])
            sf, mf, phi = independent_probabilities(
                wrapper_for(module, meta["geometry"], torch.float64), poses, cov, cache["xi1"], params,
            )
            sf32, mf32, phi32 = independent_probabilities(
                wrapper_for(module, meta["geometry"], torch.float32), poses, cov, cache["xi1"], params,
            )
            eligible = np.array([
                s["success"] and s.get("solver_called", True) is not False
                for s in meta["solver_status_infos"]
            ])
            risk = sf if row["source"] == "cc_sf" else mf
            valid = eligible & np.isfinite(risk)
            threshold = params["chance_epsilon"] + 1e-6
            violations = int(np.sum(valid & (risk > threshold)))
            item = {
                "source": row["source"], "seed": int(row["seed"]),
                "samples_independently_recalculated": len(inputs),
                "covariance_max_abs_error": float(np.max(np.abs(cov - cache_cov))),
                "sf_probability_max_abs_error": float(np.max(np.abs(sf - cache["p_sf"]))),
                "mf_probability_sum_max_abs_error": float(np.max(np.abs(mf - cache["r_mf"]))),
                "valid_denominator": int(valid.sum()), "violation_numerator": violations,
                "csv_valid_denominator": int(row["n_valid"]), "csv_violation_numerator": int(row["n_violation"]),
                "eligibility_equal_cache": bool(np.array_equal(eligible, cache["eligible"])),
                "rates_and_counts_match": violations == int(row["n_violation"]) and int(valid.sum()) == int(row["n_valid"]),
                "float32_comparison": {
                    "sf_max_absolute_probability_difference": float(np.max(np.abs(sf32 - sf))),
                    "mf_max_absolute_probability_sum_difference": float(np.max(np.abs(mf32 - mf))),
                    "sf_threshold_classification_mismatches_eligible": int(np.sum(eligible & ((sf32 > threshold) != (sf > threshold)))),
                    "mf_threshold_classification_mismatches_eligible": int(np.sum(eligible & ((mf32 > threshold) != (mf > threshold)))),
                },
                "max_adjacent_heading_delta_rad": float(np.max(np.abs(np.diff(raw["poses"][:, 2])))),
                "mf_domain_inactive_eligible": int(np.sum(eligible & (phi <= params["d_min"]))),
            }
            review["checks"].append(item)
    review["verdict"] = "passes reviewed static alignment and independent pilot arithmetic; native stage-1 metric remains unavailable"
    (OUTPUT / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(review["checks"], indent=2))


if __name__ == "__main__":
    main()
