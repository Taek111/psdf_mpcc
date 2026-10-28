"""CC-MF with an inverse-CDF residual and unchanged affine OCP rows."""

import copy
import json
from dataclasses import fields

import numpy as np
import torch

from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
from models.mf_inverse_cdf import ICDF_BRANCH_NAMES, ICDF_STATUS_NAMES


class MFInverseCDFEvaluationError(RuntimeError):
    """A requested inverse-CDF row could not be evaluated safely."""


class CCMFICDFOptimizerParam(CCMFOptimizerParam):
    def __init__(self):
        super().__init__()
        # Quantile-unit slack penalties; these are independent tuning values,
        # not a conversion of the probability-slack objective.
        self.mf_slack_linear = 100.0
        self.mf_slack_quadratic = 1.0


class CCMFICDFOptimizer(CCMFOptimizer):
    _codegen_name = "cc_mf_icdf"
    _model_name = "differential_drive_cc_mf_icdf"
    diagnostics_label = "CC-MF-ICDF"
    RESIDUAL_KIND = "inverse_cdf"
    RESIDUAL_UNIT = "standard_normal_quantile"

    CONSTRAINT_LOG_FIELDS = CCMFOptimizer.CONSTRAINT_LOG_FIELDS + (
        "residual_kind", "residual_unit", "slack_unit",
        "mf_probability_sum_nominal", "mf_icdf_residual_nominal",
        "mf_probability_sum_frozen", "mf_icdf_residual_frozen",
        "mf_probability_sum_fresh", "mf_icdf_residual_fresh",
        "mf_icdf_affine_minus_frozen", "mf_status", "mf_branch",
        "mf_tail_clipped", "feature_count",
    )
    CYCLE_LOG_FIELDS = CCMFOptimizer.CYCLE_LOG_FIELDS + (
        "residual_kind", "residual_unit", "slack_unit",
        "mf_icdf_residual_nominal", "mf_status", "mf_branch",
        "mf_tail_clipped", "mf_active", "feature_count",
    )

    def _normalize_epsilon_values(self, num_stages):
        epsilon = super()._normalize_epsilon_values(num_stages)
        if np.any(epsilon <= np.finfo(np.float64).tiny) or np.any(epsilon >= 0.5):
            raise ValueError(
                "CC-MF-ICDF requires float64.tiny < chance_epsilon < 0.5"
            )
        return epsilon

    def setup(self, param, system, reference_trajectory, obstacles):
        self.param = param
        self._normalize_epsilon_values(int(param.horizon) + 1)
        for name in ("d_min", "d_col", "d_mf_mask"):
            if not np.isfinite(float(getattr(param, name))):
                raise ValueError(f"{name} must be finite")
        return super().setup(param, system, reference_trajectory, obstacles)

    def _inactive_mf_constant(self, epsilon):
        return np.ones_like(np.asarray(epsilon, dtype=float))

    def _nominal_probability_sum(self, constraint_data, z_bar):
        del z_bar
        return constraint_data["probability_sum"].copy()

    def _raise_if_fatal_evaluation_error(self, error):
        if isinstance(error, MFInverseCDFEvaluationError):
            raise error

    @staticmethod
    def _result_arrays(result):
        arrays = {
            item.name: getattr(result, item.name).detach().cpu().numpy()
            for item in fields(result)
            if isinstance(getattr(result, item.name), torch.Tensor)
        }
        for key, names in (("status", ICDF_STATUS_NAMES), ("branch", ICDF_BRANCH_NAMES)):
            arrays[key] = np.asarray(
                [names[int(code)] for code in arrays[key]], dtype=object,
            )
        return arrays

    @staticmethod
    def _validate_result(arrays, context):
        if not np.isfinite(arrays["phi"]).all() or not np.isfinite(
            arrays["gradient"]
        ).all():
            raise MFInverseCDFEvaluationError(f"{context}: non-finite PSDF guard geometry")
        finite = (
            np.isfinite(arrays["A"]).all(axis=1)
            & np.isfinite(arrays["c"])
            & np.isfinite(arrays["residual"])
            & np.isfinite(arrays["probability_sum"])
        )
        failed = arrays["required"] & (~arrays["valid"] | ~finite)
        if failed.any():
            reasons = ", ".join(
                f"stage {i}: {arrays['status'][i]}" for i in np.flatnonzero(failed)
            )
            raise MFInverseCDFEvaluationError(f"{context}: invalid MF evaluation ({reasons})")

    def _compute_constraint_affine_data(self, z_bar):
        z_bar = np.asarray(z_bar, dtype=float)
        if z_bar.ndim != 2 or z_bar.shape[1] != 8 or not np.isfinite(z_bar).all():
            raise ValueError("z_bar must have shape (H, 8) and finite values")
        size = len(z_bar)
        epsilon = self._normalize_epsilon_values(size)
        stage_mask = self._build_mf_stage_mask(size)
        stage_mask[0] = False
        requested = (
            stage_mask
            & bool(self.param.use_row_mf)
            & bool(self.param.use_obstacle_constraint)
        )
        try:
            if (
                self.psdf_wrapper is None
                or not self.param.use_obstacle_constraint
                or not self.param.use_row_mf
            ):
                if self.psdf_wrapper is not None and self.param.use_obstacle_constraint:
                    phi, gradient = self._compute_exact_psdf(z_bar[:, :3])
                else:
                    phi = np.full(size, 1000.0)
                    gradient = np.zeros((size, 3))
                no_obstacles = self.psdf_wrapper is None or (
                    getattr(self.psdf_wrapper, "active_clusters", None) == 0
                )
                missing = np.full(size, np.nan)
                arrays = {
                    "phi": np.asarray(phi), "gradient": np.asarray(gradient),
                    "A": np.zeros((size, 8)), "c": np.ones(size),
                    "residual": missing.copy(),
                    "probability_sum": np.zeros(size) if no_obstacles else missing.copy(),
                    "feature_count": np.zeros(size, dtype=int),
                    "valid": np.zeros(size, dtype=bool),
                    "required": np.zeros(size, dtype=bool),
                    "status": np.full(size, "not_requested", dtype=object),
                    "branch": np.full(size, "inactive", dtype=object),
                    "tail_clipped": np.zeros(size, dtype=bool),
                }
                frozen = None
            else:
                result = self.psdf_wrapper.forward_mf_icdf(
                    z_bar, epsilon, float(self.param.d_min),
                    stage_requested=requested,
                    d_mf_mask=float(self.param.d_mf_mask),
                    return_frozen_features=bool(
                        self.param.debug_mf or self.param.debug_infeasibility
                    ),
                )
                arrays = self._result_arrays(result)
                frozen = result.frozen_features
            self._validate_result(arrays, "CC-MF-ICDF preprocessing")
        except MFInverseCDFEvaluationError:
            raise
        except Exception as error:
            raise MFInverseCDFEvaluationError(
                f"CC-MF-ICDF preprocessing failed: {error}"
            ) from error

        required = arrays["required"]
        A = np.zeros((size, 8))
        c = self._inactive_mf_constant(epsilon)
        A[required], c[required] = arrays["A"][required], arrays["c"][required]
        guard_A = np.zeros((size, 8))
        guard_A[:, :3] = arrays["gradient"]
        return {
            **arrays,
            "row_mf_enabled": bool(self.param.use_row_mf),
            "guard_A": guard_A,
            "guard_c": (
                arrays["phi"]
                - np.einsum("ij,ij->i", arrays["gradient"], z_bar[:, :3])
                - float(self.param.d_col)
            ),
            "mf_A_raw": arrays["A"], "mf_c_raw": arrays["c"],
            "mf_valid": arrays["valid"].copy(),
            "mf_stage_mask": stage_mask,
            "mf_A": A, "mf_c": c, "mf_mask": required.copy(),
            "epsilon": epsilon, "frozen_features": frozen,
            "nominal_state": z_bar.copy(),
        }

    def _prepare_and_solve(self, solver):
        # A preprocessing exception must not expose a previous cycle's data.
        self._constraint_data = None
        self._current_psdf = None
        self._last_nominal_probability_sum = None
        return super()._prepare_and_solve(solver)

    def compute_diagnostics(self, solver, status, mode, *, failed=False, report=False):
        try:
            states = self._read_diagnostic_states(solver)
            diagnostics = self._build_constraint_diagnostics(solver, states, status, mode)
        except MFInverseCDFEvaluationError:
            raise
        except Exception as error:
            print(f"Warning: could not compute CC-MF-ICDF diagnostics: {error}")
            return None
        name = "_last_failed_constraint_diagnostics" if failed else "_last_constraint_diagnostics"
        setattr(self, name, copy.deepcopy(diagnostics) if failed else diagnostics)
        if report:
            self.report_diagnostics(diagnostics, failed=failed)
        return diagnostics

    def _extend_constraint_diagnostics(self, diagnostics, states, fresh):
        nominal = self._constraint_data
        diagnostics.update(
            residual_kind=self.RESIDUAL_KIND,
            residual_unit=self.RESIDUAL_UNIT,
            slack_unit=self.RESIDUAL_UNIT,
            mf_status=nominal["status"].copy(),
            mf_branch=nominal["branch"].copy(),
            mf_tail_clipped=nominal["tail_clipped"].copy(),
            feature_count=nominal["feature_count"].copy(),
        )
        for suffix, data in (("nominal", nominal), ("fresh", fresh)):
            for name, source in (("mf_probability_sum", "probability_sum"),
                                 ("mf_icdf_residual", "residual")):
                diagnostics[f"{name}_{suffix}"] = None if data is None else data[source].copy()
        frozen = nominal.get("frozen_features")
        if frozen is not None:
            try:
                result = self.psdf_wrapper.evaluate_mf_icdf_frozen(
                    frozen, states, nominal["epsilon"], float(self.param.d_min),
                )
                arrays = self._result_arrays(result)
                self._validate_result(arrays, "CC-MF-ICDF frozen diagnostics")
            except MFInverseCDFEvaluationError:
                raise
            except Exception as error:
                raise MFInverseCDFEvaluationError(
                    f"CC-MF-ICDF frozen diagnostics failed: {error}"
                ) from error
            diagnostics["mf_probability_sum_frozen"] = arrays["probability_sum"]
            diagnostics["mf_icdf_residual_frozen"] = arrays["residual"]
            error = diagnostics["mf_affine_residual"] - arrays["residual"]
            error[~nominal["mf_mask"]] = np.nan
            diagnostics["mf_icdf_affine_minus_frozen"] = error
        return diagnostics

    def _record_constraint_log(self, diagnostics):
        first = len(self._constraint_log_rows)
        super()._record_constraint_log(diagnostics)
        for row in self._constraint_log_rows[first:]:
            stage = row["horizon_stage"]
            row.update(residual_kind=self.RESIDUAL_KIND,
                       residual_unit=self.RESIDUAL_UNIT, slack_unit=self.RESIDUAL_UNIT)
            for key in self.CONSTRAINT_LOG_FIELDS[len(CCMFOptimizer.CONSTRAINT_LOG_FIELDS) + 3:]:
                values = diagnostics.get(key)
                value = None if values is None else values[stage]
                if key in ("mf_status", "mf_branch"):
                    row[key] = None if value is None else str(value)
                elif key == "mf_tail_clipped":
                    row[key] = None if value is None else bool(value)
                elif key == "feature_count":
                    row[key] = None if value is None else int(value)
                else:
                    row[key] = self._finite_or_none(value)

    def _record_experiment_logs(self, mode, status, solver, diagnostics):
        super()._record_experiment_logs(mode, status, solver, diagnostics)
        row = self._cycle_log_rows[-1]
        row.update(residual_kind=self.RESIDUAL_KIND,
                   residual_unit=self.RESIDUAL_UNIT, slack_unit=self.RESIDUAL_UNIT)
        data = self._constraint_data
        for key, source in (
            ("mf_icdf_residual_nominal", "residual"), ("mf_active", "mf_mask"),
            ("mf_tail_clipped", "tail_clipped"), ("feature_count", "feature_count"),
        ):
            row[key] = self._json_vector(None if data is None else data[source])
        for key, source in (("mf_status", "status"), ("mf_branch", "branch")):
            row[key] = json.dumps([] if data is None else data[source].tolist())

    def _print_constraint_diagnostics(self, diagnostics):
        print("CC-MF-ICDF: hard PSDF Row G; soft inverse-CDF Row M and slack in quantile units.")
        for key in ("mf_probability_sum_nominal", "mf_icdf_residual_nominal",
                    "mf_affine_residual", "mf_icdf_residual_frozen",
                    "mf_icdf_residual_fresh", "mf_lower_slack"):
            print(f"CC-MF-ICDF {key}: {self._diagnostic_array_string(diagnostics.get(key))}")
