"""CC-MF with a distance-valued implicit clearance margin.

The OCP, covariance dynamics and QP recovery belong to CCMFOptimizer. Only
the preprocessing residual changes: a GPU scalar root and its analytic
Jacobian supply the same nine-parameter affine Row M interface.
"""

import copy
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from control.cc_mf_imp_errors import MFImplicitRootError
from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
from models.mf_implicit import MFImplicitRootStatus, validate_root_settings


class CCMFImpOptimizerParam(CCMFOptimizerParam):
    def __init__(self):
        super().__init__()
        self.mf_imp_root_max_iter = 12
        self.mf_imp_root_atol_m = 1e-8
        self.mf_imp_root_rtol = 1e-10
        self.mf_imp_root_prob_atol = 1e-10
        # Distance slack: cost/m and cost/m^2. Provisional values to retune;
        # equal numbers do not reproduce the probability-slack objective.
        self.mf_slack_linear = 100.0
        self.mf_slack_quadratic = 1.0


class CCMFImpOptimizer(CCMFOptimizer):
    _codegen_name = "cc_mf_imp"
    _model_name = "differential_drive_cc_mf_imp"
    diagnostics_label = "CC-MF-IMP"
    CONSTRAINT_LOG_FIELDS = CCMFOptimizer.CONSTRAINT_LOG_FIELDS + (
        "residual_kind",
        "residual_unit",
        "slack_unit",
    )

    CYCLE_LOG_FIELDS = CCMFOptimizer.CYCLE_LOG_FIELDS + (
        "residual_kind",
        "slack_unit",
        "mf_probability_sum_nominal",
        "mf_imp_margin_nominal_m",
        "mf_imp_slack_m",
        "mf_active",
        "root_valid",
        "root_status",
        "root_residual",
        "root_bracket_width_m",
        "root_error_bound_m",
        "root_iterations",
        "preprocessing_time_s",
        "control_cycle_time_s",
        "solver_time_s",
        "failure_context",
        "failure_reason",
    )
    IMPLICIT_LOG_FIELDS = (
        "solve_step",
        "horizon_stage",
        "evaluation_context",
        "residual_kind",
        "slack_unit",
        "mf_probability_sum_nominal",
        "mf_imp_margin_nominal_m",
        "mf_imp_slack_m",
        "mf_active",
        "root_valid",
        "root_status",
        "root_residual",
        "root_bracket_width_m",
        "root_error_bound_m",
        "root_iterations",
        "feature_count",
        "row_m_affine_residual_m",
        "mf_imp_margin_frozen_m",
        "mf_probability_sum_frozen",
        "mf_imp_margin_fresh_m",
        "mf_probability_sum_fresh",
    )

    def __init__(self):
        self._implicit_halted = False
        self._implicit_failure = None
        self._implicit_failure_features = None
        self._implicit_log_rows = []
        self._implicit_context = "nominal"
        self._implicit_preprocessing_time = 0.0
        self._implicit_cycle_start = None
        self._implicit_qp_statuses = []
        super().__init__()

    def cleanup(self):
        super().cleanup()
        # A cleanup is also used during exception unwinding. Only explicit
        # reset (or a new instance) releases a failed controller's latch.
        self._implicit_log_rows = []
        self._implicit_preprocessing_time = 0.0
        self._implicit_qp_statuses = []

    def reset(self):
        self.cleanup()
        self._implicit_halted = False
        self._implicit_failure = None
        self._implicit_failure_features = None

    def _check_implicit_latch(self):
        if self._implicit_halted:
            raise MFImplicitRootError(
                "CC-MF-IMP is halted after an implicit evaluation failure; "
                "call reset() or create a new controller before restarting."
            )

    def _root_options(self):
        return {
            "root_max_iter": self.param.mf_imp_root_max_iter,
            "root_atol_m": self.param.mf_imp_root_atol_m,
            "root_rtol": self.param.mf_imp_root_rtol,
            "root_prob_atol": self.param.mf_imp_root_prob_atol,
        }

    def _validate_implicit_params(self, param):
        self._normalize_epsilon_values(int(param.horizon) + 1)
        for name in ("d_min", "d_col", "d_mf_mask"):
            if not np.isfinite(float(getattr(param, name))):
                raise ValueError(f"{name} must be finite")
        validate_root_settings(**self._root_options())

    def setup(self, param, system, reference_trajectory, obstacles):
        self._check_implicit_latch()
        self.param = param
        self._validate_implicit_params(param)
        return super().setup(param, system, reference_trajectory, obstacles)

    def _normalize_epsilon_values(self, num_stages):
        epsilon = super()._normalize_epsilon_values(num_stages)
        if np.any(epsilon >= 0.5):
            raise ValueError(
                "CC-MF-IMP upper-bound Newton requires 0 < chance_epsilon < 0.5"
            )
        return epsilon

    def _inactive_mf_constant(self, epsilon):
        return np.ones_like(np.asarray(epsilon, dtype=float))

    def _nominal_probability_sum(self, constraint_data, z_bar):
        del z_bar
        return constraint_data["probability_sum"].copy()

    def _raise_if_fatal_evaluation_error(self, error):
        if isinstance(error, MFImplicitRootError):
            raise error

    @staticmethod
    def _result_arrays(result):
        """Transfer stage summaries only; feature snapshots stay on device."""
        return {
            item.name: getattr(result, item.name).detach().cpu().numpy()
            for item in fields(result)
            if isinstance(getattr(result, item.name), torch.Tensor)
        }

    @staticmethod
    def _status_names(codes):
        return np.asarray(
            [MFImplicitRootStatus(int(code)).name.lower() for code in codes],
            dtype=object,
        )

    def _abort_evaluation(
        self, message, z_bar, *, arrays=None, features=None, cause=None
    ):
        self._implicit_halted = True
        self._implicit_failure = {
            "message": str(message),
            "context": self._implicit_context,
            "solve_step": int(self._solve_count),
            "nominal_state": np.array(z_bar, copy=True),
            "epsilon": np.asarray(self.param.chance_epsilon),
            "d_min": float(self.param.d_min),
            "settings": self._root_options(),
            "stage_results": arrays,
        }
        self._implicit_failure_features = features
        error = MFImplicitRootError(str(message))
        error.context = self._implicit_context
        error.diagnostics = self._implicit_failure
        if cause is not None:
            raise error from cause
        raise error

    def _compute_constraint_affine_data(self, z_bar):
        z_bar = np.asarray(z_bar, dtype=float)
        if z_bar.ndim != 2 or z_bar.shape[1] != 8 or not np.isfinite(z_bar).all():
            raise ValueError("z_bar must have shape (H, 8) and finite values")
        size = z_bar.shape[0]
        epsilon = self._normalize_epsilon_values(size)
        stage_mask = self._build_mf_stage_mask(size)
        stage_mask[0] = False
        requested = (
            stage_mask
            & bool(self.param.use_row_mf)
            & bool(self.param.use_obstacle_constraint)
        )
        started = time.perf_counter()
        try:
            if (
                self.psdf_wrapper is None
                or not self.param.use_obstacle_constraint
                or not self.param.use_row_mf
            ):
                # A disabled Row M needs only pose geometry for the hard guard.
                no_obstacles = (
                    self.psdf_wrapper is not None
                    and self.param.use_obstacle_constraint
                    and self.psdf_wrapper.active_clusters == 0
                )
                if self.psdf_wrapper is not None and self.param.use_obstacle_constraint:
                    phi, gradient = self._compute_exact_psdf(z_bar[:, :3])
                else:
                    phi = np.full(size, 1000.0)
                    gradient = np.zeros((size, 3))
                zeros = np.zeros(size, dtype=bool)
                missing = np.full(size, np.nan)
                arrays = {
                    "phi": phi,
                    "gradient": gradient,
                    "A": np.full((size, 8), np.nan),
                    "c": missing.copy(),
                    "margin": missing.copy(),
                    "probability_sum": (
                        np.zeros(size)
                        if self.psdf_wrapper is None
                        or not self.param.use_obstacle_constraint
                        or no_obstacles
                        else missing.copy()
                    ),
                    "root_residual": missing.copy(),
                    "root_bracket_width": missing.copy(),
                    "root_error_bound": missing.copy(),
                    "root_iterations": np.zeros(size, dtype=int),
                    "feature_count": np.zeros(size, dtype=int),
                    "root_valid": zeros.copy(),
                    "mf_required": zeros.copy(),
                    "root_status": np.full(
                        size,
                        "no_obstacles" if no_obstacles else "not_requested",
                        dtype=object,
                    ),
                }
                frozen = None
            else:
                result = self.psdf_wrapper.forward_mf_imp(
                    z_bar,
                    epsilon,
                    float(self.param.d_min),
                    stage_requested=requested,
                    d_mf_mask=float(self.param.d_mf_mask),
                    return_frozen_features=bool(
                        self.param.debug_mf or self.param.debug_infeasibility
                    ),
                    **self._root_options(),
                )
                arrays = self._result_arrays(result)
                arrays["root_status"] = self._status_names(arrays["root_status"])
                phi, gradient = arrays["phi"], arrays["gradient"]
                if not np.isfinite(phi).all() or not np.isfinite(gradient).all():
                    self._abort_evaluation(
                        "non-finite signed PSDF guard geometry",
                        z_bar,
                        arrays=arrays,
                        features=getattr(result, "failure_features", None),
                    )
                required = np.asarray(arrays["mf_required"], dtype=bool)
                finite = np.isfinite(arrays["A"]).all(axis=1) & np.isfinite(
                    arrays["c"]
                )
                failed = required & (~arrays["root_valid"] | ~finite)
                if failed.any():
                    reasons = ", ".join(
                        f"stage {i}: {arrays['root_status'][i]}"
                        for i in np.flatnonzero(failed)
                    )
                    self._abort_evaluation(
                        f"CC-MF-IMP implicit evaluation failed ({reasons})",
                        z_bar,
                        arrays=arrays,
                        features=getattr(result, "failure_features", None),
                    )
                frozen = result.frozen_features
        except MFImplicitRootError:
            raise
        except Exception as error:
            self._abort_evaluation(
                "CC-MF-IMP batch evaluation failed: " + str(error),
                z_bar,
                cause=error,
            )
        finally:
            if self._implicit_context in ("nominal", "backup"):
                # CPU summary transfer above completes GPU work, so this wall
                # interval includes geometry/root/gradient execution.
                self._implicit_preprocessing_time += time.perf_counter() - started

        required = np.asarray(arrays["mf_required"], dtype=bool)
        A = np.zeros((size, 8))
        c = self._inactive_mf_constant(epsilon)
        A[required], c[required] = arrays["A"][required], arrays["c"][required]
        guard_A = np.zeros((size, 8))
        guard_A[:, :3] = gradient
        data = {
            **arrays,
            "row_mf_enabled": bool(self.param.use_row_mf),
            "guard_A": guard_A,
            "guard_c": (
                phi
                - np.einsum("ij,ij->i", gradient, z_bar[:, :3])
                - float(self.param.d_col)
            ),
            "mf_A_raw": arrays["A"],
            "mf_c_raw": arrays["c"],
            "mf_valid": arrays["root_valid"].copy(),
            "mf_stage_mask": stage_mask,
            "mf_A": A,
            "mf_c": c,
            "mf_mask": required.copy(),
            "epsilon": epsilon,
            "frozen_features": frozen,
            "nominal_state": z_bar.copy(),
        }
        return data

    def _prepare_and_solve(self, solver):
        context = self._implicit_context
        self._implicit_context = "backup" if solver is self.backup_solver else "nominal"
        try:
            status = super()._prepare_and_solve(solver)
            if status is not None:
                self._implicit_qp_statuses.append(int(status))
            return status
        finally:
            self._implicit_context = context

    def compute_diagnostics(self, solver, status, mode, *, failed=False, report=False):
        # Common diagnostics are best effort. Implicit model failures must
        # propagate through both diagnostic catch layers and abort this run.
        context = self._implicit_context
        self._implicit_context = "debug_fresh_" + str(mode)
        try:
            states = self._read_diagnostic_states(solver)
            diagnostics = self._build_constraint_diagnostics(solver, states, status, mode)
        except MFImplicitRootError:
            raise
        except Exception as error:
            print(f"Warning: could not compute CC-MF-IMP diagnostics: {error}")
            return None
        finally:
            self._implicit_context = context
        name = (
            "_last_failed_constraint_diagnostics"
            if failed
            else "_last_constraint_diagnostics"
        )
        setattr(self, name, copy.deepcopy(diagnostics) if failed else diagnostics)
        if report:
            self.report_diagnostics(diagnostics, failed=failed)
        return diagnostics

    def _extend_constraint_diagnostics(self, diagnostics, states, fresh):
        nominal = self._constraint_data
        diagnostics.update(
            {
                "residual_kind": "implicit_clearance_margin",
                "residual_unit": "m",
                "slack_unit": "m",
                "mf_imp_margin_nominal_m": nominal["margin"].copy(),
                "mf_probability_sum_nominal": nominal["probability_sum"].copy(),
                "mf_imp_margin_fresh_m": (
                    None if fresh is None else fresh["margin"].copy()
                ),
                "mf_probability_sum_fresh": (
                    None if fresh is None else fresh["probability_sum"].copy()
                ),
                "root_status": nominal["root_status"].copy(),
                "root_valid": nominal["root_valid"].copy(),
                "root_iterations": nominal["root_iterations"].copy(),
                "root_error_bound_m": nominal["root_error_bound"].copy(),
            }
        )
        frozen = nominal.get("frozen_features")
        if frozen is not None:
            context = self._implicit_context
            self._implicit_context = context.replace("debug_fresh_", "debug_frozen_", 1)
            try:
                result = self.psdf_wrapper.evaluate_mf_imp_frozen(
                    frozen,
                    states,
                    nominal["epsilon"],
                    float(self.param.d_min),
                    **self._root_options(),
                )
                arrays = self._result_arrays(result)
                arrays["root_status"] = self._status_names(arrays["root_status"])
                failed = arrays["mf_required"] & ~arrays["root_valid"]
                if failed.any():
                    self._abort_evaluation(
                        "CC-MF-IMP frozen debug root failed",
                        states,
                        arrays=arrays,
                        features=getattr(result, "failure_features", None),
                    )
                diagnostics["mf_imp_margin_frozen_m"] = arrays["margin"]
                diagnostics["mf_probability_sum_frozen"] = arrays["probability_sum"]
            except MFImplicitRootError:
                raise
            except Exception as error:
                self._abort_evaluation(
                    "CC-MF-IMP frozen debug evaluation failed: " + str(error),
                    states,
                    cause=error,
                )
            finally:
                self._implicit_context = context
        return diagnostics

    def get_implicit_log_rows(self):
        return [row.copy() for row in self._implicit_log_rows]

    def get_implicit_log_fields(self):
        return list(self.IMPLICIT_LOG_FIELDS)

    def _record_constraint_log(self, diagnostics):
        first = len(self._constraint_log_rows)
        super()._record_constraint_log(diagnostics)
        for row in self._constraint_log_rows[first:]:
            row.update(
                residual_kind="implicit_clearance_margin",
                residual_unit="m",
                slack_unit="m",
            )

    def _print_constraint_diagnostics(self, diagnostics):
        active = np.asarray(diagnostics["mf_mask"], dtype=bool)
        print(
            f"CC-MF-IMP: hard Row G; soft implicit margin Row M [m], "
            f"active={int(active.sum())}/{self.N}; lower slack [m]."
        )
        for label, key in (
            ("nominal margin [m]", "mf_imp_margin_nominal_m"),
            ("returned affine margin [m]", "mf_affine_residual"),
            ("returned frozen margin [m]", "mf_imp_margin_frozen_m"),
            ("returned fresh margin [m]", "mf_imp_margin_fresh_m"),
            ("nominal probability sum", "mf_probability_sum_nominal"),
        ):
            print(
                f"CC-MF-IMP {label}: "
                f"{self._diagnostic_array_string(diagnostics.get(key))}"
            )

    def _append_implicit_rows(
        self, data, *, solver=None, diagnostics=None, context="nominal"
    ):
        if data is None:
            return
        count = len(data["margin"])
        slacks = np.full(count, np.nan)
        affine = np.full(count, np.nan)
        if solver is not None:
            states, _ = self._read_solver_trajectory(solver)
            affine = np.einsum("ij,ij->i", data["mf_A"], states) + data["mf_c"]
            if self._last_solve_mode not in ("safe_stop", "safe_stop_stage0"):
                slacks = self._read_lower_slack_trajectory(solver)[:, 1]
        diagnostics = diagnostics or {}
        for stage in range(count):
            row = {
                "solve_step": self._solve_count,
                "horizon_stage": stage,
                "evaluation_context": context,
                "residual_kind": "implicit_clearance_margin",
                "slack_unit": "m",
                "mf_probability_sum_nominal": self._finite_or_none(
                    data["probability_sum"][stage]
                ),
                "mf_imp_margin_nominal_m": self._finite_or_none(data["margin"][stage]),
                "mf_imp_slack_m": self._finite_or_none(slacks[stage]),
                "mf_active": bool(data["mf_required"][stage]),
                "root_valid": bool(data["root_valid"][stage]),
                "root_status": str(data["root_status"][stage]),
                "root_residual": self._finite_or_none(data["root_residual"][stage]),
                "root_bracket_width_m": self._finite_or_none(
                    data["root_bracket_width"][stage]
                ),
                "root_error_bound_m": self._finite_or_none(
                    data["root_error_bound"][stage]
                ),
                "root_iterations": int(data["root_iterations"][stage]),
                "feature_count": int(data["feature_count"][stage]),
                "row_m_affine_residual_m": (
                    self._finite_or_none(affine[stage])
                    if data["mf_required"][stage]
                    else None
                ),
            }
            for name in (
                "mf_imp_margin_frozen_m",
                "mf_probability_sum_frozen",
                "mf_imp_margin_fresh_m",
                "mf_probability_sum_fresh",
            ):
                values = diagnostics.get(name)
                row[name] = None if values is None else self._finite_or_none(values[stage])
            self._implicit_log_rows.append(row)

    def _finish_solve(self, mode, status, solver, diagnostics):
        super()._finish_solve(mode, status, solver, diagnostics)
        data = self._constraint_data
        self._append_implicit_rows(data, solver=solver, diagnostics=diagnostics)
        row = self._cycle_log_rows[-1]
        row.update(
            {
                "residual_kind": "implicit_clearance_margin",
                "slack_unit": "m",
                "mf_probability_sum_nominal": row["mf_probability_sum"],
                "mf_imp_slack_m": row["mf_slack"],
                "preprocessing_time_s": self._implicit_preprocessing_time,
                "control_cycle_time_s": time.perf_counter() - self._implicit_cycle_start,
                "solver_time_s": (
                    self._cycle_solver_time if self._implicit_qp_statuses else None
                ),
                "failure_context": None,
                "failure_reason": None,
            }
        )
        for key, source in (
            ("mf_imp_margin_nominal_m", "margin"),
            ("mf_active", "mf_required"),
            ("root_valid", "root_valid"),
            ("root_residual", "root_residual"),
            ("root_bracket_width_m", "root_bracket_width"),
            ("root_error_bound_m", "root_error_bound"),
            ("root_iterations", "root_iterations"),
        ):
            row[key] = self._json_vector(data[source])
        row["root_status"] = json.dumps(data["root_status"].tolist())
        row["control_cycle_time_s"] = time.perf_counter() - self._implicit_cycle_start

    def record_control_cycle_time(self, elapsed_s):
        """Replace the direct-solver timing with the host control-loop boundary.

        The simulation calls this after obstacle updates, setup, solve, input
        selection and pose logging, before plant integration or failure dumps.
        Direct solve_nlp callers retain the inner timing from _finish_solve.
        """
        if self._cycle_log_rows:
            self._cycle_log_rows[-1]["control_cycle_time_s"] = float(elapsed_s)

    def solve_nlp(self):
        self._check_implicit_latch()
        self._implicit_preprocessing_time = 0.0
        self._implicit_cycle_start = time.perf_counter()
        self._implicit_qp_statuses = []
        self._constraint_data = None
        try:
            return super().solve_nlp()
        except MFImplicitRootError as error:
            self._implicit_halted = True
            if self._implicit_failure is None:
                self._implicit_failure = {
                    "message": str(error),
                    "context": self._implicit_context,
                }
            self._last_solve_mode = "implicit_root_failure"
            try:
                self._record_implicit_failure_cycle(error)
            except Exception as logging_error:
                # Diagnostics must never replace the exception that halts the
                # controller. The original snapshot remains exportable.
                print(
                    f"Warning: CC-MF-IMP failure-cycle logging failed: {logging_error}"
                )
            raise

    def _record_implicit_failure_cycle(self, error):
        arrays = self._implicit_failure.get("stage_results")
        self._append_implicit_rows(arrays, context=self._implicit_failure["context"])
        row = {
            "time": self._get_cycle_log_time(),
            "solve_mode": "implicit_root_failure",
            "solver_status": (
                self._implicit_qp_statuses[-1] if self._implicit_qp_statuses else None
            ),
            "u0": "[]",
            "s": self._current_s0,
            "psdf": self._finite_or_none(self._current_psdf),
            "mf_probability_sum": "[]",
            "mf_slack": "[]",
            "mf_imp_slack_m": "[]",
            "residual_kind": "implicit_clearance_margin",
            "slack_unit": "m",
            "preprocessing_time_s": self._implicit_preprocessing_time,
            "control_cycle_time_s": time.perf_counter() - self._implicit_cycle_start,
            "solver_time_s": (
                self._cycle_solver_time if self._implicit_qp_statuses else None
            ),
            "failure_context": self._implicit_failure["context"],
            "failure_reason": str(error),
        }
        if arrays is not None:
            row["mf_probability_sum"] = self._json_vector(arrays["probability_sum"])
            row["mf_probability_sum_nominal"] = row["mf_probability_sum"]
            for key, source in (
                ("mf_imp_margin_nominal_m", "margin"),
                ("mf_active", "mf_required"),
                ("root_valid", "root_valid"),
                ("root_residual", "root_residual"),
                ("root_bracket_width_m", "root_bracket_width"),
                ("root_error_bound_m", "root_error_bound"),
                ("root_iterations", "root_iterations"),
            ):
                row[key] = self._json_vector(arrays[source])
            row["root_status"] = json.dumps(arrays["root_status"].tolist())
        row["control_cycle_time_s"] = time.perf_counter() - self._implicit_cycle_start
        self._cycle_log_rows.append(row)

    @classmethod
    def _snapshot_json(cls, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return cls._snapshot_json(value.tolist())
        if isinstance(value, dict):
            return {str(k): cls._snapshot_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._snapshot_json(v) for v in value]
        if isinstance(value, (float, np.floating)):
            return cls._finite_or_none(value)
        if isinstance(value, np.generic):
            return value.item()
        return value

    def save_implicit_failure_snapshot(self, output_dir):
        if self._implicit_failure is None:
            return None
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        payload = dict(self._implicit_failure)
        features = self._implicit_failure_features
        arrays = payload.get("stage_results")
        if features is not None and arrays is not None:
            failed = arrays["mf_required"] & ~arrays["root_valid"]
            failed |= ~np.isfinite(arrays["phi"]) | ~np.isfinite(
                arrays["gradient"]
            ).all(axis=1)
            stages = np.flatnonzero(failed)
            payload["failure_stages"] = stages
            # Copy feature data to the CPU only on this failure export path,
            # keeping the dump limited to stages needed for reproduction.
            features = {
                key: (
                    value[torch.as_tensor(stages, device=value.device)]
                    if isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == len(failed)
                    else value
                )
                for key, value in features.items()
            }
        payload["features"] = features
        path = directory / f"cc_mf_imp_failure_{self._solve_count:06d}.json"
        path.write_text(
            json.dumps(
                self._snapshot_json(payload),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        return path
