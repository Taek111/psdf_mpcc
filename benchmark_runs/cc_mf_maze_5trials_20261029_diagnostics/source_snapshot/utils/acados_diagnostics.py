"""Reusable, optimizer-agnostic helpers for acados diagnostics."""

import copy

import numpy as np

__all__ = [
    "AcadosDiagnosticsMixin",
    "build_box_diagnostics",
    "collect_solver_diagnostics",
    "find_box_violations",
    "find_stage_violations",
    "format_diagnostic_array",
    "read_stage_trajectory",
    "report_solver_failure",
]


class AcadosDiagnosticsMixin:
    """Failure-safe diagnostic lifecycle with optimizer-specific hooks.

    Consumers override ``_build_constraint_diagnostics`` and may override the
    solver-collection and reporting hooks. The lifecycle keeps the latest
    snapshots in ``_last_constraint_diagnostics`` and
    ``_last_failed_constraint_diagnostics``.
    """

    diagnostics_label = "Acados"

    def _build_constraint_diagnostics(self, solver, states, status, mode):
        """Build optimizer-specific diagnostics from a solver trajectory."""
        raise NotImplementedError

    def _read_diagnostic_states(self, solver):
        """Read the state trajectory used by optimizer-specific builders."""
        return read_stage_trajectory(solver, "x", range(self.N + 1))

    def _collect_solver_diagnostics(self, solver, status, mode):
        """Collect common solver data; subclasses may extend the result."""
        return collect_solver_diagnostics(
            solver,
            status,
            mode=mode,
            backup_status=getattr(self, "_last_backup_status", None),
        )

    @staticmethod
    def _print_constraint_diagnostics(diagnostics):
        """Optionally report a completed optimizer-specific snapshot."""
        del diagnostics

    @staticmethod
    def _print_infeasibility_diagnostics(diagnostics):
        """Optionally report optimizer-specific infeasibility details."""
        del diagnostics

    def report_diagnostics(
        self,
        diagnostics,
        *,
        failed=False,
        include_constraint=True,
    ):
        """Run reporting hooks without allowing them to stop control."""
        if diagnostics is None:
            return None
        snapshot_name = "failed" if failed else "final"
        try:
            if failed:
                print(
                    f"{self.diagnostics_label} diagnostics captured before "
                    "infeasibility recovery."
                )
            if include_constraint:
                self._print_constraint_diagnostics(diagnostics)
            if failed:
                self._print_infeasibility_diagnostics(diagnostics)
        except Exception as error:
            print(f"Warning: could not report {snapshot_name} diagnostics: {error}")
        return diagnostics

    def compute_diagnostics(
        self,
        solver,
        status,
        mode,
        *,
        failed=False,
        report=False,
    ):
        """Compute optional diagnostics without allowing them to stop control."""
        snapshot_name = "failed" if failed else "final"
        try:
            states = self._read_diagnostic_states(solver)
            diagnostics = self._build_constraint_diagnostics(
                solver,
                states,
                status,
                mode,
            )
        except Exception as error:
            print(f"Warning: could not compute {snapshot_name} diagnostics: {error}")
            return None

        cache_attribute = (
            "_last_failed_constraint_diagnostics"
            if failed
            else "_last_constraint_diagnostics"
        )
        try:
            cached = copy.deepcopy(diagnostics) if failed else diagnostics
            setattr(self, cache_attribute, cached)
        except Exception as error:
            try:
                setattr(self, cache_attribute, None)
            except Exception:
                pass
            print(f"Warning: could not cache {snapshot_name} diagnostics: {error}")

        if report:
            self.report_diagnostics(diagnostics, failed=failed)
        return diagnostics


def read_stage_trajectory(solver, field, stages):
    """Stack one acados field over the supplied stage indices."""
    return np.stack(
        [np.asarray(solver.get(stage, field), dtype=float) for stage in stages],
        axis=0,
    )


def collect_solver_diagnostics(
    solver,
    status,
    *,
    mode=None,
    backup_status=None,
):
    """Collect residual and QP statistics without raising on missing APIs."""
    diagnostics = {
        "status": int(status),
        "mode": None if mode is None else str(mode),
        "backup_status": backup_status,
        "nlp_residuals": None,
        "qp_status": None,
        "qp_iterations": None,
        "statistics": None,
    }
    try:
        diagnostics["nlp_residuals"] = np.asarray(
            solver.get_residuals(recompute=True),
            dtype=float,
        )
    except Exception:
        try:
            diagnostics["nlp_residuals"] = np.asarray(
                solver.get_residuals(),
                dtype=float,
            )
        except Exception:
            pass

    for key, statistic in (("qp_status", "qp_stat"), ("qp_iterations", "qp_iter")):
        try:
            values = solver.get_stats(statistic)
            if values is not None:
                diagnostics[key] = np.asarray(values, dtype=float)
        except Exception:
            pass

    try:
        diagnostics["statistics"] = np.asarray(
            solver.get_stats("statistics"),
            dtype=float,
        )
    except Exception:
        pass
    return diagnostics


def build_box_diagnostics(prefix, values, lower, upper, names, stages):
    """Build a consistently named residual dictionary for box constraints."""
    values = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    names = np.asarray(names, dtype=object).reshape(-1)
    stages = np.asarray(stages, dtype=int).reshape(-1)
    if values.ndim != 2:
        raise ValueError("box values must be a two-dimensional array")
    if values.shape != (stages.size, names.size):
        raise ValueError("box values, stages, and names have incompatible shapes")
    if lower.size != names.size or upper.size != names.size:
        raise ValueError("box bounds and names have incompatible shapes")
    return {
        f"{prefix}_stages": stages,
        f"{prefix}_names": names,
        f"{prefix}_values": values,
        f"{prefix}_lower_bounds": lower,
        f"{prefix}_upper_bounds": upper,
        f"{prefix}_lower_residual": values - lower,
        f"{prefix}_upper_residual": upper - values,
    }


def find_stage_violations(name, residual, tolerance, *, stages=None):
    """Return one candidate dictionary for negative stage residuals, if any."""
    residual = np.asarray(residual, dtype=float).reshape(-1)
    bad = np.flatnonzero(np.isfinite(residual) & (residual < -tolerance))
    if not bad.size:
        return None
    if stages is None:
        stage_ids = bad
    else:
        stages = np.asarray(stages, dtype=int).reshape(-1)
        if stages.shape != residual.shape:
            raise ValueError("stages must contain one ID per residual")
        stage_ids = stages[bad]
    return {
        "constraint": str(name),
        "stages": stage_ids,
        "minimum_residual": float(np.nanmin(residual[bad])),
    }


def find_box_violations(bounds, prefix, tolerance):
    """Return one candidate dictionary for violated box constraints, if any."""
    residual = np.minimum(
        bounds[f"{prefix}_lower_residual"],
        bounds[f"{prefix}_upper_residual"],
    )
    bad = np.argwhere(np.isfinite(residual) & (residual < -tolerance))
    if not bad.size:
        return None
    return {
        "constraint": f"{prefix}_bounds",
        "stage_fields": [
            (
                int(bounds[f"{prefix}_stages"][row]),
                str(bounds[f"{prefix}_names"][column]),
            )
            for row, column in bad
        ],
        "minimum_residual": float(np.nanmin(residual)),
    }


def format_diagnostic_array(values):
    """Format diagnostic arrays consistently across optimizers."""
    return np.array2string(
        np.asarray(values),
        precision=5,
        suppress_small=False,
        max_line_width=200,
    )


def report_solver_failure(solver, status, *, max_iteration_message=None):
    """Print the common acados failure summary and return its diagnostics."""
    print(f"Acados solver failed with status {status}")
    if status == 2 and max_iteration_message:
        print(max_iteration_message)
    diagnostics = collect_solver_diagnostics(solver, status)
    residuals = diagnostics["nlp_residuals"]
    if residuals is not None:
        print(f"Acados residuals [stat, eq, ineq, comp]: {residuals}")
    try:
        print(f"Acados SQP iterations: {int(solver.get_stats('sqp_iter'))}")
    except Exception:
        pass
    return diagnostics
