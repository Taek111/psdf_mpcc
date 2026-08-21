"""RMPCC Row G/MF diagnostics built on the common acados lifecycle.

Only RMPCC safety-row interpretation lives here. Solver collection, caching,
and failure-safe diagnostic execution are inherited from
``utils.acados_diagnostics.AcadosDiagnosticsMixin``.
"""

import numpy as np

from utils.acados_diagnostics import (
    AcadosDiagnosticsMixin,
    build_box_diagnostics,
    find_box_violations,
    find_stage_violations,
    format_diagnostic_array,
)

__all__ = ["RMPCCDiagnosticsMixin"]


class RMPCCDiagnosticsMixin(AcadosDiagnosticsMixin):
    """Interpret generic acados diagnostics using RMPCC Row G/MF semantics."""

    diagnostics_label = "RMPCC-MF"

    _diagnostic_array_string = staticmethod(format_diagnostic_array)

    def _read_lower_slack_trajectory(self, solver):
        # Public layout is [hard Row G, soft Row M].  Only Row M has a solver
        # slack, so Row G is represented by zero at constrained stages.
        lower_slacks = np.full((self.N + 1, 2), np.nan, dtype=float)
        lower_slacks[1:, 0] = 0.0
        for stage in range(1, self.N + 1):
            try:
                stage_slacks = np.asarray(
                    solver.get(stage, "sl"),
                    dtype=float,
                ).reshape(-1)
                if stage_slacks.size:
                    lower_slacks[stage, 1] = stage_slacks[0]
            except Exception:
                continue
        return lower_slacks

    def _sample_first_interval_psdf(self, solver, states):
        num_substeps = max(int(self.param.first_interval_substeps), 1)
        pose = np.asarray(states[0, :3], dtype=float).copy()
        poses = [pose.copy()]
        if self.N:
            physical_input = np.asarray(solver.get(0, "u"), dtype=float)[:2]
            substep_dt = float(self.param.tf) / (self.N * num_substeps)
            for _ in range(num_substeps):
                pose = np.asarray(
                    self._system_dynamics.forward_dynamics(
                        pose,
                        physical_input,
                        substep_dt,
                    ),
                    dtype=float,
                )
                poses.append(np.asarray(pose, dtype=float).copy())
        poses = np.asarray(poses, dtype=float)
        phi, _ = self._compute_exact_psdf(poses)
        return poses, np.asarray(phi, dtype=float)

    def _collect_solver_diagnostics(self, solver, status, mode):
        """Extend common solver diagnostics with RMPCC backup-QP rows."""
        diagnostics = super()._collect_solver_diagnostics(solver, status, mode)
        statistics = diagnostics["statistics"]
        if (
            mode == "backup_feasible_qp"
            and statistics is not None
            and statistics.ndim == 2
            and statistics.shape[0] >= 11
            and statistics.shape[1]
        ):
            diagnostics["qp_status"] = statistics[[5, 7, 9], -1].copy()
            diagnostics["qp_iterations"] = statistics[[6, 8, 10], -1].copy()
        return diagnostics

    def _build_bound_diagnostics(self, solver, states):
        s_lower, s_upper = self._get_current_s_bounds(self.param)
        indices, lower, upper = self._get_augmented_state_bounds(
            self.param,
            s_lower,
            s_upper,
        )
        all_state_names = np.array(
            ["x", "y", "theta", "s", "P_f", "P_l", "P_psi", "P_lpsi"],
            dtype=object,
        )
        state_values = np.asarray(states[1:], dtype=float)[:, indices]

        input_lower = np.array(
            [self.param.vmin, self.param.omegamin, 0.0],
            dtype=float,
        )
        input_upper = np.array(
            [
                self.param.vmax,
                self.param.omegamax,
                self._get_effective_v_s_max(self.param),
            ],
            dtype=float,
        )
        try:
            input_values = np.stack(
                [
                    np.asarray(solver.get(stage, "u"), dtype=float)[:3]
                    for stage in range(self.N)
                ],
                axis=0,
            )
        except Exception:
            input_values = np.full((self.N, 3), np.nan, dtype=float)

        diagnostics = build_box_diagnostics(
            "state",
            state_values,
            lower,
            upper,
            all_state_names[indices],
            np.arange(1, self.N + 1),
        )
        diagnostics.update(
            build_box_diagnostics(
                "input",
                input_values,
                input_lower,
                input_upper,
                ["v", "omega", "v_s"],
                np.arange(self.N),
            )
        )
        return diagnostics

    def _classify_infeasibility_candidates(self, diagnostics):
        """Identify violated constraints on an iterate; this is not a proof."""
        tolerance = max(float(self.param.constraint_debug_tolerance), 0.0)
        candidates = []

        for name, residual in (
            ("row_g_hard_guard", diagnostics["guard_affine_residual"]),
            (
                "row_m_soft_after_slack",
                diagnostics["solver"]["qp_fixed_row_lower_residual"][:, 1],
            ),
        ):
            candidate = find_stage_violations(name, residual, tolerance)
            if candidate is not None:
                candidates.append(candidate)

        bounds = diagnostics["bounds"]
        for prefix in ("state", "input"):
            candidate = find_box_violations(bounds, prefix, tolerance)
            if candidate is not None:
                candidates.append(candidate)

        nlp_residuals = diagnostics["solver"].get("nlp_residuals")
        if nlp_residuals is not None:
            nlp_residuals = np.asarray(nlp_residuals, dtype=float).reshape(-1)
            if (
                nlp_residuals.size > 1
                and np.isfinite(nlp_residuals[1])
                and nlp_residuals[1] > tolerance
            ):
                candidates.append(
                    {
                        "constraint": "dynamics_or_stage0_equality",
                        "maximum_residual": float(nlp_residuals[1]),
                    }
                )

        inactive_mf = ~np.asarray(diagnostics["mf_mask"], dtype=bool)
        inactive_mf[0] = False
        mf_residual = np.asarray(diagnostics["mf_affine_residual"], dtype=float)
        inactive_bad = np.flatnonzero(
            inactive_mf
            & np.isfinite(mf_residual)
            & (mf_residual < -tolerance)
        )
        return {
            "tolerance": tolerance,
            "candidates": candidates,
            "no_direct_iterate_violation_found": not candidates,
            "inactive_mf_violation_stages": inactive_bad,
            "note": (
                "Candidates use the returned failed iterate. A nonzero QP status "
                "can still be numerical or caused by a QP linearization."
            ),
        }

    def _build_constraint_diagnostics(self, solver, states, status, mode):
        constraint_data = self._constraint_data
        if constraint_data is None:
            raise RuntimeError("constraint linearization is unavailable")
        d_col = float(self.param.d_col)
        d_mf_mask = float(self.param.d_mf_mask)
        phi_exact, _ = self._compute_exact_psdf(states[:, :3])
        phi_exact = np.asarray(phi_exact, dtype=float)

        guard_affine_residual = (
            np.einsum("ij,ij->i", constraint_data["guard_A"], states)
            + constraint_data["guard_c"]
        )
        guard_fresh_residual = phi_exact - d_col
        mf_affine_residual = (
            np.einsum("ij,ij->i", constraint_data["mf_A"], states)
            + constraint_data["mf_c"]
        )
        try:
            fresh = self._compute_constraint_affine_data(states)
            mf_fresh_raw_residual = (
                np.einsum("ij,ij->i", fresh["mf_A_raw"], states)
                + fresh["mf_c_raw"]
            )
            raw_valid = fresh["mf_valid"] & np.isfinite(
                fresh["mf_A_raw"]
            ).all(axis=1) & np.isfinite(fresh["mf_c_raw"])
            mf_fresh_raw_residual[~raw_valid] = np.nan
            mf_fresh_effective_residual = (
                np.einsum("ij,ij->i", fresh["mf_A"], states) + fresh["mf_c"]
            )
            mf_fresh_mask = fresh["mf_mask"].copy()
            mf_fresh_mask[0] = False
        except Exception:
            mf_fresh_raw_residual = np.full(self.N + 1, np.nan)
            mf_fresh_effective_residual = np.full(self.N + 1, np.nan)
            mf_fresh_mask = np.zeros(self.N + 1, dtype=bool)

        lower_slacks = self._read_lower_slack_trajectory(solver)
        slack_source = "solver"
        if mode == "safe_stop":
            lower_slacks[:, 1] = np.nan
            slack_source = "unavailable_safe_stop"
        row_norms = np.column_stack(
            (
                np.linalg.norm(constraint_data["guard_A"], axis=1),
                np.linalg.norm(constraint_data["mf_A"], axis=1),
            )
        )
        required_slacks = np.maximum(
            -np.column_stack((guard_affine_residual, mf_affine_residual)),
            0.0,
        )
        qp_row_residual = np.column_stack(
            (guard_affine_residual, mf_affine_residual)
        ) + lower_slacks

        for values in (
            guard_affine_residual,
            guard_fresh_residual,
            mf_affine_residual,
            mf_fresh_raw_residual,
            mf_fresh_effective_residual,
        ):
            values[0] = np.nan
        lower_slacks[0] = np.nan
        required_slacks[0] = np.nan
        qp_row_residual[0] = np.nan
        row_norms[0] = np.nan

        interval_poses, interval_phi = self._sample_first_interval_psdf(
            solver,
            states,
        )
        solver_diagnostics = self._collect_solver_diagnostics(
            solver,
            status,
            mode,
        )
        solver_diagnostics["qp_fixed_row_lower_residual"] = qp_row_residual
        future_qp_residual = qp_row_residual[1:]
        solver_diagnostics["qp_fixed_row_violation_inf"] = (
            float(np.nanmax(np.maximum(-future_qp_residual, 0.0)))
            if np.isfinite(future_qp_residual).any()
            else None
        )
        future_phi = phi_exact[1:] if self.N else phi_exact
        diagnostics = {
            "row_g_enabled": True,
            "row_mf_enabled": bool(constraint_data["row_mf_enabled"]),
            "d_mf_mask": d_mf_mask,
            "constraint_row_order": {0: "guard/recovery", 1: "MF"},
            "constraint_stages": np.arange(1, self.N + 1, dtype=int),
            "exact_current_psdf": float(constraint_data["exact_current_phi"]),
            "exact_current_guard_residual": float(
                constraint_data["exact_current_phi"] - d_col
            ),
            "guard_affine_residual": guard_affine_residual,
            "guard_fresh_exact_psdf_residual": guard_fresh_residual,
            "mf_affine_residual": mf_affine_residual,
            "mf_fresh_recomputed_residual": mf_fresh_effective_residual,
            "mf_fresh_raw_residual": mf_fresh_raw_residual,
            "mf_fresh_effective_residual": mf_fresh_effective_residual,
            "guard_lower_slack": lower_slacks[:, 0],
            "mf_lower_slack": lower_slacks[:, 1],
            "lower_slack_source": slack_source,
            "required_lower_slack": required_slacks,
            "mf_valid": constraint_data["mf_valid"].copy(),
            # This is the nominal/linearization PSDF value that actually
            # decides whether Row M is installed in the OCP at each stage.
            "mf_activation_phi": constraint_data["phi"].copy(),
            "mf_domain_eligible": constraint_data["phi"] > d_mf_mask,
            "mf_mask": constraint_data["mf_mask"].copy(),
            "mf_fresh_mask": mf_fresh_mask,
            "exact_psdf_predicted_nodes": phi_exact,
            "minimum_exact_psdf_predicted_nodes": float(np.min(future_phi)),
            "first_interval_substep_poses": interval_poses,
            "first_interval_substep_exact_psdf": interval_phi,
            "minimum_exact_psdf_first_interval_substeps": float(
                np.min(interval_phi)
            ),
            "row_coefficient_norms": row_norms,
            "bounds": self._build_bound_diagnostics(solver, states),
            "solver": solver_diagnostics,
        }
        diagnostics["infeasibility"] = self._classify_infeasibility_candidates(
            diagnostics
        )
        return diagnostics

    def _print_constraint_diagnostics(self, diagnostics):
        future = slice(1, None)
        solver = diagnostics["solver"]
        print("RMPCC-MF constraint order: 0=guard/recovery, 1=MF")
        print("RMPCC-MF Row G: enabled (always hard)")
        row_m_status = (
            "enabled"
            if diagnostics["row_mf_enabled"]
            else "disabled (trivially masked)"
        )
        print(f"RMPCC-MF Row M: {row_m_status}")
        print(
            "RMPCC-MF Row M domain rule: phi > d_mf_mask "
            f"({diagnostics['d_mf_mask']:.8g})"
        )
        print(f"RMPCC-MF exact current PSDF: {diagnostics['exact_current_psdf']:.8g}")
        self._print_pair(
            "guard residuals [affine | fresh exact]",
            diagnostics["guard_affine_residual"][future],
            diagnostics["guard_fresh_exact_psdf_residual"][future],
        )
        self._print_pair(
            "MF residuals [affine | fresh recomputed]",
            diagnostics["mf_affine_residual"][future],
            diagnostics["mf_fresh_recomputed_residual"][future],
        )
        print(
            "RMPCC-MF Row G is hard (no slack); MF lower slack: "
            f"{self._diagnostic_array_string(diagnostics['mf_lower_slack'][future])}"
        )
        if diagnostics["lower_slack_source"] != "solver":
            print(
                "RMPCC-MF required row relaxations [hard guard violation, MF slack] "
                f"({diagnostics['lower_slack_source']}): "
                f"{self._diagnostic_array_string(diagnostics['required_lower_slack'][future])}"
            )
        eligibility = (
            diagnostics["mf_valid"],
            diagnostics["mf_domain_eligible"],
            diagnostics["mf_mask"],
            diagnostics["mf_fresh_mask"],
        )
        print(
            "RMPCC-MF per-stage MF eligibility "
            "[valid | nominal phi-domain | OCP active | fresh active]: "
            + " | ".join(
                self._diagnostic_array_string(values[future].astype(int))
                for values in eligibility
            )
        )
        print(
            "RMPCC-MF exact PSDF minima [predicted nodes | first interval substeps]: "
            f"{diagnostics['minimum_exact_psdf_predicted_nodes']:.8g} | "
            f"{diagnostics['minimum_exact_psdf_first_interval_substeps']:.8g}"
        )
        print(
            "RMPCC-MF raw row coefficient norms [guard, MF]: "
            f"{self._diagnostic_array_string(diagnostics['row_coefficient_norms'][future])}"
        )
        print(
            "RMPCC-MF QP lower residuals [hard guard | soft MF after slack]: "
            f"{self._diagnostic_array_string(solver['qp_fixed_row_lower_residual'][future])}"
        )
        print(
            "RMPCC-MF solver diagnostics [NLP(stat,eq,ineq,comp) | status | iter]: "
            f"{self._diagnostic_array_string(solver['nlp_residuals'])} | "
            f"{self._diagnostic_array_string(solver['qp_status'])} | "
            f"{self._diagnostic_array_string(solver['qp_iterations'])}"
        )

    def _print_pair(self, label, left, right):
        print(
            f"RMPCC-MF {label}: "
            f"{self._diagnostic_array_string(left)} | "
            f"{self._diagnostic_array_string(right)}"
        )

    def _print_infeasibility_diagnostics(self, diagnostics):
        summary = diagnostics["infeasibility"]
        print("RMPCC-MF failed-iterate infeasibility candidates:")
        if summary["candidates"]:
            for candidate in summary["candidates"]:
                print(f"  - {candidate}")
        else:
            print(
                "  - No direct violated constraint was found; inspect QP "
                "linearization or numerical conditioning."
            )
        inactive_bad = summary["inactive_mf_violation_stages"]
        if inactive_bad.size:
            print(
                "RMPCC-MF ERROR: inactive Row M is not trivially feasible at "
                f"stages {inactive_bad.tolist()}"
            )
        else:
            print("RMPCC-MF inactive Row M stages are trivially feasible.")

        bounds = diagnostics["bounds"]
        minimums = []
        for prefix in ("state", "input"):
            residual = np.minimum(
                bounds[f"{prefix}_lower_residual"],
                bounds[f"{prefix}_upper_residual"],
            )
            minimums.append(
                float(np.nanmin(residual)) if np.isfinite(residual).any() else None
            )
        print(
            "RMPCC-MF minimum box residuals [state | input]: "
            f"{minimums[0]} | {minimums[1]}"
        )
