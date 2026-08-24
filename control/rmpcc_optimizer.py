import copy
import json
import os
import shutil
import time

import casadi as ca
import numpy as np
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from utils.acados_diagnostics import report_solver_failure
from utils.rmpcc_diagnostics import RMPCCDiagnosticsMixin
from utils.spline_path import (
    clamp_progress,
    normalize_path_data,
    progress_bounds,
    project_progress,
    stage_parameter,
)
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.geometry_utils import polygon_to_edges


class RMPCCOptimizerParam:
    def __init__(self):
        # Horizon and MPCC cost
        self.horizon = 20
        self.tf = 0.1 * self.horizon
        self.mat_Qe = np.diag([1.0, 10.0])
        self.mat_Re = np.diag([1.0, 0.01])
        self.q_s = 1.0
        self.terminal_weight = 5.0
        self.v_s_target = 0.8
        self.q_s_ref = 2.0

        # Path and progress
        self.v_s_max = 2.0
        self.enforce_v_s_not_faster_than_vmax = True
        self.path_gate_epsilon = 0.05
        self.tangent_reg_delta = 1e-6
        self.reference_weight_reg = 1e-8
        self.fd_eps = 1e-3
        self.max_segments = 20
        self.line_search_window = 0.5
        self.line_search_samples = 41
        self.s_upper_guard = 100.0
        self.use_s_state_bounds = True
        self.s_backtrack_tolerance = 0.3

        # Physical inputs
        self.vmin, self.vmax = -0.7, 0.7
        self.omegamin, self.omegamax = -1.2, 1.2

        # Safety rows: Row G is hard; only Row M has a slack.
        self.d_min = 0.001
        self.d_col = 0.001
        self.d_mf_mask = self.d_col
        self.use_row_mf = True
        self.use_obstacle_constraint = True
        self.chance_epsilon = 0.20
        self.mf_slack_linear = 1e2
        self.mf_slack_quadratic = 1e0
        self.mf_active_start_step = 1
        self.mf_active_end_step = None
        self.mf_active_terminal = True
        self.augmented_psdf_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Risk and covariance
        self.sigma_f0 = 0.0002
        self.sigma_l0 = 0.00025
        self.sigma_psi0 = 0.00025
        self.q_f0 = 0
        self.q_l0 = 0
        self.q_psi0 = 0
        self.covariance_growth_scale = 0.6
        self.alpha_f = 0.002 * self.covariance_growth_scale
        self.alpha_v = 0.0004 * self.covariance_growth_scale
        self.alpha_kappa = 0.01 * self.covariance_growth_scale
        self.beta_v = 0.02 * self.covariance_growth_scale
        self.beta_kappa = 0.008 * self.covariance_growth_scale
        self.beta_omega = 0.008 * self.covariance_growth_scale
        self.risk_cov_jitter = 1e-9

        # Animation diagnostics
        self.use_risk_visualization = True

        # Solver and recovery
        self.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.hessian_approx = "GAUSS_NEWTON"
        self.integrator_type = "ERK"
        self.nlp_solver_type = "SQP_RTI"
        self.qp_solver_iter_max = 50
        self.nlp_solver_max_iter = 50
        self.tol = 1e-4
        self.enable_backup_solver = False

        # Optional diagnostics
        self.debug_mf = False
        self.debug_infeasibility = False
        self.constraint_debug_tolerance = 1e-6
        self.first_interval_substeps = 10


class RMPCCOptimizer(RMPCCDiagnosticsMixin):
    CYCLE_LOG_FIELDS = (
        "time",
        "solve_mode",
        "solver_status",
        "u0",
        "s",
        "psdf",
        "mf_probability_sum",
        "mf_slack",
    )
    COVARIANCE_LOG_FIELDS = (
        "time",
        "stage",
        "v",
        "omega",
        "sigma_f",
        "sigma_l",
        "sigma_psi",
        "P_lpsi",
    )
    CONSTRAINT_LOG_FIELDS = (
        "solve_step",
        "horizon_stage",
        "phi",
        "d_mf_mask",
        "phi_gt_d_mf_mask",
        "mf_valid",
        "mf_active",
        "row_g_residual",
        "row_g_feasible",
        "row_m_residual_before_slack",
        "row_m_lower_slack",
        "row_m_residual_after_slack",
        "row_m_feasible_before_slack",
        "row_m_feasible_after_slack",
    )

    def __init__(self):
        self.ocp = None
        self.solver = None
        self.solver_times = []
        self.state = None
        self.reference_path_data = None
        self.N = None
        self.nx = None
        self.nu = None
        self.variables = {"x": "x", "u": "u"}

        self.psdf_wrapper = None
        self._is_initialized = False
        self.device = "cuda"
        self._system_dynamics = None
        self._system = None

        self.path_param_dim = None
        self.guard_param_dim = 4
        self.mf_param_dim = 9
        self.path_param_slice = slice(0, 0)
        self.guard_param_slice = slice(0, 0)
        self.mf_param_slice = slice(0, 0)
        self._prev_predicted_s = 0.0
        self._current_s0 = 0.0
        self._has_prev_predicted_s = False
        self._constraint_data = None
        self._last_constraint_diagnostics = None
        self._last_failed_constraint_diagnostics = None
        self._constraint_log_rows = []
        self._cycle_log_rows = []
        self._covariance_log_rows = []
        self._last_nominal_probability_sum = None
        self._last_boole_risk_visualization_data = None
        self._last_main_status = None
        self._solve_count = 0
        self.prints_compact_runtime_summary = True

        self.json_filename = "acados_ocp_rmpcc_mf.json"
        self.backup_solver = None
        self.code_export_directory = "c_generated_code_rmpcc_mf"
        self.backup_json_filename = "acados_ocp_rmpcc_mf_backup.json"
        self.backup_code_export_directory = "c_generated_code_rmpcc_mf_backup"
        self._last_solve_mode = "uninitialized"
        self._last_backup_status = None
        self._temp_files = []
        self._reset_runtime_counters()

    def _reset_runtime_counters(self):
        self._backup_feasible_qp_success_count = 0
        self._main_solver_failure_count = 0
        self._backup_solver_failure_count = 0
        self._safe_stop_count = 0
        self._plant_input_apply_count = 0

    def _has_runtime_activity(self):
        return (
            self._backup_feasible_qp_success_count > 0
            or self._safe_stop_count > 0
            or self._plant_input_apply_count > 0
        )

    def _print_runtime_summary(self):
        if not self._has_runtime_activity():
            return

        print(
            "RMPCC-MF execution summary: "
            f"SQP-WITH FEASIBLE QP={self._backup_feasible_qp_success_count}, "
            f"Safe stop={self._safe_stop_count}, "
            f"Plant-applied control inputs={self._plant_input_apply_count}"
        )

    def cleanup(self):
        self._print_runtime_summary()
        self._reset_runtime_counters()
        try:
            files = {
                *self._temp_files,
                self.json_filename,
                self.backup_json_filename,
            }
            for file_path in files:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            for directory in (
                self.code_export_directory,
                self.backup_code_export_directory,
            ):
                if os.path.isdir(directory):
                    shutil.rmtree(directory, ignore_errors=True)
        except OSError as error:
            print(f"Warning: Error during cleanup: {error}")

        self.ocp = None
        self.solver = None
        self.backup_solver = None
        self.solver_times = []
        self.psdf_wrapper = None
        self._is_initialized = False

        self.reference_path_data = None
        self._system_dynamics = None
        self._system = None
        self._temp_files = []
        self.path_param_dim = None
        self.guard_param_dim = 4
        self.mf_param_dim = 9
        self.path_param_slice = slice(0, 0)
        self.guard_param_slice = slice(0, 0)
        self.mf_param_slice = slice(0, 0)
        self._prev_predicted_s = 0.0
        self._current_s0 = 0.0
        self._has_prev_predicted_s = False
        self._constraint_data = None
        self._last_constraint_diagnostics = None
        self._last_failed_constraint_diagnostics = None
        self._constraint_log_rows = []
        self._cycle_log_rows = []
        self._covariance_log_rows = []
        self._last_nominal_probability_sum = None
        self._last_boole_risk_visualization_data = None
        self._last_main_status = None
        self._solve_count = 0
        self._last_solve_mode = "uninitialized"
        self._last_backup_status = None

    def __del__(self):
        self.cleanup()

    def reset(self):
        self.cleanup()

    def set_state(self, state):
        self.state = state

    def get_progress_info(self):
        """Return MPCC progress-state values useful for runtime logging."""
        return {
            "path_s": float(self._current_s0),
            "predicted_s": float(self._prev_predicted_s),
            "has_predicted_s": bool(self._has_prev_predicted_s),
        }

    def get_runtime_stats(self):
        return {
            "backup_feasible_qp_success_count": int(self._backup_feasible_qp_success_count),
            "main_solver_failure_count": int(self._main_solver_failure_count),
            "backup_solver_failure_count": int(self._backup_solver_failure_count),
            "safe_stop_count": int(self._safe_stop_count),
            "plant_input_apply_count": int(self._plant_input_apply_count),
            "last_solve_mode": str(self._last_solve_mode),
        }

    def get_last_constraint_diagnostics(self):
        """Return the latest fixed-row safety diagnostics.

        The cache is populated only when ``debug_mf`` is enabled for the most
        recent solve. Constraint ordering is invariant: row 0 is the signed-
        PSDF guard/recovery row and row 1 is the MF row.
        """
        if self._last_constraint_diagnostics is None:
            return None
        return copy.deepcopy(self._last_constraint_diagnostics)

    def get_last_failed_constraint_diagnostics(self):
        """Return the pre-recovery snapshot when failure debugging is enabled."""
        if self._last_failed_constraint_diagnostics is None:
            return None
        return copy.deepcopy(self._last_failed_constraint_diagnostics)

    def get_constraint_log_rows(self):
        """Return the compact per-stage inequality history for CSV export."""
        return [row.copy() for row in self._constraint_log_rows]

    def get_constraint_log_fields(self):
        return list(self.CONSTRAINT_LOG_FIELDS)

    def record_plant_input_applied(self):
        self._plant_input_apply_count += 1

    def get_cycle_log_rows(self):
        return [row.copy() for row in self._cycle_log_rows]

    def get_cycle_log_fields(self):
        return list(self.CYCLE_LOG_FIELDS)

    def get_last_boole_risk_sum_trajectory(self):
        """Return the latest nominal stage-wise Boole risk sum.

        The trajectory is evaluated at the nominal linearization points used
        for the most recent solver attempt and has ``N + 1`` entries.  ``None``
        is returned before the first attempt or after the optimizer is reset.
        A copy is returned so diagnostics cannot mutate the cached values.
        """
        if self._last_nominal_probability_sum is None:
            return None
        return np.asarray(
            self._last_nominal_probability_sum,
            dtype=float,
        ).copy()

    def get_last_boole_risk_visualization_data(self):
        """Return source-aligned nominal data for Boole-risk visualization.

        Every array comes from the same nominal linearization pass.  The
        returned dictionary contains ``nominal_poses`` with shape
        ``(N + 1, 3)`` and the one-dimensional ``boole_risk_sum``, ``epsilon``,
        and ``mf_mask`` arrays.  Copies prevent animation code from mutating
        the optimizer's latest atomic cache.  ``None`` is returned for a
        safe-stop cycle because that fallback plan was not the trajectory on
        which the cached MF sum was evaluated.
        """
        if self._last_boole_risk_visualization_data is None:
            return None
        return {
            key: np.asarray(value).copy()
            for key, value in self._last_boole_risk_visualization_data.items()
        }

    def get_covariance_log_rows(self):
        return [row.copy() for row in self._covariance_log_rows]

    def get_covariance_log_fields(self):
        return list(self.COVARIANCE_LOG_FIELDS)

    def _get_effective_v_s_max(self, param):
        v_s_max = float(param.v_s_max)
        if param.enforce_v_s_not_faster_than_vmax:
            return min(v_s_max, max(0.0, float(param.vmax)))
        return v_s_max

    def _get_progress_speed_penalty(self, param):
        v_s_target = max(float(param.v_s_target), 1e-6)
        q_s = max(float(param.q_s), 0.0)
        # For r_v_s * v_s^2 - q_s * v_s, the unconstrained minimizer is
        # q_s / (2 * r_v_s), so choose the quadratic penalty from v_s_target.
        return 0.5 * q_s / v_s_target

    def create_model(self, param):
        dt = float(param.tf) / float(max(param.horizon, 1))
        nx = 8  # [x, y, theta, s, P_f, P_l, P_psi, P_lpsi]
        nu = 3  # [v, omega, v_s]
        # Stage-varying path parameter:
        # [p_ref_x, p_ref_y, t_ref_x, t_ref_y, s_ref, kappa_ref]
        self.path_param_dim = 6

        x = ca.MX.sym("x", nx)
        xdot = ca.MX.sym("xdot", nx)
        u = ca.MX.sym("u", nu)
        p_ref_t_ref = ca.MX.sym("p_ref_t_ref", self.path_param_dim)
        p_guard = ca.MX.sym("guard_affine", self.guard_param_dim)
        p_mf = ca.MX.sym("mf_affine", self.mf_param_dim)

        self.path_param_slice = slice(0, self.path_param_dim)
        self.guard_param_slice = slice(
            self.path_param_dim,
            self.path_param_dim + self.guard_param_dim,
        )
        self.mf_param_slice = slice(
            self.path_param_dim + self.guard_param_dim,
            self.path_param_dim + self.guard_param_dim + self.mf_param_dim,
        )
        p_all = ca.vertcat(p_ref_t_ref, p_guard, p_mf)

        v = u[0]
        omega = u[1]
        v_s = u[2]

        p_ref = p_ref_t_ref[0:2]
        t_raw = p_ref_t_ref[2:4]
        s_ref = p_ref_t_ref[4]
        kappa_ref = p_ref_t_ref[5]
        t_vec = t_raw / (ca.norm_2(t_raw) + float(param.tangent_reg_delta))
        p_ref_lin = p_ref + (x[3] - s_ref) * t_vec

        dx = x[0] - p_ref_lin[0]
        dy = x[1] - p_ref_lin[1]
        # Match existing MPCC sign convention:
        # e_l = -dot(position_error, tangent)
        # e_c = dot(position_error, right-normal)
        e_l = -(dx * t_vec[0] + dy * t_vec[1])
        e_c = dx * t_vec[1] - dy * t_vec[0]
        e_s = x[3] - s_ref

        q_f_noise = ca.fmax(
            float(param.q_f0) + float(param.alpha_f) * v * v,
            0.0,
        )
        q_l_noise = ca.fmax(
            float(param.q_l0)
            + float(param.alpha_v) * v * v
            + float(param.alpha_kappa) * v * v * ca.fabs(kappa_ref),
            0.0,
        )
        q_psi_noise = ca.fmax(
            float(param.q_psi0)
            + float(param.beta_v) * v * v
            + float(param.beta_kappa) * v * v * ca.fabs(kappa_ref)
            + float(param.beta_omega) * omega * omega,
            0.0,
        )

        p_f = x[4]
        p_l = x[5]
        p_psi = x[6]
        p_lpsi = x[7]

        # Keep the existing continuous-time ERK setup by differentiating the
        # user-provided one-step covariance recursion: xdot = (x_{k+1} - x_k)/dt.
        f_expl = ca.vertcat(
            v * ca.cos(x[2]),
            v * ca.sin(x[2]),
            omega,
            v_s,
            dt * omega * omega * p_l + q_f_noise,
            dt * omega * omega * p_f + 2.0 * v * p_lpsi + dt * v * v * p_psi + q_l_noise,
            q_psi_noise,
            v * p_psi,
        )

        q_c = float(param.mat_Qe[0, 0])
        q_l = float(param.mat_Qe[1, 1])
        r_v = float(param.mat_Re[0, 0])
        r_w = float(param.mat_Re[1, 1])
        q_s = float(param.q_s)
        r_vs = self._get_progress_speed_penalty(param)
        q_s_ref = float(param.q_s_ref)
        qf = float(param.terminal_weight)

        cost_stage = (
            q_c * e_c * e_c
            + q_l * e_l * e_l
            + q_s_ref * e_s * e_s
            + r_v * u[0] * u[0]
            + r_w * u[1] * u[1]
            + r_vs * u[2] * u[2]
            - q_s * u[2]
        )
        cost_terminal = qf * (q_c * e_c * e_c + q_l * e_l * e_l + q_s_ref * e_s * e_s)

        model = AcadosModel()
        model.name = "differential_drive_rmpcc_mf"
        model.x = x
        model.xdot = xdot
        model.u = u
        model.p = p_all
        model.f_expl_expr = f_expl
        model.f_impl_expr = xdot - f_expl
        model.cost_expr_ext_cost = cost_stage
        model.cost_expr_ext_cost_e = cost_terminal
        return model

    def _build_stage_path_parameter(self, s_value):
        return stage_parameter(
            self.reference_path_data,
            s_value,
            self.param.tangent_reg_delta,
            self.param.fd_eps,
        )

    def _build_stage_param(self, stage_path_param, guard_affine, mf_affine):
        stage_param = np.zeros(
            self.path_param_dim + self.guard_param_dim + self.mf_param_dim,
            dtype=float,
        )
        stage_param[self.path_param_slice] = stage_path_param
        stage_param[self.guard_param_slice] = guard_affine
        stage_param[self.mf_param_slice] = mf_affine
        return stage_param

    def _project_to_psd(self, Sigma):
        Sigma_sym = 0.5 * (Sigma + Sigma.T)
        jitter = max(float(self.param.risk_cov_jitter), 0.0)
        eigvals, eigvecs = np.linalg.eigh(Sigma_sym)
        eigvals = np.clip(eigvals, jitter, None)
        Sigma_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return 0.5 * (Sigma_psd + Sigma_psd.T)

    def _build_initial_local_covariance(self, param):
        sigma_f0 = float(max(param.sigma_f0, 0.0))
        sigma_l0 = float(max(param.sigma_l0, 0.0))
        sigma_psi0 = float(max(param.sigma_psi0, 0.0))
        return np.diag([sigma_f0 * sigma_f0, sigma_l0 * sigma_l0, sigma_psi0 * sigma_psi0])

    def _build_initial_covariance_state(self, param):
        Sigma0 = self._build_initial_local_covariance(param)
        return np.array([Sigma0[0, 0], Sigma0[1, 1], Sigma0[2, 2], Sigma0[1, 2]], dtype=float)

    def _covariance_state_to_matrix(self, covariance_state):
        cov_state = np.asarray(covariance_state, dtype=float)
        squeeze = cov_state.ndim == 1
        cov_state = np.atleast_2d(cov_state)

        Sigma = np.zeros((cov_state.shape[0], 3, 3), dtype=float)
        Sigma[:, 0, 0] = cov_state[:, 0]
        Sigma[:, 1, 1] = cov_state[:, 1]
        Sigma[:, 2, 2] = cov_state[:, 2]
        Sigma[:, 1, 2] = cov_state[:, 3]
        Sigma[:, 2, 1] = cov_state[:, 3]
        return Sigma[0] if squeeze else Sigma

    def _matrix_to_covariance_state(self, Sigma):
        Sigma_np = np.asarray(Sigma, dtype=float)
        return np.array(
            [Sigma_np[0, 0], Sigma_np[1, 1], Sigma_np[2, 2], Sigma_np[1, 2]],
            dtype=float,
        )

    def _normalize_epsilon_values(self, num_stages):
        epsilon = np.asarray(
            self.param.chance_epsilon,
            dtype=float,
        )
        if epsilon.ndim == 0:
            epsilon = np.full((num_stages,), float(epsilon), dtype=float)
        else:
            epsilon = epsilon.reshape(-1)
            if epsilon.shape != (num_stages,):
                raise ValueError(
                    "chance_epsilon must be a scalar or have one value per stage"
                )
        if not np.isfinite(epsilon).all() or np.any(epsilon <= 0.0) or np.any(epsilon >= 1.0):
            raise ValueError("chance_epsilon values must lie strictly between 0 and 1")
        return epsilon

    def _compute_exact_psdf(self, poses):
        """Evaluate signed PSDF directly at world-frame poses."""
        poses_np = np.asarray(poses, dtype=float)
        squeeze = poses_np.ndim == 1
        poses_np = np.atleast_2d(poses_np)
        if poses_np.ndim != 2 or poses_np.shape[1] != 3:
            raise ValueError("poses must have shape (H, 3) or (3,)")
        if not np.isfinite(poses_np).all():
            raise ValueError("pose values must be finite")

        if self.psdf_wrapper is None:
            phi = np.full((poses_np.shape[0],), 1000.0, dtype=float)
            gradient = np.zeros((poses_np.shape[0], 3), dtype=float)
        else:
            poses_t = torch.as_tensor(
                poses_np,
                dtype=self.psdf_wrapper.A.dtype,
                device=self.psdf_wrapper.device,
            )
            phi_t, gradient_t = self.psdf_wrapper(poses_t)
            phi = phi_t.detach().cpu().numpy()
            gradient = gradient_t.detach().cpu().numpy()

        if not np.isfinite(phi).all() or not np.isfinite(gradient).all():
            raise ValueError("augmented PSDF returned non-finite signed-distance data")
        if squeeze:
            return float(phi[0]), gradient[0].copy()
        return phi, gradient

    def _compute_mf_affine_data(self, z_bar):
        """Return raw PSDF/MF data without applying the separated-domain mask.

        The normal path is one batched ``forward_mf`` call.  If a covariance
        issue makes that call fail, stagewise calls preserve every valid MF
        row while the pose-only PSDF path still supplies the always-active Row G.
        """
        z_bar_np = np.asarray(z_bar, dtype=float)
        if z_bar_np.ndim != 2 or z_bar_np.shape[1] != 8:
            raise ValueError("z_bar must have shape (H, 8)")
        if not np.isfinite(z_bar_np).all():
            raise ValueError("z_bar values must be finite")

        num_stages = z_bar_np.shape[0]
        epsilon = self._normalize_epsilon_values(num_stages)

        if self.psdf_wrapper is None or not self.param.use_obstacle_constraint:
            phi = np.full((num_stages,), 1000.0, dtype=float)
            gradient = np.zeros((num_stages, 3), dtype=float)
            A_mf = np.zeros((num_stages, 8), dtype=float)
            c_mf = epsilon.copy()
            stage_available = np.zeros(num_stages, dtype=bool)
            return phi, gradient, A_mf, c_mf, stage_available

        z_bar_t = torch.as_tensor(
            z_bar_np,
            dtype=self.psdf_wrapper.A.dtype,
            device=self.psdf_wrapper.device,
        )
        try:
            outputs_t = self.psdf_wrapper.forward_mf(
                z_bar_t,
                torch.as_tensor(
                    epsilon,
                    dtype=z_bar_t.dtype,
                    device=z_bar_t.device,
                ),
                float(self.param.d_min),
            )
            phi, gradient, A_mf, c_mf = tuple(
                output.detach().cpu().numpy() for output in outputs_t
            )
            stage_available = np.ones((num_stages,), dtype=bool)
        except (RuntimeError, ValueError, FloatingPointError):
            # Guard geometry is covariance-independent, so retain it even when
            # one MF stage has invalid variance data.
            phi, gradient = self._compute_exact_psdf(z_bar_np[:, :3])
            A_mf = np.full((num_stages, 8), np.nan, dtype=float)
            c_mf = np.full((num_stages,), np.nan, dtype=float)
            stage_available = np.zeros((num_stages,), dtype=bool)

            for i in range(num_stages):
                try:
                    outputs_i = self.psdf_wrapper.forward_mf(
                        z_bar_t[i : i + 1],
                        float(epsilon[i]),
                        float(self.param.d_min),
                    )
                    phi_i, gradient_i, A_i, c_i = tuple(
                        output.detach().cpu().numpy() for output in outputs_i
                    )
                    phi[i] = phi_i[0]
                    gradient[i] = gradient_i[0]
                    A_mf[i] = A_i[0]
                    c_mf[i] = c_i[0]
                    stage_available[i] = True
                except (RuntimeError, ValueError, FloatingPointError):
                    continue

        # A failed/non-finite MF stage is maskable.  Repair guard/mask geometry
        # through the exact pose interface, then fail loudly if even the signed
        # PSDF itself is invalid.  Row G is repaired independently from the MF
        # validity mask and remains active at every constrained stage.
        guard_finite = np.isfinite(phi) & np.isfinite(gradient).all(axis=1)
        # A non-finite PSDF/gradient returned by forward_mf also invalidates
        # that stage's MF data, even though Row G can be repaired independently.
        stage_available &= guard_finite
        if not guard_finite.all():
            phi_exact, gradient_exact = self._compute_exact_psdf(z_bar_np[:, :3])
            phi[~guard_finite] = phi_exact[~guard_finite]
            gradient[~guard_finite] = gradient_exact[~guard_finite]
        if not np.isfinite(phi).all() or not np.isfinite(gradient).all():
            raise ValueError("signed PSDF guard data must be finite at every stage")

        active_clusters = getattr(self.psdf_wrapper, "active_clusters", None)
        has_active_features = (
            True if active_clusters is None else bool(active_clusters > 0)
        )
        return (
            phi,
            gradient,
            A_mf,
            c_mf,
            stage_available & has_active_features,
        )

    def _build_mf_stage_mask(self, num_stages):
        """Return the configured intermediate/terminal MF stage window."""
        if num_stages <= 0:
            raise ValueError("num_stages must be positive")

        terminal_stage = num_stages - 1
        stage_mask = np.zeros(num_stages, dtype=bool)

        start_step = int(self.param.mf_active_start_step)
        if start_step < 1:
            raise ValueError("mf_active_start_step must be at least 1")
        configured_end = self.param.mf_active_end_step
        if configured_end is None:
            end_step = terminal_stage - 1
        else:
            end_step = int(configured_end)

        intermediate_last = terminal_stage - 1
        if intermediate_last >= start_step and end_step >= start_step:
            stage_indices = np.arange(num_stages)
            intermediate = (
                (stage_indices >= start_step)
                & (stage_indices <= min(end_step, intermediate_last))
            )
            stage_mask[intermediate] = True

        stage_mask[terminal_stage] = bool(self.param.mf_active_terminal)
        return stage_mask

    def _compute_constraint_affine_data(self, z_bar):
        """Build fixed Row G/Row M coefficients and the per-stage MF mask."""
        z_bar_np = np.asarray(z_bar, dtype=float)
        if z_bar_np.ndim != 2 or z_bar_np.shape[1] != 8:
            raise ValueError("z_bar must have shape (H, 8)")
        if not np.isfinite(z_bar_np).all():
            raise ValueError("z_bar values must be finite")

        num_stages = z_bar_np.shape[0]
        epsilon = self._normalize_epsilon_values(num_stages)
        row_mf_enabled = bool(self.param.use_row_mf)
        obstacle_constraint_enabled = bool(self.param.use_obstacle_constraint)

        if not obstacle_constraint_enabled:
            # Preserve the master obstacle-constraint switch even if a wrapper
            # from an earlier setup is still attached to the optimizer.
            phi = np.full((num_stages,), 1000.0, dtype=float)
            gradient = np.zeros((num_stages, 3), dtype=float)
            A_mf_raw = np.full((num_stages, 8), np.nan, dtype=float)
            c_mf_raw = np.full((num_stages,), np.nan, dtype=float)
            stage_available = np.zeros(num_stages, dtype=bool)
        elif row_mf_enabled:
            mf_data = self._compute_mf_affine_data(z_bar_np)
            phi, gradient, A_mf_raw, c_mf_raw = mf_data[:4]
            stage_available = mf_data[4] if len(mf_data) > 4 else None
        else:
            # A disabled Row M must not evaluate the covariance-dependent MF
            # model.  Retain the pose-only signed PSDF evaluation required to
            # build hard Row G, and mark raw MF data as intentionally absent.
            phi, gradient = self._compute_exact_psdf(z_bar_np[:, :3])
            phi = np.asarray(phi, dtype=float)
            gradient = np.asarray(gradient, dtype=float)
            A_mf_raw = np.full((num_stages, 8), np.nan, dtype=float)
            c_mf_raw = np.full((num_stages,), np.nan, dtype=float)
            stage_available = np.zeros(num_stages, dtype=bool)

        d_col = float(self.param.d_col)
        d_mf_mask = float(self.param.d_mf_mask)
        if not np.isfinite(d_col) or not np.isfinite(d_mf_mask):
            raise ValueError("d_col and d_mf_mask must be finite")

        guard_A = np.zeros((num_stages, 8), dtype=float)
        guard_A[:, :3] = gradient
        guard_c = phi - np.einsum("ij,ij->i", gradient, z_bar_np[:, :3]) - d_col

        finite_mf = np.isfinite(A_mf_raw).all(axis=1) & np.isfinite(c_mf_raw)
        if stage_available is None:
            stage_available = finite_mf.copy()
        else:
            stage_available = np.asarray(stage_available, dtype=bool)
            if stage_available.shape != (num_stages,):
                raise ValueError("MF availability must contain one value per stage")
        mf_valid = row_mf_enabled & stage_available & finite_mf
        mf_stage_mask = self._build_mf_stage_mask(num_stages)
        mf_mask = (phi > d_mf_mask) & mf_valid & mf_stage_mask

        A_mf = np.zeros((num_stages, 8), dtype=float)
        c_mf = epsilon.copy()
        A_mf[mf_mask] = A_mf_raw[mf_mask]
        c_mf[mf_mask] = c_mf_raw[mf_mask]

        return {
            "phi": phi,
            "gradient": gradient,
            "row_mf_enabled": row_mf_enabled,
            "guard_A": guard_A,
            "guard_c": guard_c,
            "mf_A_raw": A_mf_raw,
            "mf_c_raw": c_mf_raw,
            "mf_valid": mf_valid,
            "mf_stage_mask": mf_stage_mask,
            "mf_A": A_mf,
            "mf_c": c_mf,
            "mf_mask": mf_mask,
            "epsilon": epsilon,
        }

    def _clamp_s(self, s_value):
        return clamp_progress(self.reference_path_data, s_value)

    def _project_s_with_line_search(self, position_xy, param):
        previous_s = self._prev_predicted_s if self._has_prev_predicted_s else None
        return project_progress(
            self.reference_path_data,
            np.asarray(position_xy, dtype=float),
            previous_s,
            param.line_search_window,
            param.line_search_samples,
        )

    def _get_current_s_bounds(self, param):
        return progress_bounds(self.reference_path_data, param.s_upper_guard)

    def _get_augmented_state_bounds(self, param, s_lower, s_upper):
        idxbx = []
        lbx = []
        ubx = []

        if param.use_s_state_bounds:
            idxbx.append(3)
            lbx.append(float(s_lower))
            ubx.append(float(s_upper))

        cov_floor = max(float(param.risk_cov_jitter), 0.0)
        idxbx.extend([4, 5, 6])
        lbx.extend([cov_floor, cov_floor, cov_floor])
        ubx.extend([1e8, 1e8, 1e8])

        return (
            np.array(idxbx, dtype=np.int64),
            np.array(lbx, dtype=float),
            np.array(ubx, dtype=float),
        )

    def update_reference_path_params(self, reference_path_data, state):
        if reference_path_data is not None:
            self.reference_path_data = normalize_path_data(reference_path_data)
        if self.reference_path_data is None:
            return None

        position_xy = np.asarray(state._x[:2], dtype=float)
        self._current_s0 = self._project_s_with_line_search(position_xy, self.param)
        self._current_s0 = self._clamp_s(self._current_s0)
        if self._has_prev_predicted_s:
            backtrack_tol = max(float(self.param.s_backtrack_tolerance), 0.0)
            s_floor = self._clamp_s(self._prev_predicted_s - backtrack_tol)
            self._current_s0 = max(self._current_s0, s_floor)
        return self._current_s0

    def set_reference_trajectory(self, reference_path_data):
        if reference_path_data is None:
            return
        self.reference_path_data = normalize_path_data(reference_path_data)

    def setup_ocp(self, param):
        self.ocp = AcadosOcp()
        model = self.create_model(param)
        self.ocp.model = model

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        N = param.horizon
        self.nx = nx
        self.nu = nu
        self.N = N
        self.ocp.dims.N = N
        self.ocp.dims.np = (
            self.path_param_dim + self.guard_param_dim + self.mf_param_dim
        )
        self.ocp.parameter_values = np.zeros((self.ocp.dims.np,), dtype=float)

        self.ocp.cost.cost_type = "EXTERNAL"
        self.ocp.cost.cost_type_e = "EXTERNAL"

        effective_v_s_max = self._get_effective_v_s_max(param)
        self.ocp.constraints.lbu = np.array([param.vmin, param.omegamin, 0.0], dtype=float)
        self.ocp.constraints.ubu = np.array(
            [param.vmax, param.omegamax, effective_v_s_max],
            dtype=float,
        )
        self.ocp.constraints.idxbu = np.array([0, 1, 2], dtype=np.int64)

        x0_phys = (
            np.zeros(3, dtype=float)
            if self.state is None
            else np.asarray(self.state._x, dtype=float)
        )
        self._current_s0 = self._project_s_with_line_search(x0_phys[:2], param)
        s_lower, s_upper = self._get_current_s_bounds(param)
        self._current_s0 = float(np.clip(self._current_s0, s_lower, s_upper))
        x0_cov = self._build_initial_covariance_state(param)
        x0_full = np.array(
            [
                x0_phys[0],
                x0_phys[1],
                x0_phys[2],
                self._current_s0,
                x0_cov[0],
                x0_cov[1],
                x0_cov[2],
                x0_cov[3],
            ],
            dtype=float,
        )

        # Stage-0 equality on full augmented state.
        self.ocp.constraints.idxbx_0 = np.arange(nx, dtype=np.int64)
        self.ocp.constraints.lbx_0 = x0_full.copy()
        self.ocp.constraints.ubx_0 = x0_full.copy()

        idxbx, lbx, ubx = self._get_augmented_state_bounds(param, s_lower, s_upper)
        self.ocp.constraints.idxbx = idxbx
        self.ocp.constraints.lbx = lbx
        self.ocp.constraints.ubx = ubx
        self.ocp.constraints.idxbx_e = idxbx.copy()
        self.ocp.constraints.lbx_e = lbx.copy()
        self.ocp.constraints.ubx_e = ubx.copy()

        self.ocp.solver_options.qp_solver = param.qp_solver
        self.ocp.solver_options.hessian_approx = param.hessian_approx
        self.ocp.solver_options.integrator_type = param.integrator_type
        self.ocp.solver_options.nlp_solver_type = param.nlp_solver_type
        self.ocp.solver_options.qp_solver_iter_max = param.qp_solver_iter_max
        self.ocp.solver_options.nlp_solver_max_iter = param.nlp_solver_max_iter
        self.ocp.solver_options.tol = param.tol

        self.ocp.solver_options.qp_solver_warm_start = True
        self.ocp.solver_options.nlp_solver_warm_start_first_qp = True
        self.ocp.solver_options.tf = param.tf

        hpipm_mode = getattr(param, "hpipm_mode", None)
        if hpipm_mode is not None:
            self.ocp.solver_options.hpipm_mode = hpipm_mode

        regularize_method = getattr(param, "regularize_method", None)
        if regularize_method is not None:
            self.ocp.solver_options.regularize_method = regularize_method

    def create_solver(self, json_filename=None, code_export_directory=None):
        if code_export_directory is not None:
            self.ocp.code_export_directory = code_export_directory

        used_json_filename = self.json_filename if json_filename is None else json_filename
        solver = AcadosOcpSolver(self.ocp, json_file=used_json_filename)
        if used_json_filename not in self._temp_files:
            self._temp_files.append(used_json_filename)
        return solver

    def add_obstacle_avoidance_constraint(self, param, obstacles_geo):
        if param.use_obstacle_constraint:
            self.update_obstacles(obstacles_geo)

        if self.ocp is None:
            print("Warning: OCP is not initialized; skipping fixed safety rows")
            return

        x = self.ocp.model.x
        guard_affine = self.ocp.model.p[self.guard_param_slice]
        mf_affine = self.ocp.model.p[self.mf_param_slice]
        guard_expr = ca.dot(guard_affine[:3], x[:3]) + guard_affine[3]
        mf_expr = ca.dot(mf_affine[:8], x) + mf_affine[8]
        constraint_expr = ca.vertcat(guard_expr, mf_expr)

        self.ocp.constraints.constr_type = "BGH"
        self.ocp.constraints.constr_type_e = "BGH"
        self.ocp.dims.nh = 2
        self.ocp.dims.nh_e = 2
        self.ocp.model.con_h_expr = constraint_expr
        self.ocp.model.con_h_expr_e = constraint_expr
        self.ocp.constraints.lh = np.zeros((2,), dtype=float)
        self.ocp.constraints.uh = np.full((2,), 1e8, dtype=float)
        self.ocp.constraints.lh_e = np.zeros((2,), dtype=float)
        self.ocp.constraints.uh_e = np.full((2,), 1e8, dtype=float)

        # Row G (index 0) is deliberately absent from idxsh and is therefore
        # hard.  Row M (index 1) remains soft so the chance constraint can be
        # relaxed without weakening the geometric collision guard.
        soft_indices = np.array([1], dtype=np.int64)
        self.ocp.dims.nsh = 1
        self.ocp.dims.ns = 1
        self.ocp.dims.nsh_e = 1
        self.ocp.dims.ns_e = 1
        self.ocp.constraints.idxsh = soft_indices
        self.ocp.constraints.idxsh_e = soft_indices.copy()
        # lsh/ush are lower bounds on lower/upper slack variables, not upper
        # limits.  Zero permits a nonnegative slack without forcing one.
        self.ocp.constraints.lsh = np.zeros((1,), dtype=float)
        self.ocp.constraints.ush = np.zeros((1,), dtype=float)
        self.ocp.constraints.lsh_e = np.zeros((1,), dtype=float)
        self.ocp.constraints.ush_e = np.zeros((1,), dtype=float)

        slack_linear = np.array(
            [float(param.mf_slack_linear)],
            dtype=float,
        )
        slack_quadratic = np.array(
            [float(param.mf_slack_quadratic)],
            dtype=float,
        )
        if (
            not np.isfinite(slack_linear).all()
            or not np.isfinite(slack_quadratic).all()
            or np.any(slack_linear <= 0.0)
            or np.any(slack_quadratic <= 0.0)
        ):
            raise ValueError("MF slack penalties must be positive and finite")

        # acados expects the diagonal entries as one-dimensional vectors.
        self.ocp.cost.zl = slack_linear.copy()
        self.ocp.cost.Zl = slack_quadratic.copy()
        self.ocp.cost.zu = slack_linear.copy()
        self.ocp.cost.Zu = slack_quadratic.copy()
        self.ocp.cost.zl_e = slack_linear.copy()
        self.ocp.cost.Zl_e = slack_quadratic.copy()
        self.ocp.cost.zu_e = slack_linear.copy()
        self.ocp.cost.Zu_e = slack_quadratic.copy()
        print(
            "Added two affine safety rows at intermediate/terminal stages: "
            "0=HARD guard/recovery, 1=SOFT MF"
        )

    def add_warm_start(self, param, system, solver=None):
        solver = self.solver if solver is None else solver
        if solver is None:
            return

        dt = float(param.tf) / float(param.horizon)
        effective_v_s_max = self._get_effective_v_s_max(param)
        try:
            x_ws, u_ws = system._dynamics.nominal_safe_controller(
                self.state._x, dt, self.state._u[0], -1.0, 1.0
            )
            v_s_ws = float(np.clip(0.1, 0.0, effective_v_s_max))
            cov0 = self._build_initial_covariance_state(param)
            for i in range(self.N):
                s_ws = self._clamp_s(self._current_s0 + i * dt * v_s_ws)
                solver.set(
                    i,
                    "x",
                    np.array(
                        [
                            x_ws[0],
                            x_ws[1],
                            x_ws[2],
                            s_ws,
                            cov0[0],
                            cov0[1],
                            cov0[2],
                            cov0[3],
                        ],
                        dtype=float,
                    ),
                )
                solver.set(i, "u", np.array([u_ws[0], u_ws[1], v_s_ws], dtype=float))
            solver.set(
                self.N,
                "x",
                np.array(
                    [
                        x_ws[0],
                        x_ws[1],
                        x_ws[2],
                        self._clamp_s(self._current_s0 + self.N * dt * v_s_ws),
                        cov0[0],
                        cov0[1],
                        cov0[2],
                        cov0[3],
                    ],
                    dtype=float,
                ),
            )
        except Exception as e:
            print(f"Error in warm start: {e}")
            for i in range(self.N):
                solver.set(i, "x", np.zeros((self.nx,), dtype=float))
                solver.set(i, "u", np.zeros((self.nu,), dtype=float))
            solver.set(self.N, "x", np.zeros((self.nx,), dtype=float))

    def _copy_primal_guess(self, src_solver, dst_solver):
        if src_solver is None or dst_solver is None:
            raise ValueError("Both source and destination solvers are required.")

        for i in range(self.N):
            xi = np.asarray(src_solver.get(i, "x"), dtype=float)
            ui = np.asarray(src_solver.get(i, "u"), dtype=float)
            if not np.all(np.isfinite(xi)):
                raise ValueError(f"Non-finite state guess at stage {i}.")
            if not np.all(np.isfinite(ui)):
                raise ValueError(f"Non-finite input guess at stage {i}.")
            dst_solver.set(i, "x", xi.copy())
            dst_solver.set(i, "u", ui.copy())

        x_terminal = np.asarray(src_solver.get(self.N, "x"), dtype=float)
        if not np.all(np.isfinite(x_terminal)):
            raise ValueError("Non-finite terminal state guess.")
        dst_solver.set(self.N, "x", x_terminal.copy())

    def _read_solver_trajectory(self, solver):
        states = np.stack(
            [np.asarray(solver.get(i, "x"), dtype=float) for i in range(self.N + 1)],
            axis=0,
        )
        inputs = np.stack(
            [np.asarray(solver.get(i, "u"), dtype=float) for i in range(self.N)],
            axis=0,
        )
        if states.shape != (self.N + 1, self.nx):
            raise ValueError(f"unexpected nominal state shape {states.shape}")
        if inputs.shape != (self.N, self.nu):
            raise ValueError(f"unexpected nominal input shape {inputs.shape}")
        if not np.isfinite(states).all() or not np.isfinite(inputs).all():
            raise ValueError("nominal state and input trajectories must be finite")
        return states, inputs

    def _predict_stage_s_values(self, s0, s_lower, s_upper):
        s_values = np.zeros((self.N + 1,), dtype=float)
        s_values[0] = float(np.clip(s0, s_lower, s_upper))
        if self.N <= 0:
            return s_values

        dt = float(self.param.tf) / float(max(self.N, 1))
        effective_v_s_max = self._get_effective_v_s_max(self.param)
        for i in range(1, self.N + 1):
            s_from_state = np.nan
            try:
                s_from_state = float(self.solver.get(i, "x")[3])
            except Exception:
                pass

            try:
                v_s_prev = float(self.solver.get(i - 1, "u")[2])
            except Exception:
                v_s_prev = 0.0
            v_s_prev = float(np.clip(v_s_prev, 0.0, effective_v_s_max))
            s_from_input = s_values[i - 1] + dt * v_s_prev

            if np.isfinite(s_from_state):
                s_next = 0.5 * s_from_state + 0.5 * s_from_input
            else:
                s_next = s_from_input

            s_next = float(np.clip(s_next, s_lower, s_upper))
            s_values[i] = max(s_values[i - 1], s_next)

        return s_values

    def _set_constraint_parameters(self, solver, stage_s_values, constraint_data):
        guard_rows = np.column_stack(
            (constraint_data["guard_A"][:, :3], constraint_data["guard_c"])
        )
        mf_rows = np.column_stack(
            (constraint_data["mf_A"], constraint_data["mf_c"])
        )

        # Stage 0 has no safety constraint.  Keep its parameter slots trivial
        # and mirror that fact in the cached diagnostics data.
        constraint_data["mf_A"][0] = 0.0
        constraint_data["mf_c"][0] = constraint_data["epsilon"][0]
        constraint_data["mf_mask"][0] = False
        guard_stage_zero = np.array([0.0, 0.0, 0.0, 1.0])
        mf_stage_zero = np.zeros(self.mf_param_dim)
        mf_stage_zero[-1] = constraint_data["epsilon"][0]
        solver.set(
            0,
            "p",
            self._build_stage_param(
                self._build_stage_path_parameter(stage_s_values[0]),
                guard_stage_zero,
                mf_stage_zero,
            ),
        )
        for stage in range(1, self.N + 1):
            solver.set(
                stage,
                "p",
                self._build_stage_param(
                    self._build_stage_path_parameter(stage_s_values[stage]),
                    guard_rows[stage],
                    mf_rows[stage],
                ),
            )

    def _prepare_and_solve(self, solver):
        if solver is None:
            raise RuntimeError("Solver is not initialized.")

        # Do not let an exception in this solve attempt expose the previous
        # cycle's risk markers as if they belonged to the current frame.
        self._last_nominal_probability_sum = None
        self._last_boole_risk_visualization_data = None

        s_lower, s_upper = self._get_current_s_bounds(self.param)
        if self.state is not None:
            self.update_reference_path_params(self.reference_path_data, self.state)
            s_lower, s_upper = self._get_current_s_bounds(self.param)
            self._current_s0 = float(np.clip(self._current_s0, s_lower, s_upper))
            x0_cov = self._build_initial_covariance_state(self.param)
            x0 = np.array(
                [
                    self.state._x[0],
                    self.state._x[1],
                    self.state._x[2],
                    self._current_s0,
                    x0_cov[0],
                    x0_cov[1],
                    x0_cov[2],
                    x0_cov[3],
                ],
                dtype=float,
            )
            try:
                solver.set(0, "lbx", x0)
                solver.set(0, "ubx", x0)
            except Exception:
                print(
                    "Warning: stage-0 full-state bounds update failed; "
                    "keeping previous equality bounds."
                )
            solver.set(0, "x", x0)

        _, lbx, ubx = self._get_augmented_state_bounds(
            self.param,
            s_lower,
            s_upper,
        )
        try:
            for i in range(1, self.N):
                solver.set(i, "lbx", lbx)
                solver.set(i, "ubx", ubx)
            solver.set(self.N, "lbx", lbx)
            solver.set(self.N, "ubx", ubx)
        except Exception:
            print(
                "Warning: failed to update per-stage augmented-state bounds "
                "in solver."
            )

        try:
            effective_v_s_max = self._get_effective_v_s_max(self.param)
            cov_floor = max(float(self.param.risk_cov_jitter), 0.0)
            for i in range(1, self.N + 1):
                xi = solver.get(i, "x")
                xi[3] = float(np.clip(xi[3], s_lower, s_upper))
                xi[4:7] = np.maximum(xi[4:7], cov_floor)
                solver.set(i, "x", xi)
            for i in range(self.N):
                ui = solver.get(i, "u")
                ui[2] = float(np.clip(ui[2], 0.0, effective_v_s_max))
                solver.set(i, "u", ui)
        except Exception:
            pass

        x_guess = np.stack(
            [solver.get(i, "x") for i in range(self.N + 1)],
            axis=0,
        )
        stage_s_values = self._predict_stage_s_values(
            self._current_s0,
            s_lower,
            s_upper,
        )

        constraint_data = self._compute_constraint_affine_data(x_guess)
        nominal_mf_residual = (
            np.einsum("ij,ij->i", constraint_data["mf_A_raw"], x_guess)
            + constraint_data["mf_c_raw"]
        )
        nominal_probability_sum = constraint_data["epsilon"] - nominal_mf_residual
        nominal_probability_sum[~constraint_data["mf_valid"]] = np.nan

        exact_current_phi, _ = self._compute_exact_psdf(x_guess[0, :3])
        constraint_data["exact_current_phi"] = float(exact_current_phi)
        self._constraint_data = constraint_data
        self._set_constraint_parameters(
            solver,
            stage_s_values,
            constraint_data,
        )
        status = solver.solve()
        self._last_nominal_probability_sum = nominal_probability_sum.copy()
        self._last_boole_risk_visualization_data = {
            "nominal_poses": x_guess[:, :3].copy(),
            "boole_risk_sum": nominal_probability_sum.copy(),
            "epsilon": np.asarray(
                constraint_data["epsilon"],
                dtype=float,
            ).copy(),
            "mf_mask": np.asarray(
                constraint_data["mf_mask"],
                dtype=bool,
            ).copy(),
        }
        return status

    def _try_backup_recovery(self):
        if self.backup_solver is None:
            print("Backup solver is unavailable; falling back to safe stop.")
            return None

        main_constraint_data = copy.deepcopy(self._constraint_data)
        main_probability_sum = (
            None
            if self._last_nominal_probability_sum is None
            else self._last_nominal_probability_sum.copy()
        )
        main_visualization_data = (
            None
            if self._last_boole_risk_visualization_data is None
            else {
                key: np.asarray(value).copy()
                for key, value in self._last_boole_risk_visualization_data.items()
            }
        )

        def restore_main_diagnostics():
            self._constraint_data = main_constraint_data
            self._last_nominal_probability_sum = main_probability_sum
            self._last_boole_risk_visualization_data = main_visualization_data

        try:
            self.backup_solver.reset(reset_qp_solver_mem=1)
            try:
                self._copy_primal_guess(self.solver, self.backup_solver)
            except Exception:
                self.add_warm_start(
                    self.param,
                    self._system,
                    solver=self.backup_solver,
                )
            status = self._prepare_and_solve(self.backup_solver)
            self._last_backup_status = int(status)
            if status != 0:
                self._backup_solver_failure_count += 1
            if status == 0:
                try:
                    self._copy_primal_guess(self.backup_solver, self.solver)
                except Exception:
                    pass
                print("Recovered with backup SQP_WITH_FEASIBLE_QP solver.")
                return self.backup_solver, 0, "backup_feasible_qp"

            restore_main_diagnostics()
            try:
                self._copy_primal_guess(self.backup_solver, self.solver)
            except Exception:
                pass
            print(
                f"Backup SQP_WITH_FEASIBLE_QP returned status {status}; "
                "falling back to safe stop."
            )
        except Exception as error:
            restore_main_diagnostics()
            print(f"Warning: backup solver recovery failed: {error}")
        return None

    def _build_safe_stop_plan(self, dynamics):
        dt = float(self.param.tf) / max(self.N, 1)
        progress = self._clamp_s(self._current_s0)
        pose = np.asarray(self.state._x, dtype=float).copy()
        previous_input = np.asarray(
            getattr(self.state, "_u", np.zeros(2)),
            dtype=float,
        ).reshape(-1)
        previous_speed = float(previous_input[0]) if previous_input.size else 0.0
        states = np.zeros((self.N + 1, self.nx), dtype=float)
        inputs = np.zeros((self.N, self.nu), dtype=float)
        states[0, :4] = [*pose, progress]

        for stage in range(self.N):
            next_pose, nominal_input = dynamics.nominal_safe_controller(
                pose,
                dt,
                previous_speed,
                -1.0,
                1.0,
            )
            nominal_input = np.asarray(nominal_input, dtype=float).reshape(-1)
            speed = float(nominal_input[0]) if nominal_input.size else 0.0
            omega = float(nominal_input[1]) if nominal_input.size > 1 else 0.0
            inputs[stage] = [speed, omega, 0.0]
            states[stage + 1, :4] = [*next_pose, progress]
            pose = np.asarray(next_pose, dtype=float)
            previous_speed = speed

        states[:, 4:8] = self._build_initial_covariance_state(self.param)
        return states, inputs

    def _apply_safe_stop(self, dynamics, solver_status):
        try:
            self.solver.reset(reset_qp_solver_mem=1)
            states, inputs = self._build_safe_stop_plan(dynamics)
            for stage in range(self.N):
                self.solver.set(stage, "x", states[stage])
                self.solver.set(stage, "u", inputs[stage])
            self.solver.set(self.N, "x", states[-1])
        except Exception as error:
            print(f"Warning: infeasible recovery failed: {error}")
        return self.solver, solver_status, "safe_stop"

    def recovery_infeasible(self, solver_status):
        self._last_backup_status = None
        if self.solver is None or self.state is None:
            return self.solver, solver_status, "safe_stop"

        dynamics = self._system_dynamics
        if dynamics is None or not hasattr(dynamics, "nominal_safe_controller"):
            print(
                "Warning: nominal_safe_controller is unavailable; "
                "keeping failed solver iterate."
            )
            return self.solver, solver_status, "sqp_rti"

        print(
            f"SQP_RTI solver failed with status {solver_status}. "
            "Starting infeasible recovery."
        )
        backup_result = self._try_backup_recovery()
        if backup_result is not None:
            return backup_result
        return self._apply_safe_stop(dynamics, solver_status)

    def update_obstacles(self, obstacles_geo):
        if self.psdf_wrapper is None:
            return
        if obstacles_geo is None:
            self.psdf_wrapper.clear_clusters()
            return

        all_clusters_A = []
        all_clusters_B = []
        for obs_geo in obstacles_geo:
            edgesA, edgesB = polygon_to_edges(obs_geo)
            all_clusters_A.append(edgesA)
            all_clusters_B.append(edgesB)

        if len(all_clusters_A) == 0:
            self.psdf_wrapper.clear_clusters()
        elif len(all_clusters_A) <= self.psdf_wrapper.K_max:
            self.psdf_wrapper.update_edge_clusters(all_clusters_A, all_clusters_B)
        else:
            self.psdf_wrapper.update_edge_clusters(
                all_clusters_A[:self.psdf_wrapper.K_max],
                all_clusters_B[:self.psdf_wrapper.K_max],
            )

    def _setup_backup_solver(self, param, obstacles):
        self.backup_solver = None
        if not param.enable_backup_solver:
            print(
                "RMPCC-MF backup SQP_WITH_FEASIBLE_QP solver disabled; "
                "using nominal safe-stop recovery."
            )
            return

        main_ocp = self.ocp
        try:
            backup_param = copy.deepcopy(param)
            backup_param.nlp_solver_type = "SQP_WITH_FEASIBLE_QP"
            backup_param.hpipm_mode = "ROBUST"
            backup_param.regularize_method = "PROJECT"
            self.setup_ocp(backup_param)
            self.add_obstacle_avoidance_constraint(backup_param, obstacles)
            self.backup_solver = self.create_solver(
                json_filename=self.backup_json_filename,
                code_export_directory=self.backup_code_export_directory,
            )
            self.add_warm_start(param, self._system, solver=self.backup_solver)
        except Exception as error:
            self.backup_solver = None
            print(f"Warning: failed to initialize backup solver: {error}")
        finally:
            self.ocp = main_ocp

    def setup(self, param, system, reference_trajectory, obstacles):
        self.param = param
        self.set_state(system._state)
        self._system = system
        self._system_dynamics = system._dynamics
        self.set_reference_trajectory(reference_trajectory)

        if self._is_initialized:
            if param.use_obstacle_constraint:
                self.update_obstacles(obstacles)
            return

        print("Setting up RMPCC-MF optimizer...")
        active_stages = np.flatnonzero(
            self._build_mf_stage_mask(int(param.horizon) + 1)
        ).tolist()
        print(
            "RMPCC-MF Row M configuration: "
            f"enabled={bool(param.use_row_mf)}, "
            f"active only when phi > d_mf_mask={float(param.d_mf_mask):.8g}, "
            f"configured stages={active_stages}"
        )
        if param.use_obstacle_constraint:
            self.initialize_augmented_psdf(
                system,
                E_max=100,
                K_max=20,
                device=param.augmented_psdf_device,
            )
        else:
            self.psdf_wrapper = None
            print(
                "RMPCC-MF obstacle constraint disabled: "
                "skipping augmented PSDF initialization."
            )

        self.setup_ocp(param)
        self.add_obstacle_avoidance_constraint(param, obstacles)
        self.solver = self.create_solver(
            code_export_directory=self.code_export_directory,
        )
        self.add_warm_start(param, system, solver=self.solver)
        self._setup_backup_solver(param, obstacles)
        self._is_initialized = True

    def _report_solver_failure(self, status):
        report_solver_failure(
            self.solver,
            status,
            max_iteration_message=(
                "Acados reached the maximum SQP iterations before convergence."
            ),
        )

    def _record_recovery(self, mode):
        if mode == "backup_feasible_qp":
            self._backup_feasible_qp_success_count += 1
        elif mode == "safe_stop":
            self._safe_stop_count += 1
            print(
                "Applied nominal_safe_controller safe-stop plan after "
                "infeasible RMPCC-MF solve."
            )

    def _update_predicted_progress(self, solver, status):
        if status != 0:
            self._prev_predicted_s = self._clamp_s(self._current_s0)
            self._has_prev_predicted_s = False
            return
        try:
            stage = 1 if self.N else 0
            self._prev_predicted_s = self._clamp_s(float(solver.get(stage, "x")[3]))
            self._has_prev_predicted_s = True
        except Exception:
            self._prev_predicted_s = self._clamp_s(self._current_s0)
            self._has_prev_predicted_s = False

    @staticmethod
    def _format_runtime_value(value):
        if value is None or not np.isfinite(value):
            return "n/a"
        return f"{float(value):+.4g}"

    def _record_constraint_log(self, diagnostics):
        tolerance = max(float(self.param.constraint_debug_tolerance), 0.0)
        for stage in diagnostics["constraint_stages"]:
            stage = int(stage)
            mf_active = bool(diagnostics["mf_mask"][stage])
            row_g_residual = float(diagnostics["guard_affine_residual"][stage])

            row_m_residual = None
            row_m_slack = None
            row_m_effective_residual = None
            row_m_feasible_before_slack = None
            row_m_feasible_after_slack = None
            if mf_active:
                value = float(diagnostics["mf_affine_residual"][stage])
                if np.isfinite(value):
                    row_m_residual = value
                    row_m_feasible_before_slack = value >= -tolerance

                value = float(diagnostics["mf_lower_slack"][stage])
                if np.isfinite(value):
                    row_m_slack = value

                value = float(
                    diagnostics["solver"]["qp_fixed_row_lower_residual"][stage, 1]
                )
                if np.isfinite(value):
                    row_m_effective_residual = value
                    row_m_feasible_after_slack = value >= -tolerance

            phi = float(diagnostics["mf_activation_phi"][stage])
            self._constraint_log_rows.append(
                {
                    "solve_step": int(self._solve_count),
                    "horizon_stage": stage,
                    "phi": phi,
                    "d_mf_mask": float(diagnostics["d_mf_mask"]),
                    "phi_gt_d_mf_mask": bool(
                        diagnostics["mf_domain_eligible"][stage]
                    ),
                    "mf_valid": bool(diagnostics["mf_valid"][stage]),
                    "mf_active": mf_active,
                    "row_g_residual": row_g_residual,
                    "row_g_feasible": row_g_residual >= -tolerance,
                    "row_m_residual_before_slack": row_m_residual,
                    "row_m_lower_slack": row_m_slack,
                    "row_m_residual_after_slack": row_m_effective_residual,
                    "row_m_feasible_before_slack": row_m_feasible_before_slack,
                    "row_m_feasible_after_slack": row_m_feasible_after_slack,
                }
            )

    @staticmethod
    def _finite_or_none(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    @classmethod
    def _json_vector(cls, values):
        if values is None:
            return "[]"
        payload = [
            cls._finite_or_none(value)
            for value in np.asarray(values, dtype=float).reshape(-1)
        ]
        return json.dumps(payload, separators=(",", ":"))

    def _get_cycle_log_time(self):
        system_time = getattr(self._system, "_time", None)
        if system_time is not None and np.isfinite(system_time):
            return float(system_time)
        dt = float(self.param.tf) / float(max(self.N, 1))
        return float(self._solve_count * dt)

    def _record_experiment_logs(self, mode, status, solver, diagnostics):
        log_time = self._get_cycle_log_time()
        try:
            u0 = np.asarray(solver.get(0, "u"), dtype=float).reshape(-1)
        except Exception:
            u0 = np.full(3, np.nan, dtype=float)

        if diagnostics is not None:
            mf_slack = diagnostics.get("mf_lower_slack")
            psdf = diagnostics.get("exact_current_psdf")
        else:
            mf_slack = self._read_lower_slack_trajectory(solver)[:, 1]
            psdf = (
                self._constraint_data.get("exact_current_phi")
                if self._constraint_data is not None
                else None
            )

        main_status = self._last_main_status
        if main_status is None:
            main_status = status
        self._cycle_log_rows.append(
            {
                "time": log_time,
                "solve_mode": str(mode),
                "solver_status": int(main_status),
                "u0": self._json_vector(u0),
                "s": float(self._prev_predicted_s),
                "psdf": self._finite_or_none(psdf),
                "mf_probability_sum": self._json_vector(
                    self._last_nominal_probability_sum
                ),
                "mf_slack": self._json_vector(mf_slack),
            }
        )

        try:
            states, inputs = self._read_solver_trajectory(solver)
        except Exception:
            return

        for stage in range(self.N + 1):
            p_f, p_l, p_psi, p_lpsi = states[stage, 4:8]
            if stage < self.N:
                v_value = self._finite_or_none(inputs[stage, 0])
                omega_value = self._finite_or_none(inputs[stage, 1])
            else:
                v_value = None
                omega_value = None
            self._covariance_log_rows.append(
                {
                    "time": log_time,
                    "stage": stage,
                    "v": v_value,
                    "omega": omega_value,
                    "sigma_f": (
                        float(np.sqrt(max(p_f, 0.0)))
                        if np.isfinite(p_f)
                        else None
                    ),
                    "sigma_l": (
                        float(np.sqrt(max(p_l, 0.0)))
                        if np.isfinite(p_l)
                        else None
                    ),
                    "sigma_psi": (
                        float(np.sqrt(max(p_psi, 0.0)))
                        if np.isfinite(p_psi)
                        else None
                    ),
                    "P_lpsi": self._finite_or_none(p_lpsi),
                }
            )

    def _finish_solve(self, start_time, mode, status, solver, diagnostics):
        self._last_solve_mode = mode
        self._record_experiment_logs(mode, status, solver, diagnostics)
        solve_time = time.time() - start_time
        self.solver_times.append(solve_time)

        if mode == "safe_stop":
            outcome = "SAFE_STOP"
        elif mode == "backup_feasible_qp":
            outcome = "RECOVERED"
        elif status == 0:
            outcome = "OK"
        else:
            outcome = "FAIL"

        pose = np.asarray(self.state._x, dtype=float) if self.state is not None else None
        try:
            stage_zero_input = np.asarray(solver.get(0, "u"), dtype=float)
        except Exception:
            stage_zero_input = None

        pose_text = "pose=n/a"
        if pose is not None and pose.size >= 3:
            pose_text = f"pose=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f})"
        input_text = "u=n/a"
        if stage_zero_input is not None and stage_zero_input.size >= 2:
            input_text = f"u=({stage_zero_input[0]:.3f},{stage_zero_input[1]:.3f})"

        summary = (
            f"[RMPCC {self._solve_count:04d}] {outcome} "
            f"solve={solve_time * 1e3:.1f}ms {pose_text} {input_text} "
            f"s={self._current_s0:.3f}"
        )
        if diagnostics is not None:
            tolerance = max(float(self.param.constraint_debug_tolerance), 0.0)
            guard_residuals = diagnostics["guard_affine_residual"][1:]
            guard_min = (
                float(np.nanmin(guard_residuals))
                if np.isfinite(guard_residuals).any()
                else None
            )
            guard_status = (
                "OK" if guard_min is not None and guard_min >= -tolerance else "FAIL"
            )
            psdf_min = diagnostics["minimum_exact_psdf_predicted_nodes"]
            summary += (
                f" PSDFmin={self._format_runtime_value(psdf_min)} "
                f"G={guard_status}({self._format_runtime_value(guard_min)})"
            )

            active_mask = np.asarray(diagnostics["mf_mask"][1:], dtype=bool)
            active_count = int(np.count_nonzero(active_mask))
            if not active_count:
                summary += f" MF=inactive(active=0/{self.N})"
            else:
                qp_residual = diagnostics["solver"][
                    "qp_fixed_row_lower_residual"
                ][1:, 1]
                active_residual = np.asarray(qp_residual, dtype=float)[active_mask]
                active_slack = np.asarray(
                    diagnostics["mf_lower_slack"][1:], dtype=float
                )[active_mask]
                mf_min = (
                    float(np.nanmin(active_residual))
                    if np.isfinite(active_residual).any()
                    else None
                )
                max_slack = (
                    float(np.nanmax(active_slack))
                    if np.isfinite(active_slack).any()
                    else None
                )
                mf_status = (
                    "OK" if mf_min is not None and mf_min >= -tolerance else "FAIL"
                )
                summary += (
                    f" MF={mf_status}(active={active_count}/{self.N},"
                    f"min={self._format_runtime_value(mf_min)},"
                    f"slack={self._format_runtime_value(max_slack)})"
                )
        print(summary)

    def solve_nlp(self):
        if self.solver is None:
            raise RuntimeError("RMPCC-MF solver is not initialized. Call setup() first.")

        start = time.time()
        self._last_backup_status = None
        self._last_constraint_diagnostics = None
        self._last_failed_constraint_diagnostics = None
        # Match the working MPCC/RMPCC-PV lifecycle: linearize Row G/M at the
        # solver's retained warm-start trajectory instead of manually shifting
        # that trajectory before each RTI solve.
        status = self._prepare_and_solve(self.solver)
        self._last_main_status = int(status)
        if status != 0:
            self._main_solver_failure_count += 1
        active_solver = self.solver
        mode = "sqp_rti"

        if status != 0:
            self._report_solver_failure(status)
            use_debug = bool(self.param.debug_infeasibility)
            if use_debug:
                failed_diagnostics = self.compute_diagnostics(
                    self.solver,
                    status,
                    "sqp_rti_failed_before_recovery",
                    failed=True,
                    report=False,
                )
                if failed_diagnostics is not None:
                    self.report_diagnostics(
                        failed_diagnostics,
                        failed=True,
                        include_constraint=False,
                    )
            active_solver, status, mode = self.recovery_infeasible(status)
            self._record_recovery(mode)

        if mode == "safe_stop":
            # The safe-stop state trajectory is not the nominal trajectory on
            # which the MF sum was evaluated.  Keep the raw experiment log, but
            # suppress the spatial overlay for this cycle to avoid mixing them.
            self._last_boole_risk_visualization_data = None

        self._update_predicted_progress(active_solver, status)

        diagnostics = None
        use_debug = bool(self.param.debug_mf)
        if use_debug:
            diagnostics = self.compute_diagnostics(
                active_solver,
                status,
                mode,
                report=False,
            )
        if diagnostics is not None:
            self._record_constraint_log(diagnostics)
        self._finish_solve(start, mode, status, active_solver, diagnostics)
        self._solve_count += 1
        return AcadosSolution(active_solver, self.N)

    def initialize_augmented_psdf(self, system, E_max=100, K_max=20, device="cpu"):
        self.device = device
        if not hasattr(system, "_geometry"):
            raise ValueError("System must have _geometry attribute")

        geometry = system._geometry._geometries[0]
        vertices = geometry._region.get_ccw_vertices()
        verts_t = torch.tensor(vertices, dtype=torch.float32, device=device)
        self.psdf_wrapper = AugmentedPSDFWrapper(
            verts=verts_t,
            E_max=E_max,
            K_max=K_max,
            device=device,
        )


class AcadosSolution:
    def __init__(self, solver, horizon):
        self.solver = solver
        self.N = horizon

    def value(self, var_expr):
        if var_expr == "x":
            return np.stack([self.solver.get(i, "x") for i in range(self.N + 1)], axis=1)
        if var_expr == "u":
            return np.stack([self.solver.get(i, "u") for i in range(self.N)], axis=1)
        raise NotImplementedError("Only 'x' and 'u' supported in AcadosSolution.value")

    def get_state_trajectory(self):
        return self.value("x")

    def get_input_trajectory(self):
        return self.value("u")

    def stats(self):
        return {"return_status": "success" if self.solver.status == 0 else "failure"}
