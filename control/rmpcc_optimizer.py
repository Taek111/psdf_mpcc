import copy
import os
import time

import casadi as ca
import numpy as np
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.geometry_utils import polygon_to_edges
from planning.trajectory_generator.spline_reference_generator import build_cubic_spline_path_data


class RMPCCOptimizerParam:
    def __init__(self):
        self.horizon = 20
        self.tf = 0.1 * self.horizon

        # MPCC cost weights
        self.mat_Qe = np.diag([1.0, 10.0])   # [contour, lag] # 1.0 10.0
        self.mat_Re = np.diag([1.0, 0.01])    # [v, omega]    # 0.2, 0.01 
        self.q_s = 1.0
        self.terminal_weight = 5.0 # default 10, 50 
        # Desired unconstrained progress speed. The quadratic progress penalty is
        # chosen so q_s / (2 * r_v_s) == v_s_target.
        self.v_s_target = 0.8
        self.q_s_ref = 2.0

        # MPCC path/progress parameters
        self.v_s_max = 2.0
        # Keep the progress-state rate aligned with the physical speed limit by default.
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
        # Allow limited progress backtracking near corners so the nominal can
        # switch active faces instead of staying pinned to the pre-turn tangent.
        self.s_backtrack_tolerance = 0.3
        
        # Solver options
        self.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        # EXTERNAL cost in this MPCC formulation is more reliable with an exact Hessian.
        self.hessian_approx = "GAUSS_NEWTON"
        self.integrator_type = "ERK"
        self.nlp_solver_type = "SQP_RTI"
        self.qp_solver_iter_max = 50
        self.nlp_solver_max_iter = 50
        self.tol = 1e-4
        # The acados SQP_WITH_FEASIBLE_QP fallback can terminate the Python
        # process inside the native QP solver on difficult MF maze stages.
        # Keep it opt-in; the deterministic nominal safe-stop remains the
        # default recovery path when SQP_RTI reports infeasibility.
        self.enable_backup_solver = False


        # Physical input bounds
        self.vmin, self.vmax = -0.7, 0.7
        self.omegamin, self.omegamax = -1.2, 1.2

        # Safety constraints (PSDF style)
        self.d_min = 0.001
        # Row G uses one fixed signed-PSDF target throughout the horizon.
        # Keep this separate from d_min, which is the distance threshold used
        # inside the multi-feature probability model.
        self.d_col = 0.001
        # Row G is always active and hard.  Row M can be switched off without
        # changing the fixed two-row OCP layout; its stage parameters are then
        # replaced by the always-feasible residual chance_epsilon.
        self.use_row_mf = False
        # The unsigned feature-distance model is used only in the separated
        # domain.  Zero therefore masks contact and penetration by default.
        self.d_mf_mask = 0.0
        self.use_obstacle_constraint = True
        self.chance_epsilon = 0.20
        self.debug_mf = True
        self.augmented_psdf_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Row G is a hard lower-bound constraint.  Only Row M has a slack, with
        # the same raw-row penalty at every intermediate/terminal stage.
        self.mf_slack_linear = 1e2
        self.mf_slack_quadratic = 1e0

        # Number of equal-duration samples used to check the executed first
        # shooting interval, in addition to its measured starting pose.
        self.first_interval_substeps = 10

        # Topic 2 risk/covariance parameters.
        self.sigma_f0 = 0.0002 #0.0002
        self.sigma_l0 = 0.00025 #0.00025
        self.sigma_psi0 = 0.00025 #0.00025
        self.q_f0 = 0 #1e-5
        self.q_l0 = 0 #2.5e-5
        self.q_psi0 = 0 #2.5e-5
        self.alpha_f = 0.002
        self.alpha_v = 0.0004
        self.alpha_kappa = 0.01
        self.beta_v = 0.02
        self.beta_kappa = 0.008
        self.beta_omega = 0.008
        self.risk_cov_jitter = 1e-9

class RMPCCOptimizer:
    def __init__(self, variables=None, costs=None, dynamics_opt=None):
        self.ocp = None
        self.solver = None
        self.solver_times = []
        self.state = None
        self.reference_path_data = None
        self.N = None
        self.nx = None
        self.nu = None
        self.variables = {}
        self.costs = {} if costs is None else costs
        self.dynamics_opt = dynamics_opt

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
        self._path_params = None
        self._prev_predicted_s = 0.0
        self._current_s0 = 0.0
        self._has_prev_predicted_s = False
        self._last_mf_residuals = None
        self._last_mf_probability_sums = None
        self._last_mf_A = None
        self._last_mf_c = None
        self._last_mf_A_raw = None
        self._last_mf_c_raw = None
        self._last_mf_valid = None
        self._last_mf_mask = None
        self._last_row_mf_enabled = None
        self._last_guard_A = None
        self._last_guard_c = None
        self._last_nominal_psdf_clearances = None
        self._last_local_covariances = None
        self._last_psdf_clearances = None
        self._last_exact_current_phi = None
        self._last_constraint_diagnostics = None
        self._mf_stage_data_available = None
        self._has_shift_source = False

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
            for file_path in self._temp_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            if os.path.exists(self.json_filename):
                os.remove(self.json_filename)
            if os.path.exists(self.backup_json_filename):
                os.remove(self.backup_json_filename)
            if os.path.exists(self.code_export_directory) and os.path.isdir(self.code_export_directory):
                import shutil
                shutil.rmtree(self.code_export_directory, ignore_errors=True)
            if os.path.exists(self.backup_code_export_directory) and os.path.isdir(self.backup_code_export_directory):
                import shutil
                shutil.rmtree(self.backup_code_export_directory, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

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
        self._path_params = None
        self._prev_predicted_s = 0.0
        self._current_s0 = 0.0
        self._has_prev_predicted_s = False
        self._last_mf_residuals = None
        self._last_mf_probability_sums = None
        self._last_mf_A = None
        self._last_mf_c = None
        self._last_mf_A_raw = None
        self._last_mf_c_raw = None
        self._last_mf_valid = None
        self._last_mf_mask = None
        self._last_row_mf_enabled = None
        self._last_guard_A = None
        self._last_guard_c = None
        self._last_nominal_psdf_clearances = None
        self._last_local_covariances = None
        self._last_psdf_clearances = None
        self._last_exact_current_phi = None
        self._last_constraint_diagnostics = None
        self._mf_stage_data_available = None
        self._has_shift_source = False
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
            "safe_stop_count": int(self._safe_stop_count),
            "plant_input_apply_count": int(self._plant_input_apply_count),
            "last_solve_mode": str(self._last_solve_mode),
        }

    def get_last_mf_residual_trajectory(self):
        if self._last_mf_residuals is None:
            return None
        return np.asarray(self._last_mf_residuals, dtype=float).copy()

    def get_last_mf_probability_sum_trajectory(self):
        if self._last_mf_probability_sums is None:
            return None
        return np.asarray(self._last_mf_probability_sums, dtype=float).copy()

    def get_last_psdf_clearance_trajectory(self):
        if self._last_psdf_clearances is None:
            return None
        return np.asarray(self._last_psdf_clearances, dtype=float).copy()

    def get_last_constraint_diagnostics(self):
        """Return the latest fixed-row safety diagnostics.

        Constraint ordering is invariant throughout this dictionary:
        row 0 is the signed-PSDF guard/recovery row and row 1 is the MF row.
        """
        if self._last_constraint_diagnostics is None:
            return None
        return copy.deepcopy(self._last_constraint_diagnostics)

    def record_plant_input_applied(self):
        self._plant_input_apply_count += 1

    def _get_effective_v_s_max(self, param):
        v_s_max = float(getattr(param, "v_s_max", 0.0))
        if getattr(param, "enforce_v_s_not_faster_than_vmax", True):
            return min(v_s_max, max(0.0, float(getattr(param, "vmax", 0.0))))
        return v_s_max

    def _get_progress_speed_penalty(self, param):
        v_s_target = max(float(getattr(param, "v_s_target", 0.6)), 1e-6)
        q_s = max(float(getattr(param, "q_s", 0.0)), 0.0)
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
            float(getattr(param, "q_f0", 0.0)) + float(getattr(param, "alpha_f", 0.0)) * v * v,
            0.0,
        )
        q_l_noise = ca.fmax(
            float(getattr(param, "q_l0", 0.0))
            + float(getattr(param, "alpha_v", 0.0)) * v * v
            + float(getattr(param, "alpha_kappa", 0.0)) * v * v * ca.fabs(kappa_ref),
            0.0,
        )
        q_psi_noise = ca.fmax(
            float(getattr(param, "q_psi0", 0.0))
            + float(getattr(param, "beta_v", 0.0)) * v * v
            + float(getattr(param, "beta_kappa", 0.0)) * v * v * ca.fabs(kappa_ref)
            + float(getattr(param, "beta_omega", 0.0)) * omega * omega,
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
        q_s_ref = float(getattr(param, "q_s_ref", 0.0))
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

    def _normalize_reference_path_data(self, reference_path_data):
        if isinstance(reference_path_data, np.ndarray):
            return build_cubic_spline_path_data(reference_path_data)

        if isinstance(reference_path_data, dict):
            def _pick(keys):
                for key in keys:
                    if key in reference_path_data and reference_path_data[key] is not None:
                        return reference_path_data[key]
                return None

            seg_x_raw = _pick(["segments_x", "coeff_x", "cx"])
            seg_y_raw = _pick(["segments_y", "coeff_y", "cy"])
            s_breaks_raw = _pick(["s_breaks", "S", "s_nodes", "segment_bounds"])
            if seg_x_raw is None or seg_y_raw is None or s_breaks_raw is None:
                raise ValueError(
                    "reference_path_data dict must contain spline coefficients and s-breaks."
                )

            seg_x = np.asarray(seg_x_raw, dtype=float)
            seg_y = np.asarray(seg_y_raw, dtype=float)
            s_breaks = np.asarray(s_breaks_raw, dtype=float).reshape(-1)
            n_segments = int(reference_path_data.get("n_segments", seg_x.shape[0]))
            if seg_x.shape != seg_y.shape or seg_x.shape[1] != 4:
                raise ValueError("segments_x/segments_y must have shape [n_segments, 4].")
            if s_breaks.shape[0] < n_segments + 1:
                raise ValueError("s_breaks must have length >= n_segments + 1.")
            return {
                "segments_x": seg_x,
                "segments_y": seg_y,
                "s_breaks": s_breaks,
                "n_segments": n_segments,
            }

        raise ValueError("reference_path_data must be ndarray or dict.")

    def _reference_segment_numeric(self, s, path_data):
        n_seg = int(path_data["n_segments"])
        if n_seg <= 0:
            return None

        s_breaks = np.asarray(path_data["s_breaks"], dtype=float)
        s_min = float(s_breaks[0])
        s_max = float(s_breaks[n_seg])
        s_clamped = float(np.clip(float(s), s_min, s_max))

        seg_idx = int(np.searchsorted(s_breaks[1:n_seg + 1], s_clamped, side="right"))
        seg_idx = int(np.clip(seg_idx, 0, n_seg - 1))
        tau = s_clamped - float(s_breaks[seg_idx])

        cx = np.asarray(path_data["segments_x"][seg_idx], dtype=float)
        cy = np.asarray(path_data["segments_y"][seg_idx], dtype=float)
        return s_min, s_max, s_clamped, tau, cx, cy

    def _reference_eval_numeric(self, s, path_data):
        segment = self._reference_segment_numeric(s, path_data)
        if segment is None:
            return np.zeros((2,), dtype=float)

        _, _, _, tau, cx, cy = segment
        x_ref = cx[0] + cx[1] * tau + cx[2] * tau * tau + cx[3] * tau * tau * tau
        y_ref = cy[0] + cy[1] * tau + cy[2] * tau * tau + cy[3] * tau * tau * tau
        return np.array([x_ref, y_ref], dtype=float)

    def _reference_tangent_numeric(self, s, path_data):
        segment = self._reference_segment_numeric(s, path_data)
        if segment is None:
            return np.array([1.0, 0.0], dtype=float)

        s_min, s_max, s_clamped, tau, cx, cy = segment
        dx_ds = cx[1] + 2.0 * cx[2] * tau + 3.0 * cx[3] * tau * tau
        dy_ds = cy[1] + 2.0 * cy[2] * tau + 3.0 * cy[3] * tau * tau
        t_vec = np.array([dx_ds, dy_ds], dtype=float)

        tangent_reg = max(float(getattr(getattr(self, "param", None), "tangent_reg_delta", 1e-6)), 1e-9)
        t_norm = float(np.linalg.norm(t_vec))
        if t_norm <= tangent_reg:
            fd_eps = max(float(getattr(getattr(self, "param", None), "fd_eps", 1e-3)), 1e-4)
            s_prev = float(np.clip(s_clamped - fd_eps, s_min, s_max))
            s_next = float(np.clip(s_clamped + fd_eps, s_min, s_max))
            if s_next > s_prev + 1e-10:
                p_prev = self._reference_eval_numeric(s_prev, path_data)
                p_next = self._reference_eval_numeric(s_next, path_data)
                t_vec = p_next - p_prev
                t_norm = float(np.linalg.norm(t_vec))

        if t_norm <= tangent_reg:
            return np.array([1.0, 0.0], dtype=float)
        return t_vec / (t_norm + tangent_reg)

    def _reference_curvature_numeric(self, s, path_data):
        segment = self._reference_segment_numeric(s, path_data)
        if segment is None:
            return 0.0

        _, _, _, tau, cx, cy = segment
        dx_ds = cx[1] + 2.0 * cx[2] * tau + 3.0 * cx[3] * tau * tau
        dy_ds = cy[1] + 2.0 * cy[2] * tau + 3.0 * cy[3] * tau * tau
        ddx_ds2 = 2.0 * cx[2] + 6.0 * cx[3] * tau
        ddy_ds2 = 2.0 * cy[2] + 6.0 * cy[3] * tau

        speed_sq = dx_ds * dx_ds + dy_ds * dy_ds
        if speed_sq <= 1e-12:
            return 0.0

        return float((dx_ds * ddy_ds2 - dy_ds * ddx_ds2) / (speed_sq ** 1.5))

    def _build_stage_path_parameter(self, s_value):
        if self.reference_path_data is None:
            return np.array([0.0, 0.0, 1.0, 0.0, float(s_value), 0.0], dtype=float)
        p_ref = self._reference_eval_numeric(s_value, self.reference_path_data)
        t_ref = self._reference_tangent_numeric(s_value, self.reference_path_data)
        kappa_ref = self._reference_curvature_numeric(s_value, self.reference_path_data)
        return np.array(
            [p_ref[0], p_ref[1], t_ref[0], t_ref[1], float(s_value), kappa_ref],
            dtype=float,
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
        jitter = max(float(getattr(self.param, "risk_cov_jitter", 1e-9)), 0.0)
        eigvals, eigvecs = np.linalg.eigh(Sigma_sym)
        eigvals = np.clip(eigvals, jitter, None)
        Sigma_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return 0.5 * (Sigma_psd + Sigma_psd.T)

    def _build_initial_local_covariance(self, param):
        sigma_f0 = float(max(getattr(param, "sigma_f0", 0.0), 0.0))
        sigma_l0 = float(max(getattr(param, "sigma_l0", 0.0), 0.0))
        sigma_psi0 = float(max(getattr(param, "sigma_psi0", 0.0), 0.0))
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
        return np.array([Sigma_np[0, 0], Sigma_np[1, 1], Sigma_np[2, 2], Sigma_np[1, 2]], dtype=float)

    def _debug_mf_enabled(self, param=None):
        target_param = self.param if param is None else param
        return bool(getattr(target_param, "debug_mf", False))

    def _normalize_epsilon_values(self, num_stages):
        epsilon = np.asarray(
            getattr(self.param, "chance_epsilon", 0.20),
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

        if self.psdf_wrapper is None or not getattr(
            self.param,
            "use_obstacle_constraint",
            True,
        ):
            phi = np.full((num_stages,), 1000.0, dtype=float)
            gradient = np.zeros((num_stages, 3), dtype=float)
            A_mf = np.zeros((num_stages, 8), dtype=float)
            c_mf = epsilon.copy()
            self._mf_stage_data_available = np.zeros((num_stages,), dtype=bool)
            return phi, gradient, A_mf, c_mf

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
        self._mf_stage_data_available = stage_available & has_active_features
        return phi, gradient, A_mf, c_mf

    def _compute_constraint_affine_data(self, z_bar):
        """Build fixed Row G/Row M coefficients and the per-stage MF mask."""
        z_bar_np = np.asarray(z_bar, dtype=float)
        if z_bar_np.ndim != 2 or z_bar_np.shape[1] != 8:
            raise ValueError("z_bar must have shape (H, 8)")
        if not np.isfinite(z_bar_np).all():
            raise ValueError("z_bar values must be finite")

        self._mf_stage_data_available = None
        num_stages = z_bar_np.shape[0]
        epsilon = self._normalize_epsilon_values(num_stages)
        row_mf_enabled = bool(getattr(self.param, "use_row_mf", True))
        obstacle_constraint_enabled = bool(
            getattr(self.param, "use_obstacle_constraint", True)
        )

        if not obstacle_constraint_enabled:
            # Preserve the master obstacle-constraint switch even if a wrapper
            # from an earlier setup is still attached to the optimizer.
            phi = np.full((num_stages,), 1000.0, dtype=float)
            gradient = np.zeros((num_stages, 3), dtype=float)
            A_mf_raw = np.full((num_stages, 8), np.nan, dtype=float)
            c_mf_raw = np.full((num_stages,), np.nan, dtype=float)
            self._mf_stage_data_available = np.zeros(
                (num_stages,),
                dtype=bool,
            )
        elif row_mf_enabled:
            phi, gradient, A_mf_raw, c_mf_raw = self._compute_mf_affine_data(
                z_bar_np
            )
        else:
            # A disabled Row M must not evaluate the covariance-dependent MF
            # model.  Retain the pose-only signed PSDF evaluation required to
            # build hard Row G, and mark raw MF data as intentionally absent.
            phi, gradient = self._compute_exact_psdf(z_bar_np[:, :3])
            phi = np.asarray(phi, dtype=float)
            gradient = np.asarray(gradient, dtype=float)
            A_mf_raw = np.full((num_stages, 8), np.nan, dtype=float)
            c_mf_raw = np.full((num_stages,), np.nan, dtype=float)
            self._mf_stage_data_available = np.zeros(
                (num_stages,),
                dtype=bool,
            )

        d_col = float(getattr(self.param, "d_col", self.param.d_min))
        d_mf_mask = float(getattr(self.param, "d_mf_mask", 0.0))
        if not np.isfinite(d_col) or not np.isfinite(d_mf_mask):
            raise ValueError("d_col and d_mf_mask must be finite")

        guard_A = np.zeros((num_stages, 8), dtype=float)
        guard_A[:, :3] = gradient
        guard_c = phi - np.einsum("ij,ij->i", gradient, z_bar_np[:, :3]) - d_col

        finite_mf = np.isfinite(A_mf_raw).all(axis=1) & np.isfinite(c_mf_raw)
        stage_available = getattr(self, "_mf_stage_data_available", None)
        if stage_available is None or np.asarray(stage_available).shape != (num_stages,):
            stage_available = finite_mf.copy()
        else:
            stage_available = np.asarray(stage_available, dtype=bool)
        mf_valid = row_mf_enabled & stage_available & finite_mf
        mf_mask = row_mf_enabled & (phi > d_mf_mask) & mf_valid

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
            "mf_A": A_mf,
            "mf_c": c_mf,
            "mf_mask": mf_mask,
            "epsilon": epsilon,
        }

    def _propagate_covariance_state_batch(self, inputs, stage_s_values):
        num_stages = len(stage_s_values)
        dt = float(self.param.tf) / float(max(self.N, 1))
        cov_floor = max(float(getattr(self.param, "risk_cov_jitter", 1e-9)), 0.0)

        covariance_state = self._build_initial_covariance_state(self.param)
        covariance_batch = np.zeros((num_stages, 4), dtype=float)

        for i in range(num_stages):
            covariance_batch[i] = covariance_state
            if i >= inputs.shape[0]:
                continue

            v_i = float(inputs[i, 0])
            omega_i = float(inputs[i, 1])
            kappa_i = 0.0 if self.reference_path_data is None else self._reference_curvature_numeric(
                stage_s_values[i],
                self.reference_path_data,
            )

            q_f = max(float(getattr(self.param, "q_f0", 0.0)) + float(getattr(self.param, "alpha_f", 0.0)) * v_i * v_i, 0.0)
            q_l = max(
                float(getattr(self.param, "q_l0", 0.0))
                + float(getattr(self.param, "alpha_v", 0.0)) * v_i * v_i
                + float(getattr(self.param, "alpha_kappa", 0.0)) * v_i * v_i * abs(kappa_i),
                0.0,
            )
            q_psi = max(
                float(getattr(self.param, "q_psi0", 0.0))
                + float(getattr(self.param, "beta_v", 0.0)) * v_i * v_i
                + float(getattr(self.param, "beta_kappa", 0.0)) * v_i * v_i * abs(kappa_i)
                + float(getattr(self.param, "beta_omega", 0.0)) * omega_i * omega_i,
                0.0,
            )

            p_f, p_l, p_psi, p_lpsi = covariance_state
            covariance_state = np.array(
                [
                    p_f + (dt * omega_i) ** 2 * p_l + dt * q_f,
                    p_l + (dt * omega_i) ** 2 * p_f + 2.0 * dt * v_i * p_lpsi + (dt * v_i) ** 2 * p_psi + dt * q_l,
                    p_psi + dt * q_psi,
                    p_lpsi + dt * v_i * p_psi,
                ],
                dtype=float,
            )

            Sigma_next = self._project_to_psd(self._covariance_state_to_matrix(covariance_state))
            Sigma_next[0, 0] = max(Sigma_next[0, 0], cov_floor)
            Sigma_next[1, 1] = max(Sigma_next[1, 1], cov_floor)
            Sigma_next[2, 2] = max(Sigma_next[2, 2], cov_floor)
            covariance_state = self._matrix_to_covariance_state(Sigma_next)

        return covariance_batch


    def _closest_s_on_path_segments(self, position_xy):
        if self.reference_path_data is None:
            return 0.0

        n_seg = int(self.reference_path_data["n_segments"])
        if n_seg <= 0:
            return 0.0

        pos = np.asarray(position_xy, dtype=float).reshape(2)
        best_dist = np.inf
        best_s = float(self.reference_path_data["s_breaks"][0])

        for i in range(n_seg):
            s_left = float(self.reference_path_data["s_breaks"][i])
            s_right = float(self.reference_path_data["s_breaks"][i + 1])
            ds = max(s_right - s_left, 1e-6)

            cx = self.reference_path_data["segments_x"][i]
            cy = self.reference_path_data["segments_y"][i]
            p0 = np.array([cx[0], cy[0]], dtype=float)
            tau1 = ds
            p1 = np.array(
                [
                    cx[0] + cx[1] * tau1 + cx[2] * tau1 * tau1 + cx[3] * tau1 * tau1 * tau1,
                    cy[0] + cy[1] * tau1 + cy[2] * tau1 * tau1 + cy[3] * tau1 * tau1 * tau1,
                ],
                dtype=float,
            )

            vec = p1 - p0
            denom = float(np.dot(vec, vec))
            if denom <= 1e-12:
                t = 0.0
            else:
                t = float(np.clip(np.dot(pos - p0, vec) / denom, 0.0, 1.0))

            proj = p0 + t * vec
            dist = float(np.linalg.norm(pos - proj))
            if dist < best_dist:
                best_dist = dist
                best_s = s_left + t * ds

        return self._clamp_s(best_s)

    def _clamp_s(self, s_value):
        if self.reference_path_data is None:
            return max(0.0, float(s_value))
        s_min = float(self.reference_path_data["s_breaks"][0])
        s_max = float(self.reference_path_data["s_breaks"][int(self.reference_path_data["n_segments"])])
        return float(np.clip(s_value, s_min, s_max))

    def _project_s_with_line_search(self, position_xy, param):
        if self.reference_path_data is None:
            return 0.0

        s_min = float(self.reference_path_data["s_breaks"][0])
        s_max = float(self.reference_path_data["s_breaks"][int(self.reference_path_data["n_segments"])])
        s_closest = self._closest_s_on_path_segments(position_xy)
        center = s_closest if not self._has_prev_predicted_s else self._clamp_s(self._prev_predicted_s)

        half_window = max(float(param.line_search_window), 1e-3)
        num_samples = max(int(param.line_search_samples), 5)
        lower = max(s_min, center - half_window)
        upper = min(s_max, center + half_window)

        if upper <= lower:
            return center

        s_grid = np.linspace(lower, upper, num_samples)
        dists = np.empty_like(s_grid)
        for i, s in enumerate(s_grid):
            p_ref = self._reference_eval_numeric(s, self.reference_path_data)
            dists[i] = np.linalg.norm(position_xy - p_ref)
        s_line = float(s_grid[int(np.argmin(dists))])

        # If line-search around previous prediction is clearly worse, fall back to closest-segment estimate.
        p_line = self._reference_eval_numeric(s_line, self.reference_path_data)
        p_closest = self._reference_eval_numeric(s_closest, self.reference_path_data)
        if np.linalg.norm(position_xy - p_closest) + 1e-6 < np.linalg.norm(position_xy - p_line):
            return s_closest
        return s_line

    def _get_current_s_bounds(self, param):
        if self.reference_path_data is None:
            return 0.0, float(param.s_upper_guard)
        s_min = float(self.reference_path_data["s_breaks"][0])
        s_max = float(self.reference_path_data["s_breaks"][int(self.reference_path_data["n_segments"])])
        s_upper = min(s_max, float(param.s_upper_guard))
        if s_upper < s_min:
            s_upper = s_min
        return s_min, s_upper

    def _get_augmented_state_bounds(self, param, s_lower, s_upper):
        idxbx = []
        lbx = []
        ubx = []

        if getattr(param, "use_s_state_bounds", True):
            idxbx.append(3)
            lbx.append(float(s_lower))
            ubx.append(float(s_upper))

        cov_floor = max(float(getattr(param, "risk_cov_jitter", 1e-9)), 0.0)
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
            self.reference_path_data = self._normalize_reference_path_data(reference_path_data)
        if self.reference_path_data is None:
            return None

        position_xy = np.asarray(state._x[:2], dtype=float)
        self._current_s0 = self._project_s_with_line_search(position_xy, self.param)
        self._current_s0 = self._clamp_s(self._current_s0)
        if self._has_prev_predicted_s:
            backtrack_tol = max(float(getattr(self.param, "s_backtrack_tolerance", 0.03)), 0.0)
            s_floor = self._clamp_s(self._prev_predicted_s - backtrack_tol)
            self._current_s0 = max(self._current_s0, s_floor)
        self._path_params = None
        return self._current_s0

    def set_reference_trajectory(self, reference_path_data):
        if reference_path_data is None:
            return
        self.reference_path_data = self._normalize_reference_path_data(reference_path_data)

    def setup_ocp(self, param, reference_path_data):
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
        self.ocp.constraints.ubu = np.array([param.vmax, param.omegamax, effective_v_s_max], dtype=float)
        self.ocp.constraints.idxbu = np.array([0, 1, 2], dtype=np.int64)

        x0_phys = np.zeros((3,), dtype=float) if self.state is None else np.asarray(self.state._x, dtype=float)
        self._current_s0 = self._project_s_with_line_search(x0_phys[:2], param)
        s_lower, s_upper = self._get_current_s_bounds(param)
        self._current_s0 = float(np.clip(self._current_s0, s_lower, s_upper))
        x0_cov = self._build_initial_covariance_state(param)
        x0_full = np.array(
            [x0_phys[0], x0_phys[1], x0_phys[2], self._current_s0, x0_cov[0], x0_cov[1], x0_cov[2], x0_cov[3]],
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
            try:
                self.ocp.solver_options.hpipm_mode = hpipm_mode
            except Exception:
                pass

        regularize_method = getattr(param, "regularize_method", None)
        if regularize_method is not None:
            try:
                self.ocp.solver_options.regularize_method = regularize_method
            except Exception:
                pass

    def create_solver(self, json_filename=None, code_export_directory=None):
        if code_export_directory is not None:
            try:
                self.ocp.code_export_directory = code_export_directory
            except Exception:
                pass

        self.variables["x"] = "x"
        self.variables["u"] = "u"
        used_json_filename = self.json_filename if json_filename is None else json_filename
        solver = AcadosOcpSolver(self.ocp, json_file=used_json_filename)
        if used_json_filename not in self._temp_files:
            self._temp_files.append(used_json_filename)
        return solver

    def add_obstacle_avoidance_constraint(self, param, system, obstacles_geo):
        if getattr(param, "use_obstacle_constraint", True):
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
            [float(getattr(param, "mf_slack_linear", 1e1))],
            dtype=float,
        )
        slack_quadratic = np.array(
            [float(getattr(param, "mf_slack_quadratic", 1e0))],
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
                    np.array([x_ws[0], x_ws[1], x_ws[2], s_ws, cov0[0], cov0[1], cov0[2], cov0[3]], dtype=float),
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

    def _rollout_terminal_pose(self, pose, physical_input, dt):
        dynamics = self._system_dynamics
        if dynamics is not None and hasattr(dynamics, "forward_dynamics"):
            return np.asarray(
                dynamics.forward_dynamics(pose, physical_input, dt),
                dtype=float,
            )
        return np.array(
            [
                pose[0] + dt * physical_input[0] * np.cos(pose[2]),
                pose[1] + dt * physical_input[0] * np.sin(pose[2]),
                pose[2] + dt * physical_input[1],
            ],
            dtype=float,
        )

    def _build_shifted_nominal(self, solver, s_lower, s_upper, shift_nominal=True):
        raw_states, raw_inputs = self._read_solver_trajectory(solver)
        nominal_states = raw_states.copy()
        nominal_inputs = raw_inputs.copy()
        dt = float(self.param.tf) / float(max(self.N, 1))

        if shift_nominal and self._has_shift_source:
            if self.N > 1:
                nominal_states[1:self.N] = raw_states[2:self.N + 1]
                nominal_inputs[:-1] = raw_inputs[1:]
            terminal_input = raw_inputs[-1].copy()
            terminal_pose = self._rollout_terminal_pose(
                raw_states[-1, :3],
                terminal_input[:2],
                dt,
            )
            nominal_states[-1, :3] = terminal_pose
            nominal_states[-1, 3] = raw_states[-1, 3] + dt * terminal_input[2]
            nominal_inputs[-1] = terminal_input

        effective_v_s_max = self._get_effective_v_s_max(self.param)
        nominal_inputs[:, 2] = np.clip(
            nominal_inputs[:, 2],
            0.0,
            effective_v_s_max,
        )
        nominal_states[:, 3] = np.clip(
            nominal_states[:, 3],
            s_lower,
            s_upper,
        )

        if self.state is not None:
            nominal_states[0, :3] = np.asarray(self.state._x, dtype=float)
        nominal_states[0, 3] = float(np.clip(self._current_s0, s_lower, s_upper))

        covariance_batch = self._propagate_covariance_state_batch(
            nominal_inputs,
            nominal_states[:, 3],
        )
        nominal_states[:, 4:8] = covariance_batch
        return nominal_states, nominal_inputs

    def _prepare_and_solve(self, solver, shift_nominal=True):
        if solver is None:
            raise RuntimeError("Solver is not initialized.")

        if self.state is not None:
            self.update_reference_path_params(self.reference_path_data, self.state)
        s_lower, s_upper = self._get_current_s_bounds(self.param)
        self._current_s0 = float(np.clip(self._current_s0, s_lower, s_upper))

        z_bar, u_bar = self._build_shifted_nominal(
            solver,
            s_lower,
            s_upper,
            shift_nominal=shift_nominal,
        )
        try:
            solver.set(0, "lbx", z_bar[0])
            solver.set(0, "ubx", z_bar[0])
        except Exception:
            print("Warning: stage-0 full-state bounds update failed.")

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
            print("Warning: failed to update augmented-state bounds.")

        for i in range(self.N):
            solver.set(i, "x", z_bar[i])
            solver.set(i, "u", u_bar[i])
        solver.set(self.N, "x", z_bar[-1])

        if self.state is not None:
            measured_pose = np.asarray(self.state._x, dtype=float)
            if not np.allclose(z_bar[0, :3], measured_pose, rtol=0.0, atol=1e-12):
                raise AssertionError("shifted nominal stage 0 must match the measured pose")
        else:
            measured_pose = z_bar[0, :3]
        exact_current_phi, _ = self._compute_exact_psdf(measured_pose)

        constraint_data = self._compute_constraint_affine_data(z_bar)
        guard_affine = np.column_stack(
            (constraint_data["guard_A"][:, :3], constraint_data["guard_c"])
        )
        mf_affine = np.column_stack(
            (constraint_data["mf_A"], constraint_data["mf_c"])
        )

        self._last_exact_current_phi = float(exact_current_phi)
        self._last_guard_A = constraint_data["guard_A"].copy()
        self._last_guard_c = constraint_data["guard_c"].copy()
        self._last_mf_A = constraint_data["mf_A"].copy()
        self._last_mf_c = constraint_data["mf_c"].copy()
        self._last_mf_A_raw = constraint_data["mf_A_raw"].copy()
        self._last_mf_c_raw = constraint_data["mf_c_raw"].copy()
        self._last_mf_valid = constraint_data["mf_valid"].copy()
        self._last_mf_mask = constraint_data["mf_mask"].copy()
        self._last_row_mf_enabled = bool(constraint_data["row_mf_enabled"])
        self._last_nominal_psdf_clearances = constraint_data["phi"].copy()

        # No safety row exists at stage 0; mirror the actual trivial p values
        # in the cached QP-row data and mask used by diagnostics.
        self._last_mf_A[0] = 0.0
        self._last_mf_c[0] = constraint_data["epsilon"][0]
        self._last_mf_mask[0] = False

        # Stage 0 has a fixed state and deliberately no con_h_expr_0.  Keep its
        # safety parameter slots trivially feasible for a uniform p layout.
        stage_path_param = self._build_stage_path_parameter(z_bar[0, 3])
        guard_affine_0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        mf_affine_0 = np.zeros((self.mf_param_dim,), dtype=float)
        mf_affine_0[-1] = constraint_data["epsilon"][0]
        solver.set(
            0,
            "p",
            self._build_stage_param(
                stage_path_param,
                guard_affine_0,
                mf_affine_0,
            ),
        )

        for i in range(1, self.N):
            stage_path_param = self._build_stage_path_parameter(z_bar[i, 3])
            solver.set(
                i,
                "p",
                self._build_stage_param(
                    stage_path_param,
                    guard_affine[i],
                    mf_affine[i],
                ),
            )
        terminal_path_param = self._build_stage_path_parameter(z_bar[-1, 3])
        solver.set(
            self.N,
            "p",
            self._build_stage_param(
                terminal_path_param,
                guard_affine[-1],
                mf_affine[-1],
            ),
        )
        return solver.solve()

    def recovery_infeasible(self, solver_status):
        self._last_backup_status = None
        if self.solver is None or self.state is None:
            return self.solver, solver_status, "safe_stop"

        dynamics = self._system_dynamics
        if dynamics is None or not hasattr(dynamics, "nominal_safe_controller"):
            print("Warning: nominal_safe_controller is unavailable; keeping failed solver iterate.")
            return self.solver, solver_status, "sqp_rti"

        print(f"SQP_RTI solver failed with status {solver_status}. Starting infeasible recovery.")

        if self.backup_solver is not None:
            cache_names = (
                "_last_exact_current_phi",
                "_last_guard_A",
                "_last_guard_c",
                "_last_mf_A",
                "_last_mf_c",
                "_last_mf_A_raw",
                "_last_mf_c_raw",
                "_last_mf_valid",
                "_last_mf_mask",
                "_last_row_mf_enabled",
                "_last_nominal_psdf_clearances",
            )
            main_constraint_cache = {
                name: copy.deepcopy(getattr(self, name)) for name in cache_names
            }
            try:
                self.backup_solver.reset(reset_qp_solver_mem=1)
                try:
                    self._copy_primal_guess(self.solver, self.backup_solver)
                except Exception:
                    if self._system is not None:
                        self.add_warm_start(self.param, self._system, solver=self.backup_solver)
                backup_status = self._prepare_and_solve(
                    self.backup_solver,
                    shift_nominal=False,
                )
                self._last_backup_status = int(backup_status)
                if backup_status == 0:
                    try:
                        self._copy_primal_guess(self.backup_solver, self.solver)
                    except Exception:
                        pass
                    self._last_solve_mode = "backup_feasible_qp"
                    print("Recovered with backup SQP_WITH_FEASIBLE_QP solver.")
                    return self.backup_solver, 0, "backup_feasible_qp"
                for name, value in main_constraint_cache.items():
                    setattr(self, name, value)
                try:
                    self._copy_primal_guess(self.backup_solver, self.solver)
                except Exception:
                    pass
                self._last_solve_mode = "backup_feasible_qp"
                print(
                    f"Backup SQP_WITH_FEASIBLE_QP solver returned status {backup_status}; "
                    "falling back to safe stop."
                )
            except Exception as e:
                for name, value in main_constraint_cache.items():
                    setattr(self, name, value)
                print(f"Warning: backup solver recovery failed: {e}")
        else:
            print("Backup solver is unavailable; falling back to safe stop.")

        dt = float(self.param.tf) / float(max(self.N, 1))
        s_curr = self._clamp_s(self._current_s0)
        x_curr = np.array(self.state._x, dtype=float).copy()
        u_prev = np.asarray(getattr(self.state, "_u", np.zeros((2,), dtype=float)), dtype=float).reshape(-1)
        v_last = float(u_prev[0]) if u_prev.size > 0 else 0.0

        try:
            self.solver.reset(reset_qp_solver_mem=1)
            safe_states = np.zeros((self.N + 1, self.nx), dtype=float)
            safe_inputs = np.zeros((self.N, self.nu), dtype=float)
            safe_states[0, :4] = np.array(
                [x_curr[0], x_curr[1], x_curr[2], s_curr],
                dtype=float,
            )

            for i in range(self.N):
                x_next, u_nom = dynamics.nominal_safe_controller(x_curr, dt, v_last, -1.0, 1.0)
                u_nom = np.asarray(u_nom, dtype=float).reshape(-1)
                v_nom = float(u_nom[0]) if u_nom.size > 0 else 0.0
                omega_nom = float(u_nom[1]) if u_nom.size > 1 else 0.0
                v_s_nom = 0.0
                s_next = s_curr

                safe_inputs[i] = np.array(
                    [v_nom, omega_nom, v_s_nom],
                    dtype=float,
                )
                safe_states[i + 1, :4] = np.array(
                    [x_next[0], x_next[1], x_next[2], s_next],
                    dtype=float,
                )

                x_curr = np.asarray(x_next, dtype=float).copy()
                v_last = v_nom
                s_curr = s_next

            safe_states[:, 4:8] = self._propagate_covariance_state_batch(
                safe_inputs,
                safe_states[:, 3],
            )
            for i in range(self.N):
                self.solver.set(i, "x", safe_states[i])
                self.solver.set(i, "u", safe_inputs[i])
            self.solver.set(self.N, "x", safe_states[-1])

            self._last_solve_mode = "safe_stop"
            return self.solver, solver_status, "safe_stop"
        except Exception as e:
            print(f"Warning: infeasible recovery failed: {e}")
            self._last_solve_mode = "safe_stop"
            return self.solver, solver_status, "safe_stop"

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

    def setup(self, param, system, reference_trajectory, obstacles):
        self.param = param
        self.set_state(system._state)
        self._system = system
        self._system_dynamics = system._dynamics
        self.set_reference_trajectory(reference_trajectory)
        use_obstacle_constraint = getattr(param, "use_obstacle_constraint", True)

        if not self._is_initialized:
            print("Setting up RMPCC-MF optimizer...")
            if use_obstacle_constraint:
                self.initialize_augmented_psdf(
                    system,
                    obstacles,
                    E_max=100,
                    K_max=20,
                    device=param.augmented_psdf_device,
                )
            else:
                self.psdf_wrapper = None
                print("RMPCC-MF obstacle constraint disabled: skipping augmented PSDF initialization.")
            self.setup_ocp(param, reference_trajectory)
            # Compile the same two safety rows in every configuration.  Row G
            # remains hard; disabled obstacle processing or Row M uses trivial
            # stage parameters instead of changing the generated OCP dimensions.
            self.add_obstacle_avoidance_constraint(param, system, obstacles)
            self.solver = self.create_solver(
                code_export_directory=self.code_export_directory,
            )
            self.add_warm_start(param, system, solver=self.solver)
            self.backup_solver = None
            if getattr(param, "enable_backup_solver", False):
                main_ocp = self.ocp
                try:
                    backup_param = copy.deepcopy(param)
                    backup_param.nlp_solver_type = "SQP_WITH_FEASIBLE_QP"
                    backup_param.hpipm_mode = "ROBUST"
                    backup_param.regularize_method = "PROJECT"
                    self.setup_ocp(backup_param, reference_trajectory)
                    self.add_obstacle_avoidance_constraint(
                        backup_param,
                        system,
                        obstacles,
                    )
                    self.backup_solver = self.create_solver(
                        json_filename=self.backup_json_filename,
                        code_export_directory=self.backup_code_export_directory,
                    )
                    self.add_warm_start(param, system, solver=self.backup_solver)
                except Exception as e:
                    self.backup_solver = None
                    print(f"Warning: failed to initialize backup solver: {e}")
                finally:
                    self.ocp = main_ocp
            else:
                print(
                    "RMPCC-MF backup SQP_WITH_FEASIBLE_QP solver disabled; "
                    "using nominal safe-stop recovery."
                )
            self._is_initialized = True
        else:
            if use_obstacle_constraint:
                self.update_obstacles(obstacles)

    @staticmethod
    def _diagnostic_array_string(values):
        return np.array2string(
            np.asarray(values),
            precision=5,
            suppress_small=False,
            max_line_width=200,
        )

    def _read_lower_slack_trajectory(self, solver):
        lower_slacks = np.full((self.N + 1, 2), np.nan, dtype=float)
        # Keep the public diagnostic layout [Row G, Row M].  Row G is hard and
        # consequently has no solver slack; represent that fixed value as zero.
        lower_slacks[1:, 0] = 0.0
        for i in range(1, self.N + 1):
            try:
                stage_slacks = np.asarray(solver.get(i, "sl"), dtype=float).reshape(-1)
                if stage_slacks.size < 1:
                    continue
                lower_slacks[i, 1] = stage_slacks[0]
            except Exception:
                continue
        return lower_slacks

    def _sample_first_interval_psdf(self, solver, x_sol):
        num_substeps = max(
            int(getattr(self.param, "first_interval_substeps", 10)),
            1,
        )
        pose = np.asarray(x_sol[0, :3], dtype=float).copy()
        poses = [pose.copy()]
        if self.N >= 1:
            physical_input = np.asarray(solver.get(0, "u"), dtype=float)[:2]
            substep_dt = float(self.param.tf) / float(self.N * num_substeps)
            for _ in range(num_substeps):
                pose = self._rollout_terminal_pose(
                    pose,
                    physical_input,
                    substep_dt,
                )
                poses.append(np.asarray(pose, dtype=float).copy())
        poses = np.asarray(poses, dtype=float)
        phi, _ = self._compute_exact_psdf(poses)
        return poses, np.asarray(phi, dtype=float)

    def _collect_solver_residual_diagnostics(self, solver, status, mode):
        diagnostics = {
            "status": int(status),
            "mode": str(mode),
            "backup_status": self._last_backup_status,
            "nlp_residuals": None,
            "qp_status": None,
            "qp_iterations": None,
            "qp_residuals": None,
            "statistics": None,
        }
        try:
            diagnostics["nlp_residuals"] = np.asarray(
                solver.get_residuals(recompute=True),
                dtype=float,
            )
        except TypeError:
            try:
                diagnostics["nlp_residuals"] = np.asarray(
                    solver.get_residuals(),
                    dtype=float,
                )
            except Exception:
                pass
        except Exception:
            pass

        try:
            qp_status = solver.get_stats("qp_stat")
            if qp_status is not None:
                diagnostics["qp_status"] = np.asarray(qp_status, dtype=float)
        except Exception:
            pass
        try:
            qp_iterations = solver.get_stats("qp_iter")
            if qp_iterations is not None:
                diagnostics["qp_iterations"] = np.asarray(
                    qp_iterations,
                    dtype=float,
                )
        except Exception:
            pass

        try:
            statistics = np.asarray(solver.get_stats("statistics"), dtype=float)
            diagnostics["statistics"] = statistics
            if (
                mode == "backup_feasible_qp"
                and statistics.ndim == 2
                and statistics.shape[0] >= 11
                and statistics.shape[1] > 0
            ):
                # SQP_WITH_FEASIBLE_QP reports up to three QP attempts.  It
                # does not expose external QP KKT residuals in this build.
                diagnostics["qp_status"] = statistics[[5, 7, 9], -1].copy()
                diagnostics["qp_iterations"] = statistics[[6, 8, 10], -1].copy()
        except Exception:
            pass
        return diagnostics

    def _build_constraint_diagnostics(self, solver, x_sol, status, mode):
        d_col = float(getattr(self.param, "d_col", self.param.d_min))
        epsilon = self._normalize_epsilon_values(self.N + 1)
        phi_exact, _ = self._compute_exact_psdf(x_sol[:, :3])
        phi_exact = np.asarray(phi_exact, dtype=float)

        guard_affine_residual = (
            np.einsum("ij,ij->i", self._last_guard_A, x_sol)
            + self._last_guard_c
        )
        guard_fresh_residual = phi_exact - d_col
        mf_affine_residual = (
            np.einsum("ij,ij->i", self._last_mf_A, x_sol)
            + self._last_mf_c
        )

        try:
            fresh_data = self._compute_constraint_affine_data(x_sol)
            mf_fresh_raw_residual = (
                np.einsum("ij,ij->i", fresh_data["mf_A_raw"], x_sol)
                + fresh_data["mf_c_raw"]
            )
            fresh_raw_finite = fresh_data["mf_valid"] & (
                np.isfinite(fresh_data["mf_A_raw"]).all(axis=1)
                & np.isfinite(fresh_data["mf_c_raw"])
            )
            mf_fresh_raw_residual[~fresh_raw_finite] = np.nan
            mf_fresh_effective_residual = (
                np.einsum("ij,ij->i", fresh_data["mf_A"], x_sol)
                + fresh_data["mf_c"]
            )
            mf_fresh_mask = fresh_data["mf_mask"].copy()
            mf_fresh_mask[0] = False
        except Exception:
            mf_fresh_raw_residual = np.full((self.N + 1,), np.nan, dtype=float)
            mf_fresh_effective_residual = np.full(
                (self.N + 1,),
                np.nan,
                dtype=float,
            )
            mf_fresh_mask = np.zeros((self.N + 1,), dtype=bool)

        lower_slacks = self._read_lower_slack_trajectory(solver)
        slack_source = "solver"
        if mode == "safe_stop":
            # Safe-stop states/inputs are written manually after a failed QP;
            # any MF sl values in solver memory belong to an older iterate.
            lower_slacks[:, 1] = np.nan
            slack_source = "unavailable_safe_stop"
        row_coefficient_norms = np.column_stack(
            (
                np.linalg.norm(self._last_guard_A, axis=1),
                np.linalg.norm(self._last_mf_A, axis=1),
            )
        )
        required_lower_slacks = np.column_stack(
            (
                np.maximum(-guard_affine_residual, 0.0),
                np.maximum(-mf_affine_residual, 0.0),
            )
        )
        qp_fixed_row_lower_residual = np.column_stack(
            (guard_affine_residual, mf_affine_residual)
        ) + lower_slacks

        # Stage 0 has no safety constraint or slack.  Keep its diagnostic slots
        # explicitly NaN so trajectory indexing cannot imply otherwise.
        for values in (
            guard_affine_residual,
            guard_fresh_residual,
            mf_affine_residual,
            mf_fresh_raw_residual,
            mf_fresh_effective_residual,
        ):
            values[0] = np.nan
        lower_slacks[0] = np.nan
        required_lower_slacks[0] = np.nan
        qp_fixed_row_lower_residual[0] = np.nan
        row_coefficient_norms[0] = np.nan

        first_interval_poses, first_interval_phi = self._sample_first_interval_psdf(
            solver,
            x_sol,
        )
        solver_diagnostics = self._collect_solver_residual_diagnostics(
            solver,
            status,
            mode,
        )
        solver_diagnostics["qp_fixed_row_lower_residual"] = (
            qp_fixed_row_lower_residual
        )
        future_qp_row_residual = qp_fixed_row_lower_residual[1:]
        if np.isfinite(future_qp_row_residual).any():
            solver_diagnostics["qp_fixed_row_violation_inf"] = float(
                np.nanmax(np.maximum(-future_qp_row_residual, 0.0))
            )
        else:
            solver_diagnostics["qp_fixed_row_violation_inf"] = None
        future_phi = phi_exact[1:] if self.N >= 1 else phi_exact
        min_predicted_phi = float(np.min(future_phi))
        min_first_interval_phi = float(np.min(first_interval_phi))

        diagnostics = {
            "row_g_enabled": True,
            "row_mf_enabled": bool(self._last_row_mf_enabled),
            "constraint_row_order": {
                0: "guard/recovery",
                1: "MF",
            },
            "constraint_stages": np.arange(1, self.N + 1, dtype=int),
            "exact_current_psdf": float(self._last_exact_current_phi),
            "exact_current_guard_residual": float(self._last_exact_current_phi - d_col),
            "guard_affine_residual": guard_affine_residual,
            "guard_fresh_exact_psdf_residual": guard_fresh_residual,
            "mf_affine_residual": mf_affine_residual,
            # The effective fresh value respects the same domain mask as Row M.
            # Keep the raw unsigned-feature value separately for debugging.
            "mf_fresh_recomputed_residual": mf_fresh_effective_residual,
            "mf_fresh_raw_residual": mf_fresh_raw_residual,
            "mf_fresh_effective_residual": mf_fresh_effective_residual,
            "guard_lower_slack": lower_slacks[:, 0],
            "mf_lower_slack": lower_slacks[:, 1],
            "lower_slack_source": slack_source,
            "required_lower_slack": required_lower_slacks,
            "mf_mask": self._last_mf_mask.copy(),
            "mf_fresh_mask": mf_fresh_mask,
            "exact_psdf_predicted_nodes": phi_exact,
            "minimum_exact_psdf_predicted_nodes": min_predicted_phi,
            "first_interval_substep_poses": first_interval_poses,
            "first_interval_substep_exact_psdf": first_interval_phi,
            "minimum_exact_psdf_first_interval_substeps": min_first_interval_phi,
            "row_coefficient_norms": row_coefficient_norms,
            "solver": solver_diagnostics,
        }

        self._last_mf_residuals = mf_affine_residual.copy()
        self._last_mf_probability_sums = epsilon - mf_affine_residual
        self._last_local_covariances = self._covariance_state_to_matrix(x_sol[:, 4:8])
        self._last_psdf_clearances = phi_exact.copy()
        self._last_constraint_diagnostics = diagnostics
        return diagnostics

    def _print_constraint_diagnostics(self, diagnostics):
        future = slice(1, None)
        solver_diagnostics = diagnostics["solver"]
        print("RMPCC-MF constraint order: 0=guard/recovery, 1=MF")
        print("RMPCC-MF Row G: enabled (always hard)")
        print(
            "RMPCC-MF Row M: "
            f"{'enabled' if diagnostics['row_mf_enabled'] else 'disabled (trivially masked)'}"
        )
        print(
            "RMPCC-MF exact current PSDF: "
            f"{diagnostics['exact_current_psdf']:.8g}"
        )
        print(
            "RMPCC-MF guard residuals [affine | fresh exact]: "
            f"{self._diagnostic_array_string(diagnostics['guard_affine_residual'][future])} | "
            f"{self._diagnostic_array_string(diagnostics['guard_fresh_exact_psdf_residual'][future])}"
        )
        print(
            "RMPCC-MF MF residuals [affine | fresh recomputed]: "
            f"{self._diagnostic_array_string(diagnostics['mf_affine_residual'][future])} | "
            f"{self._diagnostic_array_string(diagnostics['mf_fresh_recomputed_residual'][future])}"
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
        print(
            "RMPCC-MF per-stage MF mask: "
            f"{self._diagnostic_array_string(diagnostics['mf_mask'][future].astype(int))}"
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
            f"{self._diagnostic_array_string(solver_diagnostics['qp_fixed_row_lower_residual'][future])}"
        )
        print(
            "RMPCC-MF solver/QP diagnostics "
            f"[NLP(stat,eq,ineq,comp) | QP KKT(if available) | status | iter]: "
            f"{self._diagnostic_array_string(solver_diagnostics['nlp_residuals'])} | "
            f"{self._diagnostic_array_string(solver_diagnostics['qp_residuals'])} | "
            f"{self._diagnostic_array_string(solver_diagnostics['qp_status'])} | "
            f"{self._diagnostic_array_string(solver_diagnostics['qp_iterations'])}"
        )

    def solve_nlp(self):
        if self.solver is None:
            raise RuntimeError("RMPCC-MF solver is not initialized. Call setup() first.")

        start = time.time()
        self._last_backup_status = None
        status = self._prepare_and_solve(self.solver)
        active_solver = self.solver
        mode = "sqp_rti"

        if status != 0:
            print(f"Acados solver failed with status {status}")
            if status == 2:
                print("Acados reached the maximum SQP iterations before meeting the configured tolerance.")
            try:
                residuals = np.asarray(self.solver.get_residuals(), dtype=float)
                print(f"Acados residuals [stat, eq, ineq, comp]: {residuals}")
            except Exception:
                pass
            try:
                sqp_iter = int(self.solver.get_stats("sqp_iter"))
                print(f"Acados SQP iterations: {sqp_iter}")
            except Exception:
                pass
            active_solver, status, mode = self.recovery_infeasible(status)
            if mode == "backup_feasible_qp":
                self._backup_feasible_qp_success_count += 1
            elif mode == "safe_stop":
                self._safe_stop_count += 1
            if mode == "safe_stop":
                print("Applied nominal_safe_controller safe-stop plan after infeasible RMPCC-MF solve.")
        if status == 0:
            try:
                if self.N >= 1:
                    self._prev_predicted_s = self._clamp_s(float(active_solver.get(1, "x")[3]))
                else:
                    self._prev_predicted_s = self._clamp_s(float(active_solver.get(0, "x")[3]))
                self._has_prev_predicted_s = True
            except Exception:
                self._prev_predicted_s = self._clamp_s(self._current_s0)
                self._has_prev_predicted_s = False
        else:
            self._prev_predicted_s = self._clamp_s(self._current_s0)
            self._has_prev_predicted_s = False

        self._has_shift_source = True

        x_sol = np.stack(
            [np.asarray(active_solver.get(i, "x"), dtype=float) for i in range(self.N + 1)],
            axis=0,
        )
        diagnostics = self._build_constraint_diagnostics(
            active_solver,
            x_sol,
            status,
            mode,
        )

        self._last_solve_mode = mode
        solve_time = time.time() - start
        self.solver_times.append(solve_time)
        if self._debug_mf_enabled():
            self._print_constraint_diagnostics(diagnostics)
            min_residual = float(np.nanmin(self._last_mf_residuals[1:]))
            max_probability_sum = float(
                np.nanmax(self._last_mf_probability_sums[1:])
            )
            min_clearance = float(
                diagnostics["minimum_exact_psdf_predicted_nodes"]
            )
            print(
                f"solver time: {solve_time}, path_s: {self._current_s0}, "
                f"mf_residual_min: {min_residual}, "
                f"mf_probability_sum_max: {max_probability_sum}, "
                f"psdf_clearance_min: {min_clearance}, solve_mode: {mode}"
            )
        else:
            print(f"solver time: {solve_time}, path_s: {self._current_s0}, solve_mode: {mode}")

        return AcadosSolution(active_solver, self.N, self.variables)

    def initialize_augmented_psdf(self, system, obstacles, E_max=100, K_max=20, device="cpu"):
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
        self.update_obstacles(obstacles)


class AcadosSolution:
    def __init__(self, solver, N, variables):
        self.solver = solver
        self.N = N
        self.variables = variables

    def value(self, var_expr):
        if var_expr == "x" or (isinstance(var_expr, str) and var_expr == "x"):
            return np.stack([self.solver.get(i, "x") for i in range(self.N + 1)], axis=1)
        if var_expr == "u" or (isinstance(var_expr, str) and var_expr == "u"):
            return np.stack([self.solver.get(i, "u") for i in range(self.N)], axis=1)
        raise NotImplementedError("Only 'x' and 'u' supported in AcadosSolution.value")

    def get_state_trajectory(self):
        return self.value("x")

    def get_input_trajectory(self):
        return self.value("u")

    def get_physical_input_trajectory(self):
        u = self.value("u")
        return u[:2, :]

    def stats(self):
        return {"return_status": "success" if self.solver.status == 0 else "failure"}
