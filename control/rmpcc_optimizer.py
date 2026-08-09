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
        self.use_obstacle_constraint = True
        self.chance_epsilon = 0.20
        self.debug_mf = True
        self.augmented_psdf_device = "cuda" if torch.cuda.is_available() else "cpu"

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
        self.mf_param_dim = 9
        self.path_param_slice = slice(0, 0)
        self.mf_param_slice = slice(0, 0)
        self._path_params = None
        self._prev_predicted_s = 0.0
        self._current_s0 = 0.0
        self._has_prev_predicted_s = False
        self._last_mf_residuals = None
        self._last_mf_probability_sums = None
        self._last_mf_A = None
        self._last_mf_c = None
        self._last_nominal_psdf_clearances = None
        self._last_local_covariances = None
        self._last_psdf_clearances = None
        self._has_shift_source = False

        self.json_filename = "acados_ocp_rmpcc_mf.json"
        self.backup_solver = None
        self.code_export_directory = "c_generated_code_rmpcc_mf"
        self.backup_json_filename = "acados_ocp_rmpcc_mf_backup.json"
        self.backup_code_export_directory = "c_generated_code_rmpcc_mf_backup"
        self._last_solve_mode = "uninitialized"
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
        self.mf_param_dim = 9
        self.path_param_slice = slice(0, 0)
        self.mf_param_slice = slice(0, 0)
        self._path_params = None
        self._prev_predicted_s = 0.0
        self._current_s0 = 0.0
        self._has_prev_predicted_s = False
        self._last_mf_residuals = None
        self._last_mf_probability_sums = None
        self._last_mf_A = None
        self._last_mf_c = None
        self._last_nominal_psdf_clearances = None
        self._last_local_covariances = None
        self._last_psdf_clearances = None
        self._has_shift_source = False
        self._last_solve_mode = "uninitialized"

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
        p_mf = ca.MX.sym("mf_affine", self.mf_param_dim)

        self.path_param_slice = slice(0, self.path_param_dim)
        self.mf_param_slice = slice(
            self.path_param_dim,
            self.path_param_dim + self.mf_param_dim,
        )
        p_all = ca.vertcat(p_ref_t_ref, p_mf)

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

    def _build_stage_param(self, stage_path_param, mf_affine):
        stage_param = np.zeros(
            self.path_param_dim + self.mf_param_dim,
            dtype=float,
        )
        stage_param[self.path_param_slice] = stage_path_param
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

    def _compute_mf_affine_data(self, z_bar):
        z_bar_np = np.asarray(z_bar, dtype=float)
        if z_bar_np.ndim != 2 or z_bar_np.shape[1] != 8:
            raise ValueError("z_bar must have shape (H, 8)")
        if not np.isfinite(z_bar_np).all():
            raise ValueError("z_bar values must be finite")

        num_stages = z_bar_np.shape[0]
        epsilon = float(getattr(self.param, "chance_epsilon", 0.20))
        if not np.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
            raise ValueError("chance_epsilon must lie strictly between 0 and 1")

        if self.psdf_wrapper is None or not getattr(
            self.param,
            "use_obstacle_constraint",
            True,
        ):
            phi = np.full((num_stages,), 1000.0, dtype=float)
            gradient = np.zeros((num_stages, 3), dtype=float)
            A_mf = np.zeros((num_stages, 8), dtype=float)
            c_mf = np.full((num_stages,), epsilon, dtype=float)
            return phi, gradient, A_mf, c_mf

        z_bar_t = torch.as_tensor(
            z_bar_np,
            dtype=self.psdf_wrapper.A.dtype,
            device=self.psdf_wrapper.device,
        )
        phi_t, gradient_t, A_mf_t, c_mf_t = self.psdf_wrapper.forward_mf(
            z_bar_t,
            epsilon,
            float(self.param.d_min),
        )
        outputs = (phi_t, gradient_t, A_mf_t, c_mf_t)
        if not all(torch.isfinite(output).all() for output in outputs):
            raise ValueError("augmented PSDF returned non-finite MF coefficients")
        return tuple(output.detach().cpu().numpy() for output in outputs)

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
        self.ocp.dims.np = self.path_param_dim + self.mf_param_dim
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
        if not getattr(param, "use_obstacle_constraint", True):
            print("Obstacle avoidance constraint is disabled by use_obstacle_constraint=False")
            return

        self.update_obstacles(obstacles_geo)

        if self.psdf_wrapper is None or self.ocp is None:
            print("Warning: augmented PSDF not initialized; skipping MF constraint")
            return

        x = self.ocp.model.x
        mf_affine = self.ocp.model.p[self.mf_param_slice]
        constraint_expr = ca.dot(mf_affine[:8], x) + mf_affine[8]

        self.ocp.constraints.constr_type = "BGH"
        self.ocp.constraints.constr_type_e = "BGH"
        self.ocp.dims.nh = 1
        self.ocp.dims.nh_e = 1
        self.ocp.model.con_h_expr = constraint_expr
        self.ocp.model.con_h_expr_e = constraint_expr
        self.ocp.constraints.lh = np.array([0.0], dtype=float)
        self.ocp.constraints.uh = np.array([1e8], dtype=float)
        self.ocp.constraints.lh_e = np.array([0.0], dtype=float)
        self.ocp.constraints.uh_e = np.array([1e8], dtype=float)
        print("Added one HARD Boole MF affine constraint at path and terminal stages")

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

        phi_bar, _, A_mf, c_mf = self._compute_mf_affine_data(z_bar)
        mf_affine = np.concatenate((A_mf, c_mf[:, None]), axis=1)
        self._last_mf_A = A_mf.copy()
        self._last_mf_c = c_mf.copy()
        self._last_nominal_psdf_clearances = phi_bar.copy()

        for i in range(self.N):
            stage_path_param = self._build_stage_path_parameter(z_bar[i, 3])
            solver.set(
                i,
                "p",
                self._build_stage_param(stage_path_param, mf_affine[i]),
            )
        terminal_path_param = self._build_stage_path_parameter(z_bar[-1, 3])
        solver.set(
            self.N,
            "p",
            self._build_stage_param(terminal_path_param, mf_affine[-1]),
        )
        return solver.solve()

    def recovery_infeasible(self, fast_status):
        if self.solver is None or self.state is None:
            return self.solver, fast_status, "safe_stop"

        dynamics = self._system_dynamics
        if dynamics is None or not hasattr(dynamics, "nominal_safe_controller"):
            print("Warning: nominal_safe_controller is unavailable; keeping failed solver iterate.")
            return self.solver, fast_status, "fast"

        print(f"Fast SQP_RTI solver failed with status {fast_status}. Starting infeasible recovery.")

        if self.backup_solver is not None:
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
                if backup_status == 0:
                    try:
                        self._copy_primal_guess(self.backup_solver, self.solver)
                    except Exception:
                        pass
                    self._last_solve_mode = "backup_feasible_qp"
                    print("Recovered with backup SQP_WITH_FEASIBLE_QP solver.")
                    return self.backup_solver, 0, "backup_feasible_qp"
                try:
                    self._copy_primal_guess(self.backup_solver, self.solver)
                except Exception:
                    pass
                self._last_solve_mode = "backup_feasible_qp"
                print(
                    f"Backup SQP_WITH_FEASIBLE_QP solver returned status {backup_status}; "
                    "falling back to safe stop."
                )
                fast_status = backup_status
            except Exception as e:
                print(f"Warning: backup solver recovery failed: {e}")
        else:
            print("Backup solver is unavailable; falling back to safe stop.")

        dt = float(self.param.tf) / float(max(self.N, 1))
        cov0 = self._build_initial_covariance_state(self.param)
        s_curr = self._clamp_s(self._current_s0)
        x_curr = np.array(self.state._x, dtype=float).copy()
        u_prev = np.asarray(getattr(self.state, "_u", np.zeros((2,), dtype=float)), dtype=float).reshape(-1)
        v_last = float(u_prev[0]) if u_prev.size > 0 else 0.0

        try:
            self.solver.reset(reset_qp_solver_mem=1)
            self.solver.set(
                0,
                "x",
                np.array([x_curr[0], x_curr[1], x_curr[2], s_curr, cov0[0], cov0[1], cov0[2], cov0[3]], dtype=float),
            )

            for i in range(self.N):
                x_next, u_nom = dynamics.nominal_safe_controller(x_curr, dt, v_last, -1.0, 1.0)
                u_nom = np.asarray(u_nom, dtype=float).reshape(-1)
                v_nom = float(u_nom[0]) if u_nom.size > 0 else 0.0
                omega_nom = float(u_nom[1]) if u_nom.size > 1 else 0.0
                v_s_nom = 0.0
                s_next = s_curr

                self.solver.set(i, "u", np.array([v_nom, omega_nom, v_s_nom], dtype=float))
                self.solver.set(
                    i + 1,
                    "x",
                    np.array([x_next[0], x_next[1], x_next[2], s_next, cov0[0], cov0[1], cov0[2], cov0[3]], dtype=float),
                )

                x_curr = np.asarray(x_next, dtype=float).copy()
                v_last = v_nom
                s_curr = s_next

            self._last_solve_mode = "safe_stop"
            return self.solver, fast_status, "safe_stop"
        except Exception as e:
            print(f"Warning: infeasible recovery failed: {e}")
            self._last_solve_mode = "safe_stop"
            return self.solver, fast_status, "safe_stop"

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
            if use_obstacle_constraint:
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
                    if use_obstacle_constraint:
                        self.add_obstacle_avoidance_constraint(backup_param, system, obstacles)
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

    def solve_nlp(self):
        if self.solver is None:
            raise RuntimeError("RMPCC-MF solver is not initialized. Call setup() first.")

        start = time.time()
        status = self._prepare_and_solve(self.solver)
        active_solver = self.solver
        mode = "fast"

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

        if self._debug_mf_enabled():
            x_sol = np.stack([active_solver.get(i, "x") for i in range(self.N + 1)], axis=0)
            if mode == "safe_stop" or self._last_mf_A is None or self._last_mf_c is None:
                phi_sol, _, A_eval, c_eval = self._compute_mf_affine_data(x_sol)
            else:
                A_eval = self._last_mf_A
                c_eval = self._last_mf_c
                if self.psdf_wrapper is None:
                    phi_sol = np.full((x_sol.shape[0],), 1000.0, dtype=float)
                else:
                    pose_t = torch.as_tensor(
                        x_sol[:, :3],
                        dtype=self.psdf_wrapper.A.dtype,
                        device=self.psdf_wrapper.device,
                    )
                    phi_t, _ = self.psdf_wrapper(pose_t)
                    phi_sol = phi_t.detach().cpu().numpy()

            residuals = np.einsum("ij,ij->i", A_eval, x_sol) + c_eval
            epsilon = float(self.param.chance_epsilon)
            self._last_mf_residuals = residuals
            self._last_mf_probability_sums = epsilon - residuals
            self._last_local_covariances = self._covariance_state_to_matrix(x_sol[:, 4:8])
            self._last_psdf_clearances = np.asarray(phi_sol, dtype=float)
        else:
            self._last_mf_residuals = None
            self._last_mf_probability_sums = None
            self._last_local_covariances = None
            self._last_psdf_clearances = None

        self._last_solve_mode = mode
        solve_time = time.time() - start
        self.solver_times.append(solve_time)
        if self._debug_mf_enabled():
            min_residual = float(np.min(self._last_mf_residuals))
            max_probability_sum = float(np.max(self._last_mf_probability_sums))
            min_clearance = float(np.min(self._last_psdf_clearances))
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
