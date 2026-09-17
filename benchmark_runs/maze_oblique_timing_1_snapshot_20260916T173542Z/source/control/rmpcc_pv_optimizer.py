import copy
import os
from statistics import NormalDist
import time

import casadi as ca
import numpy as np
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from control.analytic_psdf_casadi import AnalyticPSDFCasADi
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.geometry_utils import polygon_to_edges
from utils.acados_diagnostics import report_solver_failure
from utils.spline_path import (
    clamp_progress,
    curvature,
    normalize_path_data,
    progress_bounds,
    project_progress,
    stage_parameter,
)


class RMPCCPVOptimizerParam:
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


        # Physical input bounds
        self.vmin, self.vmax = -0.7, 0.7
        self.omegamin, self.omegamax = -1.2, 1.2

        # Safety constraints (PSDF style)
        self.d_min = 0.001
        self.use_obstacle_constraint = True
        self.use_risk_margin = True
        self.use_projected_variance_margin = True
        self.chance_epsilon = 0.20 # 0.05
        self.chance_gamma = float(NormalDist().inv_cdf(1.0 - self.chance_epsilon))
        self.risk_margin_floor = 0.0
        self.risk_margin_cap = 2.0
        # Zero-based inclusive stage window for activating the risk margin.
        self.risk_margin_active_start_step = 0
        self.risk_margin_active_end_step = 10
        self.debug_risk = True
        self.debug_print_nominal_risk_heads = False
        self.debug_print_nominal_risk_head_only_active = True

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

class RMPCCPVOptimizer:
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
        self.ped_model = None
        self._is_initialized = False
        self.device = "cuda"
        self._system_dynamics = None
        self._system = None

        self.path_param_dim = None
        self.risk_tensor_param_dim = 4
        self.risk_margin_scale_param_dim = 1
        self.psdf_param_dim = 0
        self.path_param_slice = slice(0, 0)
        self.risk_tensor_param_slice = slice(0, 0)
        self.risk_margin_scale_param_slice = slice(0, 0)
        self.psdf_param_slice = slice(0, 0)
        self._prev_predicted_s = 0.0
        self._current_s0 = 0.0
        self._has_prev_predicted_s = False
        self._last_risk_margins = None
        self._last_local_covariances = None
        self._last_risk_tensors = None
        self._last_risk_clearances = None

        self.json_filename = "acados_ocp_rmpcc_pv.json"
        self.backup_solver = None
        self.code_export_directory = "c_generated_code_rmpcc_pv"
        self.backup_json_filename = "acados_ocp_rmpcc_pv_backup.json"
        self.backup_code_export_directory = "c_generated_code_rmpcc_pv_backup"
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
            "RMPCC-PV execution summary: "
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
        self.ped_model = None
        self._is_initialized = False

        self.reference_path_data = None
        self._system_dynamics = None
        self._system = None
        self._temp_files = []
        self.path_param_dim = None
        self.risk_tensor_param_dim = 4
        self.risk_margin_scale_param_dim = 1
        self.psdf_param_dim = 0
        self.path_param_slice = slice(0, 0)
        self.risk_tensor_param_slice = slice(0, 0)
        self.risk_margin_scale_param_slice = slice(0, 0)
        self.psdf_param_slice = slice(0, 0)
        self._prev_predicted_s = 0.0
        self._current_s0 = 0.0
        self._has_prev_predicted_s = False
        self._last_risk_margins = None
        self._last_local_covariances = None
        self._last_risk_tensors = None
        self._last_risk_clearances = None
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

    def get_last_risk_margin_trajectory(self):
        if self._last_risk_margins is None:
            return None
        return np.asarray(self._last_risk_margins, dtype=float).copy()

    def get_last_risk_margin_at_stage(self, stage_idx):
        if self._last_risk_margins is None:
            return None

        stage_idx = int(stage_idx)
        if stage_idx < 0 or stage_idx >= len(self._last_risk_margins):
            return None

        return float(self._last_risk_margins[stage_idx])

    def get_last_risk_tensor_trajectory(self):
        if self._last_risk_tensors is None:
            return None
        return np.asarray(self._last_risk_tensors, dtype=float).copy()

    def get_risk_margin_active_stage_mask(self, num_stages=None):
        if getattr(self, "param", None) is None:
            return None

        if num_stages is None:
            if self.N is None:
                return None
            num_stages = self.N + 1

        num_stages = max(int(num_stages), 0)
        if num_stages == 0:
            return np.zeros((0,), dtype=bool)

        return self._build_risk_margin_scale_batch(num_stages, self.param) > 0.0

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

    def create_model(self, param, p_psdf_sym=None):
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
        p_risk_tensor = ca.MX.sym("risk_tensor", self.risk_tensor_param_dim)
        p_risk_margin_scale = ca.MX.sym("risk_margin_scale", self.risk_margin_scale_param_dim)

        self.psdf_param_dim = 0 if p_psdf_sym is None else int(p_psdf_sym.shape[0])
        self.path_param_slice = slice(0, self.path_param_dim)
        self.risk_tensor_param_slice = slice(
            self.path_param_dim,
            self.path_param_dim + self.risk_tensor_param_dim,
        )
        self.risk_margin_scale_param_slice = slice(
            self.risk_tensor_param_slice.stop,
            self.risk_tensor_param_slice.stop + self.risk_margin_scale_param_dim,
        )
        self.psdf_param_slice = slice(
            self.risk_margin_scale_param_slice.stop,
            self.risk_margin_scale_param_slice.stop + self.psdf_param_dim,
        )

        if p_psdf_sym is None:
            p_all = ca.vertcat(p_ref_t_ref, p_risk_tensor, p_risk_margin_scale)
        else:
            p_all = ca.vertcat(p_ref_t_ref, p_risk_tensor, p_risk_margin_scale, p_psdf_sym)

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
        model.name = "differential_drive_rmpcc_pv"
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
            getattr(self.param, "tangent_reg_delta", 1e-6),
            getattr(self.param, "fd_eps", 1e-3),
        )

    def _build_stage_param(self, stage_path_param, risk_terms, risk_margin_scale, psdf_stage=None):
        stage_param = np.zeros(
            (
                self.path_param_dim
                + self.risk_tensor_param_dim
                + self.risk_margin_scale_param_dim
                + self.psdf_param_dim
            ),
            dtype=float,
        )
        stage_param[self.path_param_slice] = stage_path_param
        stage_param[self.risk_tensor_param_slice] = risk_terms
        stage_param[self.risk_margin_scale_param_slice] = risk_margin_scale
        if self.psdf_param_dim > 0 and psdf_stage is not None:
            stage_param[self.psdf_param_slice] = psdf_stage
        return stage_param

    def _get_chance_gamma(self, param):
        gamma = getattr(param, "chance_gamma", None)
        if gamma is not None:
            gamma = float(gamma)
            if np.isfinite(gamma) and gamma >= 0.0:
                return gamma

        epsilon = float(np.clip(getattr(param, "chance_epsilon", 0.05), 1e-6, 0.499999))
        return float(NormalDist().inv_cdf(1.0 - epsilon))

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

    def _debug_risk_enabled(self, param=None):
        target_param = self.param if param is None else param
        return bool(getattr(target_param, "debug_risk", False))

    def _build_projected_variance_heads(self, grad_local):
        grad_np = np.asarray(grad_local, dtype=float)
        squeeze = grad_np.ndim == 1
        grad_np = np.atleast_2d(grad_np)
        heads = grad_np[:, :, None] * grad_np[:, None, :]
        return heads[0] if squeeze else heads

    def _world_to_local_gradient(self, poses, grad_world):
        poses_np = np.asarray(poses, dtype=float)
        grad_np = np.asarray(grad_world, dtype=float)
        squeeze = grad_np.ndim == 1

        poses_np = np.atleast_2d(poses_np)
        grad_np = np.atleast_2d(grad_np)

        theta = poses_np[:, 2]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        grad_local = np.zeros_like(grad_np, dtype=float)
        grad_local[:, 0] = cos_theta * grad_np[:, 0] + sin_theta * grad_np[:, 1]
        grad_local[:, 1] = -sin_theta * grad_np[:, 0] + cos_theta * grad_np[:, 1]
        grad_local[:, 2] = grad_np[:, 2]
        return grad_local[0] if squeeze else grad_local

    def _projected_variance_data_from_psdf_params(self, poses, psdf_params_batch):
        params_np = np.asarray(psdf_params_batch, dtype=float)
        squeeze = params_np.ndim == 1
        params_np = np.atleast_2d(params_np)

        if params_np.shape[1] < 7:
            raise ValueError(
                "Projected-variance mode expects PSDF Taylor params [a(3), phi(1), grad(3)] per stage."
            )

        phi_np = params_np[:, 3].copy()
        grad_world = params_np[:, 4:7].copy()
        grad_local = self._world_to_local_gradient(poses, grad_world)
        heads = self._build_projected_variance_heads(grad_local)
        terms = self._extract_projected_variance_terms(grad_local)

        if squeeze:
            return phi_np[0], heads, terms
        return phi_np, heads, terms

    def _extract_projected_variance_terms(self, grad_local):
        grad_np = np.asarray(grad_local, dtype=float)
        squeeze = grad_np.ndim == 1
        grad_np = np.atleast_2d(grad_np)

        terms = np.zeros((grad_np.shape[0], 4), dtype=float)
        terms[:, 0] = grad_np[:, 0] * grad_np[:, 0]
        terms[:, 1] = grad_np[:, 1] * grad_np[:, 1]
        terms[:, 2] = grad_np[:, 2] * grad_np[:, 2]
        terms[:, 3] = grad_np[:, 1] * grad_np[:, 2]
        return terms[0] if squeeze else terms

    def _compute_margin_from_risk_terms(self, covariance_state_batch, risk_tensor_terms_batch, param):
        cov_state = np.asarray(covariance_state_batch, dtype=float)
        risk_terms = np.asarray(risk_tensor_terms_batch, dtype=float)

        risk_var = (
            risk_terms[:, 0] * cov_state[:, 0]
            + risk_terms[:, 1] * cov_state[:, 1]
            + risk_terms[:, 2] * cov_state[:, 2]
            + 2.0 * risk_terms[:, 3] * cov_state[:, 3]
        )
        gamma = self._get_chance_gamma(param)
        floor = float(getattr(param, "risk_margin_floor", 0.0))
        cap = float(getattr(param, "risk_margin_cap", np.inf))
        return np.clip(gamma * np.sqrt(np.maximum(risk_var, 0.0)), floor, cap)

    def _is_risk_margin_active_stage(self, stage_idx, param):
        stage_idx = int(stage_idx)
        start_step = max(int(getattr(param, "risk_margin_active_start_step", 0)), 0)
        end_step = getattr(param, "risk_margin_active_end_step", None)

        if stage_idx < start_step:
            return False
        if end_step is None:
            return True

        end_step = int(end_step)
        if end_step < start_step:
            return False
        return stage_idx <= end_step

    def _build_risk_margin_scale_batch(self, num_stages, param):
        return np.array(
            [1.0 if self._is_risk_margin_active_stage(i, param) else 0.0 for i in range(num_stages)],
            dtype=float,
        )

    def _log_stage_risk_margins(self, risk_margin_batch, param):
        if not self._debug_risk_enabled(param):
            return

        margin_batch = np.asarray(risk_margin_batch, dtype=float).reshape(-1)
        if margin_batch.size == 0:
            print("risk_margin_stages(active): []")
            return

        scale_batch = self._build_risk_margin_scale_batch(margin_batch.size, param)
        stage_indices = np.flatnonzero(scale_batch > 0.0)

        if stage_indices.size == 0:
            print("risk_margin_stages(active): []")
            return

        stage_entries = [f"{int(i)}:{float(margin_batch[i]):.6f}" for i in stage_indices]
        print(f"risk_margin_stages(active): [{', '.join(stage_entries)}]")

    def _log_stage_risk_heads(self, risk_tensor_batch, param, source_label="nominal"):
        if not getattr(param, "debug_print_nominal_risk_heads", False):
            return

        risk_tensor_batch = np.asarray(risk_tensor_batch, dtype=float)
        if risk_tensor_batch.size == 0:
            print(f"{source_label}_risk_head_stages: []")
            return

        if risk_tensor_batch.ndim == 2:
            risk_tensor_batch = risk_tensor_batch[None, :, :]

        scale_batch = self._build_risk_margin_scale_batch(risk_tensor_batch.shape[0], param)
        only_active = bool(getattr(param, "debug_print_nominal_risk_head_only_active", True))
        if only_active:
            stage_indices = np.flatnonzero(scale_batch > 0.0)
        else:
            stage_indices = np.arange(risk_tensor_batch.shape[0])

        label = "active" if only_active else "all"
        if stage_indices.size == 0:
            print(f"{source_label}_risk_head_stages({label}): []")
            return

        def _format_matrix(matrix):
            rows = [", ".join(f"{float(value):.6f}" for value in row) for row in matrix]
            return "[" + " | ".join(f"[{row}]" for row in rows) + "]"

        stage_entries = [f"{int(i)}:{_format_matrix(risk_tensor_batch[i])}" for i in stage_indices]
        print(f"{source_label}_risk_head_stages({label}): [{', '.join(stage_entries)}]")

    def _compute_stage_risk_data(self, poses, psdf_params_batch=None, risk_tensor_batch=None):
        num_stages = poses.shape[0]
        zero_phi = np.zeros((num_stages,), dtype=float)
        zero_R = np.zeros((num_stages, 3, 3), dtype=float)
        zero_terms = np.zeros((num_stages, 4), dtype=float)
        if self.psdf_wrapper is None or not getattr(self.param, "use_obstacle_constraint", True):
            return zero_phi, zero_R, zero_terms

        if psdf_params_batch is None and self.ped_model is not None and self.psdf_param_dim > 0:
            psdf_params_batch = self.ped_model.get_params(poses)

        if psdf_params_batch is not None:
            phi_np, R_np, risk_terms = self._projected_variance_data_from_psdf_params(
                poses,
                psdf_params_batch,
            )
            if not getattr(self.param, "use_risk_margin", True):
                return phi_np, R_np, zero_terms
            return phi_np, R_np, risk_terms

        poses_t = torch.as_tensor(
            poses,
            dtype=self.psdf_wrapper.A.dtype,
            device=self.psdf_wrapper.device,
        )
        phi_t, grad_world_t = self.psdf_wrapper(poses_t)
        phi_np = phi_t.detach().cpu().numpy().reshape(-1)
        grad_local_np = self._world_to_local_gradient(
            poses,
            grad_world_t.detach().cpu().numpy(),
        )
        R_np = self._build_projected_variance_heads(grad_local_np)
        if not getattr(self.param, "use_risk_margin", True):
            return phi_np, R_np, zero_terms

        risk_terms = self._extract_projected_variance_terms(grad_local_np)
        return phi_np, R_np, risk_terms

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
            kappa_i = (
                0.0
                if self.reference_path_data is None
                else curvature(self.reference_path_data, stage_s_values[i])
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

    def _compute_risk_margin_batch(self, poses, inputs, stage_s_values):
        num_stages = poses.shape[0]
        zero_margin = np.zeros((num_stages,), dtype=float)

        if (
            self.psdf_wrapper is None
            or not getattr(self.param, "use_obstacle_constraint", True)
            or not getattr(self.param, "use_risk_margin", True)
        ):
            return zero_margin

        _, _, risk_terms = self._compute_stage_risk_data(poses)
        covariance_batch = self._propagate_covariance_state_batch(inputs, stage_s_values)
        margin_batch = self._compute_margin_from_risk_terms(covariance_batch, risk_terms, self.param)
        margin_batch *= self._build_risk_margin_scale_batch(num_stages, self.param)
        return margin_batch

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

    def _clamp_s(self, s_value):
        return clamp_progress(self.reference_path_data, s_value)

    def _project_s_with_line_search(self, position_xy, param):
        previous_s = self._prev_predicted_s if self._has_prev_predicted_s else None
        return project_progress(
            self.reference_path_data,
            position_xy,
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
            self.reference_path_data = normalize_path_data(reference_path_data)
        if self.reference_path_data is None:
            return None

        position_xy = np.asarray(state._x[:2], dtype=float)
        self._current_s0 = self._project_s_with_line_search(position_xy, self.param)
        self._current_s0 = self._clamp_s(self._current_s0)
        if self._has_prev_predicted_s:
            backtrack_tol = max(float(getattr(self.param, "s_backtrack_tolerance", 0.03)), 0.0)
            s_floor = self._clamp_s(self._prev_predicted_s - backtrack_tol)
            self._current_s0 = max(self._current_s0, s_floor)
        return self._current_s0

    def set_reference_trajectory(self, reference_path_data):
        if reference_path_data is None:
            return
        self.reference_path_data = normalize_path_data(reference_path_data)

    def setup_ocp(self, param, reference_path_data):
        self.ocp = AcadosOcp()
        p_psdf_sym = None
        if self.ped_model is not None:
            # PSDF bridges expose symbolic first-order Taylor parameters.
            _pose_seed = ca.MX.sym("pose_seed", 3)
            _ = self.ped_model(_pose_seed)
            p_psdf_sym = self.ped_model.get_sym_params()
        model = self.create_model(param, p_psdf_sym=p_psdf_sym)
        self.ocp.model = model

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        N = param.horizon
        self.nx = nx
        self.nu = nu
        self.N = N
        self.ocp.dims.N = N
        self.ocp.dims.np = (
            self.path_param_dim
            + self.risk_tensor_param_dim
            + self.risk_margin_scale_param_dim
            + self.psdf_param_dim
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

        if (
            getattr(getattr(self, "ped_model", None), "requires_external_shared_lib", False)
            and hasattr(self.ped_model, "shared_lib_dir")
            and hasattr(self.ped_model, "name")
        ):
            self.ocp.solver_options.model_external_shared_lib_dir = self.ped_model.shared_lib_dir
            self.ocp.solver_options.model_external_shared_lib_name = self.ped_model.name

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

        if self.ped_model is None or self.ocp is None:
            print("Warning: PED model not initialized, skipping obstacle avoidance constraint")
            return

        x = self.ocp.model.x
        d_min = float(param.d_min)
        if getattr(param, "use_risk_margin", True):
            risk_terms = self.ocp.model.p[self.risk_tensor_param_slice]
            risk_margin_scale = self.ocp.model.p[self.risk_margin_scale_param_slice][0]
            p_f = x[4]
            p_l = x[5]
            p_psi = x[6]
            p_lpsi = x[7]
            risk_var = (
                risk_terms[0] * p_f
                + risk_terms[1] * p_l
                + risk_terms[2] * p_psi
                + 2.0 * risk_terms[3] * p_lpsi
            )
            risk_margin = self._get_chance_gamma(param) * ca.sqrt(ca.fmax(risk_var, 0.0))
            floor = float(getattr(param, "risk_margin_floor", 0.0))
            cap = float(getattr(param, "risk_margin_cap", np.inf))
            if floor > 0.0:
                risk_margin = ca.fmax(risk_margin, floor)
            if np.isfinite(cap):
                risk_margin = ca.fmin(risk_margin, cap)
            risk_margin = risk_margin_scale * risk_margin
        else:
            risk_margin = 0.0

        # The active PSDF bridge injects first-order Taylor parameters through model.p.
        sdf_value = self.ped_model(x[:3])
        constraint_expr = sdf_value - risk_margin - d_min

        self.ocp.constraints.constr_type = "BGH"
        self.ocp.dims.nh = 1
        self.ocp.model.con_h_expr = constraint_expr
        self.ocp.constraints.lh = np.array([0.0], dtype=float)
        self.ocp.constraints.uh = np.array([1e8], dtype=float)
        print(f"Added HARD projected-variance obstacle constraint with d_min = {d_min}")

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

    def _prepare_and_solve(self, solver):
        if solver is None:
            raise RuntimeError("Solver is not initialized.")

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
                print("Warning: stage-0 full-state bounds update failed; keeping previous equality bounds.")
            solver.set(0, "x", x0)

        _, lbx, ubx = self._get_augmented_state_bounds(self.param, s_lower, s_upper)
        try:
            for i in range(1, self.N):
                solver.set(i, "lbx", lbx)
                solver.set(i, "ubx", ubx)
            solver.set(self.N, "lbx", lbx)
            solver.set(self.N, "ubx", ubx)
        except Exception:
            print("Warning: failed to update per-stage augmented-state bounds in solver.")

        try:
            effective_v_s_max = self._get_effective_v_s_max(self.param)
            cov_floor = max(float(getattr(self.param, "risk_cov_jitter", 1e-9)), 0.0)
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

        x_guess = np.stack([solver.get(i, "x") for i in range(self.N + 1)], axis=0)

        psdf_params_batch = None
        if self.ped_model is not None and self.psdf_param_dim > 0:
            psdf_params_batch = self.ped_model.get_params(x_guess[:, :3])

        stage_s_values = self._predict_stage_s_values(self._current_s0, s_lower, s_upper)
        _, risk_tensor_batch, risk_terms_batch = self._compute_stage_risk_data(
            x_guess[:, :3],
            psdf_params_batch=psdf_params_batch,
        )
        self._log_stage_risk_heads(risk_tensor_batch, self.param, source_label="nominal")
        risk_margin_scale_batch = self._build_risk_margin_scale_batch(self.N + 1, self.param)
        for i in range(self.N):
            stage_path_param = self._build_stage_path_parameter(stage_s_values[i])
            psdf_stage = None if psdf_params_batch is None else psdf_params_batch[i]
            stage_param = self._build_stage_param(
                stage_path_param,
                risk_terms_batch[i],
                risk_margin_scale_batch[i],
                psdf_stage=psdf_stage,
            )
            solver.set(i, "p", stage_param)
        terminal_path_param = self._build_stage_path_parameter(stage_s_values[self.N])
        psdf_terminal = None if psdf_params_batch is None else psdf_params_batch[-1]
        terminal_stage_param = self._build_stage_param(
            terminal_path_param,
            risk_terms_batch[-1],
            risk_margin_scale_batch[-1],
            psdf_stage=psdf_terminal,
        )
        solver.set(self.N, "p", terminal_stage_param)

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
                backup_status = self._prepare_and_solve(self.backup_solver)
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
            print("Setting up RMPCC-PV optimizer...")
            if use_obstacle_constraint:
                self.initialize_ped_model(system, obstacles, E_max=100, K_max=20, device="cpu")
            else:
                self.psdf_wrapper = None
                self.ped_model = None
                print("RMPCC-PV obstacle constraint disabled: skipping PSDF model initialization.")
            self.setup_ocp(param, reference_trajectory)
            if use_obstacle_constraint:
                self.add_obstacle_avoidance_constraint(param, system, obstacles)
            self.solver = self.create_solver(
                code_export_directory=self.code_export_directory,
            )
            self.add_warm_start(param, system, solver=self.solver)
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
            self._is_initialized = True
        else:
            if use_obstacle_constraint:
                self.update_obstacles(obstacles)

    def solve_nlp(self):
        if self.solver is None:
            raise RuntimeError("RMPCC-PV solver is not initialized. Call setup() first.")

        start = time.time()
        status = self._prepare_and_solve(self.solver)
        active_solver = self.solver
        mode = "fast"

        if status != 0:
            report_solver_failure(
                self.solver,
                status,
                max_iteration_message=(
                    "Acados reached the maximum SQP iterations before meeting "
                    "the configured tolerance."
                ),
            )
            active_solver, status, mode = self.recovery_infeasible(status)
            if mode == "backup_feasible_qp":
                self._backup_feasible_qp_success_count += 1
            elif mode == "safe_stop":
                self._safe_stop_count += 1
            if mode == "safe_stop":
                print("Applied nominal_safe_controller safe-stop plan after infeasible RMPCC-PV solve.")
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

        if self._debug_risk_enabled():
            x_sol = np.stack([active_solver.get(i, "x") for i in range(self.N + 1)], axis=0)
            psdf_params_sol = None
            if self.ped_model is not None and self.psdf_param_dim > 0:
                psdf_params_sol = self.ped_model.get_params(x_sol[:, :3])
            phi_sol, R_sol, risk_terms_sol = self._compute_stage_risk_data(
                x_sol[:, :3],
                psdf_params_batch=psdf_params_sol,
            )
            covariance_sol = x_sol[:, 4:8]
            risk_margin_batch = (
                self._compute_margin_from_risk_terms(covariance_sol, risk_terms_sol, self.param)
                if getattr(self.param, "use_risk_margin", True)
                else np.zeros((x_sol.shape[0],), dtype=float)
            )
            risk_margin_batch *= self._build_risk_margin_scale_batch(x_sol.shape[0], self.param)

            self._last_risk_margins = risk_margin_batch
            self._last_local_covariances = self._covariance_state_to_matrix(covariance_sol)
            self._last_risk_tensors = R_sol
            self._last_risk_clearances = phi_sol
        else:
            self._last_risk_margins = None
            self._last_local_covariances = None
            self._last_risk_tensors = None
            self._last_risk_clearances = None

        self._last_solve_mode = mode
        solve_time = time.time() - start
        self.solver_times.append(solve_time)
        if self._debug_risk_enabled():
            risk_margin_max = 0.0 if self._last_risk_margins is None else float(np.max(self._last_risk_margins))
            print(
                f"solver time: {solve_time}, path_s: {self._current_s0}, "
                f"risk_margin_max: {risk_margin_max}, solve_mode: {mode}"
            )
            self._log_stage_risk_margins(self._last_risk_margins, self.param)
        else:
            print(f"solver time: {solve_time}, path_s: {self._current_s0}, solve_mode: {mode}")

        return AcadosSolution(active_solver, self.N, self.variables)

    def initialize_ped_model(self, system, obstacles, E_max=100, K_max=20, device="cpu"):
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
        self.ped_model = AnalyticPSDFCasADi(
            self.psdf_wrapper,
            device=self.device,
            name="analytic_rmpcc_pv_psdf",
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
