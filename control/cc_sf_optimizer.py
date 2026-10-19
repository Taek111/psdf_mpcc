import copy
import os
from statistics import NormalDist
import time

import casadi as ca
import numpy as np
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from control.analytic_psdf_casadi import AnalyticPSDFCasADi
from control.cc_covariance import (
    covariance_from_log_cholesky,
    initialize_cc_noise_params,
    initial_local_covariance,
    log_cholesky_from_covariance,
    propagate_log_cholesky,
    resolve_robust_horizon,
    symbolic_dd_step,
    symbolic_log_cholesky_transition,
)
from control.mpcc_optimizer import (
    MPCCOptimizer,
    initialize_mpcc_solver_params,
    initialize_mpcc_tracking_params,
)
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from utils.acados_diagnostics import report_solver_failure


class CCSFOptimizerParam:
    def __init__(self):
        initialize_mpcc_tracking_params(self)
        initialize_mpcc_solver_params(self, integrator_type="DISCRETE")
        # None selects min(10, horizon); set horizon for full propagation.
        self.robust_horizon = None
        # Optional estimator covariance in world coordinates; otherwise use sigma_*0.
        self.initial_pose_covariance = None

        # Safety constraints (PSDF style)
        self.d_min = 0.001
        self.use_obstacle_constraint = True
        self.use_risk_margin = True
        self.use_projected_variance_margin = True
        self.chance_epsilon = 0.20 # 0.05
        self.chance_gamma = float(NormalDist().inv_cdf(1.0 - self.chance_epsilon))
        self.risk_margin_floor = 0.0
        self.risk_margin_cap = 2.0
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
        initialize_cc_noise_params(self)
        self.risk_cov_jitter = 1e-9


class CCSFOptimizer(MPCCOptimizer):
    _path_includes_curvature = True

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
        self.covariance_propagation_param_slice = slice(0, 0)
        self._prev_predicted_s = 0.0
        self._current_s0 = 0.0
        self._has_prev_predicted_s = False
        self._last_risk_margins = None
        self._last_local_covariances = None
        self._last_risk_tensors = None
        self._last_risk_clearances = None

        self.json_filename = "acados_ocp_cc_sf.json"
        self.backup_solver = None
        self.code_export_directory = "c_generated_code_cc_sf"
        self.backup_json_filename = "acados_ocp_cc_sf_backup.json"
        self.backup_code_export_directory = "c_generated_code_cc_sf_backup"
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
            "CC-SF execution summary: "
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
        self.covariance_propagation_param_slice = slice(0, 0)
        self._prev_predicted_s = 0.0
        self._current_s0 = 0.0
        self._has_prev_predicted_s = False
        self._last_risk_margins = None
        self._last_local_covariances = None
        self._last_risk_tensors = None
        self._last_risk_clearances = None
        self._last_solve_mode = "uninitialized"

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

        return np.full(
            num_stages, self._risk_margin_enabled(self.param), dtype=bool
        )

    def record_plant_input_applied(self):
        self._plant_input_apply_count += 1

    def create_model(self, param, p_psdf_sym=None):
        dt = float(param.tf) / float(max(param.horizon, 1))
        nx = 8  # [x, y, theta, s, xi_f, xi_l, xi_theta, xi_c]
        nu = 3  # [v, omega, v_s]
        # Stage-varying path parameter:
        # [p_ref_x, p_ref_y, t_ref_x, t_ref_y, s_ref, kappa_ref]
        self.path_param_dim = 6

        x = ca.MX.sym("x", nx)
        u = ca.MX.sym("u", nu)
        p_ref_t_ref = ca.MX.sym("p_ref_t_ref", self.path_param_dim)
        p_risk_tensor = ca.MX.sym("risk_tensor", self.risk_tensor_param_dim)
        p_risk_margin_scale = ca.MX.sym("risk_margin_scale", self.risk_margin_scale_param_dim)
        p_covariance_propagation = ca.MX.sym("covariance_propagation", 1)

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

        self.covariance_propagation_param_slice = slice(
            self.psdf_param_slice.stop,
            self.psdf_param_slice.stop + 1,
        )
        parameter_blocks = [p_ref_t_ref, p_risk_tensor, p_risk_margin_scale]
        if p_psdf_sym is not None:
            parameter_blocks.append(p_psdf_sym)
        p_all = ca.vertcat(*parameter_blocks, p_covariance_propagation)

        v = u[0]
        omega = u[1]
        v_s = u[2]

        kappa_ref = p_ref_t_ref[5]

        xi_next = symbolic_log_cholesky_transition(
            x[4:8], v, omega, kappa_ref, dt, param
        )
        discrete_dynamics = ca.vertcat(
            symbolic_dd_step(x[:3], v, omega, dt),
            x[3] + dt * v_s,
            ca.if_else(p_covariance_propagation[0] > 0.5, xi_next, x[4:8]),
        )

        cost_stage, cost_terminal = self._build_mpcc_cost(
            param, x, u, p_ref_t_ref
        )

        model = AcadosModel()
        model.name = "differential_drive_cc_sf"
        model.x = x
        model.u = u
        model.p = p_all
        model.disc_dyn_expr = discrete_dynamics
        model.cost_expr_ext_cost = cost_stage
        model.cost_expr_ext_cost_e = cost_terminal
        return model

    def _build_stage_param(
        self, stage_path_param, risk_terms, risk_margin_scale, psdf_stage=None, stage_idx=0
    ):
        stage_param = np.zeros(
            (
                self.path_param_dim
                + self.risk_tensor_param_dim
                + self.risk_margin_scale_param_dim
                + self.psdf_param_dim
                + 1
            ),
            dtype=float,
        )
        stage_param[self.path_param_slice] = stage_path_param
        stage_param[self.risk_tensor_param_slice] = risk_terms
        stage_param[self.risk_margin_scale_param_slice] = risk_margin_scale
        if self.psdf_param_dim > 0 and psdf_stage is not None:
            stage_param[self.psdf_param_slice] = psdf_stage
        stage_param[self.covariance_propagation_param_slice] = float(
            stage_idx < resolve_robust_horizon(self.param)
        )
        return stage_param

    def _get_chance_gamma(self, param):
        gamma = getattr(param, "chance_gamma", None)
        if gamma is not None:
            gamma = float(gamma)
            if np.isfinite(gamma) and gamma >= 0.0:
                return gamma

        epsilon = float(np.clip(getattr(param, "chance_epsilon", 0.05), 1e-6, 0.499999))
        return float(NormalDist().inv_cdf(1.0 - epsilon))

    def _build_initial_local_covariance(self, param):
        return initial_local_covariance(param, self.state)

    def _build_initial_covariance_state(self, param):
        return log_cholesky_from_covariance(self._build_initial_local_covariance(param))

    def _covariance_state_to_matrix(self, covariance_state):
        return covariance_from_log_cholesky(covariance_state)

    def _matrix_to_covariance_state(self, Sigma):
        return log_cholesky_from_covariance(Sigma)

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
        xi = np.asarray(covariance_state_batch, dtype=float)
        covariance = self._covariance_state_to_matrix(covariance_state_batch)
        risk_terms = np.asarray(risk_tensor_terms_batch, dtype=float)
        # Risk terms are [q_f^2, q_l^2, q_theta^2, q_l*q_theta]. Recover
        # a common-sign equivalent pair to evaluate ||L.T q|| without cancellation.
        q_l = np.sqrt(np.maximum(risk_terms[:, 1], 0.0))
        q_theta = np.where(risk_terms[:, 3] < 0.0, -1.0, 1.0) * np.sqrt(
            np.maximum(risk_terms[:, 2], 0.0)
        )
        risk_var = (
            risk_terms[:, 0] * covariance[:, 0, 0]
            + (np.exp(xi[:, 1]) * q_l + xi[:, 3] * q_theta) ** 2
            + np.exp(2.0 * xi[:, 2]) * risk_terms[:, 2]
        )
        if not np.all(np.isfinite(risk_var)):
            raise ValueError("Non-finite projected variance in risk margin.")
        gamma = self._get_chance_gamma(param)
        floor = float(getattr(param, "risk_margin_floor", 0.0))
        cap = float(getattr(param, "risk_margin_cap", np.inf))
        return np.clip(gamma * np.sqrt(np.maximum(risk_var, 0.0)), floor, cap)

    def _build_risk_margin_scale_batch(self, num_stages, param):
        # SF has no stage window. Keep the parameter slot for ECS beta values.
        return np.ones((num_stages,), dtype=float)

    def _risk_margin_enabled(self, param):
        return bool(
            getattr(param, "use_obstacle_constraint", True)
            and getattr(param, "use_risk_margin", True)
        )

    def _log_stage_risk_margins(self, risk_margin_batch, param):
        if not self._debug_risk_enabled(param):
            return

        margin_batch = np.asarray(risk_margin_batch, dtype=float).reshape(-1)
        if margin_batch.size == 0:
            print("risk_margin_stages(active): []")
            return

        stage_indices = (
            np.arange(margin_batch.size)
            if self._risk_margin_enabled(param)
            else np.zeros((0,), dtype=int)
        )

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

        only_active = bool(getattr(param, "debug_print_nominal_risk_head_only_active", True))
        if only_active:
            stage_indices = (
                np.arange(risk_tensor_batch.shape[0])
                if self._risk_margin_enabled(param)
                else np.zeros((0,), dtype=int)
            )
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
        """Regenerate the nominal log-Cholesky trajectory, including its frozen tail."""
        num_stages = len(stage_s_values)
        dt = float(self.param.tf) / float(max(self.N, 1))
        robust_horizon = resolve_robust_horizon(self.param)
        covariance_state = self._build_initial_covariance_state(self.param)
        covariance_batch = np.zeros((num_stages, 4), dtype=float)

        for i in range(num_stages):
            covariance_batch[i] = covariance_state
            if i >= inputs.shape[0] or i >= robust_horizon:
                continue
            kappa_i = (
                0.0
                if self.reference_path_data is None
                else float(self._build_stage_path_parameter(stage_s_values[i])[5])
            )
            covariance_state = propagate_log_cholesky(
                covariance_state, inputs[i, 0], inputs[i, 1], kappa_i, dt, self.param
            )
        return covariance_batch

    def _regenerate_covariance_guess(self, solver, stage_s_values=None):
        states = np.stack([solver.get(i, "x") for i in range(self.N + 1)], axis=0)
        inputs = np.stack([solver.get(i, "u") for i in range(self.N)], axis=0)
        if not np.all(np.isfinite(states[:, :4])) or not np.all(np.isfinite(inputs)):
            raise ValueError("Non-finite pose, progress, or control in nominal trajectory.")
        if stage_s_values is None:
            stage_s_values = states[:, 3]
        states[:, 4:8] = self._propagate_covariance_state_batch(inputs, stage_s_values)
        for i, state in enumerate(states):
            solver.set(i, "x", state)
        return states

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

    def _get_augmented_state_bounds(self, param, s_lower, s_upper):
        idxbx = []
        lbx = []
        ubx = []

        if getattr(param, "use_s_state_bounds", True):
            idxbx.append(3)
            lbx.append(float(s_lower))
            ubx.append(float(s_upper))

        return (
            np.array(idxbx, dtype=np.int64),
            np.array(lbx, dtype=float),
            np.array(ubx, dtype=float),
        )

    def setup_ocp(self, param, reference_path_data):
        self.ocp = AcadosOcp()
        p_psdf_sym = None
        if self.ped_model is not None:
            # PSDF bridges expose symbolic first-order Taylor parameters.
            _pose_seed = ca.MX.sym("pose_seed", 3)
            _ = self.ped_model(_pose_seed)
            p_psdf_sym = self.ped_model.get_sym_params()
        model = self.create_model(param, p_psdf_sym=p_psdf_sym)
        self._configure_ocp_dimensions_and_cost(param, model)
        resolve_robust_horizon(param)
        self.ocp.parameter_values[self.covariance_propagation_param_slice] = 1.0
        self._set_input_bounds(param)

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
        self.ocp.constraints.idxbx_0 = np.arange(self.nx, dtype=np.int64)
        self.ocp.constraints.lbx_0 = x0_full.copy()
        self.ocp.constraints.ubx_0 = x0_full.copy()

        idxbx, lbx, ubx = self._get_augmented_state_bounds(param, s_lower, s_upper)
        self.ocp.constraints.idxbx = idxbx
        self.ocp.constraints.lbx = lbx
        self.ocp.constraints.ubx = ubx
        self.ocp.constraints.idxbx_e = idxbx.copy()
        self.ocp.constraints.lbx_e = lbx.copy()
        self.ocp.constraints.ubx_e = ubx.copy()

        # The augmented covariance map is a one-step discrete recursion.
        self._configure_solver_options(param, integrator_type="DISCRETE")

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
            q_l = ca.sqrt(ca.fmax(risk_terms[1], 0.0))
            q_theta = ca.if_else(risk_terms[3] < 0.0, -1.0, 1.0) * ca.sqrt(
                ca.fmax(risk_terms[2], 0.0)
            )
            risk_var = (
                ca.exp(2.0 * x[4]) * risk_terms[0]
                + (ca.exp(x[5]) * q_l + x[7] * q_theta) ** 2
                + ca.exp(2.0 * x[6]) * risk_terms[2]
            )
            # Zero-gradient stages have zero margin and derivative; keep the
            # unselected square-root branch finite without adding variance.
            positive_variance = risk_var > 0.0
            safe_variance = ca.if_else(positive_variance, risk_var, 1.0)
            risk_margin = self._get_chance_gamma(param) * ca.if_else(
                positive_variance, ca.sqrt(safe_variance), 0.0
            )
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
        self.ocp.constraints.constr_type_e = "BGH"
        self.ocp.dims.nh_e = 1
        self.ocp.model.con_h_expr_e = constraint_expr
        self.ocp.constraints.lh_e = np.array([0.0], dtype=float)
        self.ocp.constraints.uh_e = np.array([1e8], dtype=float)
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
            initial_guess = np.zeros((self.nx,), dtype=float)
            if self.state is not None:
                initial_guess[:3] = self.state._x[:3]
            initial_guess[3] = self._current_s0
            initial_guess[4:8] = self._build_initial_covariance_state(param)
            for i in range(self.N):
                solver.set(i, "x", initial_guess.copy())
                solver.set(i, "u", np.zeros((self.nu,), dtype=float))
            solver.set(self.N, "x", initial_guess.copy())

        self._regenerate_covariance_guess(solver)
        for i in range(self.N + 1):
            stage_param = self.ocp.parameter_values.copy()
            stage_param[self.covariance_propagation_param_slice] = float(
                i < resolve_robust_horizon(param)
            )
            solver.set(i, "p", stage_param)

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

        effective_v_s_max = self._get_effective_v_s_max(self.param)
        for i in range(1, self.N + 1):
            xi = np.asarray(solver.get(i, "x"), dtype=float)
            if not np.all(np.isfinite(xi[:4])):
                raise ValueError(f"Non-finite pose or progress guess at stage {i}.")
            xi[3] = float(np.clip(xi[3], s_lower, s_upper))
            solver.set(i, "x", xi)
        for i in range(self.N):
            ui = np.asarray(solver.get(i, "u"), dtype=float)
            if not np.all(np.isfinite(ui)):
                raise ValueError(f"Non-finite input guess at stage {i}.")
            ui[2] = float(np.clip(ui[2], 0.0, effective_v_s_max))
            solver.set(i, "u", ui)

        stage_s_values = self._predict_stage_s_values(
            self._current_s0, s_lower, s_upper, solver
        )
        x_guess = self._regenerate_covariance_guess(solver, stage_s_values)

        psdf_params_batch = None
        if self.ped_model is not None and self.psdf_param_dim > 0:
            psdf_params_batch = self.ped_model.get_params(x_guess[:, :3])

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
                stage_idx=i,
            )
            solver.set(i, "p", stage_param)
        terminal_path_param = self._build_stage_path_parameter(stage_s_values[self.N])
        psdf_terminal = None if psdf_params_batch is None else psdf_params_batch[-1]
        terminal_stage_param = self._build_stage_param(
            terminal_path_param,
            risk_terms_batch[-1],
            risk_margin_scale_batch[-1],
            psdf_stage=psdf_terminal,
            stage_idx=self.N,
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

            self._regenerate_covariance_guess(self.solver)
            self._last_solve_mode = "safe_stop"
            return self.solver, fast_status, "safe_stop"
        except Exception as e:
            print(f"Warning: infeasible recovery failed: {e}")
            self._last_solve_mode = "safe_stop"
            return self.solver, fast_status, "safe_stop"

    def setup(self, param, system, reference_trajectory, obstacles):
        self.param = param
        self.set_state(system._state)
        self._system = system
        self._system_dynamics = system._dynamics
        self.set_reference_trajectory(reference_trajectory)
        use_obstacle_constraint = getattr(param, "use_obstacle_constraint", True)

        if not self._is_initialized:
            print("Setting up CC-SF optimizer...")
            if use_obstacle_constraint:
                self.initialize_ped_model(system, obstacles, E_max=100, K_max=20, device="cuda")
            else:
                self.psdf_wrapper = None
                self.ped_model = None
                print("CC-SF obstacle constraint disabled: skipping PSDF model initialization.")
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
            raise RuntimeError("CC-SF solver is not initialized. Call setup() first.")

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
                print("Applied nominal_safe_controller safe-stop plan after infeasible CC-SF solve.")
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

    def initialize_ped_model(self, system, obstacles, E_max=100, K_max=20, device="cuda"):
        device = self._require_cuda_device(device)
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
            name="analytic_cc_sf_psdf",
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
        """Return rows [x, y, theta, s, xi_f, xi_l, xi_theta, xi_c]."""
        return self.value("x")

    def get_input_trajectory(self):
        return self.value("u")

    def get_physical_input_trajectory(self):
        u = self.value("u")
        return u[:2, :]

    def stats(self):
        return {"return_status": "success" if self.solver.status == 0 else "failure"}
