"""ECS-MPCC with noise-free covariance propagation and polygon PSDF rows.

The ECS support is applied to locally linearized pose clearance, not to an
exact convex configuration-space obstacle. The scalar chance_epsilon is a
budget over future discrete nodes 1..N, including the terminal node. It is
shared over obstacles at each node, and allocated uniformly over time.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import time

import casadi as ca
import numpy as np
import torch
from scipy.stats import chi2

from control.cc_covariance import covariance_from_log_cholesky, resolve_robust_horizon
from control.cc_sf_optimizer import AcadosSolution, CCSFOptimizer
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.geometry_utils import polygon_to_edges
from utils.acados_diagnostics import report_solver_failure


PROCESS_NOISE_FIELDS = (
    "q_f0", "q_l0", "q_psi0", "alpha_f", "alpha_v", "alpha_kappa",
    "beta_v", "beta_kappa", "beta_omega",
)


class CCECSOptimizerParam:
    """MPCC defaults match CC-MF; only process noise is removed from covariance."""

    def __init__(self):
        self.horizon = 20
        self.tf = 0.1 * self.horizon
        self.mat_Qe = np.diag([1.0, 10.0])
        self.mat_Re = np.diag([1.0, 0.01])
        self.q_s = 1.0
        self.terminal_weight = 5.0
        self.v_s_target = 0.8
        self.q_s_ref = 2.0
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
        self.vmin, self.vmax = -0.7, 0.7
        self.omegamin, self.omegamax = -1.2, 1.2

        self.d_min = 0.001
        self.use_obstacle_constraint = True
        self.chance_epsilon = 0.20  # Entire future horizon, not per obstacle/node.
        self.robust_horizon = None  # Same min(10, horizon) default as CC-MF.
        self.initial_pose_covariance = None
        self.sigma_f0 = 0.0002
        self.sigma_l0 = 0.00025
        self.sigma_psi0 = 0.00025
        for name in PROCESS_NOISE_FIELDS:
            setattr(self, name, 0.0)

        self.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.hessian_approx = "GAUSS_NEWTON"
        self.integrator_type = "DISCRETE"
        self.nlp_solver_type = "SQP_RTI"
        self.qp_solver_iter_max = 50
        self.nlp_solver_max_iter = 50
        self.tol = 1e-4
        self.enable_backup_solver = False
        self.augmented_psdf_device = "cuda"


class _ClusterAffinePSDF:
    """Vector affine bridge: one [world gradient(3), intercept] per obstacle."""

    requires_external_shared_lib = False

    def __init__(self, capacity):
        self.capacity = capacity
        self._params = ca.MX.sym("ecs_psdf_planes", 4 * capacity)

    def get_sym_params(self):
        return self._params

    def __call__(self, pose):
        return ca.vertcat(*[
            ca.dot(self._params[4 * i:4 * i + 3], pose)
            + self._params[4 * i + 3]
            for i in range(self.capacity)
        ])


class CCECSOptimizer(CCSFOptimizer):
    """Reuse CC-SF MPCC dynamics/progress/warm start, specialize all safety paths.

    The parameter blocks retain the CC-SF plumbing: path(6), local gradients
    (3*K), ECS beta(1), affine PSDF planes(4*K), covariance propagation flag(1).
    Only nominal geometry is frozen. L(xi) stays symbolic in every hard row.
    """

    CONSTRAINT_LOG_FIELDS = (
        "time", "solve_step", "solver_status", "solve_mode", "horizon_stage",
        "obstacle_index", "active", "epsilon_horizon", "epsilon_stage", "beta",
        "nominal_psdf", "psdf", "fixed_gradient_margin", "reevaluated_margin",
        "fixed_row_residual", "reevaluated_ecs_residual",
        "P_ff", "P_ll", "P_psipsi", "P_lpsi",
    )
    COVARIANCE_LOG_FIELDS = (
        "time", "solve_step", "stage", "v", "omega", "sigma_f", "sigma_l",
        "sigma_psi", "P_ff", "P_ll", "P_psipsi", "P_lpsi",
    )
    CYCLE_LOG_FIELDS = (
        "time", "solve_step", "solve_mode", "solver_status", "main_status",
        "backup_status", "solver_time_s", "path_s", "minimum_psdf",
        "minimum_fixed_row_residual", "minimum_reevaluated_ecs_residual",
    )

    def __init__(self, variables=None, costs=None, dynamics_opt=None):
        super().__init__(variables, costs, dynamics_opt)
        self.json_filename = "acados_ocp_cc_ecs.json"
        self.code_export_directory = "c_generated_code_cc_ecs"
        self.backup_json_filename = "acados_ocp_cc_ecs_backup.json"
        self.backup_code_export_directory = "c_generated_code_cc_ecs_backup"
        self._owned_codegen_directories = set()
        self.prints_compact_runtime_summary = True
        self._clear_ecs_data()
        self._configure_cluster_capacity(1)

    def _clear_ecs_data(self):
        self._constraint_data = None
        self._last_constraint_diagnostics = None
        self._constraint_log_rows = []
        self._covariance_log_rows = []
        self._cycle_log_rows = []
        self._solve_count = 0
        self._cycle_solver_time = 0.0
        self._last_main_status = None
        self._last_backup_status = None
        self._compiled_param_signature = None

    def _configure_cluster_capacity(self, capacity):
        self.obstacle_capacity = max(1, int(capacity))
        self.risk_tensor_param_dim = 3 * self.obstacle_capacity
        self.ped_model = _ClusterAffinePSDF(self.obstacle_capacity)

    def cleanup(self):
        self._print_runtime_summary()
        self._reset_runtime_counters()
        # An unbuilt instance (including geometry-only tests) owns no files.
        for filename in self._temp_files:
            if os.path.isfile(filename):
                os.remove(filename)
        for directory in getattr(self, "_owned_codegen_directories", ()):
            if os.path.isdir(directory):
                shutil.rmtree(directory)
        self._owned_codegen_directories = set()
        self._temp_files = []
        self.ocp = self.solver = self.backup_solver = None
        self.psdf_wrapper = self.ped_model = None
        self.solver_times = []
        self._is_initialized = False
        self.reference_path_data = None
        self._system = self._system_dynamics = None
        self._prev_predicted_s = self._current_s0 = 0.0
        self._has_prev_predicted_s = False
        self._last_solve_mode = "uninitialized"
        self._last_risk_margins = self._last_local_covariances = None
        self._last_risk_tensors = self._last_risk_clearances = None
        self.N = self.nx = self.nu = None
        self._clear_ecs_data()
        self.obstacle_capacity = 1
        self.risk_tensor_param_dim = 3

    def create_solver(self, json_filename=None, code_export_directory=None):
        directory = code_export_directory or self.code_export_directory
        filename = json_filename or self.json_filename
        self._owned_codegen_directories.add(directory)
        if filename not in self._temp_files:
            self._temp_files.append(filename)
        return super().create_solver(filename, directory)

    def _print_runtime_summary(self):
        if self._has_runtime_activity():
            print(
                "CC-ECS execution summary: "
                f"Recovered={self._backup_feasible_qp_success_count}, "
                f"Safe stop={self._safe_stop_count}, "
                f"Plant-applied control inputs={self._plant_input_apply_count}"
            )

    @staticmethod
    def _risk_allocation(param):
        resolve_robust_horizon(param)
        epsilon = np.asarray(param.chance_epsilon)
        if epsilon.ndim != 0 or epsilon.dtype.kind not in "fiu":
            raise ValueError("chance_epsilon must be one scalar horizon risk budget")
        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
            raise ValueError("chance_epsilon must lie strictly between 0 and 1")
        stage_epsilon = epsilon / int(param.horizon)
        # isf avoids cancellation in 1 - epsilon for small risk budgets.
        beta = float(np.sqrt(chi2.isf(stage_epsilon, df=3)))
        if not np.isfinite(beta):
            raise ValueError("chance_epsilon is too small for a finite ECS quantile")
        return stage_epsilon, beta

    @classmethod
    def _validate_param(cls, param):
        cls._risk_allocation(param)
        if not np.isfinite(param.tf) or param.tf <= 0:
            raise ValueError("tf must be finite and positive")
        if not np.isfinite(param.d_min) or param.d_min < 0:
            raise ValueError("d_min must be finite and nonnegative")
        for name in PROCESS_NOISE_FIELDS:
            if float(getattr(param, name, 0.0)) != 0.0:
                raise ValueError(f"CC-ECS removes process noise: {name} must be zero")
        if param.integrator_type != "DISCRETE":
            raise ValueError("CC-ECS covariance propagation requires DISCRETE dynamics")

    @staticmethod
    def _build_signature(param):
        # These quantities enter stage parameters or x0 rather than generated C.
        runtime_fields = {
            "chance_epsilon", "robust_horizon", "initial_pose_covariance",
            "sigma_f0", "sigma_l0", "sigma_psi0",
        }
        return json.dumps(
            {key: value for key, value in vars(param).items() if key not in runtime_fields},
            sort_keys=True,
            default=lambda value: np.asarray(value).tolist(),
        )

    def create_model(self, param, p_psdf_sym=None):
        self._validate_param(param)
        if self.ped_model is None:
            self._configure_cluster_capacity(self.obstacle_capacity)
        if p_psdf_sym is None:
            p_psdf_sym = self.ped_model.get_sym_params()
        model = super().create_model(param, p_psdf_sym)
        model.name = "differential_drive_cc_ecs"
        return model

    def setup_ocp(self, param, reference_path_data):
        if self.ped_model is None:
            self._configure_cluster_capacity(self.obstacle_capacity)
        super().setup_ocp(param, reference_path_data)
        # Safe inactive rows before the first nominal geometry update.
        planes = np.zeros((self.obstacle_capacity, 4))
        planes[:, 3] = float(param.d_min) + 1.0
        self.ocp.parameter_values[self.psdf_param_slice] = planes.ravel()

    @staticmethod
    def _symbolic_projected_std(xi, q):
        variance = (
            (ca.exp(xi[0]) * q[0]) ** 2
            + (ca.exp(xi[1]) * q[1] + xi[3] * q[2]) ** 2
            + (ca.exp(xi[2]) * q[2]) ** 2
        )
        # P is SPD. Exactly zero gradients (padding) need finite derivatives.
        positive = variance > 0.0
        return ca.if_else(positive, ca.sqrt(ca.if_else(positive, variance, 1.0)), 0.0)

    def add_obstacle_avoidance_constraint(self, param, system, obstacles_geo):
        self.update_obstacles(obstacles_geo)
        model = self.ocp.model
        distances = self.ped_model(model.x[:3])
        gradients = model.p[self.risk_tensor_param_slice]
        beta = model.p[self.risk_margin_scale_param_slice][0]
        rows = ca.vertcat(*[
            distances[i] - float(param.d_min) - beta * self._symbolic_projected_std(
                model.x[4:8], gradients[3 * i:3 * i + 3]
            )
            for i in range(self.obstacle_capacity)
        ])
        model.con_h_expr = rows
        model.con_h_expr_e = rows
        # Stage 0 is fixed and outside the future-horizon budget.
        model.con_h_expr_0 = ca.MX.ones(self.obstacle_capacity, 1)
        for suffix in ("", "_e", "_0"):
            setattr(self.ocp.constraints, f"constr_type{suffix}", "BGH")
            setattr(self.ocp.constraints, f"lh{suffix}", np.zeros(self.obstacle_capacity))
            setattr(self.ocp.constraints, f"uh{suffix}", np.full(self.obstacle_capacity, 1e8))
        self.ocp.dims.nh = self.obstacle_capacity
        self.ocp.dims.nh_e = self.obstacle_capacity
        # No idxsh or slack costs: all future obstacle rows are hard.

    def initialize_ped_model(self, system, obstacles, E_max=None, K_max=None, device="cuda"):
        device = self._require_cuda_device(device)
        geometries = getattr(getattr(system, "_geometry", None), "_geometries", ())
        if len(geometries) != 1:
            raise ValueError("CC-ECS requires a single convex polygon robot footprint")
        vertices = np.asarray(geometries[0]._region.get_ccw_vertices(), dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("CC-ECS requires polygon vertices with shape (M>=3, 2)")
        if not np.isfinite(vertices).all():
            raise ValueError("Robot polygon vertices must be finite")
        edges = np.roll(vertices, -1, axis=0) - vertices
        next_edges = np.roll(edges, -1, axis=0)
        turns = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
        if np.any(turns < -1e-12) or not np.any(turns > 1e-12):
            raise ValueError("Robot footprint must be a CCW convex polygon")
        obstacles = [] if obstacles is None else list(obstacles)
        edge_counts = [len(obstacle.get_ccw_vertices()) for obstacle in obstacles]
        required_k = max(1, len(obstacles))
        required_e = max([1, *edge_counts])
        capacity = required_k if K_max is None else int(K_max)
        edge_capacity = required_e if E_max is None else int(E_max)
        if capacity < required_k or edge_capacity < required_e:
            raise ValueError("CC-ECS capacity must cover all supplied obstacles and edges")
        self.device = str(device)
        self.psdf_wrapper = AugmentedPSDFWrapper(
            torch.as_tensor(vertices, dtype=torch.float32, device=device),
            K_max=capacity, E_max=edge_capacity, device=device,
        )
        self._configure_cluster_capacity(capacity)
        self.update_obstacles(obstacles)

    def update_obstacles(self, obstacles_geo):
        if self.psdf_wrapper is None:
            # The simulator's PSDF logging hook runs before the first setup.
            # setup() receives the same full snapshot and initializes capacity.
            if not self._is_initialized:
                return
            if obstacles_geo is not None and len(obstacles_geo):
                if self.param.use_obstacle_constraint:
                    raise RuntimeError("CC-ECS geometry is not initialized")
            return
        obstacles = [] if obstacles_geo is None else list(obstacles_geo)
        if len(obstacles) > self.psdf_wrapper.K_max:
            raise ValueError("CC-ECS obstacle capacity exceeded; reset with the complete obstacle set")
        clusters = [polygon_to_edges(obstacle) for obstacle in obstacles]
        if any(len(a) > self.psdf_wrapper.E_max for a, _ in clusters):
            raise ValueError("CC-ECS edge capacity exceeded; reset with the complete obstacle set")
        # Never silently discard obstacles, including when replacing a larger set.
        self.psdf_wrapper.update_edge_clusters(
            [a for a, _ in clusters], [b for _, b in clusters]
        )

    @staticmethod
    def _cluster_local_gradients(poses, gradients_world):
        cosine = np.cos(poses[:, 2])[:, None]
        sine = np.sin(poses[:, 2])[:, None]
        return np.stack((
            cosine * gradients_world[..., 0] + sine * gradients_world[..., 1],
            -sine * gradients_world[..., 0] + cosine * gradients_world[..., 1],
            gradients_world[..., 2],
        ), axis=-1)

    def _evaluate_cluster_geometry(self, poses):
        poses = np.asarray(poses, dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 3 or not np.isfinite(poses).all():
            raise ValueError("Nominal poses must be finite with shape (H, 3)")
        if not self.param.use_obstacle_constraint:
            return np.empty((len(poses), 0)), np.empty((len(poses), 0, 3))
        if self.psdf_wrapper is None:
            raise RuntimeError("CC-ECS requires an initialized PSDF geometry backend")
        pose_tensor = torch.as_tensor(
            poses, dtype=self.psdf_wrapper.A.dtype, device=self.psdf_wrapper.device
        )
        distances, gradients = self.psdf_wrapper.forward_clusters(pose_tensor)
        distances = distances.detach().cpu().numpy().astype(float)
        gradients = gradients.detach().cpu().numpy().astype(float)
        count = self.psdf_wrapper.active_clusters
        if distances.shape != (len(poses), count) or gradients.shape != (len(poses), count, 3):
            raise ValueError("Unexpected per-obstacle PSDF output shape")
        if not np.isfinite(distances).all() or not np.isfinite(gradients).all():
            raise ValueError("Non-finite per-obstacle PSDF distance or gradient")
        return distances, gradients

    def _compute_constraint_data(self, states):
        states = np.asarray(states, dtype=float)
        if states.shape != (self.N + 1, 8) or not np.isfinite(states).all():
            raise ValueError("Nominal states must be finite with shape (N+1, 8)")
        covariance_from_log_cholesky(states[:, 4:8])
        distances, gradients = self._evaluate_cluster_geometry(states[:, :3])
        stage_epsilon, beta = self._risk_allocation(self.param)
        return {
            "poses": states[:, :3].copy(), "distances": distances,
            "gradients_world": gradients,
            "gradients_local": self._cluster_local_gradients(states[:, :3], gradients),
            "epsilon_stage": stage_epsilon, "beta": beta,
        }

    def _set_ecs_parameters(self, solver, stage_s_values, data):
        count = data["distances"].shape[1]
        if count > self.obstacle_capacity:
            raise ValueError("CC-ECS constraint data exceeds solver obstacle capacity")
        for stage in range(self.N + 1):
            planes = np.zeros((self.obstacle_capacity, 4))
            planes[:, 3] = self.param.d_min + 1.0
            gradients_local = np.zeros((self.obstacle_capacity, 3))
            active = stage > 0 and self.param.use_obstacle_constraint
            if active:
                gradient = data["gradients_world"][stage]
                planes[:count, :3] = gradient
                planes[:count, 3] = (
                    data["distances"][stage] - gradient @ data["poses"][stage]
                )
                gradients_local[:count] = data["gradients_local"][stage]
            solver.set(stage, "p", self._build_stage_param(
                self._build_stage_path_parameter(stage_s_values[stage]),
                gradients_local.ravel(), data["beta"] if active else 0.0,
                psdf_stage=planes.ravel(), stage_idx=stage,
            ))

    def _prepare_and_solve(self, solver):
        self._validate_param(self.param)
        if (
            self._compiled_param_signature is not None
            and self._build_signature(self.param) != self._compiled_param_signature
        ):
            raise ValueError("Changing compiled CC-ECS settings requires reset()")
        self.update_reference_path_params(self.reference_path_data, self.state)
        s_lower, s_upper = self._get_current_s_bounds(self.param)
        self._current_s0 = float(np.clip(self._current_s0, s_lower, s_upper))
        x0 = np.r_[self.state._x[:3], self._current_s0,
                   self._build_initial_covariance_state(self.param)]
        # A failed bounds/parameter update must never solve with stale state.
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)
        solver.set(0, "x", x0)
        indices, lbx, ubx = self._get_augmented_state_bounds(self.param, s_lower, s_upper)
        for stage in range(1, self.N + 1):
            if indices.size:
                solver.set(stage, "lbx", lbx)
                solver.set(stage, "ubx", ubx)
            state = np.asarray(solver.get(stage, "x"), dtype=float)
            if not np.isfinite(state[:4]).all():
                raise ValueError(f"Non-finite pose/progress guess at stage {stage}")
            state[3] = np.clip(state[3], s_lower, s_upper)
            solver.set(stage, "x", state)
        for stage in range(self.N):
            control = np.asarray(solver.get(stage, "u"), dtype=float)
            if not np.isfinite(control).all():
                raise ValueError(f"Non-finite input guess at stage {stage}")
            control[2] = np.clip(control[2], 0.0, self._get_effective_v_s_max(self.param))
            solver.set(stage, "u", control)
        stage_s = self._predict_stage_s_values(self._current_s0, s_lower, s_upper, solver)
        states = self._regenerate_covariance_guess(solver, stage_s)
        self._constraint_data = self._compute_constraint_data(states)
        self._set_ecs_parameters(solver, stage_s, self._constraint_data)
        started = time.perf_counter()
        try:
            status = int(solver.solve())
        finally:
            self._cycle_solver_time += time.perf_counter() - started
        solver.status = status
        return status

    def setup(self, param, system, reference_trajectory, obstacles):
        self._validate_param(param)
        signature = self._build_signature(param)
        if self._is_initialized and signature != self._compiled_param_signature:
            raise ValueError("Changing compiled CC-ECS settings requires reset()")
        self.param = param
        self.set_state(system._state)
        self._system = system
        self._system_dynamics = system._dynamics
        self.set_reference_trajectory(reference_trajectory)
        if self._is_initialized:
            self.update_obstacles(obstacles)
            return
        if param.use_obstacle_constraint:
            self.initialize_ped_model(
                system, obstacles, device=param.augmented_psdf_device
            )
        else:
            self._configure_cluster_capacity(1)
        self.setup_ocp(param, reference_trajectory)
        self.add_obstacle_avoidance_constraint(param, system, obstacles)
        self.solver = self.create_solver(code_export_directory=self.code_export_directory)
        self.add_warm_start(param, system)
        if param.enable_backup_solver:
            main_ocp = self.ocp
            backup_param = copy.deepcopy(param)
            backup_param.nlp_solver_type = "SQP_WITH_FEASIBLE_QP"
            backup_param.hpipm_mode = "ROBUST"
            backup_param.regularize_method = "PROJECT"
            try:
                self.setup_ocp(backup_param, reference_trajectory)
                self.add_obstacle_avoidance_constraint(backup_param, system, obstacles)
                self.backup_solver = self.create_solver(
                    json_filename=self.backup_json_filename,
                    code_export_directory=self.backup_code_export_directory,
                )
                self.add_warm_start(param, system, solver=self.backup_solver)
            finally:
                self.ocp = main_ocp
        self._compiled_param_signature = signature
        self._is_initialized = True

    def _apply_safe_stop(self, solver_status):
        dynamics = self._system_dynamics
        if dynamics is None or not hasattr(dynamics, "nominal_safe_controller"):
            raise RuntimeError("CC-ECS solver failed and nominal_safe_controller is unavailable")
        dt = self.param.tf / self.N
        states = np.zeros((self.N + 1, 8))
        inputs = np.zeros((self.N, 3))
        states[0, :4] = np.r_[self.state._x[:3], self._current_s0]
        previous_speed = float(self.state._u[0])
        for stage in range(self.N):
            next_pose, control = dynamics.nominal_safe_controller(
                states[stage, :3], dt, previous_speed, -1.0, 1.0
            )
            inputs[stage, :2] = np.asarray(control, dtype=float)[:2]
            states[stage + 1, :4] = np.r_[next_pose, self._current_s0]
            previous_speed = inputs[stage, 0]
        states[:, 4:8] = self._propagate_covariance_state_batch(inputs, states[:, 3])
        if not np.isfinite(states).all() or not np.isfinite(inputs).all():
            raise ValueError("Non-finite nominal safe-stop plan")
        self.solver.reset(reset_qp_solver_mem=1)
        for stage in range(self.N + 1):
            self.solver.set(stage, "x", states[stage])
            if stage < self.N:
                self.solver.set(stage, "u", inputs[stage])
        # reset() may reset status. Safe stop is not a successful ECS solve.
        self.solver.status = int(solver_status)
        return self.solver, int(solver_status), "safe_stop"

    def recovery_infeasible(self, solver_status):
        if self.backup_solver is not None:
            main_constraint_data = self._constraint_data
            self.backup_solver.reset(reset_qp_solver_mem=1)
            try:
                self._copy_primal_guess(self.solver, self.backup_solver)
            except ValueError:
                self.add_warm_start(self.param, self._system, solver=self.backup_solver)
            status = self._prepare_and_solve(self.backup_solver)
            self._last_backup_status = status
            if status == 0:
                self._copy_primal_guess(self.backup_solver, self.solver)
                return self.backup_solver, 0, "backup_feasible_qp"
            # Safe stop is written into the main solver, whose parameters still
            # contain the main attempt's frozen geometry.
            self._constraint_data = main_constraint_data
        return self._apply_safe_stop(solver_status)

    def _compute_margins(self, covariance_states, gradients_local):
        xi = np.asarray(covariance_states, dtype=float)
        q = np.asarray(gradients_local, dtype=float)
        covariance_from_log_cholesky(xi)  # Validate the retained SPD blocks.
        if q.ndim != 3 or q.shape[0] != len(xi) or q.shape[-1] != 3:
            raise ValueError("ECS gradients must have shape (H, K, 3)")
        if not np.isfinite(q).all():
            raise ValueError("ECS gradients must be finite")
        # q^T P q = ||L^T q||^2, retaining the sign of P_lpsi.
        projected = np.stack((
            np.exp(xi[:, 0, None]) * q[..., 0],
            np.exp(xi[:, 1, None]) * q[..., 1] + xi[:, 3, None] * q[..., 2],
            np.exp(xi[:, 2, None]) * q[..., 2],
        ), axis=-1)
        _, beta = self._risk_allocation(self.param)
        margins = beta * np.linalg.norm(projected, axis=-1)
        if not np.isfinite(margins).all():
            raise ValueError("Non-finite ECS margin")
        return margins

    def _record_experiment_logs(self, solver, status, mode):
        states = np.stack([solver.get(k, "x") for k in range(self.N + 1)])
        inputs = np.stack([solver.get(k, "u") for k in range(self.N)])
        covariance = covariance_from_log_cholesky(states[:, 4:8])
        data = self._constraint_data
        exact_distances, exact_gradients = self._evaluate_cluster_geometry(states[:, :3])
        fixed_margins = self._compute_margins(states[:, 4:8], data["gradients_local"])
        exact_margins = self._compute_margins(
            states[:, 4:8], self._cluster_local_gradients(states[:, :3], exact_gradients)
        )
        affine_distances = data["distances"] + np.einsum(
            "hki,hi->hk", data["gradients_world"], states[:, :3] - data["poses"]
        )
        fixed_residuals = affine_distances - self.param.d_min - fixed_margins
        exact_residuals = exact_distances - self.param.d_min - exact_margins
        active = np.ones(exact_distances.shape, dtype=bool)
        active[0] = False
        self._last_constraint_diagnostics = {
            "solver_status": int(status), "solve_mode": mode, "active": active,
            "nominal_psdf": data["distances"].copy(), "psdf": exact_distances,
            "fixed_gradient_margin": fixed_margins, "reevaluated_margin": exact_margins,
            "fixed_row_residual": np.where(active, fixed_residuals, np.nan),
            "reevaluated_ecs_residual": np.where(active, exact_residuals, np.nan),
            "covariance": covariance, "epsilon_stage": data["epsilon_stage"],
            "beta": data["beta"],
        }
        log_time = (self._solve_count - 1) * self.param.tf / self.N
        for stage in range(self.N + 1):
            p = covariance[stage]
            common = {"P_ff": p[0, 0], "P_ll": p[1, 1],
                      "P_psipsi": p[2, 2], "P_lpsi": p[1, 2]}
            self._covariance_log_rows.append({
                "time": log_time, "solve_step": self._solve_count, "stage": stage,
                "v": inputs[stage, 0] if stage < self.N else None,
                "omega": inputs[stage, 1] if stage < self.N else None,
                "sigma_f": np.sqrt(p[0, 0]), "sigma_l": np.sqrt(p[1, 1]),
                "sigma_psi": np.sqrt(p[2, 2]), **common,
            })
            for obstacle in range(exact_distances.shape[1]):
                is_active = bool(active[stage, obstacle])
                self._constraint_log_rows.append({
                    "time": log_time, "solve_step": self._solve_count,
                    "solver_status": int(status), "solve_mode": mode,
                    "horizon_stage": stage, "obstacle_index": obstacle,
                    "active": is_active, "epsilon_horizon": self.param.chance_epsilon,
                    "epsilon_stage": data["epsilon_stage"] if is_active else None,
                    "beta": data["beta"] if is_active else None,
                    "nominal_psdf": data["distances"][stage, obstacle],
                    "psdf": exact_distances[stage, obstacle],
                    "fixed_gradient_margin": fixed_margins[stage, obstacle] if is_active else None,
                    "reevaluated_margin": exact_margins[stage, obstacle] if is_active else None,
                    "fixed_row_residual": fixed_residuals[stage, obstacle] if is_active else None,
                    "reevaluated_ecs_residual": exact_residuals[stage, obstacle] if is_active else None,
                    **common,
                })
        self._cycle_log_rows.append({
            "time": log_time, "solve_step": self._solve_count, "solve_mode": mode,
            "solver_status": int(status), "main_status": self._last_main_status,
            "backup_status": self._last_backup_status, "solver_time_s": self._cycle_solver_time,
            "path_s": self._current_s0,
            "minimum_psdf": float(exact_distances.min()) if exact_distances.size else None,
            "minimum_fixed_row_residual": float(fixed_residuals[active].min()) if active.any() else None,
            "minimum_reevaluated_ecs_residual": float(exact_residuals[active].min()) if active.any() else None,
        })

    def solve_nlp(self):
        if self.solver is None:
            raise RuntimeError("CC-ECS solver is not initialized; call setup() first")
        self._cycle_solver_time = 0.0
        self._last_backup_status = None
        self._last_constraint_diagnostics = None
        self._solve_count += 1
        status = self._prepare_and_solve(self.solver)
        self._last_main_status = status
        active_solver, mode = self.solver, "sqp_rti"
        if status != 0:
            report_solver_failure(self.solver, status)
            active_solver, status, mode = self.recovery_infeasible(status)
            if mode == "backup_feasible_qp":
                self._backup_feasible_qp_success_count += 1
            elif mode == "safe_stop":
                self._safe_stop_count += 1
        self._has_prev_predicted_s = status == 0
        self._prev_predicted_s = self._clamp_s(
            float(active_solver.get(1, "x")[3]) if status == 0 else self._current_s0
        )
        active_solver.status = int(status)
        self._last_solve_mode = mode
        self.solver_times.append(self._cycle_solver_time)
        self._record_experiment_logs(active_solver, status, mode)
        print(
            f"[CC-ECS {self._solve_count:04d}] {mode} status={status} "
            f"solve={1e3 * self._cycle_solver_time:.1f}ms s={self._current_s0:.3f}"
        )
        return AcadosSolution(active_solver, self.N, self.variables)

    def get_last_constraint_diagnostics(self):
        return copy.deepcopy(self._last_constraint_diagnostics)

    def get_constraint_log_rows(self):
        return [row.copy() for row in self._constraint_log_rows]

    def get_constraint_log_fields(self):
        return self.CONSTRAINT_LOG_FIELDS

    def get_covariance_log_rows(self):
        return [row.copy() for row in self._covariance_log_rows]

    def get_covariance_log_fields(self):
        return self.COVARIANCE_LOG_FIELDS

    def get_cycle_log_rows(self):
        return [row.copy() for row in self._cycle_log_rows]

    def get_cycle_log_fields(self):
        return self.CYCLE_LOG_FIELDS

    def get_last_risk_margin_trajectory(self):
        # A single global-PSDF margin cannot represent these per-obstacle rows.
        # Full margins are available in constraint diagnostics and CSV output.
        return None

    def get_risk_margin_active_stage_mask(self, num_stages=None):
        if self.N is None:
            return None
        count = self.N + 1 if num_stages is None else int(num_stages)
        mask = np.ones(count, dtype=bool)
        if count:
            mask[0] = False
        if not self.param.use_obstacle_constraint:
            mask[:] = False
        return mask
