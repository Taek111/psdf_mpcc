import os
import time

import casadi as ca
import numpy as np
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from control.analytic_psdf_casadi import AnalyticPSDFCasADi
from models.geometry_utils import polygon_to_edges
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from planning.trajectory_generator.spline_reference_generator import build_cubic_spline_path_data


class MPCCOptimizerParam:
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
        self.use_soft_constraint = False
        self.slack_weight = 1e16

        # Obstacle detection parameters (kept for config compatibility)
        self.detection_window_width = 4.0
        self.detection_window_height = 4.0
        self.detection_safety_margin = 0.05
        self.detection_frequency = 10.0


class MPCCOptimizer:
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

        self.path_param_dim = None
        self.psdf_param_dim = 0
        self.path_param_slice = slice(0, 0)
        self.psdf_param_slice = slice(0, 0)
        self._path_params = None
        self._prev_predicted_s = 0.0
        self._current_s0 = 0.0
        self._has_prev_predicted_s = False

        # Local window obstacle detector placeholder (kept for compatibility)
        self.obstacle_detector = None

        self.json_filename = "acados_ocp.json"
        self._temp_files = []

    def cleanup(self):
        try:
            for file_path in self._temp_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            if os.path.exists(self.json_filename):
                os.remove(self.json_filename)
            if os.path.exists("c_generated_code") and os.path.isdir("c_generated_code"):
                import shutil
                shutil.rmtree("c_generated_code", ignore_errors=True)
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

        self.ocp = None
        self.solver = None
        self.solver_times = []
        self.psdf_wrapper = None
        self.ped_model = None
        self._is_initialized = False

        self.reference_path_data = None
        self.path_param_dim = None
        self.psdf_param_dim = 0
        self.path_param_slice = slice(0, 0)
        self.psdf_param_slice = slice(0, 0)
        self._path_params = None
        self._prev_predicted_s = 0.0
        self._current_s0 = 0.0
        self._has_prev_predicted_s = False
        self.obstacle_detector = None

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
        nx = 4  # [x, y, theta, s]
        nu = 3  # [v, omega, v_s]
        # Stage-varying path parameter: [p_ref_x, p_ref_y, t_ref_x, t_ref_y, s_ref]
        self.path_param_dim = 5

        x = ca.MX.sym("x", nx)
        xdot = ca.MX.sym("xdot", nx)
        u = ca.MX.sym("u", nu)
        p_ref_t_ref = ca.MX.sym("p_ref_t_ref", self.path_param_dim)

        self.psdf_param_dim = 0 if p_psdf_sym is None else int(p_psdf_sym.shape[0])
        self.path_param_slice = slice(0, self.path_param_dim)
        self.psdf_param_slice = slice(self.path_param_dim, self.path_param_dim + self.psdf_param_dim)

        if p_psdf_sym is None:
            p_all = p_ref_t_ref
        else:
            p_all = ca.vertcat(p_ref_t_ref, p_psdf_sym)

        f_expl = ca.vertcat(
            u[0] * ca.cos(x[2]),
            u[0] * ca.sin(x[2]),
            u[1],
            u[2],
        )

        p_ref = p_ref_t_ref[0:2]
        t_raw = p_ref_t_ref[2:4]
        s_ref = p_ref_t_ref[4]
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
        model.name = "differential_drive_mpcc"
        model.x = x
        model.xdot = xdot
        model.u = u
        model.p = p_all
        model.f_expl_expr = f_expl
        model.f_impl_expr = xdot - f_expl
        model.cost_expr_ext_cost = cost_stage
        model.cost_expr_ext_cost_e = cost_terminal
        return model

    def _build_path_data_from_polyline(self, polyline):
        return build_cubic_spline_path_data(polyline)

    def _normalize_reference_path_data(self, reference_path_data):
        if isinstance(reference_path_data, np.ndarray):
            return self._build_path_data_from_polyline(reference_path_data)

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

    def _reference_eval_numeric(self, s, path_data):
        s = float(s)
        n_seg = int(path_data["n_segments"])
        if n_seg <= 0:
            return np.zeros((2,), dtype=float)

        s_breaks = np.asarray(path_data["s_breaks"], dtype=float)
        s_min = float(s_breaks[0])
        s_max = float(s_breaks[n_seg])
        s_clamped = float(np.clip(s, s_min, s_max))

        seg_idx = int(np.searchsorted(s_breaks[1:n_seg + 1], s_clamped, side="right"))
        seg_idx = int(np.clip(seg_idx, 0, n_seg - 1))
        s_left = float(s_breaks[seg_idx])
        tau = s_clamped - s_left

        cx = np.asarray(path_data["segments_x"][seg_idx], dtype=float)
        cy = np.asarray(path_data["segments_y"][seg_idx], dtype=float)
        x_ref = cx[0] + cx[1] * tau + cx[2] * tau * tau + cx[3] * tau * tau * tau
        y_ref = cy[0] + cy[1] * tau + cy[2] * tau * tau + cy[3] * tau * tau * tau
        return np.array([x_ref, y_ref], dtype=float)

    def _reference_tangent_numeric(self, s, path_data):
        n_seg = int(path_data["n_segments"])
        if n_seg <= 0:
            return np.array([1.0, 0.0], dtype=float)

        s_breaks = np.asarray(path_data["s_breaks"], dtype=float)
        s_min = float(s_breaks[0])
        s_max = float(s_breaks[n_seg])
        s_clamped = float(np.clip(float(s), s_min, s_max))

        seg_idx = int(np.searchsorted(s_breaks[1:n_seg + 1], s_clamped, side="right"))
        seg_idx = int(np.clip(seg_idx, 0, n_seg - 1))
        s_left = float(s_breaks[seg_idx])
        tau = s_clamped - s_left

        cx = np.asarray(path_data["segments_x"][seg_idx], dtype=float)
        cy = np.asarray(path_data["segments_y"][seg_idx], dtype=float)
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

    def _build_stage_path_parameter(self, s_value):
        if self.reference_path_data is None:
            return np.array([0.0, 0.0, 1.0, 0.0, float(s_value)], dtype=float)
        p_ref = self._reference_eval_numeric(s_value, self.reference_path_data)
        t_ref = self._reference_tangent_numeric(s_value, self.reference_path_data)
        return np.array([p_ref[0], p_ref[1], t_ref[0], t_ref[1], float(s_value)], dtype=float)

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
        p_psdf_sym = None
        if self.ped_model is not None:
            # The analytic bridge exposes the same seven Taylor parameters.
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
        self.ocp.dims.np = self.path_param_dim + self.psdf_param_dim
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
        x0_full = np.array([x0_phys[0], x0_phys[1], x0_phys[2], self._current_s0], dtype=float)

        # Stage-0 equality on full augmented state.
        self.ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3], dtype=np.int64)
        self.ocp.constraints.lbx_0 = x0_full.copy()
        self.ocp.constraints.ubx_0 = x0_full.copy()

        # Optional s-bounds for intermediate and terminal stages.
        if getattr(param, "use_s_state_bounds", True):
            self.ocp.constraints.idxbx = np.array([3], dtype=np.int64)
            self.ocp.constraints.lbx = np.array([s_lower], dtype=float)
            self.ocp.constraints.ubx = np.array([s_upper], dtype=float)
            self.ocp.constraints.idxbx_e = np.array([3], dtype=np.int64)
            self.ocp.constraints.lbx_e = np.array([s_lower], dtype=float)
            self.ocp.constraints.ubx_e = np.array([s_upper], dtype=float)
        else:
            self.ocp.constraints.x0 = x0_full

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

        if reference_path_data is not None:
            self.set_reference_trajectory(reference_path_data)

    def create_solver(self):
        if (
            getattr(getattr(self, "ped_model", None), "requires_external_shared_lib", False)
            and hasattr(self.ped_model, "shared_lib_dir")
            and hasattr(self.ped_model, "name")
        ):
            self.ocp.solver_options.model_external_shared_lib_dir = self.ped_model.shared_lib_dir
            self.ocp.solver_options.model_external_shared_lib_name = self.ped_model.name

        self.variables["x"] = "x"
        self.variables["u"] = "u"
        self.solver = AcadosOcpSolver(self.ocp, json_file=self.json_filename)
        self._temp_files.append(self.json_filename)

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

        # The analytic bridge injects first-order Taylor parameters through model.p.
        sdf_value = self.ped_model(x[:3])
        constraint_expr = sdf_value - d_min

        self.ocp.constraints.constr_type = "BGH"
        self.ocp.dims.nh = 1
        self.ocp.model.con_h_expr = constraint_expr
        self.ocp.constraints.lh = np.array([0.0], dtype=float)
        self.ocp.constraints.uh = np.array([1e8], dtype=float)

        if getattr(param, "use_soft_constraint", False):
            ns = 1
            self.ocp.dims.ns = ns
            self.ocp.dims.nsh = 1
            self.ocp.constraints.idxsh = np.arange(1, dtype=np.int64)
            self.ocp.constraints.lsh = np.zeros(ns)
            self.ocp.constraints.ush = np.ones(ns) * 1e8
            slack_w = float(getattr(param, "slack_weight", 1e4))
            self.ocp.cost.Zl = np.diag([slack_w] * ns)
            self.ocp.cost.Zu = np.diag([slack_w] * ns)
            self.ocp.cost.zl = np.zeros(ns)
            self.ocp.cost.zu = np.zeros(ns)
            print(f"Obstacle avoidance softened with slack weight = {slack_w}")
        else:
            print(f"Added HARD obstacle avoidance constraint with d_min = {d_min}")

    def add_warm_start(self, param, system):
        if self.solver is None:
            return

        dt = float(param.tf) / float(param.horizon)
        effective_v_s_max = self._get_effective_v_s_max(param)
        try:
            x_ws, u_ws = system._dynamics.nominal_safe_controller(
                self.state._x, dt, self.state._u[0], -1.0, 1.0
            )
            v_s_ws = float(np.clip(0.1, 0.0, effective_v_s_max))
            for i in range(self.N):
                s_ws = self._clamp_s(self._current_s0 + i * dt * v_s_ws)
                self.solver.set(i, "x", np.array([x_ws[0], x_ws[1], x_ws[2], s_ws], dtype=float))
                self.solver.set(i, "u", np.array([u_ws[0], u_ws[1], v_s_ws], dtype=float))
            self.solver.set(
                self.N,
                "x",
                np.array(
                    [x_ws[0], x_ws[1], x_ws[2], self._clamp_s(self._current_s0 + self.N * dt * v_s_ws)],
                    dtype=float,
                ),
            )
        except Exception as e:
            print(f"Error in warm start: {e}")
            for i in range(self.N):
                self.solver.set(i, "x", np.zeros((self.nx,), dtype=float))
                self.solver.set(i, "u", np.zeros((self.nu,), dtype=float))
            self.solver.set(self.N, "x", np.zeros((self.nx,), dtype=float))

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
        self.set_reference_trajectory(reference_trajectory)
        use_obstacle_constraint = getattr(param, "use_obstacle_constraint", True)

        if not self._is_initialized:
            print("Setting up MPCC optimizer...")
            if use_obstacle_constraint:
                self.initialize_ped_model(system, obstacles, E_max=100, K_max=20, device="cpu")
            else:
                self.psdf_wrapper = None
                self.ped_model = None
                print("MPCC obstacle constraint disabled: skipping PSDF model initialization.")
            self.setup_ocp(param, reference_trajectory)
            if use_obstacle_constraint:
                self.add_obstacle_avoidance_constraint(param, system, obstacles)
            self.create_solver()
            self.add_warm_start(param, system)
            self._is_initialized = True
        else:
            self.set_reference_trajectory(reference_trajectory)
            if use_obstacle_constraint:
                self.update_obstacles(obstacles)

    def solve_nlp(self):
        if self.solver is None:
            raise RuntimeError("MPCC solver is not initialized. Call setup() first.")

        start = time.time()
        s_lower, s_upper = self._get_current_s_bounds(self.param)
        if self.state is not None:
            self.update_reference_path_params(self.reference_path_data, self.state)
            s_lower, s_upper = self._get_current_s_bounds(self.param)
            self._current_s0 = float(np.clip(self._current_s0, s_lower, s_upper))
            x0 = np.array([self.state._x[0], self.state._x[1], self.state._x[2], self._current_s0], dtype=float)
            try:
                # Preferred when idxbx_0 = [0,1,2,3]
                self.solver.set(0, "lbx", x0)
                self.solver.set(0, "ubx", x0)
            except Exception:
                # Fallback for configurations where only s is boxed.
                print("Warning: stage-0 full-state bounds update failed; falling back to s-only bound.")
                self.solver.set(0, "lbx", np.array([x0[3]], dtype=float))
                self.solver.set(0, "ubx", np.array([x0[3]], dtype=float))
            self.solver.set(0, "x", x0)

        if getattr(self.param, "use_s_state_bounds", True):
            try:
                for i in range(1, self.N):
                    self.solver.set(i, "lbx", np.array([s_lower], dtype=float))
                    self.solver.set(i, "ubx", np.array([s_upper], dtype=float))
                self.solver.set(self.N, "lbx", np.array([s_lower], dtype=float))
                self.solver.set(self.N, "ubx", np.array([s_upper], dtype=float))
            except Exception:
                print("Warning: failed to update per-stage s bounds in solver.")

        # Keep warm-start state/control in-domain before using them for stage-wise reference generation.
        try:
            effective_v_s_max = self._get_effective_v_s_max(self.param)
            for i in range(1, self.N + 1):
                xi = self.solver.get(i, "x")
                xi[3] = float(np.clip(xi[3], s_lower, s_upper))
                self.solver.set(i, "x", xi)
            for i in range(self.N):
                ui = self.solver.get(i, "u")
                ui[2] = float(np.clip(ui[2], 0.0, effective_v_s_max))
                self.solver.set(i, "u", ui)
        except Exception:
            pass

        psdf_params_batch = None
        if self.ped_model is not None and self.psdf_param_dim > 0:
            x_guess = np.stack([self.solver.get(i, "x") for i in range(self.N + 1)], axis=0)
            psdf_params_batch = self.ped_model.get_params(x_guess[:, :3])

        stage_s_values = self._predict_stage_s_values(self._current_s0, s_lower, s_upper)
        for i in range(self.N):
            stage_path_param = self._build_stage_path_parameter(stage_s_values[i])
            psdf_stage = None if psdf_params_batch is None else psdf_params_batch[i]
            stage_param = np.zeros((self.path_param_dim + self.psdf_param_dim,), dtype=float)
            stage_param[self.path_param_slice] = stage_path_param
            if self.psdf_param_dim > 0 and psdf_stage is not None:
                stage_param[self.psdf_param_slice] = psdf_stage
            self.solver.set(i, "p", stage_param)
        terminal_path_param = self._build_stage_path_parameter(stage_s_values[self.N])
        psdf_terminal = None if psdf_params_batch is None else psdf_params_batch[-1]
        terminal_stage_param = np.zeros((self.path_param_dim + self.psdf_param_dim,), dtype=float)
        terminal_stage_param[self.path_param_slice] = terminal_path_param
        if self.psdf_param_dim > 0 and psdf_terminal is not None:
            terminal_stage_param[self.psdf_param_slice] = psdf_terminal
        self.solver.set(self.N, "p", terminal_stage_param)

        status = self.solver.solve()
        solve_time = time.time() - start
        self.solver_times.append(solve_time)
        print(f"solver time: {solve_time}, path_s: {self._current_s0}")

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
            # Prevent failed iterate from poisoning next-step projection center.
            self._prev_predicted_s = self._clamp_s(self._current_s0)
            self._has_prev_predicted_s = False
        else:
            try:
                if self.N >= 1:
                    self._prev_predicted_s = self._clamp_s(float(self.solver.get(1, "x")[3]))
                else:
                    self._prev_predicted_s = self._clamp_s(float(self.solver.get(0, "x")[3]))
                self._has_prev_predicted_s = True
            except Exception:
                self._prev_predicted_s = self._clamp_s(self._current_s0)
                self._has_prev_predicted_s = False

        return AcadosSolution(self.solver, self.N, self.variables)

    def initialize_ped_model(self, system, obstacles, E_max=100, K_max=20, device="cpu"):
        self.device = device
        if not hasattr(system, "_geometry"):
            raise ValueError("System must have _geometry attribute")

        geometry = system._geometry._geometries[0]
        vertices = geometry._region.get_ccw_vertices()
        self.psdf_wrapper = AugmentedPSDFWrapper(
            verts=torch.tensor(vertices, dtype=torch.float32, device=device),
            E_max=E_max,
            K_max=K_max,
            device=device,
        )
        self.ped_model = AnalyticPSDFCasADi(
            self.psdf_wrapper,
            device=self.device,
            name="analytic_mpcc_psdf",
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
