"""SIP/zoRO dilated-polygon MPC adapted to the repository's DD model.

Based on the equations in references/sipoc/sip_paper.pdf and the algorithm
in Bosch's DilatedPolygonRobustController. The upper OCP uses acados SQP;
the lower problem uses Clarabel. Polygonal obstacles replace the upstream
point cloud, so this is an adaptation, not a reproduction of its timings.
See SIPOC_IMPLEMENTATION.md for equations, assumptions and validation.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import casadi as ca
import numpy as np

from control.sipoc_geometry import (
    ConvexPolygon,
    polygon_ellipse_distance,
    propagate_covariance,
    psd_matrix,
    rotation,
)


@dataclass
class DilatedPolygonRobustOptimizerParam:
    horizon: int = 15
    tf: float = 1.5
    mat_Q: np.ndarray = field(default_factory=lambda: np.diag([20., 20., .5]))
    mat_R: np.ndarray = field(default_factory=lambda: np.diag([5., .05]))
    terminal_weight: float = 2.0
    vmin: float = -0.6
    vmax: float = 0.6
    omegamin: float = -1.0
    omegamax: float = 1.0
    polygon_dilation_radius: float = 0.001
    d_safe: float = 0.0
    robust_scale: float = 1.0
    initial_covariance: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    # Discrete state disturbance shape, per step; illustrative, not identified.
    process_noise: np.ndarray = field(default_factory=lambda: np.diag([1e-6, 1e-6, 1e-5]))
    feedback_gain: np.ndarray = field(default_factory=lambda: np.zeros((2, 3)))
    max_outer_iterations: int = 100
    max_constraints: int = 25
    activation_distance: float = 0.5
    feasibility_tolerance: float = 1e-5
    step_tolerance: float = 1e-4
    # Relax the frozen-geometry fixed-point iteration at polygon corners.
    # Convergence still uses the undamped proposal residual.
    outer_step_length: float = 1.0
    # Slacks permit iterates to escape overlap; accepted outputs remain feasible.
    use_soft_constraint: bool = True
    slack_l1_penalty: float = 1e3
    slack_l2_penalty: float = 1e2
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    qp_solver_iter_max: int = 100
    # Match the original Python experiment's frozen-subproblem solver.
    nlp_solver_type: str = "SQP"
    nlp_solver_max_iter: int = 4
    levenberg_marquardt: float = 1e-7
    # acados SQP default used by the original generator. A looser upper
    # stationarity tolerance can exceed the outer input convergence threshold.
    tol: float = 1e-8


class DilatedPolygonSolution:
    """Immutable controller/logger view, with the complete robust solve status."""

    def __init__(self, states, inputs, success, reason, diagnostics):
        self._states = np.asarray(states).copy()
        self._inputs = np.asarray(inputs).copy()
        self.solver = SimpleNamespace(status=0 if success else 1)
        self._stats = {"success": bool(success), "return_status": reason,
                       **diagnostics}

    def value(self, variable):
        if variable == "x":
            return self._states.T.copy()
        if variable == "u":
            return self._inputs.T.copy()
        raise NotImplementedError("Only 'x' and 'u' are supported")

    def get_state_trajectory(self):
        return self.value("x")

    def get_input_trajectory(self):
        return self.value("u")

    def stats(self):
        return dict(self._stats)


class DilatedPolygonRobustController:
    """Optimizer API compatible with BaseController and test_nmpc.py.

    Uncertainty is frozen within each SQP subproblem and recomputed after it (zoRO).
    Every accepted output passes a new nonlinear geometry check at all nodes,
    including the measured initial pose and the terminal pose. Failure returns
    zero velocity with success=False; lower-solver exceptions propagate.
    """

    def __init__(self, variables=None, costs=None, dynamics_opt=None):
        self.variables = {"x": "x", "u": "u"}
        self.costs = {} if costs is None else costs
        self.dynamics_opt = dynamics_opt
        self._supplied_dynamics = dynamics_opt
        self.solver = None
        self.ocp = None
        self.state = None
        self.solver_times = []
        self._workspace = None
        self._signature = None
        self._previous_inputs = None
        self._last_covariances = None
        self._last_diagnostics = {}
        self._solve_count = 0
        self._success_count = 0
        self._safe_stop_count = 0
        self._plant_input_apply_count = 0
        self.problem_build_count = 0
        self.active_obstacle_count = 0
        self.active_constraint_pair_count = 0
        self._is_initialized = False

    def set_state(self, state):
        self.state = state

    @staticmethod
    def _validate_param(param):
        for name in ("horizon", "max_constraints", "max_outer_iterations",
                     "qp_solver_iter_max", "nlp_solver_max_iter"):
            value = getattr(param, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("tf", "feasibility_tolerance", "step_tolerance", "tol"):
            value = float(getattr(param, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (not np.isfinite(param.outer_step_length)
                or not 0 < param.outer_step_length <= 1):
            raise ValueError("outer_step_length must be in (0, 1]")
        for name in ("polygon_dilation_radius", "d_safe", "robust_scale",
                     "terminal_weight", "slack_l1_penalty", "slack_l2_penalty"):
            value = float(getattr(param, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (not np.isfinite(param.levenberg_marquardt)
                or param.levenberg_marquardt < 0):
            raise ValueError("levenberg_marquardt must be finite and nonnegative")
        if param.nlp_solver_type not in ("SQP", "SQP_RTI"):
            raise ValueError("nlp_solver_type must be SQP or SQP_RTI")
        if not np.isfinite(param.activation_distance) or param.activation_distance < 0:
            raise ValueError("activation_distance must be finite and nonnegative")
        bounds = np.array([param.vmin, param.vmax, param.omegamin, param.omegamax])
        if (not np.all(np.isfinite(bounds)) or not param.vmin <= 0 <= param.vmax
                or not param.omegamin <= 0 <= param.omegamax):
            raise ValueError("Input bounds must be finite and contain zero for safe stop")
        for name, dim in (("mat_Q", 3), ("mat_R", 2),
                          ("initial_covariance", 3), ("process_noise", 3)):
            setattr(param, name, psd_matrix(getattr(param, name), dim, name))
        gain = np.asarray(param.feedback_gain, dtype=float)
        if gain.shape != (2, 3) or not np.all(np.isfinite(gain)):
            raise ValueError("feedback_gain must be a finite 2x3 matrix")
        if np.any(gain != 0.):
            raise ValueError("SIPOC DD adaptation requires feedback_gain=0; ancillary feedback is not implemented")
        param.feedback_gain = gain

    def setup(self, param, system, reference_trajectory, obstacles):
        self._validate_param(param)
        self.param = param
        self.N, self.nx, self.nu = int(param.horizon), 3, 2
        self.dt = float(param.tf) / self.N
        if hasattr(system, "_dt") and not np.isclose(system._dt, self.dt):
            raise ValueError("SIPOC horizon timestep must match the plant timestep")
        self.set_state(system._state)
        self._x0 = np.asarray(self.state._x, dtype=float).reshape(-1).copy()
        if self._x0.shape != (3,) or not np.all(np.isfinite(self._x0)):
            raise ValueError("SIPOC requires a finite DD [x,y,theta] state")
        self.reference_trajectory = self._pack_reference(reference_trajectory)
        self.bodies = [ConvexPolygon.from_geometry(g)
                       for g in system._geometry.equiv_rep()]
        if not self.bodies:
            raise ValueError("A polygon robot footprint is required")
        self.obstacles = [ConvexPolygon.from_geometry(g) for g in obstacles]
        # Includes every disconnected convex component, preserving offsets.
        signature = self._solver_signature()
        if signature != self._signature:
            self._release_solver()
            self._configure_dynamics()
            try:
                self._build_solver()
            except Exception:
                self._release_solver()
                raise
            self._previous_inputs = None
            self._signature = signature
            self._is_initialized = True
        self.set_reference_trajectory(self.reference_trajectory)

    def _pack_reference(self, reference):
        reference = np.asarray(reference, dtype=float)
        if (reference.ndim != 2 or reference.shape[0] < 1
                or reference.shape[1] < 3 or not np.all(np.isfinite(reference))):
            raise ValueError("reference_trajectory must be a finite nonempty Nx3 array")
        reference = reference[:self.N + 1, :3].copy()
        if len(reference) < self.N + 1:
            reference = np.vstack([reference, np.repeat(reference[-1:],
                                   self.N + 1 - len(reference), axis=0)])
        headings = np.unwrap(np.r_[self._x0[2], reference[:, 2]])[1:]
        reference[:, 2] = headings
        return reference

    def _solver_signature(self):
        p = self.param
        return (self.N, p.tf, p.max_constraints, p.use_soft_constraint,
                p.mat_Q.tobytes(), p.mat_R.tobytes(), p.terminal_weight,
                p.vmin, p.vmax, p.omegamin, p.omegamax,
                p.slack_l1_penalty, p.slack_l2_penalty,
                p.qp_solver, p.qp_solver_iter_max, p.nlp_solver_type,
                p.nlp_solver_max_iter, p.levenberg_marquardt, p.tol)

    def _configure_dynamics(self):
        x, u = ca.SX.sym("x", 3), ca.SX.sym("u", 2)
        if self._supplied_dynamics is None:
            next_x = x + self.dt * ca.vertcat(u[0] * ca.cos(x[2]),
                                             u[0] * ca.sin(x[2]), u[1])
            self.dynamics_opt = ca.Function("sipoc_dd_step", [x, u], [next_x])
        else:
            next_x = self._supplied_dynamics(x, u)
        if next_x.shape != (3, 1):
            raise ValueError("dynamics_opt must map (3-state,2-input) to 3-state")
        self._jacobians = ca.Function("sipoc_dd_jacobians", [x, u],
                                     [ca.jacobian(next_x, x), ca.jacobian(next_x, u)])

    def _build_solver(self):
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

        self._workspace = tempfile.TemporaryDirectory(prefix="psdf_sipoc_")
        p, m = self.param, int(self.param.max_constraints)
        model = AcadosModel()
        model.name = "differential_drive_sipoc"
        model.x = ca.SX.sym("x", 3)
        model.u = ca.SX.sym("u", 2)
        model.disc_dyn_expr = self.dynamics_opt(model.x, model.u)
        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self.N
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W = np.block([[p.mat_Q, np.zeros((3, 2))],
                              [np.zeros((2, 3)), p.mat_R]])
        ocp.cost.W_e = p.terminal_weight * p.mat_Q
        ocp.cost.Vx = np.vstack([np.eye(3), np.zeros((2, 3))])
        ocp.cost.Vu = np.vstack([np.zeros((3, 2)), np.eye(2)])
        ocp.cost.Vx_e = np.eye(3)
        ocp.cost.yref = np.zeros(5)
        ocp.cost.yref_e = np.zeros(3)
        ocp.constraints.x0 = self._x0
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbu = np.array([p.vmin, p.omegamin])
        ocp.constraints.ubu = np.array([p.vmax, p.omegamax])
        ocp.constraints.C = np.zeros((m, 3))
        ocp.constraints.D = np.zeros((m, 2))
        ocp.constraints.lg = np.full(m, -1e8)
        ocp.constraints.ug = np.full(m, 1e8)
        ocp.constraints.C_e = np.zeros((m, 3))
        ocp.constraints.lg_e = np.full(m, -1e8)
        ocp.constraints.ug_e = np.full(m, 1e8)
        if p.use_soft_constraint:
            ocp.constraints.idxsg = np.arange(m)
            ocp.constraints.idxsg_e = np.arange(m)
            for suffix in ("", "_e"):
                setattr(ocp.cost, "zl" + suffix, np.full(m, p.slack_l1_penalty))
                setattr(ocp.cost, "zu" + suffix, np.full(m, p.slack_l1_penalty))
                setattr(ocp.cost, "Zl" + suffix, np.full(m, p.slack_l2_penalty))
                setattr(ocp.cost, "Zu" + suffix, np.full(m, p.slack_l2_penalty))
        ocp.solver_options.tf = p.tf
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = p.nlp_solver_type
        ocp.solver_options.nlp_solver_max_iter = int(p.nlp_solver_max_iter)
        ocp.solver_options.levenberg_marquardt = p.levenberg_marquardt
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.qp_solver = p.qp_solver
        ocp.solver_options.qp_solver_iter_max = p.qp_solver_iter_max
        ocp.solver_options.tol = p.tol
        ocp.solver_options.print_level = 0
        ocp.code_export_directory = self._workspace.name + "/code"
        self.ocp = ocp
        self.solver = AcadosOcpSolver(ocp, json_file=self._workspace.name + "/ocp.json")
        self.problem_build_count += 1

    def set_reference_trajectory(self, reference):
        self.reference_trajectory = np.asarray(reference).copy()
        if self.solver is not None:
            for k in range(self.N):
                self.solver.set(k, "yref", np.r_[self.reference_trajectory[k], 0., 0.])
            self.solver.set(self.N, "yref", self.reference_trajectory[self.N])

    def _rollout(self, inputs):
        states = np.empty((self.N + 1, 3))
        states[0] = self._x0
        for k, u in enumerate(inputs):
            states[k + 1] = np.asarray(self.dynamics_opt(states[k], u)).reshape(3)
        return states

    def _seed(self):
        if self._previous_inputs is not None:
            inputs = np.vstack([self._previous_inputs[1:], self._previous_inputs[-1:]])
        else:
            inputs = np.zeros((self.N, 2))
            for k in range(self.N):
                start = self._x0 if k == 0 else self.reference_trajectory[k - 1]
                target = self.reference_trajectory[k]
                direction = np.array([np.cos(start[2]), np.sin(start[2])])
                inputs[k, 0] = (target[:2] - start[:2]) @ direction / self.dt
                inputs[k, 1] = (target[2] - start[2]) / self.dt
            inputs = np.clip(inputs, [self.param.vmin, self.param.omegamin],
                             [self.param.vmax, self.param.omegamax])
        return self._rollout(inputs), inputs

    def _propagate_covariances(self, states, inputs):
        a, b = [], []
        for k, u in enumerate(inputs):
            ak, bk = self._jacobians(states[k], u)
            a.append(np.asarray(ak))
            b.append(np.asarray(bk))
        return propagate_covariance(self.param.initial_covariance, a, b,
                                    self.param.process_noise, self.param.feedback_gain)

    def _geometry_at_nodes(self, states, covariances, include_initial=False):
        """All pairs, with a conservative bounding-box skip for distant pairs.

        AABB gaps bound clearance from below even with rotation uncertainty.
        Final validation uses the same test and never silently drops a violation.
        """
        result = [[] for _ in range(self.N + 1)]
        radius = self.param.polygon_dilation_radius + self.param.d_safe
        tol = self.param.feasibility_tolerance
        for k in range(0 if include_initial else 1, self.N + 1):
            shape = self.param.robust_scale * covariances[k]
            translation_extent = np.sqrt(np.maximum(0., np.diag(shape)[:2]))
            angular_extent = np.sqrt(max(0., shape[2, 2]))
            rot = rotation(states[k, 2])
            for bi, body in enumerate(self.bodies):
                backoff = body.radius * angular_extent
                world = body.vertices @ rot.T + states[k, :2]
                extent = translation_extent + radius + backoff
                lower, upper = world.min(axis=0) - extent, world.max(axis=0) + extent
                for oi, obstacle in enumerate(self.obstacles):
                    gap = np.maximum(np.maximum(obstacle.vertices.min(axis=0) - upper,
                                                lower - obstacle.vertices.max(axis=0)), 0.)
                    if np.linalg.norm(gap) > max(self.param.activation_distance, tol):
                        continue
                    witness = polygon_ellipse_distance(states[k, :2], states[k, 2],
                                                       shape[:2, :2], body, obstacle)
                    clearance = witness.support_gap - radius - backoff
                    result[k].append((clearance, bi, oi, witness, backoff))
        return result

    @staticmethod
    def _minimum_clearance(geometry):
        return min((row[0] for stage in geometry for row in stage), default=float("inf"))

    @staticmethod
    def linearized_clearance_row(pose, witness, radius, backoff):
        """Paper Eq.57 / source C*x>=lg, with the robust residual normal."""
        normal, gamma = witness.normal, witness.gamma
        rot = rotation(pose[2])
        derivative = rot @ np.array([[0., -1.], [1., 0.]])
        c = np.r_[normal, normal @ derivative @ gamma]
        lg = (normal @ witness.obstacle_point - normal @ witness.delta
              - normal @ (rot - pose[2] * derivative) @ gamma + radius + backoff)
        return c, float(lg)

    def _update_constraints(self, states, geometry, retained):
        p, m = self.param, int(self.param.max_constraints)
        total, active_obstacles = 0, set()
        for k in range(self.N + 1):
            c, lg = np.zeros((m, 3)), np.full(m, -1e8)
            selected = []
            if k > 0:
                rows = {(r[1], r[2]): r for r in geometry[k]}
                # Whole polygon obstacles can give nonunique closest robot
                # points (parallel faces). Keep every tied support vertex and
                # retain discovered vertices to prevent alternating corner cuts.
                rot = rotation(states[k, 2])
                for (bi, oi), row in rows.items():
                    if row[0] > p.activation_distance:
                        continue
                    projections = (self.bodies[bi].vertices @ rot.T) @ row[3].normal
                    tied = np.flatnonzero(projections <= projections.min() + 1e-8)
                    retained[k].update((bi, oi, int(vi)) for vi in tied)
                # Pairs that leave the conservative window are certified distant.
                retained[k].intersection_update({key for key in retained[k]
                                                if key[:2] in rows})
                selected = [(rows[key[:2]], key[2]) for key in sorted(retained[k])]
                if len(selected) > m:
                    return False
                for i, ((_, bi, oi, witness, backoff), vi) in enumerate(selected):
                    witness = replace(witness, gamma=self.bodies[bi].vertices[vi])
                    c[i], lg[i] = self.linearized_clearance_row(
                        states[k], witness, p.polygon_dilation_radius + p.d_safe, backoff)
                    active_obstacles.add(oi)
                total += len(selected)
            self.solver.constraints_set(k, "C", c, api="new")
            self.solver.constraints_set(k, "lg", lg)
            if k < self.N:
                self.solver.set(k, "u", self._iteration_inputs[k])
            self.solver.set(k, "x", states[k])
        self.solver.set(0, "lbx", self._x0)
        self.solver.set(0, "ubx", self._x0)
        self.active_obstacle_count = len(active_obstacles)
        self.active_constraint_pair_count = total
        return True

    def solve_nlp(self):
        if self.solver is None:
            raise RuntimeError("Call setup() before solve_nlp()")
        start = time.perf_counter()
        self._solve_count += 1
        self.active_obstacle_count = self.active_constraint_pair_count = 0
        upper_time = 0.0
        status, success, reason, iteration = None, False, "maximum_iterations", 0
        iteration_history = []
        try:
            states, inputs = self._seed()
            retained = [set() for _ in range(self.N + 1)]
            covariances = self._propagate_covariances(states, inputs)
            geometry = self._geometry_at_nodes(states, covariances, include_initial=True)
            initial_gap = self._minimum_clearance([geometry[0]])
            if initial_gap < -self.param.feasibility_tolerance:
                reason = "initial_robust_clearance_violation"
            else:
                for iteration in range(1, int(self.param.max_outer_iterations) + 1):
                    self._iteration_inputs = inputs
                    if not self._update_constraints(states, geometry, retained):
                        reason = "constraint_capacity_exceeded"
                        break
                    upper_start = time.perf_counter()
                    status = int(self.solver.solve())
                    upper_time += time.perf_counter() - upper_start
                    # The original SIPOC accepts an SQP MAXITER candidate.
                    # It still must pass independent outer convergence and
                    # robust geometry checks before becoming a successful output.
                    if status != 0 and not (status == 2 and self.param.nlp_solver_type == "SQP"):
                        reason = f"acados_failure({status})"
                        break
                    new_inputs = np.stack([self.solver.get(k, "u") for k in range(self.N)])
                    if not np.all(np.isfinite(new_inputs)):
                        reason = "nonfinite_solution"
                        break
                    tol = self.param.feasibility_tolerance
                    if (np.any(new_inputs < np.array([self.param.vmin, self.param.omegamin]) - tol)
                            or np.any(new_inputs > np.array([self.param.vmax, self.param.omegamax]) + tol)):
                        reason = "input_bound_violation"
                        break
                    # Clip numerical bound tolerances, then validate the exact rollout.
                    new_inputs = np.clip(new_inputs, [self.param.vmin, self.param.omegamin],
                                         [self.param.vmax, self.param.omegamax])
                    proposed_states = self._rollout(new_inputs)
                    if not np.all(np.isfinite(proposed_states)):
                        reason = "nonfinite_rollout"
                        break
                    state_step = float(np.max(np.abs(proposed_states - states)))
                    input_step = float(np.max(np.abs(new_inputs - inputs)))
                    step = max(state_step, input_step)
                    alpha = self.param.outer_step_length
                    new_inputs = inputs + alpha * (new_inputs - inputs)
                    new_states = self._rollout(new_inputs)
                    states, inputs = new_states, new_inputs
                    covariances = self._propagate_covariances(states, inputs)
                    geometry = self._geometry_at_nodes(states, covariances, include_initial=True)
                    gap = self._minimum_clearance(geometry)
                    iteration_history.append({
                        "iteration": iteration, "state_step_inf": state_step,
                        "input_step_inf": input_step, "minimum_clearance": gap,
                        "upper_status": status, "outer_step_length": alpha,
                    })
                    if gap >= -tol and step <= self.param.step_tolerance:
                        success, reason = True, "success"
                        break
            minimum_clearance = self._minimum_clearance(geometry)
            if success:
                self._previous_inputs = inputs.copy()
                self._success_count += 1
            else:
                # Do not apply an infeasible iterate to the plant.
                inputs = np.zeros((self.N, 2))
                states = self._rollout(inputs)
                covariances = self._propagate_covariances(states, inputs)
                self._previous_inputs = None
                self._safe_stop_count += 1
            self._last_covariances = covariances.copy()
            self._last_diagnostics = {
                "outer_iterations": iteration, "upper_status": status,
                "minimum_candidate_clearance": minimum_clearance,
                "upper_solver_time_s": upper_time,
                "active_constraint_pairs": self.active_constraint_pair_count,
                "safe_stop": not success,
                "iteration_history": iteration_history,
            }
            return DilatedPolygonSolution(states, inputs, success, reason,
                                          self._last_diagnostics)
        finally:
            # One total algorithm timing per controller call, including inner solves.
            self.solver_times.append(time.perf_counter() - start)

    def get_last_covariance_trajectory(self):
        return None if self._last_covariances is None else self._last_covariances.copy()

    def get_last_risk_margin_trajectory(self):
        if self._last_covariances is None:
            return None
        radii = max(body.radius for body in self.bodies)
        shape = self.param.robust_scale * self._last_covariances
        translation = np.sqrt(np.maximum(0., np.linalg.eigvalsh(shape[:, :2, :2])[:, -1]))
        return translation + radii * np.sqrt(np.maximum(0., shape[:, 2, 2]))

    def record_plant_input_applied(self):
        self._plant_input_apply_count += 1

    def get_runtime_stats(self):
        return {"solve_calls": self._solve_count, "successful_solves": self._success_count,
                "safe_stop_count": self._safe_stop_count,
                "plant_input_apply_count": self._plant_input_apply_count,
                "problem_build_count": self.problem_build_count,
                **self._last_diagnostics}

    def _release_solver(self):
        self.solver, self.ocp = None, None
        if self._workspace is not None:
            self._workspace.cleanup()
            self._workspace = None

    def cleanup(self):
        self._release_solver()
        self._signature = None
        self._is_initialized = False
        self._previous_inputs = None
        self._last_covariances = None
        self._last_diagnostics = {}
        self.solver_times.clear()
        self._solve_count = self._success_count = self._safe_stop_count = 0
        self._plant_input_apply_count = 0
        self.problem_build_count = 0
        self.active_obstacle_count = self.active_constraint_pair_count = 0

    def reset(self):
        self.cleanup()


DilatedPolygonRobustOptimizer = DilatedPolygonRobustController
