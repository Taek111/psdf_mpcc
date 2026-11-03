import time

import casadi as ca
import numpy as np

from control.duality_optimizer_utils import (
    ReusableRegionDistanceQuery,
    freeze_value,
    validate_cutoff,
)
from models.geometry_utils import (
    ConvexRegion2D,
    get_dist_point_to_region,
)


class OBCAOptimizerParam:
    def __init__(self):
        self.horizon = 15
        # Match the common weights in NmpcDcbfOptimizerParam.
        self.mat_Q = np.diag([20.0, 20.0, 0.5])  # np.diag([100.0, 100.0, 1.0, 1.0])
        self.mat_R = np.diag([5.0, 0.0])
        self.mat_Rold = np.diag([10.0, 1.0]) * 0.0
        self.mat_dR = np.diag([1.0, 1.0]) * 0.0
        # Match PSDF d_safe; gamma_s is a separate PSDF approximation margin.
        self.margin_dist = 0.001
        self.safe_dist = 0.5
        self.use_obstacle_cutoff = True
        self.terminal_weight = 2.0
        self.vmin, self.vmax = -0.6, 0.6
        self.omegamin, self.omegamax = -1.0, 1.0


class OBCAOptimizer:
    """Enforce duality-based minimum distance at every predicted state."""

    def __init__(self, variables: dict, costs: dict, dynamics_opt):
        self.opti = None
        self.variables = variables
        self.costs = costs
        self.dynamics_opt = dynamics_opt
        self.solver_times = []
        self.active_obstacle_count = 0
        self.active_constraint_pair_count = 0
        self.problem_build_count = 0
        self._distance_query = None
        self._problem_signature = None
        self._parameters = {}
        self._dual_variables = []

    def set_state(self, state):
        self.state = state

    def initialize_variables(self, param):
        self.variables["x"] = self.opti.variable(3, param.horizon + 1)
        self.variables["u"] = self.opti.variable(2, param.horizon)

    def add_initial_condition_constraint(self):
        initial_state = self._parameters.get("x0", self.state._x)
        self.opti.subject_to(self.variables["x"][:, 0] == initial_state)

    def add_input_constraint(self, param):
        for i in range(param.horizon):
            # input constraints
            self.opti.subject_to(self.variables["u"][0, i] <= param.vmax)
            self.opti.subject_to(param.vmin <= self.variables["u"][0, i])
            self.opti.subject_to(self.variables["u"][1, i] <= param.omegamax)
            self.opti.subject_to(param.omegamin <= self.variables["u"][1, i])

    def add_input_derivative_constraint(self, param):
        # TODO: Remove this hardcoded function with timestep
        jerk_min, jerk_max = -1.0, 1.0
        omegadot_min, omegadot_max = -1.0, 1.0
        previous_input = self._parameters.get("u_prev", self.state._u)
        for i in range(param.horizon - 1):
            # input constraints
            self.opti.subject_to(self.variables["u"][0, i + 1] - self.variables["u"][0, i] <= jerk_max)
            self.opti.subject_to(self.variables["u"][0, i + 1] - self.variables["u"][0, i] >= jerk_min)
            self.opti.subject_to(self.variables["u"][1, i + 1] - self.variables["u"][1, i] <= omegadot_max)
            self.opti.subject_to(self.variables["u"][1, i + 1] - self.variables["u"][1, i] >= omegadot_min)
        self.opti.subject_to(self.variables["u"][0, 0] - previous_input[0] <= jerk_max)
        self.opti.subject_to(self.variables["u"][0, 0] - previous_input[0] >= jerk_min)
        self.opti.subject_to(self.variables["u"][1, 0] - previous_input[1] <= omegadot_max)
        self.opti.subject_to(self.variables["u"][1, 0] - previous_input[1] >= omegadot_min)

    def add_dynamics_constraint(self, param):
        for i in range(param.horizon):
            self.opti.subject_to(
                self.variables["x"][:, i + 1] == self.dynamics_opt(self.variables["x"][:, i], self.variables["u"][:, i])
            )

    def add_reference_trajectory_tracking_cost(self, param, reference_trajectory):
        self.costs["reference_trajectory_tracking"] = 0
        for i in range(param.horizon):
            reference_state = ca.reshape(reference_trajectory[i, :3], 3, 1)
            x_diff = self.variables["x"][:, i] - reference_state
            self.costs["reference_trajectory_tracking"] += ca.mtimes(x_diff.T, ca.mtimes(param.mat_Q, x_diff))
        terminal_reference = ca.reshape(reference_trajectory[-1, :3], 3, 1)
        x_diff = self.variables["x"][:, -1] - terminal_reference
        self.costs["reference_trajectory_tracking"] += param.terminal_weight * ca.mtimes(
            x_diff.T, ca.mtimes(param.mat_Q, x_diff)
        )

    def add_input_stage_cost(self, param):
        self.costs["input_stage"] = 0
        for i in range(param.horizon):
            self.costs["input_stage"] += ca.mtimes(
                self.variables["u"][:, i].T, ca.mtimes(param.mat_R, self.variables["u"][:, i])
            )

    def add_prev_input_cost(self, param):
        previous_input = self._parameters.get("u_prev", self.state._u)
        self.costs["prev_input"] = 0
        self.costs["prev_input"] += ca.mtimes(
            (self.variables["u"][:, 0] - previous_input).T,
            ca.mtimes(param.mat_Rold, (self.variables["u"][:, 0] - previous_input)),
        )

    def add_input_smoothness_cost(self, param):
        self.costs["input_smoothness"] = 0
        for i in range(param.horizon - 1):
            self.costs["input_smoothness"] += ca.mtimes(
                (self.variables["u"][:, i + 1] - self.variables["u"][:, i]).T,
                ca.mtimes(param.mat_dR, (self.variables["u"][:, i + 1] - self.variables["u"][:, i])),
            )

    def add_point_to_convex_constraint(self, param, obs_geo):
        mat_A, vec_b = obs_geo.get_convex_rep()
        # Current-pose duals are used only to initialize the NLP.
        _, lamb_curr = get_dist_point_to_region(self.state._x[0:2], mat_A, vec_b)
        lamb = self.opti.variable(mat_A.shape[0], param.horizon)
        for i in range(param.horizon):
            self.opti.subject_to(lamb[:, i] >= 0)
            self.opti.subject_to(
                ca.mtimes((ca.mtimes(mat_A, self.variables["x"][0:2, i + 1]) - vec_b).T, lamb[:, i])
                >= param.margin_dist
            )
            temp = ca.mtimes(mat_A.T, lamb[:, i])
            self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
            # warm start
            self.opti.set_initial(lamb[:, i], lamb_curr)

    def _query_region_distance(self, mat_A, vec_b, world_G, world_g):
        if self._distance_query is None:
            self._distance_query = ReusableRegionDistanceQuery()
        return self._distance_query.distance(mat_A, vec_b, world_G, world_g)

    @staticmethod
    def _geometry_representation(geometry):
        matrix, vector = geometry.get_convex_rep()
        return (
            np.asarray(matrix, dtype=float).copy(),
            np.asarray(vector, dtype=float).reshape(-1, 1).copy(),
        )

    def _make_constraint_pair(self, robot_geo, obs_geo):
        mat_A, vec_b = self._geometry_representation(obs_geo)
        robot_G, robot_g = self._geometry_representation(robot_geo)
        world_G = robot_G @ self.state.rotation().T
        world_g = world_G @ self.state.translation() + robot_g
        distance, lamb_curr, mu_curr = self._query_region_distance(
            mat_A, vec_b, world_G, world_g
        )
        return {
            "mat_A": mat_A,
            "vec_b": vec_b,
            "robot_G": robot_G,
            "robot_g": robot_g,
            "distance": distance,
            "lamb_curr": lamb_curr,
            "mu_curr": mu_curr,
        }

    def _collect_constraint_pairs(self, param, system, obstacles_geo):
        cutoff = validate_cutoff(param)
        robot_components = system._geometry.equiv_rep()
        selected_pairs = []
        active_obstacles = set()
        for obstacle_index, obs_geo in enumerate(obstacles_geo):
            for robot_comp in robot_components:
                if not isinstance(robot_comp, ConvexRegion2D):
                    raise NotImplementedError()
                pair = self._make_constraint_pair(robot_comp, obs_geo)
                if cutoff is not None and pair["distance"] > cutoff:
                    continue
                selected_pairs.append(pair)
                active_obstacles.add(obstacle_index)
        self.active_obstacle_count = len(active_obstacles)
        self.active_constraint_pair_count = len(selected_pairs)
        return selected_pairs

    def _add_convex_pair_constraint(self, param, pair):
        mat_A, vec_b = pair["mat_A"], pair["vec_b"]
        robot_G, robot_g = pair["robot_G"], pair["robot_g"]
        # OBCA distance certificate for B(x) = R(x) B + t(x), B = {y: Gy <= g}.
        lamb = self.opti.variable(mat_A.shape[0], param.horizon)
        mu = self.opti.variable(robot_G.shape[0], param.horizon)
        for i in range(param.horizon):
            robot_R = ca.hcat(
                [
                    ca.vcat(
                        [
                            ca.cos(self.variables["x"][2, i + 1]),
                            ca.sin(self.variables["x"][2, i + 1]),
                        ]
                    ),
                    ca.vcat(
                        [
                            -ca.sin(self.variables["x"][2, i + 1]),
                            ca.cos(self.variables["x"][2, i + 1]),
                        ]
                    ),
                ]
            )
            robot_T = self.variables["x"][0:2, i + 1]
            self.opti.subject_to(lamb[:, i] >= 0)
            self.opti.subject_to(mu[:, i] >= 0)
            self.opti.subject_to(
                -ca.mtimes(robot_g.T, mu[:, i]) + ca.mtimes((ca.mtimes(mat_A, robot_T) - vec_b).T, lamb[:, i])
                >= param.margin_dist
            )
            self.opti.subject_to(
                ca.mtimes(robot_G.T, mu[:, i]) + ca.mtimes(ca.mtimes(robot_R.T, mat_A.T), lamb[:, i]) == 0
            )
            temp = ca.mtimes(mat_A.T, lamb[:, i])
            self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
        self._dual_variables.append((lamb, mu))
        self._set_dual_initial(param, lamb, mu, pair)

    def _set_dual_initial(self, param, lamb, mu, pair):
        lamb_curr = np.asarray(pair["lamb_curr"], dtype=float).reshape(-1, 1)
        mu_curr = np.asarray(pair["mu_curr"], dtype=float).reshape(-1, 1)
        self.opti.set_initial(lamb, np.repeat(lamb_curr, param.horizon, axis=1))
        self.opti.set_initial(mu, np.repeat(mu_curr, param.horizon, axis=1))

    def add_convex_to_convex_constraint(self, param, robot_geo, obs_geo):
        """Add one pair unconditionally; setup handles obstacle selection."""
        pair = self._make_constraint_pair(robot_geo, obs_geo)
        self._add_convex_pair_constraint(param, pair)

    def add_obstacle_avoidance_constraint(
        self, param, system, obstacles_geo, selected_pairs=None
    ):
        if selected_pairs is None:
            selected_pairs = self._collect_constraint_pairs(param, system, obstacles_geo)
        for pair in selected_pairs:
            self._add_convex_pair_constraint(param, pair)

    def add_warm_start(self, param, system):
        # TODO: wrap params
        x_ws, u_ws = system._dynamics.nominal_safe_controller(self.state._x, 0.1, self.state._u[0], -1.0, 1.0)
        self.opti.set_initial(self.variables["x"][:, 0], self.state._x)
        for i in range(param.horizon):
            self.opti.set_initial(self.variables["x"][:, i + 1], x_ws)
            self.opti.set_initial(self.variables["u"][:, i], u_ws)

    @staticmethod
    def _pack_reference(param, reference_trajectory):
        reference = np.asarray(reference_trajectory, dtype=float)
        if (
            reference.ndim != 2
            or reference.shape[0] < param.horizon
            or reference.shape[1] < 3
        ):
            raise ValueError("Reference trajectory must have at least horizon rows and 3 columns")
        # Preserve stage references 0..N-1 and the original final reference.
        return np.vstack((reference[:param.horizon, :3], reference[-1, :3])).T

    def _graph_signature(self, param, selected_pairs):
        geometry_signature = tuple(
            freeze_value((pair["mat_A"], pair["vec_b"], pair["robot_G"], pair["robot_g"]))
            for pair in selected_pairs
        )
        return freeze_value(vars(param)), geometry_signature, id(self.dynamics_opt)

    def _build_problem(self, param, system, obstacles, selected_pairs):
        self._problem_signature = None
        self.opti = ca.Opti()
        self.variables.clear()
        self.costs.clear()
        self._dual_variables = []
        self._parameters = {
            "x0": self.opti.parameter(3),
            "u_prev": self.opti.parameter(2),
            "reference": self.opti.parameter(3, param.horizon + 1),
        }
        self.initialize_variables(param)
        self.add_initial_condition_constraint()
        self.add_input_constraint(param)
        # self.add_input_derivative_constraint(param)
        self.add_dynamics_constraint(param)
        self.add_reference_trajectory_tracking_cost(param, self._parameters["reference"].T)
        self.add_input_stage_cost(param)
        self.add_prev_input_cost(param)
        self.add_input_smoothness_cost(param)
        self.add_obstacle_avoidance_constraint(param, system, obstacles, selected_pairs)
        cost = 0
        for cost_name in self.costs:
            cost += self.costs[cost_name]
        self.opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt", option)
        self.problem_build_count += 1

    def setup(self, param, system, reference_trajectory, obstacles):
        self.set_state(system._state)
        packed_reference = self._pack_reference(param, reference_trajectory)
        selected_pairs = self._collect_constraint_pairs(param, system, obstacles)
        signature = self._graph_signature(param, selected_pairs)
        if self.opti is None or self._problem_signature != signature:
            self._build_problem(param, system, obstacles, selected_pairs)
            self._problem_signature = signature

        self.opti.set_value(self._parameters["x0"], self.state._x)
        self.opti.set_value(self._parameters["u_prev"], self.state._u)
        self.opti.set_value(self._parameters["reference"], packed_reference)
        for (lamb, mu), pair in zip(self._dual_variables, selected_pairs):
            self._set_dual_initial(param, lamb, mu, pair)
        self.add_warm_start(param, system)

    def solve_nlp(self):
        start_timer = time.perf_counter()
        opt_sol = self.opti.solve()
        solve_time = time.perf_counter() - start_timer
        self.solver_times.append(solve_time)
        print("solver time: ", solve_time)
        return opt_sol
