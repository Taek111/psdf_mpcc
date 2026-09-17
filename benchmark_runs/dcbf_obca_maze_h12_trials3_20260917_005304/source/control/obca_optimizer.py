import time

import casadi as ca
import numpy as np

from models.geometry_utils import (
    ConvexRegion2D,
    get_dist_point_to_region,
    get_dist_region_to_region,
)


class OBCAOptimizerParam:
    def __init__(self):
        self.horizon = 12
        # Match the common weights in NmpcDcbfOptimizerParam.
        self.mat_Q = np.diag([100.0, 100.0, 1.0])
        self.mat_R = np.diag([1.0, 0.0])
        self.mat_Rold = np.diag([1.0, 1.0]) * 0.0
        self.mat_dR = np.diag([1.0, 1.0]) * 0.0
        # Match PSDF d_safe; gamma_s is a separate PSDF approximation margin.
        self.margin_dist = 0.0001
        self.terminal_weight = 1.0
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

    def set_state(self, state):
        self.state = state

    def initialize_variables(self, param):
        self.variables["x"] = self.opti.variable(3, param.horizon + 1)
        self.variables["u"] = self.opti.variable(2, param.horizon)

    def add_initial_condition_constraint(self):
        self.opti.subject_to(self.variables["x"][:, 0] == self.state._x)

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
        omegadot_min, omegadot_max = -0.5, 0.5
        for i in range(param.horizon - 1):
            # input constraints
            self.opti.subject_to(self.variables["u"][0, i + 1] - self.variables["u"][0, i] <= jerk_max)
            self.opti.subject_to(self.variables["u"][0, i + 1] - self.variables["u"][0, i] >= jerk_min)
            self.opti.subject_to(self.variables["u"][1, i + 1] - self.variables["u"][1, i] <= omegadot_max)
            self.opti.subject_to(self.variables["u"][1, i + 1] - self.variables["u"][1, i] >= omegadot_min)
        self.opti.subject_to(self.variables["u"][0, 0] - self.state._u[0] <= jerk_max)
        self.opti.subject_to(self.variables["u"][0, 0] - self.state._u[0] >= jerk_min)
        self.opti.subject_to(self.variables["u"][1, 0] - self.state._u[1] <= omegadot_max)
        self.opti.subject_to(self.variables["u"][1, 0] - self.state._u[1] >= omegadot_min)

    def add_dynamics_constraint(self, param):
        for i in range(param.horizon):
            self.opti.subject_to(
                self.variables["x"][:, i + 1] == self.dynamics_opt(self.variables["x"][:, i], self.variables["u"][:, i])
            )

    def add_reference_trajectory_tracking_cost(self, param, reference_trajectory):
        self.costs["reference_trajectory_tracking"] = 0
        for i in range(param.horizon):
            x_diff = self.variables["x"][:, i] - reference_trajectory[i, :3]  # Take only first 3 elements
            self.costs["reference_trajectory_tracking"] += ca.mtimes(x_diff.T, ca.mtimes(param.mat_Q, x_diff))
        x_diff = self.variables["x"][:, -1] - reference_trajectory[-1, :3]  # Take only first 3 elements
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
        self.costs["prev_input"] = 0
        self.costs["prev_input"] += ca.mtimes(
            (self.variables["u"][:, 0] - self.state._u).T,
            ca.mtimes(param.mat_Rold, (self.variables["u"][:, 0] - self.state._u)),
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

    def add_convex_to_convex_constraint(self, param, robot_geo, obs_geo):
        mat_A, vec_b = obs_geo.get_convex_rep()
        robot_G, robot_g = robot_geo.get_convex_rep()
        # Current-pose duals are used only to initialize the NLP.
        _, lamb_curr, mu_curr = get_dist_region_to_region(
            mat_A,
            vec_b,
            np.dot(robot_G, self.state.rotation().T),
            np.dot(np.dot(robot_G, self.state.rotation().T), self.state.translation()) + robot_g,
        )
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
            # warm start
            self.opti.set_initial(lamb[:, i], lamb_curr)
            self.opti.set_initial(mu[:, i], mu_curr)

    def add_obstacle_avoidance_constraint(self, param, system, obstacles_geo):
        robot_components = system._geometry.equiv_rep()
        # Include every obstacle: a current-distance cutoff can omit obstacles
        # that the robot reaches later in the prediction horizon.
        for obs_geo in obstacles_geo:
            for robot_comp in robot_components:
                # TODO: need to add case for `add_point_convex_constraint()`
                if isinstance(robot_comp, ConvexRegion2D):
                    self.add_convex_to_convex_constraint(param, robot_comp, obs_geo)
                else:
                    raise NotImplementedError()

    def add_warm_start(self, param, system):
        # TODO: wrap params
        x_ws, u_ws = system._dynamics.nominal_safe_controller(self.state._x, 0.1, self.state._u[0], -1.0, 1.0)
        for i in range(param.horizon):
            self.opti.set_initial(self.variables["x"][:, i + 1], x_ws)
            self.opti.set_initial(self.variables["u"][:, i], u_ws)

    def setup(self, param, system, reference_trajectory, obstacles):
        self.set_state(system._state)
        self.opti = ca.Opti()
        self.initialize_variables(param)
        self.add_initial_condition_constraint()
        self.add_input_constraint(param)
        # self.add_input_derivative_constraint(param)
        self.add_dynamics_constraint(param)
        self.add_reference_trajectory_tracking_cost(param, reference_trajectory)
        self.add_input_stage_cost(param)
        self.add_prev_input_cost(param)
        self.add_input_smoothness_cost(param)
        self.add_obstacle_avoidance_constraint(param, system, obstacles)
        self.add_warm_start(param, system)

    def solve_nlp(self):
        cost = 0
        for cost_name in self.costs:
            cost += self.costs[cost_name]
        self.opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt", option)
        start_timer = time.perf_counter()
        opt_sol = self.opti.solve()
        solve_time = time.perf_counter() - start_timer
        self.solver_times.append(solve_time)
        print("solver time: ", solve_time)
        return opt_sol
