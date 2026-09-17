import time

import casadi as ca
import numpy as np

from control.duality_optimizer_utils import (
    ReusableRegionDistanceQuery,
    freeze_value,
    validate_cutoff,
)
from models.geometry_utils import *


class NmpcDcbfOptimizerParam:
    def __init__(self):
        self.horizon = 15
        self.horizon_dcbf = 10
        # Match the tracking and input weights in PSDFOptimizerParam.
        self.mat_Q = np.diag([20.0, 20.0, 0.5])  # np.diag([100.0, 100.0, 1.0, 1.0])
        self.mat_R = np.diag([5.0, 0.0])
        self.mat_Rold = np.diag([10.0, 1.0]) * 0.0
        self.mat_dR = np.diag([1.0, 1.0]) * 0.0
        self.gamma = 0.8  # 0.8
        self.pomega = 2.0
        self.margin_dist = 0.0001  # 0.0
        self.terminal_weight = 2.0  # 10.0
        self.vmin, self.vmax = -0.6, 0.6
        self.omegamin, self.omegamax = -1.0, 1.0
        self.safe_dist = 0.5
        self.use_obstacle_cutoff = True


class NmpcDbcfOptimizer:
    def __init__(self, variables: dict, costs: dict, dynamics_opt):
        self.opti = None
        self.variables = variables
        self.costs = costs
        self.dynamics_opt = dynamics_opt
        self.solver_times = []
        self.parameters = {}
        self._distance_query = None
        self._problem_cache_key = None
        self._pair_bindings = []
        self.active_obstacle_count = 0
        self.active_constraint_pair_count = 0
        self.problem_build_count = 0

    def set_state(self, state):
        self.state = state

    def initialize_variables(self, param):
        self.variables["x"] = self.opti.variable(3, param.horizon + 1)
        self.variables["u"] = self.opti.variable(2, param.horizon)

    def add_initial_condition_constraint(self):
        initial_state = self.parameters.get("x0", self.state._x)
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
        for i in range(param.horizon - 1):
            # input constraints
            self.opti.subject_to(self.variables["u"][0, i + 1] - self.variables["u"][0, i] <= jerk_max)
            self.opti.subject_to(self.variables["u"][0, i + 1] - self.variables["u"][0, i] >= jerk_min)
            self.opti.subject_to(self.variables["u"][1, i + 1] - self.variables["u"][1, i] <= omegadot_max)
            self.opti.subject_to(self.variables["u"][1, i + 1] - self.variables["u"][1, i] >= omegadot_min)
        previous_input = self.parameters.get("previous_input", self.state._u)
        self.opti.subject_to(self.variables["u"][0, 0] - previous_input[0] <= jerk_max)
        self.opti.subject_to(self.variables["u"][0, 0] - previous_input[0] >= jerk_min)
        self.opti.subject_to(self.variables["u"][1, 0] - previous_input[1] <= omegadot_max)
        self.opti.subject_to(self.variables["u"][1, 0] - previous_input[1] >= omegadot_min)

    def add_dynamics_constraint(self, param):
        for i in range(param.horizon):
            self.opti.subject_to(
                self.variables["x"][:, i + 1] == self.dynamics_opt(self.variables["x"][:, i], self.variables["u"][:, i])
            )

    def add_reference_trajectory_tracking_cost(self, param, reference_trajectory=None):
        reference = self.parameters.get("reference")
        if reference is None:
            reference = ca.DM(np.asarray(reference_trajectory)[:, :3].T)
        self.costs["reference_trajectory_tracking"] = 0
        for i in range(param.horizon):
            x_diff = self.variables["x"][:, i] - reference[:, i]
            self.costs["reference_trajectory_tracking"] += ca.mtimes(x_diff.T, ca.mtimes(param.mat_Q, x_diff))
        x_diff = self.variables["x"][:, -1] - reference[:, -1]
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
        previous_input = self.parameters.get("previous_input", self.state._u)
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

    def add_point_to_convex_constraint(self, param, obs_geo, safe_dist):
        # get current value of cbf
        mat_A, vec_b = obs_geo.get_convex_rep()
        cbf_curr, lamb_curr = get_dist_point_to_region(self.state._x[0:2], mat_A, vec_b)
        # filter obstacle if it's still far away
        if cbf_curr > safe_dist:
            return
        # duality-cbf constraints
        lamb = self.opti.variable(mat_A.shape[0], param.horizon_dcbf)
        omega = self.opti.variable(param.horizon_dcbf, 1)
        for i in range(param.horizon_dcbf):
            self.opti.subject_to(lamb[:, i] >= 0)
            self.opti.subject_to(
                ca.mtimes((ca.mtimes(mat_A, self.variables["x"][0:2, i + 1]) - vec_b).T, lamb[:, i])
                >= omega[i] * param.gamma ** (i + 1) * (cbf_curr - param.margin_dist) + param.margin_dist
            )
            temp = ca.mtimes(mat_A.T, lamb[:, i])
            self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
            self.opti.subject_to(omega[i] >= 0)
            self.costs["decay_rate_relaxing"] += param.pomega * (omega[i] - 1) ** 2
            # warm start
            self.opti.set_initial(lamb[:, i], lamb_curr)
            self.opti.set_initial(omega[i], 0.1)

    def _measure_pair(self, robot_geo, obs_geo):
        mat_A, vec_b = obs_geo.get_convex_rep()
        robot_G, robot_g = robot_geo.get_convex_rep()
        mat_A = np.asarray(mat_A, dtype=float).copy()
        vec_b = np.asarray(vec_b, dtype=float).reshape(-1, 1).copy()
        robot_G = np.asarray(robot_G, dtype=float).copy()
        robot_g = np.asarray(robot_g, dtype=float).reshape(-1, 1).copy()
        if self._distance_query is None:
            self._distance_query = ReusableRegionDistanceQuery()
        robot_world_G = robot_G @ self.state.rotation().T
        cbf_curr, lamb_curr, mu_curr = self._distance_query.distance(
            mat_A,
            vec_b,
            robot_world_G,
            robot_world_G @ self.state.translation() + robot_g,
        )
        return {
            "mat_A": mat_A,
            "vec_b": vec_b,
            "robot_G": robot_G,
            "robot_g": robot_g,
            "cbf_curr": cbf_curr,
            "lamb_curr": lamb_curr,
            "mu_curr": mu_curr,
        }

    def _select_constraint_pairs(self, param, system, obstacles_geo):
        cutoff = validate_cutoff(param)
        robot_components = system._geometry.equiv_rep()
        selected_pairs = []
        for obstacle_index, obs_geo in enumerate(obstacles_geo):
            for robot_index, robot_comp in enumerate(robot_components):
                if not isinstance(robot_comp, ConvexRegion2D):
                    raise NotImplementedError()
                pair = self._measure_pair(robot_comp, obs_geo)
                if cutoff is not None and pair["cbf_curr"] > cutoff:
                    continue
                pair["obstacle_index"] = obstacle_index
                pair["robot_index"] = robot_index
                selected_pairs.append(pair)
        self.active_obstacle_count = len({pair["obstacle_index"] for pair in selected_pairs})
        self.active_constraint_pair_count = len(selected_pairs)
        return selected_pairs

    def add_convex_to_convex_constraint(self, param, robot_geo, obs_geo, safe_dist, pair_data=None):
        pair = pair_data if pair_data is not None else self._measure_pair(robot_geo, obs_geo)
        if safe_dist is not None and pair["cbf_curr"] > safe_dist:
            return
        mat_A, vec_b = pair["mat_A"], pair["vec_b"]
        robot_G, robot_g = pair["robot_G"], pair["robot_g"]
        # The graph can be reused, but the barrier's current clearance cannot.
        cbf_curr = self.opti.parameter()
        # duality-cbf constraints
        lamb = self.opti.variable(mat_A.shape[0], param.horizon_dcbf)
        mu = self.opti.variable(robot_G.shape[0], param.horizon_dcbf)
        omega = self.opti.variable(param.horizon_dcbf, 1)  # Fixed dimension
        binding = {"cbf_curr": cbf_curr, "lamb": lamb, "mu": mu, "omega": omega}
        self._pair_bindings.append(binding)
        for i in range(param.horizon_dcbf):
            robot_R = ca.hcat(
                [
                    ca.vcat(
                        [
                            ca.cos(self.variables["x"][2, i + 1]),  # Changed from index 3 to 2
                            ca.sin(self.variables["x"][2, i + 1]),  # Changed from index 3 to 2
                        ]
                    ),
                    ca.vcat(
                        [
                            -ca.sin(self.variables["x"][2, i + 1]),  # Changed from index 3 to 2
                            ca.cos(self.variables["x"][2, i + 1]),  # Changed from index 3 to 2
                        ]
                    ),
                ]
            )
            robot_T = self.variables["x"][0:2, i + 1]
            self.opti.subject_to(lamb[:, i] >= 0)
            self.opti.subject_to(mu[:, i] >= 0)
            self.opti.subject_to(
                -ca.mtimes(robot_g.T, mu[:, i]) + ca.mtimes((ca.mtimes(mat_A, robot_T) - vec_b).T, lamb[:, i])
                >= omega[i] * param.gamma ** (i + 1) * (cbf_curr - param.margin_dist) + param.margin_dist
            )
            self.opti.subject_to(
                ca.mtimes(robot_G.T, mu[:, i]) + ca.mtimes(ca.mtimes(robot_R.T, mat_A.T), lamb[:, i]) == 0
            )
            temp = ca.mtimes(mat_A.T, lamb[:, i])
            self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
            self.opti.subject_to(omega[i] >= 0)
            self.costs["decay_rate_relaxing"] += param.pomega * (omega[i] - 1) ** 2
        self._update_pair_values(binding, pair, param)

    def _update_pair_values(self, binding, pair, param):
        self.opti.set_value(binding["cbf_curr"], pair["cbf_curr"])
        for i in range(param.horizon_dcbf):
            self.opti.set_initial(binding["lamb"][:, i], pair["lamb_curr"])
            self.opti.set_initial(binding["mu"][:, i], pair["mu_curr"])
            self.opti.set_initial(binding["omega"][i], 0.1)

    def add_obstacle_avoidance_constraint(self, param, system, obstacles_geo, selected_pairs=None):
        self.costs["decay_rate_relaxing"] = 0
        if selected_pairs is None:
            selected_pairs = self._select_constraint_pairs(param, system, obstacles_geo)
        for pair in selected_pairs:
            self.add_convex_to_convex_constraint(param, None, None, None, pair_data=pair)

    def add_warm_start(self, param, system):
        # TODO: wrap params
        x_ws, u_ws = system._dynamics.nominal_safe_controller(self.state._x, 0.1, self.state._u[0], -1.0, 1.0)
        self.opti.set_initial(self.variables["x"][:, 0], self.state._x)
        for i in range(param.horizon):
            self.opti.set_initial(self.variables["x"][:, i + 1], x_ws)
            self.opti.set_initial(self.variables["u"][:, i], u_ws)

    def _build_problem(self, param, selected_pairs):
        self._problem_cache_key = None
        self.opti = ca.Opti()
        self.variables.clear()
        self.costs.clear()
        self._pair_bindings = []
        self.parameters = {
            "x0": self.opti.parameter(3),
            "previous_input": self.opti.parameter(2),
            "reference": self.opti.parameter(3, param.horizon + 1),
        }
        self.initialize_variables(param)
        self.add_initial_condition_constraint()
        self.add_input_constraint(param)
        # self.add_input_derivative_constraint(param)
        self.add_dynamics_constraint(param)
        self.add_reference_trajectory_tracking_cost(param)
        self.add_input_stage_cost(param)
        self.add_prev_input_cost(param)
        self.add_input_smoothness_cost(param)
        self.add_obstacle_avoidance_constraint(param, None, None, selected_pairs=selected_pairs)
        self.opti.minimize(sum(self.costs.values()))
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt", option)
        self.problem_build_count += 1

    def setup(self, param, system, reference_trajectory, obstacles):
        self.set_state(system._state)
        reference = np.asarray(reference_trajectory, dtype=float)
        if reference.ndim != 2 or reference.shape[0] < param.horizon or reference.shape[1] < 3:
            raise ValueError("DCBF reference must have at least horizon rows and three state columns.")
        # Retain the existing terminal reference even when its horizon differs from N.
        reference_values = np.vstack((reference[:param.horizon, :3], reference[-1, :3])).T
        selected_pairs = self._select_constraint_pairs(param, system, obstacles)
        # State, input, reference, and current clearances are parameters, not
        # structural cache inputs. Numerical halfspaces detect geometry edits.
        pair_geometry = tuple(
            freeze_value((pair["mat_A"], pair["vec_b"], pair["robot_G"], pair["robot_g"]))
            for pair in selected_pairs
        )
        cache_key = (freeze_value(vars(param)), pair_geometry, id(self.dynamics_opt))
        if self.opti is None or self._problem_cache_key != cache_key:
            self._build_problem(param, selected_pairs)
            self._problem_cache_key = cache_key
        self.opti.set_value(self.parameters["x0"], self.state._x)
        self.opti.set_value(self.parameters["previous_input"], self.state._u)
        self.opti.set_value(self.parameters["reference"], reference_values)
        for binding, pair in zip(self._pair_bindings, selected_pairs):
            self._update_pair_values(binding, pair, param)
        self.add_warm_start(param, system)

    def solve_nlp(self):
        start_timer = time.perf_counter()
        opt_sol = self.opti.solve()
        solve_time = time.perf_counter() - start_timer
        self.solver_times.append(solve_time)
        print("solver time: ", solve_time)
        return opt_sol
