import datetime

import casadi as ca
import numpy as np

from models.geometry_utils import *

class NmpcOptimizerParam:
    def __init__(self):
        self.horizon = 11
        self.horizon_dcbf = 11 # 6
        
        self.mat_Q = np.diag([100.0, 100.0, 0.0]) #np.diag([100.0, 100.0, 1.0, 1.0])
        self.mat_R = np.diag([20.0, 0.0])
        self.mat_Rold = np.diag([10.0, 1.0]) * 0.0  # np.diag([1.0, 1.0, 1.0]) * 0.0 
        self.mat_dR = np.diag([1.0, 1.0]) * 0.0 # np.diag([1.0, 1.0, 1.0]) * 0.0
        
        self.gamma = 0.8 # 0.8
        self.pomega = 10.0
        self.margin_dist = 0.01 # 0.0
        self.terminal_weight = 1.0 # 10.0


class NmpcOptimizer:
    def __init__(self, variables: dict, costs: dict, dynamics_opt):
        self.opti = None
        self.variables = variables
        self.costs = costs
        self.dynamics_opt = dynamics_opt
        self.solver_times = []

    def set_state(self, state):
        self.state = state

    def initialize_variables(self, param):
        self.variables["x"] = self.opti.variable(3, param.horizon + 1) # x y theta
        self.variables["u"] = self.opti.variable(2, param.horizon) # v omega

    def add_initial_condition_constraint(self):
        self.opti.subject_to(self.variables["x"][:, 0] == self.state._x)

    def add_input_constraint(self, param):
        vmin, vmax = -0.5, 0.5
        omegamin, omegamax = -1.2, 1.2
        for i in range(param.horizon):
            # input constraints
            self.opti.subject_to(self.variables["u"][0, i] <= vmax)
            self.opti.subject_to(vmin <= self.variables["u"][0, i])
            self.opti.subject_to(self.variables["u"][1, i] <= omegamax)
            self.opti.subject_to(omegamin <= self.variables["u"][1, i])

    def add_input_derivative_constraint(self, param):
        amin, amax = -0.2, 0.2
        omegadot_min, omegadot_max = -0.8, 0.8
        for i in range(param.horizon - 1):
            # input derivative constraints
            self.opti.subject_to(self.variables["u"][0, i + 1] - self.variables["u"][0, i] <= amax)
            self.opti.subject_to(self.variables["u"][0, i + 1] - self.variables["u"][0, i] >= amin)
            self.opti.subject_to(self.variables["u"][1, i + 1] - self.variables["u"][1, i] <= omegadot_max)
            self.opti.subject_to(self.variables["u"][1, i + 1] - self.variables["u"][1, i] >= omegadot_min)
        self.opti.subject_to(self.variables["u"][0, 0] - self.state._u[0] <= amax)
        self.opti.subject_to(self.variables["u"][0, 0] - self.state._u[0] >= amin)
        self.opti.subject_to(self.variables["u"][1, 0] - self.state._u[1] <= omegadot_max)
        self.opti.subject_to(self.variables["u"][1, 0] - self.state._u[1] >= omegadot_min)

    def add_dynamics_constraint(self, param):
        for i in range(param.horizon):
            self.opti.subject_to(
                self.variables["x"][:, i + 1] == self.dynamics_opt(self.variables["x"][:, i], self.variables["u"][:, i])
            )

    def add_reference_trajectory_tracking_cost(self, param, reference_trajectory):
        self.costs["reference_trajectory_tracking"] = 0
        for i in range(param.horizon - 1):
            x_diff = self.variables["x"][:, i] - reference_trajectory[i, :]
            self.costs["reference_trajectory_tracking"] += ca.mtimes(x_diff.T, ca.mtimes(param.mat_Q, x_diff))
        x_diff = self.variables["x"][:, -1] - reference_trajectory[-1, :]
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

    def add_obstacle_avoidance_constraint(self, param, system, obstacles_geo):
        # TODO : Implement the obstacle avoidance constraint with psdf
        pass
            


    def add_warm_start(self, param, system):
        # TODO : Adjust the arguments of nominal_safe_controller to be compatible with DDRturnDynamics 
        x_ws, u_ws = system._dynamics.nominal_safe_controller(self.state._x, 0.1,self.state._u[0], -1.0, 1.0)
        print("x_ws, u_ws :", x_ws, u_ws)
        for i in range(param.horizon):
            self.opti.set_initial(self.variables["x"][:, i + 1], x_ws)
            self.opti.set_initial(self.variables["u"][:, i], u_ws)

    def setup(self, param, system, reference_trajectory, obstacles):
        self.set_state(system._state)
        self.opti = ca.Opti()
        self.initialize_variables(param)
        self.add_initial_condition_constraint()
        self.add_input_constraint(param)
        self.add_input_derivative_constraint(param)
        self.add_dynamics_constraint(param)
        self.add_reference_trajectory_tracking_cost(param, reference_trajectory)
        self.add_input_stage_cost(param)
        self.add_prev_input_cost(param)
        self.add_input_smoothness_cost(param)
        self.add_obstacle_avoidance_constraint(param, system, obstacles)
        self.add_warm_start(param, system)

    def solve_nlp(self):
        print("Solving NMPC optimization problem")
        cost = 0
        for cost_name in self.costs:
            cost += self.costs[cost_name]
        self.opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0, "ipopt.linear_solver": "mumps"}
        start_timer = datetime.datetime.now()
        self.opti.solver("ipopt", option)
        opt_sol = self.opti.solve()
        print("return code : ", opt_sol.stats()["return_status"])
        end_timer = datetime.datetime.now()
        delta_timer = end_timer - start_timer
        self.solver_times.append(delta_timer.total_seconds())
        print("solver time: ", delta_timer.total_seconds())
        return opt_sol
