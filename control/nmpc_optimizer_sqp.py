import datetime
import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import time
import os

class NmpcOptimizerSqpParam:
    def __init__(self):
        self.horizon = 11
        self.horizon_dcbf = 11  # 6
        
        self.mat_Q = np.diag([100.0, 100.0, 0.0])  # np.diag([100.0, 100.0, 1.0, 1.0])
        self.mat_R = np.diag([20.0, 0.0])
        self.mat_Rold = np.diag([10.0, 1.0]) * 0.0  # np.diag([1.0, 1.0, 1.0]) * 0.0 
        self.mat_dR = np.diag([1.0, 1.0]) * 0.0  # np.diag([1.0, 1.0, 1.0]) * 0.0
        
        self.gamma = 0.8  # 0.8
        self.pomega = 10.0
        self.margin_dist = 0.01  # 0.0
        self.terminal_weight = 1.0  # 10.0
        
        # SQP specific parameters
        self.tf = 0.1 * self.horizon  # total time horizon
        self.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # qp solver to be used
        self.hessian_approx = 'GAUSS_NEWTON'  # Hessian approximation
        self.integrator_type = 'ERK'  # explicit Runge-Kutta
        self.nlp_solver_type = 'SQP_RTI'  # SQP solver
        self.qp_solver_iter_max = 50  # maximum iterations for QP solver
        self.nlp_solver_max_iter = 20  # maximum iterations for NLP solver
        self.tol = 1e-4  # convergence tolerance
        
        # Input constraints
        self.vmin, self.vmax = -0.5, 0.5
        self.omegamin, self.omegamax = -1.2, 1.2
        
        # Input derivative constraints
        self.amin, self.amax = -0.2, 0.2
        self.omegadot_min, self.omegadot_max = -0.8, 0.8


class NmpcOptimizerSqp:
    def __init__(self, variables=None, costs=None, dynamics_opt=None):
        self.ocp = None
        self.solver = None
        self.solver_times = []
        self.state = None
        self.reference_trajectory = None
        self.N = None
        self.nx = None
        self.nu = None
        self.variables = {}
        self.costs = {} if costs is None else costs
        self.dynamics_opt = dynamics_opt
        
        # Use common JSON filename - will be cleaned up after each test
        self.json_filename = "acados_ocp.json"
        
        # Store files to cleanup
        self._temp_files = []

    def cleanup(self):
        """Clean up temporary files and reset state"""
        try:
            # Clean up temporary files
            for file_path in self._temp_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Clean up JSON file
            if os.path.exists(self.json_filename):
                os.remove(self.json_filename)
                
            # Clean up acados generated directories
            cleanup_dirs = [
                "c_generated_code",
            ]
            for dir_name in cleanup_dirs:
                if os.path.exists(dir_name) and os.path.isdir(dir_name):
                    import shutil
                    shutil.rmtree(dir_name, ignore_errors=True)
            
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
        
        # Reset internal state
        self.ocp = None
        self.solver = None
        self.solver_times = []

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

    def reset(self):
        """Reset optimizer state for new test"""
        self.cleanup()

    def set_state(self, state):
        self.state = state

    def set_reference_trajectory(self, reference_trajectory):
        """Set the reference trajectory for the optimizer"""
        self.reference_trajectory = reference_trajectory
        if self.solver is not None and reference_trajectory is not None:
            # Set stage costs
            for i in range(self.N):
                yref = np.zeros(self.nx + self.nu)
                yref[:self.nx] = reference_trajectory[i, :]
                self.solver.set(i, "yref", yref)
            
            # Set terminal cost
            yref_e = reference_trajectory[-1, :]
            self.solver.set(self.N, "yref", yref_e)

    def create_model(self, param):
        """Create the acados model for the robot dynamics"""
        # State and input dimensions
        nx = 3  # x, y, theta
        nu = 2  # v, omega
        
        # Symbolic variables
        x = ca.SX.sym('x', nx)
        xdot = ca.SX.sym('xdot', nx)
        u = ca.SX.sym('u', nu)
        
        # Dynamics - differential drive robot
        f_expl = ca.vertcat(
            u[0] * ca.cos(x[2]),  # x_dot = v * cos(theta)
            u[0] * ca.sin(x[2]),  # y_dot = v * sin(theta)
            u[1]                  # theta_dot = omega
        )
        
        # Create acados model
        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = 'differential_drive_nmpc'
        
        return model

    def setup_ocp(self, param, reference_trajectory):
        """Setup the optimal control problem"""
        self.ocp = AcadosOcp()
        
        # Create model
        model = self.create_model(param)
        self.ocp.model = model
        
        # Dimensions
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        N = param.horizon
        self.N = N
        self.nx = nx
        self.nu = nu
        
        # Set dimensions
        self.ocp.dims.N = N
        
        # Set cost
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        
        # Weight matrices
        Q = param.mat_Q
        R = param.mat_R
        R_old = param.mat_Rold
        dR = param.mat_dR
        
        # Cost matrices for reference tracking and input
        self.ocp.cost.W = np.block([[Q, np.zeros((nx, nu))],
                                   [np.zeros((nu, nx)), R]])
        # Terminal cost (weighted by terminal_weight)
        self.ocp.cost.W_e = Q * param.terminal_weight
        
        # Reference tracking
        self.ocp.cost.Vx = np.zeros((nx + nu, nx))
        self.ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        self.ocp.cost.Vu = np.zeros((nx + nu, nu))
        self.ocp.cost.Vu[nx:, :] = np.eye(nu)
        self.ocp.cost.Vx_e = np.eye(nx)
        
        # Set reference trajectory
        self.ocp.cost.yref = np.zeros((nx + nu,))
        self.ocp.cost.yref_e = reference_trajectory[-1, :] if reference_trajectory.size > 0 else np.zeros(nx)
        
        # Set constraints
        # Input constraints
        self.ocp.constraints.lbu = np.array([param.vmin, param.omegamin])
        self.ocp.constraints.ubu = np.array([param.vmax, param.omegamax])
        self.ocp.constraints.idxbu = np.array([0, 1])
        
        # Initial state constraint
        self.ocp.constraints.x0 = self.state._x
        
        # Add input derivative constraints (rate constraints)
        # This requires setting up a custom constraint in acados
        # For simplicity, we'll use soft constraints via cost function
        
        # Set options
        self.ocp.solver_options.qp_solver = param.qp_solver
        self.ocp.solver_options.hessian_approx = param.hessian_approx
        self.ocp.solver_options.integrator_type = param.integrator_type
        self.ocp.solver_options.nlp_solver_type = param.nlp_solver_type
        
        # Set prediction horizon
        self.ocp.solver_options.tf = param.tf
        
        # Store reference trajectory
        self.reference_trajectory = reference_trajectory

    def create_solver(self):
        """Create the acados solver"""
        self.solver = AcadosOcpSolver(self.ocp, json_file=self.json_filename)
        self._temp_files.append(self.json_filename)
        self.variables["x"] = "x"
        self.variables["u"] = "u"

    def add_obstacle_avoidance_constraint(self, param, system, obstacles_geo):
        pass
        # TODO: Implement the obstacle avoidance constraint with PED-Net
        # This would require adding nonlinear constraints to the acados problem
        # May need a different approach than in the original CasADi implementation

    def add_warm_start(self, param, system):
        """Add warm start based on nominal safe controller"""
        if self.solver is None:
            return
        
        # Get warm start trajectory from nominal safe controller
        try:
            x_ws, u_ws = system._dynamics.nominal_safe_controller(
                self.state._x, 0.1, self.state._u[0], -1.0, 1.0
            )
            print("x_ws, u_ws :", x_ws, u_ws)
            
            # Set warm start for all horizons
            for i in range(self.N):
                # Set state warm start
                self.solver.set(i, "x", x_ws)
                # Set control warm start
                self.solver.set(i, "u", u_ws)
                
            # Set final state warm start
            self.solver.set(self.N, "x", x_ws)
            
        except Exception as e:
            print(f"Error in warm start: {e}")

    def setup(self, param, system, reference_trajectory, obstacles):
        """Setup the complete optimization problem"""
        self.set_state(system._state)
        self.setup_ocp(param, reference_trajectory)
        self.create_solver()
        self.set_reference_trajectory(reference_trajectory)
        self.add_obstacle_avoidance_constraint(param, system, obstacles)
        self.add_warm_start(param, system)

    def solve_nlp(self):
        """Solve the NLP using SQP"""
    
        # Ensure the initial state constraint is updated
        if self.state is not None:
            self.solver.set(0, "lbx", self.state._x)
            self.solver.set(0, "ubx", self.state._x)
        
        # Solve the optimization problem
        start = time.time()
        status = self.solver.solve()
        end = time.time()
        solve_time = end - start
        
        # Record solver time
        self.solver_times.append(solve_time)
        print("solver time: ", solve_time)
        
        if status != 0:
            print(f"Acados solver failed with status {status}")
        
        return AcadosSolution(self.solver, self.N, self.variables)


class AcadosSolution:
    """Helper class to mimic casadi solution interface"""
    
    def __init__(self, solver, N, variables):
        self.solver = solver
        self.N = N
        self.variables = variables
    
    def value(self, var_expr):
        """Get the value of a variable expression"""
        if var_expr == "x" or (isinstance(var_expr, str) and var_expr == "x"):
            return np.stack([self.solver.get(i, "x") for i in range(self.N + 1)], axis=1)
        elif var_expr == "u" or (isinstance(var_expr, str) and var_expr == "u"):
            return np.stack([self.solver.get(i, "u") for i in range(self.N)], axis=1)
        else:
            raise NotImplementedError("Only 'x' and 'u' supported in AcadosSolution.value")
    
    def get_state_trajectory(self):
        """Get the optimized state trajectory"""
        return self.value("x")
    
    def get_input_trajectory(self):
        """Get the optimized input trajectory"""
        return self.value("u")
    
    def stats(self):
        """Get solver statistics"""
        return {"return_status": "success" if self.solver.status == 0 else "failure"}
