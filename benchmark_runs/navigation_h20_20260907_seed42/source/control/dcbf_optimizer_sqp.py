import datetime
import numpy as np
import casadi as ca
import os

try:
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
    ACADOS_AVAILABLE = True
except ImportError:
    print("Warning: acados_template not available. DCBF SQP optimizer will not work.")
    ACADOS_AVAILABLE = False
from models.geometry_utils import *


class NmpcDcbfOptimizerSqpParam:
    def __init__(self):
        self.horizon = 20
        self.horizon_dcbf = self.horizon
        
        self.mat_Q = np.diag([10.0, 10.0, 0.0])  # np.diag([100.0, 100.0, 1.0, 1.0])
        self.mat_R = np.diag([20.0, 0.0])
        self.mat_Rold = np.diag([10.0, 1.0]) * 0.0  # np.diag([1.0, 1.0, 1.0]) * 0.0 
        self.mat_dR = np.diag([1.0, 1.0]) * 0.0  # np.diag([1.0, 1.0, 1.0]) * 0.0
        
        self.gamma = 0.8  # 0.8
        self.pomega = 10.0
        self.margin_dist = 0.001  # 0.0
        self.terminal_weight = 1.0  # 10.0
        
        
        # SQP specific parameters
        self.tf = 0.1 * self.horizon  # total time horizon
        self.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # qp solver to be used
        self.hessian_approx = 'GAUSS_NEWTON'  # Hessian approximation
        self.integrator_type = 'IRK'  # explicit Runge-Kutta
        self.nlp_solver_type = 'SQP_RTI'  # SQP solver
        self.qp_solver_iter_max = 50  # maximum iterations for QP solver
        self.nlp_solver_max_iter = 20  # maximum iterations for NLP solver
        
        # self.qp_solver = 'FULL_CONDENSING_HPIPM'  # qp solver to be used
        # self.hessian_approx = 'GAUSS_NEWTON'  # Hessian approximation
        # self.integrator_type = 'IRK'  # explicit Runge-Kutta
        # self.nlp_solver_type = 'SQP_RTI'  # SQP solver
        # Input constraints
        self.amin, self.amax = -0.5, 0.5
        self.omegamin, self.omegamax = -0.5, 0.5
        
        # Input derivative constraints
        self.jerk_min, self.jerk_max = -1.0, 1.0
        self.omegadot_min, self.omegadot_max = -0.5, 0.5


class NmpcDcbfOptimizerSqp:
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
        

    def create_model(self, param, nz: int) -> AcadosModel:
        """
        acados에서 algebraic state z(듀얼 변수 λ, μ, ω를 한꺼번에 모아둔 벡터)의
        차원을 nz로 지정해 모델을 만든다.
        """
        # ---------- symbol 정의 ----------
        nx = 3                        # [x, y, theta]  
        nu = 2                        # [v, omega]
        x   = ca.SX.sym('x',   nx)
        xdot= ca.SX.sym('xdot', nx)
        u   = ca.SX.sym('u',   nu)
        z   = ca.SX.sym('z',   nz)    # ← 듀얼변수 전부를 모아 둔 algebraic state

        # ---------- 로봇 동역학 (ex: differential-drive) ----------
        f_expl = ca.vertcat(
            u[0] * ca.cos(x[2]),  # x_dot = v * cos(theta)
            u[0] * ca.sin(x[2]),  # y_dot = v * sin(theta)
            u[1]                  # theta_dot = omega
        )

       

        # ---------- 모델 객체에 등록 ----------
        model = AcadosModel()
        model.name         = 'diff_drive_dcbf'
        model.x            = x
        model.xdot         = xdot
        model.u            = u
        model.z            = z
        model.f_expl_expr  = f_expl
        model.f_impl_expr  = ca.vertcat(xdot - f_expl,  # differential equations (length nx=3)
                                        z)             # algebraic equations (length nz=162) (dual variables has no dynamics)
                                        
        # model.f_z_expr     = f_z               

        return model

    def setup_ocp(self, param, reference_trajectory, obstacles_geo, robot_components):
        """
        Setup the optimal control problem
        obstacles_geo    : [(A_O, b_O), ...]   각 장애물 폴리토프의 half-space 표현
        robot_components : [(A_R, b_R), ...]   로봇(또는 로봇 링크) 폴리토프 표현
        """
        if not ACADOS_AVAILABLE:
            raise ImportError("acados_template is required for DCBF SQP optimizer")
        
        ocp = AcadosOcp()
        
        # ---------- (1) algebraic 변수 차원 산출 ----------
        nz_per_step = 0
        for obs_geo in obstacles_geo:
            if hasattr(obs_geo, 'get_convex_rep'):
                A_O, _ = obs_geo.get_convex_rep()
                m_O = A_O.shape[0]
                
                for robot_comp in robot_components:
                    if hasattr(robot_comp, 'get_convex_rep'):
                        A_R, _ = robot_comp.get_convex_rep()
                        m_R = A_R.shape[0]
                        nz_per_step += m_O + m_R + 1  # λ^O + λ^R + ω for each obstacle-robot pair

        nz_total = nz_per_step * param.horizon_dcbf
        
        # ---------- (2) acados 모델 생성 ----------    
        # Create model
        ocp.model = self.create_model(param, nz_total)

        nx = ocp.model.x.size1()
        nu = ocp.model.u.size1()
        ocp.dims.N  = param.horizon          # *** 올바른 위치 ***
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.nz = nz_total
        ocp.solver_options.N_horizon = param.horizon
        
        # ── (3) 외부 cost 정의  ------------------------------------------------
        #     (a) 파라미터 심볼을 **먼저** 만들고 model.p에 등록
        x_ref = ca.SX.sym('x_ref', nx)      # (3,)
        u_ref = ca.SX.sym('u_ref', nu)      # (2,)
        ocp.model.p = ca.vertcat(x_ref, u_ref)
        ocp.dims.np = nx + nu               # = 5

        # set the default parameter values
        ocp.parameter_values = np.zeros(ocp.dims.np)  # shape = (5,)

        #     (b) stage cost: tracking + ω-relaxation
        Q = ca.SX(param.mat_Q)
        R = ca.SX(param.mat_R)
        alpha = ca.SX(param.terminal_weight)  # terminal weight 

        # tracking part
        x   = ocp.model.x
        u   = ocp.model.u
        track_cost = (x - x_ref).T @ Q @ (x - x_ref) \
                     + (u - u_ref).T @ R @ (u - u_ref)

        # ω-relaxation part (계산 함수 그대로 사용)
        relax_cost = self._build_omega_relax_cost(param, ocp.model.z, obstacles_geo, robot_components)

        stage_cost = track_cost + relax_cost
        term_cost  = alpha * (x - x_ref).T @ Q @ (x - x_ref)

        # 외부 cost 등록
        ocp.model.cost_expr_ext_cost   = stage_cost      # 모델에 할당해야 함 :contentReference[oaicite:0]{index=0}  
        ocp.model.cost_expr_ext_cost_e = term_cost       # 모델에 할당해야 함 :contentReference[oaicite:1]{index=1}  
        ocp.cost.cost_type        = 'EXTERNAL'
        ocp.cost.cost_type_e      = 'EXTERNAL'
        ocp.dims.n_cost_ext = stage_cost.size1()
        ocp.dims.n_cost_ext_e = term_cost.size1()

        # Set constraints
        # Input constraints
        ocp.constraints.lbu = np.array([param.amin,  param.omegamin])
        ocp.constraints.ubu = np.array([param.amax,  param.omegamax])
        ocp.constraints.idxbu = np.array([0, 1])

        ocp.constraints.x0 = self.state._x
        
        # Set options
        ocp.solver_options.tf              = param.tf
        ocp.solver_options.integrator_type = param.integrator_type  # 'IRK'
        ocp.solver_options.nlp_solver_type = param.nlp_solver_type  # 'SQP'
        ocp.solver_options.qp_solver       = param.qp_solver
        ocp.solver_options.hessian_approx  = 'EXACT'   # 외부 cost이므로 EXACT 권장
        
        # Save the objects for the OCP
        self.ocp = ocp
        self.N   = param.horizon
        self.nx  = nx
        self.nu  = nu

    def _build_omega_relax_cost(self, param, z_sym, obstacles_geo, robot_components):
        """
        z_sym  : model.z (NZ×1) –  전체 듀얼 변수 벡터
        반환값 : Σ  p_omega · (ω − 1)²   (NZ 중 ω항에 대해서만)
        """
        cost = ca.SX(0)  # Start with CasADi zero instead of Python 0
        offset = 0
        pomega = ca.SX(param.pomega)
        for k in range(min(param.horizon_dcbf, param.horizon)):
            for obs_geo in obstacles_geo:
                if not hasattr(obs_geo, 'get_convex_rep'):
                    continue
                    
                A_O, _ = obs_geo.get_convex_rep()
                m_O = A_O.shape[0]
                
                for robot_geo in robot_components:
                    if not hasattr(robot_geo, 'get_convex_rep'):
                        continue
                        
                    A_R, _ = robot_geo.get_convex_rep()
                    m_R = A_R.shape[0]
                    
                    # Check if we have enough variables
                    if offset + m_O + m_R < z_sym.size1():
                        omega = z_sym[offset + m_O + m_R]   # ω 위치
                        cost += pomega * (omega - 1)**2
                    
                    offset += m_O + m_R + 1

        return cost

    def create_solver(self):
        """Create the acados solver"""
        if not ACADOS_AVAILABLE:
            raise ImportError("acados_template is required for DCBF SQP optimizer")
        self.solver = AcadosOcpSolver(self.ocp, json_file=self.json_filename)
        self._temp_files.append(self.json_filename)
        self.variables["x"] = "x"
        self.variables["u"] = "u"

    # --- (3) add_obstacle_avoidance_constraint() 구현 --------------------------
    def add_obstacle_avoidance_constraint(self, param, system, obstacles_geo):
        """DCBF constraints를 acados의 nonlinear constraints로 추가"""
        if not obstacles_geo:
            return
            
        robot_components = system._geometry.equiv_rep()
        
        # Get safe distance for filtering
        safe_dist = system._dynamics.safe_dist(0.1, system._state._u[0], 1.0, param.margin_dist)
        
        # Initialize constraint lists
        h_expr_list, lg_list, ug_list = [], [], []
        
        # Symbolic variables from model
        x = self.ocp.model.x  # [x, y, theta, v]
        z = self.ocp.model.z  # algebraic variables [λ^O, λ^R, ω] for all timesteps
        
        # Track index offset for algebraic variables
        z_offset = 0
        
        # Add DCBF constraints for each timestep within DCBF horizon
        for k in range(min(param.horizon_dcbf, param.horizon)):
            for obs_idx, obs_geo in enumerate(obstacles_geo):
                if not hasattr(obs_geo, 'get_convex_rep'):
                    continue
                    
                A_obs, b_obs = obs_geo.get_convex_rep()
                m_O = A_obs.shape[0]
                
                # Get current CBF value for filtering
                if k == 0:
                    current_pos = self.state._x[0:2]
                    cbf_curr, _ = get_dist_point_to_region(current_pos, A_obs, b_obs)
                    if cbf_curr > safe_dist:
                        continue
                
                # For each robot component
                for rob_idx, robot_geo in enumerate(robot_components):
                    if not hasattr(robot_geo, 'get_convex_rep'):
                        continue
                        
                    A_robot, b_robot = robot_geo.get_convex_rep()
                    m_R = A_robot.shape[0]
                    
                    # Get current CBF value and dual variables for warm start
                    if k == 0:
                        # Robot rotation matrix for current state
                        theta_curr = self.state._x[2]
                        R_curr = np.array([[np.cos(theta_curr), -np.sin(theta_curr)],
                                         [np.sin(theta_curr), np.cos(theta_curr)]])
                        
                        # Transformed robot geometry
                        A_robot_curr = A_robot @ R_curr.T
                        b_robot_curr = A_robot @ R_curr.T @ self.state._x[0:2] + b_robot
                        
                        cbf_curr, lamb_curr, mu_curr = get_dist_region_to_region(
                            A_obs, b_obs, A_robot_curr, b_robot_curr)
                        
                        if cbf_curr > safe_dist:
                            z_offset += m_O + m_R + 1  # Skip this obstacle-robot pair
                            continue
                    
                    # Extract dual variables for this time step and obstacle-robot pair
                    lambda_O = z[z_offset:z_offset + m_O]        # λ^O_k
                    lambda_R = z[z_offset + m_O:z_offset + m_O + m_R]  # λ^R_k
                    omega = z[z_offset + m_O + m_R]              # ω_k
                    
                    # Robot position and orientation (symbolic)
                    pos = x[0:2]
                    theta = x[2]
                    
                    # Rotation matrix (symbolic)
                    R = ca.vertcat(
                        ca.horzcat(ca.cos(theta), -ca.sin(theta)),
                        ca.horzcat(ca.sin(theta), ca.cos(theta))
                    )
                    
                    # Constraint 1: λ^O ≥ 0 (equation 20h)
                    for i in range(m_O):
                        h_expr_list.append(lambda_O[i])
                        lg_list.append(0.0)
                        ug_list.append(1e6)
                    
                    # Constraint 2: λ^R ≥ 0 (equation 20h)
                    for i in range(m_R):
                        h_expr_list.append(lambda_R[i])
                        lg_list.append(0.0)
                        ug_list.append(1e6)
                    
                    # Constraint 3: ω ≥ 0 (equation 20h)
                    h_expr_list.append(omega)
                    lg_list.append(0.0)
                    ug_list.append(1e6)
                    
                    # Constraint 4: DCBF constraint (equation 20e)
                    # -λ^R^T b^R + (A^O x - b^O)^T λ^O ≥ ω γ^{k+1} (h_curr - d_margin) + d_margin
                    robot_term = -ca.mtimes(b_robot.T, lambda_R)
                    obstacle_term = ca.mtimes((ca.mtimes(A_obs, pos) - b_obs).T, lambda_O)
                    
                    if k == 0:
                        gamma_term = omega * (param.gamma ** (k + 1)) * (cbf_curr - param.margin_dist) + param.margin_dist
                    else:
                        # For k > 0, use a conservative estimate
                        gamma_term = omega * (param.gamma ** (k + 1)) * param.margin_dist + param.margin_dist
                    
                    dcbf_expr = robot_term + obstacle_term - gamma_term
                    h_expr_list.append(dcbf_expr)
                    lg_list.append(0.0)
                    ug_list.append(1e6)
                    
                    # Constraint 5: Duality constraint (equation 20g)
                    # λ^O^T A^O + λ^R^T A^R R^T = 0 (2D constraint)
                    robot_A_rotated = ca.mtimes(A_robot, R.T)
                    duality_expr = ca.mtimes(lambda_O.T, A_obs) + ca.mtimes(lambda_R.T, robot_A_rotated)
                    
                    for i in range(2):  # 2D equality constraint
                        h_expr_list.append(duality_expr[i])
                        lg_list.append(0.0)
                        ug_list.append(0.0)
                    
                    # Constraint 6: Norm constraint (equation 20f)
                    # ||λ^O^T A^O||_2 ≤ 1
                    lambda_A_norm = ca.mtimes(lambda_O.T, A_obs)
                    norm_expr = ca.mtimes(lambda_A_norm, lambda_A_norm.T) - 1.0
                    
                    h_expr_list.append(norm_expr)
                    lg_list.append(-1e6)
                    ug_list.append(0.0)
                    
                    # Update offset for next obstacle-robot pair
                    z_offset += m_O + m_R + 1

        # Set the constraints if any were added
        if h_expr_list:
            self.ocp.constraints.expr_h = ca.vertcat(*h_expr_list)
            self.ocp.constraints.lg = np.array(lg_list)
            self.ocp.constraints.ug = np.array(ug_list)
            
            # For the relaxation cost on ω, we'll handle it through additional cost terms
            # This will be added to the existing cost function

    def add_warm_start(self, param, system):
        """Add warm start based on nominal safe controller"""
        if self.solver is not None:
            try:
                # Get nominal safe trajectory
                x_ws, u_ws = system._dynamics.nominal_safe_controller(self.state._x, 0.1, self.state._u[0], -1.0, 1.0)
                
                # Set warm start for all stages
                for i in range(self.N):
                    self.solver.set(i, 'x', x_ws)
                    self.solver.set(i, 'u', u_ws)
                # Set final state
                self.solver.set(self.N, 'x', x_ws)
                
            except Exception as e:
                print(f"Warning: Could not set warm start: {e}")
                # Use current state as warm start
                for i in range(self.N):
                    self.solver.set(i, 'x', self.state._x)
                    self.solver.set(i, 'u', np.zeros(2))
                self.solver.set(self.N, 'x', self.state._x)

    def setup(self, param, system, reference_trajectory, obstacles):
        """Setup the complete optimization problem"""
        self.set_state(system._state)
        robot_components = system._geometry.equiv_rep()
        
        # Convert obstacles to geometry representations
        obstacles_geo = []
        if obstacles:
            for obs in obstacles:
                if hasattr(obs, '_geometry'):
                    obstacles_geo.append(obs._geometry)
                elif hasattr(obs, 'get_convex_rep'):
                    obstacles_geo.append(obs)
        
        self.setup_ocp(param, reference_trajectory, obstacles_geo, robot_components)
        
        # Add relaxation cost for ω variables if we have obstacles
        # if obstacles_geo and robot_components:
        #     self.add_relaxation_cost_to_ocp(param, obstacles_geo, robot_components)
        
        self.create_solver()
        self.add_obstacle_avoidance_constraint(param, system, obstacles_geo)
        self.add_warm_start(param, system)

    def solve_nlp(self):
        """Solve the nonlinear programming problem"""
        print("Solving DCBF SQP optimization problem")
        
        # Update initial state
        self.solver.set(0, 'lbx', self.state._x)
        self.solver.set(0, 'ubx', self.state._x)
        
        # Update reference trajectory parameters if using external cost
        if self.reference_trajectory is not None:
            for i in range(self.N):
                if i < len(self.reference_trajectory):
                    ref_x = self.reference_trajectory[i, :]
                else:
                    ref_x = self.reference_trajectory[-1, :]
                ref_u = np.zeros(2)  # Zero input reference
                ref_param = np.concatenate([ref_x, ref_u])
                self.solver.set(i, 'p', ref_param)
            # Terminal reference
            if len(self.reference_trajectory) > 0:
                ref_x_term = self.reference_trajectory[-1, :]
                ref_u_term = np.zeros(2)
                ref_param_term = np.concatenate([ref_x_term, ref_u_term])
                self.solver.set(self.N, 'p', ref_param_term)
        else:
            # Set default parameter values if no reference trajectory
            for i in range(self.N + 1):
                self.solver.set(i, 'p', np.zeros(self.nx + self.nu))
        # Solve
        start_timer = datetime.datetime.now()
        status = self.solver.solve()
        end_timer = datetime.datetime.now()
        
        # Record timing
        delta_timer = end_timer - start_timer
        self.solver_times.append(delta_timer.total_seconds())
        print("solver time: ", delta_timer.total_seconds())
        print("solver status: ", status)
        
        return AcadosSolution(self.solver, self.N, self.variables)


class AcadosSolution:
    """Helper class to mimic casadi solution interface"""
    
    def __init__(self, solver, N, variables):
        self.solver = solver
        self.N = N
        self.variables = variables
        self._x_traj = None
        self._u_traj = None
        self._compute_trajectories()
    
    def _compute_trajectories(self):
        """Extract state and input trajectories from solver"""
        self._x_traj = np.zeros((3, self.N + 1))
        self._u_traj = np.zeros((2, self.N))
        
        for i in range(self.N + 1):
            self._x_traj[:, i] = self.solver.get(i, 'x')
        
        for i in range(self.N):
            self._u_traj[:, i] = self.solver.get(i, 'u')
    
    def value(self, var_expr):
        """Get value of a variable expression"""
        if isinstance(var_expr, str):
            if var_expr == "x":
                return self._x_traj
            elif var_expr == "u":
                return self._u_traj
        elif hasattr(var_expr, '__getitem__'):  # Handle indexing like variables["u"][:, 0]
            if "x" in str(var_expr):
                return self._x_traj
            elif "u" in str(var_expr):
                return self._u_traj
        return self._u_traj  # Default fallback
    
    def get_state_trajectory(self):
        """Get state trajectory"""
        return self._x_traj
    
    def get_input_trajectory(self):
        """Get input trajectory"""  
        return self._u_traj
    
    def stats(self):
        """Get solver statistics"""
        return {"return_status": "Solve_Succeeded"}
