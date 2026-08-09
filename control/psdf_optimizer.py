import datetime
import casadi as ca
import numpy as np
import torch
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import time
import os
import json
import hashlib
import sys
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.geometry_utils import polygon_to_edges
from control.analytic_psdf_casadi import AnalyticPSDFCasADi

class PSDFOptimizerParam:
    def __init__(self):
        self.horizon = 20
        self.mat_Q = np.diag([50.0, 50.0, 1.0]) 
        self.mat_R = np.diag([5.0, 0.05])  
        
        self.terminal_weight = 1.0  # 10.0
        
        # SQP specific parameters
        self.tf = 0.1 * self.horizon  # total time horizon
        self.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # qp solver to be used
        self.hessian_approx = 'GAUSS_NEWTON'  # Hessian approximation 'GAUSS_NEWTON'
        self.integrator_type = 'ERK'  # explicit Runge-Kutta
        self.nlp_solver_type = 'SQP_RTI'  # SQP solver
        self.qp_solver_iter_max = 50  # maximum iterations for QP solver
        self.nlp_solver_max_iter = 20  # maximum iterations for NLP solver
        self.tol = 1e-4  # convergence tolerance
        
        # Input constraints
        self.vmin, self.vmax = -0.6, 0.6
        self.omegamin, self.omegamax = -1.2, 1.2

        # Safety distance parameter
        self.d_safe = 0.0001  # minimum safe distance to obstacles
        
        # Soft-constraint parameters
        # If `use_soft_constraint` is True, the obstacle avoidance constraint will be softened
        # by introducing a non-negative slack variable that is penalised in the cost function
        # with quadratic weight `slack_weight`.
        self.use_soft_constraint = False
        self.slack_weight = 1e16
        
        # Obstacle detection parameters
        self.detection_window_width = 4.0   # local window width [m]
        self.detection_window_height = 4.0  # local window height [m]
        self.detection_safety_margin = 0.05 # safety margin for obstacle caps [m]
        self.detection_frequency = 10.0     # obstacle detection frequency [Hz]

        # Reuse generated acados code when possible.
        # If relevant code/config changes are detected, solver is rebuilt automatically.
        self.reuse_acados_codegen = True


class PSDFOptimizer:
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
        self.psdf_wrapper = None  # AugmentedPSDFWrapper instance
        self._is_initialized = False
        
        # Local window obstacle detector
        self.obstacle_detector = None
        
        # Use common JSON filename - will be cleaned up after each test
        self.json_filename = "acados_ocp.json"
        self.codegen_meta_filename = "acados_psdf_codegen_meta.json"
        self._default_code_export_dir = "c_generated_code"
        
        # Store files to cleanup
        self._temp_files = []

    def cleanup(self, purge_codegen=False):
        """Clean up temporary files and reset state"""
        try:
            # Clean up temporary files
            for file_path in self._temp_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            if purge_codegen:
                if os.path.exists(self.json_filename):
                    os.remove(self.json_filename)
                if os.path.exists(self.codegen_meta_filename):
                    os.remove(self.codegen_meta_filename)

                cleanup_dirs = [self._default_code_export_dir]
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
        self.psdf_wrapper = None
        self.ped_model = None
        self._is_initialized = False

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
        x = ca.MX.sym('x', nx)
        xdot = ca.MX.sym('xdot', nx)
        u = ca.MX.sym('u', nu)
        
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
        model.name = 'differential_drive_psdf'
        
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
        yref_init = np.zeros((nx + nu,))
        if reference_trajectory.size > 0:
            yref_init[:nx] = reference_trajectory[0, :]
        self.ocp.cost.yref = yref_init
        self.ocp.cost.yref_e = reference_trajectory[-1, :]
        
        # Set constraints
        # Input constraints
        self.ocp.constraints.lbu = np.array([param.vmin, param.omegamin])
        self.ocp.constraints.ubu = np.array([param.vmax, param.omegamax])
        self.ocp.constraints.idxbu = np.array([0, 1])
        
        # Initial state constraint
        if self.state is not None:
            self.ocp.constraints.x0 = self.state._x
        else:
            self.ocp.constraints.x0 = np.zeros(nx)
        
        # Add input derivative constraints (rate constraints)
        # This requires setting up a custom constraint in acados
        # For simplicity, we'll use soft constraints via cost function
        
        # Set options
        self.ocp.solver_options.qp_solver = param.qp_solver
        self.ocp.solver_options.hessian_approx = param.hessian_approx
        self.ocp.solver_options.integrator_type = param.integrator_type
        self.ocp.solver_options.nlp_solver_type = param.nlp_solver_type
        self.ocp.solver_options.qp_solver_iter_max = param.qp_solver_iter_max 
        self.ocp.solver_options.nlp_solver_max_iter = param.nlp_solver_max_iter
        self.ocp.solver_options.tol = param.tol
        # Set warm start options
        self.ocp.solver_options.qp_solver_warm_start = True  
        self.ocp.solver_options.nlp_solver_warm_start_first_qp = True  
        # Set prediction horizon
        self.ocp.solver_options.tf = param.tf
        
        # Store reference trajectory
        self.reference_trajectory = reference_trajectory

        

    def create_solver(self):
        if (
            getattr(getattr(self, "ped_model", None), "requires_external_shared_lib", False)
            and hasattr(self.ped_model, "shared_lib_dir")
            and hasattr(self.ped_model, "name")
        ):
            self.ocp.solver_options.model_external_shared_lib_dir = self.ped_model.shared_lib_dir
            self.ocp.solver_options.model_external_shared_lib_name = self.ped_model.name
       
        """Create the acados solver"""
        self.variables["x"] = "x"
        self.variables["u"] = "u"
        reuse_codegen = bool(getattr(self.param, "reuse_acados_codegen", True))
        build_signature = self._compute_build_signature()

        if reuse_codegen and self._can_reuse_codegen(build_signature):
            print("Reusing cached acados PSDF build artifacts (no recompile).")
            self.solver = AcadosOcpSolver(
                None,
                json_file=self.json_filename,
                generate=False,
                build=False,
            )
        else:
            if reuse_codegen:
                print("acados PSDF cache miss or source/config changed: rebuilding.")
            else:
                print("acados PSDF codegen cache disabled: rebuilding.")
            self.solver = AcadosOcpSolver(self.ocp, json_file=self.json_filename)
            if reuse_codegen:
                self._write_codegen_meta(build_signature)

    @staticmethod
    def _to_jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [PSDFOptimizer._to_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): PSDFOptimizer._to_jsonable(v) for k, v in value.items()}
        return value

    @staticmethod
    def _hash_file(path):
        if not os.path.exists(path) or not os.path.isfile(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _shared_lib_ext():
        if os.name == "nt":
            return ".dll"
        if sys.platform == "darwin":
            return ".dylib"
        return ".so"

    def _get_code_export_dir(self):
        if self.ocp is not None and getattr(self.ocp, "code_export_directory", None):
            return self.ocp.code_export_directory
        return self._default_code_export_dir

    def _get_solver_shared_lib_path(self):
        model_name = "differential_drive_psdf"
        if self.ocp is not None and getattr(self.ocp, "model", None) is not None:
            model_name = self.ocp.model.name
        lib_prefix = "" if os.name == "nt" else "lib"
        lib_name = f"{lib_prefix}acados_ocp_solver_{model_name}{self._shared_lib_ext()}"
        return os.path.join(self._get_code_export_dir(), lib_name)

    @staticmethod
    def _safe_dim_int(value):
        if value is None:
            return 0
        try:
            return int(value)
        except Exception:
            return 0

    def _compute_build_signature(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        tracked_files = {
            "control/psdf_optimizer.py": os.path.abspath(__file__),
            "control/analytic_psdf_casadi.py": os.path.join(
                project_root, "control", "analytic_psdf_casadi.py"
            ),
            "models/augmented_psdf_wrapper.py": os.path.join(
                project_root, "models", "augmented_psdf_wrapper.py"
            ),
            "models/augmented_psdf.py": os.path.join(
                project_root, "models", "augmented_psdf.py"
            ),
        }
        file_hashes = {k: self._hash_file(v) for k, v in tracked_files.items()}

        dims = self.ocp.dims if self.ocp is not None else None
        payload = {
            "signature_version": 2,
            "model_name": self.ocp.model.name if self.ocp is not None else "differential_drive_psdf",
            "dims": {
                "N": self._safe_dim_int(getattr(dims, "N", None)),
                "np": self._safe_dim_int(getattr(dims, "np", None)),
                "nh": self._safe_dim_int(getattr(dims, "nh", None)),
                "ns": self._safe_dim_int(getattr(dims, "ns", None)),
            },
            "param": self._to_jsonable(getattr(self, "param", None).__dict__ if hasattr(self, "param") else {}),
            "file_hashes": file_hashes,
        }
        payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload_bytes).hexdigest()

    def _write_codegen_meta(self, build_signature):
        meta = {
            "build_signature": build_signature,
            "json_file": self.json_filename,
            "shared_lib": self._get_solver_shared_lib_path(),
            "code_export_dir": self._get_code_export_dir(),
            "updated_at": datetime.datetime.now().isoformat(),
        }
        with open(self.codegen_meta_filename, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    def _read_codegen_meta(self):
        if not os.path.exists(self.codegen_meta_filename):
            return None
        try:
            with open(self.codegen_meta_filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _can_reuse_codegen(self, build_signature):
        if not os.path.exists(self.json_filename):
            return False
        if not os.path.exists(self._get_code_export_dir()):
            return False
        if not os.path.exists(self._get_solver_shared_lib_path()):
            return False

        meta = self._read_codegen_meta()
        if meta is None:
            return False
        if meta.get("build_signature") != build_signature:
            return False
        return True

    def update_obstacles(self, obstacles_geo): 
        """Update the obstacles in the PSDF wrapper (legacy method)"""
        if self.psdf_wrapper is not None:
            all_clusters_A = []
            all_clusters_B = []
            for obs_geo in obstacles_geo:
                edgesA, edgesB = polygon_to_edges(obs_geo)
                all_clusters_A.append(edgesA)
                all_clusters_B.append(edgesB)

            # Update the augmented PSDF wrapper with all clusters.
            if len(all_clusters_A) <= self.psdf_wrapper.K_max:
                self.psdf_wrapper.update_edge_clusters(all_clusters_A, all_clusters_B)
            else:
                # Use only first K_max clusters
                self.psdf_wrapper.update_edge_clusters(all_clusters_A[:self.psdf_wrapper.K_max], all_clusters_B[:self.psdf_wrapper.K_max])
        else:
            print("Warning: PSDF wrapper not initialized, skipping obstacle update")                

    def add_obstacle_avoidance_constraint(self, param, system, obstacles_geo):
        
        self.update_obstacles(obstacles_geo)

        # Add obstacle avoidance constraint using PED model
        if self.ped_model is not None and self.ocp is not None:
            # Define constraint function: h(x) = sdf(x, y, theta) - d_safe >= 0
            x = self.ocp.model.x  # state variables [x, y, theta]
            
            sdf_value = self.ped_model(x)
            constraint_expr = sdf_value - param.d_safe
            
            # Set up nonlinear constraint
            self.ocp.constraints.constr_type = 'BGH'
            
            # --- expose real-time Taylor parameters as model parameter vector ---
            p_sym = self.ped_model.get_sym_params()  # (np, 1)
            self.ocp.model.p = p_sym
            self.ocp.dims.np = p_sym.shape[0]
            # Provide default parameter values (will be overwritten each iteration)
            self.ocp.parameter_values = np.zeros((p_sym.shape[0], ))

            # For path constraints (applied at each time step)
            nh = 1  # number of constraints (scalar sdf)
            self.ocp.dims.nh = nh

            # Set constraint expression (depends on p)
            self.ocp.model.con_h_expr = constraint_expr

            # Set constraint bounds: h(x) >= 0 (hard lower bound)
            self.ocp.constraints.lh = np.array([0.0])
            self.ocp.constraints.uh = np.array([1e8])  # effectively no upper bound

            # ------------------------------------------------------------------
            # Soft-constraint handling using slack variables in acados
            # ------------------------------------------------------------------
            if getattr(param, "use_soft_constraint", False):
                # Number of soft constraints (lower bound violation only)
                ns = nh
                self.ocp.dims.ns = ns      # total number of slacks
                self.ocp.dims.nsh = nh     # slacks for nonlinear path constraints

                # Indices of the constraints that are softened (0-based)
                self.ocp.constraints.idxsh = np.arange(nh, dtype=np.int64)

                # Slack bounds (0 <= s <= large)
                self.ocp.constraints.lsh = np.zeros(ns)
                self.ocp.constraints.ush = np.ones(ns) * 1e8

                # Quadratic cost on slack variables
                slack_w = float(getattr(param, "slack_weight", 1e4))
                self.ocp.cost.Zl = np.diag([slack_w] * ns)
                self.ocp.cost.Zu = np.diag([slack_w] * ns)
                # Linear term (optional)
                self.ocp.cost.zl = np.zeros(ns)
                self.ocp.cost.zu = np.zeros(ns)

                print(f"Obstacle avoidance constraint softened with slack weight = {slack_w}")
            else:
                print(f"Added HARD obstacle avoidance constraint with d_safe = {param.d_safe}")
        else:
            print("Warning: PED model not initialized, skipping obstacle avoidance constraint")

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
        self.param = param
        self.set_state(system._state)

        if not self._is_initialized:
            print("Setting up PSDF optimizer...")
            # Setup the OCP Problem
            self.setup_ocp(param, reference_trajectory)
            self.initialize_ped_model(system, obstacles, E_max=100, device="cpu") # Set the PED model in the OCP

            self.add_obstacle_avoidance_constraint(param, system, obstacles)
            # Create the acados solver
            self.create_solver()
            self.add_warm_start(param, system)
            self._is_initialized = True
            
        
        else:
            # Set the reference trajectory
            self.set_reference_trajectory(reference_trajectory)
            # Obstacle update is now handled in the control loop (e.g., test_nmpc.py)
            # before this setup method is called.

    def solve_nlp(self):
        """Solve the NLP using SQP"""
        start = time.time()
        # Ensure the initial state constraint is updated
        if self.state is not None:
            self.solver.set(0, "lbx", self.state._x)
            self.solver.set(0, "ubx", self.state._x)
        
        # ------------------------------------------------------------------
        # Update real-time Taylor parameters for each stage (batch evaluation)
        # ------------------------------------------------------------------
        if self.ped_model is not None:
            # Gather current guessed states (warm start or previous solution)
            x_guess = np.stack([self.solver.get(i, "x") for i in range(self.N + 1)], axis=0)
            params_batch = self.ped_model.get_params(x_guess)  # (N+1, np)
            # Set parameters for each stage
            for i in range(self.N):
                self.solver.set(i, "p", params_batch[i])
            self.solver.set(self.N, "p", params_batch[-1])

        
        status = self.solver.solve()
        end = time.time()
        solve_time = end - start
        
        # Record solver time
        self.solver_times.append(solve_time)
        print("solver time: ", solve_time)
        
        if status != 0:
            print(f"Acados solver failed with status {status}")
            
        
        return AcadosSolution(self.solver, self.N, self.variables)


    def initialize_ped_model(self, system, obstacles, E_max=100, K_max = 20, device="cpu"):
        """
        시스템의 geometry로부터 AugmentedPSDFWrapper 초기화 및 obstacle detector 설정
        
        Args:
            system: robot system with geometry
            obstacles: initial obstacles list
            E_max: maximum number of edges to handle
            K_max: maximum number of clusters to handle
            device: torch device ("cpu" or "cuda")
        """
        self.device = device

        if hasattr(system, '_geometry'):
            geometry = system._geometry._geometries[0]  # Get the first geometry
            vertices = geometry._region.get_ccw_vertices() # Get vertices in counter-clockwise order
            
            print(f"vertices.shape: {vertices.shape}")
            self.psdf_wrapper = AugmentedPSDFWrapper(
                verts=torch.tensor(vertices, dtype=torch.float32, device=device),
                E_max=E_max,
                K_max=K_max,
                device=device
            )
            
            print(f"AugmentedPSDFWrapper initialized with vertices shape: {vertices.shape}")
            
        else:
            raise ValueError("System must have _geometry attribute")

        self.ped_model = AnalyticPSDFCasADi(
            self.psdf_wrapper,
            device=self.device,
            name="analytic_psdf",
        )


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
