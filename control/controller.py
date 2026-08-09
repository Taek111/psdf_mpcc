class BaseController:
    def __init__(self, optimizer, opt_param):
        self._optimizer = optimizer
        self._param = opt_param
        self._opt_sol = None
        self._is_setup = False
        self._last_solver_status_info = {
            "raw_status": None,
            "status_code": None,
            "success": None,
        }

    @staticmethod
    def _infer_success(raw_status, status_code):
        if status_code is not None:
            return int(status_code) == 0

        if raw_status is None:
            return None

        normalized = str(raw_status).strip().lower()
        if normalized in ("success", "solve_succeeded", "solved"):
            return True
        if any(token in normalized for token in ("fail", "infeasible", "error", "exception")):
            return False
        return None

    def _update_last_solver_status_info(self, exc=None):
        raw_status = None
        status_code = None

        if exc is not None:
            raw_status = f"{type(exc).__name__}: {exc}"
        elif self._opt_sol is not None and hasattr(self._opt_sol, "stats"):
            try:
                stats = self._opt_sol.stats()
            except Exception:
                stats = {}
            if isinstance(stats, dict):
                raw_status = stats.get("return_status", None)

        solver_obj = None
        if self._opt_sol is not None and hasattr(self._opt_sol, "solver"):
            solver_obj = self._opt_sol.solver
        elif hasattr(self._optimizer, "solver"):
            solver_obj = self._optimizer.solver

        if solver_obj is not None and hasattr(solver_obj, "status"):
            try:
                status_code = int(solver_obj.status)
            except Exception:
                status_code = None

        self._last_solver_status_info = {
            "raw_status": raw_status,
            "status_code": status_code,
            "success": self._infer_success(raw_status, status_code),
        }

    def get_last_solver_status_info(self):
        return dict(self._last_solver_status_info)

    def _setup_optimizer_once(self, system, local_trajectory, obstacles):
        if not self._is_setup:
            self._optimizer.setup(self._param, system, local_trajectory, obstacles)
            self._is_setup = True

    def generate_control_input(self, system, global_path, local_trajectory, obstacles):
        self._optimizer.setup(self._param, system, local_trajectory, obstacles)
        try:
            self._opt_sol = self._optimizer.solve_nlp()
        except Exception as exc:
            self._update_last_solver_status_info(exc=exc)
            raise

        self._update_last_solver_status_info()
        try:
            u_opt = self._opt_sol.value("u")
            u_stage0 = u_opt[:, 0]
            if hasattr(self._optimizer, "record_plant_input_applied"):
                self._optimizer.record_plant_input_applied()
            print(f"plant_input_stage0: full={u_stage0}, physical={u_stage0[:2]}")
            return u_stage0
                  
        except Exception:
            u_stage0 = self._opt_sol.value(self._optimizer.variables["u"][:, 0])
            if hasattr(self._optimizer, "record_plant_input_applied"):
                self._optimizer.record_plant_input_applied()
            print(f"plant_input_stage0: full={u_stage0}, physical={u_stage0[:2]}")
            return u_stage0

    def logging(self, logger):
        # SQP 기반 optimizer
        if hasattr(self._opt_sol, 'get_state_trajectory'):
            logger._xtrajs.append(self._opt_sol.get_state_trajectory().T)
            logger._utrajs.append(self._opt_sol.get_input_trajectory().T)
        else:
            # CasADi 기반 optimizer
            logger._xtrajs.append(self._opt_sol.value(self._optimizer.variables["x"]).T)
            logger._utrajs.append(self._opt_sol.value(self._optimizer.variables["u"]).T)

        risk_margin_traj = None
        if hasattr(self._optimizer, "get_last_risk_margin_trajectory"):
            try:
                risk_margin_traj = self._optimizer.get_last_risk_margin_trajectory()
            except Exception:
                risk_margin_traj = None
        logger._risk_margin_trajs.append(risk_margin_traj)
        logger._solver_status_infos.append(self.get_last_solver_status_info())
