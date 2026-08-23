class SystemLogger:
    def __init__(self):
        self._xs = []
        self._us = []

class ControllerLogger:
    def __init__(self):
        self._xtrajs = []
        self._utrajs = []
        self._risk_margin_trajs = []
        self._solver_status_infos = []
        self._computation_times = []

class LocalPlannerLogger:
    def __init__(self):
        self._trajs = []

class GlobalPlannerLogger:
    def __init__(self):
        self._paths = []
