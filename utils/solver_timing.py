"""Use the same solve_nlp wall-time boundary for every MPCC controller."""

from functools import wraps
from time import perf_counter


SOLVE_NLP_TIMING_METHODS = frozenset({"mpcc", "cc_sf", "cc_mf", "cc_ecs"})


def record_solve_nlp_time(solve_nlp):
    """Record one duration in seconds per call, including recovery and failures.

    External setup, planning, controller logging and result export are outside
    this boundary. With use_debug enabled, diagnostic work inside solve_nlp is
    included too; fair comparisons must use use_debug=False for every method.
    """
    @wraps(solve_nlp)
    def timed_solve(self, *args, **kwargs):
        start = perf_counter()
        try:
            return solve_nlp(self, *args, **kwargs)
        finally:
            self.solver_times.append(perf_counter() - start)

    return timed_solve
