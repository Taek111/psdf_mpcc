"""Shared cache and distance-query helpers for duality-based optimizers."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import casadi as ca
import numpy as np


def freeze_value(value: Any) -> Any:
    """Represent numerical configuration values as deterministic cache keys."""
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, np.generic):
            # Extended precision NumPy scalars may have no Python equivalent.
            return ("numpy_scalar", value.dtype.str, str(value))
        return freeze_value(scalar)
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            value.dtype.str,
            value.shape,
            tuple(freeze_value(item) for item in value.flat),
        )
    if isinstance(value, dict):
        items = [
            (freeze_value(key), freeze_value(item)) for key, item in value.items()
        ]
        return ("dict", tuple(sorted(items, key=lambda item: repr(item[0]))))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(freeze_value(item) for item in value))
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        # Hex strings also give NaN/inf stable equality for cache comparisons.
        return ("float", value.hex())
    if isinstance(value, complex):
        return ("complex", freeze_value(value.real), freeze_value(value.imag))
    if isinstance(value, (str, bytes)):
        return (type(value).__name__, value)
    raise TypeError(f"Unsupported cache-signature value: {type(value).__name__}")


def validate_cutoff(param: Any) -> Optional[float]:
    """Return the obstacle cutoff, or None when obstacle filtering is disabled."""
    if not getattr(param, "use_obstacle_cutoff", True):
        return None
    try:
        cutoff = float(getattr(param, "safe_dist", 0.5))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("safe_dist must be a finite, nonnegative number") from exc
    if not np.isfinite(cutoff) or cutoff < 0:
        raise ValueError("safe_dist must be a finite, nonnegative number")
    return cutoff


@dataclass
class _DistanceProblem:
    opti: Any
    parameters: Tuple[Any, Any, Any, Any]
    distance_expression: Any
    dual_expressions: Tuple[Any, Any]


class ReusableRegionDistanceQuery:
    """Reuse one parameterized distance problem for each pair of region shapes.

    Regions are supplied as halfspaces ``A @ point <= b``. The objective,
    solver options, and normalized duals match ``get_dist_region_to_region``.
    Initial guesses are left at their defaults, as in the original helper.
    """

    def __init__(self) -> None:
        self._problems: Dict[Tuple[int, int, int], _DistanceProblem] = {}
        self.build_count = 0

    @staticmethod
    def _validate_region(
        mat_a: Any, vec_b: Any, name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            mat_a = np.asarray(mat_a, dtype=float)
            vec_b = np.asarray(vec_b, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} halfspaces must be numerical arrays") from exc
        if mat_a.ndim != 2 or min(mat_a.shape) < 1:
            raise ValueError(
                f"{name} A must have shape (positive rows, positive dimensions)"
            )
        rows = mat_a.shape[0]
        if vec_b.shape not in ((rows,), (rows, 1)):
            raise ValueError(f"{name} b must have shape ({rows},) or ({rows}, 1)")
        if not np.all(np.isfinite(mat_a)) or not np.all(np.isfinite(vec_b)):
            raise ValueError(f"{name} halfspaces must contain only finite values")
        return mat_a, vec_b.reshape(rows, 1)

    def _build_problem(
        self, rows1: int, rows2: int, dimensions: int
    ) -> _DistanceProblem:
        opti = ca.Opti()
        mat_a1 = opti.parameter(rows1, dimensions)
        vec_b1 = opti.parameter(rows1, 1)
        mat_a2 = opti.parameter(rows2, dimensions)
        vec_b2 = opti.parameter(rows2, 1)
        point1 = opti.variable(dimensions, 1)
        point2 = opti.variable(dimensions, 1)
        constraint1 = ca.mtimes(mat_a1, point1) <= vec_b1
        constraint2 = ca.mtimes(mat_a2, point2) <= vec_b2
        opti.subject_to(constraint1)
        opti.subject_to(constraint2)
        dist_vec = point1 - point2
        opti.minimize(ca.mtimes(dist_vec.T, dist_vec))
        opti.solver(
            "ipopt", {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        )
        problem = _DistanceProblem(
            opti=opti,
            parameters=(mat_a1, vec_b1, mat_a2, vec_b2),
            distance_expression=ca.norm_2(dist_vec),
            dual_expressions=(opti.dual(constraint1), opti.dual(constraint2)),
        )
        self.build_count += 1
        return problem

    def distance(
        self, mat_a1: Any, vec_b1: Any, mat_a2: Any, vec_b2: Any
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Return distance and normalized halfspace duals for both regions."""
        mat_a1, vec_b1 = self._validate_region(mat_a1, vec_b1, "First region")
        mat_a2, vec_b2 = self._validate_region(mat_a2, vec_b2, "Second region")
        if mat_a1.shape[1] != mat_a2.shape[1]:
            raise ValueError("Both regions must have the same spatial dimension")
        key = (mat_a1.shape[0], mat_a2.shape[0], mat_a1.shape[1])
        if key not in self._problems:
            self._problems[key] = self._build_problem(*key)
        problem = self._problems[key]
        values = (mat_a1, vec_b1, mat_a2, vec_b2)
        for parameter, value in zip(problem.parameters, values):
            problem.opti.set_value(parameter, value)
        solution = problem.opti.solve()
        dist = float(solution.value(problem.distance_expression))
        if not np.isfinite(dist):
            raise ValueError("Region distance solver returned a nonfinite distance")
        if dist > 0:
            lamb = np.asarray(
                solution.value(problem.dual_expressions[0]), dtype=float
            ).reshape(-1)
            mu = np.asarray(
                solution.value(problem.dual_expressions[1]), dtype=float
            ).reshape(-1)
            lamb = lamb / (2 * dist)
            mu = mu / (2 * dist)
            if not np.all(np.isfinite(lamb)) or not np.all(np.isfinite(mu)):
                raise ValueError("Region distance solver returned nonfinite dual variables")
        else:
            lamb = np.zeros(mat_a1.shape[0])
            mu = np.zeros(mat_a2.shape[0])
        return dist, lamb, mu
