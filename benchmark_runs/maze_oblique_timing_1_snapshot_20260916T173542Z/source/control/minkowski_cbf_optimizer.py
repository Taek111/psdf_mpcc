"""Pointwise Minkowski signed-distance CLF-CBF-QP optimizer.

The geometry and signed-distance construction follow Chen et al.,
"Control Barrier Functions via Minkowski Operations for Safe Navigation
among Polytopic Sets".  The control-affine constraint is adapted to this
repository's kinematic differential-drive model:

    state = [x, y, theta], input = [v, omega].

For separated sets, translation derivatives use the exact
``-z_star / ||z_star||`` result and the heading derivative uses the paper's
reduced KKT system.  For penetration/contact, the first normalized facet
attaining ``min(b)`` supplies a deterministic one-sided gradient.  Facet
switches and matching ambiguity retain the exact signed-distance value and
fall back only for the heading derivative, which is reported as nonsmooth.
These penetration and nonsmooth policies are implementation choices because
the paper leaves their rigorous analysis as future work.

The soft CLF uses the bearing from the current position to the pointwise
reference.  This avoids the zero-control-direction stall of a Cartesian pose
quadratic for nonholonomic differential-drive kinematics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import osqp
from scipy import sparse
from scipy.spatial import ConvexHull, QhullError

from models.geometry_utils import ConvexRegion2D


_OSQP_SOLVED_STATUS_VALUES = (1,)


@dataclass
class MinkowskiCBFOptimizerParam:
    """Minimal numerical and controller parameters."""

    d_safe: float = 1e-4
    gamma: float = 1.0
    epsilon: float = 1e-6

    vmin: float = -0.6
    vmax: float = 0.6
    omegamin: float = -1.2
    omegamax: float = 1.2

    control_weight_v: float = 1.0
    control_weight_omega: float = 0.2
    clf_rate: float = 1.0
    clf_heading_weight: float = 0.25
    clf_slack_weight: float = 100.0

    geometry_tol: float = 1e-9
    containment_tol: float = 1e-8
    active_tol: float = 2e-6
    dual_tol: float = 1e-8
    contact_tol: float = 1e-8
    facet_match_tol: float = 5e-4
    theta_fd_eps: float = 1e-5
    kkt_condition_max: float = 1e10

    osqp_eps_abs: float = 1e-8
    osqp_eps_rel: float = 1e-8
    osqp_max_iter: int = 10000
    osqp_polishing: bool = True
    verbose: bool = False


@dataclass
class ConfigurationObstacle:
    """Normalized H-representation and vertices of one configuration obstacle."""

    A: np.ndarray
    b: np.ndarray
    vertices: np.ndarray


@dataclass
class SignedDistanceResult:
    """Signed-distance result for one robot-component/obstacle pair."""

    signed_distance: float
    h: float
    gradient: np.ndarray
    branch: str
    active_indices: Tuple[int, ...]
    nonsmooth: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    critical_point: np.ndarray = field(default_factory=lambda: np.zeros(2))
    configuration_obstacle: Optional[ConfigurationObstacle] = None


@dataclass
class _RawSignedDistance:
    signed_distance: float
    branch: str
    active_indices: Tuple[int, ...]
    critical_point: np.ndarray
    dual: np.ndarray
    diagnostics: Dict[str, Any]


def _as_vertices(vertices: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(vertices, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 3:
        raise ValueError(f"{name} must contain at least three 2D vertices")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _deduplicate_points(points: np.ndarray, tol: float) -> np.ndarray:
    unique: List[np.ndarray] = []
    for point in np.asarray(points, dtype=float):
        if not any(np.linalg.norm(point - existing) <= tol for existing in unique):
            unique.append(point.copy())
    if len(unique) < 3:
        raise ValueError("A nondegenerate polygon requires at least three unique points")
    return np.asarray(unique, dtype=float)


def _sort_vertices_ccw(vertices: np.ndarray) -> np.ndarray:
    centroid = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
    ordered = vertices[np.argsort(angles)]
    twice_area = np.sum(
        ordered[:, 0] * np.roll(ordered[:, 1], -1)
        - ordered[:, 1] * np.roll(ordered[:, 0], -1)
    )
    if abs(twice_area) <= np.finfo(float).eps:
        raise ValueError("Polygon vertices are collinear")
    if twice_area < 0.0:
        ordered = ordered[::-1]
    return ordered


def _normalize_hrep(
    mat_a: np.ndarray,
    vec_b: np.ndarray,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mat_a = np.asarray(mat_a, dtype=float)
    vec_b = np.asarray(vec_b, dtype=float).reshape(-1)
    if mat_a.ndim != 2 or mat_a.shape[1] != 2 or mat_a.shape[0] != vec_b.size:
        raise ValueError("Invalid 2D H-representation")

    norms = np.linalg.norm(mat_a, axis=1)
    valid = norms > tol
    if np.count_nonzero(valid) < 3:
        raise ValueError("Configuration obstacle has fewer than three valid facets")

    normalized_a = mat_a[valid] / norms[valid, None]
    normalized_b = vec_b[valid] / norms[valid]

    order = np.argsort(np.arctan2(normalized_a[:, 1], normalized_a[:, 0]))
    normalized_a = normalized_a[order]
    normalized_b = normalized_b[order]

    normal_tol = max(100.0 * tol, 1e-8)
    kept_a: List[np.ndarray] = []
    kept_b: List[float] = []
    for row, bound in zip(normalized_a, normalized_b):
        duplicate_index = None
        for index, existing in enumerate(kept_a):
            cross = abs(existing[0] * row[1] - existing[1] * row[0])
            if cross <= normal_tol and float(np.dot(existing, row)) > 0.0:
                duplicate_index = index
                break
        if duplicate_index is None:
            kept_a.append(row.copy())
            kept_b.append(float(bound))
        elif bound < kept_b[duplicate_index]:
            kept_a[duplicate_index] = row.copy()
            kept_b[duplicate_index] = float(bound)

    result_a = np.asarray(kept_a, dtype=float)
    result_b = np.asarray(kept_b, dtype=float)
    order = np.argsort(np.arctan2(result_a[:, 1], result_a[:, 0]))
    return result_a[order], result_b[order]


def _build_configuration_obstacle(
    robot_vertices_local: Sequence[Sequence[float]],
    obstacle_region: ConvexRegion2D,
    pose: Sequence[float],
    tol: float = 1e-9,
) -> ConfigurationObstacle:
    """Build O (+) (-R(x)) from existing geometry objects."""

    robot_local = _as_vertices(robot_vertices_local, "robot_vertices_local")
    pose_array = np.asarray(pose, dtype=float).reshape(-1)
    if pose_array.size != 3 or not np.all(np.isfinite(pose_array)):
        raise ValueError("pose must be a finite [x, y, theta] vector")
    if not isinstance(obstacle_region, ConvexRegion2D):
        raise TypeError("obstacle_region must be a ConvexRegion2D")
    if not hasattr(obstacle_region, "get_ccw_vertices"):
        raise TypeError("obstacle_region must provide get_ccw_vertices()")

    obstacle_vertices = _as_vertices(
        obstacle_region.get_ccw_vertices(),
        "obstacle_vertices",
    )

    theta = float(pose_array[2])
    rotation = np.array(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ],
        dtype=float,
    )
    robot_world = robot_local @ rotation.T + pose_array[:2]
    reflected_robot = -robot_world

    pairwise_sums = (
        obstacle_vertices[:, None, :] + reflected_robot[None, :, :]
    ).reshape(-1, 2)
    pairwise_sums = _deduplicate_points(pairwise_sums, tol)

    try:
        hull = ConvexHull(pairwise_sums)
    except QhullError as exc:
        raise ValueError("Configuration obstacle is degenerate") from exc
    raw_a = hull.equations[:, :2]
    raw_b = -hull.equations[:, 2]
    mat_a, vec_b = _normalize_hrep(raw_a, raw_b, tol)
    vertices = _sort_vertices_ccw(pairwise_sums[hull.vertices])

    feasibility_tol = max(100.0 * tol, 1e-7)
    if np.any(mat_a @ vertices.T - vec_b[:, None] > feasibility_tol):
        raise ValueError("Normalized CO H-representation excludes a hull vertex")

    return ConfigurationObstacle(A=mat_a, b=vec_b, vertices=vertices)


def _configure_osqp(
    problem: osqp.OSQP,
    param: MinkowskiCBFOptimizerParam,
    **kwargs: Any,
) -> None:
    problem.setup(
        verbose=bool(param.verbose),
        eps_abs=float(param.osqp_eps_abs),
        eps_rel=float(param.osqp_eps_rel),
        max_iter=int(param.osqp_max_iter),
        polishing=bool(param.osqp_polishing),
        **kwargs,
    )


def _require_osqp_solution(result: Any, context: str) -> None:
    status_value = int(result.info.status_val)
    if status_value not in _OSQP_SOLVED_STATUS_VALUES:
        raise RuntimeError(
            f"{context} failed: status={result.info.status}, "
            f"status_val={status_value}"
        )


def _independent_active_indices(
    mat_a: np.ndarray,
    dual: np.ndarray,
    candidates: Sequence[int],
    tol: float,
) -> Tuple[int, ...]:
    ordered = sorted(candidates, key=lambda index: (-dual[index], index))
    selected: List[int] = []
    current_rank = 0
    for index in ordered:
        trial = selected + [int(index)]
        trial_rank = int(np.linalg.matrix_rank(mat_a[trial], tol=tol))
        if trial_rank > current_rank:
            selected.append(int(index))
            current_rank = trial_rank
        if current_rank >= 2:
            break
    if not selected:
        raise RuntimeError("Projection QP did not provide an independent active facet")
    return tuple(sorted(selected))


def _solve_projection_qp(
    co: ConfigurationObstacle,
    param: MinkowskiCBFOptimizerParam,
) -> _RawSignedDistance:
    problem = osqp.OSQP()
    mat_p = sparse.csc_matrix(2.0 * np.eye(2))
    vec_q = np.zeros(2)
    mat_constraint = sparse.csc_matrix(co.A)
    lower = np.full(co.b.shape, -np.inf)
    upper = co.b.copy()
    _configure_osqp(
        problem,
        param,
        P=mat_p,
        q=vec_q,
        A=mat_constraint,
        l=lower,
        u=upper,
    )
    result = problem.solve(raise_error=False)
    _require_osqp_solution(result, "Minkowski projection QP")

    critical_point = np.asarray(result.x, dtype=float).reshape(2)
    raw_dual = np.asarray(result.y, dtype=float).reshape(-1)
    if np.min(raw_dual) < -10.0 * param.dual_tol:
        raise RuntimeError("OSQP returned a negative upper-bound dual")
    dual = np.maximum(raw_dual, 0.0)
    slack = co.A @ critical_point - co.b

    active_candidates = np.flatnonzero(
        (np.abs(slack) <= param.active_tol) & (dual > param.dual_tol)
    )
    residual_active = np.flatnonzero(np.abs(slack) <= param.active_tol)
    if active_candidates.size == 0:
        closest = int(np.argmin(np.abs(slack)))
        if abs(slack[closest]) > 10.0 * param.active_tol:
            raise RuntimeError("Projection QP has no numerically active constraint")
        active_candidates = np.asarray([closest], dtype=int)

    active = _independent_active_indices(
        co.A,
        dual,
        active_candidates,
        max(param.geometry_tol, 1e-12),
    )
    stationarity = 2.0 * critical_point + co.A.T @ dual
    complementarity = dual * slack
    distance = float(np.linalg.norm(critical_point))
    primal_residual = float(max(0.0, np.max(slack)))
    stationarity_residual = float(np.linalg.norm(stationarity, ord=np.inf))
    residual_limit = max(
        10.0 * param.osqp_eps_abs,
        10.0 * param.osqp_eps_rel * max(1.0, distance),
        param.geometry_tol,
    )
    if primal_residual > residual_limit or stationarity_residual > residual_limit:
        raise RuntimeError(
            "Projection QP residual check failed: "
            f"primal={primal_residual}, stationarity={stationarity_residual}"
        )

    return _RawSignedDistance(
        signed_distance=distance,
        branch="separated",
        active_indices=active,
        critical_point=critical_point,
        dual=dual,
        diagnostics={
            "qp_status": str(result.info.status),
            "qp_iterations": int(result.info.iter),
            "primal_residual": primal_residual,
            "dual_min": float(np.min(dual)),
            "stationarity_residual": stationarity_residual,
            "complementarity_residual": float(
                np.linalg.norm(complementarity, ord=np.inf)
            ),
            "active_candidate_count": int(active_candidates.size),
            "residual_active_count": int(residual_active.size),
            "degenerate_active": bool(residual_active.size > len(active)),
        },
    )


def _raw_signed_distance_from_co(
    co: ConfigurationObstacle,
    param: MinkowskiCBFOptimizerParam,
) -> _RawSignedDistance:
    origin_inside = bool(np.all(co.b >= -param.containment_tol))
    if not origin_inside:
        return _solve_projection_qp(co, param)

    row_norms = np.linalg.norm(co.A, axis=1)
    ratios = co.b / row_norms
    minimum_ratio = float(np.min(ratios))
    depth = max(0.0, minimum_ratio)
    tied_indices = tuple(
        int(index)
        for index in np.flatnonzero(
            np.abs(ratios - minimum_ratio) <= param.active_tol
        )
    )
    selected = int(np.argmin(ratios))
    active = (selected,) + tuple(
        index for index in tied_indices if index != selected
    )
    normal = co.A[selected]
    critical_point = depth * normal / float(np.linalg.norm(normal))
    branch = "contact" if depth <= param.contact_tol else "penetrating"

    return _RawSignedDistance(
        signed_distance=-depth,
        branch=branch,
        active_indices=active,
        critical_point=critical_point,
        dual=np.zeros(co.A.shape[0]),
        diagnostics={
            "penetration_depth": depth,
            "penetration_ratio_min": minimum_ratio,
            "penetration_tie_count": len(active),
        },
    )


def _match_facet(
    base_normal: np.ndarray,
    base_bound: float,
    co: ConfigurationObstacle,
    param: MinkowskiCBFOptimizerParam,
) -> Tuple[int, bool, Dict[str, Any]]:
    dots = np.clip(co.A @ base_normal, -1.0, 1.0)
    angles = np.arccos(dots)
    candidates = np.flatnonzero(angles <= param.facet_match_tol)
    if candidates.size == 0:
        raise RuntimeError(
            f"No perturbed facet matches base normal; min_angle={np.min(angles)}"
        )

    ordered = sorted(
        (int(index) for index in candidates),
        key=lambda index: (
            float(angles[index]),
            abs(float(co.b[index] - base_bound)),
            index,
        ),
    )
    selected = ordered[0]
    ambiguous = len(ordered) > 1
    return selected, ambiguous, {
        "selected": selected,
        "candidate_count": len(ordered),
        "angle": float(angles[selected]),
    }


def _active_constraint_derivatives(
    robot_vertices_local: np.ndarray,
    obstacle_region: ConvexRegion2D,
    pose: np.ndarray,
    co: ConfigurationObstacle,
    active_indices: Sequence[int],
    param: MinkowskiCBFOptimizerParam,
) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
    active = tuple(int(index) for index in active_indices)
    mat_a = co.A[list(active)]
    derivative_a = np.zeros((len(active), 2, 3), dtype=float)
    derivative_b = np.zeros((len(active), 3), dtype=float)
    derivative_b[:, 0] = -mat_a[:, 0]
    derivative_b[:, 1] = -mat_a[:, 1]

    step = float(param.theta_fd_eps)
    if step <= 0.0:
        raise ValueError("theta_fd_eps must be positive")
    pose_plus = pose.copy()
    pose_minus = pose.copy()
    pose_plus[2] += step
    pose_minus[2] -= step
    co_plus = _build_configuration_obstacle(
        robot_vertices_local,
        obstacle_region,
        pose_plus,
        param.geometry_tol,
    )
    co_minus = _build_configuration_obstacle(
        robot_vertices_local,
        obstacle_region,
        pose_minus,
        param.geometry_tol,
    )

    plus_indices: List[int] = []
    minus_indices: List[int] = []
    match_details: List[Dict[str, Any]] = []
    nonsmooth = False
    for local_index, base_index in enumerate(active):
        plus_index, plus_ambiguous, plus_detail = _match_facet(
            co.A[base_index],
            float(co.b[base_index]),
            co_plus,
            param,
        )
        minus_index, minus_ambiguous, minus_detail = _match_facet(
            co.A[base_index],
            float(co.b[base_index]),
            co_minus,
            param,
        )
        plus_indices.append(plus_index)
        minus_indices.append(minus_index)
        nonsmooth = nonsmooth or plus_ambiguous or minus_ambiguous
        match_details.append({"plus": plus_detail, "minus": minus_detail})

        derivative_a[local_index, :, 2] = (
            co_plus.A[plus_index] - co_minus.A[minus_index]
        ) / (2.0 * step)
        derivative_b[local_index, 2] = (
            co_plus.b[plus_index] - co_minus.b[minus_index]
        ) / (2.0 * step)

    if len(set(plus_indices)) != len(plus_indices):
        nonsmooth = True
    if len(set(minus_indices)) != len(minus_indices):
        nonsmooth = True

    return derivative_a, derivative_b, nonsmooth, {
        "facet_matches": match_details,
        "plus_indices": tuple(plus_indices),
        "minus_indices": tuple(minus_indices),
    }


def _signed_distance_only(
    robot_vertices_local: np.ndarray,
    obstacle_region: ConvexRegion2D,
    pose: np.ndarray,
    param: MinkowskiCBFOptimizerParam,
) -> float:
    co = _build_configuration_obstacle(
        robot_vertices_local,
        obstacle_region,
        pose,
        param.geometry_tol,
    )
    return _raw_signed_distance_from_co(co, param).signed_distance


def _finite_difference_distance_component(
    robot_vertices_local: np.ndarray,
    obstacle_region: ConvexRegion2D,
    pose: np.ndarray,
    axis: int,
    step: float,
    param: MinkowskiCBFOptimizerParam,
) -> float:
    pose_plus = pose.copy()
    pose_minus = pose.copy()
    pose_plus[axis] += step
    pose_minus[axis] -= step
    value_plus = _signed_distance_only(
        robot_vertices_local,
        obstacle_region,
        pose_plus,
        param,
    )
    value_minus = _signed_distance_only(
        robot_vertices_local,
        obstacle_region,
        pose_minus,
        param,
    )
    return float((value_plus - value_minus) / (2.0 * step))


def _separated_gradient(
    robot_vertices_local: np.ndarray,
    obstacle_region: ConvexRegion2D,
    pose: np.ndarray,
    co: ConfigurationObstacle,
    raw: _RawSignedDistance,
    param: MinkowskiCBFOptimizerParam,
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    distance = float(np.linalg.norm(raw.critical_point))
    if distance <= param.contact_tol:
        raise RuntimeError("Separated branch reached the contact tolerance")
    direction = raw.critical_point / distance
    # Translating the robot by dp translates the configuration obstacle by
    # -dp, hence the exact smooth translation gradient is -z/||z||.
    gradient = np.array([-direction[0], -direction[1], 0.0], dtype=float)
    active = raw.active_indices
    mat_a = co.A[list(active)]
    dual = raw.dual[list(active)]
    nonsmooth = False
    derivative_diagnostics: Dict[str, Any] = {}
    kkt_condition = np.inf
    theta_scalar_fd_fallback = False

    try:
        if np.any(dual <= param.dual_tol):
            raise RuntimeError("Reduced KKT contains a weakly active constraint")
        if np.linalg.matrix_rank(mat_a, tol=param.geometry_tol) != len(active):
            raise RuntimeError("Reduced KKT active rows are rank deficient")

        derivative_a, derivative_b, nonsmooth, derivative_diagnostics = (
            _active_constraint_derivatives(
                robot_vertices_local,
                obstacle_region,
                pose,
                co,
                active,
                param,
            )
        )
        dimension = 2
        active_count = len(active)
        # Divide the strict-active complementarity rows by lambda.  This is
        # algebraically equivalent to the paper's differentiated KKT system,
        # but avoids scaling the constraint rows by small dual values.
        kkt = np.block(
            [
                [2.0 * np.eye(dimension), mat_a.T],
                [mat_a, np.zeros((active_count, active_count))],
            ]
        )
        kkt_condition = float(np.linalg.cond(kkt))
        if not np.isfinite(kkt_condition) or (
            kkt_condition > param.kkt_condition_max
        ):
            raise RuntimeError(
                f"Reduced KKT is ill-conditioned: cond={kkt_condition}"
            )
        right_hand_side = np.zeros((dimension + active_count, 3), dtype=float)
        for state_index in range(3):
            right_hand_side[:dimension, state_index] = np.sum(
                derivative_a[:, :, state_index] * dual[:, None],
                axis=0,
            )
            right_hand_side[dimension:, state_index] = (
                derivative_a[:, :, state_index] @ raw.critical_point
                - derivative_b[:, state_index]
            )

        solution = np.linalg.solve(kkt, -right_hand_side)
        jacobian = solution[:dimension, :]
        gradient[2] = float(direction @ jacobian[:, 2])
    except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
        nonsmooth = True
        theta_scalar_fd_fallback = True
        derivative_diagnostics = {
            **derivative_diagnostics,
            "theta_fallback_reason": f"{type(exc).__name__}: {exc}",
        }

    if nonsmooth:
        gradient[2] = _finite_difference_distance_component(
            robot_vertices_local,
            obstacle_region,
            pose,
            2,
            param.theta_fd_eps,
            param,
        )
        theta_scalar_fd_fallback = True

    return np.asarray(gradient, dtype=float), nonsmooth, {
        **derivative_diagnostics,
        "kkt_condition": kkt_condition,
        "translation_gradient_closed_form": True,
        "theta_scalar_fd_fallback": theta_scalar_fd_fallback,
    }


def _penetration_gradient(
    robot_vertices_local: np.ndarray,
    obstacle_region: ConvexRegion2D,
    pose: np.ndarray,
    co: ConfigurationObstacle,
    raw: _RawSignedDistance,
    param: MinkowskiCBFOptimizerParam,
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    selected = int(raw.active_indices[0])
    # With normalized facets, depth=b_k and sd=-b_k on a fixed active facet.
    # Since db/dp=-a, the exact translation gradient is the selected normal.
    gradient = np.array([co.A[selected, 0], co.A[selected, 1], 0.0])
    nonsmooth = len(raw.active_indices) > 1 or raw.branch == "contact"
    derivative_diagnostics: Dict[str, Any] = {}
    try:
        derivative_a, derivative_b, match_nonsmooth, derivative_diagnostics = (
            _active_constraint_derivatives(
                robot_vertices_local,
                obstacle_region,
                pose,
                co,
                (selected,),
                param,
            )
        )
        del derivative_a
        nonsmooth = nonsmooth or match_nonsmooth
        gradient[2] = -derivative_b[0, 2]
    except (RuntimeError, ValueError) as exc:
        nonsmooth = True
        derivative_diagnostics = {
            **derivative_diagnostics,
            "theta_fallback_reason": f"{type(exc).__name__}: {exc}",
        }
    if nonsmooth:
        gradient[2] = _finite_difference_distance_component(
            robot_vertices_local,
            obstacle_region,
            pose,
            2,
            param.theta_fd_eps,
            param,
        )

    return gradient, nonsmooth, {
        **derivative_diagnostics,
        "penetration_gradient_policy": "fixed-normalized-active-facet",
        "translation_gradient_closed_form": True,
        "theta_scalar_fd_fallback": bool(nonsmooth),
    }


def _signed_distance_and_gradient(
    robot_vertices_local: Sequence[Sequence[float]],
    obstacle_region: ConvexRegion2D,
    pose: Sequence[float],
    param: MinkowskiCBFOptimizerParam,
) -> SignedDistanceResult:
    """Compute signed distance, CBF value, and state gradient for one pair."""

    robot_local = _as_vertices(robot_vertices_local, "robot_vertices_local")
    pose_array = np.asarray(pose, dtype=float).reshape(-1)
    if pose_array.size != 3:
        raise ValueError("pose must have shape (3,)")
    co = _build_configuration_obstacle(
        robot_local,
        obstacle_region,
        pose_array,
        param.geometry_tol,
    )
    raw = _raw_signed_distance_from_co(co, param)
    diagnostics = dict(raw.diagnostics)

    try:
        if raw.branch == "separated":
            gradient, nonsmooth, gradient_diagnostics = _separated_gradient(
                robot_local,
                obstacle_region,
                pose_array,
                co,
                raw,
                param,
            )
        else:
            gradient, nonsmooth, gradient_diagnostics = _penetration_gradient(
                robot_local,
                obstacle_region,
                pose_array,
                co,
                raw,
                param,
            )
    except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
        if raw.branch == "separated":
            distance = float(np.linalg.norm(raw.critical_point))
            if distance <= param.contact_tol:
                raise RuntimeError("Cannot select a finite contact gradient") from exc
            translation_gradient = -raw.critical_point / distance
        else:
            translation_gradient = co.A[int(raw.active_indices[0])]
        gradient = np.array(
            [
                translation_gradient[0],
                translation_gradient[1],
                _finite_difference_distance_component(
                    robot_local,
                    obstacle_region,
                    pose_array,
                    2,
                    param.theta_fd_eps,
                    param,
                ),
            ],
            dtype=float,
        )
        nonsmooth = True
        gradient_diagnostics = {
            "theta_scalar_fd_fallback": True,
            "translation_gradient_closed_form": True,
            "fallback_reason": f"{type(exc).__name__}: {exc}",
        }

    if not np.all(np.isfinite(gradient)):
        raise RuntimeError("Signed-distance gradient contains NaN or Inf")
    diagnostics.update(gradient_diagnostics)
    diagnostics["nonsmooth"] = bool(nonsmooth)

    return SignedDistanceResult(
        signed_distance=float(raw.signed_distance),
        h=float(raw.signed_distance - param.d_safe),
        gradient=np.asarray(gradient, dtype=float).reshape(3),
        branch=raw.branch,
        active_indices=raw.active_indices,
        nonsmooth=bool(nonsmooth),
        diagnostics=diagnostics,
        critical_point=raw.critical_point.copy(),
        configuration_obstacle=co,
    )


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _validate_param(param: MinkowskiCBFOptimizerParam) -> None:
    if param.d_safe < 0.0:
        raise ValueError("d_safe must be nonnegative")
    if param.gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if param.epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")
    if param.vmin > param.vmax or param.omegamin > param.omegamax:
        raise ValueError("Invalid input bounds")
    positive_values = {
        "control_weight_v": param.control_weight_v,
        "control_weight_omega": param.control_weight_omega,
        "clf_rate": param.clf_rate,
        "clf_heading_weight": param.clf_heading_weight,
        "clf_slack_weight": param.clf_slack_weight,
        "geometry_tol": param.geometry_tol,
        "containment_tol": param.containment_tol,
        "active_tol": param.active_tol,
        "dual_tol": param.dual_tol,
        "contact_tol": param.contact_tol,
        "facet_match_tol": param.facet_match_tol,
        "theta_fd_eps": param.theta_fd_eps,
        "kkt_condition_max": param.kkt_condition_max,
        "osqp_eps_abs": param.osqp_eps_abs,
        "osqp_eps_rel": param.osqp_eps_rel,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0.0]
    if invalid:
        raise ValueError(f"Parameters must be positive: {invalid}")
    if param.osqp_max_iter <= 0:
        raise ValueError("osqp_max_iter must be positive")


class _MinkowskiCBFSolution:
    """Small solution wrapper matching BaseController's expected interface."""

    def __init__(
        self,
        state_trajectory: np.ndarray,
        input_trajectory: np.ndarray,
        solver_status: str,
        diagnostics: Dict[str, Any],
    ) -> None:
        self._state_trajectory = np.asarray(state_trajectory, dtype=float)
        self._input_trajectory = np.asarray(input_trajectory, dtype=float)
        self._solver_status = str(solver_status)
        self.diagnostics = diagnostics

    def value(self, variable: str) -> np.ndarray:
        if variable == "x":
            return self._state_trajectory.copy()
        if variable == "u":
            return self._input_trajectory.copy()
        raise NotImplementedError("Only 'x' and 'u' are available")

    def get_state_trajectory(self) -> np.ndarray:
        return self.value("x")

    def get_input_trajectory(self) -> np.ndarray:
        return self.value("u")

    def stats(self) -> Dict[str, Any]:
        return {
            "return_status": "success",
            "solver_status": self._solver_status,
        }


class MinkowskiCBFOptimizer:
    """Pointwise hard-CBF/soft-CLF safety controller."""

    def __init__(
        self,
        variables: Optional[Dict[str, Any]] = None,
        costs: Optional[Dict[str, Any]] = None,
        dynamics_opt: Any = None,
    ) -> None:
        del costs, dynamics_opt
        self.variables = {"x": "x", "u": "u"}
        if variables:
            self.variables.update(variables)
        self.prints_compact_runtime_summary = False
        self.solver: Optional[osqp.OSQP] = None
        self.solver_times: List[float] = []
        self.param: Optional[MinkowskiCBFOptimizerParam] = None
        self.system: Any = None
        self.state: Any = None
        self.reference_trajectory: Optional[np.ndarray] = None
        self.obstacles: List[ConvexRegion2D] = []
        self.robot_components: List[np.ndarray] = []
        self.last_pair_results: List[SignedDistanceResult] = []
        self.last_diagnostics: Dict[str, Any] = {}
        self._last_solution: Optional[_MinkowskiCBFSolution] = None
        self._is_setup = False

    def reset(self) -> None:
        self.solver = None
        self.solver_times = []
        self.param = None
        self.system = None
        self.state = None
        self.reference_trajectory = None
        self.obstacles = []
        self.robot_components = []
        self.last_pair_results = []
        self.last_diagnostics = {}
        self._last_solution = None
        self._is_setup = False

    def setup(
        self,
        param: MinkowskiCBFOptimizerParam,
        system: Any,
        reference_trajectory: Sequence[Sequence[float]],
        obstacles: Sequence[ConvexRegion2D],
    ) -> None:
        _validate_param(param)
        if not hasattr(system, "_state") or not hasattr(system._state, "_x"):
            raise TypeError("system must provide system._state._x")
        if not hasattr(system, "_geometry") or not hasattr(
            system._geometry,
            "equiv_rep",
        ):
            raise TypeError("system must provide system._geometry.equiv_rep()")
        if not hasattr(system, "_dynamics") or not hasattr(
            system._dynamics,
            "forward_dynamics",
        ):
            raise TypeError("system must provide forward_dynamics()")

        state_vector = np.asarray(system._state._x, dtype=float).reshape(-1)
        if state_vector.size != 3 or not np.all(np.isfinite(state_vector)):
            raise ValueError("MinkowskiCBFOptimizer requires a finite 3-state pose")

        components: List[np.ndarray] = []
        for component in system._geometry.equiv_rep():
            if not isinstance(component, ConvexRegion2D) or not hasattr(
                component,
                "get_ccw_vertices",
            ):
                raise TypeError(
                    "MinkowskiCBFOptimizer supports only ConvexRegion2D "
                    "robot components"
                )
            components.append(
                _as_vertices(component.get_ccw_vertices(), "robot_component")
            )
        if not components:
            raise ValueError("Robot geometry has no convex component")

        obstacle_list = list(obstacles)
        for obstacle in obstacle_list:
            if not isinstance(obstacle, ConvexRegion2D) or not hasattr(
                obstacle,
                "get_ccw_vertices",
            ):
                raise TypeError(
                    "MinkowskiCBFOptimizer supports only ConvexRegion2D obstacles"
                )

        if reference_trajectory is None:
            reference = state_vector.reshape(1, 3)
        else:
            reference = np.asarray(reference_trajectory, dtype=float)
            if reference.size == 0:
                reference = state_vector.reshape(1, 3)
            if reference.ndim != 2 or reference.shape[1] < 3:
                raise ValueError("reference_trajectory must have shape (N, >=3)")
            if not np.all(np.isfinite(reference[:, :3])):
                raise ValueError("reference_trajectory contains NaN or Inf")

        self.param = param
        self.system = system
        self.state = system._state
        self.reference_trajectory = reference[:, :3].copy()
        self.obstacles = obstacle_list
        self.robot_components = components
        self._is_setup = True

    @staticmethod
    def _input_matrix(state: np.ndarray) -> np.ndarray:
        theta = float(state[2])
        return np.array(
            [
                [math.cos(theta), 0.0],
                [math.sin(theta), 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )

    def _clf_value_and_gradient(
        self,
        state: np.ndarray,
        reference: np.ndarray,
    ) -> Tuple[float, np.ndarray, float, str]:
        if self.param is None:
            raise RuntimeError("Optimizer is not set up")

        position_error = state[:2] - reference[:2]
        distance_squared = float(position_error @ position_error)
        if distance_squared > max(self.param.geometry_tol**2, 1e-12):
            desired_heading = math.atan2(
                float(reference[1] - state[1]),
                float(reference[0] - state[0]),
            )
            heading_mode = "bearing"
            heading_error = _wrap_angle(float(state[2] - desired_heading))
            heading_error_gradient = np.array(
                [
                    position_error[1] / distance_squared,
                    -position_error[0] / distance_squared,
                    1.0,
                ],
                dtype=float,
            )
        else:
            heading_mode = "reference_heading"
            heading_error = _wrap_angle(float(state[2] - reference[2]))
            heading_error_gradient = np.array([0.0, 0.0, 1.0])

        gradient = np.array(
            [position_error[0], position_error[1], 0.0],
            dtype=float,
        )
        gradient += (
            self.param.clf_heading_weight
            * heading_error
            * heading_error_gradient
        )
        value = 0.5 * (
            distance_squared
            + self.param.clf_heading_weight * heading_error**2
        )
        return float(value), gradient, float(heading_error), heading_mode

    def _solve_control_qp(
        self,
        state: np.ndarray,
        pair_results: Sequence[SignedDistanceResult],
    ) -> Tuple[np.ndarray, float, str, Dict[str, Any]]:
        if self.param is None or self.reference_trajectory is None:
            raise RuntimeError("Optimizer is not set up")
        param = self.param
        input_matrix = self._input_matrix(state)
        reference = self.reference_trajectory[-1, :3]

        (
            clf_value,
            gradient_v,
            clf_heading_error,
            clf_heading_mode,
        ) = self._clf_value_and_gradient(
            state,
            reference,
        )
        clf_control_row = gradient_v @ input_matrix

        rows: List[np.ndarray] = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        lower: List[float] = [param.vmin, param.omegamin, 0.0]
        upper: List[float] = [param.vmax, param.omegamax, np.inf]

        cbf_rows: List[np.ndarray] = []
        cbf_lower_bounds: List[float] = []
        for result in pair_results:
            control_row = np.asarray(result.gradient @ input_matrix, dtype=float)
            row = np.array([control_row[0], control_row[1], 0.0])
            lower_bound = float(param.epsilon - param.gamma * result.h)
            rows.append(row)
            lower.append(lower_bound)
            upper.append(np.inf)
            cbf_rows.append(row)
            cbf_lower_bounds.append(lower_bound)

        clf_row = np.array(
            [clf_control_row[0], clf_control_row[1], -1.0],
            dtype=float,
        )
        clf_upper = float(-param.clf_rate * clf_value)
        rows.append(clf_row)
        lower.append(-np.inf)
        upper.append(clf_upper)

        hessian = 2.0 * np.diag(
            [
                param.control_weight_v,
                param.control_weight_omega,
                param.clf_slack_weight,
            ]
        )
        problem = osqp.OSQP()
        _configure_osqp(
            problem,
            param,
            P=sparse.csc_matrix(hessian),
            q=np.zeros(3),
            A=sparse.csc_matrix(np.vstack(rows)),
            l=np.asarray(lower, dtype=float),
            u=np.asarray(upper, dtype=float),
        )
        qp_result = problem.solve(raise_error=False)
        self.solver = problem
        _require_osqp_solution(qp_result, "Minkowski CLF-CBF control QP")

        decision = np.asarray(qp_result.x, dtype=float).reshape(3)
        if not np.all(np.isfinite(decision)):
            raise RuntimeError("Minkowski control QP returned NaN or Inf")
        control = decision[:2]
        slack = float(decision[2])
        cbf_margins = [
            float(row @ decision - bound)
            for row, bound in zip(cbf_rows, cbf_lower_bounds)
        ]
        clf_margin = float(clf_upper - clf_row @ decision)
        bound_margin = min(
            control[0] - param.vmin,
            param.vmax - control[0],
            control[1] - param.omegamin,
            param.omegamax - control[1],
            slack,
        )
        feasibility_tolerance = max(
            10.0 * param.osqp_eps_abs,
            10.0
            * param.osqp_eps_rel
            * max(1.0, float(np.linalg.norm(decision, ord=np.inf))),
            param.geometry_tol,
        )
        minimum_cbf_margin = (
            float(np.min(cbf_margins)) if cbf_margins else np.inf
        )
        if min(minimum_cbf_margin, clf_margin, bound_margin) < (
            -feasibility_tolerance
        ):
            raise RuntimeError(
                "Minkowski control QP residual check failed: "
                f"cbf={minimum_cbf_margin}, clf={clf_margin}, "
                f"bounds={bound_margin}"
            )

        diagnostics = {
            "solver_status": str(qp_result.info.status),
            "solver_iterations": int(qp_result.info.iter),
            "decision": decision.copy(),
            "reference": reference.copy(),
            "clf_value": clf_value,
            "clf_gradient": gradient_v.copy(),
            "clf_control_row": np.asarray(clf_control_row, dtype=float),
            "clf_heading_error": clf_heading_error,
            "clf_heading_mode": clf_heading_mode,
            "clf_margin": clf_margin,
            "cbf_margins": np.asarray(cbf_margins, dtype=float),
            "minimum_cbf_margin": minimum_cbf_margin,
            "minimum_bound_margin": float(bound_margin),
            "feasibility_tolerance": feasibility_tolerance,
        }
        return control, slack, str(qp_result.info.status), diagnostics

    def solve_nlp(self) -> _MinkowskiCBFSolution:
        if not self._is_setup or self.param is None or self.system is None:
            raise RuntimeError("setup() must be called before solve_nlp()")

        start = time.perf_counter()
        try:
            state = np.asarray(self.state._x, dtype=float).reshape(3).copy()
            pair_results: List[SignedDistanceResult] = []
            for component in self.robot_components:
                for obstacle in self.obstacles:
                    pair_results.append(
                        _signed_distance_and_gradient(
                            component,
                            obstacle,
                            state,
                            self.param,
                        )
                    )

            control, clf_slack, solver_status, control_diagnostics = (
                self._solve_control_qp(state, pair_results)
            )
            timestep = float(getattr(self.system, "_dt", 0.1))
            next_state = np.asarray(
                self.system._dynamics.forward_dynamics(
                    state,
                    control,
                    timestep,
                ),
                dtype=float,
            ).reshape(3)

            state_trajectory = np.column_stack([state, next_state])
            input_trajectory = control.reshape(2, 1)
            self.last_pair_results = pair_results
            self.last_diagnostics = {
                **control_diagnostics,
                "clf_slack": clf_slack,
                "pair_count": len(pair_results),
                "signed_distances": np.asarray(
                    [result.signed_distance for result in pair_results],
                    dtype=float,
                ),
                "cbf_values": np.asarray(
                    [result.h for result in pair_results],
                    dtype=float,
                ),
                "pair_branches": tuple(
                    result.branch for result in pair_results
                ),
                "pair_nonsmooth": tuple(
                    result.nonsmooth for result in pair_results
                ),
            }
            solution = _MinkowskiCBFSolution(
                state_trajectory=state_trajectory,
                input_trajectory=input_trajectory,
                solver_status=solver_status,
                diagnostics=self.last_diagnostics,
            )
            self._last_solution = solution
            return solution
        finally:
            self.solver_times.append(time.perf_counter() - start)

    def get_last_signed_distance(self) -> Optional[float]:
        if not self.last_pair_results:
            return None
        return float(
            min(result.signed_distance for result in self.last_pair_results)
        )

    def get_last_cbf_results(self) -> List[SignedDistanceResult]:
        return list(self.last_pair_results)
