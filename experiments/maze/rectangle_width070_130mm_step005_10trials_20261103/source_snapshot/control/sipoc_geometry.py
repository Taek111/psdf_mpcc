"""Convex geometry and ellipsoid propagation for the SIPOC robust controller.

The lower problem minimizes the distance between ``p + R B + E(P)`` and a
convex obstacle, where ``E(P) = {L z: ||z|| <= 1}`` and ``L L.T = P``.
This is the dilated-polygon distance problem in ``references/sipoc/sip_paper.pdf``
and the mobile-robot experiment's ellipsoid-to-polygon QCQP, extended to a
polygon obstacle instead of one obstacle point.  The implementation uses a
second-order cone formulation and remains valid for singular or zero P.

Uncertainty propagation follows the experiment's discrete linearized update
``P_next = (A + B K) P (A + B K).T + W``.  P is an ellipsoid shape matrix;
callers must apply any confidence scaling to a statistical covariance before
using it as a geometric uncertainty set.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize_scalar
from scipy.spatial import ConvexHull, QhullError

try:
    import clarabel
except ImportError:
    clarabel = None


def rotation(theta: float) -> np.ndarray:
    """Return the planar body-to-world rotation."""
    theta = float(theta)
    if not np.isfinite(theta):
        raise ValueError("heading must be finite")
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array([[cosine, -sine], [sine, cosine]])


def psd_matrix(
    matrix: np.ndarray, dimension: Optional[int] = None, name: str = "shape"
) -> np.ndarray:
    """Validate a symmetric PSD matrix and remove roundoff-sized negative modes."""
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if dimension is not None and value.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain finite values")
    scale = max(1.0, float(np.linalg.norm(value, ord=2)))
    tolerance = 1e-10 * scale
    if np.max(np.abs(value - value.T), initial=0.0) > tolerance:
        raise ValueError(f"{name} must be symmetric")
    value = (value + value.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if eigenvalues.min(initial=0.0) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if np.any(eigenvalues < 0.0):
        value = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return value


def psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Return an eigenfactor L with L L.T = P, including rank-deficient P."""
    value = psd_matrix(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    return eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))


@dataclass(frozen=True)
class ConvexPolygon:
    """A bounded, full-dimensional convex polygon in CCW vertex order."""

    vertices: np.ndarray
    A: np.ndarray
    b: np.ndarray

    @property
    def radius(self) -> float:
        """Maximum vertex radius about the model origin, including offsets."""
        return float(np.linalg.norm(self.vertices, axis=1).max())

    @classmethod
    def from_vertices(cls, vertices: np.ndarray) -> "ConvexPolygon":
        """Construct the convex hull; repeated/interior input points are allowed."""
        points = np.asarray(vertices, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("polygon vertices must have shape (N, 2), N >= 3")
        if not np.all(np.isfinite(points)):
            raise ValueError("polygon vertices must be finite")
        points = np.unique(points, axis=0)
        try:
            hull = ConvexHull(points)
        except QhullError as exc:
            raise ValueError("polygon must have a nonzero planar area") from exc
        points = points[hull.vertices].copy()
        edges = np.roll(points, -1, axis=0) - points
        normals = np.column_stack((edges[:, 1], -edges[:, 0]))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        bounds = np.einsum("ij,ij->i", normals, points)
        points.setflags(write=False)
        normals.setflags(write=False)
        bounds.setflags(write=False)
        return cls(points, normals, bounds)

    @classmethod
    def from_halfspaces(cls, A: np.ndarray, b: np.ndarray) -> "ConvexPolygon":
        """Enumerate a bounded polygon from A x <= b without optional pypoman."""
        normals = np.asarray(A, dtype=float)
        bounds = np.asarray(b, dtype=float).reshape(-1)
        if normals.ndim != 2 or normals.shape != (len(bounds), 2):
            raise ValueError("polygon halfspaces must have A shape (N, 2) and b length N")
        if not np.all(np.isfinite(normals)) or not np.all(np.isfinite(bounds)):
            raise ValueError("polygon halfspaces must be finite")
        lengths = np.linalg.norm(normals, axis=1)
        zero = lengths <= np.finfo(float).tiny
        if np.any(bounds[zero] < 0.0):
            raise ValueError("polygon halfspaces are infeasible")
        normals, bounds, lengths = normals[~zero], bounds[~zero], lengths[~zero]
        normals, bounds = normals / lengths[:, None], bounds / lengths
        # A few finite intersections alone do not establish boundedness.
        for direction in np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]):
            result = linprog(direction, A_ub=normals, b_ub=bounds,
                             bounds=[(None, None), (None, None)], method="highs")
            if not result.success:
                raise ValueError("polygon halfspaces must define a nonempty bounded region")
        points = []
        feasibility_tolerance = 1e-9 * max(1.0, float(np.max(np.abs(bounds), initial=0.0)))
        for first, second in combinations(range(len(bounds)), 2):
            pair = normals[[first, second]]
            if abs(float(np.linalg.det(pair))) <= 1e-12:
                continue
            point = np.linalg.solve(pair, bounds[[first, second]])
            if np.all(normals @ point <= bounds + feasibility_tolerance):
                points.append(point)
        return cls.from_vertices(np.asarray(points))

    @classmethod
    def from_geometry(cls, geometry) -> "ConvexPolygon":
        """Adapt repository regions by duck typing without importing robot models."""
        if isinstance(geometry, cls):
            return geometry
        if hasattr(geometry, "get_convex_rep"):
            normals, bounds = geometry.get_convex_rep()
            return cls.from_halfspaces(normals, bounds)
        if hasattr(geometry, "get_ccw_vertices"):
            return cls.from_vertices(geometry.get_ccw_vertices())
        if hasattr(geometry, "vertices"):
            return cls.from_vertices(geometry.vertices)
        return cls.from_vertices(geometry)

    # Convenient name used by callers adapting existing polygon objects.
    from_polygon = from_geometry


@dataclass(frozen=True)
class DistanceWitness:
    """Lower-problem witnesses and an exact support gap along a unit normal.

    ``normal`` points from the obstacle to the dilated robot.  ``delta`` is
    the translational uncertainty at the witness, so the envelope gradient
    must use ``p + R gamma + delta - obstacle_point``.  In overlap, distance
    is zero and support_gap is nonpositive; the normal is a recovery direction.
    """

    distance: float
    gamma: np.ndarray
    obstacle_point: np.ndarray
    delta: np.ndarray
    normal: np.ndarray
    support_gap: float
    overlapping: bool


def _support_gap(
    normal: np.ndarray, position: np.ndarray, rotated_vertices: np.ndarray,
    shape: np.ndarray, obstacle: ConvexPolygon,
) -> float:
    variance = max(0.0, float(normal @ shape @ normal))
    return float(normal @ position + np.min(rotated_vertices @ normal)
                 - np.max(obstacle.vertices @ normal) - np.sqrt(variance))


def _recovery_normal(
    position: np.ndarray, R: np.ndarray, shape: np.ndarray,
    body: ConvexPolygon, obstacle: ConvexPolygon,
) -> np.ndarray:
    """Choose a nondegenerate support direction even when closest points coincide."""
    angles = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    edge_normals = np.vstack((body.A @ R.T, obstacle.A))
    edge_angles = np.arctan2(edge_normals[:, 1], edge_normals[:, 0])
    angles = np.concatenate((angles, edge_angles, edge_angles + np.pi))
    center_delta = position + np.mean(body.vertices @ R.T, axis=0) - np.mean(
        obstacle.vertices, axis=0)
    if np.linalg.norm(center_delta) > 1e-12:
        angles = np.append(angles, np.arctan2(center_delta[1], center_delta[0]))
    rotated = body.vertices @ R.T

    def gap_at(angle):
        normal = np.array([np.cos(angle), np.sin(angle)])
        return _support_gap(normal, position, rotated, shape, obstacle)

    gaps = np.array([gap_at(angle) for angle in angles])
    best_index = int(np.argmax(gaps))
    best_angle, best_gap = float(angles[best_index]), float(gaps[best_index])
    # Edge normals retain the exact nonsmooth candidates; refinement improves
    # curved ellipse directions between them without replacing their values.
    spacing = 2.0 * np.pi / 64.0
    for index in np.argsort(gaps)[-8:]:
        angle = float(angles[index])
        result = minimize_scalar(lambda value: -gap_at(value),
                                 bounds=(angle - spacing, angle + spacing),
                                 method="bounded", options={"xatol": 1e-10})
        if result.success and -float(result.fun) > best_gap:
            best_angle, best_gap = float(result.x), -float(result.fun)
    return np.array([np.cos(best_angle), np.sin(best_angle)])


def polygon_ellipse_distance(
    position: np.ndarray, heading: float, shape_xy: np.ndarray,
    body: ConvexPolygon, obstacle: ConvexPolygon,
) -> DistanceWitness:
    """Solve min 1/2 ||p + R gamma + L z - q||^2 with ||z|| <= 1.

    Body/obstacle halfspaces constrain gamma/q.  Clarabel failures and
    infeasible returned witnesses are errors, never nominal substitutions.
    """
    if clarabel is None:
        raise RuntimeError("Dilated polygon distance requires the 'clarabel' package")
    body, obstacle = ConvexPolygon.from_geometry(body), ConvexPolygon.from_geometry(obstacle)
    position = np.asarray(position, dtype=float).reshape(-1)
    if position.shape != (2,) or not np.all(np.isfinite(position)):
        raise ValueError("position must have two finite coordinates")
    shape = psd_matrix(shape_xy, dimension=2, name="translation uncertainty shape")
    R, L = rotation(heading), psd_sqrt(shape)
    # Solve in a translated frame: q_relative = q_world - position.  This
    # removes the large global p term from the objective and keeps witnesses
    # and support gaps accurate when the same map has a large world offset.
    relative_vertices = obstacle.vertices - position
    # This is b_relative = b - A p, evaluated from rebased vertices to avoid
    # subtracting two large halfspace offsets.  The hull halfspaces are tight.
    relative_bounds = np.max(obstacle.A @ relative_vertices.T, axis=1)
    relative_obstacle = ConvexPolygon(relative_vertices, obstacle.A, relative_bounds)
    residual_map = np.hstack((R, L, -np.eye(2)))
    quadratic = sparse.triu(sparse.csc_matrix(residual_map.T @ residual_map), format="csc")
    linear = np.zeros(6)
    count = len(body.b) + len(obstacle.b)
    constraints = np.zeros((count + 3, 6))
    constraints[:len(body.b), :2] = body.A
    constraints[len(body.b):count, 4:] = obstacle.A
    # b - A v = [1, z] belongs to the Lorentz cone.
    constraints[count + 1:, 2:4] = -np.eye(2)
    bounds = np.concatenate((body.b, relative_bounds, [1.0, 0.0, 0.0]))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 150
    settings.tol_gap_abs = 1e-10
    settings.tol_gap_rel = 1e-10
    settings.tol_feas = 1e-10
    cones = [clarabel.NonnegativeConeT(count), clarabel.SecondOrderConeT(3)]
    try:
        solver = clarabel.DefaultSolver(quadratic, linear,
                                        sparse.csc_matrix(constraints), bounds,
                                        cones, settings)
        solution = solver.solve()
    except Exception as exc:
        raise RuntimeError("dilated polygon distance solve failed") from exc
    if str(solution.status) not in {"Solved", "AlmostSolved"}:
        raise RuntimeError(f"dilated polygon distance solve failed: {solution.status}")
    variables = np.asarray(solution.x, dtype=float)
    gamma, z, relative_point = variables[:2], variables[2:4], variables[4:]
    feasibility_scale = max(1.0, body.radius,
                            float(np.linalg.norm(np.ptp(obstacle.vertices, axis=0))),
                            float(np.linalg.norm(L, ord=2)))
    tolerance = 2e-7 * feasibility_scale
    if (not np.all(np.isfinite(variables))
            or np.max(body.A @ gamma - body.b) > tolerance
            or np.max(obstacle.A @ relative_point - relative_bounds) > tolerance
            or np.linalg.norm(z) > 1.0 + 2e-7):
        raise RuntimeError("dilated polygon distance returned an infeasible witness")
    delta = L @ z
    residual = R @ gamma + delta - relative_point
    distance = float(np.linalg.norm(residual))
    overlapping = distance <= 1e-7 * feasibility_scale
    if overlapping:
        normal = _recovery_normal(np.zeros(2), R, shape, body, relative_obstacle)
        # Coincident closest points have zero derivative.  Exact support
        # vertices and the worst ellipse point yield a valid recovery plane.
        gamma = body.vertices[int(np.argmin((body.vertices @ R.T) @ normal))].copy()
        relative_point = relative_vertices[int(np.argmax(relative_vertices @ normal))].copy()
        uncertainty_radius = np.sqrt(max(0.0, float(normal @ shape @ normal)))
        delta = (-shape @ normal / uncertainty_radius if uncertainty_radius > 0.0
                 else np.zeros(2))
        distance = 0.0
    else:
        normal = residual / distance
    gap = _support_gap(normal, np.zeros(2), body.vertices @ R.T, shape, relative_obstacle)
    obstacle_point = relative_point + position
    return DistanceWitness(distance, gamma.copy(), obstacle_point.copy(),
                           delta.copy(), normal.copy(), gap, overlapping)


def propagate_covariance(
    initial: np.ndarray, state_jacobians: Sequence[np.ndarray],
    input_jacobians: Optional[Sequence[np.ndarray]], process_covariance: np.ndarray,
    feedback_gain: Optional[np.ndarray] = None,
    step_scales: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Propagate an ellipsoid shape/covariance along supplied discrete Jacobians.

    ``feedback_gain`` follows SIPOC's plus-sign convention ``A + B K``.
    ``step_scales`` scales W relative to the interval where W was calibrated.
    With equal sample periods its values are all one.
    """
    initial = psd_matrix(initial, name="initial uncertainty")
    dimension = initial.shape[0]
    noise = psd_matrix(process_covariance, dimension, name="process uncertainty")
    count = len(state_jacobians)
    if input_jacobians is not None and len(input_jacobians) != count:
        raise ValueError("state and input Jacobian horizons must match")
    if feedback_gain is not None and input_jacobians is None:
        raise ValueError("feedback requires input Jacobians")
    scales = np.ones(count) if step_scales is None else np.asarray(step_scales, dtype=float)
    if scales.shape != (count,) or np.any(scales < 0.0) or not np.all(np.isfinite(scales)):
        raise ValueError("step_scales must be finite, nonnegative and match the horizon")
    gain = None if feedback_gain is None else np.asarray(feedback_gain, dtype=float)
    result = np.empty((count + 1, dimension, dimension))
    result[0] = initial
    for stage, jacobian in enumerate(state_jacobians):
        A = np.asarray(jacobian, dtype=float)
        if A.shape != (dimension, dimension) or not np.all(np.isfinite(A)):
            raise ValueError("state Jacobians must be finite square state matrices")
        if input_jacobians is not None:
            B = np.asarray(input_jacobians[stage], dtype=float)
            if B.ndim != 2 or B.shape[0] != dimension or not np.all(np.isfinite(B)):
                raise ValueError("input Jacobians must have finite state-by-input shape")
            if gain is not None:
                if gain.shape != (B.shape[1], dimension) or not np.all(np.isfinite(gain)):
                    raise ValueError("feedback gain must have finite input-by-state shape")
                A = A + B @ gain
        propagated = A @ result[stage] @ A.T + scales[stage] * noise
        result[stage + 1] = psd_matrix(propagated, dimension, name="propagated uncertainty")
    return result
