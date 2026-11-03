"""U-OBCA for a rectangular robot and deterministic static polygon obstacles.

This specializes Eqs. (19), (22), and (29) of references/u-obca.pdf. Pose
noise is zero-mean Gaussian, with position independent of heading. Covariance
matrices are evaluated at the warm-start trajectory and frozen during each
solve (Remark 11); the mean pose and separating normal remain NLP variables.
"""

import math
from statistics import NormalDist

import casadi as ca
import numpy as np

from control.obca_optimizer import OBCAOptimizer, OBCAOptimizerParam
from models.geometry_utils import ConvexRegion2D


_VARIANCE_EPSILON = 1e-20


def wasserstein_eta(alpha: float, radius: float) -> float:
    """Solve Eq. (12) for a Gaussian reference by scalar bisection.

    ``radius`` uses the paper's Mahalanobis-scaled Wasserstein distance;
    it is a dimensionless ambiguity radius, not a geometric clearance.
    """
    alpha, radius = float(alpha), float(radius)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 0.5:
        raise ValueError("alpha must be in (0, 0.5]")
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("Wasserstein radius must be finite and nonnegative")
    z = -NormalDist().inv_cdf(alpha)
    if radius == 0.0:
        return z

    phi_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    lower = z
    # phi(eta) - eta * (1 - Phi(eta)) >= 0 gives this upper bound.
    upper = (radius + phi_z) / alpha
    if not math.isfinite(upper):
        raise ValueError("Risk level and radius give an unrepresentable eta")
    for _ in range(80):
        eta = 0.5 * lower + 0.5 * upper
        cdf_difference = alpha - 0.5 * math.erfc(eta / math.sqrt(2.0))
        phi_eta = math.exp(-0.5 * eta * eta) / math.sqrt(2.0 * math.pi)
        if eta * cdf_difference + phi_eta - phi_z < radius:
            lower = eta
        else:
            upper = eta
    return upper


def rotation_moments(
    heading: float, heading_variance: float, vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and covariance of R(heading + w) @ vector for Gaussian w.

    Heading and its variance use radians and radians squared, respectively.
    The covariance formula remains accurate for very small heading variance.
    """
    heading, variance = float(heading), float(heading_variance)
    vector = np.asarray(vector, dtype=float).reshape(2)
    if not math.isfinite(heading) or not np.all(np.isfinite(vector)):
        raise ValueError("Heading and vector must be finite")
    if not math.isfinite(variance) or variance < 0.0:
        raise ValueError("Heading variance must be finite and nonnegative")
    cosine, sine = math.cos(heading), math.sin(heading)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    rotated = rotation @ vector
    perpendicular = np.array([-rotated[1], rotated[0]])
    parallel_variance = 0.5 * math.expm1(-variance) ** 2
    perpendicular_variance = -0.5 * math.expm1(-2.0 * variance)
    covariance = (
        parallel_variance * np.outer(rotated, rotated)
        + perpendicular_variance * np.outer(perpendicular, perpendicular)
    )
    return math.exp(-0.5 * variance) * rotated, covariance


class UOBCAOptimizerParam(OBCAOptimizerParam):
    def __init__(self):
        super().__init__()
        # Per obstacle and predicted stage, not a whole-trajectory risk bound.
        self.alpha = 0.01
        self.wasserstein_radius = 0.01
        # The inherited current-pose distance cutoff does not account for risk.
        self.use_obstacle_cutoff = False


class UOBCAOptimizer(OBCAOptimizer):
    """Reuse OBCA's dynamics, costs, solver, and graph with U-OBCA certificates.

    Call ``set_pose_covariance(P)`` before the usual ``setup(...)`` to supply
    world-frame pose covariance. It persists until replaced and defaults to
    zero (deterministic OBCA). No covariance propagation is performed here.
    Only one body-axis-aligned rectangular footprint is supported; its origin
    need not be at its center. Obstacles must be known static convex polygons.
    """

    def __init__(self, variables: dict, costs: dict, dynamics_opt):
        super().__init__(variables, costs, dynamics_opt)
        self._pose_covariance = np.zeros((3, 3))
        self._previous_u = None
        self._risk_signature = None
        self._eta = None
        self._separation_covariances = []

    def set_pose_covariance(self, covariance: np.ndarray) -> None:
        """Set covariance of [x, y, heading] in world coordinates (m, rad).

        Accept (3, 3) for all stages or (horizon, 3, 3), whose entry i applies
        to x[:, i + 1]. Position x/y correlation is allowed; position/heading
        correlation violates the independence assumption and is rejected.
        Roundoff within 1e-12 is tolerated when checking symmetry and PSD.
        Changing these values does not rebuild the NLP graph.
        """
        covariance = np.asarray(covariance, dtype=float).copy()
        if covariance.shape != (3, 3) and not (
            covariance.ndim == 3
            and covariance.shape[0] > 0
            and covariance.shape[1:] == (3, 3)
        ):
            raise ValueError("Pose covariance must have shape (3, 3) or (horizon, 3, 3)")
        matrices = covariance.reshape(-1, 3, 3)
        if not np.all(np.isfinite(matrices)):
            raise ValueError("Pose covariance must be finite")
        if not np.allclose(matrices, matrices.transpose(0, 2, 1), atol=1e-12, rtol=0):
            raise ValueError("Pose covariance must be symmetric")
        if np.any(np.abs(matrices[:, :2, 2]) > 1e-12):
            raise ValueError("U-OBCA requires independent position and heading noise")
        for matrix in matrices:
            matrix[:] = 0.5 * (matrix + matrix.T)
            eigenvalues, eigenvectors = np.linalg.eigh(matrix[:2, :2])
            if eigenvalues[0] < -1e-12 or matrix[2, 2] < -1e-12:
                raise ValueError("Pose covariance must be positive semidefinite")
            # Remove only tolerated roundoff, keeping the assumed blocks PSD.
            matrix[:2, :2] = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
            matrix[2, 2] = max(matrix[2, 2], 0.0)
            matrix[:2, 2] = matrix[2, :2] = 0.0
        self._pose_covariance = covariance

    @classmethod
    def _canonical_rectangle(cls, robot_geo):
        """Normalize and order the actual halfspaces as [I; -I], never a box fit."""
        if not isinstance(robot_geo, ConvexRegion2D):
            raise ValueError("U-OBCA requires a body-axis-aligned rectangle")
        matrix, vector = cls._geometry_representation(robot_geo)
        if matrix.shape != (4, 2) or vector.shape != (4, 1):
            raise ValueError("U-OBCA requires a body-axis-aligned rectangle")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
            raise ValueError("Rectangle halfspaces must be finite")
        scales = np.linalg.norm(matrix, axis=1)
        if np.any(scales == 0.0):
            raise ValueError("Rectangle halfspace normals must be nonzero")
        normals = matrix / scales[:, None]
        canonical = np.vstack((np.eye(2), -np.eye(2)))
        indices = np.argmax(normals @ canonical.T, axis=1)
        if len(set(indices)) != 4 or not np.allclose(
            normals, canonical[indices], atol=1e-10, rtol=0
        ):
            raise ValueError("U-OBCA requires a body-axis-aligned rectangle")
        bounds = np.empty((4, 1))
        bounds[indices] = vector / scales[:, None]
        if np.any(bounds[:2] + bounds[2:] <= 0.0):
            raise ValueError("Rectangle must have positive width and length")
        return canonical, bounds

    def _collect_constraint_pairs(self, param, system, obstacles_geo):
        components = system._geometry.equiv_rep()
        if len(components) != 1:
            raise ValueError("U-OBCA supports one rectangular robot footprint")
        self._canonical_rectangle(components[0])
        return super()._collect_constraint_pairs(param, system, obstacles_geo)

    def _make_constraint_pair(self, robot_geo, obs_geo):
        mat_A, vec_b = self._geometry_representation(obs_geo)
        robot_G, robot_g = self._canonical_rectangle(robot_geo)
        world_G = robot_G @ self.state.rotation().T
        world_g = world_G @ self.state.translation() + robot_g
        distance, lamb_curr, mu_curr = self._query_region_distance(
            mat_A, vec_b, world_G, world_g
        )
        return {
            "mat_A": mat_A, "vec_b": vec_b,
            "robot_G": robot_G, "robot_g": robot_g,
            "distance": distance, "lamb_curr": lamb_curr, "mu_curr": mu_curr,
        }

    def initialize_variables(self, param):
        super().initialize_variables(param)
        alpha, radius = float(param.alpha), float(param.wasserstein_radius)
        if not math.isfinite(alpha) or not 0.0 < alpha <= 0.5:
            raise ValueError("alpha must be in (0, 0.5]")
        if self._risk_signature != (alpha, radius):
            self._eta = [
                wasserstein_eta(fraction * alpha, radius)
                for fraction in (0.2, 0.2, 0.6)
            ]
            self._risk_signature = (alpha, radius)
        self._parameters["heading_mean_scale"] = self.opti.parameter(1, param.horizon)
        self._parameters["rotation_covariance_x"] = self.opti.parameter(2, 2 * param.horizon)
        self._parameters["rotation_covariance_y"] = self.opti.parameter(2, 2 * param.horizon)
        self._separation_covariances = []

    def _add_chance_constraint(self, slack, normal, covariance, eta):
        variance = ca.mtimes([normal.T, covariance, normal])
        # Keep residuals in the original units for IPOPT's feasibility tolerance.
        # The conservative 1e-10 std floor gives finite derivatives at n = 0.
        # Bypass it for zero covariance to recover deterministic OBCA exactly.
        # fmax removes possible negative roundoff in a PSD quadratic form.
        sigma = ca.if_else(
            ca.trace(covariance) > 0,
            ca.sqrt(ca.fmax(variance, 0.0) + _VARIANCE_EPSILON),
            0.0,
        )
        self.opti.subject_to(slack >= eta * sigma)

    def _add_convex_pair_constraint(self, param, pair):
        mat_A, vec_b = pair["mat_A"], pair["vec_b"]
        g_plus, g_minus = pair["robot_g"][:2], pair["robot_g"][2:]
        lamb = self.opti.variable(mat_A.shape[0], param.horizon)
        xi_dual = self.opti.variable(2, param.horizon)
        covariance_3 = self.opti.parameter(2, 2 * param.horizon)
        self._separation_covariances.append((covariance_3, g_plus.reshape(2)))
        for i in range(param.horizon):
            heading = self.variables["x"][2, i + 1]
            cosine, sine = ca.cos(heading), ca.sin(heading)
            rotation = ca.vertcat(ca.horzcat(cosine, -sine), ca.horzcat(sine, cosine))
            mean_rotation = self._parameters["heading_mean_scale"][i] * rotation
            normal = mat_A.T @ lamb[:, i]
            self.opti.subject_to(lamb[:, i] >= 0)
            self.opti.subject_to(xi_dual[:, i] >= 0)
            self.opti.subject_to(ca.dot(normal, normal) <= 1)
            for axis, key in enumerate(("rotation_covariance_x", "rotation_covariance_y")):
                slack = xi_dual[axis, i] - ca.dot(normal, mean_rotation[:, axis])
                self._add_chance_constraint(
                    slack, normal, self._parameters[key][:, 2 * i:2 * i + 2], self._eta[axis]
                )
            separation_slack = (
                ca.dot(normal, self.variables["x"][:2, i + 1] + mean_rotation @ g_plus)
                - ca.dot(vec_b, lamb[:, i])
                - ca.dot(g_plus + g_minus, xi_dual[:, i])
                - param.margin_dist
            )
            self._add_chance_constraint(
                separation_slack, normal, covariance_3[:, 2 * i:2 * i + 2], self._eta[2]
            )
        self._dual_variables.append((lamb, xi_dual))
        self._set_dual_initial(param, lamb, xi_dual, pair)

    def _set_dual_initial(self, param, lamb, xi_dual, pair):
        lamb_curr = np.maximum(np.asarray(pair["lamb_curr"]).reshape(-1, 1), 0.0)
        # Canonical mu = [xi - R.T @ n; xi]. This is only an initial guess,
        # not a chance-feasible certificate when uncertainty is nonzero.
        xi_curr = np.maximum(np.asarray(pair["mu_curr"]).reshape(4, 1)[2:], 0.0)
        self.opti.set_initial(lamb, np.repeat(lamb_curr, param.horizon, axis=1))
        self.opti.set_initial(xi_dual, np.repeat(xi_curr, param.horizon, axis=1))

    def setup(self, param, system, reference_trajectory, obstacles):
        if self._pose_covariance.ndim == 3 and self._pose_covariance.shape[0] != param.horizon:
            raise ValueError("Stage covariance count must equal horizon")
        super().setup(param, system, reference_trajectory, obstacles)

    def add_warm_start(self, param, system):
        if self._previous_u is not None and self._previous_u.shape == (2, param.horizon):
            controls = np.column_stack((self._previous_u[:, 1:], self._previous_u[:, -1]))
        else:
            controls = np.repeat(
                np.asarray(self.state._u).reshape(2, 1), param.horizon, axis=1
            )
        controls = np.clip(
            controls, [[param.vmin], [param.omegamin]], [[param.vmax], [param.omegamax]]
        )
        poses = np.empty((3, param.horizon + 1))
        poses[:, 0] = np.asarray(self.state._x).reshape(3)
        for i in range(param.horizon):
            poses[:, i + 1] = np.asarray(
                self.dynamics_opt(poses[:, i], controls[:, i])
            ).reshape(3)
        self.opti.set_initial(self.variables["x"], poses)
        self.opti.set_initial(self.variables["u"], controls)

        covariances = np.broadcast_to(self._pose_covariance, (param.horizon, 3, 3))
        self.opti.set_value(
            self._parameters["heading_mean_scale"],
            np.exp(-0.5 * covariances[:, 2, 2]).reshape(1, -1),
        )
        for axis, key in enumerate(("rotation_covariance_x", "rotation_covariance_y")):
            blocks = [
                rotation_moments(poses[2, i + 1], covariances[i, 2, 2], np.eye(2)[axis])[1]
                for i in range(param.horizon)
            ]
            self.opti.set_value(self._parameters[key], np.hstack(blocks))
        for parameter, g_plus in self._separation_covariances:
            blocks = [
                covariances[i, :2, :2]
                + rotation_moments(poses[2, i + 1], covariances[i, 2, 2], g_plus)[1]
                for i in range(param.horizon)
            ]
            self.opti.set_value(parameter, np.hstack(blocks))

    def solve_nlp(self):
        try:
            solution = super().solve_nlp()
        except Exception:
            self._previous_u = None
            raise
        self._previous_u = np.asarray(
            solution.value(self.variables["u"]), dtype=float
        ).reshape(2, -1)
        return solution
