"""Check U-OBCA moments and certificates without running an NLP solver."""

import math
import unittest
from statistics import NormalDist
from types import SimpleNamespace
from unittest.mock import patch

import casadi as ca
import numpy as np

from control.obca_optimizer import OBCAOptimizer, OBCAOptimizerParam
from control.uobca_optimizer import (
    UOBCAOptimizer,
    UOBCAOptimizerParam,
    rotation_moments,
    wasserstein_eta,
)
from models.geometry_utils import RectangleRegion


class UncertaintyMomentsTest(unittest.TestCase):
    def test_rotation_moments_match_gaussian_quadrature(self):
        nodes, weights = np.polynomial.hermite.hermgauss(64)
        weights /= np.sqrt(np.pi)
        # The offset rectangle's g+ is not a unit vector or a half-size.
        for vector in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.11, 0.06])):
            for heading, variance in ((0.0, 0.0), (0.7, 0.03), (-1.1, 1.5)):
                with self.subTest(vector=vector, heading=heading, variance=variance):
                    angles = heading + np.sqrt(2.0 * variance) * nodes
                    rotated = np.column_stack((
                        np.cos(angles) * vector[0] - np.sin(angles) * vector[1],
                        np.sin(angles) * vector[0] + np.cos(angles) * vector[1],
                    ))
                    expected_mean = weights @ rotated
                    centered = rotated - expected_mean
                    expected_covariance = (centered.T * weights) @ centered
                    mean, covariance = rotation_moments(heading, variance, vector)
                    np.testing.assert_allclose(mean, expected_mean, atol=1e-13)
                    np.testing.assert_allclose(covariance, expected_covariance, atol=1e-13)

    def test_tiny_heading_variance_preserves_positive_covariance(self):
        variance = 1e-12
        _, covariance = rotation_moments(0.0, variance, np.array([1.0, 0.0]))
        self.assertGreater(covariance[0, 0], 0.0)
        np.testing.assert_allclose(
            np.diag(covariance),
            [0.5 * np.expm1(-variance) ** 2, -0.5 * np.expm1(-2.0 * variance)],
            rtol=1e-12,
            atol=0.0,
        )
        _, zero_covariance = rotation_moments(0.7, 0.0, np.array([0.11, 0.06]))
        np.testing.assert_array_equal(zero_covariance, np.zeros((2, 2)))

    def test_wasserstein_eta_satisfies_equation_and_monotonicity(self):
        normal = NormalDist()
        for alpha in (0.002, 0.05, 0.5):
            quantile = normal.inv_cdf(1.0 - alpha)
            self.assertAlmostEqual(wasserstein_eta(alpha, 0.0), quantile)
            previous_eta = quantile
            for radius in (0.001, 0.01, 0.1):
                eta = wasserstein_eta(alpha, radius)
                residual = (
                    eta * (normal.cdf(eta) - normal.cdf(quantile))
                    + normal.pdf(eta) - normal.pdf(quantile)
                )
                self.assertAlmostEqual(residual, radius, delta=1e-10)
                self.assertGreater(eta, previous_eta)
                previous_eta = eta
        self.assertGreater(wasserstein_eta(0.01, 0.01), wasserstein_eta(0.1, 0.01))

    def test_invalid_risk_parameters_are_rejected(self):
        for alpha, radius in (
            (0.0, 0.01), (-0.1, 0.01), (0.5001, 0.01),
            (np.nan, 0.01), (0.01, -0.1), (0.01, np.inf),
        ):
            with self.subTest(alpha=alpha, radius=radius):
                with self.assertRaises(ValueError):
                    wasserstein_eta(alpha, radius)


class UOBCAConstraintsTest(unittest.TestCase):
    @staticmethod
    def make_case(optimizer_type=UOBCAOptimizer, horizon=3, heading=0.0):
        param_type = (
            UOBCAOptimizerParam if optimizer_type is UOBCAOptimizer else OBCAOptimizerParam
        )
        param = param_type()
        param.horizon = horizon
        param.margin_dist = 0.02
        param.use_obstacle_cutoff = False
        state = SimpleNamespace(_x=np.array([-1.0, 0.0, heading]), _u=np.zeros(2))
        state.rotation = lambda: np.array([
            [np.cos(state._x[2]), -np.sin(state._x[2])],
            [np.sin(state._x[2]), np.cos(state._x[2])],
        ])
        state.translation = lambda: state._x[:2].reshape(2, 1)
        robot = RectangleRegion(-0.04, 0.11, -0.025, 0.06)
        obstacle = RectangleRegion(1.0, 2.0, -2.0, 2.0)
        system = SimpleNamespace(
            _state=state,
            _geometry=SimpleNamespace(equiv_rep=lambda: [robot]),
        )
        optimizer = optimizer_type({}, {}, dynamics_opt=lambda x, u: x)
        return optimizer, param, system, robot, obstacle

    @staticmethod
    def distance_result(mat_a, vec_b, world_g, world_b):
        # Exact horizontal duals, in whatever order the robot rows use.
        lamb = np.array([1.0, 0.0, 0.0, 0.0])
        mu = np.maximum(world_g[:, 0], 0.0)
        return 0.5, lamb, mu

    def build_certificate(self, optimizer, param, system, robot, obstacle):
        optimizer.opti = ca.Opti()
        optimizer.initialize_variables(param)
        optimizer.set_state(system._state)
        with patch.object(optimizer, "_query_region_distance", side_effect=self.distance_result):
            optimizer.add_convex_to_convex_constraint(param, robot, obstacle)
        if isinstance(optimizer, UOBCAOptimizer):
            optimizer.add_warm_start(param, system)

    @staticmethod
    def initial_value(optimizer, expression):
        return np.asarray(optimizer.opti.debug.value(expression, optimizer.opti.initial()))

    def certificate_is_feasible(self, optimizer, poses):
        optimizer.opti.set_initial(optimizer.variables["x"], poses)
        values = self.initial_value(optimizer, optimizer.opti.g).reshape(-1)
        lower = self.initial_value(optimizer, optimizer.opti.lbg).reshape(-1)
        upper = self.initial_value(optimizer, optimizer.opti.ubg).reshape(-1)
        self.assertGreater(values.size, 0)
        return bool(np.all(values >= lower - 1e-9) and np.all(values <= upper + 1e-9))

    def test_zero_noise_matches_obca_at_every_predicted_state(self):
        for horizon in (1, 3):
            for heading in (0.0, np.pi / 4.0, np.pi / 2.0):
                with self.subTest(horizon=horizon, heading=heading):
                    optimizers = []
                    for optimizer_type in (OBCAOptimizer, UOBCAOptimizer):
                        case = self.make_case(optimizer_type, horizon, heading)
                        self.build_certificate(*case)
                        optimizers.append(case[0])
                    param, system, robot = case[1:4]
                    support_x = robot.right * math.cos(heading) - robot.down * math.sin(heading)
                    poses = np.repeat(system._state._x[:, None], horizon + 1, axis=1)
                    poses[0, 1:] = 1.0 - support_x - param.margin_dist - 0.001
                    for optimizer in optimizers:
                        self.assertTrue(self.certificate_is_feasible(optimizer, poses))
                        self.assertEqual(optimizer.costs, {})
                    for stage in range(1, horizon + 1):
                        for clearance in (param.margin_dist - 0.001, -0.01):
                            unsafe = poses.copy()
                            unsafe[0, stage] = 1.0 - support_x - clearance
                            for optimizer in optimizers:
                                self.assertFalse(self.certificate_is_feasible(optimizer, unsafe))

    def test_terminal_position_uncertainty_tightens_clearance(self):
        case = self.make_case()
        optimizer, param, system, robot, _ = case
        sigma = 0.01
        covariance = np.zeros((param.horizon, 3, 3))
        covariance[-1, 0, 0] = sigma ** 2
        optimizer.set_pose_covariance(covariance)
        self.build_certificate(*case)
        tightening = wasserstein_eta(0.6 * param.alpha, param.wasserstein_radius) * sigma
        poses = np.repeat(system._state._x[:, None], param.horizon + 1, axis=1)
        poses[0, 1:] = 1.0 - robot.right - param.margin_dist - 0.001
        poses[0, -1] -= tightening
        self.assertTrue(self.certificate_is_feasible(optimizer, poses))
        poses[0, -1] += 0.002
        self.assertFalse(self.certificate_is_feasible(optimizer, poses))
        # A larger violation of the nominal clearance must also be rejected.
        poses[0, -1] = 1.0 - robot.right - param.margin_dist + tightening + 0.001
        self.assertFalse(self.certificate_is_feasible(optimizer, poses))

    def test_small_position_uncertainty_still_tightens_clearance(self):
        case = self.make_case()
        optimizer, param, system, robot, _ = case
        sigma = 1e-6
        optimizer.set_pose_covariance(np.diag([sigma ** 2, 0.0, 0.0]))
        self.build_certificate(*case)
        tightening = wasserstein_eta(0.6 * param.alpha, param.wasserstein_radius) * sigma
        poses = np.repeat(system._state._x[:, None], param.horizon + 1, axis=1)
        poses[0, -1] = 1.0 - robot.right - param.margin_dist - 0.5 * tightening
        # Squared residuals would be below the 1e-9 absolute tolerance here.
        self.assertFalse(self.certificate_is_feasible(optimizer, poses))
        poses[0, -1] -= tightening
        self.assertTrue(self.certificate_is_feasible(optimizer, poses))

    def test_heading_uncertainty_requires_tightened_dual_positivity(self):
        case = self.make_case()
        optimizer, param, system, _, _ = case
        optimizer.set_pose_covariance(np.diag([0.0, 0.0, 0.01]))
        self.build_certificate(*case)
        poses = np.repeat(system._state._x[:, None], param.horizon + 1, axis=1)
        self.assertFalse(self.certificate_is_feasible(optimizer, poses))
        _, xi = optimizer._dual_variables[0]
        lateral_sigma = np.sqrt(-0.5 * np.expm1(-0.02))
        xi_values = np.zeros((2, param.horizon))
        eta = wasserstein_eta(0.2 * param.alpha, param.wasserstein_radius)
        xi_values[1, :] = eta * lateral_sigma + 0.001
        optimizer.opti.set_initial(xi, xi_values)
        self.assertTrue(self.certificate_is_feasible(optimizer, poses))

    def test_zero_noise_constraint_jacobian_is_finite(self):
        case = self.make_case()
        self.build_certificate(*case)
        optimizer = case[0]
        jacobian = ca.jacobian(optimizer.opti.g, optimizer.opti.x)
        self.assertTrue(np.isfinite(self.initial_value(optimizer, jacobian)).all())

    def test_covariance_refresh_reuses_problem_graph(self):
        optimizer, param, system, _, obstacle = self.make_case()
        reference = np.repeat(system._state._x[None, :], param.horizon + 1, axis=0)
        with patch.object(optimizer, "_query_region_distance", side_effect=self.distance_result):
            optimizer.setup(param, system, reference, [obstacle])
            original_opti = optimizer.opti
            before = self.initial_value(optimizer, optimizer.opti.g)
            optimizer.set_pose_covariance(np.diag([0.0001, 0.0004, 0.0]))
            optimizer.setup(param, system, reference, [obstacle])
            after = self.initial_value(optimizer, optimizer.opti.g)
        self.assertIs(optimizer.opti, original_opti)
        self.assertEqual(optimizer.problem_build_count, 1)
        self.assertGreater(np.max(np.abs(after - before)), 1e-8)

    def test_shifted_controls_roll_out_stage_heading_moments(self):
        case = self.make_case()
        optimizer, param, system, _, _ = case
        dt, heading_variance = 0.2, 0.01

        def dynamics(x, u):
            return ca.vertcat(
                x[0] + dt * u[0] * ca.cos(x[2]),
                x[1] + dt * u[0] * ca.sin(x[2]),
                x[2] + dt * u[1],
            )

        optimizer.dynamics_opt = dynamics
        optimizer.set_pose_covariance(np.diag([0.0, 0.0, heading_variance]))
        self.build_certificate(*case)
        solved_controls = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        solution = SimpleNamespace(value=lambda expression: solved_controls)
        with patch.object(OBCAOptimizer, "solve_nlp", return_value=solution):
            optimizer.solve_nlp()

        system._state._x = np.array([-0.9, 0.02, 0.4])
        optimizer.add_warm_start(param, system)
        expected_controls = np.column_stack((solved_controls[:, 1:], solved_controls[:, -1]))
        expected_poses = np.empty((3, param.horizon + 1))
        expected_poses[:, 0] = system._state._x
        for i in range(param.horizon):
            x, y, heading = expected_poses[:, i]
            speed, turn_rate = expected_controls[:, i]
            expected_poses[:, i + 1] = [
                x + dt * speed * np.cos(heading),
                y + dt * speed * np.sin(heading),
                heading + dt * turn_rate,
            ]
        np.testing.assert_allclose(
            self.initial_value(optimizer, optimizer.variables["u"]), expected_controls
        )
        np.testing.assert_allclose(
            self.initial_value(optimizer, optimizer.variables["x"]), expected_poses
        )

        # Test stage moments through their constraints. Covariance stays at the
        # rollout heading while the mean follows the current decision variable.
        _, xi = optimizer._dual_variables[0]
        eta = wasserstein_eta(0.2 * param.alpha, param.wasserstein_radius)
        for decision_headings in (expected_poses[2, 1:], np.zeros(param.horizon)):
            poses = expected_poses.copy()
            poses[2, 1:] = decision_headings
            xi_values = np.zeros((2, param.horizon))
            for i in range(param.horizon):
                for axis, vector in enumerate(np.eye(2)):
                    mean, _ = rotation_moments(decision_headings[i], heading_variance, vector)
                    _, covariance = rotation_moments(
                        expected_poses[2, i + 1], heading_variance, vector
                    )
                    threshold = -mean[0] + eta * np.sqrt(covariance[0, 0])
                    xi_values[axis, i] = max(0.0, threshold) + 0.001
            optimizer.opti.set_initial(xi, xi_values)
            self.assertTrue(self.certificate_is_feasible(optimizer, poses))
            for stage in range(param.horizon):
                unsafe_xi = xi_values.copy()
                unsafe_xi[1, stage] -= 0.002
                optimizer.opti.set_initial(xi, unsafe_xi)
                self.assertFalse(self.certificate_is_feasible(optimizer, poses))

    def test_covariance_validation_rejects_unsupported_cross_terms(self):
        optimizer = UOBCAOptimizer({}, {}, dynamics_opt=None)
        optimizer.set_pose_covariance(np.zeros((3, 3)))
        optimizer.set_pose_covariance(np.diag([0.01, 0.02, 0.03]))
        cross_covariance = np.eye(3)
        cross_covariance[0, 2] = cross_covariance[2, 0] = 0.01
        asymmetric = np.eye(3)
        asymmetric[0, 1] = 0.1
        for covariance in (
            cross_covariance, asymmetric, np.diag([-0.01, 0.0, 0.0]),
            np.full((3, 3), np.nan), np.eye(2),
        ):
            with self.subTest(covariance=covariance):
                with self.assertRaises(ValueError):
                    optimizer.set_pose_covariance(covariance)

    def test_canonical_rectangle_preserves_scaled_offset_geometry(self):
        robot = RectangleRegion(-0.04, 0.11, -0.025, 0.06)
        matrix, bounds = robot.get_convex_rep()
        order = [3, 1, 0, 2]
        scales = np.array([2.0, 0.5, 4.0, 3.0])[:, None]
        scaled_rectangle = matrix[order] * scales, bounds[order] * scales
        with patch.object(robot, "get_convex_rep", return_value=scaled_rectangle):
            canonical_matrix, canonical_bounds = UOBCAOptimizer._canonical_rectangle(robot)
        np.testing.assert_array_equal(canonical_matrix, np.vstack((np.eye(2), -np.eye(2))))
        np.testing.assert_allclose(
            np.asarray(canonical_bounds).reshape(-1), [0.11, 0.06, 0.04, 0.025]
        )
        rotation = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
        for invalid_matrix, invalid_bounds in (
            (matrix @ rotation.T, bounds),
            (matrix[:3], bounds[:3]),
        ):
            invalid_rectangle = invalid_matrix, invalid_bounds
            with patch.object(robot, "get_convex_rep", return_value=invalid_rectangle):
                with self.assertRaises((ValueError, NotImplementedError)):
                    UOBCAOptimizer._canonical_rectangle(robot)


if __name__ == "__main__":
    unittest.main()
