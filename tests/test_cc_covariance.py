"""Regression checks for the shared CC covariance representation and prediction."""

import unittest
from types import SimpleNamespace

import casadi as ca
import numpy as np

from control.cc_covariance import (
    covariance_from_log_cholesky,
    initial_local_covariance,
    log_cholesky_from_covariance,
    propagate_log_cholesky,
    resolve_robust_horizon,
    symbolic_covariance_from_log_cholesky,
    symbolic_log_cholesky_transition,
)
from models.dd_dynamics import symbolic_dd_step


class CCCovarianceTest(unittest.TestCase):
    def setUp(self):
        self.param = SimpleNamespace(
            horizon=20,
            robust_horizon=None,
            sigma_f0=0.02,
            sigma_l0=0.03,
            sigma_psi0=0.04,
            initial_pose_covariance=None,
            q_f0=0.001,
            q_l0=0.002,
            q_psi0=0.003,
            alpha_f=0.01,
            alpha_v=0.02,
            alpha_kappa=0.03,
            beta_v=0.04,
            beta_kappa=0.05,
            beta_omega=0.06,
        )

    def test_signed_cross_factor_roundtrip_single_and_batch(self):
        states = np.array([
            [np.log(0.02), np.log(0.03), np.log(0.04), -0.06],
            [np.log(0.04), np.log(0.02), np.log(0.03), 0.05],
        ])
        for xi in (states[0], states, states.reshape(1, 2, 4)):
            with self.subTest(shape=xi.shape):
                covariance = covariance_from_log_cholesky(xi)
                np.testing.assert_allclose(
                    log_cholesky_from_covariance(covariance), xi, atol=1e-14
                )
                self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0.0))
                np.testing.assert_array_equal(covariance[..., 0, 1:], 0.0)
                np.testing.assert_allclose(
                    covariance[..., 1, 2], xi[..., 3] * np.exp(xi[..., 1])
                )

    def test_estimator_covariance_rotates_then_retains_body_blocks(self):
        factor = np.array([[0.2, 0.0, 0.0], [0.03, 0.3, 0.0], [0.02, -0.04, 0.1]])
        world_covariance = factor @ factor.T
        self.param.initial_pose_covariance = world_covariance.copy()
        theta = 0.7
        state = SimpleNamespace(_x=np.array([1.0, -2.0, theta]))
        cosine, sine = np.cos(theta), np.sin(theta)
        transform = np.array([
            [cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]
        ])
        expected = transform.T @ world_covariance @ transform
        expected[0, 1:] = 0.0
        expected[1:, 0] = 0.0
        np.testing.assert_allclose(initial_local_covariance(self.param, state), expected)
        np.testing.assert_array_equal(self.param.initial_pose_covariance, world_covariance)
        with self.assertRaises(ValueError):
            initial_local_covariance(self.param)

    def test_numpy_and_symbolic_prediction_match_rk4_pose_jacobian(self):
        xi_symbol = ca.MX.sym("xi", 4)
        v_symbol = ca.MX.sym("v")
        omega_symbol = ca.MX.sym("omega")
        curvature_symbol = ca.MX.sym("curvature")
        dt = 0.15
        transition = ca.Function(
            "covariance_transition",
            [xi_symbol, v_symbol, omega_symbol, curvature_symbol],
            [symbolic_log_cholesky_transition(
                xi_symbol, v_symbol, omega_symbol, curvature_symbol, dt, self.param
            )],
        )
        reconstruction = ca.Function(
            "covariance_reconstruction", [xi_symbol],
            [symbolic_covariance_from_log_cholesky(xi_symbol)],
        )
        pose_symbol = ca.MX.sym("pose", 3)
        next_pose = symbolic_dd_step(pose_symbol, v_symbol, omega_symbol, dt)
        pose_step_and_jacobian = ca.Function(
            "rk4_pose_step_and_jacobian", [pose_symbol, v_symbol, omega_symbol],
            [next_pose, ca.jacobian(next_pose, pose_symbol)],
        )
        cases = (
            (0.0, 0.4, -0.7, -0.3),
            (0.7, -0.5, 0.6, 0.2),
            (-2.3, 0.4, 1.1, 0.0),
            (2.9, 0.4, 0.0, 8.0),
            (-0.5, 0.0, -0.7, 3.0),
            (0.6, 0.4, 1e-9, -1.0),
            (1.2, 0.4, np.pi / (2.0 * dt), -2.0),
        )
        for cross_factor in (-0.06, 0.05):
            for theta, v, omega, curvature in cases:
                with self.subTest(cross=cross_factor, theta=theta, v=v, omega=omega):
                    xi = np.array([
                        np.log(0.02), np.log(0.03), np.log(0.04), cross_factor
                    ])
                    covariance = covariance_from_log_cholesky(xi)
                    pose_next, jacobian = pose_step_and_jacobian(
                        [1.0, -2.0, theta], v, omega,
                    )
                    transforms = []
                    for heading in (theta, float(pose_next[2])):
                        cosine, sine = np.cos(heading), np.sin(heading)
                        transforms.append(np.array([
                            [cosine, -sine, 0.0],
                            [sine, cosine, 0.0],
                            [0.0, 0.0, 1.0],
                        ]))
                    matrix = transforms[1].T @ np.asarray(jacobian) @ transforms[0]
                    param = self.param
                    q_f = param.q_f0 + param.alpha_f * v**2
                    q_l = (
                        param.q_l0 + param.alpha_v * v**2
                        + param.alpha_kappa * abs(v * omega)
                    )
                    q_heading = (
                        param.q_psi0 + param.beta_v * v**2
                        + param.beta_kappa * abs(v * omega)
                        + param.beta_omega * omega**2
                    )
                    expected_covariance = (
                        matrix @ covariance @ matrix.T
                        + dt * np.diag([q_f, q_l, q_heading])
                    )
                    expected_covariance[0, 1:] = 0.0
                    expected_covariance[1:, 0] = 0.0
                    numeric_xi = propagate_log_cholesky(xi, v, omega, curvature, dt, param)
                    symbolic_xi = np.asarray(transition(xi, v, omega, curvature)).reshape(4)
                    np.testing.assert_allclose(numeric_xi, symbolic_xi, rtol=1e-13, atol=1e-14)
                    np.testing.assert_allclose(
                        covariance_from_log_cholesky(numeric_xi), expected_covariance,
                        rtol=1e-13, atol=1e-15,
                    )
                    np.testing.assert_allclose(np.asarray(reconstruction(xi)), covariance)

    def test_noise_free_pure_rotation_preserves_isotropic_position_covariance(self):
        xi = np.r_[np.log([0.03, 0.03, 0.02]), 0.0]
        expected = covariance_from_log_cholesky(xi)
        for omega in (-1.2, 0.7):
            with self.subTest(omega=omega):
                state = xi.copy()
                for _ in range(20):
                    state = propagate_log_cholesky(
                        state, 0.0, omega, 8.0, 0.15, SimpleNamespace(),
                    )
                    np.testing.assert_allclose(
                        covariance_from_log_cholesky(state), expected,
                        rtol=1e-13, atol=1e-15,
                    )

    def test_curvature_noise_uses_absolute_physical_turning_and_ignores_reference(self):
        xi = np.r_[np.log([0.02, 0.03, 0.04]), -0.06]
        v = np.array([0.4, -0.4, 0.4, -0.4, 0.0, 0.4])
        omega = np.array([0.7, 0.7, -0.7, -0.7, 0.7, 0.0])
        dt = 0.15
        param = SimpleNamespace(alpha_kappa=0.03, beta_kappa=0.05)
        baseline = covariance_from_log_cholesky(propagate_log_cholesky(
            xi, v, omega, 0.0, dt, SimpleNamespace(),
        ))
        expected_noise = np.zeros((len(v), 3, 3))
        expected_noise[:, 1, 1] = dt * param.alpha_kappa * np.abs(v * omega)
        expected_noise[:, 2, 2] = dt * param.beta_kappa * np.abs(v * omega)
        for curvature in (0.0, -8.0, 8.0):
            with self.subTest(curvature=curvature):
                result = covariance_from_log_cholesky(propagate_log_cholesky(
                    xi, v, omega, curvature, dt, param,
                ))
                np.testing.assert_allclose(
                    result - baseline, expected_noise, rtol=1e-13, atol=1e-15,
                )
        # Preserve the old API's broadcasting even when only kappa is batched.
        curvature_batch = np.array([0.0, -8.0, 8.0])
        states = propagate_log_cholesky(xi, v[0], omega[0], curvature_batch, dt, param)
        expected_state = propagate_log_cholesky(xi, v[0], omega[0], 0.0, dt, param)
        np.testing.assert_allclose(states, np.tile(expected_state, (3, 1)))

    def test_zero_noise_cancellation_keeps_positive_conditional_variance(self):
        param = SimpleNamespace()
        xi = np.array([np.log(0.2), np.log(0.3), np.log(1e-12), 0.4])
        result = propagate_log_cholesky(xi, 0.7, 0.0, 0.0, 0.2, param)
        expected_lateral = (0.3 + 0.2 * 0.7 * 0.4)**2 + (0.2 * 0.7 * 1e-12)**2
        expected_conditional = 0.3**2 * 1e-24 / expected_lateral
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(np.exp(2.0 * result[2]), expected_conditional, rtol=1e-13)
        zero_motion = propagate_log_cholesky(xi, 0.0, 0.0, 0.0, 0.2, param)
        np.testing.assert_allclose(zero_motion, xi, rtol=1e-13, atol=1e-14)

    def test_invalid_covariance_and_numeric_range_are_rejected(self):
        for covariance in (
            np.diag([0.0, 1.0, 1.0]),
            np.diag([1.0, -1.0, 1.0]),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 2.0, 1.0]]),
            np.full((3, 3), np.nan),
        ):
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                log_cholesky_from_covariance(covariance)
        for xi in ([0.0, 0.0, np.nan, 0.0], [1000.0, 0.0, 0.0, 0.0], [-1000.0, 0.0, 0.0, 0.0]):
            with self.subTest(xi=xi), self.assertRaises(ValueError):
                covariance_from_log_cholesky(xi)
        self.param.sigma_f0 = 0.0
        with self.assertRaises(ValueError):
            initial_local_covariance(self.param)
        for coefficient in (-1.0, np.inf, np.nan):
            self.param.alpha_f = coefficient
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                propagate_log_cholesky(np.zeros(4), 0.4, 0.3, 0.2, 0.1, self.param)

    def test_robust_horizon_default_and_explicit_bounds(self):
        self.assertEqual(resolve_robust_horizon(self.param), 10)
        self.param.horizon = 3
        self.assertEqual(resolve_robust_horizon(self.param), 3)
        for horizon in (1, 3, np.int64(2)):
            self.param.robust_horizon = horizon
            self.assertEqual(resolve_robust_horizon(self.param), horizon)
        for horizon in (0, 4, 1.5, True):
            self.param.robust_horizon = horizon
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                resolve_robust_horizon(self.param)


if __name__ == "__main__":
    unittest.main()
