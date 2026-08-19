import unittest

import casadi as ca
import numpy as np
import torch

from control.analytic_psdf_casadi import AnalyticPSDFCasADi
from models.augmented_psdf import PSDF
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper


class AugmentedPSDFTest(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.verts = torch.tensor(
            [[-0.075, -0.045], [0.075, -0.045], [0.075, 0.045], [-0.075, 0.045]],
            dtype=self.dtype,
        )

    def rectangle_edges(self, left, right, bottom, top):
        points = torch.tensor(
            [[left, bottom], [right, bottom], [right, top], [left, top]],
            dtype=self.dtype,
        )
        return points, torch.roll(points, -1, 0)

    def evaluate_box(self, bounds):
        edges_A, edges_B = self.rectangle_edges(*bounds)
        mask = torch.ones((1, 4), dtype=torch.bool)
        model = PSDF(self.verts)
        pose = torch.zeros((1, 3), dtype=self.dtype)
        return model(edges_A.unsqueeze(0), edges_B.unsqueeze(0), mask, pose)

    def test_forward_hard_sat_regression_values(self):
        cases = {
            "separated": ((0.2, 0.3, -0.05, 0.05), 0.125),
            "touching": ((0.075, 0.175, -0.05, 0.05), 0.0),
            "partial_overlap": ((0.05, 0.15, -0.05, 0.05), -0.025),
            "containment": ((-0.2, 0.2, -0.2, 0.2), -0.245),
        }
        for name, (bounds, expected_phi) in cases.items():
            with self.subTest(name=name):
                phi, gradient = self.evaluate_box(bounds)
                self.assertEqual(phi.shape, (1,))
                self.assertEqual(gradient.shape, (1, 3))
                self.assertAlmostEqual(phi.item(), expected_phi, places=8)
                self.assertTrue(torch.isfinite(gradient).all())

    def test_analytic_gradient_matches_central_difference(self):
        points = torch.tensor(
            [[0.2, -0.08], [0.34, -0.02], [0.30, 0.12], [0.17, 0.09]],
            dtype=self.dtype,
        )
        A = points.unsqueeze(0)
        B = torch.roll(points, -1, 0).unsqueeze(0)
        mask = torch.ones((1, 4), dtype=torch.bool)
        pose = torch.tensor([[0.01, 0.02, 0.13]], dtype=self.dtype)
        model = PSDF(self.verts)

        _, gradient = model(A, B, mask, pose)
        finite_difference = torch.zeros(3, dtype=self.dtype)
        step = 1e-6
        for dimension in range(3):
            pose_plus = pose.clone()
            pose_minus = pose.clone()
            pose_plus[0, dimension] += step
            pose_minus[0, dimension] -= step
            phi_plus, _ = model(A, B, mask, pose_plus)
            phi_minus, _ = model(A, B, mask, pose_minus)
            finite_difference[dimension] = (phi_plus - phi_minus) / (2.0 * step)

        torch.testing.assert_close(gradient[0], finite_difference, atol=2e-6, rtol=2e-5)

    def test_wrapper_normalizes_pose_shapes(self):
        edges_A, edges_B = self.rectangle_edges(0.2, 0.3, -0.05, 0.05)
        wrapper = AugmentedPSDFWrapper(self.verts, K_max=3, E_max=6)
        wrapper.update_edge_clusters([edges_A], [edges_B])

        for pose in (
            torch.zeros(3, dtype=self.dtype),
            torch.zeros((3, 1), dtype=self.dtype),
            torch.zeros((2, 3), dtype=self.dtype),
        ):
            with self.subTest(shape=tuple(pose.shape)):
                phi, gradient = wrapper(pose)
                expected_batch = 2 if pose.shape == (2, 3) else 1
                self.assertEqual(phi.shape, (expected_batch,))
                self.assertEqual(gradient.shape, (expected_batch, 3))

        self.assertFalse(hasattr(wrapper, "forward_with_gradient"))
        self.assertFalse(hasattr(wrapper, "forward_pose"))

    def test_wrapper_empty_and_stale_cluster_behavior(self):
        edges_A, edges_B = self.rectangle_edges(0.2, 0.3, -0.05, 0.05)
        other_A, other_B = self.rectangle_edges(-0.3, -0.2, -0.05, 0.05)
        wrapper = AugmentedPSDFWrapper(self.verts, K_max=3, E_max=6)

        empty_phi, empty_gradient = wrapper(torch.zeros((2, 3), dtype=self.dtype))
        torch.testing.assert_close(empty_phi, torch.full((2,), 1000.0, dtype=self.dtype))
        torch.testing.assert_close(empty_gradient, torch.zeros((2, 3), dtype=self.dtype))

        wrapper.update_edge_clusters([edges_A, other_A], [edges_B, other_B])
        self.assertEqual(wrapper.active_clusters, 2)
        wrapper.update_edge_clusters([edges_A], [edges_B])
        self.assertEqual(wrapper.active_clusters, 1)
        self.assertFalse(wrapper.mask[1:].any())
        self.assertEqual(wrapper.A[1:].count_nonzero().item(), 0)
        self.assertEqual(wrapper.B[1:].count_nonzero().item(), 0)

    def test_tensor_cluster_update_uses_explicit_mask(self):
        edges_A, edges_B = self.rectangle_edges(0.2, 0.3, -0.05, 0.05)
        padded_A = torch.zeros((1, 6, 2), dtype=self.dtype)
        padded_B = torch.zeros((1, 6, 2), dtype=self.dtype)
        padded_A[:, :4] = edges_A
        padded_B[:, :4] = edges_B
        mask = torch.zeros((1, 6), dtype=torch.bool)
        mask[:, :4] = True

        wrapper = AugmentedPSDFWrapper(self.verts, K_max=2, E_max=6)
        wrapper.update_edge_clusters(padded_A, padded_B, mask)
        phi, gradient = wrapper(torch.zeros((1, 3), dtype=self.dtype))
        self.assertAlmostEqual(phi.item(), 0.125, places=8)
        self.assertEqual(gradient.shape, (1, 3))

    def test_analytic_bridge_parameter_layout_and_affine_value(self):
        edges_A, edges_B = self.rectangle_edges(0.2, 0.3, -0.05, 0.05)
        wrapper = AugmentedPSDFWrapper(self.verts, K_max=2, E_max=6)
        wrapper.update_edge_clusters([edges_A], [edges_B])
        bridge = AnalyticPSDFCasADi(wrapper, name="test_analytic_psdf")

        pose = np.array([0.01, -0.02, 0.1], dtype=float)
        batch = np.stack((pose, pose + np.array([0.01, 0.0, 0.0])))
        single_params = bridge.get_params(pose)
        batch_params = bridge.get_params(batch)
        self.assertEqual(single_params.shape, (7,))
        self.assertEqual(batch_params.shape, (2, 7))
        np.testing.assert_allclose(single_params[:3], pose)

        pose_symbol = ca.MX.sym("pose", 3, 1)
        affine_function = ca.Function(
            "analytic_psdf_affine_test",
            [pose_symbol, bridge.get_sym_params()],
            [bridge(pose_symbol)],
        )
        nominal_value = float(affine_function(pose, single_params))
        self.assertAlmostEqual(nominal_value, single_params[3], places=10)

        delta = np.array([2e-3, -1e-3, 5e-4])
        perturbed_value = float(affine_function(pose + delta, single_params))
        expected_value = single_params[3] + single_params[4:7] @ delta
        self.assertAlmostEqual(perturbed_value, expected_value, places=10)

    def test_invalid_inputs_fail_before_solver_use(self):
        wrapper = AugmentedPSDFWrapper(self.verts, K_max=2, E_max=6)
        bridge = AnalyticPSDFCasADi(wrapper)
        with self.assertRaises(ValueError):
            wrapper(torch.zeros((1, 4), dtype=self.dtype))
        with self.assertRaises(ValueError):
            bridge.get_params(np.array([0.0, np.nan, 0.0]))
        with self.assertRaises(ValueError):
            bridge.get_params(np.zeros((2, 4)))

    def test_mf_affine_identity_and_stage_epsilon(self):
        edges_A, edges_B = self.rectangle_edges(0.09, 0.19, -0.06, 0.06)
        wrapper = AugmentedPSDFWrapper(self.verts, K_max=2, E_max=6)
        wrapper.update_edge_clusters([edges_A], [edges_B])
        z_bar = torch.zeros((3, 8), dtype=self.dtype)
        z_bar[:, 2] = torch.tensor([0.0, 0.1, -0.2], dtype=self.dtype)
        z_bar[:, 4:7] = 4e-4
        epsilon = torch.tensor([0.1, 0.2, 0.3], dtype=self.dtype)

        phi, gradient, A_mf, c_mf = wrapper.forward_mf(
            z_bar,
            epsilon,
            d_min=0.001,
        )
        self.assertEqual(phi.shape, (3,))
        self.assertEqual(gradient.shape, (3, 3))
        self.assertEqual(A_mf.shape, (3, 8))
        self.assertEqual(c_mf.shape, (3,))
        self.assertTrue(torch.isfinite(A_mf).all())
        residual = (A_mf * z_bar).sum(-1) + c_mf

        _, _, d, q, feature_mask = wrapper.psdf._forward_impl(
            wrapper.A[: wrapper.active_clusters],
            wrapper.B[: wrapper.active_clusters],
            wrapper.mask[: wrapper.active_clusters],
            z_bar[:, :3],
            return_features=True,
        )
        q_f, q_l, q_theta = q.unbind(-1)
        sigma_sq = (
            q_f.square() * z_bar[:, 4, None]
            + q_l.square() * z_bar[:, 5, None]
            + q_theta.square() * z_bar[:, 6, None]
            + 2.0 * q_l * q_theta * z_bar[:, 7, None]
        )
        sigma = torch.where(feature_mask, sigma_sq, torch.ones_like(sigma_sq)).sqrt()
        zeta = (0.001 - d) / sigma
        cdf = 0.5 * (1.0 + torch.erf(zeta / np.sqrt(2.0)))
        expected_h = epsilon - torch.where(
            feature_mask,
            cdf,
            torch.zeros_like(cdf),
        ).sum(-1)
        torch.testing.assert_close(residual, expected_h, atol=1e-12, rtol=1e-12)

    def test_mf_analytic_row_matches_fixed_feature_central_difference(self):
        edges_A, edges_B = self.rectangle_edges(0.09, 0.19, -0.06, 0.06)
        wrapper = AugmentedPSDFWrapper(self.verts, K_max=2, E_max=6)
        wrapper.update_edge_clusters([edges_A], [edges_B])
        z_bar = torch.tensor(
            [[0.0, 0.0, 0.3, 0.0, 4e-4, 5e-4, 6e-4, 2e-5]],
            dtype=self.dtype,
        )
        epsilon = 0.2
        d_min = 0.001
        _, _, A_mf, _ = wrapper.forward_mf(z_bar, epsilon, d_min)
        _, _, d, q, feature_mask = wrapper.psdf._forward_impl(
            wrapper.A[: wrapper.active_clusters],
            wrapper.B[: wrapper.active_clusters],
            wrapper.mask[: wrapper.active_clusters],
            z_bar[:, :3],
            return_features=True,
        )

        theta_bar = z_bar[0, 2]
        cos_theta = theta_bar.cos()
        sin_theta = theta_bar.sin()

        def fixed_feature_residual(z):
            delta = z[:3] - z_bar[0, :3]
            delta_body = torch.stack(
                [
                    cos_theta * delta[0] + sin_theta * delta[1],
                    -sin_theta * delta[0] + cos_theta * delta[1],
                    delta[2],
                ]
            )
            mean = d[0] + (q[0] * delta_body).sum(-1)
            q_f, q_l, q_theta = q[0].unbind(-1)
            sigma_sq = (
                q_f.square() * z[4]
                + q_l.square() * z[5]
                + q_theta.square() * z[6]
                + 2.0 * q_l * q_theta * z[7]
            )
            sigma = torch.where(
                feature_mask[0],
                sigma_sq,
                torch.ones_like(sigma_sq),
            ).sqrt()
            zeta = (d_min - mean) / sigma
            cdf = 0.5 * (1.0 + torch.erf(zeta / np.sqrt(2.0)))
            return epsilon - torch.where(
                feature_mask[0],
                cdf,
                torch.zeros_like(cdf),
            ).sum()

        finite_difference = torch.zeros(8, dtype=self.dtype)
        for dimension in (0, 1, 2, 4, 5, 6, 7):
            step = 1e-6 if dimension < 3 else 1e-7
            z_plus = z_bar[0].clone()
            z_minus = z_bar[0].clone()
            z_plus[dimension] += step
            z_minus[dimension] -= step
            finite_difference[dimension] = (
                fixed_feature_residual(z_plus) - fixed_feature_residual(z_minus)
            ) / (2.0 * step)

        torch.testing.assert_close(A_mf[0], finite_difference, atol=3e-6, rtol=2e-5)

    def test_mf_empty_obstacle_is_feasible(self):
        wrapper = AugmentedPSDFWrapper(self.verts, K_max=2, E_max=6)
        z_bar = torch.zeros((2, 8), dtype=self.dtype)
        z_bar[:, 4:7] = 1e-4
        phi, gradient, A_mf, c_mf = wrapper.forward_mf(z_bar, 0.2, 0.001)
        torch.testing.assert_close(phi, torch.full((2,), 1000.0, dtype=self.dtype))
        torch.testing.assert_close(gradient, torch.zeros((2, 3), dtype=self.dtype))
        torch.testing.assert_close(A_mf, torch.zeros((2, 8), dtype=self.dtype))
        torch.testing.assert_close(c_mf, torch.full((2,), 0.2, dtype=self.dtype))


if __name__ == "__main__":
    unittest.main()
