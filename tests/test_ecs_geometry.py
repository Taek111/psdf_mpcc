"""Geometry contracts for the per-obstacle ECS linearization interface."""

import unittest

import torch

from models.augmented_psdf import PSDF
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper


class ECSGeometryTest(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.verts = torch.tensor(
            [[-0.075, -0.045], [0.075, -0.045], [0.075, 0.045], [-0.075, 0.045]],
            dtype=self.dtype,
        )
        self.A = torch.tensor(
            [
                [[0.2, -0.08], [0.34, -0.02], [0.30, 0.12], [0.17, 0.09]],
                [[-0.43, 0.13], [-0.29, 0.17], [-0.32, 0.31], [-0.46, 0.26]],
            ],
            dtype=self.dtype,
        )
        self.B = torch.roll(self.A, -1, dims=1)
        self.mask = torch.ones((2, 4), dtype=torch.bool)
        self.poses = torch.tensor(
            [[0.01, 0.02, 0.13], [-0.02, -0.01, -0.21]], dtype=self.dtype
        )
        self.model = PSDF(self.verts)

    def wrapper(self):
        wrapper = AugmentedPSDFWrapper(self.verts, K_max=4, E_max=7)
        wrapper.update_edge_clusters(self.A, self.B, self.mask)
        return wrapper

    def test_cluster_min_and_active_gradient_match_scalar_forward(self):
        distances, gradients = self.model.forward_clusters(
            self.A, self.B, self.mask, self.poses
        )
        phi, gradient = self.model(self.A, self.B, self.mask, self.poses)

        self.assertEqual(distances.shape, (2, 2))
        self.assertEqual(gradients.shape, (2, 2, 3))
        active_cluster = distances.argmin(dim=1)
        stage = torch.arange(self.poses.shape[0])
        torch.testing.assert_close(distances[stage, active_cluster], phi)
        torch.testing.assert_close(gradients[stage, active_cluster], gradient)

        # Every returned row belongs to its own obstacle, including obstacles
        # that never attain the global minimum for this nominal trajectory.
        for cluster in range(self.A.shape[0]):
            single_phi, single_gradient = self.model(
                self.A[cluster : cluster + 1],
                self.B[cluster : cluster + 1],
                self.mask[cluster : cluster + 1],
                self.poses,
            )
            torch.testing.assert_close(distances[:, cluster], single_phi)
            torch.testing.assert_close(gradients[:, cluster], single_gradient)

    def test_each_world_gradient_matches_pose_central_difference(self):
        _, gradients = self.model.forward_clusters(
            self.A, self.B, self.mask, self.poses
        )
        step = 1e-6
        numerical = torch.zeros_like(gradients)
        for stage in range(self.poses.shape[0]):
            for dimension in range(3):
                plus = self.poses.clone()
                minus = self.poses.clone()
                plus[stage, dimension] += step
                minus[stage, dimension] -= step
                d_plus, _ = self.model.forward_clusters(
                    self.A, self.B, self.mask, plus
                )
                d_minus, _ = self.model.forward_clusters(
                    self.A, self.B, self.mask, minus
                )
                numerical[stage, :, dimension] = (
                    d_plus[stage] - d_minus[stage]
                ) / (2.0 * step)

        torch.testing.assert_close(gradients, numerical, atol=5e-6, rtol=5e-5)

    def test_cluster_order_and_duplicate_obstacles_are_preserved(self):
        distances, gradients = self.model.forward_clusters(
            self.A, self.B, self.mask, self.poses
        )
        order = torch.tensor([1, 0, 1])
        reordered_d, reordered_g = self.model.forward_clusters(
            self.A[order], self.B[order], self.mask[order], self.poses
        )
        torch.testing.assert_close(reordered_d, distances[:, order])
        torch.testing.assert_close(reordered_g, gradients[:, order])

    def test_wrapper_masks_edge_padding_and_removes_stale_cluster_rows(self):
        wrapper = self.wrapper()
        distances, gradients = wrapper.forward_clusters(self.poses)
        expected_d, expected_g = self.model.forward_clusters(
            self.A, self.B, self.mask, self.poses
        )
        torch.testing.assert_close(distances, expected_d)
        torch.testing.assert_close(gradients, expected_g)

        wrapper.update_edge_clusters([self.A[1]], [self.B[1]])
        single_d, single_g = wrapper.forward_clusters(self.poses)
        self.assertEqual(single_d.shape, (2, 1))
        self.assertEqual(single_g.shape, (2, 1, 3))
        torch.testing.assert_close(single_d[:, 0], expected_d[:, 1])
        torch.testing.assert_close(single_g[:, 0], expected_g[:, 1])

    def test_wrapper_normalizes_single_pose_inputs(self):
        wrapper = self.wrapper()
        expected_d, expected_g = wrapper.forward_clusters(self.poses[:1])
        for pose in (self.poses[0], self.poses[0].reshape(3, 1)):
            with self.subTest(shape=tuple(pose.shape)):
                distances, gradients = wrapper.forward_clusters(pose)
                torch.testing.assert_close(distances, expected_d)
                torch.testing.assert_close(gradients, expected_g)

    def test_empty_obstacles_have_no_rows_and_preserve_scalar_contract(self):
        wrapper = self.wrapper()
        wrapper.clear_clusters()
        distances, gradients = wrapper.forward_clusters(self.poses)
        self.assertEqual(distances.shape, (2, 0))
        self.assertEqual(gradients.shape, (2, 0, 3))
        self.assertEqual(distances.dtype, self.dtype)
        self.assertEqual(distances.device, self.poses.device)

        phi, gradient = wrapper(self.poses)
        torch.testing.assert_close(phi, torch.full((2,), 1000.0, dtype=self.dtype))
        torch.testing.assert_close(gradient, torch.zeros((2, 3), dtype=self.dtype))

        # Direct callers need not allocate even a padded edge slot when K=0.
        empty_edges = torch.empty((0, 0, 2), dtype=self.dtype)
        empty_mask = torch.empty((0, 0), dtype=torch.bool)
        direct_d, direct_g = self.model.forward_clusters(
            empty_edges, empty_edges, empty_mask, self.poses
        )
        self.assertEqual(direct_d.shape, (2, 0))
        self.assertEqual(direct_g.shape, (2, 0, 3))

    def test_cluster_reads_do_not_change_existing_mf_outputs(self):
        wrapper = self.wrapper()
        z_bar = torch.zeros((2, 8), dtype=self.dtype)
        z_bar[:, :3] = self.poses
        z_bar[:, 4:7] = torch.tensor([0.02, 0.025, 0.03], dtype=self.dtype).log()
        z_bar[:, 7] = -0.006
        before = wrapper.forward_mf(z_bar, epsilon=0.2, d_min=0.001)
        wrapper.forward_clusters(self.poses)
        after = wrapper.forward_mf(z_bar, epsilon=0.2, d_min=0.001)
        self.assertEqual(len(after), 4)
        for old_value, new_value in zip(before, after):
            torch.testing.assert_close(old_value, new_value, atol=0.0, rtol=0.0)
        phi, gradient = wrapper(self.poses)
        torch.testing.assert_close(after[0], phi)
        torch.testing.assert_close(after[1], gradient)

    def test_cluster_signed_distance_keeps_penetration_and_containment(self):
        bounds = [
            (0.2, 0.3, -0.05, 0.05),
            (0.05, 0.15, -0.05, 0.05),
            (-0.2, 0.2, -0.2, 0.2),
        ]
        A = torch.tensor(
            [
                [[left, bottom], [right, bottom], [right, top], [left, top]]
                for left, right, bottom, top in bounds
            ],
            dtype=self.dtype,
        )
        distances, gradients = self.model.forward_clusters(
            A,
            torch.roll(A, -1, dims=1),
            torch.ones((3, 4), dtype=torch.bool),
            torch.zeros((1, 3), dtype=self.dtype),
        )
        expected = torch.tensor([[0.125, -0.025, -0.245]], dtype=self.dtype)
        torch.testing.assert_close(distances, expected, atol=1e-8, rtol=1e-8)
        self.assertTrue(torch.isfinite(gradients).all())

    def test_invalid_inputs_are_rejected_including_empty_snapshots(self):
        wrapper = AugmentedPSDFWrapper(self.verts, K_max=2, E_max=4)
        with self.assertRaises(ValueError):
            wrapper.forward_clusters(torch.zeros((1, 4), dtype=self.dtype))
        with self.assertRaises(ValueError):
            wrapper.forward_clusters(torch.tensor([0.0, float("nan"), 0.0]))
        with self.assertRaises(ValueError):
            self.model.forward_clusters(
                self.A, self.B, torch.zeros_like(self.mask), self.poses
            )
        with self.assertRaises(TypeError):
            self.model.forward_clusters(
                self.A, self.B, self.mask, self.poses.to(torch.float32)
            )
        with self.assertRaises(ValueError):
            self.model.forward_clusters(
                self.A, self.B, self.mask[:, :3], self.poses
            )


if __name__ == "__main__":
    unittest.main()
