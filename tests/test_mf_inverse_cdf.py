"""Focused inverse-CDF MF math and wrapper-contract tests (no solver builds)."""

import math
import unittest

import torch

from models.augmented_psdf import PSDF
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.mf_inverse_cdf import (
    ICDF_PROBABILITY_FLOOR,
    MFInverseCDFBranch,
    MFInverseCDFStatus,
    compute_mf_icdf,
    evaluate_mf_icdf_frozen,
)


class MFInverseCDFTest(unittest.TestCase):
    dtype = torch.float64

    def state(self, H=1, device="cpu"):
        z = torch.zeros((H, 8), dtype=self.dtype, device=device)
        z[:, 2] = 0.37
        z[:, 4:7] = torch.tensor([0.025, 0.03, 0.035], device=device).to(self.dtype).log()
        z[:, 7] = -0.004
        return z

    def evaluate(self, d, q, z=None, mask=None, epsilon=0.2, **kwargs):
        d = torch.as_tensor(d, dtype=self.dtype)
        q = torch.as_tensor(q, dtype=self.dtype, device=d.device)
        if z is None:
            z = self.state(d.shape[0], d.device)
        if mask is None:
            mask = torch.ones_like(d, dtype=torch.bool)
        phi = kwargs.pop("phi", torch.ones(d.shape[0], dtype=self.dtype, device=d.device))
        gradient = kwargs.pop("gradient", torch.zeros((d.shape[0], 3), dtype=self.dtype,
                                                    device=d.device))
        d_min = kwargs.pop("d_min", 0.0)
        return compute_mf_icdf(phi, gradient, d, q, mask, z, epsilon, d_min, **kwargs)

    @staticmethod
    def reference_residual(z, nominal, d, q, epsilon):
        theta = nominal[2]
        world_q = torch.stack(
            (theta.cos() * q[:, 0] - theta.sin() * q[:, 1],
             theta.sin() * q[:, 0] + theta.cos() * q[:, 1], q[:, 2]), dim=-1,
        )
        distance = d + (world_q * (z[:3] - nominal[:3])).sum(-1)
        L = torch.diag(z[4:7].exp())
        L = L + torch.nn.functional.pad(z[7].reshape(1, 1), (1, 1, 2, 0))
        sigma = (q @ L).square().sum(-1).sqrt()
        R = (0.5 * torch.special.erfc(distance / (math.sqrt(2.0) * sigma))).sum()
        gamma = -torch.special.ndtri(torch.as_tensor(epsilon, dtype=z.dtype, device=z.device))
        R_safe = R.clamp(min=ICDF_PROBABILITY_FLOOR, max=0.5)
        return torch.where(
            R > 0.5, -math.sqrt(2.0 * math.pi) * (R - 0.5) - gamma,
            -torch.special.ndtri(R_safe) - gamma,
        )

    def test_nominal_identity_and_fixed_geometry_derivative(self):
        z = self.state()
        d = torch.tensor([[0.035, 0.05]], dtype=self.dtype)
        q = torch.tensor([[[0.8, 0.6, -0.2], [-0.6, 0.8, 0.4]]], dtype=self.dtype)
        result = self.evaluate(d, q, z, return_frozen_features=True)
        self.assertTrue(result.valid.all())
        torch.testing.assert_close((result.A * z).sum(-1) + result.c, result.residual)
        differentiable_z = z[0].clone().requires_grad_(True)
        expected_value = self.reference_residual(differentiable_z, z[0], d[0], q[0], 0.2)
        expected_A, = torch.autograd.grad(expected_value, differentiable_z)
        torch.testing.assert_close(result.residual[0], expected_value)
        torch.testing.assert_close(result.A[0], expected_A, atol=2e-12, rtol=2e-12)
        finite_difference = torch.zeros(8, dtype=self.dtype)
        for dimension in range(8):
            step = 1e-7
            plus, minus = z.clone(), z.clone()
            plus[0, dimension] += step
            minus[0, dimension] -= step
            h_plus = evaluate_mf_icdf_frozen(result.frozen_features, plus, 0.2, 0.0).residual
            h_minus = evaluate_mf_icdf_frozen(result.frozen_features, minus, 0.2, 0.0).residual
            finite_difference[dimension] = (h_plus[0] - h_minus[0]) / (2 * step)
        torch.testing.assert_close(result.A[0], finite_difference, atol=2e-8, rtol=2e-8)
        self.assertEqual(result.A[0, 3], 0.0)

    def test_single_feature_cancellation_in_tiny_probability_tail(self):
        z = torch.zeros((1, 8), dtype=self.dtype)
        result = self.evaluate([[30.0]], [[[1.0, 0.0, 0.0]]], z)
        gamma = -torch.special.ndtri(torch.tensor(0.2, dtype=self.dtype))
        self.assertTrue(result.valid.item())
        self.assertFalse(result.tail_clipped.item())
        self.assertGreater(result.probability_sum.item(), 0.0)
        self.assertLess(result.probability_sum.item(), 1e-190)
        torch.testing.assert_close(result.residual, (30.0 - gamma).reshape(1),
                                   atol=2e-12, rtol=2e-12)
        expected_A = torch.tensor([[1.0, 0.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0]],
                                  dtype=self.dtype)
        torch.testing.assert_close(result.A, expected_A, atol=1e-10, rtol=1e-10)

    def test_boundary_row_is_positive_multiple_of_probability_row(self):
        z = torch.zeros((1, 8), dtype=self.dtype)
        z[0, :4] = torch.tensor([0.3, -0.2, 0.37, 0.5], dtype=self.dtype)
        probabilities = torch.tensor([[0.08, 0.12]], dtype=self.dtype)
        d = -torch.special.ndtri(probabilities)
        q = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=self.dtype)
        mask = torch.ones_like(d, dtype=torch.bool)
        epsilon = torch.tensor([0.2], dtype=self.dtype)
        result = self.evaluate(d, q, z, mask=mask, epsilon=epsilon)
        A_probability, h_probability = PSDF._compute_A(d, q, mask, z, epsilon, 0.0)
        c_probability = PSDF._compute_c(A_probability, h_probability, z)
        threshold_quantile = torch.special.ndtri(epsilon)
        density = torch.exp(-0.5 * threshold_quantile.square()) / math.sqrt(2 * math.pi)
        torch.testing.assert_close(result.probability_sum, epsilon, atol=1e-15, rtol=1e-15)
        torch.testing.assert_close(result.residual, torch.zeros_like(epsilon),
                                   atol=1e-15, rtol=0)
        torch.testing.assert_close(result.A, A_probability / density[:, None],
                                   atol=2e-14, rtol=2e-14)
        torch.testing.assert_close(result.c, c_probability / density,
                                   atol=2e-14, rtol=2e-14)

    def test_single_feature_fixed_covariance_residual_is_affine_in_pose(self):
        z = self.state()
        d = torch.tensor([[0.04]], dtype=self.dtype)
        q = torch.tensor([[[0.8, 0.6, 0.3]]], dtype=self.dtype)
        mask = torch.ones_like(d, dtype=torch.bool)
        epsilon = torch.tensor([0.2], dtype=self.dtype)
        result = self.evaluate(d, q, z, mask=mask, epsilon=epsilon,
                               return_frozen_features=True)
        A_probability, h_probability = PSDF._compute_A(d, q, mask, z, epsilon, 0.0)
        c_probability = PSDF._compute_c(A_probability, h_probability, z)
        probability_errors = []
        for displacement in ([0.01, 0.0, 0.02], [-0.01, 0.003, -0.01],
                             [0.002, -0.003, 0.03]):
            candidate = z.clone()
            candidate[0, :3] += torch.tensor(displacement, dtype=self.dtype)
            frozen = evaluate_mf_icdf_frozen(result.frozen_features, candidate, epsilon, 0.0)
            self.assertEqual(frozen.branch.item(), MFInverseCDFBranch.INVERSE_CDF)
            affine_residual = (result.A * candidate).sum(-1) + result.c
            torch.testing.assert_close(affine_residual, frozen.residual,
                                       atol=2e-14, rtol=2e-14)
            probability_affine = (A_probability * candidate).sum(-1) + c_probability
            probability_errors.append(
                (probability_affine - (epsilon - frozen.probability_sum)).abs().item()
            )
        self.assertGreater(max(probability_errors), 1e-3)

    def test_tiny_epsilon_does_not_round_one_minus_epsilon(self):
        result = self.evaluate([[0.03]], [[[1.0, 0.0, 0.0]]], epsilon=1e-100)
        self.assertTrue(result.valid.item())
        self.assertTrue(torch.isfinite(result.residual).all())
        self.assertLess(result.residual.item(), 0.0)

    def test_extension_handles_probability_sum_above_one(self):
        z = torch.zeros((1, 8), dtype=self.dtype)
        distance = -torch.special.ndtri(torch.tensor(0.75, dtype=self.dtype)).item()
        d = [[distance, distance]]
        q = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        result = self.evaluate(d, q, z)
        self.assertTrue(result.valid.item())
        self.assertEqual(result.branch.item(), MFInverseCDFBranch.LINEAR_EXTENSION)
        torch.testing.assert_close(result.probability_sum, torch.tensor([1.5], dtype=self.dtype))
        expected = -math.sqrt(2 * math.pi) - (-torch.special.ndtri(torch.tensor(0.2, dtype=self.dtype)))
        torch.testing.assert_close(result.residual[0], expected)
        candidate = z[0].clone().requires_grad_(True)
        value = self.reference_residual(candidate, z[0], torch.tensor(d[0], dtype=self.dtype),
                                        torch.tensor(q[0], dtype=self.dtype), 0.2)
        expected_A, = torch.autograd.grad(value, candidate)
        torch.testing.assert_close(result.A[0], expected_A)

    def test_half_probability_join_is_value_and_gradient_continuous(self):
        z = torch.zeros((3, 8), dtype=self.dtype)
        distances = torch.tensor([[1e-8], [0.0], [-1e-8]], dtype=self.dtype)
        q = torch.tensor([[[1.0, 0.0, 0.0]]] * 3, dtype=self.dtype)
        result = self.evaluate(distances, q, z)
        self.assertTrue(result.valid.all())
        self.assertEqual(result.branch[0], MFInverseCDFBranch.INVERSE_CDF)
        self.assertEqual(result.branch[1], MFInverseCDFBranch.INVERSE_CDF)
        self.assertEqual(result.branch[2], MFInverseCDFBranch.LINEAR_EXTENSION)
        torch.testing.assert_close(result.A[:, 0], torch.ones(3, dtype=self.dtype),
                                   atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(result.residual[[0, 2]], result.residual[1].expand(2),
                                   atol=2e-8, rtol=0)

    def test_underflow_clips_residual_and_gradient_consistently(self):
        z = torch.zeros((1, 8), dtype=self.dtype)
        result = self.evaluate([[40.0]], [[[1.0, 0.0, 0.0]]], z,
                               return_frozen_features=True)
        self.assertTrue(result.valid.item())
        self.assertTrue(result.tail_clipped.item())
        self.assertEqual(result.branch.item(), MFInverseCDFBranch.TAIL_CLIPPED)
        self.assertEqual(result.probability_sum.item(), 0.0)
        torch.testing.assert_close(result.A, torch.zeros_like(z))
        expected = -torch.special.ndtri(torch.tensor(ICDF_PROBABILITY_FLOOR, dtype=self.dtype))
        expected += torch.special.ndtri(torch.tensor(0.2, dtype=self.dtype))
        torch.testing.assert_close(result.residual[0], expected)
        plus, minus = z.clone(), z.clone()
        plus[0, 0], minus[0, 0] = 1e-4, -1e-4
        h_plus = evaluate_mf_icdf_frozen(result.frozen_features, plus, 0.2, 0.0).residual
        h_minus = evaluate_mf_icdf_frozen(result.frozen_features, minus, 0.2, 0.0).residual
        torch.testing.assert_close(h_plus, h_minus, atol=0, rtol=0)
        # The snapshot retains geometry, not the clipped zero derivative.
        candidate = z.clone()
        candidate[0, 0] = -39.0
        recovered = evaluate_mf_icdf_frozen(result.frozen_features, candidate, 0.2, 0.0)
        self.assertFalse(recovered.tail_clipped.item())
        self.assertAlmostEqual(recovered.A[0, 0].item(), 1.0)

    def test_masked_nan_padding_does_not_change_result(self):
        result = self.evaluate([[0.03, float("nan")]],
                               [[[1.0, 0.0, 0.0], [float("nan")] * 3]],
                               mask=torch.tensor([[True, False]]))
        expected = self.evaluate([[0.03]], [[[1.0, 0.0, 0.0]]])
        self.assertTrue(result.valid.item())
        torch.testing.assert_close(result.A, expected.A)
        torch.testing.assert_close(result.c, expected.c)

    def test_empty_features_are_distinct_from_underflow_and_no_obstacles(self):
        d, q = torch.empty((2, 0), dtype=self.dtype), torch.empty((2, 0, 3), dtype=self.dtype)
        missing = self.evaluate(d, q)
        self.assertTrue(missing.required.all())
        self.assertFalse(missing.valid.any())
        self.assertTrue((missing.status == MFInverseCDFStatus.NO_VALID_FEATURES).all())
        self.assertTrue(torch.isnan(missing.A).all())
        absent = self.evaluate(d, q, no_obstacles=True, return_frozen_features=True)
        self.assertFalse(absent.required.any())
        self.assertTrue((absent.status == MFInverseCDFStatus.NO_OBSTACLES).all())
        torch.testing.assert_close(absent.c, torch.ones(2, dtype=self.dtype))
        torch.testing.assert_close(absent.A, torch.zeros((2, 8), dtype=self.dtype))
        self.assertTrue(torch.isnan(absent.residual).all())

    def test_d_min_controls_strict_domain_gate(self):
        phi = torch.tensor([0.00005, 0.000075, 0.0001, 0.00015], dtype=self.dtype)
        result = self.evaluate(
            [[0.03]] * 4, [[[1.0, 0.0, 0.0]]] * 4,
            phi=phi, d_min=0.0001,
        )
        self.assertEqual(result.required.tolist(), [False, False, False, True])
        self.assertEqual(result.valid.tolist(), [False, False, False, True])
        self.assertTrue((result.status[:3] == MFInverseCDFStatus.DOMAIN_INACTIVE).all())
        torch.testing.assert_close(result.A[:3], torch.zeros((3, 8), dtype=self.dtype))
        torch.testing.assert_close(result.c[:3], torch.ones(3, dtype=self.dtype))

    def test_stage_gates_and_invalid_inputs_are_separate(self):
        z = self.state(6)
        z[2, 4] = float("nan")
        d = torch.full((6, 1), 0.03, dtype=self.dtype)
        q = torch.tensor([[[1.0, 0.0, 0.0]]] * 6, dtype=self.dtype)
        q[5] = 0.0
        mask = torch.ones((6, 1), dtype=torch.bool)
        mask[4] = False
        phi = torch.tensor([0.0, 1.0, 1.0, float("nan"), 1.0, 1.0], dtype=self.dtype)
        result = self.evaluate(d, q, z, mask=mask, phi=phi,
                               stage_requested=torch.tensor([True, False, True, True, True, True]))
        expected = [MFInverseCDFStatus.DOMAIN_INACTIVE, MFInverseCDFStatus.NOT_REQUESTED,
                    MFInverseCDFStatus.INVALID_NOMINAL, MFInverseCDFStatus.INVALID_GEOMETRY,
                    MFInverseCDFStatus.NO_VALID_FEATURES, MFInverseCDFStatus.INVALID_VARIANCE]
        self.assertEqual(result.status.tolist(), expected)
        self.assertFalse(result.required[:2].any())
        self.assertTrue(result.required[2:].all())
        self.assertTrue(torch.isnan(result.c[2:]).all())

    def test_frozen_gate_stays_required_after_candidate_crosses_domain(self):
        z = self.state()
        initial = self.evaluate([[0.03]], [[[1.0, 0.0, 0.0]]], z,
                                phi=torch.tensor([0.01], dtype=self.dtype),
                                gradient=torch.tensor([[1.0, 0.0, 0.0]], dtype=self.dtype),
                                return_frozen_features=True)
        candidate = z.clone()
        candidate[0, 0] = -0.1
        candidate[0, 2] += 0.6
        frozen = evaluate_mf_icdf_frozen(initial.frozen_features, candidate, 0.2, 0.0)
        self.assertLess(frozen.phi.item(), 0.001)
        self.assertTrue(frozen.required.item())
        self.assertTrue(frozen.valid.item())
        expected = self.reference_residual(candidate[0], z[0],
                                            torch.tensor([0.03], dtype=self.dtype),
                                            torch.tensor([[1.0, 0.0, 0.0]], dtype=self.dtype), 0.2)
        torch.testing.assert_close(frozen.residual[0], expected)

    def test_wrapper_keeps_original_float64_state_and_existing_api(self):
        verts = torch.tensor([[-0.075, -0.045], [0.075, -0.045],
                              [0.075, 0.045], [-0.075, 0.045]], dtype=torch.float32)
        wrapper = AugmentedPSDFWrapper(verts, K_max=2, E_max=4)
        edges = torch.tensor([[0.12, -0.08], [0.2, -0.08], [0.2, 0.08], [0.12, 0.08]],
                              dtype=torch.float32)
        wrapper.update_edge_clusters([edges], [torch.roll(edges, -1, 0)])
        z = self.state(2)
        z[:, 4] = -3.600000012345
        z[:, 7] = 0.001234567890123
        result = wrapper.forward_mf_icdf(z, 0.2, 0.0, return_frozen_features=True)
        self.assertEqual(result.A.dtype, torch.float64)
        self.assertEqual(result.phi.dtype, torch.float32)
        torch.testing.assert_close(result.frozen_features["z_bar"], z, atol=0, rtol=0)
        ordinary_phi, ordinary_gradient = wrapper(z[:, :3])
        torch.testing.assert_close(result.phi, ordinary_phi, atol=0, rtol=0)
        torch.testing.assert_close(result.gradient, ordinary_gradient, atol=0, rtol=0)
        legacy = wrapper.forward_mf(z, 0.2, 0.0)
        self.assertEqual(len(legacy), 4)
        self.assertEqual(legacy[2].dtype, torch.float32)

    def test_epsilon_shape_and_range_validation(self):
        for epsilon in (0.0, 0.5, 1.0, float("nan"), ICDF_PROBABILITY_FLOOR):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                self.evaluate([[0.03]], [[[1.0, 0.0, 0.0]]], epsilon=epsilon)
        invalid = self.evaluate([[0.03]], [[[1.0, 0.0, 0.0]]],
                                epsilon=torch.tensor([0.5], dtype=self.dtype))
        self.assertEqual(invalid.status.item(), MFInverseCDFStatus.INVALID_EPSILON)
        with self.assertRaises(ValueError):
            self.evaluate([[0.03]], [[[1.0, 0.0, 0.0]]], epsilon=[0.2, 0.3])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_cuda_agree_on_valid_extension_and_underflow(self):
        z = torch.zeros((3, 8), dtype=self.dtype)
        d = torch.tensor([[1.0, 2.0], [-1.0, -1.0], [40.0, 41.0]], dtype=self.dtype)
        q = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]] * 3, dtype=self.dtype)
        cpu = self.evaluate(d, q, z)
        gpu = self.evaluate(d.cuda(), q.cuda(), z.cuda())
        for field in ("A", "c", "residual", "probability_sum", "valid", "required",
                      "status", "branch", "tail_clipped"):
            torch.testing.assert_close(getattr(cpu, field), getattr(gpu, field).cpu(),
                                       atol=2e-12, rtol=2e-12)


if __name__ == "__main__":
    unittest.main()
