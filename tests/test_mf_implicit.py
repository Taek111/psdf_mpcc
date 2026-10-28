"""Mathematical contracts for the frozen-feature implicit MF model.

These tests use synthetic features, so geometry branch changes cannot enter
the finite differences. They do not build acados solvers or run simulations.
"""

import math
import unittest

import torch

from models.augmented_psdf import PSDF
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.mf_implicit import compute_mf_imp, evaluate_mf_imp_frozen, root_status_name


class MFImplicitKernelTest(unittest.TestCase):
    def setUp(self):
        self.z = torch.tensor(
            [[0.12, -0.07, 0.63, 0.18, math.log(0.02),
              math.log(0.03), math.log(0.04), 0.007]],
            dtype=torch.float64,
        )
        self.d = torch.tensor([[0.04, 0.055, 0.07]], dtype=torch.float64)
        self.q = torch.tensor(
            [[[1.0, 0.2, 0.3], [-0.4, 1.0, -0.1], [0.7, -0.8, 0.2]]],
            dtype=torch.float64,
        )
        self.mask = torch.ones_like(self.d, dtype=torch.bool)
        self.epsilon = 0.12
        self.d_min = 0.001

    def evaluate(self, *, d=None, q=None, mask=None, z=None, epsilon=None, **kwargs):
        z = self.z if z is None else z
        d = self.d if d is None else d
        q = self.q if q is None else q
        mask = self.mask if mask is None else mask
        return compute_mf_imp(
            torch.full((len(z),), 0.2, dtype=torch.float64),
            torch.zeros((len(z), 3), dtype=torch.float64),
            d, q, mask, z,
            self.epsilon if epsilon is None else epsilon,
            self.d_min,
            stage_requested=torch.ones(len(z), dtype=torch.bool),
            **kwargs,
        )

    @staticmethod
    def sigma(z, q):
        uf = z[:, 4, None].exp() * q[..., 0]
        ul = z[:, 5, None].exp() * q[..., 1]
        ut = z[:, 6, None].exp() * q[..., 2]
        coupled = ul + z[:, 7, None] * q[..., 2]
        return (uf.square() + coupled.square() + ut.square()).sqrt()

    def probability(self, d, q, z, margin):
        sigma = self.sigma(z, q)
        return (0.5 * torch.special.erfc(
            (d - self.d_min - margin[:, None]) / (math.sqrt(2.0) * sigma)
        )).sum(-1)

    def test_single_feature_matches_gaussian_margin_and_analytic_gradient(self):
        d, q = self.d[:, :1], self.q[:, :1]
        result = self.evaluate(d=d, q=q, mask=self.mask[:, :1])
        self.assertTrue(bool(result.root_valid.all()))
        self.assertEqual(int(result.root_iterations[0]), 0)
        sigma = self.sigma(self.z, q)
        gamma = -torch.special.ndtri(torch.tensor(self.epsilon, dtype=torch.float64))
        expected = d[:, 0] - self.d_min - gamma * sigma[:, 0]
        torch.testing.assert_close(result.margin, expected, atol=1e-13, rtol=1e-12)

        theta = self.z[0, 2]
        expected_pose = torch.stack((
            theta.cos() * q[0, 0, 0] - theta.sin() * q[0, 0, 1],
            theta.sin() * q[0, 0, 0] + theta.cos() * q[0, 0, 1],
            q[0, 0, 2],
        ))
        torch.testing.assert_close(result.A[0, :3], expected_pose)
        uf = self.z[0, 4].exp() * q[0, 0, 0]
        ul = self.z[0, 5].exp() * q[0, 0, 1]
        ut = self.z[0, 6].exp() * q[0, 0, 2]
        coupled = ul + self.z[0, 7] * q[0, 0, 2]
        sigma_gradient = torch.stack((
            uf.square(), ul * coupled, ut.square(), q[0, 0, 2] * coupled,
        )) / sigma[0, 0]
        torch.testing.assert_close(result.A[0, 4:], -gamma * sigma_gradient)
        self.assertEqual(float(result.A[0, 3]), 0.0)
        torch.testing.assert_close((result.A * self.z).sum(-1) + result.c, expected)

    def test_duplicate_features_preserve_probability_sum(self):
        count = 8
        d = self.d[:, :1].expand(1, count).clone()
        q = self.q[:, :1].expand(1, count, 3).clone()
        result = self.evaluate(d=d, q=q, mask=torch.ones_like(d, dtype=torch.bool))
        expected = d[:, 0] - self.d_min + self.sigma(self.z, q)[:, 0] * (
            torch.special.ndtri(torch.tensor(self.epsilon / count, dtype=torch.float64))
        )
        self.assertTrue(bool(result.root_valid.all()))
        torch.testing.assert_close(result.margin, expected, atol=1e-10, rtol=1e-10)
        self.assertEqual(int(result.feature_count[0]), count)
        self.assertGreater(int(result.root_iterations[0]), 0)

    def test_near_zero_single_feature_root_survives_endpoint_cancellation(self):
        q = self.q[:, :1]
        sigma = self.sigma(self.z, q)
        gamma = -torch.special.ndtri(torch.tensor(self.epsilon, dtype=torch.float64))
        # b + sigma*Q(epsilon) cancels near zero. An ulp of the margin
        # alone is much smaller than the rounding scale of those summands.
        d = self.d_min + gamma * sigma
        result = self.evaluate(d=d, q=q, mask=self.mask[:, :1])
        self.assertTrue(bool(result.root_valid.all()))
        self.assertEqual(int(result.root_iterations[0]), 0)
        self.assertLess(float(result.margin.abs().max()), 1e-12)
        probability = self.probability(d, q, self.z, result.margin)
        self.assertLessEqual(float((probability - self.epsilon).abs().max()), 1e-10)
        radius = result.root_error_bound
        self.assertTrue(bool(torch.isfinite(radius).all() & (radius > 0).all()))
        self.assertTrue(bool((self.probability(d, q, self.z, result.margin - radius) <= self.epsilon).all()))
        self.assertTrue(bool((self.probability(d, q, self.z, result.margin + radius) >= self.epsilon).all()))

    def test_common_distance_shift_is_exact(self):
        baseline = self.evaluate()
        shifted = self.evaluate(d=self.d + 0.035)
        self.assertTrue(bool(baseline.root_valid.all() & shifted.root_valid.all()))
        torch.testing.assert_close(shifted.margin, baseline.margin + 0.035, atol=1e-11, rtol=1e-10)
        torch.testing.assert_close(shifted.A, baseline.A, atol=1e-10, rtol=1e-9)

    def test_returned_root_passes_both_probability_and_local_bracket_checks(self):
        result = self.evaluate()
        self.assertTrue(bool(result.root_valid.all()))
        radius = result.root_error_bound
        self.assertTrue(bool(torch.isfinite(radius).all() & (radius > 0).all()))
        residual = self.probability(self.d, self.q, self.z, result.margin) - self.epsilon
        self.assertLessEqual(float(residual.abs().max()), 1e-10)
        self.assertTrue(bool((self.probability(self.d, self.q, self.z, result.margin - radius) <= self.epsilon).all()))
        self.assertTrue(bool((self.probability(self.d, self.q, self.z, result.margin + radius) >= self.epsilon).all()))
        torch.testing.assert_close(result.root_residual, residual, atol=1e-13, rtol=1e-8)
        f0 = self.probability(self.d, self.q, self.z, torch.zeros(1, dtype=torch.float64))
        torch.testing.assert_close(result.probability_sum, f0)

    def test_gradient_matches_frozen_model_central_differences(self):
        nominal = self.evaluate()
        self.assertTrue(bool(nominal.root_valid.all()))
        theta = self.z[0, 2]
        world_q = self.q.clone()
        world_q[..., 0] = theta.cos() * self.q[..., 0] - theta.sin() * self.q[..., 1]
        world_q[..., 1] = theta.sin() * self.q[..., 0] + theta.cos() * self.q[..., 1]
        step = 1e-6
        for coordinate in (0, 1, 2, 4, 5, 6, 7):
            with self.subTest(coordinate=coordinate):
                plus, minus = self.z.clone(), self.z.clone()
                plus[0, coordinate] += step
                minus[0, coordinate] -= step
                d_plus, d_minus = self.d, self.d
                if coordinate < 3:
                    d_plus = self.d + step * world_q[..., coordinate]
                    d_minus = self.d - step * world_q[..., coordinate]
                forward = self.evaluate(d=d_plus, z=plus)
                backward = self.evaluate(d=d_minus, z=minus)
                self.assertTrue(bool(forward.root_valid.all() & backward.root_valid.all()))
                numerical = (forward.margin - backward.margin) / (2.0 * step)
                torch.testing.assert_close(numerical[0], nominal.A[0, coordinate], atol=2e-7, rtol=2e-5)

    def test_boundary_tangent_matches_original_probability_residual(self):
        initial = self.evaluate()
        d = self.d - initial.margin[:, None]
        result = self.evaluate(d=d)
        self.assertTrue(bool(result.root_valid.all()))
        self.assertLess(float(result.margin.abs().max()), 1e-10)
        epsilon = torch.tensor([self.epsilon], dtype=torch.float64)
        original_A, _ = PSDF._compute_A(d, self.q, self.mask, self.z, epsilon, self.d_min)
        sigma = self.sigma(self.z, self.q)
        zeta = (d - self.d_min) / sigma
        derivative = (torch.exp(-0.5 * zeta.square()) / (math.sqrt(2.0 * math.pi) * sigma)).sum(-1)
        torch.testing.assert_close(result.A, original_A / derivative[:, None], atol=1e-8, rtol=1e-7)

    def test_frozen_reassessment_keeps_nominal_frame_features_and_snapshot(self):
        nominal = self.evaluate(return_frozen_features=True)
        snapshot = nominal.frozen_features
        original = {key: value.clone() for key, value in snapshot.items() if isinstance(value, torch.Tensor)}
        shifted = self.z.clone()
        shifted[:, 2] += 0.4
        shifted[:, 4] += 0.1
        reassessed = evaluate_mf_imp_frozen(snapshot, shifted, self.epsilon, self.d_min)
        distance = self.d + (
            snapshot['pose_feature_gradient'] * (shifted[:, :3] - self.z[:, :3])[:, None, :]
        ).sum(-1)
        expected = compute_mf_imp(
            snapshot['phi'], snapshot['gradient'], distance, self.q, self.mask,
            shifted, self.epsilon, self.d_min,
            required_override=snapshot['mf_required'],
            pose_feature_gradient=snapshot['pose_feature_gradient'],
        )
        self.assertTrue(bool(reassessed.root_valid.all() & expected.root_valid.all()))
        torch.testing.assert_close(reassessed.margin, expected.margin)
        torch.testing.assert_close(reassessed.A, expected.A)
        for key, value in original.items():
            torch.testing.assert_close(snapshot[key], value)

    def test_invalid_padding_cannot_poison_valid_features(self):
        expected = self.evaluate()
        d = torch.cat((self.d, torch.tensor([[float('nan'), float('inf')]], dtype=torch.float64)), dim=1)
        q = torch.cat((self.q, torch.full((1, 2, 3), float('nan'), dtype=torch.float64)), dim=1)
        mask = torch.cat((self.mask, torch.zeros((1, 2), dtype=torch.bool)), dim=1)
        padded = self.evaluate(d=d, q=q, mask=mask)
        self.assertTrue(bool(padded.root_valid.all()))
        torch.testing.assert_close(padded.margin, expected.margin)
        torch.testing.assert_close(padded.A, expected.A)
        self.assertEqual(int(padded.feature_count[0]), 3)

    def test_batched_masks_keep_unrequested_domain_and_failed_stages_distinct(self):
        z = self.z.expand(4, -1).clone()
        d = self.d.expand(4, -1).clone()
        q = self.q.expand(4, -1, -1).clone()
        mask = self.mask.expand(4, -1).clone()
        mask[3] = False
        result = compute_mf_imp(
            torch.tensor([.2, .2, 0.0, .2], dtype=torch.float64),
            torch.zeros((4, 3), dtype=torch.float64), d, q, mask, z,
            torch.tensor([.05, .1, .2, .15], dtype=torch.float64), self.d_min,
            stage_requested=torch.tensor([False, True, True, True]),
        )
        self.assertEqual(
            [root_status_name(int(code)) for code in result.root_status],
            ['not_requested', 'success', 'domain_inactive', 'no_valid_features'],
        )
        torch.testing.assert_close(result.mf_required, torch.tensor([False, True, False, True]))
        torch.testing.assert_close(result.root_valid, torch.tensor([False, True, False, False]))
        torch.testing.assert_close(result.c[[0, 2]], torch.ones(2, dtype=torch.float64))
        self.assertTrue(bool(torch.isnan(result.A[3]).all()))
        self.assertTrue(bool(torch.isnan(result.margin[[0, 2, 3]]).all()))
        self.assertTrue(bool(torch.isfinite(result.margin[1])))

    def test_required_empty_or_zero_variance_features_fail_explicitly(self):
        for q, mask, expected_status in (
            (self.q, torch.zeros_like(self.mask), 'no_valid_features'),
            (torch.zeros_like(self.q), self.mask, 'invalid_variance'),
        ):
            with self.subTest(expected_status=expected_status):
                result = self.evaluate(q=q, mask=mask)
                self.assertFalse(bool(result.root_valid[0]))
                self.assertEqual(root_status_name(int(result.root_status[0])), expected_status)
                self.assertTrue(bool(torch.isnan(result.margin).all()))
                self.assertTrue(bool(torch.isnan(result.A).all()))
                self.assertTrue(bool(torch.isnan(result.c).all()))

    def test_unsupported_epsilon_and_invalid_tolerances_raise(self):
        for epsilon in (0.0, 0.5, 0.8, float('nan')):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                self.evaluate(epsilon=epsilon)
        for options in ({'root_max_iter': 0}, {'root_atol_m': 0.0},
                        {'root_rtol': -1.0}, {'root_prob_atol': float('inf')}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.evaluate(**options)

    def test_iteration_limit_does_not_accept_unconverged_upper_bound(self):
        count = 12
        d = self.d[:, :1].expand(1, count).clone()
        q = self.q[:, :1].expand(1, count, 3).clone()
        result = self.evaluate(d=d, q=q, mask=torch.ones_like(d, dtype=torch.bool), root_max_iter=1)
        self.assertFalse(bool(result.root_valid[0]))
        self.assertTrue(bool(torch.isnan(result.margin).all()))
        self.assertTrue(bool(torch.isnan(result.root_error_bound).all()))
        self.assertEqual(int(result.root_iterations[0]), 1)

    def test_negative_margin_is_valid_and_preserves_chance_condition_sign(self):
        result = self.evaluate(d=self.d - 0.08)
        self.assertTrue(bool(result.root_valid.all()))
        self.assertLess(float(result.margin[0]), 0.0)
        self.assertGreater(float(result.probability_sum[0]), self.epsilon)

    def test_empty_wrapper_distinguishes_missing_root_from_positive_sentinel(self):
        verts = torch.tensor([[-.1, -.05], [.1, -.05], [.1, .05], [-.1, .05]], dtype=torch.float32)
        wrapper = AugmentedPSDFWrapper(verts, K_max=1, E_max=4)
        result = wrapper.forward_mf_imp(self.z, self.epsilon, self.d_min)
        self.assertEqual(result.A.dtype, torch.float64)
        self.assertEqual(float(result.probability_sum[0]), 0.0)
        self.assertTrue(bool(torch.isnan(result.margin).all()))
        self.assertFalse(bool(result.root_valid[0]))
        self.assertEqual(root_status_name(int(result.root_status[0])), 'no_obstacles')
        self.assertEqual(float(result.c[0]), 1.0)


if __name__ == '__main__':
    unittest.main()
