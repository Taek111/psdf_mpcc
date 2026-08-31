import unittest

import torch

from models.psdf import PSDF


class PSDFTest(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.verts = torch.tensor(
            [
                [-0.5, -0.5],
                [0.5, -0.5],
                [0.5, 0.5],
                [-0.5, 0.5],
            ],
            dtype=self.dtype,
        )
        self.model = PSDF(self.verts)

    def rectangle_edges(self, left, right, bottom, top):
        points = torch.tensor(
            [
                [left, bottom],
                [right, bottom],
                [right, top],
                [left, top],
            ],
            dtype=self.dtype,
        )
        return points, torch.roll(points, shifts=-1, dims=0)

    def evaluate_box(self, bounds):
        edges_a, edges_b = self.rectangle_edges(*bounds)
        mask = torch.ones((1, 4), dtype=torch.bool)
        pose = torch.zeros((1, 3), dtype=self.dtype, requires_grad=True)
        value = self.model(
            edges_a.unsqueeze(0),
            edges_b.unsqueeze(0),
            mask,
            pose,
        )
        value.sum().backward()
        return value, pose.grad

    def test_hard_min_penetration_returns_exact_sat_overlap(self):
        cases = {
            "separated": ((1.0, 2.0, -1.0, 1.0), 0.5),
            "touching": ((0.5, 1.5, -1.0, 1.0), 0.0),
            "penetrating_0.1": ((0.4, 1.4, -1.0, 1.0), -0.1),
            "penetrating_0.3": ((0.2, 1.2, -1.0, 1.0), -0.3),
        }

        for name, (bounds, expected) in cases.items():
            with self.subTest(name=name):
                value, gradient = self.evaluate_box(bounds)
                self.assertEqual(value.shape, (1,))
                self.assertAlmostEqual(value.item(), expected, places=8)
                self.assertTrue(torch.isfinite(value).all())
                self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":
    unittest.main()
