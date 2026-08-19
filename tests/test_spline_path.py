import unittest
from types import SimpleNamespace

import numpy as np

from control.mpcc_optimizer import MPCCOptimizer
from control.rmpcc_pv_optimizer import RMPCCPVOptimizer
from planning.trajectory_generator.spline_reference_generator import (
    build_cubic_spline_path_data as planner_build_path,
)
from utils.spline_path import (
    build_cubic_spline_path_data,
    clamp_progress,
    curvature,
    evaluate,
    evaluate_spline_path,
    normalize_path_data,
    progress_bounds,
    project_progress,
    slice_path_data,
    stage_parameter,
    tangent,
)


class SplinePathUtilityTest(unittest.TestCase):
    def setUp(self):
        self.points = np.array(
            [[0.0, 0.0], [0.5, 0.2], [1.0, 0.0]],
            dtype=float,
        )
        self.path = build_cubic_spline_path_data(self.points)

    def test_planner_reexports_canonical_builder(self):
        self.assertIs(planner_build_path, build_cubic_spline_path_data)

    def test_scalar_and_vector_evaluation_are_consistent(self):
        samples = np.linspace(0.0, self.path["s_breaks"][-1], 9)
        vector = evaluate_spline_path(self.path, samples)
        scalar = np.stack([evaluate(self.path, value) for value in samples])
        np.testing.assert_allclose(vector, scalar, atol=1e-14, rtol=0.0)

    def test_normalize_accepts_aliases_and_validates_shapes(self):
        aliases = {
            "cx": self.path["segments_x"],
            "cy": self.path["segments_y"],
            "S": self.path["s_breaks"],
        }
        normalized = normalize_path_data(aliases)
        np.testing.assert_allclose(normalized["segments_x"], self.path["segments_x"])
        self.assertEqual(normalized["n_segments"], self.path["n_segments"])
        with self.assertRaises(ValueError):
            normalize_path_data({"cx": np.zeros((2, 3)), "cy": np.zeros((2, 3)), "S": [0, 1, 2]})

    def test_path_frame_supports_both_mpcc_layouts(self):
        progress = float(self.path["s_breaks"][1])
        frame_5 = stage_parameter(self.path, progress, include_curvature=False)
        frame_6 = stage_parameter(self.path, progress, include_curvature=True)
        self.assertEqual(frame_5.shape, (5,))
        self.assertEqual(frame_6.shape, (6,))
        np.testing.assert_allclose(frame_6[:5], frame_5)
        self.assertAlmostEqual(frame_6[-1], curvature(self.path, progress))
        self.assertGreater(np.linalg.norm(tangent(self.path, progress)), 0.99)

    def test_controller_adapters_keep_default_tangent_options(self):
        for optimizer_type, expected_size in (
            (MPCCOptimizer, 5),
            (RMPCCPVOptimizer, 6),
        ):
            optimizer = optimizer_type()
            optimizer.param = SimpleNamespace()
            optimizer.reference_path_data = None
            self.assertEqual(
                optimizer._build_stage_path_parameter(0.2).shape,
                (expected_size,),
            )

    def test_progress_helpers_clamp_project_and_slice(self):
        maximum = float(self.path["s_breaks"][-1])
        self.assertEqual(clamp_progress(self.path, -1.0), 0.0)
        self.assertEqual(clamp_progress(self.path, maximum + 1.0), maximum)
        self.assertEqual(progress_bounds(self.path, maximum / 2), (0.0, maximum / 2))
        position = evaluate(self.path, maximum * 0.7)
        projected = project_progress(self.path, position, None, maximum, 101)
        self.assertLess(abs(projected - maximum * 0.7), 0.05)
        local = slice_path_data(self.path, maximum * 0.25, maximum * 0.25, 1)
        self.assertEqual(local["n_segments"], 1)


if __name__ == "__main__":
    unittest.main()
