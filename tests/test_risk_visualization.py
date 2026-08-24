import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

import numpy as np

from control.rmpcc_optimizer import RMPCCOptimizerParam
from sim.simulation_mpc import simulation_mpc
from test_nmpc import build_parser


class BooleRiskVisualizationTest(unittest.TestCase):
    def test_active_stages_are_encoded_by_budget_utilization(self):
        risk_frame = {
            "nominal_poses": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.1, 0.0],
                    [2.0, 0.2, 0.0],
                    [3.0, 0.3, 0.0],
                ]
            ),
            "boole_risk_sum": np.array([0.0, 0.1, 0.2, 0.4]),
            "epsilon": np.full(4, 0.2),
            "mf_mask": np.array([False, True, True, True]),
        }

        visual = simulation_mpc._prepare_boole_risk_visual_data(risk_frame)

        np.testing.assert_allclose(
            visual["offsets"],
            np.array([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]]),
        )
        np.testing.assert_allclose(visual["normalized_risk"], [0.5, 1.0, 1.0])
        np.testing.assert_allclose(visual["sizes"], [58.0, 100.0, 100.0])
        np.testing.assert_array_equal(
            visual["budget_exceeded"],
            [False, True, True],
        )

    def test_inactive_and_invalid_stages_are_not_drawn(self):
        risk_frame = {
            "nominal_poses": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [np.nan, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ]
            ),
            "boole_risk_sum": np.array(
                [0.0, 0.01, 0.02, 0.03, -1e-12, -0.01]
            ),
            "epsilon": np.full(6, 0.05),
            "mf_mask": np.array([False, False, True, True, True, True]),
        }

        visual = simulation_mpc._prepare_boole_risk_visual_data(risk_frame)

        np.testing.assert_allclose(visual["offsets"], [[2.0, 0.0], [4.0, 0.0]])
        np.testing.assert_allclose(visual["normalized_risk"], [0.4, 0.0])
        np.testing.assert_array_equal(visual["budget_exceeded"], [False, False])

    def test_stagewise_epsilon_normalizes_equal_budget_usage_equally(self):
        risk_frame = {
            "nominal_poses": np.zeros((3, 3)),
            "boole_risk_sum": np.array([0.0, 0.01, 0.02]),
            "epsilon": np.array([0.1, 0.02, 0.04]),
            "mf_mask": np.array([False, True, True]),
        }

        visual = simulation_mpc._prepare_boole_risk_visual_data(risk_frame)

        np.testing.assert_allclose(visual["normalized_risk"], [0.5, 0.5])
        np.testing.assert_allclose(visual["sizes"], [58.0, 58.0])

    def test_missing_frame_returns_empty_marker_data(self):
        visual = simulation_mpc._prepare_boole_risk_visual_data(None)

        self.assertEqual(visual["offsets"].shape, (0, 2))
        self.assertEqual(visual["sizes"].shape, (0,))
        self.assertEqual(visual["normalized_risk"].shape, (0,))
        self.assertEqual(visual["budget_exceeded"].shape, (0,))

    def test_optimizer_param_controls_visualization_without_override(self):
        opt_param = RMPCCOptimizerParam()
        controller = SimpleNamespace(_param=opt_param)
        simulation = SimpleNamespace(
            _robot=SimpleNamespace(_controller=controller)
        )

        self.assertFalse(
            simulation_mpc._resolve_use_risk_visualization(simulation)
        )

        opt_param.use_risk_visualization = True
        self.assertTrue(
            simulation_mpc._resolve_use_risk_visualization(simulation)
        )
        self.assertFalse(
            simulation_mpc._resolve_use_risk_visualization(
                simulation,
                override=False,
            )
        )

    def test_rmpcc_config_block_can_override_visualization_param(self):
        opt_param = RMPCCOptimizerParam()

        with redirect_stdout(io.StringIO()):
            simulation_mpc._apply_param_overrides(
                opt_param,
                {"use_risk_visualization": True},
                "RMPCC",
            )

        self.assertTrue(opt_param.use_risk_visualization)

    def test_cli_uses_only_the_new_risk_visualization_switch(self):
        parser = build_parser()

        defaults = parser.parse_args([])
        self.assertIsNone(defaults.use_risk_visualization)
        self.assertFalse(hasattr(defaults, "plot_covariance"))
        self.assertTrue(
            parser.parse_args(["--risk-visualization"]).use_risk_visualization
        )

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--plot-covariance"])


if __name__ == "__main__":
    unittest.main()
