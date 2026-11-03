"""Shared controller configuration without solver generation or navigation."""

import io
from contextlib import redirect_stdout
from statistics import NormalDist
import unittest

import numpy as np

from control.cc_ecs_optimizer import (
    CCECSOptimizer,
    CCECSOptimizerParam,
    PROCESS_NOISE_FIELDS,
)
from control.cc_mf_optimizer import CCMFOptimizerParam
from control.cc_sf_optimizer import CCSFOptimizer, CCSFOptimizerParam
from control.mpcc_optimizer import MPCCOptimizerParam
from sim.simulation_mpc import simulation_mpc


CONTROLLERS = (
    ("mpcc", MPCCOptimizerParam),
    ("cc_sf", CCSFOptimizerParam),
    ("cc_mf", CCMFOptimizerParam),
    ("cc_ecs", CCECSOptimizerParam),
)


class MPCCConfigTest(unittest.TestCase):
    def setUp(self):
        self.sf_optimizer = CCSFOptimizer()
        self.addCleanup(self.sf_optimizer.cleanup)

    def apply(self, method, param_type, config):
        param = param_type()
        output = io.StringIO()
        with redirect_stdout(output):
            simulation_mpc._apply_mpcc_param_overrides(param, config, method)
        return param, output.getvalue()

    def test_common_values_reach_every_applicable_controller_and_keep_arrays_independent(self):
        config = {"mpcc": {
            "a_max": .27, "r_delta_omega": .2, "d_min": .002,
            "chance_epsilon": .15, "sigma_psi0": .0003,
            "covariance_proximal_weight": .025,
            "mat_Qe": [[3., 0.], [0., 11.]],
            "mat_Re": [[2., 0.], [0., .03]],
        }}
        params = []
        for method, param_type in CONTROLLERS:
            with self.subTest(method=method):
                param, output = self.apply(method, param_type, config)
                params.append(param)
                self.assertEqual((param.a_max, param.r_delta_omega, param.d_min), (.27, .2, .002))
                self.assertIsInstance(param.mat_Qe, np.ndarray)
                self.assertIsInstance(param.mat_Re, np.ndarray)
                np.testing.assert_array_equal(param.mat_Qe, config["mpcc"]["mat_Qe"])
                np.testing.assert_array_equal(param.mat_Re, config["mpcc"]["mat_Re"])
                if method == "mpcc":
                    self.assertFalse(hasattr(param, "chance_epsilon"))
                    self.assertFalse(hasattr(param, "covariance_proximal_weight"))
                    self.assertIn("ignores CC-only settings", output)
                else:
                    self.assertEqual(param.chance_epsilon, .15)
                    self.assertEqual(param.sigma_psi0, .0003)
                    self.assertEqual(param.covariance_proximal_weight, .025)
                if method == "cc_sf":
                    self.assertIsNone(param.chance_gamma)
                    self.assertAlmostEqual(
                        self.sf_optimizer._get_chance_gamma(param), NormalDist().inv_cdf(.85),
                    )
        params[0].mat_Qe[0, 0] = 99.
        self.assertEqual(config["mpcc"]["mat_Qe"][0][0], 3.)
        for param in params[1:]:
            self.assertEqual(param.mat_Qe[0, 0], 3.)

    def test_shared_block_rejects_method_only_fields(self):
        config = {"mpcc": {
            "mf_slack_linear": 5., "chance_gamma": 9., "risk_margin_cap": .5,
            "unknown_setting": True,
        }}
        for method, param_type in CONTROLLERS:
            with self.subTest(method=method):
                param, output = self.apply(method, param_type, config)
                defaults = param_type()
                for field in config["mpcc"]:
                    self.assertIn(field, output)
                    if hasattr(defaults, field):
                        self.assertEqual(getattr(param, field), getattr(defaults, field))
                    else:
                        self.assertFalse(hasattr(param, field))
                self.assertIn("ignoring non-common keys", output)

    def test_legacy_local_common_values_win_with_migration_warning(self):
        param, output = self.apply("cc_sf", CCSFOptimizerParam, {
            "mpcc": {"a_max": .2, "chance_epsilon": .15},
            "cc_sf": {
                "a_max": .4, "chance_epsilon": .12,
                "mat_Re": [[2., 0.], [0., .04]], "risk_margin_cap": 1.5,
            },
        })
        self.assertEqual(param.a_max, .4)
        self.assertEqual(param.chance_epsilon, .12)
        self.assertIsNone(param.chance_gamma)
        self.assertAlmostEqual(
            self.sf_optimizer._get_chance_gamma(param), NormalDist().inv_cdf(.88),
        )
        self.assertIsInstance(param.mat_Re, np.ndarray)
        self.assertEqual(param.mat_Re[1, 1], .04)
        self.assertEqual(param.risk_margin_cap, 1.5)
        self.assertIn("move them to 'mpcc'", output)
        self.assertIn("a_max", output)

    def test_explicit_sf_gamma_is_preserved_when_epsilon_changes(self):
        param, _ = self.apply("cc_sf", CCSFOptimizerParam, {
            "mpcc": {"chance_epsilon": .15},
            "cc_sf": {"chance_gamma": 2.5},
        })
        with redirect_stdout(io.StringIO()):
            simulation_mpc._apply_param_overrides(param, {"chance_epsilon": .12}, "CC-SF")
        self.assertEqual(param.chance_gamma, 2.5)
        self.assertEqual(self.sf_optimizer._get_chance_gamma(param), 2.5)

    def test_method_blocks_do_not_leak_between_controllers(self):
        param, output = self.apply("cc_sf", CCSFOptimizerParam, {
            "cc_mf": {"r_delta_omega": 99., "mf_slack_linear": 2.},
            "cc_sf": {"mf_slack_linear": 8., "risk_margin_cap": 1.25},
        })
        self.assertEqual(param.r_delta_omega, CCSFOptimizerParam().r_delta_omega)
        self.assertEqual(param.risk_margin_cap, 1.25)
        self.assertFalse(hasattr(param, "mf_slack_linear"))
        self.assertIn("ignoring unknown CC-SF config keys: mf_slack_linear", output)

    def test_ecs_ignores_shared_noise_but_keeps_other_common_settings(self):
        common = {name: .001 for name in PROCESS_NOISE_FIELDS}
        common.update({"covariance_growth_scale": .5, "a_max": .25, "chance_epsilon": .12})
        param, output = self.apply("cc_ecs", CCECSOptimizerParam, {"mpcc": common})
        CCECSOptimizer._validate_param(param)
        for name in PROCESS_NOISE_FIELDS:
            self.assertEqual(getattr(param, name), 0., msg=name)
            self.assertIn(name, output)
        self.assertEqual(param.covariance_growth_scale, CCECSOptimizerParam().covariance_growth_scale)
        self.assertEqual(param.a_max, .25)
        self.assertEqual(param.chance_epsilon, .12)
        self.assertIn("keeps Q=0", output)
        for method, param_type in CONTROLLERS[1:3]:
            noisy, _ = self.apply(method, param_type, {"mpcc": common})
            for name in PROCESS_NOISE_FIELDS:
                self.assertEqual(getattr(noisy, name), .001, msg=name)

    def test_ecs_explicit_nonzero_local_noise_still_fails_validation(self):
        param, output = self.apply("cc_ecs", CCECSOptimizerParam, {
            "mpcc": {"beta_omega": .001},
            "cc_ecs": {"beta_omega": .002},
        })
        self.assertEqual(param.beta_omega, .002)
        self.assertIn("overrides common parameters", output)
        with self.assertRaisesRegex(ValueError, "beta_omega must be zero"):
            CCECSOptimizer._validate_param(param)


if __name__ == "__main__":
    unittest.main()
