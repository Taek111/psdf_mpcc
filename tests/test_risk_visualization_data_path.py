import unittest

import numpy as np

from control.controller import BaseController
from control.rmpcc_optimizer import RMPCCOptimizer, RMPCCOptimizerParam
from sim.logger import ControllerLogger


class _FakeSolution:
    def __init__(self, state_trajectory, input_trajectory):
        self._state_trajectory = np.asarray(state_trajectory, dtype=float)
        self._input_trajectory = np.asarray(input_trajectory, dtype=float)

    def get_state_trajectory(self):
        return self._state_trajectory.copy()

    def get_input_trajectory(self):
        return self._input_trajectory.copy()


class _OptimizerWithoutBooleRiskAccessor:
    pass


class _FakePrepareSolver:
    def __init__(self, states, inputs):
        self.states = np.asarray(states, dtype=float).copy()
        self.inputs = np.asarray(inputs, dtype=float).copy()

    def get(self, stage, field):
        if field == "x":
            return self.states[stage].copy()
        if field == "u":
            return self.inputs[stage].copy()
        raise KeyError(field)

    def set(self, stage, field, value):
        if field == "x":
            self.states[stage] = np.asarray(value, dtype=float)
        elif field == "u":
            self.inputs[stage] = np.asarray(value, dtype=float)
        elif field not in ("lbx", "ubx", "p"):
            raise KeyError(field)

    @staticmethod
    def solve():
        return 0


class _ResetOnlySolver:
    @staticmethod
    def reset(**_):
        return None


class RiskVisualizationDataPathTest(unittest.TestCase):
    @staticmethod
    def _controller_with_solution(optimizer):
        controller = BaseController(optimizer, opt_param=None)
        controller._opt_sol = _FakeSolution(
            state_trajectory=np.zeros((3, 2)),
            input_trajectory=np.zeros((2, 1)),
        )
        return controller

    def test_optimizer_accessor_returns_isolated_copy(self):
        optimizer = RMPCCOptimizer()
        self.assertIsNone(optimizer.get_last_boole_risk_sum_trajectory())
        self.assertIsNone(optimizer.get_last_boole_risk_visualization_data())

        expected = np.array([np.nan, 0.01, 0.02, 0.03])
        optimizer.N = expected.size - 1
        optimizer._last_nominal_probability_sum = expected.copy()

        returned = optimizer.get_last_boole_risk_sum_trajectory()

        self.assertEqual(returned.shape, (optimizer.N + 1,))
        np.testing.assert_equal(returned, expected)
        returned[1] = 99.0
        np.testing.assert_equal(
            optimizer.get_last_boole_risk_sum_trajectory(),
            expected,
        )

        optimizer.reset()
        self.assertIsNone(optimizer.get_last_boole_risk_sum_trajectory())
        self.assertIsNone(optimizer.get_last_boole_risk_visualization_data())

        optimizer._last_nominal_probability_sum = expected.copy()
        optimizer._last_boole_risk_visualization_data = {
            "nominal_poses": np.zeros((expected.size, 3)),
            "boole_risk_sum": expected.copy(),
            "epsilon": np.full(expected.size, 0.05),
            "mf_mask": np.ones(expected.size, dtype=bool),
        }
        optimizer.cleanup()
        self.assertIsNone(optimizer.get_last_boole_risk_sum_trajectory())
        self.assertIsNone(optimizer.get_last_boole_risk_visualization_data())

    def test_prepare_caches_source_aligned_visualization_bundle(self):
        optimizer = RMPCCOptimizer()
        optimizer.N = 2
        optimizer.nx = 8
        optimizer.nu = 3
        optimizer.param = RMPCCOptimizerParam()
        optimizer.param.horizon = optimizer.N
        optimizer.state = None

        states = np.array(
            [
                [0.0, 0.1, 0.2, 0.0, 1e-4, 2e-4, 3e-4, 0.0],
                [0.5, 0.2, 0.3, 0.2, 2e-4, 3e-4, 4e-4, 0.0],
                [1.0, 0.3, 0.4, 0.4, 3e-4, 4e-4, 5e-4, 0.0],
            ]
        )
        solver = _FakePrepareSolver(states, np.zeros((optimizer.N, 3)))
        epsilon = np.array([0.05, 0.04, 0.03])
        expected_risk = np.array([0.01, 0.02, np.nan])
        constraint_data = {
            "mf_A_raw": np.zeros((optimizer.N + 1, optimizer.nx)),
            "mf_c_raw": epsilon - np.array([0.01, 0.02, 0.005]),
            "epsilon": epsilon.copy(),
            "mf_valid": np.array([True, True, False]),
            "mf_mask": np.array([True, True, False]),
        }

        optimizer._get_current_s_bounds = lambda _: (0.0, 1.0)
        optimizer._get_augmented_state_bounds = lambda *_: (
            None,
            np.zeros(optimizer.nx),
            np.full(optimizer.nx, 10.0),
        )
        optimizer._predict_stage_s_values = lambda *_: np.zeros(optimizer.N + 1)
        optimizer._compute_constraint_affine_data = lambda _: constraint_data
        optimizer._compute_exact_psdf = lambda _: (1.0, np.zeros(3))

        def set_constraint_parameters(_, __, data):
            data["mf_mask"][0] = False

        optimizer._set_constraint_parameters = set_constraint_parameters

        self.assertEqual(optimizer._prepare_and_solve(solver), 0)
        cached = optimizer.get_last_boole_risk_visualization_data()

        self.assertEqual(
            set(cached),
            {"nominal_poses", "boole_risk_sum", "epsilon", "mf_mask"},
        )
        np.testing.assert_allclose(cached["nominal_poses"], states[:, :3])
        np.testing.assert_allclose(
            cached["boole_risk_sum"],
            expected_risk,
            equal_nan=True,
        )
        np.testing.assert_equal(cached["epsilon"], epsilon)
        np.testing.assert_array_equal(
            cached["mf_mask"],
            np.array([False, True, False]),
        )
        for key in cached:
            self.assertEqual(cached[key].shape[0], optimizer.N + 1)

        cached["nominal_poses"][1, 0] = 99.0
        cached["boole_risk_sum"][1] = 99.0
        cached["epsilon"][1] = 99.0
        cached["mf_mask"][1] = False
        fresh = optimizer.get_last_boole_risk_visualization_data()
        np.testing.assert_allclose(fresh["nominal_poses"], states[:, :3])
        np.testing.assert_allclose(
            fresh["boole_risk_sum"],
            expected_risk,
            equal_nan=True,
        )
        np.testing.assert_equal(fresh["epsilon"], epsilon)
        np.testing.assert_array_equal(
            fresh["mf_mask"],
            np.array([False, True, False]),
        )

    def test_failed_prepare_does_not_reuse_previous_cycle_risk(self):
        optimizer = RMPCCOptimizer()
        optimizer.param = RMPCCOptimizerParam()
        optimizer._last_nominal_probability_sum = np.array([0.01, 0.02])
        optimizer._last_boole_risk_visualization_data = {
            "nominal_poses": np.zeros((2, 3)),
            "boole_risk_sum": np.array([0.01, 0.02]),
            "epsilon": np.full(2, 0.05),
            "mf_mask": np.array([False, True]),
        }

        def fail_before_constraint_evaluation(_):
            raise RuntimeError("synthetic prepare failure")

        optimizer._get_current_s_bounds = fail_before_constraint_evaluation

        with self.assertRaisesRegex(RuntimeError, "synthetic prepare failure"):
            optimizer._prepare_and_solve(object())

        self.assertIsNone(optimizer.get_last_boole_risk_sum_trajectory())
        self.assertIsNone(optimizer.get_last_boole_risk_visualization_data())

    def test_failed_backup_restores_main_nominal_risk_bundle(self):
        optimizer = RMPCCOptimizer()
        optimizer.solver = object()
        optimizer.backup_solver = _ResetOnlySolver()
        optimizer._constraint_data = {"source": "main"}
        main_sum = np.array([0.01, 0.02])
        main_bundle = {
            "nominal_poses": np.zeros((2, 3)),
            "boole_risk_sum": main_sum.copy(),
            "epsilon": np.full(2, 0.05),
            "mf_mask": np.array([False, True]),
        }
        optimizer._last_nominal_probability_sum = main_sum.copy()
        optimizer._last_boole_risk_visualization_data = {
            key: value.copy() for key, value in main_bundle.items()
        }
        optimizer._copy_primal_guess = lambda *_: None

        def failed_backup_prepare(_):
            optimizer._constraint_data = {"source": "backup"}
            optimizer._last_nominal_probability_sum = np.array([0.4, 0.5])
            optimizer._last_boole_risk_visualization_data = {
                "nominal_poses": np.ones((2, 3)),
                "boole_risk_sum": np.array([0.4, 0.5]),
                "epsilon": np.full(2, 0.05),
                "mf_mask": np.array([False, True]),
            }
            return 1

        optimizer._prepare_and_solve = failed_backup_prepare

        self.assertIsNone(optimizer._try_backup_recovery())

        self.assertEqual(optimizer._constraint_data, {"source": "main"})
        np.testing.assert_equal(
            optimizer.get_last_boole_risk_sum_trajectory(),
            main_sum,
        )
        restored_bundle = optimizer.get_last_boole_risk_visualization_data()
        for key, expected in main_bundle.items():
            np.testing.assert_equal(restored_bundle[key], expected)

    def test_controller_logger_stores_visualization_bundle_per_cycle(self):
        optimizer = RMPCCOptimizer()
        expected = np.array([np.nan, 0.015, 0.025])
        expected_visualization_data = {
            "nominal_poses": np.arange(9, dtype=float).reshape(3, 3),
            "boole_risk_sum": expected.copy(),
            "epsilon": np.full(3, 0.05),
            "mf_mask": np.array([False, True, True]),
        }
        optimizer._last_boole_risk_visualization_data = {
            key: value.copy()
            for key, value in expected_visualization_data.items()
        }
        controller = self._controller_with_solution(optimizer)
        logger = ControllerLogger()

        controller.logging(logger)

        self.assertEqual(len(logger._boole_risk_visualization_data), 1)
        logged_data = logger._boole_risk_visualization_data[0]
        for key, expected_value in expected_visualization_data.items():
            np.testing.assert_equal(logged_data[key], expected_value)
        optimizer._last_boole_risk_visualization_data["boole_risk_sum"][1] = 0.5
        np.testing.assert_equal(
            logged_data["boole_risk_sum"],
            expected_visualization_data["boole_risk_sum"],
        )

    def test_controller_logger_records_none_without_boole_risk_accessor(self):
        controller = self._controller_with_solution(
            _OptimizerWithoutBooleRiskAccessor()
        )
        logger = ControllerLogger()

        controller.logging(logger)

        self.assertEqual(logger._boole_risk_visualization_data, [None])


if __name__ == "__main__":
    unittest.main()
