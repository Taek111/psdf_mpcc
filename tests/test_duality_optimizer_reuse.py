"""Check cutoff and parameter refresh without running an optimization solver."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import casadi as ca
import numpy as np

from control.dcbf_optimizer import NmpcDbcfOptimizer, NmpcDcbfOptimizerParam
from control.duality_optimizer_utils import ReusableRegionDistanceQuery, validate_cutoff
from control.obca_optimizer import OBCAOptimizer, OBCAOptimizerParam
from models.geometry_utils import RectangleRegion


class DualityOptimizerReuseTest(unittest.TestCase):
    optimizer_types = (
        (OBCAOptimizer, OBCAOptimizerParam),
        (NmpcDbcfOptimizer, NmpcDcbfOptimizerParam),
    )

    def make_case(self, optimizer_type, param_type):
        param = param_type()
        param.horizon = 3
        if hasattr(param, "horizon_dcbf"):
            param.horizon_dcbf = 2
        param.mat_Rold = np.eye(2)
        state = SimpleNamespace(_x=np.zeros(3), _u=np.zeros(2))
        state.rotation = lambda: np.array([
            [np.cos(state._x[2]), -np.sin(state._x[2])],
            [np.sin(state._x[2]), np.cos(state._x[2])],
        ])
        state.translation = lambda: state._x[:2].reshape(2, 1)
        robot = RectangleRegion(-0.075, 0.075, -0.035, 0.035)
        system = SimpleNamespace(
            _state=state,
            _geometry=SimpleNamespace(equiv_rep=lambda: [robot]),
            _dynamics=SimpleNamespace(
                nominal_safe_controller=lambda x, *args: (x.copy(), np.zeros(2)),
            ),
        )

        def dynamics(x, u):
            return ca.vertcat(
                x[0] + 0.1 * u[0] * ca.cos(x[2]),
                x[1] + 0.1 * u[0] * ca.sin(x[2]),
                x[2] + 0.1 * u[1],
            )

        optimizer = optimizer_type({}, {}, dynamics)
        obstacles = [RectangleRegion(1.0, 2.0, -2.0, 2.0)]
        return optimizer, param, system, obstacles, np.zeros((20, 3))

    @staticmethod
    def distance_result(distance):
        return distance, np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0])

    @staticmethod
    def initial_value(optimizer, expression):
        return np.asarray(optimizer.opti.debug.value(expression, optimizer.opti.initial()))

    def test_reuse_refreshes_state_reference_and_previous_input(self):
        for optimizer_type, param_type in self.optimizer_types:
            with self.subTest(optimizer=optimizer_type.__name__):
                optimizer, param, system, obstacles, reference = self.make_case(optimizer_type, param_type)
                with patch.object(ReusableRegionDistanceQuery, "distance", return_value=self.distance_result(0.4)):
                    optimizer.setup(param, system, reference, obstacles)
                    first_opti = optimizer.opti
                    system._state._x = np.array([0.1, 0.2, 0.3])
                    system._state._u = np.array([0.3, 0.0])
                    reference[:, 0] = 0.4
                    optimizer.setup(param, system, reference, obstacles)

                self.assertIs(optimizer.opti, first_opti)
                self.assertEqual(optimizer.problem_build_count, 1)
                self.assertEqual(optimizer.active_obstacle_count, 1)
                self.assertEqual(optimizer.active_constraint_pair_count, 1)
                optimizer.opti.set_initial(optimizer.variables["x"], np.zeros((3, param.horizon + 1)))
                optimizer.opti.set_initial(optimizer.variables["u"], np.zeros((2, param.horizon)))
                residual = self.initial_value(optimizer, optimizer.opti.g[:3] - optimizer.opti.lbg[:3])
                np.testing.assert_allclose(residual.reshape(-1), -system._state._x)
                reference_cost = float(self.initial_value(optimizer, optimizer.costs["reference_trajectory_tracking"]))
                self.assertAlmostEqual(reference_cost, (param.horizon + param.terminal_weight) * param.mat_Q[0, 0] * 0.4**2)
                previous_input_cost = float(self.initial_value(optimizer, optimizer.costs["prev_input"]))
                self.assertAlmostEqual(previous_input_cost, 0.3**2)

    def test_cutoff_boundary_and_empty_problem_reuse(self):
        for optimizer_type, param_type in self.optimizer_types:
            with self.subTest(optimizer=optimizer_type.__name__):
                optimizer, param, system, obstacles, reference = self.make_case(optimizer_type, param_type)
                self.assertEqual(param.safe_dist, 0.5)
                with patch.object(ReusableRegionDistanceQuery, "distance", return_value=self.distance_result(0.5)):
                    optimizer.setup(param, system, reference, obstacles)
                included_opti = optimizer.opti
                self.assertEqual(optimizer.active_constraint_pair_count, 1)
                with patch.object(ReusableRegionDistanceQuery, "distance", return_value=self.distance_result(0.5001)):
                    optimizer.setup(param, system, reference, obstacles)
                    empty_opti = optimizer.opti
                    self.assertIsNot(empty_opti, included_opti)
                    self.assertEqual(optimizer.active_constraint_pair_count, 0)
                    system._state._x[0] = 0.1
                    optimizer.setup(param, system, reference, obstacles)
                self.assertIs(optimizer.opti, empty_opti)
                self.assertEqual(optimizer.problem_build_count, 2)

    def test_cutoff_can_be_disabled(self):
        for optimizer_type, param_type in self.optimizer_types:
            with self.subTest(optimizer=optimizer_type.__name__):
                optimizer, param, system, obstacles, reference = self.make_case(optimizer_type, param_type)
                param.use_obstacle_cutoff = False
                with patch.object(ReusableRegionDistanceQuery, "distance", return_value=self.distance_result(2.0)):
                    optimizer.setup(param, system, reference, obstacles)
                self.assertEqual(optimizer.active_obstacle_count, 1)

    def test_geometry_and_cost_changes_invalidate_problem(self):
        for optimizer_type, param_type in self.optimizer_types:
            with self.subTest(optimizer=optimizer_type.__name__):
                optimizer, param, system, obstacles, reference = self.make_case(optimizer_type, param_type)
                with patch.object(ReusableRegionDistanceQuery, "distance", return_value=self.distance_result(0.4)):
                    optimizer.setup(param, system, reference, obstacles)
                    original_opti = optimizer.opti
                    obstacles[0].left += 0.1
                    optimizer.setup(param, system, reference, obstacles)
                    moved_opti = optimizer.opti
                    self.assertIsNot(moved_opti, original_opti)
                    param.mat_Q[0, 0] += 1.0
                    optimizer.setup(param, system, reference, obstacles)
                self.assertIsNot(optimizer.opti, moved_opti)
                self.assertEqual(optimizer.problem_build_count, 3)

    def test_dcbf_current_distance_updates_without_rebuilding(self):
        optimizer, param, system, obstacles, reference = self.make_case(NmpcDbcfOptimizer, NmpcDcbfOptimizerParam)
        residuals = []
        for distance in (0.2, 0.3):
            with patch.object(ReusableRegionDistanceQuery, "distance", return_value=self.distance_result(distance)):
                optimizer.setup(param, system, reference, obstacles)
            residuals.append(self.initial_value(optimizer, optimizer.opti.g - optimizer.opti.lbg).reshape(-1))
        self.assertEqual(optimizer.problem_build_count, 1)
        finite = np.isfinite(residuals[0]) & np.isfinite(residuals[1])
        self.assertGreater(np.max(np.abs(residuals[1][finite] - residuals[0][finite])), 1e-6)


class ReusableDistanceQueryTest(unittest.TestCase):
    def test_reuses_shape_and_refreshes_geometry_without_a_solver(self):
        query = ReusableRegionDistanceQuery()
        mat_a, vec_b = RectangleRegion(-1.0, 1.0, -1.0, 1.0).get_convex_rep()

        def fake_solve(opti):
            return SimpleNamespace(value=lambda expression: opti.debug.value(expression, opti.initial()))

        with patch.object(ca.Opti, "solve", autospec=True, side_effect=fake_solve):
            distance, lamb, mu = query.distance(mat_a, vec_b, mat_a, vec_b)
            self.assertEqual(distance, 0.0)
            np.testing.assert_array_equal(lamb, np.zeros(4))
            np.testing.assert_array_equal(mu, np.zeros(4))
            query.distance(mat_a, vec_b + 1.0, mat_a, vec_b)
            self.assertEqual(query.build_count, 1)
            problem = next(iter(query._problems.values()))
            actual_b = problem.opti.debug.value(problem.parameters[1], problem.opti.initial())
            np.testing.assert_allclose(np.asarray(actual_b).reshape(-1), (vec_b + 1.0).reshape(-1))
            query.distance(np.vstack([mat_a, mat_a[:1]]), np.vstack([vec_b, vec_b[:1]]), mat_a, vec_b)
            self.assertEqual(query.build_count, 2)

    def test_invalid_cutoff_is_rejected(self):
        for cutoff in (-0.1, np.nan, np.inf, None):
            with self.subTest(cutoff=cutoff):
                with self.assertRaises(ValueError):
                    validate_cutoff(SimpleNamespace(safe_dist=cutoff, use_obstacle_cutoff=True))
        self.assertIsNone(validate_cutoff(SimpleNamespace(use_obstacle_cutoff=False)))


if __name__ == "__main__":
    unittest.main()
