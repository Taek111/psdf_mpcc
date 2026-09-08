import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from control.controller import BaseController
from control.minkowski_cbf_optimizer import (
    ConfigurationObstacle,
    MinkowskiCBFOptimizer,
    MinkowskiCBFOptimizerParam,
    _build_configuration_obstacle,
    _raw_signed_distance_from_co,
    _signed_distance_and_gradient,
)
from models.dd import DifferentialDriveDynamics
from models.geometry_utils import RectangleRegion
from sim.logger import ControllerLogger
from sim.simulation_mpc import simulation_mpc


def _rectangle_vertices(left, right, bottom, top):
    return np.array(
        [
            [left, bottom],
            [right, bottom],
            [right, top],
            [left, top],
        ],
        dtype=float,
    )


def _configuration_vertices(configuration_obstacle):
    """Read CO vertices without coupling tests to a specific container type."""
    names = ("vertices", "vertices_ccw", "co_vertices")
    for name in names:
        if hasattr(configuration_obstacle, name):
            vertices = getattr(configuration_obstacle, name)
            break
        if isinstance(configuration_obstacle, dict) and name in configuration_obstacle:
            vertices = configuration_obstacle[name]
            break
    else:
        if isinstance(configuration_obstacle, np.ndarray):
            vertices = configuration_obstacle
        elif isinstance(configuration_obstacle, (tuple, list)):
            candidates = []
            for value in configuration_obstacle:
                array = np.asarray(value)
                if array.ndim == 2 and array.shape[1] == 2 and array.shape[0] >= 3:
                    candidates.append(array)
            if not candidates:
                raise AssertionError("configuration obstacle does not expose Nx2 vertices")
            vertices = candidates[0]
        else:
            raise AssertionError("configuration obstacle does not expose vertices")

    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
        raise AssertionError(f"expected CO vertices with shape (N, 2), got {vertices.shape}")
    return vertices


def _set_first_existing(obj, names, value, required=True):
    for name in names:
        if hasattr(obj, name):
            setattr(obj, name, value)
            return name
    if required:
        raise AssertionError(f"none of the expected parameter names exist: {names}")
    return None


def _get_first_existing(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


class _StateStub:
    def __init__(self, pose):
        self._x = np.asarray(pose, dtype=float)
        self._u = np.zeros(2, dtype=float)

    def translation(self):
        return self._x[:2].reshape(2, 1)

    def rotation(self):
        theta = float(self._x[2])
        return np.array(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ],
            dtype=float,
        )


class _GeometryStub:
    def __init__(self, region):
        self._regions = (
            list(region) if isinstance(region, (list, tuple)) else [region]
        )
        self._geometries = [
            SimpleNamespace(_region=component) for component in self._regions
        ]

    def equiv_rep(self):
        return list(self._regions)


class _SystemStub:
    def __init__(self, pose, robot_region):
        self._state = _StateStub(pose)
        self._geometry = _GeometryStub(robot_region)
        self._dynamics = DifferentialDriveDynamics()
        self._dt = 0.1


class MinkowskiGeometryTest(unittest.TestCase):
    def setUp(self):
        self.square_vertices = _rectangle_vertices(-0.5, 0.5, -0.5, 0.5)
        self.param = MinkowskiCBFOptimizerParam()
        _set_first_existing(self.param, ("d_safe", "safe_distance"), 0.05)

    def test_configuration_obstacle_matches_axis_aligned_minkowski_sum(self):
        obstacle = RectangleRegion(1.0, 2.0, -1.0, 1.0)
        configuration_obstacle = _build_configuration_obstacle(
            self.square_vertices,
            obstacle,
            np.zeros(3),
            tol=1e-10,
        )
        vertices = _configuration_vertices(configuration_obstacle)

        np.testing.assert_allclose(vertices.min(axis=0), [0.5, -1.5], atol=1e-9)
        np.testing.assert_allclose(vertices.max(axis=0), [2.5, 1.5], atol=1e-9)
        self.assertTrue(np.isfinite(vertices).all())

    def test_configuration_hrep_is_normalized_minimal_and_order_invariant(self):
        robot_vertices = _rectangle_vertices(-0.5, 0.5, -0.2, 0.2)
        reordered_vertices = np.roll(robot_vertices[::-1], shift=1, axis=0)
        obstacle = RectangleRegion(1.2, 2.0, -0.8, 0.9)
        pose = np.array([0.1, -0.05, 0.37])

        original = _build_configuration_obstacle(
            robot_vertices, obstacle, pose, tol=1e-10
        )
        reordered = _build_configuration_obstacle(
            reordered_vertices, obstacle, pose, tol=1e-10
        )

        self.assertEqual(original.A.ndim, 2)
        self.assertEqual(original.A.shape[1], 2)
        self.assertEqual(original.b.shape, (original.A.shape[0],))
        self.assertEqual(original.vertices.shape[0], original.A.shape[0])
        np.testing.assert_allclose(
            np.linalg.norm(original.A, axis=1),
            np.ones(original.A.shape[0]),
            atol=1e-10,
        )
        self.assertLessEqual(
            float(np.max(original.A @ original.vertices.T - original.b[:, None])),
            1e-8,
        )

        for first_index in range(original.A.shape[0]):
            for second_index in range(first_index + 1, original.A.shape[0]):
                first = original.A[first_index]
                second = original.A[second_index]
                cross = abs(first[0] * second[1] - first[1] * second[0])
                same_direction = float(np.dot(first, second)) > 0.0
                self.assertFalse(cross <= 1e-8 and same_direction)

        original_vertices = np.asarray(
            sorted(map(tuple, np.round(original.vertices, decimals=9)))
        )
        reordered_vertices = np.asarray(
            sorted(map(tuple, np.round(reordered.vertices, decimals=9)))
        )
        np.testing.assert_allclose(original_vertices, reordered_vertices, atol=1e-9)

    def test_signed_distance_covers_separation_touch_and_penetration(self):
        cases = (
            ("separated", RectangleRegion(1.0, 2.0, -1.0, 1.0), 0.5),
            ("touching", RectangleRegion(0.5, 1.5, -1.0, 1.0), 0.0),
            ("penetrating", RectangleRegion(0.4, 1.4, -1.0, 1.0), -0.1),
        )

        for name, obstacle, expected_signed_distance in cases:
            with self.subTest(name=name):
                result = _signed_distance_and_gradient(
                    self.square_vertices,
                    obstacle,
                    np.zeros(3),
                    self.param,
                )
                self.assertAlmostEqual(result.signed_distance, expected_signed_distance, places=7)
                self.assertAlmostEqual(
                    result.h,
                    expected_signed_distance
                    - float(_get_first_existing(self.param, ("d_safe", "safe_distance"))),
                    places=7,
                )
                self.assertEqual(np.asarray(result.gradient).shape, (3,))
                self.assertTrue(np.isfinite(result.gradient).all())
                self.assertIsInstance(result.diagnostics, dict)

        separated = _signed_distance_and_gradient(
            self.square_vertices, cases[0][1], np.zeros(3), self.param
        )
        penetrating = _signed_distance_and_gradient(
            self.square_vertices, cases[2][1], np.zeros(3), self.param
        )
        self.assertIn(str(separated.branch).lower(), {"separated", "separation", "qp"})
        self.assertIn(
            str(penetrating.branch).lower(),
            {"penetrating", "penetration", "collision", "lp"},
        )
        np.testing.assert_allclose(
            np.asarray(penetrating.gradient)[:2],
            [-1.0, 0.0],
            atol=2e-5,
        )
        self.assertGreater(len(np.asarray(separated.active_indices).reshape(-1)), 0)
        self.assertGreater(len(np.asarray(penetrating.active_indices).reshape(-1)), 0)

    def test_penetration_near_tie_keeps_exact_argmin_facet_first(self):
        configuration_obstacle = ConfigurationObstacle(
            A=np.array(
                [[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]]
            ),
            b=np.array([1.0, 0.100001, 1.0, 0.1]),
            vertices=np.array(
                [[-1.0, -0.100001], [1.0, -0.100001], [1.0, 0.1], [-1.0, 0.1]]
            ),
        )
        raw = _raw_signed_distance_from_co(configuration_obstacle, self.param)

        self.assertEqual(raw.active_indices[0], 3)
        self.assertEqual(set(raw.active_indices), {1, 3})
        np.testing.assert_allclose(raw.critical_point, [0.0, 0.1], atol=1e-12)

    def test_smooth_gradient_matches_analytic_support_derivative(self):
        robot_vertices = _rectangle_vertices(-0.5, 0.5, -0.2, 0.2)
        obstacle = RectangleRegion(1.2, 2.0, -2.0, 2.0)
        theta = 0.27
        pose = np.array([0.0, 0.0, theta])

        result = _signed_distance_and_gradient(robot_vertices, obstacle, pose, self.param)

        expected_distance = 1.2 - (0.5 * math.cos(theta) + 0.2 * math.sin(theta))
        expected_theta_gradient = 0.5 * math.sin(theta) - 0.2 * math.cos(theta)
        self.assertAlmostEqual(result.signed_distance, expected_distance, places=6)
        np.testing.assert_allclose(result.gradient[:2], [-1.0, 0.0], atol=2e-5)
        self.assertAlmostEqual(result.gradient[2], expected_theta_gradient, delta=2e-4)
        self.assertFalse(bool(result.nonsmooth))

        eps = 1e-6
        plus = _signed_distance_and_gradient(
            robot_vertices,
            obstacle,
            pose + np.array([0.0, 0.0, eps]),
            self.param,
        ).signed_distance
        minus = _signed_distance_and_gradient(
            robot_vertices,
            obstacle,
            pose - np.array([0.0, 0.0, eps]),
            self.param,
        ).signed_distance
        self.assertAlmostEqual(result.gradient[2], (plus - minus) / (2.0 * eps), delta=5e-4)

    def test_projection_diagnostics_report_small_kkt_residuals(self):
        robot_vertices = _rectangle_vertices(-0.5, 0.5, -0.2, 0.2)
        obstacle = RectangleRegion(1.2, 2.0, -2.0, 2.0)
        result = _signed_distance_and_gradient(
            robot_vertices,
            obstacle,
            np.array([0.0, 0.0, 0.27]),
            self.param,
        )

        self.assertEqual(result.branch, "separated")
        self.assertIn("solved", str(result.diagnostics["qp_status"]).lower())
        residual_limit = max(
            50.0 * float(self.param.osqp_eps_abs),
            50.0 * float(self.param.osqp_eps_rel),
            1e-7,
        )
        self.assertLessEqual(result.diagnostics["primal_residual"], residual_limit)
        self.assertLessEqual(
            result.diagnostics["stationarity_residual"], residual_limit
        )
        self.assertLessEqual(
            result.diagnostics["complementarity_residual"], residual_limit
        )
        self.assertGreaterEqual(
            result.diagnostics["dual_min"],
            -10.0 * float(self.param.dual_tol),
        )

    def test_translation_gradient_matches_finite_difference_across_orientations(self):
        robot_vertices = _rectangle_vertices(-0.5, 0.5, -0.2, 0.2)
        cases = (
            (-0.55, RectangleRegion(1.2, 2.0, -2.0, 2.0)),
            (0.31, RectangleRegion(1.2, 2.0, -2.0, 2.0)),
            (-0.33, RectangleRegion(-2.0, 2.0, 1.2, 2.0)),
            (0.58, RectangleRegion(-2.0, 2.0, 1.2, 2.0)),
        )
        eps = 1e-6

        for theta, obstacle in cases:
            with self.subTest(theta=theta):
                pose = np.array([0.0, 0.0, theta])
                result = _signed_distance_and_gradient(
                    robot_vertices, obstacle, pose, self.param
                )
                finite_difference = np.zeros(2)
                for axis in range(2):
                    offset = np.zeros(3)
                    offset[axis] = eps
                    plus = _signed_distance_and_gradient(
                        robot_vertices, obstacle, pose + offset, self.param
                    ).signed_distance
                    minus = _signed_distance_and_gradient(
                        robot_vertices, obstacle, pose - offset, self.param
                    ).signed_distance
                    finite_difference[axis] = (plus - minus) / (2.0 * eps)

                self.assertFalse(bool(result.nonsmooth))
                np.testing.assert_allclose(
                    result.gradient[:2],
                    finite_difference,
                    atol=3e-5,
                    rtol=3e-5,
                )

    def test_parallel_facets_are_reported_without_nonfinite_values(self):
        robot_vertices = _rectangle_vertices(-0.5, 0.5, -0.2, 0.2)
        obstacle = RectangleRegion(1.2, 2.0, -2.0, 2.0)
        first = _signed_distance_and_gradient(
            robot_vertices, obstacle, np.zeros(3), self.param
        )
        second = _signed_distance_and_gradient(
            robot_vertices, obstacle, np.zeros(3), self.param
        )

        self.assertTrue(bool(first.nonsmooth))
        self.assertTrue(np.isfinite(first.gradient).all())
        self.assertTrue(np.isfinite(first.signed_distance))
        self.assertEqual(first.branch, second.branch)
        np.testing.assert_allclose(first.gradient, second.gradient, atol=1e-12)
        self.assertGreater(len(np.asarray(first.active_indices).reshape(-1)), 0)
        self.assertGreater(len(np.asarray(second.active_indices).reshape(-1)), 0)

    def test_ill_conditioned_kkt_uses_theta_only_fallback(self):
        robot_vertices = _rectangle_vertices(-0.5, 0.5, -0.2, 0.2)
        obstacle = RectangleRegion(1.2, 2.0, -2.0, 2.0)
        param = MinkowskiCBFOptimizerParam(kkt_condition_max=1.0)
        result = _signed_distance_and_gradient(
            robot_vertices,
            obstacle,
            np.array([0.0, 0.0, 0.27]),
            param,
        )

        self.assertTrue(result.nonsmooth)
        self.assertTrue(result.diagnostics["theta_scalar_fd_fallback"])
        np.testing.assert_allclose(result.gradient[:2], [-1.0, 0.0], atol=2e-5)


class MinkowskiOptimizerContractTest(unittest.TestCase):
    @staticmethod
    def _make_system():
        robot_region = RectangleRegion(-0.2, 0.2, -0.1, 0.1)
        return _SystemStub(np.zeros(3), robot_region), robot_region

    @staticmethod
    def _reference(target_x=0.8):
        reference = np.zeros((20, 3), dtype=float)
        reference[:, 0] = np.linspace(0.05, target_x, reference.shape[0])
        return reference

    def test_base_controller_and_solution_wrapper_contract(self):
        system, _ = self._make_system()
        obstacle = RectangleRegion(2.0, 2.5, -0.5, 0.5)
        param = MinkowskiCBFOptimizerParam()
        optimizer = MinkowskiCBFOptimizer()
        controller = BaseController(optimizer, param)

        control = np.asarray(
            controller.generate_control_input(
                system,
                global_path=None,
                local_trajectory=self._reference(),
                obstacles=[obstacle],
            ),
            dtype=float,
        )

        self.assertEqual(control.shape, (2,))
        self.assertTrue(np.isfinite(control).all())
        solution = controller._opt_sol
        self.assertTrue(hasattr(solution, "value"))
        self.assertTrue(hasattr(solution, "stats"))
        self.assertTrue(hasattr(solution, "get_state_trajectory"))
        self.assertTrue(hasattr(solution, "get_input_trajectory"))

        state_trajectory = np.asarray(solution.value("x"), dtype=float)
        input_trajectory = np.asarray(solution.value("u"), dtype=float)
        self.assertEqual(state_trajectory.shape, (3, 2))
        self.assertEqual(input_trajectory.shape, (2, 1))
        np.testing.assert_allclose(solution.get_state_trajectory(), state_trajectory)
        np.testing.assert_allclose(solution.get_input_trajectory(), input_trajectory)

        stats = solution.stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("return_status", stats)
        self.assertTrue(controller.get_last_solver_status_info()["success"])
        self.assertEqual(len(optimizer.solver_times), 1)
        self.assertTrue(np.isfinite(optimizer.solver_times[0]))
        self.assertGreaterEqual(optimizer.solver_times[0], 0.0)
        self.assertGreaterEqual(optimizer.last_diagnostics["clf_margin"], -5e-6)
        self.assertGreaterEqual(
            optimizer.last_diagnostics["minimum_bound_margin"],
            -5e-6,
        )

        logger = ControllerLogger()
        controller.logging(logger)
        self.assertEqual(len(logger._xtrajs), 1)
        self.assertEqual(len(logger._utrajs), 1)
        self.assertEqual(logger._xtrajs[0].shape, (2, 3))
        self.assertEqual(logger._utrajs[0].shape, (1, 2))

        vmin = float(_get_first_existing(param, ("vmin", "v_min"), -np.inf))
        vmax = float(_get_first_existing(param, ("vmax", "v_max"), np.inf))
        omegamin = float(_get_first_existing(param, ("omegamin", "omega_min"), -np.inf))
        omegamax = float(_get_first_existing(param, ("omegamax", "omega_max"), np.inf))
        self.assertGreaterEqual(control[0], vmin - 1e-8)
        self.assertLessEqual(control[0], vmax + 1e-8)
        self.assertGreaterEqual(control[1], omegamin - 1e-8)
        self.assertLessEqual(control[1], omegamax + 1e-8)

    def test_control_satisfies_repository_three_state_cbf_row(self):
        system, robot_region = self._make_system()
        obstacle = RectangleRegion(0.3, 0.7, -0.5, 0.5)
        param = MinkowskiCBFOptimizerParam()
        _set_first_existing(param, ("d_safe", "safe_distance"), 0.05)
        gamma_name = _set_first_existing(param, ("gamma", "cbf_gamma"), 0.1)
        epsilon_name = _set_first_existing(param, ("epsilon", "cbf_epsilon"), 0.0)
        _set_first_existing(
            param, ("use_cbf_slack", "soften_cbf"), False, required=False
        )

        optimizer = MinkowskiCBFOptimizer()
        optimizer.setup(param, system, self._reference(target_x=1.0), [obstacle])
        solution = optimizer.solve_nlp()
        control = np.asarray(solution.value("u"), dtype=float)[:, 0]

        sdf_result = _signed_distance_and_gradient(
            robot_region.get_ccw_vertices(),
            obstacle,
            system._state._x,
            param,
        )
        theta = float(system._state._x[2])
        state_derivative = np.array(
            [
                control[0] * math.cos(theta),
                control[0] * math.sin(theta),
                control[1],
            ]
        )
        gamma = float(getattr(param, gamma_name))
        epsilon = float(getattr(param, epsilon_name))
        cbf_left_hand_side = float(
            np.dot(sdf_result.gradient, state_derivative) + gamma * sdf_result.h
        )

        self.assertGreaterEqual(cbf_left_hand_side, epsilon - 5e-6)
        self.assertTrue(np.isfinite(control).all())

    def test_feasible_penetrating_state_moves_toward_recovery(self):
        system, robot_region = self._make_system()
        obstacle = RectangleRegion(0.15, 0.55, -0.5, 0.5)
        param = MinkowskiCBFOptimizerParam(d_safe=0.01, gamma=2.0, epsilon=0.0)
        optimizer = MinkowskiCBFOptimizer()
        reference = self._reference(target_x=1.0)

        initial_h = _signed_distance_and_gradient(
            robot_region.get_ccw_vertices(),
            obstacle,
            system._state._x,
            param,
        ).h
        h_values = [initial_h]
        for _ in range(5):
            optimizer.setup(param, system, reference, [obstacle])
            solution = optimizer.solve_nlp()
            self.assertGreaterEqual(
                optimizer.last_diagnostics["minimum_cbf_margin"],
                -5e-6,
            )
            system._state._x = solution.value("x")[:, -1]
            h_values.append(
                _signed_distance_and_gradient(
                    robot_region.get_ccw_vertices(),
                    obstacle,
                    system._state._x,
                    param,
                ).h
            )

        self.assertLess(initial_h, 0.0)
        self.assertGreater(h_values[-1], initial_h)
        self.assertTrue(np.all(np.diff(h_values) >= -1e-8))

    def test_no_obstacle_solve_and_reset_are_reusable(self):
        system, _ = self._make_system()
        param = MinkowskiCBFOptimizerParam()
        optimizer = MinkowskiCBFOptimizer()

        optimizer.setup(param, system, self._reference(), [])
        first_solution = optimizer.solve_nlp()
        self.assertTrue(np.isfinite(first_solution.value("u")).all())
        self.assertEqual(optimizer.last_diagnostics["pair_count"], 0)
        self.assertEqual(optimizer.get_last_cbf_results(), [])
        self.assertIsNone(optimizer.get_last_signed_distance())
        self.assertEqual(len(optimizer.solver_times), 1)

        optimizer.reset()
        self.assertEqual(optimizer.solver_times, [])
        self.assertEqual(optimizer.get_last_cbf_results(), [])
        self.assertIsNone(optimizer.get_last_signed_distance())
        with self.assertRaisesRegex(RuntimeError, r"setup\(\)"):
            optimizer.solve_nlp()

        optimizer.setup(param, system, self._reference(), [])
        second_solution = optimizer.solve_nlp()
        self.assertTrue(np.isfinite(second_solution.value("u")).all())
        self.assertEqual(len(optimizer.solver_times), 1)

    def test_bearing_clf_has_control_authority_for_lateral_goal(self):
        robot_region = RectangleRegion(-0.2, 0.2, -0.1, 0.1)
        system = _SystemStub(np.array([0.0, 1.0, 0.0]), robot_region)
        reference = np.zeros((2, 3), dtype=float)
        optimizer = MinkowskiCBFOptimizer()
        optimizer.setup(MinkowskiCBFOptimizerParam(), system, reference, [])
        solution = optimizer.solve_nlp()

        control = solution.value("u")[:, 0]
        self.assertEqual(optimizer.last_diagnostics["clf_heading_mode"], "bearing")
        self.assertGreater(
            np.linalg.norm(optimizer.last_diagnostics["clf_control_row"]),
            1e-6,
        )
        self.assertGreater(np.linalg.norm(control), 1e-6)
        self.assertGreaterEqual(optimizer.last_diagnostics["clf_margin"], -5e-6)

    def test_bearing_clf_gradient_matches_finite_difference(self):
        optimizer = MinkowskiCBFOptimizer()
        optimizer.param = MinkowskiCBFOptimizerParam()
        state = np.array([0.2, -0.4, 0.3])
        reference = np.array([1.0, 0.2, 1.1])
        value, gradient, _, mode = optimizer._clf_value_and_gradient(
            state,
            reference,
        )

        finite_difference = np.zeros(3)
        step = 1e-6
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = step
            value_plus = optimizer._clf_value_and_gradient(
                state + offset,
                reference,
            )[0]
            value_minus = optimizer._clf_value_and_gradient(
                state - offset,
                reference,
            )[0]
            finite_difference[axis] = (value_plus - value_minus) / (2.0 * step)

        self.assertGreater(value, 0.0)
        self.assertEqual(mode, "bearing")
        np.testing.assert_allclose(gradient, finite_difference, atol=2e-6)

    def test_unsupported_geometry_is_rejected_explicitly(self):
        system, _ = self._make_system()
        optimizer = MinkowskiCBFOptimizer()
        param = MinkowskiCBFOptimizerParam()

        system._geometry = _GeometryStub([None])
        with self.assertRaisesRegex(TypeError, "ConvexRegion2D"):
            optimizer.setup(param, system, self._reference(), [])

        system, _ = self._make_system()
        with self.assertRaisesRegex(TypeError, "ConvexRegion2D"):
            optimizer.setup(param, system, self._reference(), [None])

    def test_hard_cbf_infeasibility_is_propagated(self):
        system, _ = self._make_system()
        obstacle = RectangleRegion(-0.1, 0.3, -0.5, 0.5)
        param = MinkowskiCBFOptimizerParam()
        param.d_safe = 0.05
        param.epsilon = 1e-3
        param.vmin = 0.0
        param.vmax = 0.0
        param.omegamin = 0.0
        param.omegamax = 0.0
        optimizer = MinkowskiCBFOptimizer()
        optimizer.setup(param, system, self._reference(), [obstacle])

        with self.assertRaisesRegex(RuntimeError, r"control QP failed"):
            optimizer.solve_nlp()
        self.assertEqual(len(optimizer.solver_times), 1)

    def test_multiple_obstacle_component_pairs_are_order_independent(self):
        components = [
            RectangleRegion(-0.2, -0.02, -0.1, 0.1),
            RectangleRegion(0.02, 0.2, -0.1, 0.1),
        ]
        obstacles = [
            RectangleRegion(1.2, 1.7, -1.0, 1.0),
            RectangleRegion(-1.0, 1.0, 1.2, 1.7),
        ]

        def solve(component_order, obstacle_order):
            system = _SystemStub(np.array([0.0, 0.0, 0.23]), component_order)
            optimizer = MinkowskiCBFOptimizer()
            optimizer.setup(
                MinkowskiCBFOptimizerParam(),
                system,
                self._reference(),
                obstacle_order,
            )
            solution = optimizer.solve_nlp()
            distances = sorted(
                result.signed_distance for result in optimizer.get_last_cbf_results()
            )
            return (
                np.asarray(solution.value("u"), dtype=float),
                np.asarray(distances, dtype=float),
                optimizer.last_diagnostics,
            )

        forward_control, forward_distances, forward_diagnostics = solve(
            components, obstacles
        )
        reverse_control, reverse_distances, reverse_diagnostics = solve(
            list(reversed(components)),
            list(reversed(obstacles)),
        )

        self.assertEqual(forward_diagnostics["pair_count"], 4)
        self.assertEqual(reverse_diagnostics["pair_count"], 4)
        np.testing.assert_allclose(
            forward_distances, reverse_distances, atol=1e-8, rtol=1e-8
        )
        np.testing.assert_allclose(
            forward_control, reverse_control, atol=2e-6, rtol=2e-6
        )


class MinkowskiSimulationIntegrationTest(unittest.TestCase):
    def test_simulation_selects_optimizer_and_applies_config_for_one_tick(self):
        test_simulation = simulation_mpc()
        with patch.object(
            simulation_mpc,
            "save_trial_history_to_csv",
            autospec=True,
        ) as save_history:
            test_simulation.mpc_test(
                maze_type="s_path",
                robot_shape="rectangle",
                optimizer_type="minkowski_cbf",
                dynamics_type="differential_drive",
                path_planner="astar",
                simulation_time=0.05,
                config={"minkowski_cbf": {"gamma": 1.25}},
            )

        controller = test_simulation.robot._controller
        self.assertIsInstance(controller._optimizer, MinkowskiCBFOptimizer)
        self.assertIsInstance(controller._param, MinkowskiCBFOptimizerParam)
        self.assertAlmostEqual(controller._param.gamma, 1.25)
        self.assertEqual(len(controller._optimizer.solver_times), 1)
        self.assertEqual(controller._opt_sol.value("u").shape, (2, 1))
        self.assertEqual(controller._opt_sol.value("x").shape, (3, 2))
        self.assertAlmostEqual(test_simulation.robot._system._time, 0.1)
        self.assertEqual(test_simulation.last_run_outcome["status"], "timeout")
        save_history.assert_called_once()


if __name__ == "__main__":
    unittest.main()
