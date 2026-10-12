"""SIP collision geometry and ellipsoid propagation mathematical oracles.

These tests solve small convex distance problems only. They do not construct an
acados solver or run navigation trials.
"""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import casadi as ca
import numpy as np

from control.dilated_polygon_robust_optimizer import (
    DilatedPolygonRobustController,
    DilatedPolygonRobustOptimizerParam,
)
from control.sipoc_geometry import (
    ConvexPolygon,
    polygon_ellipse_distance,
    propagate_covariance,
    psd_matrix,
    psd_sqrt,
    rotation,
)


def _rectangle(left, right, bottom, top):
    return ConvexPolygon.from_vertices(
        [[left, bottom], [right, bottom], [right, top], [left, top]]
    )


def _anisotropic_fixture():
    """Construct an exact optimum from independent convex support conditions."""
    body = _rectangle(-1.0, 1.0, -1.0, 1.0)
    normal = np.array([-0.6, -0.8])
    shape = np.array([[0.16, 0.075], [0.075, 0.09]])
    gamma = np.ones(2)
    delta = -(shape @ normal) / np.sqrt(normal @ shape @ normal)
    closest_obstacle = gamma + delta - normal  # Clearance is exactly one.
    obstacle = ConvexPolygon.from_vertices(
        closest_obstacle + np.array([[0, 0], [0.2, 0], [0.2, 0.2], [0, 0.2]])
    )
    return body, obstacle, shape, normal, gamma, delta, closest_obstacle


class SIPConvexGeometryTest(unittest.TestCase):
    def test_unordered_polygon_hull_contains_input_and_removes_interior_points(self):
        points = np.array([[1, 1], [-1, -1], [0, 0], [1, -1], [-1, 1], [1, 1]])
        body = ConvexPolygon.from_vertices(points)
        self.assertEqual(body.vertices.shape, (4, 2))
        self.assertLessEqual(np.max(body.A @ points.T - body.b[:, None]), 1e-10)
        self.assertAlmostEqual(body.radius, np.sqrt(2.0), places=10)

    def test_scaled_redundant_halfspaces_preserve_rectangle(self):
        # 2x <= 2, -3x <= 3, 4y <= 8, -5y <= 10, x+y <= 10.
        polygon = ConvexPolygon.from_halfspaces(
            np.array([[2, 0], [-3, 0], [0, 4], [0, -5], [1, 1]]),
            np.array([2, 3, 8, 10, 10]),
        )
        np.testing.assert_allclose(polygon.vertices.min(axis=0), [-1, -2], atol=1e-9)
        np.testing.assert_allclose(polygon.vertices.max(axis=0), [1, 2], atol=1e-9)

    def test_degenerate_polygon_is_rejected(self):
        with self.assertRaises(ValueError):
            ConvexPolygon.from_vertices([[0, 0], [1, 0], [2, 0]])

    def test_zero_uncertainty_rectangle_distance(self):
        body = _rectangle(-1, 1, -1, 1)
        obstacle = _rectangle(3, 4, -0.5, 0.5)
        witness = polygon_ellipse_distance(np.zeros(2), 0.0, np.zeros((2, 2)), body, obstacle)
        self.assertAlmostEqual(witness.distance, 2.0, places=6)
        self.assertAlmostEqual(witness.support_gap, 2.0, places=6)
        np.testing.assert_allclose(witness.normal, [-1.0, 0.0], atol=2e-6)
        np.testing.assert_allclose(witness.delta, 0.0, atol=1e-10)
        self.assertFalse(witness.overlapping)

    def test_isotropic_ellipse_reduces_clearance_by_its_radius(self):
        body = _rectangle(-1, 1, -1, 1)
        obstacle = _rectangle(3, 4, -0.5, 0.5)
        witness = polygon_ellipse_distance(np.zeros(2), 0.0, 0.04 * np.eye(2), body, obstacle)
        self.assertAlmostEqual(witness.distance, 1.8, places=6)
        self.assertAlmostEqual(witness.support_gap, 1.8, places=6)
        self.assertTrue(np.isfinite(witness.delta).all())

    def test_singular_ellipse_preserves_uncertainty_in_nonzero_axis(self):
        body = _rectangle(-1, 1, -1, 1)
        obstacle = _rectangle(3, 4, -0.5, 0.5)
        for shape, expected in ((np.diag([0.09, 0.0]), 1.7), (np.diag([0.0, 0.64]), 2.0)):
            with self.subTest(shape=shape):
                witness = polygon_ellipse_distance(np.zeros(2), 0.0, shape, body, obstacle)
                self.assertAlmostEqual(witness.distance, expected, places=6)
                self.assertAlmostEqual(witness.support_gap, expected, places=6)

    def test_rotated_rectangle_against_wall_matches_support_extent(self):
        body = _rectangle(-1.0, 1.0, -0.25, 0.25)
        obstacle = _rectangle(3, 4, -10, 10)
        heading = np.pi / 3
        witness = polygon_ellipse_distance(np.zeros(2), heading, np.zeros((2, 2)), body, obstacle)
        expected = 3.0 - abs(np.cos(heading)) - 0.25 * abs(np.sin(heading))
        self.assertAlmostEqual(witness.distance, expected, places=6)

    def test_anisotropic_witness_matches_known_convex_support_optimum(self):
        body, obstacle, shape, normal, gamma, delta, closest = _anisotropic_fixture()
        witness = polygon_ellipse_distance(np.zeros(2), 0.0, shape, body, obstacle)
        self.assertAlmostEqual(witness.distance, 1.0, places=6)
        self.assertAlmostEqual(witness.support_gap, 1.0, places=6)
        np.testing.assert_allclose(witness.normal, normal, atol=2e-6)
        np.testing.assert_allclose(witness.gamma, gamma, atol=2e-6)
        np.testing.assert_allclose(witness.delta, delta, atol=2e-6)
        np.testing.assert_allclose(witness.obstacle_point, closest, atol=2e-6)
        # Ellipsoid displacement changes the normal for this anisotropic case.
        nominal_residual = witness.gamma - witness.obstacle_point
        nominal_normal = nominal_residual / np.linalg.norm(nominal_residual)
        self.assertGreater(np.linalg.norm(nominal_normal - normal), 0.02)

    def test_envelope_gradient_matches_finite_difference_of_lower_level_value(self):
        body, obstacle, shape, _, _, _, _ = _anisotropic_fixture()
        witness = polygon_ellipse_distance(np.zeros(2), 0.0, shape, body, obstacle)
        rotation_derivative = np.array([[0.0, -1.0], [1.0, 0.0]])
        gradient = np.r_[witness.normal, witness.normal @ rotation_derivative @ witness.gamma]
        epsilon = 1e-4
        numerical_gradient = np.zeros(3)
        for index in range(3):
            perturbation = np.zeros(3)
            perturbation[index] = epsilon
            plus = polygon_ellipse_distance(perturbation[:2], perturbation[2], shape, body, obstacle)
            minus = polygon_ellipse_distance(-perturbation[:2], -perturbation[2], shape, body, obstacle)
            numerical_gradient[index] = (plus.distance - minus.distance) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical_gradient, atol=2e-5)

    def test_world_frame_rotation_and_translation_preserve_clearance(self):
        body, obstacle, shape, _, _, _, _ = _anisotropic_fixture()
        original = polygon_ellipse_distance(np.zeros(2), 0.0, shape, body, obstacle)
        heading = 0.73
        matrix = rotation(heading)
        translation = np.array([2.3, -4.7])
        transformed_obstacle = ConvexPolygon.from_vertices(obstacle.vertices @ matrix.T + translation)
        transformed = polygon_ellipse_distance(
            translation, heading, matrix @ shape @ matrix.T, body, transformed_obstacle
        )
        self.assertAlmostEqual(transformed.distance, original.distance, places=6)
        np.testing.assert_allclose(transformed.normal, matrix @ original.normal, atol=2e-6)
        np.testing.assert_allclose(transformed.delta, matrix @ original.delta, atol=2e-6)

    def test_large_world_origin_preserves_local_geometry_and_clearance(self):
        body, obstacle, shape, normal, _, delta, _ = _anisotropic_fixture()
        translation = np.array([1e6, -2e6])
        shifted_obstacle = ConvexPolygon.from_vertices(obstacle.vertices + translation)
        witness = polygon_ellipse_distance(translation, 0.0, shape, body, shifted_obstacle)
        self.assertAlmostEqual(witness.distance, 1.0, places=6)
        self.assertAlmostEqual(witness.support_gap, 1.0, places=6)
        self.assertFalse(witness.overlapping)
        np.testing.assert_allclose(witness.normal, normal, atol=2e-6)
        np.testing.assert_allclose(witness.delta, delta, atol=2e-6)

    def test_small_obstacle_entirely_inside_body_is_detected(self):
        body = _rectangle(-1, 1, -1, 1)
        obstacle = _rectangle(-0.05, 0.05, -0.05, 0.05)
        witness = polygon_ellipse_distance(np.zeros(2), 0.0, np.zeros((2, 2)), body, obstacle)
        self.assertLessEqual(witness.distance, 1e-7)
        self.assertTrue(witness.overlapping)
        self.assertLessEqual(witness.support_gap, 1e-7)
        self.assertTrue(np.isfinite(witness.normal).all())


class SIPEllipsoidPropagationTest(unittest.TestCase):
    def test_psd_square_root_handles_zero_isotropic_and_rank_one_shapes(self):
        rank_one_direction = np.array([0.3, -0.4])
        for matrix in (
            np.zeros((2, 2)),
            np.eye(2) * 0.04,
            np.outer(rank_one_direction, rank_one_direction),
        ):
            with self.subTest(matrix=matrix):
                square_root = psd_sqrt(matrix)
                self.assertTrue(np.isfinite(square_root).all())
                np.testing.assert_allclose(square_root @ square_root.T, matrix, atol=1e-12)

    def test_negative_uncertainty_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            psd_matrix(np.diag([0.01, -0.1]), dimension=2)

    def test_covariance_propagation_matches_independent_closed_form(self):
        initial = np.diag([0.04, 0.09, 0.01])
        process = np.diag([0.002, 0.005, 0.001])
        diagonal = np.array([0.8, 1.2, 0.9])
        horizon = 4
        matrices = np.repeat(np.diag(diagonal)[None, :, :], horizon, axis=0)
        inputs = np.zeros((horizon, 3, 2))
        trajectory = propagate_covariance(initial, matrices, inputs, process)
        self.assertEqual(np.shape(trajectory), (horizon + 1, 3, 3))
        for step in range(horizon + 1):
            expected_diagonal = diagonal ** (2 * step) * np.diag(initial)
            expected_diagonal += np.diag(process) * sum(diagonal ** (2 * j) for j in range(step))
            np.testing.assert_allclose(trajectory[step], np.diag(expected_diagonal), atol=1e-12)

    def test_feedback_uses_closed_loop_sensitivity(self):
        initial = np.diag([0.04, 0.09, 0.01])
        inputs = np.array([[[1, 0], [0, 1], [0, 0]]], dtype=float)
        feedback = np.array([[-0.5, 0, 0], [0, -0.25, 0]], dtype=float)
        trajectory = propagate_covariance(
            initial, np.eye(3)[None, :, :], inputs, np.zeros((3, 3)), feedback_gain=feedback
        )
        np.testing.assert_allclose(trajectory[1], np.diag([0.01, 0.050625, 0.01]), atol=1e-12)

    def test_nonuniform_steps_scale_additive_noise(self):
        process = np.diag([0.002, 0.003, 0.004])
        matrices = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
        trajectory = propagate_covariance(
            np.zeros((3, 3)), matrices, np.zeros((3, 3, 2)), process, step_scales=[1, 2, 0.5]
        )
        for step, scale in enumerate([0, 1, 3, 3.5]):
            np.testing.assert_allclose(trajectory[step], scale * process, atol=1e-12)


class _RecordingUpperSolver:
    """Prescribed upper iterates; lower geometry and rollout stay real."""

    def __init__(self, horizon, input_batches, statuses):
        self.horizon = horizon
        self.input_batches = input_batches
        self.statuses = statuses
        self.solve_calls = 0
        self.values = {}
        self.constraints = {}
        self.inputs = np.zeros((horizon, 2))

    def set(self, stage, field, value):
        self.values[stage, field] = np.asarray(value).copy()

    def constraints_set(self, stage, field, value, **kwargs):
        self.constraints[stage, field] = np.asarray(value).copy()

    def solve(self):
        index = self.solve_calls
        self.solve_calls += 1
        self.inputs = np.asarray(self.input_batches[min(index, len(self.input_batches) - 1)]).copy()
        return self.statuses[min(index, len(self.statuses) - 1)]

    def get(self, stage, field):
        if field != "u":
            raise AssertionError("The accepted state must come from the exact rollout")
        return self.inputs[stage].copy()


class _ControllerWithoutAcadosBuild(DilatedPolygonRobustController):
    def __init__(self, input_batches, statuses=(0,), dynamics_opt=None):
        super().__init__(dynamics_opt=dynamics_opt)
        self.input_batches = input_batches
        self.statuses = statuses

    def _build_solver(self):
        self.solver = _RecordingUpperSolver(self.N, self.input_batches, self.statuses)
        self.problem_build_count += 1


class SIPControllerContractTest(unittest.TestCase):
    def _build(self, inputs=None, obstacles=(), pose=(0, 0, 0), param=None,
               statuses=(0,), dynamics_opt=None):
        if param is None:
            param = DilatedPolygonRobustOptimizerParam()
            param.horizon, param.tf = 2, 0.2
            param.process_noise = np.zeros((3, 3))
            param.initial_covariance = np.zeros((3, 3))
            param.max_outer_iterations = 3
            # Prescribed full-step iterates keep the existing contract fixtures
            # independent of damping; dedicated tests below exercise alpha=0.5.
            param.outer_step_length = 1.0
        if inputs is None:
            inputs = [np.zeros((param.horizon, 2))]
        body = _rectangle(-0.1, 0.1, -0.05, 0.05)
        system = SimpleNamespace(
            _state=SimpleNamespace(_x=np.asarray(pose, dtype=float), _u=np.zeros(2)),
            _dt=param.tf / param.horizon,
            _geometry=SimpleNamespace(equiv_rep=lambda: [body]),
        )
        reference = np.repeat(np.asarray(pose, dtype=float)[None, :], param.horizon + 1, axis=0)
        optimizer = _ControllerWithoutAcadosBuild(inputs, statuses, dynamics_opt=dynamics_opt)
        optimizer.setup(param, system, reference, list(obstacles))
        return optimizer, param, system, reference

    def test_linearized_row_includes_translation_and_angular_backoffs(self):
        body, obstacle, shape, normal, _, _, _ = _anisotropic_fixture()
        heading = 0.0
        witness = polygon_ellipse_distance(np.zeros(2), heading, shape, body, obstacle)
        radius, theta_radius = 0.07, 0.03
        backoff = body.radius * theta_radius
        c, lower = DilatedPolygonRobustController.linearized_clearance_row(
            np.zeros(3), witness, radius, backoff
        )
        np.testing.assert_allclose(c[:2], normal, atol=2e-6)
        self.assertAlmostEqual(-lower, 1.0 - radius - backoff, places=6)
        self.assertAlmostEqual(c[2], 0.6 - 0.8, places=6)

    def test_supplied_euler_map_jacobian_propagates_heading_into_position(self):
        x, u = ca.SX.sym("x", 3), ca.SX.sym("u", 2)
        dynamics = ca.Function("test_dd_euler", [x, u], [
            x + 0.1 * ca.vertcat(u[0] * ca.cos(x[2]), u[0] * ca.sin(x[2]), u[1])
        ])
        optimizer, param, _, _ = self._build(pose=(0, 0, np.pi / 2), dynamics_opt=dynamics)
        param.initial_covariance = np.diag([0.04, 0.09, 0.01])
        inputs = np.array([[0.5, 0.2], [0.5, 0.2]])
        states = optimizer._rollout(inputs)
        covariances = optimizer._propagate_covariances(states, inputs)
        expected_first = np.array([[0.040025, 0, -0.0005], [0, 0.09, 0], [-0.0005, 0, 0.01]])
        np.testing.assert_allclose(covariances[1], expected_first, atol=1e-12)

    def test_geometry_verifies_first_and_terminal_nodes(self):
        obstacle = _rectangle(0.3, 0.4, -0.2, 0.2)
        optimizer, _, _, _ = self._build(obstacles=[obstacle])
        covariances = np.zeros((3, 3, 3))
        for node in (1, 2):
            with self.subTest(node=node):
                states = np.zeros((3, 3))
                states[node, 0] = 0.25
                geometry = optimizer._geometry_at_nodes(states, covariances, include_initial=True)
                self.assertLess(optimizer._minimum_clearance(geometry), 0.0)
                self.assertLess(geometry[node][0][0], 0.0)

    def test_obstacle_distant_from_current_pose_is_checked_at_future_nodes(self):
        optimizer, _, _, _ = self._build(obstacles=[_rectangle(2.0, 2.1, -0.2, 0.2)])
        states = np.array([[0, 0, 0], [0, 0, 0], [1.95, 0, 0]], dtype=float)
        geometry = optimizer._geometry_at_nodes(states, np.zeros((3, 3, 3)), include_initial=True)
        self.assertEqual(geometry[0], [])
        self.assertLess(geometry[-1][0][0], 0.0)

    def test_terminal_collision_never_applies_successful_upper_iterate(self):
        optimizer, _, _, _ = self._build(
            inputs=[np.array([[0.5, 0], [0.5, 0]])],
            obstacles=[_rectangle(0.19, 0.3, -0.2, 0.2)],
        )
        solution = optimizer.solve_nlp()
        self.assertFalse(solution.stats()["success"])
        self.assertGreater(solution.solver.status, 0)
        np.testing.assert_array_equal(solution.value("u"), np.zeros((2, 2)))
        self.assertLess(solution.stats()["minimum_candidate_clearance"], 0.0)

    def test_initial_uncertainty_overlap_returns_failed_zero_velocity(self):
        optimizer, param, _, _ = self._build(obstacles=[_rectangle(0.12, 0.2, -0.2, 0.2)])
        # The nominal 20 mm clearance is smaller than the 30 mm uncertainty.
        param.initial_covariance = np.diag([0.03 ** 2, 0.0, 0.0])
        solution = optimizer.solve_nlp()
        self.assertEqual(solution.stats()["return_status"], "initial_robust_clearance_violation")
        self.assertFalse(solution.stats()["success"])
        self.assertEqual(optimizer.solver.solve_calls, 0)
        np.testing.assert_array_equal(solution.value("u"), np.zeros((2, 2)))

    def test_constraint_capacity_overflow_fails_without_silent_obstacle_drop(self):
        obstacles = [_rectangle(0.2, 0.3, -0.3, -0.1), _rectangle(0.2, 0.3, 0.1, 0.3)]
        optimizer, param, _, _ = self._build(obstacles=obstacles)
        param.max_constraints = 1
        solution = optimizer.solve_nlp()
        self.assertFalse(solution.stats()["success"])
        self.assertEqual(solution.stats()["return_status"], "constraint_capacity_exceeded")
        self.assertEqual(optimizer.solver.solve_calls, 0)

    def test_parallel_face_enforces_both_angular_support_directions(self):
        # The true clearance has a cusp: d(theta)=0.2-0.05|theta|+O(theta^2).
        # A midpoint witness has zero yaw derivative and misses both sides.
        optimizer, _, _, _ = self._build(obstacles=[_rectangle(0.3, 0.4, -0.2, 0.2)])
        self.assertTrue(optimizer.solve_nlp().stats()["success"])
        for stage in (1, 2):
            rows = optimizer.solver.constraints[stage, "C"]
            bounds = optimizer.solver.constraints[stage, "lg"]
            active = rows[bounds > -1e6]
            self.assertEqual(len(active), 2)
            np.testing.assert_allclose(active[:, :2], [[-1, 0], [-1, 0]], atol=2e-6)
            np.testing.assert_allclose(np.sort(active[:, 2]), [-0.05, 0.05], atol=2e-6)

    def test_vertex_rows_are_counted_in_capacity_even_for_one_obstacle(self):
        optimizer, param, _, _ = self._build(obstacles=[_rectangle(0.3, 0.4, -0.2, 0.2)])
        param.max_constraints = 1
        solution = optimizer.solve_nlp()
        self.assertFalse(solution.stats()["success"])
        self.assertEqual(solution.stats()["return_status"], "constraint_capacity_exceeded")
        self.assertEqual(optimizer.solver.solve_calls, 0)

    def test_upper_solver_failure_has_positive_complete_status_and_zero_velocity(self):
        optimizer, _, _, _ = self._build(statuses=(7,))
        solution = optimizer.solve_nlp()
        self.assertFalse(solution.stats()["success"])
        self.assertEqual(solution.stats()["upper_status"], 7)
        self.assertEqual(solution.stats()["return_status"], "acados_failure(7)")
        self.assertGreater(solution.solver.status, 0)
        np.testing.assert_array_equal(solution.value("u"), np.zeros((2, 2)))

    def test_sqp_maxiter_candidate_requires_feasible_outer_convergence(self):
        inputs = np.array([[0.4, 0], [0.4, 0]])
        optimizer, param, _, _ = self._build(
            inputs=[inputs], statuses=(2,),
            obstacles=[_rectangle(0.3, 0.4, -0.2, 0.2)],
        )
        solution = optimizer.solve_nlp()
        stats = solution.stats()
        self.assertTrue(stats["success"])
        self.assertEqual(solution.solver.status, 0)
        self.assertEqual(stats["upper_status"], 2)
        self.assertEqual(stats["outer_iterations"], 2)
        self.assertGreater(stats["iteration_history"][0]["input_step_inf"],
                           param.step_tolerance)
        self.assertLessEqual(stats["iteration_history"][-1]["input_step_inf"],
                             param.step_tolerance)
        self.assertGreaterEqual(stats["minimum_candidate_clearance"],
                                -param.feasibility_tolerance)
        np.testing.assert_allclose(solution.value("u"), inputs.T, atol=1e-12)
        np.testing.assert_allclose(solution.value("x")[0], [0.0, 0.04, 0.08], atol=1e-12)

    def test_converged_sqp_maxiter_candidate_with_collision_is_rejected(self):
        optimizer, param, _, _ = self._build(
            inputs=[np.array([[0.5, 0], [0.5, 0]])], statuses=(2,),
            obstacles=[_rectangle(0.19, 0.3, -0.2, 0.2)],
        )
        solution = optimizer.solve_nlp()
        stats = solution.stats()
        self.assertFalse(stats["success"])
        self.assertEqual(stats["return_status"], "maximum_iterations")
        self.assertEqual(stats["upper_status"], 2)
        self.assertEqual(stats["outer_iterations"], param.max_outer_iterations)
        # The last iterate is stationary, but its exact terminal rollout collides.
        self.assertEqual(stats["iteration_history"][-1]["state_step_inf"], 0.0)
        self.assertEqual(stats["iteration_history"][-1]["input_step_inf"], 0.0)
        self.assertLess(stats["minimum_candidate_clearance"],
                        -param.feasibility_tolerance)
        self.assertGreater(solution.solver.status, 0)
        np.testing.assert_array_equal(solution.value("u"), np.zeros((2, 2)))

    def test_feasible_sqp_maxiter_candidate_without_outer_convergence_is_rejected(self):
        inputs = [np.array([[speed, 0], [speed, 0]]) for speed in (0.2, 0.4, 0.2)]
        optimizer, param, _, _ = self._build(
            inputs=inputs, statuses=(2,),
            obstacles=[_rectangle(0.3, 0.4, -0.2, 0.2)],
        )
        solution = optimizer.solve_nlp()
        stats = solution.stats()
        self.assertFalse(stats["success"])
        self.assertEqual(stats["return_status"], "maximum_iterations")
        self.assertGreater(stats["minimum_candidate_clearance"], 0.0)
        for iteration in stats["iteration_history"]:
            self.assertGreater(iteration["input_step_inf"], param.step_tolerance)
        np.testing.assert_array_equal(solution.value("u"), np.zeros((2, 2)))

    def test_rti_maxiter_status_is_rejected_even_for_zero_feasible_step(self):
        param = DilatedPolygonRobustOptimizerParam(
            horizon=2, tf=0.2, max_outer_iterations=3,
            process_noise=np.zeros((3, 3)), nlp_solver_type="SQP_RTI",
        )
        optimizer, _, _, _ = self._build(param=param, statuses=(2,))
        solution = optimizer.solve_nlp()
        self.assertFalse(solution.stats()["success"])
        self.assertEqual(solution.stats()["upper_status"], 2)
        self.assertEqual(solution.stats()["return_status"], "acados_failure(2)")
        self.assertEqual(optimizer.solver.solve_calls, 1)
        self.assertGreater(solution.solver.status, 0)
        np.testing.assert_array_equal(solution.value("u"), np.zeros((2, 2)))

    def test_sqp_maxiter_does_not_relax_numerical_clearance_tolerance(self):
        tolerance = DilatedPolygonRobustOptimizerParam().feasibility_tolerance
        inputs = np.array([[0.5, 0], [0.5, 0]])
        for expected_gap, expected_success in ((-0.5 * tolerance, True),
                                                (-5.0 * tolerance, False)):
            with self.subTest(expected_gap=expected_gap):
                # Exact terminal x=0.1, body support=0.1, dilation=0.001.
                wall_x = 0.201 + expected_gap
                optimizer, param, _, _ = self._build(
                    inputs=[inputs], statuses=(2,),
                    obstacles=[_rectangle(wall_x, 0.4, -0.2, 0.2)],
                )
                solution = optimizer.solve_nlp()
                self.assertEqual(param.feasibility_tolerance, tolerance)
                self.assertEqual(solution.stats()["success"], expected_success)
                self.assertAlmostEqual(solution.stats()["minimum_candidate_clearance"],
                                       expected_gap, places=7)
                if not expected_success:
                    np.testing.assert_array_equal(solution.value("u"), np.zeros((2, 2)))

    def test_outer_step_length_default_preserves_convergence_and_clearance_limits(self):
        param = DilatedPolygonRobustOptimizerParam()
        self.assertEqual(param.outer_step_length, 1.0)
        self.assertEqual(param.tol, 1e-8)
        self.assertEqual(param.step_tolerance, 1e-4)
        self.assertEqual(param.feasibility_tolerance, 1e-5)

    def test_outer_step_length_rejects_zero_nonfinite_and_out_of_range_values(self):
        for alpha in (0.0, -0.5, 1.000001, np.nan, np.inf, -np.inf):
            with self.subTest(alpha=alpha):
                param = DilatedPolygonRobustOptimizerParam(
                    horizon=2, tf=0.2, outer_step_length=alpha,
                )
                with self.assertRaisesRegex(ValueError, "outer_step_length"):
                    self._build(param=param)

    def test_damped_inputs_use_exact_dd_rollout_and_matching_covariance(self):
        proposed_inputs = np.array([[0.4, 0.6], [0.2, -0.4]])
        damped_inputs = proposed_inputs * 0.5
        param = DilatedPolygonRobustOptimizerParam(
            horizon=2, tf=0.2, max_outer_iterations=3, outer_step_length=0.5,
            process_noise=np.zeros((3, 3)), initial_covariance=np.diag([0, 0, 0.01]),
        )
        optimizer, _, _, _ = self._build(
            inputs=[proposed_inputs, damped_inputs], param=param, pose=(0.1, -0.2, 0.3),
        )
        solution = optimizer.solve_nlp()
        self.assertTrue(solution.stats()["success"])
        self.assertEqual(solution.stats()["outer_iterations"], 2)
        np.testing.assert_allclose(solution.value("u"), damped_inputs.T, atol=1e-12)
        first = np.array([0.1 + 0.02 * np.cos(0.3),
                          -0.2 + 0.02 * np.sin(0.3), 0.33])
        second = first + np.array([0.01 * np.cos(0.33),
                                   0.01 * np.sin(0.33), -0.02])
        expected_states = np.vstack([[0.1, -0.2, 0.3], first, second])
        np.testing.assert_allclose(solution.value("x"), expected_states.T, atol=1e-12)
        # Position/heading covariance comes from the damped speed 0.2, not 0.4.
        sensitivity = np.array([-0.02 * np.sin(0.3), 0.02 * np.cos(0.3), 1.0])
        expected_first_covariance = 0.01 * np.outer(sensitivity, sensitivity)
        np.testing.assert_allclose(optimizer.get_last_covariance_trajectory()[1],
                                   expected_first_covariance, atol=1e-12)

    def test_tiny_outer_step_cannot_make_unconverged_proposal_successful(self):
        param = DilatedPolygonRobustOptimizerParam(
            horizon=2, tf=0.2, max_outer_iterations=1, outer_step_length=1e-8,
            process_noise=np.zeros((3, 3)),
        )
        optimizer, _, _, _ = self._build(
            inputs=[np.array([[0.4, 0], [0.4, 0]])], param=param,
        )
        solution = optimizer.solve_nlp()
        stats = solution.stats()
        self.assertFalse(stats["success"])
        self.assertEqual(stats["return_status"], "maximum_iterations")
        residual = stats["iteration_history"][0]["input_step_inf"]
        self.assertAlmostEqual(residual, 0.4)
        self.assertGreater(residual, param.step_tolerance)
        self.assertLess(param.outer_step_length * residual, param.step_tolerance)
        np.testing.assert_array_equal(solution.value("u"), np.zeros((2, 2)))

    def test_lower_solver_exception_propagates_and_records_one_call_duration(self):
        optimizer, _, _, _ = self._build(obstacles=[_rectangle(0.3, 0.4, -0.2, 0.2)])
        with patch("control.dilated_polygon_robust_optimizer.polygon_ellipse_distance",
                   side_effect=RuntimeError("lower problem failed")):
            with self.assertRaisesRegex(RuntimeError, "lower problem failed"):
                optimizer.solve_nlp()
        self.assertEqual(len(optimizer.solver_times), 1)
        self.assertGreaterEqual(optimizer.solver_times[0], 0.0)

    def test_one_total_duration_per_controller_call_across_outer_iterations(self):
        optimizer, _, _, _ = self._build(inputs=[np.array([[0.4, 0], [0.4, 0]])])
        solution = optimizer.solve_nlp()
        self.assertTrue(solution.stats()["success"])
        self.assertEqual(optimizer.solver.solve_calls, 2)
        self.assertEqual(solution.stats()["outer_iterations"], 2)
        self.assertEqual(len(optimizer.solver_times), 1)
        self.assertGreaterEqual(optimizer.solver_times[0], solution.stats()["upper_solver_time_s"])
        # Exact plant states are recomputed from the accepted velocity inputs.
        np.testing.assert_allclose(solution.value("x")[0], [0.0, 0.04, 0.08], atol=1e-12)

    def test_completed_solution_keeps_status_and_values_after_later_failure(self):
        optimizer, _, _, _ = self._build()
        first_solution = optimizer.solve_nlp()
        first_states = first_solution.value("x")
        optimizer.solver.statuses = (7,)
        second_solution = optimizer.solve_nlp()
        self.assertTrue(first_solution.stats()["success"])
        self.assertEqual(first_solution.solver.status, 0)
        self.assertFalse(second_solution.stats()["success"])
        np.testing.assert_array_equal(first_solution.value("x"), first_states)

    def test_solver_is_reused_and_cleanup_drops_warm_start(self):
        optimizer, param, system, reference = self._build()
        solver = optimizer.solver
        optimizer.solve_nlp()
        optimizer.setup(param, system, reference, [])
        self.assertIs(optimizer.solver, solver)
        self.assertEqual(optimizer.problem_build_count, 1)
        self.assertIsNotNone(optimizer._previous_inputs)
        optimizer.cleanup()
        self.assertIsNone(optimizer.solver)
        self.assertIsNone(optimizer._previous_inputs)
        self.assertEqual(optimizer.solver_times, [])
        self.assertEqual(optimizer.problem_build_count, 0)
        optimizer.setup(param, system, reference, [])
        self.assertEqual(optimizer.problem_build_count, 1)

    def test_changed_sqp_options_rebuild_solver_without_codegen(self):
        for name, value in (("nlp_solver_type", "SQP_RTI"),
                            ("nlp_solver_max_iter", 5),
                            ("levenberg_marquardt", 2e-7)):
            with self.subTest(option=name):
                optimizer, param, system, reference = self._build()
                original_solver = optimizer.solver
                setattr(param, name, value)
                optimizer.setup(param, system, reference, [])
                self.assertIsNot(optimizer.solver, original_solver)
                self.assertEqual(optimizer.problem_build_count, 2)
                rebuilt_solver = optimizer.solver
                optimizer.setup(param, system, reference, [])
                self.assertIs(optimizer.solver, rebuilt_solver)
                self.assertEqual(optimizer.problem_build_count, 2)

    def test_list_valued_config_matrices_are_normalized_before_build(self):
        param = DilatedPolygonRobustOptimizerParam()
        param.horizon, param.tf = 2, 0.2
        param.mat_Q = [[20, 0, 0], [0, 20, 0], [0, 0, 0.5]]
        param.mat_R = [[5, 0], [0, 0.05]]
        param.initial_covariance = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        param.process_noise = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        optimizer, _, _, _ = self._build(param=param)
        self.assertIsInstance(optimizer.param.mat_Q, np.ndarray)
        self.assertTrue(optimizer.solve_nlp().stats()["success"])

    def test_nonzero_feedback_assumption_is_rejected_without_plant_feedback(self):
        param = DilatedPolygonRobustOptimizerParam()
        param.horizon, param.tf = 2, 0.2
        param.feedback_gain = np.array([[-0.1, 0, 0], [0, 0, 0]])
        with self.assertRaises(ValueError):
            self._build(param=param)

    def test_prediction_and_plant_timestep_mismatch_is_rejected(self):
        optimizer, param, system, reference = self._build()
        system._dt = 0.2
        with self.assertRaisesRegex(ValueError, "timestep"):
            optimizer.setup(param, system, reference, [])


@unittest.skipUnless(os.environ.get("SIPOC_RUN_ACADOS_SMOKE") == "1",
                     "Set SIPOC_RUN_ACADOS_SMOKE=1 to compile and check the real upper solver")
class SIPAcadosSmokeTest(unittest.TestCase):
    def test_nominal_and_robust_wall_clearance_with_reused_solver(self):
        param = DilatedPolygonRobustOptimizerParam(
            horizon=3, tf=0.3, max_constraints=6,
            process_noise=np.zeros((3, 3)),
        )
        body = _rectangle(-0.1, 0.1, -0.05, 0.05)
        obstacle = _rectangle(0.15, 1.0, -1.0, 1.0)
        system = SimpleNamespace(
            _state=SimpleNamespace(_x=np.zeros(3), _u=np.zeros(2)),
            _dt=0.1,
            _geometry=SimpleNamespace(equiv_rep=lambda: [body]),
        )
        reference = np.array([[0, 0, 0], [0.03, 0, 0], [0.06, 0, 0], [0.09, 0, 0]], dtype=float)
        optimizer = DilatedPolygonRobustController()
        workspace = None
        try:
            optimizer.setup(param, system, reference, [obstacle])
            workspace = Path(optimizer._workspace.name)
            self.assertTrue(workspace.is_dir())
            nominal = optimizer.solve_nlp()
            self.assertTrue(nominal.stats()["success"], nominal.stats())
            self.assertGreaterEqual(nominal.stats()["minimum_candidate_clearance"],
                                    -param.feasibility_tolerance)
            nominal_terminal_x = nominal.value("x")[0, -1]
            self.assertLessEqual(nominal_terminal_x, 0.04901)

            param.initial_covariance = np.diag([25e-6, 4e-6, 1e-4])
            param.process_noise = np.diag([1e-6, 1e-6, 1e-5])
            optimizer.setup(param, system, reference, [obstacle])
            robust = optimizer.solve_nlp()
            self.assertTrue(robust.stats()["success"], robust.stats())
            self.assertGreaterEqual(robust.stats()["minimum_candidate_clearance"],
                                    -param.feasibility_tolerance)
            robust_terminal_x = robust.value("x")[0, -1]
            self.assertLess(robust_terminal_x, 0.043)
            self.assertLess(robust_terminal_x, nominal_terminal_x)
            self.assertEqual(optimizer.problem_build_count, 1)
            self.assertEqual(len(optimizer.solver_times), 2)
        finally:
            optimizer.cleanup()
        self.assertIsNotNone(workspace)
        self.assertFalse(workspace.exists())


if __name__ == "__main__":
    unittest.main()
