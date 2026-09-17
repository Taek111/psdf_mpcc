"""Check OBCA certificates against known geometry without solving an NLP."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import casadi as ca
import numpy as np

from control.obca_optimizer import OBCAOptimizer, OBCAOptimizerParam
from models.geometry_utils import RectangleRegion


class OBCAConstraintsTest(unittest.TestCase):
    def build_optimizer(self, horizon, heading=0.0):
        param = OBCAOptimizerParam()
        param.horizon = horizon
        param.margin_dist = 0.02
        param.use_obstacle_cutoff = False
        rotation = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)],
        ])
        state = SimpleNamespace(
            _x=np.array([-1.0, 0.0, heading]),
            rotation=lambda: rotation,
            translation=lambda: np.array([[-1.0], [0.0]]),
        )
        optimizer = OBCAOptimizer({}, {}, dynamics_opt=None)
        optimizer.opti = ca.Opti()
        optimizer.initialize_variables(param)
        optimizer.set_state(state)
        return optimizer, param

    def certificate_is_feasible(self, optimizer, poses):
        opti = optimizer.opti
        opti.set_initial(optimizer.variables["x"], poses)
        values = np.asarray(opti.debug.value(opti.g, opti.initial())).reshape(-1)
        lower = np.asarray(opti.debug.value(opti.lbg, opti.initial())).reshape(-1)
        upper = np.asarray(opti.debug.value(opti.ubg, opti.initial())).reshape(-1)
        self.assertGreater(values.size, 0, "Obstacle constraints must not be omitted")
        return bool(np.all(values >= lower - 1e-9) and np.all(values <= upper + 1e-9))

    def test_rectangle_clearance_at_every_predicted_state(self):
        robot = RectangleRegion(-0.075, 0.075, -0.045, 0.045)
        obstacle = RectangleRegion(1.0, 2.0, -2.0, 2.0)
        system = SimpleNamespace(_geometry=SimpleNamespace(equiv_rep=lambda: [robot]))

        for horizon in (1, 3):
            for heading in (0.0, np.pi / 4, np.pi / 2):
                with self.subTest(horizon=horizon, heading=heading):
                    optimizer, param = self.build_optimizer(horizon, heading)
                    cosine, sine = np.cos(heading), np.sin(heading)
                    half_extent_x = 0.075 * cosine + 0.045 * sine
                    # Exact horizontal separating normal for these rectangles.
                    # Disable selection to test the certificate from a far pose.
                    current_distance = 2.0 - half_extent_x
                    lamb = np.array([1.0, 0.0, 0.0, 0.0])
                    mu = np.array([0.0, sine, cosine, 0.0])
                    with patch(
                        "control.duality_optimizer_utils.ReusableRegionDistanceQuery.distance",
                        return_value=(current_distance, lamb, mu),
                    ):
                        optimizer.add_obstacle_avoidance_constraint(param, system, [obstacle])

                    poses = np.zeros((3, horizon + 1))
                    poses[2, :] = heading
                    poses[:, 0] = optimizer.state._x
                    poses[0, 1:] = 1.0 - half_extent_x - (param.margin_dist + 0.001)
                    # A trajectory may approach the margin from a far initial pose
                    # without requiring a DCBF decay-relaxation penalty.
                    self.assertTrue(self.certificate_is_feasible(optimizer, poses))
                    self.assertEqual(optimizer.costs, {})

                    for step in range(1, horizon + 1):
                        for clearance in (param.margin_dist - 0.001, -0.01):
                            unsafe_poses = poses.copy()
                            unsafe_poses[0, step] = 1.0 - half_extent_x - clearance
                            self.assertFalse(self.certificate_is_feasible(optimizer, unsafe_poses))

    def test_point_clearance_includes_terminal_state(self):
        optimizer, param = self.build_optimizer(horizon=3)
        obstacle = RectangleRegion(1.0, 2.0, -2.0, 2.0)
        with patch(
            "control.obca_optimizer.get_dist_point_to_region",
            return_value=(2.0, np.array([1.0, 0.0, 0.0, 0.0])),
        ):
            optimizer.add_point_to_convex_constraint(param, obstacle)

        poses = np.zeros((3, param.horizon + 1))
        poses[:, 0] = optimizer.state._x
        poses[0, 1:] = 1.0 - (param.margin_dist + 0.001)
        self.assertTrue(self.certificate_is_feasible(optimizer, poses))
        self.assertEqual(optimizer.costs, {})
        poses[0, -1] = 1.0 - (param.margin_dist - 0.001)
        self.assertFalse(self.certificate_is_feasible(optimizer, poses))


if __name__ == "__main__":
    unittest.main()
