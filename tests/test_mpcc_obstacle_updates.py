"""Obstacle ownership without CUDA, solver compilation or navigation."""

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from control.cc_ecs_optimizer import CCECSOptimizer, CCECSOptimizerParam
from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
from control.cc_sf_optimizer import CCSFOptimizer, CCSFOptimizerParam
from control.mpcc_optimizer import MPCCOptimizer, MPCCOptimizerParam
from models.dd import DifferentialDriveRectangleGeometry
from models.dd_dynamics import DifferentialDriveDynamics
from models.geometry_utils import RectangleRegion


class MPCCObstacleUpdatesTest(unittest.TestCase):
    def test_first_and_later_setups_upload_once_and_keep_public_refresh(self):
        system = SimpleNamespace(
            _state=SimpleNamespace(_x=np.zeros(3), _u=np.zeros(2)),
            _geometry=SimpleNamespace(_geometries=[
                DifferentialDriveRectangleGeometry(
                    length=0.15, width=0.08, rear_dist=0.0,
                ),
            ]),
            _dynamics=DifferentialDriveDynamics,
        )
        obstacle = RectangleRegion(0.5, 0.6, -0.2, 0.2)
        moved_obstacle = RectangleRegion(0.7, 0.8, -0.2, 0.2)
        for optimizer_type, param_type in (
            (MPCCOptimizer, MPCCOptimizerParam),
            (CCSFOptimizer, CCSFOptimizerParam),
            (CCECSOptimizer, CCECSOptimizerParam),
            (CCMFOptimizer, CCMFOptimizerParam),
        ):
            with self.subTest(method=optimizer_type.__name__):
                optimizer = optimizer_type()
                self.addCleanup(optimizer.cleanup)
                param = param_type()
                param.horizon = 3
                param.tf = 0.3
                if hasattr(param, "robust_horizon"):
                    param.robust_horizon = param.horizon
                if hasattr(param, "enable_backup_solver"):
                    param.enable_backup_solver = True

                def fake_create_solver(*args, **kwargs):
                    solver = SimpleNamespace()
                    optimizer.solver = solver
                    return solver

                with (
                    # Keep real CPU geometry and symbolic OCP construction;
                    # replace only device policy, native build and warm start.
                    patch.object(optimizer, "_require_cuda_device", return_value="cpu"),
                    patch.object(optimizer, "create_solver", side_effect=fake_create_solver),
                    patch.object(optimizer, "add_warm_start"),
                    patch.object(
                        optimizer, "update_obstacles", wraps=optimizer.update_obstacles,
                    ) as update,
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertTrue(optimizer.updates_obstacles_in_setup)
                    for expected_count, snapshot in enumerate(
                        ([obstacle], [moved_obstacle], []), start=1,
                    ):
                        optimizer.setup(param, system, None, snapshot)
                        self.assertEqual(update.call_count, expected_count)
                        update.assert_called_with(snapshot)
                        self.assertEqual(
                            optimizer.psdf_wrapper.active_clusters, len(snapshot),
                        )
                    self.assertTrue(optimizer._is_initialized)
                    if optimizer_type is not MPCCOptimizer:
                        self.assertIsNotNone(optimizer.backup_solver)

                    # Direct users of the public constraint helper still get
                    # its original refresh unless they explicitly opt out.
                    arguments = (
                        (param, [obstacle])
                        if isinstance(optimizer, CCMFOptimizer)
                        else (param, system, [obstacle])
                    )
                    optimizer.add_obstacle_avoidance_constraint(*arguments)
                    self.assertEqual(update.call_count, 4)
                    self.assertEqual(optimizer.psdf_wrapper.active_clusters, 1)
                    optimizer.add_obstacle_avoidance_constraint(
                        *arguments, update_obstacle_data=False,
                    )
                    self.assertEqual(update.call_count, 4)


if __name__ == "__main__":
    unittest.main()
