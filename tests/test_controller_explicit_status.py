"""Controller status snapshots must survive skipped solves and solver resets."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from control.controller import BaseController


class ControllerExplicitStatusTest(unittest.TestCase):
    @staticmethod
    def make_controller(stats, native_status=0):
        solver = SimpleNamespace(status=native_status)
        optimizer = SimpleNamespace(solver=solver)
        controller = BaseController(optimizer, None)
        controller._opt_sol = SimpleNamespace(
            solver=solver, stats=lambda: stats.copy(),
        )
        return controller

    def test_skipped_safety_stop_does_not_inherit_native_success(self):
        controller = self.make_controller({
            "return_status": "safe_stop_stage0",
            "status_code": None,
            "success": False,
            "solver_called": False,
        })

        controller._update_last_solver_status_info()

        self.assertEqual(controller.get_last_solver_status_info(), {
            "raw_status": "safe_stop_stage0",
            "status_code": None,
            "success": False,
            "solver_called": False,
        })

    def test_solution_snapshot_survives_native_solver_status_change(self):
        for captured_status, later_status in ((4, 0), (0, 4)):
            with self.subTest(captured_status=captured_status):
                controller = self.make_controller({
                    "return_status": "failure" if captured_status else "success",
                    "status_code": captured_status,
                    "success": captured_status == 0,
                    "solver_called": True,
                }, native_status=later_status)

                controller._update_last_solver_status_info()
                result = controller.get_last_solver_status_info()

                self.assertEqual(result["status_code"], captured_status)
                self.assertEqual(result["success"], captured_status == 0)
                self.assertIs(result["solver_called"], True)

    def test_legacy_solution_retains_native_status_precedence(self):
        controller = self.make_controller({
            "return_status": "success", "success": True,
        }, native_status=4)

        controller._update_last_solver_status_info()

        self.assertEqual(controller.get_last_solver_status_info(), {
            "raw_status": "success", "status_code": 4, "success": False,
        })

    def test_legacy_solution_without_native_solver_uses_reported_success(self):
        controller = BaseController(SimpleNamespace(), None)
        controller._opt_sol = SimpleNamespace(stats=lambda: {
            "return_status": "custom_result", "success": True,
        })

        controller._update_last_solver_status_info()

        self.assertEqual(controller.get_last_solver_status_info(), {
            "raw_status": "custom_result", "status_code": None, "success": True,
        })

    def test_optional_solver_called_does_not_leak_to_later_legacy_result(self):
        controller = self.make_controller({
            "return_status": "safe_stop_stage0", "status_code": None,
            "success": False, "solver_called": False,
        })
        controller._update_last_solver_status_info()
        controller._opt_sol.stats = lambda: {"return_status": "success"}

        controller._update_last_solver_status_info()

        self.assertEqual(controller.get_last_solver_status_info(), {
            "raw_status": "success", "status_code": 0, "success": True,
        })

    def test_solve_exception_clears_previous_solution_status_and_metadata(self):
        controller = self.make_controller({
            "return_status": "success", "status_code": 0,
            "success": True, "solver_called": True,
        })
        controller._update_last_solver_status_info()
        controller._optimizer.setup = Mock()
        controller._optimizer.solve_nlp = Mock(side_effect=RuntimeError("failed"))

        with self.assertRaisesRegex(RuntimeError, "failed"):
            controller.generate_control_input(None, None, None, None)

        self.assertEqual(controller.get_last_solver_status_info(), {
            "raw_status": "RuntimeError: failed",
            "status_code": None,
            "success": False,
        })


if __name__ == "__main__":
    unittest.main()
