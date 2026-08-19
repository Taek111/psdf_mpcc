import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from utils.acados_diagnostics import (
    AcadosDiagnosticsMixin,
    build_box_diagnostics,
    collect_solver_diagnostics,
    find_box_violations,
    find_stage_violations,
    read_stage_trajectory,
    report_solver_failure,
)
from utils.rmpcc_diagnostics import RMPCCDiagnosticsMixin


class _FakeAcadosSolver:
    def get(self, stage, field):
        if field == "x":
            return np.array([stage, stage + 0.5])
        raise KeyError(field)

    @staticmethod
    def get_residuals(recompute=False):
        del recompute
        return np.array([0.1, 0.2, 0.3, 0.4])

    @staticmethod
    def get_stats(name):
        values = {
            "qp_stat": [0],
            "qp_iter": [4],
            "sqp_iter": 2,
            "statistics": np.arange(12).reshape(3, 4),
        }
        return values[name]


class _NoRecomputeSolver(_FakeAcadosSolver):
    @staticmethod
    def get_residuals(recompute=None):
        if recompute is not None:
            raise RuntimeError("recompute is unavailable")
        return np.array([1.0, 2.0, 3.0, 4.0])


class _Uncopyable:
    def __deepcopy__(self, memo):
        del memo
        raise RuntimeError("cannot copy")


class _DiagnosticsConsumer(AcadosDiagnosticsMixin):
    N = 0

    def __init__(self):
        self._last_constraint_diagnostics = None
        self._last_failed_constraint_diagnostics = None

    @staticmethod
    def _build_constraint_diagnostics(*args):
        del args
        return {"value": _Uncopyable()}

    @staticmethod
    def _print_constraint_diagnostics(diagnostics):
        del diagnostics

    @staticmethod
    def _print_infeasibility_diagnostics(diagnostics):
        del diagnostics


class AcadosDiagnosticsUtilityTest(unittest.TestCase):
    def setUp(self):
        self.solver = _FakeAcadosSolver()

    def test_reads_stage_trajectory(self):
        values = read_stage_trajectory(self.solver, "x", range(3))
        np.testing.assert_allclose(values, [[0, 0.5], [1, 1.5], [2, 2.5]])

    def test_rmpcc_diagnostics_extend_common_lifecycle(self):
        self.assertTrue(issubclass(RMPCCDiagnosticsMixin, AcadosDiagnosticsMixin))
        self.assertIn(
            "_collect_solver_diagnostics",
            RMPCCDiagnosticsMixin.__dict__,
        )

    def test_collects_available_solver_statistics(self):
        diagnostics = collect_solver_diagnostics(
            self.solver,
            2,
            mode="fast",
            backup_status=1,
        )
        self.assertEqual(diagnostics["status"], 2)
        self.assertEqual(diagnostics["mode"], "fast")
        self.assertEqual(diagnostics["backup_status"], 1)
        np.testing.assert_allclose(diagnostics["nlp_residuals"], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(diagnostics["qp_iterations"], [4])

    def test_residual_collection_falls_back_when_recompute_fails(self):
        diagnostics = collect_solver_diagnostics(_NoRecomputeSolver(), 0)
        np.testing.assert_allclose(diagnostics["nlp_residuals"], [1, 2, 3, 4])

    def test_builds_and_classifies_box_residuals(self):
        bounds = build_box_diagnostics(
            "input",
            [[-0.1, 0.5], [0.2, 1.2]],
            [0.0, 0.0],
            [1.0, 1.0],
            ["u0", "u1"],
            [0, 1],
        )
        violation = find_box_violations(bounds, "input", 1e-9)
        self.assertEqual(violation["constraint"], "input_bounds")
        self.assertEqual(violation["stage_fields"], [(0, "u0"), (1, "u1")])

    def test_finds_negative_stage_residuals(self):
        violation = find_stage_violations(
            "row",
            [0.0, -0.2, 0.1],
            1e-6,
            stages=[2, 4, 6],
        )
        self.assertEqual(violation["constraint"], "row")
        np.testing.assert_array_equal(violation["stages"], [4])
        self.assertIsNone(find_stage_violations("row", [0.0, 0.1], 1e-6))
        with self.assertRaises(ValueError):
            find_stage_violations("row", [0.0, -0.1], 1e-6, stages=[1])

    def test_common_failure_report_is_non_throwing(self):
        output = io.StringIO()
        with redirect_stdout(output):
            diagnostics = report_solver_failure(
                self.solver,
                2,
                max_iteration_message="iteration limit",
            )
        self.assertIn("iteration limit", output.getvalue())
        self.assertEqual(diagnostics["status"], 2)

    def test_failed_snapshot_cache_error_does_not_escape(self):
        consumer = _DiagnosticsConsumer()
        output = io.StringIO()
        with redirect_stdout(output):
            diagnostics = consumer.compute_diagnostics(
                self.solver,
                1,
                "failed",
                failed=True,
            )
        self.assertIn("could not cache failed diagnostics", output.getvalue())
        self.assertIsInstance(diagnostics["value"], _Uncopyable)
        self.assertIsNone(consumer._last_failed_constraint_diagnostics)

    def test_reporting_hook_error_does_not_escape(self):
        consumer = _DiagnosticsConsumer()

        def fail_to_report(_):
            raise RuntimeError("formatting failed")

        consumer._print_infeasibility_diagnostics = fail_to_report
        output = io.StringIO()
        with redirect_stdout(output):
            result = consumer.report_diagnostics(
                {"value": 1},
                failed=True,
                include_constraint=False,
            )

        self.assertEqual(result, {"value": 1})
        self.assertIn("could not report failed diagnostics", output.getvalue())


if __name__ == "__main__":
    unittest.main()
