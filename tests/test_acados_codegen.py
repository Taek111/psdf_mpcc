"""Build-path isolation regressions; solver generation and CUDA are mocked.

Prepared for an explicitly requested test run, without navigation trials.
"""

import importlib
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


OPTIMIZERS = (
    ("control.psdf_optimizer", "PSDFOptimizer", False),
    ("control.mpcc_optimizer", "MPCCOptimizer", False),
    ("control.cc_mf_optimizer", "CCMFOptimizer", True),
    ("control.cc_sf_optimizer", "CCSFOptimizer", True),
    # ECS inherits solver construction from CC-SF.
    ("control.cc_ecs_optimizer", "CCECSOptimizer", True),
)


def write_build_artifacts(ocp, *, json_file):
    directory = Path(ocp.code_export_directory)
    directory.mkdir(parents=True)
    (directory / "solver.so").write_bytes(b"mock solver")
    Path(json_file).write_text("{}", encoding="utf-8")
    return SimpleNamespace()


class AcadosCodegenTest(unittest.TestCase):
    def make_optimizer(self, module_name, class_name):
        optimizer_type = getattr(importlib.import_module(module_name), class_name)
        optimizer = optimizer_type()
        self.addCleanup(optimizer.cleanup)
        optimizer.ocp = SimpleNamespace(solver_options=SimpleNamespace())
        if class_name == "PSDFOptimizer":
            optimizer.param = SimpleNamespace(reuse_acados_codegen=False)
            signature = patch.object(optimizer, "_compute_build_signature", return_value="test")
            signature.start()
            self.addCleanup(signature.stop)
        return optimizer

    @staticmethod
    def solver_module(module_name):
        return "control.cc_sf_optimizer" if module_name.endswith("cc_ecs_optimizer") else module_name

    def test_concurrent_instances_keep_main_backup_and_metadata_isolated(self):
        for module_name, class_name, has_backup in OPTIMIZERS:
            with self.subTest(optimizer=class_name):
                first = self.make_optimizer(module_name, class_name)
                second = self.make_optimizer(module_name, class_name)
                with patch(
                    f"{self.solver_module(module_name)}.AcadosOcpSolver",
                    side_effect=write_build_artifacts,
                ):
                    for optimizer in (first, second):
                        solver = optimizer.create_solver()
                        if solver is not None:
                            optimizer.solver = solver
                        if has_backup:
                            optimizer.backup_solver = optimizer.create_solver(
                                json_filename=optimizer.backup_json_filename,
                                code_export_directory=optimizer.backup_code_export_directory,
                            )
                        Path(optimizer.codegen_meta_filename).write_text("{}", encoding="utf-8")
                first_root = Path(first.json_filename).parent
                second_root = Path(second.json_filename).parent
                self.assertTrue(first_root.is_absolute())
                self.assertNotEqual(first_root, second_root)
                second_files = {p.relative_to(second_root) for p in second_root.rglob("*")}
                first.cleanup()
                first.cleanup()
                self.assertFalse(first_root.exists())
                self.assertEqual(
                    second_files, {p.relative_to(second_root) for p in second_root.rglob("*")},
                )
                second.cleanup()
                self.assertFalse(second_root.exists())

    def test_failed_build_is_removed_and_next_build_uses_a_fresh_workspace(self):
        for module_name, class_name, _ in OPTIMIZERS:
            with self.subTest(optimizer=class_name):
                optimizer = self.make_optimizer(module_name, class_name)
                failed_root = Path(optimizer.json_filename).parent

                def fail_after_writing(ocp, *, json_file):
                    write_build_artifacts(ocp, json_file=json_file)
                    raise RuntimeError("mock compile failure")

                with patch(
                    f"{self.solver_module(module_name)}.AcadosOcpSolver",
                    side_effect=fail_after_writing,
                ):
                    with self.assertRaisesRegex(RuntimeError, "mock compile failure"):
                        optimizer.create_solver()
                optimizer.reset()
                self.assertFalse(failed_root.exists())
                optimizer.ocp = SimpleNamespace(solver_options=SimpleNamespace())
                with patch(
                    f"{self.solver_module(module_name)}.AcadosOcpSolver",
                    side_effect=write_build_artifacts,
                ):
                    optimizer.create_solver()
                new_root = Path(optimizer.json_filename).parent
                self.assertNotEqual(new_root, failed_root)
                self.assertTrue(new_root.is_dir())
                optimizer.cleanup()
                self.assertFalse(new_root.exists())

    def test_normal_process_exit_removes_abandoned_workspace(self):
        # Exercise TemporaryDirectory's exit fallback without importing solvers.
        result = subprocess.run(
            [sys.executable, "-c", "\n".join((
                "from pathlib import Path",
                "from utils.acados_codegen import AcadosCodegenMixin",
                "owner = AcadosCodegenMixin()",
                "path = Path(owner.json_filename)",
                "path.write_text('{}', encoding='utf-8')",
                "print(path.parent)",
            ))],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True, timeout=10,
        )
        self.assertFalse(Path(result.stdout.strip()).exists())


if __name__ == "__main__":
    unittest.main()
