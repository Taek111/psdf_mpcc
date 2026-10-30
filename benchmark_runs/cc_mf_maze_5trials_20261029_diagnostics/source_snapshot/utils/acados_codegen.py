"""Private, temporary acados build paths for each optimizer instance."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory


class AcadosCodegenMixin:
    """Keep main/backup solvers and cache metadata in one owned directory.

    Paths are allocated lazily, so constructing or cleaning up an unbuilt
    optimizer does not touch the filesystem. TemporaryDirectory also removes
    abandoned builds at garbage collection or normal interpreter shutdown.
    After cleanup, the next build gets a new directory. Callers must release
    their solver objects before calling _cleanup_codegen_workspace().
    """

    _codegen_name = "ocp"

    def _codegen_path(self, filename: str) -> str:
        workspace = getattr(self, "_acados_workspace", None)
        if workspace is None:
            workspace = TemporaryDirectory(
                prefix=f"psdf_mpcc_{self._codegen_name}_{os.getpid()}_",
            )
            self._acados_workspace = workspace
        return str(Path(workspace.name).resolve() / filename)

    @property
    def json_filename(self) -> str:
        return self._codegen_path(f"acados_ocp_{self._codegen_name}.json")

    @property
    def code_export_directory(self) -> str:
        return self._codegen_path("code")

    @property
    def backup_json_filename(self) -> str:
        return self._codegen_path(f"acados_ocp_{self._codegen_name}_backup.json")

    @property
    def backup_code_export_directory(self) -> str:
        return self._codegen_path("code_backup")

    @property
    def codegen_meta_filename(self) -> str:
        return self._codegen_path(f"acados_{self._codegen_name}_codegen_meta.json")

    def _cleanup_codegen_workspace(self) -> None:
        workspace = getattr(self, "_acados_workspace", None)
        if workspace is not None:
            try:
                workspace.cleanup()
            except OSError as error:
                # Retain ownership so a later cleanup can retry.
                print(f"Warning: Could not remove acados workspace {workspace.name}: {error}")
            else:
                self._acados_workspace = None
