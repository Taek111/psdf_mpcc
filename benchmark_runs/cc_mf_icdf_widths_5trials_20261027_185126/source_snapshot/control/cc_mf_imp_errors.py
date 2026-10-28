"""Lightweight fail-stop exception shared by implicit MF entry points."""


class MFImplicitRootError(RuntimeError):
    """An implicit evaluation failed; the current control run must stop."""

    def __init__(self, message, *, context=None, diagnostics=None):
        super().__init__(message)
        self.context = context
        self.diagnostics = diagnostics
        # The trial entry point fills this before releasing optimizer resources.
        self.trial_result = None
