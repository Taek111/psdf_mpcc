"""Supported controller options shared by configuration and CLI entry points."""

SUPPORTED_OPTIMIZER_TYPES = (
    "obca",
    "psdf",
    "mpcc",
    "cc_mf",
    "cc_mf_imp",
    "cc_sf",
    "dcbf",
    "cc_ecs",
    "sipoc",
    "uobca",
)
DEFAULT_OPTIMIZER_TYPE = "psdf"


def normalize_optimizer_type(optimizer_type):
    """Resolve the explicit CC-MF-IMP alias without rewriting other names."""
    return "cc_mf_imp" if optimizer_type == "cc-mf-imp" else optimizer_type
