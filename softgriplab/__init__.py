"""Mechanics-informed benchmarks for compliant robot grasping."""

from importlib import import_module

__version__ = "0.5.0"
__all__ = ["Geometry", "StandardLinearSolid", "CompressionTrace", "fit_material"]


def __getattr__(name):
    # Robot/continuum commands do not need the SciPy identification stack.
    modules = {
        "Geometry": "materials",
        "StandardLinearSolid": "materials",
        "CompressionTrace": "identification",
        "fit_material": "identification",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{modules[name]}", __name__), name)
    globals()[name] = value
    return value
