"""Mechanics-informed benchmarks for compliant robot grasping."""

from .materials import Geometry, StandardLinearSolid
from .identification import CompressionTrace, fit_material

__version__ = "0.2.0"
__all__ = ["Geometry", "StandardLinearSolid", "CompressionTrace", "fit_material"]
