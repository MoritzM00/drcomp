"""Dimensionality Reduction Comparison."""

from .dimensionality_reducer import (
    DimensionalityReducer,
    MetricsDict,
    estimate_intrinsic_dimension,
)

__all__ = ["DimensionalityReducer", "estimate_intrinsic_dimension", "MetricsDict"]
