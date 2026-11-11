"""RBDO benchmark problems and the registry mapping benchmark ids to their classes."""

from .base import (
    TrussBridge,
    Cantilever2d,
    Cantilever4d,
    FourBranches,
    BenchmarkBase,
    LinearStackup,
    ThreeNonlinear,
    HighlyNonlinear,
    BracketStructure,
    Cantilever2dMeanShift,
    Cantilever4dMeanShift,
    ColumnUnderCompression,
    LinearStackupMeanShift,
    register_benchmark,
)

__all__ = [
    "BenchmarkBase",
    "BracketStructure",
    "Cantilever2d",
    "Cantilever2dMeanShift",
    "Cantilever4d",
    "Cantilever4dMeanShift",
    "ColumnUnderCompression",
    "FourBranches",
    "HighlyNonlinear",
    "LinearStackup",
    "LinearStackupMeanShift",
    "ThreeNonlinear",
    "TrussBridge",
    "register_benchmark",
]
