"""Sampling of the design space."""

from . import sampling
from .sampling import SobolTrainingPoints

__all__ = [
    "SobolTrainingPoints",
    "sampling",
]
