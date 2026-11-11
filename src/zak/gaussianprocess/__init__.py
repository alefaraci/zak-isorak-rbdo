"""Kriging surrogates of the limit-state functions."""

from .gp import GPRBoTorch, ListGPRBoTorch

__all__ = ["GPRBoTorch", "ListGPRBoTorch"]
