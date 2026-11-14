"""Topology optimisation with PyTorch and PETSc."""

from .config import CantileverConfig
from .runner import OptimizationResult, optimise

__all__ = ["CantileverConfig", "OptimizationResult", "optimise"]
