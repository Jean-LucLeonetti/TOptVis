"""Differentiable, normalised smooth Heaviside density projection."""

import math

import torch


def heaviside_projection(density: torch.Tensor, beta: float, eta: float) -> torch.Tensor:
    """Sharpen a density field at eta; beta=0 gives the identity exactly.

    PyTorch differentiates this pointwise operation before propagating through
    the Helmholtz adjoint. Beta and eta remain fixed during an optimisation.
    """
    if not math.isfinite(beta) or beta < 0 or not math.isfinite(eta) or not 0 < eta < 1:
        raise ValueError("Projection requires finite beta >= 0 and 0 < eta < 1")
    if beta == 0:
        return density
    offset = math.tanh(beta * eta)
    denominator = offset + math.tanh(beta * (1 - eta))
    return (offset + torch.tanh(beta * (density - eta))) / denominator
