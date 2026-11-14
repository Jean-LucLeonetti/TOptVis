"""Differentiable stiffness interpolation of the physical material density."""

import torch

from .config import CantileverConfig


def elastic_modulus(density: torch.Tensor, config: CantileverConfig) -> torch.Tensor:
    """SIMP uses exponent p; RAMP uses rational penalty q (q=0 is linear)."""
    if config.material_model == "simp":
        fraction = density.pow(config.penalisation)
    else:
        fraction = density / (1 + config.penalisation * (1 - density))
    return config.minimum_modulus + (config.young_modulus - config.minimum_modulus) * fraction
