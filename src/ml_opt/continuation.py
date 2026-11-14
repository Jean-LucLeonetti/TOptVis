"""Iteration-based, capped geometric continuation of projection sharpness."""

from .config import CantileverConfig


def next_beta(config: CantileverConfig, beta: float, completed_iterations: int) -> float:
    """Increase after each full interval; the initial evaluation is iteration zero."""
    if (config.projection_schedule == "fixed" or completed_iterations == 0 or
            completed_iterations % config.projection_beta_interval):
        return beta
    # Avoid overflow even for a very large finite multiplier.
    if beta >= config.projection_beta_max / config.projection_beta_multiplier:
        return config.projection_beta_max
    return beta * config.projection_beta_multiplier
