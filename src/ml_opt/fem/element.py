"""Unit-Young-modulus plane-stress Q4 stiffness using 2x2 Gauss integration."""

import numpy as np


def q4_stiffness(width: float, height: float, poisson: float, thickness: float) -> np.ndarray:
    constitutive = np.array([[1, poisson, 0], [poisson, 1, 0],
                            [0, 0, (1 - poisson) / 2]]) / (1 - poisson**2)
    stiffness = np.zeros((8, 8))
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    for xi in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
        for eta in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
            dx = signs[:, 0] * (1 + signs[:, 1] * eta) / (2 * width)
            dy = signs[:, 1] * (1 + signs[:, 0] * xi) / (2 * height)
            strain = np.zeros((3, 8))
            strain[0, 0::2] = dx
            strain[1, 1::2] = dy
            strain[2, 0::2] = dy
            strain[2, 1::2] = dx
            stiffness += strain.T @ constitutive @ strain * width * height * thickness / 4
    return stiffness
