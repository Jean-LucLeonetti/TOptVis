"""Maximum phase sizes from compact averaging and relaxed morphology.

Lazarov & Wang (2017), section 5, DOI: 10.1016/j.cma.2017.02.018.
The disk diameter is in model units. Outside the design domain is void.
"""

import math

import numpy as np
from scipy.fft import irfftn, next_fast_len, rfftn
import torch
from torch.autograd.function import once_differentiable

from .fem.mesh import StructuredMesh


class _Average(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, operator):
        ctx.operator = operator
        return torch.from_numpy(operator.average(density.detach().numpy()))

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        # A symmetric kernel with zero extension/cropping gives a symmetric map.
        return torch.from_numpy(ctx.operator.average(gradient.detach().numpy())), None


class MorphologicalSize:
    def __init__(self, mesh: StructuredMesh, diameter: float, beta: float, tolerance: float):
        self.mesh, self.beta, self.tolerance = mesh, beta, tolerance
        radius = diameter / 2
        dx, dy = mesh.length / mesh.nx, mesh.height / mesh.ny
        sx, sy = math.ceil(radius/dx), math.ceil(radius/dy)
        x, y = np.meshgrid(np.arange(-sx, sx+1)*dx, np.arange(-sy, sy+1)*dy)
        kernel = np.maximum(0, 1 - np.hypot(x, y)/radius)
        kernel /= kernel.sum()
        self._slice = (slice(sy, sy+mesh.ny), slice(sx, sx+mesh.nx))
        self._shape = (next_fast_len(mesh.ny+2*sy), next_fast_len(mesh.nx+2*sx))
        self._kernel_fft = rfftn(kernel, s=self._shape)

    def average(self, density: np.ndarray) -> np.ndarray:
        image = np.asarray(density, dtype=np.float64).reshape(self.mesh.ny, self.mesh.nx)
        result = irfftn(rfftn(image, s=self._shape) * self._kernel_fft, s=self._shape)
        return result[self._slice].copy().ravel()

    def eroded(self, density: torch.Tensor) -> torch.Tensor:
        if density.device.type != "cpu" or density.dtype != torch.float64:
            raise ValueError("Maximum-size erosion requires CPU float64 density")
        # Preserve the linear adjoint at fully solid/void points. Clipping FFT
        # roundoff at 1 would incorrectly zero the derivative of solid interiors.
        averaged = _Average.apply(density, self)
        return 1 - torch.tanh(self.beta * (1 - averaged)) / math.tanh(self.beta)

    def remaining_void(self, density: torch.Tensor) -> torch.Tensor:
        """Void left after material dilation at threshold zero.

        Convolve material with zero extension, so outside remains void. Eroding
        1-density with zero extension would incorrectly make the exterior solid.
        """
        if density.device.type != "cpu" or density.dtype != torch.float64:
            raise ValueError("Maximum-size dilation requires CPU float64 density")
        averaged = _Average.apply(density, self)
        return 1 - torch.tanh(self.beta * averaged) / math.tanh(self.beta)


class MaximumSolidSize(MorphologicalSize):
    """Solid-size operator retained for existing callers."""
