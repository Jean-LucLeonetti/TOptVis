"""Pinned mmapy algorithms with broadcast scaling and a compiled Newton loop.

Bind the upstream routines to a private globals dictionary. ``diags`` uses
broadcasting; multi-constraint subproblems use the compiled primal-dual loop.
Equations and stopping tests are retained without mutating the installed package.
The mmapy dependency is pinned because this adapter uses its internal subsolv.
"""

from functools import cache, update_wrapper
from types import FunctionType

import mmapy.mma as upstream
import numpy as np


class _Diagonal:
    def __init__(self, values, offset=0):
        if offset != 0:
            raise ValueError("Only main-diagonal scaling is supported")
        self.values = np.asarray(values)

    def dot(self, array):
        array = np.asarray(array)
        return self.values.reshape((-1,) + (1,) * (array.ndim - 1)) * array

    def __add__(self, array):
        result = np.array(array, copy=True)
        index = np.arange(self.values.size)
        result[index, index] += self.values
        return result


@cache
def _compiled_kernel():
    try:
        from ._subproblem import subsolv
    except ImportError:
        # Existing environments remain usable until dependencies are updated.
        return None
    return subsolv


def warmup_optimizer():
    """Compile/load one float64 signature before the browser enables Run."""
    kernel = _compiled_kernel()
    if kernel is None:
        return False
    n, m = 3, 2
    column = np.ones((n, 1))
    rows = np.ones((m, 1))
    kernel(m, n, 1e-7, 0*column, column, .1*column, .9*column,
           .1*column, .1*column, np.full((m, n), .1), np.full((m, n), .1),
           1.0, 0*rows, 10*rows, 1000*rows, rows)
    return True


def _bind_functions(*, compiled=True):
    namespace = dict(upstream.mmasub.__globals__)
    namespace["diags"] = _Diagonal
    for name in ("subsolv", "mmasub", "gcmmasub"):
        function = getattr(upstream, name)
        clone = FunctionType(function.__code__, namespace, function.__name__,
                             function.__defaults__, function.__closure__)
        clone.__kwdefaults__ = function.__kwdefaults__
        namespace[name] = update_wrapper(clone, function)
    if compiled:
        reference = namespace["subsolv"]
        def subsolv(m, *args, **kwargs):
            kernel = _compiled_kernel() if m > 1 else None
            if kernel is None:
                return reference(m, *args, **kwargs)
            contiguous = tuple(np.ascontiguousarray(value) if isinstance(value, np.ndarray) else value
                               for value in args)
            return kernel(m, *contiguous, **kwargs)
        namespace["subsolv"] = subsolv
    return namespace["mmasub"], namespace["gcmmasub"]


mmasub, gcmmasub = _bind_functions()
