"""mmapy 0.3.1 algorithms without sparse diagonal scaling allocations.

Bind the upstream routines to a private globals dictionary: only ``diags``
changes. This keeps mmapy's equations, stopping tests, and return values intact
without mutating the installed package or its process-wide module globals.
The mmapy dependency is pinned because this adapter uses its internal subsolv.
"""

from functools import update_wrapper
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


def _bind_functions():
    namespace = dict(upstream.mmasub.__globals__)
    namespace["diags"] = _Diagonal
    for name in ("subsolv", "mmasub", "gcmmasub"):
        function = getattr(upstream, name)
        clone = FunctionType(function.__code__, namespace, function.__name__,
                             function.__defaults__, function.__closure__)
        clone.__kwdefaults__ = function.__kwdefaults__
        namespace[name] = update_wrapper(clone, function)
    return namespace["mmasub"], namespace["gcmmasub"]


mmasub, gcmmasub = _bind_functions()
