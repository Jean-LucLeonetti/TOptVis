# SPDX-License-Identifier: GPL-3.0-or-later
"""Fused evaluation of the pinned MMA primal-dual residual equations."""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def residual_norm(m, n, x, y, z, lam, xsi, eta, mu, zet, s,
                  low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d, epsi):
    inv_upper, inv_lower = np.empty(n), np.empty(n)
    squared, maximum = 0.0, 0.0
    for j in range(n):
        inv_upper[j] = 1.0/(upp[j, 0]-x[j, 0])
        inv_lower[j] = 1.0/(x[j, 0]-low[j, 0])
        plam, qlam = p0[j, 0], q0[j, 0]
        for i in range(m):
            plam += P[i, j]*lam[i, 0]
            qlam += Q[i, j]*lam[i, 0]
        rx = plam*inv_upper[j]**2-qlam*inv_lower[j]**2-xsi[j, 0]+eta[j, 0]
        rxi = xsi[j, 0]*(x[j, 0]-alfa[j, 0])-epsi
        reta = eta[j, 0]*(beta[j, 0]-x[j, 0])-epsi
        squared += rx*rx+rxi*rxi+reta*reta
        maximum = max(maximum, abs(rx), abs(rxi), abs(reta))
    alam = 0.0
    for i in range(m):
        upper, lower = 0.0, 0.0
        for j in range(n):
            upper += P[i, j]*inv_upper[j]
            lower += Q[i, j]*inv_lower[j]
        ry = c[i, 0]+d[i, 0]*y[i, 0]-mu[i, 0]-lam[i, 0]
        rl = upper+lower-a[i, 0]*z[0, 0]-y[i, 0]+s[i, 0]-b[i, 0]
        rmu = mu[i, 0]*y[i, 0]-epsi
        rs = lam[i, 0]*s[i, 0]-epsi
        squared += ry*ry+rl*rl+rmu*rmu+rs*rs
        maximum = max(maximum, abs(ry), abs(rl), abs(rmu), abs(rs))
        alam += a[i, 0]*lam[i, 0]
    rz, rzet = a0-zet[0, 0]-alam, zet[0, 0]*z[0, 0]-epsi
    squared += rz*rz+rzet*rzet
    maximum = max(maximum, abs(rz), abs(rzet))
    if np.isnan(squared):
        return np.nan, np.nan
    return np.sqrt(squared), maximum
