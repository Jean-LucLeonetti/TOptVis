# SPDX-License-Identifier: GPL-3.0-or-later
# Derived from mmapy 0.3.1 (GCMMA-MMA-Python), Arjen Deetman;
# original MMA algorithm by Krister Svanberg.
# https://github.com/arjendeetman/GCMMA-MMA-Python
# See GPL-3.0-or-later.txt in this package.
"""Compiled pinned MMA primal-dual loop, with unchanged barrier/stopping tests.

Changes from upstream: dense diagonal broadcasting, scalar residual norms and
NumPy's LAPACK-backed solve permit compilation. fastmath stays disabled.
"""

import numpy as np
from numba import njit

from ._residual import residual_norm


@njit(cache=True, fastmath=False)
def subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    een = np.ones((n, 1))
    eem = np.ones((m, 1))
    epsi = 1
    epsvecn = epsi * een
    epsvecm = epsi * eem
    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = np.array([[1.0]])
    lam = eem.copy()
    xsi = een / (x - alfa)
    xsi = np.maximum(xsi, een)
    eta = een / (beta - x)
    eta = np.maximum(eta, een)
    mu = np.maximum(eem, 0.5 * c)
    zet = np.array([[1.0]])
    s = eem.copy()
    itera = 0
    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem
        residunorm, residumax = residual_norm(m, n, x, y, z, lam, xsi, eta, mu, zet, s, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d, epsi)
        ittt = 0
        while residumax > 0.9 * epsi and ittt < 200:
            ittt += 1
            itera += 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2
            xlinv2 = een / xl2
            plam = p0 + np.dot(P.T, lam)
            qlam = q0 + np.dot(Q.T, lam)
            gvec = np.dot(P, uxinv1) + np.dot(Q, xlinv1)
            GG = (uxinv2 * P.T).T - (xlinv2 * Q.T).T
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - np.dot(a.T, lam) - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam
            diagx = plam / ux3 + qlam / xl3
            diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = een / diagx
            diagy = d + mu / y
            diagyinv = eem / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv
            if m < n:
                blam = dellam + dely / diagy - np.dot(GG, delx / diagx)
                bb = np.concatenate((blam, delz), axis=0)
                Alam = np.asarray(np.diag(diaglamyi.flatten()) + (GG * diagxinv.T).dot(GG.T))
                AAr1 = np.concatenate((Alam, a), axis=1)
                AAr2 = np.concatenate((a, -zet / z), axis=0).T
                AA = np.concatenate((AAr1, AAr2), axis=0)
                solut = np.linalg.solve(AA, bb)
                dlam = solut[0:m].copy()
                dz = solut[m:m + 1].copy()
                dx = -delx / diagx - np.dot(GG.T, dlam) / diagx
            else:
                diaglamyiinv = eem / diaglamyi
                dellamyi = dellam + dely / diagy
                Axx = np.asarray(np.diag(diagx.flatten()) + (diaglamyiinv * GG).T.dot(GG))
                azz = zet / z + np.dot(a.T, a / diaglamyi)
                axz = np.dot(-GG.T, a / diaglamyi)
                bx = delx + np.dot(GG.T, dellamyi / diaglamyi)
                bz = delz - np.dot(a.T, dellamyi / diaglamyi)
                AAr1 = np.concatenate((Axx, axz), axis=1)
                AAr2 = np.concatenate((axz.T, azz), axis=1)
                AA = np.concatenate((AAr1, AAr2), axis=0)
                bb = np.concatenate((-bx, -bz), axis=0)
                solut = np.linalg.solve(AA, bb)
                dx = solut[0:n].copy()
                dz = solut[n:n + 1].copy()
                dlam = np.dot(GG, dx) / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi
            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsvecn / (x - alfa) - xsi * dx / (x - alfa)
            deta = -eta + epsvecn / (beta - x) + eta * dx / (beta - x)
            dmu = -mu + epsvecm / y - mu * dy / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm / lam - s * dlam / lam
            xx = np.concatenate((y, z, lam, xsi, eta, mu, zet, s), axis=0)
            dxx = np.concatenate((dy, dz, dlam, dxsi, deta, dmu, dzet, ds), axis=0)
            stepxx = -1.01 * dxx / xx
            stmxx = np.max(stepxx)
            stepalfa = -1.01 * dx / (x - alfa)
            stmalfa = np.max(stepalfa)
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta = np.max(stepbeta)
            stmalbe = np.maximum(stmalfa, stmbeta)
            stmalbexx = np.maximum(stmalbe, stmxx)
            stminv = np.maximum(stmalbexx, 1.0)
            steg = 1.0 / stminv
            xold = x.copy()
            yold = y.copy()
            zold = z.copy()
            lamold = lam.copy()
            xsiold = xsi.copy()
            etaold = eta.copy()
            muold = mu.copy()
            zetold = zet.copy()
            sold = s.copy()
            itto = 0
            resinew = 2 * residunorm
            while resinew > residunorm and itto < 50:
                itto += 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                resinew, new_residual_max = residual_norm(m, n, x, y, z, lam, xsi, eta, mu, zet, s, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d, epsi)
                steg = steg / 2
            residunorm = resinew
            residumax = new_residual_max
            steg = 2 * steg
        epsi = 0.1 * epsi
    xmma = x.copy()
    ymma = y.copy()
    zmma = z.copy()
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s
    return (xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma)
