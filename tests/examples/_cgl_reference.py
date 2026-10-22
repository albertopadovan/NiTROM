"""Independent compact-monomial reference for package fitting equivalence tests."""

import itertools
from math import factorial

import numpy as np
import scipy.linalg as sla
from scipy.integrate import solve_ivp

DEGREE = {"linear": 1, "quadratic": 2, "cubic": 3}


def ddt_fd4(Z, dt):
    """4th-order central differences along axis 1; drops 2 samples at each end."""
    return (Z[:, :-4] - 8 * Z[:, 1:-3] + 8 * Z[:, 3:-1] - Z[:, 4:]) / (12 * dt)


def poly_index(r, p):
    """Unique monomials of degree p in r variables."""
    return np.array(list(itertools.combinations_with_replacement(range(r), p)))


def poly_features(Z, idx):
    F = Z[idx[:, 0]]
    for k in range(1, idx.shape[1]):
        F = F * Z[idx[:, k]]
    return F


class ROM:
    """dz/dt = A_r z + c^(p-1) H_r z^p (+ f_r(t)). For trajectories normalized by
    their IC amplitude c the degree-p term picks up c^(p-1); physical coordinates: c = 1."""

    def __init__(self, Ar, Hr=None, degree=2):
        self.Ar = Ar
        self.p = degree
        self.Hr = Hr if (Hr is not None and Hr.size) else None
        self.idx = poly_index(Ar.shape[0], degree) if self.Hr is not None else None

    def rhs(self, t, z, c=1.0, forcing=None):
        f = self.Ar @ z
        if self.Hr is not None:
            f += c ** (self.p - 1) * (
                self.Hr @ poly_features(z[:, None], self.idx)[:, 0]
            )
        if forcing is not None:
            f += forcing(t)
        return f

    def integrate(self, z0, time, c=1.0, forcing=None, blowup=1e6):
        """DOP853; stops (NaN afterwards) if ||z|| exceeds blowup*max(||z0||, 1)."""
        zmax = blowup * max(np.linalg.norm(z0), 1.0)

        def event(t, z, c, f):
            return np.linalg.norm(z) - zmax

        event.terminal = True
        sol = solve_ivp(
            self.rhs,
            (time[0], time[-1]),
            z0,
            t_eval=time,
            args=(c, forcing),
            method="DOP853",
            rtol=1e-9,
            atol=1e-12,
            events=event,
        )
        Z = np.full((len(z0), len(time)), np.nan)
        Z[:, : sol.y.shape[1]] = sol.y
        return Z


def opinf(
    trajs, Psi, dt, model="quadratic", reg=0.0, amps=None, C=None, normalize_energy=True
):
    """Fit dz/dt = A_r z (+ c^(p-1) H_r z^p) to z = Psi^T x~ (x~: amplitude-normalized
    trajectories with IC amplitudes `amps`) by least squares; only the nonlinear
    block is (Tikhonov) regularized, using the Frobenius norm of its symmetric
    dense tensor. A unique monomial with multiplicity m contributes h**2/m,
    so mixed terms receive the same penalty as their dense representation.
    dz/dt from 4th-order finite differences.

    The fit uses the original trajectories: checkpointing and Lall averaging
    belong to the balancing, not here.

    With `normalize_energy` (default) each trajectory's residual is weighted by
    1/E_j with E_j = <||C x~||^2> its time-averaged **output** energy, so the
    cost is sum_j (1/E_j) sum_i ||dz/dt - f(z)||^2.  Using the output and not
    the state matches the NiTROM cost (3.7) and nitrom_cost(), so the fit, the
    regularization selection and NiTROM itself all weight trajectories the same
    way.  Without the weighting the trajectories enter equally and the fit is
    dominated by the most energetic ones."""
    r = Psi.shape[1]
    amps = np.ones(len(trajs)) if amps is None else np.asarray(amps)
    Zs = [Psi.T @ Xj for Xj in trajs]
    Z = np.concatenate([Zj[:, 2:-2] for Zj in Zs], axis=1)
    dZ = np.concatenate([ddt_fd4(Zj, dt) for Zj in Zs], axis=1)
    p = DEGREE[model]
    if p == 1:
        D = Z
    else:
        cp = np.concatenate(
            [
                np.full(Zj.shape[1] - 4, c ** (p - 1))
                for Zj, c in zip(Zs, amps, strict=False)
            ]
        )
        D = np.concatenate((Z, cp * poly_features(Z, poly_index(r, p))), axis=0)
    if normalize_energy:
        if C is None:
            raise ValueError(
                "normalize_energy weights by the output energy <||C x~||^2>, "
                "so the output operator C must be given."
            )
        # Column weights 1/sqrt(E_j); the regularization block below is left
        # unscaled, so `reg` keeps its meaning relative to the weighted fit.
        w = np.concatenate(
            [
                np.full(
                    Zj.shape[1] - 4,
                    1.0 / np.sqrt(np.mean(np.sum((C @ Xj) ** 2, axis=0))),
                )
                for Xj, Zj in zip(trajs, Zs, strict=False)
            ]
        )
        D, dZ = D * w, dZ * w
    d = D.shape[0]
    Gam = np.zeros(d)
    if p > 1:
        multiplicities = np.array(
            [
                factorial(p)
                / np.prod(
                    [factorial(int(c)) for c in np.unique(term, return_counts=True)[1]]
                )
                for term in poly_index(r, p)
            ]
        )
        Gam[r:] = np.sqrt(reg / multiplicities)
    operators = sla.lstsq(
        np.concatenate((D.T, np.diag(Gam))), np.concatenate((dZ.T, np.zeros((d, r))))
    )[0].T
    return ROM(operators[:, :r], operators[:, r:] if p > 1 else None, p)
