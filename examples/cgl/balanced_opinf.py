"""
Operator Inference for the balanced-OpInf study (see balanced_opinf.pdf).

The balancing itself now lives in the library, as
nitrom.utils.compute_data_driven_balancing; what remains here is the Operator
Inference side and the diagnostics that consume the balancing output.

Operator Inference: dz/dt = A_r z + c^(p-1) H_r (z x ... x z) + B_r u, z = Psi^T x~, for
trajectories x~ = x/c normalized by their IC amplitude c and a polynomial nonlinearity
of degree p (p = 2 for Navier-Stokes, p = 3 for the CGL).
"""

import itertools
import time as timer

import numpy as np
import scipy.linalg as sla
from scipy.integrate import solve_ivp


def check_balancing(Phi, Psi, Sig, info, r=None):
    """Biorthogonality and balancing, from the factors only:
    Psi^T W_c Psi = (Psi^T X)(Psi^T X)^T with Psi^T X = Sigma^{-1/2} L^T H, and
    Phi^T W_o Phi = (Q^T Phi)^T (Q^T Phi)."""
    r = len(Sig) if r is None else min(r, len(Sig))
    Ph, Ps, S = Phi[:, :r], Psi[:, :r], Sig[:r]
    PsX = (info["L"][:, :r]/np.sqrt(S)).T @ info["H"]
    QtPh = info["Q"].T @ Ph
    return dict(biorth=np.linalg.norm(Ps.T @ Ph - np.eye(r)),
                Wc=np.linalg.norm(PsX @ PsX.T - np.diag(S))/np.linalg.norm(S),
                Wo=np.linalg.norm(QtPh.T @ QtPh - np.diag(S))/np.linalg.norm(S))


def pod_basis(trajs, r):
    return sla.svd(np.concatenate(trajs, axis=1), full_matrices=False)[0][:, :r]


# ----------------------------------------------------------------------------
# Operator Inference
# ----------------------------------------------------------------------------

DEGREE = {"linear": 1, "quadratic": 2, "cubic": 3}


def ddt_fd4(Z, dt):
    """4th-order central differences along axis 1; drops 2 samples at each end."""
    return (Z[:, :-4] - 8*Z[:, 1:-3] + 8*Z[:, 3:-1] - Z[:, 4:])/(12*dt)


def poly_index(r, p):
    """Unique monomials of degree p in r variables."""
    return np.array(list(itertools.combinations_with_replacement(range(r), p)))


def poly_features(Z, idx):
    F = Z[idx[:, 0]]
    for k in range(1, idx.shape[1]):
        F = F*Z[idx[:, k]]
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
            f += c**(self.p - 1)*(self.Hr @ poly_features(z[:, None], self.idx)[:, 0])
        if forcing is not None:
            f += forcing(t)
        return f

    def integrate(self, z0, time, c=1.0, forcing=None, blowup=1e6):
        """DOP853; stops (NaN afterwards) if ||z|| exceeds blowup*max(||z0||, 1)."""
        zmax = blowup*max(np.linalg.norm(z0), 1.0)
        event = lambda t, z, c, f: np.linalg.norm(z) - zmax
        event.terminal = True
        sol = solve_ivp(self.rhs, (time[0], time[-1]), z0, t_eval=time, args=(c, forcing),
                        method="DOP853", rtol=1e-9, atol=1e-12, events=event)
        Z = np.full((len(z0), len(time)), np.nan)
        Z[:, :sol.y.shape[1]] = sol.y
        return Z


def opinf(trajs, Psi, dt, model="quadratic", reg=0.0, amps=None, C=None,
          normalize_energy=True):
    """Fit dz/dt = A_r z (+ c^(p-1) H_r z^p) to z = Psi^T x~ (x~: amplitude-normalized
    trajectories with IC amplitudes `amps`) by least squares; only the nonlinear
    block is (Tikhonov) regularized. dz/dt from 4th-order finite differences.

    The fit uses the original trajectories: checkpointing and Lall averaging
    belong to the balancing, not here.

    With `normalize_energy` (default) each trajectory's residual is weighted by
    1/E_j with E_j = <||C x~||^2> its time-averaged **output** energy, so the
    cost is sum_j (1/E_j) sum_i ||dz/dt - f(z)||^2.  Using the output and not
    the state matches the NiTROM cost (3.7) and nitrom_cost(), so the fit, the
    regularization selection and NiTROM itself all weight trajectories the same
    way.  Without the weighting the trajectories enter equally and the fit is
    dominated by the most energetic ones: here <||C x~||^2> spans ~63x, with the
    near-linear alpha = 0.1 trajectories largest and the saturated alpha = 5
    ones smallest, so the very data added to capture saturation would carry the
    least weight."""
    r = Psi.shape[1]
    amps = np.ones(len(trajs)) if amps is None else np.asarray(amps)
    Zs = [Psi.T @ Xj for Xj in trajs]
    Z = np.concatenate([Zj[:, 2:-2] for Zj in Zs], axis=1)
    dZ = np.concatenate([ddt_fd4(Zj, dt) for Zj in Zs], axis=1)
    p = DEGREE[model]
    if p == 1:
        D = Z
    else:
        cp = np.concatenate([np.full(Zj.shape[1] - 4, c**(p - 1)) for Zj, c in zip(Zs, amps)])
        D = np.concatenate((Z, cp*poly_features(Z, poly_index(r, p))), axis=0)
    if normalize_energy:
        if C is None:
            raise ValueError(
                "normalize_energy weights by the output energy <||C x~||^2>, "
                "so the output operator C must be given."
            )
        # Column weights 1/sqrt(E_j); the regularization block below is left
        # unscaled, so `reg` keeps its meaning relative to the weighted fit.
        w = np.concatenate([np.full(Zj.shape[1] - 4,
                                    1.0/np.sqrt(np.mean(np.sum((C @ Xj)**2, axis=0))))
                            for Xj, Zj in zip(trajs, Zs)])
        D, dZ = D*w, dZ*w
    d = D.shape[0]
    Gam = np.zeros(d)
    Gam[r:] = np.sqrt(reg)
    O = sla.lstsq(np.concatenate((D.T, np.diag(Gam))),
                  np.concatenate((dZ.T, np.zeros((d, r)))))[0].T
    return ROM(O[:, :r], O[:, r:] if p > 1 else None, p)


REG_GRID = 10.0**np.arange(-2, 9, 1)


def nitrom_cost(rom, Phi, Psi, trajs, time, C, amps=None):
    """
    NiTROM trajectory cost, eq. (3.7):

        J = sum_j (1/alpha_j) sum_i ||y^(j)(t_i) - C Phi z^(j)(t_i)||^2,

    with alpha_j the time-averaged output energy of trajectory j.  Each ratio is
    scale invariant, so it makes no difference whether the trajectories are
    given in physical or amplitude-normalized coordinates.  The constant
    N_traj * N of (3.7) is omitted: it does not affect which model wins.

    Returns inf if the ROM blows up, so the caller can reject it.
    """
    amps = np.ones(len(trajs)) if amps is None else amps
    J = 0.0
    for Xj, c in zip(trajs, amps):
        Z = rom.integrate(Psi.T @ Xj[:, 0], time[:Xj.shape[1]], c)
        yh = C @ (Phi @ Z)
        if not np.all(np.isfinite(yh)):
            return np.inf
        y = C @ Xj
        J += np.sum((y - yh)**2)/np.mean(np.sum(y**2, axis=0))
    return J


def opinf_select(trajs, Phi, Psi, dt, time, amps, C, model="quadratic",
                 grid=REG_GRID, verbose=False):
    """OpInf with the nonlinear-term regularization chosen from `grid` by the
    NiTROM trajectory cost (3.7) on the training data -- the same quantity
    NiTROM itself minimizes, and normalized per trajectory by its average output
    energy, so it is consistent with the 1/E_j weighting inside opinf().
    Testing data are never used.  Returns (rom, reg, J, state error)."""
    def score(rom):
        return (nitrom_cost(rom, Phi, Psi, trajs, time, C, amps),
                rel_error(rom, Phi, Psi, trajs, time, amps)[0])

    if model == "linear":
        rom = opinf(trajs, Psi, dt, "linear", C=C)
        return (rom, 0.0) + score(rom)
    best = (np.inf, None, None, np.inf)
    for reg in grid:
        t0 = timer.time()
        rom = opinf(trajs, Psi, dt, model, reg, amps, C)
        J, e = score(rom)
        if verbose:
            print(f"      reg {reg:.0e}: J {J:.4e}, state error {e:.3e} "
                  f"({timer.time() - t0:.1f} s)", flush=True)
        if np.isfinite(J) and J < best[0]:
            best = (J, rom, reg, e)
    return best[1], best[2], best[0], best[3]


def rel_error(rom, Phi, Psi, trajs, time, amps=None):
    """Energy-weighted relative state error over trajectories,
    sqrt(sum_j ||X_j - Phi Z_j||^2 / sum_j ||X_j||^2); also per-trajectory errors
    and the ROM trajectories."""
    amps = np.ones(len(trajs)) if amps is None else amps
    num, den, errs, Zs = 0.0, 0.0, [], []
    for Xj, c in zip(trajs, amps):
        Z = rom.integrate(Psi.T @ Xj[:, 0], time[:Xj.shape[1]], c)
        Zs.append(Z)
        ej = np.linalg.norm(Xj - Phi @ Z)
        num, den = num + ej**2, den + np.linalg.norm(Xj)**2
        errs.append(ej/np.linalg.norm(Xj))
    return np.sqrt(num/den), np.array(errs), Zs
