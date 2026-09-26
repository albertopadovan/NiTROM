"""
Operator Inference for the lid-driven cavity (SIADS 2024, section 5).

The balancing itself lives in the library, as
nitrom.utils.compute_data_driven_balancing; what remains here is the Operator
Inference side and the diagnostics that consume the balancing output.

Operator Inference: dz/dt = A_r z + H_r (z x z) + B_r w, z = Psi^T q, fitted to
the seven impulse responses of eq. (5.5).  The cavity nonlinearity is quadratic
(p = 2), and the observable is the whole state, y = q, so C = I throughout --
passing C = None means exactly that and avoids ever forming a 19800 x 19800
identity.
"""

import itertools
import time as timer

import numpy as np
import scipy.linalg as sla
from scipy.integrate import solve_ivp


def apply_C(C, X):
    """C X, with C = None meaning the identity (y = q, as in section 5)."""
    return X if C is None else C @ X


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
    """Leading r POD modes.  With N = 19800 and ~2800 snapshots the economy SVD
    of the stacked snapshot matrix (~450 MB) is cheaper than the method of
    snapshots plus a lift, so it is done directly."""
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
    """dz/dt = A_r z + c^(p-1) H_r z^p (+ f_r(t)).  For trajectories normalized
    by their IC amplitude c the degree-p term picks up c^(p-1); in physical
    coordinates c = 1, which is how train_models.py uses it."""

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
    """Fit dz/dt = A_r z (+ c^(p-1) H_r z^p) to z = Psi^T x by least squares; only
    the nonlinear block is (Tikhonov) regularized.  dz/dt from 4th-order finite
    differences.

    The fit uses the original trajectories: checkpointing belongs to the
    balancing, not here.

    With `normalize_energy` (default) each trajectory's residual is weighted by
    1/E_j with E_j = <||C x||^2> its time-averaged energy -- "we normalize the
    trajectories by their time-averaged energy" in section 5.1.  Here C = I, so
    this is the state energy, and it matters a lot: the beta = 1 impulse carries
    ~10^4 times the energy of the beta = 0.01 one, so an unweighted fit would
    see only the largest few.
    """
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
        w = np.concatenate([np.full(Zj.shape[1] - 4,
                                    1.0/np.sqrt(np.mean(np.sum(apply_C(C, Xj)**2, axis=0))))
                            for Xj, Zj in zip(trajs, Zs)])
        D, dZ = D*w, dZ*w
    d = D.shape[0]
    Gam = np.zeros(d)
    Gam[r:] = np.sqrt(reg)
    A = np.concatenate((D.T, np.diag(Gam)))
    b = np.concatenate((dZ.T, np.zeros((d, r))))
    # gelsy (QR with column pivoting) instead of the default gelsd (SVD): the
    # Tikhonov block makes the system full rank, so the two agree, but gelsd
    # intermittently fails to converge on the badly scaled balanced-basis
    # feature matrices at large lambda.
    try:
        O = sla.lstsq(A, b, lapack_driver="gelsy")[0].T
    except np.linalg.LinAlgError:
        O = sla.lstsq(A, b, lapack_driver="gelsd")[0].T
    return ROM(O[:, :r], O[:, r:] if p > 1 else None, p)


# The paper uses lambda = 1e-3; the grid brackets it by five decades either way.
REG_GRID = 10.0**np.arange(-8, 5, 1.0)


def nitrom_cost(rom, Phi, Psi, trajs, time, C=None, amps=None):
    """
    NiTROM trajectory cost, eq. (3.7):

        J = sum_j (1/alpha_j) sum_i ||y^(j)(t_i) - C Phi z^(j)(t_i)||^2,

    with alpha_j the time-averaged energy of trajectory j.  Each ratio is scale
    invariant, so physical or amplitude-normalized coordinates give the same
    value.  The constant N_traj * N of (3.7) is omitted: it does not affect
    which model wins.

    Returns inf if the ROM blows up, so the caller can reject it.
    """
    amps = np.ones(len(trajs)) if amps is None else amps
    J = 0.0
    for Xj, c in zip(trajs, amps):
        Z = rom.integrate(Psi.T @ Xj[:, 0], time[:Xj.shape[1]], c)
        if not np.all(np.isfinite(Z)):
            return np.inf
        yh = apply_C(C, Phi @ Z)
        y = apply_C(C, Xj)
        J += np.sum((y - yh)**2)/np.mean(np.sum(y**2, axis=0))
    return J


def opinf_select(trajs, Phi, Psi, dt, time, amps, C=None, model="quadratic",
                 grid=REG_GRID, verbose=False):
    """OpInf with the nonlinear-term regularization chosen from `grid` by the
    NiTROM trajectory cost (3.7) on the training data -- the same quantity
    NiTROM itself minimizes, and normalized per trajectory by its average
    energy, so it is consistent with the 1/E_j weighting inside opinf().
    Testing data are never used.  Returns (rom, reg, J, state error)."""
    def score(rom):
        """(NiTROM cost, relative state error) from a single set of ROM solves."""
        J = num = den = 0.0
        for Xj, c in zip(trajs, amps):
            Z = rom.integrate(Psi.T @ Xj[:, 0], time[:Xj.shape[1]], c)
            if not np.all(np.isfinite(Z)):
                return np.inf, np.inf
            Xh = Phi @ Z
            y, yh = apply_C(C, Xj), apply_C(C, Xh)
            J += np.sum((y - yh)**2)/np.mean(np.sum(y**2, axis=0))
            num += np.linalg.norm(Xj - Xh)**2
            den += np.linalg.norm(Xj)**2
        return J, np.sqrt(num/den)

    amps = np.ones(len(trajs)) if amps is None else amps
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
        if not np.all(np.isfinite(Z)):
            errs.append(np.inf)
            continue
        ej = np.linalg.norm(Xj - Phi @ Z)
        num, den = num + ej**2, den + np.linalg.norm(Xj)**2
        errs.append(ej/np.linalg.norm(Xj))
    return np.sqrt(num/den) if den else np.inf, np.array(errs), Zs
