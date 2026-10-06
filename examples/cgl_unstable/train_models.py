"""
Cubic OpInf ROMs for the unstable CGL (mu0 = 0.41), in x - m coordinates:

  Vanilla  : POD basis of the training trajectories (Phi = Psi)
  Balanced : Lall-averaged data-driven balancing
             (nitrom.utils.compute_data_driven_balancing), one family per
             impulse amplitude, checkpoints every 1 time unit for t <= 9

Both fit dz/dt = A_r z + H_r (z x z x z) by least squares on z = Psi^T x.
Two choices of the time derivative (--derivs):

  reproj (default): re-projection, dz/dt = Psi^T f(m + D z) with the decoder
      D = Phi (Psi^T Phi)^-1 and f the CGL right-hand side, evaluated at every
      projected training snapshot.  The data are then Markovian in z and, the
      model class containing the projected dynamics, the unregularized fit
      recovers the intrusive Petrov-Galerkin ROM Psi^T f(m + D z) (up to the
      O(||m||) constant/quadratic terms a cubic-only model cannot carry, ~1e-5
      here).  No regularization; the recovery is checked against the intrusive
      right-hand side at random z.
  fd: 4th-order finite differences of Psi^T x, the cubic block
      Tikhonov-regularized and the regularization chosen from a grid by the
      trajectory cost on the training data (balanced_opinf.opinf_select of
      examples/cgl).  The model is cubic only -- no constant or quadratic term --
which is consistent because m ~ 0 (the cycle is a rotating wave; see
generate_data.py): in x - m coordinates with m = 0 the CGL is exactly
linear + cubic.

Unlike examples/cgl, the trajectories are in physical units and the output is
the whole state (C = I): the question here is whether the ROMs reproduce the
state on the limit cycle and on the transients towards it, not a sensor
signal.

Usage: python train_models.py [--r 5] [--derivs reproj|fd]
Saves roms_r<r>_<derivs>.npy (dict name -> Phi, Psi, Ar, Hr, reg, Sigma).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "cgl"))  # after this dir: local names win
from balanced_opinf import (ROM, check_balancing, nitrom_cost, opinf_select,  # noqa: E402
                            pod_basis, poly_features, poly_index, rel_error)
from cgl import CGL  # noqa: E402

from nitrom.backend import set_backend  # noqa: E402
from nitrom.utils import compute_data_driven_balancing  # noqa: E402

set_backend("numpy")

DATA = os.path.join(HERE, "data")
M_CKPT, Q_CKPT = 10, 5             # checkpoints at t = 0, 1, ..., 9


class Pool:
    """Minimal TrainingPool view for compute_data_driven_balancing."""

    def __init__(self, X, time):
        self.X, self.time = X, time
        self.world_size, self.rank, self.comm = 1, 0, None
        self.N = X.shape[1]


def opinf_reproj(trajs, Phi, Psi, m, cgl):
    """Unregularized cubic OpInf on re-projected data: z = Psi^T x,
    dz/dt = Psi^T f(m + D z), D = Phi (Psi^T Phi)^-1.  Returns (rom, D)."""
    r = Psi.shape[1]
    D = Phi @ np.linalg.inv(Psi.T @ Phi)
    idx = poly_index(r, 3)
    Z = np.concatenate([Psi.T @ X for X in trajs], axis=1)
    Q = m[:, None] + D @ Z
    F = Psi.T @ (cgl.A @ Q + cgl.nonlinearity(Q))
    Dat = np.concatenate((Z, poly_features(Z, idx)), axis=0)
    O = np.linalg.lstsq(Dat.T, F.T, rcond=None)[0].T
    return ROM(O[:, :r], O[:, r:], 3), D, np.linalg.cond(Dat)


def pg_check(rom, Psi, D, m, cgl, trajs, n=200, seed=0):
    """Relative mismatch between the fitted and the intrusive PG right-hand
    side at random z of the size of the training data."""
    Z = np.concatenate([Psi.T @ X for X in trajs], axis=1)
    rng = np.random.default_rng(seed)
    Zr = Z[:, rng.integers(Z.shape[1], size=n)] * rng.uniform(0.5, 1.5, n)
    Q = m[:, None] + D @ Zr
    F = Psi.T @ (cgl.A @ Q + cgl.nonlinearity(Q))
    Fh = np.stack([rom.rhs(0.0, z) for z in Zr.T], axis=1)
    return np.linalg.norm(F - Fh)/np.linalg.norm(F)


def load(kind="train"):
    alphas = np.load(os.path.join(DATA, f"{kind}_alpha.npy"))
    trajs = [np.load(os.path.join(DATA, f"{kind}_traj_{j:03d}.npy"))
             for j in range(len(alphas))]
    labels = np.load(os.path.join(DATA, f"{kind}_labels.npy"))
    return trajs, alphas, labels, np.load(os.path.join(DATA, "time.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--derivs", choices=["reproj", "fd"], default="reproj")
    args = ap.parse_args()
    R = args.r

    trajs, alphas, _, time = load("train")
    dt = time[1] - time[0]
    X = np.stack(trajs)
    C = np.eye(X.shape[1])                           # output = whole state
    print(f"{len(trajs)} training trajectories, N = {X.shape[1]}, "
          f"{len(time)} snapshots, dt = {dt:g}, amplitudes {sorted(set(alphas))}")

    bases = {}
    groups = [list(np.where(alphas == a)[0]) for a in np.unique(alphas)]
    Phi, Psi, Sig, info = compute_data_driven_balancing(
        Pool(X, time), n_checkpoints=M_CKPT, checkpoint_stride=Q_CKPT,
        families=groups)
    print(f"Balanced (Lall-averaged), families by amplitude: m = "
          f"{[f['n_ic'] for f in info['families']]}, M ranks kept "
          f"{[f['rank_M_kept'] for f in info['families']]}")
    print(f"   Sigma[:8] = {np.array2string(Sig[:8], precision=1)}; checks "
          + ", ".join(f"{k} {v:.1e}" for k, v in
                      check_balancing(Phi, Psi, Sig, info, 3*R).items()))
    bases["Balanced"] = (Phi[:, :R], Psi[:, :R], Sig)
    U = pod_basis(trajs, R)
    bases["Vanilla (POD)"] = (U, U, None)

    roms = {}
    m = np.load(os.path.join(DATA, "setup.npz"))["m"]
    cgl = CGL(mu0=float(np.load(os.path.join(DATA, "setup.npz"))["mu0"]))
    print(f"\nr = {R} cubic OpInf, derivatives: {args.derivs}")
    for name, (Ph, Ps, Sg) in bases.items():
        if args.derivs == "reproj":
            rom, D, cond = opinf_reproj(trajs, Ph, Ps, m, cgl)
            reg = 0.0
            J = nitrom_cost(rom, D, Ps, trajs, time, C)
            e = rel_error(rom, D, Ps, trajs, time)[0] if np.isfinite(J) else np.inf
            print(f"   {name:<14} data matrix cond {cond:.1e}; |rhs - PG rhs|/|PG rhs| "
                  f"at random z: {pg_check(rom, Ps, D, m, cgl, trajs):.1e}")
        else:
            rom, reg, J, e = opinf_select(trajs, Ph, Ps, dt, time, None, C, "cubic")
            if rom is None:
                print(f"   {name:<14} every regularization blew up on the training data")
                continue
        ev = np.linalg.eigvals(rom.Ar)
        lead = ev[np.argmax(ev.real)]
        roms[name] = dict(Phi=Ph, Psi=Ps, Ar=rom.Ar, Hr=rom.Hr, reg=reg, Sigma=Sg,
                          J=J, err=e)
        print(f"   {name:<14} reg {reg:.0e}, trajectory cost {J:.4e}, state error "
              f"{e:.3f}; leading eig of A_r {lead.real:+.4f} {lead.imag:+.4f}i")
    out = f"roms_r{R}_{args.derivs}.npy"
    np.save(os.path.join(HERE, out), roms, allow_pickle=True)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
