"""
r = 50 quadratic OpInf ROMs for the lid-driven cavity at Re = 8300
(Padovan, Vollmer & Bodony, SIADS 2024, section 5), trained on the seven
impulse responses of eq. (5.5) in data/ (generate_data_bopinf.py):

  Balanced (Lall-averaged)  : amplitude families keyed on the parent trajectory
                              (one per beta), Gramians averaged over them
  POD                       : leading POD modes of the training trajectories
                              -- the paper's own choice for OpInf

The balancing comes from nitrom.utils.compute_data_driven_balancing; only the
Operator Inference pieces are local (balanced_opinf.py).  The observable is the
whole state, y = q, so C = I and the energy weighting is the state energy.

Everything is done at the full state dimension N = 19800: the balancing never
forms an N x N Gramian, only the factors, so the memory cost is the checkpoint
windows themselves (~4.3 GB here).

Saves roms_r50.npy (dict name -> Phi, Psi, Ar, Hr, reg, Sigma).
"""

import os

import numpy as np

from balanced_opinf import check_balancing, pod_basis, opinf_select

from nitrom.backend import set_backend
from nitrom.utils import compute_data_driven_balancing

set_backend("numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
R = 50                             # paper: 50 POD modes, 99.6% of the variance
M_CKPT, Q_CKPT = 20, 2             # 20 ICs per trajectory, every 2nd snapshot
# rank_M is left to the default: M's spectrum has no clean gap here, so the cut
# is made by relative threshold (rcond = 1e-4) rather than at a fixed rank.


class Pool:
    """Minimal TrainingPool view: compute_data_driven_balancing needs only the
    raw trajectories, the time grid and the (single-rank) layout."""

    def __init__(self, X, time):
        self.X, self.time = X, time
        self.world_size, self.rank, self.comm = 1, 0, None
        self.N = X.shape[1]


def vorticity(flow, q):
    """Vorticity dv/dx - du/dy at interior cell corners, in physical orientation
    (the solver's y axis points down, hence the flip)."""
    U = q[:flow.szu].reshape(flow.rowsu, flow.colsu)
    V = q[flow.szu:].reshape(flow.rowsv, flow.colsv)
    dvdx = (V[:, 1:] - V[:, :-1])/flow.dx           # (Ny-1, Nx-1) at interior corners
    dudy = (U[1:, :] - U[:-1, :])/flow.dy
    return np.flipud(-(dvdx - dudy))


def load_train():
    betas = np.load(os.path.join(DATA, "setup.npz"))["betas"]
    trajs = [np.load(os.path.join(DATA, f"train_traj_{j:03d}.npy"))
             for j in range(len(betas))]
    return trajs, betas, np.load(os.path.join(DATA, "time.npy"))


def main():
    trajs_n, betas, time = load_train()
    dt = time[1] - time[0]
    # Stored trajectories are divided by |beta| (so every one starts at +-B).
    # Work in physical units: one (A_r, H_r) then serves every amplitude with no
    # c factor, and the family binning keys off the raw norm.
    trajs = [abs(b)*T for b, T in zip(betas, trajs_n)]
    X_raw = np.stack(trajs)
    print(f"{len(betas)} training trajectories, N = {X_raw.shape[1]}, "
          f"{X_raw.shape[2]} snapshots, dt = {dt:g}, betas = {list(betas)}")

    bases = {}
    # One family per *signed* beta.  The cavity nonlinearity is quadratic, so
    # the response to -beta is not the mirror of the response to +beta: the two
    # explore genuinely different regions of state space and carry different
    # observability energies, which is precisely what an amplitude family is
    # meant to separate.  Pooling them by |beta| would average over that
    # difference.  Here each beta is unique, so this is one family per
    # trajectory: 7 families of M_CKPT initial conditions.
    groups = [[j] for j in range(len(betas))]
    Phi, Psi, Sig, info = compute_data_driven_balancing(
        Pool(X_raw, time), n_checkpoints=M_CKPT,
        checkpoint_stride=Q_CKPT, families=groups,
    )
    print(f"Balanced (Lall-averaged), one family per signed beta: "
          f"m = {[f['n_ic'] for f in info['families']]}, "
          f"M ranks kept {[f['rank_M_kept'] for f in info['families']]}, "
          f"window {info['window_length']} of {X_raw.shape[2]}")
    print(f"   Sigma[:8] = {np.array2string(Sig[:8], precision=3)}; checks "
          + ", ".join(f"{k} {v:.1e}"
                      for k, v in check_balancing(Phi, Psi, Sig, info, R).items()))
    if len(Sig) < R:
        raise ValueError(
            f"only {len(Sig)} balanced modes exist (span of {M_CKPT} checkpoints "
            f"x {len(betas)} trajectories) but r = {R} was requested; increase "
            f"M_CKPT."
        )
    bases["Balanced (Lall-averaged)"] = (Phi[:, :R], Psi[:, :R], Sig)
    del X_raw                      # 445 MB copy; the window set is already freed

    U = pod_basis(trajs, R)
    bases["POD"] = (U, U, None)

    roms = {}
    print(f"\nr = {R} quadratic OpInf (regularization chosen by the NiTROM cost (3.7))")
    for name, (Ph, Ps, Sg) in bases.items():
        rom, reg, J, e = opinf_select(trajs, Ph, Ps, dt, time, None, None,
                                      "quadratic", verbose=True)
        roms[name] = dict(Phi=Ph, Psi=Ps, Ar=rom.Ar,
                          Hr=rom.Hr if rom.Hr is not None else np.zeros((R, 0)),
                          reg=reg, Sigma=Sg)
        print(f"   {name:<26} reg {reg:.0e}, NiTROM cost {J:.4e}, state error {e:.3f}")
    np.save(os.path.join(HERE, f"roms_r{R}.npy"), roms, allow_pickle=True)
    print(f"saved -> {os.path.join(HERE, f'roms_r{R}.npy')}")


if __name__ == "__main__":
    main()
