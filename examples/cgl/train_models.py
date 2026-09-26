"""
r = 5 cubic OpInf ROMs for the CGL, trained on all amplitude-normalized
training trajectories (data/, from generate_data.py):

  Balanced (Lall-averaged)  : amplitude families keyed on the parent trajectory,
                              Gramians averaged over them
  POD                       : leading POD modes of the training trajectories

The balancing itself comes from nitrom.utils.compute_data_driven_balancing; only
the Operator Inference pieces are local (balanced_opinf.py).

Saves roms_r5.npy (dict name -> Phi, Psi, Ar, Hr, reg, Sigma) and figures/modes_r5.png.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cgl import CGL
from balanced_opinf import check_balancing, pod_basis, opinf_select

from nitrom.backend import set_backend
from nitrom.utils import compute_data_driven_balancing

set_backend("numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
R = 5
M_CKPT, Q_CKPT = 10, 5             # snapshots 0, 5, ..., 45 (t = 0, 1, ..., 9)
# rank_M is left to the default: M's spectrum has no clean gap here, so the cut
# is made by relative threshold (rcond = 1e-4) rather than at a fixed rank.


class Pool:
    """Minimal TrainingPool view: compute_data_driven_balancing needs only the
    raw trajectories, the time grid and the (single-rank) layout.  A real
    TrainingPool would also synthesize and cache time derivatives on disk, which
    the balancing never uses."""

    def __init__(self, X, time):
        self.X, self.time = X, time
        self.world_size, self.rank, self.comm = 1, 0, None
        self.N = X.shape[1]


def load_train():
    alphas = np.load(os.path.join(DATA, "train_alpha.npy"))
    trajs = [np.load(os.path.join(DATA, f"train_traj_{j:03d}.npy")) for j in range(len(alphas))]
    return trajs, alphas, np.load(os.path.join(DATA, "time.npy"))


def main():
    trajs, alphas, time = load_train()
    dt = time[1] - time[0]
    # The trajectories are stored divided by their IC amplitude; the family
    # binning keys off the *raw* norm, so undo that scaling for the pool.
    X_raw = np.stack([a*T for a, T in zip(alphas, trajs)])

    bases = {}
    # Families keyed on the parent trajectory: every checkpoint stays with the
    # amplitude its parent was launched at, rather than migrating to a lower
    # family as it decays (which binning by the checkpoint's own norm would do).
    groups = [list(np.where(alphas == a)[0]) for a in np.unique(alphas)]
    Phi, Psi, Sig, info = compute_data_driven_balancing(
        Pool(X_raw, time), n_checkpoints=M_CKPT,
        checkpoint_stride=Q_CKPT, families=groups,
    )
    print(f"Balanced (Lall-averaged), families by parent alpha "
          f"{[float(a) for a in np.unique(alphas)]}: "
          f"m = {[f['n_ic'] for f in info['families']]}, "
          f"M ranks kept {[f['rank_M_kept'] for f in info['families']]}")
    print(f"   Sigma[:8] = {np.array2string(Sig[:8], precision=1)}; checks "
          + ", ".join(f"{k} {v:.1e}" for k, v in check_balancing(Phi, Psi, Sig, info, 30).items()))
    bases["Balanced (Lall-averaged)"] = (Phi[:, :R], Psi[:, :R], Sig)

    U = pod_basis(trajs, R)
    bases["POD"] = (U, U, None)

    # Output operator y = C q: Gaussian sensor at branch II, eq. (4.2).
    cgl_ = CGL()
    g = np.exp(-((cgl_.x + cgl_.xbar)/cgl_.s)**2)
    C = np.zeros((2, 2*cgl_.nx))
    C[0, :cgl_.nx], C[1, cgl_.nx:] = g, g

    roms = {}
    print(f"\nr = {R} cubic OpInf (regularization chosen by the NiTROM cost (3.7))")
    for name, (Ph, Ps, Sig) in bases.items():
        rom, reg, J, e = opinf_select(trajs, Ph, Ps, dt, time, alphas, C, "cubic")
        roms[name] = dict(Phi=Ph, Psi=Ps, Ar=rom.Ar, Hr=rom.Hr if rom.Hr is not None else np.zeros((R, 0)),
                          reg=reg, Sigma=Sig)
        print(f"   {name:<26} reg {reg:.0e}, NiTROM cost {J:.4e}, state error {e:.3f}")
    np.save(os.path.join(HERE, f"roms_r{R}.npy"), roms, allow_pickle=True)

    # Modes
    cgl = CGL()
    nx, x = cgl.nx, cgl.x
    cols = [(r"$\Psi$ " + n, b[1]) for n, b in bases.items() if n != "POD"] + \
           [(r"$\Phi$ " + n, b[0]) for n, b in bases.items() if n != "POD"] + [("POD modes", bases["POD"][0])]
    fig, ax = plt.subplots(R, len(cols), figsize=(4*len(cols), 1.6*R), sharex=True)
    for c, (title, V) in enumerate(cols):
        for i in range(R):
            v = V[:, i]/np.abs(V[:, i]).max()
            a = ax[i, c]
            a.plot(x, v[:nx], "tab:blue", lw=1.1, label="Re")
            a.plot(x, v[nx:], "tab:orange", lw=1.1, label="Im")
            a.plot(x, np.hypot(v[:nx], v[nx:]), "k", lw=0.7, label="|.|")
            for xb in [cgl.xbar, -cgl.xbar]:
                a.axvline(xb, color="gray", ls="--", lw=0.6)
            a.set_xlim(-30, 30); a.set_yticks([])
            if c == 0:
                a.set_ylabel(f"mode {i+1}")
            if i == 0:
                a.set_title(title, fontsize=9)
    ax[0, 0].legend(fontsize=6, loc="upper right")
    for a in ax[-1]:
        a.set_xlabel("$x$")
    fig.suptitle(f"r = {R} bases (dashed: branches I, II)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", f"modes_r{R}.png"), dpi=130)


if __name__ == "__main__":
    main()
