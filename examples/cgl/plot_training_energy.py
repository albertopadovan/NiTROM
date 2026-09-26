"""
Energy of the NiTROM training set over time, before and after the per-trajectory
energy normalization of the cost (3.7).

The training set is what train_nitrom.py actually feeds the optimizer: the 8
impulse responses of eq. (4.5), each cut into M_CKPT sliding windows of equal
length (checkpoint_windows), every window treated as starting at t = 0.

Left column:  raw energy.  The windows span orders of magnitude -- both because
              beta ranges over {-1, 0.01, 0.1, 1} and because a window that
              starts late begins after much of the transient has already
              decayed.  An unweighted cost would be dominated by the largest.
Right column: the same divided by that window's own time average, which is the
              1/alpha_j weighting in (3.7).  Every window then carries the same
              total weight, so late, low-amplitude windows count as much as the
              energetic early ones.

Top row is the state energy ||x(t)||^2; bottom row the output energy
||y(t)||^2 = ||C x(t)||^2, which is the quantity the cost is actually built
from (alpha_j is the time average of the bottom-left curves).

Writes figures/training_energy.png.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cgl import CGL
from train_nitrom import checkpoint_windows, M_CKPT, Q_CKPT, N_SNAP

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def main():
    cgl = CGL()
    nx = cgl.nx
    g = np.exp(-((cgl.x + cgl.xbar)/cgl.s)**2)          # sensor at branch II
    C = np.zeros((2, 2*nx))
    C[0, :nx], C[1, nx:] = g, g

    alphas = np.load(os.path.join(DATA, "train_alpha.npy"))
    t_full = np.load(os.path.join(DATA, "time.npy"))[:N_SNAP]
    # Physical (un-normalized) trajectories, as train_nitrom.py builds them.
    X = np.stack([a*np.load(os.path.join(DATA, f"train_traj_{j:03d}.npy"))[:, :N_SNAP]
                  for j, a in enumerate(alphas)])

    W, L = checkpoint_windows(X, M_CKPT, Q_CKPT)
    t = t_full[:L]
    parent = np.repeat(alphas, M_CKPT)                  # beta of each window
    ckpt = np.tile(np.arange(M_CKPT), len(alphas))      # which checkpoint

    Ex = np.sum(W**2, axis=1)                           # ||x(t)||^2 per window
    Ey = np.sum(np.matmul(C, W)**2, axis=1)             # ||y(t)||^2 per window
    alpha = Ey.mean(axis=1)                             # alpha_j of eq. (3.7)

    uniq = np.unique(alphas)
    colors = dict(zip(uniq, plt.cm.viridis(np.linspace(0, 0.85, len(uniq)))))

    print(f"NiTROM training set: {W.shape[0]} windows "
          f"({len(alphas)} trajectories x {M_CKPT} checkpoints, stride {Q_CKPT}), "
          f"{L} snapshots each, t in [0, {t[-1]:g}]")
    # y(0) is ~0 (the sensor is at branch II, the IC at branch I), so peak
    # growth is reported against the window's own time average, not against y(0).
    print(f"\n{'beta':>6} {'<||x||^2>':>12} {'<||y||^2> = alpha_j':>22} "
          f"{'peak ||y||^2/alpha_j':>22}")
    for a in uniq:
        s = parent == a
        print(f"{a:>6g} {Ex[s].mean():>12.4e} {alpha[s].mean():>22.4e} "
              f"{(Ey[s].max(axis=1)/alpha[s]).mean():>22.1f}")
    print(f"\nalpha_j spread over all windows: {alpha.max()/alpha.min():>8.2f}x")
    print(f"  ... by amplitude only (checkpoint 0): "
          f"{alpha[ckpt == 0].max()/alpha[ckpt == 0].min():.2f}x")
    print(f"  ... by checkpoint only (beta = 1):    "
          f"{alpha[parent == 1].max()/alpha[parent == 1].min():.4f}x")
    tot = np.sum(Ey, axis=1)/alpha
    print(f"after 1/alpha_j weighting, per-window totals span "
          f"{tot.max()/tot.min():.3f}x (1.0 = equal weight)")

    fig, ax = plt.subplots(2, 2, figsize=(13, 7.6), sharex=True,
                           constrained_layout=True)
    seen = set()
    for j in range(W.shape[0]):
        a = parent[j]
        lab = rf"$\beta = {a:g}$" if a not in seen else None
        seen.add(a)
        # beta = -1 and beta = +1 have identical energies (the cubic is odd, so
        # x_{-beta} = -x_{beta}); dash the former so it is not hidden underneath.
        kw = dict(color=colors[a], lw=0.9, alpha=0.75,
                  ls="--" if a < 0 else "-")
        ax[0, 0].semilogy(t, Ex[j], label=lab, **kw)
        ax[0, 1].semilogy(t, Ex[j]/Ex[j].mean(), **kw)
        ax[1, 0].semilogy(t, Ey[j], **kw)
        ax[1, 1].semilogy(t, Ey[j]/alpha[j], **kw)
    ax[0, 0].set_ylabel(r"$\|x(t)\|^2$")
    ax[0, 0].set_title("state energy (raw)", fontsize=10)
    ax[0, 1].set_title(r"state energy $/\ \langle\|x\|^2\rangle_t$", fontsize=10)
    ax[1, 0].set_ylabel(r"$\|y(t)\|^2$")
    ax[1, 0].set_title("output energy (raw)", fontsize=10)
    ax[1, 1].set_title(r"output energy $/\ \alpha_j$ -- the weighting in (3.7)",
                       fontsize=10)
    # y(0) is zero to roundoff (~1e-45: the sensor at branch II sees nothing of
    # an IC at branch I), so an automatic log range would span 50 decades and
    # hide everything.  Clip the output panels to 8 decades below their peak.
    for a_, dat in ((ax[1, 0], Ey), (ax[1, 1], Ey/alpha[:, None])):
        hi = dat.max()
        a_.set_ylim(hi*1e-8, hi*3)
    for a_ in ax[-1]:
        a_.set_xlabel("$t$")
    for a_ in ax.flat:
        a_.set_xlim(0, t[-1])
        a_.grid(alpha=0.25, which="both")
    ax[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"NiTROM training set: {W.shape[0]} checkpoint windows, "
                 f"energy before and after the $1/\\alpha_j$ normalization",
                 fontsize=11)
    out = os.path.join(HERE, "figures", "training_energy.png")
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
