"""
Spectra behind the cavity balancing:

Left:  singular values of M = X_0^T X_0 for each amplitude family, normalized by
       the largest.  M is the Gram matrix of that family's initial conditions
       (M_CKPT checkpoints of one trajectory here), and M^+ is truncated at
       `rcond`; where that cut lands decides how much of span(X_0) survives into
       the restricted observability factor Q.  Checkpoints taken at nearby times
       are nearly collinear, so this spectrum decaying smoothly -- rather than
       dropping to zero -- is the normal case and the reason a threshold is
       needed at all.
Right: the Hankel singular values Sigma of the balancing.  Their count is capped
       by the number of initial conditions (M_CKPT x n_traj), which is what
       limits how large r can be.

Writes figures/spectra_cavity.png.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_models import load_train, Pool, M_CKPT, Q_CKPT, R, HERE

from nitrom.backend import set_backend
from nitrom.utils import compute_data_driven_balancing

set_backend("numpy")

RCOND = 1e-4            # the default used by compute_data_driven_balancing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=M_CKPT,
                    help="checkpoints per trajectory (default: %(default)s)")
    ap.add_argument("--q", type=int, default=Q_CKPT,
                    help="snapshot stride between checkpoints (default: %(default)s)")
    args = ap.parse_args()
    m_ckpt, q_ckpt = args.m, args.q

    trajs_n, betas, time = load_train()
    X = np.stack([abs(b)*T for b, T in zip(betas, trajs_n)])
    Phi, Psi, Sig, info = compute_data_driven_balancing(
        Pool(X, time), n_checkpoints=m_ckpt, checkpoint_stride=q_ckpt,
        families=[[j] for j in range(len(betas))],
    )
    del X

    fams = info["families"]
    print(f"{len(fams)} families of {m_ckpt} ICs (stride {q_ckpt}), "
          f"window {info['window_length']}")
    print(f"{'beta':>7} {'rank kept':>10} {'sM[last kept]/sM[0]':>21} {'sM[-1]/sM[0]':>14}")
    for b, f in zip(betas, fams):
        s = f["sM"]/f["sM"][0]
        print(f"{b:>7g} {f['rank_M_kept']:>10d} {s[f['rank_M_kept']-1]:>21.2e} "
              f"{s[-1]:>14.2e}")
    print(f"\n{len(Sig)} Hankel singular values; "
          f"Sigma[{R-1}]/Sigma[0] = {Sig[R-1]/Sig[0]:.2e}")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0, 0.88, len(fams)))
    for b, f, c in zip(betas, fams, colors):
        s = f["sM"]/f["sM"][0]
        ax[0].semilogy(np.arange(1, len(s) + 1), s, "o-", color=c, ms=4, lw=1.1,
                       label=rf"$\beta = {b:g}$")
        ax[0].semilogy(f["rank_M_kept"], s[f["rank_M_kept"] - 1], "x", color=c,
                       ms=10, mew=2)
    ax[0].axhline(RCOND, color="k", ls="--", lw=1,
                  label=rf"rcond = {RCOND:g}")
    ax[0].set_xlabel("index")
    ax[0].set_ylabel(r"$\sigma_k(M)\,/\,\sigma_1(M)$")
    ax[0].set_title(r"$M = X_0^\top X_0$ per family ($\times$ = last rank kept)",
                    fontsize=10)
    ax[0].legend(fontsize=8, ncol=2)

    ax[1].semilogy(np.arange(1, len(Sig) + 1), Sig/Sig[0], "o-",
                   color="tab:green", ms=4, lw=1.2)
    ax[1].axvline(R, color="k", ls="--", lw=1, label=rf"$r = {R}$")
    ax[1].set_xlabel("index $k$")
    ax[1].set_ylabel(r"$\sigma_k\,/\,\sigma_1$")
    ax[1].set_title(f"Hankel singular values ({len(Sig)} available, "
                    f"{m_ckpt}$\\times${len(betas)} ICs)", fontsize=10)
    ax[1].legend(fontsize=9)
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.suptitle("Cavity balancing spectra", fontsize=11)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", f"spectra_cavity_m{m_ckpt}_q{q_ckpt}.png")
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
