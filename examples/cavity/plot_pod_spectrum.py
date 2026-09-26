"""
POD spectrum of the energy-normalized training data, for choosing the
pre-projection subspace.

Why pre-project.  NiTROM optimizes Phi on Grassmann and Psi on Stiefel in the
ambient R^19800, where nothing constrains the iterates to be divergence free.
The training snapshots *are* divergence free, so any basis built from them spans
a divergence-free subspace; restricting the whole pipeline -- balancing, OpInf,
NiTROM -- to the span of the leading POD modes therefore enforces (5.2) by
construction.  This is what section 5.1 of the paper does with 200 modes
(>99.99% of the variance).

Each trajectory is first divided by the square root of its time-averaged energy,
so every impulse contributes equally regardless of beta.  Without that, the
beta = +-1 responses carry ~10^4 times the energy of the beta = 0.01 one and the
POD modes describe only the large-amplitude behaviour.

Writes figures/pod_spectrum.png and pod_prespectrum.npz (singular values).

Usage: python plot_pod_spectrum.py [--r-pre 200]
"""

import argparse
import os

import numpy as np
import scipy.linalg as sla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cavity import Cavity
from train_models import load_train, HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-pre", type=int, default=200,
                    help="pre-projection rank to mark on the plot")
    args = ap.parse_args()

    trajs_n, betas, time = load_train()
    trajs = [abs(b)*T for b, T in zip(betas, trajs_n)]
    # normalize each trajectory by its time-averaged energy, so all amplitudes
    # weigh equally (the same 1/alpha_j convention as the NiTROM cost)
    alpha = np.array([np.mean(np.sum(X**2, axis=0)) for X in trajs])
    Xn = [X/np.sqrt(a) for X, a in zip(trajs, alpha)]
    print(f"{len(trajs)} trajectories, time-averaged energies "
          f"{np.array2string(alpha, precision=2)}")
    print(f"  spread before normalization: {alpha.max()/alpha.min():.1f}x, "
          f"after: 1.0x")

    A = np.concatenate(Xn, axis=1)
    print(f"snapshot matrix {A.shape}; computing the SVD ...", flush=True)
    U, s, _ = sla.svd(A, full_matrices=False)

    e = np.cumsum(s**2)/np.sum(s**2)
    print(f"\n{'variance kept':>14}{'modes needed':>14}")
    for f in (0.9, 0.99, 0.999, 0.9999, 0.99999):
        k = int(np.searchsorted(e, f)) + 1
        print(f"{f:>14.5%}{k:>14d}")
    print(f"\nat r = {args.r_pre}: {e[args.r_pre-1]:.6%} of the variance, "
          f"sigma_{args.r_pre}/sigma_1 = {s[args.r_pre-1]/s[0]:.2e}")

    # divergence of the leading modes: they should inherit it from the data
    cav = Cavity()
    divs = [cav.divergence(U[:, i]) for i in (0, 9, 49, 99, args.r_pre - 1)]
    print(f"relative divergence of modes 1, 10, 50, 100, {args.r_pre}: "
          + ", ".join(f"{d:.1e}" for d in divs))

    np.savez(os.path.join(HERE, "pod_prespectrum.npz"), s=s, alpha=alpha)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    k = np.arange(1, len(s) + 1)
    ax[0].semilogy(k, s/s[0], color="tab:blue", lw=1.4)
    ax[0].axvline(args.r_pre, color="k", ls="--", lw=1,
                  label=rf"$r_{{\rm pre}} = {args.r_pre}$")
    ax[0].set_xlabel("index $k$")
    ax[0].set_ylabel(r"$\sigma_k\,/\,\sigma_1$")
    ax[0].set_title("POD singular values, energy-normalized data", fontsize=10)
    ax[0].legend(fontsize=9)

    ax[1].semilogy(k, 1 - e, color="tab:red", lw=1.4)
    for f, c in ((1e-3, "0.6"), (1e-4, "0.4")):
        ax[1].axhline(f, color=c, ls=":", lw=1)
        ax[1].text(len(s)*0.55, f*1.3, f"{1-f:.2%} of the variance",
                   fontsize=8, color=c)
    ax[1].axvline(args.r_pre, color="k", ls="--", lw=1)
    ax[1].set_xlabel("index $k$")
    ax[1].set_ylabel(r"$1 - \sum_{j\leq k}\sigma_j^2/\sum_j\sigma_j^2$")
    ax[1].set_title("variance not captured", fontsize=10)
    for a in ax:
        a.grid(alpha=0.3, which="both")
        a.set_xlim(1, len(s))
    fig.suptitle(f"Cavity: POD spectrum of the {len(trajs)} energy-normalized "
                 f"training impulses ({A.shape[1]} snapshots)", fontsize=11)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", "pod_spectrum.png")
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
