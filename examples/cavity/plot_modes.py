"""
Balanced modes of the cavity: vorticity of the leading direct (Phi) and adjoint
(Psi) modes from compute_data_driven_balancing.

The pair is biorthogonal, Psi^T Phi = I, and the two carry different
information: Phi spans the directions the flow actually evolves along (the
controllable subspace), while Psi says which directions must be *measured* to
predict that evolution (the observable subspace).  For a strongly non-normal
flow like the lid-driven cavity at Re = 8300 they look nothing alike -- Psi
concentrates upstream of Phi, near where perturbations must be seeded to be
amplified -- and that mismatch is exactly what an orthogonal (POD) projection
cannot represent.

Also saves the balancing to balancing_r{R}.npy so train_models.py-scale runs do
not have to recompute it.

Writes figures/balanced_modes_cavity.png.
"""

import argparse
import os
import resource
import time as timer

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cavity import Cavity
from compute_preprojection import load_preproj
from train_models import load_train, Pool, vorticity, M_CKPT, Q_CKPT, R, HERE
from balanced_opinf import check_balancing

from nitrom.backend import set_backend
from nitrom.utils import compute_data_driven_balancing

set_backend("numpy")

N_SHOW = 10             # leading modes to plot
NCOL = 5


def step(msg, t0):
    """Progress line with elapsed time and peak RSS.  The balancing holds every
    checkpoint window in memory at once, so watching RSS is the way to tell a
    slow run from one that has started swapping."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9
    print(f"[{timer.time() - t0:6.1f} s | peak {rss:5.2f} GB] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=M_CKPT,
                    help="checkpoints per trajectory (default: %(default)s)")
    ap.add_argument("--q", type=int, default=Q_CKPT,
                    help="snapshot stride between checkpoints (default: %(default)s)")
    ap.add_argument("--pre", type=int, default=0,
                    help="build the balancing in the span of this many POD "
                         "pre-projection modes instead of the raw R^19800.  "
                         "Phi and Psi are lifted back for plotting and saving, "
                         "and the reduced factors are stored alongside so "
                         "training can stay in the pre-projected coordinates.")
    ap.add_argument("--t0", type=float, default=0.0,
                    help="start the checkpoint set at this time instead of 0. "
                         "The t = 0 states are the raw actuator profile +-B, a "
                         "sharply localized structure unlike anything the flow "
                         "visits later and identical across trajectories; "
                         "keeping it in X_0 distorts the oblique projector.")
    args = ap.parse_args()
    m_ckpt, q_ckpt = args.m, args.q

    t0 = timer.time()
    trajs_n, betas, time = load_train()
    X = np.stack([abs(b)*T for b, T in zip(betas, trajs_n)])
    off = int(np.searchsorted(time, args.t0)) if args.t0 > 0 else 0
    if off:
        X, time = X[:, :, off:], time[off:]
        step(f"dropping the first {off} snapshot(s): X_0 now starts at "
             f"t = {time[0]:g}", t0)
    Upre = None
    if args.pre:
        Upre = load_preproj(args.pre)
        X = np.stack([Upre.T @ X[j] for j in range(X.shape[0])])
        step(f"pre-projected onto {args.pre} POD modes -> state dimension "
             f"{X.shape[1]}", t0)
    step(f"loaded {X.shape} training data", t0)
    n_ic = len(betas)*m_ckpt
    L = X.shape[2] - (m_ckpt - 1)*q_ckpt
    gb = n_ic*X.shape[1]*L*8/1e9
    step(f"balancing {len(betas)} trajectories x {m_ckpt} checkpoints "
         f"(stride {q_ckpt}) = {n_ic} ICs, window {L} of {X.shape[2]}", t0)
    step(f"  the window set needs ~{gb:.1f} GB; this is the long step, "
         f"no output until it finishes", t0)
    Phi, Psi, Sig, info = compute_data_driven_balancing(
        Pool(X, time), n_checkpoints=m_ckpt, checkpoint_stride=q_ckpt,
        families=[[j] for j in range(len(betas))],
    )
    step(f"balancing done: Phi {Phi.shape}, {len(Sig)} Hankel values", t0)
    del X
    Phi_red = Psi_red = None
    r_keep = min(R, len(Sig))
    print(f"M ranks kept {[f['rank_M_kept'] for f in info['families']]}")
    print(f"{len(Sig)} Hankel singular values; Sigma[:8] = "
          f"{np.array2string(Sig[:8], precision=3)}")
    print(f"Sigma_k/Sigma_1 at " + ", ".join(
        f"k={k}: {Sig[k-1]/Sig[0]:.2e}" for k in (10, 20, 30, 40, 50)
        if k <= len(Sig)))
    # the factors in `info` live in whatever space the balancing ran in, so the
    # checks must use the reduced Phi/Psi when pre-projecting
    Pc, Sc = Phi, Psi          # still in the balancing's own coordinates here
    print("checks: " + ", ".join(
        f"{k} {v:.1e}"
        for k, v in check_balancing(Pc, Sc, Sig, info, r_keep).items()))
    if Upre is not None:
        # keep the reduced factors for training, lift for plotting/evaluation
        Phi_red, Psi_red = Phi, Psi
        Phi, Psi = Upre @ Phi, Upre @ Psi
        step(f"lifted to the full state: Phi {Phi.shape}", t0)

    out_bal = dict(Phi=Phi[:, :r_keep], Psi=Psi[:, :r_keep], Sigma=Sig,
                   m_ckpt=m_ckpt, q_ckpt=q_ckpt, t0=args.t0, r_pre=args.pre)
    if Phi_red is not None:
        out_bal.update(Phi_red=Phi_red[:, :r_keep], Psi_red=Psi_red[:, :r_keep])
    name = (f"balancing_r{R}.npy" if not args.pre
            else f"balancing_pre{args.pre}_r{R}.npy")
    np.save(os.path.join(HERE, name), out_bal, allow_pickle=True)
    step(f"saved {name} ({r_keep} modes); now plotting modes", t0)

    cav = Cavity()
    n = min(N_SHOW, Phi.shape[1])
    nrow = 2*int(np.ceil(n/NCOL))
    fig, ax = plt.subplots(nrow, NCOL, figsize=(2.6*NCOL, 2.7*nrow), squeeze=False)
    for i in range(n):
        for blk, (nm, V) in enumerate([(r"\psi", Psi), (r"\phi", Phi)]):
            a = ax[blk*(nrow//2) + i//NCOL, i % NCOL]
            w = vorticity(cav.flow, V[:, i])
            lim = np.abs(w).max()
            a.imshow(w, origin="lower", extent=[0, 1, 0, 1], cmap="bwr",
                     vmin=-lim, vmax=lim)
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(rf"${nm}_{{{i+1}}}$  ($\sigma_{{{i+1}}}$ = {Sig[i]:.2f})",
                        fontsize=9)
    for a in ax.flat:
        if not a.images:
            a.axis("off")
    fig.suptitle(rf"Cavity balanced modes (vorticity), {m_ckpt} checkpoints/trajectory: "
                 rf"rows 1-2 adjoint $\Psi$, rows 3-4 direct $\Phi$", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", "balanced_modes_cavity"
                       + (f"_pre{args.pre}" if args.pre else "") + ".png")
    fig.savefig(out, dpi=130)
    step(f"saved -> {out}", t0)


if __name__ == "__main__":
    main()
