"""Vorticity of the balanced trial (Phi) and test (Psi) modes.

compute_balancing.py builds (Phi, Psi) for the weighted state q~ = W^{1/2} q.
Both are shown as physical fields on the ROM domain by undoing the weight:

    phi = W^{-1/2} Phi~   -- the velocity field the mode adds to q;
    psi = W^{-1/2} Psi~   -- the field whose W-inner product with q gives the
                             latent coordinate, z = Psi~^T q~ = <psi, q>_W.

So phi shows where a mode lives and psi where a perturbation has to be to
excite it.  Each panel has its own symmetric colour scale.

Usage: python plot_balanced_modes.py [--modes 1 2 3 4 10 20]
Writes figures/airfoil_balanced_modes.png.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseflow import rd_bu_r_with_white_center
from rom_domain import X_BOUNDS, Y_BOUNDS, crop_mask, vorticity

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balancing", default=os.path.join(HERE, "balancing"))
    ap.add_argument("--modes", type=int, nargs="+", default=[1, 2, 3, 4, 10, 20],
                    help="1-based mode numbers")
    ap.add_argument("--xlim", type=float, nargs=2, default=X_BOUNDS)
    ap.add_argument("--ylim", type=float, nargs=2, default=(-1.5, 1.5))
    args = ap.parse_args()

    bal = np.load(os.path.join(args.balancing, "balancing.npz"), allow_pickle=True)
    Sig, sqrt_w = bal["Sigma"], bal["sqrt_w"]
    with np.load(str(bal["baseflow"])) as snap:
        coords = {c: snap[c] for c in ("xu", "yu", "xv", "yv", "xi", "eta")}
    dmask = crop_mask(coords)
    full = bal["Phi"].shape[0] == dmask.size     # full-mesh balancing?
    cmap = rd_bu_r_with_white_center()

    n = len(args.modes)
    fig, axes = plt.subplots(n, 2, figsize=(16, 1.1 + 2.0*n),
                             constrained_layout=True, squeeze=False)
    for row, m in enumerate(args.modes):
        for col, (name, key) in enumerate((("\\Phi", "Phi"), ("\\Psi", "Psi"))):
            field = np.ascontiguousarray(bal[key][:, m - 1])/sqrt_w
            X, Y, om = vorticity(field[dmask] if full else field, coords)
            view = ((X >= args.xlim[0]) & (X <= args.xlim[1])
                    & (Y >= args.ylim[0]) & (Y <= args.ylim[1]))
            lim = float(np.percentile(np.abs(om[view]), 99.5)) or 1.0
            ax = axes[row, col]
            ax.pcolormesh(X, Y, om, cmap=cmap, vmin=-lim, vmax=lim,
                          shading="auto", rasterized=True)
            ax.fill(coords["xi"], coords["eta"], color="k", zorder=5)
            ax.set_xlim(args.xlim)
            ax.set_ylim(args.ylim)
            ax.set_aspect("equal")
            ax.set_title(rf"${name}_{{{m}}}$" + (rf"  ($\sigma_{{{m}}} = "
                         rf"{Sig[m-1]:.3g}$)" if col == 0 else ""),
                         fontsize=11, loc="left")
    fig.suptitle("Balanced modes (vorticity; trial $\\Phi$ left, test $\\Psi$ "
                 "right)", fontsize=13)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.balancing))
    out = os.path.join(HERE, "figures", f"airfoil_modes_{tag}_"
                       f"{min(args.modes)}-{max(args.modes)}.png")
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
