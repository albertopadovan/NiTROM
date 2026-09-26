"""Perturbation vorticity snapshots of a FOM training trajectory.

One figure per trajectory, one panel per requested time, over the ROM domain
D = [-4, 13] x [-3, 3] only (rom_domain.py).  The perturbation
decays by orders of magnitude over a run, so every panel has its own
symmetric colour scale (99.5th percentile of |omega| over D), printed in
the panel.

Usage: python plot_trajectory_snapshots.py [--traj 1 7]
                                           [--times 0 0.4 2 6 12 16 20 30]
Writes figures/airfoil_traj_%03d_snapshots.png.
"""

from __future__ import annotations

import argparse
import glob
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
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data"))
    ap.add_argument("--traj", type=int, nargs="+", default=None,
                    help="trajectory indices (default: all present)")
    ap.add_argument("--times", type=float, nargs="+",
                    default=[0.0, 0.4, 2.0, 6.0, 12.0, 16.0, 20.0, 30.0])
    ap.add_argument("--xlim", type=float, nargs=2, default=X_BOUNDS,
                    help="zoom window (colour scales are then set over it)")
    ap.add_argument("--ylim", type=float, nargs=2, default=Y_BOUNDS)
    ap.add_argument("--ncol", type=int, default=2)
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    args = ap.parse_args()

    meta = np.load(os.path.join(args.data, "meta.npz"))
    t, params = meta["time"], meta["parameters"]
    with np.load(str(meta["baseflow"])) as snap:
        coords = {c: snap[c] for c in ("xu", "yu", "xv", "yv", "xi", "eta")}
    mask = crop_mask(coords)

    trajs = args.traj or sorted(
        int(os.path.basename(f)[6:9])
        for f in glob.glob(os.path.join(args.data, "fluct_[0-9][0-9][0-9].npy")))
    cmap = rd_bu_r_with_white_center()
    idx = [int(np.argmin(np.abs(t - s))) for s in args.times]

    for k in trajs:
        Q = np.load(os.path.join(args.data, f"fluct_{k:03d}.npy"), mmap_mode="r")
        beta, xc, yc = params[k]
        ncol = args.ncol
        nrow = -(-len(idx) // ncol)
        aspect = (args.ylim[1] - args.ylim[0])/(args.xlim[1] - args.xlim[0])
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(16, (16/ncol)*aspect*nrow + 0.9),
                                 constrained_layout=True, squeeze=False)
        for ax, i in zip(axes.flat, idx):
            X, Y, om = vorticity(np.array(Q[:, i])[mask], coords)
            view = ((X >= args.xlim[0]) & (X <= args.xlim[1])
                    & (Y >= args.ylim[0]) & (Y <= args.ylim[1]))
            lim = float(np.percentile(np.abs(om[view]), 99.5)) or 1.0
            ax.pcolormesh(X, Y, om, cmap=cmap, vmin=-lim, vmax=lim,
                          shading="auto", rasterized=True)
            ax.fill(coords["xi"], coords["eta"], color="k", zorder=5)
            ax.plot(xc, yc, "o", ms=4, mfc="white", mec="k", zorder=6)
            ax.set_xlim(args.xlim)
            ax.set_ylim(args.ylim)
            ax.set_aspect("equal")
            ax.set_title(f"$t = {t[i]:.4g}$", fontsize=11, loc="left")
            ax.text(0.99, 0.95, rf"$|\omega'| \leq {lim:.2g}$",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10,
                    bbox=dict(fc="white", ec="none", alpha=0.8))
        for ax in axes.flat[len(idx):]:
            ax.axis("off")
        fig.suptitle(rf"Trajectory {k}: perturbation vorticity, $\beta = {beta:g}$ "
                     f"at ({xc:+.3f}, {yc:+.3f})", fontsize=13)
        out = os.path.join(HERE, "figures", f"airfoil_traj_{k:03d}_snapshots{args.tag}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
