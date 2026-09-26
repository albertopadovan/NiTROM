"""Vorticity snapshots of one impulse response: FOM next to each ROM.

One row per requested time, one column for the FOM and one per model.  ROMs
are forecast from z(0) = Psi^T q~(0) and reconstructed in physical units,
q = W^{-1/2} Phi (Psi^T Phi)^{-1} z, then shown on the ROM domain D.  Each row
uses the FOM's colour scale (99.5th percentile of |omega'| over D), so an
under- or over-predicted amplitude shows up directly.

Usage: python plot_rom_snapshots.py models/a.pkl [models/b.pkl ...]
            [--data fom_data_testing] [--traj 1] [--times 1 4 8 11 15 19 30]
Writes figures/airfoil_snapshots_<data>_<traj>.png.
"""

from __future__ import annotations

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap, TwoSlopeNorm

from rom_domain import X_BOUNDS, Y_BOUNDS, crop_mask, vorticity
from train_gas_opinf_balanced import forecast

HERE = os.path.dirname(os.path.abspath(__file__))
YLIM = (-1.5, 1.5)


def rd_bu_r_with_white_center() -> ListedColormap:
    """RdBu_r with a pure-white band at zero (as plot_baseflow.py, which
    imports incompreso; this script only needs NiTROM)."""
    colors = plt.get_cmap("RdBu_r", 512)(np.linspace(0.0, 1.0, 512))
    center = len(colors) // 2
    colors[center - 1: center + 1] = (1.0, 1.0, 1.0, 1.0)
    return ListedColormap(colors, name="RdBu_r_white_center")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="checkpoint .pkl files")
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data_testing"))
    ap.add_argument("--traj", type=int, default=1)
    ap.add_argument("--times", type=float, nargs="+",
                    default=[1, 4, 8, 11, 15, 19, 30])
    ap.add_argument("--balancing", default=os.path.join(HERE, "balancing_full"))
    ap.add_argument("--scale", choices=["trajectory", "row"], default="row",
                    help="colour limits: min/max of the FOM vorticity over the "
                         "whole trajectory (all panels alike), or per row")
    ap.add_argument("--scale-from", type=float, default=0.0,
                    help="with --scale trajectory: take the FOM min/max over "
                         "t >= this only (the t = 0 impulse is ~100x the "
                         "later vorticity and would wash every panel out)")
    ap.add_argument("--saturate", type=float, default=1.0,
                    help="with --scale trajectory: colour limits are this "
                         "fraction of the FOM min/max (values beyond saturate)")
    ap.add_argument("--tag", default="", help="suffix for the output name")
    args = ap.parse_args()

    bal = np.load(os.path.join(args.balancing, "balancing.npz"), allow_pickle=True)
    mask, sw = bal["mask"], bal["sqrt_w"]
    if not mask.all():
        raise SystemExit("expects the full-mesh balancing (balancing_full)")
    meta = np.load(os.path.join(args.data, "meta.npz"), allow_pickle=True)
    t, params = meta["time"], meta["parameters"]
    with np.load(str(meta["baseflow"])) as snap:
        coords = {c: snap[c] for c in ("xu", "yu", "xv", "yv", "xi", "eta")}
    dmask = crop_mask(coords)

    Q = np.load(os.path.join(args.data, f"fluct_{args.traj:03d}.npy"),
                mmap_mode="r")
    x0 = np.ascontiguousarray(Q[:, 0])*sw                  # q~(0)
    idx = [int(np.argmin(np.abs(t - s))) for s in args.times]

    # physical-unit fields at the requested times: FOM, then each model
    cols = [("FOM", [np.array(Q[:, i]) for i in idx])]
    for path in args.models:
        with open(path, "rb") as f:
            ck = pickle.load(f)
        A, H = [np.asarray(x) for x in ck["tensors"]]
        Phi, Psi = np.asarray(ck["Phi"]), np.asarray(ck["Psi"])
        Z = forecast(A, H, Psi.T @ x0, t)
        dec = Phi @ np.linalg.inv(Psi.T @ Phi)
        cols.append((os.path.splitext(os.path.basename(path))[0],
                     [(dec @ Z[:, i])/sw for i in idx]))

    cmap = rd_bu_r_with_white_center()
    if args.scale == "trajectory":
        # min/max of the FOM vorticity over every snapshot of the trajectory
        lo, hi = np.inf, -np.inf
        for i in np.flatnonzero(t >= args.scale_from - 1e-9):
            om = vorticity(np.array(Q[:, i])[dmask], coords)[2]
            lo, hi = min(lo, float(om.min())), max(hi, float(om.max()))
        norm_all = TwoSlopeNorm(vcenter=0.0, vmin=args.saturate*lo,
                                vmax=args.saturate*hi)
        print(f"FOM vorticity over the trajectory (t >= {args.scale_from:g}): "
              f"min {lo:.3g}, max {hi:.3g}")
    nr, nc = len(idx), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(5.6*nc, 1.35*nr + 0.8),
                             constrained_layout=True, squeeze=False)
    for row, i in enumerate(idx):
        oms = [vorticity(fields[row][dmask], coords) for _, fields in cols]
        lim = float(np.percentile(np.abs(oms[0][2]), 99.5)) or 1.0
        norm = (norm_all if args.scale == "trajectory"
                else TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim))
        for c, ((name, _), (X, Y, om)) in enumerate(zip(cols, oms)):
            ax = axes[row, c]
            pcm = ax.pcolormesh(X, Y, om, cmap=cmap, norm=norm,
                                shading="auto", rasterized=True)
            ax.fill(coords["xi"], coords["eta"], color="k", zorder=5)
            ax.set_xlim(X_BOUNDS)
            ax.set_ylim(YLIM)
            ax.set_aspect("equal")
            ax.set_xticks([] if row < nr - 1 else [-4, 0, 4, 8, 12])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"$t = {t[i]:.1f}$", fontsize=10)
                if args.scale == "row":
                    ax.text(0.99, 0.93, rf"$|\omega'| \leq {lim:.2g}$",
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=8,
                            bbox=dict(fc="white", ec="none", alpha=0.8))
    beta = params[args.traj, 0]
    lab = str(meta["labels"][args.traj]) if "labels" in meta.files else ""
    fig.suptitle(f"{os.path.basename(os.path.normpath(args.data))} "
                 f"trajectory {args.traj}: {lab}, $\\beta = {beta:.3f}$ -- "
                 "perturbation vorticity, "
                 + ("colour limits = FOM min/max over the trajectory"
                    + (f" (t >= {args.scale_from:g})" if args.scale_from > 0 else "")
                    + (f" x {args.saturate:g}" if args.saturate != 1 else "")
                    if args.scale == "trajectory"
                    else "each row on the FOM's colour scale"),
                 fontsize=11)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures",
                       f"airfoil_snapshots_{os.path.basename(os.path.normpath(args.data))}"
                       f"_{args.traj:03d}{args.tag}.png")
    if args.scale == "trajectory":
        fig.colorbar(pcm, ax=axes, shrink=0.5, pad=0.01, label=r"$\omega'$",
                     extend="both" if args.saturate < 1 else "neither")
    fig.savefig(out, dpi=140)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
