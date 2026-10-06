"""Snapshots of a training trajectory's perturbation q' = q - q_b on the
mid-span plane: u' and v' (w' vanishes there by the z-symmetry of the
impulses) at several times, one row per time.

Fields are read from fom_data/fluct_KKK.npy (time-major, float32) on their own
staggered grids -- no interpolation, so the full flow's boundary values do not
leak into the perturbation.  The z-plane is the u/v cell-centre plane nearest
z = 0.  Colour limits are symmetric per panel (the energy decays by decades).

Usage (incompreso venv):
    python plot_trajectory_snapshots.py [--traj 2] [--times 0 0.4 2 6 10 13]
                                        [--xlim -1 10] [--ylim -1.5 1.5]
Writes figures/trajectory_snapshots_KKK.png.
"""

import argparse
import contextlib
import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import incompreso  # noqa: E402

from case import DEFAULT_CONFIG, Case  # noqa: E402
from make_initial_conditions import staggered_coords  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIONS = ("leading edge", "midchord", "trailing edge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="case config (wing: config.yaml, plate: config_plate.yaml)")
    ap.add_argument("--data", default=None, help="trajectory directory (default: the case's)")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--times", type=float, nargs="+", default=[0, 0.4, 2, 6, 10, 13])
    ap.add_argument("--xlim", type=float, nargs=2, default=[-1.0, 10.0])
    ap.add_argument("--ylim", type=float, nargs=2, default=[-1.5, 1.5])
    args = ap.parse_args()

    C = Case(args.config)
    args.data = args.data or C.fom_data
    with contextlib.redirect_stdout(io.StringIO()):
        mesh = incompreso.parse_input_file(C.post_config)["mesh"]
    grids = staggered_coords(mesh)                                  # [(x, y, z) for u, v, w]
    sizes = [len(x)*len(y)*len(z) for x, y, z in grids]
    meta = np.load(os.path.join(args.data, "meta.npz"))
    t, params = meta["time"], meta["parameters"]
    beta = params[args.traj, 0]
    nb = len(np.unique(params[:, 0]))
    Q = np.load(os.path.join(args.data, f"fluct_{args.traj:03d}.npy"), mmap_mode="r")

    px, py = C.midspan_outline()

    aspect = (args.ylim[1] - args.ylim[0])/(args.xlim[1] - args.xlim[0])
    fig, axes = plt.subplots(len(args.times), 2, figsize=(15, len(args.times)*7.0*aspect + 1.0),
                             sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    for row, ts in zip(axes, args.times):
        i = int(np.argmin(np.abs(t - ts)))
        qi = np.asarray(Q[i], dtype=np.float64)
        for c, ax in enumerate(row):                                # c = 0: u', c = 1: v'
            x, y, z = grids[c]
            start = sum(sizes[:c])
            F = qi[start:start + sizes[c]].reshape(len(z), len(y), len(x))
            k = int(np.argmin(np.abs(z)))
            ix = np.where((x >= args.xlim[0]) & (x <= args.xlim[1]))[0]
            iy = np.where((y >= args.ylim[0]) & (y <= args.ylim[1]))[0]
            f = F[k][np.ix_(iy, ix)]
            lim = max(float(np.abs(f).max()), 1e-30)
            X, Y = np.meshgrid(x[ix], y[iy])
            cf = ax.contourf(X, Y, f, levels=np.linspace(-lim, lim, 41), cmap="RdBu_r")
            fig.colorbar(cf, ax=ax, ticks=[-lim, 0, lim], format="%.1e", pad=0.01)
            ax.plot(px, py, "k-", lw=1.5)
            ax.set_aspect("equal")
            ax.set_title(f"${'uv'[c]}'$, t = {t[i]:g}", fontsize=9, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("$x$")
    for ax in axes[:, 0]:
        ax.set_ylabel("$y$")
    fig.suptitle(f"Trajectory {args.traj}: {STATIONS[args.traj // nb]}, $\\beta$ = {beta:g}  "
                 f"(perturbation on z = {z[k]:.3g})")
    os.makedirs(C.figures, exist_ok=True)
    out = os.path.join(C.figures, f"trajectory_snapshots_{args.traj:03d}.png")
    fig.savefig(out, dpi=110)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
