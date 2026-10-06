"""Perturbation u', v', w' of one training-trajectory snapshot on
  (1) (x, y) planes at a few spanwise stations z (default -0.5, 0, 0.5: tip,
      mid-span, tip), and
  (2) (y, z) cross-stream planes at a few streamwise stations x.

Fields are read from <fom_data>/fluct_KKK.npy (time-major, float32; the
.tmp.npy of a trajectory still being written also works, for the snapshots
already stored) on their own staggered grids -- no interpolation, so the full
flow's boundary values do not leak into the perturbation.  Each plane is the
component's grid plane nearest the requested coordinate.  The grid
coordinates come from the trajectory's base-flow snapshot (meta.npz), so
nothing is set up.  Colour limits are symmetric per panel.

Usage (incompreso venv):
    python plot_fluct_planes.py [--config config_traj.yaml] [--traj 2] [--time 12.6]
                                [--z -0.5 0 0.5] [--x 0.75 2 5 8 11]
Writes <figures>/fluct_xy_KKK_t<T>.png and fluct_yz_KKK_t<T>.png.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from case import DEFAULT_CONFIG, Case  # noqa: E402

COMP = "uvw"


def load_snapshot(C, traj, time):
    meta = np.load(os.path.join(C.fom_data, "meta.npz"))
    tsave = meta["time"]
    k = int(np.argmin(np.abs(tsave - time)))
    f = os.path.join(C.fom_data, f"fluct_{traj:03d}.npy")
    if not os.path.exists(f):
        f += ".tmp.npy"                         # trajectory still running
    Q = np.load(f, mmap_mode="r")
    q = np.asarray(Q[k], dtype=np.float64)
    if not np.any(q):
        raise SystemExit(f"snapshot t = {tsave[k]:g} not written yet")
    with np.load(str(meta["baseflow"])) as s:
        grids = [tuple(s[f"{a}{c}"][1:-1] for a in "xyz") for c in COMP]
    fields, o = [], 0
    for x, y, z in grids:
        n = x.size*y.size*z.size
        fields.append(q[o:o + n].reshape(z.size, y.size, x.size))
        o += n
    return float(tsave[k]), grids, fields


def panel(ax, fig, X, Y, F, title):
    m = np.abs(F).max() or 1.0
    cf = ax.contourf(X, Y, F, levels=np.linspace(-m, m, 41), cmap="RdBu_r", extend="both")
    cb = fig.colorbar(cf, ax=ax, shrink=0.9)
    cb.formatter.set_powerlimits((-2, 2))
    cb.update_ticks()
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_aspect("equal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--time", type=float, default=12.6)
    ap.add_argument("--z", type=float, nargs="+", default=[-0.5, 0.0, 0.5])
    ap.add_argument("--x", type=float, nargs="+", default=[0.75, 2.0, 5.0, 8.0, 11.0])
    ap.add_argument("--xlim", type=float, nargs=2, default=[-1.0, 15.0])
    ap.add_argument("--ylim", type=float, nargs=2, default=[-2.0, 2.0])
    ap.add_argument("--half", type=float, default=1.5, help="(y, z) planes: |y|, |z| <= HALF")
    args = ap.parse_args()
    C = Case(args.config)
    t, grids, fields = load_snapshot(C, args.traj, args.time)
    ox, oy = C.midspan_outline()
    os.makedirs(C.figures, exist_ok=True)
    tag = f"{args.traj:03d}_t{t:05.2f}"

    # (x, y) planes
    fig, axs = plt.subplots(len(args.z), 3, figsize=(18, 2.6*len(args.z) + 0.6),
                            squeeze=False, constrained_layout=True)
    for r, z0 in enumerate(args.z):
        for c, ((x, y, z), F) in enumerate(zip(grids, fields)):
            ix = (x >= args.xlim[0]) & (x <= args.xlim[1])
            iy = (y >= args.ylim[0]) & (y <= args.ylim[1])
            kz = int(np.argmin(np.abs(z - z0)))
            ax = axs[r, c]
            panel(ax, fig, x[ix], y[iy], F[kz][np.ix_(iy, ix)],
                  f"{COMP[c]}', z = {z[kz]:+.4f}")
            ax.fill(ox, oy, color="0.3", lw=0)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
    fig.suptitle(f"{C.name}: perturbation, trajectory {args.traj}, t = {t:g}")
    out = os.path.join(C.figures, f"fluct_xy_{tag}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved -> {out}")

    # (y, z) planes
    h = args.half
    fig, axs = plt.subplots(len(args.x), 3, figsize=(13, 3.6*len(args.x)),
                            squeeze=False, constrained_layout=True)
    for r, x0 in enumerate(args.x):
        for c, ((x, y, z), F) in enumerate(zip(grids, fields)):
            iy, iz = np.abs(y) <= h, np.abs(z) <= h
            kx = int(np.argmin(np.abs(x - x0)))
            ax = axs[r, c]
            panel(ax, fig, y[iy], z[iz], F[:, :, kx][np.ix_(iz, iy)],
                  f"{COMP[c]}', x = {x[kx]:.3f}")
            ax.set_xlabel("y")
            ax.set_ylabel("z")
    fig.suptitle(f"{C.name}: perturbation, trajectory {args.traj}, t = {t:g}")
    out = os.path.join(C.figures, f"fluct_yz_{tag}.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
