"""Velocity components u, v, w on the symmetry plane z = 0 of a 3D plate
snapshot (filled contours, one panel each), with the plate's chord line
drawn in.

Velocities are interpolated to the cell nodes with incompreso's
interpolate_to_nodes; the slice is the node plane nearest z = 0.  The setup is
parsed from config_post.yaml (mesh and BCs only).

Usage (incompreso venv):
    python plot_velocity_slices.py [--snapshot N] [--xlim X0 X1] [--ylim Y0 Y1]
Defaults: the latest snapshot in data/, x in [-2, 10], y in [-2.5, 2.5].
Writes figures/velocity_z0_<N>.png.
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import incompreso  # noqa: E402
from incompreso.backend import to_numpy, xp  # noqa: E402
from incompreso.utils.post_process import interpolate_to_nodes  # noqa: E402

from case import DEFAULT_CONFIG, Case  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="case config (wing: config.yaml, plate: config_plate.yaml)")
    ap.add_argument("--snapshot", type=int, default=None)
    ap.add_argument("--xlim", type=float, nargs=2, default=[-2.0, 10.0])
    ap.add_argument("--ylim", type=float, nargs=2, default=[-2.5, 2.5])
    args = ap.parse_args()
    C = Case(args.config)
    n = args.snapshot
    if n is None:
        n = max(int(re.search(r"snapshot_(\d+)\.npz", f).group(1))
                for f in glob.glob(os.path.join(C.data, "snapshot_*.npz")))

    sim = incompreso.parse_input_file(C.post_config)
    mesh, bcs = sim["mesh"], sim["bcs"]
    q, t = incompreso.load_restart(os.path.join(C.data, f"snapshot_{n:06d}.npz"))
    un, vn, wn = interpolate_to_nodes(t, xp.asarray(q), mesh, bcs)   # (nz+1, ny+1, nx+1)
    x, y, z = (to_numpy(a) for a in (mesh.x, mesh.y, mesh.z))
    k = int(np.argmin(np.abs(z)))
    ix = np.where((x >= args.xlim[0]) & (x <= args.xlim[1]))[0]
    iy = np.where((y >= args.ylim[0]) & (y <= args.ylim[1]))[0]
    fields = [to_numpy(f[k])[np.ix_(iy, ix)] for f in (un, vn, wn)]
    print(f"snapshot {n}: t = {float(t):g}, slice z = {z[k]:.3g}", flush=True)

    px, py = C.midspan_outline()

    X, Y = np.meshgrid(x[ix], y[iy])
    aspect = (args.ylim[1] - args.ylim[0]) / (args.xlim[1] - args.xlim[0])
    fig, axes = plt.subplots(3, 1, figsize=(11, 3*11*aspect + 1.2), sharex=True,
                             constrained_layout=True)
    for ax, f, name in zip(axes, fields, ("u", "v", "w")):
        if name == "u":
            levels, cmap = np.linspace(f.min(), f.max(), 41), "viridis"
        else:
            c = max(float(np.abs(f).max()), 1e-12)
            levels, cmap = np.linspace(-c, c, 41), "RdBu_r"
        cf = ax.contourf(X, Y, f, levels=levels, cmap=cmap)
        if name == "u":
            ax.contour(X, Y, f, levels=[0.0], colors="w", linewidths=0.8)
        fig.colorbar(cf, ax=ax, label=f"${name}$", format="%.2g", pad=0.01)
        ax.plot(px, py, "k-", lw=2.0)
        ax.set_aspect("equal")
        ax.set_ylabel("$y$")
        ax.set_title(f"${name}$" + ("  (white: $u = 0$)" if name == "u" else ""),
                     fontsize=10, loc="left")
    axes[-1].set_xlabel("$x$")
    fig.suptitle(f"{C.name.capitalize()}, z = {z[k]:.3g} plane, t = {float(t):g}")
    os.makedirs(C.figures, exist_ok=True)
    out = os.path.join(C.figures, f"velocity_z0_{n:06d}.png")
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
