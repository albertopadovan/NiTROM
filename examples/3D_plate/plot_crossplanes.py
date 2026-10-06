"""u, v, w on cross-stream planes x = const of a 3D plate state (default: the
base flow base_flow.npz), one row per station, one column per component.
u is shown with its u = 0 contour (backflow), v and w with symmetric limits.

Velocities are interpolated to the cell nodes with incompreso's
interpolate_to_nodes; each plane is the node plane nearest the requested x.

Usage (incompreso venv):
    python plot_crossplanes.py [--state base_flow.npz] [--x 1.5 3 5 8 10] [--half 2]
Writes figures/crossplanes_<state>.png.
"""

import argparse
import os

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
    ap.add_argument("--state", default=None,
                    help="snapshot .npz (default: the latest in the case's data directory)")
    ap.add_argument("--x", type=float, nargs="+", default=[1.5, 3.0, 5.0, 8.0, 10.0])
    ap.add_argument("--half", type=float, default=2.0, help="plot |y|, |z| <= HALF")
    args = ap.parse_args()

    C = Case(args.config)
    if args.state is None:
        import glob
        args.state = max(glob.glob(os.path.join(C.data, "snapshot_*.npz")))
    sim = incompreso.parse_input_file(C.post_config)
    mesh, bcs = sim["mesh"], sim["bcs"]
    q, t = incompreso.load_restart(os.path.join(HERE, args.state))
    un, vn, wn = interpolate_to_nodes(t, xp.asarray(q), mesh, bcs)   # (nz+1, ny+1, nx+1)
    x, y, z = (to_numpy(a) for a in (mesh.x, mesh.y, mesh.z))
    jy = np.where(np.abs(y) <= args.half)[0]
    kz = np.where(np.abs(z) <= args.half)[0]
    Y, Z = np.meshgrid(y[jy], z[kz])
    print(f"{args.state}: t = {float(t):g}", flush=True)

    fig, axes = plt.subplots(len(args.x), 3, figsize=(12, 3.6*len(args.x)),
                             constrained_layout=True, squeeze=False)
    for row, xs in zip(axes, args.x):
        i = int(np.argmin(np.abs(x - xs)))
        u, v, w = (to_numpy(f[:, :, i])[np.ix_(kz, jy)] for f in (un, vn, wn))
        print(f"  x = {x[i]:.3g}: u in [{u.min():.3f}, {u.max():.3f}], "
              f"|v| <= {np.abs(v).max():.3f}, |w| <= {np.abs(w).max():.3f}, "
              f"backflow cells {int((u < 0).sum())}", flush=True)
        for ax, f, name in zip(row, (u, v, w), "uvw"):
            if name == "u":
                cf = ax.contourf(Y, Z, f, levels=np.linspace(f.min(), f.max(), 41), cmap="viridis")
                if f.min() < 0:
                    ax.contour(Y, Z, f, levels=[0.0], colors="r", linewidths=1.0)
            else:
                c = max(float(np.abs(f).max()), 1e-12)
                cf = ax.contourf(Y, Z, f, levels=np.linspace(-c, c, 41), cmap="RdBu_r")
            fig.colorbar(cf, ax=ax, format="%.2g", pad=0.01)
            ax.set_aspect("equal")
            ax.set_title(f"${name}$, x = {x[i]:.3g}" + ("  (red: u = 0)" if name == "u" and f.min() < 0 else ""),
                         fontsize=9, loc="left")
            ax.set_xlabel("$y$")
            ax.set_ylabel("$z$")
    fig.suptitle(f"{C.name.capitalize()}: cross-stream planes, {os.path.basename(args.state)}, t = {float(t):g}")
    os.makedirs(C.figures, exist_ok=True)
    out = os.path.join(C.figures, f"crossplanes_{os.path.splitext(os.path.basename(args.state))[0]}.png")
    fig.savefig(out, dpi=110)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
