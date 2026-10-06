"""Q-criterion isosurface of a 3D plate snapshot, coloured by the streamwise
vorticity omega_x, with the plate drawn as a solid surface.

Q and omega_x come from incompreso's post-processing utilities
(compute_q_criterion, compute_vorticity), on the cell-node grid.  The setup
is parsed from config_post.yaml (mesh and BCs only, no operators).

Usage (incompreso venv):
    python plot_qcriterion.py [--config config.yaml] [--snapshot N] [--q Q] [--clim C]
Defaults: the latest snapshot in data/, Q = 1 (U_inf = c = 1 units),
colour limits +-C with C = 5.  Writes figures/qcriterion_<N>.png.
"""

import argparse
import glob
import os
import re

import numpy as np

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
import pyvista as pv  # noqa: E402

import incompreso  # noqa: E402
from incompreso.backend import to_numpy  # noqa: E402

from case import DEFAULT_CONFIG, Case  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CROP = dict(x=(-1.0, 6.0), y=(-2.0, 2.0), z=(-1.5, 1.5))


def latest_snapshot(data):
    idx = [int(re.search(r"snapshot_(\d+)\.npz", f).group(1))
           for f in glob.glob(os.path.join(data, "snapshot_*.npz"))]
    if not idx:
        raise FileNotFoundError(f"no snapshots in {data}")
    return max(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="case config (wing: config.yaml, plate: config_plate.yaml)")
    ap.add_argument("--snapshot", type=int, default=None)
    ap.add_argument("--q", type=float, default=1.0, help="Q isosurface level")
    ap.add_argument("--clim", type=float, default=5.0, help="omega_x colour limit")
    args = ap.parse_args()
    C = Case(args.config)
    n = latest_snapshot(C.data) if args.snapshot is None else args.snapshot

    sim = incompreso.parse_input_file(C.post_config)
    mesh, bcs = sim["mesh"], sim["bcs"]
    path = os.path.join(C.data, f"snapshot_{n:06d}.npz")
    q, t = incompreso.load_restart(path)
    q = incompreso.backend.xp.asarray(q)
    print(f"snapshot {n}: t = {float(t):g}", flush=True)

    Q = to_numpy(incompreso.compute_q_criterion(t, q, mesh, bcs)[0])
    wx = to_numpy(incompreso.compute_vorticity(t, q, mesh, bcs)[0])

    x, y, z = (to_numpy(a) for a in (mesh.x, mesh.y, mesh.z))
    ix = np.where((x >= CROP["x"][0]) & (x <= CROP["x"][1]))[0]
    iy = np.where((y >= CROP["y"][0]) & (y <= CROP["y"][1]))[0]
    iz = np.where((z >= CROP["z"][0]) & (z <= CROP["z"][1]))[0]
    sl = np.ix_(iz, iy, ix)
    Qc, wxc = Q[sl], wx[sl]
    pos = Qc[Qc > 0]
    print(f"Q in crop: max {Qc.max():.3g}; positive-Q percentiles 50/90/99 = "
          f"{np.percentile(pos, 50):.3g}/{np.percentile(pos, 90):.3g}/"
          f"{np.percentile(pos, 99):.3g}; |omega_x| max {np.abs(wxc).max():.3g}",
          flush=True)

    Zn, Yn, Xn = np.meshgrid(z[iz], y[iy], x[ix], indexing="ij")
    grid = pv.StructuredGrid(Xn, Yn, Zn)
    grid["Q"] = Qc.ravel(order="F")
    grid["omega_x"] = wxc.ravel(order="F")
    iso = grid.contour([args.q], scalars="Q")
    print(f"isosurface Q = {args.q:g}: {iso.n_points} points", flush=True)

    g = C.body()
    if C.name == "plate":                       # single-layer marker sheet
        m = int(round(np.sqrt(g["xi"].size)))
        body = pv.StructuredGrid(*(g[k].reshape(m, m) for k in ("xi", "eta", "zeta")))
    else:                                       # closed, convex wing: hull of the markers
        body = pv.PolyData(np.column_stack([g["xi"], g["eta"], g["zeta"]])
                           ).delaunay_3d().extract_surface()

    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=[2000, 1300])
    pl.set_background("white")
    pl.add_mesh(iso, scalars="omega_x", cmap="RdBu_r", clim=[-args.clim, args.clim],
                smooth_shading=True, specular=0.4, specular_power=20,
                scalar_bar_args=dict(title="omega_x", color="black", vertical=True,
                                     position_x=0.9, position_y=0.25, height=0.5))
    pl.add_mesh(body, color=[80, 80, 80], smooth_shading=True)
    pl.add_text(f"t = {float(t):g},  Q = {args.q:g}", color="black", font_size=14)
    if C.name == "wing":                        # whole span plus the tip-vortex wake
        pl.camera_position = [(-3.5, 4.0, 6.5), (2.0, -0.3, 0.0), (0.0, 1.0, 0.0)]
    else:
        pl.camera_position = [(-1.5, 2.2, 3.4), (1.6, -0.3, 0.0), (0.0, 1.0, 0.0)]
    os.makedirs(C.figures, exist_ok=True)
    out = os.path.join(C.figures, f"qcriterion_{n:06d}.png")
    pl.screenshot(out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
