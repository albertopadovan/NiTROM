"""Balanced modes of the 3D wing training data: primal Phi and adjoint Psi.

The modes are r_l-vectors in the lossless coordinates (compute_rom_bases.py);
the full-space weighted basis vectors are U Phi_j and U Psi_j (one pass over
the trajectory files, rom_compress.Compressed.lift).  Both are shown as the
physical fields

    phi_j = W^{-1/2} U Phi_j,     psi_j = W^{-1/2} U Psi_j,

so that z_j = Psi_j^T q~ = <psi_j, q'>_W and q' ~ sum_j z_j phi_j.  The lifted
fields are cached in <data>/modes_balanced.npz (float32), so replotting does
not touch the trajectories again.

Figures (figures_wing/):
  balanced_modes_xy.png   (x, y) plane z = 0: u, v of phi_j and psi_j
  balanced_modes_yz.png   (y, z) planes where each field's energy peaks in x
  balanced_modes_Q_<j>.png  Q-criterion isosurfaces of phi_j and psi_j
                            (Q at --q-frac of its max), coloured by omega_x
Q and omega_x use incompreso's post-processing with homogeneous boundary
values (config_post_perturbation.yaml).

Usage (incompreso venv, GPU for the lift):
    python plot_balanced_modes.py [--modes 0 1 4 10] [--q-modes 0 4]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from incompreso.backend import to_numpy, xp  # noqa: E402

from rom_compress import HERE, Compressed  # noqa: E402

COMP = "uvw"


def lifted_modes(data_dir, modes):
    """Physical fields phi_j, psi_j (float32, (n_modes, N)), cached."""
    cache = os.path.join(data_dir, "modes_balanced.npz")
    if os.path.exists(cache):
        c = np.load(cache)
        if set(modes) <= set(c["modes"].tolist()):
            idx = [c["modes"].tolist().index(j) for j in modes]
            return c["phi"][idx], c["psi"][idx], c["Sigma"]
    comp = Compressed(os.path.join(data_dir, "compressed.npz"))
    b = np.load(os.path.join(data_dir, "bases.npz"), allow_pickle=True)
    A = np.concatenate([b["Phi_bal"][:, modes], b["Psi_bal"][:, modes]], axis=1)
    L = comp.lift(A, xp)                                   # (N, 2 m) weighted
    inv = (1/comp.data.sqrt_w).astype(np.float32)[:, None]
    m = len(modes)
    phi = np.ascontiguousarray((L[:, :m]*inv).T)
    psi = np.ascontiguousarray((L[:, m:]*inv).T)
    np.savez(cache, modes=np.array(modes), phi=phi, psi=psi, Sigma=b["Sigma_bal"])
    print(f"cached -> {cache}")
    return phi, psi, b["Sigma_bal"]


def split(q, grids):
    out, o = [], 0
    for x, y, z in grids:
        n = x.size*y.size*z.size
        out.append(q[o:o + n].reshape(z.size, y.size, x.size))
        o += n
    return out


def panel(ax, fig, X, Y, F, title):
    m = float(np.abs(F).max()) or 1.0
    cf = ax.contourf(X, Y, F/m, levels=np.linspace(-1, 1, 41), cmap="RdBu_r")
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_aspect("equal")
    return cf


def q_figure(field, label, mesh, bcs, body, q_frac, out):
    import incompreso
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    import pyvista as pv
    q = xp.asarray(field, dtype=xp.float64)
    Q = to_numpy(incompreso.compute_q_criterion(0.0, q, mesh, bcs)[0])
    wx = to_numpy(incompreso.compute_vorticity(0.0, q, mesh, bcs)[0])
    x, y, z = (to_numpy(a) for a in (mesh.x, mesh.y, mesh.z))
    ix = np.where((x >= -1.5) & (x <= 14.0))[0]
    iy = np.where(np.abs(y) <= 2.0)[0]
    iz = np.where(np.abs(z) <= 2.0)[0]
    sl = np.ix_(iz, iy, ix)
    Qc, wxc = Q[sl], wx[sl]
    level = q_frac*Qc.max()
    Zn, Yn, Xn = np.meshgrid(z[iz], y[iy], x[ix], indexing="ij")
    grid = pv.StructuredGrid(Xn, Yn, Zn)
    grid["Q"] = Qc.ravel(order="F")
    grid["omega_x"] = wxc.ravel(order="F")
    iso = grid.contour([level], scalars="Q")
    c = float(np.percentile(np.abs(wxc[Qc > level]), 95)) if np.any(Qc > level) else 1.0
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=[2000, 1100])
    pl.set_background("white")
    if iso.n_points:
        pl.add_mesh(iso, scalars="omega_x", cmap="RdBu_r", clim=[-c, c], smooth_shading=True,
                    scalar_bar_args=dict(title="omega_x", color="black", vertical=True,
                                         position_x=0.9, position_y=0.25, height=0.5))
    pl.add_mesh(body, color=[80, 80, 80], smooth_shading=True)
    pl.add_text(f"{label}:  Q = {q_frac:g} max Q", color="black", font_size=14)
    pl.camera_position = [(-3.0, 4.5, 8.0), (5.0, -0.3, 0.0), (0.0, 1.0, 0.0)]
    pl.screenshot(out)
    pl.close()                  # several off-screen plotters in one process render black otherwise
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data_wing_re1000_train"))
    ap.add_argument("--modes", type=int, nargs="+", default=[0, 1, 4, 10])
    ap.add_argument("--q-modes", type=int, nargs="*", default=[0, 4])
    ap.add_argument("--q-frac", type=float, default=0.05)
    ap.add_argument("--xlim", type=float, nargs=2, default=[-1.0, 15.0])
    ap.add_argument("--ylim", type=float, nargs=2, default=[-1.5, 1.5])
    ap.add_argument("--half", type=float, default=1.2)
    ap.add_argument("--q-render", nargs=2, metavar=("FIELD", "MODE"), default=None,
                    help=argparse.SUPPRESS)   # internal: one Q figure per process
    args = ap.parse_args()
    if args.q_render is not None:
        render_one(args, args.q_render[0], int(args.q_render[1]))
        return
    modes = sorted(set(args.modes) | set(args.q_modes))
    phi, psi, Sig = lifted_modes(args.data, modes)
    meta = np.load(os.path.join(args.data, "meta.npz"))
    with np.load(str(meta["baseflow"])) as s:
        grids = [tuple(s[f"{a}{c}"][1:-1] for a in "xyz") for c in COMP]
    from case import Case
    C = Case(os.path.join(HERE, "config_re1000.yaml"))
    ox, oy = C.midspan_outline()
    figs = os.path.join(HERE, "figures_wing")
    os.makedirs(figs, exist_ok=True)
    rows = [modes.index(j) for j in args.modes]

    # (x, y) at z = 0: u, v of phi and psi (w = 0 there by the z-symmetry)
    fig, axs = plt.subplots(len(rows), 4, figsize=(22, 1.9*len(rows) + 0.8),
                            squeeze=False, constrained_layout=True)
    for r, i in enumerate(rows):
        j = modes[i]
        for c, (name, f, comp) in enumerate((("phi", phi, 0), ("phi", phi, 1),
                                             ("psi", psi, 0), ("psi", psi, 1))):
            x, y, z = grids[comp]
            F = split(f[i], grids)[comp]
            kz = int(np.argmin(np.abs(z)))
            sx = (x >= args.xlim[0]) & (x <= args.xlim[1])
            sy = (y >= args.ylim[0]) & (y <= args.ylim[1])
            cf = panel(axs[r, c], fig, x[sx], y[sy], F[kz][np.ix_(sy, sx)],
                       f"{name}_{j} {COMP[comp]}, z = {z[kz]:+.4f}  (Sigma_{j}/Sigma_0 = "
                       f"{Sig[j]/Sig[0]:.3g})")
            axs[r, c].fill(ox, oy, color="0.3", lw=0)
    fig.colorbar(cf, ax=axs, shrink=0.6, label="field / max |field|")
    fig.suptitle("Balanced modes (Re = 1000 wing), plane z = 0: primal phi_j (left), adjoint psi_j (right)")
    out = os.path.join(figs, "balanced_modes_xy.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved -> {out}")

    # (y, z) where each field's u-energy peaks in x: u and w of phi and psi
    fig, axs = plt.subplots(len(rows), 4, figsize=(14, 3.3*len(rows)),
                            squeeze=False, constrained_layout=True)
    h = args.half
    for r, i in enumerate(rows):
        j = modes[i]
        for c, (name, f, comp) in enumerate((("phi", phi, 0), ("phi", phi, 2),
                                             ("psi", psi, 0), ("psi", psi, 2))):
            Fu = split(f[i], grids)[0]
            xu = grids[0][0]
            xpk = xu[int(np.argmax((Fu.astype(np.float64)**2).sum(axis=(0, 1))))]
            x, y, z = grids[comp]
            F = split(f[i], grids)[comp]
            kx = int(np.argmin(np.abs(x - xpk)))
            sy, sz = np.abs(y) <= h, np.abs(z) <= h
            cf = panel(axs[r, c], fig, y[sy], z[sz], F[:, :, kx][np.ix_(sz, sy)],
                       f"{name}_{j} {COMP[comp]}, x = {x[kx]:.3f}")
            axs[r, c].set_xlabel("y")
            axs[r, c].set_ylabel("z")
    fig.colorbar(cf, ax=axs, shrink=0.6, label="field / max |field|")
    fig.suptitle("Balanced modes: (y, z) planes at the x where each field's u-energy peaks")
    out = os.path.join(figs, "balanced_modes_yz.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"saved -> {out}")

    # Q figures: off-screen VTK (Mesa) renders only the first plotter of a
    # process, so each figure is drawn by a fresh process (fields are cached)
    for j in args.q_modes:
        for name in ("phi", "psi"):
            subprocess.run([sys.executable, os.path.abspath(__file__), "--data", args.data,
                            "--q-frac", str(args.q_frac), "--q-render", name, str(j)],
                           check=True)


def render_one(args, name, j):
    import incompreso
    import pyvista as pv
    from case import Case
    phi, psi, _ = lifted_modes(args.data, [j])
    field = (phi if name == "phi" else psi)[0]
    sim = incompreso.parse_input_file(os.path.join(HERE, "config_post_perturbation.yaml"))
    g = Case(os.path.join(HERE, "config_re1000.yaml")).body()
    body = pv.PolyData(np.column_stack([g["xi"], g["eta"], g["zeta"]])
                       ).delaunay_3d().extract_surface()
    q_figure(field, f"{name}_{j}", sim["mesh"], sim["bcs"], body, args.q_frac,
             os.path.join(HERE, "figures_wing", f"balanced_modes_Q_{name}{j}.png"))

if __name__ == "__main__":
    main()
