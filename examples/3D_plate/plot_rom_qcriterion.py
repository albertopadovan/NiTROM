"""Q-criterion of the perturbation: FOM vs the balanced and POD ROMs.

For each chosen trajectory and time, three perturbation fields q':
  FOM       the stored snapshot (fluct_KKK.npy);
  ROM_b,p   the forecast reconstruction W^{-1/2} U Phi z(t), z from the
            OpInf model (train_opinf.py checkpoint) integrated from
            z(0) = Psi^T x(0); U Phi z is lifted through the data (one pass
            for all ROM states, rom_compress.Compressed.lift).
Q and omega_x come from incompreso's post-processing with homogeneous boundary
values (config_post_perturbation.yaml).  The isosurface level, Q = q_frac *
max Q_FOM, and the omega_x colour limits are taken from the FOM snapshot at
that time and used unchanged for both ROMs.  Off-screen VTK renders only one
plotter per process, so each panel is drawn by its own subprocess from the
cropped Q / omega_x cached in --cache; the panels are then tiled per
trajectory (rows FOM / balanced / POD, columns times).

Usage (incompreso venv, GPU):
    python plot_rom_qcriterion.py [--trajs 4 1] [--times 1 4 8 12 16]
           [--models opinf_balanced opinf_pod] [--r 50] [--q-frac 0.05]
Writes figures_wing/rom_qcriterion_traj<K>.png.
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = "/tmp/claude-70968/-work-nvme-bgqw-padovan3-NiTROM/5e962897-efa8-4230-be03-030fe5750624/scratchpad/rom_q"
CROP = dict(x=(-1.5, 14.0), y=(-2.0, 2.0), z=(-2.0, 2.0))
CAMERA = [(-3.0, 4.5, 8.0), (5.0, -0.3, 0.0), (0.0, 1.0, 0.0)]


def forecast(A, H, z0, time):
    from scipy.integrate import solve_ivp
    r = len(z0)
    H2 = H.reshape(r, r*r)
    sol = solve_ivp(lambda t, z: A @ z + H2 @ np.outer(z, z).ravel(),
                    (time[0], time[-1]), z0, t_eval=time, method="DOP853",
                    rtol=1e-9, atol=1e-12)
    return sol.y


def compute_fields(args):
    import incompreso
    from incompreso.backend import to_numpy, xp
    from rom_compress import Compressed

    comp = Compressed(os.path.join(args.data, "compressed.npz"))
    X, time, params = comp.X, comp.time, comp.params
    idx = [int(np.argmin(np.abs(time - t))) for t in args.times]
    sqrt_w = comp.data.sqrt_w

    cols, keys = [], []
    for name in args.models:
        with open(os.path.join(args.model_dir, f"{name}_r{args.r}.pkl"), "rb") as f:
            m = pickle.load(f)
        A = m["tensors"][0]
        r = A.shape[0]
        H = m["tensors"][1] if len(m["tensors"]) > 1 else np.zeros((r, r*r))
        for k in args.trajs:
            Z = forecast(A, H, m["Psi"].T @ X[k][:, 0], time)
            for i in idx:
                cols.append(m["Phi"] @ Z[:, i])
                keys.append((name, k, i))
    print(f"lifting {len(cols)} ROM states (one pass over the data)", flush=True)
    L = comp.lift(np.stack(cols, axis=1), xp)                    # weighted, (N, n)

    sim = incompreso.parse_input_file(os.path.join(HERE, "config_post_perturbation.yaml"))
    mesh, bcs = sim["mesh"], sim["bcs"]
    x, y, z = (to_numpy(a) for a in (mesh.x, mesh.y, mesh.z))
    ix = np.where((x >= CROP["x"][0]) & (x <= CROP["x"][1]))[0]
    iy = np.where((y >= CROP["y"][0]) & (y <= CROP["y"][1]))[0]
    iz = np.where((z >= CROP["z"][0]) & (z <= CROP["z"][1]))[0]
    sl = np.ix_(iz, iy, ix)
    os.makedirs(args.cache, exist_ok=True)
    np.savez(os.path.join(args.cache, "grid.npz"), x=x[ix], y=y[iy], z=z[iz])

    def save(field, tag):
        q = xp.asarray(field, dtype=xp.float64)
        Q = to_numpy(incompreso.compute_q_criterion(0.0, q, mesh, bcs)[0])[sl]
        wx = to_numpy(incompreso.compute_vorticity(0.0, q, mesh, bcs)[0])[sl]
        np.savez(os.path.join(args.cache, f"{tag}.npz"), Q=Q.astype(np.float32),
                 wx=wx.astype(np.float32))
        return Q, wx

    levels = {}
    for k in args.trajs:
        Qk = np.load(os.path.join(args.data, f"fluct_{k:03d}.npy"), mmap_mode="r")
        for i in idx:
            Q, wx = save(np.asarray(Qk[i], dtype=np.float64), f"fom_{k}_{i}")
            level = args.q_frac*float(Q.max())
            sel = Q > level
            clim = float(np.percentile(np.abs(wx[sel]), 95)) if sel.any() else 1.0
            levels[(k, i)] = (level, clim)
            print(f"  traj {k}, t = {time[i]:g}: FOM max Q {Q.max():.3g}, level {level:.3g}, "
                  f"|omega_x| clim {clim:.3g}", flush=True)
    for j, (name, k, i) in enumerate(keys):
        save(L[:, j].astype(np.float64)/sqrt_w, f"{name}_{k}_{i}")
    np.save(os.path.join(args.cache, "levels.npy"),
            np.array([(k, i, *levels[(k, i)]) for (k, i) in levels]))
    return time, idx, levels


def render(args):
    """One panel, in its own process."""
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    import pyvista as pv
    from case import Case
    tag, level, clim, focal, out = args.render
    fx, fy, fz = (float(v) for v in focal.split(","))
    g = np.load(os.path.join(args.cache, "grid.npz"))
    d = np.load(os.path.join(args.cache, f"{tag}.npz"))
    Zn, Yn, Xn = np.meshgrid(g["z"], g["y"], g["x"], indexing="ij")
    grid = pv.StructuredGrid(Xn, Yn, Zn)
    grid["Q"] = d["Q"].ravel(order="F")
    grid["omega_x"] = d["wx"].ravel(order="F")
    iso = grid.contour([float(level)], scalars="Q")
    b = Case(os.path.join(HERE, "config_re1000.yaml")).body()
    body = pv.PolyData(np.column_stack([b["xi"], b["eta"], b["zeta"]])
                       ).delaunay_3d().extract_surface()
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=[1400, 800])
    pl.set_background("white")
    c = float(clim)
    if iso.n_points:
        pl.add_mesh(iso, scalars="omega_x", cmap="RdBu_r", clim=[-c, c], smooth_shading=True,
                    show_scalar_bar=False)
    pl.add_mesh(body, color=[80, 80, 80], smooth_shading=True)
    off = np.array(CAMERA[0]) - np.array(CAMERA[1])           # same view direction,
    off *= 5.5/np.linalg.norm(off)                              # closer, following the packet
    pl.camera_position = [tuple(np.array([fx, fy, fz]) + off), (fx, fy, fz), CAMERA[2]]
    pl.screenshot(out)


def tile(args, time, idx, levels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    rows = [("FOM", "fom")] + [(n.replace("opinf_", "").replace("_", " "), n)
                               for n in args.models]
    for k in args.trajs:
        fig, axs = plt.subplots(len(rows), len(idx), figsize=(4.2*len(idx), 2.7*len(rows)),
                                squeeze=False, constrained_layout=True)
        for c, i in enumerate(idx):
            level, clim = levels[(k, i)]
            imgs = [plt.imread(os.path.join(args.cache, f"{tag}_{k}_{i}.png")) for _, tag in rows]
            mask = np.any([im[..., :3].min(axis=2) < 0.98 for im in imgs], axis=0)
            rr, cc = np.where(mask)
            pad = 10
            box = (slice(max(rr.min() - pad, 0), rr.max() + pad),
                   slice(max(cc.min() - pad, 0), cc.max() + pad))
            for r, (label, tag) in enumerate(rows):
                img = imgs[r][box]
                ax = axs[r, c]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"t = {time[i]:g}\nQ = {level:.2g}, |$\\omega_x$| $\\leq$ {clim:.2g}",
                                 fontsize=9)
                if c == 0:
                    ax.set_ylabel(label, fontsize=11)
        params = np.load(os.path.join(args.data, "meta.npz"))["parameters"]
        fig.colorbar(ScalarMappable(Normalize(-1, 1), "RdBu_r"), ax=axs, shrink=0.5,
                     label=r"$\omega_x$ / (FOM colour limit of that column)")
        fig.suptitle(f"Perturbation Q-criterion, trajectory {k} (beta = {params[k, 0]:g}, "
                     f"station ({params[k, 1]:+.2f}, {params[k, 2]:+.2f})); level and colours "
                     f"from the FOM at each time", fontsize=11)
        out = os.path.join(HERE, "figures_wing", f"rom_qcriterion_traj{k}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"saved -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajs", type=int, nargs="+", default=[4, 1])
    ap.add_argument("--times", type=float, nargs="+", default=[1, 4, 8, 12, 16])
    ap.add_argument("--models", nargs="+", default=["opinf_balanced", "opinf_pod"])
    ap.add_argument("--r", type=int, default=50)
    ap.add_argument("--q-frac", type=float, default=0.05)
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data_wing_re1000_train"))
    ap.add_argument("--model-dir", default=os.path.join(HERE, "models_wing_re1000"))
    ap.add_argument("--cache", default=SCRATCH)
    ap.add_argument("--render", nargs=5, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--rerender", action="store_true",
                    help="reuse the cached Q / omega_x fields (no data pass)")
    args = ap.parse_args()
    if args.render is not None:
        render(args)
        return

    if args.rerender:
        time = np.load(os.path.join(args.data, "meta.npz"))["time"]
        idx = [int(np.argmin(np.abs(time - t))) for t in args.times]
        levels = {(int(k), int(i)): (lv, cl) for k, i, lv, cl in
                  np.load(os.path.join(args.cache, "levels.npy"))}
    else:
        time, idx, levels = compute_fields(args)
    g = np.load(os.path.join(args.cache, "grid.npz"))
    for k in args.trajs:
        for i in idx:
            level, clim = levels[(k, i)]
            fom = np.load(os.path.join(args.cache, f"fom_{k}_{i}.npz"))
            w = np.where(fom["Q"] > level, fom["Q"], 0.0)          # FOM packet centroid
            zc, yc, xc = (float((w.sum(axis=ax)*c).sum()/w.sum()) for ax, c in
                          (((1, 2), g["z"]), ((0, 2), g["y"]), ((0, 1), g["x"])))
            focal = f"{xc},{yc},{zc}"
            for tag in ["fom"] + args.models:
                name = f"{tag}_{k}_{i}"
                subprocess.run([sys.executable, os.path.abspath(__file__), "--cache", args.cache,
                                "--render", name, str(level), str(clim), focal,
                                os.path.join(args.cache, f"{name}.png")], check=True)
    tile(args, time, idx, levels)


if __name__ == "__main__":
    main()
