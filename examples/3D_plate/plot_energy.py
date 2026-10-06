"""Perturbation energy E(t) = <q', q'>_W of the training trajectories, with
the weighted inner product of the stretched mesh: W = V / V_min, V the dual
cell volume of each u, v, w DOF (make_initial_conditions.cell_volume_weights).

Uses the energies saved by generate_trajectories.py (fom_data/energy_KKK.npy,
computed in float64 on the GPU during the march); ``--check K`` recomputes two
snapshots of trajectory K from the stored float32 fields as a cross-check.

Usage (incompreso venv):
    python plot_energy.py [--data fom_data] [--check 2]
Writes figures/trajectory_energy.png.
"""

import argparse
import glob
import io
import contextlib
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from case import DEFAULT_CONFIG, Case  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIONS = ("leading edge", "midchord", "trailing edge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="case config (wing: config.yaml, plate: config_plate.yaml)")
    ap.add_argument("--data", default=None, help="trajectory directory (default: the case's)")
    ap.add_argument("--check", type=int, default=None,
                    help="recompute E for this trajectory from its stored fields")
    args = ap.parse_args()
    C = Case(args.config)
    args.data = args.data or C.fom_data

    meta = np.load(os.path.join(args.data, "meta.npz"))
    t, params = meta["time"], meta["parameters"]
    ks = sorted(int(re.search(r"energy_(\d+)\.npy", f).group(1))
                for f in glob.glob(os.path.join(args.data, "energy_*.npy")))
    betas = np.unique(params[:, 0])

    if args.check is not None:
        import incompreso
        from make_initial_conditions import cell_volume_weights
        with contextlib.redirect_stdout(io.StringIO()):
            mesh = incompreso.parse_input_file(C.post_config)["mesh"]
        w = cell_volume_weights(mesh)
        Q = np.load(os.path.join(args.data, f"fluct_{args.check:03d}.npy"), mmap_mode="r")
        E = np.load(os.path.join(args.data, f"energy_{args.check:03d}.npy"))
        for i in (0, len(t)//4):
            qi = np.asarray(Q[i], dtype=np.float64)
            Ei = float(np.dot(w*qi, qi))
            print(f"check traj {args.check}, t = {t[i]:g}: from fields {Ei:.6e}, "
                  f"saved {E[i]:.6e} (rel diff {abs(Ei - E[i])/E[i]:.1e})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    styles = {b: s for b, s in zip(betas, ("-", "--", ":"))}
    for k in ks:
        beta, x0, y0 = params[k]
        E = np.load(os.path.join(args.data, f"energy_{k:03d}.npy"))
        j = k // len(betas)
        kw = dict(color=f"C{j}", ls=styles[beta], lw=1.6,
                  label=f"{STATIONS[j]}, $\\beta$ = {beta:g}")
        axes[0].semilogy(t, E, **kw)
        axes[1].semilogy(t, E/E[0], **kw)
    axes[0].set_ylabel(r"$E = \langle q', q' \rangle_W$")
    axes[1].set_ylabel(r"$E / E(0)$")
    for ax in axes:
        ax.set_xlabel("$t$")
        ax.grid(alpha=0.3, which="both")
        ax.set_xlim(t[0], t[-1])
    axes[1].legend(fontsize=8)
    fig.suptitle(f"3D {C.name} training trajectories ({len(ks)} of {len(params)}): "
                 "perturbation energy, W = V / V_min")
    os.makedirs(C.figures, exist_ok=True)
    out = os.path.join(C.figures, "trajectory_energy.png")
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
