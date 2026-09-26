"""Perturbation energy of the FOM training trajectories -- figure 10(b).

E_pert(t) = <q, q>_W with W the cell volume over the smallest cell volume,
the normalisation the paper states for figure 10(b), summed over the ROM
domain D = [-4, 13] x [-3, 3] only (see rom_domain.py) -- the same data the
ROMs are trained on.  Plots whatever ``generate_trajectories.py`` has written
so far, so it can be run while the remaining trajectories are still marching.

Usage: python plot_training_energy.py [--data fom_data/]
Writes figures/airfoil_training_energy.png.
"""

from __future__ import annotations

import argparse
import glob
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_initial_conditions import cell_volume_weights
from rom_domain import crop_mask

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data"))
    args = ap.parse_args()

    meta = np.load(os.path.join(args.data, "meta.npz"))
    t, params = meta["time"], meta["parameters"]
    with np.load(str(meta["baseflow"])) as snap:
        mesh = SimpleNamespace(**{c: snap[c] for c in ("xu", "yu", "xv", "yv")})
        mask = crop_mask(snap)
    w = cell_volume_weights(mesh)[mask]

    files = sorted(glob.glob(os.path.join(args.data, "fluct_[0-9][0-9][0-9].npy")))
    if not files:
        raise SystemExit(f"no trajectories in {args.data} yet")

    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    print(f"{'traj':>4}{'beta':>6}{'x_c':>9}{'y_c':>9}{'E(0)':>10}"
          f"{'max E':>10}{'t_max':>7}{'gain':>8}{'E(end)':>10}")
    for f in files:
        k = int(os.path.basename(f)[6:9])
        Q = np.load(f, mmap_mode="r")[mask]
        E = np.einsum("it,i,it->t", Q, w, Q)
        beta, xc, yc = params[k]
        print(f"{k:>4}{beta:>6g}{xc:>9.4f}{yc:>9.4f}{E[0]:>10.3e}{E.max():>10.3e}"
              f"{t[E.argmax()]:>7.1f}{E.max()/E[0]:>8.0f}{E[-1]:>10.3e}")
        ax.plot(t, E, "k")
        ax.annotate(f"{k}", (t[E.argmax()], E.max()), textcoords="offset points",
                    xytext=(3, 3), fontsize=8)

    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$E_{\mathrm{pert}}$")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", "airfoil_training_energy.png")
    fig.savefig(out, dpi=200)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
