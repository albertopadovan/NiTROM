"""
Vorticity snapshots of the three GAS-guaranteed ROMs against the FOM for one
impulse response, training or held-out.

Rows: FOM / GAS-OpInf (POD) / GAS-OpInf (balanced) / GasNiTROM.

Colour scale is symmetric and set per column by the FOM field at that time, so
each column is comparable to the FOM but columns are not comparable to each
other.

Usage: python plot_snapshots_gas.py [--r 30] [--beta 0.901] [--on test]
Writes figures/snapshots_gas_r<r>_<train|test>_beta<beta>.png.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cavity import Cavity
from plot_energy_gas import load_gas_models
from train_models import load_train, vorticity, HERE, DATA

TIMES = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--beta", type=float, default=0.901)
    ap.add_argument("--on", choices=["train", "test"], default="test")
    args = ap.parse_args()
    r = args.r

    trajs_n, betas, time = load_train()
    if args.on == "test":
        betas = np.load(os.path.join(DATA, "setup.npz"))["betas_test"]
        trajs = [abs(b)*np.load(os.path.join(DATA, f"test_traj_{j:03d}.npy")
                                ).astype(np.float64)
                 for j, b in enumerate(betas)]
    else:
        trajs = [abs(b)*T for b, T in zip(betas, trajs_n)]
    j = int(np.argmin(np.abs(np.asarray(betas) - args.beta)))
    X = trajs[j]
    print(f"{args.on} trajectory {j}: beta = {betas[j]:g}, r = {r}, "
          f"peak ||x||^2 = {np.sum(X**2, axis=0).max():.3e}")

    cav = Cavity()
    rows = [("FOM", X)]
    for nm, Ph, Ps, rom, ev, _, _ in load_gas_models(r):
        Z = rom.integrate(Ps.T @ X[:, 0], time, 1.0)
        ok = np.all(np.isfinite(Z), axis=0)
        Xh = Ph @ np.where(ok, Z, np.nan)
        err = (np.linalg.norm(X - Xh)/np.linalg.norm(X) if ok.all() else np.inf)
        print(f"  {nm:<28} err {err:.3f}"
              + ("" if ok.all() else
                 f"  diverges at t = {time[np.argmax(~ok)]:.1f}"))
        rows.append((nm.replace(" (", "\n("), Xh))

    idx = [int(np.argmin(np.abs(time - t))) for t in TIMES]
    fig, ax = plt.subplots(len(rows), len(idx),
                           figsize=(2.35*len(idx), 2.45*len(rows)),
                           squeeze=False)
    for c, kk in enumerate(idx):
        lim = np.abs(vorticity(cav.flow, X[:, kk])).max()
        for rr, (label, F) in enumerate(rows):
            a = ax[rr, c]
            q = F[:, kk]
            if np.all(np.isfinite(q)):
                a.imshow(vorticity(cav.flow, q), origin="lower",
                         extent=[0, 1, 0, 1], cmap="bwr", vmin=-lim, vmax=lim)
            else:
                a.text(0.5, 0.5, "diverged", ha="center", va="center",
                       fontsize=11, color="0.4", transform=a.transAxes)
            a.set_xticks([])
            a.set_yticks([])
            if rr == 0:
                a.set_title(f"$t = {time[kk]:g}$", fontsize=11)
            if c == 0:
                a.set_ylabel(label, fontsize=8)
    what = "held-out" if args.on == "test" else "training"
    fig.suptitle(rf"Cavity vorticity, {what} impulse $\beta = {betas[j]:g}$, "
                 rf"$r = {r}$ (colour scale set by the FOM in each column)",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures",
                       f"snapshots_gas_r{r}_{args.on}_beta{betas[j]:g}.png")
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
