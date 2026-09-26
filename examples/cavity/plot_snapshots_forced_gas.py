"""
Vorticity snapshots of the three GAS-guaranteed ROMs against the FOM for one
sinusoidally forced case: zero initial condition, w(t) = amp sin(k omega t)
through the actuator B.

Rows: FOM / GAS-OpInf (POD) / GAS-OpInf (balanced) / GasNiTROM.  The reduced
input operator is B_r = Psi^T B exactly; nothing about the forced response was
fitted, and none of the models saw forced data in training.

Colour scale is symmetric and set per column by the FOM field at that time, so
each column is comparable to the FOM but columns are not comparable to each
other.

Usage: python plot_snapshots_forced_gas.py [--r 30] [--omega 1] [--k 2] [--w-amp 2.0]
Writes figures/snapshots_forced_gas_r<r>_A<amp>_w<omega>_k<k>.png.
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

TIMES = [2.0, 5.0, 10.0, 20.0, 30.0, 40.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--w-amp", type=float, default=2.0)
    args = ap.parse_args()
    r, omega, k, amp = args.r, args.omega, args.k, args.w_amp

    _, _, time = load_train()
    setup = np.load(os.path.join(DATA, "setup.npz"))
    legacy = abs(amp - float(setup["w_amp"])) < 1e-12
    name = (f"forced_w{omega:g}_k{k}.npy" if legacy
            else f"forced_A{amp:g}_w{omega:g}_k{k}.npy")
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found -- run "
                                f"python generate_forced.py --amp {amp:g}")
    Y = np.load(path).astype(np.float64)
    print(f"forced case omega = {omega:g}, k = {k}, amplitude {amp:g}; "
          f"FOM peak ||x||^2 = {np.sum(Y**2, axis=0).max():.3e}")

    cav = Cavity()
    rows = [("FOM", Y)]
    for nm, Ph, Ps, rom, ev, _, _ in load_gas_models(r):
        b_r = Ps.T @ (amp*cav.B)
        Z = rom.integrate(np.zeros(r), time, 1.0,
                          forcing=lambda t, b=b_r: np.sin(k*omega*t)*b)
        ok = np.all(np.isfinite(Z), axis=0)
        Yh = Ph @ np.where(ok, Z, np.nan)
        err = (np.linalg.norm(Y - Yh)/np.linalg.norm(Y) if ok.all() else np.inf)
        print(f"  {nm:<28} err {err:.3f}"
              + ("" if ok.all() else
                 f"  diverges at t = {time[np.argmax(~ok)]:.1f}"))
        rows.append((nm.replace(" (", "\n("), Yh))

    idx = [int(np.argmin(np.abs(time - t))) for t in TIMES]
    fig, ax = plt.subplots(len(rows), len(idx),
                           figsize=(2.35*len(idx), 2.45*len(rows)),
                           squeeze=False)
    for c, kk in enumerate(idx):
        lim = np.abs(vorticity(cav.flow, Y[:, kk])).max()
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
    fig.suptitle(rf"Cavity vorticity, forced $w = {amp:g}\sin({k}\omega t)$, "
                 rf"$\omega = {omega:g}$, $r = {r}$ "
                 rf"(colour scale set by the FOM in each column)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures",
                       f"snapshots_forced_gas_r{r}_A{amp:g}_w{omega:g}_k{k}.png")
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
