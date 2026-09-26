"""
Vorticity snapshots of the r = 20 models against the FOM, for one training
impulse response.

Rows, top to bottom:
  FOM             the Navier-Stokes field
  NiTROM          Phi (Psi^T Phi)^-1 z(t), bases and tensors both optimized
  balanced OpInf  the Lall-balanced basis with operators fitted by OpInf
  POD OpInf       the POD basis with operators fitted by OpInf

Colour scale is symmetric and set per column by the FOM field at that time, so
each column is comparable to the FOM but columns are not comparable to each
other.

Usage: python plot_snapshots.py [--r 20] [--beta 1.0] [--reg-pod 1e3] [--reg-bal 1e0]
Writes figures/snapshots_cavity_r<r>_beta<beta>.png.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cavity import Cavity
from balanced_opinf import ROM, opinf
from plot_energy_cavity import load_nitrom
from train_pod_opinf import get_pod
from train_models import load_train, vorticity, HERE

TIMES = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=20)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--reg-pod", type=float, default=1e3)
    ap.add_argument("--reg-bal", type=float, default=1e0)
    args = ap.parse_args()
    r = args.r

    trajs_n, betas, time = load_train()
    trajs = [abs(b)*T for b, T in zip(betas, trajs_n)]
    dt = time[1] - time[0]
    j = int(np.argmin(np.abs(betas - args.beta)))
    X = trajs[j]
    print(f"trajectory {j}: beta = {betas[j]:g}, r = {r}")

    d = np.load(os.path.join(HERE, "balancing_r50.npy"), allow_pickle=True).item()
    rows = [("FOM", X)]

    nitrom_path = os.path.join(HERE, f"roms_r{r}_nitrom_balanced.npy")
    if os.path.exists(nitrom_path):
        n = load_nitrom(nitrom_path, r)
        models = [(f"NiTROM ({n['n_iters']} it)", n["dec"], n["Psi"],
                   ROM(n["Ar"], n["Hr"], 2))]
    else:
        models = []
        print(f"note: {os.path.basename(nitrom_path)} not found")

    for name, (Ph, Ps, reg) in [
        ("balanced OpInf", (d["Phi"][:, :r], d["Psi"][:, :r], args.reg_bal)),
        ("POD OpInf", (get_pod(trajs, r), None, args.reg_pod)),
    ]:
        Ps = Ph if Ps is None else Ps
        models.append((f"{name}\n" + rf"$\lambda$ = {reg:.0e}", Ph, Ps,
                       opinf(trajs, Ps, dt, "quadratic", reg=reg, C=None)))

    for name, Ph, Ps, rom in models:
        Z = rom.integrate(Ps.T @ X[:, 0], time, 1.0)
        ok = np.all(np.isfinite(Z), axis=0)
        if not ok.all():
            print(f"  {name.splitlines()[0]}: diverges at t = "
                  f"{time[np.argmax(~ok)]:.1f}")
        rows.append((name, Ph @ np.where(ok, Z, np.nan)))

    cav = Cavity()
    idx = [int(np.argmin(np.abs(time - t))) for t in TIMES]
    fig, ax = plt.subplots(len(rows), len(idx),
                           figsize=(2.35*len(idx), 2.45*len(rows)),
                           squeeze=False)
    for c, k in enumerate(idx):
        lim = np.abs(vorticity(cav.flow, X[:, k])).max()    # FOM sets the scale
        for rr, (label, F) in enumerate(rows):
            a = ax[rr, c]
            q = F[:, k]
            if np.all(np.isfinite(q)):
                a.imshow(vorticity(cav.flow, q), origin="lower",
                         extent=[0, 1, 0, 1], cmap="bwr", vmin=-lim, vmax=lim)
            else:
                a.text(0.5, 0.5, "diverged", ha="center", va="center",
                       fontsize=11, color="0.4", transform=a.transAxes)
            a.set_xticks([])
            a.set_yticks([])
            if rr == 0:
                a.set_title(f"$t = {time[k]:g}$", fontsize=11)
            if c == 0:
                a.set_ylabel(label, fontsize=9)
    fig.suptitle(rf"Cavity vorticity, $\beta = {betas[j]:g}$, $r = {r}$ "
                 rf"(colour scale set by the FOM in each column)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures",
                       f"snapshots_cavity_r{r}_beta{betas[j]:g}.png")
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
