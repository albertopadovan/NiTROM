"""
Vorticity snapshots of the ROMs against the FOM for one sinusoidally forced
case: zero initial condition, w(t) = amp sin(k omega t) through the actuator B.

Rows: FOM / NiTROM / balanced OpInf / POD OpInf.  The reduced input operator is
B_r = Psi^T B exactly (see plot_energy_forced.py); nothing about the forced
response was fitted.

Colour scale is symmetric and set per column by the FOM field at that time.

Usage: python plot_snapshots_forced.py [--r 30] [--omega 1] [--k 4] [--w-amp 0.3]
Writes figures/snapshots_forced_r<r>_A<amp>_w<omega>_k<k>.png.
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
from train_models import load_train, vorticity, HERE, DATA

TIMES = [2.0, 5.0, 10.0, 20.0, 30.0, 40.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--w-amp", type=float, default=0.3)
    ap.add_argument("--reg-pod", type=float, default=1e-2)
    ap.add_argument("--reg-bal", type=float, default=1e2)
    args = ap.parse_args()
    r, omega, k, amp = args.r, args.omega, args.k, args.w_amp

    trajs_n, betas, time = load_train()
    trajs_fit = [abs(b)*T for b, T in zip(betas, trajs_n)]
    dt = time[1] - time[0]

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
    d = np.load(os.path.join(HERE, "balancing_r50.npy"), allow_pickle=True).item()
    U = get_pod(trajs_fit, r)
    rows = [("FOM", Y)]

    models = []
    nitrom_path = os.path.join(HERE, f"roms_r{r}_nitrom_balanced.npy")
    if os.path.exists(nitrom_path):
        n = load_nitrom(nitrom_path, r)
        models.append((f"NiTROM ({n['n_iters']} it)", n["dec"], n["Psi"],
                       ROM(n["Ar"], n["Hr"], 2)))
    for nm, Ph, Ps, reg in [
        ("balanced OpInf", d["Phi"][:, :r], d["Psi"][:, :r], args.reg_bal),
        ("POD OpInf", U, U, args.reg_pod),
    ]:
        models.append((f"{nm}\n" + rf"$\lambda$ = {reg:.0e}", Ph, Ps,
                       opinf(trajs_fit, Ps, dt, "quadratic", reg=reg, C=None)))

    for nm, Ph, Ps, rom in models:
        b_r = Ps.T @ (amp*cav.B)
        Z = rom.integrate(np.zeros(r), time, 1.0,
                          forcing=lambda t, b=b_r: np.sin(k*omega*t)*b)
        ok = np.all(np.isfinite(Z), axis=0)
        if not ok.all():
            print(f"  {nm.splitlines()[0]}: diverges at t = "
                  f"{time[np.argmax(~ok)]:.1f}")
        rows.append((nm, Ph @ np.where(ok, Z, np.nan)))

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
                a.set_ylabel(label, fontsize=9)
    fig.suptitle(rf"Cavity vorticity, forced $w = {amp:g}\sin({k}\omega t)$, "
                 rf"$\omega = {omega:g}$, $r = {r}$ "
                 rf"(colour scale set by the FOM in each column)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures",
                       f"snapshots_forced_r{r}_A{amp:g}_w{omega:g}_k{k}.png")
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
