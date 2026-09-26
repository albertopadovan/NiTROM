"""
Average error over time, eq. (3.9), for the pre-projected GAS-OpInf models:

    e(t) = (1/N_traj) sum_j (1/alpha_j) ||q^(j)(t) - qhat^(j)(t)||^2,

with alpha_j the time-averaged energy of trajectory j, so amplitudes spanning
~2700x contribute comparably.  Both panels score against the FULL Navier-Stokes
solution: the models are built in the 1000-mode pre-projected space, and their
predictions are lifted back before the comparison, so the pre-projection error
is charged to the model.

Usage: python plot_avg_error_cavity.py [--r 30] [--pre 1000]
Writes figures/avg_error_gas_r<r>.png.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from balanced_opinf import ROM
from plot_energy_cavity import load_nitrom
from train_models import load_train, HERE, DATA

STYLES = {"pod": ("tab:blue", "-."), "balanced": ("tab:olive", ":")}


def avg_error(rom, Phi, Psi, trajs, time):
    """e(t) of eq. (3.9); a diverged trajectory contributes its own energy."""
    nt = len(time)
    e = np.zeros(nt)
    nbad = 0
    for X in trajs:
        alpha = np.mean(np.sum(X**2, axis=0))
        Z = rom.integrate(Psi.T @ X[:, 0], time, 1.0)
        ok = np.all(np.isfinite(Z), axis=0)
        Xh = Phi @ np.where(ok, Z, 0.0)
        if not ok.all():
            nbad += 1
            Xh[:, ~ok] = 0.0          # no prediction there; charge the full energy
        e += np.sum((X - Xh)**2, axis=0)/alpha
    return e/len(trajs), nbad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--pre", type=int, default=1000)
    args = ap.parse_args()
    r = args.r

    trajs_n, betas, time = load_train()
    sets = {"training": ([abs(b)*T for b, T in zip(betas, trajs_n)], betas)}
    bt = np.load(os.path.join(DATA, "setup.npz"))["betas_test"]
    sets["held-out"] = (
        [abs(b)*np.load(os.path.join(DATA, f"test_traj_{j:03d}.npy")).astype(np.float64)
         for j, b in enumerate(bt)], bt)

    models = []
    for basis, (col, ls) in STYLES.items():
        path = os.path.join(HERE, f"opinf_gas_{basis}_pre{args.pre}_r{r}.npy")
        if not os.path.exists(path):
            print(f"note: {os.path.basename(path)} missing")
            continue
        d = np.load(path, allow_pickle=True).item()
        models.append((f"GAS-OpInf, {basis} ({d['epochs']} ep)",
                       np.asarray(d["Phi"]), np.asarray(d["Psi"]),
                       ROM(np.asarray(d["Ar"]), np.asarray(d["Hr"]), 2),
                       np.linalg.eigvals(np.asarray(d["Ar"])).real.max(),
                       col, ls))

    gpath = os.path.join(HERE, f"roms_r{r}_gasnitrom_pre{args.pre}.npy")
    if os.path.exists(gpath):
        g = load_nitrom(gpath, r)     # oblique decode; its bases are not biorthogonal
        models.append((f"GasNiTROM ({g['n_iters']} it)", g["dec"], g["Psi"],
                       ROM(g["Ar"], g["Hr"], 2),
                       np.linalg.eigvals(g["Ar"]).real.max(),
                       "tab:brown", (0, (3, 1, 1, 1))))
    else:
        print(f"note: {os.path.basename(gpath)} missing")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True,
                           constrained_layout=True)
    for a, (what, (trajs, bs)) in zip(ax, sets.items()):
        print(f"\n=== {what} ({len(trajs)} trajectories, "
              f"beta = {list(np.round(np.asarray(bs, float), 3))}) ===")
        for name, Phi, Psi, rom, ev, col, ls in models:
            e, nbad = avg_error(rom, Phi, Psi, trajs, time)
            lab = f"{name}" + (f"  [{nbad} diverged]" if nbad else "")
            a.semilogy(time, e, color=col, ls=ls, lw=1.7, label=lab)
            print(f"  {name:<28} maxRe(A) {ev:+.4f}   e(40) = {e[-1]:.3e}   "
                  f"time-mean e = {np.mean(e):.3e}"
                  + (f"   {nbad}/{len(trajs)} diverged" if nbad else ""))
        a.axhline(1.0, color="k", lw=0.9, ls="--")
        a.text(time[-1], 1.15, "predicting zero", ha="right", fontsize=8)
        a.set_title(f"{what} impulses", fontsize=10)
        a.set_xlabel("$t$")
        a.set_xlim(0, time[-1])
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8, loc="lower right")
    ax[0].set_ylabel(r"$e(t)$")
    fig.suptitle(f"Cavity, r = {r}: average error (3.9) of the pre-projected "
                 f"GAS-OpInf models, scored against the full FOM", fontsize=11)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", f"avg_error_gas_r{r}.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
