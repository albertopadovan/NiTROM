"""
Sinusoidally forced response of the three globally-asymptotically-stable ROMs:
GAS-OpInf on POD, GAS-OpInf on the balanced basis, and GasNiTROM.

Zero initial condition, w(t) = amp sin(k omega t) through the actuator B, for
the (omega, k) pairs of the paper's Fig. 9.  Nothing about the forced response
was fitted: every model was trained on unforced, decaying impulse responses, and
the reduced input operator is exactly B_r = Psi^T B, since projecting
dq/dt = f(q) + B w gives dz/dt = Psi^T f(q) + (Psi^T B) w.

This is the strictest test available here -- the forcing never lets the state
decay, so a model that is merely over-damped cannot hide, and amplitudes beyond
the training data probe the quadratic term directly.

Usage: python plot_forced_gas.py [--r 30] [--w-amp 0.1]
       (amplitudes other than 0.1 need generate_forced.py --amp <a> first)
Writes figures/forced_gas_r<r>[_A<amp>].png.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cavity import Cavity
from plot_energy_gas import load_gas_models
from train_models import load_train, HERE, DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--w-amp", type=float, default=None)
    ap.add_argument("--pre", type=int, default=0,
                    help="use the pre-projected models (opinf_gas_*_pre<N>_*)")
    args = ap.parse_args()
    r = args.r

    _, _, time = load_train()
    setup = np.load(os.path.join(DATA, "setup.npz"))
    forced = setup["forced"]
    amp = float(setup["w_amp"]) if args.w_amp is None else args.w_amp
    legacy = abs(amp - float(setup["w_amp"])) < 1e-12

    def path_for(omega, k):
        name = (f"forced_w{omega:g}_k{k}.npy" if legacy
                else f"forced_A{amp:g}_w{omega:g}_k{k}.npy")
        p = os.path.join(DATA, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} not found -- run "
                                    f"python generate_forced.py --amp {amp:g}")
        return p

    cav = Cavity()
    B = cav.B
    models = load_gas_models(r, pre=args.pre)

    fig, ax = plt.subplots(2, 3, figsize=(15, 7.2), sharex=True,
                           constrained_layout=True)
    print(f"r = {r}, forcing amplitude {amp:g}, zero initial condition\n")
    print(f"{'case':>16}" + "".join(f"{m[0].split(' (')[0]:>24}" for m in models))
    agg = {m[0]: [0.0, 0.0] for m in models}
    for i, (omega, k) in enumerate(forced):
        k = int(k)
        a = ax.flat[i]
        Y = np.load(path_for(omega, k)).astype(np.float64)
        a.semilogy(time, np.sum(Y**2, axis=0), "k", lw=2.2,
                   label="FOM" if i == 0 else None)
        line = f"{f'w={omega:g}, k={k}':>16}"
        for name, Ph, Ps, rom, ev, col, ls in models:
            b_r = Ps.T @ (amp*B)
            Z = rom.integrate(np.zeros(r), time, 1.0,
                              forcing=lambda t, b=b_r, kk=k, om=omega:
                                  np.sin(kk*om*t)*b)
            ok = np.all(np.isfinite(Z), axis=0)
            Yh = Ph @ np.where(ok, Z, 0.0)
            E = np.where(ok, np.sum(Yh**2, axis=0), np.nan)
            err = (np.linalg.norm(Y - Yh)/np.linalg.norm(Y) if ok.all()
                   else np.inf)
            a.semilogy(time, E, color=col, ls=ls, lw=1.6,
                       label=name if i == 0 else None)
            line += (f"{err:>24.3f}" if np.isfinite(err) else f"{'diverged':>24}")
            if np.isfinite(err):
                agg[name][0] += np.linalg.norm(Y - Yh)**2
                agg[name][1] += np.linalg.norm(Y)**2
        print(line)
        a.set_title(rf"$\omega = {omega:g}$, $k = {k}$   "
                    rf"($w = {amp:g}\,\sin({k}\omega t)$)", fontsize=10)
        a.grid(alpha=0.3, which="both")
    print(f"{'aggregate':>16}" + "".join(
        f"{np.sqrt(agg[m[0]][0]/agg[m[0]][1]):>24.3f}"
        if agg[m[0]][1] else f"{'-':>24}" for m in models))
    ax.flat[0].legend(fontsize=7, loc="lower right")
    for a in ax[-1]:
        a.set_xlabel("$t$")
    for a in ax[:, 0]:
        a.set_ylabel(r"$\|x(t)\|^2$")
    fig.suptitle(f"Cavity, r = {r}: GAS-guaranteed ROMs under sinusoidal "
                 f"forcing of amplitude {amp:g} (zero IC, no forced training data)",
                 fontsize=11)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    tag = "" if legacy else f"_A{amp:g}"
    pre = f"_pre{args.pre}" if args.pre else ""
    out = os.path.join(HERE, "figures", f"forced_gas_r{r}{pre}{tag}.png")
    fig.savefig(out, dpi=140)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
