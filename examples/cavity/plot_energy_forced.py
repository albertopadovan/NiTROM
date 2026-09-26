"""
Energy ||x(t)||^2 of the r-dimensional ROMs against the FOM on the SINUSOIDALLY
FORCED cases of data/forced_* (the paper's Fig. 9): zero initial condition,
w(t) = W_AMP sin(k omega t) entering through the actuator B.

This is the hardest test available here.  Every model was fitted on decaying
impulse responses with no input at all, so the forcing probes dynamics the
training data never contained -- and it never lets the state decay, so a model
that is merely over-damped cannot hide.

The reduced input operator is not inferred: with z = Psi^T q and
dq/dt = f(q) + B w, projecting gives dz/dt = Psi^T f(q) + (Psi^T B) w, so
B_r = Psi^T B exactly.  Each model uses its own Psi.

Usage: python plot_energy_forced.py [--r 30] [--reg-pod 1e-2] [--reg-bal 1e2]
Writes figures/energy_forced_r<r>.png.
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
from train_models import load_train, HERE, DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--reg-pod", type=float, default=1e-2)
    ap.add_argument("--reg-bal", type=float, default=1e2)
    ap.add_argument("--w-amp", type=float, default=None,
                    help="forcing amplitude; default is the one in setup.npz "
                         "(0.1).  Other amplitudes come from generate_forced.py.")
    args = ap.parse_args()
    r = args.r

    trajs_n, betas, time = load_train()
    trajs_fit = [abs(b)*T for b, T in zip(betas, trajs_n)]
    dt = time[1] - time[0]

    setup = np.load(os.path.join(DATA, "setup.npz"))
    forced = setup["forced"]
    w_amp = float(setup["w_amp"]) if args.w_amp is None else args.w_amp
    # The original 0.1 set kept its unprefixed names; everything else is
    # written by generate_forced.py with the amplitude in the filename.
    legacy = abs(w_amp - float(setup["w_amp"])) < 1e-12

    def forced_path(omega, k):
        name = (f"forced_w{omega:g}_k{k}.npy" if legacy
                else f"forced_A{w_amp:g}_w{omega:g}_k{k}.npy")
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found -- run: python generate_forced.py "
                f"--amp {w_amp:g}")
        return path
    cav = Cavity()
    B = cav.B                                   # unit-norm, divergence-free

    d = np.load(os.path.join(HERE, "balancing_r50.npy"), allow_pickle=True).item()
    U = get_pod(trajs_fit, r)
    models = []

    nitrom_path = os.path.join(HERE, f"roms_r{r}_nitrom_balanced.npy")
    if os.path.exists(nitrom_path):
        n = load_nitrom(nitrom_path, r)
        models.append((f"NiTROM ({n['n_iters']} it)", n["dec"], n["Psi"],
                       ROM(n["Ar"], n["Hr"], 2), "tab:purple", (0, (5, 1))))
    for name, Ph, Ps, reg, col, ls in [
        ("balanced OpInf", d["Phi"][:, :r], d["Psi"][:, :r], args.reg_bal,
         "tab:red", "--"),
        ("POD OpInf", U, U, args.reg_pod, "tab:blue", "-."),
    ]:
        models.append((rf"{name} ($\lambda$ = {reg:.0e})", Ph, Ps,
                       opinf(trajs_fit, Ps, dt, "quadratic", reg=reg, C=None),
                       col, ls))

    fig, ax = plt.subplots(2, 3, figsize=(15, 7.2), sharex=True,
                           constrained_layout=True)
    print(f"r = {r}, forcing amplitude {w_amp:g}, zero initial condition\n")
    hdr = f"{'case':>16}" + "".join(f"{m[0].split(' (')[0]:>22}" for m in models)
    print(hdr)
    for i, (omega, k) in enumerate(forced):
        a = ax.flat[i]
        Y = np.load(forced_path(omega, int(k))).astype(np.float64)
        E = np.sum(Y**2, axis=0)
        a.semilogy(time, E, "k", lw=2.2, label="FOM")
        line = f"{f'w={omega:g}, k={int(k)}':>16}"
        for name, Ph, Ps, rom, col, ls in models:
            b_r = Ps.T @ (w_amp*B)
            Z = rom.integrate(np.zeros(r), time, 1.0,
                              forcing=lambda t, b=b_r, kk=int(k), om=omega:
                                  np.sin(kk*om*t)*b)
            ok = np.all(np.isfinite(Z), axis=0)
            Eh = np.where(ok, np.sum((Ph @ np.where(ok, Z, 0.0))**2, axis=0),
                          np.nan)
            err = (np.linalg.norm(Y[:, ok] - Ph @ Z[:, ok])
                   / np.linalg.norm(Y[:, ok])) if ok.all() else np.inf
            a.semilogy(time, Eh, color=col, ls=ls, lw=1.5,
                       label=f"{name} (err {err:.2f})" if np.isfinite(err)
                       else f"{name} (diverged)")
            if not ok.all():
                a.axvline(time[np.argmax(~ok)], color=col, lw=0.8, alpha=0.6)
            line += f"{err:>22.3f}" if np.isfinite(err) else f"{'diverged':>22}"
        print(line)
        a.set_title(rf"$\omega = {omega:g}$, $k = {int(k)}$   "
                    rf"($w = {w_amp:g}\sin({int(k)}\omega t)$)", fontsize=10)
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=7, loc="lower right")
    for a in ax[-1]:
        a.set_xlabel("$t$")
    for a in ax[:, 0]:
        a.set_ylabel(r"$\|x(t)\|^2$")
    fig.suptitle(f"Cavity: sinusoidally forced response, r = {r} "
                 f"(zero IC; no forced data was used in training)", fontsize=11)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    tag = "" if legacy else f"_A{w_amp:g}"
    out = os.path.join(HERE, "figures", f"energy_forced_r{r}{tag}.png")
    fig.savefig(out, dpi=140)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
