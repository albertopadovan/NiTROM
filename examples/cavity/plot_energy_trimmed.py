"""
Energy ||x(t)||^2 of the balanced-OpInf ROM fitted WITHOUT the early transient
(t < T_MIN), against the FOM and against the same fit on all snapshots.

Three model curves per panel:
  all t, from t=0        the reference balanced OpInf (lambda chosen so it
                         stays bounded over the window)
  t>=T_MIN, from t=0     the trimmed fit asked to propagate an initial
                         condition from a regime it never saw
  t>=T_MIN, from t=T_MIN the trimmed fit restarted inside its own fit window --
                         the question it was actually trained to answer

Dropping t < 3 makes the linear operator stable (max Re eig goes from +0.0078
to about -0.008) but costs accuracy, and it cannot repair the initial condition:
the balanced basis reconstructs the impulse IC with ~190% error, so any forecast
begun at t = 0 starts in the wrong place regardless of the operator.

Usage: python plot_energy_trimmed.py [--r 30] [--t-min 3] [--reg-all 1e2] [--reg-trim 1e1]
Writes figures/energy_trimmed_r<r>.png.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from balanced_opinf import opinf
from train_models import load_train, HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--t-min", type=float, default=3.0)
    ap.add_argument("--reg-all", type=float, default=1e2)
    ap.add_argument("--reg-trim", type=float, default=1e1)
    args = ap.parse_args()
    r = args.r

    trajs_n, betas, time = load_train()
    trajs = [abs(b)*T for b, T in zip(betas, trajs_n)]
    dt = time[1] - time[0]
    i0 = int(np.searchsorted(time, args.t_min))
    d = np.load(os.path.join(HERE, "balancing_r50.npy"), allow_pickle=True).item()
    Phi, Psi = d["Phi"][:, :r], d["Psi"][:, :r]

    rom_all = opinf(trajs, Psi, dt, "quadratic", reg=args.reg_all, C=None)
    rom_cut = opinf([X[:, i0:] for X in trajs], Psi, dt, "quadratic",
                    reg=args.reg_trim, C=None)
    print(f"r = {r}, trimming t < {time[i0]:g} "
          f"({len(time)-i0} of {len(time)} snapshots kept for the fit)")
    for tag, rom in (("all t  ", rom_all), ("t>=t_min", rom_cut)):
        print(f"  {tag} max Re eig(Ar) = "
              f"{np.linalg.eigvals(rom.Ar).real.max():+.4f}, "
              f"||Hr|| = {np.linalg.norm(rom.Hr):.2e}")

    # (label, rom, start index, colour, linestyle)
    # Both models are launched from the true initial condition at t = 0.  The
    # trimmed model's lambda was selected on how well it propagates a state at
    # t = t_min (see sweep_opinf.py --t-min), but it is judged here on the same
    # impulse-response task as everything else, which is the honest comparison:
    # its fit window excluded t < t_min, so z(0) = Psi^T q(0) is outside it.
    specs = [
        (rf"all $t$ fit ($\lambda$={args.reg_all:.0e})",
         rom_all, 0, "tab:red", "--"),
        (rf"$t\geq{args.t_min:g}$ fit ($\lambda$={args.reg_trim:.0e})",
         rom_cut, 0, "tab:green", "-"),
        (rf"$t\geq{args.t_min:g}$ fit, restarted at $t={args.t_min:g}$ (reference)",
         rom_cut, i0, "tab:green", ":"),
    ]

    fig, ax = plt.subplots(2, 4, figsize=(18, 7.2), sharex=True,
                           constrained_layout=True)
    print(f"\n{'beta':>7}" + "".join(f"{lbl.split(' (')[0][:22]:>24}"
                                     for lbl, *_ in specs))
    for j, (b, X) in enumerate(zip(betas, trajs)):
        a = ax.flat[j]
        a.semilogy(time, np.sum(X**2, axis=0), "k", lw=2.2, label="FOM")
        line = f"{b:>7g}"
        for lbl, rom, start, col, ls in specs:
            Xw, tw = X[:, start:], time[:X.shape[1]-start]
            Z = rom.integrate(Psi.T @ Xw[:, 0], tw, 1.0)
            ok = np.all(np.isfinite(Z), axis=0)
            Xh = Phi @ np.where(ok, Z, 0.0)
            E = np.where(ok, np.sum(Xh**2, axis=0), np.nan)
            err = (np.linalg.norm(Xw - Xh)/np.linalg.norm(Xw) if ok.all()
                   else np.inf)
            a.semilogy(time[start:], E, color=col, ls=ls, lw=1.5,
                       label=f"{lbl} (err {err:.2f})" if np.isfinite(err)
                       else f"{lbl} (diverged)")
            line += (f"{err:>24.3f}" if np.isfinite(err)
                     else f"{'diverged':>24}")
        print(line)
        a.set_title(rf"$\beta = {b:g}$", fontsize=10)
        a.grid(alpha=0.3, which="both")
        if j == 0:
            a.legend(fontsize=6.5, loc="lower left")
    ax.flat[-1].axis("off")
    for a in ax[-1]:
        a.set_xlabel("$t$")
    for a in ax[:, 0]:
        a.set_ylabel(r"$\|x(t)\|^2$")
    fig.suptitle(f"Cavity, r = {r}: balanced OpInf fitted with and without the "
                 f"early transient (t < {args.t_min:g})", fontsize=11)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures",
                       f"energy_trimmed_r{r}_tmin{args.t_min:g}.png")
    fig.savefig(out, dpi=140)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
