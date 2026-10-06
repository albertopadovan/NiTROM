"""
Generalization in the *location* of the impulse.

Every training impulse is the same Gaussian, centred at branch I
(x = xbar ~ -8.25, width s = 1.6).  Here the Gaussian is moved by delta along
x -- same width, unit-norm, amplitudes beta inside the training range -- and
the r = 5 ROMs are compared with the FOM: relative state error over
[0, T] and the asymptotic amplitude (does the ROM still land on the cycle?).
The overlap <g_delta, g_train> of the shifted and the training profile says
how far "outside the training support" each case is.

Usage: python test_shifted_support.py [--r 5] [--derivs reproj|fd]
Writes figures/shifted_support_r<r>_<derivs>.png.
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "cgl"))  # after this dir: local names win
from balanced_opinf import ROM  # noqa: E402
from cgl import CGL, TimeStepper  # noqa: E402

from generate_data import DT, MU0, NSAVE, T  # noqa: E402

DATA = os.path.join(HERE, "data")
DELTAS = [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0]
BETAS = [0.3, 1.0]


def gaussian(c, delta):
    g = np.exp(-((c.x - (c.xbar + delta))/c.s)**2)
    v = np.concatenate((g, np.zeros(c.nx)))
    return v/np.linalg.norm(v)


def run(args):
    delta, beta = args
    c = CGL(mu0=MU0)
    return TimeStepper(c, DT).integrate(beta*gaussian(c, delta), T, NSAVE)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--derivs", choices=["reproj", "fd"], default="reproj")
    args = ap.parse_args()
    tag = f"r{args.r}_{args.derivs}"
    roms = np.load(os.path.join(HERE, f"roms_{tag}.npy"), allow_pickle=True).item()
    m = np.load(os.path.join(DATA, "setup.npz"))["m"]
    amp_lc = float(np.load(os.path.join(DATA, "limit_cycle.npz"))["amp_mean"])
    time = np.load(os.path.join(DATA, "time.npy"))
    c = CGL(mu0=MU0)
    g0 = gaussian(c, 0.0)

    cases = [(d, b) for d in DELTAS for b in BETAS]
    with Pool(len(cases)) as pool:
        Qs = pool.map(run, cases)

    models = {n: (ROM(d["Ar"], d["Hr"], 3), d["Psi"],
                  d["Phi"] @ np.linalg.inv(d["Psi"].T @ d["Phi"]))
              for n, d in roms.items()}
    print(f"{'delta':>6}{'beta':>6}{'overlap':>9}" + "".join(
        f"{n + ' err':>18}{'final ||x||':>13}" for n in models)
        + f"{'FOM final':>11}")
    fig, axes = plt.subplots(len(BETAS), len(DELTAS), figsize=(18, 5.5),
                             constrained_layout=True, squeeze=False)
    for (d, b), Q in zip(cases, Qs):
        X = Q - m[:, None]
        row = []
        ax = axes[BETAS.index(b), DELTAS.index(d)]
        ax.semilogy(time, np.linalg.norm(X, axis=0), "k", lw=2, label="FOM")
        for k, (name, (rom, Psi, D)) in enumerate(models.items()):
            Z = rom.integrate(Psi.T @ X[:, 0], time)
            Xh = D @ Z
            if np.all(np.isfinite(Xh)):
                err = np.linalg.norm(X - Xh)/np.linalg.norm(X)
                fin = np.linalg.norm(Xh[:, -1])
                ax.semilogy(time, np.linalg.norm(Xh, axis=0), color=f"C{k}",
                            lw=1.3, label=name)
            else:
                err, fin = np.inf, np.inf
            row.append(f"{err:>18.3f}{fin:>13.3f}")
        ax.axhline(amp_lc, color="gray", ls=":", lw=1)
        ax.set_title(f"$\\delta$ = {d:+g}, $\\beta$ = {b:g} "
                     f"(overlap {g0 @ gaussian(c, d):.2f})", fontsize=9)
        print(f"{d:>+6g}{b:>6g}{g0 @ gaussian(c, d):>9.2f}" + "".join(row)
              + f"{np.linalg.norm(X[:, -1]):>11.3f}")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Impulse moved by $\\delta$ from branch I (training location); "
                 "dotted: FOM cycle amplitude")
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", f"shifted_support_{tag}.png")
    fig.savefig(out, dpi=120)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
