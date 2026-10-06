"""
Sensor output y(t) of the FOM and the ROMs, for a few training, testing and
shifted-support impulses.

y = Re q at a Gaussian sensor at branch II (the output of examples/cgl,
eq. (4.2)), applied to x - m.  Each case gets two panels: the transient
(t <= T_EARLY) and the last stretch of the window (t >= T - T_LATE), where
the trajectories sit on the limit cycle and a phase error shows up as a
shift of the oscillation.

Usage: python plot_outputs.py [--r 5] [--derivs reproj|fd]
Writes figures/outputs_r<r>_<derivs>.png.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "cgl"))  # after this dir: local names win
from balanced_opinf import ROM  # noqa: E402
from cgl import CGL, TimeStepper  # noqa: E402

from generate_data import DT, MU0, NSAVE, T  # noqa: E402
from test_shifted_support import gaussian  # noqa: E402
from train_models import load  # noqa: E402

DATA = os.path.join(HERE, "data")
T_EARLY, T_LATE = 150.0, 60.0
SHIFTED = [(-4.0, 1.0), (4.0, 1.0)]       # (delta, beta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--derivs", choices=["reproj", "fd"], default="reproj")
    args = ap.parse_args()
    tag = f"r{args.r}_{args.derivs}"
    roms = np.load(os.path.join(HERE, f"roms_{tag}.npy"), allow_pickle=True).item()
    m = np.load(os.path.join(DATA, "setup.npz"))["m"]
    c = CGL(mu0=MU0)
    sensor = np.exp(-((c.x + c.xbar)/c.s)**2)       # branch II
    C = np.concatenate((sensor, np.zeros(c.nx)))

    tr, _, tr_lab, time = load("train")
    te, _, te_lab, _ = load("test")
    cases = [(tr[j], tr_lab[j]) for j in (2, 4, 6, 0)]  # beta = 0.01, 0.1, 1, -1 (Re)
    cases += [(te[j], te_lab[j]) for j in range(len(te))]
    for d, b in SHIFTED:
        Q = TimeStepper(c, DT).integrate(b*gaussian(c, d), T, NSAVE)[1]
        cases.append((Q - m[:, None], f"shifted $\\delta$ = {d:+g}, $\\beta$ = {b:g}"))

    models = {n: (ROM(v["Ar"], v["Hr"], 3), v["Psi"],
                  v["Phi"] @ np.linalg.inv(v["Psi"].T @ v["Phi"]))
              for n, v in roms.items()}
    early, late = time <= T_EARLY, time >= time[-1] - T_LATE
    fig, axes = plt.subplots(len(cases), 2, figsize=(14, 1.9*len(cases)),
                             constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[2.5, 1]))
    for (X, lab), (a0, a1) in zip(cases, axes):
        y = C @ X
        for ax, w in ((a0, early), (a1, late)):
            ax.plot(time[w], y[w], "k", lw=2, label="FOM")
        for k, (name, (rom, Psi, D)) in enumerate(models.items()):
            yh = C @ (D @ rom.integrate(Psi.T @ X[:, 0], time))
            for ax, w in ((a0, early), (a1, late)):
                ax.plot(time[w], yh[w], color=f"C{k}", lw=1.2, label=name)
        a0.set_ylabel("$y$")
        a0.set_title(lab, fontsize=9, loc="left")
        a0.set_xlim(0, T_EARLY)
        a1.set_xlim(time[-1] - T_LATE, time[-1])
        a1.set_ylim(-1.4*np.abs(y[late]).max(), 1.4*np.abs(y[late]).max())
    axes[0, 0].legend(fontsize=7, ncol=3)
    axes[-1, 0].set_xlabel("Time $t$ (transient)")
    axes[-1, 1].set_xlabel("Time $t$ (end of window, on the cycle)")
    fig.suptitle(f"Sensor output at branch II, r = {args.r} cubic OpInf "
                 f"({'re-projected: Petrov-Galerkin' if args.derivs == 'reproj' else 'finite differences'})")
    out = os.path.join(HERE, "figures", f"outputs_{tag}.png")
    fig.savefig(out, dpi=110)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
