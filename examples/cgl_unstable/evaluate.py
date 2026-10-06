"""
Does a cubic OpInf ROM of the unstable CGL keep the limit cycle, and the
off-attractor dynamics?

  1. Off-attractor: ||x(t) - m|| of the FOM and of each ROM (decoded,
     Phi (Psi^T Phi)^-1 z) for every training and testing impulse, plus the
     relative state error over the window.
  2. Limit cycle: every ROM is integrated far past the training horizon
     (to T_LONG) from the training initial conditions; on the last stretch the
     asymptotic amplitude ||x||, its fluctuation and the rotation frequency
     are compared with the FOM cycle (data/limit_cycle.npz), and the cycle's
     |q(x)| profile is plotted against the FOM's.

Usage: python evaluate.py [--r 5] [--derivs reproj|fd]
Writes figures/norm_vs_time_r<r>_<derivs>.png and figures/limit_cycle_r<r>_<derivs>.png.
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
from cgl import CGL  # noqa: E402

from train_models import load  # noqa: E402

DATA = os.path.join(HERE, "data")
T_LONG = 4000.0


def decoder(d):
    return d["Phi"] @ np.linalg.inv(d["Psi"].T @ d["Phi"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--derivs", choices=["reproj", "fd"], default="reproj")
    args = ap.parse_args()
    tag = f"r{args.r}_{args.derivs}"
    roms = np.load(os.path.join(HERE, f"roms_{tag}.npy"), allow_pickle=True).item()
    lc = np.load(os.path.join(DATA, "limit_cycle.npz"))
    cgl = CGL(mu0=float(lc["mu0"]))
    nx = cgl.nx
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

    sets = {"train": load("train"), "test": load("test")}
    models = {name: (ROM(d["Ar"], d["Hr"], 3), d["Psi"], decoder(d))
              for name, d in roms.items()}
    colors = dict(zip(models, ["C0", "C1", "C2"]))

    # ---- 1. off-attractor: norm vs time, state errors ----------------------
    n_tr, n_te = len(sets["train"][0]), len(sets["test"][0])
    ncol = 4
    nrow = -(-(n_tr + n_te) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.7*nrow),
                             constrained_layout=True, squeeze=False)
    k = 0
    print(f"relative state error ||X - X_rom|| / ||X|| over t in [0, "
          f"{sets['train'][3][-1]:g}]")
    print(f"{'':48s}" + "".join(f"{n:>16s}" for n in models))
    for kind, (trajs, alphas, labels, time) in sets.items():
        for X, lab in zip(trajs, labels):
            ax = axes.flat[k]
            ax.semilogy(time, np.linalg.norm(X, axis=0), "k", lw=2, label="FOM")
            row = []
            for name, (rom, Psi, D) in models.items():
                Z = rom.integrate(Psi.T @ X[:, 0], time)
                Xh = D @ Z
                ok = np.all(np.isfinite(Xh))
                row.append(np.linalg.norm(X - Xh)/np.linalg.norm(X) if ok else np.inf)
                ax.semilogy(time, np.linalg.norm(Xh, axis=0), color=colors[name],
                            lw=1.3, label=name)
            ax.axhline(float(lc["amp_mean"]), color="gray", ls=":", lw=1)
            ax.set_title(f"{kind}: {lab}", fontsize=8)
            ax.set_xlim(time[0], time[-1])
            print(f"{(kind + ': ' + str(lab))[:48]:48s}"
                  + "".join(f"{e:>16.3f}" for e in row))
            k += 1
    for ax in axes.flat[k:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\|x - m\|$")
    for ax in axes[-1]:
        ax.set_xlabel("Time $t$")
    fig.suptitle(f"Unstable CGL ($\\mu_0$ = {float(lc['mu0'])}): FOM vs r = {args.r} "
                 "cubic OpInf (dotted: FOM limit-cycle amplitude)")
    fig.savefig(os.path.join(HERE, "figures", f"norm_vs_time_{tag}.png"), dpi=130)

    # ---- 2. long-time behaviour: does the ROM settle on a cycle? ------------
    trajs, _, labels, time = sets["train"]
    dts = time[1] - time[0]
    t_long = dts*np.arange(int(round(T_LONG/dts)) + 1)
    late = t_long >= T_LONG - 300.0
    print(f"\nFOM limit cycle: ||x|| = {float(lc['amp_mean']):.4f}, omega = "
          f"{float(lc['omega']):.4f}")
    print(f"ROMs integrated to t = {T_LONG:g} from every training IC; stats over the "
          f"last 300 time units")
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    ax2[0].plot(cgl.x, lc["profile"], "k", lw=2.2, label="FOM cycle")
    for name, (rom, Psi, D) in models.items():
        amps, oms, flucts = [], [], []
        prof = None
        for X in trajs:
            Z = rom.integrate(Psi.T @ X[:, 0], t_long)
            if not np.all(np.isfinite(Z)):
                amps.append(np.inf)
                continue
            Xl = D @ Z[:, late]
            nrm = np.linalg.norm(Xl, axis=0)
            amps.append(nrm.mean())
            flucts.append((nrm.max() - nrm.min())/max(nrm.mean(), 1e-300))
            mod = np.hypot(Xl[:nx], Xl[nx:])
            i = int(np.argmax(mod[:, -1]))
            if nrm.mean() > 1e-6:
                ph = np.unwrap(np.angle(Xl[i] + 1j*Xl[nx + i]))
                oms.append(np.polyfit(t_long[late], ph, 1)[0])
            if prof is None:
                prof = mod[:, -1]
        amps = np.array(amps)
        print(f"   {name:<14}: asymptotic ||x|| per IC "
              f"{np.array2string(amps, precision=3)}; omega "
              f"{np.array2string(np.array(oms), precision=4)}; max fluctuation "
              f"{max(flucts) if flucts else np.nan:.1e}")
        if prof is not None:
            ax2[0].plot(cgl.x, prof, color=colors[name], lw=1.3, label=name)
        Z = rom.integrate(Psi.T @ trajs[0][:, 0], t_long)
        ax2[1].semilogy(t_long, np.linalg.norm(D @ Z, axis=0), color=colors[name],
                        lw=1.2, label=name)
    ax2[1].axhline(float(lc["amp_mean"]), color="k", ls=":", lw=1.2,
                   label="FOM cycle amplitude")
    ax2[1].axvline(time[-1], color="gray", ls="--", lw=0.8)
    ax2[0].set_xlabel("$x$"); ax2[0].set_ylabel(r"$|q(x)|$ on the cycle")
    ax2[0].set_xlim(-40, 40); ax2[0].legend(fontsize=8)
    ax2[1].set_xlabel("Time $t$ (dashed: end of training window)")
    ax2[1].set_ylabel(r"$\|x - m\|$"); ax2[1].legend(fontsize=8)
    ax2[1].set_title(f"{labels[0]}, integrated to t = {T_LONG:g}", fontsize=9)
    fig2.savefig(os.path.join(HERE, "figures", f"limit_cycle_{tag}.png"), dpi=130)
    print(f"saved -> figures/norm_vs_time_{tag}.png, figures/limit_cycle_{tag}.png")


if __name__ == "__main__":
    main()
