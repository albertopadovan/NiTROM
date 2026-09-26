"""
Average testing error over time -- Figure 4(a) of Padovan, Vollmer & Bodony,
SIADS 2024, reproduced for the models of train_models.py / train_nitrom.py.

The paper tests on 50 impulse responses B u(t) = beta B e_j delta(t) of eq.
(4.5), with beta drawn uniformly from [-1, 1] (generate_data.py writes
data/ens_*), and reports

    e(t) = (1/N_traj) sum_j (1/alpha_j) || y^(j)(t) - yhat^(j)(t) ||^2,   (3.9)

with alpha_j as in the cost (3.7) -- for the CGL, the time-averaged output
energy of the jth trajectory -- so trajectories of very different amplitude
contribute comparably.  yhat = C Phi (Psi^T Phi)^-1 z with z solving the latent
ODE from z(0) = Psi^T x(0).

TrOOP and the POD-Galerkin model of the paper are not reproduced here; the
curves are the OpInf ROMs (balanced and POD) and whatever NiTROM refinements
exist on disk.

Writes figures/avg_testing_error_r5.png.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cgl import CGL
from balanced_opinf import ROM
from plot_results import decoder, load_roms, NITROM_GLOB

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
R = 5
T_PLOT = 200.0          # the paper's Figure 4(a) spans [0, 500]; 200 covers the
                        # transient and matches the window plot_results.py uses


def output_operator():
    """y = C q: Gaussian sensor at branch II, eq. (4.2), in Re and Im."""
    cgl = CGL()
    g = np.exp(-((cgl.x + cgl.xbar)/cgl.s)**2)
    C = np.zeros((2, 2*cgl.nx))
    C[0, :cgl.nx], C[1, cgl.nx:] = g, g
    return C


def main():
    C = output_operator()
    time = np.load(os.path.join(DATA, "time.npy"))
    nt = int(round(T_PLOT/(time[1] - time[0]))) + 1
    t = time[:nt]

    betas = np.load(os.path.join(DATA, "ens_beta.npy"))
    # Physical trajectories: one (A_r, H_r) serves every amplitude, no c factor.
    X = [np.load(os.path.join(DATA, f"ens_traj_{j:03d}.npy"))[:, :nt]
         for j in range(len(betas))]
    Y = [C @ Xj for Xj in X]
    alpha = np.array([np.mean(np.sum(Yj**2, axis=0)) for Yj in Y])

    roms, styles = load_roms(os.path.join(HERE, f"roms_r{R}.npy"),
                             os.path.join(HERE, NITROM_GLOB.format(r=R)))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    print(f"Average testing error, eq. (3.9), {len(betas)} impulse responses, "
          f"beta ~ U[-1, 1], r = {R}")
    for name, (col, ls) in styles.items():
        d = roms[name]
        P = decoder(d)
        rom = ROM(d["Ar"], d["Hr"], 3)
        e = np.zeros(nt)
        n_blow = 0
        for Xj, Yj, aj in zip(X, Y, alpha):
            yh = C @ (P @ rom.integrate(d["Psi"].T @ Xj[:, 0], t, 1.0))
            bad = ~np.all(np.isfinite(yh), axis=0)
            if bad.any():
                n_blow += 1
                # A diverged trajectory contributes its own energy from the
                # point of divergence: the model predicts nothing there, and
                # dropping it would flatter the model that blew up.
                yh[:, bad] = 0.0
            e += np.sum((Yj - yh)**2, axis=0)/aj
        e /= len(betas)
        note = f"  [{n_blow} diverged]" if n_blow else ""
        ax.semilogy(t, e, color=col, ls=ls, lw=1.6, label=f"{name}{note}")
        print(f"   {name:<26} e(T) = {e[-1]:.3e}, "
              f"time-mean e = {np.mean(e):.3e}{note}")

    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$e(t)$")
    ax.set_xlim(0, T_PLOT)
    ax.set_title(f"Average error over {len(betas)} testing impulse responses, "
                 rf"$\beta \sim U[-1, 1]$, $r = {R}$", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out = os.path.join(HERE, "figures", f"avg_testing_error_r{R}.png")
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
