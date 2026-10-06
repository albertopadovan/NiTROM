"""Perturbation energy E(t) = <q', q'>_W (W = V/V_min) of the wing's
leading-edge, beta = 1 trajectories at several Reynolds numbers, read from
the energy_KKK.npy files generate_trajectories.py writes next to each
trajectory.  Log scale on the left, linear on the right.

Usage (incompreso venv):
    python plot_energy_compare.py [--traj 2]
Writes figures_wing/energy_vs_Re_KKK.png.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = (("Re = 600", "fom_data_wing"),
        ("Re = 1000", "fom_data_wing_re1000"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=int, default=2)
    args = ap.parse_args()

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for label, d in RUNS:
        if not os.path.exists(os.path.join(HERE, d, f"energy_{args.traj:03d}.npy")):
            print(f"{label}: {d}/ not found, skipped")
            continue
        t = np.load(os.path.join(HERE, d, "meta.npz"))["time"]
        E = np.load(os.path.join(HERE, d, f"energy_{args.traj:03d}.npy"))
        k = E.argmax() if E.argmax() > 0 else E[5:].argmax() + 5
        lab = f"{label} (peak {E[k]:.3g} at t = {t[k]:g})"
        axs[0].semilogy(t, E, label=lab)
        axs[1].plot(t, E, label=lab)
        print(f"{label}: min {E.min():.3e} at t = {t[E.argmin()]:g}, "
              f"wake peak {E[k]:.3e} at t = {t[k]:g}, E(end) = {E[-1]:.3e}")
    for ax in axs:
        ax.axhline(1.0, color="0.5", lw=0.8, ls="--")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$E = \langle q', q' \rangle_W$")
        ax.grid(alpha=0.3, which="both")
        ax.set_xlim(0, t[-1])
    axs[1].set_ylim(0, None)
    axs[0].legend(fontsize=9)
    fig.suptitle(r"Wing, leading-edge impulse, $\beta = 1$ (z-symmetric): "
                 r"perturbation energy, W = V/V$_{min}$  (dashed: E(0))")
    out = os.path.join(HERE, "figures_wing", f"energy_vs_Re_{args.traj:03d}.png")
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
