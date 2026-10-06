"""Perturbation energy of the 3D wing training trajectories, unscaled and
scaled by each trajectory's time-averaged energy.

E_k(t) = <q'_k, q'_k>_W, W = V/V_min over the full domain, as written by
generate_trajectories.py (energy_KKK.npy, computed in float64 at run time).
The scaling is the one the airfoil ROMs are trained with
(examples/airfoil/compute_preprojection.py): Ebar_k = mean_t E_k(t), the
per-trajectory weight, so E_k/Ebar_k shows the trajectories on an equal
footing.  Plots whatever trajectories exist so far.

Usage (incompreso venv):
    python plot_training_energy.py [--data fom_data_wing_re1000_train]
Writes figures_wing/training_energy_<data>.png.
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIONS = ("leading edge", "midchord", "trailing edge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data_wing_re1000_train"))
    args = ap.parse_args()

    meta = np.load(os.path.join(args.data, "meta.npz"))
    t, params = meta["time"], meta["parameters"]
    files = sorted(glob.glob(os.path.join(args.data, "energy_[0-9][0-9][0-9].npy")))
    if not files:
        raise SystemExit(f"no trajectories in {args.data} yet")
    n_beta = len(np.unique(params[:, 0]))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    print(f"{'traj':>4}{'station':>15}{'beta':>6}{'E(0)':>11}{'Ebar':>11}"
          f"{'max E (t>1)':>13}{'t_max':>7}{'E(end)':>11}")
    for f in files:
        k = int(os.path.basename(f)[7:10])
        E = np.load(f)
        beta = params[k, 0]
        station = STATIONS[k // n_beta]
        Ebar = E.mean()
        late = t > 1.0
        i = np.flatnonzero(late)[E[late].argmax()]
        print(f"{k:>4}{station:>15}{beta:>6g}{E[0]:>11.3e}{Ebar:>11.3e}"
              f"{E[i]:>13.3e}{t[i]:>7.1f}{E[-1]:>11.3e}")
        label = f"{k}: {station}, $\\beta$ = {beta:g}"
        axs[0].semilogy(t, E, label=label)
        axs[1].semilogy(t, E/Ebar, label=label)
    axs[0].set_ylabel(r"$E_k = \langle q'_k, q'_k \rangle_W$")
    axs[1].set_ylabel(r"$E_k / \bar E_k$,   $\bar E_k = \mathrm{mean}_t\, E_k$")
    axs[0].set_title("unscaled", fontsize=10)
    axs[1].set_title("scaled by the time-averaged energy", fontsize=10)
    for ax in axs:
        ax.set_xlabel("t")
        ax.set_xlim(t[0], t[-1])
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(r"Wing, Re = 1000: training-trajectory energy, W = V/V$_{min}$")
    out = os.path.join(HERE, "figures_wing",
                       f"training_energy_{os.path.basename(os.path.normpath(args.data))}.png")
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
