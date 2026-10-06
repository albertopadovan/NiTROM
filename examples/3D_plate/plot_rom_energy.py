"""Perturbation energy of the 3D wing ROMs against the FOM, per training
trajectory.

Each model (train_opinf.py checkpoint) is forecast from z(0) = Psi^T x(0) and
its energy is ||Phi z(t)||^2 -- exactly the W-energy of the reconstructed
state, the lossless coordinates being orthonormal -- next to the FOM's
||x(t)||^2 = <q', q'>_W.

Usage (python with nitrom on the path):
    python plot_rom_energy.py [--kind opinf] [--r 50] [--bases pod balanced]
Writes figures_wing/rom_energy_<kind>_r<r>.png.
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from train_opinf import forecast  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIONS = ("leading edge", "midchord", "trailing edge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="opinf", help="opinf or gas_opinf")
    ap.add_argument("--linear", action="store_true", help="the linear models (tag _linear)")
    ap.add_argument("--r", type=int, default=50)
    ap.add_argument("--bases", nargs="+", default=["pod", "balanced"])
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data_wing_re1000_train"))
    ap.add_argument("--models", default=os.path.join(HERE, "models_wing_re1000"))
    args = ap.parse_args()

    c = np.load(os.path.join(args.data, "compressed.npz"), allow_pickle=True)
    X, time, params = c["X"], c["time"], c["parameters"]
    n_traj = len(X)
    ncol = min(n_traj, 3)                        # station x beta grid
    nrow = -(-n_traj//ncol)
    fig, axg = plt.subplots(nrow, ncol, figsize=(5.2*ncol, 3.9*nrow), squeeze=False,
                            constrained_layout=True)
    axs = axg.ravel()
    for ax in axs[n_traj:]:
        ax.set_visible(False)
    for k in range(n_traj):
        axs[k].semilogy(time, (X[k]**2).sum(0), "k", lw=2, label="FOM")
    for basis, color in zip(args.bases, ("C0", "C3", "C2")):
        lin = "_linear" if args.linear else ""
        path = os.path.join(args.models, f"{args.kind}_{basis}{lin}_r{args.r}.pkl")
        with open(path, "rb") as f:
            m = pickle.load(f)
        A = m["tensors"][0]
        r_m = A.shape[0]
        H = m["tensors"][1] if len(m["tensors"]) > 1 else np.zeros((r_m, r_m*r_m))
        Phi, Psi = m["Phi"], m["Psi"]
        for k in range(n_traj):
            Z = forecast(A, H, Psi.T @ X[k][:, 0], time)
            Erom = ((Phi @ Z)**2).sum(0)
            err = np.sqrt(np.nansum(((X[k] - Phi @ Z)**2).sum(0))/(X[k]**2).sum())
            axs[k].semilogy(time, Erom, color=color, ls="--",
                            label=f"{args.kind}{' linear' if args.linear else ''} {basis} "
                                  f"r={args.r} (rel. err {err:.2f})")
            print(f"{basis:>9} traj {k}: max E_rom {np.nanmax(Erom):.3e} (FOM "
                  f"{(X[k]**2).sum(0).max():.3e}), time-integrated rel. error {err:.3f}")
    for k, ax in enumerate(axs[:n_traj]):
        beta, x0, y0 = params[k]
        n_beta = len(np.unique(params[:, 0]))
        ax.set_title(f"traj {k}: {STATIONS[k // n_beta] if n_beta else ''}, "
                     f"$\\beta$ = {beta:g}", fontsize=10)
        ax.set_xlabel("t")
        ax.set_ylabel(r"$E = \langle q', q' \rangle_W$")
        ax.set_xlim(time[0], time[-1])
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.suptitle(f"Wing, Re = 1000: ROM vs FOM perturbation energy ({args.kind}, r = {args.r})")
    out = os.path.join(HERE, "figures_wing",
                       f"rom_energy_{args.kind}{'_linear' if args.linear else ''}_r{args.r}.png")
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
