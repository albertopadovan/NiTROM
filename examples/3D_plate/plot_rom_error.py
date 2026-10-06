"""Forecast error vs time of the 3D wing ROMs, per training trajectory.

    e_k(t) = ||q~_k(t) - Phi z(t)|| / sqrt(Ebar_k)

(the trajectory's RMS norm -- the same per-trajectory normalization as the
OpInf cost; a pointwise ||q~ - Phi z|| / ||q~|| blows up once the packet has
left).  Dotted: the oblique projection error ||q~ - Phi Psi^T q~|| / sqrt(Ebar_k)
of each basis, the floor of any ROM on it.  Exact in the lossless coordinates.

Usage (python with nitrom on the path):
    python plot_rom_error.py [--r 50] [--models opinf_balanced_linear opinf_balanced opinf_pod]
Writes figures_wing/rom_error_r<r>.png.
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
    ap.add_argument("--r", type=int, default=50)
    ap.add_argument("--models", nargs="+",
                    default=["opinf_balanced_linear", "opinf_balanced", "opinf_pod"])
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data_wing_re1000_train"))
    ap.add_argument("--model-dir", default=os.path.join(HERE, "models_wing_re1000"))
    args = ap.parse_args()

    c = np.load(os.path.join(args.data, "compressed.npz"), allow_pickle=True)
    X, time, params, Ebar = c["X"], c["time"], c["parameters"], c["energy"]
    n_traj = len(X)
    n_beta = len(np.unique(params[:, 0]))
    ncol = min(n_traj, 3)
    nrow = -(-n_traj//ncol)
    fig, axg = plt.subplots(nrow, ncol, figsize=(5.2*ncol, 3.9*nrow), squeeze=False,
                            constrained_layout=True)
    axs = axg.ravel()
    for ax in axs[n_traj:]:
        ax.set_visible(False)

    floors_done = set()
    colors = {"balanced": "C3", "pod": "C0"}
    styles = ["-", "--", "-."]
    for im, name in enumerate(args.models):
        with open(os.path.join(args.model_dir, f"{name}_r{args.r}.pkl"), "rb") as f:
            m = pickle.load(f)
        A = m["tensors"][0]
        r = A.shape[0]
        H = m["tensors"][1] if len(m["tensors"]) > 1 else np.zeros((r, r*r))
        Phi, Psi, basis = m["Phi"], m["Psi"], m["basis"]
        lin = len(m["tensors"]) == 1
        label = f"{basis} {'linear' if lin else 'quadratic'}"
        ls = styles[sum(1 for n in args.models[:im] if basis in n) % len(styles)]
        for k in range(n_traj):
            Z = forecast(A, H, Psi.T @ X[k][:, 0], time)
            e = np.linalg.norm(X[k] - Phi @ Z, axis=0)/np.sqrt(Ebar[k])
            axs[k].semilogy(time, e, color=colors[basis], ls=ls, label=label)
            if basis not in floors_done:
                ep = np.linalg.norm(X[k] - Phi @ (Psi.T @ X[k]), axis=0)/np.sqrt(Ebar[k])
                axs[k].semilogy(time, ep, color=colors[basis], ls=":", lw=1,
                                label=f"{basis} projection")
        floors_done.add(basis)
    for k, ax in enumerate(axs[:n_traj]):
        beta = params[k, 0]
        ax.semilogy(time, np.linalg.norm(X[k], axis=0)/np.sqrt(Ebar[k]), "k", lw=1.5,
                    label=r"FOM $\|\tilde q\|$")
        ax.set_title(f"traj {k}: {STATIONS[k // n_beta]}, $\\beta$ = {beta:g}", fontsize=10)
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\|\tilde q - \Phi z\| / \sqrt{\bar E_k}$")
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(1e-4, 5)
        ax.grid(alpha=0.3, which="both")
    axs[0].legend(fontsize=7)
    fig.suptitle(f"Wing, Re = 1000: forecast error vs time (r = {args.r}); black: FOM norm, "
                 f"dotted: projection floor")
    out = os.path.join(HERE, "figures_wing", f"rom_error_r{args.r}.png")
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
