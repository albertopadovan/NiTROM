"""Average training error vs time of balanced ROMs -- the paper's eq. (40).

    e(t) = (1/N_traj) sum_j || q~_j(t) - q^_j(t) ||^2 / e_j,avg,
    e_j,avg = time average of || q~_j ||^2,

with q^_j the ROM forecast from z(0) = Psi^T q~_j(0), decoded with
Phi (Psi^T Phi)^{-1}.  The "zero" curve is the trivial model q^ = 0, i.e. the
instantaneous energy normalized by its time average: a model above it does
worse than predicting nothing.  Everything is on the mesh the balancing was
built on, in the weighted state q~ = W^{1/2} q.

Usage: python plot_rom_error.py models/a.pkl [models/b.pkl ...]
                                [--balancing balancing_full] [--out NAME]
Writes figures/<NAME>.png (default: airfoil_rom_error).
"""

from __future__ import annotations

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_gas_opinf_balanced import forecast

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="checkpoint .pkl files")
    ap.add_argument("--balancing", default=os.path.join(HERE, "balancing_full"))
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data"))
    ap.add_argument("--out", default="airfoil_rom_error")
    args = ap.parse_args()

    bal = np.load(os.path.join(args.balancing, "balancing.npz"), allow_pickle=True)
    red = np.load(os.path.join(args.balancing,
                               f"reduced_r{bal['Phi'].shape[1]}.npz"))
    t, sq, e_avg = red["time"], red["sq"], red["energy"]
    n_traj = len(e_avg)
    mask, sqrt_w = bal["mask"], bal["sqrt_w"]
    X = [np.load(os.path.join(args.data, f"fluct_{k:03d}.npy"),
                 mmap_mode="r")[mask]*sqrt_w[:, None] for k in range(n_traj)]
    print(f"loaded {n_traj} trajectories, N = {X[0].shape[0]}")

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    zero = (sq/e_avg[:, None]).mean(0)
    ax.plot(t, zero, "k:", lw=1.5, label="Zero")
    print(f"{'model':<52}{'time-avg e(t)':>14}")
    print(f"{'zero':<52}{zero.mean():>14.4f}")
    for m, path in enumerate(args.models):
        with open(path, "rb") as f:
            ck = pickle.load(f)
        A, H = [np.asarray(x) for x in ck["tensors"]]
        Phi, Psi = np.asarray(ck["Phi"]), np.asarray(ck["Psi"])
        D = np.linalg.inv(Psi.T @ Phi)
        PhiD = Phi @ D                                   # decoder
        G = PhiD.T @ PhiD
        e = np.zeros_like(t)
        for k in range(n_traj):
            Z = forecast(A, H, Psi.T @ np.ascontiguousarray(X[k][:, 0]), t)
            if not np.all(np.isfinite(Z)):
                e[:] = np.inf
                break
            # ||q - PhiD z||^2 = ||q||^2 - 2 z^T (PhiD^T q) + z^T G z
            e2 = sq[k] - 2*(Z*(PhiD.T @ X[k])).sum(0) + (Z*(G @ Z)).sum(0)
            e += np.maximum(e2, 0.0)/e_avg[k]
        e /= n_traj
        label = os.path.splitext(os.path.basename(path))[0]
        ax.plot(t, e, color=f"C{m}", lw=1.6, label=label)
        print(f"{label:<52}{e.mean():>14.4f}")

    ax.set_yscale("log")
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("average error $e(t)$, eq. (40)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", f"{args.out}.png")
    fig.savefig(out, dpi=170)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
