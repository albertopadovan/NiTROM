"""Singular values of a training trajectory in the weighted inner product.

POD by the method of snapshots on qtilde = W^{1/2} q', W = V/V_min (the same
weighting as the energy, make_initial_conditions.cell_volume_weights), so
the singular values are those of the W-orthonormal POD: with
Qt = [qtilde(t_0), ..., qtilde(t_{m-1})] (n x m),

    K = Qt^T Qt  (m x m),   sigma_i = sqrt(lambda_i(K)).

K is accumulated on the GPU over column chunks of the time-major memmap
(m x n float32), so the 32 GB trajectory is never held in memory.

Usage (incompreso venv):
    python pod_trajectory.py [--config config_traj.yaml] [--traj 2]
Writes <figures>/pod_sv_KKK.png and <fom_data>/pod_sv_KKK.npy.
"""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from incompreso.backend import to_numpy, xp  # noqa: E402

from case import DEFAULT_CONFIG, Case  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK = 4_000_000                       # DOFs per chunk (1.6 GB of float32 for 100 snapshots)


def cell_volume_weights_from_snapshot(path):
    """V/V_min of every u, v, w DOF from a base-flow snapshot's grids
    (identical to make_initial_conditions.cell_volume_weights, without
    parsing a setup)."""
    vols = []
    with np.load(path) as s:
        for c in "uvw":
            x, y, z = (s[f"{a}{c}"][1:-1] for a in "xyz")
            vols.append((np.gradient(z)[:, None, None] * np.gradient(y)[None, :, None]
                         * np.gradient(x)[None, None, :]).ravel())
    V = np.concatenate(vols)
    return V/V.min()


def gram_time_major(Q, w):
    """K = Qt Qt^T for time-major Q (m, n), Qt = Q W^{1/2}, on the device."""
    m, n = Q.shape
    K = xp.zeros((m, m))
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        X = xp.asarray(np.ascontiguousarray(Q[:, a:b]), dtype=xp.float64)
        X *= xp.asarray(np.sqrt(w[a:b]))[None, :]
        K += X @ X.T
    return to_numpy(K)


def singular_values(K):
    lam = np.linalg.eigvalsh(K)[::-1]
    return np.sqrt(np.clip(lam, 0.0, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data", default=None, help="trajectory directory (default: the case's)")
    ap.add_argument("--traj", type=int, default=2)
    args = ap.parse_args()
    C = Case(args.config)
    data = args.data or C.fom_data

    meta = np.load(os.path.join(data, "meta.npz"))
    t3 = meta["time"]
    Q = np.load(os.path.join(data, f"fluct_{args.traj:03d}.npy"), mmap_mode="r")
    w = cell_volume_weights_from_snapshot(str(meta["baseflow"]))
    if w.size != Q.shape[1]:
        raise SystemExit("weights do not match the trajectory")
    t0 = time.time()
    K = gram_time_major(Q, w)
    s3 = singular_values(K)
    print(f"3D trajectory {args.traj}: {Q.shape[0]} snapshots x {Q.shape[1]} DOFs, "
          f"Gram matrix in {time.time() - t0:.0f} s; trace check: "
          f"sum sigma^2 = {np.sum(s3**2):.4e} = sum_k E(t_k) = "
          f"{np.load(os.path.join(data, f'energy_{args.traj:03d}.npy')).sum():.4e}")
    np.save(os.path.join(data, f"pod_sv_{args.traj:03d}.npy"), s3)

    curves = [(f"3D wing (traj {args.traj}), t <= {t3[-1]:g}", s3)]

    print(f"\n{'r':>4}" + "".join(f"{lab.split(',')[0]:>22}" for lab, _ in curves)
          + "   (sigma_r/sigma_1;  energy captured by r modes)")
    for r in (1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 60):
        row = f"{r:>4}"
        for _, s in curves:
            if r <= s.size:
                row += f"   {s[r - 1]/s[0]:8.2e} {np.sum(s[:r]**2)/np.sum(s**2):8.5f}"
        print(row)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for (lab, s), mk in zip(curves, ("o", "s")):
        r = np.arange(1, s.size + 1)
        axs[0].semilogy(r, s/s[0], mk, ms=3.5, label=lab)
        axs[1].semilogy(r, 1.0 - np.cumsum(s**2)/np.sum(s**2) + 1e-17, mk, ms=3.5, label=lab)
    axs[0].set_ylabel(r"$\sigma_r/\sigma_1$")
    axs[1].set_ylabel(r"$1 - \sum_{i\leq r}\sigma_i^2 / \sum_i \sigma_i^2$")
    for ax in axs:
        ax.set_xlabel("r")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(r"POD of $\tilde q = W^{1/2} q'$ (W = V/V$_{min}$): leading-edge impulse, $\beta = 1$")
    out = os.path.join(C.figures, f"pod_sv_{args.traj:03d}.png")
    fig.savefig(out, dpi=130)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
