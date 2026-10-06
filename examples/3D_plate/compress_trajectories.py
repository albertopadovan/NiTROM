"""Lossless POD compression of the 3D training trajectories (see rom_compress.py).

One streamed pass over the trajectory files (TSQR on the GPU), then the SVD
of the m x m R factor.  Only singular values below ``--tol`` (relative; the
float32 storage of the data is ~6e-8 per entry) are dropped, and the
discarded energy is printed.  Checks: ||x_k(t)||^2 against the W-energy
generate_trajectories.py computed in float64 at run time.

Usage (incompreso venv, GPU):
    python compress_trajectories.py [--data fom_data_wing_re1000_train]
                                    [--trajs 0 1 2] [--tol 1e-10]
Writes <data>/compressed.npz: X (n_traj, r_l, n_t), S, V, scale, energy, ...
(a few MB; the basis itself is implicit, U = Y V S^{-1}).
"""

from __future__ import annotations

import argparse
import os
import time as timer

import numpy as np

from incompreso.backend import xp

from rom_compress import HERE, SnapshotData, tsqr_R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data_wing_re1000_train"))
    ap.add_argument("--trajs", type=int, nargs="+", default=None,
                    help="trajectories to compress (default: every fluct_KKK.npy present)")
    ap.add_argument("--tol", type=float, default=1e-10,
                    help="drop singular values below tol * S_0")
    args = ap.parse_args()

    meta = np.load(os.path.join(args.data, "meta.npz"))
    params = meta["parameters"]
    trajs = args.trajs or [k for k in range(len(params))
                           if os.path.exists(os.path.join(args.data, f"fluct_{k:03d}.npy"))]
    E = np.stack([np.load(os.path.join(args.data, f"energy_{k:03d}.npy")) for k in trajs])
    Ebar = E.mean(axis=1)
    scale = 1/np.sqrt(Ebar)
    data = SnapshotData(args.data, trajs, scale)
    print(f"trajectories {trajs}: N = {data.N}, n_t = {data.n_t}, m = {data.m}; "
          f"time-averaged W-energy {np.array2string(Ebar, precision=3)}")

    t0 = timer.time()
    R = tsqr_R(data, xp)
    R = R.get() if hasattr(R, "get") else R
    print(f"TSQR: {timer.time() - t0:.0f} s")

    _, S, Vt = np.linalg.svd(R)
    keep = S > args.tol*S[0]
    r_l = int(keep.sum())
    disc = np.sum(S[~keep]**2)/np.sum(S**2)
    print(f"singular values: S_0 = {S[0]:.4e}, S_{data.m - 1}/S_0 = {S[-1]/S[0]:.2e}; "
          f"kept r_l = {r_l} of {data.m} (S/S_0 > {args.tol:g}), "
          f"discarded energy fraction {disc:.2e}")
    for r in (1, 10, 50, 100, 200, 300, 400, 500):
        if r <= data.m:
            print(f"    S_{r - 1}/S_0 = {S[r - 1]/S[0]:.3e}   "
                  f"1 - captured(r={r}) = {1 - np.sum(S[:r]**2)/np.sum(S**2):.3e}")
    S, V = S[:r_l], Vt[:r_l].T                                   # (r_l,), (m, r_l)

    # coordinates x = U^T q~ of the UNscaled snapshots: (S V^T)[:, block k] / s_k
    C = S[:, None]*V.T
    X = np.stack([C[:, j*data.n_t:(j + 1)*data.n_t]/scale[j] for j in range(len(trajs))])

    print(f"\n{'traj':>4}{'beta':>6}{'max | ||x||^2 / E_W - 1 |':>28}")
    for j, k in enumerate(trajs):
        rel = np.abs((X[j]**2).sum(0)/E[j] - 1).max()
        print(f"{k:>4}{params[k, 0]:>6g}{rel:>28.2e}")

    out = os.path.join(args.data, "compressed.npz")
    np.savez(out, X=X, S=S, V=V, scale=scale, energy=Ebar, trajs=np.array(trajs),
             time=data.time, parameters=params[trajs], tol=args.tol,
             data_dir=os.path.abspath(args.data), baseflow=data.baseflow)
    print(f"saved -> {out}  (X: {X.shape})")


if __name__ == "__main__":
    main()
