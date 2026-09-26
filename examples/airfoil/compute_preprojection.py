"""POD pre-projection of the airfoil training data (section 5.1).

Turns the full-grid perturbation trajectories written by
``generate_trajectories.py`` into the reduced files the training scripts
load from ``trajectories/``:

1. crop to the ROM domain [-4, 13] x [-3, 3], which keeps the outflow
   artifacts out of the data (rom_domain.py);
2. POD in the cell-volume inner product <a, b> = a^T V b, by the method of
   snapshots.  By default each trajectory is first divided by the square root
   of its time-averaged energy, so the beta = 0.1 runs are not drowned out by
   the beta = 2 ones (energy ratio 400); ``--no-normalize`` does the POD on the
   raw snapshots instead, as the cavity example does;
3. project the RAW trajectories onto the first r modes, X = Phi^T V q.

Phi is V-orthonormal, so ||X||^2 = q^T V q and E_pert = ||X||^2 / V_min.  The
per-trajectory weight is the time-averaged ||X||^2, which the training
pool uses to normalise each trajectory's contribution to the cost.

Writes (all read by train_*.py, read_results.py and the movie scripts):
    trajectories/traj_%03d.npy     (r, n_t) POD coefficients
    trajectories/weight_%03d.npy   [time-averaged ||X||^2]
    trajectories/time.npy, parameters.npy  (beta, x_c, y_c) per trajectory
    trajectories/decoder.npz       Phi_project (n_crop, r), q_base, n_u, n_v
    weights.npy                    full-grid cell volumes V

Usage: python compute_preprojection.py [--r 300] [--no-normalize]
"""

from __future__ import annotations

import argparse
import glob
import os
from types import SimpleNamespace

import numpy as np

from make_initial_conditions import cell_volumes
from rom_domain import X_BOUNDS, Y_BOUNDS, crop_mask

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data"))
    ap.add_argument("--out", default=os.path.join(HERE, "trajectories"))
    ap.add_argument("--r", type=int, default=300, help="number of POD modes kept")
    ap.add_argument("--no-normalize", action="store_true",
                    help="POD on raw snapshots, not energy-normalised ones")
    args = ap.parse_args()

    meta = np.load(os.path.join(args.data, "meta.npz"))
    t, params = meta["time"], meta["parameters"]
    n_traj = len(params)
    files = [os.path.join(args.data, f"fluct_{k:03d}.npy") for k in range(n_traj)]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"{len(missing)} trajectories missing, e.g. {missing[0]}")

    with np.load(str(meta["baseflow"])) as snap:
        mesh = SimpleNamespace(**{c: snap[c] for c in ("xu", "yu", "xv", "yv")})
        mask = crop_mask(snap)
        q_base = snap["q"][mask]
        xu, yu = snap["xu"][1:-1], snap["yu"][1:-1]
        n_u_full = len(xu)*len(yu)
    V_full = cell_volumes(mesh)
    V = V_full[mask]
    n_u = int(mask[:n_u_full].sum())
    n_v = int(mask[n_u_full:].sum())
    print(f"ROM domain {X_BOUNDS} x {Y_BOUNDS}: n_u = {n_u}, n_v = {n_v}, "
          f"n = {V.size} of {V_full.size}")

    # ---- snapshot matrix -------------------------------------------------
    n_t = len(t)
    Y = np.empty((V.size, n_traj*n_t))
    energy = np.empty(n_traj)
    for k, f in enumerate(files):
        Qk = np.load(f, mmap_mode="r")
        if Qk.shape != (V_full.size, n_t):
            raise SystemExit(f"{f} has shape {Qk.shape}, expected "
                             f"{(V_full.size, n_t)}")
        Y[:, k*n_t:(k + 1)*n_t] = Qk[mask]
        blk = Y[:, k*n_t:(k + 1)*n_t]
        energy[k] = np.mean(((blk*blk)*V[:, None]).sum(0))
        print(f"  loaded {os.path.basename(f)}: beta = {params[k, 0]:g}, "
              f"time-averaged E_pert = {energy[k]/V_full.min():.3e}")

    scale = (np.ones(n_traj) if args.no_normalize else 1/np.sqrt(energy))
    for k in range(n_traj):
        Y[:, k*n_t:(k + 1)*n_t] *= scale[k]

    # ---- method of snapshots in the V inner product ----------------------
    sqV = np.sqrt(V)
    Ys = Y*sqV[:, None]
    G = Ys.T @ Ys
    del Ys
    lam, U = np.linalg.eigh(G)
    lam, U = lam[::-1], U[:, ::-1]
    if lam[args.r - 1] <= lam[0]*1e-13:
        raise SystemExit(f"mode {args.r} is at round-off; lower --r")
    frac = np.cumsum(lam)/lam.sum()
    print(f"\nPOD ({'energy-normalised' if not args.no_normalize else 'raw'} "
          f"snapshots): variance captured by 50 modes = {100*frac[49]:.2f}%, "
          f"by {args.r} modes = {100*frac[args.r - 1]:.5f}%  "
          f"(paper: 96.3% and > 99.99%)")

    Phi = Y @ (U[:, :args.r]/np.sqrt(lam[:args.r]))
    for k in range(n_traj):                       # undo the normalisation
        Y[:, k*n_t:(k + 1)*n_t] /= scale[k]
    PhiTV = np.ascontiguousarray((Phi*V[:, None]).T)
    orth = np.abs(PhiTV @ Phi - np.eye(args.r)).max()
    print(f"max |Phi^T V Phi - I| = {orth:.1e}")

    # ---- project and save ------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    for f in glob.glob(os.path.join(args.out, "deriv_[0-9][0-9][0-9].npy")):
        os.remove(f)          # TrainingPool caches these; they would be stale
    print(f"\n{'traj':>4}{'beta':>6}{'weight':>12}{'captured':>11}")
    for k in range(n_traj):
        Yk = Y[:, k*n_t:(k + 1)*n_t]
        X = PhiTV @ Yk
        e_full = ((Yk*Yk)*V[:, None]).sum()
        captured = (X*X).sum()/e_full
        weight = np.mean((X*X).sum(0))
        np.save(os.path.join(args.out, f"traj_{k:03d}.npy"), X)
        np.save(os.path.join(args.out, f"weight_{k:03d}.npy"), [weight])
        print(f"{k:>4}{params[k, 0]:>6g}{weight:>12.3e}{100*captured:>10.4f}%")
    np.save(os.path.join(args.out, "time.npy"), t)
    np.save(os.path.join(args.out, "parameters.npy"), params)
    np.savez(os.path.join(args.out, "decoder.npz"), Phi_project=Phi,
             q_base=q_base, n_u=n_u, n_v=n_v, pod_eigenvalues=lam,
             normalized=not args.no_normalize)
    np.save(os.path.join(HERE, "weights.npy"), V_full)
    print(f"\nsaved -> {os.path.relpath(args.out, HERE)}/ and weights.npy")


if __name__ == "__main__":
    main()
