"""Data-driven balanced basis for the airfoil, straight from the FOM trajectories.

The airfoil analogue of the cavity's train_models.py: the Lall-averaged
empirical balancing of nitrom.utils.compute_data_driven_balancing, applied to
the nine impulse responses written by generate_trajectories.py.  There is no
POD pre-projection -- the balancing works at the full (cropped) state
dimension, and since it only ever forms Gramian factors, never an n x n
matrix, that is affordable even at n ~ 6e5.

State.  The paper's weighted state q~ = W^{1/2} q (eq. 42), W = V / V_min,
restricted to the ROM domain D = [-4, 13] x [-3, 3] (rom_domain.py).  In q~ the energy inner
product is the Euclidean one, which is what the balancing assumes.

Families.  Lall-Marsden-Glavaski averaging groups initial conditions of equal
amplitude, because for a nonlinear system observability energy depends on
amplitude and not only direction.  Here the three stations share each beta,
so there are three families (beta = 0.1, 1, 2), each holding the checkpoints
of three trajectories.  Every checkpoint inherits its parent's family.

Normalization (Lall).  Every window is divided by the norm of its own initial
condition, so each column of X_0 is a unit vector -- Lall's 1/c^2 Gramian
scaling -- and the three beta families are averaged with weight 1/3.  At t = 0
that norm is exactly beta (B_f has unit W-norm, so ||q~(0)|| = beta);
checkpoints are normalized by their own norm, as in the cavity, while
keeping their parent's family.

Checkpoints.  25 per trajectory, every 4th snapshot, so initial conditions
span t in [0, 19.2] and each window is 104 snapshots (20.8 time units) long.
The cavity's every-2nd-snapshot choice does NOT carry over: Psi lies in the span
of the checkpoint states, and the airfoil perturbation is a wave packet
convected down the wake, so checkpoints confined to t <= 7.6 never see where
the packet is at its energy peak (t ~ 12, x ~ 5-11).  On six trajectories
that basis missed 100% of the state from t ~ 12 on (r = 50 time-integrated
projection error ~75%); spreading the checkpoints to t <= 19 brought it to
4-12%.  Denser checkpoints (40 at stride 3) did worse: shorter windows.

Also stores, for the leading r_save modes, everything the reduced fits need
without touching the full data again: z_j = Psi^T q~_j, Phi^T q~_j, Phi^T Phi
and ||q~_j(t)||^2 (the last three give exact full-state reconstruction errors).

Memory: the data (~8.7 GB) plus one family's staged windows (75 windows of
104 snapshots, ~38 GB).

Usage: python compute_balancing.py [--domain rom|full] [--r-save 100]
                                   [--m-ckpt 25] [--q-ckpt 4]
Writes <out>/balancing.npz and <out>/reduced_r<r_save>.npz, with <out> =
balancing/ for the ROM domain and balancing_full/ for the whole mesh.
"""

from __future__ import annotations

import argparse
import os
import time as timer
from types import SimpleNamespace

import numpy as np

from make_initial_conditions import cell_volume_weights
from rom_domain import X_BOUNDS, Y_BOUNDS, crop_mask

from nitrom.backend import set_backend
from nitrom.utils import compute_data_driven_balancing

set_backend("numpy")

HERE = os.path.dirname(os.path.abspath(__file__))


def n_u_full_of(snap):
    return (len(snap["xu"]) - 2)*(len(snap["yu"]) - 2)


def n_v_full_of(snap):
    return (len(snap["xv"]) - 2)*(len(snap["yv"]) - 2)


class Pool:
    """Minimal TrainingPool view: compute_data_driven_balancing needs only the
    raw trajectories, the time grid and the (single-rank) layout."""

    def __init__(self, X, time):
        self.X, self.time = X, time
        self.world_size, self.rank, self.comm = 1, 0, None
        self.N = X.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data"))
    ap.add_argument("--domain", choices=["rom", "full"], default="rom",
                    help="ROM domain D (paper) or the whole simulation mesh; "
                         "the full mesh is what re-projection needs, since "
                         "Phi z must be a complete solver state")
    ap.add_argument("--out", default=None,
                    help="default: balancing/ (rom) or balancing_full/ (full)")
    ap.add_argument("--r-save", type=int, default=100,
                    help="balanced modes to keep on disk")
    ap.add_argument("--m-ckpt", type=int, default=25,
                    help="checkpoints (initial conditions) per trajectory")
    ap.add_argument("--q-ckpt", type=int, default=4,
                    help="snapshots between checkpoints")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(HERE, "balancing" if args.domain == "rom"
                                else "balancing_full")

    meta = np.load(os.path.join(args.data, "meta.npz"))
    time, params = meta["time"], meta["parameters"]
    n_traj, n_t = len(params), len(time)
    files = [os.path.join(args.data, f"fluct_{k:03d}.npy") for k in range(n_traj)]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"{len(missing)} trajectories missing, e.g. {missing[0]}")

    with np.load(str(meta["baseflow"])) as snap:
        mesh = SimpleNamespace(**{c: snap[c] for c in ("xu", "yu", "xv", "yv")})
        mask = (crop_mask(snap) if args.domain == "rom"
                else np.ones(n_u_full_of(snap) + n_v_full_of(snap), bool))
        n_u_full = (len(snap["xu"]) - 2)*(len(snap["yu"]) - 2)
    sqrt_w = np.sqrt(cell_volume_weights(mesh))[mask]
    N = int(mask.sum())
    print((f"ROM domain {X_BOUNDS} x {Y_BOUNDS}" if args.domain == "rom"
           else "full simulation mesh") + f": N = {N}, weighted state "
          f"q~ = W^(1/2) q")

    t0 = timer.time()
    X = np.empty((n_traj, N, n_t))
    for k, f in enumerate(files):
        X[k] = np.load(f, mmap_mode="r")[mask]*sqrt_w[:, None]
    energy = ((X*X).sum(axis=1)).mean(axis=1)        # <||q~||^2>_t = <E_pert>_t
    print(f"loaded {n_traj} trajectories x {n_t} snapshots "
          f"({X.nbytes/1e9:.1f} GB) in {timer.time() - t0:.0f} s")

    betas = np.unique(params[:, 0])
    families = [[k for k in range(n_traj) if params[k, 0] == b] for b in betas]
    print(f"families (one per beta): "
          + ", ".join(f"beta = {b:g}: {fam}" for b, fam in zip(betas, families)))

    t0 = timer.time()
    Phi, Psi, Sig, info = compute_data_driven_balancing(
        Pool(X, time), n_checkpoints=args.m_ckpt,
        checkpoint_stride=args.q_ckpt, families=families, normalize=True)
    print(f"balancing: {timer.time() - t0:.0f} s, {len(Sig)} balanced modes, "
          f"M ranks kept {[f['rank_M_kept'] for f in info['families']]}, "
          f"window {info['window_length']} of {n_t} snapshots")
    print(f"Hankel singular values [:10] = "
          f"{np.array2string(Sig[:10], precision=3)}")

    R = min(args.r_save, len(Sig))
    if R < args.r_save:
        print(f"WARNING: only {len(Sig)} balanced modes exist; keeping all")
    Phi, Psi = np.ascontiguousarray(Phi[:, :R]), np.ascontiguousarray(Psi[:, :R])
    biorth = np.abs(Psi.T @ Phi - np.eye(R)).max()
    print(f"max |Psi^T Phi - I| over {R} modes = {biorth:.1e}")

    # Reduced quantities for the fits: all 2-D GEMMs (the venv's threaded
    # OpenBLAS returns garbage for 1-D dot products on strided views).
    Z = np.stack([Psi.T @ X[k] for k in range(n_traj)])        # (n_traj, R, n_t)
    PhiTX = np.stack([Phi.T @ X[k] for k in range(n_traj)])
    PhiTPhi = Phi.T @ Phi
    sq = (X*X).sum(axis=1)                                      # (n_traj, n_t)

    # How much of each trajectory the oblique projection Phi Psi^T keeps.
    print(f"\nprojection error ||q~ - Phi Psi^T q~|| / ||q~|| (time-integrated):")
    print(f"{'r':>6}" + "".join(f"{k:>9}" for k in range(n_traj)))
    for r in (10, 20, 30, 50, 75, 100):
        if r > R:
            break
        row = []
        for k in range(n_traj):
            z = Z[k, :r]
            e2 = (sq[k].sum() - 2*(z*PhiTX[k, :r]).sum()
                  + (z*(PhiTPhi[:r, :r] @ z)).sum())
            row.append(np.sqrt(max(e2, 0.0)/sq[k].sum()))
        print(f"{r:>6}" + "".join(f"{e:>9.2e}" for e in row))

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "balancing.npz"),
             Phi=Phi, Psi=Psi, Sigma=Sig, sqrt_w=sqrt_w, mask=mask,
             n_u=int(mask[:n_u_full].sum()), n_v=int(mask[n_u_full:].sum()),
             m_ckpt=args.m_ckpt, q_ckpt=args.q_ckpt, families=np.array(
                 [np.array(f) for f in families], dtype=object),
             sM=np.array([f["sM"] for f in info["families"]], dtype=object),
             baseflow=str(meta["baseflow"]), domain=args.domain)
    np.savez(os.path.join(args.out, f"reduced_r{R}.npz"),
             Z=Z, PhiTX=PhiTX, PhiTPhi=PhiTPhi, sq=sq, energy=energy,
             time=time, parameters=params)
    print(f"\nsaved -> {args.out}/balancing.npz, reduced_r{R}.npz")


if __name__ == "__main__":
    main()
