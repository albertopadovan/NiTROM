"""POD and data-driven balanced bases of the 3D training data, computed in the
lossless coordinates of compress_trajectories.py.

The coordinates x = U^T q~ (U orthonormal, no truncation) preserve every
Euclidean inner product of the weighted snapshots, so both bases are the
full-dimension ones exactly, expressed as r_l-vectors (lift with
rom_compress.Compressed.lift):

  POD       Phi = Psi = leading left singular vectors of the snapshots, each
            trajectory scaled by 1/sqrt(Ebar_k) (the airfoil's compute_POD
            with normalize=True);
  balanced  nitrom's compute_data_driven_balancing on the coordinate
            trajectories (as the airfoil's compute_balancing.py with its exact
            QR pre-projection): Lall families by beta, normalized windows,
            checkpoints every --q-ckpt snapshots (default 2: every 0.4) up to
            t = --t-ckpt-max (default 18; later states of the beta = 0.01 run
            are dominated by the ~1e-5 base-flow drift, which the per-window
            normalization would weight like real perturbations).

Also prints the time-integrated projection error ||q~ - Phi Psi^T q~||/||q~||
per trajectory, exact in these coordinates.

Usage (python with nitrom on the path; numpy only):
    python compute_rom_bases.py [--data fom_data_wing_re1000_train] [--r-save 100]
Writes <data>/bases.npz.
"""

from __future__ import annotations

import argparse
import os
import time as timer

import numpy as np

from nitrom.backend import set_backend
from nitrom.utils import compute_data_driven_balancing

set_backend("numpy")

HERE = os.path.dirname(os.path.abspath(__file__))


class Pool:
    """Minimal TrainingPool view for compute_data_driven_balancing."""

    def __init__(self, X, time):
        self.X, self.time = X, time
        self.world_size, self.rank, self.comm = 1, 0, None
        self.N = X.shape[1]


def projection_errors(X, Phi, Psi, rs):
    rows = []
    for r in rs:
        P = Phi[:, :r] @ Psi[:, :r].T
        rows.append([np.linalg.norm(Xk - P @ Xk)/np.linalg.norm(Xk) for Xk in X])
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data_wing_re1000_train"))
    ap.add_argument("--r-save", type=int, default=100)
    ap.add_argument("--q-ckpt", type=int, default=2, help="snapshots between checkpoints")
    ap.add_argument("--t-ckpt-max", type=float, default=18.0, help="last checkpoint time")
    args = ap.parse_args()

    c = np.load(os.path.join(args.data, "compressed.npz"), allow_pickle=True)
    X, time, params, Ebar = c["X"], c["time"], c["parameters"], c["energy"]
    n_traj, r_l, n_t = X.shape
    dt = time[1] - time[0]
    m_ckpt = int(np.floor(args.t_ckpt_max/(args.q_ckpt*dt) + 1e-9)) + 1
    R = min(args.r_save, r_l)
    print(f"{n_traj} trajectories in r_l = {r_l} lossless coordinates, {n_t} snapshots")

    # POD of the energy-normalized snapshots
    Yn = np.concatenate([X[k]/np.sqrt(Ebar[k]) for k in range(n_traj)], axis=1)
    U_pod, S_pod, _ = np.linalg.svd(Yn, full_matrices=False)
    Phi_pod = np.ascontiguousarray(U_pod[:, :R])
    frac = np.cumsum(S_pod**2)/np.sum(S_pod**2)
    print("POD (normalized snapshots): 1 - captured energy at r = "
          + ", ".join(f"{r}: {1 - frac[r - 1]:.2e}" for r in (10, 20, 30, 50, 75, 100) if r <= r_l))

    # data-driven balancing, Lall families by beta
    betas = np.unique(params[:, 0])
    families = [[k for k in range(n_traj) if params[k, 0] == b] for b in betas]
    print("families: " + ", ".join(f"beta = {b:g}: {f}" for b, f in zip(betas, families)))
    t0 = timer.time()
    Phi_bal, Psi_bal, Sig, info = compute_data_driven_balancing(
        Pool(X, time), families=families, normalize=True,
        n_checkpoints=m_ckpt, checkpoint_stride=args.q_ckpt)
    Rb = min(R, len(Sig))
    Phi_bal, Psi_bal = Phi_bal[:, :Rb], Psi_bal[:, :Rb]
    print(f"checkpoints: {m_ckpt} per trajectory at t = 0, {args.q_ckpt*dt:g}, ..., "
          f"{(m_ckpt - 1)*args.q_ckpt*dt:g}")
    print(f"balancing: {timer.time() - t0:.1f} s, {len(Sig)} balanced modes, window "
          f"{info['window_length']} snapshots; Hankel SVs [:8] = "
          f"{np.array2string(Sig[:8], precision=3)}; "
          f"max |Psi^T Phi - I| = {np.abs(Psi_bal.T @ Phi_bal - np.eye(Rb)).max():.1e}")

    rs = [r for r in (10, 20, 30, 50, 75, 100) if r <= Rb]
    for name, Ph, Ps in (("POD", Phi_pod, Phi_pod), ("balanced", Phi_bal, Psi_bal)):
        E = projection_errors(X, Ph, Ps, rs)
        print(f"\n{name}: ||q~ - Phi Psi^T q~|| / ||q~|| (time-integrated)")
        print(f"{'r':>6}" + "".join(f"{f'traj {k}':>11}" for k in range(n_traj)))
        for r, row in zip(rs, E):
            print(f"{r:>6}" + "".join(f"{e:>11.2e}" for e in row))

    out = os.path.join(args.data, "bases.npz")
    np.savez(out, Phi_pod=Phi_pod, S_pod=S_pod, Phi_bal=Phi_bal, Psi_bal=Psi_bal,
             Sigma_bal=Sig, families=np.array([np.array(f) for f in families], dtype=object),
             m_ckpt=m_ckpt, q_ckpt=args.q_ckpt, t_ckpt_max=args.t_ckpt_max)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
