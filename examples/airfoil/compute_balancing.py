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

Non-uniformly sampled data (fom_data_merged: 0.02 up to t = 4, then 0.2):
the fine early samples serve to enrich X_0 -- 20 checkpoints in [0, 2) from
the fast early dynamics plus 20 spread over [0.2, 20] -- while every window
uses the same uniform reference grid tau = 0, 0.2, ..., T_w (rectangle rule),
T_w = t_end - t_c,max.  Window samples t_c + tau that are not stored (only
after t = 4, for checkpoints off the 0.2 grid) are Lagrange-interpolated from
the 0.2 data, in the wake phase where that is accurate to < 1%.  The reduced
quantities saved for the fits stay on the uniform 0.2 grid.

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


def preproject(X, use_gpu=True, n_check=40, seed=0):
    r"""Exact reduction of the snapshot data for the balancing.

    Every quantity of the balancing -- windows (interpolated ones included),
    checkpoint states X_0, Q, H, Phi, Psi -- is a linear combination of the
    snapshots, and only their Euclidean inner products enter.  With the
    stacked snapshots Y = U C (U orthonormal columns, NO truncation),
    x_i^T x_j = c_i^T c_j, so balancing the coefficient trajectories and
    lifting Phi = U Phi_c, Psi = U Psi_c is the full-size balancing exactly.
    U comes from a Householder QR of Y rather than POD by the method of
    snapshots: QR is exact to machine precision, while the snapshot Gram
    matrix squares the conditioning.  (POD modes would only rotate U, to which
    the balancing is invariant.)

    :param X: (n_traj, N, n_t) weighted snapshots
    :returns: (C, lift, err): coefficient trajectories (n_traj, m, n_t) with
        m = n_traj*n_t, a function A -> U A (host arrays), and the relative
        reconstruction error of n_check random snapshots
    """
    n_traj, N, n_t = X.shape
    m = n_traj*n_t
    cols = np.random.default_rng(seed).choice(m, size=min(n_check, m),
                                              replace=False)
    if use_gpu:
        import cupy as cp
        Y = cp.empty((N, m))
        for k in range(n_traj):
            Y[:, k*n_t:(k + 1)*n_t] = cp.asarray(X[k])
        U, Rm = cp.linalg.qr(Y, mode="reduced")
        Yc = Y[:, cols]
        err = float(cp.linalg.norm(U @ Rm[:, cols] - Yc)/cp.linalg.norm(Yc))
        del Y, Yc
        C = cp.asnumpy(Rm)
        lift = lambda A: cp.asnumpy(U @ cp.asarray(np.ascontiguousarray(A)))
    else:
        import scipy.linalg as sla
        Y = np.concatenate([X[k] for k in range(n_traj)], axis=1)
        U, C = sla.qr(Y, mode="economic")
        err = float(np.linalg.norm(U @ C[:, cols] - Y[:, cols])
                    / np.linalg.norm(Y[:, cols]))
        del Y
        lift = lambda A: U @ np.ascontiguousarray(A)
    C = np.ascontiguousarray(
        np.stack([C[:, k*n_t:(k + 1)*n_t] for k in range(n_traj)]))
    return C, lift, err


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
                    help="snapshots between checkpoints (uniform data)")
    # non-uniformly sampled data (e.g. fom_data_merged): the fine early
    # samples enrich X_0 with checkpoints from the fast early dynamics
    ap.add_argument("--ckpt-early", type=float, nargs=3, default=[0.0, 2.0, 20],
                    metavar=("T0", "T1", "N"),
                    help="N checkpoints evenly spaced on [T0, T1)")
    ap.add_argument("--ckpt-late", type=float, nargs=3, default=[0.2, 20.0, 20],
                    metavar=("T0", "T1", "N"),
                    help="N checkpoints evenly spaced on [T0, T1] (snapped to "
                         "the nearest sample; duplicates of the early set "
                         "dropped)")
    ap.add_argument("--window-dt", type=float, default=0.2,
                    help="uniform spacing of the common window (rectangle rule)")
    ap.add_argument("--preproject", choices=["qr-gpu", "qr-cpu", "none"],
                    default="qr-gpu",
                    help="balance the exact (untruncated) QR coefficients of "
                         "the snapshots instead of the full state; identical "
                         "result, dimension m = n_traj*n_t instead of N")
    ap.add_argument("--interp-order", type=int, default=6,
                    help="Lagrange order where t_c + tau is not a stored sample")
    ap.add_argument("--dt-reduced", type=float, default=0.2,
                    help="grid of the saved reduced quantities (the NiTROM / "
                         "OpInf grid)")
    args = ap.parse_args()

    meta = np.load(os.path.join(args.data, "meta.npz"))
    time, params = meta["time"], meta["parameters"]
    n_traj, n_t = len(params), len(time)
    uniform = np.allclose(np.diff(time), time[1] - time[0], rtol=1e-9, atol=1e-12)
    if args.out is None:
        args.out = os.path.join(
            HERE, ("balancing" if args.domain == "rom" else "balancing_full")
            + ("" if uniform else "_quad"))
    # reduced quantities stay on the uniform NiTROM / OpInf grid
    keep = (np.arange(n_t) if uniform else np.flatnonzero(
        np.isclose(np.round(time/args.dt_reduced)*args.dt_reduced, time,
                   rtol=0, atol=1e-9)))
    print(f"time grid: {'uniform' if uniform else 'non-uniform'}, {n_t} "
          f"samples; reduced quantities on {len(keep)} of them")
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
    energy = ((X[:, :, keep]**2).sum(axis=1)).mean(axis=1)   # <||q~||^2>_t
    print(f"loaded {n_traj} trajectories x {n_t} snapshots "
          f"({X.nbytes/1e9:.1f} GB) in {timer.time() - t0:.0f} s")
    lift = None
    if args.preproject != "none":
        t0 = timer.time()
        X, lift, err = preproject(X, use_gpu=(args.preproject == "qr-gpu"))
        print(f"pre-projection ({args.preproject}): N = {N} -> m = {X.shape[1]}, "
              f"relative reconstruction error {err:.1e}, {timer.time() - t0:.0f} s")

    betas = np.unique(params[:, 0])
    families = [[k for k in range(n_traj) if params[k, 0] == b] for b in betas]
    print(f"families (one per beta): "
          + ", ".join(f"beta = {b:g}: {fam}" for b, fam in zip(betas, families)))

    t0 = timer.time()
    if uniform:
        quad = None
        kw = dict(n_checkpoints=args.m_ckpt, checkpoint_stride=args.q_ckpt)
    else:
        snap = lambda tt: [float(time[np.argmin(np.abs(time - x))]) for x in tt]
        a0, a1, na = args.ckpt_early
        b0, b1, nb = args.ckpt_late
        early = snap(np.linspace(a0, a1, int(na), endpoint=False))
        late = [t for t in snap(np.linspace(b0, b1, int(nb)))
                if not np.any(np.isclose(t, early, atol=1e-9))]
        t_ck = np.array(sorted(set(early) | set(late)))
        T_w = time[-1] - t_ck[-1]
        L = int(np.floor(T_w/args.window_dt + 1e-9)) + 1
        nodes = args.window_dt*np.arange(L)
        weights = np.full(L, args.window_dt)
        quad = dict(nodes=nodes, weights=weights, checkpoint_times=t_ck,
                    interp_order=args.interp_order)
        kw = dict(quadrature=quad)
        print(f"checkpoints: {len(early)} in [{a0:g}, {a1:g}) + {len(late)} in "
              f"[{b0:g}, {b1:g}] = {len(t_ck)} per trajectory: "
              f"{np.array2string(t_ck, precision=2, max_line_width=200)}")
        print(f"reference window: uniform, tau = 0..{nodes[-1]:g} every "
              f"{args.window_dt:g} ({L} samples, rectangle rule); order-"
              f"{args.interp_order} interpolation only where t_c + tau is not "
              f"a stored sample")
    Phi, Psi, Sig, info = compute_data_driven_balancing(
        Pool(X, time), families=families, normalize=True, **kw)
    print(f"balancing: {timer.time() - t0:.0f} s, {len(Sig)} balanced modes, "
          f"M ranks kept {[f['rank_M_kept'] for f in info['families']]}, "
          f"window {info['window_length']} "
          + ("snapshots" if uniform else "quadrature nodes"))
    print(f"Hankel singular values [:10] = "
          f"{np.array2string(Sig[:10], precision=3)}")

    R = min(args.r_save, len(Sig))
    if R < args.r_save:
        print(f"WARNING: only {len(Sig)} balanced modes exist; keeping all")
    Phi, Psi = np.ascontiguousarray(Phi[:, :R]), np.ascontiguousarray(Psi[:, :R])

    # Reduced quantities for the fits: all 2-D GEMMs (the venv's threaded
    # OpenBLAS returns garbage for 1-D dot products on strided views).  With
    # the pre-projection they come straight from the coefficients -- U has
    # orthonormal columns, so Psi^T x = Psi_c^T c and so on.
    Xr = X[:, :, keep]                                          # NiTROM grid
    del X
    Z = np.stack([Psi.T @ Xr[k] for k in range(n_traj)])       # (n_traj, R, n_t)
    PhiTX = np.stack([Phi.T @ Xr[k] for k in range(n_traj)])
    PhiTPhi = Phi.T @ Phi
    sq = (Xr*Xr).sum(axis=1)                                    # (n_traj, n_t)
    if lift is not None:
        t0 = timer.time()
        Phi, Psi = lift(Phi), lift(Psi)
        print(f"lifted Phi, Psi to N = {Phi.shape[0]} in {timer.time() - t0:.0f} s")
    biorth = np.abs(Psi.T @ Phi - np.eye(R)).max()
    print(f"max |Psi^T Phi - I| over {R} modes = {biorth:.1e}")

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
             baseflow=str(meta["baseflow"]), domain=args.domain,
             data=os.path.abspath(args.data), preproject=args.preproject,
             quadrature=np.array(quad, dtype=object))
    np.savez(os.path.join(args.out, f"reduced_r{R}.npz"),
             Z=Z, PhiTX=PhiTX, PhiTPhi=PhiTPhi, sq=sq, energy=energy,
             time=time[keep], parameters=params)
    print(f"\nsaved -> {args.out}/balancing.npz, reduced_r{R}.npz")


if __name__ == "__main__":
    main()
