"""Re-projected time derivatives of the latent training data (3D wing).

The 3D version of examples/airfoil/compute_reprojected_derivatives.py.  For a
rank-r basis pair (Phi, Psi) from compute_rom_bases.py (r_l-vectors in the
lossless coordinates, full-space bases U Phi, U Psi), every training snapshot
gives z = Psi^T x (x its lossless coordinates) and

    dz/dt = (U Psi)^T W^{1/2} [ P f(q_b + W^{-1/2} (U Phi) z) - P f(q_b) ],

with f the solver's momentum right-hand side and P the linear part of its
constraint projection, P v = enforce_constraints(v) - enforce_constraints(0).
POD: Psi = Phi (Galerkin); balanced: the oblique pair, Psi^T Phi = I
(Petrov-Galerkin).  The pairs sample that projected model itself.

Cost.  The full-space bases U Phi and U Psi of every requested basis are
formed in ONE pass over the trajectory files (rom_compress.Compressed.lift)
and kept as float32 (N x r: 16 GB at r = 50); each basis pair then sits on
the GPU and every state costs two chunked float64 GEMVs plus one solver
right-hand side -- no further passes over the data.

Usage (incompreso venv, GPU):
    python compute_reprojected_derivatives.py --basis pod balanced [--r 50]
           [--data fom_data_wing_re1000_train] [--config config_re1000.yaml]
Writes <data>/latent_<basis>_r<r>_reproj/{traj,deriv,weight}_%03d.npy and
time.npy (TrainingPool layout).
"""

from __future__ import annotations

import argparse
import os
import time as timer

import numpy as np

from incompreso import parse_input_file
from incompreso.backend import GPU, to_backend, xp

from rom_compress import HERE, Compressed

CHUNK = 4_000_000          # rows per float64 GEMV block on the GPU


def matvec(M, z):
    """M z for a float32 device matrix M (N, r), in float64 by row blocks."""
    out = xp.empty(M.shape[0])
    zz = xp.asarray(z, dtype=xp.float64)
    for a in range(0, M.shape[0], CHUNK):
        out[a:a + CHUNK] = M[a:a + CHUNK].astype(xp.float64) @ zz
    return out


def rmatvec(M, g):
    """M^T g for a float32 device matrix M (N, r) and float64 g, by row blocks."""
    acc = xp.zeros(M.shape[1])
    for a in range(0, M.shape[0], CHUNK):
        acc += M[a:a + CHUNK].astype(xp.float64).T @ g[a:a + CHUNK]
    return acc


def gram(Ma, Mb):
    """Ma^T Mb (r x r), float64, by row blocks."""
    acc = xp.zeros((Ma.shape[1], Mb.shape[1]))
    for a in range(0, Ma.shape[0], CHUNK):
        acc += Ma[a:a + CHUNK].astype(xp.float64).T @ Mb[a:a + CHUNK].astype(xp.float64)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", choices=["pod", "balanced"], nargs="+", required=True)
    ap.add_argument("--r", type=int, default=50)
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data_wing_re1000_train"))
    ap.add_argument("--config", default=os.path.join(HERE, "config_re1000.yaml"))
    args = ap.parse_args()
    r = args.r

    comp = Compressed(os.path.join(args.data, "compressed.npz"))
    bases = np.load(os.path.join(args.data, "bases.npz"), allow_pickle=True)
    X, time = comp.X, comp.time
    n_traj, _, n_t = X.shape

    # every distinct r_l-basis matrix needed, lifted together in one data pass
    pairs, cols = {}, []
    for basis in args.basis:
        if basis == "pod":
            Phi = Psi = bases["Phi_pod"][:, :r]
            pairs[basis] = (Phi, Psi, len(cols), len(cols))
            cols.append(Phi)
        else:
            Phi, Psi = bases["Phi_bal"][:, :r], bases["Psi_bal"][:, :r]
            pairs[basis] = (Phi, Psi, len(cols), len(cols) + 1)
            cols += [Phi, Psi]
        if Phi.shape[1] < r:
            raise SystemExit(f"only {Phi.shape[1]} {basis} modes saved")
    t0 = timer.time()
    L = comp.lift(np.concatenate(cols, axis=1), xp)        # host float32 (N, r*len(cols))
    print(f"lifted {len(cols)} basis matrices (N x {r}) in one pass: "
          f"{timer.time() - t0:.0f} s", flush=True)

    sim = parse_input_file(args.config)
    sp = sim["time_stepper"].spatial_operators
    ib = sim["time_stepper"].immersed_body
    sqrt_w = to_backend(comp.data.sqrt_w)
    with np.load(comp.baseflow) as s:
        qb = to_backend(s["q"])
    P0 = ib.enforce_constraints(0.0, xp.zeros_like(qb)).copy()

    def Pf(q):
        f = sp.evaluate_right_hand_side(0.0, q).copy()
        return ib.enforce_constraints(0.0, f) - P0

    Pf_b = Pf(qb)
    print(f"backend: {'GPU' if GPU else 'CPU'}; ||P f(q_b)|| = "
          f"{float(xp.linalg.norm(Pf_b)):.2e}; {n_traj*n_t} states per basis", flush=True)

    for basis in args.basis:
        Phi, Psi, i_phi, i_psi = pairs[basis]
        PhiG = xp.asarray(np.ascontiguousarray(L[:, i_phi*r:(i_phi + 1)*r]))
        PsiG = PhiG if i_psi == i_phi else xp.asarray(
            np.ascontiguousarray(L[:, i_psi*r:(i_psi + 1)*r]))
        biorth = float(xp.abs(gram(PsiG, PhiG) - xp.eye(r)).max())
        print(f"\n=== {basis}, r = {r}: full-space max |(U Psi)^T (U Phi) - I| = "
              f"{biorth:.1e} (float32 storage of the lifted bases) ===", flush=True)

        Z = np.stack([Psi.T @ X[k] for k in range(n_traj)])        # (n_traj, r, n_t)
        dZ = np.empty_like(Z)
        t0 = timer.time()
        for k in range(n_traj):
            for i in range(n_t):
                q = qb + matvec(PhiG, Z[k, :, i])/sqrt_w
                dZ[k, :, i] = (rmatvec(PsiG, sqrt_w*(Pf(q) - Pf_b))).get() if GPU \
                    else rmatvec(PsiG, sqrt_w*(Pf(q) - Pf_b))
            print(f"  traj {k}: {n_t} states [{timer.time() - t0:.0f} s]", flush=True)
        del PhiG, PsiG
        if GPU:
            xp.get_default_memory_pool().free_all_blocks()

        out = os.path.join(args.data, f"latent_{basis}_r{r}_reproj")
        os.makedirs(out, exist_ok=True)
        dt = time[1] - time[0]
        print(f"{'traj':>4}{'||dz/dt|| mean':>16}{'||fd - reproj||/||reproj||':>28}")
        for k in range(n_traj):
            fd = (Z[k, :, :-4] - 8*Z[k, :, 1:-3] + 8*Z[k, :, 3:-1] - Z[k, :, 4:])/(12*dt)
            rel = np.linalg.norm(fd - dZ[k, :, 2:-2])/np.linalg.norm(dZ[k, :, 2:-2])
            np.save(os.path.join(out, f"traj_{k:03d}.npy"), np.ascontiguousarray(Z[k]))
            np.save(os.path.join(out, f"deriv_{k:03d}.npy"), np.ascontiguousarray(dZ[k]))
            np.save(os.path.join(out, f"weight_{k:03d}.npy"), [comp.energy[k]])
            print(f"{k:>4}{np.linalg.norm(dZ[k], axis=0).mean():>16.3e}{rel:>28.3e}")
        np.save(os.path.join(out, "time.npy"), time)
        print(f"{n_traj*n_t} re-projected derivatives in {timer.time() - t0:.0f} s -> "
              f"{os.path.relpath(out, HERE)}/", flush=True)


if __name__ == "__main__":
    main()
