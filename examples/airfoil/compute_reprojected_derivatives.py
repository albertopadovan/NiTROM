"""Re-projected time derivatives of the latent training data.

For every training snapshot q~ = W^{1/2} q (full mesh), with z = Psi^T q~,

    dz/dt = Psi^T W^{1/2} [ P f(q_b + W^{-1/2} Phi z) - P f(q_b) ],

where f is the solver's momentum right-hand side (evaluate_right_hand_side:
convection + viscosity, no pressure) and P the linear part of its constraint
projection (Leray + no-slip on the body), P v = enforce_constraints(v) -
enforce_constraints(0).  enforce_constraints is affine -- it also imposes the
u = 1 inflow and the body velocity -- so only its linear part acts on a time
derivative.  P f is the solver's exact time derivative: one RK2 step from a
training snapshot agrees with it to O(dt) (4e-3 at dt = 2e-3, 1e-3 at 5e-4).
P f(q_b) is 5e-9 (the base flow is converged); subtracting it keeps the
quadratic ROM free of a constant term.

These (z, dz/dt) pairs are samples of the Petrov-Galerkin model of (Phi, Psi)
itself, and f is exactly quadratic, so OpInf on them should reach round-off.

Needs the full-mesh balancing (compute_balancing.py --domain full): Phi z has
to be a complete solver state.  Runs on the GPU if incompreso's backend is.

Usage: python compute_reprojected_derivatives.py [--r 50]
Writes balancing_full/latent_r<r>_reproj/{traj,deriv,weight}_%03d.npy and
time.npy -- z, dz/dt, <||q~||^2> -- in the layout TrainingPool reads.
"""

from __future__ import annotations

import argparse
import os
import time as timer

import numpy as np

from incompreso import parse_input_file
from incompreso.backend import GPU, to_backend, to_numpy, xp

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=50)
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--balancing", default=os.path.join(HERE, "balancing_full"))
    args = ap.parse_args()
    r = args.r

    bal = np.load(os.path.join(args.balancing, "balancing.npz"), allow_pickle=True)
    if str(bal["domain"]) != "full":
        raise SystemExit("re-projection needs the full-mesh balancing "
                         "(compute_balancing.py --domain full)")
    R = bal["Phi"].shape[1]
    red = np.load(os.path.join(args.balancing, f"reduced_r{R}.npz"))
    Z_all, time, energy = red["Z"][:, :r], red["time"], red["energy"]
    n_traj, _, n_t = Z_all.shape

    sim = parse_input_file(args.config)
    sp = sim["time_stepper"].spatial_operators
    ib = sim["time_stepper"].immersed_body

    Phi = to_backend(np.ascontiguousarray(bal["Phi"][:, :r]))
    Psi = to_backend(np.ascontiguousarray(bal["Psi"][:, :r]))
    sqrt_w = to_backend(bal["sqrt_w"])
    qb = to_backend(np.load(str(bal["baseflow"]))["q"])
    P0 = ib.enforce_constraints(0.0, xp.zeros_like(qb)).copy()

    def Pf(q):
        f = sp.evaluate_right_hand_side(0.0, q).copy()
        return ib.enforce_constraints(0.0, f) - P0

    Pf_b = Pf(qb)
    print(f"backend: {'GPU' if GPU else 'CPU'}; r = {r}; "
          f"||P f(q_b)|| = {float(xp.linalg.norm(Pf_b)):.2e}")

    out = os.path.join(args.balancing, f"latent_r{r}_reproj")
    os.makedirs(out, exist_ok=True)
    t0 = timer.time()
    print(f"{'traj':>4}{'||dz/dt|| mean':>16}{'||fd - reproj||/||reproj||':>28}")
    for k in range(n_traj):
        Z = Z_all[k]
        dZ = np.empty_like(Z)
        for i in range(n_t):
            q = qb + (Phi @ to_backend(np.ascontiguousarray(Z[:, i])))/sqrt_w
            dZ[:, i] = to_numpy(Psi.T @ (sqrt_w*(Pf(q) - Pf_b)))
        # compare with the 4th-order finite differences the fd fit uses
        # (interior points only, where the stencil is centred)
        fd = (Z[:, :-4] - 8*Z[:, 1:-3] + 8*Z[:, 3:-1] - Z[:, 4:])/(12*(time[1] - time[0]))
        rel = np.linalg.norm(fd - dZ[:, 2:-2])/np.linalg.norm(dZ[:, 2:-2])
        np.save(os.path.join(out, f"traj_{k:03d}.npy"), np.ascontiguousarray(Z))
        np.save(os.path.join(out, f"deriv_{k:03d}.npy"), dZ)
        np.save(os.path.join(out, f"weight_{k:03d}.npy"), [energy[k]])
        print(f"{k:>4}{np.linalg.norm(dZ, axis=0).mean():>16.3e}{rel:>28.3e}",
              flush=True)
    np.save(os.path.join(out, "time.npy"), time)
    print(f"{n_traj*n_t} re-projected derivatives in {timer.time() - t0:.0f} s "
          f"-> {os.path.relpath(out, HERE)}/")


if __name__ == "__main__":
    main()
