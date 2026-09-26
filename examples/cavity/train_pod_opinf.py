"""
POD-based Operator Inference for the cavity, sweeping the regularization.

Fits dz/dt = A_r z + H_r (z x z) with z = U^T q on the seven training impulse
responses of eq. (5.5), for each lambda in the grid, then forecasts all seven
trajectories over the full window and reports:

  maxRe(Ar)  largest real part of the eigenvalues of A_r.  The cavity at
             Re = 8300 is linearly stable, so a positive value means the fit has
             invented growth -- the instability the paper reports in section 5.1.
  J          the NiTROM cost (3.7), each trajectory normalized by its
             time-averaged energy
  err        relative state error over all trajectories

The POD basis is cached in pod_r<r>.npy (the SVD is the slow part, ~10 s);
the winning model is saved to opinf_pod_r<r>.npy.

Usage: python train_pod_opinf.py [--r 20]
"""

import argparse
import os

import numpy as np

from balanced_opinf import opinf, pod_basis
from train_models import load_train, HERE

REG_GRID = 10.0**np.arange(-6, 8, 1.0)


def forecast(rom, Phi, Psi, trajs, time):
    """Integrate every trajectory; return (J, relative state error, n_diverged)."""
    J = num = den = 0.0
    nbad = 0
    for X in trajs:
        Z = rom.integrate(Psi.T @ X[:, 0], time, 1.0)
        den += np.linalg.norm(X)**2
        if not np.all(np.isfinite(Z)):
            nbad += 1
            continue
        Xh = Phi @ Z
        J += np.sum((X - Xh)**2)/np.mean(np.sum(X**2, axis=0))
        num += np.linalg.norm(X - Xh)**2
    if nbad:
        return np.inf, np.inf, nbad
    return J, np.sqrt(num/den), 0


def get_pod(trajs, r):
    """Leading r POD modes, cached: the SVD dominates the runtime."""
    path = os.path.join(HERE, f"pod_r{r}.npy")
    if os.path.exists(path):
        U = np.load(path)
        if U.shape[1] >= r:
            return U[:, :r]
    U = pod_basis(trajs, r)
    np.save(path, U)
    return U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=20)
    args = ap.parse_args()
    r = args.r

    trajs_n, betas, time = load_train()
    trajs = [abs(b)*T for b, T in zip(betas, trajs_n)]
    dt = time[1] - time[0]
    U = get_pod(trajs, r)

    den = sum(np.linalg.norm(X)**2 for X in trajs)
    proj = np.sqrt(sum(np.linalg.norm(X - U @ (U.T @ X))**2 for X in trajs)/den)
    print(f"POD r = {r}: {len(trajs)} trajectories, {len(time)} snapshots, "
          f"dt = {dt:g}")
    print(f"projection error of the training data: {proj:.4f}  "
          f"(the ROM cannot do better than this)\n")

    print(f"{'lambda':>9} {'maxRe(Ar)':>11} {'||Ar||':>9} {'||Hr||':>10} "
          f"{'J (3.7)':>12} {'err':>9}  status")
    best = None
    for reg in REG_GRID:
        rom = opinf(trajs, U, dt, "quadratic", reg=reg, C=None)
        J, err, nbad = forecast(rom, U, U, trajs, time)
        ev = np.linalg.eigvals(rom.Ar).real.max()
        st = "ok" if nbad == 0 else f"{nbad}/{len(trajs)} diverged"
        Js = f"{J:12.4e}" if np.isfinite(J) else f"{'inf':>12}"
        es = f"{err:9.4f}" if np.isfinite(err) else f"{'-':>9}"
        print(f"{reg:9.0e} {ev:+11.4f} {np.linalg.norm(rom.Ar):9.2f} "
              f"{np.linalg.norm(rom.Hr):10.2e} {Js} {es}  {st}")
        if np.isfinite(J) and (best is None or J < best[0]):
            best = (J, reg, rom, err)

    if best is None:
        print("\nevery regularization diverged; nothing saved.")
        return
    J, reg, rom, err = best
    print(f"\nbest: lambda = {reg:.0e}, J = {J:.4e}, relative state error = {err:.4f}")
    out = os.path.join(HERE, f"opinf_pod_r{r}.npy")
    np.save(out, dict(Phi=U, Psi=U, Ar=rom.Ar, Hr=rom.Hr, reg=reg, Sigma=None),
            allow_pickle=True)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
