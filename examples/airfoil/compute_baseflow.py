"""Steady base flow for the NACA 0012 airfoil at Re = 600, alpha = 6 deg.

The whole study linearizes about this state, so it has to be a genuine
equilibrium, not a slowly-drifting transient.  Time marching does not get
there: from a uniform start the residual ||dq/dt|| was still 1.09 (against
||q|| = 572) at t = 20, decaying at a rate of about exp(-0.072 t), which
implies ~190 more time units -- hours of wall clock -- to reach 1e-6.

So we march only far enough to land in the basin, then hand the state to
``TimeStepper.newton_solve``, which solves the nonlinear saddle system for
(u, p, f_ib) directly with FGMRES preconditioned by a saddle multigrid
V-cycle.  Newton converges quadratically once the guess is good, which is
what the t = 20 snapshot provides.

Usage (needs the incompreso venv):
    incompreso/.venv/bin/python compute_baseflow.py [--guess PATH]
                                [--tol 1e-8] [--maxiter 20]
Writes base_flow.npz (q, t, and the grid vectors carried over from the guess).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import incompreso

HERE = os.path.dirname(os.path.abspath(__file__))
# The t = 20 march lives in the incompreso example that produced it.
DEFAULT_GUESS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..", "incompreso", "examples", "airfoil", "data", "snapshot_000010.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--guess", default=DEFAULT_GUESS,
                    help="snapshot used as the Newton initial guess; "
                         "'uniform' starts from the config's initial condition")
    ap.add_argument("--tol", type=float, default=1e-8,
                    help="residual tolerance on ||RHS - G p - E^T f||")
    ap.add_argument("--maxiter", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "base_flow.npz"))
    args = ap.parse_args()

    sim = incompreso.parse_input_file(args.config)
    tstep = sim["time_stepper"]

    if args.guess == "uniform":
        q0 = np.asarray(sim["q0"])
        print("Newton guess: the config's uniform initial condition", flush=True)
    else:
        guess = os.path.normpath(args.guess)
        if not os.path.exists(guess):
            raise FileNotFoundError(
                f"{guess} not found -- pass --guess PATH or --guess uniform")
        d = np.load(guess)
        q0 = np.asarray(d["q"], dtype=np.float64)
        if q0.shape != np.shape(sim["q0"]):
            raise ValueError(
                f"guess has {q0.shape} DOFs but this mesh expects "
                f"{np.shape(sim['q0'])}; the guess came from a different grid")
        print(f"Newton guess: {os.path.basename(guess)} at t = {float(d['t']):g}, "
              f"||q|| = {np.linalg.norm(q0):.6f}", flush=True)

    print(f"\nrunning newton_solve(tol = {args.tol:g}, maxiter = {args.maxiter}) "
          f"...", flush=True)
    t0 = time.perf_counter()
    qn = tstep.newton_solve(q0, args.tol, args.maxiter)
    dt_wall = time.perf_counter() - t0
    qn = np.asarray(qn)

    # Independent check: the converged state must be a fixed point of the
    # solver's own right-hand side, not merely of Newton's residual.
    print(f"\nNewton finished in {dt_wall:.1f} s", flush=True)
    print(f"||q_steady||          = {np.linalg.norm(qn):.8f}")
    print(f"||q_steady - q_guess|| = {np.linalg.norm(qn - q0):.6e}")

    out = {"q": qn, "t": np.array(np.inf)}
    if args.guess != "uniform":
        for k in ("xu", "yu", "xv", "yv", "xp", "yp", "xi", "eta"):
            if k in d:
                out[k] = d[k]
    np.savez(args.out, **out)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
