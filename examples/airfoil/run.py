"""March the NACA 0012 flow (Re = 600, alpha = 6 deg) to its steady base flow.

Newton is unavailable here: ``TimeStepper.newton_solve`` builds a saddle
multigrid preconditioner whose coarse-grid transfers are block-diagonal over
[u; v; p_pinned] only, so ``R · J · P`` fails with a dimension mismatch of
exactly 2 x n_markers (796 here) as soon as an immersed body adds force DOFs
to the augmented system.  So we time march.

The march is a continuation, not a fresh start.  ``config.yaml`` points its
initial condition at ``baseflow_seed.npz``, a state already marched to t = 20
whose residual ||dq/dt|| was 1.09 against ||q|| = 572, decaying at roughly
exp(-0.072 t); reaching 1e-6 therefore needs ~190 further time units, which is
where the config's t1 = 220 comes from.  Watch the residual with
``check_convergence.py`` and stop early once it plateaus -- there is no point
paying for time units past convergence.

Restarting is the config's job, not this script's: a ``type: file`` initial
condition hands back the state, the time to resume from, the snapshot index to
continue numbering at, and the body marker positions.  To extend a finished
march, point ``initial_condition.path`` at the newest snapshot and raise t1.

Usage (needs the incompreso venv):
    ../../../incompreso/.venv/bin/python run.py
    ... run.py --t1 300                # march further without editing the YAML
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from incompreso import parse_input_file

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--t1", type=float, default=None, help="override config t1")
    ap.add_argument("--mjump", type=int, default=None,
                    help="override config mjump (steps between snapshots)")
    args = ap.parse_args()

    sim = parse_input_file(args.config)
    t0 = sim["t0"]
    t1 = sim["t1"] if args.t1 is None else args.t1
    mjump = sim["mjump"] if args.mjump is None else args.mjump
    dt = sim["time_stepper"].dt

    if t1 <= t0:
        raise ValueError(
            f"t1 = {t1:g} is not beyond the start time t0 = {t0:g} -- the "
            f"initial condition already reached t0, so there is nothing to "
            f"march.  Raise t1 in the config or pass --t1.")

    print(f"start: t = {t0:g}, ||q|| = {np.linalg.norm(np.asarray(sim['q0'])):.6f}, "
          f"snapshot numbering continues from {sim['start_step']}", flush=True)
    print(f"marching t = {t0:g} -> {t1:g}  ({int(round((t1 - t0)/dt))} steps of "
          f"dt = {dt:g}), saving every {mjump} steps (= {mjump*dt:g} time "
          f"units) to {sim['save_path']}", flush=True)

    sim["time_stepper"].solve(t0, t1, mjump, sim["q0"],
                              save_path=sim["save_path"],
                              start_step=sim["start_step"])
    print("\nmarch finished; check convergence with "
          "`python check_convergence.py`")


if __name__ == "__main__":
    main()
