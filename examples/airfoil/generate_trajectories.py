"""March the nine airfoil training impulses of section 5.1 on the full-order model.

Each trajectory is an unforced run from q(0) = q_b + beta * B_f, with q_b the
converged base flow, B_f one of the three unit-norm Leray-projected Gaussians
written by ``make_initial_conditions.py`` and beta in {0.1, 1.0, 2.0}.  The
paper saves 200 snapshots at spacing 0.2, i.e. t = 0, 0.2, ..., 39.8 -- the
same convention as the cavity data, whose first snapshot is the initial
condition.

What is stored is the full-grid PERTURBATION q(t) - q_b, one array of shape
(n, 200) per trajectory.  Cropping to the ROM domain, weighting and the POD
pre-projection all happen in ``compute_preprojection.py``, so those choices
can be revisited without re-running the FOM.

``TimeStepper.solve`` writes an .npz and a .vtr for every snapshot, which for
1800 snapshots of 674k DOFs is mostly wasted disk, so the RK2 step is repeated
here and snapshots are collected in memory instead.  The step itself is the
solver's own (``evaluate_right_hand_side`` plus ``enforce_constraints``); the
loop is only valid for a stationary body, which is checked.

Runs on either incompreso backend; on the GPU only one snapshot per save
crosses to the host.  Trajectories that already exist are skipped, so an
interrupted job can simply be resubmitted, and ``--only`` splits the nine runs
across jobs.

Usage (needs the incompreso venv, with NiTROM on PYTHONPATH):
    PYTHONPATH=../../src ../../../incompreso/.venv/bin/python \
        generate_trajectories.py [--baseflow bflowdata/snapshot_000020.npz]
Writes fom_data/fluct_%03d.npy and fom_data/meta.npz.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from incompreso import parse_input_file
from incompreso.backend import GPU, to_backend, to_numpy, xp

from make_initial_conditions import cell_volume_weights

HERE = os.path.dirname(os.path.abspath(__file__))

T_FINAL = 40.0          # 200 snapshots at spacing 0.2
DT_SAVE = 0.2


def march(tstep, q0, n_steps, mjump):
    """RK2 exactly as ``TimeStepper.solve`` does it for a fixed body.

    Returns the host array of states at steps 0, mjump, 2*mjump, ...
    (n_steps must be a multiple of mjump; the final step is not saved).
    """
    spops, dt = tstep.spatial_operators, tstep.dt
    n_save = n_steps // mjump
    out = np.empty((q0.shape[0], n_save))

    q = tstep.enforce_constraints(0.0, q0.copy())
    qs1, rhs = xp.zeros_like(q), xp.zeros_like(q)
    for i in range(n_steps):
        if i % mjump == 0:
            if xp.isnan(q).any() or float(xp.max(xp.abs(q))) > 1e4:
                raise ValueError(f"blow up detected at t = {i*dt:g}")
            out[:, i // mjump] = to_numpy(q)
        t, t_mid = i*dt, (i + 0.5)*dt
        rhs[:] = spops.evaluate_right_hand_side(t, q)
        qs1[:] = q + dt/2*rhs
        qs1[:] = tstep.enforce_constraints(t_mid, qs1)
        rhs[:] = spops.evaluate_right_hand_side(t_mid, qs1)
        q += dt*rhs
        q[:] = tstep.enforce_constraints((i + 1)*dt, q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--baseflow",
                    default=os.path.join(HERE, "bflowdata", "snapshot_000020.npz"),
                    help="converged steady state q_b (see check_convergence.py)")
    ap.add_argument("--out", default=os.path.join(HERE, "fom_data"))
    ap.add_argument("--only", type=int, nargs="+", default=None,
                    help="trajectory indices to run (default: all nine)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sim = parse_input_file(args.config)
    tstep = sim["time_stepper"]
    ib = tstep.immersed_body
    if tstep.scheme != "RK2":
        raise SystemExit(f"this script repeats the RK2 step; config uses "
                         f"{tstep.scheme}")
    if ib is None or ib.is_body_moving:
        raise SystemExit("expected a stationary immersed body")

    dt = tstep.dt
    mjump = int(round(DT_SAVE/dt))
    n_steps = int(round(T_FINAL/dt))
    if abs(mjump*dt - DT_SAVE) > 1e-12 or n_steps % mjump:
        raise SystemExit(f"dt = {dt:g} does not divide the save spacing "
                         f"{DT_SAVE:g} and final time {T_FINAL:g}")
    tsave = DT_SAVE*np.arange(n_steps // mjump)

    # The base flow must be a steady state of THIS discretisation, not just
    # of the same config: check the mesh and that q_b is (nearly) stationary.
    with np.load(args.baseflow) as snap:
        qb_host = snap["q"].astype(np.float64)
        same_mesh = all(np.array_equal(snap[c], to_numpy(getattr(tstep.spatial_operators.mesh, c)))
                        for c in ("xu", "yu", "xv", "yv"))
    if not same_mesh:
        raise SystemExit(f"{args.baseflow} was computed on a different mesh")
    qb = to_backend(qb_host)
    res = to_numpy(tstep.spatial_operators.evaluate_right_hand_side(0.0, qb))
    # the RHS of a steady state is a pure gradient that the projection
    # removes, so test the projected one-step change instead
    step = to_numpy(tstep.enforce_constraints(dt, qb + dt*to_backend(res))) - qb_host
    print(f"backend: {'GPU (CuPy)' if GPU else 'CPU (NumPy)'}")
    print(f"base flow {os.path.relpath(args.baseflow, HERE)}: "
          f"||q_b|| = {np.linalg.norm(qb_host):.6f}, "
          f"||dq/dt|| ~ {np.linalg.norm(step)/dt:.2e} (one Euler step)")

    Bf = np.load(os.path.join(HERE, "forcing", "Bf_profiles.npy"))
    stations = np.load(os.path.join(HERE, "forcing", "stations.npy"))
    betas = np.load(os.path.join(HERE, "forcing", "betas.npy"))
    if Bf.shape[0] != qb_host.size:
        raise SystemExit("forcing/Bf_profiles.npy does not match the mesh; "
                         "rerun make_initial_conditions.py")

    # Station-major order: trajectory k = 3*station + beta index.
    params = np.array([(beta, *stations[j])
                       for j in range(len(stations)) for beta in betas])
    which = range(len(params)) if args.only is None else args.only

    # Paper, figure 10(b): E_pert = <q, q>_W with W = V / V_min.
    w = cell_volume_weights(tstep.spatial_operators.mesh)

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "meta.npz"), time=tsave, parameters=params,
             baseflow=os.path.abspath(args.baseflow), dt=dt, mjump=mjump)
    print(f"{n_steps} steps of dt = {dt:g} per trajectory, saving "
          f"{len(tsave)} snapshots at t = 0, {DT_SAVE:g}, ..., {tsave[-1]:g}\n")

    for k in which:
        fname = os.path.join(args.out, f"fluct_{k:03d}.npy")
        beta, x0, y0 = params[k]
        j = k // len(betas)
        if os.path.exists(fname) and not args.overwrite:
            print(f"[{k}] {fname} exists, skipping")
            continue

        t0 = time.time()
        q0 = to_backend(qb_host + beta*Bf[:, j])
        Q = march(tstep, q0, n_steps, mjump)
        Q -= qb_host[:, None]
        E = np.einsum("it,i,it->t", Q, w, Q)

        tmp = fname + ".tmp.npy"
        np.save(tmp, Q)
        os.replace(tmp, fname)
        print(f"[{k}] beta = {beta:g} at ({x0:+.4f}, {y0:+.4f}): "
              f"E(0) = {E[0]:.3e}, max E = {E.max():.3e} at t = "
              f"{tsave[E.argmax()]:g}, E(end) = {E[-1]:.3e}  "
              f"[{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
