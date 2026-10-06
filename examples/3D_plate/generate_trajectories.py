"""March the nine 3D-plate training impulses on the full-order model.

The 3D analogue of examples/airfoil/generate_trajectories.py.  Each trajectory
is an unforced run from q(0) = q_b + beta * B_f, with q_b the converged steady
base flow, B_f one of the three unit-W-norm Leray-projected profiles written
by make_initial_conditions.py, and beta in {0.01, 0.1, 1}.

What is stored is the PERTURBATION q(t) - q_b on the full grid.  A plate state
has ~45M DOFs (0.36 GB in float64), so snapshots are streamed straight to a
memory-mapped .npy instead of being held in memory, by default in float32,
and TIME-MAJOR: fom_data/fluct_KKK.npy has shape (n_snapshots, n_dof)
(transpose of the airfoil's (n_dof, n_snapshots)).  With the defaults
(T = 40, every 0.2: 200 snapshots) that is ~36 GB per trajectory.

The time step is the solver's own AB2 (bootstrapped with one RK2 step, as in
TimeStepper.solve), repeated here so snapshots go to the memmap rather than to
per-snapshot .npz/.vtr files.  Trajectories that exist are skipped; ``--only``
splits the nine runs across jobs.

Usage (incompreso venv, GPU):
    python generate_trajectories.py --baseflow data/snapshot_NNNNNN.npz
                                    [--t-final 40] [--dt-save 0.2]
                                    [--dtype float32] [--only K ...]
Writes <out>/fluct_KKK.npy, energy_KKK.npy and meta.npz; <out> defaults to
fom_data_wing/ (wing, config.yaml) or fom_data/ (plate, config_plate.yaml).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from incompreso import parse_input_file
from incompreso.backend import GPU, to_backend, to_numpy, xp

from case import DEFAULT_CONFIG, Case
from make_initial_conditions import cell_volume_weights, forcing_dir, make_profiles

HERE = os.path.dirname(os.path.abspath(__file__))


def march_ab2(tstep, q0, qb, n_steps, mjump, out, w):
    """AB2 as in TimeStepper.solve for a fixed body (first step RK2).  Writes
    the perturbation q(t) - qb at steps 0, mjump, 2*mjump, ... into out[k]
    (host array or memmap); returns the W-energy of each saved state."""
    spops, dt, enforce = tstep.spatial_operators, tstep.dt, tstep.enforce_constraints
    w_dev = to_backend(w)
    q = enforce(0.0, q0.copy())
    rhs, rhs_prev = xp.zeros_like(q), xp.zeros_like(q)
    energy = np.empty(out.shape[0])
    t_start = time.time()
    for i in range(n_steps):
        if i % mjump == 0:
            k = i // mjump
            if xp.isnan(q).any() or float(xp.max(xp.abs(q))) > 1e4:
                raise ValueError(f"blow up detected at t = {i*dt:g}")
            dq = q - qb
            energy[k] = float(xp.dot(w_dev*dq, dq))
            out[k] = to_numpy(dq).astype(out.dtype, copy=False)
            print(f"    t = {i*dt:7.2f}  E = {energy[k]:.4e}  "
                  f"[{time.time() - t_start:.0f} s]", flush=True)
        t = i*dt
        if i == 0:
            rhs[:] = spops.evaluate_right_hand_side(t, q)
            rhs_prev[:] = rhs
            qs1 = enforce(t + dt/2, q + dt/2*rhs)
            rhs[:] = spops.evaluate_right_hand_side(t + dt/2, qs1)
            q += dt*rhs
        else:
            rhs[:] = spops.evaluate_right_hand_side(t, q)
            q += dt*(1.5*rhs - 0.5*rhs_prev)
            rhs_prev[:] = rhs
        q[:] = enforce(t + dt, q)
    return energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="case config (wing: config.yaml, plate: config_plate.yaml)")
    ap.add_argument("--baseflow", required=True,
                    help="converged steady snapshot q_b (e.g. data/snapshot_NNNNNN.npz)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: fom_data_wing/ or fom_data/ for the plate)")
    ap.add_argument("--t-final", type=float, default=40.0)
    ap.add_argument("--dt-save", type=float, default=0.2)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--only", type=int, nargs="+", default=None,
                    help="trajectory indices to run (default: all nine)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--z-center", type=float, default=0.0,
                    help="spanwise centre of the impulses (0: z-symmetric); off-centre "
                         "impulses and trajectories go to <forcing>_z<zc>/, <out>_z<zc>/")
    args = ap.parse_args()
    C = Case(args.config)
    forcing = forcing_dir(C, args.z_center)
    args.out = args.out or (C.fom_data if args.z_center == 0.0
                            else f"{C.fom_data}_z{args.z_center:g}")

    sim = parse_input_file(args.config)
    tstep = sim["time_stepper"]
    ib = tstep.immersed_body
    if tstep.scheme != "AB2":
        raise SystemExit(f"this script repeats the AB2 step; config uses {tstep.scheme}")
    if ib is None or ib.is_body_moving:
        raise SystemExit("expected a stationary immersed body")

    dt = tstep.dt
    mjump = int(round(args.dt_save/dt))
    n_steps = int(round(args.t_final/dt))
    if abs(mjump*dt - args.dt_save) > 1e-12 or n_steps % mjump:
        raise SystemExit(f"dt = {dt:g} does not divide dt_save = {args.dt_save:g} "
                         f"and t_final = {args.t_final:g}")
    tsave = args.dt_save*np.arange(n_steps // mjump)

    mesh = tstep.spatial_operators.mesh
    with np.load(args.baseflow) as snap:
        qb_host = snap["q"].astype(np.float64)
        same_mesh = all(np.array_equal(snap[c], to_numpy(getattr(mesh, c)))
                        for c in ("xu", "yu", "zu", "xv", "yv", "zv", "xw", "yw", "zw"))
    if not same_mesh:
        raise SystemExit(f"{args.baseflow} was computed on a different mesh")
    qb = to_backend(qb_host)
    res = tstep.spatial_operators.evaluate_right_hand_side(0.0, qb)
    step = to_numpy(tstep.enforce_constraints(dt, qb + dt*res)) - qb_host
    print(f"backend: {'GPU (CuPy)' if GPU else 'CPU (NumPy)'}")
    print(f"base flow {os.path.relpath(args.baseflow, HERE)}: ||q_b|| = "
          f"{np.linalg.norm(qb_host):.6f}, ||dq/dt||/||q_b|| ~ "
          f"{np.linalg.norm(step)/dt/np.linalg.norm(qb_host):.2e} (one Euler step)")

    if not os.path.exists(os.path.join(forcing, "Bf_profiles.npy")):
        print(f"\n{os.path.basename(forcing)}/Bf_profiles.npy missing: building the "
              f"impulses on this setup (make_initial_conditions.make_profiles)\n")
        make_profiles(sim, C, z_center=args.z_center)
    Bf = np.load(os.path.join(forcing, "Bf_profiles.npy"))
    stations = np.load(os.path.join(forcing, "stations.npy"))
    betas = np.load(os.path.join(forcing, "betas.npy"))
    if Bf.shape[0] != qb_host.size:
        raise SystemExit("forcing/Bf_profiles.npy does not match the mesh; "
                         "rerun make_initial_conditions.py")

    # Station-major order: trajectory k = 3*station + beta index.
    params = np.array([(beta, *stations[j]) for j in range(len(stations)) for beta in betas])
    which = range(len(params)) if args.only is None else args.only
    w = cell_volume_weights(mesh)

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "meta.npz"), time=tsave, parameters=params,
             baseflow=os.path.abspath(args.baseflow), dt=dt, mjump=mjump,
             layout="time-major (n_snapshots, n_dof)", dtype=args.dtype,
             z_center=args.z_center, forcing=os.path.abspath(forcing))
    gb = len(tsave)*qb_host.size*np.dtype(args.dtype).itemsize/1e9
    print(f"{n_steps} steps of dt = {dt:g} per trajectory, saving {len(tsave)} "
          f"snapshots at t = 0, {args.dt_save:g}, ..., {tsave[-1]:g} "
          f"({gb:.1f} GB per trajectory, {args.dtype})\n")

    for k in which:
        fname = os.path.join(args.out, f"fluct_{k:03d}.npy")
        beta, x0, y0 = params[k]
        j = k // len(betas)
        if os.path.exists(fname) and not args.overwrite:
            print(f"[{k}] {fname} exists, skipping")
            continue
        t0 = time.time()
        tmp = fname + ".tmp.npy"
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=args.dtype,
                                       shape=(len(tsave), qb_host.size))
        E = march_ab2(tstep, to_backend(qb_host + beta*Bf[:, j]), qb, n_steps, mjump, mm, w)
        mm.flush()
        del mm
        os.replace(tmp, fname)
        np.save(os.path.join(args.out, f"energy_{k:03d}.npy"), E)
        print(f"[{k}] beta = {beta:g} at ({x0:+.4f}, {y0:+.4f}): E(0) = {E[0]:.3e}, "
              f"max E = {E.max():.3e} at t = {tsave[E.argmax()]:g}, E(end) = {E[-1]:.3e}  "
              f"[{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
