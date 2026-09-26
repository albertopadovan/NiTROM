"""Finely sampled start of the training trajectories, merged with the rest.

The balancing needs the fast transient right after the impulse, which the
0.2-spaced training snapshots under-resolve (the energy drops ~5x between
t = 0 and the first snapshot at 0.2).  This re-marches every training
trajectory of fom_data/ from the same initial condition q_b + beta B_f over
[0, T_fine], saving every DT_FINE, and splices those samples in front of the
existing coarse ones:

    t = 0, 0.02, ..., T_fine - 0.02  (new)  +  T_fine, ..., 39.8  (existing)

As a consistency check the fine run also saves t = T_fine, which must agree
with the existing snapshot there.  fom_data/ itself is left untouched (NiTROM
and OpInf keep the uniform 0.2 grid); the merged, non-uniformly sampled
trajectories go to fom_data_merged/ in the same layout.

Usage (incompreso venv, NiTROM importable):
    INCOMPRESO_BACKEND=gpu python generate_fine_segment.py [--t-fine 0.2]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from incompreso import parse_input_file
from incompreso.backend import GPU, to_backend

from generate_trajectories import march

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data"))
    ap.add_argument("--out", default=os.path.join(HERE, "fom_data_merged"))
    ap.add_argument("--t-fine", type=float, default=0.2,
                    help="end of the finely sampled segment")
    ap.add_argument("--dt-fine", type=float, default=0.02)
    args = ap.parse_args()

    meta = np.load(os.path.join(args.data, "meta.npz"))
    t_coarse, params = meta["time"], meta["parameters"]
    qb = np.load(str(meta["baseflow"]))["q"].astype(np.float64)
    Bf = np.load(os.path.join(HERE, "forcing", "Bf_profiles.npy"))
    betas = np.load(os.path.join(HERE, "forcing", "betas.npy"))

    sim = parse_input_file(args.config)
    tstep = sim["time_stepper"]
    dt = tstep.dt
    if abs(dt - float(meta["dt"])) > 1e-15:
        raise SystemExit(f"config dt {dt} differs from the data's {meta['dt']}")
    mjump = int(round(args.dt_fine/dt))
    n_fine = int(round(args.t_fine/args.dt_fine))       # samples before T_fine
    i_coarse = int(np.argmin(np.abs(t_coarse - args.t_fine)))
    if abs(mjump*dt - args.dt_fine) > 1e-12 or \
            abs(t_coarse[i_coarse] - args.t_fine) > 1e-12:
        raise SystemExit("T_fine / dt_fine must be multiples of dt and T_fine "
                         "a coarse snapshot time")
    t_fine = args.dt_fine*np.arange(n_fine)
    t_merged = np.concatenate([t_fine, t_coarse[i_coarse:]])

    os.makedirs(args.out, exist_ok=True)
    print(f"backend: {'GPU' if GPU else 'CPU'}; fine segment t = 0..{args.t_fine:g} "
          f"every {args.dt_fine:g} ({n_fine} new samples); merged grid has "
          f"{len(t_merged)} snapshots")
    print(f"{'traj':>4}{'beta':>6}{'|dq(0)|/|q(0)|':>16}"
          f"{f'|dq({args.t_fine:g})|/|q|':>16}{'time':>8}")
    for k in range(len(params)):
        beta = params[k, 0]
        j = k // len(betas)
        Qc = np.load(os.path.join(args.data, f"fluct_{k:03d}.npy"), mmap_mode="r")
        t0 = time.time()
        # one sample past the segment, at T_fine, to check against the data
        Qf = march(tstep, to_backend(qb + beta*Bf[:, j]),
                   (n_fine + 1)*mjump, mjump)
        Qf -= qb[:, None]
        e0 = np.linalg.norm(Qf[:, 0] - Qc[:, 0])/np.linalg.norm(Qc[:, 0])
        e1 = (np.linalg.norm(Qf[:, n_fine] - Qc[:, i_coarse])
              / np.linalg.norm(Qc[:, i_coarse]))
        print(f"{k:>4}{beta:>6g}{e0:>16.1e}{e1:>16.1e}{time.time()-t0:>7.0f}s",
              flush=True)
        if max(e0, e1) > 1e-10:
            raise SystemExit(f"trajectory {k}: the fine run does not reproduce "
                             f"the stored one; not merging")
        merged = np.concatenate([Qf[:, :n_fine], Qc[:, i_coarse:]], axis=1)
        tmp = os.path.join(args.out, f"fluct_{k:03d}.tmp.npy")
        np.save(tmp, merged)
        os.replace(tmp, os.path.join(args.out, f"fluct_{k:03d}.npy"))

    np.savez(os.path.join(args.out, "meta.npz"), time=t_merged,
             parameters=params, baseflow=str(meta["baseflow"]), dt=dt,
             t_fine=args.t_fine, dt_fine=args.dt_fine, dt_coarse=0.2,
             source=os.path.abspath(args.data))
    print(f"saved -> {os.path.relpath(args.out, HERE)}/ "
          f"({len(params)} trajectories x {len(t_merged)} snapshots)")


if __name__ == "__main__":
    main()
