"""Testing impulse responses for the airfoil ROMs.

Same construction as the training data (make_initial_conditions.py +
generate_trajectories.py): the equation-(41) Gaussian on u, its modified Leray
projection (linear part), scaled to unit W-norm, added as beta * B_f to the
converged base flow and marched unforced with the solver's RK2 to t = 39.8,
200 snapshots.  Two sets:

  A  the training station --known-station (default: midchord) at --n-known
     amplitudes drawn uniformly from (beta_min, beta_max) of the training set;
  B  new stations between midchord and trailing edge (--new-xc, chord
     fractions), at the same standoff above the upper surface as the paper's
     stations (y/c = y_surface + 0.0165; the paper's midchord and trailing-edge
     points sit 0.0168 and 0.016 above the NACA 0012 surface), each at
     --n-per-new random amplitudes.

Amplitudes use a fixed seed, so the set is reproducible.  As a check, the
profile of the known station is rebuilt from scratch and compared with the
saved training profile.

Usage (incompreso venv, NiTROM importable):
    INCOMPRESO_BACKEND=gpu python generate_testing_trajectories.py
Writes fom_data_testing/fluct_%03d.npy and fom_data_testing/meta.npz, the same
layout as fom_data/.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from incompreso import parse_input_file
from incompreso.backend import GPU, to_backend, to_numpy

from generate_trajectories import DT_SAVE, T_FINAL, march
from make_initial_conditions import cell_volume_weights, gaussian_profile
from plot_baseflow import AIRFOIL_ALPHA, FORCING_STATIONS

HERE = os.path.dirname(os.path.abspath(__file__))
STANDOFF = 0.0165


def naca_half_thickness(xc, t=0.12):
    """NACA 00xx half-thickness at chord fraction xc (closed trailing edge
    variant is not needed at these stations)."""
    return 5*t*(0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*xc**2
                + 0.2843*xc**3 - 0.1015*xc**4)


def body_to_physical(xc, yc):
    """(x/c, y/c) in the body frame -> physical coordinates: half-chord at the
    origin, rotated nose-up by the angle of attack (as forcing_locations)."""
    a = np.deg2rad(AIRFOIL_ALPHA)
    xb = xc - 0.5
    return xb*np.cos(a) + yc*np.sin(a), -xb*np.sin(a) + yc*np.cos(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--baseflow",
                    default=os.path.join(HERE, "bflowdata", "snapshot_000020.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "fom_data_testing"))
    ap.add_argument("--known-station", default="midchord",
                    choices=[s[0] for s in FORCING_STATIONS])
    ap.add_argument("--n-known", type=int, default=5)
    ap.add_argument("--new-xc", type=float, nargs="*", default=[0.65, 0.80])
    ap.add_argument("--n-per-new", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    train_betas = np.load(os.path.join(HERE, "forcing", "betas.npy"))
    b_lo, b_hi = float(train_betas.min()), float(train_betas.max())
    rng = np.random.default_rng(args.seed)

    # ---- the test cases: (set, label, x/c, y/c, beta) ---------------------
    cases = []
    xk, yk = {s[0]: (s[1], s[2]) for s in FORCING_STATIONS}[args.known_station]
    for b in rng.uniform(b_lo, b_hi, args.n_known):
        cases.append(("A", args.known_station, xk, yk, float(b)))
    for xc in args.new_xc:
        yc = float(naca_half_thickness(xc) + STANDOFF)
        for b in rng.uniform(b_lo, b_hi, args.n_per_new):
            cases.append(("B", f"x/c={xc:g}", xc, yc, float(b)))

    # ---- solver, base flow, Leray projection of a perturbation ------------
    sim = parse_input_file(args.config)
    tstep = sim["time_stepper"]
    ib, mesh = tstep.immersed_body, tstep.spatial_operators.mesh
    dt = tstep.dt
    mjump, n_steps = int(round(DT_SAVE/dt)), int(round(T_FINAL/dt))
    tsave = DT_SAVE*np.arange(n_steps // mjump)
    qb = np.load(args.baseflow)["q"].astype(np.float64)
    w = cell_volume_weights(mesh)
    P0 = to_numpy(ib.enforce_constraints(0.0, to_backend(np.zeros_like(qb))))

    def unit_profile(xc, yc):
        x0, y0 = body_to_physical(xc, yc)
        raw = gaussian_profile(mesh, x0, y0, "u")
        proj = to_numpy(ib.enforce_constraints(0.0, to_backend(raw))) - P0
        return proj/np.sqrt(np.sum(w*proj*proj)), (x0, y0)

    print(f"backend: {'GPU' if GPU else 'CPU'}; training amplitudes span "
          f"[{b_lo:g}, {b_hi:g}]")
    # check: the known station rebuilt here must match the saved training B_f
    Bf_train = np.load(os.path.join(HERE, "forcing", "Bf_profiles.npy"))
    j = [s[0] for s in FORCING_STATIONS].index(args.known_station)
    Bk, _ = unit_profile(xk, yk)
    diff = np.sqrt(np.sum(w*(Bk - Bf_train[:, j])**2))
    print(f"check: rebuilt {args.known_station} profile vs training B_f, "
          f"W-norm difference {diff:.1e}")
    if diff > 1e-8:
        raise SystemExit("profile construction does not reproduce the training "
                         "data; refusing to generate an inconsistent test set")

    os.makedirs(args.out, exist_ok=True)
    params = np.array([(c[4], *body_to_physical(c[2], c[3]), c[2], c[3])
                       for c in cases])       # beta, x0, y0, x/c, y/c
    np.savez(os.path.join(args.out, "meta.npz"), time=tsave, parameters=params,
             sets=np.array([c[0] for c in cases]),
             labels=np.array([c[1] for c in cases]),
             columns=np.array(["beta", "x0", "y0", "x_over_c", "y_over_c"]),
             baseflow=os.path.abspath(args.baseflow), dt=dt, mjump=mjump,
             seed=args.seed)
    print(f"{len(cases)} test trajectories, {n_steps} steps of dt = {dt:g}:")
    for k, (st, lab, xc, yc, b) in enumerate(cases):
        print(f"  [{k}] set {st}  {lab:>14}  (x/c, y/c) = ({xc:.3f}, {yc:.4f})"
              f"  beta = {b:.4f}")

    for k, (st, lab, xc, yc, beta) in enumerate(cases):
        fname = os.path.join(args.out, f"fluct_{k:03d}.npy")
        if os.path.exists(fname) and not args.overwrite:
            print(f"[{k}] exists, skipping")
            continue
        t0 = time.time()
        Bf, _ = (Bk, None) if (st == "A") else unit_profile(xc, yc)
        Q = march(tstep, to_backend(qb + beta*Bf), n_steps, mjump)
        Q -= qb[:, None]
        E = np.einsum("it,i,it->t", Q, w, Q)
        tmp = fname + ".tmp.npy"
        np.save(tmp, Q)
        os.replace(tmp, fname)
        print(f"[{k}] set {st} {lab}, beta = {beta:.4f}: E(0) = {E[0]:.3e}, "
              f"max E = {E.max():.3e} at t = {tsave[E.argmax()]:g}  "
              f"[{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
