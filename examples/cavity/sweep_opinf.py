"""
Sweep the OpInf regularization for the cavity on both bases at a given r.

For each lambda: fit dz/dt = A_r z + H_r (z x z), forecast all seven training
impulse responses over the full window, and report

  maxRe(Ar)  largest real part of eig(A_r).  The cavity at Re = 8300 is linearly
             stable, so a positive value means the fit invented growth -- the
             instability reported in section 5.1 of the paper.
  J          NiTROM cost (3.7), each trajectory normalized by its time-averaged
             energy
  err        relative state error over all trajectories
  proj       projection error of the training data onto the basis: the floor no
             ROM on that basis can beat

The winner per basis is saved to opinf_<basis>_r<r>.npy.

Usage: python sweep_opinf.py [--r 30] [--basis both|pod|balanced]
"""

import argparse
import os

import numpy as np

from balanced_opinf import opinf
from compute_preprojection import load_train_reduced, pod_basis_reduced
from train_pod_opinf import get_pod, forecast, REG_GRID
from train_models import load_train, HERE


def forecast_from(rom, Phi, Psi, trajs, time, i0):
    """Forecast starting at snapshot i0 instead of t = 0.

    Dropping the first snapshots from the FIT still leaves the model judged on
    a window it never saw; restarting the forecast at the same instant asks the
    fairer question -- how well does it propagate a state that lies inside the
    regime it was fitted on?
    """
    J = num = den = 0.0
    nbad = 0
    for X in trajs:
        Xw = X[:, i0:]
        Z = rom.integrate(Psi.T @ Xw[:, 0], time[:Xw.shape[1]], 1.0)
        den += np.linalg.norm(Xw)**2
        if not np.all(np.isfinite(Z)):
            nbad += 1
            continue
        Xh = Phi @ Z
        J += np.sum((Xw - Xh)**2)/np.mean(np.sum(Xw**2, axis=0))
        num += np.linalg.norm(Xw - Xh)**2
    if nbad:
        return np.inf, np.inf, nbad
    return J, np.sqrt(num/den), 0


def sweep(name, Phi, Psi, trajs, time, dt, den, trajs_fit=None, i0=0,
          Psi_fit=None):
    proj = np.sqrt(sum(np.linalg.norm(X - Phi @ (Psi.T @ X))**2
                       for X in trajs)/den)
    R1, R2 = np.linalg.qr(Phi, mode="r"), np.linalg.qr(Psi, mode="r")
    print(f"\n=== {name}, r = {Phi.shape[1]} ===")
    print(f"projection error {proj:.4f}   ||Phi Psi^T||_2 = "
          f"{np.linalg.norm(R1 @ R2.T, 2):.2f}")
    print(f"{'lambda':>9} {'maxRe(Ar)':>11} {'||Ar||':>9} {'||Hr||':>10} "
          f"{'J (3.7)':>12} {'err':>9}"
          + (f" {'err|t>=t0':>9}" if i0 else "") + "  status")
    trajs_fit = trajs if trajs_fit is None else trajs_fit
    Psi_fit = Psi if Psi_fit is None else Psi_fit   # encoder in the FIT space
    best = None
    for reg in REG_GRID:
        try:
            rom = opinf(trajs_fit, Psi_fit, dt, "quadratic", reg=reg, C=None)
        except np.linalg.LinAlgError as exc:      # one bad lambda must not kill the sweep
            print(f"{reg:9.0e} {'least squares failed: ' + str(exc):>55}")
            continue
        J, err, nbad = forecast(rom, Phi, Psi, trajs, time)
        ev = np.linalg.eigvals(rom.Ar).real.max()
        st = "ok" if nbad == 0 else f"{nbad}/{len(trajs)} diverged"
        Js = f"{J:12.4e}" if np.isfinite(J) else f"{'inf':>12}"
        es = f"{err:9.4f}" if np.isfinite(err) else f"{'-':>9}"
        extra = ""
        if i0:
            _, err2, nb2 = forecast_from(rom, Phi, Psi, trajs, time, i0)
            extra = (f" {err2:9.4f}" if np.isfinite(err2) else f" {'-':>9}")
        print(f"{reg:9.0e} {ev:+11.4f} {np.linalg.norm(rom.Ar):9.2f} "
              f"{np.linalg.norm(rom.Hr):10.2e} {Js} {es}{extra}  {st}")
        if np.isfinite(J) and (best is None or J < best[0]):
            best = (J, reg, rom, err)
    if best is None:
        print(f"-> {name}: every regularization diverged; nothing saved.")
        return None
    J, reg, rom, err = best
    print(f"-> {name}: best lambda = {reg:.0e}, J = {J:.4e}, err = {err:.4f} "
          f"(projection floor {proj:.4f})")
    return dict(Phi=Phi, Psi=Psi, Ar=rom.Ar, Hr=rom.Hr, reg=reg, J=J, err=err,
                proj=proj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--basis", choices=["both", "pod", "balanced"],
                    default="both")
    ap.add_argument("--pre", type=int, default=0,
                    help="fit in the span of this many POD pre-projection modes "
                         "(divergence-free by construction).  Models are still "
                         "EVALUATED against the full Navier-Stokes solution, so "
                         "the pre-projection error is charged to the model.")
    ap.add_argument("--t-min", type=float, default=0.0,
                    help="drop snapshots before this time from the FIT.  The "
                         "early transient is where the oblique projection is "
                         "worst (the balanced basis reconstructs the impulse "
                         "IC with 190%% error), so those samples may be "
                         "poisoning the least squares.")
    args = ap.parse_args()
    r = args.r

    if args.pre:
        red, trajs, betas, time, U = load_train_reduced(args.pre)
        print(f"fitting in the {args.pre}-mode pre-projected space, "
              f"evaluating against the full FOM")
    else:
        trajs_n, betas, time = load_train()
        trajs = [abs(b)*T for b, T in zip(betas, trajs_n)]
        red, U = trajs, None
    dt = time[1] - time[0]
    den = sum(np.linalg.norm(X)**2 for X in trajs)
    i0 = int(np.searchsorted(time, args.t_min)) if args.t_min > 0 else 0
    trajs_fit = [X[:, i0:] for X in trajs] if i0 else trajs
    if i0:
        print(f"fitting on t >= {time[i0]:g} only: {trajs_fit[0].shape[1]} of "
              f"{trajs[0].shape[1]} snapshots per trajectory\n"
              f"('err' is still the forecast from t = 0 over the full window; "
              f"'err|t>=t0' restarts at t = {time[i0]:g})")

    want = ["balanced", "pod"] if args.basis == "both" else [args.basis]
    for name in want:
        if args.pre:
            if name == "balanced":
                bf = os.path.join(HERE, f"balancing_pre{args.pre}_r50.npy")
                d = np.load(bf, allow_pickle=True).item()
                Phi_f, Psi_f = d["Phi"][:, :r], d["Psi"][:, :r]
                Phi_r, Psi_r = d["Phi_red"][:, :r], d["Psi_red"][:, :r]
            else:
                Phi_r = Psi_r = pod_basis_reduced(r, args.pre)
                Phi_f = Psi_f = U[:, :r]        # U e_i is the i-th POD mode
        elif name == "balanced":
            d = np.load(os.path.join(HERE, "balancing_r50.npy"),
                        allow_pickle=True).item()
            Phi_f, Psi_f = d["Phi"][:, :r], d["Psi"][:, :r]
            Phi_r, Psi_r = Phi_f, Psi_f
        else:
            Phi_f = Psi_f = Phi_r = Psi_r = get_pod(trajs, r)
        # fit on the reduced data, score against the full trajectories
        fit_src = [X[:, i0:] for X in red] if i0 else red
        out = sweep(name, Phi_f, Psi_f, trajs, time, dt, den, fit_src, i0,
                    Psi_fit=Psi_r)
        if out is not None:
            out.update(Phi_red=Phi_r, Psi_red=Psi_r, r_pre=args.pre)
            tag = f"_pre{args.pre}" if args.pre else ""
            np.save(os.path.join(HERE, f"opinf_{name}{tag}_r{r}.npy"), out,
                    allow_pickle=True)
            print(f"   saved -> opinf_{name}{tag}_r{r}.npy")


if __name__ == "__main__":
    main()
