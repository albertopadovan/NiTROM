"""
A learned initialization map, so the trimmed balanced-OpInf ROM can be used as a
predictor from the initial condition alone.

The problem this solves.  Fitting the ROM on t >= TAU only (train_gas_opinf /
sweep_opinf --t-min) makes the linear operator stable and the model accurate --
restarted at t = TAU from the true state it reaches a relative error of ~0.3.
But started from z(0) = Psi^T q(0) it diverges, because the balanced basis
misrepresents the impulse initial condition (||Phi Psi^T q0 - q0||/||q0|| ~ 1.9)
and because t < TAU was excluded from the fit, so z(0) lies outside the regime
the quadratic term was ever constrained on.

The fix.  Do not ask the ROM to reproduce the early transient at all.  Instead
learn, directly from data, where the initial condition *ends up*:

    beta = B^T q(0)          (every impulse here is q(0) = beta B, ||B|| = 1)
    z(TAU) ~ sum_p c_p beta^p,   p = 1..P

fitted by least squares over the training trajectories.  Then a prediction for a
new beta is: evaluate the map, and integrate the ROM from t = TAU.  The ROM is
never time-stepped while fitting -- this is a regression on snapshots, so it
stays Operator-Inference-like rather than becoming NiTROM.

Early error is accepted by construction: nothing is claimed for t < TAU.  The
question is whether the trajectory shadows the FOM afterwards.

Three initializations are compared on t >= TAU:
    naive     z = Psi^T q(0), integrated from t = 0 (what fails today)
    learned   z = sum_p c_p beta^p                 (this script)
    oracle    z = Psi^T q(TAU), the true state     (the unreachable best case)

Usage: python train_encoder_map.py [--r 30] [--tau 3] [--degree 3] [--reg-trim 2e-2]
"""

import argparse
import os

import numpy as np

from cavity import Cavity
from balanced_opinf import opinf
from train_models import load_train, HERE, DATA


def fit_map(betas, Z_tau, degree):
    """Least-squares coefficients of z(TAU) ~ sum_p c_p beta^p, p = 1..degree.

    No constant term: beta = 0 must map to z = 0, since a zero impulse leaves
    the flow on the base state.

    :returns: ``C`` of shape ``(r, degree)``
    """
    V = np.stack([betas**p for p in range(1, degree + 1)], axis=1)   # (n, degree)
    return np.linalg.lstsq(V, Z_tau.T, rcond=None)[0].T              # (r, degree)


def apply_map(C, beta):
    degree = C.shape[1]
    return C @ np.array([beta**p for p in range(1, degree + 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--reg-trim", type=float, default=2e-2)
    args = ap.parse_args()
    r = args.r

    trajs_n, betas, time = load_train()
    trajs = [abs(b)*T for b, T in zip(betas, trajs_n)]
    dt = time[1] - time[0]
    i0 = int(np.searchsorted(time, args.tau))
    tau = time[i0]

    d = np.load(os.path.join(HERE, "balancing_r50.npy"), allow_pickle=True).item()
    Phi, Psi = d["Phi"][:, :r], d["Psi"][:, :r]
    rom = opinf([X[:, i0:] for X in trajs], Psi, dt, "quadratic",
                reg=args.reg_trim, C=None)
    print(f"r = {r}, tau = {tau:g}, ROM fitted on t >= {tau:g} "
          f"(lambda = {args.reg_trim:.0e}), "
          f"max Re eig(Ar) = {np.linalg.eigvals(rom.Ar).real.max():+.4f}")

    # --- fit the initialization map on the training trajectories only
    Z_tau = np.stack([Psi.T @ X[:, i0] for X in trajs], axis=1)      # (r, n)
    C = fit_map(np.asarray(betas, dtype=float), Z_tau, args.degree)
    resid = np.linalg.norm(Z_tau - np.stack([apply_map(C, b) for b in betas], 1))
    print(f"initialization map: degree {args.degree} in beta, "
          f"{len(betas)} training samples, "
          f"relative fit residual {resid/np.linalg.norm(Z_tau):.3e}")

    cav = Cavity()
    B = cav.B

    def evaluate(tag, beta_list, traj_list):
        print(f"\n{tag}")
        print(f"{'beta':>8}{'naive (t>=tau)':>18}{'learned':>12}{'oracle':>12}")
        agg = {k: [0.0, 0.0] for k in ("naive", "learned", "oracle")}
        for b, X in zip(beta_list, traj_list):
            Xw, tw = X[:, i0:], time[:X.shape[1] - i0]
            row = f"{b:>8.3f}"
            for key, z0, t_grid, cmp in (
                # naive: integrate from t = 0, then compare only on t >= tau
                ("naive", Psi.T @ X[:, 0], time[:X.shape[1]], "tail"),
                ("learned", apply_map(C, float(np.dot(B, X[:, 0]))), tw, "all"),
                ("oracle", Psi.T @ Xw[:, 0], tw, "all"),
            ):
                Z = rom.integrate(z0, t_grid, 1.0)
                ok = np.all(np.isfinite(Z), axis=0)
                if not ok.all():
                    row += f"{'diverged':>18}" if key == "naive" else f"{'diverged':>12}"
                    agg[key] = None if agg[key] is None else None
                    continue
                Xh = Phi @ Z
                Xh = Xh[:, i0:] if cmp == "tail" else Xh
                e = np.linalg.norm(Xw - Xh)/np.linalg.norm(Xw)
                row += (f"{e:>18.4f}" if key == "naive" else f"{e:>12.4f}")
                if agg[key] is not None:
                    agg[key][0] += np.linalg.norm(Xw - Xh)**2
                    agg[key][1] += np.linalg.norm(Xw)**2
            print(row)
        out = {}
        for k, v in agg.items():
            out[k] = np.sqrt(v[0]/v[1]) if v else np.inf
        print(f"{'aggregate':>8}" + "".join(
            f"{out[k]:>18.4f}" if k == "naive" and np.isfinite(out[k])
            else f"{'diverged':>18}" if k == "naive"
            else f"{out[k]:>12.4f}" if np.isfinite(out[k]) else f"{'diverged':>12}"
            for k in ("naive", "learned", "oracle")))
        return out

    evaluate("TRAINING impulses (the map was fitted on these)", betas, trajs)

    bt = np.load(os.path.join(DATA, "setup.npz"))["betas_test"]
    tt = [abs(b)*np.load(os.path.join(DATA, f"test_traj_{j:03d}.npy")).astype(np.float64)
          for j, b in enumerate(bt)]
    evaluate("HELD-OUT impulses (the map has never seen these beta)", bt, tt)


if __name__ == "__main__":
    main()
