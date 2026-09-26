"""How close the marched airfoil flow is to a genuine steady state.

``run.py`` cannot tell you when to stop, because the march has no built-in
residual monitor.  This reads the snapshots it wrote and reports

    ||dq/dt|| ~ ||q_k - q_{k-1}|| / (t_k - t_{k-1}),

together with the relative form ||dq/dt|| / ||q||, which is the number that
matters: the training impulses have amplitudes beta in {0.1, 1, 2} against a
base flow of norm ~572, so a base flow still drifting at 1e-3 relative would
contaminate the smallest impulse responses outright.

It also extrapolates: the residual decays exponentially once the transient is
gone, so fitting the last few points gives an estimate of how much further the
march has to run.

Usage: python check_convergence.py [--data data/] [--target 1e-6]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="snapshot directory; default is the config's save_path")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--target", type=float, default=1e-6,
                    help="residual ||dq/dt|| you want to reach")
    ap.add_argument("--fit-last", type=int, default=5,
                    help="how many trailing points to fit the decay rate on")
    args = ap.parse_args()

    # Follow the config's save_path unless told otherwise, so this keeps
    # working when that is retargeted.  Read as plain YAML: no incompreso
    # import, so this runs outside the solver's venv.
    data = args.data
    if data is None:
        import yaml
        with open(args.config) as fh:
            data = yaml.safe_load(fh).get("output", {}).get("save_path", "data/")
        if not os.path.isabs(data):
            data = os.path.join(HERE, data)

    files = sorted(glob.glob(os.path.join(data, "snapshot_*.npz")))
    if len(files) < 2:
        raise SystemExit(f"need at least 2 snapshots in {data}, found "
                         f"{len(files)}")

    ts, res, nrm = [], [], []
    prev_q = prev_t = None
    print(f"{'file':>24}{'t':>9}{'||q||':>14}{'||dq/dt||':>13}{'rel':>11}")
    for f in files:
        d = np.load(f)
        q, t = np.asarray(d["q"], dtype=np.float64), float(d["t"])
        n = np.linalg.norm(q)
        if prev_q is not None and t > prev_t:
            r = np.linalg.norm(q - prev_q)/(t - prev_t)
            ts.append(t)
            res.append(r)
            nrm.append(n)
            print(f"{os.path.basename(f):>24}{t:>9.2f}{n:>14.6f}"
                  f"{r:>13.3e}{r/n:>11.2e}")
        else:
            print(f"{os.path.basename(f):>24}{t:>9.2f}{n:>14.6f}"
                  f"{'-':>13}{'-':>11}")
        prev_q, prev_t = q, t

    ts, res = np.asarray(ts), np.asarray(res)
    if len(ts) < 2:
        return
    print(f"\ncurrent residual {res[-1]:.3e} "
          f"({res[-1]/nrm[-1]:.2e} relative) at t = {ts[-1]:g}")
    if res[-1] <= args.target:
        print(f"converged: already at or below the {args.target:g} target.")
        return

    # Exponential fit on the trailing points: res ~ C exp(-lambda t).
    k = min(args.fit_last, len(ts))
    if k >= 2 and np.all(res[-k:] > 0):
        lam, logC = np.polyfit(ts[-k:], np.log(res[-k:]), 1)
        lam = -lam
        if lam > 0:
            t_need = (logC - np.log(args.target))/lam
            print(f"decay rate from the last {k} points: "
                  f"lambda = {lam:.4f} per time unit")
            print(f"extrapolated: residual {args.target:g} at t ~ {t_need:.0f}, "
                  f"i.e. {t_need - ts[-1]:.0f} more time units to march")
        else:
            print(f"the residual is NOT decaying over the last {k} points "
                  f"(fitted rate {lam:+.4f}); the march is not converging.")


if __name__ == "__main__":
    main()
