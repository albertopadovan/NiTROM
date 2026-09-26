"""
Extra sinusoidally forced FOM runs at a chosen amplitude.

generate_data_bopinf.py produces the paper's Fig. 9 cases at w_amp = 0.1; this
script regenerates the same (omega, k) pairs at any amplitude, so the ROMs can
be pushed outside the regime the original forced data covers.  Zero initial
condition, w(t) = amp sin(k omega t) through the actuator B.

Files are named forced_A<amp>_w<omega>_k<k>.npy.  The amplitude 0.1 set keeps
its original names (forced_w<omega>_k<k>.npy) so nothing already on disk is
overwritten.

Usage: python generate_forced.py --amp 0.3
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np

from cavity import Cavity, NSTEP, HERE

DATA = os.path.join(HERE, "data")
T, NSAVE = 40.0, NSTEP//10           # sampling interval 0.1, as in the rest


def forced(args):
    omega, k, amp = args
    cav = Cavity()
    _, Q = cav.integrate(np.zeros(cav.N), T, NSAVE,
                         forcing=lambda t: amp*np.sin(k*omega*t))
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", type=float, required=True)
    ap.add_argument("--omega", type=float, default=None,
                    help="generate a single (omega, k) case instead of the six "
                         "in setup.npz -- e.g. omega = 1, k = 1, which the "
                         "original set does not contain")
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    setup = np.load(os.path.join(DATA, "setup.npz"))
    if args.omega is not None or args.k is not None:
        if args.omega is None or args.k is None:
            ap.error("--omega and --k must be given together")
        cases = [(args.omega, args.k, args.amp)]
    else:
        cases = [(float(o), int(k), args.amp) for o, k in setup["forced"]]
    print(f"{len(cases)} forced runs at amplitude {args.amp:g}: "
          f"{[(o, k) for o, k, _ in cases]}", flush=True)
    with Pool(min(10, len(cases))) as pool:
        Qs = pool.map(forced, cases)
    for (o, k, _), Q in zip(cases, Qs):
        path = os.path.join(DATA, f"forced_A{args.amp:g}_w{o:g}_k{k}.npy")
        np.save(path, Q.astype(np.float32))
        print(f"  saved {os.path.basename(path)}  "
              f"peak ||x||^2 = {np.sum(Q**2, axis=0).max():.3e}")


if __name__ == "__main__":
    main()
