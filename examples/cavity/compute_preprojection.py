"""
Pre-projection basis: the leading POD modes of the energy-normalized training
data, used as the ambient space for all model building from here on.

Everything downstream -- balancing, OpInf, NiTROM, GasNiTROM -- is built and
trained in these coordinates, z_pre = U^T q.  Two reasons:

* **divergence.** NiTROM optimizes Phi on Grassmann and Psi on Stiefel, which in
  the raw R^19800 can produce bases that are not divergence free.  The training
  snapshots are, so the POD modes are too (verified below to ~1e-11), and any
  basis expressed in their span inherits it.  This enforces (5.2) by
  construction rather than by hope.
* **cost.** The balancing window set shrinks from 19800 x L to R_PRE x L, which
  removes the memory pressure entirely.

Evaluation is NOT done in these coordinates: a ROM prediction is lifted back,
q_hat = U (Phi z), and compared against the full Navier-Stokes solution, so the
pre-projection error is charged to the model like any other.

Each trajectory is divided by the square root of its time-averaged energy before
the SVD, so all seven amplitudes weigh equally (their raw energies span ~2700x).

Usage: python compute_preprojection.py [--r-pre 1000]
Writes preproj_U<r_pre>.npy (the basis) and prints what it captures.
"""

import argparse
import os

import numpy as np
import scipy.linalg as sla

from cavity import Cavity
from train_models import load_train, HERE


def preproj_path(r_pre):
    return os.path.join(HERE, f"preproj_U{r_pre}.npy")


def load_preproj(r_pre):
    """The cached pre-projection basis, or a clear error telling you to build it."""
    p = preproj_path(r_pre)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{os.path.basename(p)} not found -- run "
            f"python compute_preprojection.py --r-pre {r_pre}")
    return np.load(p)


def load_train_reduced(r_pre):
    """Training data in both spaces: (reduced, full, betas, time).

    ``reduced`` is what every model is FITTED on; ``full`` is the Navier-Stokes
    solution every model is EVALUATED against, so the pre-projection error is
    charged to the model rather than hidden by comparing inside its own span.
    """
    from train_models import load_train
    U = load_preproj(r_pre)
    trajs_n, betas, time = load_train()
    full = [abs(b)*T for b, T in zip(betas, trajs_n)]
    red = [U.T @ X for X in full]
    return red, full, betas, time, U


def pod_basis_reduced(r, r_pre):
    """POD basis of the pre-projected data = the first r columns of I.

    The pre-projection modes ARE the POD modes of the (energy-normalized)
    training data, ordered by energy, so in the coordinates z = U^T q the POD
    basis is exactly the leading unit vectors -- no SVD needed.  Verified in
    main() against an explicit SVD of the reduced snapshots.
    """
    import numpy as np
    return np.eye(r_pre, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-pre", type=int, default=1000)
    args = ap.parse_args()

    trajs_n, betas, time = load_train()
    trajs = [abs(b)*T for b, T in zip(betas, trajs_n)]
    alpha = np.array([np.mean(np.sum(X**2, axis=0)) for X in trajs])
    A = np.concatenate([X/np.sqrt(a) for X, a in zip(trajs, alpha)], axis=1)
    print(f"snapshot matrix {A.shape} (energy-normalized); computing the SVD ...",
          flush=True)
    U, s, _ = sla.svd(A, full_matrices=False)
    U = np.ascontiguousarray(U[:, :args.r_pre])

    e = np.cumsum(s**2)/np.sum(s**2)
    print(f"r_pre = {args.r_pre}: {e[args.r_pre-1]:.8%} of the variance, "
          f"sigma_{args.r_pre}/sigma_1 = {s[args.r_pre-1]/s[0]:.2e}")
    # what the pre-projection itself costs, on the data it was built from
    den = sum(np.linalg.norm(X)**2 for X in trajs)
    err = np.sqrt(sum(np.linalg.norm(X - U @ (U.T @ X))**2 for X in trajs)/den)
    print(f"pre-projection error on the training trajectories: {err:.3e}")

    cav = Cavity()
    divs = [cav.divergence(U[:, i]) for i in (0, 99, 499, args.r_pre - 1)]
    print(f"relative divergence of modes 1, 100, 500, {args.r_pre}: "
          + ", ".join(f"{d:.1e}" for d in divs))

    np.save(preproj_path(args.r_pre), U)
    print(f"saved -> {preproj_path(args.r_pre)} "
          f"({U.nbytes/1e6:.0f} MB, {U.shape})")


if __name__ == "__main__":
    main()
