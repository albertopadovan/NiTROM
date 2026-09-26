"""GasNiTROM for the airfoil in balanced coordinates.

Minimizes the NiTROM trajectory cost over the GAS parameters and the bases,

    J = sum_j (1/alpha_j) sum_i || q~_j(t_i) - Phi (Psi^T Phi)^-1 z_j(t_i) ||^2,

on the whole training window (t in [0, 40]) from the first iteration, with the
same setup as the cavity's train_gasnitrom_bopinf.py: Phi on Grassmann, Psi on
Stiefel, discrete adjoint, y = q~.

Data: the nine FOM trajectories scaled by sqrt(W), q~ = W^{1/2} q, on the
mesh the balancing used (balancing_full/ by default: the whole mesh, as the
re-projected OpInf models use), with the mask and sqrt(W) read from its
balancing.npz.  alpha_j is the time-averaged energy <||q~_j||^2>.

Initialization (--init):
  gas_opinf  models/gas_opinf_balanced_r<r>.pkl -- its Phi, Psi and GAS
             parameters, used as they are;
  opinf      models/opinf_balanced_r<r>.pkl -- its Phi, Psi and (A, H),
             retracted onto the GAS manifold first in the Lyapunov metric
             (exact in A for a Hurwitz seed; P = I would force monotone
             energy decay and wipe out the transient growth).
Progress is saved every --save-every iterations to
models/gas_nitrom_balanced_from_<init>_r<r>.pkl, and a rerun resumes from that
file unless --fresh is given.

Coordinate descent (--alternate N): N iterations on the operators with the
bases frozen, then N on the bases with the operators frozen, and so on.

Run serially:
    python train_gasnitrom_balanced.py --init opinf --r 50 --iters 1000
or trajectory-parallel (at most one rank per trajectory, so up to 9):
    mpiexec -n 9 python train_gasnitrom_balanced.py --init opinf --r 50 --iters 1000
"""

from __future__ import annotations

import argparse
import os
import pickle
import time as timer

import numpy as np

from train_gas_opinf_balanced import forecast

from nitrom.backend import mpi_allreduce_scalar, mpi_rank_size, set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.optimization import NitromModule, train
from nitrom.projections import LinearProjection
from nitrom.roms.param_registry import ParamRegistry

set_backend("numpy")
RANK, WORLD = mpi_rank_size()
HERE = os.path.dirname(os.path.abspath(__file__))
POLY_COMP = [1, 2]


def printr(*a, **k):
    if RANK == 0:
        print(*a, **k, flush=True)


def gcost(module):
    """Globally reduced cost: each rank's forward() is a partial sum."""
    c = float(module())
    return mpi_allreduce_scalar(c) if WORLD > 1 else c


class FOM:
    """y = q~: the output is the state, and its adjoint is the identity, so
    NitromModule never has to form an N x N output Jacobian.  output_is_state
    also lets it evaluate the mismatch in Gram form, without ever storing the
    (ntraj, N, nt) reconstruction or error."""

    output_is_state = True

    def compute_output(self, q):
        return q

    def compute_output_derivative(self, q):          # pragma: no cover
        raise NotImplementedError("y = q~: use apply_output_adjoint")

    def apply_output_adjoint(self, e, q):
        return e


def orthonormalize_psi(Psi, model):
    r"""Equivalent GAS model whose test basis has orthonormal columns.

    The Stiefel optimizer maps its starting point onto the manifold with a
    sign-fixed QR, and z = Psi^T q~, so a non-orthonormal Psi would silently
    change the latent coordinates under unchanged operators (the cost jumped
    from 0.643 to 0.933 at iteration 0).  As in the cavity: with the same
    sign-fixed QR Psi = Psi_o R, the new coordinates are z' = T z, T = R^-T,
    and the model moves with them, A' = T A T^-1, H'(z',z') = T H(T^-1 z',
    T^-1 z').  For the GAS parameterization A = ((K - K^T) - R~) Q~,
    H[:, :, k] = (S_k - S_k^T) Q~ that is exactly
        K' = T K T^T,  R' = R T^-1,  Q' = Q T^T,
        S'[:, :, k'] = sum_k T^-1[k, k'] T S[:, :, k] T^T,
    so the model -- and the cost -- are unchanged.  Phi needs nothing: the
    decoder Phi (Psi^T Phi)^-1 absorbs T.

    :returns: ``(Psi_o, gas_params')``
    """
    Psi_o, Rq = np.linalg.qr(Psi)
    sg = np.sign(np.diag(Rq))
    sg[sg == 0] = 1.0
    Psi_o, Rq = Psi_o*sg[None, :], Rq*sg[:, None]      # the optimizer's QR
    T = np.linalg.inv(Rq).T
    Ti = np.linalg.inv(T)
    names = list(model.param_names)
    if names[:4] != ["K", "R", "Q", "S"]:
        raise ValueError(f"unexpected GAS parameters {names}")
    K, R, Q, S = [np.array(t) for t in model.get_params()[:4]]
    S_new = np.einsum("kc,ab,bdk,ed->aec", Ti, T, S, T, optimize=True)
    return Psi_o, [T @ K @ T.T, R @ Ti, Q @ T.T, S_new] + \
        [np.array(t) for t in model.get_params()[4:]]


class Data:
    """Minimal TrainingData view (X, time, weights, forcing_fns): no
    time-derivative files, which NiTROM never uses."""

    def __init__(self, X, time, weights):
        self.X, self.time, self.weights = X, time, weights
        self.forcing_fns = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=50)
    ap.add_argument("--init", choices=["opinf", "gas_opinf"], default="opinf",
                    help="starting model: OpInf retracted onto the GAS "
                         "manifold, or the trained GAS-OpInf model")
    ap.add_argument("--iters", type=int, default=1000,
                    help="L-BFGS iterations in this run")
    ap.add_argument("--save-every", type=int, default=50,
                    help="checkpoint interval (iterations)")
    ap.add_argument("--n-substeps", type=int, default=15,
                    help="ROM time steps between two snapshots")
    ap.add_argument("--alternate", type=int, default=0,
                    help="coordinate descent: alternate blocks of this many "
                         "iterations on the operators (GAS parameters, bases "
                         "frozen) and on the bases (Phi, Psi, operators "
                         "frozen), operators first; a checkpoint is saved "
                         "after every block.  0 = joint optimization")
    ap.add_argument("--reg", type=float, default=0.0)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore a saved GasNiTROM model; restart from GAS-OpInf")
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data"))
    ap.add_argument("--balancing", default=os.path.join(HERE, "balancing_full"))
    ap.add_argument("--models", default=os.path.join(HERE, "models"))
    args = ap.parse_args()
    r = args.r

    # --- data: q~ = sqrt(W) q on the ROM domain, this rank's trajectories --
    bal = np.load(os.path.join(args.balancing, "balancing.npz"), allow_pickle=True)
    mask, sqrt_w = bal["mask"], bal["sqrt_w"]
    red = np.load(os.path.join(args.balancing,
                               f"reduced_r{bal['Phi'].shape[1]}.npz"))
    time, alpha = red["time"], red["energy"]
    n_traj, n_snap = len(alpha), len(time)
    if WORLD > n_traj:
        raise SystemExit(f"{WORLD} ranks for {n_traj} trajectories")
    mine = list(range(RANK, n_traj, WORLD))
    t0 = timer.time()
    X = np.stack([np.load(os.path.join(args.data, f"fluct_{k:03d}.npy"),
                          mmap_mode="r")[mask]*sqrt_w[:, None] for k in mine])
    printr(f"{n_traj} trajectories x {n_snap} snapshots, N = {X.shape[1]}, "
           f"{WORLD} rank(s); rank 0 loaded {len(mine)} in "
           f"{timer.time() - t0:.0f} s")
    # TrainingData's convention: the cost averages over trajectories and
    # snapshots, each trajectory normalized by its time-averaged energy.
    data = Data(X, time, alpha[mine]*n_traj*n_snap)

    # --- initialization -----------------------------------------------------
    out_path = os.path.join(args.models,
                            f"gas_nitrom_balanced_from_{args.init}_r{r}.pkl")
    src_path = os.path.join(args.models, f"{args.init}_balanced_r{r}.pkl")
    resuming = not args.fresh and os.path.exists(out_path)
    with open(out_path if resuming else src_path, "rb") as f:
        ck = pickle.load(f)
    n_done = ck.get("n_iters", 0) if resuming else 0
    history = ck.get("history", dict(loss=[], gradnorm=[])) if resuming else \
        dict(loss=[], gradnorm=[])
    printr(f"starting from {os.path.basename(out_path if resuming else src_path)}"
           + (f" ({n_done} iterations so far)" if resuming else ""))

    if resuming or args.init == "gas_opinf":
        gas_params = [np.array(t) for t in ck["gas_params"]]
    else:
        A0, H0 = [np.array(t) for t in ck["tensors"]]
        seed = GasPolynomialModel(r, POLY_COMP, dtype=np.float64)
        seed.retract_general_tensors_to_gas_tensors([A0, H0], use_P_I=False)
        gas_params = [np.array(t) for t in seed.get_params()]
        A1 = np.asarray(seed.assemble_gas_tensors()[0])
        printr(f"OpInf retracted onto the GAS manifold: max Re eig(A) "
               f"{np.linalg.eigvals(A0).real.max():+.4f} -> "
               f"{np.linalg.eigvals(A1).real.max():+.4f}, ||A - A_opinf||/"
               f"||A_opinf|| = {np.linalg.norm(A1 - A0)/np.linalg.norm(A0):.3f}")
    model = GasPolynomialModel(r, POLY_COMP, dtype=np.float64,
                               gas_params=gas_params)
    Psi = np.ascontiguousarray(ck["Psi"])
    orth_err = np.abs(Psi.T @ Psi - np.eye(r)).max()
    if orth_err > 1e-10:
        A0, H0 = [np.array(t) for t in model.assemble_gas_tensors()]
        Psi, gas_params = orthonormalize_psi(Psi, model)
        model = GasPolynomialModel(r, POLY_COMP, dtype=np.float64,
                                   gas_params=gas_params)
        A1, H1 = [np.array(t) for t in model.assemble_gas_tensors()]
        # check against the dense transformation of the old tensors
        _, Rq = np.linalg.qr(np.ascontiguousarray(ck["Psi"]))
        Rq = Rq*np.where(np.diag(Rq) < 0, -1.0, 1.0)[:, None]
        T = np.linalg.inv(Rq).T
        Ti = np.linalg.inv(T)
        A_ref = T @ A0 @ Ti
        H_ref = np.einsum("ai,ijk,jb,kc->abc", T, H0, Ti, Ti, optimize=True)
        printr(f"orthonormalized Psi (|Psi^T Psi - I| was {orth_err:.1e}); "
               f"GAS model transformed exactly: |A' - T A T^-1|/|A'| = "
               f"{np.abs(A1 - A_ref).max()/np.abs(A_ref).max():.1e}, "
               f"|H' - T H(T^-1., T^-1.)|/|H'| = "
               f"{np.abs(H1 - H_ref).max()/np.abs(H_ref).max():.1e}")
    projection = LinearProjection([np.ascontiguousarray(ck["Phi"]), Psi])
    registry = ParamRegistry(model, projection)
    gasnit = NitromModule(data, registry, fom=FOM(), reg=args.reg,
                          n_substeps=args.n_substeps, adjoint_method="discrete")
    gasnit.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])
    printr(f"initial cost: {gcost(gasnit):.6e}")

    def save(n_iters, t_elapsed):
        gasnit._sync_to_registry()
        if RANK != 0:
            return
        A, H = [np.array(t) for t in model.model.get_params()]
        ckpt = dict(kind="gas_nitrom", init=args.init, r=r, poly_comp=POLY_COMP,
                    forcing_config=None, basis="balanced",
                    Phi=np.array(projection.Phi), Psi=np.array(projection.Psi),
                    tensors=[A, H],
                    gas_params=[np.array(t) for t in model.get_params()],
                    reg=args.reg, n_iters=n_iters, history=history,
                    time=ck.get("time", 0.0) + t_elapsed if resuming else t_elapsed)
        tmp = out_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(ckpt, f)
        os.replace(tmp, out_path)
        print(f"saved -> {os.path.relpath(out_path, HERE)} "
              f"({n_iters} iterations)", flush=True)

    # --- train in chunks so progress is never lost --------------------------
    bases = ["Phi", "Psi"]
    ops = [p for p in registry.names if p not in bases]
    chunk = args.alternate if args.alternate > 0 else args.save_every
    t_start = timer.perf_counter()
    left, block = args.iters, 0
    while left > 0:
        n = min(chunk, left)
        if args.alternate > 0:
            on_ops = block % 2 == 0
            gasnit.set_learnable(*(ops if on_ops else bases))
            gasnit.set_unlearnable(*(bases if on_ops else ops))
            printr(f"--- block {block + 1}: {n} iterations on the "
                   f"{'operators ' + str(ops) if on_ops else 'bases (Phi, Psi)'}")
            block += 1
        train(gasnit, n_epochs=n, lr=1.0, optimizer_type="lbfgs",
              print_every=1, tol=1e-14)
        history["loss"].extend(gasnit.loss_history)
        history["gradnorm"].extend(gasnit.gradnorm_history)
        n_done += n
        left -= n
        save(n_done, timer.perf_counter() - t_start)

    # --- forecast the training impulses with the final model ----------------
    A, H = [np.array(t) for t in model.model.get_params()]
    Phi, Psi = np.array(projection.Phi), np.array(projection.Psi)
    dec = Phi @ np.linalg.inv(Psi.T @ Phi)        # iterates are not biorthogonal
    printr(f"final cost {gcost(gasnit):.6e}; max Re eig(A) = "
           f"{np.linalg.eigvals(A).real.max():+.6f}")
    for i, k in enumerate(mine):
        Z = forecast(A, H, Psi.T @ X[i][:, 0], time)
        err = (np.linalg.norm(X[i] - dec @ Z)/np.linalg.norm(X[i])
               if np.all(np.isfinite(Z)) else np.inf)
        print(f"  traj {k}: relative forecast error {err:.4f}", flush=True)


if __name__ == "__main__":
    main()
