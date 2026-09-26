"""
GasNiTROM for the cavity: the NiTROM trajectory cost minimized over GAS-
constrained latent dynamics and the projection bases.

    J = sum_j (1/alpha_j) sum_i || q^(j)(t_i) - Phi (Psi^T Phi)^-1 z^(j)(t_i) ||^2

with z solving dz/dt = A z + H(z, z), where (A, H) are not free but assembled
from GAS parameters (K, R, Q, S),

    A = ((K - K^T) - R^-1 R^-T) Qtil,   H_ijk = (S_ilk - S_lik) Qtil_lj,

so every iterate is globally asymptotically stable by construction.  Phi moves
on Grassmann and Psi on Stiefel, as in train_nitrom_bopinf.py.

Why this combination.  On this basis the two ingredients fail separately and for
opposite reasons: the exact Petrov-Galerkin operators (train_reprojection.py)
have a well-damped linear part but a quadratic term that pumps energy, so they
diverge at large amplitude; and GAS-OpInf contains that divergence but, because
it minimizes the *one-step residual*, converges back toward those same operators
and inherits their bias (state error 1.05 retracted -> 1.30 at 100 epochs ->
2.13 at 1000).  GasNiTROM replaces that objective with the trajectory error the
model is actually judged on, while keeping the stability guarantee -- and unlike
plain NiTROM it cannot wander to an unstable A on the way.

Initialization is the saved GAS-OpInf model (opinf_gas_r<r>.npy), using its
stored GAS parameters directly so nothing is lost to a re-retraction.

y = q here (section 5 observes the whole state), so C = I: `FOM` returns the
state unchanged and supplies `apply_output_adjoint`, which lets NitromModule
apply C^T without forming the 19800 x 19800 identity.

Run serially:
    python train_gasnitrom_bopinf.py --r 30 --iters 200
or trajectory-parallel (at most one rank per trajectory, so 7 here):
    mpiexec -n 7 python train_gasnitrom_bopinf.py --r 30 --iters 200

Cost and gradient norm print every PRINT_EVERY iterations, on rank 0.
"""

import argparse
import os
import shutil

import numpy as np

from balanced_opinf import ROM
from train_gas_opinf import tensor_to_sym
from train_nitrom_bopinf import FOM, Data, printr, gcost
from train_pod_opinf import forecast
from train_models import load_train, HERE
from compute_preprojection import load_train_reduced, load_preproj

from nitrom.backend import mpi_rank_size, set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.optimization import NitromModule, train
from nitrom.projections import LinearProjection
from nitrom.roms.param_registry import ParamRegistry

set_backend("numpy")
RANK, WORLD = mpi_rank_size()

R = 30
DEGREE = 2
N_SUBSTEPS = 5
N_EPOCHS = 200
PRINT_EVERY = 5
REG = 0.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="GasNiTROM refinement of the GAS-OpInf model for the cavity.")
    ap.add_argument("--r", type=int, default=R)
    ap.add_argument("--basis", choices=["balanced", "pod"], default="balanced",
                    help="which GAS-OpInf model to start from "
                         "(opinf_gas_<basis>_r<r>.npy)")
    ap.add_argument("--pre", type=int, default=0,
                    help="train in the span of this many POD pre-projection "
                         "modes.  Phi and Psi then live in R^r_pre, so every "
                         "iterate is divergence free by construction; the saved "
                         "model carries both the reduced factors and their lift "
                         "to R^19800, and evaluation uses the full FOM.")
    ap.add_argument("--iters", type=int, default=N_EPOCHS)
    ap.add_argument("--n-substeps", type=int, default=N_SUBSTEPS)
    ap.add_argument("--freeze-bases", action="store_true",
                    help="optimize only the GAS parameters, holding Phi and Psi "
                         "fixed")
    ap.add_argument("--optimizer", choices=["lbfgs", "adam", "sgd"],
                    default="lbfgs",
                    help="search-direction rule; the step always comes from a "
                         "strong-Wolfe line search.  adam rescales each "
                         "coordinate by its RMS gradient, which directly "
                         "targets the block mismatch in the GAS "
                         "parameterization: Q's relative gradient is ~10x "
                         "that of K, R and S, so one scalar L-BFGS step size "
                         "cannot suit both.")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any saved GasNiTROM model and restart from the "
                         "GAS-OpInf one")
    return ap.parse_args(argv)


def main(args=None):
    args = parse_args() if args is None else args
    r = args.r

    if args.pre:
        red, full, betas, time, U = load_train_reduced(args.pre)
        X = np.stack(red)                 # the cost is minimized in this space
        X_full = np.stack(full)           # only used for the final scoring
        printr(f"training in the {args.pre}-mode pre-projected space "
               f"(state dimension {X.shape[1]}); Phi and Psi are therefore "
               f"divergence free by construction")
    else:
        trajs_n, betas, time = load_train()
        X = np.stack([abs(b)*T for b, T in zip(betas, trajs_n)])
        X_full, U = X, None
    n_traj_global, _, n_snap = X.shape
    if WORLD > n_traj_global:
        raise ValueError(f"{WORLD} ranks for {n_traj_global} trajectories; "
                         f"use at most one rank per trajectory.")
    mine = list(range(RANK, n_traj_global, WORLD))
    printr(f"rank layout: {WORLD} rank(s), "
           f"{[len(range(k, n_traj_global, WORLD)) for k in range(WORLD)]} "
           f"trajectories each")

    tag = "" if args.basis == "balanced" else f"_{args.basis}"
    tag += f"_pre{args.pre}" if args.pre else ""
    out_path = os.path.join(HERE, f"roms_r{r}_gasnitrom{tag}.npy")
    resuming = not args.fresh and os.path.exists(out_path)
    if resuming:
        if RANK == 0:
            shutil.copyfile(out_path, out_path + ".prev")
        d = np.load(out_path, allow_pickle=True).item()
        printr(f"resuming from {os.path.basename(out_path)} "
               f"({d.get('n_iters', 0)} iterations so far; previous -> .prev)")
    else:
        pre_tag = f"_pre{args.pre}" if args.pre else ""
        src = os.path.join(HERE, f"opinf_gas_{args.basis}{pre_tag}_r{r}.npy")
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"{os.path.basename(src)} not found -- run "
                f"python train_gas_opinf.py --r {r} --basis {args.basis} "
                + (f"--pre {args.pre} " if args.pre else "") + "--seed opinf")
        d = np.load(src, allow_pickle=True).item()
        printr(f"starting from the GAS-OpInf model ({d['epochs']} epochs, "
               f"derivs = {d.get('derivs', 'fd')}, seed = {d.get('seed', '?')}), "
               f"state error {d['err']:.4f}, max Re eig(A) {d['max_re_eig']:+.6f}")
    if "gas_params" not in d:
        raise KeyError("no gas_params in the checkpoint -- re-run "
                       "train_gas_opinf.py so they are saved")

    # in pre-projected mode the OPTIMIZED bases are the reduced ones
    kphi, kpsi = ("Phi_red", "Psi_red") if args.pre else ("Phi", "Psi")
    Phi = np.ascontiguousarray(d[kphi])
    Psi = np.ascontiguousarray(d[kpsi])
    gas_init = [np.ascontiguousarray(q) for q in d["gas_params"]]

    model = GasPolynomialModel(r, [1, DEGREE], dtype=np.float64,
                               gas_params=gas_init)
    A0, H0 = [np.asarray(t) for t in model.assemble_gas_tensors()]
    printr(f"assembled from the stored GAS parameters: max Re eig(A) = "
           f"{np.linalg.eigvals(A0).real.max():+.6f}, "
           f"||A|| = {np.linalg.norm(A0):.2f}, ||H|| = {np.linalg.norm(H0):.2e}")
    if not resuming:
        # the assembled tensors must reproduce the saved ones exactly
        printr(f"  round-trip check vs the saved tensors: "
               f"|dA| = {np.abs(A0 - np.asarray(d['Ar_dense'])).max():.2e}, "
               f"|dH| = {np.abs(H0 - np.asarray(d['Hr_dense'])).max():.2e}")

    # alpha_j = time-averaged output energy; with y = q that is the state
    # energy.  Taken from the FULL trajectories so the cost keeps the same scale
    # as the non-pre-projected runs (the two differ by ~1e-7 here anyway).
    alpha = np.mean(np.sum(X_full**2, axis=1), axis=1)
    weights = alpha*n_traj_global*n_snap
    data = Data(X[mine], time, weights[mine])

    projection = LinearProjection([Phi, Psi])
    registry = ParamRegistry(model, projection)
    gasnit = NitromModule(data, registry, fom=FOM(), reg=REG,
                          n_substeps=args.n_substeps,
                          adjoint_method="discrete")
    if args.freeze_bases:
        gasnit.set_unlearnable("Phi", "Psi")
    else:
        gasnit.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])
    printr(f"trainable: {[n for n, v in zip(registry.names, gasnit.is_learnable.values()) if v]}"
           f"  frozen: {[n for n, v in zip(registry.names, gasnit.is_learnable.values()) if not v]}")
    printr(f"training set: {n_traj_global} trajectories x {n_snap} snapshots, "
           f"{args.n_substeps} substeps")
    printr(f"initial cost: {gcost(gasnit):.6e}")

    train(gasnit, n_epochs=args.iters, lr=1.0, optimizer_type=args.optimizer,
          print_every=PRINT_EVERY, tol=1e-12)

    printr(f"final cost: {gcost(gasnit):.6e}")
    gasnit._sync_to_registry()
    A, H = [np.asarray(t) for t in model.assemble_gas_tensors()]
    ev = np.linalg.eigvals(A).real.max()
    printr(f"max Re eig(A) = {ev:+.6f}  (negative by construction)")

    if RANK == 0:
        Ph = np.asarray(projection.Phi)
        Ps = np.asarray(projection.Psi)
        Ph_red, Ps_red = Ph, Ps
        # Score in the SAME space the cost was minimized in: the r_pre-dimensional
        # pre-projected system.  Lifting to R^19800 and comparing against the raw
        # Navier-Stokes solution would add the pre-projection error (3.5e-7 here)
        # to every model equally, which is not what the optimizer is being judged
        # on.  The lifted Phi and Psi are still saved for plotting.
        dec = Ph @ np.linalg.inv(Ps.T @ Ph)     # NiTROM iterates are not biorthogonal
        J, err, nbad = forecast(ROM(A, tensor_to_sym(H, r), 2), dec, Ps,
                                [X[j] for j in range(n_traj_global)], time)
        if args.pre:
            Ph, Ps = U @ Ph, U @ Ps             # lift only for the saved file
        where = (f"in the {args.pre}-dim pre-projected space" if args.pre
                 else "in the full space")
        print(f"forecast on the training impulses ({where}): J = {J:.4e}, "
              f"rel state error = {err:.4f}"
              + (f", {nbad}/{n_traj_global} diverged" if nbad else ""))
        prev = d.get("n_iters", 0) if resuming else 0
        np.save(out_path, dict(
            Phi=Ph, Psi=Ps, Phi_red=Ph_red, Psi_red=Ps_red,
            Ar=A, Hr=H, reg=REG, r=r, err=err, J=J, r_pre=args.pre,
            max_re_eig=ev, n_iters=prev + args.iters,
            gas_params=[np.asarray(q) for q in model.get_params()]),
            allow_pickle=True)
        print(f"saved -> {out_path}  (cumulative iterations: {prev + args.iters})")


if __name__ == "__main__":
    main()
