"""
GAS-OpInf on the Lall-balanced basis for the cavity.

Same Operator Inference least-squares problem as balanced_opinf.opinf, but the
latent tensors are not free: they are parameterized so that the ROM is globally
asymptotically stable *by construction*,

    A = ((K - K^T) - R^-1 R^-T) Qtil,   H_ijk = (S_ilk - S_lik) Qtil_lj,
    Qtil = Q^-1 Q^-T,

with (K, R, Q, S) the free parameters (GasPolynomialModel).  That directly
targets the failure mode every unconstrained fit here has shown: lambda only
regularizes H_r, so nothing stops A_r from acquiring a positive eigenvalue, and
the balanced OpInf model at r = 30 has max Re eig = +0.0078 -- invisible over
the t <= 40 training window but divergent by t = 339.

Pipeline, following the GAS-OpInf recipe:
  1. project the training trajectories onto the balanced basis, z = Psi^T x,
     with dz/dt by 4th-order central differences (2 samples dropped per end);
  2. solve the unconstrained OpInf problem in closed form for a seed (A, H);
  3. retract that seed onto the GAS manifold to initialize (K, R, Q, S);
  4. minimize the OpInf residual over the GAS parameters with L-BFGS.

Per-trajectory weights are 1/<||x||^2>, the same time-averaged-energy
normalization used everywhere else here, so the fit is comparable to the
unconstrained one.

Usage: python train_gas_opinf.py [--r 30] [--epochs 2000] [--reg 0.0]
Saves opinf_gas_r<r>.npy.
"""

import argparse
import os
import shutil
import time as timer

import numpy as np

from balanced_opinf import ROM, ddt_fd4, poly_index
from train_nitrom_bopinf import sym_to_tensor
from train_pod_opinf import forecast
from train_models import load_train, HERE, DATA
from compute_preprojection import load_train_reduced, pod_basis_reduced

from nitrom.backend import set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import OpInfModule, solve_opinf, train
from nitrom.projections.linear_projection import LinearProjection
from nitrom.training_data import TrainingData, TrainingPool

set_backend("numpy")


def tensor_to_sym(T, r, degree=2):
    """Dense (r,)*(degree+1) tensor -> the (r, n_monomials) form ROM expects."""
    import itertools
    idx = poly_index(r, degree)
    Hs = np.zeros((r, len(idx)))
    for m, tup in enumerate(idx):
        for perm in set(itertools.permutations(tuple(tup))):
            Hs[:, m] += T[(slice(None),) + perm]
    return Hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--reg", type=float, default=0.0)
    ap.add_argument("--basis", choices=["balanced", "pod"], default="balanced")
    ap.add_argument("--pre", type=int, default=0,
                    help="fit in the span of this many POD pre-projection modes; "
                         "evaluation is still against the full FOM")
    ap.add_argument("--derivs", choices=["fd", "reproj"], default="fd",
                    help="how dz/dt for the OpInf residual is obtained: 4th-order "
                         "finite differences of z, or RE-PROJECTION, dz/dt = "
                         "Psi^T f(Phi z), which is exact.  With re-projected "
                         "derivatives the residual has a zero-residual solution "
                         "(the Petrov-Galerkin tensors), so minimizing it over "
                         "the GAS manifold asks for the best GAS approximation "
                         "of the exact projected dynamics -- a well-posed "
                         "target, unlike the noisy FD residual.")
    ap.add_argument("--optimizer", choices=["lbfgs", "adam", "sgd"],
                    default="lbfgs",
                    help="search-direction rule; the step always comes from a "
                         "strong-Wolfe line search.  adam rescales each "
                         "coordinate by its RMS gradient, which directly "
                         "targets the block mismatch in the GAS "
                         "parameterization: Q's relative gradient is ~10x "
                         "that of K, R and S, so one scalar L-BFGS step size "
                         "cannot suit both.")
    ap.add_argument("--seed", choices=["opinf", "reproj"], default="opinf",
                    help="where the initial (A, H) come from before retraction: "
                         "the closed-form OpInf fit, or the re-projected "
                         "(exact Petrov-Galerkin) tensors of "
                         "train_reprojection.py")
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
    if args.pre:
        if args.basis == "pod":
            Phi_r = Psi_r = pod_basis_reduced(r, args.pre)
            Phi, Psi = U[:, :r], U[:, :r]
            print(f"POD basis, r = {r} (in reduced coordinates it is the "
                  f"leading {r} unit vectors)")
        else:
            bf = os.path.join(HERE, f"balancing_pre{args.pre}_r50.npy")
            d = np.load(bf, allow_pickle=True).item()
            Phi, Psi = d["Phi"][:, :r], d["Psi"][:, :r]
            Phi_r, Psi_r = d["Phi_red"][:, :r], d["Psi_red"][:, :r]
            print(f"Lall-balanced basis, r = {r} ({d['m_ckpt']} checkpoints, "
                  f"stride {d['q_ckpt']}, pre-projected)")
    elif args.basis == "pod":
        from train_pod_opinf import get_pod
        Phi = Psi = Phi_r = Psi_r = get_pod(trajs, r)
        print(f"POD basis, r = {r} (orthogonal: Phi = Psi)")
    else:
        d = np.load(os.path.join(HERE, "balancing_r50.npy"),
                    allow_pickle=True).item()
        Phi, Psi = d["Phi"][:, :r], d["Psi"][:, :r]
        Phi_r, Psi_r = Phi, Psi
        print(f"Lall-balanced basis, r = {r} ({d['m_ckpt']} checkpoints, "
              f"stride {d['q_ckpt']})")

    # 1. reduced trajectories + finite-difference derivatives on disk, which is
    #    what TrainingPool reads.  The interior samples only (fd4 drops 2/end).
    red_trajs = red if args.pre else trajs          # what the residual is fit on
    red = os.path.join(DATA, f"reduced_gas_{args.basis}_pre{args.pre}_r{r}")
    shutil.rmtree(red, ignore_errors=True)
    os.makedirs(red)
    if args.derivs == "reproj":
        from cavity import Cavity, BC
        cav = Cavity()
        BC_PERT = [0]*8
        t0 = timer.time()
    fit_src = red_trajs if args.pre else trajs
    for j, (X, Xf) in enumerate(zip(fit_src, trajs)):
        Z = Psi_r.T @ X
        if args.derivs == "reproj":
            # Re-projection works in the pre-projected setting too: the state
            # Phi z is built in the FULL space (Phi is the lifted basis), the
            # FOM right-hand side is evaluated there, and the result is
            # projected back with the same lifted Psi.  Psi^T = Psi_red^T U^T,
            # so this is exactly Psi_red^T U^T f(U Phi_red z).
            Xr = Phi @ Z                       # re-projected states, full space
            F = Psi.T @ np.column_stack(
                [cav.fom.evaluate_fom_dynamics(Xr[:, k], BC, BC_PERT)
                 for k in range(Xr.shape[1])])   # Psi is the lifted basis
            Zs, Fs, tsave = Z, F, time         # no stencil, nothing dropped
        else:
            Zs, Fs, tsave = Z[:, 2:-2], ddt_fd4(Z, dt), time[2:-2]
        np.save(os.path.join(red, f"traj_{j:03d}.npy"), Zs)
        np.save(os.path.join(red, f"deriv_{j:03d}.npy"), Fs)
        # 1/<||x||^2>: the time-averaged-energy normalization used throughout
        np.save(os.path.join(red, f"weight_{j:03d}.npy"),
                np.array(1.0/np.mean(np.sum(Xf**2, axis=0))))
    np.save(os.path.join(red, "time.npy"), tsave)
    if args.derivs == "reproj":
        print(f"re-projected derivatives: {len(trajs)*len(time)} FOM "
              f"right-hand sides in {timer.time()-t0:.1f} s")

    pool = TrainingPool(n_traj=len(trajs),
                        fname_traj=os.path.join(red, "traj_%03d.npy"),
                        fname_time=os.path.join(red, "time.npy"),
                        dtype=np.float64,
                        fname_derivs=os.path.join(red, "deriv_%03d.npy"),
                        fname_weights=os.path.join(red, "weight_%03d.npy"))
    data = TrainingData(pool, which_trajs=list(range(len(trajs))),
                        percent_time_length=1.0, leggauss_deg=5, nsave_rom=1)
    identity = LinearProjection([np.eye(r), np.eye(r)])

    # 2. seed tensors before retraction
    if args.seed == "reproj":
        rp = os.path.join(HERE, f"opinf_reproj_{args.basis}"
                          + (f"_pre{args.pre}" if args.pre else "")
                          + f"_r{r}.npy")
        if not os.path.exists(rp):
            raise FileNotFoundError(
                f"{os.path.basename(rp)} not found -- run "
                f"python train_reprojection.py --r {r} --basis {args.basis} "
                + (f"--pre {args.pre} " if args.pre else "") + "--sweep")
        rd = np.load(rp, allow_pickle=True).item()
        # stored in the symmetric-monomial layout; the GAS retraction wants the
        # dense (r, r, r) tensor
        A0, H0 = np.asarray(rd["Ar"]), sym_to_tensor(np.asarray(rd["Hr"]), r)
        print(f"seed: re-projected (exact Petrov-Galerkin) tensors, "
              f"fit residual {rd['residual']:.2e}")
    else:
        seedmod = PolynomialModel(r, [1, 2], dtype=np.float64)
        solve_opinf(OpInfModule(data, seedmod, identity, reg=args.reg))
        A0, H0 = [np.asarray(p) for p in seedmod.get_params()[:2]]
        print("seed: closed-form unconstrained OpInf")
    ev0 = np.linalg.eigvals(A0).real.max()
    print(f"  max Re eig(A) = {ev0:+.4f}, "
          f"||A|| = {np.linalg.norm(A0):.2f}, ||H|| = {np.linalg.norm(H0):.2e}")

    # 3. retract onto the GAS manifold
    gas = GasPolynomialModel(r, [1, 2], dtype=np.float64)
    t0 = timer.time()
    gas.retract_general_tensors_to_gas_tensors([A0, H0], use_P_I=True)
    A1, H1 = [np.asarray(t) for t in gas.assemble_gas_tensors()]
    print(f"after retraction ({timer.time()-t0:.1f} s): max Re eig(A) = "
          f"{np.linalg.eigvals(A1).real.max():+.4f}, "
          f"||A - A_seed||/||A_seed|| = {np.linalg.norm(A1-A0)/np.linalg.norm(A0):.3f}")

    # 4. train the GAS parameters on the same OpInf residual
    module = OpInfModule(data, gas, identity, reg=args.reg)
    print(f"\ninitial OpInf residual: {float(module()):.6e}")
    train(module, n_epochs=args.epochs, lr=1.0, optimizer_type=args.optimizer,
          print_every=max(1, args.epochs//20), tol=1e-12)
    print(f"final OpInf residual:   {float(module()):.6e}")

    A, H = [np.asarray(t) for t in gas.assemble_gas_tensors()]
    ev = np.linalg.eigvals(A).real.max()
    print(f"\nGAS model: max Re eig(A) = {ev:+.6f}  "
          f"(negative by construction)")
    print(f"           ||A|| = {np.linalg.norm(A):.2f}, ||H|| = {np.linalg.norm(H):.2e}")

    rom = ROM(A, tensor_to_sym(H, r), 2)
    J, err, nbad = forecast(rom, Phi, Psi, trajs, time)
    print(f"forecast on the training impulses: J = {J:.4e}, "
          f"rel state error = {err:.4f}"
          + (f", {nbad}/{len(trajs)} diverged" if nbad else ""))

    tag = f"_pre{args.pre}" if args.pre else ""
    out = os.path.join(HERE, f"opinf_gas_{args.basis}{tag}_r{r}.npy")
    print(f"(seed = {args.seed})")
    # The GAS parameters themselves are saved, not just the assembled tensors:
    # train_gasnitrom_bopinf.py starts from them directly, so no retraction (and
    # no loss of the fit) is needed to hand the model over.
    np.save(out, dict(Phi=Phi, Psi=Psi, Ar=A, Hr=tensor_to_sym(H, r),
                      Ar_dense=A, Hr_dense=H, reg=args.reg, J=J, err=err,
                      max_re_eig=ev, epochs=args.epochs,
                      gas_params=[np.asarray(q) for q in gas.get_params()],
                      derivs=args.derivs, seed=args.seed, r_pre=args.pre,
                      Phi_red=Phi_r, Psi_red=Psi_r), allow_pickle=True)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
