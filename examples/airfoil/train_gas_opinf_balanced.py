"""GAS-OpInf for the airfoil in balanced coordinates.

Same recipe as train_opinf.py (and the paper, section 5.1), with the POD basis
replaced by the data-driven balanced pair (Phi, Psi) of compute_balancing.py:

  1. latent trajectories z_j = Psi^T q~_j (precomputed by compute_balancing.py)
     with time derivatives either from nitrom's 4th-order finite differences
     (--derivs fd) or re-projected through the solver,
     dz/dt = Psi^T W^{1/2} P f(q_b + W^{-1/2} Phi z)  (--derivs reproj, the
     default; see compute_reprojected_derivatives.py, which must run first).
     Re-projected pairs are samples of the Petrov-Galerkin model itself, so
     the unregularized OpInf fit should be exact to round-off -- printed below
     as a check;
  2. closed-form OpInf (lambda = 1e-3 on H) as the seed;
  3. retraction of the seed onto the GAS manifold, by default in the Lyapunov
     metric A^T P + P A = -I (--retraction lyap).  For a Hurwitz seed that
     reproduces A exactly (only H is projected); --retraction identity fixes
     P = I, which forces sym(A) < 0 -- monotone energy decay, i.e. no
     transient growth -- and here damps the model to nothing;
  4. L-BFGS on the OpInf residual over the GAS parameters (lambda = 1.7e-5).

Each trajectory's residual is divided by its time-averaged energy <||q~||^2>
-- the pool's weight files hold that energy, and OpInfModule divides by it.

The model is then forecast from z_j(0) on every training impulse and scored in
the full weighted state, ||q~ - Phi z||, without reloading any FOM data.

Usage: python train_gas_opinf_balanced.py [--r 50] [--derivs reproj|fd]
                                          [--reg-seed 1e-3] [--reg 1.7e-5]
Writes models/{opinf,gas_opinf}_balanced[_fd]_r<r>.pkl (the checkpoint format
of train_opinf.py, plus Psi) and the matching _history.pkl.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import time as timer

import numpy as np
from scipy.integrate import solve_ivp

from nitrom.backend import set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import OpInfModule, solve_opinf, train
from nitrom.projections.linear_projection import LinearProjection
from nitrom.training_data import TrainingData, TrainingPool

set_backend("numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
POLY_COMP = [1, 2]


def forecast(A, H, z0, time, blowup=1e6):
    """Integrate dz/dt = A z + H : z z; NaN after a blow-up."""
    r = len(z0)
    H2 = H.reshape(r, r*r)
    zmax = blowup*max(np.linalg.norm(z0), 1.0)
    event = lambda t, z: np.linalg.norm(z) - zmax
    event.terminal = True
    sol = solve_ivp(lambda t, z: A @ z + H2 @ np.outer(z, z).ravel(),
                    (time[0], time[-1]), z0, t_eval=time, method="DOP853",
                    rtol=1e-9, atol=1e-12, events=event)
    Z = np.full((r, len(time)), np.nan)
    Z[:, :sol.y.shape[1]] = sol.y
    return Z


def score(A, H, red, r):
    """Per-trajectory relative error ||q~ - Phi z|| / ||q~|| over the whole
    window, and the NiTROM cost sum_j sum_t ||q~ - Phi z||^2 / <||q~||^2>."""
    time, errs, J = red["time"], [], 0.0
    G = red["PhiTPhi"][:r, :r]
    for k in range(len(red["parameters"])):
        Z = forecast(A, H, red["Z"][k, :r, 0], time)
        if not np.all(np.isfinite(Z)):
            errs.append(np.inf)
            J = np.inf
            continue
        e2 = (red["sq"][k] - 2*(Z*red["PhiTX"][k, :r]).sum(0)
              + (Z*(G @ Z)).sum(0))                 # ||q~(t) - Phi z(t)||^2
        e2 = np.maximum(e2, 0.0)
        errs.append(np.sqrt(e2.sum()/red["sq"][k].sum()))
        J += e2.sum()/red["energy"][k]
    return np.array(errs), J


def save_checkpoint(path, kind, r, Phi, Psi, tensors, gas_params=None, **extra):
    ckpt = dict(kind=kind, r=r, poly_comp=POLY_COMP, forcing_config=None,
                basis="balanced", Phi=np.asarray(Phi), Psi=np.asarray(Psi),
                tensors=[np.asarray(t) for t in tensors], gas_params=gas_params,
                **extra)
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"saved -> {os.path.relpath(path, HERE)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=50)
    ap.add_argument("--reg-seed", type=float, default=1e-3,
                    help="OpInf regularization on H (paper: 1e-3)")
    ap.add_argument("--reg", type=float, default=1.7e-5,
                    help="GAS-OpInf regularization on H (paper: 1.7e-5)")
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--derivs", choices=["reproj", "fd"], default="reproj")
    ap.add_argument("--retraction", choices=["lyap", "identity"], default="lyap",
                    help="metric of the GAS retraction: Lyapunov P, or P = I")
    ap.add_argument("--balancing", default=os.path.join(HERE, "balancing_full"))
    ap.add_argument("--models", default=os.path.join(HERE, "models"))
    args = ap.parse_args()
    r = args.r

    bal = np.load(os.path.join(args.balancing, "balancing.npz"), allow_pickle=True)
    R = bal["Phi"].shape[1]
    if r > R:
        raise SystemExit(f"only {R} balanced modes saved; rerun "
                         f"compute_balancing.py --r-save {r}")
    red = dict(np.load(os.path.join(args.balancing, f"reduced_r{R}.npz")))
    params, time = red["parameters"], red["time"]
    n_traj = len(params)
    print(f"balanced basis: r = {r} of {R} saved, Sigma[{r-1}]/Sigma[0] = "
          f"{bal['Sigma'][r-1]/bal['Sigma'][0]:.2e}; {n_traj} trajectories")

    # 1. latent trajectories on disk, the way TrainingPool reads them.  With
    #    fd it computes and caches the finite-difference derivatives itself;
    #    with reproj they were written by compute_reprojected_derivatives.py.
    if args.derivs == "reproj":
        rdir = os.path.join(args.balancing, f"latent_r{r}_reproj")
        if not os.path.exists(os.path.join(rdir, f"deriv_{n_traj-1:03d}.npy")):
            raise SystemExit(f"{os.path.relpath(rdir, HERE)} is incomplete -- "
                             f"run compute_reprojected_derivatives.py --r {r}")
    else:
        rdir = os.path.join(args.balancing, f"latent_r{r}")
        shutil.rmtree(rdir, ignore_errors=True)
        os.makedirs(rdir)
        for k in range(n_traj):
            np.save(os.path.join(rdir, f"traj_{k:03d}.npy"),
                    np.ascontiguousarray(red["Z"][k, :r]))
            np.save(os.path.join(rdir, f"weight_{k:03d}.npy"), [red["energy"][k]])
        np.save(os.path.join(rdir, "time.npy"), time)
    pool = TrainingPool(n_traj=n_traj,
                        fname_traj=os.path.join(rdir, "traj_%03d.npy"),
                        fname_time=os.path.join(rdir, "time.npy"),
                        dtype=np.float64,
                        fname_weights=os.path.join(rdir, "weight_%03d.npy"),
                        fname_derivs=os.path.join(rdir, "deriv_%03d.npy"))
    data = TrainingData(pool, which_trajs=list(range(n_traj)),
                        percent_time_length=1.0, leggauss_deg=5, nsave_rom=1)
    identity = LinearProjection([np.eye(r), np.eye(r)])
    Phi, Psi = bal["Phi"][:, :r], bal["Psi"][:, :r]
    models_dir = args.models
    os.makedirs(models_dir, exist_ok=True)
    tag = ("" if args.derivs == "reproj" else "_fd") + \
        ("" if args.retraction == "lyap" else "_PI")
    print(f"time derivatives: {args.derivs}")

    def fit_residual(A, H):
        """Relative residual ||A z + H:zz - dz/dt|| / ||dz/dt|| on the fit data."""
        num = den = 0.0
        for Zk, dZk in zip(pool.X, pool.dX):
            Zk, dZk = np.asarray(Zk), np.asarray(dZk)
            quad = (H.reshape(r, r*r) @ np.einsum("it,jt->ijt", Zk, Zk)
                    .reshape(r*r, -1))
            num += np.sum((A @ Zk + quad - dZk)**2)
            den += np.sum(dZk**2)
        return np.sqrt(num/den)

    if args.derivs == "reproj":
        chk = PolynomialModel(r, POLY_COMP, dtype=np.float64)
        solve_opinf(OpInfModule(data, chk, identity, reg=0.0))
        print(f"check: unregularized OpInf fit residual = "
              f"{fit_residual(*[np.array(t) for t in chk.get_params()]):.2e} "
              f"(round-off expected for exact re-projection)")

    # 2. OpInf seed
    print(f"\n=== OpInf (lambda = {args.reg_seed:g}) ===")
    seedmod = PolynomialModel(r, POLY_COMP, dtype=np.float64)
    solve_opinf(OpInfModule(data, seedmod, identity, reg=args.reg_seed))
    A0, H0 = [np.array(t) for t in seedmod.get_params()]
    print(f"fit residual {fit_residual(A0, H0):.2e}")
    errs, J = score(A0, H0, red, r)
    print(f"max Re eig(A) = {np.linalg.eigvals(A0).real.max():+.4f}; forecast: "
          f"J = {J:.4e}, rel error per traj {np.array2string(errs, precision=3)}")
    save_checkpoint(os.path.join(models_dir, f"opinf_balanced{tag}_r{r}.pkl"),
                    "opinf", r, Phi, Psi, [A0, H0], reg=args.reg_seed, derivs=args.derivs,
                    J=J, errors=errs)

    # 3. retract onto the GAS manifold
    print(f"\n=== GAS-OpInf (lambda = {args.reg:g}) ===")
    seed = GasPolynomialModel(r, POLY_COMP, dtype=np.float64)
    seed.retract_general_tensors_to_gas_tensors(
        [A0, H0], use_P_I=(args.retraction == "identity"))
    gas_model = GasPolynomialModel(r, POLY_COMP, dtype=np.float64,
                                   gas_params=[*seed.get_params()])
    A1 = np.asarray(gas_model.assemble_gas_tensors()[0])
    print(f"after retraction ({args.retraction}): max Re eig(A) = "
          f"{np.linalg.eigvals(A1).real.max():+.4f}, "
          f"||A - A_seed||/||A_seed|| = "
          f"{np.linalg.norm(A1 - A0)/np.linalg.norm(A0):.3f}")

    # 4. train the GAS parameters on the OpInf residual
    gas = OpInfModule(data, gas_model, identity, reg=args.reg)
    t0 = timer.perf_counter()
    train(gas, n_epochs=args.epochs, lr=1.0, optimizer_type="lbfgs",
          print_every=max(1, args.epochs//40), tol=1e-14)
    t_train = timer.perf_counter() - t0
    A, H = [np.array(t) for t in gas_model.model.get_params()]
    print(f"fit residual {fit_residual(A, H):.2e}")
    errs, J = score(A, H, red, r)
    print(f"training time {t_train:.1f} s; max Re eig(A) = "
          f"{np.linalg.eigvals(A).real.max():+.6f} (negative by construction)")
    print(f"forecast: J = {J:.4e}, rel error per traj "
          f"{np.array2string(errs, precision=3)}")

    save_checkpoint(os.path.join(models_dir, f"gas_opinf_balanced{tag}_r{r}.pkl"),
                    "gas", r, Phi, Psi, [A, H],
                    gas_params=[np.array(t) for t in gas_model.get_params()],
                    reg=args.reg, derivs=args.derivs,
                    retraction=args.retraction, J=J, errors=errs)
    with open(os.path.join(models_dir, f"gas_opinf_balanced{tag}_r{r}_history.pkl"),
              "wb") as f:
        pickle.dump(dict(iters=np.arange(len(gas.loss_history)),
                         loss=gas.loss_history, gradnorm=gas.gradnorm_history,
                         time=t_train), f)


if __name__ == "__main__":
    main()
