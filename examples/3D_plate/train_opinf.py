"""OpInf and GAS-OpInf for the 3D wing on POD or balanced coordinates.

The airfoil recipe (examples/airfoil/train_gas_opinf_balanced.py), on the
latent data of compute_reprojected_derivatives.py:

  1. z_k = Psi^T x_k with re-projected derivatives (--derivs reproj, default)
     or nitrom's 4th-order finite differences (--derivs fd); the unregularized
     fit residual is printed (round-off expected for exact re-projection);
  2. closed-form OpInf (lambda = --reg-seed on H);
  3. retraction onto the GAS manifold (Lyapunov metric);
  4. L-BFGS on the OpInf residual over the GAS parameters (lambda = --reg).

Each trajectory's residual is divided by its time-averaged energy (the pool's
weights).  Models are forecast from z_k(0) and scored in the full weighted
state, ||q~ - U Phi z|| / ||q~||, exactly, from the lossless coordinates
(||q~ - U Phi z|| = ||x - Phi z||): no FOM data is reloaded.

Usage (python with nitrom on the path):
    python train_opinf.py --basis pod|balanced [--r 50] [--derivs reproj|fd]
Writes models_wing_re1000/{opinf,gas_opinf}_<basis>[_fd]_r<r>.pkl (+ history).
Bases are r_l-vectors in the lossless coordinates (lift with
rom_compress.Compressed.lift).
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


def score(A, H, X, Phi, Psi, energy, time):
    """Per-trajectory relative error ||x - Phi z|| / ||x|| over the window,
    and the cost sum_k sum_t ||x - Phi z||^2 / Ebar_k."""
    errs, J = [], 0.0
    for k in range(len(X)):
        Z = forecast(A, H, Psi.T @ X[k][:, 0], time)
        if not np.all(np.isfinite(Z)):
            errs.append(np.inf)
            J = np.inf
            continue
        e2 = ((X[k] - Phi @ Z)**2).sum(0)
        errs.append(np.sqrt(e2.sum()/(X[k]**2).sum()))
        J += e2.sum()/energy[k]
    return np.array(errs), J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", choices=["pod", "balanced"], required=True)
    ap.add_argument("--r", type=int, default=50)
    ap.add_argument("--reg-seed", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1.7e-5)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--derivs", choices=["reproj", "fd"], default="reproj")
    ap.add_argument("--data", default=os.path.join(HERE, "fom_data_wing_re1000_train"))
    ap.add_argument("--models", default=os.path.join(HERE, "models_wing_re1000"))
    ap.add_argument("--skip-gas", action="store_true", help="closed-form OpInf only")
    ap.add_argument("--linear", action="store_true",
                    help="linear model dz/dt = A z (no H; nitrom regularizes only H, so "
                         "this is unregularized least squares); closed-form OpInf only")
    args = ap.parse_args()
    r = args.r
    poly_comp = [1] if args.linear else POLY_COMP

    c = np.load(os.path.join(args.data, "compressed.npz"), allow_pickle=True)
    X, time, params, energy = c["X"], c["time"], c["parameters"], c["energy"]
    bases = np.load(os.path.join(args.data, "bases.npz"), allow_pickle=True)
    if args.basis == "pod":
        Phi = Psi = bases["Phi_pod"][:, :r]
    else:
        Phi, Psi = bases["Phi_bal"][:, :r], bases["Psi_bal"][:, :r]
    n_traj = len(X)
    print(f"{args.basis} basis, r = {r} (lossless dimension {X.shape[1]}); "
          f"{n_traj} trajectories; derivatives: {args.derivs}")

    if args.derivs == "reproj":
        rdir = os.path.join(args.data, f"latent_{args.basis}_r{r}_reproj")
        if not os.path.exists(os.path.join(rdir, f"deriv_{n_traj - 1:03d}.npy")):
            raise SystemExit(f"{rdir} is incomplete -- run compute_reprojected_derivatives.py "
                             f"--basis {args.basis} --r {r}")
    else:
        rdir = os.path.join(args.data, f"latent_{args.basis}_r{r}")
        shutil.rmtree(rdir, ignore_errors=True)
        os.makedirs(rdir)
        for k in range(n_traj):
            np.save(os.path.join(rdir, f"traj_{k:03d}.npy"), np.ascontiguousarray(Psi.T @ X[k]))
            np.save(os.path.join(rdir, f"weight_{k:03d}.npy"), [energy[k]])
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
    os.makedirs(args.models, exist_ok=True)
    tag = (f"{args.basis}{'' if args.derivs == 'reproj' else '_fd'}"
           f"{'_linear' if args.linear else ''}_r{r}")

    def tensors(model):
        """(A, H) of a fitted model; H = 0 for the linear one."""
        t = [np.array(x) for x in model.get_params()]
        return (t[0], np.zeros((r, r*r))) if args.linear else tuple(t)

    def fit_residual(A, H):
        num = den = 0.0
        for Zk, dZk in zip(pool.X, pool.dX):
            Zk, dZk = np.asarray(Zk), np.asarray(dZk)
            quad = H.reshape(r, r*r) @ np.einsum("it,jt->ijt", Zk, Zk).reshape(r*r, -1)
            num += np.sum((A @ Zk + quad - dZk)**2)
            den += np.sum(dZk**2)
        return np.sqrt(num/den)

    def save(kind, tensors, **extra):
        path = os.path.join(args.models, f"{kind}_{tag}.pkl")
        with open(path, "wb") as f:
            pickle.dump(dict(kind=kind, r=r, poly_comp=poly_comp, basis=args.basis,
                             Phi=Phi, Psi=Psi, tensors=[np.asarray(t) for t in tensors],
                             coordinates=os.path.join(args.data, "compressed.npz"),
                             **extra), f)
        print(f"saved -> {os.path.relpath(path, HERE)}")

    if args.derivs == "reproj":
        chk = PolynomialModel(r, poly_comp, dtype=np.float64)
        solve_opinf(OpInfModule(data, chk, identity, reg=0.0))
        print(f"check: unregularized OpInf fit residual = "
              f"{fit_residual(*tensors(chk)):.2e} "
              f"({'linear fit of a quadratic model: not round-off' if args.linear else 'round-off expected for exact re-projection'})")

    print(f"\n=== {'linear ' if args.linear else ''}OpInf"
          f"{'' if args.linear else f' (lambda = {args.reg_seed:g})'} ===")
    seedmod = PolynomialModel(r, poly_comp, dtype=np.float64)
    solve_opinf(OpInfModule(data, seedmod, identity, reg=args.reg_seed))
    A0, H0 = tensors(seedmod)
    errs, J = score(A0, H0, X, Phi, Psi, energy, time)
    print(f"fit residual {fit_residual(A0, H0):.2e}; max Re eig(A) = "
          f"{np.linalg.eigvals(A0).real.max():+.4f}; forecast J = {J:.4e}, "
          f"rel error per traj {np.array2string(errs, precision=3)}")
    save("opinf", [A0] if args.linear else [A0, H0], reg=args.reg_seed, derivs=args.derivs,
         J=J, errors=errs)
    if args.skip_gas or args.linear:
        return

    print(f"\n=== GAS-OpInf (lambda = {args.reg:g}) ===")
    seed = GasPolynomialModel(r, POLY_COMP, dtype=np.float64)
    seed.retract_general_tensors_to_gas_tensors([A0, H0], use_P_I=False)
    gas_model = GasPolynomialModel(r, POLY_COMP, dtype=np.float64,
                                   gas_params=[*seed.get_params()])
    gas = OpInfModule(data, gas_model, identity, reg=args.reg)
    t0 = timer.perf_counter()
    train(gas, n_epochs=args.epochs, lr=1.0, optimizer_type="lbfgs",
          print_every=max(1, args.epochs//40), tol=1e-14)
    t_train = timer.perf_counter() - t0
    A, H = [np.array(t) for t in gas_model.model.get_params()]
    errs, J = score(A, H, X, Phi, Psi, energy, time)
    print(f"training {t_train:.1f} s; fit residual {fit_residual(A, H):.2e}; max Re eig(A) = "
          f"{np.linalg.eigvals(A).real.max():+.6f}; forecast J = {J:.4e}, "
          f"rel error per traj {np.array2string(errs, precision=3)}")
    save("gas_opinf", [A, H], gas_params=[np.array(t) for t in gas_model.get_params()],
         reg=args.reg, derivs=args.derivs, J=J, errors=errs)
    with open(os.path.join(args.models, f"gas_opinf_{tag}_history.pkl"), "wb") as f:
        pickle.dump(dict(iters=np.arange(len(gas.loss_history)), loss=gas.loss_history,
                         gradnorm=gas.gradnorm_history, time=t_train), f)


if __name__ == "__main__":
    main()
