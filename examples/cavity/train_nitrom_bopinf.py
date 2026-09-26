"""
NiTROM refinement of the balanced (Lall-averaged) OpInf ROM for the cavity.

Starts from the r-dimensional quadratic OpInf ROM built on the balanced basis
(compute_data_driven_balancing + balanced_opinf.opinf) and minimizes the NiTROM
trajectory cost

    J = sum_j (1/alpha_j) sum_i || y^(j)(t_i) - C Phi z^(j)(t_i) ||^2,

where z solves the latent ODE from z(0) = Psi^T q^(j)(0).  Here y = q (section 5
observes the whole state), so C = I: `FOM` returns the state unchanged and
supplies `apply_output_adjoint`, which lets NitromModule apply C^T without ever
forming the 19800 x 19800 identity.

Phi and Psi are free by default, on the manifolds the paper uses: Phi on
Grassmann (only its span matters), Psi on Stiefel.  Pass --freeze-bases to move
only the latent tensors A_r and H_r.

Two conversions matter and are checked at start-up:

* **physical coordinates.** balanced_opinf fits x~ = x/c with a c^(p-1) on the
  degree-p term.  Undoing that scaling, dz/dt = A z + H (z x z) with the *same*
  A, H and no amplitude factor, so NiTROM is fed raw trajectories and
  PolynomialModel needs no per-trajectory scaling.
* **tensor layout.** balanced_opinf stores H_r against the r(r+1)/2 unique
  quadratic monomials; PolynomialModel contracts a dense (r, r, r) tensor.  The
  weight of each monomial is spread over its distinct permutations.

Run serially:
    python train_nitrom_bopinf.py
or trajectory-parallel over MPI (at most one rank per trajectory, so 7 here):
    mpiexec -n 7 python train_nitrom_bopinf.py

Under MPI each rank holds a shard of the trajectories and returns a *partial*
cost and gradient; train() all-reduces them over COMM_WORLD.  The weights carry
the global N_traj * N factor of eq. (3.7) for exactly that reason -- using the
local count would rescale the cost with the rank count.

Cost and gradient norm print every PRINT_EVERY iterations, on rank 0.
"""

import argparse
import itertools
import os
import shutil

import numpy as np

from balanced_opinf import ROM, opinf, poly_index
from train_models import load_train, HERE

from nitrom.backend import mpi_allreduce_scalar, mpi_rank_size, set_backend
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import NitromModule, train
from nitrom.projections import LinearProjection
from nitrom.roms.param_registry import ParamRegistry

set_backend("numpy")
RANK, WORLD = mpi_rank_size()

R = 20
DEGREE = 2                         # quadratic, eq. (5.4)
REG_BAL = 1e0                      # lambda of the OpInf model we start from
# ROM sub-steps between snapshots (dt = 0.1/N_SUBSTEPS).  ||A_r|| ~ 25 here, so
# the latent timescale is ~0.04 and dt = 0.1 alone is too coarse; check with
# --check-substeps before trusting a long run.
N_SUBSTEPS = 5
N_EPOCHS = 200
PRINT_EVERY = 5
REG = 0.0                          # Tikhonov weight on H_r inside NiTROM


def printr(*a, **k):
    if RANK == 0:
        print(*a, **k)


def gcost(module):
    """Globally reduced cost: each rank's forward() is a partial sum."""
    c = float(module())
    return mpi_allreduce_scalar(c) if WORLD > 1 else c


class FOM:
    """Full-state observable, y = q (section 5).

    `apply_output_adjoint` is the reason this is tractable: NitromModule uses it
    instead of forming C = I of size N x N (3.1 GB here).  For y = q the output
    Jacobian is the identity, so its transpose applied to an error is that error.
    """

    def compute_output(self, q):
        return q

    def compute_output_derivative(self, q):          # pragma: no cover
        raise NotImplementedError(
            "y = q: use apply_output_adjoint; forming C = I would be N x N."
        )

    def apply_output_adjoint(self, e, q):
        return e


class Data:
    """Minimal TrainingData view: NitromModule reads X, time, weights and
    forcing_fns.  Built directly so that no time-derivative files are
    synthesized on disk -- NiTROM never uses dX."""

    def __init__(self, X, time, weights):
        self.X, self.time, self.weights = X, time, weights
        self.forcing_fns = []


def sym_to_tensor(Hs, r, degree=DEGREE):
    """
    Dense (r,) * (degree + 1) tensor equivalent to the symmetric-monomial form.

    ``Hs[:, m]`` multiplies the monomial of ``poly_index``; the dense tensor must
    reproduce that when contracted with z ``degree`` times, so each monomial's
    weight is split evenly over its distinct index permutations.
    """
    T = np.zeros((r,) + (r,)*degree)
    for m, tup in enumerate(poly_index(r, degree)):
        perms = set(itertools.permutations(tuple(tup)))
        for p in perms:
            T[(slice(None),) + p] += Hs[:, m]/len(perms)
    return T


def orthonormalize_psi(Phi, Psi, Ar, Hr):
    r"""
    Equivalent model whose test basis has orthonormal columns.

    riemannian_optimize projects the starting point onto its manifold with a QR,
    so a Stiefel-constrained Psi is orthonormalized before the first step.  That
    is not a harmless re-scaling: z = Psi^T q, so it changes the latent
    coordinates and would invalidate A_r and H_r unless they move with it.

    Writing the QR as Psi = Psi_o R, the new coordinates are z' = T z with
    T = R^-T, so A' = T A T^-1 and H'(z',z') = T H(T^-1 z', T^-1 z').  Phi is
    left alone: the decode Phi (Psi^T Phi)^-1 is invariant under Phi -> Phi Q,
    so Grassmann's own QR costs nothing and needs no correction.

    :returns: ``(Psi_o, Ar_new, Hr_new)`` -- the same input-output model
    """
    Psi_o, Rq = np.linalg.qr(Psi)
    T = np.linalg.inv(Rq).T
    Ti = np.linalg.inv(T)
    # One index at a time.  The equivalent one-shot einsum has six indices of
    # size r, and numpy's default optimize=False walks all r^6 of them: 0.14 s
    # at r = 20 but ~35 s at r = 50, on every MPI rank, before anything prints.
    H = np.tensordot(T, Hr, axes=(1, 0))        # (a, j, k)
    H = np.tensordot(H, Ti, axes=(1, 0))        # (a, k, b)
    H = np.tensordot(H, Ti, axes=(1, 0))        # (a, b, c)
    return Psi_o, T @ Ar @ Ti, H


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="NiTROM refinement of the balanced OpInf ROM for the cavity.")
    ap.add_argument("--r", type=int, default=R)
    ap.add_argument("--iters", type=int, default=N_EPOCHS)
    ap.add_argument("--reg-bal", type=float, default=REG_BAL,
                    help="lambda of the OpInf model used as the starting point")
    ap.add_argument("--freeze-bases", action="store_true",
                    help="optimize only A_r and H_r, holding Phi and Psi fixed")
    ap.add_argument("--n-substeps", type=int, default=N_SUBSTEPS)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any saved NiTROM model and restart from OpInf")
    ap.add_argument("--from-gas", action="store_true",
                    help="initialize from the balanced GAS-OpInf model "
                         "(opinf_gas_balanced_r<r>.npy) instead of the OpInf "
                         "fit.  Its assembled (A, H) are stable and already "
                         "trajectory-reasonable, so NiTROM starts from a much "
                         "better point -- but the GAS constraint is dropped "
                         "here, so nothing keeps A stable during training.")
    ap.add_argument("--zero-tensors", action="store_true",
                    help="start from A_r = H_r = 0 with both bases orthonormalized, "
                         "instead of from the OpInf ROM")
    return ap.parse_args(argv)


def main(args=None):
    args = parse_args() if args is None else args
    r = args.r

    trajs_n, betas, time = load_train()
    # Physical (un-normalized) trajectories: in physical coordinates one (A, H)
    # serves every amplitude, with no c factor.
    X = np.stack([abs(b)*T for b, T in zip(betas, trajs_n)])
    dt = time[1] - time[0]
    n_traj_global, _, n_snap = X.shape

    if WORLD > n_traj_global:
        raise ValueError(
            f"{WORLD} ranks for {n_traj_global} trajectories; use at most one "
            f"rank per trajectory."
        )
    mine = list(range(RANK, n_traj_global, WORLD))
    printr(f"rank layout: {WORLD} rank(s), "
           f"{[len(range(k, n_traj_global, WORLD)) for k in range(WORLD)]} "
           f"trajectories each")

    tag = "_fromgas" if args.from_gas else ""
    out_path = os.path.join(HERE, f"roms_r{r}_nitrom_balanced{tag}.npy")
    resuming = not args.fresh and os.path.exists(out_path)
    if resuming:
        if RANK == 0:
            shutil.copyfile(out_path, out_path + ".prev")
        d = np.load(out_path, allow_pickle=True).item()
        Phi, Psi = np.ascontiguousarray(d["Phi"]), np.ascontiguousarray(d["Psi"])
        Ar, Hr = np.ascontiguousarray(d["Ar"]), np.ascontiguousarray(d["Hr"])
        printr(f"resuming from {os.path.basename(out_path)} "
               f"({d.get('n_iters', 0)} iterations so far; previous -> .prev)")
    else:
        b = np.load(os.path.join(HERE, "balancing_r50.npy"),
                    allow_pickle=True).item()
        if b["Phi"].shape[1] < r:
            raise ValueError(f"balancing_r50.npy holds only {b['Phi'].shape[1]} "
                             f"modes; rerun plot_modes.py for r = {r}.")
        Phi = np.ascontiguousarray(b["Phi"][:, :r])
        Psi = np.ascontiguousarray(b["Psi"][:, :r])
        if args.from_gas:
            gp = os.path.join(HERE, f"opinf_gas_balanced_r{r}.npy")
            if not os.path.exists(gp):
                raise FileNotFoundError(
                    f"{os.path.basename(gp)} not found -- run "
                    f"python train_gas_opinf.py --r {r} --basis balanced "
                    f"--derivs reproj --seed reproj")
            g = np.load(gp, allow_pickle=True).item()
            Phi = np.ascontiguousarray(g["Phi"])
            Psi = np.ascontiguousarray(g["Psi"])
            Ar = np.ascontiguousarray(g["Ar_dense"])
            Hr = np.ascontiguousarray(g["Hr_dense"])
            printr(f"starting from the balanced GAS-OpInf model "
                   f"({g['epochs']} epochs, derivs = {g.get('derivs', 'fd')}, "
                   f"seed = {g.get('seed', '?')}): state error {g['err']:.4f}, "
                   f"max Re eig(A) {g['max_re_eig']:+.6f}")
            printr("  the GAS parameterization is NOT carried over: from here "
                   "A and H are free, so stability is no longer guaranteed")
        elif args.zero_tensors:
            # Nothing to keep consistent: with A_r = H_r = 0 the latent model
            # carries no coordinates, so BOTH bases can simply be orthonormalized
            # by QR -- Phi onto Grassmann, Psi onto Stiefel -- with no
            # compensating transformation of the tensors.
            Phi = np.ascontiguousarray(np.linalg.qr(Phi)[0])
            Psi = np.ascontiguousarray(np.linalg.qr(Psi)[0])
            Ar = np.zeros((r, r))
            Hr = np.zeros((r,)*(DEGREE + 1))
            S = Psi.T @ Phi
            printr(f"starting from the balanced bases with ZERO tensors "
                   f"(r = {r}, {b['m_ckpt']} checkpoints stride {b['q_ckpt']})")
            printr(f"both bases orthonormalized: "
                   f"||Phi^T Phi - I|| = {np.linalg.norm(Phi.T @ Phi - np.eye(r)):.2e}, "
                   f"||Psi^T Psi - I|| = {np.linalg.norm(Psi.T @ Psi - np.eye(r)):.2e}")
            # The decode is Phi (Psi^T Phi)^-1, so the two subspaces must not be
            # close to orthogonal; cond(Psi^T Phi) is 1/cos(largest principal angle).
            sv = np.linalg.svd(S, compute_uv=False)
            printr(f"Psi^T Phi: cond {sv[0]/sv[-1]:.2e}, smallest singular value "
                   f"{sv[-1]:.3e} (= cos of the largest principal angle)")
        else:
            trajs = [X[j] for j in range(n_traj_global)]
            rom0 = opinf(trajs, Psi, dt, "quadratic", reg=args.reg_bal, C=None)
            Ar = np.ascontiguousarray(rom0.Ar)
            Hr = sym_to_tensor(rom0.Hr, r)
            printr(f"starting from the balanced OpInf ROM "
                   f"(r = {r}, lambda = {args.reg_bal:.0e}, "
                   f"{b['m_ckpt']} checkpoints stride {b['q_ckpt']})")
            # Check the conversions reproduce the balanced-OpInf right-hand side.
            z = np.random.default_rng(0).standard_normal(r)
            ref = rom0.rhs(0.0, z, c=1.0)
            got = Ar @ z + np.einsum("abc,b,c->a", Hr, z, z)
            printr(f"tensor conversion check: max |dRHS| = "
                   f"{np.abs(ref - got).max():.3e}")

    # Orthonormalize Psi whenever we start from an OpInf fit, frozen bases or
    # not.  With the bases free it is required (the Stiefel retraction would
    # otherwise silently replace the model); with them frozen it is still worth
    # doing, because the balanced Psi carries sigma^-1/2 column scaling, which
    # spreads the latent coordinates over orders of magnitude and leaves A_r and
    # H_r badly scaled for L-BFGS.  orthonormalize_psi transforms the tensors to
    # match, so the model is unchanged either way.
    if not (args.zero_tensors and not resuming) and not args.freeze_bases:
        # Move to coordinates where Psi is orthonormal, so the Stiefel
        # projection of the starting point is a no-op rather than a silent
        # change of model.  A resumed Psi is already orthonormal.
        dev = np.linalg.norm(Psi.T @ Psi - np.eye(r))
        if dev > 1e-10:
            Psi, Ar, Hr = orthonormalize_psi(Phi, Psi, Ar, Hr)
            printr(f"orthonormalized Psi (was off Stiefel by {dev:.2e})")
        else:
            printr(f"Psi already on Stiefel ({dev:.2e}); coordinates unchanged")

    # alpha_j = time-averaged output energy; with y = q that is the state energy.
    alpha = np.mean(np.sum(X**2, axis=1), axis=1)
    weights = alpha*n_traj_global*n_snap
    data = Data(X[mine], time, weights[mine])

    model = PolynomialModel(r, [1, DEGREE], dtype=np.float64, tensors=[Ar, Hr])
    projection = LinearProjection([Phi, Psi])
    registry = ParamRegistry(model, projection)
    nitrom = NitromModule(data, registry, fom=FOM(), reg=REG,
                          n_substeps=args.n_substeps,
                          adjoint_method="discrete")
    if args.freeze_bases:
        nitrom.set_unlearnable("Phi", "Psi")
    else:
        nitrom.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])

    printr(f"trainable: {[n for n, v in zip(registry.names, nitrom.is_learnable.values()) if v]}"
           f"  frozen: {[n for n, v in zip(registry.names, nitrom.is_learnable.values()) if not v]}")
    printr(f"manifolds: {dict(zip(registry.names, nitrom.get_manifold_types()))}")
    printr(f"training set: {n_traj_global} trajectories x {n_snap} snapshots, "
           f"dt = {dt:g}, {args.n_substeps} substeps")
    printr(f"initial cost: {gcost(nitrom):.6e}")

    train(nitrom, n_epochs=args.iters, lr=1.0, optimizer_type="lbfgs",
          print_every=PRINT_EVERY, tol=1e-12)

    printr(f"final cost: {gcost(nitrom):.6e}")
    nitrom._sync_to_registry()
    if RANK == 0:
        prev_iters = (np.load(out_path, allow_pickle=True).item().get("n_iters", 0)
                      if resuming else 0)
        np.save(out_path, dict(
            Phi=np.asarray(projection.Phi), Psi=np.asarray(projection.Psi),
            Ar=np.asarray(model.A_1), Hr=np.asarray(model.A_2),
            reg=REG, reg_bal=args.reg_bal, r=r,
            n_iters=prev_iters + args.iters), allow_pickle=True)
        print(f"saved -> {out_path}  "
              f"(cumulative iterations: {prev_iters + args.iters})")


if __name__ == "__main__":
    main()
