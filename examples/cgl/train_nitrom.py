"""
NiTROM refinement of the Lall-averaged balanced ROM.

Starts from the r = 5 cubic ROM that train_models.py builds on the Lall-averaged
balanced basis and minimizes the NiTROM trajectory cost

    J = sum_j (1/alpha_j) sum_i || y^(j)(t_i) - C Phi z^(j)(t_i) ||^2,

where z solves the latent ODE from z(0) = Psi^T q^(j)(0).  By default the latent
tensors A_r, H_r and both bases move, Phi on Grassmann and Psi on Stiefel; with
FREEZE_BASES only A_r and H_r do, which asks whether the balanced basis is
limited by its dynamics or by the subspace itself.

With RESUME (the default) a previous run's saved model is the starting point
instead, so training can be continued in further batches of N_EPOCHS iterations.

Two conversions matter here and are checked when starting from the OpInf ROM:

* **physical coordinates.** balanced_opinf fits x~ = x/c with a c^2 on the cubic
  term.  Undoing that scaling, dz/dt = A z + H (z x z x z) with the *same* A, H
  and no amplitude factor, so NiTROM is fed raw (un-normalized) trajectories and
  PolynomialModel needs no per-trajectory scaling.
* **tensor layout.** balanced_opinf stores H_r against the 35 unique cubic
  monomials; PolynomialModel contracts a dense (r, r, r, r) tensor.  The weight
  of each monomial is spread over its distinct permutations.

Run serially:
    python train_nitrom.py
or trajectory-parallel over MPI (at most one rank per trajectory):
    mpiexec -n 4 python train_nitrom.py

Under MPI each rank holds a shard of the trajectories and returns a *partial*
cost and gradient; train() all-reduces them over COMM_WORLD, so the reduced
values equal the serial ones.  The weights carry the global N_traj * N factor
of eq. (3.7) for exactly that reason -- using the local count would rescale the
cost with the rank count.

Cost and gradient norm print every PRINT_EVERY iterations, on rank 0.
"""

import argparse
import itertools
import os
import shutil

import numpy as np

from cgl import CGL
from balanced_opinf import ROM, poly_features, poly_index

from nitrom.backend import mpi_allreduce_scalar, mpi_rank_size, set_backend
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import NitromModule, train
from nitrom.projections import LinearProjection
from nitrom.roms.param_registry import ParamRegistry

set_backend("numpy")
RANK, WORLD = mpi_rank_size()
HERE = os.path.dirname(os.path.abspath(__file__))


def printr(*a, **k):
    if RANK == 0:
        print(*a, **k)


def gcost(module):
    """Globally reduced cost: each rank's forward() is a partial sum."""
    c = float(module())
    return mpi_allreduce_scalar(c) if WORLD > 1 else c
DATA = os.path.join(HERE, "data")

# OpInf ROMs in roms_r{R}.npy that a run can be initialized from, keyed by the
# tag given on the command line.  Each init gets its own output file and its own
# label, so runs from different starting points never overwrite one another.
INITS = {"lall": "Balanced (Lall-averaged)",
         "pod": "POD"}
LABELS = {"lall": "NiTROM (Lall init)",
          "pod": "NiTROM (POD init)"}
DEFAULT_INIT = "lall"
BASIS = INITS[DEFAULT_INIT]        # default init; also read by bench_mpi_scaling.py
R = 5
# Warm start.  With RESUME and roms_r{R}_nitrom.npy present, training continues
# from that saved iterate instead of the train_models.py balanced ROM; the
# previous file is copied to .prev first so a diverging continuation cannot
# destroy a good model.  Set False to restart from the balanced ROM.
RESUME = True
# With FREEZE_BASES the optimizer moves only A_r and H_r.  Otherwise Phi and Psi
# are optimized too, on the manifolds the paper uses: Phi on Grassmann (only its
# span matters), Psi on Stiefel.
FREEZE_BASES = False
# Forecast horizons.  A single entry trains in one shot on that window; several
# entries run the continuation of section 5.1 of the paper, minimizing on each
# t <= T in turn and warm-starting from the previous stage (e.g.
# [25.0, 50.0, 100.0, 200.0]).  200 is the window plot_results.py scores on.
HORIZONS = [200.0]
N_SNAP = 1001                      # full window, t <= 200 (used by the benchmark)
# ROM sub-steps between snapshots (dt = 0.2/N_SUBSTEPS).  The forward solve takes
# n_snap*N_SUBSTEPS Python-level RK4 steps, so this multiplies runtime directly.
# At 3 the cost and gradient agree with a ~17x finer solve to 7e-7 and 1e-6, far
# below anything the optimizer resolves, for ~4.7x the speed.
N_SUBSTEPS = 3
N_EPOCHS = 300                     # L-BFGS iterations per horizon (--iters)
# Checkpointing (--ckpt-m/--ckpt-q).  m = 1 is the default: NiTROM trains on the
# 8 full trajectories.  Setting m > 1 cuts each trajectory into m equal-length
# sliding windows, as train_models.py does when building the balanced basis,
# which puts NiTROM on the same footing as the basis in number of initial
# conditions -- at m times the cost per iteration, and only from a starting
# model that is stable on the mid-transient checkpoint ICs (an OpInf ROM is;
# a NiTROM model trained on the full trajectories alone is not).
M_CKPT, Q_CKPT = 1, 5
PRINT_EVERY = 5
REG = 0.0                # Tikhonov weight on H_r inside NiTROM


class FOM:
    """Output map only: NiTROM needs y = C q and its (constant) Jacobian."""

    def __init__(self, C):
        self.C = C

    def compute_output(self, q):
        return np.matmul(self.C, q)

    def compute_output_derivative(self, q):
        return self.C


class Data:
    """Minimal TrainingData view: NitromModule reads X, time, weights and
    forcing_fns.  Built directly so that no time-derivative files are
    synthesized on disk -- NiTROM never uses dX."""

    def __init__(self, X, time, weights):
        self.X, self.time, self.weights = X, time, weights
        self.forcing_fns = []


def orthonormalize_psi(Phi, Psi, Ar, Hr):
    r"""
    Equivalent model whose test basis has orthonormal columns.

    riemannian_optimize projects the starting point onto its manifold with a QR,
    so a Stiefel-constrained Psi is orthonormalized before the first step.  That
    is not a harmless re-scaling: z = Psi^T q, so it changes the latent
    coordinates and would invalidate A_r and H_r unless they move with it.

    Writing the QR as Psi = Psi_o R, the new coordinates are z' = T z with
    T = R^-T, so A' = T A T^-1 and H'(z',z',z') = T H(T^-1 z', ...).  Phi is left
    alone: the decode Phi (Psi^T Phi)^-1 is invariant under Phi -> Phi Q, so
    Grassmann's own QR costs nothing and needs no correction.

    :returns: ``(Psi_o, Ar_new, Hr_new)`` -- the same input-output model
    """
    Psi_o, Rq = np.linalg.qr(Psi)
    T = np.linalg.inv(Rq).T
    Ti = np.linalg.inv(T)
    Ar_new = T @ Ar @ Ti
    # optimize=True: the default walks all r^8 index combinations, which is
    # harmless at r = 5 but explodes with r (see the cavity version's note).
    Hr_new = np.einsum("ai,ijkl,jb,kc,ld->abcd", T, Hr, Ti, Ti, Ti, optimize=True)
    return Psi_o, Ar_new, Hr_new


def checkpoint_windows(X, m, q):
    """
    Sliding equal-length windows starting at snapshots 0, q, ..., (m-1)q.

    The CGL is autonomous, so the state at snapshot k*q is a legitimate initial
    condition at t = 0 and the window that follows it is a trajectory in its own
    right.  All windows are given the same length L = n_snap - (m-1)q, so a
    single time grid serves them all; checkpoints that would run past the end
    are dropped.  This is the same construction compute_data_driven_balancing
    uses to enlarge the span of X_0, at zero extra simulation cost.

    :returns: ``(windows, L)`` with windows of shape ``(n_traj*m, n, L)``,
        ordered trajectory-major so that MPI's strided shard mixes parents.
    """
    n_snap = X.shape[2]
    starts = [k*q for k in range(m) if k*q < n_snap]
    L = n_snap - starts[-1]
    W = np.stack([Xj[:, s:s + L] for Xj in X for s in starts])
    return W, L


def sym_to_tensor(Hs, r, degree=3):
    """
    Dense (r,) * (degree + 1) tensor equivalent to the symmetric-monomial form.

    ``Hs[:, m]`` multiplies the monomial z_a z_b z_c of ``poly_index``; the dense
    tensor must reproduce that when contracted with z ``degree`` times, so each
    monomial's weight is split evenly over its distinct index permutations.
    """
    T = np.zeros((r,) + (r,)*degree)
    for m, triple in enumerate(poly_index(r, degree)):
        perms = set(itertools.permutations(tuple(triple)))
        for p in perms:
            T[(slice(None),) + p] += Hs[:, m]/len(perms)
    return T


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="NiTROM refinement of an OpInf ROM for the CGL example.")
    ap.add_argument("init", nargs="?", default=DEFAULT_INIT, choices=sorted(INITS),
                    help="OpInf ROM to start from (default: %(default)s)")
    ap.add_argument("--iters", type=int, default=N_EPOCHS,
                    help="L-BFGS iterations per horizon (default: %(default)s)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any saved NiTROM model for this init and "
                         "restart from the OpInf ROM")
    ap.add_argument("--ckpt-m", type=int, default=M_CKPT, metavar="M",
                    help="checkpoints per trajectory (1 disables; default: "
                         "%(default)s, matching train_models.py)")
    ap.add_argument("--ckpt-q", type=int, default=Q_CKPT, metavar="Q",
                    help="snapshot stride between checkpoints (default: %(default)s)")
    return ap.parse_args(argv)


def main(args=None):
    args = parse_args() if args is None else args
    basis, label = INITS[args.init], LABELS[args.init]
    n_epochs = args.iters

    cgl = CGL()
    nx = cgl.nx
    c = np.exp(-((cgl.x + cgl.xbar)/cgl.s)**2)      # sensor at branch II, eq. (4.2)
    C = np.zeros((2, 2*nx))
    C[0, :nx], C[1, nx:] = c, c
    fom = FOM(C)

    alphas = np.load(os.path.join(DATA, "train_alpha.npy"))
    time_full = np.load(os.path.join(DATA, "time.npy"))[:N_SNAP]
    # Raw (un-normalized) trajectories: in physical coordinates one (A, H) serves
    # every amplitude, with no c^2 factor.
    X = np.stack([a*np.load(os.path.join(DATA, f"train_traj_{j:03d}.npy"))[:, :N_SNAP]
                  for j, a in enumerate(alphas)])

    out_path = os.path.join(HERE, f"roms_r{R}_nitrom_{args.init}.npy")
    resuming = RESUME and not args.fresh and os.path.exists(out_path)
    d = np.load(os.path.join(HERE, f"roms_r{R}.npy"), allow_pickle=True).item()[basis]
    if resuming:
        if RANK == 0:
            shutil.copyfile(out_path, out_path + ".prev")
        d = np.load(out_path, allow_pickle=True).item()[label]
        printr(f"resuming {label} from {os.path.basename(out_path)} "
               f"({d.get('n_iters', 0)} iterations so far; previous saved as .prev)")
    else:
        printr(f"starting {label} from the '{basis}' OpInf ROM")

    Phi, Psi = np.ascontiguousarray(d["Phi"]), np.ascontiguousarray(d["Psi"])
    Ar = np.ascontiguousarray(d["Ar"])
    # A resumed Hr is already the dense tensor NiTROM stores; a fresh one is in
    # balanced_opinf's symmetric-monomial layout and must be converted.
    Hr = np.ascontiguousarray(d["Hr"])
    if Hr.ndim == 2:
        Hr = sym_to_tensor(Hr, R)
        # Check the two conversions reproduce the balanced-OpInf right-hand side.
        rng = np.random.default_rng(0)
        z = rng.standard_normal(R)
        ref = ROM(d["Ar"], d["Hr"], 3).rhs(0.0, z, c=1.0)
        got = Ar @ z + np.einsum("abcd,b,c,d->a", Hr, z, z, z)
        printr(f"tensor conversion check: max |dRHS| = {np.abs(ref - got).max():.3e}")

    if not FREEZE_BASES:
        # Move to coordinates where Psi is orthonormal, so the Stiefel
        # projection of the starting point is a no-op rather than a silent
        # change of model.  Phi needs no correction.  A resumed Psi comes off
        # the Stiefel optimizer already orthonormal, so this is then skipped.
        dev = np.linalg.norm(Psi.T @ Psi - np.eye(R))
        if dev > 1e-10:
            Psi, Ar, Hr = orthonormalize_psi(Phi, Psi, Ar, Hr)
            printr(f"orthonormalized Psi (was off Stiefel by {dev:.2e})")
        else:
            printr(f"Psi already on Stiefel ({dev:.2e}); coordinates unchanged")

    model = projection = None
    for stage, t_end in enumerate(HORIZONS, 1):
        n_snap = int(np.searchsorted(time_full, t_end)) + 1
        Xw = X[:, :, :n_snap]
        if args.ckpt_m > 1:
            Xw, n_snap = checkpoint_windows(Xw, args.ckpt_m, args.ckpt_q)
        time = time_full[:n_snap]
        n_traj_global = Xw.shape[0]

        # Trajectory parallelism: each rank keeps a strided shard and returns a
        # partial cost/gradient, which train() all-reduces.
        if WORLD > n_traj_global:
            raise ValueError(
                f"{WORLD} ranks for {n_traj_global} trajectories; use at most "
                f"one rank per trajectory."
            )
        mine = list(range(RANK, n_traj_global, WORLD))

        # alpha_j is the time-averaged output energy of *each window*, so every
        # window -- including the late, low-amplitude ones -- contributes on the
        # same footing.  alpha_j and the eq. (3.7) factor are recomputed per
        # stage, so costs are not comparable between stages.
        Yw = np.matmul(C, Xw)
        alpha = np.mean(np.sum(Yw**2, axis=1), axis=1)
        weights = alpha*n_traj_global*n_snap
        data = Data(Xw[mine], time, weights[mine])

        model = PolynomialModel(R, [1, 3], dtype=np.float64, tensors=[Ar, Hr])
        projection = LinearProjection([Phi, Psi])
        registry = ParamRegistry(model, projection)
        nitrom = NitromModule(data, registry, fom=fom, reg=REG,
                              n_substeps=N_SUBSTEPS, adjoint_method="discrete")
        if FREEZE_BASES:
            nitrom.set_unlearnable("Phi", "Psi")   # only A_r, H_r move
        else:
            nitrom.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])

        if stage == 1:
            printr(f"training set: {X.shape[0]} trajectories x "
                   f"{args.ckpt_m if args.ckpt_m > 1 else 1} checkpoint(s) "
                   f"(stride {args.ckpt_q}) = {n_traj_global} windows of "
                   f"{n_snap} snapshots")
            printr(f"rank layout: {WORLD} rank(s), "
                   f"{[len(range(k, n_traj_global, WORLD)) for k in range(WORLD)]} windows each")
            printr(f"trainable: {[n for n, v in zip(registry.names, nitrom.is_learnable.values()) if v]}"
                   f"  frozen: {[n for n, v in zip(registry.names, nitrom.is_learnable.values()) if not v]}")
            printr(f"manifolds: {dict(zip(registry.names, nitrom.get_manifold_types()))}")
        printr(f"\n=== stage {stage}/{len(HORIZONS)}: t <= {t_end:g} "
               f"({n_snap} snapshots), {n_epochs} iterations ===")
        printr(f"initial cost: {gcost(nitrom):.6e}")

        train(nitrom, n_epochs=n_epochs, lr=1.0, optimizer_type="lbfgs",
              print_every=PRINT_EVERY, tol=1e-12)

        printr(f"stage {stage} final cost: {gcost(nitrom):.6e}")
        # Carry the iterate into the next, longer window.
        nitrom._sync_to_registry()
        Ar = np.ascontiguousarray(np.asarray(model.A_1))
        Hr = np.ascontiguousarray(np.asarray(model.A_3))
        Phi = np.ascontiguousarray(np.asarray(projection.Phi))
        Psi = np.ascontiguousarray(np.asarray(projection.Psi))

    out = dict(d)
    out.update(Phi=np.asarray(projection.Phi), Psi=np.asarray(projection.Psi),
               Ar=np.asarray(model.A_1),
               Hr=np.asarray(model.A_3), reg=REG,
               n_iters=out.get("n_iters", 0) + n_epochs*len(HORIZONS))
    if RANK == 0:
        np.save(out_path, {label: out}, allow_pickle=True)
        print(f"saved -> {out_path}  (cumulative iterations: {out['n_iters']})")


if __name__ == "__main__":
    main()
