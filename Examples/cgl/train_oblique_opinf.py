r"""
Oblique operator inference for the CGL equation.

Standard Operator Inference reduces dimension by *orthogonally* projecting onto
POD modes.  Section 4 of Padovan, Vollmer & Bodony (SIADS 2024) attributes
OpInf's loss of accuracy on the CGL to exactly that: the CGL is strongly
non-normal, and ROMs of non-normal systems need a carefully chosen **oblique**
projection.  Concretely, on this problem the leading five POD modes capture
99.1% of the training snapshots but only 9.8% of the impulse initial condition
-- the energetic downstream response dominates the POD, while the input acts
upstream at branch I.  An orthogonally-projected ROM therefore starts from
almost nothing.

:class:`~nitrom.optimization.ObliqueOpInfModule` keeps the derivative-matching
cost of OpInf but learns the test basis alongside the reduced tensors:

.. math::

    J = \sum_{j} \frac{1}{\alpha_j} \sum_{i}
        \bigl\| S\Psi^\top \dot{q}^{(j)}(t_i)
                - S\,g\bigl(\Psi^\top q^{(j)}(t_i)\bigr) \bigr\|^2
        + \lambda \|\mathrm{Mat}(H_r)\|_F^2 ,
    \qquad S = (\Psi^\top \Phi)^{-1}.

Here :math:`\Psi` is charted as :math:`\Psi = \Phi + W N`
(:class:`~nitrom.projections.ObliqueChartProjection`), with :math:`\Phi` the
leading :math:`r` POD modes and :math:`W` the next :math:`k` -- the "left-over"
modes.  Since :math:`W^\top\Phi = 0`, this gives :math:`\Psi^\top\Phi = I`
identically, so :math:`S = I`: no matrix inverse to condition, no
transversality constraint, and the single unknown :math:`N \in
\mathbb{R}^{k\times r}` is unconstrained (Euclidean, no Stiefel retraction).
The trailing POD modes are precisely the directions :math:`\Phi` is missing, so
they are what lets :math:`\Psi` see the initial condition.

The optimization starts from the trained OpInf model with :math:`N = 0`, i.e.
:math:`\Psi = \Phi`, so the initial cost equals OpInf's exactly -- any decrease
is attributable to the oblique projection.  :math:`\lambda = 10^9`, as for
OpInf.

Run with ``python train_oblique_opinf.py`` (after ``train_opinf.py``), or
trajectory-parallel with ``mpiexec -n 4 python train_oblique_opinf.py``.
"""
import os
import pickle

import fom_class
import numpy as np

from nitrom.backend import mpi_allreduce_scalar, mpi_rank_size, set_backend
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import ObliqueOpInfModule, solve_oblique_opinf
from nitrom.projections import ObliqueChartProjection
from nitrom.roms.param_registry import ParamRegistry
from nitrom.training_data import TrainingData, TrainingPool
from nitrom.utils import compute_POD

set_backend("numpy")
dtype = np.float64
rank, world_size = mpi_rank_size()


def printr(*a, **k):
    if rank == 0:
        print(*a, **k)


def gcost(m):
    c = float(m())
    return mpi_allreduce_scalar(c) if world_size > 1 else c


traj_path = "./trajectories/"
models_dir = "./models/"
n_traj = 8
r = 5
poly_comp = [1, 3]
LAMBDA = 1.0e9
K_CHART = 11  # number of trailing POD modes spanning the chart W

# %% Full-order model (rebuilt exactly as in generate_data.py)

x, dx, A, B_fom, C = fom_class.build_operators(L=30.0, n=301, dtype=dtype)

# %% Load the training trajectories

pool = TrainingPool(
    n_traj=n_traj,
    fname_traj=traj_path + "traj_%03d.npy",
    fname_time=traj_path + "time.npy",
    dtype=dtype,
    fname_weights=traj_path + "weight_%03d.npy",
    fname_derivs=traj_path + "deriv_%03d.npy",
)

training_data = TrainingData(
    pool,
    which_trajs=list(range(n_traj)),
    percent_time_length=1.0,
    leggauss_deg=5,
    nsave_rom=1,
)

# %% Initialize from the trained OpInf model (Phi = POD, Psi = Phi)

with open(os.path.join(models_dir, "opinf_model.pkl"), "rb") as f:
    ckpt = pickle.load(f)
Phi = np.ascontiguousarray(np.asarray(ckpt["Phi"], dtype=dtype))
tensors = [np.ascontiguousarray(np.asarray(t, dtype=dtype)) for t in ckpt["tensors"]]
printr(f"initialized from OpInf: ||A_r|| = {np.linalg.norm(tensors[0]):.4e}, "
       f"||H_r|| = {np.linalg.norm(tensors[1]):.4e}")

# The chart W: POD modes r ... r + k - 1, orthogonal to Phi by construction.
U, S_pod, _ = compute_POD(pool, normalize=True)
W = np.ascontiguousarray(U[:, r:r + K_CHART])
printr(f"chart: W spans POD modes {r + 1}-{r + K_CHART} "
       f"({K_CHART * r} free parameters in N)")

model = PolynomialModel(r, poly_comp, dtype=dtype, tensors=tensors)
# N = 0  =>  Psi = Phi, i.e. the orthogonal projection OpInf already uses.
projection = ObliqueChartProjection([Phi, W, np.zeros((K_CHART, r), dtype=dtype)])
registry = ParamRegistry(model, projection)

oblique = ObliqueOpInfModule(training_data, registry, reg=LAMBDA)

printr(f"\n=== Oblique OpInf (lambda = {LAMBDA:.1e}) ===")
printr(f"trainable: {[n for n, _ in oblique.named_parameters()]}")
printr(f"initial cost: {gcost(oblique):.6e}")
# Alternating least squares: closed-form tensor solve, then Levenberg-Marquardt
# on the chart coefficients.  A joint quasi-Newton descent stalls here -- the
# two blocks are coupled through z = Psi^T q, so the latent data move whenever
# Psi does.
solve_oblique_opinf(oblique, n_sweeps=30, verbose=(rank == 0))
printr(f"final cost:   {gcost(oblique):.6e}")

# %% Save

N_opt = np.asarray(oblique.N)
Psi = projection.Psi
printr(f"||N|| = {np.linalg.norm(N_opt):.4e}")
printr(f"Psi^T Phi == I: "
       f"{np.allclose(Psi.T @ Phi, np.eye(r), atol=1e-10)} "
       f"(residual {np.linalg.norm(Psi.T @ Phi - np.eye(r)):.2e})")
# How much of the impulse each projector retains.  The orthogonal projector
# can only shrink it; the oblique one is not a contraction, so a value above 1
# simply means Phi Psi^T amplifies the branch-I input -- which is the point.
q0_train = np.load(traj_path + "q0.npy")
cap_phi = np.linalg.norm(Phi @ (Phi.T @ q0_train.T)) / np.linalg.norm(q0_train)
cap_psi = np.linalg.norm(Phi @ (Psi.T @ q0_train.T)) / np.linalg.norm(q0_train)
printr(f"||P q0|| / ||q0|| on the training impulses: "
       f"orthogonal {cap_phi:.4f} -> oblique {cap_psi:.4f}")

if rank == 0:
    ckpt_out = {
        "kind": "oblique_opinf",
        "r": r,
        "poly_comp": poly_comp,
        "Phi": Phi,
        "Psi": Psi,
        "W": W,
        "N": N_opt,
        "B_r": Psi.T @ B_fom,  # fixed reduced input operator, set by projection
        "tensors": [np.asarray(getattr(oblique, n)) for n in model.param_names],
    }
    path = os.path.join(models_dir, "oblique_opinf_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(ckpt_out, f)
    print(f"saved -> {path}")
    with open(os.path.join(models_dir, "oblique_opinf_history.pkl"), "wb") as f:
        pickle.dump({"loss": oblique.loss_history}, f)
