r"""
Operator inference (and POD-Galerkin) for the CGL equation.

Section 4 of Padovan, Vollmer & Bodony (SIADS 2024) seeks cubic ROMs of size
:math:`r = 5`, since the leading five POD modes of the training data hold about
98% of the variance.  The Operator Inference cost is

.. math::

    J = \sum_{j} \frac{1}{\alpha_j} \sum_{i}
        \bigl\| \dot{\hat{z}}^{(j)}(t_i) - A_r \hat{z}^{(j)}(t_i)
                - H_r : \hat{z}^{(j)}(t_i)^{\otimes 3} \bigr\|^2
        + \lambda \|\mathrm{Mat}(H_r)\|_F^2 ,

with :math:`\hat{z} = \Phi^\top q` the POD coefficients and
:math:`\lambda = 10^9` (the value found best by the sweep reported in the
paper).  Both models use the orthogonal projection :math:`\Psi = \Phi`.

The training impulses vanish for :math:`t > 0`, so the reduced input operator
:math:`B_r` is never constrained by the training cost.  It is set by projection
after the fact, :math:`B_r = \Phi^\top B`, and stored in the checkpoint for the
sinusoidal test in ``read_results.py``.

Run with ``python train_opinf.py``, or trajectory-parallel with MPI::

    mpiexec -n 4 python train_opinf.py
"""

import os
import pickle

import fom_class
import numpy as np

from nitrom.backend import mpi_allreduce_scalar, mpi_rank_size, set_backend
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import OpInfModule, solve_opinf
from nitrom.projections.linear_projection import LinearProjection
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
r = 5  # reduced dimension
poly_comp = [1, 3]  # linear + cubic, as required by -a |q|^2 q
LAMBDA = 1.0e9  # regularization on the reduced fourth-order tensor

if rank == 0:
    os.makedirs(models_dir, exist_ok=True)

# %% Full-order model (rebuilt exactly as in generate_data.py)

x, dx, A, B_fom, C = fom_class.build_operators(L=30.0, n=301, dtype=dtype)
fom = fom_class.full_order_model(A, B_fom, C, dtype=dtype)

# %% Load the training trajectories

pool = TrainingPool(
    n_traj=n_traj,
    fname_traj=traj_path + "traj_%03d.npy",
    fname_time=traj_path + "time.npy",
    dtype=dtype,
    fname_weights=traj_path + "weight_%03d.npy",
    fname_derivs=traj_path + "deriv_%03d.npy",
)

# %% POD basis (rank r)

U, S, _ = compute_POD(pool, normalize=True)
Phi = np.ascontiguousarray(U[:, :r])  # (N, r)
energy = np.cumsum(S**2) / np.sum(S**2)
printr(f"POD: r = {r} captures {100 * energy[r - 1]:.2f}% of the variance "
       f"(first 8: {np.array2string(100 * energy[:8], precision=2)})")

projection = LinearProjection([Phi, Phi])  # orthogonal (Psi = Phi)

training_data = TrainingData(
    pool,
    which_trajs=list(range(n_traj)),
    percent_time_length=1.0,
    leggauss_deg=5,
    nsave_rom=1,
)


def save_checkpoint(tensors, kind, path, Psi=None):
    """Save a self-contained ROM checkpoint (bases + physical operators)."""
    Psi = Phi if Psi is None else Psi
    B_r = Psi.T @ B_fom  # fixed reduced input operator, set by projection
    ckpt = {
        "kind": kind,
        "r": r,
        "poly_comp": poly_comp,
        "Phi": np.asarray(Phi),
        "Psi": np.asarray(Psi),
        "B_r": np.asarray(B_r),
        "tensors": [np.asarray(t) for t in tensors],
    }
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"saved -> {path}")


# %% 0) POD-Galerkin: project the known FOM operators onto Phi

printr("\n=== POD-Galerkin ===")
(Ar, Hr), (Br, Cr) = fom.assemble_petrov_galerkin_tensors(Phi, Phi)
printr(f"||A_r|| = {np.linalg.norm(Ar):.4e}, ||H_r|| = {np.linalg.norm(Hr):.4e}")
if rank == 0:
    save_checkpoint([Ar, Hr], "galerkin", os.path.join(models_dir, "galerkin_model.pkl"))

# %% 1) Operator inference

printr(f"\n=== OpInf (lambda = {LAMBDA:.1e}) ===")
# The ROM right-hand side is linear in (A_r, H_r), so the regularized OpInf
# problem is a linear least-squares problem with a closed-form solution --
# solve it directly rather than iterating.
opinf_model = PolynomialModel(r, poly_comp, dtype=dtype)
opinf = OpInfModule(training_data, opinf_model, projection, reg=LAMBDA)
printr(f"cost at H = A = 0: {gcost(opinf):.6e}")
solve_opinf(opinf)
printr(f"cost after direct solve: {gcost(opinf):.6e}")
data_cost = gcost(opinf) - LAMBDA * np.linalg.norm(opinf.rom.A_3) ** 2
printr(f"  data-fit term {data_cost:.6e} | "
       f"regularizer {LAMBDA * np.linalg.norm(opinf.rom.A_3) ** 2:.6e}")
printr(f"||A_r|| = {np.linalg.norm(opinf.rom.A_1):.4e}, "
       f"||H_r|| = {np.linalg.norm(opinf.rom.A_3):.4e}")
if rank == 0:
    save_checkpoint(opinf.rom.get_params(), "opinf",
                    os.path.join(models_dir, "opinf_model.pkl"))
