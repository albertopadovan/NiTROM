import os
import pickle
import time

import fom_class
import numpy as np

from nitrom.backend import mpi_allreduce_scalar, mpi_rank_size, set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import OpInfModule, train
from nitrom.projections.linear_projection import LinearProjection
from nitrom.training_data import TrainingData, TrainingPool
from nitrom.utils import compute_POD

# Pure-numpy CPU run (lightweight; no torch).  Trajectory-parallel with MPI:
#     mpiexec -n 4 python train_opinf.py
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
models_dir = "./models_discrete_adjoint/"
n_traj = 4
r = 2  # reduced dimension
poly_comp = [1, 2]

if rank == 0:
    os.makedirs(models_dir, exist_ok=True)


def save_checkpoint(tensors, kind, path, gas_params=None):
    """Save a self-contained ROM checkpoint (POD basis + physical operators)."""
    ckpt = {
        "kind": kind,
        "r": r,
        "poly_comp": poly_comp,
        "forcing_config": forcing_config,
        "Phi": np.asarray(Phi),
        "tensors": [np.asarray(t) for t in tensors],
        "gas_params": gas_params,
    }
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"saved -> {path}")


# %% Load the trajectories into a TrainingPool

# rank/world_size are auto-detected from MPI, so they need not be passed here.
pool = TrainingPool(
    n_traj=n_traj,
    fname_traj=traj_path + "traj_%03d.npy",
    fname_time=traj_path + "time.npy",
    dtype=dtype,
    fname_weights=traj_path + "weight_%03d.npy",
    fname_forcing=traj_path + "forcing_%03d.pkl",
    fname_derivs=traj_path + "deriv_%03d.npy",
)

# %% POD basis (rank r)

U, _, _ = compute_POD(pool, normalize=True)
Phi = U[:, :r]  # (N, r)
projection = LinearProjection([Phi, Phi])  # orthogonal (Psi = Phi)

# B = encode(B_fom) (fixed, not learned); the toy FOM is driven by B_fom=ones(N,1).
B_fom = np.ones((Phi.shape[0], 1), dtype=dtype)
B_r = projection.encode(B_fom.T).T  # fixed reduced input operator, (r, m)
forcing_config = {"forcing_exists": True, "B": B_r, "m": B_fom.shape[1]}

training_data = TrainingData(
    pool,
    which_trajs=list(range(n_traj)),
    percent_time_length=1.0,
    leggauss_deg=5,
    nsave_rom=1,
)

# %% 0) POD-Galerkin model: project the (known) FOM operators onto Phi.

printr("=== POD-Galerkin ===")
beta = 20.0
A2 = np.diag(np.array([-1.0, -2.0, -5.0], dtype=dtype))
A3 = np.zeros((3, 3, 3), dtype=dtype)
A3[:, :, -1] = np.diag(np.array([beta, beta, 0.0], dtype=dtype))
B = np.ones((3, 1), dtype=dtype)
C = np.ones((1, 3), dtype=dtype)
fom = fom_class.full_order_model(A2, A3, B, C, dtype=dtype)

# Galerkin projection (Psi = Phi): (A_r, H_r), (B_r, C_r).
(A2r, A3r), (Br, _) = fom.assemble_petrov_galerkin_tensors(Phi, Phi)
if rank == 0:
    save_checkpoint([A2r, A3r, Br], "galerkin", os.path.join(models_dir, "galerkin_model.pkl"))

# %% 1) Train standard operator inference

printr("\n=== OpInf ===")
opinf_model = PolynomialModel(r, poly_comp, dtype=dtype, forcing_config=forcing_config)
opinf = OpInfModule(training_data, opinf_model, projection, reg=2.983647e-07)
opinf.set_unlearnable("B")  # B = Phi^T B_fom is fixed, not trained
printr(f"initial cost: {gcost(opinf):.6e}")
train(opinf, n_epochs=200, lr=1.0, optimizer_type="lbfgs", print_every=1, tol=1e-14)
printr(f"final cost:   {gcost(opinf):.6e}")
opinf._sync_to_rom()
if rank == 0:
    save_checkpoint(opinf.rom.get_params(), "opinf", os.path.join(models_dir, "opinf_model.pkl"))

# %% 2) Train GAS-constrained OpInf, initialized from the OpInf operators

printr("\n=== GAS-OpInf (initialized from OpInf) ===")
# Retract the trained OpInf operators (A = A_1, H = A_2) onto the GAS manifold.
seed = GasPolynomialModel(r, poly_comp, dtype=dtype)
seed.retract_general_tensors_to_gas_tensors([opinf.A_1, opinf.A_2])
gas_init = [*seed.get_params(), np.copy(opinf.B)]

gas_model = GasPolynomialModel(
    r, poly_comp, dtype=dtype, gas_params=gas_init, forcing_config=forcing_config,
)
gas = OpInfModule(training_data, gas_model, projection, reg=0.0)
gas.set_unlearnable("B")  # B = Phi^T B_fom is fixed, not trained
printr(f"initial cost: {gcost(gas):.6e}")
t0 = time.perf_counter()
train(gas, n_epochs=1000, lr=1.0, optimizer_type="lbfgs", print_every=1, tol=1e-14)
gas_opinf_time = time.perf_counter() - t0
printr(f"training time: {gas_opinf_time:.4f} s")
printr(f"final cost:   {gcost(gas):.6e}")
gas._sync_to_rom()
gas_params = [np.asarray(t) for t in gas.rom.get_params()]
if rank == 0:
    save_checkpoint(
        gas.rom.model.get_params(), "gas",
        os.path.join(models_dir, "gas_opinf_model.pkl"), gas_params=gas_params,
    )
gas_opinf_loss = gas.loss_history
gas_opinf_gradnorm = gas.gradnorm_history
gas_opinf_iters = np.arange(len(gas_opinf_loss))
gas_opinf_dict = { "iters": gas_opinf_iters,
                    "loss": gas_opinf_loss,
                    "gradnorm": gas_opinf_gradnorm,
                    "time": gas_opinf_time }
if rank == 0:
    with open(os.path.join(models_dir, "gas_opinf_history.pkl"), "wb") as f:
        pickle.dump(gas_opinf_dict, f)
