import os
import pickle
import time
import numpy as np

import fom_class
from nitrom.backend import mpi_allreduce_scalar, mpi_rank_size, set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import OpInfModule, solve_opinf, train
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry
from nitrom.training_data import TrainingData, TrainingPool
from nitrom.utils import compute_POD

# Pure-numpy CPU run (lightweight; no torch). Trajectory-parallel with MPI:
set_backend("numpy")
dtype = np.float64
rank, world_size = mpi_rank_size()


def printr(*a, **k):
    if rank == 0:
        print(*a, **k)


def gcost(m):
    c = float(m())
    return mpi_allreduce_scalar(c) if world_size > 1 else c


# Cavity physical dimensions/parameters
Lx = 1
Ly = 1
Nx = 100
Ny = 100
dx = Lx / Nx
dy = Ly / Ny
Re = 8300

n = 400
dt = 1.0 / n

# Setup the FOM
fom = fom_class.fom_class()

# Trajectory details
traj_path = "./trajectories/"
parameters = np.load(traj_path + "parameters.npy")
n_traj = len(parameters)

r = 50  # reduced dimension
poly_comp = [1, 2]

# Load the trajectories into a TrainingPool
pool = TrainingPool(
    n_traj=n_traj,
    fname_traj=traj_path + "traj_%03d.npy",
    fname_time=traj_path + "time.npy",
    dtype=dtype,
    fname_weights=traj_path + "weight_%03d.npy",
)

# Compute POD basis (rank r) of the pre-projected snapshots (dimension 300)
U, _, _ = compute_POD(pool, normalize=True)
Phi = U[:, :r]  # (300, r)
projection = LinearProjection([Phi, Phi])  # orthogonal (Psi = Phi)

models_dir = "./models/"


def save_checkpoint(tensors, kind, path, gas_params=None) -> None:
    """Save a self-contained ROM checkpoint (POD basis + physical operators)."""
    ckpt = {
        "kind": kind,
        "r": r,
        "poly_comp": poly_comp,
        "forcing_config": None,
        "Phi": np.asarray(Phi),
        "tensors": [np.asarray(t) for t in tensors],
        "gas_params": gas_params,
    }
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"saved -> {path}")


training_data = TrainingData(
    pool,
    which_trajs=list(range(n_traj)),
    percent_time_length=0.5,
    leggauss_deg=5,
    nsave_rom=1,
)

# 1) Solve standard OpInf analytically
printr("\n=== OpInf ===")
opinf_model = PolynomialModel(r, poly_comp, dtype=dtype)
opinf = OpInfModule(training_data, opinf_model, projection, reg=1e-3)
solve_opinf(opinf)

if rank == 0:
    os.makedirs(models_dir, exist_ok=True)
    save_checkpoint(
        [np.copy(np.asarray(t)) for t in opinf_model.get_params()],
        "opinf",
        os.path.join(models_dir, "opinf_model.pkl"),
    )
A2r, A3r = opinf_model.get_params()

# 2) Train GAS-constrained OpInf, initialized from Galerkin
printr("\n=== GAS-OpInf ===")
seed = GasPolynomialModel(r, poly_comp, dtype=dtype)
seed.retract_general_tensors_to_gas_tensors([A2r, A3r], use_P_I=True)
gas_init = [*seed.get_params()]

gas_model = GasPolynomialModel(
    r,
    poly_comp,
    dtype=dtype,
    gas_params=gas_init,
)
gas = OpInfModule(training_data, gas_model, projection, reg=1e-3)

t0 = time.perf_counter()
train(gas, n_epochs=10000, lr=1.0, optimizer_type="lbfgs", print_every=259, tol=1e-14)
gas_opinf_time = time.perf_counter() - t0
printr(f"training time: {gas_opinf_time:.4f} s")

if rank == 0:
    gas_params = [np.copy(np.asarray(t)) for t in gas_model.get_params()]
    best_gas_physical_tensors = [
        np.copy(np.asarray(t)) for t in gas_model.model.get_params()
    ]
    save_checkpoint(
        best_gas_physical_tensors,
        "gas",
        os.path.join(models_dir, "gas_opinf_model.pkl"),
        gas_params=gas_params,
    )

    gas_opinf_dict = {
        "iters": np.arange(len(gas.loss_history)),
        "loss": gas.loss_history,
        "gradnorm": gas.gradnorm_history,
        "time": gas_opinf_time,
    }
    with open(os.path.join(models_dir, "gas_opinf_history.pkl"), "wb") as f:
        pickle.dump(gas_opinf_dict, f)
