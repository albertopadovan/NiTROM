import os
import pickle
import time

import fom_class
import numpy as np

from nitrom.backend import mpi_comm_world, mpi_rank_size, set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import NitromModule, OpInfModule
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry
from nitrom.training_data import TrainingData, TrainingPool
from nitrom.utils import compute_POD

# Pure-numpy CPU run (lightweight; no torch). Trajectory-parallel with MPI:
set_backend("numpy")
dtype = np.float64
rank, world_size = mpi_rank_size()


def printr(*args, **kwargs) -> None:
    """Print on rank 0 only."""
    if rank == 0:
        print(*args, **kwargs)


def mpi_max(v: float) -> float:
    """Reduce a per-rank timing to the slowest rank (the parallel wall time)."""
    if world_size == 1:
        return v
    from mpi4py import MPI

    return mpi_comm_world().allreduce(float(v), op=MPI.MAX)


traj_path = "./trajectories/"
models_dir = "./models/"
r = 75  # reduced dimension
poly_comp = [1, 2]

# Setup the FOM
fom = fom_class.fom_class()

# Trajectory details
parameters = np.load(traj_path + "parameters.npy")
n_traj = len(parameters)

# Load the trajectories into a TrainingPool
pool = TrainingPool(
    n_traj=n_traj,
    fname_traj=traj_path + "traj_%03d.npy",
    fname_time=traj_path + "time.npy",
    dtype=dtype,
    fname_weights=traj_path + "weight_%03d.npy",
    fname_derivs=traj_path + "deriv_%03d.npy",
)

# Compute POD basis (rank r) of the pre-projected snapshots (dimension 300)
U, _, _ = compute_POD(pool, normalize=True)
Phi = U[:, :r]  # (300, r)
projection = LinearProjection([Phi, Phi])  # orthogonal (Psi = Phi)

training_data = TrainingData(
    pool,
    which_trajs=list(range(n_traj)),
    percent_time_length=1.0,
    leggauss_deg=5,
    nsave_rom=15,
)

# Initialize seed for models
ckpt_name = "opinf_model.pkl"
ckpt_path = os.path.join(models_dir, ckpt_name)
with open(ckpt_path, "rb") as f:
    ckpt = pickle.load(f)
tensors = [np.asarray(t, dtype=dtype) for t in ckpt["tensors"]]
A2r, A3r = tensors

seed = GasPolynomialModel(r, poly_comp, dtype=dtype)
seed.retract_general_tensors_to_gas_tensors([A2r, A3r])
gas_init = [*seed.get_params()]

# 1) Setup GasOpInf
gas_model = GasPolynomialModel(
    r,
    poly_comp,
    dtype=dtype,
    gas_params=gas_init,
)
gasopinf = OpInfModule(training_data, gas_model, projection, reg=1e-6)

# 2) Setup NiTROM
nitrom_model = PolynomialModel(
    r,
    poly_comp,
    dtype=dtype,
    tensors=(A2r, A3r),
)
registry = ParamRegistry(nitrom_model, projection)
nitrom = NitromModule(training_data, registry, fom=fom, n_substeps=15)

# 3) Setup GasNiTROM
gas_nitrom_model = GasPolynomialModel(
    r,
    poly_comp,
    dtype=dtype,
    gas_params=gas_init,
)
registry_gas = ParamRegistry(gas_nitrom_model, projection)
gasnitrom = NitromModule(training_data, registry_gas, fom=fom, n_substeps=15)


def time_module(module, name, num_calls=50):
    # Warm-up calls
    _ = module()
    _ = module.gradient()

    # Time cost evaluations (each rank times its own trajectory shard)
    t0 = time.perf_counter()
    for _ in range(num_calls):
        _ = module()
    t_cost = mpi_max((time.perf_counter() - t0) / num_calls)

    # Time gradient evaluations
    t0 = time.perf_counter()
    for _ in range(num_calls):
        _ = module.gradient()
    t_grad = mpi_max((time.perf_counter() - t0) / num_calls)

    printr(f"{name:<12} | {t_cost:12.6f} | {t_grad:12.6f}")
    return t_cost, t_grad


if __name__ == "__main__":
    num_calls = 50
    printr(f"Timing average of {num_calls} calls for cost and gradient functions...\n")
    printr(f"Running with {world_size} MPI rank(s)")
    printr(f"{'Method':<12} | {'Avg Cost (s)':<12} | {'Avg Grad (s)':<12}")
    printr("-" * 48)

    cost_gasopinf, grad_gasopinf = time_module(gasopinf, "GasOpInf", num_calls)
    cost_nitrom, grad_nitrom = time_module(nitrom, "NiTROM", num_calls)
    cost_gasnitrom, grad_gasnitrom = time_module(gasnitrom, "GasNiTROM", num_calls)
