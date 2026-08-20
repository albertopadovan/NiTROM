import os
import pickle
import time

import fom_class
import numpy as np
import torch

from nitrom.backend import set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import NitromModule, OpInfModule
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry
from nitrom.training_data import TrainingData, TrainingPool
from nitrom.utils import compute_POD

# Ensure we run in torch backend
set_backend("torch")
dtype = torch.float64
device = "cuda" if torch.cuda.is_available() else "cpu"

traj_path = "./trajectories/"
models_dir = "./models/"
r = 50  # reduced dimension
poly_comp = [1, 2]

# Setup the FOM
fom = fom_class.fom_class()

# Trajectory details
parameters = torch.tensor(
    np.load(traj_path + "parameters.npy"), dtype=dtype, device=device
)
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
tensors = [torch.tensor(t, dtype=dtype, device=device) for t in ckpt["tensors"]]
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

    # Time cost evaluations
    t0 = time.perf_counter()
    for _ in range(num_calls):
        _ = module()
    t_cost = (time.perf_counter() - t0) / num_calls / n_traj

    # Time gradient evaluations
    t0 = time.perf_counter()
    for _ in range(num_calls):
        _ = module.gradient()
    t_grad = (time.perf_counter() - t0) / num_calls / n_traj

    print(f"{name:<12} | {t_cost:12.6f} | {t_grad:12.6f}")
    return t_cost, t_grad


if __name__ == "__main__":
    num_calls = 5
    print(f"Timing average of {num_calls} calls for cost and gradient functions...\n")
    print(f"{'Method':<12} | {'Avg Cost (s)':<12} | {'Avg Grad (s)':<12}")
    print("-" * 48)

    cost_gasopinf, grad_gasopinf = time_module(gasopinf, "GasOpInf", num_calls)
    cost_nitrom, grad_nitrom = time_module(nitrom, "NiTROM", num_calls)
    cost_gasnitrom, grad_gasnitrom = time_module(gasnitrom, "GasNiTROM", num_calls)
