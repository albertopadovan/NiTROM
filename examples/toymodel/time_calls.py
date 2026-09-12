import os
import pickle
import time
import numpy as np

import fom_class
from nitrom.backend import set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import OpInfModule, NitromModule
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry
from nitrom.training_data import TrainingData, TrainingPool
from nitrom.utils import compute_POD

# Ensure we run in numpy backend
set_backend("numpy")
dtype = np.float64

# Configuration matching train_nitrom.py and train_opinf.py
traj_path = "./trajectories/"
n_traj = 4
r = 2  # reduced dimension
poly_comp = [1, 2]

# Load trajectories
pool = TrainingPool(
    n_traj=n_traj,
    fname_traj=traj_path + "traj_%03d.npy",
    fname_time=traj_path + "time.npy",
    dtype=dtype,
    fname_weights=traj_path + "weight_%03d.npy",
    fname_forcing=traj_path + "forcing_%03d.pkl",
    fname_derivs=traj_path + "deriv_%03d.npy",
)

# Compute POD basis
U, _, _ = compute_POD(pool, normalize=True)
Phi = U[:, :r]  # (N, r)
projection = LinearProjection([Phi, Phi])  # orthogonal (Psi = Phi)

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

# Setup FOM
beta = 20.0
A2 = np.diag(np.array([-1.0, -2.0, -5.0], dtype=dtype))
A3 = np.zeros((3, 3, 3), dtype=dtype)
A3[:, :, -1] = np.diag(np.array([beta, beta, 0.0], dtype=dtype))
B_op = np.ones((3, 1), dtype=dtype)
C = np.ones((1, 3), dtype=dtype)
fom = fom_class.full_order_model(A2, A3, B_op, C, dtype=dtype)

(A2r, A3r), (Br, _) = fom.assemble_petrov_galerkin_tensors(Phi, Phi)

# Initialize seed for GAS-based models
seed = GasPolynomialModel(r, poly_comp, dtype=dtype)
seed.retract_general_tensors_to_gas_tensors([A2r, A3r])
gas_init = [*seed.get_params(), np.copy(Br)]

# 1) Setup GasOpInf
gas_model = GasPolynomialModel(
    r, poly_comp, dtype=dtype, gas_params=gas_init, forcing_config=forcing_config,
)
gasopinf = OpInfModule(training_data, gas_model, projection, reg=1e-6)

# 2) Setup NiTROM
nitrom_model = PolynomialModel(
    r, poly_comp, dtype=dtype, forcing_config=forcing_config, tensors=(A2r, A3r, Br),
)
registry = ParamRegistry(nitrom_model, projection)
nitrom = NitromModule(training_data, registry, fom=fom, n_substeps=15, adjoint_method='continuous')

# 3) Setup GasNiTROM
gas_nitrom_model = GasPolynomialModel(
    r, poly_comp, dtype=dtype, gas_params=gas_init, forcing_config=forcing_config,
)
registry_gas = ParamRegistry(gas_nitrom_model, projection)
gasnitrom = NitromModule(training_data, registry_gas, fom=fom, n_substeps=15, adjoint_method='continuous')


def time_module(module, name, num_calls=50):
    # Warm-up calls
    _ = module()
    _ = module.gradient()

    # Time cost evaluations
    t0 = time.perf_counter()
    for _ in range(num_calls):
        _ = module()
    t_cost = (time.perf_counter() - t0) / num_calls

    # Time gradient evaluations
    t0 = time.perf_counter()
    for _ in range(num_calls):
        _ = module.gradient()
    t_grad = (time.perf_counter() - t0) / num_calls

    print(f"{name:<12} | {t_cost:12.6e} | {t_grad:12.6e}")
    return t_cost, t_grad


if __name__ == "__main__":
    num_calls = 50
    print(f"Timing average of {num_calls} calls for cost and gradient functions...\n")
    print(f"{'Method':<12} | {'Avg Cost (s)':<12} | {'Avg Grad (s)':<12}")
    print("-" * 48)

    cost_gasopinf, grad_gasopinf = time_module(gasopinf, "GasOpInf", num_calls)
    cost_nitrom, grad_nitrom = time_module(nitrom, "NiTROM", num_calls)
    cost_gasnitrom, grad_gasnitrom = time_module(gasnitrom, "GasNiTROM", num_calls)
