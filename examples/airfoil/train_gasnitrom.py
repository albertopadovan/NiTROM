import os
import pickle
import time

import fom_class
import numpy as np

from nitrom.backend import mpi_allreduce_scalar, mpi_rank_size, set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.optimization import NitromModule, train
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry
from nitrom.training_data import TrainingData, TrainingPool

# Pure-numpy CPU run (lightweight; no torch). Trajectory-parallel with MPI:
set_backend("numpy")
dtype = np.float64
rank, world_size = mpi_rank_size()

traj_path = "./trajectories/"
models_dir = "./models/"
r = 75  # reduced dimension
poly_comp = [1, 2]

# Initialization model for GAS-NiTROM: "gas_opinf" or "nitrom".
init_model = "gas_opinf"

if rank == 0:
    os.makedirs(models_dir, exist_ok=True)


def printr(*args, **kwargs) -> None:
    """Print on rank 0 only."""
    if rank == 0:
        print(*args, **kwargs)


def gcost(module) -> float:
    """Cost summed across ranks."""
    c = float(module())
    return mpi_allreduce_scalar(c) if world_size > 1 else c


def save_checkpoint(tensors, kind, path, Phi, Psi, gas_params=None) -> None:
    """Save a self-contained NiTROM ROM checkpoint (trial/test bases + operators)."""
    ckpt = {
        "kind": kind,
        "r": r,
        "poly_comp": poly_comp,
        "forcing_config": None,
        "Phi": np.asarray(Phi),
        "Psi": np.asarray(Psi),
        "tensors": [np.asarray(t) for t in tensors],
        "gas_params": gas_params,
    }
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"saved -> {path}")


def set_optimize_bases(module):
    all_params = module.param_names
    bases = ["Phi", "Psi"]
    ops = [p for p in all_params if p not in bases]
    module.set_learnable(*bases)
    if ops:
        module.set_unlearnable(*ops)


def set_optimize_operators(module):
    all_params = module.param_names
    bases = ["Phi", "Psi"]
    ops = [p for p in all_params if p not in bases]
    module.set_unlearnable(*bases)
    if ops:
        module.set_learnable(*ops)


def update_module_training_data(module, td):
    module.training_data = td
    module.time = td.time
    module.forcing_fns = td.forcing_fns
    module.weights = td.weights


# Setup the flow & FOM
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
)

# Alternating optimization parameters
n_outer_iterations = 20
epochs_bases = 25
epochs_operators = 25

# Loop percent_time_length from 5% to 100% in steps of 5%
percents = np.arange(0.05, 1.01, 0.05)


# Train GAS-NiTROM
printr(f"\n=== GAS-NiTROM (initialized from {init_model}) ===")

ckpt_name = {
    "gas_opinf": "gas_opinf_model.pkl",
    "nitrom": "nitrom_model.pkl",
}[init_model]
ckpt_path = os.path.join(models_dir, ckpt_name)
printr(f"Loading initialization from {ckpt_path}...")
with open(ckpt_path, "rb") as f:
    ckpt = pickle.load(f)

init_Phi = np.asarray(ckpt["Phi"], dtype=dtype)
init_Psi = np.asarray(ckpt.get("Psi", ckpt["Phi"]), dtype=dtype)

if init_model == "gas_opinf":
    gas_init = [np.asarray(t, dtype=dtype) for t in ckpt["gas_params"]]
else:
    tensors = [np.asarray(t, dtype=dtype) for t in ckpt["tensors"]]
    seed = GasPolynomialModel(r, poly_comp, dtype=dtype)
    seed.retract_general_tensors_to_gas_tensors(tensors[:2], use_P_I=True)
    gas_init = [*seed.get_params()]

gas_nitrom_model = GasPolynomialModel(
    r,
    poly_comp,
    dtype=dtype,
    gas_params=gas_init,
)

projection_gas = LinearProjection([init_Phi, init_Psi])
registry_gas = ParamRegistry(gas_nitrom_model, projection_gas)
# Start with percent_time_length = 0.05
td_gas_init = TrainingData(
    pool,
    which_trajs=list(range(n_traj)),
    percent_time_length=0.05,
    leggauss_deg=5,
    nsave_rom=15,
)
gas_nitrom = NitromModule(
    td_gas_init, registry_gas, fom=fom, n_substeps=15, adjoint_method="discrete"
)
gas_nitrom.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])

all_gas_nitrom_loss = []
all_gas_nitrom_gradnorm = []

printr(f"initial cost: {gcost(gas_nitrom):.6e}")

t0_gas = time.perf_counter()

for p_idx, percent in enumerate(percents):
    printr(
        f"\n--- GAS-NiTROM Stage {p_idx + 1}/{len(percents)} (Time length: {percent * 100:.1f}%) ---"
    )
    td_slice = TrainingData(
        pool,
        which_trajs=list(range(n_traj)),
        percent_time_length=percent,
        leggauss_deg=5,
        nsave_rom=15,
    )
    update_module_training_data(gas_nitrom, td_slice)

    # Alternating optimization
    for outer_iter in range(n_outer_iterations):
        printr(f"  Outer Iteration {outer_iter + 1}/{n_outer_iterations}:")

        # Optimize bases (Phi, Psi) only
        printr("    Optimizing bases (Phi, Psi)...")
        set_optimize_bases(gas_nitrom)
        train(
            gas_nitrom,
            n_epochs=epochs_bases,
            lr=1.0,
            optimizer_type="lbfgs",
            print_every=1,
            tol=1e-14,
        )
        all_gas_nitrom_loss.extend(gas_nitrom.loss_history)
        all_gas_nitrom_gradnorm.extend(gas_nitrom.gradnorm_history)

        # Optimize operators only
        printr("    Optimizing operators...")
        set_optimize_operators(gas_nitrom)
        train(
            gas_nitrom,
            n_epochs=epochs_operators,
            lr=1.0,
            optimizer_type="lbfgs",
            print_every=1,
            tol=1e-14,
        )
        all_gas_nitrom_loss.extend(gas_nitrom.loss_history)
        all_gas_nitrom_gradnorm.extend(gas_nitrom.gradnorm_history)

gas_nitrom_time = time.perf_counter() - t0_gas
printr(f"training time: {gas_nitrom_time:.4f} s")
printr(f"final cost:   {gcost(gas_nitrom):.6e}")

gas_nitrom._sync_to_registry()
if rank == 0:
    gas_params = [np.asarray(t) for t in gas_nitrom_model.get_params()]
    save_checkpoint(
        gas_nitrom_model.model.get_params(),
        "gas_nitrom",
        os.path.join(models_dir, "gas_nitrom_model.pkl"),
        gas_nitrom.projection.Phi,
        gas_nitrom.projection.Psi,
        gas_params=gas_params,
    )

gas_nitrom_loss = np.array(all_gas_nitrom_loss)
gas_nitrom_gradnorm = np.array(all_gas_nitrom_gradnorm)
gas_nitrom_iters = np.arange(len(gas_nitrom_loss))
gas_nitrom_dict = {
    "iters": gas_nitrom_iters,
    "loss": gas_nitrom_loss,
    "gradnorm": gas_nitrom_gradnorm,
    "time": gas_nitrom_time,
}
if rank == 0:
    with open(os.path.join(models_dir, "gas_nitrom_history.pkl"), "wb") as f:
        pickle.dump(gas_nitrom_dict, f)
