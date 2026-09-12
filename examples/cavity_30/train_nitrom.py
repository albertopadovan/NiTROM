import os
import pickle
import time
import numpy as np

import classes_cavity
from nitrom.backend import mpi_allreduce_scalar, mpi_rank_size, set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import NitromModule, train
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry
from nitrom.training_data import TrainingData, TrainingPool
from nitrom.utils import compute_POD

# Pure-numpy CPU run (lightweight; no torch). Trajectory-parallel with MPI:
set_backend("numpy")
dtype = np.float64
rank, world_size = mpi_rank_size()

traj_path = "./trajectories/"
models_dir = "./models/"
r = 30  # reduced dimension
poly_comp = [1, 2]
shift_start_times = [0.0]  # custom start times for each shift window

# Initialization model for GAS-NiTROM: "galerkin", "gas_opinf", or "nitrom".
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


# Cavity physical dimensions/parameters
Lx = 1
Ly = 1
Nx = 100
Ny = 100
dx = Lx/Nx
dy = Ly/Ny
Re = 8300

n = 400
dt = 1.0/n

# Setup the flow & FOM
flow = classes_cavity.flow_class(Lx, Ly, Nx, Ny, Re)
lops = classes_cavity.linear_operators_2D(flow, dt)
flow.q_sbf = np.load("bflow_Re%d_Nx%d_Ny%d.npy" % (Re, Nx, Ny))
fom = classes_cavity.fom_class(flow, lops)
fom.assemble_forcing_profile(0.95, 0.05)
B = fom.f.copy()  # shape (19700,)

# Trajectory details
amps = np.load(traj_path + "amps.npy")
n_traj = len(amps)
phi_pre = np.load(traj_path + "phi_pre.npy")  # (19700, 200)

# Load the trajectories into a TrainingPool
pool = TrainingPool(
    n_traj=n_traj,
    fname_traj=traj_path + "traj_%03d.npy",
    fname_time=traj_path + "time.npy",
    dtype=dtype,
    fname_weights=traj_path + "weight_%03d.npy",
    fname_derivs=traj_path + "deriv_%03d.npy",
    shift_start_times=shift_start_times,
)

# Compute POD basis (rank r) of the pre-projected snapshots (dimension 200)
U, _, _ = compute_POD(pool, normalize=True)
Phi = U[:, :r]  # (200, r)
projection = LinearProjection([Phi, Phi])  # orthogonal (Psi = Phi)

training_data = TrainingData(
    pool,
    which_trajs=list(range(n_traj)),
    percent_time_length=1.0,
    leggauss_deg=5,
    nsave_rom=15,
    shift_start_times=shift_start_times,
)

# Galerkin projection (Psi = Phi): (A_r, H_r)
phi_tot = phi_pre @ Phi
psi_tot = phi_pre @ Phi
(A2r, A3r), _ = fom.assemble_petrov_galerkin_tensors(phi_tot, psi_tot, B, [0,0,1,0,0,0,0,0])

if rank == 0:
    save_checkpoint(
        [A2r, A3r], "galerkin",
        os.path.join(models_dir, "galerkin_model.pkl"), Phi, Phi,
    )

# %% 1) Train standard NiTROM
printr("\n=== NiTROM ===")
nitrom_model = PolynomialModel(
    r, poly_comp, dtype=dtype, tensors=(A2r, A3r),
)
registry = ParamRegistry(nitrom_model, projection)

# Start with percent_time_length = 0.05
td_init = TrainingData(
    pool,
    which_trajs=list(range(n_traj)),
    percent_time_length=0.05,
    leggauss_deg=5,
    nsave_rom=15,
    shift_start_times=shift_start_times,
)
nitrom = NitromModule(td_init, registry, fom=fom, n_substeps=15, adjoint_method='discrete')
nitrom.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])

# Alternating optimization parameters
n_outer_iterations = 20
epochs_bases = 25
epochs_operators = 25

all_nitrom_loss = []
all_nitrom_gradnorm = []

printr(f"initial cost: {gcost(nitrom):.6e}")

t0 = time.perf_counter()

# Loop percent_time_length from 5% to 100% in steps of 5%
percents = np.arange(0.05, 1.05, 0.05)
for p_idx, percent in enumerate(percents):
    printr(f"\n--- NiTROM Stage {p_idx + 1}/{len(percents)} (Time length: {percent * 100:.1f}%) ---")
    td_slice = TrainingData(
        pool,
        which_trajs=list(range(n_traj)),
        percent_time_length=percent,
        leggauss_deg=5,
        nsave_rom=15,
        shift_start_times=shift_start_times,
    )
    update_module_training_data(nitrom, td_slice)
    
    # Alternating optimization
    for outer_iter in range(n_outer_iterations):
        printr(f"  Outer Iteration {outer_iter + 1}/{n_outer_iterations}:")
        
        # Optimize bases (Phi, Psi) only
        printr("    Optimizing bases (Phi, Psi)...")
        set_optimize_bases(nitrom)
        train(nitrom, n_epochs=epochs_bases, lr=1.0, optimizer_type="lbfgs", print_every=1, tol=1e-14)
        all_nitrom_loss.extend(nitrom.loss_history)
        all_nitrom_gradnorm.extend(nitrom.gradnorm_history)
        
        # Optimize operators only
        printr("    Optimizing operators...")
        set_optimize_operators(nitrom)
        train(nitrom, n_epochs=epochs_operators, lr=1.0, optimizer_type="lbfgs", print_every=1, tol=1e-14)
        all_nitrom_loss.extend(nitrom.loss_history)
        all_nitrom_gradnorm.extend(nitrom.gradnorm_history)

nitrom_time = time.perf_counter() - t0
printr(f"training time: {nitrom_time:.4f} s")
printr(f"final cost:   {gcost(nitrom):.6e}")

nitrom._sync_to_registry()
if rank == 0:
    save_checkpoint(
        nitrom_model.get_params(), "nitrom",
        os.path.join(models_dir, "nitrom_model.pkl"),
        nitrom.projection.Phi, nitrom.projection.Psi,
    )

nitrom_loss = np.array(all_nitrom_loss)
nitrom_gradnorm = np.array(all_nitrom_gradnorm)
nitrom_iters = np.arange(len(nitrom_loss))
nitrom_dict = {
    "iters": nitrom_iters,
    "loss": nitrom_loss,
    "gradnorm": nitrom_gradnorm,
    "time": nitrom_time
}
if rank == 0:
    with open(os.path.join(models_dir, "nitrom_history.pkl"), "wb") as f:
        pickle.dump(nitrom_dict, f)

# %% 2) Train GAS-NiTROM
printr(f"\n=== GAS-NiTROM (initialized from {init_model}) ===")

ckpt_name = {
    "galerkin": "galerkin_model.pkl",
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
    r, poly_comp, dtype=dtype, gas_params=gas_init,
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
    shift_start_times=shift_start_times,
)
gas_nitrom = NitromModule(td_gas_init, registry_gas, fom=fom, n_substeps=15, adjoint_method='discrete')
gas_nitrom.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])

all_gas_nitrom_loss = []
all_gas_nitrom_gradnorm = []

printr(f"initial cost: {gcost(gas_nitrom):.6e}")

t0_gas = time.perf_counter()

# Loop percent_time_length from 5% to 100% in steps of 5%
for p_idx, percent in enumerate(percents):
    printr(f"\n--- GAS-NiTROM Stage {p_idx + 1}/{len(percents)} (Time length: {percent * 100:.1f}%) ---")
    td_slice = TrainingData(
        pool,
        which_trajs=list(range(n_traj)),
        percent_time_length=percent,
        leggauss_deg=5,
        nsave_rom=15,
        shift_start_times=shift_start_times,
    )
    update_module_training_data(gas_nitrom, td_slice)
    
    # Alternating optimization
    for outer_iter in range(n_outer_iterations):
        printr(f"  Outer Iteration {outer_iter + 1}/{n_outer_iterations}:")
        
        # Optimize bases (Phi, Psi) only
        printr("    Optimizing bases (Phi, Psi)...")
        set_optimize_bases(gas_nitrom)
        train(gas_nitrom, n_epochs=epochs_bases, lr=1.0, optimizer_type="lbfgs", print_every=1, tol=1e-14)
        all_gas_nitrom_loss.extend(gas_nitrom.loss_history)
        all_gas_nitrom_gradnorm.extend(gas_nitrom.gradnorm_history)
        
        # Optimize operators only
        printr("    Optimizing operators...")
        set_optimize_operators(gas_nitrom)
        train(gas_nitrom, n_epochs=epochs_operators, lr=1.0, optimizer_type="lbfgs", print_every=1, tol=1e-14)
        all_gas_nitrom_loss.extend(gas_nitrom.loss_history)
        all_gas_nitrom_gradnorm.extend(gas_nitrom.gradnorm_history)

gas_nitrom_time = time.perf_counter() - t0_gas
printr(f"training time: {gas_nitrom_time:.4f} s")
printr(f"final cost:   {gcost(gas_nitrom):.6e}")

gas_nitrom._sync_to_registry()
if rank == 0:
    gas_params = [np.asarray(t) for t in gas_nitrom_model.get_params()]
    save_checkpoint(
        gas_nitrom_model.model.get_params(), "gas_nitrom",
        os.path.join(models_dir, "gas_nitrom_model.pkl"),
        gas_nitrom.projection.Phi, gas_nitrom.projection.Psi, gas_params=gas_params,
    )

gas_nitrom_loss = np.array(all_gas_nitrom_loss)
gas_nitrom_gradnorm = np.array(all_gas_nitrom_gradnorm)
gas_nitrom_iters = np.arange(len(gas_nitrom_loss))
gas_nitrom_dict = {
    "iters": gas_nitrom_iters,
    "loss": gas_nitrom_loss,
    "gradnorm": gas_nitrom_gradnorm,
    "time": gas_nitrom_time
}
if rank == 0:
    with open(os.path.join(models_dir, "gas_nitrom_history.pkl"), "wb") as f:
        pickle.dump(gas_nitrom_dict, f)
