import os
import pickle
import time

import fom_class
import numpy as np

from nitrom.backend import mpi_allreduce_scalar, mpi_rank_size, set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import NitromModule, train
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry
from nitrom.training_data import TrainingData, TrainingPool
from nitrom.utils import compute_POD

# Pure-numpy CPU run (lightweight; no torch).  Trajectory-parallel with MPI:
#     mpiexec -n 4 python train_nitrom.py
set_backend("numpy")
dtype = np.float64
rank, world_size = mpi_rank_size()

traj_path = "./trajectories/"
adjoint_method = 'continuous'
models_dir = f"./models_{adjoint_method}_adjoint/"
n_traj = 4
r = 2  # reduced dimension
poly_comp = [1, 2]

# Initialization model for GAS-NiTROM: "galerkin", "gas_opinf", or "nitrom".
# "gas_opinf" requires train_opinf.py to have been run first (writes
# ./models/gas_opinf_model.pkl).
init_model = "gas_opinf"

if rank == 0:
    os.makedirs(models_dir, exist_ok=True)


def printr(*args, **kwargs) -> None:
    """Print on rank 0 only."""
    if rank == 0:
        print(*args, **kwargs)


def gcost(module) -> float:
    """Cost summed across ranks (``module()`` returns the local partial cost)."""
    c = float(module())
    return mpi_allreduce_scalar(c) if world_size > 1 else c


def save_checkpoint(tensors, kind, path, Phi, Psi, gas_params=None) -> None:
    """Save a self-contained NiTROM ROM checkpoint (trial/test bases + operators)."""
    ckpt = {
        "kind": kind,
        "r": r,
        "poly_comp": poly_comp,
        "forcing_config": forcing_config,
        "Phi": np.asarray(Phi),
        "Psi": np.asarray(Psi),
        "tensors": [np.asarray(t) for t in tensors],
        "gas_params": gas_params,
    }
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"saved -> {path}")


# %% Load the (rank-sharded) trajectories into a TrainingPool

pool = TrainingPool(
    n_traj=n_traj,
    fname_traj=traj_path + "traj_%03d.npy",
    fname_time=traj_path + "time.npy",
    dtype=dtype,
    fname_weights=traj_path + "weight_%03d.npy",
    fname_forcing=traj_path + "forcing_%03d.pkl",
    fname_derivs=traj_path + "deriv_%03d.npy",
)

# %% POD basis (rank r).  compute_POD gathers on root and, by default,
# broadcasts the basis to every rank -- so Phi is identical everywhere.

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
    nsave_rom=15,
)

# %% Setup the FOM
beta = 20.0
A2 = np.diag(np.array([-1.0, -2.0, -5.0], dtype=dtype))
A3 = np.zeros((3, 3, 3), dtype=dtype)
A3[:, :, -1] = np.diag(np.array([beta, beta, 0.0], dtype=dtype))
B_op = np.ones((3, 1), dtype=dtype)
C = np.ones((1, 3), dtype=dtype)
fom = fom_class.full_order_model(A2, A3, B_op, C, dtype=dtype)

# Galerkin projection (Psi = Phi): (A_r, H_r), (B_r, C_r).
(A2r, A3r), (Br, _) = fom.assemble_petrov_galerkin_tensors(Phi, Phi)
if rank == 0:
    save_checkpoint(
        [A2r, A3r, Br], "galerkin",
        os.path.join(models_dir, "galerkin_model.pkl"), Phi, Phi,
    )

# %% 1) Train standard NiTROM.  Phi lives on the Grassmann manifold, Psi on the
# Stiefel manifold; train() then uses the Riemannian L-BFGS (retraction +
# vector transport) for those, and the strong-Wolfe/Armijo step throughout.
printr("\n=== NiTROM ===")
nitrom_model = PolynomialModel(
    r, poly_comp, dtype=dtype, forcing_config=forcing_config, tensors=(A2r, A3r, Br),
)
registry = ParamRegistry(nitrom_model, projection)
nitrom = NitromModule(training_data, registry, fom=fom, n_substeps=15, adjoint_method=adjoint_method)
nitrom.set_unlearnable("B")  # B = Phi^T B_fom is fixed, not trained
nitrom.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])

printr(f"initial cost: {gcost(nitrom):.6e}")
t0 = time.perf_counter()
train(nitrom, n_epochs=400, lr=1.0, optimizer_type="lbfgs", print_every=1, tol=1e-14)
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

nitrom_loss = nitrom.loss_history
nitrom_gradnorm = nitrom.gradnorm_history
nitrom_iters = np.arange(len(nitrom_loss))
nitrom_dict = {"iters": nitrom_iters,
                "loss": nitrom_loss,
                "gradnorm": nitrom_gradnorm,
                "time": nitrom_time}
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
    # Retract the general operator tensors (A, H) onto the GAS manifold.
    seed = GasPolynomialModel(r, poly_comp, dtype=dtype)
    seed.retract_general_tensors_to_gas_tensors(tensors[:2])
    gas_init = [*seed.get_params(), np.copy(tensors[2])]

gas_nitrom_model = GasPolynomialModel(
    r, poly_comp, dtype=dtype, gas_params=gas_init, forcing_config=forcing_config,
)

# Start the projection from the loaded bases.
projection_gas = LinearProjection([init_Phi, init_Psi])
registry_gas = ParamRegistry(gas_nitrom_model, projection_gas)
gas_nitrom = NitromModule(training_data, registry_gas, fom=fom, n_substeps=15, adjoint_method=adjoint_method)
gas_nitrom.set_unlearnable("B")
gas_nitrom.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])

printr(f"initial cost: {gcost(gas_nitrom):.6e}")
t0_gas = time.perf_counter()
train(gas_nitrom, n_epochs=400, lr=1.0, optimizer_type="lbfgs", print_every=1, tol=1e-14)
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

gas_nitrom_loss = gas_nitrom.loss_history
gas_nitrom_gradnorm = gas_nitrom.gradnorm_history
gas_nitrom_iters = np.arange(len(gas_nitrom_loss))
gas_nitrom_dict = {"iters": gas_nitrom_iters,
                "loss": gas_nitrom_loss,
                "gradnorm": gas_nitrom_gradnorm,
                "time": gas_nitrom_time}
if rank == 0:
    with open(os.path.join(models_dir, "gas_nitrom_history.pkl"), "wb") as f:
        pickle.dump(gas_nitrom_dict, f)