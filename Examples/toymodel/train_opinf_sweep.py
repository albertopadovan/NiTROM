import os
import pickle

import fom_class
import numpy as np

from nitrom.backend import mpi_allreduce_scalar, mpi_rank_size, set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import OpInfModule, solve_opinf, train, NitromModule
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry
from nitrom.training_data import TrainingData, TrainingPool
from nitrom.utils import compute_POD

# Pure-numpy CPU run (lightweight; no torch). Trajectory-parallel with MPI:
#     mpiexec -n 4 python train_opinf_sweep.py
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
n_traj = 4
r = 2  # reduced dimension
poly_comp = [1, 2]

# Load the trajectories into a TrainingPool
pool = TrainingPool(
    n_traj=n_traj,
    fname_traj=traj_path + "traj_%03d.npy",
    fname_time=traj_path + "time.npy",
    dtype=dtype,
    fname_weights=traj_path + "weight_%03d.npy",
    fname_forcing=traj_path + "forcing_%03d.pkl",
    fname_derivs=traj_path + "deriv_%03d.npy",
)

# POD basis (rank r)
U, _, _ = compute_POD(pool, normalize=True)
Phi = U[:, :r]  # (N, r)
projection = LinearProjection([Phi, Phi])  # orthogonal (Psi = Phi)

# B = encode(B_fom) (fixed, not learned); the toy FOM is driven by B_fom=ones(N,1).
B_fom = np.ones((Phi.shape[0], 1), dtype=dtype)
B_r = projection.encode(B_fom.T).T  # fixed reduced input operator, (r, m)
forcing_config = {"forcing_exists": True, "B": B_r, "m": B_fom.shape[1]}

models_dir = "./models_discrete_adjoint/"


def save_checkpoint(tensors, kind, path, gas_params=None) -> None:
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


training_data = TrainingData(
    pool,
    which_trajs=list(range(n_traj)),
    percent_time_length=1.0,
    leggauss_deg=5,
    nsave_rom=1,
)

# Setup the FOM
beta = 20.0
A2 = np.diag(np.array([-1.0, -2.0, -5.0], dtype=dtype))
A3 = np.zeros((3, 3, 3), dtype=dtype)
A3[:, :, -1] = np.diag(np.array([beta, beta, 0.0], dtype=dtype))
B = np.ones((3, 1), dtype=dtype)
C = np.ones((1, 3), dtype=dtype)
fom = fom_class.full_order_model(A2, A3, B, C, dtype=dtype)

(A2r, A3r), (Br, _) = fom.assemble_petrov_galerkin_tensors(Phi, Phi)

# Sweep range
regs = np.logspace(-12, -4, 130)

best_opinf_reg = None
best_opinf_cost = float("inf")
best_opinf_tensors = None

best_gas_reg = None
best_gas_cost = float("inf")
best_gas_params = None
best_gas_physical_tensors = None

printr(f"Running sweep over {len(regs)} regularization parameters from 1e-8 to 1e-2...")
printr("-" * 75)
printr(f"{'Regularization':<20} | {'OpInf NiTROM Cost':<22} | {'GAS-OpInf NiTROM Cost':<22}")
printr("-" * 75)

gas_init = None
for reg in regs:
    # 1) Solve standard OpInf analytically
    opinf_model = PolynomialModel(r, poly_comp, dtype=dtype, forcing_config=forcing_config)
    opinf = OpInfModule(training_data, opinf_model, projection, reg=reg)
    opinf.set_unlearnable("B")
    solve_opinf(opinf)

    # Evaluate standard OpInf NiTROM-based cost
    opinf_registry = ParamRegistry(opinf_model, projection)
    opinf_nitrom = NitromModule(training_data, opinf_registry, fom=fom, n_substeps=30)
    opinf_cost = gcost(opinf_nitrom)

    if opinf_cost < best_opinf_cost:
        best_opinf_cost = opinf_cost
        best_opinf_reg = reg
        best_opinf_tensors = [np.copy(np.asarray(t)) for t in opinf_model.get_params()]

    # 2) Train GAS-constrained OpInf, initialized from Galerkin
    if gas_init is None:
        seed = GasPolynomialModel(r, poly_comp, dtype=dtype)
        seed.retract_general_tensors_to_gas_tensors([A2r, A3r], optimize_F=True, F_cond_penalty=1e-2)
        gas_init = [*seed.get_params(), np.copy(B_r)]

    gas_model = GasPolynomialModel(
        r, poly_comp, dtype=dtype, gas_params=gas_init, forcing_config=forcing_config,
    )
    gas = OpInfModule(training_data, gas_model, projection, reg=reg)
    gas.set_unlearnable("B")

    # Train GAS-OpInf silently
    train(gas, n_epochs=1000, lr=1.0, optimizer_type="lbfgs", print_every=0, tol=1e-10)

    # Save the current optimized parameters for the next iteration (warm-start)
    gas_init = [np.copy(np.asarray(t)) for t in gas_model.get_params()]

    # Evaluate GAS-OpInf NiTROM-based cost
    gas_registry = ParamRegistry(gas_model, projection)
    gas_nitrom = NitromModule(training_data, gas_registry, fom=fom, n_substeps=30)
    gas_cost = gcost(gas_nitrom)

    if gas_cost < best_gas_cost:
        best_gas_cost = gas_cost
        best_gas_reg = reg
        best_gas_params = [np.copy(np.asarray(t)) for t in gas_model.get_params()]
        best_gas_physical_tensors = [np.copy(np.asarray(t)) for t in gas_model.model.get_params()]

    printr(f"{reg:20.6e} | {opinf_cost:22.6e} | {gas_cost:22.6e}")

printr("-" * 75)
printr(f"Optimal standard OpInf regularization: {best_opinf_reg:.6e} with NiTROM Cost: {best_opinf_cost:.6e}")
printr(f"Optimal GAS-OpInf regularization:      {best_gas_reg:.6e} with NiTROM Cost: {best_gas_cost:.6e}")
printr("-" * 75)

if rank == 0:
    os.makedirs(models_dir, exist_ok=True)
    if best_opinf_tensors is not None:
        save_checkpoint(best_opinf_tensors, "opinf", os.path.join(models_dir, "opinf_model.pkl"))
    if best_gas_params is not None:
        save_checkpoint(
            best_gas_physical_tensors, "gas",
            os.path.join(models_dir, "gas_opinf_model.pkl"), gas_params=best_gas_params,
        )
