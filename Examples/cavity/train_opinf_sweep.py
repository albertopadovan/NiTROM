import os
import pickle
import numpy as np

import classes_cavity
from nitrom.backend import mpi_allreduce_scalar, mpi_rank_size, set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import OpInfModule, solve_opinf, train, NitromModule
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
traj_path = "./trajectories/"
amps = np.load(traj_path + "amps.npy")
n_traj = len(amps)
phi_pre = np.load(traj_path + "phi_pre.npy")  # (19700, 200)

r = 50  # reduced dimension
poly_comp = [1, 2]

# Load the trajectories into a TrainingPool
pool = TrainingPool(
    n_traj=n_traj,
    fname_traj=traj_path + "traj_%03d.npy",
    fname_time=traj_path + "time.npy",
    dtype=dtype,
    fname_weights=traj_path + "weight_%03d.npy",
    fname_derivs=traj_path + "deriv_%03d.npy",
)

# Compute POD basis (rank r) of the pre-projected snapshots (dimension 200)
U, _, _ = compute_POD(pool, normalize=True)
Phi = U[:, :r]  # (200, r)
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

# Galerkin projection (Psi = Phi): (A_r, H_r)
phi_tot = phi_pre @ Phi
psi_tot = phi_pre @ Phi
(A2r, A3r), _ = fom.assemble_petrov_galerkin_tensors(phi_tot, psi_tot, B, [0,0,1,0,0,0,0,0])

# Sweep range
regs = np.logspace(1, 5, 50)

best_opinf_reg = None
best_opinf_cost = float("inf")
best_opinf_tensors = None

best_gas_reg = None
best_gas_cost = float("inf")
best_gas_params = None
best_gas_physical_tensors = None

printr(f"Running sweep over {len(regs)} regularization parameters from {regs[0]} to {regs[-1]}...")
printr("-" * 75)
printr(f"{'Regularization':<20} | {'OpInf NiTROM Cost':<22} | {'GAS-OpInf NiTROM Cost':<22}")
printr("-" * 75)

gas_init = None

for reg in regs:
    # 1) Solve standard OpInf analytically
    opinf_model = PolynomialModel(r, poly_comp, dtype=dtype)
    opinf = OpInfModule(training_data, opinf_model, projection, reg=reg)
    solve_opinf(opinf)

    # Evaluate standard OpInf NiTROM-based cost
    opinf_registry = ParamRegistry(opinf_model, projection)
    opinf_nitrom = NitromModule(training_data, opinf_registry, fom=fom, n_substeps=15)
    opinf_cost = gcost(opinf_nitrom)

    if opinf_cost < best_opinf_cost:
        best_opinf_cost = opinf_cost
        best_opinf_reg = reg
        best_opinf_tensors = [np.copy(np.asarray(t)) for t in opinf_model.get_params()]

    # 2) Train GAS-constrained OpInf, initialized from the previous solution (warm-start) or Galerkin
    epochs = 2500
    if gas_init is None:
        seed = GasPolynomialModel(r, poly_comp, dtype=dtype)
        seed.retract_general_tensors_to_gas_tensors([A2r, A3r], use_P_I=True)
        gas_init = [*seed.get_params()]
        epochs = 8000

    gas_model = GasPolynomialModel(
        r, poly_comp, dtype=dtype, gas_params=gas_init,
    )
    gas = OpInfModule(training_data, gas_model, projection, reg=reg)

    # Train GAS-OpInf silently
    train(gas, n_epochs=epochs, lr=1.0, optimizer_type="lbfgs", print_every=100, tol=1e-10)

    # Save the current optimized parameters for the next iteration (warm-start)
    gas_init = [np.copy(np.asarray(t)) for t in gas_model.get_params()]

    # Evaluate GAS-OpInf NiTROM-based cost
    gas_registry = ParamRegistry(gas_model, projection)
    gas_nitrom = NitromModule(training_data, gas_registry, fom=fom, n_substeps=15)
    gas_cost = gcost(gas_nitrom)

    if gas_cost < best_gas_cost:
        best_gas_cost = gas_cost
        best_gas_reg = reg
        best_gas_params = [np.copy(np.asarray(t)) for t in gas_model.get_params()]
        best_gas_physical_tensors = [np.copy(np.asarray(t)) for t in gas_model.model.get_params()]
        best_gas_loss = np.copy(np.asarray(gas.loss_history))
        best_gas_gradnorm = np.copy(np.asarray(gas.gradnorm_history))

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
        gas_opinf_dict = {
            "iters": np.arange(len(best_gas_loss)),
            "loss": best_gas_loss,
            "gradnorm": best_gas_gradnorm
        }
        with open(os.path.join(models_dir, "gas_opinf_history.pkl"), "wb") as f:
            pickle.dump(gas_opinf_dict, f)
