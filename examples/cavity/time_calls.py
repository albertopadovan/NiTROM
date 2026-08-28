import time
import numpy as np

import classes_cavity
from nitrom.backend import mpi_comm_world, mpi_rank_size, set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import OpInfModule, NitromModule
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
r = 50  # reduced dimension
poly_comp = [1, 2]

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
)

# Galerkin projection (Psi = Phi): (A_r, H_r)
phi_tot = phi_pre @ Phi
psi_tot = phi_pre @ Phi
(A2r, A3r), _ = fom.assemble_petrov_galerkin_tensors(phi_tot, psi_tot, B, [0,0,1,0,0,0,0,0])

# Initialize seed for GAS-based models
seed = GasPolynomialModel(r, poly_comp, dtype=dtype)
seed.retract_general_tensors_to_gas_tensors([A2r, A3r])
gas_init = [*seed.get_params()]

# 1) Setup GasOpInf
gas_model = GasPolynomialModel(
    r, poly_comp, dtype=dtype, gas_params=gas_init,
)
gasopinf = OpInfModule(training_data, gas_model, projection, reg=1e-6)

# 2) Setup NiTROM
nitrom_model = PolynomialModel(
    r, poly_comp, dtype=dtype, tensors=(A2r, A3r),
)
registry = ParamRegistry(nitrom_model, projection)
nitrom = NitromModule(training_data, registry, fom=fom, n_substeps=15)

# 3) Setup GasNiTROM
gas_nitrom_model = GasPolynomialModel(
    r, poly_comp, dtype=dtype, gas_params=gas_init,
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
