import os
import pickle
import time

import classes_cavity
import numpy as np

from nitrom.backend import mpi_allreduce_scalar, mpi_rank_size, set_backend
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
r = 50  # reduced dimension
poly_comp = [1, 2]
shift_start_times = [0.0]  # custom start times for each shift window

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


def save_training_checkpoint(
    path,
    p_idx,
    outer_iter,
    phase,
    loss_history,
    gradnorm_history,
    elapsed_time,
    tensors=None,
    gas_params=None,
    Phi=None,
    Psi=None,
) -> None:
    """Save resumable training state: position in the schedule, current model
    parameters, and loss/gradnorm histories. Written atomically so a job that
    is killed mid-write can never leave a corrupt checkpoint behind."""
    ckpt = {
        "p_idx": p_idx,
        "outer_iter": outer_iter,
        "phase": phase,
        "tensors": [np.asarray(t) for t in tensors] if tensors is not None else None,
        "gas_params": (
            [np.asarray(t) for t in gas_params] if gas_params is not None else None
        ),
        "Phi": np.asarray(Phi),
        "Psi": np.asarray(Psi),
        "loss_history": list(loss_history),
        "gradnorm_history": list(gradnorm_history),
        "elapsed_time": elapsed_time,
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(ckpt, f)
    os.replace(tmp_path, path)
    print(
        f"checkpoint -> {path} "
        f"(stage {p_idx + 1}, outer iter {outer_iter + 1}, phase={phase})"
    )


def load_training_checkpoint(path):
    """Load resumable training state, or None if no checkpoint exists yet."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


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
dx = Lx / Nx
dy = Ly / Ny
Re = 8300

n = 400
dt = 1.0 / n

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
printr(n_traj)
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

training_data = TrainingData(
    pool,
    which_trajs=list(range(n_traj)),
    percent_time_length=0.5,
    leggauss_deg=5,
    nsave_rom=15,
    shift_start_times=shift_start_times,
)

# Galerkin projection (Psi = Phi): (A_r, H_r)
phi_tot = phi_pre @ Phi
psi_tot = phi_pre @ Phi
(A2r, A3r), _ = fom.assemble_petrov_galerkin_tensors(
    phi_tot, psi_tot, B, [0, 0, 1, 0, 0, 0, 0, 0]
)

if rank == 0:
    save_checkpoint(
        [A2r, A3r],
        "galerkin",
        os.path.join(models_dir, "galerkin_model.pkl"),
        Phi,
        Phi,
    )

# Train standard NiTROM
printr("\n=== NiTROM ===")
checkpoint_path = os.path.join(models_dir, "nitrom_checkpoint.pkl")

# Alternating optimization parameters
n_outer_iterations = 20
epochs_bases = 25
epochs_operators = 25

# Loop percent_time_length from 5% to 50% in steps of 5%
percents = np.arange(0.05, 0.51, 0.05)

resume_state = load_training_checkpoint(checkpoint_path)
if resume_state is not None:
    start_p_idx = resume_state["p_idx"]
    start_outer_iter = resume_state["outer_iter"]
    start_phase = resume_state["phase"]
    if start_phase == "operators_done":
        # Both phases of that outer iteration are done; continue with the next one.
        start_outer_iter += 1
        start_phase = None
    printr(
        f"Resuming from checkpoint {checkpoint_path} "
        f"(stage {start_p_idx + 1}/{len(percents)}, "
        f"outer iter {start_outer_iter + 1}/{n_outer_iterations})"
    )
    tensors = [np.asarray(t, dtype=dtype) for t in resume_state["tensors"]]
    init_Phi = np.asarray(resume_state["Phi"], dtype=dtype)
    init_Psi = np.asarray(resume_state["Psi"], dtype=dtype)
    all_nitrom_loss = list(resume_state["loss_history"])
    all_nitrom_gradnorm = list(resume_state["gradnorm_history"])
    prior_elapsed = resume_state["elapsed_time"]
else:
    tensors = [A2r, A3r]
    init_Phi = Phi  # POD basis computed above
    init_Psi = Phi
    all_nitrom_loss = []
    all_nitrom_gradnorm = []
    prior_elapsed = 0.0
    start_p_idx = 0
    start_outer_iter = 0
    start_phase = None

nitrom_model = PolynomialModel(
    r,
    poly_comp,
    dtype=dtype,
    tensors=tensors,
)
projection = LinearProjection([init_Phi, init_Psi])
registry = ParamRegistry(nitrom_model, projection)

td_init = TrainingData(
    pool,
    which_trajs=list(range(n_traj)),
    percent_time_length=percents[start_p_idx],
    leggauss_deg=5,
    nsave_rom=15,
    shift_start_times=shift_start_times,
)
nitrom = NitromModule(
    td_init, registry, fom=fom, n_substeps=15, adjoint_method="discrete"
)
nitrom.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])

printr(f"initial cost: {gcost(nitrom):.6e}")

t0 = time.perf_counter()


def elapsed_time() -> float:
    return prior_elapsed + (time.perf_counter() - t0)


def checkpoint(p_idx, outer_iter, phase) -> None:
    if rank == 0:
        nitrom._sync_to_registry()
        save_training_checkpoint(
            checkpoint_path,
            p_idx=p_idx,
            outer_iter=outer_iter,
            phase=phase,
            tensors=nitrom_model.get_params(),
            Phi=nitrom.projection.Phi,
            Psi=nitrom.projection.Psi,
            loss_history=all_nitrom_loss,
            gradnorm_history=all_nitrom_gradnorm,
            elapsed_time=elapsed_time(),
        )


for p_idx, percent in enumerate(percents):
    if p_idx < start_p_idx:
        continue
    printr(
        f"\n--- NiTROM Stage {p_idx + 1}/{len(percents)} (Time length: {percent * 100:.1f}%) ---"
    )
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
    outer_start = start_outer_iter if p_idx == start_p_idx else 0
    for outer_iter in range(outer_start, n_outer_iterations):
        printr(f"  Outer Iteration {outer_iter + 1}/{n_outer_iterations}:")

        skip_bases = (
            p_idx == start_p_idx
            and outer_iter == start_outer_iter
            and start_phase == "bases_done"
        )
        if not skip_bases:
            # Optimize bases (Phi, Psi) only
            printr("    Optimizing bases (Phi, Psi)...")
            set_optimize_bases(nitrom)
            train(
                nitrom,
                n_epochs=epochs_bases,
                lr=1.0,
                optimizer_type="lbfgs",
                print_every=1,
                tol=1e-14,
            )
            all_nitrom_loss.extend(nitrom.loss_history)
            all_nitrom_gradnorm.extend(nitrom.gradnorm_history)
            checkpoint(p_idx, outer_iter, "bases_done")

        # Optimize operators only
        printr("    Optimizing operators...")
        set_optimize_operators(nitrom)
        train(
            nitrom,
            n_epochs=epochs_operators,
            lr=1.0,
            optimizer_type="lbfgs",
            print_every=1,
            tol=1e-14,
        )
        all_nitrom_loss.extend(nitrom.loss_history)
        all_nitrom_gradnorm.extend(nitrom.gradnorm_history)
        checkpoint(p_idx, outer_iter, "operators_done")

nitrom_time = elapsed_time()
printr(f"training time: {nitrom_time:.4f} s")
printr(f"final cost:   {gcost(nitrom):.6e}")

nitrom._sync_to_registry()
if rank == 0:
    save_checkpoint(
        nitrom_model.get_params(),
        "nitrom",
        os.path.join(models_dir, "nitrom_model.pkl"),
        nitrom.projection.Phi,
        nitrom.projection.Psi,
    )
    # Training finished; the resume checkpoint is no longer needed.
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

nitrom_loss = np.array(all_nitrom_loss)
nitrom_gradnorm = np.array(all_nitrom_gradnorm)
nitrom_iters = np.arange(len(nitrom_loss))
nitrom_dict = {
    "iters": nitrom_iters,
    "loss": nitrom_loss,
    "gradnorm": nitrom_gradnorm,
    "time": nitrom_time,
}
if rank == 0:
    with open(os.path.join(models_dir, "nitrom_history.pkl"), "wb") as f:
        pickle.dump(nitrom_dict, f)
