"""Movies of ROM predictions on the last training trajectory.

For the last trajectory in the training set, integrates each trained ROM
(OpInf, GasOpInf, NiTROM, GasNiTROM) forward from the true initial condition
and animates the resulting perturbation vorticity field (base flow excluded)
over the ROM's cropped domain, side by side with the true trajectory.
"""

import os
import pickle
import shutil

import fom_class
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from nitrom.backend import set_backend
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.plotting import set_plot_style
from nitrom.projections.linear_projection import LinearProjection
from nitrom.time_steppers.time_stepper import solve_ivp
from nitrom.training_data import TrainingPool

set_backend("numpy")
set_plot_style()

dtype = np.float64
rtol = 1e-4
atol = 1e-8

traj_path = "./trajectories/"
models_dir = "./models/"
movies_dir = "./movies/"
snapshot_fname = "snapshot_000074.npz"

# Spatial restriction used before the POD and stored in decoder.npz.
X_BOUNDS = (-4.0, 13.0)
Y_BOUNDS = (-3.0, 3.0)

# Frame stride for the movies; trim to taste.
FRAME_STRIDE = 1
FPS = 15

ROM_LABELS = {
    "opinf": "OpInf",
    "gas_opinf": "GasOpInf",
    "nitrom": "NiTROM",
    "gas_nitrom": "GasNiTROM",
}


def load_rom(fname):
    with open(os.path.join(models_dir, fname), "rb") as f:
        ckpt = pickle.load(f)
    rom = PolynomialModel(
        ckpt["r"],
        ckpt["poly_comp"],
        dtype=dtype,
        tensors=[np.asarray(t, dtype=dtype) for t in ckpt["tensors"]],
        forcing_config=ckpt["forcing_config"],
    )
    Phi = np.asarray(ckpt["Phi"], dtype=dtype)
    Psi = np.asarray(ckpt.get("Psi", ckpt["Phi"]), dtype=dtype)
    return rom, Phi, Psi


# --- Load trajectories and the decoder used to build them ---
fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"
parameters = np.load(traj_path + "parameters.npy")
n_traj = len(parameters)

decoder = np.load(traj_path + "decoder.npz")
phi_pre = decoder["Phi_project"]  # (n_fom, 300)
n_u = int(decoder["n_u"])
n_v = int(decoder["n_v"])

pool = TrainingPool(
    n_traj=n_traj,
    fname_traj=fname_traj,
    fname_time=fname_time,
    dtype=dtype,
    fname_weights=fname_weight,
    fname_derivs=fname_deriv,
)

# --- Mesh and airfoil outline, taken from a representative flow snapshot ---
snap = np.load(snapshot_fname)
xu_full, yu_full = snap["xu"][1:-1], snap["yu"][1:-1]
xv_full, yv_full = snap["xv"][1:-1], snap["yv"][1:-1]
xi, eta = snap["xi"], snap["eta"]

xu = xu_full[(xu_full >= X_BOUNDS[0]) & (xu_full < X_BOUNDS[1])]
yu = yu_full[(yu_full > Y_BOUNDS[0]) & (yu_full < Y_BOUNDS[1])]
xv = xv_full[(xv_full > X_BOUNDS[0]) & (xv_full < X_BOUNDS[1])]
yv = yv_full[(yv_full > Y_BOUNDS[0]) & (yv_full < Y_BOUNDS[1])]

rowsu, colsu = len(yu), len(xu)
rowsv, colsv = len(yv), len(xv)
if rowsu * colsu != n_u or rowsv * colsv != n_v:
    raise ValueError(
        "The configured airfoil crop does not match decoder.npz: "
        f"crop gives n_u={rowsu * colsu}, n_v={rowsv * colsv}; "
        f"decoder contains n_u={n_u}, n_v={n_v}."
    )

# Vorticity is evaluated where both cropped staggered fields have all the
# neighboring samples required by the finite difference. The outer u columns
# are omitted because the corresponding v samples lie outside the ROM crop.
X_vort, Y_vort = np.meshgrid(xu[1:-1], yv)
XLIM = (float(X_vort.min()), float(X_vort.max()))
YLIM = (float(Y_vort.min()), float(Y_vort.max()))
dx_local = np.diff(xv)  # (colsu - 2,)
dy_local = np.diff(yu)  # (rowsv,)


def split_fields(q):
    u = q[:n_u].reshape(rowsu, colsu)
    v = q[n_u:].reshape(rowsv, colsv)
    return u, v


def vorticity(q):
    u, v = split_fields(q)
    dv_dx = (v[:, 1:] - v[:, :-1]) / dx_local[None, :]
    du_dy = (u[1:, :] - u[:-1, :]) / dy_local[:, None]
    return dv_dx - du_dy[:, 1:-1]


# --- Load the trained ROMs ---
rom_oi, Phi_oi, Psi_oi = load_rom("opinf_model.pkl")
proj_oi = LinearProjection([Phi_oi, Psi_oi])

rom_oi_gs, Phi_oi_gs, Psi_oi_gs = load_rom("gas_opinf_model.pkl")
proj_oi_gs = LinearProjection([Phi_oi_gs, Psi_oi_gs])

rom_nit, Phi_nit, Psi_nit = load_rom("nitrom_model.pkl")
proj_nit = LinearProjection([Phi_nit, Psi_nit])

rom_nit_gs, Phi_nit_gs, Psi_nit_gs = load_rom("gas_nitrom_model.pkl")
proj_nit_gs = LinearProjection([Phi_nit_gs, Psi_nit_gs])

roms = {
    "opinf": (rom_oi, proj_oi, Psi_oi),
    "gas_opinf": (rom_oi_gs, proj_oi_gs, Psi_oi_gs),
    "nitrom": (rom_nit, proj_nit, Psi_nit),
    "gas_nitrom": (rom_nit_gs, proj_nit_gs, Psi_nit_gs),
}

# --- Integrate each ROM on the last training trajectory ---
k_last = n_traj - 1
print(f"Using training trajectory {k_last} (parameters={parameters[k_last]})")

t_eval = pool.time
t0, tf = float(t_eval[0]), float(t_eval[-1])
dt = float(t_eval[1] - t_eval[0])

X_true = pool.X[k_last]  # (300, T)

predictions = {"truth": X_true}
for name, (rom, proj, Psi) in roms.items():
    z0 = Psi.T @ X_true[:, 0]
    sol_r = solve_ivp(
        rom.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45", rtol=rtol, atol=atol
    )
    predictions[name] = proj.decode(sol_r.T).T  # (300, T)

# --- Render one side-by-side (truth vs. ROM) movie per ROM ---
os.makedirs(movies_dir, exist_ok=True)

frame_idx = np.arange(0, len(t_eval), FRAME_STRIDE)

writer_cls, ext = (
    (animation.FFMpegWriter, "mp4")
    if shutil.which("ffmpeg")
    else (animation.PillowWriter, "gif")
)


def field_stack(pred_300):
    q_pert = phi_pre @ pred_300  # (n_fom, T)
    return np.stack([vorticity(q_pert[:, j]) for j in frame_idx], axis=0)


vort_truth = field_stack(predictions["truth"])
vmax = float(np.percentile(np.abs(vort_truth), 99))
vmin = -vmax

for name in roms:
    vort_pred = field_stack(predictions[name])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.2), constrained_layout=True)
    meshes = []
    for ax, title, field0 in zip(axes, ["Truth", ROM_LABELS[name]], [vort_truth[0], vort_pred[0]]):
        ax.set_aspect("equal")
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_title(title)
        ax.set_xlabel(r"$x$")
        mesh = ax.pcolormesh(
            X_vort, Y_vort, field0, cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="gouraud"
        )
        ax.fill(xi, eta, color="0.2", zorder=5)
        meshes.append(mesh)
    axes[0].set_ylabel(r"$y$")
    fig.colorbar(meshes[1], ax=axes, shrink=0.85, label="Perturbation vorticity")
    time_text = axes[0].text(
        0.02, 0.92, "", transform=axes[0].transAxes, fontsize=8, color="k"
    )

    def update(frame, meshes=meshes, time_text=time_text):
        meshes[0].set_array(vort_truth[frame].ravel())
        meshes[1].set_array(vort_pred[frame].ravel())
        time_text.set_text(f"t = {t_eval[frame_idx[frame]]:.1f}")
        return (*meshes, time_text)

    anim = animation.FuncAnimation(fig, update, frames=len(frame_idx), blit=False)
    outfile = os.path.join(movies_dir, f"airfoil_prediction_{name}.{ext}")
    anim.save(outfile, writer=writer_cls(fps=FPS))
    plt.close(fig)
    print(f"Saved {outfile}")
