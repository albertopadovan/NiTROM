"""Movie of the FOM trajectory (last training trajectory).

For the last trajectory in the training set, animates the true (projected)
perturbation vorticity field (base flow excluded) over the full domain. This
is the FOM-only counterpart to `make_prediction_movies.py`, which also
integrates and animates each trained ROM alongside the truth.
"""

import os
import shutil

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from nitrom.backend import set_backend
from nitrom.plotting import set_plot_style
from nitrom.training_data import TrainingPool

set_backend("numpy")
set_plot_style()

dtype = np.float64

traj_path = "./trajectories/"
movies_dir = "./movies/"
snapshot_fname = "snapshot_000074.npz"

# Frame stride for the movie; trim to taste.
FRAME_STRIDE = 1
FPS = 15

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
xu, yu = snap["xu"][1:-1], snap["yu"][1:-1]
xv, yv = snap["xv"][1:-1], snap["yv"][1:-1]
xi, eta = snap["xi"], snap["eta"]

rowsu, colsu = len(yu), len(xu)
rowsv, colsv = len(yv), len(xv)
assert rowsu * colsu == n_u, "u-grid does not match decoder's n_u"
assert rowsv * colsv == n_v, "v-grid does not match decoder's n_v"

# Vorticity lives at the corners of the (u, v) staggered grid.
X_vort, Y_vort = np.meshgrid(xu, yv)
XLIM = (float(X_vort.min()), float(X_vort.max()))
YLIM = (float(Y_vort.min()), float(Y_vort.max()))
dx_local = np.diff(xv)  # (colsu,)
dy_local = np.diff(yu)  # (rowsv,)


def split_fields(q):
    u = q[:n_u].reshape(rowsu, colsu)
    v = q[n_u:].reshape(rowsv, colsv)
    return u, v


def vorticity(q):
    u, v = split_fields(q)
    dv_dx = (v[:, 1:] - v[:, :-1]) / dx_local[None, :]
    du_dy = (u[1:, :] - u[:-1, :]) / dy_local[:, None]
    return dv_dx - du_dy


# --- Grab the last training trajectory ---
k_last = n_traj - 1
print(f"Using training trajectory {k_last} (parameters={parameters[k_last]})")

t_eval = pool.time
X_true = pool.X[k_last]  # (300, T)

# --- Render the FOM (truth) movie ---
os.makedirs(movies_dir, exist_ok=True)

frame_idx = np.arange(0, len(t_eval), FRAME_STRIDE)

writer_cls, ext = (
    (animation.FFMpegWriter, "mp4")
    if shutil.which("ffmpeg")
    else (animation.PillowWriter, "gif")
)

q_pert = phi_pre @ X_true  # (n_fom, T)
vort_truth = np.stack([vorticity(q_pert[:, j]) for j in frame_idx], axis=0)
vmax = float(np.percentile(np.abs(vort_truth), 99)) / 10
vmin = -vmax

fig, ax = plt.subplots(figsize=(5.8, 3.2), constrained_layout=True)
ax.set_aspect("equal")
ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
ax.set_title("FOM (truth)")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
mesh = ax.pcolormesh(
    X_vort,
    Y_vort,
    vort_truth[0],
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
    shading="gouraud",
)
ax.fill(xi, eta, color="0.2", zorder=5)
fig.colorbar(mesh, ax=ax, shrink=0.85, label="Perturbation vorticity")
time_text = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=8, color="k")


def update(frame):
    mesh.set_array(vort_truth[frame].ravel())
    time_text.set_text(f"t = {t_eval[frame_idx[frame]]:.1f}")
    return (mesh, time_text)


anim = animation.FuncAnimation(fig, update, frames=len(frame_idx), blit=False)
outfile = os.path.join(movies_dir, f"airfoil_fom.{ext}")
anim.save(outfile, writer=writer_cls(fps=FPS))
plt.close(fig)
print(f"Saved {outfile}")
