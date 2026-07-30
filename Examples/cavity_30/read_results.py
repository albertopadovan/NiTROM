import os
import pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

import classes_cavity
import time_steppers as tstep
import post_process as pp
from nitrom.backend import set_backend
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.plotting import COLORS, STYLES, set_plot_style
from nitrom.projections.linear_projection import LinearProjection
from nitrom.time_steppers.time_stepper import solve_ivp
from nitrom.training_data import TrainingPool

# Pure-numpy CPU run.
set_backend("numpy")
set_plot_style()

dtype = np.float64

FIG_WIDTH = 3.4
FIG_WIDTH_WIDE = 6.8
FIG_HEIGHT = 2.6
TRAINING_SHADE = "#ececec"
rtol = 1e-4
atol = 1e-8


def make_figure(*, wide=False):
    width = FIG_WIDTH_WIDE if wide else FIG_WIDTH
    return plt.subplots(figsize=(width, FIG_HEIGHT))


def style_axes(ax, *, xlabel="", ylabel="", xlim=None, ylim=None, log_y=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        bottom, top = ylim
        ax.set_ylim(bottom=bottom, top=top)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which="major", color="#d7d7d7", linewidth=0.55, alpha=0.7)
    ax.grid(which="minor", color="#efefef", linewidth=0.4, alpha=0.8)


def add_training_window(ax, x_end=20.0):
    ax.axvspan(0.0, x_end, color=TRAINING_SHADE, alpha=0.9, zorder=0)
    ax.text(
        0.25,
        0.12,
        "Training window",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor="#6f6f6f", linewidth=0.6, alpha=0.95, boxstyle="round,pad=0.25"),
    )


def save_figure(fig, stem):
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/{stem}.eps", format="eps")
    fig.savefig(f"figures/{stem}.png", format="png")
    plt.close(fig)


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
dt_orig = dt

# Setup the flow & FOM
flow = classes_cavity.flow_class(Lx, Ly, Nx, Ny, Re)
lops = classes_cavity.linear_operators_2D(flow, dt)
flow.q_sbf = np.load("bflow_Re%d_Nx%d_Ny%d.npy" % (Re, Nx, Ny))
fom = classes_cavity.fom_class(flow, lops)
fom.assemble_forcing_profile(0.95, 0.05)
B = fom.f.copy()  # shape (19700,)

# Trajectory details
traj_path = "./trajectories/"
which = 'train'  # 'train' or 'test'

if which == 'train':
    fname_traj = traj_path + "traj_%03d.npy"
    fname_weight = traj_path + "weight_%03d.npy"
    fname_deriv = traj_path + "deriv_%03d.npy"
    fname_time = traj_path + "time.npy"
    amps = np.load(traj_path + "amps.npy")
else:
    fname_traj = traj_path + "traj_%03d_testing.npy"
    fname_weight = traj_path + "weight_%03d_testing.npy"
    fname_deriv = traj_path + "deriv_%03d_testing.npy"
    fname_time = traj_path + "time.npy"
    amps = np.load(traj_path + "amps_testing.npy")

phi_pre = np.load(traj_path + "phi_pre.npy")  # (19700, 200)
n_traj = len(amps)
n = phi_pre.shape[-1]

# Load the trajectories into a TrainingPool
pool = TrainingPool(
    n_traj=n_traj,
    fname_traj=fname_traj,
    fname_time=fname_time,
    dtype=dtype,
    fname_weights=fname_weight,
    fname_derivs=fname_deriv,
)

# Load the trained ROMs
models_dir = "./models/"


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


rom_pod, Phi_pod, Psi_pod = load_rom("galerkin_model.pkl")
proj_pod = LinearProjection([Phi_pod, Psi_pod])

rom_oi, Phi_oi, Psi_oi = load_rom("opinf_model.pkl")
proj_oi = LinearProjection([Phi_oi, Psi_oi])

rom_oi_gs, Phi_oi_gs, Psi_oi_gs = load_rom("gas_opinf_model.pkl")
proj_oi_gs = LinearProjection([Phi_oi_gs, Psi_oi_gs])

rom_nit, Phi_nit, Psi_nit = load_rom("nitrom_model.pkl")
proj_nit = LinearProjection([Phi_nit, Psi_nit])

rom_nit_gs, Phi_nit_gs, Psi_nit_gs = load_rom("gas_nitrom_model.pkl")
proj_nit_gs = LinearProjection([Phi_nit_gs, Psi_nit_gs])

r = rom_pod._r

# --- 1) Energy of perturbations (only plotted for training) ---
if which == 'train':
    fig, ax = make_figure()
    for k in range(pool.my_n_traj):
        Qk = pool.X[k]
        energy_k = np.linalg.norm(Qk, axis=0)**2
        ax.plot(pool.time, energy_k, color='k', alpha=0.85)
    style_axes(ax, xlabel=r'Time $t$', ylabel='Energy of perturbations', xlim=(0.0, pool.time[-1]))
    ax.set_ylim(bottom=0)
    add_training_window(ax, x_end=20.0)
    save_figure(fig, "cavity_energy_perturbations")

# --- 2) Validation Error ---
t_eval = pool.time
t0, tf = float(t_eval[0]), float(t_eval[-1])
dt = float(t_eval[1] - t_eval[0])

error_pod = np.zeros_like(t_eval)
error_oi = np.zeros_like(t_eval)
error_oi_gs = np.zeros_like(t_eval)
error_nit = np.zeros_like(t_eval)
error_nit_gs = np.zeros_like(t_eval)

for k in range(n_traj):
    mean_en = np.mean(np.linalg.norm(pool.X[k], axis=0)**2)
    z0 = Psi_pod.T @ pool.X[k, :, 0]

    # POD-Gal.
    sol_pod_r = solve_ivp(rom_pod.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45", rtol=rtol, atol=atol)
    sol_pod = proj_pod.decode(sol_pod_r.T).T
    error_pod += np.linalg.norm(sol_pod - pool.X[k], axis=0)**2 / mean_en / n_traj

    # OpInf
    z0_oi = Psi_oi.T @ pool.X[k, :, 0]
    sol_oi_r = solve_ivp(rom_oi.evaluate_rhs, z0_oi, t0, tf, dt, t_eval, "rk45", rtol=rtol, atol=atol)
    sol_oi = proj_oi.decode(sol_oi_r.T).T
    error_oi += np.linalg.norm(sol_oi - pool.X[k], axis=0)**2 / mean_en / n_traj

    # GasOpInf
    z0_oi_gs = Psi_oi_gs.T @ pool.X[k, :, 0]
    sol_oi_gs_r = solve_ivp(rom_oi_gs.evaluate_rhs, z0_oi_gs, t0, tf, dt, t_eval, "rk45", rtol=rtol, atol=atol)
    sol_oi_gs = proj_oi_gs.decode(sol_oi_gs_r.T).T
    error_oi_gs += np.linalg.norm(sol_oi_gs - pool.X[k], axis=0)**2 / mean_en / n_traj

    # NiTROM
    z0_nit = Psi_nit.T @ pool.X[k, :, 0]
    sol_nit_r = solve_ivp(rom_nit.evaluate_rhs, z0_nit, t0, tf, dt, t_eval, "rk45", rtol=rtol, atol=atol)
    sol_nit = proj_nit.decode(sol_nit_r.T).T
    error_nit += np.linalg.norm(sol_nit - pool.X[k], axis=0)**2 / mean_en / n_traj

    # GasNiTROM
    z0_nit_gs = Psi_nit_gs.T @ pool.X[k, :, 0]
    sol_nit_gs_r = solve_ivp(rom_nit_gs.evaluate_rhs, z0_nit_gs, t0, tf, dt, t_eval, "rk45", rtol=rtol, atol=atol)
    sol_nit_gs = proj_nit_gs.decode(sol_nit_gs_r.T).T
    error_nit_gs += np.linalg.norm(sol_nit_gs - pool.X[k], axis=0)**2 / mean_en / n_traj


# Plot full validation errors
fig, ax = make_figure(wide=False)
ax.semilogy(t_eval, error_pod, label='POD-Gal.', color=COLORS["galerkin"], linestyle=STYLES["galerkin"])
ax.semilogy(t_eval, error_oi, label='OpInf', color=COLORS["opinf"], linestyle=STYLES["notgas"])
ax.semilogy(t_eval, error_oi_gs, label='GasOpInf', color=COLORS["opinf"], linestyle=STYLES["gas"])
ax.semilogy(t_eval, error_nit, label='NiTROM', color=COLORS["nitrom"], linestyle=STYLES["notgas"])
ax.semilogy(t_eval, error_nit_gs, label='GasNiTROM', color=COLORS["nitrom"], linestyle=STYLES["gas"])
style_axes(ax, xlabel='Time $t$', ylabel='Error', xlim=(0.0, float(t_eval[-1])), ylim=(1e-3, 1e3), log_y=True)
# ax.legend(loc='upper right', ncol=3, columnspacing=1.0, handletextpad=0.5)
save_figure(fig, f'cavity_30_error_{which}_full')

# --- 3) Sinusoidal Forcing ---
time_np = dt_orig * np.arange(0, 80 * n, 1)
nsave = 5
amp = 0.9
energies = []
ks = [1, 2, 4]

# For filename, format the amp variable (e.g. 0.1 -> 0p1)
amp_str = str(amp).replace('.', 'p')

# For tracking final simulation snapshot
sol_pod = None
sol_oi = None
sol_oi_gs = None
sol_nit = None
sol_nit_gs = None
dataf = None
tsavef = None

# For tracking contour plot simulation snapshot
harmonic_contour = 1
sol_pod_contour = None
sol_oi_contour = None
sol_oi_gs_contour = None
sol_nit_contour = None
sol_nit_gs_contour = None
dataf_contour = None
tsavef_contour = None

for harmonic in ks:
    freq = 1.00 * harmonic
    tf_f = np.arange(0, 2 * np.pi / freq, dt)
    fint = sp.interpolate.interp1d(tf_f, amp * np.sin(freq * tf_f), kind='linear', fill_value="extrapolate")
    print("Forcing frequency %.2f..." % freq)

    qic = flow.q_sbf.copy()
    dataf, tsavef = tstep.solver_2D(
        flow, lops, qic, time_np, nsave, [0, 1, 1, 0, 0, 0, 0, 0],
        [1], [fint], [2 * np.pi / freq], vol_forcing=B
    )
    energy_true = np.linalg.norm(dataf - qic.reshape(-1, 1), axis=0)**2

    z0 = np.zeros(r)

    def make_forcing_fn(Psi_):
        F_spatial = Psi_.T @ (phi_pre.T @ B)
        return [lambda t, Fs=F_spatial, f=freq: Fs * amp * np.sin(f * t)]

    t0_f, tf_f = float(tsavef[0]), float(tsavef[-1])
    dt_f = float(tsavef[1] - tsavef[0])

    # POD
    sol_pod_r = solve_ivp(rom_pod.evaluate_rhs, z0, t0_f, tf_f, dt_f, tsavef, "rk45", external_forcing=make_forcing_fn(Psi_pod), rtol=rtol, atol=atol)
    sol_pod = proj_pod.decode(sol_pod_r.T).T
    energy_pod = np.linalg.norm(sol_pod, axis=0)**2

    # OpInf
    sol_oi_r = solve_ivp(rom_oi.evaluate_rhs, z0, t0_f, tf_f, dt_f, tsavef, "rk45", external_forcing=make_forcing_fn(Psi_oi), rtol=rtol, atol=atol)
    sol_oi = proj_oi.decode(sol_oi_r.T).T
    energy_oi = np.linalg.norm(sol_oi, axis=0)**2

    # GasOpInf
    sol_oi_gs_r = solve_ivp(rom_oi_gs.evaluate_rhs, z0, t0_f, tf_f, dt_f, tsavef, "rk45", external_forcing=make_forcing_fn(Psi_oi_gs), rtol=rtol, atol=atol)
    sol_oi_gs = proj_oi_gs.decode(sol_oi_gs_r.T).T
    energy_oi_gs = np.linalg.norm(sol_oi_gs, axis=0)**2

    # NiTROM
    sol_nit_r = solve_ivp(rom_nit.evaluate_rhs, z0, t0_f, tf_f, dt_f, tsavef, "rk45", external_forcing=make_forcing_fn(Psi_nit), rtol=rtol, atol=atol)
    sol_nit = proj_nit.decode(sol_nit_r.T).T
    energy_nit = np.linalg.norm(sol_nit, axis=0)**2

    # GasNiTROM
    sol_nit_gs_r = solve_ivp(rom_nit_gs.evaluate_rhs, z0, t0_f, tf_f, dt_f, tsavef, "rk45", external_forcing=make_forcing_fn(Psi_nit_gs), rtol=rtol, atol=atol)
    sol_nit_gs = proj_nit_gs.decode(sol_nit_gs_r.T).T
    energy_nit_gs = np.linalg.norm(sol_nit_gs, axis=0)**2

    energy_lst = [energy_true, energy_pod, energy_oi, energy_oi_gs, energy_nit, energy_nit_gs]
    energies.append(energy_lst)

    if harmonic == harmonic_contour:
        sol_pod_contour = sol_pod.copy() if sol_pod is not None else None
        sol_oi_contour = sol_oi.copy() if sol_oi is not None else None
        sol_oi_gs_contour = sol_oi_gs.copy() if sol_oi_gs is not None else None
        sol_nit_contour = sol_nit.copy() if sol_nit is not None else None
        sol_nit_gs_contour = sol_nit_gs.copy() if sol_nit_gs is not None else None
        dataf_contour = dataf.copy() if dataf is not None else None
        tsavef_contour = tsavef.copy() if tsavef is not None else None

colors = ['k', COLORS["galerkin"], COLORS["opinf"], COLORS["opinf"], COLORS["nitrom"], COLORS["nitrom"]]
lstyle = ['-', STYLES["galerkin"], STYLES["notgas"], STYLES["gas"], STYLES["notgas"], STYLES["gas"]]

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6.8, 6.6), constrained_layout=True)
for k in range(len(energies)):
    for i, vec in enumerate(energies[k]):
        ax[k].plot(tsavef, vec, color=colors[i], linestyle=lstyle[i])
    if k < len(energies) - 1:
        ax[k].set_xticklabels([])
    ax[k].text(
        0.03, 0.65, rf'$k = {ks[k]}$',
        transform=ax[k].transAxes,
        ha='left', va='bottom',
        fontsize=22,
    )
    style_axes(ax[k], xlabel='' if k < 2 else r'Time $t$', ylabel='Energy' if k == 1 else '', xlim=(0.0, tsavef[-1]))
    ax[k].grid(which="minor", visible=False)
    if k == 0:
        ax[k].set_ylim(0, energies[k][-1].max()*1.1)
    elif k == 1:
        ax[k].set_ylim(0, energies[k][-2].max()*1.1)
    else:
        ax[k].set_ylim(0, energies[k][-1].max()*1.1)
    ax[k].xaxis.label.set_fontsize(16)
    ax[k].xaxis.set_tick_params(labelsize=16)
    ax[k].yaxis.label.set_fontsize(16)
    ax[k].yaxis.set_tick_params(labelsize=16)

save_figure(fig, f"cavity_30_forcing_{amp_str}_energy")

# --- 4) Snapshot contour plots ---
idx = np.argmin(np.abs(tsavef_contour - 30))
ii = 2  # index for vorticity output
X, Y, fields = pp.output_fields(flow, dataf_contour[:, idx] - flow.q_sbf)
fields_window = np.flipud(fields[ii])[:39, :]
vmin = np.min(fields_window)
vmax = -vmin
print(vmin, vmax)

snapshots = [
    ("FOM", dataf_contour[:, idx] - flow.q_sbf),
    ("POD-Gal.", phi_pre @ sol_pod_contour[:, idx]),
    ("OpInf", phi_pre @ sol_oi_contour[:, idx]),
    ("GasOpInf", phi_pre @ sol_oi_gs_contour[:, idx]),
    ("NiTROM", phi_pre @ sol_nit_contour[:, idx]),
    ("GasNiTROM", phi_pre @ sol_nit_gs_contour[:, idx]),
]

fig, axes = plt.subplots(3, 2, figsize=(FIG_WIDTH_WIDE, 5.0), constrained_layout=True)
axes = axes.ravel()

for idx_subplot, (ax, (title, state_vec)) in enumerate(zip(axes, snapshots)):
    X, Y, fields = pp.output_fields(flow, state_vec)
    if title == "OpInf" and amp_str == '0p9':
        fields[ii] = np.zeros_like(fields[ii])
        title += " (blew up)"
    cf = ax.contourf(
        X[ii][:39, :],
        Y[ii][:39, :],
        np.flipud(fields[ii])[:39, :],
        levels=100,
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax
    )
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(direction='out', top=False, right=False)
    
    if idx_subplot < 4:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r'$x$')
    
    if idx_subplot % 2 != 0:
        ax.set_yticks([])
    else:
        ax.set_ylabel(r'$y$')
    
    ax.text(
        0.5, 0.65, title,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=20,
    )
    ax.xaxis.label.set_fontsize(16)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.label.set_fontsize(16)
    ax.yaxis.set_tick_params(labelsize=16)

save_figure(fig, f"cavity_30_forcing_{amp_str}_k{harmonic_contour}_snapshot_all")

# --- 5) Training History Plots ---
with open(os.path.join(models_dir, "gas_opinf_history.pkl"), "rb") as f:
    hist_gas_opinf = pickle.load(f)
with open(os.path.join(models_dir, "nitrom_history.pkl"), "rb") as f:
    hist_nitrom = pickle.load(f)
with open(os.path.join(models_dir, "gas_nitrom_history.pkl"), "rb") as f:
    hist_gas_nitrom = pickle.load(f)

# Timings
time_gas_opinf = hist_gas_opinf["time"]
time_nitrom = hist_nitrom["time"]
time_gas_nitrom = hist_gas_nitrom["time"]
print("GasOpInf time (hours):", time_gas_opinf/60/60)
print("NiTROM time (hours):", time_nitrom/60/60)
print("GasNiTROM time (hours):", time_gas_nitrom/60/60)

# Cost vs Iteration Plot
fig, ax1 = make_figure()
ax2 = ax1.twinx()

l1 = ax1.semilogy(hist_nitrom["iters"], hist_nitrom["loss"], label='NiTROM', color=COLORS["nitrom"], linestyle=STYLES["notgas"])
l2 = ax1.semilogy(hist_gas_nitrom["iters"], hist_gas_nitrom["loss"], label='GasNiTROM', color=COLORS["nitrom"], linestyle=STYLES["gas"])
style_axes(ax1, xlabel='Iteration', ylabel=r'$J_{\text{NiTROM}}$', log_y=True)
ax1.yaxis.label.set_color(COLORS["nitrom"])
ax1.tick_params(axis='y', colors=COLORS["nitrom"])

l3 = ax2.semilogy(hist_gas_opinf["iters"], hist_gas_opinf["loss"], label='GasOpInf', color=COLORS["opinf"], linestyle=STYLES["gas"])
ax2.set_ylabel(r'$J_{\text{OpInf}}$', color=COLORS["opinf"])
ax2.tick_params(axis='y', colors=COLORS["opinf"])
ax2.set_yscale("log")
ax2.yaxis.set_major_locator(LogLocator(base=10.0))
ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax2.yaxis.set_minor_formatter(NullFormatter())

lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

save_figure(fig, 'cost_history_cavity_30')

# Gradient Norm vs Iteration Plot
fig, ax = make_figure()
ax.semilogy(hist_nitrom["iters"], hist_nitrom["gradnorm"], label='NiTROM', color=COLORS["nitrom"], linestyle=STYLES["notgas"], linewidth=1.0)
ax.semilogy(hist_gas_nitrom["iters"], hist_gas_nitrom["gradnorm"], label='GasNiTROM', color=COLORS["nitrom"], linestyle=STYLES["gas"], linewidth=1.0)
ax.semilogy(hist_gas_opinf["iters"], hist_gas_opinf["gradnorm"], label='GasOpInf', color=COLORS["opinf"], linestyle=STYLES["gas"], linewidth=1.0)
style_axes(ax, xlabel='Iteration', ylabel='Gradient Norm', log_y=True)
# ax.legend(loc='upper right')

save_figure(fig, 'gradnorm_history_cavity_30')

print("All figures plotted and saved successfully!")
