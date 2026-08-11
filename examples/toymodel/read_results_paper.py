import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

import fom_class
from nitrom.backend import set_backend
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.plotting import COLORS, STYLES, set_plot_style
from nitrom.projections.linear_projection import LinearProjection
from nitrom.time_steppers.time_stepper import solve_ivp

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


def style_axes(ax, *, xlabel, ylabel, xlim=None, ylim=None, log_y=False):
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


def add_training_window(ax, x_end=10.0):
    ax.axvspan(0.0, x_end, color=TRAINING_SHADE, alpha=0.9, zorder=0)
    ax.text(
        0.16,
        0.88,
        "Training window",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor="#6f6f6f", linewidth=0.6, alpha=0.95, boxstyle="round,pad=0.25"),
    )

adjoint_method = 'continuous'
def save_figure(fig, stem):
    os.makedirs(f"figures_{adjoint_method}_adjoint", exist_ok=True)
    fig.savefig(f"figures_{adjoint_method}_adjoint/{stem}.eps", format="eps")
    fig.savefig(f"figures_{adjoint_method}_adjoint/{stem}.png", format="png")
    plt.close(fig)


# Instantiate the full-order model class
n = 3
beta = 20.0
A2 = np.diag(np.array([-1.0, -2.0, -5.0], dtype=dtype))
A3 = np.zeros((n, n, n), dtype=dtype)
A3[:, :, -1] = np.diag(np.array([beta, beta, 0.0], dtype=dtype))
B = np.ones((n, 1), dtype=dtype)
C = np.ones((1, n), dtype=dtype)

fom = fom_class.full_order_model(A2, A3, B, C, dtype=dtype)

# Load the trained ROMs
models_dir = f"./models_{adjoint_method}_adjoint/"


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

# --- 1) Average Error Validation ---
max_val = 5 / 20
rng = np.random.default_rng(0)
betas = rng.random(100).astype(dtype) * 0.999 * max_val

t_eval = np.linspace(0.0, 30.0, 300, dtype=dtype)
t0, tf = float(t_eval[0]), float(t_eval[-1])
dt = float((t_eval[1] - t_eval[0]) / 10)  # dt = 0.01

idx_10 = np.argmin(np.abs(t_eval - 10.0))
error_pod = np.zeros_like(t_eval)
error_oi = np.zeros_like(t_eval)
error_oi_gs = np.zeros_like(t_eval)
error_nit = np.zeros_like(t_eval)
error_nit_gs = np.zeros_like(t_eval)

for k in range(len(betas)):
    b = betas[k]
    # Forcing for FOM is b * np.ones(n, dtype=dtype)
    u_fom = b * np.ones(n, dtype=dtype)
    x0 = np.zeros(n, dtype=dtype)
    z0 = np.zeros(r, dtype=dtype)

    sol = solve_ivp(fom.evaluate_fom_dynamics, x0, t0, tf, dt, t_eval, "rk45", u=u_fom, rtol=rtol, atol=atol)
    y_fom = fom.compute_output(sol)

    id_ss = np.array([-b / (-1 + 4 * b), -b / (-2 + 4 * b), b / 5], dtype=dtype)
    weight = np.linalg.norm(fom.compute_output(id_ss)) ** 2

    # For ROMs, the forcing is lambda t: np.array([b], dtype=dtype)
    forcing_fns = [lambda t, val=b: np.array([val], dtype=dtype)]

    sol_pod_r = solve_ivp(
        rom_pod.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45",
        external_forcing=forcing_fns, rtol=rtol, atol=atol
    )
    sol_pod = proj_pod.decode(sol_pod_r.T).T
    y_pod = fom.compute_output(sol_pod)
    error_pod += np.linalg.norm(y_pod - y_fom, axis=0) ** 2 / weight / len(betas)

    sol_oi_r = solve_ivp(
        rom_oi.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45",
        external_forcing=forcing_fns, rtol=rtol, atol=atol
    )
    sol_oi = proj_oi.decode(sol_oi_r.T).T
    y_oi = fom.compute_output(sol_oi)
    error_oi += np.linalg.norm(y_oi - y_fom, axis=0) ** 2 / weight / len(betas)

    sol_oi_gs_r = solve_ivp(
        rom_oi_gs.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45",
        external_forcing=forcing_fns, rtol=rtol, atol=atol
    )
    sol_oi_gs = proj_oi_gs.decode(sol_oi_gs_r.T).T
    y_oi_gs = fom.compute_output(sol_oi_gs)
    error_oi_gs += np.linalg.norm(y_oi_gs - y_fom, axis=0) ** 2 / weight / len(betas)

    sol_nit_r = solve_ivp(
        rom_nit.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45",
        external_forcing=forcing_fns, rtol=rtol, atol=atol
    )
    sol_nit = proj_nit.decode(sol_nit_r.T).T
    y_nit = fom.compute_output(sol_nit)
    error_nit += np.linalg.norm(y_nit - y_fom, axis=0) ** 2 / weight / len(betas)

    sol_nit_gs_r = solve_ivp(
        rom_nit_gs.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45",
        external_forcing=forcing_fns, rtol=rtol, atol=atol
    )
    sol_nit_gs = proj_nit_gs.decode(sol_nit_gs_r.T).T
    y_nit_gs = fom.compute_output(sol_nit_gs)
    error_nit_gs += np.linalg.norm(y_nit_gs - y_fom, axis=0) ** 2 / weight / len(betas)


# --- 1a) Average Training Error ---
betas_train = np.array([0.01, 0.1, 0.2, 0.248], dtype=dtype)
t_eval_train = t_eval[:idx_10]
t0_train, tf_train = float(t_eval_train[0]), float(t_eval_train[-1])
dt_train = float((t_eval_train[1] - t_eval_train[0]) / 10)

error_pod_train = np.zeros_like(t_eval_train)
error_oi_train = np.zeros_like(t_eval_train)
error_oi_gs_train = np.zeros_like(t_eval_train)
error_nit_train = np.zeros_like(t_eval_train)
error_nit_gs_train = np.zeros_like(t_eval_train)

for k in range(len(betas_train)):
    b = betas_train[k]
    u_fom = b * np.ones(n, dtype=dtype)
    x0 = np.zeros(n, dtype=dtype)
    z0 = np.zeros(r, dtype=dtype)

    sol = solve_ivp(fom.evaluate_fom_dynamics, x0, t0_train, tf_train, dt_train, t_eval_train, "rk45", u=u_fom, rtol=rtol, atol=atol)
    y_fom = fom.compute_output(sol)

    id_ss = np.array([-b / (-1 + 4 * b), -b / (-2 + 4 * b), b / 5], dtype=dtype)
    weight = np.linalg.norm(fom.compute_output(id_ss)) ** 2

    forcing_fns = [lambda t, val=b: np.array([val], dtype=dtype)]

    sol_pod_r = solve_ivp(
        rom_pod.evaluate_rhs, z0, t0_train, tf_train, dt_train, t_eval_train, "rk45",
        external_forcing=forcing_fns, rtol=rtol, atol=atol
    )
    sol_pod = proj_pod.decode(sol_pod_r.T).T
    y_pod = fom.compute_output(sol_pod)
    error_pod_train += np.linalg.norm(y_pod - y_fom, axis=0) ** 2 / weight / len(betas_train)

    sol_oi_r = solve_ivp(
        rom_oi.evaluate_rhs, z0, t0_train, tf_train, dt_train, t_eval_train, "rk45",
        external_forcing=forcing_fns, rtol=rtol, atol=atol
    )
    sol_oi = proj_oi.decode(sol_oi_r.T).T
    y_oi = fom.compute_output(sol_oi)
    error_oi_train += np.linalg.norm(y_oi - y_fom, axis=0) ** 2 / weight / len(betas_train)

    sol_oi_gs_r = solve_ivp(
        rom_oi_gs.evaluate_rhs, z0, t0_train, tf_train, dt_train, t_eval_train, "rk45",
        external_forcing=forcing_fns, rtol=rtol, atol=atol
    )
    sol_oi_gs = proj_oi_gs.decode(sol_oi_gs_r.T).T
    y_oi_gs = fom.compute_output(sol_oi_gs)
    error_oi_gs_train += np.linalg.norm(y_oi_gs - y_fom, axis=0) ** 2 / weight / len(betas_train)

    sol_nit_r = solve_ivp(
        rom_nit.evaluate_rhs, z0, t0_train, tf_train, dt_train, t_eval_train, "rk45",
        external_forcing=forcing_fns, rtol=rtol, atol=atol
    )
    sol_nit = proj_nit.decode(sol_nit_r.T).T
    y_nit = fom.compute_output(sol_nit)
    error_nit_train += np.linalg.norm(y_nit - y_fom, axis=0) ** 2 / weight / len(betas_train)

    sol_nit_gs_r = solve_ivp(
        rom_nit_gs.evaluate_rhs, z0, t0_train, tf_train, dt_train, t_eval_train, "rk45",
        external_forcing=forcing_fns, rtol=rtol, atol=atol
    )
    sol_nit_gs = proj_nit_gs.decode(sol_nit_gs_r.T).T
    y_nit_gs = fom.compute_output(sol_nit_gs)
    error_nit_gs_train += np.linalg.norm(y_nit_gs - y_fom, axis=0) ** 2 / weight / len(betas_train)

# Plot errors
fig, ax = make_figure()
ax.semilogy(t_eval_train, error_pod_train, label='POD-Gal.', color=COLORS["galerkin"], linestyle=STYLES["galerkin"])
ax.semilogy(t_eval_train, error_oi_train, label='OpInf', color=COLORS["opinf"], linestyle=STYLES["notgas"])
ax.semilogy(t_eval_train, error_oi_gs_train, label='GasOpInf', color=COLORS["opinf"], linestyle=STYLES["gas"])
ax.semilogy(t_eval_train, error_nit_train, label='NiTROM', color=COLORS["nitrom"], linestyle=STYLES["notgas"])
ax.semilogy(t_eval_train, error_nit_gs_train, label='GasNiTROM', color=COLORS["nitrom"], linestyle=STYLES["gas"])
style_axes(ax, xlabel='Time $t$', ylabel='Average error $e(t)$', xlim=(0.0, 10.0), ylim=(None, 1e-1), log_y=True)
save_figure(fig, 'error_toymodel_10')

fig, ax = make_figure(wide=True)
ax.semilogy(t_eval, error_pod, label='POD-Gal.', color=COLORS["galerkin"], linestyle=STYLES["galerkin"])
ax.semilogy(t_eval, error_oi, label='OpInf', color=COLORS["opinf"], linestyle=STYLES["notgas"])
ax.semilogy(t_eval, error_oi_gs, label='GasOpInf', color=COLORS["opinf"], linestyle=STYLES["gas"])
ax.semilogy(t_eval, error_nit, label='NiTROM', color=COLORS["nitrom"], linestyle=STYLES["notgas"])
ax.semilogy(t_eval, error_nit_gs, label='GasNiTROM', color=COLORS["nitrom"], linestyle=STYLES["gas"])
style_axes(ax, xlabel='Time $t$', ylabel='Average error $e(t)$', xlim=(0.0, 30.0), ylim=(None, 1e1), log_y=True)
# add_training_window(ax)
ax.legend(loc='lower right', ncol=2, columnspacing=1.0, handletextpad=0.5)
save_figure(fig, 'error_toymodel_30')

# --- 2) Dynamic Response (Low Input) ---
t_eval = np.linspace(0.0, 30.0, 1000, dtype=dtype)
t0, tf = float(t_eval[0]), float(t_eval[-1])
dt = float((t_eval[1] - t_eval[0]) / 10)  # dt = 0.003
idx_10 = np.argmin(np.abs(t_eval - 10.0))

u_fun_low = lambda t: np.array([0.45 * (np.sin(t) + np.cos(2.0 * t))], dtype=dtype)
fom_forcing_low = lambda t: B[:, 0] * (0.45 * (np.sin(t) + np.cos(2.0 * t)))

x0 = np.zeros(n, dtype=dtype)
z0 = np.zeros(r, dtype=dtype)

sol_fom = solve_ivp(fom.evaluate_fom_dynamics, x0, t0, tf, dt, t_eval, "rk45", u=fom_forcing_low, rtol=rtol, atol=atol)
y_fom_low = fom.compute_output(sol_fom)

sol_pod_r = solve_ivp(
    rom_pod.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45",
    external_forcing=[u_fun_low], rtol=rtol, atol=atol
)
y_pod_low = fom.compute_output(proj_pod.decode(sol_pod_r.T).T)

sol_oi_r = solve_ivp(
    rom_oi.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45",
    external_forcing=[u_fun_low], rtol=rtol, atol=atol
)
y_oi_low = fom.compute_output(proj_oi.decode(sol_oi_r.T).T)

sol_oi_gs_r = solve_ivp(
    rom_oi_gs.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45",
    external_forcing=[u_fun_low], rtol=rtol, atol=atol
)
y_oi_gs_low = fom.compute_output(proj_oi_gs.decode(sol_oi_gs_r.T).T)

sol_nit_r = solve_ivp(
    rom_nit.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45",
    external_forcing=[u_fun_low], rtol=rtol, atol=atol
)
y_nit_low = fom.compute_output(proj_nit.decode(sol_nit_r.T).T)

sol_nit_gs_r = solve_ivp(
    rom_nit_gs.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45",
    external_forcing=[u_fun_low], rtol=rtol, atol=atol
)
y_nit_gs_low = fom.compute_output(proj_nit_gs.decode(sol_nit_gs_r.T).T)

# Plot low response
fig, ax = make_figure(wide=True)
ax.plot(t_eval, y_fom_low[0, :], color='k', linewidth=2.4, label='FOM')
ax.plot(t_eval, y_pod_low[0, :], color=COLORS["galerkin"], linestyle=STYLES["galerkin"], label='POD-Gal.')
ax.plot(t_eval, y_oi_low[0, :], color=COLORS["opinf"], linestyle=STYLES["notgas"], label='OpInf')
ax.plot(t_eval, y_nit_low[0, :], color=COLORS["nitrom"], linestyle=STYLES["notgas"], label='NiTROM')
ax.plot(t_eval, y_oi_gs_low[0, :], color=COLORS["opinf"], linestyle=STYLES["gas"], label='GasOpInf')
ax.plot(t_eval, y_nit_gs_low[0, :], color=COLORS["nitrom"], linestyle=STYLES["gas"], label='GasNiTROM')
style_axes(ax, xlabel='Time $t$', ylabel='$y(t)$', xlim=(0.0, 30.0), ylim=(-1.0, 9.0))
ax.xaxis.label.set_fontsize(16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.label.set_fontsize(16)
ax.yaxis.set_tick_params(labelsize=16)
save_figure(fig, 'response_toymodel_low_30')

# --- 3) Dynamic Response (High Input) ---
u_fun_high = lambda t: np.array([0.75 * (np.sin(t) + np.cos(2.0 * t))], dtype=dtype)
fom_forcing_high = lambda t: B[:, 0] * (0.65 * (np.sin(t) + np.cos(2.0 * t)))

sol_fom = solve_ivp(fom.evaluate_fom_dynamics, x0, t0, tf, dt, t_eval, "rk45", u=fom_forcing_high, rtol=rtol, atol=atol)
y_fom_high = fom.compute_output(sol_fom)

sol_pod_r = solve_ivp(
    rom_pod.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45",
    external_forcing=[u_fun_high], rtol=rtol, atol=atol
)
y_pod_high = fom.compute_output(proj_pod.decode(sol_pod_r.T).T)

sol_oi_r = solve_ivp(
    rom_oi.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45",
    external_forcing=[u_fun_high], rtol=rtol, atol=atol
)
y_oi_high = fom.compute_output(proj_oi.decode(sol_oi_r.T).T)

sol_oi_gs_r = solve_ivp(
    rom_oi_gs.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45",
    external_forcing=[u_fun_high], rtol=rtol, atol=atol
)
y_oi_gs_high = fom.compute_output(proj_oi_gs.decode(sol_oi_gs_r.T).T)

sol_nit_r = solve_ivp(
    rom_nit.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45",
    external_forcing=[u_fun_high], rtol=rtol, atol=atol
)
y_nit_high = fom.compute_output(proj_nit.decode(sol_nit_r.T).T)

sol_nit_gs_r = solve_ivp(
    rom_nit_gs.evaluate_rhs, z0, t0, tf, dt, t_eval, "rk45",
    external_forcing=[u_fun_high], rtol=rtol, atol=atol
)
y_nit_gs_high = fom.compute_output(proj_nit_gs.decode(sol_nit_gs_r.T).T)

# Plot high response
fig, ax = make_figure(wide=True)
ax.plot(t_eval, y_fom_high[0, :], color='k', linewidth=2.4, label='FOM')
ax.plot(t_eval, y_pod_high[0, :], color=COLORS["galerkin"], linestyle=STYLES["galerkin"], label='POD-Gal.')
ax.plot(t_eval, y_nit_high[0, :], color=COLORS["nitrom"], linestyle=STYLES["notgas"], label='NiTROM')
ax.plot(t_eval, y_nit_gs_high[0, :], color=COLORS["nitrom"], linestyle=STYLES["gas"], label='GasNiTROM')
ax.plot(t_eval, y_oi_high[0, :], color=COLORS["opinf"], linestyle=STYLES["notgas"], label='OpInf')
ax.plot(t_eval, y_oi_gs_high[0, :], color=COLORS["opinf"], linestyle=STYLES["gas"], label='GasOpInf')
style_axes(
    ax,
    xlabel='Time $t$',
    ylabel='$y(t)$',
    xlim=(0.0, 30.0),
    ylim=(y_fom_high[0, :].min() * 2, 12.0),
)
ax.xaxis.label.set_fontsize(16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.label.set_fontsize(16)
ax.yaxis.set_tick_params(labelsize=16)
save_figure(fig, 'response_toymodel_high_30')

# --- 4) Training History Plots ---
# Load histories
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
print("GasOpInf time:", time_gas_opinf)
print("NiTROM time:", time_nitrom)
print("GasNiTROM time:", time_gas_nitrom)

# Cost vs Iteration Plot
fig, ax1 = make_figure()
ax2 = ax1.twinx()

# Plot NiTROM costs on left y-axis
l1 = ax1.semilogy(hist_nitrom["iters"], hist_nitrom["loss"], label='NiTROM', color=COLORS["nitrom"], linestyle=STYLES["notgas"])
l2 = ax1.semilogy(hist_gas_nitrom["iters"], hist_gas_nitrom["loss"], label='GasNiTROM', color=COLORS["nitrom"], linestyle=STYLES["gas"])
style_axes(ax1, xlabel='Iteration', ylabel=r'$J_{\text{NiTROM}}$', log_y=True)
ax1.yaxis.label.set_color(COLORS["nitrom"])
ax1.tick_params(axis='y', colors=COLORS["nitrom"])

# Plot GasOpInf cost on right y-axis
l3 = ax2.semilogy(hist_gas_opinf["iters"], hist_gas_opinf["loss"], label='GasOpInf', color=COLORS["opinf"], linestyle=STYLES["gas"])
ax2.set_ylabel(r'$J_{\text{OpInf}}$', color=COLORS["opinf"])
ax2.tick_params(axis='y', colors=COLORS["opinf"])
ax2.set_yscale("log")
ax2.yaxis.set_major_locator(LogLocator(base=10.0))
ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax2.yaxis.set_minor_formatter(NullFormatter())

# Combine legends
lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

save_figure(fig, 'cost_history_toymodel')

# Gradient Norm vs Iteration Plot
fig, ax = make_figure()
ax.semilogy(hist_nitrom["iters"], hist_nitrom["gradnorm"], label='NiTROM', color=COLORS["nitrom"], linestyle=STYLES["notgas"])
ax.semilogy(hist_gas_nitrom["iters"], hist_gas_nitrom["gradnorm"], label='GasNiTROM', color=COLORS["nitrom"], linestyle=STYLES["gas"])
ax.semilogy(hist_gas_opinf["iters"], hist_gas_opinf["gradnorm"], label='GasOpInf', color=COLORS["opinf"], linestyle=STYLES["gas"])
style_axes(ax, xlabel='Iteration', ylabel='Gradient Norm', log_y=True)
# ax.legend(loc='lower right')

save_figure(fig, 'gradnorm_history_toymodel')

print("All figures plotted and saved successfully!")
