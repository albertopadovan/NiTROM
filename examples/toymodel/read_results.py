import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from nitrom.backend import set_backend
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.plotting import COLORS, set_plot_style
from nitrom.projections.linear_projection import LinearProjection
from nitrom.time_steppers.time_stepper import solve_ivp

# Pure-numpy CPU run.
set_backend("numpy")
set_plot_style()

dtype = np.float64
models_dir = "./models_continuous_adjoint/"
n = 3  # FOM dimension

# %% Full-order model operators (same toy model as generate_data.py)

beta = 20.0
A2 = np.diag(np.array([-1.0, -2.0, -5.0], dtype=dtype))
A3 = np.zeros((n, n, n), dtype=dtype)
A3[:, :, -1] = np.diag(np.array([beta, beta, 0.0], dtype=dtype))
B = np.ones((n, 1), dtype=dtype)
C = np.ones((1, n), dtype=dtype)  # output operator y = C x


def fom_rhs(t, x, forcing):
    """Step-forced FOM dynamics, batched over the leading axis (f = B u)."""
    return x @ A2.T + np.einsum("ijk,...j,...k->...i", A3, x, x) + forcing


# %% Load the trained ROMs (each is a PolynomialModel plus its POD basis Phi).
# GAS models are stored via their reconstructed (A, H, B) operators, so they
# load as ordinary PolynomialModels too.


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


models = {}
available_models = [
    (r"POD-Galerkin", "galerkin_model.pkl", "red", "solid"),
    (r"OpInf", "opinf_model.pkl", "blue", "dotted"),
    (r"GAS-OpInf", "gas_opinf_model.pkl", "green", "dashed"),
    (r"NiTROM", "nitrom_model.pkl", "#e78ac3", "dashdot"),
    (r"GAS-NiTROM", "gas_nitrom_model.pkl", "#a6d854", (0, (3, 1, 1, 1))),
]

for label, fname, color, style in available_models:
    if os.path.exists(os.path.join(models_dir, fname)):
        models[label] = (load_rom(fname), color, style)

# %% Random step-response trajectories (constant forcing u = b), batched.
# Initial condition is x(0) = 0, as in the training data (generate_data.py);
# each trajectory is driven by a constant input of amplitude b_j.

time = np.linspace(0.0, 10.0, 200, dtype=dtype)
t0, tf = float(time[0]), float(time[-1])
dt_sub = float((time[1] - time[0]) / 5)  # 5 RK4 sub-steps per snapshot

n_test = 100
max_amplitude = 0.24
rng = np.random.default_rng(0)
amplitudes = (max_amplitude * rng.random(n_test)).astype(dtype)

x0 = np.zeros((n_test, n), dtype=dtype)
# FOM forcing field B u = b * ones(n) per trajectory (constant in time).
forcing_field = amplitudes[:, None] * B[:, 0]  # (n_test, n)
# ROM forcing: one callable per trajectory returning the m=1 input u(t) = b.
forcing_fns = [
    (lambda t, b=float(b): b * np.ones(1, dtype=dtype))
    for b in amplitudes
]

X_fom = solve_ivp(fom_rhs, x0, t0, tf, dt_sub, time, "rk4", forcing=forcing_field)
Y_fom = np.einsum("on,bnt->bot", C, X_fom)  # FOM output (n_test, n_out, nt)

# Per-trajectory weight alpha_j = ||C x_ss(b_j)||^2, the steady-state output
# energy (same definition as the training weights in generate_data.py).
ratio = beta / 5 * amplitudes
x_ss = np.stack(
    [-amplitudes / (-1 + ratio), -amplitudes / (-2 + ratio), amplitudes / 5],
    axis=1,
)  # (n_test, n)
alpha = np.linalg.norm(x_ss @ C.T, axis=1) ** 2  # (n_test,)

# e(t) = (1/N) sum_j ||y_j(t) - yhat_j(t)||^2 / alpha_j
avg_error = {}
for name, ((rom, Phi, Psi), _color, _style) in models.items():
    r = Phi.shape[1]
    z0 = np.zeros((n_test, r), dtype=dtype)
    Z_rom = solve_ivp(
        rom.evaluate_rhs, z0, t0, tf, dt_sub, time, "rk4",
        external_forcing=forcing_fns,
    )
    proj = LinearProjection([Phi, Psi])
    Z_flat = np.transpose(Z_rom, (0, 2, 1)).reshape(-1, r)
    X_flat = proj.decode(Z_flat)
    X_rom = np.transpose(X_flat.reshape(n_test, len(time), -1), (0, 2, 1))
    Y_rom = np.einsum("on,bnt->bot", C, X_rom)  # ROM output (n_test, n_out, nt)
    sq_err = np.linalg.norm(Y_fom - Y_rom, axis=1) ** 2  # (n_test, nt)
    avg_error[name] = (sq_err / alpha[:, None]).mean(axis=0)  # (nt,)

# %% Plot the trajectory-averaged error over time

fig, ax = plt.subplots(figsize=(6.0, 4.0))
for name, (_, color, style) in models.items():
    ax.semilogy(time, avg_error[name], color=color, linestyle=style, label=name)
ax.set_xlabel(r"Time $t$")
ax.set_ylabel(r"Average error $e(t)$")
ax.set_title(rf"Step response (averaged over {n_test} trajectories)")
ax.set_xlim(t0, tf)
ax.set_ylim(top=1e-1, bottom=1e-7)
ax.legend()

figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)
out_path = os.path.join(figures_dir, "rom_step_error.pdf")
fig.savefig(out_path)
print(f"saved -> {out_path}")
