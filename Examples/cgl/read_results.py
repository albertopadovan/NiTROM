r"""
Post-processing for the CGL example: reproduce Figures 4 and 5 of Padovan,
Vollmer & Bodony (SIADS 2024).

Produces three figures in ``figures/``:

``cgl_average_error.pdf``
    Trajectory-averaged testing error

    .. math::

        e(t) = \frac{1}{N_{traj}} \sum_j \frac{1}{\alpha_j}
               \bigl\| \mathbf{y}^{(j)}(t) - \hat{\mathbf{y}}^{(j)}(t)\bigr\|^2

    over the 50 random impulse responses (paper Figure 4a).

``cgl_impulse_response.pdf``
    Real part of the output for a representative testing impulse, against the
    ground truth (paper Figure 4b).

``cgl_sinusoidal_response.pdf``
    Real part of the output under sinusoidal forcing at :math:`\omega` and
    :math:`2\omega` -- inputs never seen during training (paper Figure 5).

Run after ``generate_data.py``, ``train_opinf.py`` and
``train_oblique_opinf.py``.
"""

import os
import pickle

import fom_class
import matplotlib.pyplot as plt
import numpy as np

from nitrom.backend import set_backend
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.plotting import set_plot_style
from nitrom.projections.linear_projection import LinearProjection
from nitrom.time_steppers.time_stepper import solve_ivp

set_backend("numpy")
set_plot_style()

dtype = np.float64
models_dir = "./models/"
test_path = "./testing/"
figures_dir = "./figures/"
os.makedirs(figures_dir, exist_ok=True)

x, dx, A, B_fom, C = fom_class.build_operators(L=30.0, n=301, dtype=dtype)

# %% Load the trained ROMs

AVAILABLE = [
    ("POD-Galerkin", "galerkin_model.pkl", "tab:red", "solid"),
    ("OpInf", "opinf_model.pkl", "tab:blue", "dotted"),
    ("Oblique OpInf", "oblique_opinf_model.pkl", "tab:green", "dashed"),
]


def load_rom(fname, forcing=False):
    """Rebuild a ROM from a checkpoint, optionally with its input operator."""
    with open(os.path.join(models_dir, fname), "rb") as f:
        ckpt = pickle.load(f)
    tensors = [np.asarray(t, dtype=dtype) for t in ckpt["tensors"]]
    fc = None
    if forcing:
        B_r = np.asarray(ckpt["B_r"], dtype=dtype)
        fc = {"forcing_exists": True, "B": B_r, "m": B_r.shape[1]}
        tensors = tensors + [B_r]
    rom = PolynomialModel(
        ckpt["r"], ckpt["poly_comp"], dtype=dtype,
        tensors=tensors, forcing_config=fc,
    )
    Phi = np.asarray(ckpt["Phi"], dtype=dtype)
    Psi = np.asarray(ckpt.get("Psi", ckpt["Phi"]), dtype=dtype)
    return rom, Phi, Psi


models = {
    label: (fname, color, style)
    for label, fname, color, style in AVAILABLE
    if os.path.exists(os.path.join(models_dir, fname))
}
missing = [f for _, f, _, _ in AVAILABLE if not os.path.exists(os.path.join(models_dir, f))]
if missing:
    print(f"note: skipping missing checkpoints {missing}")


def rom_outputs(rom, Phi, Psi, q0, time, forcing_fns=None, n_sub=50):
    """Integrate a ROM from ambient initial conditions and return y(t)."""
    proj = LinearProjection([Phi, Psi])
    r = Phi.shape[1]
    z0 = proj.encode(q0)  # (m, r)
    dt = float(time[1] - time[0]) / n_sub
    Z = solve_ivp(
        rom.evaluate_rhs, z0, float(time[0]), float(time[-1]), dt, time, "rk4",
        external_forcing=forcing_fns,
    )  # (m, r, nt)
    Z_flat = np.transpose(Z, (0, 2, 1)).reshape(-1, r)
    X = np.transpose(
        proj.decode(Z_flat).reshape(Z.shape[0], len(time), -1), (0, 2, 1)
    )
    return np.einsum("on,bnt->bot", C, X)  # (m, 2, nt)


# %% Figure 1: trajectory-averaged testing error

q0_test = np.load(test_path + "q0.npy")
Y_test = np.load(test_path + "outputs.npy")
alpha_test = np.load(test_path + "weights.npy")  # alpha_j = <||y||^2>_t
time_test = np.load(test_path + "time.npy")
n_test = q0_test.shape[0]

print(f"evaluating {len(models)} ROMs on {n_test} testing impulse responses ...")
avg_error, Y_roms = {}, {}
for label, (fname, _c, _s) in models.items():
    rom, Phi, Psi = load_rom(fname)
    Y_rom = rom_outputs(rom, Phi, Psi, q0_test, time_test)
    Y_roms[label] = Y_rom
    sq_err = np.sum((Y_test - Y_rom) ** 2, axis=1)  # (n_test, nt)
    avg_error[label] = (sq_err / alpha_test[:, None]).mean(axis=0)
    print(f"  {label:<16s} time-averaged e(t) = {avg_error[label].mean():.4e}")

fig, ax = plt.subplots(figsize=(6.0, 4.0))
for label, (_f, color, style) in models.items():
    ax.semilogy(time_test, avg_error[label], color=color, linestyle=style, label=label)
ax.set_xlabel(r"Time $t$")
ax.set_ylabel(r"Average error $e(t)$")
ax.set_xlim(float(time_test[0]), float(time_test[-1]))
ax.legend()
fig.tight_layout()
out = os.path.join(figures_dir, "cgl_average_error.pdf")
fig.savefig(out)
print(f"saved -> {out}")

# %% Figure 2: representative impulse response

# Pick the trajectory with the largest impulse amplitude (strongest nonlinearity).
rep = int(np.argmax(np.abs(Y_test).max(axis=(1, 2))))
mask = time_test <= 200.0

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(time_test[mask], Y_test[rep, 0, mask], color="black", lw=2.5,
        label="FOM", zorder=1)
for label, (_f, color, style) in models.items():
    ax.plot(time_test[mask], Y_roms[label][rep, 0, mask],
            color=color, linestyle=style, label=label, zorder=2)
ax.set_xlabel(r"Time $t$")
ax.set_ylabel(r"$\mathrm{Re}\,(y(t))$")
ax.set_xlim(0.0, 200.0)
ax.legend(ncol=2)
fig.tight_layout()
out = os.path.join(figures_dir, "cgl_impulse_response.pdf")
fig.savefig(out)
print(f"saved -> {out}")

# %% Figure 3: sinusoidal forcing at omega and 2 omega (unseen inputs)

Y_sin = np.load(test_path + "sin_outputs.npy")
alpha_sin = np.load(test_path + "sin_weights.npy")
time_sin = np.load(test_path + "sin_time.npy")
meta = np.load(test_path + "sin_meta.npy")
omega, ks = float(meta[2]), [int(k) for k in meta[3:]]
amp, Bv_norm = float(meta[0]), float(meta[1])
v_sin = np.load(test_path + "sin_v.npy")


def sin_input(k):
    """ROM-level input u(t) = amp sin(k omega t) v / ||Bv||."""
    return lambda t, k=k: amp * np.sin(k * omega * t) * v_sin / Bv_norm


forcing_all = [sin_input(k) for k in ks]

print(f"\nevaluating sinusoidal response at omega = {omega:.4f} "
      f"and {ks[-1]}omega (unseen during training) ...")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
for i, (k, ax) in enumerate(zip(ks, axes)):
    ax.plot(time_sin, Y_sin[i, 0], color="black", lw=2.5, label="FOM", zorder=1)
    q0 = np.zeros((1, A.shape[0]), dtype=dtype)
    for label, (fname, color, style) in models.items():
        rom, Phi, Psi = load_rom(fname, forcing=True)
        Y_rom = rom_outputs(rom, Phi, Psi, q0, time_sin,
                            forcing_fns=[forcing_all[i]])
        ax.plot(time_sin, Y_rom[0, 0], color=color, linestyle=style,
                label=label, zorder=2)
        rel = np.linalg.norm(Y_sin[i] - Y_rom[0]) / np.linalg.norm(Y_sin[i])
        print(f"  k = {k}: {label:<16s} relative output error = {rel:.4e}")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$\mathrm{Re}\,(y(t))$")
    ax.set_title(rf"$\omega_f = {k}\omega$" if k > 1 else r"$\omega_f = \omega$")
    ax.set_xlim(float(time_sin[0]), float(time_sin[-1]))
    # Scale to the ground truth.  The resonant response is an order of
    # magnitude larger than the 2*omega one, and an unstable ROM would
    # otherwise flatten the comparison we actually care about.
    lim = 1.6 * float(np.abs(Y_sin[i, 0]).max())
    ax.set_ylim(-lim, lim)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4,
           bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = os.path.join(figures_dir, "cgl_sinusoidal_response.pdf")
fig.savefig(out)
print(f"saved -> {out}")
