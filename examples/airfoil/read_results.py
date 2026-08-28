import os
import pickle
import re
import tempfile

import fom_class
import incompreso
import matplotlib.pyplot as plt
import numpy as np
import yaml
from incompreso.backend import to_numpy
from incompreso.utils.post_process import compute_vorticity
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

from nitrom.backend import set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
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


def save_figure(fig, stem):
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/{stem}.eps", format="eps")
    fig.savefig(f"figures/{stem}.png", format="png")
    plt.close(fig)


def detect_blowup(energy, energy_true, factor=1000.0):
    """Flag a rollout as blown up: non-finite energy, or energy that grows
    far past the truth trajectory's peak (> `factor`x), which is how a
    diverging polynomial ROM shows up here (it stays finite, just huge)."""
    return bool(
        np.any(~np.isfinite(energy)) or np.max(energy) > factor * np.max(energy_true)
    )


# Setup the FOM
fom = fom_class.fom_class()

# Trajectory details
traj_path = "./trajectories/"
which = "test"  # 'train' or 'test'

if which == "train":
    fname_traj = traj_path + "traj_%03d.npy"
    fname_weight = traj_path + "weight_%03d.npy"
    fname_deriv = traj_path + "deriv_%03d.npy"
    fname_time = traj_path + "time.npy"
    parameters = np.load(traj_path + "parameters.npy")
else:
    fname_traj = traj_path + "traj_%03d_testing.npy"
    fname_weight = traj_path + "weight_%03d_testing.npy"
    fname_deriv = traj_path + "deriv_%03d_testing.npy"
    fname_time = traj_path + "time.npy"
    parameters = np.load(traj_path + "parameters_testing.npy")

phi_pre = np.load(traj_path + "decoder.npz")["Phi_project"]  # (n_fom, 300)
n_traj = len(parameters)
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


def load_checkpoint(fname, r, poly_comp):
    with open(os.path.join(models_dir, fname), "rb") as f:
        ckpt = pickle.load(f)
    rom = PolynomialModel(
        r,
        poly_comp,
        dtype=dtype,
        tensors=[np.asarray(t, dtype=dtype) for t in ckpt["tensors"]],
        forcing_config=None,
    )
    Phi = np.asarray(ckpt["Phi"], dtype=dtype)
    Psi = np.asarray(ckpt.get("Psi", ckpt["Phi"]), dtype=dtype)
    return rom, Phi, Psi


def load_checkpoint_gas(fname, r, poly_comp):
    with open(os.path.join(models_dir, fname), "rb") as f:
        ckpt = pickle.load(f)
    rom = GasPolynomialModel(
        r,
        poly_comp,
        dtype=dtype,
        gas_params=ckpt["gas_params"],
        forcing_config=None,
    )
    Phi = np.asarray(ckpt["Phi"], dtype=dtype)
    Psi = np.asarray(ckpt.get("Psi", ckpt["Phi"]), dtype=dtype)
    return rom, Phi, Psi


rom_oi, Phi_oi, Psi_oi = load_rom("opinf_model.pkl")
proj_oi = LinearProjection([Phi_oi, Psi_oi])
Phi_pod = Phi_oi.copy()
Psi_pod = Psi_oi.copy()

rom_oi_gs, Phi_oi_gs, Psi_oi_gs = load_rom("gas_opinf_model.pkl")
proj_oi_gs = LinearProjection([Phi_oi_gs, Psi_oi_gs])

rom_nit, Phi_nit, Psi_nit = load_rom("nitrom_model.pkl")
proj_nit = LinearProjection([Phi_nit, Psi_nit])

rom_nit_gs, Phi_nit_gs, Psi_nit_gs = load_rom("gas_nitrom_model.pkl")
proj_nit_gs = LinearProjection([Phi_nit_gs, Psi_nit_gs])

r = rom_oi._r

# --- 1) Energy of perturbations (only plotted for training) ---
if which == "train":
    fig, ax = make_figure()
    for k in range(pool.my_n_traj):
        Qk = pool.X[k]
        energy_k = np.linalg.norm(Qk, axis=0) ** 2
        ax.plot(pool.time, energy_k, color="k", alpha=0.85)
    style_axes(
        ax,
        xlabel=r"Time $t$",
        ylabel="Energy of perturbations",
        xlim=(0.0, pool.time[-1]),
    )
    ax.set_ylim(bottom=0)
    save_figure(fig, "airfoil_energy_perturbations")

# --- 2) Validation Error ---
t_eval = pool.time
t0, tf = float(t_eval[0]), float(t_eval[-1])
dt = float(t_eval[1] - t_eval[0])

error_pod = np.zeros_like(t_eval)
error_oi = np.zeros_like(t_eval)
error_oi_gs = np.zeros_like(t_eval)
error_nit = np.zeros_like(t_eval)
error_nit_gs = np.zeros_like(t_eval)
error_zeros = np.zeros_like(t_eval)

zero_pred = np.zeros((n, len(t_eval)), dtype=dtype)

for k in range(n_traj):
    mean_en = np.mean(np.linalg.norm(pool.X[k], axis=0) ** 2)
    z0 = Psi_pod.T @ pool.X[k, :, 0]

    # OpInf
    z0_oi = Psi_oi.T @ pool.X[k, :, 0]
    sol_oi_r = solve_ivp(
        rom_oi.evaluate_rhs, z0_oi, t0, tf, dt, t_eval, "rk45", rtol=rtol, atol=atol
    )
    sol_oi = proj_oi.decode(sol_oi_r.T).T
    error_oi += np.linalg.norm(sol_oi - pool.X[k], axis=0) ** 2 / mean_en / n_traj

    # GasOpInf
    z0_oi_gs = Psi_oi_gs.T @ pool.X[k, :, 0]
    sol_oi_gs_r = solve_ivp(
        rom_oi_gs.evaluate_rhs,
        z0_oi_gs,
        t0,
        tf,
        dt,
        t_eval,
        "rk45",
        rtol=rtol,
        atol=atol,
    )
    sol_oi_gs = proj_oi_gs.decode(sol_oi_gs_r.T).T
    error_oi_gs += np.linalg.norm(sol_oi_gs - pool.X[k], axis=0) ** 2 / mean_en / n_traj

    # NiTROM
    z0_nit = Psi_nit.T @ pool.X[k, :, 0]
    sol_nit_r = solve_ivp(
        rom_nit.evaluate_rhs, z0_nit, t0, tf, dt, t_eval, "rk45", rtol=rtol, atol=atol
    )
    sol_nit = proj_nit.decode(sol_nit_r.T).T
    error_nit += np.linalg.norm(sol_nit - pool.X[k], axis=0) ** 2 / mean_en / n_traj

    # GasNiTROM
    z0_nit_gs = Psi_nit_gs.T @ pool.X[k, :, 0]
    sol_nit_gs_r = solve_ivp(
        rom_nit_gs.evaluate_rhs,
        z0_nit_gs,
        t0,
        tf,
        dt,
        t_eval,
        "rk45",
        rtol=rtol,
        atol=atol,
    )
    sol_nit_gs = proj_nit_gs.decode(sol_nit_gs_r.T).T
    error_nit_gs += (
        np.linalg.norm(sol_nit_gs - pool.X[k], axis=0) ** 2 / mean_en / n_traj
    )

    error_zeros += np.linalg.norm(zero_pred - pool.X[k], axis=0) ** 2 / mean_en / n_traj


# Plot training validation errors
fig, ax = make_figure(wide=False)
ax.semilogy(
    t_eval,
    error_oi,
    label="OpInf",
    color=COLORS["opinf"],
    linestyle=STYLES["notgas"],
)
ax.semilogy(
    t_eval,
    error_oi_gs,
    label="GasOpInf",
    color=COLORS["opinf"],
    linestyle=STYLES["gas"],
)
ax.semilogy(
    t_eval,
    error_nit,
    label="NiTROM",
    color=COLORS["nitrom"],
    linestyle=STYLES["notgas"],
)
ax.semilogy(
    t_eval,
    error_nit_gs,
    label="GasNiTROM",
    color=COLORS["nitrom"],
    linestyle=STYLES["gas"],
)
ax.semilogy(
    t_eval,
    error_zeros,
    label="Zero",
    color="k",
    linestyle="-",
)
style_axes(
    ax,
    xlabel="Time $t$",
    ylabel="Error",
    xlim=(0.0, float(t_eval[-1])),
    ylim=(1e-6, 1e2),
    log_y=True,
)
ax.legend()
save_figure(fig, f"airfoil_error_{which}")


# --- 3) Sinusoidal Forcing ---
forcing_path = "./traj_forcing/"

decoder_full = np.load(traj_path + "decoder.npz")
q_base = decoder_full["q_base"]  # (n_fom,)
n_u = int(decoder_full["n_u"])
n_v = int(decoder_full["n_v"])

# Finite-volume weights the POD modes are orthonormal under, i.e.
# Phi_project.T @ diag(fv_weights) @ Phi_project == I. Needed to encode a
# full-order forcing vector into the 300-dim projected space (Phi_project's
# columns are not orthonormal in the plain Euclidean inner product on this
# non-uniform grid).
fv_weights = np.load("weights.npy")  # (n_fom,)

# forcing_profiles.npy stacks one n_fom-dim spatial forcing profile per
# traj_forcing case file (columns line up 1:1 with the sorted file list);
# the profile is location-only -- amplitude and frequency are applied on
# top of it below from the case filename ("<location>_amp<amp>_k<harmonic>").
forcing_profiles = np.load(forcing_path + "forcing_profiles.npy")  # (n_fom, n_cases)
t_forcing = np.load(forcing_path + "time.npy")
t0_f, tf_f = float(t_forcing[0]), float(t_forcing[-1])
dt_f = float(t_forcing[1] - t_forcing[0])

case_re = re.compile(r"^(?P<location>.+)_amp(?P<amp>[\d.]+)_k(?P<harmonic>\d+)\.npy$")
case_files = sorted(
    f
    for f in os.listdir(forcing_path)
    if f.endswith(".npy") and f not in ("time.npy", "forcing_profiles.npy")
)
assert len(case_files) == forcing_profiles.shape[1], (
    "forcing_profiles.npy columns must line up with the traj_forcing case files"
)

case_groups = {}
for i, fname in enumerate(case_files):
    m = case_re.match(fname)
    if m is None:
        raise ValueError(f"Unrecognized traj_forcing filename: {fname}")
    location, amp, harmonic = (
        m.group("location"),
        float(m.group("amp")),
        int(m.group("harmonic")),
    )
    case = dict(
        fname=fname,
        location=location,
        amp=amp,
        harmonic=harmonic,
        B=forcing_profiles[:, i],
    )
    case_groups.setdefault((location, amp), []).append(case)
for group in case_groups.values():
    group.sort(key=lambda c: c["harmonic"])

roms = {
    "OpInf": (rom_oi, proj_oi, Psi_oi, COLORS["opinf"], STYLES["notgas"]),
    "GasOpInf": (rom_oi_gs, proj_oi_gs, Psi_oi_gs, COLORS["opinf"], STYLES["gas"]),
    "NiTROM": (rom_nit, proj_nit, Psi_nit, COLORS["nitrom"], STYLES["notgas"]),
    "GasNiTROM": (rom_nit_gs, proj_nit_gs, Psi_nit_gs, COLORS["nitrom"], STYLES["gas"]),
}

# --- Mesh, BCs, and airfoil outline (rebuilt from config.yaml), used for
# vorticity contour snapshots ---
# config.yaml's initial_condition points at a solver restart file that
# lives alongside the raw incompreso run (not checked into this repo);
# postprocessing here only needs mesh/bcs/immersed_body -- the states come
# from the ROMs -- so swap in a zero IC before building the sim objects.
with open("config.yaml") as f:
    _cfg_dict = yaml.safe_load(f)
_cfg_dict["initial_condition"] = {"type": "zero"}
with tempfile.NamedTemporaryFile(
    mode="w", suffix=".yaml", dir=".", delete=False
) as _tmp:
    yaml.safe_dump(_cfg_dict, _tmp)
    _tmp_config_path = _tmp.name
try:
    sim = incompreso.parse_input_file(_tmp_config_path)
finally:
    os.remove(_tmp_config_path)
mesh = sim["mesh"]
bcs = sim["bcs"]
ib = sim["immersed_body"]
xi, eta = to_numpy(ib.xi), to_numpy(ib.eta)

n_fom_expected = mesh.u_int.size + mesh.v_int.size + mesh.w_int.size
assert q_base.size == n_fom_expected, (
    "decoder.npz's q_base doesn't match the mesh built from config.yaml"
)

# Vorticity is only affine (not linear) in q once boundary conditions are
# applied, so the perturbation vorticity is the *difference* of the two
# full-state vorticities rather than compute_vorticity(perturbation, ...)
# directly (which would wrongly impose the full-flow BCs on a
# perturbation field).
omega_base, X_vort, Y_vort = compute_vorticity(0.0, q_base, mesh, bcs)
omega_base = to_numpy(omega_base)
X_vort, Y_vort = to_numpy(X_vort), to_numpy(Y_vort)


def vorticity(q_pert):
    omega_full, _, _ = compute_vorticity(0.0, q_base + q_pert, mesh, bcs)
    return to_numpy(omega_full) - omega_base


harmonic_contour = 2

for (location, amp), group in sorted(case_groups.items()):
    amp_str = str(amp).replace(".", "p")

    energies_by_row = []  # one dict (name -> energy(t)) per harmonic, in group order
    blowup_by_row = []  # one dict (name -> bool) per harmonic, in group order
    contour_states = (
        None  # name -> full-order perturbation snapshot, for harmonic_contour
    )
    contour_blowup = None  # name -> bool, for harmonic_contour

    for case in group:
        # 0.8026 Hz is the base (vortex shedding) frequency the harmonics were
        # generated at; convert to rad/s for use inside sin(f * t) below.
        freq = 2 * np.pi * 0.8026 * float(case["harmonic"])
        B = case["B"]
        dataf = np.load(
            forcing_path + case["fname"]
        )  # (n_fom, T), full state (base + perturbation)
        # Raw full-order field -- unlike pool.X/proj.decode(...) output, this
        # isn't already in the weighted-orthonormal 300-dim space, so the
        # energy norm needs the finite-volume weights explicitly.
        energy_true = np.sum(fv_weights[:, None] * (dataf - q_base[:, None]) ** 2, axis=0)

        z0 = np.zeros(r)
        energies_row = {"Truth": energy_true}
        blowup_row = {}
        preds_full = {}
        for name, (rom, proj, Psi, color, ls) in roms.items():
            F_spatial = Psi.T @ (phi_pre.T @ (fv_weights * B))
            sol_r = solve_ivp(
                rom.evaluate_rhs,
                z0,
                t0_f,
                tf_f,
                dt_f,
                t_forcing,
                "rk45",
                external_forcing=[
                    lambda t, Fs=F_spatial, f=freq, a=amp: Fs * a * np.sin(f * t)
                ],
                rtol=rtol,
                atol=atol,
            )
            sol_300 = proj.decode(sol_r.T).T  # (300, T), weighted-orthonormal coords
            sol_full = phi_pre @ sol_300  # (n_fom, T), perturbation only, for vorticity
            # sol_300 is already isometric to the full-order weighted norm
            # (Phi_project.T @ diag(fv_weights) @ Phi_project == I), so no
            # explicit weighting is needed here, unlike energy_true above.
            energy = np.linalg.norm(sol_300, axis=0) ** 2
            energies_row[name] = energy
            blowup_row[name] = detect_blowup(energy, energy_true)
            preds_full[name] = sol_full

        energies_by_row.append(energies_row)
        blowup_by_row.append(blowup_row)
        if any(blowup_row.values()):
            print(
                f"  Blew up ({location}, amp={amp}, k={case['harmonic']}): "
                f"{[name for name, blew_up in blowup_row.items() if blew_up]}"
            )

        if case["harmonic"] == harmonic_contour:
            contour_states = {"Truth": dataf - q_base[:, None], **preds_full}
            contour_blowup = {"Truth": False, **blowup_row}

    # --- Energy figure: one row per harmonic ---
    fig, ax = plt.subplots(
        nrows=len(group),
        ncols=1,
        figsize=(FIG_WIDTH_WIDE, 2.2 * len(group)),
        constrained_layout=True,
    )
    ax = np.atleast_1d(ax)
    for row_idx, (case, energies_row) in enumerate(zip(group, energies_by_row)):
        blowup_row = blowup_by_row[row_idx]
        ax[row_idx].plot(
            t_forcing, energies_row["Truth"], color="k", linestyle="-", label="Truth"
        )
        for name, (_, _, _, color, ls) in roms.items():
            ax[row_idx].plot(
                t_forcing, energies_row[name], color=color, linestyle=ls, label=name
            )
        ax[row_idx].text(
            0.03,
            0.65,
            rf"$k = {case['harmonic']}$",
            transform=ax[row_idx].transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
        )
        style_axes(
            ax[row_idx],
            xlabel="" if row_idx < len(group) - 1 else r"Time $t$",
            ylabel="Energy" if row_idx == len(group) // 2 else "",
            xlim=(0.0, float(t_forcing[-1])),
        )
        # Limit the plot height to the highest stable (non-blown-up) model
        # so a diverging ROM doesn't squash the rest of the curves.
        stable_names = ["Truth"] + [name for name in roms if not blowup_row[name]]
        row_max = max(np.max(energies_row[name]) for name in stable_names)
        # Extra headroom on the top row so the legend doesn't overlap the curves.
        headroom = 1.35 if row_idx == 0 else 1.1
        ax[row_idx].set_ylim(bottom=0, top=row_max * headroom)
    ax[0].legend(loc="upper right", fontsize=7)
    save_figure(fig, f"airfoil_{location}_amp{amp_str}_forcing_energy")

    # --- Snapshot contour plots: perturbation vorticity at the final time ---
    if contour_states is not None:
        fig, axes = plt.subplots(
            3, 2, figsize=(FIG_WIDTH_WIDE, 5.0), constrained_layout=True
        )
        axes = axes.ravel()

        vort_fields = {}
        for title, state in contour_states.items():
            field = vorticity(state[:, -1])
            if contour_blowup[title]:
                field = np.zeros_like(field)
                title += " (blew up)"
            vort_fields[title] = field
        vmax = max(np.max(np.abs(field)) for field in vort_fields.values())
        vmin = -vmax

        for ax_i, (title, field) in zip(axes, vort_fields.items()):
            # contourf (not pcolormesh/gouraud) so the EPS export in save_figure works --
            # Ghostscript's PS/EPS distiller can't handle gouraud-shaded meshes.
            cf = ax_i.contourf(
                X_vort, Y_vort, field, levels=100, cmap="RdBu_r", vmin=vmin, vmax=vmax
            )
            ax_i.fill(xi, eta, color="0.2", zorder=5)
            ax_i.set_aspect("equal", adjustable="box")
            ax_i.set_xlim(-1, 5)
            ax_i.set_ylim(-2, 2)
            ax_i.set_title(title, fontsize=10)
            ax_i.tick_params(direction="out", top=False, right=False)
        for ax_i in axes[len(vort_fields) :]:
            ax_i.axis("off")

        fig.colorbar(
            cf, ax=axes[: len(vort_fields)], shrink=0.85, label="Perturbation vorticity"
        )
        save_figure(
            fig, f"airfoil_{location}_amp{amp_str}_k{harmonic_contour}_snapshot_all"
        )

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
print("GasOpInf time (hours):", time_gas_opinf / 60 / 60)
print("NiTROM time (hours):", time_nitrom / 60 / 60)
print("GasNiTROM time (hours):", time_gas_nitrom / 60 / 60)

# Cost vs Iteration Plot
fig, ax1 = make_figure()
ax2 = ax1.twinx()

l1 = ax1.semilogy(
    hist_nitrom["iters"],
    hist_nitrom["loss"],
    label="NiTROM",
    color=COLORS["nitrom"],
    linestyle=STYLES["notgas"],
)
l2 = ax1.semilogy(
    hist_gas_nitrom["iters"],
    hist_gas_nitrom["loss"],
    label="GasNiTROM",
    color=COLORS["nitrom"],
    linestyle=STYLES["gas"],
)
style_axes(ax1, xlabel="Iteration", ylabel=r"$J_{\text{NiTROM}}$", log_y=True)
ax1.yaxis.label.set_color(COLORS["nitrom"])
ax1.tick_params(axis="y", colors=COLORS["nitrom"])

l3 = ax2.semilogy(
    hist_gas_opinf["iters"],
    hist_gas_opinf["loss"],
    label="GasOpInf",
    color=COLORS["opinf"],
    linestyle=STYLES["gas"],
)
ax2.set_ylabel(r"$J_{\text{OpInf}}$", color=COLORS["opinf"])
ax2.tick_params(axis="y", colors=COLORS["opinf"])
ax2.set_yscale("log")
ax2.yaxis.set_major_locator(LogLocator(base=10.0))
ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax2.yaxis.set_minor_formatter(NullFormatter())

lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right")

save_figure(fig, "cost_history_cavity")

# Gradient Norm vs Iteration Plot
fig, ax = make_figure()
ax.semilogy(
    hist_nitrom["iters"],
    hist_nitrom["gradnorm"],
    label="NiTROM",
    color=COLORS["nitrom"],
    linestyle=STYLES["notgas"],
    linewidth=1.0,
)
ax.semilogy(
    hist_gas_nitrom["iters"],
    hist_gas_nitrom["gradnorm"],
    label="GasNiTROM",
    color=COLORS["nitrom"],
    linestyle=STYLES["gas"],
    linewidth=1.0,
)
ax.semilogy(
    hist_gas_opinf["iters"],
    hist_gas_opinf["gradnorm"],
    label="GasOpInf",
    color=COLORS["opinf"],
    linestyle=STYLES["gas"],
    linewidth=1.0,
)
style_axes(ax, xlabel="Iteration", ylabel="Gradient Norm", log_y=True)
# ax.legend(loc='upper right')

save_figure(fig, "gradnorm_history_cavity")

print("All figures plotted and saved successfully!")
