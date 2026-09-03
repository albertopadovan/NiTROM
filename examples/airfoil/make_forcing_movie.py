"""Movie of a sinusoidally forced trajectory: FOM truth vs. every trained ROM.

Picks one case out of `traj_forcing/` (default: the trailing-edge, amplitude
1.80, k = 1 case), integrates each trained ROM (OpInf, GasOpInf, NiTROM,
GasNiTROM) from rest under the same analytic sinusoidal input, and animates
the resulting perturbation vorticity fields next to the true one.

The case is selected with `--case`, e.g.

    python make_forcing_movie.py --case leading_edge_amp0.30_k2

and `--list` prints the cases available in `traj_forcing/`.
"""

import argparse
import os
import pickle
import re
import shutil

import matplotlib.animation as animation
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
rtol = 1e-4
atol = 1e-8

traj_path = "./trajectories/"
forcing_path = "./traj_forcing/"
models_dir = "./models/"
movies_dir = "./movies/"
snapshot_fname = "snapshot_000074.npz"

DEFAULT_CASE = "trailing_edge_amp0.30_k1"

# Spatial restriction used before the POD and stored in decoder.npz; the saved
# forcing cases and finite-volume weights still live on the original full mesh.
X_BOUNDS = (-4.0, 13.0)
Y_BOUNDS = (-3.0, 3.0)

# Window actually rendered, and frame pacing; trim to taste.
XLIM = (-1.0, 5.0)
YLIM = (-2.0, 2.0)
FRAME_STRIDE = 1
FPS = 10

# Color scale: symmetric about zero, set from this percentile of |vorticity|
# over the truth trajectory.
VORT_PERCENTILE = 99.5

ROM_FILES = {
    "OpInf": "opinf_model.pkl",
    "GasOpInf": "gas_opinf_model.pkl",
    "NiTROM": "nitrom_model.pkl",
    "GasNiTROM": "gas_nitrom_model.pkl",
}

case_re = re.compile(r"^(?P<location>.+)_amp(?P<amp>[\d.]+)_k(?P<harmonic>\d+)\.npy$")


def available_cases():
    return sorted(
        f[:-4] for f in os.listdir(forcing_path) if case_re.match(f) is not None
    )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--case",
    default=DEFAULT_CASE,
    help=f"traj_forcing case to animate (default: {DEFAULT_CASE})",
)
parser.add_argument(
    "--list", action="store_true", help="list the available cases and exit"
)
args = parser.parse_args()

cases = available_cases()
if args.list:
    print("Available forcing cases:")
    for name in cases:
        print(f"  {name}")
    raise SystemExit(0)

case_name = args.case[:-4] if args.case.endswith(".npy") else args.case
if case_name not in cases:
    raise SystemExit(
        f"Unknown forcing case {case_name!r}. Available cases: {', '.join(cases)}"
    )
case_idx = cases.index(case_name)
print(f"Animating forcing case {case_name} (index {case_idx})")


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
    return rom, LinearProjection([Phi, Psi]), Psi


def blowup_mask(energy, energy_true, factor=1000.0):
    """Per-time-step mask of where a rollout has blown up: non-finite energy,
    or energy grown far past the truth trajectory's peak (> `factor`x), which
    is how a diverging polynomial ROM shows up here (it stays finite, just
    huge). Accumulated forward in time, since a diverged rollout never
    recovers -- unlike `read_results.py`, which only needs a single flag per
    case, a movie should keep showing the frames from before the divergence.
    """
    bad = ~np.isfinite(energy) | (energy > factor * np.max(energy_true))
    return np.maximum.accumulate(bad)


# --- Decoder and the spatial crop it was built with ---
decoder = np.load(traj_path + "decoder.npz")
phi_pre = decoder["Phi_project"]  # (n_fom, 300)
q_base = decoder["q_base"]  # (n_fom,)
n_u = int(decoder["n_u"])
n_v = int(decoder["n_v"])

snap = np.load(snapshot_fname)
xu_full, yu_full = snap["xu"][1:-1], snap["yu"][1:-1]
xv_full, yv_full = snap["xv"][1:-1], snap["yv"][1:-1]
xi, eta = snap["xi"], snap["eta"]

u_x_keep = (xu_full >= X_BOUNDS[0]) & (xu_full < X_BOUNDS[1])
u_y_keep = (yu_full > Y_BOUNDS[0]) & (yu_full < Y_BOUNDS[1])
v_x_keep = (xv_full > X_BOUNDS[0]) & (xv_full < X_BOUNDS[1])
v_y_keep = (yv_full > Y_BOUNDS[0]) & (yv_full < Y_BOUNDS[1])

xu, yu = xu_full[u_x_keep], yu_full[u_y_keep]
xv, yv = xv_full[v_x_keep], yv_full[v_y_keep]

if len(yu) * len(xu) != n_u or len(yv) * len(xv) != n_v:
    raise ValueError(
        "The configured airfoil crop does not match decoder.npz: "
        f"crop gives n_u={len(yu) * len(xu)}, n_v={len(yv) * len(xv)}; "
        f"decoder contains n_u={n_u}, n_v={n_v}."
    )


def restrict_to_rom_domain(q):
    """Restrict full-grid velocity data to the ROM's cropped domain."""
    q = np.asarray(q)
    if q.shape[0] == q_base.size:
        return q

    n_u_full = len(yu_full) * len(xu_full)
    n_v_full = len(yv_full) * len(xv_full)
    if q.shape[0] != n_u_full + n_v_full:
        raise ValueError(
            f"Expected {n_u_full + n_v_full} full-grid or {q_base.size} "
            f"cropped velocity entries, got {q.shape[0]}."
        )

    trailing_shape = q.shape[1:]
    u_full = q[:n_u_full].reshape((len(yu_full), len(xu_full)) + trailing_shape)
    v_full = q[n_u_full:].reshape((len(yv_full), len(xv_full)) + trailing_shape)
    u = u_full[u_y_keep][:, u_x_keep]
    v = v_full[v_y_keep][:, v_x_keep]
    return np.concatenate(
        (u.reshape((n_u,) + trailing_shape), v.reshape((n_v,) + trailing_shape)),
        axis=0,
    )


# Vorticity is evaluated only where both cropped staggered fields have the
# neighboring samples required by the finite difference. The outer u columns
# are omitted because the corresponding v samples lie outside the ROM crop.
X_vort, Y_vort = np.meshgrid(xu[1:-1], yv)
dx_local = np.diff(xv)
dy_local = np.diff(yu)


def vorticity(q_pert):
    u = q_pert[:n_u].reshape(len(yu), len(xu))
    v = q_pert[n_u:].reshape(len(yv), len(xv))
    dv_dx = np.diff(v, axis=1) / dx_local[None, :]
    du_dy = np.diff(u, axis=0) / dy_local[:, None]
    return dv_dx - du_dy[:, 1:-1]


# --- Forcing metadata for the selected case ---
# Finite-volume weights the POD modes are orthonormal under, i.e.
# Phi_project.T @ diag(fv_weights) @ Phi_project == I. Needed to encode a
# full-order forcing vector into the 300-dim projected space (Phi_project's
# columns are not orthonormal in the plain Euclidean inner product on this
# non-uniform grid).
fv_weights = restrict_to_rom_domain(np.load("weights.npy"))  # (n_fom,)

# forcing_profiles_projected.npy stacks the full-grid constraint-projected
# spatial forcing profile for each traj_forcing case (columns line up 1:1 with
# the sorted case list). Project before restricting to the ROM crop so the
# reduced forcing represents the field that survives the FOM projection. The
# ROM evaluates the sinusoidal input analytically from its stored amplitude
# and angular frequency; interpolating forcing_signal.npy would alias the
# higher harmonics because that file is sampled only at snapshot times.
forcing_profiles = np.load(forcing_path + "forcing_profiles_projected.npy")
forcing_amplitudes = np.load(forcing_path + "forcing_amplitude.npy")
forcing_omegas = np.load(forcing_path + "forcing_omega.npy")
assert forcing_profiles.shape[1] == len(cases), (
    "forcing_profiles_projected.npy columns must line up with the "
    "traj_forcing case files"
)
assert forcing_amplitudes.shape == forcing_omegas.shape == (len(cases),), (
    "forcing_amplitude.npy and forcing_omega.npy must contain one entry per "
    "traj_forcing case"
)

B = restrict_to_rom_domain(forcing_profiles[:, case_idx])
forcing_amplitude = float(forcing_amplitudes[case_idx])
forcing_omega = float(forcing_omegas[case_idx])
if not np.isclose(forcing_amplitude, float(case_re.match(case_name + ".npy")["amp"])):
    raise ValueError(
        f"Forcing amplitude metadata for {case_name} is {forcing_amplitude:g}, "
        f"but its filename specifies {case_re.match(case_name + '.npy')['amp']}."
    )

t_forcing = np.load(forcing_path + "time.npy")
t0_f, tf_f = float(t_forcing[0]), float(t_forcing[-1])
dt_f = float(t_forcing[1] - t_forcing[0])

# Truth: full state (base + perturbation) on the full grid.
data_f = restrict_to_rom_domain(np.load(forcing_path + case_name + ".npy"))
pert_true = data_f - q_base[:, None]  # (n_fom, T)
del data_f
# Raw full-order field -- unlike proj.decode(...) output, this isn't already in
# the weighted-orthonormal 300-dim space, so the energy norm needs the
# finite-volume weights explicitly.
energy_true = np.sum(fv_weights[:, None] * pert_true**2, axis=0)

# --- Integrate every ROM under the same sinusoidal input ---
roms = {name: load_rom(fname) for name, fname in ROM_FILES.items()}

preds_full = {}
blowup = {"Truth": np.zeros(len(t_forcing), dtype=bool)}
for name, (rom, proj, Psi) in roms.items():
    F_spatial = Psi.T @ (phi_pre.T @ (fv_weights * B))
    z0 = np.zeros(rom._r)
    sol_r = solve_ivp(
        rom.evaluate_rhs,
        z0,
        t0_f,
        tf_f,
        dt_f,
        t_forcing,
        "rk45",
        external_forcing=[
            lambda t, Fs=F_spatial, a=forcing_amplitude, omega=forcing_omega: (
                Fs * a * np.sin(omega * t)
            )
        ],
        rtol=rtol,
        atol=atol,
    )
    sol_300 = proj.decode(sol_r.T).T  # (300, T), weighted-orthonormal coords
    # sol_300 is already isometric to the full-order weighted norm, so no
    # explicit weighting is needed here, unlike energy_true above.
    blowup[name] = blowup_mask(np.linalg.norm(sol_300, axis=0) ** 2, energy_true)
    preds_full[name] = phi_pre @ sol_300  # (n_fom, T), perturbation only
    if blowup[name].any():
        t_blow = float(t_forcing[np.argmax(blowup[name])])
        print(f"  {name} blows up at t = {t_blow:.1f}; later frames are blanked.")

# --- Render one movie with the truth and every ROM side by side ---
os.makedirs(movies_dir, exist_ok=True)

frame_idx = np.arange(0, len(t_forcing), FRAME_STRIDE)
states = {"Truth": pert_true, **preds_full}
vort = {
    name: np.stack([vorticity(state[:, j]) for j in frame_idx], axis=0)
    for name, state in states.items()
}
for name, blew_up in blowup.items():
    vort[name][blew_up[frame_idx]] = 0.0


def panel_title(name):
    """Label a panel, noting when (if ever) that rollout diverges."""
    if not blowup[name].any():
        return name
    return f"{name} (diverges at $t = {t_forcing[np.argmax(blowup[name])]:.0f}$)"


vmax = float(np.percentile(np.abs(vort["Truth"]), VORT_PERCENTILE))
vmin = -vmax

writer_cls, ext = (
    (animation.FFMpegWriter, "mp4")
    if shutil.which("ffmpeg")
    else (animation.PillowWriter, "gif")
)

ncols = 3
nrows = int(np.ceil(len(vort) / ncols))
fig, axes = plt.subplots(
    nrows, ncols, figsize=(3.1 * ncols, 2.4 * nrows), constrained_layout=True
)
axes = np.atleast_1d(axes).ravel()

meshes = []
for ax, (name, field) in zip(axes, vort.items()):
    mesh = ax.pcolormesh(
        X_vort,
        Y_vort,
        field[0],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        shading="gouraud",
    )
    ax.fill(xi, eta, color="0.2", zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_title(panel_title(name), fontsize=10)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.tick_params(direction="out", top=False, right=False)
    meshes.append(mesh)
for ax in axes[len(vort) :]:
    ax.axis("off")

fig.colorbar(
    meshes[0], ax=axes[: len(vort)], shrink=0.85, label="Perturbation vorticity"
)
fig.suptitle(
    rf"{case_name.replace('_', ' ')}:  $a = {forcing_amplitude:g}$,  "
    rf"$\omega = {forcing_omega:.3f}$",
    fontsize=11,
)
time_text = axes[0].text(
    0.02, 0.9, "", transform=axes[0].transAxes, fontsize=8, color="k", zorder=10
)

fields = list(vort.values())


def update(frame):
    for mesh, field in zip(meshes, fields):
        mesh.set_array(field[frame].ravel())
    time_text.set_text(f"t = {t_forcing[frame_idx[frame]]:.1f}")
    return (*meshes, time_text)


anim = animation.FuncAnimation(fig, update, frames=len(frame_idx), blit=False)
outfile = os.path.join(movies_dir, f"airfoil_forcing_{case_name}.{ext}")
anim.save(outfile, writer=writer_cls(fps=FPS))
plt.close(fig)
print(f"Saved {outfile}")
