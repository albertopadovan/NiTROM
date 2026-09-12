"""Plot base-flow vorticity and the three forcing-station centers.

Usage:
    python examples/airfoil/plot_baseflow.py

Writes ``figures/airfoil_baseflow.{png,eps}`` beside this script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from incompreso.backend import to_numpy
from incompreso.utils.immersed_body import make_airfoil
from matplotlib import ticker

from nitrom.backend import set_backend
from nitrom.plotting import set_plot_style

# Match the plotting setup used by read_results.py.
set_backend("numpy")
set_plot_style()
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
SNAPSHOT_PATH = HERE / "snapshot_000074.npz"
FIG_DIR = HERE / "figures"

AIRFOIL_THICKNESS = 0.12
AIRFOIL_N_POINTS = 200
AIRFOIL_ALPHA = 6.0
FORCING_STANDOFF = 0.015
FORCING_STATIONS = (
    ("leading edge", 0.02),
    ("midchord", 0.50),
    ("trailing edge", 0.98),
)
FORCING_X_OVER_C = dict(FORCING_STATIONS)

BASEFLOW_XLIM = (-2.0, 5.0)
BASEFLOW_YLIM = (-1.5, 1.5)


def rd_bu_r_with_white_center() -> ListedColormap:
    """Return RdBu_r with a pure-white band at zero vorticity."""
    colors = plt.get_cmap("RdBu_r", 512)(np.linspace(0.0, 1.0, 512))
    center = len(colors) // 2
    colors[center - 1 : center + 1] = (1.0, 1.0, 1.0, 1.0)
    return ListedColormap(colors, name="RdBu_r_white_center")


def forcing_locations() -> list[tuple[str, float, float]]:
    """Return the physical centers of the three forcing stations."""
    xi0, _, _ = make_airfoil(AIRFOIL_THICKNESS, AIRFOIL_N_POINTS, 0.0)
    x_upper = to_numpy(xi0)[:AIRFOIL_N_POINTS] + 0.5

    xi, eta, _ = make_airfoil(AIRFOIL_THICKNESS, AIRFOIL_N_POINTS, AIRFOIL_ALPHA)
    xi, eta = to_numpy(xi), to_numpy(eta)

    locations = []
    for label, chord_fraction in FORCING_STATIONS:
        idx = int(np.argmin(np.abs(x_upper - chord_fraction)))
        idx = min(max(idx, 1), AIRFOIL_N_POINTS - 2)

        tx = xi[idx + 1] - xi[idx - 1]
        ty = eta[idx + 1] - eta[idx - 1]
        tangent_norm = np.hypot(tx, ty)
        nx, ny = -ty / tangent_norm, tx / tangent_norm
        x0 = xi[idx] + FORCING_STANDOFF * nx
        y0 = eta[idx] + FORCING_STANDOFF * ny
        locations.append((label, float(x0), float(y0)))

    return locations


def baseflow_vorticity(snapshot: np.lib.npyio.NpzFile):
    """Return corner-grid coordinates and vorticity from a flow snapshot."""
    xu = snapshot["xu"][1:-1]
    yu = snapshot["yu"][1:-1]
    xv = snapshot["xv"][1:-1]
    yv = snapshot["yv"][1:-1]

    n_u = len(yu) * len(xu)
    u = snapshot["q"][:n_u].reshape(len(yu), len(xu))
    v = snapshot["q"][n_u:].reshape(len(yv), len(xv))

    dv_dx = np.diff(v, axis=1) / np.diff(xv)[None, :]
    du_dy = np.diff(u, axis=0) / np.diff(yu)[:, None]
    X, Y = np.meshgrid(xu, yv)
    return X, Y, dv_dx - du_dy


def main() -> None:
    with np.load(SNAPSHOT_PATH) as snapshot:
        X, Y, omega = baseflow_vorticity(snapshot)
        xi, eta = snapshot["xi"], snapshot["eta"]

    fig_b, ax_b = plt.subplots(figsize=(9, 4), constrained_layout=True)
    lim = float(np.percentile(np.abs(omega), 99.5))
    levels = np.linspace(-lim, lim, 200)
    cf = ax_b.contourf(
        X,
        Y,
        omega,
        levels=levels,
        cmap=rd_bu_r_with_white_center(),
        extend="both",
    )
    ax_b.fill(xi, eta, color="k", zorder=5)

    locations = forcing_locations()
    alpha_rad = np.deg2rad(AIRFOIL_ALPHA)
    print("Forcing locations:")
    for label, x0, y0 in locations:
        y_airfoil = np.sin(alpha_rad) * x0 + np.cos(alpha_rad) * y0
        print(f"  {label}: x/c = {FORCING_X_OVER_C[label]:.6f}, y/c = {y_airfoil:.6f}")

    for _, x0, y0 in locations:
        ax_b.plot(
            x0,
            y0,
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=7,
        )

    ax_b.set_aspect("equal")
    ax_b.set_xlim(*BASEFLOW_XLIM)
    ax_b.set_ylim(*BASEFLOW_YLIM)
    ax_b.set_xlabel(r"$x / c$")
    ax_b.set_ylabel(r"$y / c$")
    ax_b.xaxis.label.set_fontsize(22)
    ax_b.xaxis.set_tick_params(labelsize=22)
    ax_b.yaxis.label.set_fontsize(22)
    ax_b.yaxis.set_tick_params(labelsize=22)
    cbar = fig_b.colorbar(
        cf,
        ax=ax_b,
        shrink=0.9,
        format="%.1f",
        pad=0.02,
    )
    cbar.ax.tick_params(labelsize=22)
    cbar.locator = ticker.MaxNLocator(nbins=7)
    cbar.update_ticks()
    locs = cbar.get_ticks()
    labels = [label.get_text() for label in cbar.ax.get_yticklabels()]
    new_locs = locs[1:-1]
    new_labels = labels[1:-1]
    cbar.set_ticks(new_locs)
    cbar.set_ticklabels(new_labels)

    FIG_DIR.mkdir(exist_ok=True)
    f_base = FIG_DIR / "airfoil_baseflow.png"
    f_base_eps = FIG_DIR / "airfoil_baseflow.eps"
    fig_b.savefig(f_base, dpi=300)
    fig_b.savefig(f_base_eps)
    plt.close(fig_b)
    print(f"wrote {f_base} and {f_base_eps}")


if __name__ == "__main__":
    main()
