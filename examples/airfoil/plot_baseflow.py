"""Plot base-flow vorticity and the three forcing-station centers.

Usage:
    python examples/airfoil/plot_baseflow.py [--snapshot PATH]

Writes ``figures/airfoil_baseflow.{png,eps}`` beside this script.
"""

from __future__ import annotations

import argparse
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
# Section 5.1 of the GasNiTROM paper: (x/c, y/c) in the body frame, with x/c
# measured from the leading edge along the chord and y/c normal to it.
FORCING_STATIONS = (
    ("leading edge", 0.02, 0.04),
    ("midchord", 0.50, 0.07),
    ("trailing edge", 0.98, 0.02),
)
FORCING_X_OVER_C = {label: xc for label, xc, _ in FORCING_STATIONS}

BASEFLOW_XLIM = (-2.0, 5.0)
BASEFLOW_YLIM = (-1.5, 1.5)


def rd_bu_r_with_white_center() -> ListedColormap:
    """Return RdBu_r with a pure-white band at zero vorticity."""
    colors = plt.get_cmap("RdBu_r", 512)(np.linspace(0.0, 1.0, 512))
    center = len(colors) // 2
    colors[center - 1 : center + 1] = (1.0, 1.0, 1.0, 1.0)
    return ListedColormap(colors, name="RdBu_r_white_center")


def forcing_locations() -> list[tuple[str, float, float]]:
    """Return the physical centers of the three forcing stations.

    The body-frame (x/c, y/c) of each station is shifted so the half-chord
    sits at the origin and then rotated nose-up by the angle of attack, the
    same map that takes the leading edge to the first airfoil marker.
    """
    alpha = np.deg2rad(AIRFOIL_ALPHA)
    ca, sa = np.cos(alpha), np.sin(alpha)

    xi, eta, _ = make_airfoil(AIRFOIL_THICKNESS, AIRFOIL_N_POINTS, AIRFOIL_ALPHA)
    xi, eta = to_numpy(xi), to_numpy(eta)
    i_le = int(np.argmin(xi))
    if not np.allclose((xi[i_le], eta[i_le]), (-0.5 * ca, 0.5 * sa), atol=1e-6):
        raise RuntimeError("airfoil rotation convention changed; the forcing "
                           "stations would no longer sit above the surface")

    locations = []
    for label, xc, yc in FORCING_STATIONS:
        xb = xc - 0.5
        locations.append((label, float(xb * ca + yc * sa),
                          float(-xb * sa + yc * ca)))

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH,
                    help="flow snapshot to plot (default: %(default)s)")
    ap.add_argument("--full-domain", action="store_true",
                    help="show the whole mesh instead of the near-body crop")
    args = ap.parse_args()
    if not args.snapshot.exists():
        raise SystemExit(f"{args.snapshot} not found -- pass --snapshot PATH")

    with np.load(args.snapshot) as snapshot:
        X, Y, omega = baseflow_vorticity(snapshot)
        xi, eta = snapshot["xi"], snapshot["eta"]

    # The full domain is 21 x 8, so the near-body aspect ratio would squash it.
    figsize = (16, 6.5) if args.full_domain else (9, 4)
    fig_b, ax_b = plt.subplots(figsize=figsize, constrained_layout=True)
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
    if args.full_domain:
        ax_b.set_xlim(float(X.min()), float(X.max()))
        ax_b.set_ylim(float(Y.min()), float(Y.max()))
    else:
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
    # Keep the two views in separate files so one does not clobber the other.
    stem = "airfoil_baseflow" + ("_full" if args.full_domain else "")
    f_base = FIG_DIR / f"{stem}.png"
    f_base_eps = FIG_DIR / f"{stem}.eps"
    fig_b.savefig(f_base, dpi=300)
    fig_b.savefig(f_base_eps)
    plt.close(fig_b)
    print(f"wrote {f_base} and {f_base_eps}")


if __name__ == "__main__":
    main()
