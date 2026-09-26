"""Initial conditions for the nine airfoil impulse trajectories.

Equation (41) of the paper puts a Gaussian into the momentum equation,

    B(x, y) = exp{-1250 [(x - x_c)^2 + (y - y_c)^2]},

at three stations along the upper surface (leading edge, midchord, trailing
edge).  The cavity case states that the analogous input "enters the
x-momentum equation", and the airfoil input is introduced as being of the
same form, so the Gaussian is placed on the u component; ``--component v``
overrides that.

The raw Gaussian is NOT a usable initial condition: it is neither
divergence free nor compatible with no-slip on the immersed body, so the
solver would project it on the first step anyway and the trajectory would
not start from the state we thought.  Section 5 says the pressure and the
immersed-body forces are removed "via modified Leray projection", which is
exactly ``ImmersedBody.enforce_constraints`` -- it imposes both constraints
at once.  We apply it here, then scale the result to unit norm, so that
q(0) = beta * B_f has ||q(0)|| = |beta| by construction.

The norm is the cell-volume-weighted one the paper adopts, <q, q>_W with W
the ratio of local cell volume to the smallest cell volume, because the mesh
is strongly stretched and an unweighted norm would let the huge far-field
cells dominate.  Dual-cell widths come from centred differences of the
staggered coordinate vectors, which is exact wherever the grid is uniform.

Usage (needs the incompreso venv, with NiTROM on PYTHONPATH):
    PYTHONPATH=../../src ../../../incompreso/.venv/bin/python \
        make_initial_conditions.py [--component u] [--norm W]
Writes forcing/Bf_profiles.npy, forcing/stations.npy, forcing/betas.npy and
figures/airfoil_initial_conditions.png.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from incompreso import parse_input_file

from plot_baseflow import forcing_locations, rd_bu_r_with_white_center

HERE = os.path.dirname(os.path.abspath(__file__))

GAUSS_WIDTH = 1250.0                  # the 1250 of equation (41)
BETAS = (0.1, 1.0, 2.0)               # section 5.1


def cell_volume_weights(mesh):
    """Diagonal of W: local dual-cell area over the smallest one.

    u and v live on different staggered grids, so each gets its own dual
    cell.  ``np.gradient`` of a coordinate vector returns the centred
    spacing (x[i+1] - x[i-1])/2, which is the finite-volume dual width and
    is exact in the uniform near-body region.
    """
    xu, yu = np.asarray(mesh.xu)[1:-1], np.asarray(mesh.yu)[1:-1]
    xv, yv = np.asarray(mesh.xv)[1:-1], np.asarray(mesh.yv)[1:-1]
    Vu = np.outer(np.gradient(yu), np.gradient(xu)).ravel()
    Vv = np.outer(np.gradient(yv), np.gradient(xv)).ravel()
    V = np.concatenate([Vu, Vv])
    return V/V.min()


def gaussian_profile(mesh, x0, y0, component):
    """The raw equation-(41) Gaussian, on the u or v staggered grid."""
    xu, yu = np.asarray(mesh.xu)[1:-1], np.asarray(mesh.yu)[1:-1]
    xv, yv = np.asarray(mesh.xv)[1:-1], np.asarray(mesh.yv)[1:-1]
    n_u, n_v = len(yu)*len(xu), len(yv)*len(xv)
    q = np.zeros(n_u + n_v)
    if component == "u":
        X, Y = np.meshgrid(xu, yu)
        q[:n_u] = np.exp(-GAUSS_WIDTH*((X - x0)**2 + (Y - y0)**2)).ravel()
    else:
        X, Y = np.meshgrid(xv, yv)
        q[n_u:] = np.exp(-GAUSS_WIDTH*((X - x0)**2 + (Y - y0)**2)).ravel()
    return q


def split(mesh, q):
    """(u, v) fields reshaped from the stacked state vector."""
    xu, yu = np.asarray(mesh.xu)[1:-1], np.asarray(mesh.yu)[1:-1]
    xv, yv = np.asarray(mesh.xv)[1:-1], np.asarray(mesh.yv)[1:-1]
    n_u = len(yu)*len(xu)
    return (q[:n_u].reshape(len(yu), len(xu)),
            q[n_u:].reshape(len(yv), len(xv)),
            (xu, yu, xv, yv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--component", choices=["u", "v"], default="u",
                    help="momentum equation the Gaussian enters")
    ap.add_argument("--norm", choices=["W", "l2"], default="W",
                    help="norm in which B_f is scaled to unity")
    ap.add_argument("--half-width", type=float, default=0.35,
                    help="half-width of the zoom box around each station")
    args = ap.parse_args()

    sim = parse_input_file(args.config)
    tstep = sim["time_stepper"]
    spops, ib, mesh = tstep.spatial_operators, tstep.immersed_body, None
    mesh = spops.mesh
    if ib is None:
        raise SystemExit("this config has no immersed body; the modified "
                         "Leray projection needs one")

    # ``enforce_constraints`` is AFFINE, not linear: it also imposes the
    # domain boundary conditions, so P(0) is the freestream field the u = 1
    # inflow injects (||P(0)|| = 580 here, with ||D P(0)|| = 0.58).  Applying
    # it straight to a Gaussian would return base flow plus perturbation and
    # amplify the "perturbation" by ~365x.  Perturbations obey homogeneous
    # BCs, so what we want is the linear part, P(q) - P(0).  That it IS the
    # linear part is checked below rather than assumed.
    n_dof = np.shape(sim["q0"])[0]
    P0 = np.asarray(ib.enforce_constraints(0.0, np.zeros(n_dof)),
                    dtype=np.float64)

    def leray(q):
        """Modified Leray projection of a PERTURBATION field."""
        return np.asarray(ib.enforce_constraints(0.0, q),
                          dtype=np.float64) - P0

    w = cell_volume_weights(mesh)
    sqrt_w = np.sqrt(w)

    def norm(q):
        return (np.linalg.norm(sqrt_w*q) if args.norm == "W"
                else np.linalg.norm(q))

    # Affinity check: if P were not affine, P(q) - P(0) would not be the
    # linear part and every profile below would be wrong.
    rng = np.random.default_rng(0)
    g = rng.standard_normal(n_dof)*1e-3
    qb = np.asarray(sim["q0"], dtype=np.float64)
    d1, d2 = leray(g), (np.asarray(ib.enforce_constraints(0.0, qb + g),
                                   dtype=np.float64)
                        - np.asarray(ib.enforce_constraints(0.0, qb),
                                     dtype=np.float64))
    rel = np.linalg.norm(d1 - d2)/np.linalg.norm(d1)
    if rel > 1e-6:
        raise SystemExit(f"enforce_constraints is not affine (base-state "
                         f"dependence {rel:.2e}); P(q) - P(0) is not the "
                         f"projection of a perturbation")
    print(f"projection is affine to {rel:.1e}; using its linear part "
          f"P(q) - P(0)\n")

    stations = forcing_locations()
    print(f"Gaussian on the {args.component}-momentum equation, width "
          f"exp(-{GAUSS_WIDTH:g} r^2), scaled to unit {args.norm} norm\n")

    raws, projs = [], []
    print(f"{'station':>15}{'x0':>9}{'y0':>9}{'||div|| raw':>14}"
          f"{'||div|| proj':>14}{'energy kept':>13}")
    for label, x0, y0 in stations:
        raw = gaussian_profile(mesh, x0, y0, args.component)
        proj = leray(raw)

        # ``evaluate_divergence_integral`` reads the mesh's own field state,
        # so use the assembled divergence matrix to test an arbitrary vector.
        d_raw = np.linalg.norm(spops.D @ raw)
        d_prj = np.linalg.norm(spops.D @ proj)
        kept = norm(proj)/norm(raw)
        print(f"{label:>15}{x0:>9.4f}{y0:>9.4f}{d_raw:>14.3e}{d_prj:>14.3e}"
              f"{kept:>13.4f}")

        n = norm(proj)
        if n == 0.0:
            raise SystemExit(f"the projection annihilated the {label} Gaussian")
        raws.append(raw)
        projs.append(proj/n)

    Bf = np.column_stack(projs)
    os.makedirs(os.path.join(HERE, "forcing"), exist_ok=True)
    np.save(os.path.join(HERE, "forcing", "Bf_profiles.npy"), Bf)
    np.save(os.path.join(HERE, "forcing", "stations.npy"),
            np.array([(x, y) for _, x, y in stations]))
    np.save(os.path.join(HERE, "forcing", "betas.npy"), np.array(BETAS))
    print(f"\nB_f {Bf.shape}: unit {args.norm} norm "
          f"({[float(f'{norm(Bf[:, j]):.6f}') for j in range(Bf.shape[1])]})")
    print(f"the nine initial conditions are q(0) = beta * B_f with beta in "
          f"{list(BETAS)}")
    print(f"saved -> forcing/Bf_profiles.npy, stations.npy, betas.npy")

    # ---- plot: what the projection does to each Gaussian -----------------
    fig, ax = plt.subplots(2, 3, figsize=(15, 6.4), constrained_layout=True)
    cmap = rd_bu_r_with_white_center()
    for k, (label, x0, y0) in enumerate(stations):
        u_r, v_r, (xu, yu, xv, yv) = split(mesh, raws[k])
        u_p, v_p, _ = split(mesh, projs[k])
        Xu, Yu = np.meshgrid(xu, yu)
        Xv, Yv = np.meshgrid(xv, yv)

        raw_f, Xr, Yr = ((u_r, Xu, Yu) if args.component == "u"
                         else (v_r, Xv, Yv))
        a = ax[0, k]
        lim = np.abs(raw_f).max()
        a.contourf(Xr, Yr, raw_f, levels=np.linspace(-lim, lim, 100),
                   cmap=cmap, extend="both")
        a.set_title(f"{label}: raw Gaussian ({args.component})", fontsize=10)

        # after projection the field is a genuine 2-component velocity, so
        # show its speed -- the component plot alone would hide the v that
        # the divergence-free constraint necessarily creates.
        a = ax[1, k]
        # u is (len(yu), len(xu)) and v is (len(yv), len(xv)); averaging u
        # down a row and v across a column lands both on the same
        # (len(yv), len(xu)) corner grid.
        u_on_c = 0.5*(u_p[:-1, :] + u_p[1:, :])
        v_on_c = 0.5*(v_p[:, :-1] + v_p[:, 1:])
        speed = np.hypot(u_on_c, v_on_c)
        Xc, Yc = np.meshgrid(xu, yv)
        cf = a.contourf(Xc, Yc, speed, levels=60, cmap="magma_r")
        fig.colorbar(cf, ax=a, shrink=0.85)
        a.set_title(f"{label}: after Leray projection (speed)", fontsize=10)

        for a in (ax[0, k], ax[1, k]):
            a.plot(x0, y0, "o", ms=6, mfc="white", mec="k", mew=1.2, zorder=7)
            a.set_xlim(x0 - args.half_width, x0 + args.half_width)
            a.set_ylim(y0 - args.half_width, y0 + args.half_width)
            a.set_aspect("equal")
            a.set_xlabel("$x/c$")
    for a in ax[:, 0]:
        a.set_ylabel("$y/c$")
    fig.suptitle("Airfoil impulse initial conditions: equation (41) Gaussians "
                 "before and after the modified Leray projection", fontsize=12)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", "airfoil_initial_conditions.png")
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
