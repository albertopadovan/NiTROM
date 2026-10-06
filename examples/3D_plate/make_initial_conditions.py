"""Initial conditions for the nine 3D impulse trajectories (wing or plate:
the case is the one the ``--config`` describes, see case.py).

The 3D analogue of examples/airfoil/make_initial_conditions.py: a Gaussian in
the (x, y) plane, placed above the suction side at three stations (leading
edge, midchord, trailing edge), times a smooth top hat in the span,

    B(x, y, z) = exp{-1250 [(x - x_c)^2 + (y - y_c)^2]} h(z),

    h(z) = 1 for |z| <= Z_FLAT,  quintic smoothstep down to 0 at |z| = Z_EDGE,
           0 beyond,

so the impulse is supported in -Z_EDGE <= z <= Z_EDGE (0.1) and C^2 smooth.
It enters the u-momentum equation (``--component`` overrides).  h is even in
z, so these impulses only excite the spanwise-symmetric dynamics.

Stations (case.Case.stations):
  wing : the 2D airfoil's three stations, body-frame (x/c, y/c) = (0.02, 0.04),
         (0.50, 0.07), (0.98, 0.02) with the half-chord at the origin, mapped
         through the markers' nose-up rotation -- the 2D case's impulses;
  plate: chord fraction s from the leading edge and normal offset d
         (``--offset``) above the suction side, x_c = s cos(ALPHA) + d sin(ALPHA),
         y_c = -s sin(ALPHA) + d cos(ALPHA); d keeps the Gaussian clear of the
         zero-thickness plate's delta support (~1.5 cells = 0.019).

As for the airfoil, the raw profile is projected with the modified Leray
projection (``enforce_constraints``: divergence free and no slip on the
plate), using its linear part P(q) - P(0) -- affinity is checked -- and
scaled to unit W-norm, <q, q>_W with W the cell volume over the smallest
cell volume, so q(0) = q_b + beta * B_f has ||q(0) - q_b||_W = |beta|.

Usage (incompreso venv):
    python make_initial_conditions.py [--config config.yaml] [--component u] [--offset 0.05]
Writes <forcing>/Bf_profiles.npy (n, 3), stations.npy, betas.npy, with
<forcing> = forcing_wing/ (wing) or forcing/ (plate).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from incompreso import parse_input_file
from incompreso.backend import GPU, to_backend, to_numpy

from case import DEFAULT_CONFIG, Case

HERE = os.path.dirname(os.path.abspath(__file__))

GAUSS_WIDTH = 1250.0                     # as for the airfoil
Z_FLAT, Z_EDGE = 0.06, 0.10              # spanwise top hat: flat to 0.06, zero beyond 0.1
BETAS = (0.1, 1.0, 2.0)


def staggered_coords(mesh):
    """Interior (x, y, z) coordinate vectors of the u, v and w grids (host)."""
    return [tuple(to_numpy(getattr(mesh, f"{a}{c}"))[1:-1] for a in "xyz")
            for c in "uvw"]


def cell_volumes(mesh):
    """Dual-cell volume of every u, v, w DOF, stacked like the state vector
    (each component C-ordered as (z, y, x))."""
    vols = []
    for x, y, z in staggered_coords(mesh):
        vols.append((np.gradient(z)[:, None, None] * np.gradient(y)[None, :, None]
                     * np.gradient(x)[None, None, :]).ravel())
    return np.concatenate(vols)


def cell_volume_weights(mesh):
    V = cell_volumes(mesh)
    return V/V.min()


def top_hat(z, z_center=0.0):
    """1 for |z - z_center| <= Z_FLAT, quintic smoothstep to 0 at Z_EDGE, 0 beyond."""
    s = np.clip((Z_EDGE - np.abs(z - z_center))/(Z_EDGE - Z_FLAT), 0.0, 1.0)
    return s**3*(10.0 - 15.0*s + 6.0*s**2)


def gaussian_profile(mesh, x0, y0, component, z_center=0.0):
    grids = staggered_coords(mesh)
    sizes = [len(x)*len(y)*len(z) for x, y, z in grids]
    q = np.zeros(sum(sizes))
    c = "uvw".index(component)
    x, y, z = grids[c]
    B = (top_hat(z, z_center)[:, None, None]
         * np.exp(-GAUSS_WIDTH*((x[None, None, :] - x0)**2 + (y[None, :, None] - y0)**2)))
    start = sum(sizes[:c])
    q[start:start + sizes[c]] = B.ravel()
    return q


def forcing_dir(C, z_center=0.0):
    """Impulse directory: C.forcing for the mid-span (z-symmetric) impulses,
    C.forcing + '_z<zc>' for impulses centred off mid-span."""
    return C.forcing if z_center == 0.0 else f"{C.forcing}_z{z_center:g}"


def make_profiles(sim, C, component="u", offset=0.05, z_center=0.0):
    """Build, project and normalize the three station profiles on an already
    parsed setup ``sim`` (so a caller that also marches, e.g.
    generate_trajectories.py, sets up the operators and the exact Schur
    inverse only once).  The spanwise top hat is centred at ``z_center``.
    Saves them to forcing_dir(C, z_center) and returns them (n, 3)."""
    tstep = sim["time_stepper"]
    spops, ib = tstep.spatial_operators, tstep.immersed_body
    mesh = spops.mesh
    if ib is None or ib.is_body_moving:
        raise SystemExit("expected a stationary immersed body")
    n_dof = sim["q0"].shape[0]
    if cell_volumes(mesh).size != n_dof:
        raise SystemExit("staggered-grid sizes do not match the state vector")

    def enforce(q):
        return to_numpy(ib.enforce_constraints(0.0, to_backend(q))).astype(np.float64, copy=False)

    print(f"backend: {'GPU (CuPy)' if GPU else 'CPU (NumPy)'}")
    P0 = enforce(np.zeros(n_dof))

    def leray(q):
        """Modified Leray projection of a PERTURBATION (linear part P(q) - P(0))."""
        return enforce(q) - P0

    rng = np.random.default_rng(0)
    g = rng.standard_normal(n_dof)*1e-3
    qb = to_numpy(sim["q0"]).astype(np.float64, copy=False)
    d1, d2 = leray(g), enforce(qb + g) - enforce(qb)
    rel = np.linalg.norm(d1 - d2)/np.linalg.norm(d1)
    if rel > 1e-6:
        raise SystemExit(f"enforce_constraints is not affine ({rel:.2e})")
    print(f"projection is affine to {rel:.1e}; using P(q) - P(0)\n")

    sqrt_w = np.sqrt(cell_volume_weights(mesh))
    st = C.stations(offset)
    print(f"Gaussian exp(-{GAUSS_WIDTH:g} r^2) in (x, y) x top hat in z "
          f"(flat |z - {z_center:g}| <= {Z_FLAT:g}, zero beyond {Z_EDGE:g}) on {component}, "
          f"unit W-norm; {C.name}, alpha = {C.alpha:g} deg\n")
    print(f"{'station':>15}{'x0':>9}{'y0':>9}{'||div|| raw':>14}{'||div|| proj':>14}{'W-energy kept':>15}")
    profiles = []
    for label, x0, y0 in st:
        raw = gaussian_profile(mesh, x0, y0, component, z_center)
        proj = leray(raw)
        d_raw = float(np.linalg.norm(to_numpy(spops.D @ to_backend(raw))))
        d_prj = float(np.linalg.norm(to_numpy(spops.D @ to_backend(proj))))
        n = np.linalg.norm(sqrt_w*proj)
        print(f"{label:>15}{x0:>9.4f}{y0:>9.4f}{d_raw:>14.3e}{d_prj:>14.3e}"
              f"{n/np.linalg.norm(sqrt_w*raw):>15.4f}")
        if n == 0.0:
            raise SystemExit(f"the projection annihilated the {label} profile")
        profiles.append(proj/n)

    Bf = np.column_stack(profiles)
    out = forcing_dir(C, z_center)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "Bf_profiles.npy"), Bf)
    np.save(os.path.join(out, "stations.npy"), np.array([(x, y) for _, x, y in st]))
    np.save(os.path.join(out, "betas.npy"), np.array(BETAS))
    np.save(os.path.join(out, "z_center.npy"), np.array(z_center))
    print(f"\nsaved {os.path.basename(out)}/Bf_profiles.npy ({n_dof} x {len(profiles)}), "
          f"stations.npy, betas.npy {BETAS}\n", flush=True)
    return Bf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="case config (wing: config.yaml, plate: config_plate.yaml)")
    ap.add_argument("--component", choices=list("uvw"), default="u")
    ap.add_argument("--offset", type=float, default=0.05,
                    help="normal distance of the stations above the suction side")
    ap.add_argument("--z-center", type=float, default=0.0,
                    help="spanwise centre of the top hat (0: z-symmetric impulses)")
    args = ap.parse_args()
    make_profiles(parse_input_file(args.config), Case(args.config),
                  args.component, args.offset, args.z_center)


if __name__ == "__main__":
    main()
