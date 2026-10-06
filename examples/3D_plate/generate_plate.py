"""Surface markers for a thin rectangular plate at a fixed angle of attack.

Plate: chord c = 1 (x), span s = 1 (z, from -0.5 to 0.5), a single marker
layer, leading edge centred at (0, 0, 0) -- the geometry of
GreensFunctionPC_Paper_Examples/3D_pitching_plate, frozen at angle of attack
ALPHA instead of pitching.  The plate is rotated about the spanwise axis
through the leading edge, nose up (trailing edge down, the same sign
convention as incompreso's airfoil `alpha`):

    x = xi cos(alpha),   y = -xi sin(alpha),   z = zeta.

Marker spacing ~ the minimum mesh spacing (0.0125, see config.yaml).

Usage: python generate_plate.py [--alpha 15]
Writes plate_geometry.npz (xi, eta, zeta, dS) next to this script.
"""

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C, S = 1.0, 1.0          # chord, span
D = 0.0125               # marker spacing ~ min mesh spacing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=15.0, help="angle of attack [deg]")
    args = ap.parse_args()

    nx = int(np.ceil(C/D)) + 1
    nz = int(np.ceil(S/D)) + 1
    x_1d = np.linspace(0.0, C, nx)
    z_1d = np.linspace(-S/2, S/2, nz)
    Xg, Zg = np.meshgrid(x_1d, z_1d)
    dA = np.full_like(Xg, (x_1d[1] - x_1d[0])*(z_1d[1] - z_1d[0]))
    dA[0, :] *= 0.5; dA[-1, :] *= 0.5        # half area at edges, quarter at corners
    dA[:, 0] *= 0.5; dA[:, -1] *= 0.5

    a = np.radians(args.alpha)
    xi = np.cos(a)*Xg.ravel()
    eta = -np.sin(a)*Xg.ravel()
    zeta = Zg.ravel()
    dS = dA.ravel()

    print(f"Plate: chord {C}, span {S}, alpha = {args.alpha:g} deg (about the LE)")
    print(f"Markers: {nx} x {nz} = {xi.size}, spacing {x_1d[1] - x_1d[0]:.4f}")
    print(f"Trailing edge at (x, y) = ({xi.max():.4f}, {eta.min():.4f})")
    print(f"Area sum(dS) = {dS.sum():.6f} (expected {C*S:.6f})")
    out = os.path.join(HERE, "plate_geometry.npz")
    np.savez(out, xi=xi, eta=eta, zeta=zeta, dS=dS, alpha=args.alpha)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
