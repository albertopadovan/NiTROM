"""Surface markers for a finite NACA 00xx wing at a fixed angle of attack --
the 3D version of the 2D airfoil case (examples/airfoil: NACA 0012, 6 deg).

Built with incompreso's own make_wing (the generator behind `shape: wing`):
the 2D NACA section (half-chord at the origin, chord 1, rotated nose up by
ALPHA about the half-chord, as in the 2D runs) extruded over -Lz/2 <= z <=
Lz/2, with the two end caps filled so every marker has dS ~ d^2.  Saving the
markers lets the post-processing and impulse scripts know the geometry; the
config can use either `shape: file` with this file or `shape: wing` with the
same thickness / alpha / Lz (the mesh spacing d must then match).

Usage (incompreso venv):
    python generate_wing.py [--alpha 6] [--thickness 0.12] [--span 1] [--d 0.0125]
Writes wing_geometry.npz (xi, eta, zeta, dS, alpha, thickness, span).
"""

import argparse
import os

import numpy as np

from incompreso.backend import to_numpy
from incompreso.utils.immersed_body import make_wing

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=6.0, help="angle of attack [deg]")
    ap.add_argument("--thickness", type=float, default=0.12)
    ap.add_argument("--span", type=float, default=1.0)
    ap.add_argument("--d", type=float, default=0.0125, help="marker spacing ~ min mesh spacing")
    args = ap.parse_args()

    xi, eta, zeta, dS = (to_numpy(a) for a in make_wing(args.thickness, args.alpha, args.span, args.d))
    mid = np.abs(zeta) < 0.25*args.span                   # side surface, away from the caps
    i_le, i_te = np.argmin(xi[mid]), np.argmax(xi[mid])
    print(f"NACA 00{round(100*args.thickness):02d} wing, alpha = {args.alpha:g} deg, span {args.span:g}, d = {args.d:g}")
    print(f"markers: {xi.size} (Schur size {3*xi.size}, dense inverse "
          f"{(3*xi.size)**2*8/1e9:.1f} GB)")
    print(f"leading edge ({xi[mid][i_le]:+.4f}, {eta[mid][i_le]:+.4f}), trailing edge "
          f"({xi[mid][i_te]:+.4f}, {eta[mid][i_te]:+.4f}); "
          f"x in [{xi.min():.3f}, {xi.max():.3f}], y in [{eta.min():.3f}, {eta.max():.3f}], "
          f"z in [{zeta.min():.3f}, {zeta.max():.3f}]")
    print(f"area sum(dS) = {dS.sum():.4f}")
    out = os.path.join(HERE, "wing_geometry.npz")
    np.savez(out, xi=xi, eta=eta, zeta=zeta, dS=dS, alpha=args.alpha,
             thickness=args.thickness, span=args.span)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
