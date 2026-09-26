"""The ROM domain D = [-4, 13] x [-3, 3] and restriction of fields to it.

Section 5 of the paper truncates the models' domain to D, a subset of the
simulation domain [-5, 16] x [-4, 4], to keep the small spurious artifacts
that form at the outflow out of the data; D then holds nx x ny = 706 x 430
grid points.  That count is the cells whose centres lie strictly inside D,
so every staggered grid here keeps the points strictly inside D as well:
u on 706 x 430 faces, v on 706 x 429 faces, n = 606,454 ~ 2 nx ny.  (A
non-strict test would add the column of u faces lying exactly on x = -4,
the boundary of D, and give 707.)

Everything that plots trajectories or trains a ROM goes through ``crop_mask``
(state vectors) or ``vorticity`` (plots), so the domain is defined once.
"""

from __future__ import annotations

import numpy as np

X_BOUNDS = (-4.0, 13.0)
Y_BOUNDS = (-3.0, 3.0)


def _inside(a, bounds):
    return (a > bounds[0]) & (a < bounds[1])


def grid_masks(coords):
    """1-D keep-masks (u_x, u_y, v_x, v_y) over the interior staggered coords.

    ``coords`` is anything indexable by "xu", "yu", "xv", "yv" holding the
    solver's coordinate vectors WITH their two ghost points (a snapshot .npz).
    """
    xu, yu = coords["xu"][1:-1], coords["yu"][1:-1]
    xv, yv = coords["xv"][1:-1], coords["yv"][1:-1]
    return (_inside(xu, X_BOUNDS), _inside(yu, Y_BOUNDS),
            _inside(xv, X_BOUNDS), _inside(yv, Y_BOUNDS))


def rom_coords(coords):
    """Interior (xu, yu, xv, yv) restricted to D."""
    ux, uy, vx, vy = grid_masks(coords)
    return (coords["xu"][1:-1][ux], coords["yu"][1:-1][uy],
            coords["xv"][1:-1][vx], coords["yv"][1:-1][vy])


def crop_mask(coords):
    """Boolean mask over the stacked full-grid (u, v) state selecting D."""
    ux, uy, vx, vy = grid_masks(coords)
    return np.concatenate([(uy[:, None] & ux[None, :]).ravel(),
                           (vy[:, None] & vx[None, :]).ravel()])


def vorticity(q, coords):
    """Vorticity dv/dx - du/dy of a D-restricted state q, at cell corners.

    Corners sit at (xu, yv).  dv/dx is only defined between two v columns, so
    the u columns outside [xv[0], xv[-1]] are dropped.  Returns (X, Y, omega).
    """
    xu, yu, xv, yv = rom_coords(coords)
    n_u = len(yu)*len(xu)
    u = q[:n_u].reshape(len(yu), len(xu))
    v = q[n_u:].reshape(len(yv), len(xv))
    keep = (xu > xv[0]) & (xu < xv[-1])
    dv_dx = np.diff(v, axis=1)/np.diff(xv)[None, :]
    du_dy = (np.diff(u, axis=0)/np.diff(yu)[:, None])[:, keep]
    X, Y = np.meshgrid(xu[keep], yv)
    return X, Y, dv_dx - du_dy
