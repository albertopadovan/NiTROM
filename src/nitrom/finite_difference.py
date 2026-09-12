"""High-order finite-difference differentiation of trajectory snapshots.

Used by :class:`nitrom.training_data.TrainingPool` to synthesize time
derivatives from trajectory snapshots when no precomputed derivative file is
available on disk.
"""

from __future__ import annotations

import numpy as np


def fd_weights(z: float, x: np.ndarray, m: int = 1) -> np.ndarray:
    """Finite-difference weights for the ``m``-th derivative at ``z``.

    Implements Fornberg's algorithm (Fornberg, 1988, "Generation of Finite
    Difference Formulas on Arbitrarily Spaced Grids"), so the same code path
    handles uniform and non-uniform node spacing ``x`` alike.

    :returns: length-``len(x)`` weight vector ``c`` such that
        ``f^{(m)}(z) ~= c @ f(x)``.
    """
    n = len(x) - 1
    c1 = 1.0
    c4 = x[0] - z
    C = np.zeros((n + 1, m + 1))
    C[0, 0] = 1.0
    for i in range(1, n + 1):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    C[i, k] = c1 * (k * C[i - 1, k - 1] - c5 * C[i - 1, k]) / c2
                C[i, 0] = -c1 * c5 * C[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                C[j, k] = (c4 * C[j, k] - k * C[j, k - 1]) / c3
            C[j, 0] = c4 * C[j, 0] / c3
        c1 = c2
    return C[:, m]


def _stencil_indices(i: int, n: int, width: int) -> np.ndarray:
    """Indices of a ``width``-point stencil centered on ``i``, clipped to ``[0, n)``."""
    width = min(width, n)
    lo = i - width // 2
    lo = max(0, min(lo, n - width))
    return np.arange(lo, lo + width)


def time_derivative(X: np.ndarray, t: np.ndarray, stencil_size: int = 5) -> np.ndarray:
    """First time-derivative of snapshots ``X`` sampled at times ``t``.

    Each point uses a ``stencil_size``-point finite-difference stencil
    (Fornberg weights recomputed from the true node spacing, so this works
    for non-uniform time grids too); the default of 5 gives fourth-order
    accuracy on a uniform grid. Stencils are one-sided near the ends of the
    trajectory and centered elsewhere.

    :param X: snapshots, shape ``(..., n_snapshots)`` -- differentiated
        along the last axis.
    :param t: sample times, shape ``(n_snapshots,)``.
    :param stencil_size: number of points per stencil (clipped to
        ``n_snapshots`` if the trajectory is shorter than that).
    :returns: array shaped like ``X`` holding ``dX/dt``.
    """
    n_snapshots = X.shape[-1]
    if n_snapshots < 2:
        raise ValueError(
            f"Need at least 2 snapshots to estimate a time derivative, got {n_snapshots}."
        )
    dX = np.empty_like(X)
    for i in range(n_snapshots):
        idx = _stencil_indices(i, n_snapshots, stencil_size)
        w = fd_weights(t[i], t[idx], m=1)
        dX[..., i] = X[..., idx] @ w
    return dX
