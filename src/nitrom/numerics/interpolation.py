"""Local interpolation on uniform and nonuniform time grids."""

from typing import Any

from ..backend import get_backend


def lagrange_stencils(t_samples, t_eval, order: int):
    r"""
    Local barycentric Lagrange interpolation weights on a (possibly non-uniform)
    sample grid.

    Every evaluation time uses the contiguous block of ``order`` samples that
    is most compact around it (smallest maximal distance), so on a grid that
    switches spacing the stencil leans toward the finely sampled side.  A time
    that coincides with a sample (to round-off) gets that sample exactly.

    :param t_samples: increasing sample times, length ``n``
    :param t_eval: evaluation times, inside ``[t_samples[0], t_samples[-1]]``
    :param order: points per stencil (polynomial degree ``order - 1``)
    :returns: ``(start, W)``: stencil ``e`` uses samples
        ``start[e] : start[e] + order`` with weights ``W[e]``
    """
    import numpy as np

    t = np.asarray(t_samples, dtype=float)
    te = np.atleast_1d(np.asarray(t_eval, dtype=float))
    n, p = len(t), int(order)
    if p < 1 or p > n:
        raise ValueError(f"order must be in [1, {n}], got {order}.")
    span = t[-1] - t[0]
    tol = 1e-10 * max(span, 1.0)
    if te.min() < t[0] - tol or te.max() > t[-1] + tol:
        raise ValueError(
            f"evaluation times [{te.min():g}, {te.max():g}] leave the sample "
            f"range [{t[0]:g}, {t[-1]:g}]."
        )
    start = np.empty(len(te), dtype=int)
    W = np.zeros((len(te), p))
    for e, tq in enumerate(te):
        i = int(np.searchsorted(t, tq))
        cand = np.arange(max(0, i - p), min(i, n - p) + 1)
        width = np.maximum(tq - t[cand], t[cand + p - 1] - tq)
        s0 = int(cand[np.argmin(width)])
        x = t[s0 : s0 + p]
        start[e] = s0
        hit = np.flatnonzero(np.abs(x - tq) <= tol)
        if hit.size:
            W[e, hit[0]] = 1.0
            continue
        bw = np.array(
            [1.0 / np.prod([x[k] - x[m] for m in range(p) if m != k]) for k in range(p)]
        )
        c = bw / (tq - x)
        W[e] = c / c.sum()
    return start, W


def interp_quadratic(t_eval: Any, t_data: Any, y_data: Any, *, backend=None) -> Any:
    r"""
    Piecewise quadratic (3-point Lagrange) interpolation of uniformly
    sampled data.

    For each query point in *t_eval*, the three nearest data points are
    used to build a degree-2 Lagrange polynomial:

    .. math::

        p(t) = \sum_{j=0}^{2} y_j \prod_{\substack{m=0 \\ m \neq j}}^{2}
               \frac{t - t_m}{t_j - t_m}

    The data in *y_data* may have arbitrary leading dimensions (e.g.
    ``(n, n_data)`` or ``(B, n, n_data)``); interpolation is always
    performed along the **last** axis.

    :param t_eval: query times of shape ``(n_eval,)``
    :param t_data: data times of shape ``(n_data,)``, must be sorted
    :param y_data: data values with time along the last axis, shape ``(..., n_data)``
    :returns: interpolated values, shape ``(..., n_eval)``
    :rtype: backend array
    """
    bkend = backend or get_backend()
    # Exact hit: querying at the data points themselves is the identity, so
    # skip the interpolation entirely.
    if t_eval.shape == t_data.shape and bool(bkend.array_equal(t_eval, t_data)):
        return y_data

    n_data = t_data.shape[0]

    # Find the index of the right neighbour for each query point.
    idx = bkend.clip(bkend.searchsorted(t_data, t_eval), 1, n_data - 1)

    # Centre index of the 3-point stencil, clamped so i-1, i, i+1 are valid.
    ic = bkend.clip(idx, 1, n_data - 2)

    t0 = t_data[ic - 1]  # (n_eval,)
    t1 = t_data[ic]
    t2 = t_data[ic + 1]

    # Lagrange basis values at t_eval
    L0 = ((t_eval - t1) * (t_eval - t2)) / ((t0 - t1) * (t0 - t2))
    L1 = ((t_eval - t0) * (t_eval - t2)) / ((t1 - t0) * (t1 - t2))
    L2 = ((t_eval - t0) * (t_eval - t1)) / ((t2 - t0) * (t2 - t1))

    # Gather data values at the stencil points: (..., n_eval)
    y0 = y_data[..., ic - 1]
    y1 = y_data[..., ic]
    y2 = y_data[..., ic + 1]

    return y0 * L0 + y1 * L1 + y2 * L2
