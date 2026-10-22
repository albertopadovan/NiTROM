"""Quadrature rules for trajectory integrals."""


def composite_gauss_legendre(breaks, n_per_panel: int):
    r"""
    Composite Gauss-Legendre rule on the panels ``[breaks[i], breaks[i+1]]``.

    Exact for polynomials of degree ``2 n_per_panel - 1`` on every panel.
    Panels may have any widths, so a rule can be fine where the integrand
    changes quickly (e.g. just after an impulse) and coarse elsewhere.

    :param breaks: increasing panel end points, length ``n_panels + 1``
    :param n_per_panel: nodes per panel
    :returns: ``(nodes, weights)``, float64 numpy arrays, nodes ascending
    """
    import numpy as np

    breaks = np.asarray(breaks, dtype=float)
    if breaks.ndim != 1 or len(breaks) < 2 or np.any(np.diff(breaks) <= 0):
        raise ValueError("breaks must be a strictly increasing 1-D array.")
    x, w = np.polynomial.legendre.leggauss(int(n_per_panel))
    a, b = breaks[:-1, None], breaks[1:, None]
    nodes = (0.5 * (b - a) * x[None, :] + 0.5 * (a + b)).ravel()
    weights = (0.5 * (b - a) * w[None, :]).ravel()
    return nodes, weights
