"""Riemannian L-BFGS on non-trivial objectives.

These exercise the full geometry -- retraction, tangent projection, and vector
transport of the curvature memory -- by driving the optimizer to a *known*
optimum on the Stiefel and Grassmann manifolds (and, as a sanity check, on the
trivial Euclidean manifold and a mixed product manifold).  The canonical test
is the Rayleigh problem

    min_{X in St(N, r)}  -tr(X^T A X),

whose minimizer spans the top-r eigenspace of the symmetric matrix ``A`` and
whose optimal value is minus the sum of the r largest eigenvalues.
"""

import numpy as np
import pytest

from nitrom.backend import set_backend
from nitrom.optimization.manifold_optimization import (
    inner,
    project,
    retract,
    riemannian_lbfgs,
    strong_wolfe_line_search,
    to_manifold,
    transport,
)


@pytest.fixture(autouse=True)
def _numpy_backend():
    # riemannian_lbfgs works in numpy; restore torch for the rest of the suite.
    set_backend("numpy")
    yield
    set_backend("torch")


def _rayleigh(N, r, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((N, N))
    A = 0.5 * (M + M.T)
    evals = np.sort(np.linalg.eigvalsh(A))[::-1]
    return A, -float(evals[:r].sum())


@pytest.mark.parametrize("manifold", ["stiefel", "grassmann"])
def test_rayleigh_reaches_top_eigenspace(manifold):
    """Minimizer stays orthonormal and attains the top-r eigenvalue sum."""
    N, r = 8, 3
    A, f_opt = _rayleigh(N, r, seed=0)

    def cost_fn(xs):
        X = xs[0]
        return -float((X * (A @ X)).sum())

    def rgrad_fn(xs):
        X = xs[0]
        return [project(X, -2.0 * (A @ X), manifold)]

    rng = np.random.default_rng(1)
    X0 = to_manifold(rng.standard_normal((N, r)), manifold)
    xs, f = riemannian_lbfgs(
        cost_fn, rgrad_fn, [X0], [manifold],
        max_iter=300, gtol=1e-12, ftol=1e-15,
    )
    X = xs[0]
    assert np.allclose(X.T @ X, np.eye(r), atol=1e-10)  # on the manifold
    assert f == pytest.approx(f_opt, abs=1e-6)          # global optimum


def test_euclidean_quadratic():
    """All-Euclidean product manifold reduces to ordinary L-BFGS on a quadratic."""
    A = np.diag([1.0, 3.0, 10.0])
    b = np.array([1.0, -2.0, 0.5])
    x_star = np.linalg.solve(A, b)

    def cost_fn(xs):
        x = xs[0]
        return float(0.5 * x @ (A @ x) - b @ x)

    def rgrad_fn(xs):
        return [A @ xs[0] - b]

    xs, _ = riemannian_lbfgs(
        cost_fn, rgrad_fn, [np.zeros(3)], ["euclidean"],
        max_iter=100, gtol=1e-13, ftol=1e-18,
    )
    assert np.allclose(xs[0], x_star, atol=1e-8)


def test_product_manifold_stiefel_and_euclidean():
    """Mixed Stiefel + Euclidean factors are optimized jointly to their optima."""
    N, r = 6, 2
    A, f_opt = _rayleigh(N, r, seed=3)
    c = np.array([2.0, -1.0])

    def cost_fn(xs):
        X, y = xs
        return -float((X * (A @ X)).sum()) + 0.5 * float((y - c) @ (y - c))

    def rgrad_fn(xs):
        X, y = xs
        return [project(X, -2.0 * (A @ X), "stiefel"), y - c]

    rng = np.random.default_rng(2)
    X0 = to_manifold(rng.standard_normal((N, r)), "stiefel")
    xs, f = riemannian_lbfgs(
        cost_fn, rgrad_fn, [X0, np.zeros(2)], ["stiefel", "euclidean"],
        max_iter=300, gtol=1e-12, ftol=1e-15,
    )
    X, y = xs
    assert np.allclose(X.T @ X, np.eye(r), atol=1e-10)  # Stiefel factor on manifold
    assert np.allclose(y, c, atol=1e-7)                 # Euclidean factor at optimum
    assert f == pytest.approx(f_opt, abs=1e-6)


@pytest.mark.parametrize("manifold", ["stiefel", "grassmann"])
def test_strong_wolfe_conditions_hold(manifold):
    """The returned step satisfies both strong-Wolfe conditions along R_x(t d)."""
    N, r = 8, 3
    A, _ = _rayleigh(N, r, seed=11)
    c1, c2 = 1e-4, 0.9

    def cost_fn(xs):
        X = xs[0]
        return -float((X * (A @ X)).sum())

    def rgrad_fn(xs):
        X = xs[0]
        return [project(X, -2.0 * (A @ X), manifold)]

    rng = np.random.default_rng(4)
    X0 = to_manifold(rng.standard_normal((N, r)), manifold)
    g0 = rgrad_fn([X0])
    d = [-g0[0]]  # steepest-descent direction (a valid descent direction)
    f0 = cost_fn([X0])
    dphi0 = inner(g0[0], d[0])

    t, y, f, g = strong_wolfe_line_search(
        cost_fn, rgrad_fn, [X0], d, [manifold], f0, g0, c1=c1, c2=c2
    )
    # Armijo (sufficient decrease):
    assert f <= f0 + c1 * t * dphi0 + 1e-12
    # Strong curvature, with the transported directional derivative:
    dphi_t = inner(g[0], transport(y[0], d[0], manifold))
    assert abs(dphi_t) <= -c2 * dphi0 + 1e-9
    # The trial point genuinely came from the retraction (stays on-manifold).
    assert np.allclose(y[0].T @ y[0], np.eye(r), atol=1e-10)
    assert np.allclose(y[0], retract(X0, t * d[0], manifold), atol=1e-12)


def test_transport_keeps_history_consistent():
    """A longer run on a curved manifold converges (would stall/diverge if the
    L-BFGS history were not vector-transported between tangent spaces)."""
    N, r = 12, 4
    A, f_opt = _rayleigh(N, r, seed=7)

    def cost_fn(xs):
        X = xs[0]
        return -float((X * (A @ X)).sum())

    def rgrad_fn(xs):
        X = xs[0]
        return [project(X, -2.0 * (A @ X), "stiefel")]

    rng = np.random.default_rng(5)
    X0 = to_manifold(rng.standard_normal((N, r)), "stiefel")
    xs, f = riemannian_lbfgs(
        cost_fn, rgrad_fn, [X0], ["stiefel"],
        max_iter=500, history_size=20, gtol=1e-12, ftol=1e-15,
    )
    assert np.allclose(xs[0].T @ xs[0], np.eye(r), atol=1e-10)
    assert f == pytest.approx(f_opt, abs=1e-6)
