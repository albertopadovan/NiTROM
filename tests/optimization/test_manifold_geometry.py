"""Direct unit tests for the matrix-manifold geometry primitives.

Unlike the optimizer tests (which exercise projection / retraction / transport
*indirectly* through convergence), these assert the defining algebraic
properties of each primitive in isolation, so a bug is caught even when
optimization happens to still converge.  Vector transport gets the most
scrutiny: its whole purpose is to move a tangent vector from one tangent space
into another, so the central check is that a transported vector genuinely lands
in the *target* tangent space.

Manifolds under test:

* Stiefel  St(N, r): tangent at X is ``{ T : X^T T + T^T X = 0 }`` (X^T T skew);
* Grassmann Gr(N, r): (horizontal) tangent at X is ``{ T : X^T T = 0 }``;
* Euclidean: every operation is the trivial vector-space one.
"""

import numpy as np
import pytest

from nitrom.backend import set_backend
from nitrom.optimization.manifold_optimization import (
    inner,
    project,
    retract,
    to_manifold,
    transport,
)

N, R = 8, 3
MATRIX_MANIFOLDS = ["stiefel", "grassmann"]


@pytest.fixture(autouse=True)
def _numpy_backend():
    set_backend("numpy")
    yield
    set_backend("torch")


def _point(seed, manifold, n=N, r=R):
    """A random point on the manifold (orthonormal for Stiefel/Grassmann)."""
    rng = np.random.default_rng(seed)
    return to_manifold(rng.standard_normal((n, r)), manifold)


def _tangent(X, seed, manifold):
    """A random tangent vector at X (an ambient vector projected to T_X)."""
    rng = np.random.default_rng(seed)
    return project(X, rng.standard_normal(X.shape), manifold)


def _assert_tangent(X, T, manifold, atol=1e-11):
    """Assert T lies in the tangent space at X."""
    XtT = X.T @ T
    if manifold == "grassmann":
        assert np.allclose(XtT, 0.0, atol=atol)            # X^T T = 0
    else:
        assert np.allclose(XtT + XtT.T, 0.0, atol=atol)    # X^T T skew-symmetric


# --------------------------------------------------------------------------
# Tangent projection (= Riemannian gradient under the embedded metric)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("manifold", MATRIX_MANIFOLDS)
def test_projection_lands_in_tangent_space(manifold):
    X = _point(0, manifold)
    G = np.random.default_rng(1).standard_normal((N, R))
    _assert_tangent(X, project(X, G, manifold), manifold)


@pytest.mark.parametrize("manifold", MATRIX_MANIFOLDS)
def test_projection_is_idempotent(manifold):
    """Projecting a vector that is already tangent leaves it unchanged."""
    X = _point(0, manifold)
    T = _tangent(X, 2, manifold)
    assert np.allclose(project(X, T, manifold), T, atol=1e-12)


@pytest.mark.parametrize("manifold", MATRIX_MANIFOLDS)
def test_projection_removes_only_normal_component(manifold):
    """G - project(G) is orthogonal (Frobenius) to every tangent vector."""
    X = _point(3, manifold)
    G = np.random.default_rng(4).standard_normal((N, R))
    normal = G - project(X, G, manifold)
    for s in range(5, 9):
        assert inner(normal, _tangent(X, s, manifold)) == pytest.approx(0.0, abs=1e-11)


def test_projection_euclidean_is_identity():
    G = np.random.default_rng(5).standard_normal((4, 2))
    assert np.array_equal(project(np.zeros((4, 2)), G, "euclidean"), G)


# --------------------------------------------------------------------------
# Retraction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("manifold", MATRIX_MANIFOLDS)
def test_retraction_stays_on_manifold(manifold):
    X = _point(4, manifold)
    Y = retract(X, 0.3 * _tangent(X, 5, manifold), manifold)
    assert np.allclose(Y.T @ Y, np.eye(R), atol=1e-12)


@pytest.mark.parametrize("manifold", MATRIX_MANIFOLDS)
def test_retraction_at_zero_is_identity(manifold):
    X = _point(6, manifold)
    assert np.allclose(retract(X, np.zeros_like(X), manifold), X, atol=1e-12)


@pytest.mark.parametrize("manifold", MATRIX_MANIFOLDS)
def test_retraction_is_first_order(manifold):
    """A retraction satisfies (R_X(t xi) - X)/t -> xi as t -> 0."""
    X = _point(7, manifold)
    xi = _tangent(X, 8, manifold)
    xi = xi / np.linalg.norm(xi)
    t = 1e-6
    approx = (retract(X, t * xi, manifold) - X) / t
    assert np.allclose(approx, xi, atol=1e-4)


def test_retraction_euclidean_is_addition():
    X = np.random.default_rng(9).standard_normal((4, 2))
    xi = np.random.default_rng(10).standard_normal((4, 2))
    assert np.allclose(retract(X, xi, "euclidean"), X + xi)


# --------------------------------------------------------------------------
# Vector transport  (the primary focus)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("manifold", MATRIX_MANIFOLDS)
def test_transport_lands_in_target_tangent_space(manifold):
    """The core property: a vector tangent at X, transported to Y, is tangent
    at Y.  A transport that forgot to re-project (or projected at the wrong
    point) would fail this."""
    X = _point(11, manifold)
    xi = _tangent(X, 12, manifold)
    Y = retract(X, 0.5 * _tangent(X, 13, manifold), manifold)
    _assert_tangent(Y, transport(Y, xi, manifold), manifold)


@pytest.mark.parametrize("manifold", MATRIX_MANIFOLDS)
def test_transport_of_ambient_vector_lands_in_target(manifold):
    """Even an arbitrary ambient vector transports into the target tangent
    space (transport-by-projection accepts any input)."""
    Y = _point(14, manifold)
    v = np.random.default_rng(15).standard_normal((N, R))
    _assert_tangent(Y, transport(Y, v, manifold), manifold)


@pytest.mark.parametrize("manifold", MATRIX_MANIFOLDS)
def test_transport_onto_same_point_is_identity(manifold):
    """Transporting a tangent vector into its own tangent space is a no-op."""
    X = _point(16, manifold)
    xi = _tangent(X, 17, manifold)
    assert np.allclose(transport(X, xi, manifold), xi, atol=1e-12)


@pytest.mark.parametrize("manifold", [*MATRIX_MANIFOLDS, "euclidean"])
def test_transport_is_linear(manifold):
    """Vector transport is a linear map: T(a u + b v) = a T(u) + b T(v)."""
    if manifold == "euclidean":
        rng = np.random.default_rng(18)
        Y = rng.standard_normal((N, R))
        u = rng.standard_normal((N, R))
        v = rng.standard_normal((N, R))
    else:
        X = _point(18, manifold)
        Y = retract(X, 0.4 * _tangent(X, 19, manifold), manifold)
        u = _tangent(X, 20, manifold)
        v = _tangent(X, 21, manifold)
    a, b = 2.0, -3.0
    lhs = transport(Y, a * u + b * v, manifold)
    rhs = a * transport(Y, u, manifold) + b * transport(Y, v, manifold)
    assert np.allclose(lhs, rhs, atol=1e-12)


@pytest.mark.parametrize("manifold", MATRIX_MANIFOLDS)
def test_transport_is_contractive_in_frobenius_norm(manifold):
    """Transport by orthogonal projection never increases the norm."""
    X = _point(22, manifold)
    xi = _tangent(X, 23, manifold)
    Y = retract(X, 0.6 * _tangent(X, 24, manifold), manifold)
    Txi = transport(Y, xi, manifold)
    assert inner(Txi, Txi) <= inner(xi, xi) + 1e-12


def test_transport_euclidean_is_identity():
    Y = np.random.default_rng(25).standard_normal((5, 2))
    xi = np.random.default_rng(26).standard_normal((5, 2))
    assert np.array_equal(transport(Y, xi, "euclidean"), xi)


# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------

def test_inner_is_symmetric_and_frobenius():
    U = np.random.default_rng(27).standard_normal((4, 3))
    V = np.random.default_rng(28).standard_normal((4, 3))
    assert inner(U, V) == pytest.approx(inner(V, U))
    assert inner(U, V) == pytest.approx(float(np.sum(U * V)))


# --------------------------------------------------------------------------
# Riemannian gradient consistency (project + retract, jointly, vs finite diff)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("manifold", MATRIX_MANIFOLDS)
def test_riemannian_gradient_matches_finite_difference(manifold):
    """<rgrad, xi> == d/dt f(R_X(t xi))|_0, via a centered difference.  This
    couples the projection (the Riemannian gradient) and the retraction: a bug
    in either breaks the identity."""
    rng = np.random.default_rng(29)
    M = rng.standard_normal((N, N))
    A = 0.5 * (M + M.T)

    def f(Z):
        return -float((Z * (A @ Z)).sum())

    X = _point(30, manifold)
    rgrad = project(X, -2.0 * (A @ X), manifold)
    xi = _tangent(X, 31, manifold)
    xi = xi / np.linalg.norm(xi)
    t = 1e-6
    fd = (f(retract(X, t * xi, manifold)) - f(retract(X, -t * xi, manifold))) / (2 * t)
    assert inner(rgrad, xi) == pytest.approx(fd, abs=1e-6)
