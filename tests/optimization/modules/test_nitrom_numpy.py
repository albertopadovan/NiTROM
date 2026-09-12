"""Finite-difference gradient checks for :class:`NitromModule` on the **numpy** backend.

The sibling :mod:`test_nitrom` suite runs under the autouse ``set_backend("torch")``
fixture in ``tests/conftest.py``, so it never exercises the numpy code path -- which
is the one production runs on (MPI trajectory parallelism uses the numpy backend).
The two backends are not interchangeable in practice: ``numpy.linalg.solve`` dropped
the ``(..., M, M) @ (..., M)`` vector-batch form in NumPy 2.0 that torch still
accepts, which silently broke ``backward_euler`` on numpy until
:meth:`nitrom.backend.Backend.solve` normalised it.  These tests pin that path down.

Everything here is numpy-native: the module stand-ins, the finite-difference
reference, and the comparison.

Note the deliberate asymmetry with :mod:`test_nitrom`: there, the *continuous*
adjoint is checked against the (finite-difference-verified) discrete adjoint,
which is far cheaper.  Here it is checked against an actual finite difference.
Keeping one independent finite-difference check of the continuous adjoint in the
suite means a hypothetical bug shared by both adjoint implementations still has
something to fail against.  These cases are small enough that it is affordable.
"""

import numpy as np
import pytest

from nitrom.backend import set_backend
from nitrom.latent_space_models.atr_polynomial_model import AtrPolynomialModel
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import NitromModule
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry

N, R, NTRAJ, NT, M, NO = 5, 2, 2, 4, 1, 3


@pytest.fixture(autouse=True)
def _pin_numpy_backend():
    """Override the session-wide torch pin from ``tests/conftest.py``."""
    set_backend("numpy")
    yield
    set_backend("torch")


class _Data:
    """Minimal :class:`TrainingData` stand-in (X, dX, weights, time, forcing)."""

    def __init__(self, X, weights=None, forcing_fns=None):
        self.X = X
        self.dX = None
        ntraj, _, nt = X.shape
        self.weights = np.ones(ntraj) if weights is None else weights
        self.time = np.linspace(0.0, 1.0, nt)
        self.forcing_fns = forcing_fns or []


class _LinearOutputFOM:
    """Full-order model with a linear output ``y = C x``."""

    def __init__(self, C):
        self.C = C

    def compute_output(self, q):
        return np.matmul(self.C, q)

    def compute_output_derivative(self, q):
        return self.C


def _random_basis(n, r, seed):
    return np.linalg.qr(np.random.default_rng(seed).standard_normal((n, r)))[0]


def _data(forcing=False):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((NTRAJ, N, NT))
    weights = 1.0 + 0.5 * rng.random(NTRAJ)
    ffns = (
        [(lambda t, c=0.5 * (i + 1): np.full(M, c)) for i in range(NTRAJ)]
        if forcing
        else None
    )
    return _Data(X, weights=weights, forcing_fns=ffns)


# Same reasoning (and the same measured convergence rates) as the _n_substeps
# note in test_nitrom.py: the discrete adjoint is exact on any grid, the
# continuous one converges at the order of the stepper.
_N_SUBSTEPS_CONTINUOUS = {"rk4": 25, "rk2": 100, "backward_euler": 50}


def _n_substeps(adjoint_method: str, time_stepper: str = "rk4") -> int:
    if adjoint_method == "discrete":
        return 10
    return _N_SUBSTEPS_CONTINUOUS[time_stepper]


def _module(model, forcing=False, n_substeps=10, **kw):
    registry = ParamRegistry(model, LinearProjection(
        [_random_basis(N, R, 1), _random_basis(N, R, 2)]
    ))
    data = _data(forcing)
    module = NitromModule(
        data, registry,
        fom=_LinearOutputFOM(np.random.default_rng(4).standard_normal((NO, N))),
        reg=0.01, n_substeps=n_substeps, n_leggauss=5, **kw,
    )
    module.forcing_fns = data.forcing_fns
    return module


def _finite_diff_grad(module, eps=1e-6):
    """Central finite difference of ``module()`` w.r.t. every parameter."""
    fd = []
    for p in module.parameters():
        g = np.zeros_like(p)
        flat, gflat = p.reshape(-1), g.reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            jp = float(module())
            flat[i] = orig - eps
            jm = float(module())
            flat[i] = orig
            gflat[i] = (jp - jm) / (2.0 * eps)
        fd.append(g)
    return fd


def _assert_grad_close(analytic, numeric, rtol, atol):
    assert len(analytic) == len(numeric)
    for i, (a, n) in enumerate(zip(analytic, numeric, strict=True)):
        assert a.shape == n.shape, f"param {i}: {a.shape} vs {n.shape}"
        np.testing.assert_allclose(
            a, n, rtol=rtol, atol=atol, err_msg=f"param {i} gradient mismatch"
        )


def _poly_model(forcing=False, seed=2):
    rng = np.random.default_rng(seed)
    tensors = [0.3 * rng.standard_normal((R, R)), 0.2 * rng.standard_normal((R, R, R))]
    if forcing:
        tensors.append(0.3 * rng.standard_normal((R, M)))
    return PolynomialModel(
        R, [1, 2], dtype=np.float64, tensors=tensors,
        forcing_config={"forcing_exists": True, "m": M} if forcing else None,
    )


def _gas_params(seed, extra_forcing):
    rng = np.random.default_rng(seed)
    eye = np.eye(R)
    params = [
        0.3 * rng.standard_normal((R, R)),          # K
        eye + 0.1 * rng.standard_normal((R, R)),    # R
        eye + 0.1 * rng.standard_normal((R, R)),    # Q
        0.2 * rng.standard_normal((R, R, R)),       # S
    ]
    if extra_forcing:
        params.append(0.3 * rng.standard_normal((R, M)))
    return params


@pytest.mark.parametrize("adjoint_method", ["discrete", "continuous"])
@pytest.mark.parametrize("time_stepper", ["rk4", "rk2", "backward_euler"])
def test_gradient_polynomial_rom(adjoint_method, time_stepper):
    # The continuous adjoint agrees with the finite difference only to the
    # discretization error of the forward solve, and backward_euler is first
    # order -- the deviation halves as n_substeps doubles -- so that pairing
    # needs a finer sub-step grid.  See the note on _N_SUBSTEPS in test_nitrom.py.
    fine = adjoint_method == "continuous" and time_stepper == "backward_euler"
    module = _module(
        _poly_model(), time_stepper=time_stepper, adjoint_method=adjoint_method,
        n_substeps=_n_substeps(adjoint_method, time_stepper),
    )
    rtol, atol = (3e-2, 3e-2) if fine else (1e-4, 1e-6)
    _assert_grad_close(module.gradient(), _finite_diff_grad(module), rtol, atol)


@pytest.mark.parametrize("adjoint_method", ["discrete", "continuous"])
@pytest.mark.parametrize("time_stepper", ["rk4", "backward_euler"])
def test_gradient_gas_rom(adjoint_method, time_stepper):
    model = GasPolynomialModel(R, [1, 2], dtype=np.float64,
                               gas_params=_gas_params(3, False))
    fine = adjoint_method == "continuous" and time_stepper == "backward_euler"
    module = _module(model, time_stepper=time_stepper, adjoint_method=adjoint_method,
                     n_substeps=_n_substeps(adjoint_method, time_stepper))
    # Diagonal entries S[i, i, k] cancel in the H = S - S^T assembly, so their true
    # gradient is exactly 0 while the finite difference picks up solver-tolerance
    # noise there -- see the same allowance in test_nitrom.py.
    if fine:
        rtol, atol = 3e-2, 3e-2
    elif time_stepper == "backward_euler":
        rtol, atol = 1e-4, 5e-6
    else:
        rtol, atol = 1e-4, 1e-6
    _assert_grad_close(module.gradient(), _finite_diff_grad(module), rtol, atol)


@pytest.mark.parametrize("adjoint_method", ["discrete", "continuous"])
def test_gradient_atr_rom(adjoint_method):
    rng = np.random.default_rng(5)
    params = _gas_params(5, False)
    params += [0.3 * rng.standard_normal(R), 0.3 * rng.standard_normal(R)]  # Bhat, m
    model = AtrPolynomialModel(R, [1, 2], dtype=np.float64, atr_params=params)
    module = _module(model, adjoint_method=adjoint_method,
                     n_substeps=_n_substeps(adjoint_method))
    _assert_grad_close(module.gradient(), _finite_diff_grad(module), 1e-4, 1e-6)


@pytest.mark.parametrize("adjoint_method", ["discrete", "continuous"])
@pytest.mark.parametrize("kind", ["poly", "gas"])
def test_gradient_with_forcing(kind, adjoint_method):
    """Exercises the grad_B path, which the non-forcing cases skip entirely."""
    if kind == "poly":
        model = _poly_model(forcing=True)
    else:
        model = GasPolynomialModel(
            R, [1, 2], dtype=np.float64, gas_params=_gas_params(3, True),
            forcing_config={"forcing_exists": True, "m": M},
        )
    module = _module(model, forcing=True, adjoint_method=adjoint_method,
                     n_substeps=_n_substeps(adjoint_method))
    _assert_grad_close(module.gradient(), _finite_diff_grad(module), 1e-4, 1e-6)


def test_multiple_trajectories_batched():
    """More than one trajectory per rank -- the batched einsum/VJP path."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((4, N, NT))
    registry = ParamRegistry(_poly_model(), LinearProjection(
        [_random_basis(N, R, 1), _random_basis(N, R, 2)]
    ))
    module = NitromModule(
        _Data(X), registry,
        fom=_LinearOutputFOM(rng.standard_normal((NO, N))),
        reg=0.01, n_substeps=10, adjoint_method="discrete",
    )
    _assert_grad_close(module.gradient(), _finite_diff_grad(module), 1e-4, 1e-6)
