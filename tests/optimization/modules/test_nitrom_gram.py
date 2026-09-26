"""Gram-form output mismatch in :class:`NitromModule` against the general path.

With a :class:`LinearProjection` and a FOM whose output is the state
(``output_is_state = True``), :class:`NitromModule` evaluates the full-space
mismatch as ``||x||^2 - 2 (Sz)^T Phi^T x + (Sz)^T Phi^T Phi (Sz)`` and never
stores the reconstruction.  It is an exact rewrite of the same cost, so the cost
and every gradient block must match the general path (``gram=False``) to
round-off, on both backends and for both adjoints.
"""

import numpy as np
import pytest
import torch

from nitrom.backend import set_backend
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.optimization import NitromModule
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry

N, R, NTRAJ, NT = 7, 3, 3, 6


class _StateFOM:
    """y = x, declared so the module may use the Gram form."""

    output_is_state = True

    def compute_output(self, q):
        return q

    def compute_output_derivative(self, q):  # pragma: no cover
        raise NotImplementedError

    def apply_output_adjoint(self, e, q):
        return e


class _Data:
    def __init__(self, X, weights, time):
        self.X, self.weights, self.time = X, weights, time
        self.dX = None
        self.forcing_fns = []


def _build(backend, gram, adjoint="discrete"):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((NTRAJ, N, NT))
    weights = 1.0 + rng.random(NTRAJ)
    time = np.linspace(0.0, 1.0, NT)
    Phi = np.linalg.qr(rng.standard_normal((N, R)))[0]
    Psi = np.linalg.qr(Phi + 0.3 * rng.standard_normal((N, R)))[0]  # oblique
    if backend == "torch":
        conv = lambda a: torch.as_tensor(a, dtype=torch.float64)
    else:
        conv = lambda a: a
    model = GasPolynomialModel(R, [1, 2], dtype=np.float64 if backend == "numpy"
                               else torch.float64)
    proj = LinearProjection([conv(Phi), conv(Psi)])
    return NitromModule(
        _Data(conv(X), conv(weights), conv(time)), ParamRegistry(model, proj),
        fom=_StateFOM(), n_substeps=10, adjoint_method=adjoint, gram=gram,
    )


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("adjoint", ["discrete", "continuous"])
def test_gram_matches_general_path(backend, adjoint):
    set_backend(backend)
    try:
        slow, fast = _build(backend, False, adjoint), _build(backend, True, adjoint)
        # identical random GAS parameters in both modules
        fast.registry.model.update_params(slow.registry.model.get_params())
        for name in fast.registry.names:
            setattr(fast, name, getattr(slow, name))
        assert not slow.gram and fast.gram

        c0, c1 = float(slow()), float(fast())
        assert abs(c1 - c0) <= 1e-13 * abs(c0)
        for name, g0, g1 in zip(slow.registry.names, slow.gradient(),
                                fast.gradient()):
            g0, g1 = _np(g0), _np(g1)
            assert np.abs(g1 - g0).max() <= 1e-12 * max(np.abs(g0).max(), 1e-300), name
    finally:
        set_backend("torch")


def test_gram_cache_follows_phi_changes():
    """Phi^T X is cached; an in-place change of Phi must invalidate it."""
    set_backend("numpy")
    try:
        fast, slow = _build("numpy", True), _build("numpy", False)
        fast.registry.model.update_params(slow.registry.model.get_params())
        for name in fast.registry.names:
            setattr(fast, name, getattr(slow, name))
        float(fast())                               # fill the cache
        for mod in (fast, slow):
            mod.Phi[...] = mod.Phi + 0.1 * np.sin(np.arange(mod.Phi.size)
                                                  ).reshape(mod.Phi.shape)
        c0, c1 = float(slow()), float(fast())
        assert abs(c1 - c0) <= 1e-13 * abs(c0)
    finally:
        set_backend("torch")


def test_gram_requires_state_output():
    class _Other:
        def compute_output(self, q):
            return q

    set_backend("numpy")
    try:
        with pytest.raises(ValueError):
            m = _build("numpy", None)
            NitromModule(m.training_data, m.registry, fom=_Other(), gram=True)
        assert not NitromModule(m.training_data, m.registry, fom=_Other()).gram
    finally:
        set_backend("torch")
