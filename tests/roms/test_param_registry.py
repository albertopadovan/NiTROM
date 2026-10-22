"""Shared parameters remain tied and ordered across both components."""

from types import SimpleNamespace

import numpy as np
import pytest

from nitrom.backend import set_backend
from nitrom.projections import LinearProjection
from nitrom.roms import ParamRegistry


def test_shared_parameter_scatter_and_consistency(backend):
    phi = backend.asarray(np.eye(4, 2))
    projection = LinearProjection([phi, backend.copy(phi)])
    model = SimpleNamespace(
        backend=backend,
        param_names=["A", "Phi"],
        A=backend.eye(2, dtype=backend.float64),
        Phi=backend.copy(phi),
    )
    model.get_params = lambda: [model.A, model.Phi]

    def update(values):
        model.A, model.Phi = values

    model.update_params = update
    registry = ParamRegistry(model, projection)
    assert registry.names == ["A", "Phi", "Psi"]
    assert registry.shared_names == ["Phi"]
    registry.assert_shared_consistent()
    values = [backend.copy(value) for value in registry.values()]
    values[1] = values[1] * 2.0
    registry.scatter(values)
    assert model.Phi is projection.Phi
    assert model.Phi is values[1]
    pytest.importorskip("torch")
    set_backend("numpy" if backend.is_torch else "torch")
    registry.assert_shared_consistent()
    model.Phi = model.Phi * 2.0
    with pytest.raises(ValueError, match="has drifted"):
        registry.assert_shared_consistent()
