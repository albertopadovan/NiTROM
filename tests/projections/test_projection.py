"""Tests for the abstract Projection class."""

import pytest

from nitrom.projections.projection import Projection

torch = pytest.importorskip("torch")

REQUIRED_METHODS = (
    "get_params",
    "update",
    "encode",
    "decode",
    "vjp_encode",
    "vjp_decode",
    "vjp_decode_state",
)


class _CompleteProjection(Projection):
    def get_params(self):
        return []

    def update(self, *args, **kwargs):
        pass

    def encode(self, q):
        return q

    def decode(self, z):
        return z

    def vjp_encode(self, q, v, *args, **kwargs):
        return ()

    def vjp_decode(self, z, v, *args, **kwargs):
        return ()

    def vjp_decode_state(self, z, v, *args, **kwargs):
        return z


class TestProjectionAbstract:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError, match="abstract"):
            Projection(n=4, r=4, param_names=[])

    @pytest.mark.parametrize("missing_method", REQUIRED_METHODS)
    def test_must_implement_method(self, missing_method):
        implementations = {
            name: getattr(_CompleteProjection, name)
            for name in REQUIRED_METHODS
            if name != missing_method
        }
        Incomplete = type("Incomplete", (Projection,), implementations)
        assert Incomplete.__abstractmethods__ == {missing_method}
        with pytest.raises(TypeError, match=rf"\b{missing_method}\b"):
            Incomplete(n=4, r=4, param_names=[])

    def test_complete_subclass_instantiates(self):
        proj = _CompleteProjection(n=4, r=4, param_names=[])
        x = torch.randn(4)
        assert proj.encode(x).shape == x.shape
        assert proj.decode(x).shape == x.shape


pytestmark = pytest.mark.usefixtures("torch_backend")
