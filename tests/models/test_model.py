"""Tests for the abstract Model class."""

import pytest
import torch
from nitrom.latent_space_models.model import Model


def _make_complete_class():
    """Return a minimal concrete subclass of Model."""

    class Complete(Model):
        def get_params(self):
            return []

        def update_params(self, *args, **kwargs):
            pass

        def evaluate_rhs(self, t, z, **kwargs):
            return torch.zeros_like(z)

        def evaluate_adjoint_rhs(self, t, z, Z, **kwargs):
            return torch.zeros_like(z)

        def vjp_evaluate_rhs(self, z, v):
            return []

    return Complete


class TestModelAbstract:
    """Verify that Model cannot be instantiated and enforces the interface."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Model(4, [])

    def test_must_implement_get_params(self):
        class Incomplete(Model):
            def update_params(self, *args, **kwargs):
                pass

            def evaluate_rhs(self, t, z, **kwargs):
                return torch.zeros_like(z)

            def evaluate_adjoint_rhs(self, t, z, Z, **kwargs):
                return torch.zeros_like(z)

            def vjp_evaluate_rhs(self, z, v):
                return []

        with pytest.raises(TypeError):
            Incomplete(4, [])

    def test_must_implement_update_params(self):
        class Incomplete(Model):
            def get_params(self):
                return []

            def evaluate_rhs(self, t, z, **kwargs):
                return torch.zeros_like(z)

            def evaluate_adjoint_rhs(self, t, z, Z, **kwargs):
                return torch.zeros_like(z)

            def vjp_evaluate_rhs(self, z, v):
                return []

        with pytest.raises(TypeError):
            Incomplete(4, [])

    def test_must_implement_evaluate_rhs(self):
        class Incomplete(Model):
            def get_params(self):
                return []

            def update_params(self, *args, **kwargs):
                pass

            def evaluate_adjoint_rhs(self, t, z, Z, **kwargs):
                return torch.zeros_like(z)

            def vjp_evaluate_rhs(self, z, v):
                return []

        with pytest.raises(TypeError):
            Incomplete(4, [])

    def test_must_implement_evaluate_adjoint_rhs(self):
        class Incomplete(Model):
            def get_params(self):
                return []

            def update_params(self, *args, **kwargs):
                pass

            def evaluate_rhs(self, t, z, **kwargs):
                return torch.zeros_like(z)

            def vjp_evaluate_rhs(self, z, v):
                return []

        with pytest.raises(TypeError):
            Incomplete(4, [])

    def test_must_implement_vjp_evaluate_rhs(self):
        class Incomplete(Model):
            def get_params(self):
                return []

            def update_params(self, *args, **kwargs):
                pass

            def evaluate_rhs(self, t, z, **kwargs):
                return torch.zeros_like(z)

            def evaluate_adjoint_rhs(self, t, z, Z, **kwargs):
                return torch.zeros_like(z)

        with pytest.raises(TypeError):
            Incomplete(4, [])

    def test_complete_subclass_instantiates(self):
        Complete = _make_complete_class()
        model = Complete(4, [])
        z = torch.randn(4)
        assert model.param_names == []
        assert model.state_dimension == 4
        assert model.evaluate_rhs(0.0, z).shape == z.shape
        assert model.evaluate_adjoint_rhs(0.0, z, z).shape == z.shape
