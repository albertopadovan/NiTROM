"""Tests for the abstract Projection class."""

import pytest
import torch
from nitrom.projections.projection import Projection


class TestProjectionAbstract:

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Projection()

    def test_must_implement_update(self):
        class Incomplete(Projection):
            def encode(self, q):
                return q

            def decode(self, z):
                return z

            def vjp_encode(self, q, v, *args, **kwargs):
                return ()

            def vjp_decode(self, z, v, *args, **kwargs):
                return ()

        with pytest.raises(TypeError):
            Incomplete()

    def test_must_implement_encode(self):
        class Incomplete(Projection):
            def update(self, *args, **kwargs):
                pass

            def decode(self, z):
                return z

            def vjp_encode(self, q, v, *args, **kwargs):
                return ()

            def vjp_decode(self, z, v, *args, **kwargs):
                return ()

        with pytest.raises(TypeError):
            Incomplete()

    def test_must_implement_decode(self):
        class Incomplete(Projection):
            def update(self, *args, **kwargs):
                pass

            def encode(self, q):
                return q

            def vjp_encode(self, q, v, *args, **kwargs):
                return ()

            def vjp_decode(self, z, v, *args, **kwargs):
                return ()

        with pytest.raises(TypeError):
            Incomplete()

    def test_must_implement_vjp_encode(self):
        class Incomplete(Projection):
            def update(self, *args, **kwargs):
                pass

            def encode(self, q):
                return q

            def decode(self, z):
                return z

            def vjp_decode(self, z, v, *args, **kwargs):
                return ()

        with pytest.raises(TypeError):
            Incomplete()

    def test_must_implement_vjp_decode(self):
        class Incomplete(Projection):
            def update(self, *args, **kwargs):
                pass

            def encode(self, q):
                return q

            def decode(self, z):
                return z

            def vjp_encode(self, q, v, *args, **kwargs):
                return ()

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_subclass_instantiates(self):
        class Complete(Projection):
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

        proj = Complete(n=4, r=4, param_names=[])
        x = torch.randn(4)
        assert proj.encode(x).shape == x.shape
        assert proj.decode(x).shape == x.shape
