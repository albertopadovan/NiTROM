"""Shared pytest configuration.

The library default backend is numpy, but the bulk of the test suite is written
against torch (it constructs torch tensors and, for a few tests, uses autograd).
This autouse fixture pins the torch backend before every test so those tests run
as written; the backend-specific tests in ``test_backend.py`` set/restore the
backend explicitly within each test, which composes fine with this pin.
"""

import pytest

from nitrom.backend import set_backend


@pytest.fixture(autouse=True)
def _pin_torch_backend():
    set_backend("torch")
    yield
