"""Explicit backend fixtures and isolation of process-global defaults."""

import pytest

from nitrom.backend import get_backend, set_backend


@pytest.fixture(autouse=True)
def _restore_backend():
    previous = get_backend()
    yield
    set_backend(previous.name)


@pytest.fixture
def numpy_backend():
    previous = get_backend()
    backend = set_backend("numpy")
    yield backend
    set_backend(previous.name)


@pytest.fixture
def torch_backend():
    pytest.importorskip("torch")
    previous = get_backend()
    backend = set_backend("torch")
    yield backend
    set_backend(previous.name)


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    if request.param == "torch":
        pytest.importorskip("torch")
    previous = get_backend()
    backend = set_backend(request.param)
    yield backend
    set_backend(previous.name)
