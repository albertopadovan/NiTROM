"""Select NumPy or PyTorch for newly constructed NiTROM objects.

``set_backend`` changes the construction default. Existing objects retain their
backend. Torch is imported only when its adapter is selected.
"""

from ._base import Backend

__all__ = ["Backend", "get_backend", "set_backend"]

_BACKENDS: dict[str, Backend] = {}
_ACTIVE: Backend | None = None


def set_backend(name: str) -> Backend:
    """Select the default array backend (``"numpy"`` or ``"torch"``)."""
    global _ACTIVE
    if name not in ("numpy", "torch"):
        raise ValueError(f"backend must be 'numpy' or 'torch', got {name!r}.")
    if name not in _BACKENDS:
        if name == "numpy":
            from ._numpy import NumpyBackend

            _BACKENDS[name] = NumpyBackend()
        else:
            from ._torch import TorchBackend

            _BACKENDS[name] = TorchBackend()
    _ACTIVE = _BACKENDS[name]
    return _ACTIVE


def get_backend() -> Backend:
    """Return the construction default, initially NumPy."""
    if _ACTIVE is None:
        return set_backend("numpy")
    return _ACTIVE
