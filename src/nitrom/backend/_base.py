"""Shared array operations and backend identity."""

import importlib


class Backend:
    """Array operations used by models, projections, solvers, and training."""

    def __init__(self, name):
        self.name = name
        self.xp = importlib.import_module(name)
        self.float64 = self.xp.float64
        self.float32 = self.xp.float32

    def validate_arrays(self, **arrays):
        """Reject arrays from another backend rather than silently moving data."""
        array_type = self.xp.Tensor if self.is_torch else self.xp.ndarray
        for name, array in arrays.items():
            if array is not None and not isinstance(array, array_type):
                raise TypeError(
                    f"{name} must be a {self.name} array, got {type(array).__name__}"
                )

    def validate_components(self, *components):
        """Require model, projection, and data to use the same array backend."""
        for component in components:
            other = getattr(component, "backend", self)
            if other.name != self.name:
                raise ValueError(f"Incompatible backends: {self.name} and {other.name}")
            for name in ("X", "dX", "weights", "time"):
                array = getattr(component, name, None)
                if array is not None:
                    self.validate_arrays(**{name: array})

    @property
    def is_torch(self) -> bool:
        return self.name == "torch"

    @property
    def is_numpy(self) -> bool:
        return self.name == "numpy"

    def zeros_like(self, x):
        return self.xp.zeros_like(x)

    def atleast_1d(self, x):
        return self.xp.atleast_1d(x)

    def outer(self, a, b):
        return self.xp.outer(a, b)

    def inv(self, x):
        return self.xp.linalg.inv(x)

    def cholesky(self, x):
        """Lower-triangular Cholesky factor ``L`` with ``L L^T = x``."""
        return self.xp.linalg.cholesky(x)

    def eigvals(self, x):
        return self.xp.linalg.eigvals(x)

    def eigh(self, x):
        """Eigendecomposition of a symmetric matrix -> ``(eigvals, eigvecs)``."""
        return self.xp.linalg.eigh(x)

    def svd(self, x, full_matrices=False):
        """Economy SVD by default -> ``(U, S, Vh)``."""
        return self.xp.linalg.svd(x, full_matrices=full_matrices)

    def qr(self, x):
        return self.xp.linalg.qr(x)

    def __getattr__(self, name):
        """Forward identical-API ops (sqrt, abs, where, diag, ...) to ``xp``."""
        try:
            xp = object.__getattribute__(self, "xp")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(xp, name)
