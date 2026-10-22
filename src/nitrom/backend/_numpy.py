"""Numpy array adapter."""

from ._base import Backend
from ._einsum import NumpyEinsum, _resolve_c_einsum


class NumpyBackend(NumpyEinsum, Backend):
    """NiTROM operations on numpy arrays."""

    def __init__(self):
        super().__init__("numpy")
        self._einsum_paths = {}
        self._c_einsum = _resolve_c_einsum(self.xp)

    def zeros(self, shape, dtype=None, device="cpu"):
        return self.xp.zeros(shape, dtype=dtype)

    def eye(self, n, dtype=None, device="cpu"):
        return self.xp.eye(n, dtype=dtype)

    def asarray(self, x, dtype=None, device="cpu"):
        """Convert ``x`` to this backend's array type (and optionally recast)."""
        return self.xp.asarray(x, dtype=dtype)

    def randn(self, shape, dtype=None, device="cpu"):
        """Standard-normal array of the given shape."""
        out = self.xp.random.standard_normal(shape)
        return out.astype(dtype) if dtype is not None else out

    def copy(self, x):
        """Copy the array without changing its dtype or device."""
        return x.copy()

    def permute(self, x, axes):
        """Reorder array axes."""
        return self.xp.transpose(x, axes)

    def ascontiguous(self, x):
        """Return a contiguous array, copying only when needed."""
        return self.xp.ascontiguousarray(x)

    def device_of(self, x):
        """Return the array device."""
        return "cpu"

    def to_numpy(self, x):
        """Detach ``x`` to a plain numpy array (for e.g. scipy interop)."""
        return self.xp.asarray(x)

    def vector_norm(self, x, axis=None):
        """Euclidean norm: flattened (``axis=None``) or along one axis."""
        return self.xp.linalg.norm(x, axis=axis)

    def solve(self, a, b):
        """Solve ``a x = b``, with ``b`` a batch of vectors or of matrices.

        NumPy 2.0 dropped the ambiguous ``(..., M, M) @ (..., M)`` vector-batch
        form of ``linalg.solve`` (it now reads a 2-D ``b`` as a single matrix),
        while torch still accepts it.  Normalise to the torch semantics so
        callers -- e.g. the batched Newton solve and the implicit-stage adjoint
        in :mod:`nitrom.time_steppers` -- behave the same on both backends.
        """
        if b.ndim == a.ndim - 1:
            return self.xp.linalg.solve(a, b[..., None])[..., 0]
        return self.xp.linalg.solve(a, b)

    def stack(self, arrays, axis=0):
        return self.xp.stack(arrays, axis=axis)

    def concatenate(self, arrays, axis=0):
        return self.xp.concatenate(arrays, axis=axis)

    def arange(self, n, dtype=None, device="cpu"):
        return self.xp.arange(n, dtype=dtype)

    def empty(self, shape, dtype=None, device="cpu"):
        return self.xp.empty(shape, dtype=dtype)

    def linspace(self, start, stop, num, dtype=None, device="cpu"):
        return self.xp.linspace(start, stop, num, dtype=dtype)

    def repeat_interleave(self, x, n):
        """Repeat each element of ``x`` ``n`` times."""
        return self.xp.repeat(x, n)

    def sum(self, x, axis=None):
        """Sum over ``axis`` (all elements if ``axis`` is None)."""
        if axis is None:
            return x.sum()
        return x.sum(axis=axis)

    def no_grad(self):
        """No-op context manager for backend-neutral numerical code."""
        import contextlib

        return contextlib.nullcontext()

    def tensordot(self, a, b, axes):
        return self.xp.tensordot(a, b, axes=axes)

    def searchsorted(self, sorted_seq, values):
        return self.xp.searchsorted(sorted_seq, values)

    def clip(self, x, lo, hi):
        return self.xp.clip(x, lo, hi)

    def array_equal(self, a, b):
        return self.xp.array_equal(a, b)

    def mH(self, x):
        """Conjugate transpose of the last two axes (plain transpose for real)."""
        return self.xp.conj(self.xp.swapaxes(x, -1, -2))

    def jacobian(self, f, t_eval, y, args, kwargs):
        """State Jacobian, with independent rows for batched right-hand sides."""
        n = y.shape[-1]
        dev = self.device_of(y)
        # numpy: forward finite difference (columns of J), batched together.
        eps = 1e-7
        f0 = f(t_eval, y, *args, **kwargs)
        if y.ndim == 1:
            J = self.zeros((n, n), dtype=y.dtype, device=dev)
            for j in range(n):
                yp = self.copy(y)
                yp[j] += eps
                J[:, j] = (f(t_eval, yp, *args, **kwargs) - f0) / eps
            return J

        B_size = y.shape[0]
        J = self.zeros((B_size, n, n), dtype=y.dtype, device=dev)
        for j in range(n):
            yp = self.copy(y)
            yp[:, j] += eps
            # f is batched and each output row depends only on its own input row.
            J[:, :, j] = (f(t_eval, yp, *args, **kwargs) - f0) / eps
        return J
