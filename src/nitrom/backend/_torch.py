"""Torch array adapter."""

from ._base import Backend


class TorchBackend(Backend):
    """NiTROM operations on torch arrays."""

    def __init__(self):
        super().__init__("torch")

    def einsum(self, equation, *operands):
        return self.xp.einsum(equation, *operands)

    def zeros(self, shape, dtype=None, device="cpu"):
        return self.xp.zeros(shape, dtype=dtype, device=device)

    def eye(self, n, dtype=None, device="cpu"):
        return self.xp.eye(n, dtype=dtype, device=device)

    def asarray(self, x, dtype=None, device="cpu"):
        """Convert ``x`` to this backend's array type (and optionally recast)."""
        return self.xp.as_tensor(x, dtype=dtype, device=device)

    def randn(self, shape, dtype=None, device="cpu"):
        """Standard-normal array of the given shape."""
        return self.xp.randn(shape, dtype=dtype, device=device)

    def copy(self, x):
        """Copy the array without changing its dtype or device."""
        return x.clone()

    def permute(self, x, axes):
        """Reorder array axes."""
        return x.permute(*axes)

    def ascontiguous(self, x):
        """Return a contiguous array, copying only when needed."""
        return x.contiguous()

    def device_of(self, x):
        """Return the array device."""
        return x.device

    def to_numpy(self, x):
        """Detach ``x`` to a plain numpy array (for e.g. scipy interop)."""
        return x.detach().cpu().numpy()

    def vector_norm(self, x, axis=None):
        """Euclidean norm: flattened (``axis=None``) or along one axis."""
        if axis is None:
            return self.xp.linalg.vector_norm(x)
        return self.xp.linalg.vector_norm(x, dim=axis)

    def solve(self, a, b):
        """Solve ``a x = b`` for vectors, matrices, or batches of either."""
        return self.xp.linalg.solve(a, b)

    def stack(self, arrays, axis=0):
        return self.xp.stack(arrays, dim=axis)

    def concatenate(self, arrays, axis=0):
        return self.xp.cat(arrays, dim=axis)

    def arange(self, n, dtype=None, device="cpu"):
        return self.xp.arange(n, dtype=dtype, device=device)

    def empty(self, shape, dtype=None, device="cpu"):
        return self.xp.empty(shape, dtype=dtype, device=device)

    def linspace(self, start, stop, num, dtype=None, device="cpu"):
        return self.xp.linspace(start, stop, num, dtype=dtype, device=device)

    def repeat_interleave(self, x, n):
        """Repeat each element of ``x`` ``n`` times."""
        return x.repeat_interleave(n)

    def sum(self, x, axis=None):
        """Sum over ``axis`` (all elements if ``axis`` is None)."""
        if axis is None:
            return x.sum()
        return x.sum(dim=axis)

    def no_grad(self):
        """Disable autograd."""
        return self.xp.no_grad()

    def tensordot(self, a, b, axes):
        return self.xp.tensordot(a, b, dims=axes)

    def searchsorted(self, sorted_seq, values):
        return self.xp.searchsorted(sorted_seq.contiguous(), values)

    def clip(self, x, lo, hi):
        return self.xp.clamp(x, lo, hi)

    def array_equal(self, a, b):
        return self.xp.equal(a, b)

    def mH(self, x):
        """Conjugate transpose of the last two axes (plain transpose for real)."""
        return x.mH

    def jacobian(self, f, t_eval, y, args, kwargs):
        """State Jacobian, with independent rows for batched right-hand sides."""
        n = y.shape[-1]
        dev = self.device_of(y)
        import torch
        from torch.func import jacrev

        y_detached = y.detach().requires_grad_(True)

        def single_f(u):
            return f(t_eval, u, *args, **kwargs)

        if y.ndim == 1:
            return jacrev(single_f)(y_detached).detach()

        B_size = y.shape[0]

        # Each output row depends only on its own input row, so the batched
        # Jacobian is block diagonal.  One jacrev over the whole batch and then
        # taking the diagonal blocks costs a single RHS trace; differentiating
        # row by row instead re-evaluates the *full* batched RHS once per row,
        # i.e. O(B^2) work for the same numbers.  The full Jacobian is
        # (B, n, B, n), so fall back to the per-row loop when that would be
        # large.
        if B_size * B_size * n * n <= 1 << 20:
            J_full = jacrev(single_f)(y_detached).detach()  # (B, n, B, n)
            idx = torch.arange(B_size, device=J_full.device)
            return J_full[idx, :, idx, :]

        J = self.zeros((B_size, n, n), dtype=y.dtype, device=dev)
        for b in range(B_size):

            def single_f_batched(u, b=b):
                y_list = [
                    y_detached[i].detach() if i != b else u for i in range(B_size)
                ]
                return f(t_eval, torch.stack(y_list, dim=0), *args, **kwargs)[b]

            J[b] = jacrev(single_f_batched)(y_detached[b]).detach()
        return J
