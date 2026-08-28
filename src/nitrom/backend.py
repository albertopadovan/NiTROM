"""Array-backend abstraction for NiTROM.

NiTROM's math (operator inference, NiTROM adjoint, projections, time steppers)
is written against a small :class:`Backend` shim rather than ``torch`` directly,
so the same code runs on either **NumPy** (lightweight, ideal for small CPU
problems) or **PyTorch** (autograd, GPU).  The active backend is process-global
and chosen with :func:`set_backend`::

    from nitrom.backend import set_backend
    set_backend("numpy")   # or "torch"

Objects bind to whatever backend is active *when they are constructed*, so a
model built under NumPy stays NumPy even if the global backend later changes.

``torch`` is imported lazily (only when the torch backend or the distributed
helpers are used), so importing this module does not pull torch into a
NumPy-only session.
"""

import importlib
import os

# The default backend.  The whole stack is backend-agnostic, so NumPy (light on
# CPU / small problems) is the default; select torch explicitly with
# ``set_backend("torch")`` for autograd/GPU.
_DEFAULT_BACKEND = "numpy"

_BACKENDS: dict[str, "Backend"] = {}
_ACTIVE: "Backend | None" = None


class Backend:
    """Thin adapter exposing a unified array API over NumPy or PyTorch.

    Only the operations whose API differs between the two libraries are wrapped
    here; identically-named ops (``einsum``, ``zeros_like``, ``outer``, ...) are
    forwarded to :attr:`xp`, the underlying array module.
    """

    def __init__(self, name: str):
        if name == "torch":
            self.xp = importlib.import_module("torch")
        elif name == "numpy":
            self.xp = importlib.import_module("numpy")
        else:
            raise ValueError(
                f"backend must be 'numpy' or 'torch', got {name!r}."
            )
        self.name = name
        self.float64 = self.xp.float64
        self.float32 = self.xp.float32
        # Cache of numpy einsum contraction paths, keyed by equation + operand
        # shapes (see :meth:`einsum`).
        self._einsum_paths = {}

    @property
    def is_torch(self) -> bool:
        return self.name == "torch"

    @property
    def is_numpy(self) -> bool:
        return self.name == "numpy"

    # -- array creation (torch carries a device; numpy does not) -------------
    def zeros(self, shape, dtype=None, device="cpu"):
        if self.is_torch:
            return self.xp.zeros(shape, dtype=dtype, device=device)
        return self.xp.zeros(shape, dtype=dtype)

    def eye(self, n, dtype=None, device="cpu"):
        if self.is_torch:
            return self.xp.eye(n, dtype=dtype, device=device)
        return self.xp.eye(n, dtype=dtype)

    def asarray(self, x, dtype=None, device="cpu"):
        """Convert ``x`` to this backend's array type (and optionally recast)."""
        if self.is_torch:
            return self.xp.as_tensor(x, dtype=dtype, device=device)
        return self.xp.asarray(x, dtype=dtype)

    # -- identically-named ops, forwarded -----------------------------------
    def zeros_like(self, x):
        return self.xp.zeros_like(x)

    #: Naive-contraction cost above which deriving an optimized path pays off.
    #: Below it numpy's path machinery costs more than the contraction itself;
    #: measured crossover for the polynomial RHS at batch 9 sits between
    #: ``r = 12`` (cost 1.6e4, naive wins) and ``r = 16`` (cost 3.7e4,
    #: optimized wins).
    _EINSUM_OPTIMIZE_MIN_COST = 20_000

    #: Cap on :attr:`_einsum_paths`.  Keys include operand shapes, so a caller
    #: that varies batch sizes freely could otherwise grow the cache without
    #: bound; the hot equations are few, so a flush is cheap and rare.
    _EINSUM_CACHE_MAX = 512

    @staticmethod
    def _einsum_naive_cost(equation, operands):
        """Cost of the *unoptimized* contraction.

        ``numpy.einsum`` without a path runs one C loop over the product of
        every distinct index dimension, so that product is the cost to beat.

        This only steers a performance choice, so every ``zip`` here is
        deliberately non-strict: a subscript that does not line up with its
        operand should make the estimate wrong, never raise.
        """
        lhs = equation.split("->")[0].replace(" ", "")
        dims = {}
        broadcast = 1
        for subscript, op in zip(lhs.split(","), operands, strict=False):
            shape = getattr(op, "shape", ())
            if "..." in subscript:
                head, tail = subscript.split("...")
                n_ell = len(shape) - len(head) - len(tail)
                size = 1
                for s in shape[len(head): len(head) + n_ell]:
                    size *= s
                broadcast = max(broadcast, size)
                labelled = [
                    *zip(head, shape, strict=False),
                    *zip(tail, shape[len(head) + n_ell:], strict=False),
                ]
            else:
                labelled = zip(subscript, shape, strict=False)
            for c, s in labelled:
                dims[c] = s
        cost = broadcast
        for s in dims.values():
            cost *= s
        return cost

    def einsum(self, equation, *operands):
        """Contract ``operands`` according to ``equation``.

        ``numpy.einsum`` defaults to ``optimize=False``, which evaluates a
        multi-operand contraction with a single naive C loop rather than
        decomposing it into BLAS calls.  NiTROM's hot kernels are exactly that
        shape -- ``'abc,db,dc->da'`` for the polynomial RHS, ``'da,db,dc->abc'``
        for its parameter VJP -- and at ``r = 50`` the naive loop costs an order
        of magnitude more (measured 970 us vs 74 us).

        Optimizing is not free, though: deriving and parsing a path costs tens
        of microseconds, which *exceeds* the whole contraction at small ``r``.
        So the choice is made per contraction from its naive cost and cached
        per ``(equation, operand shapes)``, leaving low-dimensional ROMs on the
        path they were already taking.

        ``torch.einsum`` already chooses its own contraction order, so the
        torch backend forwards unchanged.
        """
        if self.is_torch:
            return self.xp.einsum(equation, *operands)

        try:
            # A list comprehension, not a generator: this key is built on every
            # contraction and at small r it is a visible fraction of one.
            key = (equation, *[op.shape for op in operands])
            path = self._einsum_paths[key]
        except AttributeError:  # an operand without .shape -- not worth caching
            return self.xp.einsum(equation, *operands)
        except KeyError:
            cost = self._einsum_naive_cost(equation, operands)
            path = (
                self.xp.einsum_path(equation, *operands, optimize="optimal")[0]
                if cost >= self._EINSUM_OPTIMIZE_MIN_COST
                else False
            )
            if len(self._einsum_paths) >= self._EINSUM_CACHE_MAX:
                self._einsum_paths.clear()
            self._einsum_paths[key] = path
        return self.xp.einsum(equation, *operands, optimize=path)

    def atleast_1d(self, x):
        return self.xp.atleast_1d(x)

    def outer(self, a, b):
        return self.xp.outer(a, b)

    def randn(self, shape, dtype=None, device="cpu"):
        """Standard-normal array of the given shape."""
        if self.is_torch:
            return self.xp.randn(shape, dtype=dtype, device=device)
        out = self.xp.random.standard_normal(shape)
        return out.astype(dtype) if dtype is not None else out

    def copy(self, x):
        """A copy of ``x`` (``clone`` on torch, ``copy`` on numpy)."""
        return x.clone() if self.is_torch else x.copy()

    def permute(self, x, axes):
        """Permute the axes of ``x`` (``permute`` on torch, ``transpose`` on numpy)."""
        return x.permute(*axes) if self.is_torch else self.xp.transpose(x, axes)

    def device_of(self, x):
        """The device of array ``x`` (always ``"cpu"`` for numpy)."""
        return x.device if self.is_torch else "cpu"

    def to_numpy(self, x):
        """Detach ``x`` to a plain numpy array (for e.g. scipy interop)."""
        if self.is_torch:
            return x.detach().cpu().numpy()
        return self.xp.asarray(x)

    # -- linalg (the norm API differs) --------------------------------------
    def vector_norm(self, x, axis=None):
        """Euclidean norm: flattened (``axis=None``) or along one axis."""
        if self.is_torch:
            if axis is None:
                return self.xp.linalg.vector_norm(x)
            return self.xp.linalg.vector_norm(x, dim=axis)
        return self.xp.linalg.norm(x, axis=axis)

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

    def solve(self, a, b):
        return self.xp.linalg.solve(a, b)

    def svd(self, x, full_matrices=False):
        """Economy SVD by default -> ``(U, S, Vh)``."""
        return self.xp.linalg.svd(x, full_matrices=full_matrices)

    def qr(self, x):
        return self.xp.linalg.qr(x)

    # -- shape / creation ops whose kwarg names differ ----------------------
    def stack(self, arrays, axis=0):
        if self.is_torch:
            return self.xp.stack(arrays, dim=axis)
        return self.xp.stack(arrays, axis=axis)

    def concatenate(self, arrays, axis=0):
        if self.is_torch:
            return self.xp.cat(arrays, dim=axis)
        return self.xp.concatenate(arrays, axis=axis)

    def arange(self, n, dtype=None, device="cpu"):
        if self.is_torch:
            return self.xp.arange(n, dtype=dtype, device=device)
        return self.xp.arange(n, dtype=dtype)

    def empty(self, shape, dtype=None, device="cpu"):
        if self.is_torch:
            return self.xp.empty(shape, dtype=dtype, device=device)
        return self.xp.empty(shape, dtype=dtype)

    def linspace(self, start, stop, num, dtype=None, device="cpu"):
        if self.is_torch:
            return self.xp.linspace(start, stop, num, dtype=dtype, device=device)
        return self.xp.linspace(start, stop, num, dtype=dtype)

    def repeat_interleave(self, x, n):
        """Repeat each element of ``x`` ``n`` times."""
        return x.repeat_interleave(n) if self.is_torch else self.xp.repeat(x, n)

    def sum(self, x, axis=None):
        """Sum over ``axis`` (all elements if ``axis`` is None)."""
        if axis is None:
            return x.sum()
        return x.sum(dim=axis) if self.is_torch else x.sum(axis=axis)

    def no_grad(self):
        """No-op context manager on numpy; ``torch.no_grad`` on torch."""
        if self.is_torch:
            return self.xp.no_grad()
        import contextlib
        return contextlib.nullcontext()

    def tensordot(self, a, b, axes):
        if self.is_torch:
            return self.xp.tensordot(a, b, dims=axes)
        return self.xp.tensordot(a, b, axes=axes)

    def searchsorted(self, sorted_seq, values):
        if self.is_torch:
            return self.xp.searchsorted(sorted_seq.contiguous(), values)
        return self.xp.searchsorted(sorted_seq, values)

    def clip(self, x, lo, hi):
        if self.is_torch:
            return self.xp.clamp(x, lo, hi)
        return self.xp.clip(x, lo, hi)

    def array_equal(self, a, b):
        return self.xp.equal(a, b) if self.is_torch else self.xp.array_equal(a, b)

    def mH(self, x):
        """Conjugate transpose of the last two axes (plain transpose for real)."""
        if self.is_torch:
            return x.mH
        return self.xp.conj(self.xp.swapaxes(x, -1, -2))

    def __getattr__(self, name):
        """Forward identical-API ops (sqrt, abs, where, diag, ...) to ``xp``."""
        try:
            xp = object.__getattribute__(self, "xp")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(xp, name)


def set_backend(name: str) -> Backend:
    """Select the process-global array backend (``"numpy"`` or ``"torch"``)."""
    global _ACTIVE
    if name not in _BACKENDS:
        _BACKENDS[name] = Backend(name)
    _ACTIVE = _BACKENDS[name]
    return _ACTIVE


def get_backend() -> Backend:
    """Return the active backend, initializing the default on first use."""
    if _ACTIVE is None:
        return set_backend(_DEFAULT_BACKEND)
    return _ACTIVE


def mpi_comm_world():
    """The MPI ``COMM_WORLD`` communicator, or ``None`` if MPI is unavailable."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except Exception:
        return None


def _resolve_comm(comm):
    """Return ``comm`` if given, else ``COMM_WORLD`` (raises if MPI missing)."""
    if comm is not None:
        return comm
    from mpi4py import MPI
    return MPI.COMM_WORLD


def comm_rank_size(comm):
    """``(rank, size)`` of a communicator, whichever flavor it is.

    Handles both APIs so the same ``comm`` argument works on either backend:

    * **mpi4py** communicator -> ``Get_rank()`` / ``Get_size()`` (capital ``G``);
    * **torch.distributed** process group (NCCL on GPU, or gloo) -> queried via
      ``torch.distributed.get_rank(group=comm)`` / ``get_world_size(group=comm)``,
      falling back to the group's ``rank()`` / ``size()`` methods.

    A torch process group has **no** ``Get_rank``/``Get_size`` -- that naming is
    mpi4py-only -- which is why we dispatch on the available interface.
    """
    if hasattr(comm, "Get_rank"):  # mpi4py communicator
        return comm.Get_rank(), comm.Get_size()
    try:  # torch.distributed process group
        import torch.distributed as dist
        return dist.get_rank(group=comm), dist.get_world_size(group=comm)
    except Exception:
        return comm.rank(), comm.size()


def mpi_rank_size(comm=None):
    """``(rank, size)`` of ``comm`` (default ``COMM_WORLD``); ``(0, 1)`` if MPI
    is unavailable."""
    try:
        comm = _resolve_comm(comm)
        return comm.Get_rank(), comm.Get_size()
    except Exception:
        return 0, 1


def distributed_rank_size():
    """Auto-detect ``(rank, world_size)`` for the active backend.

    * **NumPy**: from the MPI ``COMM_WORLD`` (``mpi4py``); ``(0, 1)`` if MPI is
      unavailable or the job is single-process (e.g. plain ``python``, no
      ``mpiexec``).
    * **PyTorch**: from ``torch.distributed`` when a process group is
      initialized, otherwise the ``torchrun`` environment (``RANK`` /
      ``WORLD_SIZE``), otherwise ``(0, 1)``.

    Lets data structures (e.g. :class:`~nitrom.training_data.TrainingPool`)
    shard correctly under ``mpiexec``/``torchrun`` even when the caller forgets
    to pass ``rank``/``world_size`` explicitly.
    """
    if get_backend().is_numpy:
        return mpi_rank_size()
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def mpi_allreduce_sum(x, comm=None):
    """Allreduce(SUM) a numpy array across ``comm`` (default ``COMM_WORLD``)."""
    import numpy as np
    from mpi4py import MPI

    comm = _resolve_comm(comm)
    x = np.ascontiguousarray(x)
    out = np.empty_like(x)
    comm.Allreduce(x, out, op=MPI.SUM)
    return out


def mpi_allreduce_scalar(v, comm=None):
    """Allreduce(SUM) a python scalar across ``comm`` (default ``COMM_WORLD``)."""
    from mpi4py import MPI

    return _resolve_comm(comm).allreduce(float(v), op=MPI.SUM)


def mpi_gather(obj, comm=None):
    """Gather python objects to root (list on rank 0, ``None`` elsewhere)."""
    return _resolve_comm(comm).gather(obj, root=0)


def mpi_bcast(obj, comm=None):
    """Broadcast a python object from root to all ranks."""
    return _resolve_comm(comm).bcast(obj, root=0)


def setup_distributed():
    r"""
    Initialize the PyTorch distributed process group for multi-GPU or
    multi-CPU training.

    When launched via ``torchrun``, the environment variables ``RANK``,
    ``LOCAL_RANK``, and ``WORLD_SIZE`` are read automatically.  The backend
    is selected based on hardware: **nccl** when CUDA is available,
    **gloo** otherwise.

    In single-process mode (no ``torchrun``), falls back to a local
    CUDA or CPU device with rank 0 and world size 1.

    :returns: ``(device, rank, world_size)``
    :rtype: tuple[torch.device, int, int]
    """
    import torch

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        init_kwargs = {
            "backend": backend,
            "init_method": "env://",
            "rank": rank,
            "world_size": world_size,
        }
        if backend == "nccl":
            init_kwargs["device_id"] = device

        torch.distributed.init_process_group(**init_kwargs)

        if backend == "nccl":
            torch.distributed.barrier(device_ids=[local_rank])
        else:
            torch.distributed.barrier()

        return device, rank, world_size
    else:
        # single-process fallback
        if torch.cuda.is_available():
            return torch.device("cuda"), 0, 1
        else:
            return torch.device("cpu"), 0, 1


def cleanup_distributed() -> None:
    """
    Destroy the distributed process group if one is active.

    Should be called at the end of the training script to release
    distributed resources cleanly.
    """
    import torch

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
