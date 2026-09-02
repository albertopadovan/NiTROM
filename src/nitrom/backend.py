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


def _resolve_c_einsum(xp):
    """Bind numpy's raw C einsum, bypassing the Python dispatcher.

    ``numpy.einsum`` runs an array-function dispatcher and kwargs validation
    before reaching the C loop.  :meth:`Backend.einsum` calls it in a tight loop
    where that prologue is a visible fraction of the contraction, so prefer the
    C entry point and fall back to the public API if numpy ever moves it.
    """
    for mod_name in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            return importlib.import_module(mod_name).c_einsum
        except (ImportError, AttributeError):
            continue
    return xp.einsum


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
        # Cache of compiled numpy einsum contraction plans, keyed by equation +
        # operand shapes (see :meth:`einsum`).
        self._einsum_paths = {}
        self._c_einsum = None if self.is_torch else _resolve_c_einsum(self.xp)

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

    #: Naive-contraction cost above which compiling a plan pays off.  Below it
    #: the replay's own transpose/reshape/``dot`` calls cost more than letting
    #: the C loop run once.
    #:
    #: Replay moved the real crossover well below this: measured on Sapphire
    #: Rapids, ``'abc,db,dc->da'`` at ``r = 8``, batch 7 (cost 3.6e3) already
    #: favours the plan, and at ``r = 12``, batch 7 it wins 6.6 us to 13.7 us.
    #: The bound is deliberately left where it was anyway.  Decomposing a
    #: contraction into BLAS reassociates its sums, so lowering it perturbs
    #: results at the 1e-16 level, and for the ``r = 50`` models this library is
    #: trained at it buys nothing -- every hot kernel is already far above the
    #: line.  Lower it only alongside a small-``r`` benchmark that shows the win.
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
        of magnitude more.

        Passing ``optimize=<cached path>`` does *not* buy that speedup at
        NiTROM's sizes, though.  numpy re-derives its entire contraction list on
        every call whenever ``optimize`` is not ``False``; a cached path only
        skips the order *search*, leaving the subscript parsing and
        BLAS-eligibility bookkeeping to run per call.  Trajectory-parallel runs
        give each rank a batch of one, where that overhead exceeds the
        arithmetic outright -- on the cavity gradient a cached path measured
        *slower* end to end than no optimization at all (2.57 s vs 2.48 s).

        So the contraction is *compiled* once per ``(equation, operand shapes)``
        into a plan of transpose/reshape/``dot`` steps (see
        :meth:`_compile_einsum_plan`) and replayed, and contractions too small
        to be worth decomposing go straight to the C entry point rather than
        back through numpy's Python wrapper.  Same cavity gradient, one Sapphire
        Rapids core: 2.57 s to 1.40 s at batch 1, 3.37 s to 2.39 s at batch 7.

        ``torch.einsum`` already chooses its own contraction order, so the
        torch backend forwards unchanged.
        """
        if self.is_torch:
            return self.xp.einsum(equation, *operands)

        try:
            # A list comprehension, not a generator: this key is built on every
            # contraction and at small r it is a visible fraction of one.
            key = (equation, *[op.shape for op in operands])
            plan = self._einsum_paths[key]
        except AttributeError:  # an operand without .shape -- not worth caching
            return self.xp.einsum(equation, *operands)
        except KeyError:
            cost = self._einsum_naive_cost(equation, operands)
            plan = (
                self._compile_einsum_plan(equation, operands)
                if cost >= self._EINSUM_OPTIMIZE_MIN_COST
                else None
            )
            if len(self._einsum_paths) >= self._EINSUM_CACHE_MAX:
                self._einsum_paths.clear()
            self._einsum_paths[key] = plan

        if plan is None:
            return self._c_einsum(equation, *operands)

        ops = list(operands)
        for inds, arg, perm in plan:
            tmp = [ops.pop(i) for i in inds]
            if arg.__class__ is str:
                ops.append(self._c_einsum(arg, *tmp))
                continue
            axes_a, shape_a, axes_b, shape_b, out_shape = arg
            a, b = tmp
            if axes_a is not None:
                a = a.transpose(axes_a)
            if axes_b is not None:
                b = b.transpose(axes_b)
            view = self.xp.dot(a.reshape(shape_a), b.reshape(shape_b))
            view = view.reshape(out_shape)
            # ``transpose`` leaves a non-contiguous view, which is what numpy's
            # own optimized einsum hands back too (it closes with ``asanyarray``
            # at ``order='K'``).  The buffer underneath is this step's fresh
            # ``dot`` output, so nothing else aliases it.
            ops.append(view if perm is None else view.transpose(perm))
        return ops[0]

    def _compile_einsum_plan(self, equation, operands):
        """Compile ``equation`` into a replayable list of contraction steps.

        Each step is ``(inds, arg, perm)``: ``inds`` are the operand slots it
        consumes (reverse-sorted, so popping them in order is safe), and ``arg``
        is either an einsum subscript string -- evaluated by the C loop, for the
        steps numpy judged ineligible for BLAS -- or the pre-solved
        ``tensordot`` decomposition ``(axes_a, shape_a, axes_b, shape_b,
        out_shape)``.  ``perm`` is a trailing axis permutation, or ``None``.

        Pre-solving the tensordot matters as much as caching the path: at these
        sizes ``numpy.tensordot`` spends more time deriving its transposes and
        reshapes than BLAS spends on the multiply.  Every shape is known at
        compile time, so all of it is hoisted out of the hot loop and only
        ``transpose``/``reshape``/``dot`` remain.

        Returns ``None`` when no plan can be built, so the caller falls back.
        """
        try:
            _, contraction_list = self.xp.einsum_path(
                equation, *operands, optimize="optimal", einsum_call=True
            )
        except TypeError:
            # ``einsum_call`` is numpy-internal; if it ever goes away, degrade
            # to the naive contraction rather than failing.
            return None

        shapes = [tuple(op.shape) for op in operands]
        steps = []
        try:
            for inds, idx_rm, einsum_str, _remaining, blas in contraction_list:
                if "..." in einsum_str:
                    return None  # broadcasting: leave the shape algebra to numpy
                in_str, out_str = einsum_str.split("->")
                in_subs = in_str.split(",")
                tmp_shapes = [shapes.pop(i) for i in inds]

                dims = {}
                for sub, shape in zip(in_subs, tmp_shapes, strict=True):
                    for c, s in zip(sub, shape, strict=True):
                        dims[c] = s
                shapes.append(tuple(dims[c] for c in out_str))

                if not blas:
                    steps.append((tuple(inds), einsum_str, None))
                    continue

                left, right = in_subs
                # numpy pairs the contracted axes by sorted label; match it
                # exactly so the plan reproduces its result bit for bit.
                removed = sorted(idx_rm)
                arg = self._plan_tensordot(
                    tmp_shapes[0],
                    tmp_shapes[1],
                    [left.index(s) for s in removed],
                    [right.index(s) for s in removed],
                )
                tensor_result = left + right
                for s in idx_rm:
                    tensor_result = tensor_result.replace(s, "")
                perm = (
                    tuple(tensor_result.index(c) for c in out_str)
                    if tensor_result != out_str
                    else None
                )
                steps.append((tuple(inds), arg, perm))
        except (ValueError, IndexError, KeyError):
            # Any subscript this shape algebra does not model (repeated or
            # unpaired labels in a layout we did not anticipate) is a reason to
            # hand the contraction back to numpy, never to fail the solve.
            return None
        return tuple(steps)

    @staticmethod
    def _plan_tensordot(shape_a, shape_b, axes_a, axes_b):
        """Pre-solve ``numpy.tensordot``'s shape algebra for known shapes.

        Mirrors the body of :func:`numpy.tensordot`: the contracted axes move to
        the end of ``a`` and the front of ``b``, both operands collapse to 2-D,
        and the matrix product is reshaped back.  A permutation that is already
        the identity is returned as ``None`` so the replay can skip the call.
        """
        nda, ndb = len(shape_a), len(shape_b)
        notin_a = [k for k in range(nda) if k not in axes_a]
        notin_b = [k for k in range(ndb) if k not in axes_b]

        n_contract = 1
        for k in axes_a:
            n_contract *= shape_a[k]
        rows = 1
        for k in notin_a:
            rows *= shape_a[k]
        cols = 1
        for k in notin_b:
            cols *= shape_b[k]

        newaxes_a = tuple(notin_a + list(axes_a))
        newaxes_b = tuple(list(axes_b) + notin_b)
        out_shape = tuple(
            [shape_a[k] for k in notin_a] + [shape_b[k] for k in notin_b]
        )
        return (
            None if newaxes_a == tuple(range(nda)) else newaxes_a,
            (rows, n_contract),
            None if newaxes_b == tuple(range(ndb)) else newaxes_b,
            (n_contract, cols),
            out_shape,
        )

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

    def ascontiguous(self, x):
        """``x`` in C-contiguous layout, without copying if it already is.

        :meth:`einsum` hands back a transposed view of its ``dot`` output (see
        the note there), so a tensor *assembled* from contractions can carry a
        permuted stride order.  That is free to produce and fine to consume
        once, but an operand stored for reuse is re-read by every contraction
        that follows, and the plans in :meth:`einsum` reshape their operands --
        which silently copies whenever the layout is not contiguous.  Making
        the layout canonical at the write path pays for the copy once instead
        of on every evaluation.
        """
        if self.is_torch:
            return x.contiguous()
        return self.xp.ascontiguousarray(x)

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
        """Solve ``a x = b``, with ``b`` a batch of vectors or of matrices.

        NumPy 2.0 dropped the ambiguous ``(..., M, M) @ (..., M)`` vector-batch
        form of ``linalg.solve`` (it now reads a 2-D ``b`` as a single matrix),
        while torch still accepts it.  Normalise to the torch semantics so
        callers -- e.g. the batched Newton solve and the implicit-stage adjoint
        in :mod:`nitrom.time_steppers` -- behave the same on both backends.
        """
        if not self.is_torch and b.ndim == a.ndim - 1:
            return self.xp.linalg.solve(a, b[..., None])[..., 0]
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
