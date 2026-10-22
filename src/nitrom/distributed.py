"""MPI and Torch process groups and collectives; imported only when used."""

import os
import sys

from .backend import get_backend


def mpi_comm_world():
    """Return the active MPI communicator without starting MPI in serial jobs.

    Respect an MPI runtime initialized by the caller, or initialize under a
    recognized MPI launcher. Explicit communicators never need auto-detection.
    """
    launch_variables = (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_RANK",
        "MPI_LOCALNRANKS",
        "MV2_COMM_WORLD_SIZE",
    )
    if "mpi4py.MPI" not in sys.modules and not any(
        name in os.environ for name in launch_variables
    ):
        return None
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
        comm = mpi_comm_world() if comm is None else comm
        if comm is None:
            return 0, 1
        return comm.Get_rank(), comm.Get_size()
    except Exception:
        return 0, 1


def distributed_rank_size(*, backend=None, comm=None):
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
    if comm is not None:
        return comm_rank_size(comm)
    backend = backend or get_backend()
    if backend.is_numpy:
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


def torch_group_root(comm=None):
    """Global rank corresponding to group-local rank zero."""
    import torch.distributed as dist

    return 0 if comm is None else dist.get_global_rank(comm, 0)
