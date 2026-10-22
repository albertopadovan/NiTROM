"""Proper orthogonal decomposition of local or distributed trajectories."""

from typing import Any

from ..backend import get_backend
from ..distributed import torch_group_root


def compute_pod(
    pool, normalize: bool = False, broadcast: bool = True, *, backend=None
) -> tuple[Any | None, Any | None, Any | None]:
    r"""
    Compute the proper orthogonal decomposition (POD) of the training data.

    Each rank reshapes its local trajectory snapshots
    (``pool.X`` of shape ``(my_n_traj, N, n_snapshots)``) into a
    ``(N, my_n_traj * n_snapshots)`` matrix.  These are gathered onto the
    root rank (rank 0) and concatenated, in rank order, into the global
    snapshot matrix

    .. math::

        X \in \mathbb{R}^{N \times (n_\text{traj} \cdot n_\text{snaps})},

    on which the economy SVD :math:`X = U\,\mathrm{diag}(S)\,V^\top` is
    computed.  The columns of :math:`U` are the POD modes.

    :param pool: training-data pool holding the (possibly rank-distributed)
        trajectories in ``pool.X``
    :type pool: TrainingPool
    :param normalize: if ``True``, scale each trajectory ``k`` by
        :math:`1/\sqrt{w_k}` (``pool.weights[k]``) before assembling the
        snapshot matrix
    :type normalize: bool
    :param broadcast: if ``True`` (default), the modes ``U`` and singular
        values ``S`` are broadcast from root to **every** rank; the temporal
        coefficients ``V`` stay on root.  If ``False``, all factors live only
        on root.
    :type broadcast: bool
    :returns: the economy SVD ``(U, S, V)``.  In a distributed run, ``V`` is
        ``None`` off root; ``U`` and ``S`` are also ``None`` off root unless
        ``broadcast`` is ``True``.
    :rtype: tuple
    """
    bkend = backend or getattr(pool, "backend", None) or get_backend()
    X = pool.X
    if normalize:
        # Scale each trajectory by 1 / sqrt(weight).
        X = X / bkend.sqrt(pool.weights).reshape(-1, 1, 1)

    # Local snapshot matrix: (N, my_n_traj * n_snapshots).
    X_local = bkend.permute(X, (1, 0, 2)).reshape(pool.N, -1)

    # Distributed POD: torch uses torch.distributed (gloo); numpy uses MPI.
    distributed = False
    if pool.world_size > 1:
        if bkend.is_torch:
            import torch.distributed as dist

            distributed = dist.is_available() and dist.is_initialized()
        else:
            distributed = True

    if not distributed:
        U, S, Vh = bkend.svd(X_local, full_matrices=False)
        return U, S, bkend.mH(Vh)

    if bkend.is_numpy:
        import numpy as np

        from ..distributed import mpi_bcast, mpi_gather

        # Gather each rank's local snapshot matrix on root and SVD there, over
        # the same communicator the pool sharded the trajectories on.
        comm = pool.comm
        gathered = mpi_gather(np.ascontiguousarray(X_local), comm=comm)
        if pool.rank == 0:
            Xg = np.concatenate(gathered, axis=1)
            U, S, Vh = np.linalg.svd(Xg, full_matrices=False)
            V = Vh.conj().T
        else:
            U = S = V = None
        if broadcast:
            U = mpi_bcast(U, comm=comm)
            S = mpi_bcast(S, comm=comm)  # V (temporal coefficients) stays on root
        return U, S, V

    import torch
    import torch.distributed as dist

    # Gather each rank's local matrix onto root (sizes differ across ranks
    # when n_traj is not divisible by world_size) and compute the SVD there.
    gather_list = [None] * pool.world_size if pool.rank == 0 else None
    dist.gather_object(
        X_local, gather_list, dst=torch_group_root(pool.comm), group=pool.comm
    )
    if pool.rank == 0:
        X = torch.cat(
            [g.to(device=pool.device, dtype=pool.dtype) for g in gather_list],
            dim=1,
        )
        U, S, Vh = bkend.svd(X, full_matrices=False)
        V = bkend.mH(Vh)
    else:
        U, S, V = None, None, None

    if broadcast:
        # Distribute the modes and singular values to every rank.
        k = min(pool.N, pool.n_traj * pool.n_snapshots)
        if pool.rank == 0:
            U, S = U.contiguous(), S.contiguous()
        else:
            U = torch.empty((pool.N, k), device=pool.device, dtype=pool.dtype)
            S = torch.empty((k,), device=pool.device, dtype=pool.dtype)
        dist.broadcast(U, src=torch_group_root(pool.comm), group=pool.comm)
        dist.broadcast(S, src=torch_group_root(pool.comm), group=pool.comm)

    return U, S, V
