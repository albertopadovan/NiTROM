from typing import Any

from .backend import get_backend


def compute_POD(
    pool,
    normalize: bool = False,
    broadcast: bool = True,
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
    bkend = get_backend()
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

        from .backend import mpi_bcast, mpi_gather

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
    dist.gather_object(X_local, gather_list, dst=0)
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
        dist.broadcast(U, src=0)
        dist.broadcast(S, src=0)

    return U, S, V


def interp_quadratic(t_eval: Any, t_data: Any, y_data: Any) -> Any:
    r"""
    Piecewise quadratic (3-point Lagrange) interpolation of uniformly
    sampled data.

    For each query point in *t_eval*, the three nearest data points are
    used to build a degree-2 Lagrange polynomial:

    .. math::

        p(t) = \sum_{j=0}^{2} y_j \prod_{\substack{m=0 \\ m \neq j}}^{2}
               \frac{t - t_m}{t_j - t_m}

    The data in *y_data* may have arbitrary leading dimensions (e.g.
    ``(n, n_data)`` or ``(B, n, n_data)``); interpolation is always
    performed along the **last** axis.

    :param t_eval: query times of shape ``(n_eval,)``
    :param t_data: data times of shape ``(n_data,)``, must be sorted
    :param y_data: data values with time along the last axis, shape ``(..., n_data)``
    :returns: interpolated values, shape ``(..., n_eval)``
    :rtype: backend array
    """
    bkend = get_backend()
    # Exact hit: querying at the data points themselves is the identity, so
    # skip the interpolation entirely.
    if t_eval.shape == t_data.shape and bool(bkend.array_equal(t_eval, t_data)):
        return y_data

    n_data = t_data.shape[0]

    # Find the index of the right neighbour for each query point.
    idx = bkend.clip(bkend.searchsorted(t_data, t_eval), 1, n_data - 1)

    # Centre index of the 3-point stencil, clamped so i-1, i, i+1 are valid.
    ic = bkend.clip(idx, 1, n_data - 2)

    t0 = t_data[ic - 1]  # (n_eval,)
    t1 = t_data[ic]
    t2 = t_data[ic + 1]

    # Lagrange basis values at t_eval
    L0 = ((t_eval - t1) * (t_eval - t2)) / ((t0 - t1) * (t0 - t2))
    L1 = ((t_eval - t0) * (t_eval - t2)) / ((t1 - t0) * (t1 - t2))
    L2 = ((t_eval - t0) * (t_eval - t1)) / ((t2 - t0) * (t2 - t1))

    # Gather data values at the stencil points: (..., n_eval)
    y0 = y_data[..., ic - 1]
    y1 = y_data[..., ic]
    y2 = y_data[..., ic + 1]

    return y0 * L0 + y1 * L1 + y2 * L2
