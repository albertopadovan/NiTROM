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


def _checkpoint_indices(n_snapshots: int, n_checkpoints: int, stride: int) -> list[int]:
    r"""
    Snapshot indices used as checkpoints, and implicitly the common window
    length.

    The dynamics are autonomous, so a state sampled at :math:`t_c` *is* a valid
    initial condition at :math:`t = 0`.  Checkpoints are therefore taken at
    ``0, q, 2q, ...`` and every resulting trajectory is the **same-length**
    window that follows its own checkpoint -- a sliding window of width
    ``L = n_snapshots - k_max``.  A common width is what makes every entry of
    :math:`G` an integral over the identical horizon, i.e. a single quadratic
    form; windows of differing length would truncate :math:`G_{ij}` at
    :math:`\min(L_i, L_j)` and break that.

    Checkpoints that would leave fewer than two samples are dropped, so
    ``n_checkpoints`` and ``stride`` trade directly against ``L``.

    :param n_snapshots: snapshots per trajectory
    :param n_checkpoints: requested number of checkpoints ``m`` (including the
        original initial condition at index 0)
    :param stride: spacing ``q`` between checkpoints
    :returns: the kept checkpoint indices, ascending
    """
    if n_checkpoints < 1 or stride < 1:
        raise ValueError(
            f"n_checkpoints and checkpoint_stride must be >= 1, got "
            f"{n_checkpoints} and {stride}."
        )
    idx = [j * stride for j in range(n_checkpoints) if j * stride <= n_snapshots - 2]
    if not idx:
        raise ValueError(
            f"No usable checkpoint: {n_snapshots} snapshots with stride "
            f"{stride} leaves no window of length >= 2."
        )
    return idx


def _assign_families(norms, amplitude_ranges):
    """
    Group initial conditions into amplitude families by binning their **raw**
    norms (before any normalization) into the caller's ranges.

    Lall-Marsden-Glavaski averaging requires that cross terms only ever pair
    trajectories of comparable amplitude, which is what the binning enforces.

    :param norms: raw norm of every candidate initial condition
    :param amplitude_ranges: list of ``(lo, hi)`` pairs, matched half-open
        ``[lo, hi)``; ``None`` puts everything in a single family
    :returns: ``(groups, dropped)`` -- a list of index lists, one per range,
        and the indices falling in no range
    """
    if amplitude_ranges is None:
        return [list(range(len(norms)))], []
    groups = [[] for _ in amplitude_ranges]
    dropped = []
    for i, nrm in enumerate(norms):
        for g, (lo, hi) in enumerate(amplitude_ranges):
            if lo <= nrm < hi:
                groups[g].append(i)
                break
        else:
            dropped.append(i)
    return groups, dropped


def _family_gramian_G(windows, dt):
    r"""
    :math:`G_{ij} = \sum_k \tilde{x}^{(i)}(t_k)^\top \tilde{x}^{(j)}(t_k)\,\Delta t`
    for one family.

    Every window has the same length, so the double sum over space and time is a
    plain inner product of the flattened windows,
    :math:`G = \Delta t\, W_f W_f^\top` with :math:`W_f` the ``(m, N L)``
    matrix whose rows are the flattened windows.  Since ``windows`` is one
    contiguous ``(m, N, L)`` block, that reshape is a view and the whole
    Gramian is a single GEMM: one streaming pass over the family's data, no
    temporaries.  (The earlier version accumulated the same quantity over
    column chunks, rebuilding each chunk with a ``concatenate`` -- that copied
    the family's data once per chunk and dominated the runtime.)

    The ``n x n`` Gramian this factors is still never formed.

    :param windows: contiguous ``(m, N, L)`` array of the family's windows
    :param dt: sampling interval
    :returns: ``G`` of shape ``(m, m)``
    """
    Wf = windows.reshape(windows.shape[0], -1)
    return dt * (Wf @ Wf.T)


def _stage_family(views, scales, buf):
    """Copy one family's scaled windows into ``buf`` and return the used slice.

    The windows are scaled slices of the training data, so they need not be
    materialized for the whole IC set at once: only the Gramian needs them
    contiguous (to reshape into a single GEMM), and it needs one family at a
    time.  Staging through a buffer sized by the largest family therefore cuts
    peak memory by roughly the number of families.  Everything else downstream
    -- ``Q^T w`` and the ``Phi`` accumulation -- works directly on the strided
    views with the scale applied to the small result instead.
    """
    for i, (v, sc) in enumerate(zip(views, scales)):
        buf[i] = sc * v
    return buf[:len(views)]


def _restricted_observability_factor(G, X0, rank_M, rcond):
    r"""
    Factor :math:`Q` with
    :math:`W_{o,X_0} = X_0 M^\dagger G M^\dagger X_0^\top = QQ^\top`,
    where :math:`M = X_0^\top X_0`.

    From the generalized eigenproblem :math:`GU = MU\Lambda` with
    :math:`U^\top MU = I`, one gets :math:`Q = X_0 U \Lambda^{1/2}`.
    :math:`M^\dagger` keeps the eigenvalues above ``rcond`` times the largest,
    and at most ``rank_M`` of them.  Truncation is not optional in practice:
    checkpoints taken close together are nearly collinear, so ``M``'s spectrum
    decays smoothly, and keeping the tail injects grid-scale noise into
    :math:`\Psi`.

    :returns: ``(Q, info)`` with ``Q`` of shape ``(N, k)``
    """
    bkend = get_backend()
    M = X0.T @ X0
    s, V = bkend.eigh(M)
    order = list(range(s.shape[0] - 1, -1, -1))  # descending, portable
    s, V = s[order], V[:, order]
    keep = int((s > rcond * s[0]).sum())
    if rank_M is not None:
        keep = min(keep, rank_M)
    if keep < 1:
        raise ValueError(
            f"M has no eigenvalue above rcond ({rcond:g}) times the largest."
        )
    T = V[:, :keep] / bkend.sqrt(s[:keep])
    lam, W = bkend.eigh(T.T @ G @ T)
    order = list(range(lam.shape[0] - 1, -1, -1))
    lam, W = bkend.clip(lam[order], 0.0, None), W[:, order]
    Q = X0 @ (T @ W * bkend.sqrt(lam))
    return Q, dict(sM=s, rank_M_kept=keep, Lambda=lam, n_ic=X0.shape[1])


def _balance_from_windows(fam_views, fam_scales, fam_X0, weights, dt,
                          rank_M, rcond):
    r"""
    Lall-Marsden-Glavaski averaged balancing, in factored form.

    With per-family weights :math:`w_c`,
    :math:`W_c = \sum_c w_c X^{(c)}X^{(c)\top} = XX^\top` and
    :math:`W_o = \sum_c w_c Q^{(c)}Q^{(c)\top} = QQ^\top`, so the factors are
    the weighted concatenations.  The balancing transformation follows from the
    SVD :math:`Q^\top X = L\Sigma R^\top`:
    :math:`\Phi = XR\Sigma^{-1/2}`, :math:`\Psi = QL\Sigma^{-1/2}`, with
    :math:`\Psi^\top\Phi = I`.

    Neither :math:`W_c` nor :math:`W_o` is ever formed.
    """
    bkend = get_backend()

    # One staging buffer, sized by the largest family, reused for each Gramian.
    v0 = fam_views[0][0]
    mmax = max(len(v) for v in fam_views)
    buf = bkend.zeros((mmax,) + tuple(v0.shape), device=bkend.device_of(v0),
                      dtype=v0.dtype)

    Qs, fam_info = [], []
    for views, scales, X0, w in zip(fam_views, fam_scales, fam_X0, weights):
        G = _family_gramian_G(_stage_family(views, scales, buf), dt)
        Qc, info = _restricted_observability_factor(G, X0, rank_M, rcond)
        Qs.append(bkend.sqrt(bkend.asarray(w, dtype=Qc.dtype,
                                           device=bkend.device_of(Qc))) * Qc)
        fam_info.append(info)
    Q = bkend.concatenate(Qs, axis=1)

    # H = Q^T X, one block per trajectory.  X is never assembled: each block is
    # a scaled slice of Q^T applied to the window it came from.
    blocks, coeffs = [], []
    for views, scs, w in zip(fam_views, fam_scales, weights):
        for view, sc in zip(views, scs):
            f = sc * float(w * dt) ** 0.5
            blocks.append(f * (Q.T @ view))     # strided view: no copy needed
            coeffs.append(f)
    H = bkend.concatenate(blocks, axis=1)

    Lsv, Sig, Rt = bkend.svd(H, full_matrices=False)
    rmax = int((Sig > 1e-12 * Sig[0]).sum())
    Lsv, Sig, Rt = Lsv[:, :rmax], Sig[:rmax], Rt[:rmax]

    # Phi = X R Sigma^{-1/2}, accumulated window by window.
    Phi, col = None, 0
    flat_views = [v for views in fam_views for v in views]
    for win, f in zip(flat_views, coeffs):
        ncols = win.shape[1]
        contrib = win @ Rt[:, col:col + ncols].T
        Phi = f * contrib if Phi is None else Phi + f * contrib
        col += ncols
    Phi = Phi / bkend.sqrt(Sig)
    Psi = Q @ (Lsv / bkend.sqrt(Sig))
    return Phi, Psi, Sig, dict(Q=Q, L=Lsv, H=H, families=fam_info)


def compute_data_driven_balancing(
    pool,
    rank_M: int | None = None,
    n_checkpoints: int = 1,
    checkpoint_stride: int = 1,
    amplitude_ranges: list | None = None,
    families: list | None = None,
    normalize: bool = True,
    family_weights: Any = None,
    rcond: float = 1e-4,
    broadcast: bool = True,
):
    r"""
    Data-driven balanced truncation from trajectory data alone (balanced
    operator inference).

    Builds a biorthogonal pair :math:`(\Phi, \Psi)`, :math:`\Psi^\top\Phi = I`,
    that balances the *empirical* Gramians of the training trajectories.  The
    full-order operator is never used, and -- unlike textbook balancing --
    neither :math:`n \times n` Gramian is ever formed: only their factors.

    **Controllability.** :math:`W_c = XX^\top` with
    :math:`X = \sqrt{\Delta t}\,[\text{all windows}]`.

    **Observability.** Adjoint trajectories are unavailable, so :math:`W_o` is
    estimated on :math:`\mathrm{span}(X_0)`:

    .. math::

        W_{o,X_0} = X_0 M^\dagger G M^\dagger X_0^\top = QQ^\top,
        \qquad M = X_0^\top X_0,
        \qquad G_{ij} = \sum_k x^{(i)\top}(t_k)x^{(j)}(t_k)\,\Delta t .

    :math:`M^\dagger` is truncated at ``rank_M`` -- see
    :func:`_restricted_observability_factor` for why that matters.

    **Checkpointing.** The dynamics are autonomous, so any sampled state is a
    legitimate initial condition at :math:`t = 0`.  Checkpoints at snapshots
    ``0, q, 2q, ...`` therefore enlarge :math:`\mathrm{span}(X_0)` at no
    simulation cost.  Each one contributes the equal-length window that follows
    it, so every :math:`G_{ij}` covers the same horizon; ``n_checkpoints`` and
    ``checkpoint_stride`` trade against the window length
    ``L = n_snapshots - (m-1)q``.

    **Amplitude families.** For a nonlinear system the observability energy
    depends on the amplitude of an initial condition and not only its
    direction, so a single :math:`G` mixing amplitudes is inconsistent.
    Following Lall, Marsden and Glavaski, initial conditions are binned by
    their raw norm into ``amplitude_ranges``, :math:`G` is formed within each
    family, and the resulting positive semidefinite Gramians are averaged.
    With one family this reduces to the linear case.

    :param pool: training-data pool holding the (possibly rank-distributed)
        trajectories in ``pool.X`` of shape ``(my_n_traj, N, n_snapshots)``
    :type pool: TrainingPool
    :param rank_M: hard cap on the rank of :math:`M^\dagger`.  ``None`` (the
        default) lets ``rcond`` decide, which is usually what you want: the
        spectrum of :math:`M` often has no clean gap, so a relative threshold
        is more robust than a fixed rank.  Inspect
        ``info["families"][c]["sM"]`` to check where the cut landed.
    :type rank_M: int or None
    :param n_checkpoints: number of checkpoints ``m`` per trajectory, counting
        the original initial condition
    :type n_checkpoints: int
    :param checkpoint_stride: snapshots ``q`` skipped between checkpoints
    :type checkpoint_stride: int
    :param amplitude_ranges: ``[(lo, hi), ...]`` bins on the raw initial-condition
        norm, matched half-open; ``None`` uses a single family
    :type amplitude_ranges: list or None
    :param families: explicit families given as lists of **trajectory** indices,
        e.g. ``[[0, ..., 9], [10, ..., 19]]``.  Every checkpoint inherits its
        parent trajectory's family, so a decayed checkpoint stays with the
        amplitude it was launched at, rather than migrating to a lower bin as
        it would under ``amplitude_ranges``.  Takes precedence over
        ``amplitude_ranges`` when both are given.
    :type families: list or None
    :param normalize: if ``True`` (default), divide every trajectory by the norm
        of its own initial condition, so all columns of :math:`X_0` are unit
        norm.  This is Lall's :math:`1/c^2` Gramian scaling.
    :type normalize: bool
    :param family_weights: per-family weights :math:`w_c`; defaults to
        :math:`1/s`
    :param rcond: relative floor on the retained eigenvalues of :math:`M`;
        eigenvalues below ``rcond * max`` are discarded.  The default
        ``1e-4`` is the working choice: below roughly this level the
        checkpoints are too nearly collinear to carry information, and
        retaining them puts grid-scale noise into :math:`\Psi`.
    :type rcond: float
    :param broadcast: if ``True`` (default), ``Phi``, ``Psi`` and ``Sigma`` are
        broadcast to every rank; otherwise they live only on root
    :type broadcast: bool
    :returns: ``(Phi, Psi, Sigma, info)``.  ``Phi`` and ``Psi`` are ``(N, R)``
        with :math:`\Psi^\top\Phi = I`, ``Sigma`` holds the Hankel singular
        values, and ``info`` carries ``Q``, ``L``, ``H``, the per-family
        diagnostics, the checkpoint indices and the window length.  Off root,
        entries are ``None`` unless ``broadcast``.
    :rtype: tuple
    """
    bkend = get_backend()

    # --- gather the trajectories on root, as compute_POD does ---------------
    distributed = False
    if pool.world_size > 1:
        if bkend.is_torch:
            import torch.distributed as dist
            distributed = dist.is_available() and dist.is_initialized()
        else:
            distributed = True

    if not distributed:
        X_all = pool.X
        on_root = True
    elif bkend.is_numpy:
        import numpy as np

        from .backend import mpi_gather
        gathered = mpi_gather(np.ascontiguousarray(pool.X), comm=pool.comm)
        on_root = pool.rank == 0
        X_all = np.concatenate(gathered, axis=0) if on_root else None
    else:
        import torch
        import torch.distributed as dist
        gather_list = [None] * pool.world_size if pool.rank == 0 else None
        dist.gather_object(pool.X, gather_list, dst=0)
        on_root = pool.rank == 0
        X_all = (
            torch.cat([g.to(device=pool.device, dtype=pool.dtype)
                       for g in gather_list], dim=0)
            if on_root else None
        )

    Phi = Psi = Sig = info = None
    if on_root:
        n_traj, _, nt = X_all.shape
        dt = float(pool.time[1] - pool.time[0])
        ks = _checkpoint_indices(nt, n_checkpoints, checkpoint_stride)
        L = nt - ks[-1]

        # Candidate initial conditions: every (trajectory, checkpoint) pair.
        # A window is a scaled slice of the training data, so it is kept as a
        # (strided) VIEW plus its scale rather than copied out.  Materializing
        # all of them would cost n_ic * N * L * 8 bytes -- 11 GB for 210 windows
        # of a 19800-state cavity -- where the data itself is under half a GB.
        ics, norms, views, scales = [], [], [], []
        for j in range(n_traj):
            for k in ks:
                col = X_all[j, :, k]
                nrm = float(bkend.vector_norm(col))
                scale = 1.0 / nrm if (normalize and nrm > 0.0) else 1.0
                ics.append(scale * col)
                norms.append(nrm)
                views.append(X_all[j, :, k:k + L])
                scales.append(scale)

        if families is not None:
            # Explicit parent grouping: every checkpoint of trajectory j
            # inherits j's family.  `ics` was built trajectory-major.
            n_ck = len(ks)
            groups = [
                [j * n_ck + c for j in grp for c in range(n_ck)]
                for grp in families
            ]
            flat = [i for g in groups for i in g]
            if len(set(flat)) != len(flat):
                raise ValueError("families must not repeat a trajectory index.")
            dropped = [i for i in range(len(ics)) if i not in set(flat)]
        else:
            groups, dropped = _assign_families(norms, amplitude_ranges)
        groups = [g for g in groups if g]
        if not groups:
            raise ValueError(
                "No initial condition fell inside amplitude_ranges "
                f"{amplitude_ranges}; raw norms span "
                f"[{min(norms):.3e}, {max(norms):.3e}]."
            )
        s = len(groups)
        w = ([1.0 / s] * s if family_weights is None
             else [float(x) for x in family_weights])
        if len(w) != s:
            raise ValueError(
                f"family_weights has {len(w)} entries but {s} non-empty "
                f"families were formed."
            )

        fam_views = [[views[i] for i in g] for g in groups]
        fam_scales = [[scales[i] for i in g] for g in groups]
        fam_X0 = [
            bkend.concatenate([ics[i].reshape(-1, 1) for i in g], axis=1)
            for g in groups
        ]
        Phi, Psi, Sig, info = _balance_from_windows(
            fam_views, fam_scales, fam_X0, w, dt, rank_M, rcond
        )
        info.update(
            checkpoints=ks, window_length=L, dt=dt, weights=w,
            n_dropped=len(dropped),
            family_amplitudes=[[norms[i] for i in g] for g in groups],
        )

    if not distributed or not broadcast:
        return Phi, Psi, Sig, info

    if bkend.is_numpy:
        from .backend import mpi_bcast
        return (mpi_bcast(Phi, comm=pool.comm), mpi_bcast(Psi, comm=pool.comm),
                mpi_bcast(Sig, comm=pool.comm), info)

    import torch
    import torch.distributed as dist
    meta = [None if not on_root else int(Sig.shape[0])]
    dist.broadcast_object_list(meta, src=0)
    R = meta[0]
    if not on_root:
        Phi = torch.empty((pool.N, R), device=pool.device, dtype=pool.dtype)
        Psi = torch.empty((pool.N, R), device=pool.device, dtype=pool.dtype)
        Sig = torch.empty((R,), device=pool.device, dtype=pool.dtype)
    else:
        Phi, Psi, Sig = Phi.contiguous(), Psi.contiguous(), Sig.contiguous()
    dist.broadcast(Phi, src=0)
    dist.broadcast(Psi, src=0)
    dist.broadcast(Sig, src=0)
    return Phi, Psi, Sig, info


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
