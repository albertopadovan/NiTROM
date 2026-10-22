"""Data-driven balancing using factored empirical Gramians."""

from typing import Any

from ..backend import get_backend
from ..distributed import torch_group_root
from ..numerics import lagrange_stencils


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


def _window_maps(ks, L, stencils=None, sqrt_w=None):
    r"""
    Every window as a sparse linear map on its trajectory's samples.

    Window column ``l`` of the window starting at checkpoint ``k`` is
    ``sum_m coef[l, m] * X_j[:, idx[l, m]]``: on uniform data a single sample,
    ``idx = k + l`` with ``coef = 1``; with a quadrature, the local Lagrange
    stencil of the node ``t_k + tau_l`` times :math:`\sqrt{\omega_l}`.
    Windows are never materialized -- everything downstream works on
    trajectory-level products (:math:`X_j^\top X_{j'}`, :math:`Q^\top X_j`,
    :math:`X_j C_j`) and these maps.

    :returns: ``{k: (idx, coef)}`` with ``idx`` (int) and ``coef`` of shape
        ``(L, p)``
    """
    import numpy as np

    maps = {}
    for k in ks:
        if stencils is None:
            maps[k] = (k + np.arange(L)[:, None], np.ones((L, 1)))
        else:
            st, W = stencils[k]
            maps[k] = (
                st[:, None] + np.arange(W.shape[1])[None, :],
                W * sqrt_w[:, None],
            )
    return maps


def _apply_map(A, idx, coef):
    """``A`` (rows, n_t) times a window map: ``(rows, L)``,
    ``out[:, l] = sum_m coef[l, m] A[:, idx[l, m]]``."""
    return (A[:, idx] * coef[None]).sum(-1)


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


def _family_gramian_G(fam, maps, X_all, dt, C=None, *, backend=None):
    r"""
    :math:`G_{ii'} = \sum_l \Delta t\, \tilde{x}^{(i)}(\tau_l)^\top
    \tilde{x}^{(i')}(\tau_l)` for one family, without forming any window.

    Window :math:`i` is :math:`s_i X_j A_i` -- its trajectory's samples
    :math:`X_j` (``N x n_t``) times its sparse map :math:`A_i` (``n_t x L``,
    :func:`_window_maps`) times its normalization :math:`s_i` -- so

    .. math::

        G_{ii'} = \Delta t\, s_i s_{i'} \sum_l A_i[:, l]^\top
                  S_{jj'}\, A_{i'}[:, l], \qquad S_{jj'} = X_j^\top X_{j'} .

    The only full-dimension work is one snapshot Gram matrix :math:`S_{jj'}`
    (``n_t x n_t``) per pair of trajectories in the family: each snapshot is
    read once per partner trajectory, however many windows contain it, and
    nothing of size ``N`` is copied.  The rest is gathers on ``n_t``-sized
    arrays.  (The previous version staged the ``(m, N, L)`` block of
    overlapping windows and took one GEMM, which duplicated every snapshot in
    up to ``L/q`` windows.)

    :param fam: the family's initial conditions as ``(j, k, s)``: trajectory,
        checkpoint, normalization
    :param maps: :func:`_window_maps` output
    :param X_all: ``(n_traj, N, n_t)`` trajectories (backend array)
    :param dt: rectangle-rule weight (``1`` when the quadrature weights are
        folded into the maps)
    :param C: output operator ``(p, N)``; ``G`` is then built from the outputs
        :math:`y = C\tilde x` (:math:`S_{jj'} = (CX_j)^\top CX_{j'}`).  ``None``
        observes the full state (:math:`C = I`).
    :returns: ``G`` of shape ``(m, m)``, float64 numpy
    """
    import numpy as np

    bkend = backend or get_backend()
    trajs = sorted({j for j, _, _ in fam})
    Yt = {a: (X_all[a] if C is None else C @ X_all[a]) for a in trajs}
    S = {}
    for ia, a in enumerate(trajs):
        for b in trajs[ia:]:
            Sab = np.asarray(bkend.to_numpy(Yt[a].T @ Yt[b]), dtype=float)
            S[(a, b)], S[(b, a)] = Sab, Sab.T
    m = len(fam)
    G = np.empty((m, m))
    rows = {a: [i for i, (j, _, _) in enumerate(fam) if j == a] for a in trajs}
    L = next(iter(maps.values()))[0].shape[0]
    lcol = np.arange(L)[None, :, None]
    for i2, (j2, k2, s2) in enumerate(fam):
        idx2, c2 = maps[k2]
        for a in trajs:
            T = _apply_map(S[(a, j2)], idx2, c2)  # (n_t, L)
            ri = rows[a]
            idx1 = np.stack([maps[fam[i][1]][0] for i in ri])  # (n_r, L, p)
            c1 = np.stack([maps[fam[i][1]][1] for i in ri])
            s1 = np.array([fam[i][2] for i in ri])
            G[ri, i2] = dt * s1 * s2 * (T[idx1, lcol] * c1).sum(axis=(1, 2))
    return G


def _restricted_observability_factor(G, X0, rank_M, rcond, *, backend=None):
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
    bkend = backend or get_backend()
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


def _balance_from_windows(
    fams,
    maps,
    X_all,
    fam_X0,
    weights,
    dt,
    rank_M,
    rcond,
    imps,
    imap,
    dt_R,
    C=None,
    *,
    backend=None,
):
    r"""
    Lall-Marsden-Glavaski averaged balancing, in factored form.

    **Reachability** -- the Gramian of the actuators, from the impulse responses
    only (no checkpoints): with :math:`x^{(j)}(0) = \beta_j b_j`,

    .. math::

        W_R = \sum_a w_a \sum_{j \in F_a} \sum_{k} \omega_k\,
              \frac{x^{(j)}(t_k)\,x^{(j)}(t_k)^\top}{\lVert x^{(j)}(t_0)\rVert^2}
            = XX^\top ,

    over the whole trajectory (``imps``, ``imap``; :math:`\omega_k = \Delta t`
    on a uniform grid).

    **Observability** -- per family, from the checkpoint windows,
    :math:`\widetilde W_O = \sum_a w_a X_0^{(a)} M_a^{+} G^{(a)} M_a^{+}
    X_0^{(a)\top} = QQ^\top`.

    The balancing transformation follows from the SVD
    :math:`H = Q^\top X = L\Sigma R^\top`: :math:`\Phi = XR\Sigma^{-1/2}`,
    :math:`\Psi = QL\Sigma^{-1/2}`, with :math:`\Psi^\top\Phi = I`.

    Neither Gramian, nor :math:`X`, nor any window is ever formed.  Impulse
    response :math:`j` enters :math:`X` as :math:`f_j X_j A` (its samples times
    the time-weight map, :math:`f_j = s_j\sqrt{w_a\,\Delta t_R}`), so

    * :math:`H = Q^\top X` takes one product :math:`P_j = Q^\top X_j` per
      trajectory;
    * :math:`\Phi = \sum_j X_j C_j \Sigma^{-1/2}` with
      :math:`C_j = f_j A R_j^\top` (``n_t x R``): one product per trajectory.

    :param fams: per family, its checkpoint initial conditions as ``(j, k, s)``
        (observability)
    :param maps: :func:`_window_maps` output for the checkpoint windows
    :param X_all: ``(n_traj, N, n_t)`` trajectories (backend array)
    :param fam_X0: per family, the ``(N, m_c)`` normalized initial conditions
    :param dt: time weight of the windows (``1`` when folded into ``maps``)
    :param imps: per family, its impulse responses as ``(j, s)`` (reachability)
    :param imap: ``(idx, coef)`` map of a whole trajectory (time weights)
    :param dt_R: time weight of the impulse responses (``1`` when folded into
        ``imap``)
    :param C: output operator for :math:`G` (``None``: the full state)
    """
    import numpy as np

    bkend = backend or get_backend()
    dtype, device = X_all.dtype, bkend.device_of(X_all)

    Qs, fam_info = [], []
    for fam, X0, w in zip(fams, fam_X0, weights, strict=False):
        G = bkend.asarray(
            _family_gramian_G(fam, maps, X_all, dt, C, backend=bkend),
            dtype=dtype,
            device=device,
        )
        Qc, info = _restricted_observability_factor(G, X0, rank_M, rcond, backend=bkend)
        Qs.append(
            bkend.sqrt(bkend.asarray(w, dtype=Qc.dtype, device=bkend.device_of(Qc)))
            * Qc
        )
        fam_info.append(info)
    Q = bkend.concatenate(Qs, axis=1)

    # H = Q^T X over the impulse responses, from P_j = Q^T X_j.
    idx, c = imap
    used = sorted({j for imp in imps for j, _ in imp})
    P = {j: np.asarray(bkend.to_numpy(Q.T @ X_all[j]), dtype=float) for j in used}
    cols, blocks = [], []
    for imp, w in zip(imps, weights, strict=False):
        for j, sc in imp:
            f = sc * float(w * dt_R) ** 0.5
            blocks.append(f * _apply_map(P[j], idx, c))
            cols.append((j, f))
    H = bkend.asarray(np.concatenate(blocks, axis=1), dtype=dtype, device=device)

    Lsv, Sig, Rt = bkend.svd(H, full_matrices=False)
    rmax = int((Sig > 1e-12 * Sig[0]).sum())
    Lsv, Sig, Rt = Lsv[:, :rmax], Sig[:rmax], Rt[:rmax]

    # Phi = X R Sigma^{-1/2} = sum_j X_j C_j Sigma^{-1/2}.
    Rt_h = np.asarray(bkend.to_numpy(Rt), dtype=float)
    n_t = X_all.shape[2]
    L = idx.shape[0]
    Phi, col = None, 0
    for j, f in cols:
        Cj = np.zeros((n_t, rmax))
        contrib = f * c[..., None] * Rt_h[:, col : col + L].T[:, None, :]  # (L, p, R)
        np.add.at(Cj, idx.ravel(), contrib.reshape(-1, rmax))
        term = X_all[j] @ bkend.asarray(Cj, dtype=dtype, device=device)
        Phi = term if Phi is None else Phi + term
        col += L
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
    quadrature: dict | None = None,
    output_operator: Any = None,
    *,
    backend=None,
):
    r"""
    Data-driven balanced truncation from trajectory data alone (balanced
    operator inference).

    Builds a biorthogonal pair :math:`(\Phi, \Psi)`, :math:`\Psi^\top\Phi = I`,
    that balances the *empirical* Gramians of the training trajectories.  The
    full-order operator is never used, and -- unlike textbook balancing --
    neither :math:`n \times n` Gramian is ever formed: only their factors.

    **Reachability.** :math:`W_R = XX^\top` is the empirical Gramian of the
    actuators: the impulse responses themselves, over the whole sampled
    horizon, each normalized by its initial amplitude and weighted by its
    family (Lall's :math:`(1/s)\sum_k \alpha_k^{-2}\sum_j \int x x^\top dt`),
    :math:`X = [\sqrt{w_a\omega_k}\, x^{(j)}(t_k)/\lVert x^{(j)}(t_0)\rVert]`.
    Checkpoints are **not** used here -- they only enrich the observability
    estimate below.  (Checkpoint windows in :math:`W_R` would give the Gramian
    of the unit-norm checkpoint directions instead of the actuators', and
    count overlapping time intervals several times.)

    **Observability.** Adjoint trajectories are unavailable, so :math:`W_o` is
    estimated on :math:`\mathrm{span}(X_0)`:

    .. math::

        W_{o,X_0} = X_0 M^\dagger G M^\dagger X_0^\top = QQ^\top,
        \qquad M = X_0^\top X_0,
        \qquad G_{ij} = \sum_k y^{(i)\top}(t_k)y^{(j)}(t_k)\,\Delta t ,

    with :math:`y = Cx` the outputs (``output_operator``; the full state,
    :math:`C = I`, by default).

    :math:`M^\dagger` is truncated at ``rank_M`` -- see
    :func:`_restricted_observability_factor` for why that matters.

    **Checkpointing** (observability only). The dynamics are autonomous, so any
    sampled state is a legitimate initial condition at :math:`t = 0`.
    Checkpoints at snapshots ``0, q, 2q, ...`` therefore enlarge
    :math:`\mathrm{span}(X_0)` at no simulation cost.  Each one contributes the equal-length window that follows
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
    :param normalize: if ``True`` (default), divide every checkpoint window by
        the norm of its own initial state, so all columns of :math:`X_0` are
        unit norm, and every impulse response by the norm of its initial
        condition.  This is Lall's :math:`1/c^2` Gramian scaling.
    :type normalize: bool
    :param family_weights: per-family weights :math:`w_c`; defaults to
        :math:`1/s`
    :param rcond: relative floor on the retained eigenvalues of :math:`M`;
        eigenvalues below ``rcond * max`` are discarded.  The default
        ``1e-4`` is the working choice: below roughly this level the
        checkpoints are too nearly collinear to carry information, and
        retaining them puts grid-scale noise into :math:`\Psi`.
    :type rcond: float
    :param quadrature: time quadrature for non-uniformly sampled data.
        ``None`` (default) assumes a uniform ``pool.time`` and uses windows of
        consecutive snapshots with the rectangle rule ``dt``.  Otherwise a dict

        * ``nodes``, ``weights`` -- a rule on the window horizon
          :math:`\tau \in [0, T_w]` (e.g. :func:`composite_gauss_legendre`,
          fine where the dynamics are fast);
        * ``checkpoint_times`` -- checkpoint times, each a sample time of
          ``pool.time`` (replaces ``n_checkpoints``/``checkpoint_stride``);
        * ``interp_order`` (default 6) -- points of the local Lagrange
          interpolation.

        Every window is then evaluated at the same :math:`\tau` nodes,
        :math:`x(t_c + \tau_q)`, by local Lagrange interpolation of the
        samples (which may be unevenly spaced), and scaled by
        :math:`\sqrt{\omega_q}`.  All windows sharing one :math:`\tau`-grid
        is what keeps :math:`G` a Gramian; the weights replace ``dt`` in
        :math:`G`.  The impulse responses of :math:`W_R` use every sample of
        ``pool.time`` with the weights :math:`\omega_k = (t_{k+1} -
        t_{k-1})/2` (one-sided at the ends), which are ``dt`` on a uniform grid.  With nodes at the sample offsets and ``weights = dt`` on
        a uniform grid this reproduces the default path to round-off.
        Windows are rebuilt when needed rather than stored.
    :type quadrature: dict or None
    :param output_operator: matrix :math:`C` of shape ``(p, N)`` defining the
        outputs :math:`y = Cx` whose energy :math:`G` measures; ``None``
        (default) observes the full state.  Only :math:`G` (hence
        :math:`\widetilde W_O`) depends on it.
    :type output_operator: array or None
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
    bkend = backend or getattr(pool, "backend", None) or get_backend()

    # --- gather the trajectories on root, as compute_pod does ---------------
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

        from ..distributed import mpi_gather

        gathered = mpi_gather(np.ascontiguousarray(pool.X), comm=pool.comm)
        on_root = pool.rank == 0
        X_all = np.concatenate(gathered, axis=0) if on_root else None
    else:
        import torch
        import torch.distributed as dist

        gather_list = [None] * pool.world_size if pool.rank == 0 else None
        dist.gather_object(
            pool.X, gather_list, dst=torch_group_root(pool.comm), group=pool.comm
        )
        on_root = pool.rank == 0
        X_all = (
            torch.cat(
                [g.to(device=pool.device, dtype=pool.dtype) for g in gather_list], dim=0
            )
            if on_root
            else None
        )

    Phi = Psi = Sig = info = None
    if on_root:
        n_traj, _, nt = X_all.shape
        if quadrature is None:
            dt = float(pool.time[1] - pool.time[0])
            ks = _checkpoint_indices(nt, n_checkpoints, checkpoint_stride)
            L = nt - ks[-1]
        else:
            import numpy as np

            t_h = np.asarray(bkend.to_numpy(pool.time), dtype=float)
            tau = np.asarray(quadrature["nodes"], dtype=float)
            omega = np.asarray(quadrature["weights"], dtype=float)
            if tau.shape != omega.shape or np.any(omega <= 0):
                raise ValueError(
                    "quadrature nodes/weights must match and weights be positive."
                )
            tol = 1e-9 * max(t_h[-1] - t_h[0], 1.0)
            ks = []
            for tc in quadrature["checkpoint_times"]:
                i = int(np.argmin(np.abs(t_h - tc)))
                if abs(t_h[i] - tc) > tol:
                    raise ValueError(
                        f"checkpoint time {tc:g} is not a sample time of pool.time."
                    )
                ks.append(i)
            order = int(quadrature.get("interp_order", 6))
            stencils = {k: lagrange_stencils(t_h, t_h[k] + tau, order) for k in ks}
            sqrt_w = np.sqrt(omega)
            dt = 1.0  # the weights are folded into the windows
            L = len(tau)

        # Candidate initial conditions: every (trajectory, checkpoint) pair,
        # kept as (trajectory, checkpoint, normalization).  Its window is never
        # materialized: it is the trajectory's samples times a sparse map
        # (_window_maps), and every product the balancing needs is taken at
        # trajectory level.
        maps = _window_maps(
            ks,
            L,
            None if quadrature is None else stencils,
            None if quadrature is None else sqrt_w,
        )
        ics, norms, ic_keys = [], [], []
        for j in range(n_traj):
            for k in ks:
                col = X_all[j, :, k]
                nrm = float(bkend.vector_norm(col))
                scale = 1.0 / nrm if (normalize and nrm > 0.0) else 1.0
                ics.append(scale * col)
                norms.append(nrm)
                ic_keys.append((j, k, scale))

        if families is not None:
            # Explicit parent grouping: every checkpoint of trajectory j
            # inherits j's family.  `ics` was built trajectory-major.
            n_ck = len(ks)
            groups = [
                [j * n_ck + c for j in grp for c in range(n_ck)] for grp in families
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
        w = (
            [1.0 / s] * s
            if family_weights is None
            else [float(x) for x in family_weights]
        )
        if len(w) != s:
            raise ValueError(
                f"family_weights has {len(w)} entries but {s} non-empty "
                f"families were formed."
            )

        # Reachability: the impulse responses, over every sample, each in the
        # family of its earliest checkpoint (that of t = 0 when it is one) and
        # normalized by its initial amplitude.
        import numpy as np

        if quadrature is None:
            imap, dt_R = (np.arange(nt)[:, None], np.ones((nt, 1))), dt
        else:
            om = np.empty(nt)
            om[1:-1] = 0.5 * (t_h[2:] - t_h[:-2])
            om[0], om[-1] = t_h[1] - t_h[0], t_h[-1] - t_h[-2]
            imap, dt_R = (np.arange(nt)[:, None], np.sqrt(om)[:, None]), 1.0
        fam_of = {i: a for a, g in enumerate(groups) for i in g}
        imps = [[] for _ in groups]
        for j in range(n_traj):
            i0 = j * len(ks)  # (j, ks[0]): earliest checkpoint
            if i0 in fam_of:
                nrm0 = float(bkend.vector_norm(X_all[j, :, 0]))
                imps[fam_of[i0]].append(
                    (j, 1.0 / nrm0 if (normalize and nrm0 > 0.0) else 1.0)
                )

        fams = [[ic_keys[i] for i in g] for g in groups]
        fam_X0 = [
            bkend.concatenate([ics[i].reshape(-1, 1) for i in g], axis=1)
            for g in groups
        ]
        Cmat = (
            None
            if output_operator is None
            else bkend.asarray(
                output_operator, dtype=X_all.dtype, device=bkend.device_of(X_all)
            )
        )
        Phi, Psi, Sig, info = _balance_from_windows(
            fams,
            maps,
            X_all,
            fam_X0,
            w,
            dt,
            rank_M,
            rcond,
            imps,
            imap,
            dt_R,
            Cmat,
            backend=bkend,
        )
        info.update(
            checkpoints=ks,
            window_length=L,
            dt=dt if quadrature is None else None,
            quadrature=None
            if quadrature is None
            else dict(nodes=tau, weights=omega, interp_order=order),
            weights=w,
            n_dropped=len(dropped),
            family_amplitudes=[[norms[i] for i in g] for g in groups],
        )

    if not distributed or not broadcast:
        return Phi, Psi, Sig, info

    if bkend.is_numpy:
        from ..distributed import mpi_bcast

        return (
            mpi_bcast(Phi, comm=pool.comm),
            mpi_bcast(Psi, comm=pool.comm),
            mpi_bcast(Sig, comm=pool.comm),
            info,
        )

    import torch
    import torch.distributed as dist

    meta = [None if not on_root else int(Sig.shape[0])]
    dist.broadcast_object_list(meta, src=torch_group_root(pool.comm), group=pool.comm)
    R = meta[0]
    if not on_root:
        Phi = torch.empty((pool.N, R), device=pool.device, dtype=pool.dtype)
        Psi = torch.empty((pool.N, R), device=pool.device, dtype=pool.dtype)
        Sig = torch.empty((R,), device=pool.device, dtype=pool.dtype)
    else:
        Phi, Psi, Sig = Phi.contiguous(), Psi.contiguous(), Sig.contiguous()
    dist.broadcast(Phi, src=torch_group_root(pool.comm), group=pool.comm)
    dist.broadcast(Psi, src=torch_group_root(pool.comm), group=pool.comm)
    dist.broadcast(Sig, src=torch_group_root(pool.comm), group=pool.comm)
    return Phi, Psi, Sig, info
