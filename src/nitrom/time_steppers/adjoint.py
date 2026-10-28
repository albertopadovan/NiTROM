"""Discrete adjoints of Runge-Kutta steps."""

from collections.abc import Callable
from typing import Any, Literal

from ..backend import get_backend
from ._methods import _BUTCHER_TABLEAUS
from ._stages import _rk_step


def solve_adjoint_ivp_discrete(
    f: Callable[..., Any],
    vjp_z: Callable[..., Any],
    vjp_theta: Callable[..., Any],
    Zint: Any,
    sub_t: Any,
    lam_init: Any,
    method: Literal["rk4", "rk2", "backward_euler", "rk45"] = "rk4",
    newton_tol: float = 1e-8,
    newton_max_iter: int = 20,
    *args,
    collector: Any = None,
    stages_G: Any = None,
    backend=None,
    **kwargs,
) -> tuple[Any, list[Any] | None]:
    r"""
    Propagate the adjoint state backward in time through the discrete RK solver
    stages and accumulate the parameter VJPs.

    :param f: right-hand side evaluate_rhs: ``(t, z, *args, **kwargs) -> dz``
    :param vjp_z: VJP w.r.t state (evaluate_adjoint_rhs): ``(t, w, z, *args, **kwargs) -> dz_vjp``
    :param vjp_theta: VJP w.r.t parameters: ``(z, w, t) -> list_of_theta_vjps``
    :param Zint: forward trajectory states over the sub-grid: ``(B, n, n_substeps + 1)``
    :param sub_t: time values of the sub-grid: ``(n_substeps + 1,)``
    :param lam_init: initial adjoint seed state: ``(B, n)``
    :param method: ``"rk4"``, ``"rk2"``, or ``"backward_euler"``.
        ``"rk45"`` raises :class:`NotImplementedError`.
    :param stages_G: forward Runge-Kutta stage states from
        :func:`solve_fixed_ivp`, shape ``(n_substeps, s, B, n)`` (or ``(n_substeps, s, n)`` unbatched).  When given,
        the stages are read from here instead of being recomputed, which for
        implicit methods also skips a full Newton solve per stage.
    :param collector: optional sink with an ``add(z, w, t)`` method.  When given,
        stage contributions are buffered there for a single bulk contraction
        instead of being reduced per stage, and the returned gradient list is
        ``None`` -- the caller reads the totals off the collector.
    :returns: tuple of the final adjoint state and the accumulated parameter
        gradients, or ``None`` for the gradients when *collector* is used
    """
    if method == "rk45" or method not in _BUTCHER_TABLEAUS:
        raise NotImplementedError(f"Discrete adjoint not implemented for {method}")

    bkend = backend or get_backend()
    lam = bkend.copy(lam_init)

    tableau = _BUTCHER_TABLEAUS[method]
    c = tableau["c"]
    b = tableau["b"]
    A = tableau["A"]
    s = len(c)

    n_substeps = len(sub_t) - 1
    param_grads = None

    dtype = Zint.dtype
    dev = bkend.device_of(Zint)
    n = Zint.shape[-2]

    eye = bkend.eye(n, dtype=dtype, device=dev)
    if Zint.ndim == 3:
        eye = eye[None]  # (1, n, n) for batch broadcasting

    # Host copy of the sub-grid (see solve_fixed_ivp).
    sub_t_h = bkend.to_numpy(sub_t)
    for j in range(n_substeps - 1, -1, -1):
        z_n = Zint[..., j]
        t_n = float(sub_t_h[j])
        h_j = float(sub_t_h[j + 1] - sub_t_h[j])

        # 1. Forward stages: reuse the cached ones, or rebuild them locally.
        if stages_G is not None:
            stages_g = [stages_G[j, i] for i in range(s)]
        else:
            _, stages_g, _ = _rk_step(
                f,
                t_n,
                z_n,
                h_j,
                tableau,
                newton_tol,
                newton_max_iter,
                args,
                kwargs,
                backend=bkend,
            )

        # 2. Initialize adjoint variables for the step
        bar_k = [lam * (h_j * b[i]) for i in range(s)]
        bar_g = [None] * s

        lam_x = bkend.copy(lam)

        # 3. Backward stage loop
        for i in range(s - 1, -1, -1):
            if bar_k[i] is not None:
                if A[i][i] != 0.0:  # Implicit stage solve (transposed system)
                    J_f = bkend.jacobian(f, t_n + c[i] * h_j, stages_g[i], args, kwargs)
                    if J_f.ndim == 2:
                        J_F_T = eye - (h_j * A[i][i]) * bkend.permute(J_f, (1, 0))
                    else:
                        J_F_T = eye - (h_j * A[i][i]) * bkend.permute(J_f, (0, 2, 1))
                    w_i = bkend.solve(J_F_T, bar_k[i])
                else:  # Explicit stage
                    w_i = bar_k[i]

                # Compute state VJP: dz_i = VJP_z(t_n + c_i * h_j, w_i, g_i)
                dz_i = vjp_z(t_n + c[i] * h_j, w_i, stages_g[i], *args, **kwargs)
                bar_g[i] = dz_i

                # Compute parameter VJP
                if collector is not None:
                    collector.add(stages_g[i], w_i, t_n + c[i] * h_j)
                else:
                    dtheta_i = vjp_theta(stages_g[i], w_i, t_n + c[i] * h_j)
                    if param_grads is None:
                        param_grads = [bkend.zeros_like(p) for p in dtheta_i]
                    for idx, g_param in enumerate(dtheta_i):
                        param_grads[idx] = param_grads[idx] + g_param

            if bar_g[i] is not None:
                lam_x = lam_x + bar_g[i]
                for idx_j in range(i):
                    if A[i][idx_j] != 0.0:
                        term = bar_g[i] * (h_j * A[i][idx_j])
                        bar_k[idx_j] = (
                            term if bar_k[idx_j] is None else bar_k[idx_j] + term
                        )

        lam = lam_x

    return lam, param_grads
