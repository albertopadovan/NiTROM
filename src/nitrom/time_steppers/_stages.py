"""Shared Runge-Kutta stages and implicit Newton solves."""

from collections.abc import Callable
from typing import Any

from ..backend import get_backend


def _newton_solve(
    f: Callable[..., Any],
    t_eval: float,
    y0: Any,
    rhs_const: Any,
    dt: float,
    alpha: float,
    newton_tol: float = 1e-8,
    newton_max_iter: int = 20,
    *args,
    backend=None,
    **kwargs,
) -> Any:
    r"""
    Solve the implicit system for y:

        y - rhs_const - dt * alpha * f(t_eval, y, *args, **kwargs) = 0

    using Newton-Raphson.
    """
    bkend = backend or get_backend()
    y = bkend.copy(y0)
    n = y.shape[-1]

    eye = bkend.eye(n, dtype=y.dtype, device=bkend.device_of(y))
    if y.ndim == 2:
        eye = eye[None]  # (1, n, n) for batched broadcasting

    res_norm = None
    for _ in range(newton_max_iter):
        f_val = f(t_eval, y, *args, **kwargs)
        F_val = y - rhs_const - dt * alpha * f_val

        if y.ndim == 1:
            res_norm = bkend.vector_norm(F_val)
        else:
            res_norm = bkend.vector_norm(F_val, axis=-1).max()
        res_val = (
            float(res_norm.detach()) if hasattr(res_norm, "detach") else float(res_norm)
        )
        if res_val < newton_tol:
            break

        J_f = bkend.jacobian(f, t_eval, y, args, kwargs)
        J_F = eye - (dt * alpha) * J_f
        delta_y = bkend.solve(J_F, -F_val)
        y = y + delta_y
    else:
        res_val = (
            float(res_norm.detach()) if hasattr(res_norm, "detach") else float(res_norm)
        )
        raise RuntimeError(
            f"Newton solver failed to converge within {newton_max_iter} "
            f"iterations. Final residual norm: {res_val:.2e}"
        )

    return y


def _rk_step(
    f: Callable[..., Any],
    t: float,
    x: Any,
    dt: float,
    tableau: dict,
    newton_tol: float,
    newton_max_iter: int,
    args: tuple,
    kwargs: dict,
    *,
    backend=None,
) -> tuple[Any, list, list]:
    """One Runge-Kutta step.

    Returns ``(x_next, stages_g, stages_k)`` -- the stage *states* and stage
    *derivatives* alongside the updated state.  :func:`step` discards the
    stages; :func:`solve_fixed_ivp` can retain ``stages_g`` so the discrete
    adjoint does not have to rebuild them.
    """
    bkend = backend or get_backend()
    c, b, A = tableau["c"], tableau["b"], tableau["A"]
    s = len(c)

    stages_g, stages_k = [], []
    for i in range(s):
        val = None
        for j in range(i):
            if A[i][j] != 0.0:
                term = A[i][j] * stages_k[j]
                val = term if val is None else val + term
        const_i = x if val is None else x + dt * val

        if A[i][i] != 0.0:  # Implicit stage
            f_guess = f(t, x, *args, **kwargs)
            y0 = const_i + dt * A[i][i] * f_guess
            g_i = _newton_solve(
                f,
                t + c[i] * dt,
                y0,
                const_i,
                dt,
                A[i][i],
                newton_tol,
                newton_max_iter,
                *args,
                **kwargs,
                backend=bkend,
            )
            k_i = f(t + c[i] * dt, g_i, *args, **kwargs)
        else:  # Explicit stage
            g_i = const_i
            k_i = f(t + c[i] * dt, g_i, *args, **kwargs)
        stages_g.append(g_i)
        stages_k.append(k_i)

    val_next = None
    for i in range(s):
        if b[i] != 0.0:
            term = b[i] * stages_k[i]
            val_next = term if val_next is None else val_next + term
    x_next = x if val_next is None else x + dt * val_next

    return x_next, stages_g, stages_k
