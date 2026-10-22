"""Single-step, fixed-grid, and adaptive Runge-Kutta integration."""

from collections.abc import Callable
from typing import Any, Literal, NamedTuple

from ..backend import get_backend
from ..numerics import interp_quadratic
from ._methods import _BUTCHER_TABLEAUS
from ._stages import _rk_step


def step(
    f: Callable[..., Any],
    t: float,
    x: Any,
    dt: float,
    method: Literal["rk4", "rk2", "backward_euler", "rk45"] = "rk4",
    newton_tol: float = 1e-8,
    newton_max_iter: int = 20,
    return_error: bool = False,
    *args,
    backend=None,
    **kwargs,
) -> Any:
    r"""
    Advance the state by one step using an explicit or implicit Runge-Kutta method.

    :param f: right-hand side :math:`f(t, x, \ldots)`
    :param t: current time
    :param x: current state of shape ``(n,)`` or ``(B, n)``
    :param dt: time-step size
    :param method: ``"rk4"``, ``"rk2"``, ``"backward_euler"``, or ``"rk45"``
    :param newton_tol: tolerance for the Newton solver (implicit methods only)
    :param newton_max_iter: max iterations for the Newton solver (implicit only)
    :param return_error: if True, also return the estimated local error (rk45 only)
    :param args: extra positional arguments forwarded to *f*
    :param kwargs: extra keyword arguments forwarded to *f*
    :returns: state at time :math:`t + \Delta t` (or tuple of state and error if return_error is True)
    """
    bkend = backend or get_backend()
    if method in _BUTCHER_TABLEAUS:
        tableau = _BUTCHER_TABLEAUS[method]
        s = len(tableau["c"])
        x_next, _stages_g, stages_k = _rk_step(
            f,
            t,
            x,
            dt,
            tableau,
            newton_tol,
            newton_max_iter,
            args,
            kwargs,
            backend=bkend,
        )

        if return_error:
            if "b_err" not in tableau:
                raise ValueError(
                    f"Method '{method}' does not support error estimation."
                )
            val_err = None
            for i in range(s):
                if tableau["b_err"][i] != 0.0:
                    term = tableau["b_err"][i] * stages_k[i]
                    val_err = term if val_err is None else val_err + term
            x_err = dt * val_err
            return x_next, x_err
    else:
        raise ValueError(f"Unknown integration method: {method}")
    return x_next


def solve_ivp(
    f: Callable[..., Any],
    x0: Any,
    t0: float,
    tf: float,
    dt: float,
    t_eval: Any,
    method: Literal["rk4", "rk2", "backward_euler", "rk45"] = "rk4",
    newton_tol: float = 1e-8,
    newton_max_iter: int = 20,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    *args,
    backend=None,
    **kwargs,
) -> Any:
    r"""
    Integrate an ODE IVP with the specified Runge-Kutta method and return the
    solution interpolated at the requested evaluation times.

    The integrator steps on a uniform grid with spacing close to *dt*
    (adjusted so that :math:`t_f - t_0` is an exact multiple), stores the
    solution at a sub-sampled rate, and interpolates onto *t_eval*.

    If method is "rk45", an adaptive step-size Runge-Kutta Dormand-Prince 5(4)
    stepper is used.

    :param f: right-hand side :math:`f(t, x, \ldots)`
    :param x0: initial condition of shape ``(n,)`` or ``(B, n)``
    :param t0: initial time
    :param tf: final time
    :param dt: desired time-step size (will be adjusted slightly; or initial step size for rk45)
    :param t_eval: times at which to return the solution, shape ``(n_eval,)``
    :param method: ``"rk4"``, ``"rk2"``, ``"backward_euler"``, or ``"rk45"``
    :param newton_tol: tolerance for the Newton solver (implicit methods only)
    :param newton_max_iter: max iterations for the Newton solver (implicit only)
    :param atol: absolute error tolerance (rk45 only)
    :param rtol: relative error tolerance (rk45 only)
    :param args: extra positional arguments forwarded to *f*
    :param kwargs: extra keyword arguments forwarded to *f*
    :returns: solution of shape ``(n, n_eval)`` or ``(B, n, n_eval)``
    """
    bkend = backend or get_backend()
    dev = bkend.device_of(x0)
    dtype = x0.dtype
    t0, tf, dt = float(t0), float(tf), float(dt)

    x = bkend.copy(x0)

    if method == "rk45":
        tsave_list = [t0]
        xsave_list = [x]
        t = t0
        h = dt
        h_min = 1e-15

        while t < tf:
            h_step = tf - t if t + h > tf else h

            x_next, x_err = step(
                f,
                t,
                x,
                h_step,
                "rk45",
                newton_tol,
                newton_max_iter,
                True,
                *args,
                **kwargs,
                backend=bkend,
            )

            # Error estimation
            abs_x = bkend.xp.abs(x)
            abs_x_next = bkend.xp.abs(x_next)
            scale = atol + rtol * bkend.xp.maximum(abs_x, abs_x_next)
            ratio = x_err / scale
            mean_sq = bkend.xp.mean(ratio**2, axis=-1)
            err_norm = bkend.xp.sqrt(mean_sq)

            if hasattr(err_norm, "detach"):
                err_norm_detached = err_norm.detach()
            else:
                err_norm_detached = err_norm

            err_val = (
                float(bkend.xp.max(err_norm_detached))
                if hasattr(err_norm_detached, "ndim") and err_norm_detached.ndim > 0
                else float(err_norm_detached)
            )

            if err_val <= 1.0:
                t = t + h_step
                x = x_next
                tsave_list.append(t)
                xsave_list.append(x)

                if err_val == 0.0:
                    factor = 10.0
                else:
                    factor = 0.9 * (err_val**-0.2)
                    factor = min(max(factor, 0.2), 10.0)
                h = h_step * factor
            else:
                factor = 0.9 * (err_val**-0.2)
                factor = min(max(factor, 0.2), 10.0)
                h = h_step * factor
                if h < h_min:
                    raise RuntimeError(
                        f"Adaptive RK45 step size became too small (h = {h:.2e} < {h_min:.2e}) "
                        f"at t = {t:.4f} without converging."
                    )

        # Safeguard to ensure we have at least 3 points for quadratic interpolation
        if len(tsave_list) < 3:
            t_mid = 0.5 * (tsave_list[0] + tsave_list[1])
            x_mid = 0.5 * (xsave_list[0] + xsave_list[1])
            tsave_list.insert(1, t_mid)
            xsave_list.insert(1, x_mid)

        tsave = bkend.asarray(tsave_list, dtype=dtype, device=dev)
        X = bkend.stack(xsave_list, axis=-1)
        return interp_quadratic(t_eval, tsave, X, backend=bkend)

    # Fixed-step: integrate on a uniform grid, sub-sample, then interpolate.
    nt_sim = round((tf - t0) / dt)
    dt_grid = (tf - t0) / nt_sim
    dteval_min = float((t_eval[1:] - t_eval[:-1]).min())
    save_every = max(round(dteval_min / dt_grid), 1)

    sol = solve_fixed_ivp(
        f,
        x0,
        t0,
        tf,
        dt_grid,
        method,
        newton_tol,
        newton_max_iter,
        save_every,
        False,
        *args,
        **kwargs,
        backend=bkend,
    )
    return interp_quadratic(t_eval, sol.t, sol.X, backend=bkend)


class FixedStepSolution(NamedTuple):
    """Fixed-step solution retained on the integration grid.

    :param t: grid times, shape ``(n_save,)``
    :param X: states, shape ``(B, n, n_save)`` or ``(n, n_save)``
    :param stages: per-step Runge-Kutta stage states, shape
        ``(n_step, s, B, n)`` or ``(n_step, s, n)``; ``None`` unless requested.
        Step and stage lead so that a single stage is a contiguous block: the
        backward sweep reads one per adjoint evaluation, and with the state
        axes trailing instead the read is a gather with a stride of the whole
        trajectory.
    """

    t: Any
    X: Any
    stages: Any = None


def solve_fixed_ivp(
    f: Callable[..., Any],
    x0: Any,
    t0: float,
    tf: float,
    dt: float,
    method: Literal["rk4", "rk2", "backward_euler"] = "rk4",
    newton_tol: float = 1e-8,
    newton_max_iter: int = 20,
    save_every: int = 1,
    with_stages: bool = False,
    *args,
    backend=None,
    **kwargs,
) -> FixedStepSolution:
    r"""
    Integrate on a fixed grid and return the solution *on that grid*.

    :func:`solve_ivp` interpolates onto caller-supplied times and discards the
    integration grid.  The adjoint sweep needs the grid itself -- otherwise each
    measurement interval has to be re-integrated from an interpolated restart --
    so this variant hands it back.  ``with_stages`` additionally retains the
    Runge-Kutta stage states, which lets the discrete adjoint skip rebuilding
    them.

    Only fixed-step methods are supported; ``"rk45"`` chooses its own steps and
    has no such grid.

    :param save_every: keep every ``save_every``-th step (stages, when
        requested, are always kept for every step)
    :param with_stages: also return the per-step stage states
    :returns: :class:`FixedStepSolution`
    """
    if method == "rk45":
        raise ValueError(
            "solve_fixed_ivp requires a fixed-step method; rk45 chooses its "
            "own steps. Use solve_ivp instead."
        )
    if method not in _BUTCHER_TABLEAUS:
        raise ValueError(f"Unknown integration method: {method}")

    bkend = backend or get_backend()
    dev = bkend.device_of(x0)
    dtype = x0.dtype
    batched = x0.ndim == 2
    t0, tf, dt = float(t0), float(tf), float(dt)

    nt_sim = round((tf - t0) / dt)
    dt = (tf - t0) / nt_sim
    tsim = dt * bkend.arange(nt_sim + 1, dtype=dtype, device=dev) + t0

    tableau = _BUTCHER_TABLEAUS[method]
    s_stages = len(tableau["c"])

    x = bkend.copy(x0)
    n_save = len(tsim[::save_every])

    if batched:
        B, n = x0.shape
        X = bkend.zeros((B, n, n_save), dtype=dtype, device=dev)
        X[:, :, 0] = x
        G = (
            bkend.zeros((nt_sim, s_stages, B, n), dtype=dtype, device=dev)
            if with_stages
            else None
        )
    else:
        n = x0.shape[0]
        X = bkend.zeros((n, n_save), dtype=dtype, device=dev)
        X[:, 0] = x
        G = (
            bkend.zeros((nt_sim, s_stages, n), dtype=dtype, device=dev)
            if with_stages
            else None
        )

    # Host copy of the grid: float() of a device scalar would force a device
    # sync on every step.
    tsim_h = bkend.to_numpy(tsim)
    for i in range(1, nt_sim + 1):
        t = float(tsim_h[i - 1])
        x, stages_g, _stages_k = _rk_step(
            f,
            t,
            x,
            dt,
            tableau,
            newton_tol,
            newton_max_iter,
            args,
            kwargs,
            backend=bkend,
        )
        if G is not None:
            G[i - 1] = bkend.stack(stages_g, axis=0)
        if i % save_every == 0:
            if batched:
                X[:, :, i // save_every] = x
            else:
                X[:, i // save_every] = x

    return FixedStepSolution(t=tsim[::save_every], X=X, stages=G)
