from collections.abc import Callable
from typing import Any, Literal, NamedTuple

from nitrom.backend import get_backend
from nitrom.utils import interp_quadratic


_BUTCHER_TABLEAUS = {
    "rk4": {
        "c": [0.0, 0.5, 0.5, 1.0],
        "b": [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0],
        "A": [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ]
    },
    "rk2": {
        "c": [0.0, 1.0],
        "b": [0.5, 0.5],
        "A": [
            [0.0, 0.0],
            [1.0, 0.0]
        ]
    },
    "backward_euler": {
        "c": [1.0],
        "b": [1.0],
        "A": [
            [1.0]
        ]
    },
    "rk45": {
        "c": [0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0],
        "b": [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0],
        "b_err": [71.0/57600.0, 0.0, -71.0/16695.0, 71.0/1920.0, -17253.0/339200.0, 22.0/525.0, -1.0/40.0],
        "A": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0],
            [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0],
            [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0],
            [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]
        ]
    }
}


def _jacobian_f(f, t_eval, y, args, kwargs, bkend):
    """Jacobian of ``f(t_eval, .)`` at ``y`` -- ``(n, n)`` or batched ``(B, n, n)``.

    Uses ``torch.func.jacrev`` (exact) on the torch backend and a forward
    finite difference on the numpy backend.
    """
    n = y.shape[-1]
    dev = bkend.device_of(y)

    if bkend.is_torch:
        import torch
        from torch.func import jacrev

        y_detached = y.detach().requires_grad_(True)

        def single_f(u):
            return f(t_eval, u, *args, **kwargs)

        if y.ndim == 1:
            return jacrev(single_f)(y_detached).detach()

        B_size = y.shape[0]

        # Each output row depends only on its own input row, so the batched
        # Jacobian is block diagonal.  One jacrev over the whole batch and then
        # taking the diagonal blocks costs a single RHS trace; differentiating
        # row by row instead re-evaluates the *full* batched RHS once per row,
        # i.e. O(B^2) work for the same numbers.  The full Jacobian is
        # (B, n, B, n), so fall back to the per-row loop when that would be
        # large.
        if B_size * B_size * n * n <= 1 << 20:
            J_full = jacrev(single_f)(y_detached).detach()  # (B, n, B, n)
            idx = torch.arange(B_size, device=J_full.device)
            return J_full[idx, :, idx, :]

        J = bkend.zeros((B_size, n, n), dtype=y.dtype, device=dev)
        for b in range(B_size):
            def single_f_batched(u, b=b):
                y_list = [
                    y_detached[i].detach() if i != b else u
                    for i in range(B_size)
                ]
                return f(t_eval, torch.stack(y_list, dim=0), *args, **kwargs)[b]
            J[b] = jacrev(single_f_batched)(y_detached[b]).detach()
        return J

    # numpy: forward finite difference (columns of J), batched together.
    eps = 1e-7
    f0 = f(t_eval, y, *args, **kwargs)
    if y.ndim == 1:
        J = bkend.zeros((n, n), dtype=y.dtype, device=dev)
        for j in range(n):
            yp = bkend.copy(y)
            yp[j] += eps
            J[:, j] = (f(t_eval, yp, *args, **kwargs) - f0) / eps
        return J

    B_size = y.shape[0]
    J = bkend.zeros((B_size, n, n), dtype=y.dtype, device=dev)
    for j in range(n):
        yp = bkend.copy(y)
        yp[:, j] += eps
        # f is batched and each output row depends only on its own input row.
        J[:, :, j] = (f(t_eval, yp, *args, **kwargs) - f0) / eps
    return J


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
    **kwargs,
) -> Any:
    r"""
    Solve the implicit system for y:

        y - rhs_const - dt * alpha * f(t_eval, y, *args, **kwargs) = 0

    using Newton-Raphson.
    """
    bkend = get_backend()
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
        res_val = float(res_norm.detach()) if hasattr(res_norm, "detach") else float(res_norm)
        if res_val < newton_tol:
            break

        J_f = _jacobian_f(f, t_eval, y, args, kwargs, bkend)
        J_F = eye - (dt * alpha) * J_f
        delta_y = bkend.solve(J_F, -F_val)
        y = y + delta_y
    else:
        res_val = float(res_norm.detach()) if hasattr(res_norm, "detach") else float(res_norm)
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
) -> tuple[Any, list, list]:
    """One Runge-Kutta step.

    Returns ``(x_next, stages_g, stages_k)`` -- the stage *states* and stage
    *derivatives* alongside the updated state.  :func:`evolve` discards the
    stages; :func:`solve_ivp_dense` can retain ``stages_g`` so the discrete
    adjoint does not have to rebuild them.
    """
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
                f, t + c[i] * dt, y0, const_i, dt, A[i][i],
                newton_tol, newton_max_iter, *args, **kwargs,
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


def evolve(
    f: Callable[..., Any],
    t: float,
    x: Any,
    dt: float,
    method: Literal["rk4", "rk2", "backward_euler", "rk45"] = "rk4",
    newton_tol: float = 1e-8,
    newton_max_iter: int = 20,
    return_error: bool = False,
    *args,
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
    if method in _BUTCHER_TABLEAUS:
        tableau = _BUTCHER_TABLEAUS[method]
        s = len(tableau["c"])
        x_next, _stages_g, stages_k = _rk_step(
            f, t, x, dt, tableau, newton_tol, newton_max_iter, args, kwargs,
        )

        if return_error:
            if "b_err" not in tableau:
                raise ValueError(f"Method '{method}' does not support error estimation.")
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
    bkend = get_backend()
    dev = bkend.device_of(x0)
    dtype = x0.dtype
    batched = x0.ndim == 2
    t0, tf, dt = float(t0), float(tf), float(dt)

    x = bkend.copy(x0)

    if method == "rk45":
        tsave_list = [t0]
        xsave_list = [x]
        t = t0
        h = dt
        h_min = 1e-15

        while t < tf:
            if t + h > tf:
                h_step = tf - t
            else:
                h_step = h

            x_next, x_err = evolve(
                f, t, x, h_step, method="rk45", return_error=True,
                *args, **kwargs
            )

            # Error estimation
            abs_x = bkend.xp.abs(x)
            abs_x_next = bkend.xp.abs(x_next)
            scale = atol + rtol * bkend.xp.maximum(abs_x, abs_x_next)
            ratio = x_err / scale
            mean_sq = bkend.xp.mean(ratio ** 2, axis=-1)
            err_norm = bkend.xp.sqrt(mean_sq)

            if hasattr(err_norm, "detach"):
                err_norm_detached = err_norm.detach()
            else:
                err_norm_detached = err_norm

            err_val = float(bkend.xp.max(err_norm_detached)) if hasattr(err_norm_detached, "ndim") and err_norm_detached.ndim > 0 else float(err_norm_detached)

            if err_val <= 1.0:
                t = t + h_step
                x = x_next
                tsave_list.append(t)
                xsave_list.append(x)

                if err_val == 0.0:
                    factor = 10.0
                else:
                    factor = 0.9 * (err_val ** -0.2)
                    factor = min(max(factor, 0.2), 10.0)
                h = h_step * factor
            else:
                factor = 0.9 * (err_val ** -0.2)
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
        return interp_quadratic(t_eval, tsave, X)

    # Fixed-step: integrate on a uniform grid, sub-sample, then interpolate.
    nt_sim = int(round((tf - t0) / dt))
    dt_grid = (tf - t0) / nt_sim
    dteval_min = float((t_eval[1:] - t_eval[:-1]).min())
    save_every = max(int(round(dteval_min / dt_grid)), 1)

    sol = solve_ivp_dense(
        f, x0, t0, tf, dt_grid, method, newton_tol, newton_max_iter,
        save_every=save_every, *args, **kwargs,
    )
    return interp_quadratic(t_eval, sol.t, sol.X)


class DenseSolution(NamedTuple):
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


def solve_ivp_dense(
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
    **kwargs,
) -> DenseSolution:
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
    :returns: :class:`DenseSolution`
    """
    if method == "rk45":
        raise ValueError(
            "solve_ivp_dense requires a fixed-step method; rk45 chooses its "
            "own steps. Use solve_ivp instead."
        )
    if method not in _BUTCHER_TABLEAUS:
        raise ValueError(f"Unknown integration method: {method}")

    bkend = get_backend()
    dev = bkend.device_of(x0)
    dtype = x0.dtype
    batched = x0.ndim == 2
    t0, tf, dt = float(t0), float(tf), float(dt)

    nt_sim = int(round((tf - t0) / dt))
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

    for i in range(1, nt_sim + 1):
        t = float(tsim[i - 1])
        x, stages_g, _stages_k = _rk_step(
            f, t, x, dt, tableau, newton_tol, newton_max_iter, args, kwargs,
        )
        if G is not None:
            G[i - 1] = bkend.stack(stages_g, axis=0)
        if i % save_every == 0:
            if batched:
                X[:, :, i // save_every] = x
            else:
                X[:, i // save_every] = x

    return DenseSolution(t=tsim[::save_every], X=X, stages=G)


def solve_adjoint_ivp_discrete(
    f: Callable[..., Any],
    vjp_z: Callable[..., Any],
    vjp_theta: Callable[..., Any],
    Zint: Any,
    sub_t: Any,
    h: float,
    lam_init: Any,
    method: Literal["rk4", "rk2", "backward_euler", "rk45"] = "rk4",
    newton_tol: float = 1e-8,
    newton_max_iter: int = 20,
    *args,
    collector: Any = None,
    stages_G: Any = None,
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
    :param h: step size (ignored if non-uniform, computed from sub_t instead)
    :param lam_init: initial adjoint seed state: ``(B, n)``
    :param method: ``"rk4"``, ``"rk2"``, ``"backward_euler"``, or ``"rk45"``
    :param stages_G: forward Runge-Kutta stage states from
        :func:`solve_ivp_dense`, shape ``(B, n, n_substeps, s)``.  When given,
        the stages are read from here instead of being recomputed, which for
        implicit methods also skips a full Newton solve per stage.
    :param collector: optional sink with an ``add(z, w, t)`` method.  When given,
        stage contributions are buffered there for a single bulk contraction
        instead of being reduced per stage, and the returned gradient list is
        ``None`` -- the caller reads the totals off the collector.
    :returns: tuple of the final adjoint state and the accumulated parameter
        gradients, or ``None`` for the gradients when *collector* is used
    """
    bkend = get_backend()
    lam = bkend.copy(lam_init) if hasattr(lam_init, "copy") else lam_init * 1.0
    
    if method not in _BUTCHER_TABLEAUS:
        raise NotImplementedError(f"Discrete adjoint not implemented for {method}")
        
    tableau = _BUTCHER_TABLEAUS[method]
    c = tableau["c"]
    b = tableau["b"]
    A = tableau["A"]
    s = len(c)
    
    n_substeps = len(sub_t) - 1
    param_grads = None
    
    dtype = Zint.dtype
    dev = bkend.device_of(Zint)
    n = Zint.shape[1]
    
    eye = bkend.eye(n, dtype=dtype, device=dev)
    if Zint.ndim == 3:
        eye = eye[None]  # (1, n, n) for batch broadcasting
        
    for j in range(n_substeps - 1, -1, -1):
        z_n = Zint[:, :, j]
        t_n = float(sub_t[j])
        h_j = float(sub_t[j+1] - sub_t[j])
        
        # 1. Forward stages: reuse the cached ones, or rebuild them locally.
        if stages_G is not None:
            stages_g = [stages_G[j, i] for i in range(s)]
        else:
            stages_g = []
            stages_k = []
            for i in range(s):
                val = None
                for idx_j in range(i):
                    if A[i][idx_j] != 0.0:
                        term = A[i][idx_j] * stages_k[idx_j]
                        val = term if val is None else val + term
                const_i = z_n if val is None else z_n + h_j * val

                if A[i][i] != 0.0:  # Implicit stage
                    f_guess = f(t_n, z_n, *args, **kwargs)
                    y0 = const_i + h_j * A[i][i] * f_guess
                    g_i = _newton_solve(
                        f, t_n + c[i] * h_j, y0, const_i, h_j, A[i][i],
                        newton_tol, newton_max_iter, *args, **kwargs,
                    )
                    k_i = f(t_n + c[i] * h_j, g_i, *args, **kwargs)
                else:  # Explicit stage
                    g_i = const_i
                    k_i = f(t_n + c[i] * h_j, g_i, *args, **kwargs)
                stages_g.append(g_i)
                stages_k.append(k_i)
            
        # 2. Initialize adjoint variables for the step
        bar_k = [lam * (h_j * b[i]) for i in range(s)]
        bar_g = [None] * s
        
        lam_x = bkend.copy(lam) if hasattr(lam, "copy") else lam * 1.0
        
        # 3. Backward stage loop
        for i in range(s - 1, -1, -1):
            if bar_k[i] is not None:
                if A[i][i] != 0.0:  # Implicit stage solve (transposed system)
                    J_f = _jacobian_f(f, t_n + c[i] * h_j, stages_g[i], args, kwargs, bkend)
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
                        bar_k[idx_j] = term if bar_k[idx_j] is None else bar_k[idx_j] + term
                        
        lam = lam_x
        
    return lam, param_grads
