"""Trajectory integration and bounded retention of adjoint base flows."""

from typing import TYPE_CHECKING, Any

import numpy as np

from ...time_steppers import FixedStepSolution, solve_fixed_ivp, solve_ivp

if TYPE_CHECKING:
    from .nitrom_module import NitromModule


def _time_runs(module: "NitromModule") -> list | None:
    """Runs of equal measurement intervals, ``[(k0, k1, delta)]``: the
    intervals ``[t_{k-1}, t_k]``, ``k0 < k <= k1``, all have length
    ``delta`` (to 1e-9 relative).  One run on a uniform grid; ``None``
    for ``rk45``, which has no fixed sub-step grid."""
    if module.time_stepper == "rk45":
        return None
    time = np.asarray(module.backend.to_numpy(module.time), dtype=float)
    cached = getattr(module, "_time_runs_cache", None)
    if cached is not None and np.array_equal(cached[0], time):
        return cached[1]
    d = np.diff(time)
    tol = 1e-9 * float(np.max(np.abs(d)))
    runs, k0 = [], 0
    for k in range(1, len(d) + 1):
        if k == len(d) or abs(d[k] - d[k0]) > tol:
            runs.append((k0, k, float(time[k] - time[k0]) / (k - k0)))
            k0 = k
    module._time_runs_cache = (time, runs)
    return runs


def _dense_fits(module: "NitromModule") -> bool:
    """Whether the dense solve (states and stages on every sub-step)
    stays under ``max_dense_bytes``."""
    ntraj = module.training_data.X.shape[0]
    r = module.model.state_dimension
    nt_sim = (len(module.time) - 1) * module.n_substeps
    itemsize = getattr(module.training_data.X.dtype, "itemsize", 8)
    n_stages = 4  # upper bound over the fixed-step tableaus
    return (
        ntraj * r * itemsize * ((nt_sim + 1) + nt_sim * n_stages)
        <= module.max_dense_bytes
    )


def _solve_piecewise(
    module: "NitromModule",
    z0: Any,
    runs: list,
    save_every: int,
    with_stages: bool = False,
) -> FixedStepSolution:
    """Fixed-step solve over a non-uniform grid, one :func:`solve_fixed_ivp`
    per run of equal intervals with ``h = delta / n_substeps``; the runs are
    concatenated so that sub-step ``(k - 1) * n_substeps`` is snapshot
    ``k - 1`` (``save_every = 1``), or the snapshots alone are kept
    (``save_every = n_substeps``)."""
    bkend = module.backend
    time = np.asarray(bkend.to_numpy(module.time), dtype=float)
    ef = module.forcing_fns or None
    ts, Xs, Gs, z = [], [], [], z0
    for i, (k0, k1, delta) in enumerate(runs):
        sol = solve_fixed_ivp(
            module.model.evaluate_rhs,
            z,
            float(time[k0]),
            float(time[k1]),
            delta / module.n_substeps,
            module.time_stepper,
            save_every=save_every,
            with_stages=with_stages,
            external_forcing=ef,
            backend=module.backend,
        )
        z = sol.X[..., -1]
        first = 0 if i == 0 else 1  # the run's start is the previous run's end
        ts.append(sol.t[first:])
        Xs.append(sol.X[..., first:])
        if with_stages:
            Gs.append(sol.stages)
    return FixedStepSolution(
        t=bkend.concatenate(ts, axis=0),
        X=bkend.concatenate(Xs, axis=-1),
        stages=bkend.concatenate(Gs, axis=0) if with_stages else None,
    )


def _dense_grid(module: "NitromModule") -> float | None:
    """Uniform sub-step size, if the snapshot grid lands exactly on sub-steps.

    The per-interval path integrates each ``[t_{k-1}, t_k]`` with its own
    ``h = (t_k - t_{k-1}) / n_substeps``.  A single global solve uses one
    uniform ``dt``, which is a *different discretization* unless the snapshot
    grid is uniform -- so this returns ``None`` on a non-uniform grid rather
    than silently changing the answer.  ``rk45`` has no fixed sub-step grid
    at all.

    :returns: the sub-step size, or ``None`` if the dense path does not apply
    """
    if module.time_stepper == "rk45":
        return None

    time = module.time
    nt = len(time)
    if nt < 2:
        return None

    time = np.asarray(module.backend.to_numpy(time), dtype=float)
    t0, tf = float(time[0]), float(time[-1])
    span = tf - t0
    if span <= 0.0:
        return None

    # Reproduce solve_fixed_ivp's own grid arithmetic exactly.
    nt_sim = round(span / ((float(time[1]) - t0) / module.n_substeps))
    if nt_sim != (nt - 1) * module.n_substeps:
        return None
    dt = span / nt_sim

    tol = 1e-9 * span
    for k in range(nt):
        if abs(float(time[k]) - (t0 + k * module.n_substeps * dt)) > tol:
            return None

    # Memory guard: (ntraj, r, nt_sim + 1) plus the stage cache.
    ntraj = module.training_data.X.shape[0]
    r = module.model.state_dimension
    itemsize = getattr(module.training_data.X.dtype, "itemsize", 8)
    n_stages = 4  # upper bound over the fixed-step tableaus
    nbytes = ntraj * r * itemsize * ((nt_sim + 1) + nt_sim * n_stages)
    if nbytes > module.max_dense_bytes:
        return None

    return dt


def solve_at_measurements(module: "NitromModule", z0: Any) -> Any:
    """Return latent states as ``(ntraj, r, nt)`` at measurement times."""
    runs = _time_runs(module)
    if runs is None or len(runs) == 1:
        dt = (module.time[1] - module.time[0]) / module.n_substeps
        Z = solve_ivp(
            module.model.evaluate_rhs,
            z0,
            module.time[0],
            module.time[-1],
            dt,
            module.time,
            module.time_stepper,
            atol=module.atol,
            rtol=module.rtol,
            external_forcing=module.forcing_fns or None,
            backend=module.backend,
        )  # (ntraj, r, nt)
    else:
        Z = _solve_piecewise(module, z0, runs, save_every=module.n_substeps).X

    return Z


def solve_for_adjoint(
    module: "NitromModule", z0: Any
) -> tuple[Any, FixedStepSolution | None]:
    """Retain integration stages when the configured memory budget permits."""
    bkend = module.backend
    time = module.time
    time_h = np.asarray(bkend.to_numpy(time), dtype=float)
    dt = (time[1] - time[0]) / module.n_substeps
    ef = module.forcing_fns or None
    # When the snapshot grid lands on sub-steps, one dense solve serves
    # both the measurement values and every interval of the backward
    # sweep, which otherwise re-integrates each interval from scratch.
    runs = _time_runs(module)
    dense_dt = _dense_grid(module)
    want_stages = module.adjoint_method == "discrete"
    if runs is not None and len(runs) > 1:
        # Non-uniform grid: per-run fixed-step solves on the sub-step
        # grid of the backward sweep (h_k = delta_k / n_substeps).
        if _dense_fits(module):
            dense = _solve_piecewise(
                module, z0, runs, save_every=1, with_stages=want_stages
            )
            Z = dense.X[:, :, :: module.n_substeps]  # (ntraj, r, nt)
        else:
            dense = None
            Z = _solve_piecewise(module, z0, runs, save_every=module.n_substeps).X
    elif dense_dt is not None:
        dense = solve_fixed_ivp(
            module.model.evaluate_rhs,
            z0,
            float(time_h[0]),
            float(time_h[-1]),
            dense_dt,
            module.time_stepper,
            save_every=1,
            with_stages=want_stages,
            external_forcing=ef,
            backend=module.backend,
        )
        Z = dense.X[:, :, :: module.n_substeps]  # (ntraj, r, nt)
    else:
        dense = None
        Z = solve_ivp(
            module.model.evaluate_rhs,
            z0,
            time[0],
            time[-1],
            dt,
            time,
            module.time_stepper,
            atol=module.atol,
            rtol=module.rtol,
            external_forcing=ef,
            backend=module.backend,
        )  # (ntraj, r, nt)

    return Z, dense
