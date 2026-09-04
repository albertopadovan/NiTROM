r"""
Alternating least squares (ALS) for oblique operator inference.

:class:`~nitrom.optimization.ObliqueOpInfModule` minimizes

.. math::

    J = \sum_{j} \frac{1}{w_j} \sum_{i}
        \bigl\| S\Psi^\top \dot{q}^{(j)}(t_i)
                - S\,g\bigl(\Psi^\top q^{(j)}(t_i), u(t_i)\bigr)\bigr\|^2
        + \lambda \|H_r\|_F^2

over the latent-model tensors *and* the test basis.  A joint quasi-Newton
descent handles that badly: the two parameter blocks are coupled through
:math:`z = \Psi^\top q`, the latent data move whenever :math:`\Psi` moves, and
the resulting landscape stalls a line search early.

The block structure is much friendlier than the joint problem:

* **with the encoder fixed**, the ROM right-hand side is *linear* in its
  tensors, so the tensor block is a regularized linear least-squares problem
  with a closed-form solution (:func:`~nitrom.optimization.solve_opinf`);
* **with the tensors fixed**, the residual is a smooth nonlinear least-squares
  function of the encoder parameters -- for a degree-:math:`d` model,
  :math:`z = \Psi^\top q` enters with degree :math:`d` -- which is exactly what
  Levenberg--Marquardt is built for.

Alternating the two is both far cheaper per sweep and far more reliable than
descending on everything at once.

This solver requires the encoder to be charted as
:math:`\Psi = \Phi + W N` (:class:`~nitrom.projections.ObliqueChartProjection`).
That is what makes the tensor block a *linear* least-squares problem: the chart
guarantees :math:`\Psi^\top\Phi = I`, hence :math:`S = I`, so at fixed
:math:`N` the oblique cost coincides with the ordinary operator-inference cost
in the coordinates :math:`z = \Psi^\top q`.  With a free :math:`\Psi` the
factor :math:`S = (\Psi^\top\Phi)^{-1}` would reintroduce the coupling and the
closed form would be lost.
"""

from typing import Any

from ..backend import get_backend
from ..projections.oblique_chart_projection import ObliqueChartProjection
from .modules.oblique_opinf_module import ObliqueOpInfModule
from .modules.opinf_module import OpInfModule
from .opinf_solver import solve_opinf


def _solve_tensor_block(module: ObliqueOpInfModule) -> None:
    """
    Closed-form update of the latent-model tensors at the current encoder.

    Builds an :class:`OpInfModule` view of the problem in the current latent
    coordinates -- legitimate precisely because the chart forces ``S = I``, so
    the two costs agree at fixed ``N`` -- solves it exactly, and copies the
    result back into ``module``'s registered parameters.
    """
    bkend = module.backend
    module._sync_to_registry()

    view = OpInfModule(
        module.training_data, module.rom, module.projection, reg=module.reg
    )
    # Carry any frozen tensors (e.g. a fixed input operator B) into the solve.
    for name in module.rom.param_names:
        if name in module.is_learnable and not module.is_learnable[name]:
            view.set_unlearnable(name)
    solve_opinf(view)

    for name, value in zip(module.rom.param_names, module.rom.get_params()):
        module.register_parameter(name, bkend.copy(value))


def _latent_jacobians(module: ObliqueOpInfModule, Z: Any) -> Any:
    r"""
    Jacobian of the latent right-hand side at every row of ``Z``.

    ``evaluate_adjoint_rhs(t, v, Z)`` applies :math:`J^\top`, so seeding it with
    the basis vector :math:`e_a` returns row :math:`a` of :math:`J`.  Sweeping
    :math:`a` over the :math:`r` basis vectors therefore builds the full
    Jacobian in ``r`` batched calls.

    :param module: the module being fitted
    :param Z: latent states, shape ``(M, r)``
    :returns: ``Jg`` of shape ``(M, r, r)`` with
        ``Jg[m, a, c] = d g_a / d z_c`` at ``Z[m]``
    """
    bkend = module.backend
    M, r = Z.shape
    dev, dtype = bkend.device_of(Z), Z.dtype
    nt = module.nt
    forced = module.forcing_fns is not None and len(module.forcing_fns) > 0

    Jg = bkend.zeros((M, r, r), device=dev, dtype=dtype)
    for a in range(r):
        seed = bkend.zeros((M, r), device=dev, dtype=dtype)
        seed[:, a] = 1.0
        if forced:
            # An additive input B u(t) does not enter the Jacobian, but the
            # model is free to make it time dependent, so honour the per-
            # snapshot time as the module itself does.
            for i in range(nt):
                Jg[i::nt, a, :] = module.rom.evaluate_adjoint_rhs(
                    module.time[i], seed[i::nt], Z[i::nt]
                )
        else:
            Jg[:, a, :] = module.rom.evaluate_adjoint_rhs(0.0, seed, Z)
    return Jg


def _solve_encoder_block(module: ObliqueOpInfModule, **lm_kwargs) -> None:
    r"""
    Levenberg--Marquardt update of the chart coefficients ``N`` at fixed
    tensors.

    The residual is a sum of squares but *not* linear in ``N``: with
    :math:`\Psi = \Phi + W N`, both :math:`z = \Psi^\top q` and
    :math:`\dot{z} = \Psi^\top\dot{q}` are affine in :math:`N`, and the
    degree-:math:`d` term of the model feeds :math:`z` in :math:`d` times, so
    the residual is a degree-:math:`d` polynomial in :math:`N` (cubic for the
    CGL model).  That is nonlinear least squares, which is what LM is for.

    The Jacobian is supplied analytically.  Writing :math:`U = XW` and
    :math:`\dot{U} = \dot{X}W` (both constant, since :math:`X`, :math:`\dot{X}`
    and :math:`W` are fixed), differentiating
    :math:`\rho_{m,a} = \sqrt{w_m}\,\bigl(\dot{z}_{m,a} - g_a(z_m)\bigr)` gives

    .. math::

        \frac{\partial \rho_{m,a}}{\partial N_{pq}}
            = \sqrt{w_m}\left(
                \delta_{aq}\,\dot{U}_{m,p}
                - [J_g(z_m)]_{a,q}\, U_{m,p}
              \right),

    since :math:`\partial z_{m,c}/\partial N_{pq} = \delta_{cq} U_{m,p}`.

    The Tikhonov term does not involve ``N``, so it is constant here and is
    left out of both the residual and the Jacobian.
    """
    import numpy as np
    from scipy.optimize import least_squares

    bkend = module.backend
    proj = module.projection
    r = module.rom.state_dimension
    k = proj.W.shape[1]
    shape = (k, r)

    sqrt_w = bkend.sqrt(module.w_row)          # (M, 1)
    U = module.X @ proj.W                      # (M, k), constant
    dU = module.dX @ proj.W                    # (M, k), constant
    eye_r = bkend.eye(r, dtype=module.N.dtype, device=bkend.device_of(module.N))

    def _set(nvec):
        module.register_parameter(
            "N",
            bkend.asarray(nvec.reshape(shape), dtype=module.N.dtype,
                          device=bkend.device_of(module.N)),
        )
        module._sync_to_registry()

    def residual(nvec):
        _set(nvec)
        _, R = module._residual()
        return np.asarray(bkend.to_numpy(sqrt_w * R)).ravel()

    def jacobian(nvec):
        _set(nvec)
        Z, _ = module._residual()
        Jg = _latent_jacobians(module, Z)                    # (M, r, r)
        # d rho[m, a] / d N[p, q], assembled as (M, r, k, r).
        T = bkend.einsum("mp,aq->mapq", dU, eye_r) - bkend.einsum(
            "maq,mp->mapq", Jg, U
        )
        T = T * sqrt_w[:, :, None, None]
        return np.asarray(bkend.to_numpy(T)).reshape(T.shape[0] * r, k * r)

    opts = dict(method="lm", jac=jacobian, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    opts.update(lm_kwargs)
    sol = least_squares(
        residual, np.asarray(bkend.to_numpy(module.N)).ravel(), **opts
    )
    _set(sol.x)  # leave the module holding the accepted iterate


def solve_oblique_opinf(
    module: ObliqueOpInfModule,
    n_sweeps: int = 30,
    tol: float = 1e-12,
    verbose: bool = False,
    **lm_kwargs: Any,
) -> ObliqueOpInfModule:
    r"""
    Fit an :class:`ObliqueOpInfModule` by alternating least squares.

    Each sweep solves the latent-model tensors in closed form at the current
    encoder, then refines the chart coefficients :math:`N` by
    Levenberg--Marquardt at the resulting tensors.  Sweep zero's tensor solve
    is, by construction, exactly ordinary operator inference (``N`` starts
    whatever it was initialized to; ``N = 0`` gives :math:`\Psi = \Phi`), so the
    cost after the first solve is the OpInf cost and every later sweep can only
    reduce it.

    The module is updated in place and also returned.

    :param module: the module to fit; its projection must be an
        :class:`~nitrom.projections.ObliqueChartProjection`
    :type module: ObliqueOpInfModule
    :param n_sweeps: maximum number of alternating sweeps
    :type n_sweeps: int
    :param tol: stop when the relative decrease in cost over one sweep falls
        below this
    :type tol: float
    :param verbose: print the cost after each sweep
    :type verbose: bool
    :param lm_kwargs: extra keyword arguments forwarded to
        :func:`scipy.optimize.least_squares` for the encoder block
    :returns: the fitted module, with ``loss_history`` set to the per-sweep
        costs
    :rtype: ObliqueOpInfModule
    :raises TypeError: if the projection is not an ``ObliqueChartProjection``
    :raises NotImplementedError: on a non-numpy backend (the encoder block uses
        :func:`scipy.optimize.least_squares`)
    """
    if not isinstance(module.projection, ObliqueChartProjection):
        raise TypeError(
            "solve_oblique_opinf requires an ObliqueChartProjection: the "
            "closed-form tensor solve relies on the chart's Psi^T Phi = I. "
            f"Got {type(module.projection).__name__}; use train() instead."
        )
    if not get_backend().is_numpy:
        raise NotImplementedError(
            "solve_oblique_opinf's encoder block uses scipy.optimize."
            "least_squares and is available on the numpy backend only."
        )

    history = []
    _solve_tensor_block(module)
    history.append(float(module()))
    if verbose:
        print(f"  sweep  0 (OpInf): J = {history[-1]:.6e}")

    for sweep in range(1, n_sweeps + 1):
        _solve_encoder_block(module, **lm_kwargs)
        _solve_tensor_block(module)
        history.append(float(module()))
        if verbose:
            print(f"  sweep {sweep:2d}:         J = {history[-1]:.6e}")
        if abs(history[-2] - history[-1]) <= tol * max(1.0, abs(history[-2])):
            if verbose:
                print(f"  converged after {sweep} sweeps")
            break

    module.loss_history = history
    return module
