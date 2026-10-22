from typing import Any

import numpy as np

from nitrom.roms.param_registry import ParamRegistry
from nitrom.training_data import TrainingData

from ...projections.linear_projection import LinearProjection
from ._adjoints import accumulate_adjoint
from ._losses import decoder_sources, trajectory_cost
from ._trajectory import solve_at_measurements, solve_for_adjoint
from .base import InferenceModule


class NitromModule(InferenceModule):
    r"""
    NiTROM training module backed by a :class:`ParamRegistry`.

    Unlike :class:`OpInfModule`, which owns a single latent-space model,
    this module receives a :class:`ParamRegistry` that already holds both
    the latent-space :class:`Model` and the :class:`Projection`.  The
    module mirrors the registry's parameters as trainable parameters
    (shared parameters appear exactly once) and pushes them back into the
    underlying model and projection through :meth:`ParamRegistry.scatter`,
    which keeps shared parameters tied by construction.

    :param training_data: training data (trajectories ``X``, time grid,
        forcing callables, weights, and optionally ``time_weights``: per-sample
        weights :math:`c_i` of shape ``(nt,)`` or ``(ntraj, nt)`` -- the rows of
        this rank's trajectories -- e.g. the quadrature weights of a
        non-uniform time grid)
    :type training_data: TrainingData
    :param registry: registry of the ROM parameters spanning the
        latent-space model and the projection
    :type registry: ParamRegistry
    :param fom: optional full-order model, e.g. for output evaluation.  Must
        provide ``compute_output(q)`` and ``compute_output_derivative(q)``; it
        may additionally provide ``apply_output_adjoint(e, q)`` to apply the
        output Jacobian-transpose without forming the ``(no, N)`` matrix.
    :param reg: Tikhonov regularization weight
    :type reg: float
    :param gram: evaluate the output mismatch in Gram form (see below).
        ``None`` (default) uses it whenever it applies, ``False`` forces the
        general path, ``True`` requires it.
    :type gram: bool or None

    **Non-uniform time grids.**  Every measurement interval
    :math:`[t_{k-1}, t_k]` is integrated with ``n_substeps`` fixed steps of
    :math:`h_k = (t_k - t_{k-1})/n_\mathrm{substeps}`, in the cost and in the
    adjoint alike, so the gradient is that of the cost on any grid.  On a
    uniform grid this is one fixed-step solve; on a non-uniform one, one
    fixed-step solve per run of equal intervals (e.g. two for a fine start
    followed by a coarse tail), concatenated on the sub-step grid.

    **Per-sample weights.**  With ``time_weights`` the cost is

    .. math::

        J = \sum_j \alpha_j^{-1} \sum_i c_i
            \lVert y^{(j)}(t_i) - \hat{y}^{(j)}(t_i) \rVert^2,

    i.e. every snapshot scaled by :math:`\sqrt{c_i}` (with trapezoidal
    :math:`c_i`, a quadrature of the time integral of the mismatch).

    **Gram form.**  With a :class:`LinearProjection` (decoder
    :math:`\hat{q} = \Phi S z`, :math:`S = (\Psi^\top\Phi)^{-1}`) and a
    FOM whose output is the state (``fom.output_is_state = True``), the
    full-space mismatch expands exactly as

    .. math::

        \lVert x - \Phi S z\rVert^2 = \lVert x\rVert^2
            - 2\,(Sz)^\top(\Phi^\top x) + (Sz)^\top(\Phi^\top\Phi)(Sz),

    and the adjoint source and decoder gradient only ever need
    :math:`\Phi^\top X`, :math:`X (SZ)^\top` and :math:`\Phi^\top\Phi`.
    That is the same full-space error -- the part of the data outside
    :math:`\mathrm{span}(\Phi)` still enters through :math:`\lVert x\rVert^2`
    -- but neither the reconstruction nor the error is ever stored as an
    ``(ntraj, N, nt)`` array.  :math:`\lVert x\rVert^2` is computed once, and
    :math:`\Phi^\top X` is reused while :math:`\Phi` does not change (e.g.
    while the bases are frozen).

    .. note::

        The Gram form **requires linearity**: a decoder that is linear in
        :math:`z` (:class:`LinearProjection`) and an output that is the state
        itself (the FOM sets ``output_is_state = True``).  Any other
        combination -- a nonlinear decoder, a linear output ``y = C x``, or a
        nonlinear output map -- uses the general path, which forms the
        ``(ntraj, N, nt)`` reconstruction and mismatch explicitly.  The two
        paths compute the same cost and gradients to round-off (see
        ``tests/optimization/modules/test_nitrom_gram.py``).

        The Gram form obtains the mismatch by cancellation, so its relative
        round-off is about :math:`\varepsilon\,\lVert x\rVert^2 /
        \lVert x - \hat x\rVert^2`: negligible at ordinary training errors,
        but a model reconstructing the data to ~1e-6 would lose digits.  Pass
        ``gram=False`` to force the general path in that regime.
    """

    def __init__(
        self,
        training_data: TrainingData,
        registry: ParamRegistry,
        fom=None,
        reg: float = 0.0,
        n_substeps: int = 100,
        time_stepper: str = "rk4",
        n_leggauss: int = 5,
        adjoint_method: str = "discrete",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        gram: bool | None = None,
    ) -> None:
        super().__init__(backend=registry.backend)
        self.backend.validate_components(training_data, registry.projection)

        if time_stepper not in ("rk2", "rk4", "backward_euler", "rk45"):
            raise ValueError(
                f"time_stepper must be 'rk2', 'rk4', 'backward_euler', or 'rk45', "
                f"got {time_stepper!r}."
            )

        if adjoint_method not in ("discrete", "continuous"):
            raise ValueError(
                f"adjoint_method must be 'discrete' or 'continuous', "
                f"got {adjoint_method!r}."
            )

        self.registry = registry
        self.training_data = training_data
        self.fom = fom
        self.reg = reg
        self.n_substeps = n_substeps
        self.time_stepper = time_stepper
        self.n_leggauss = n_leggauss
        self.adjoint_method = adjoint_method
        self.atol = atol
        self.rtol = rtol
        gram_ok = isinstance(registry.projection, LinearProjection) and bool(
            getattr(fom, "output_is_state", False)
        )
        if gram and not gram_ok:
            raise ValueError(
                "gram=True needs a LinearProjection and a FOM with "
                "output_is_state = True."
            )
        self.gram = gram_ok if gram is None else bool(gram)

        # Convenience handles into the registry's components
        self.model = registry.model
        self.projection = registry.projection
        self.backend = self.model.backend
        bkend = self.backend

        # Gauss-Legendre nodes / weights on [-1, 1] for quadrature of the
        # adjoint parameter-gradient integrals.
        nodes, weights = np.polynomial.legendre.leggauss(n_leggauss)
        self._gl_nodes = bkend.asarray(
            nodes, device=self.model.device, dtype=self.model.dtype
        )
        self._gl_weights = bkend.asarray(
            weights, device=self.model.device, dtype=self.model.dtype
        )

        # Commonly used training data (mirrors OpInfModule conventions)
        self.forcing_fns = getattr(training_data, "forcing_fns", None)
        self.time = getattr(training_data, "time", None)
        self.weights = getattr(training_data, "weights", None)

        # Register one parameter per registered parameter, in registry order.
        for name in self.registry.names:
            self.register_parameter(name, bkend.copy(self.registry.value(name)))

    @property
    def param_names(self) -> list[str]:
        """Names of the trainable parameters, in registry order."""
        return self.registry.names

    def parameter_list(self) -> list:
        """Current trainable parameter tensors, in registry order."""
        return [getattr(self, name) for name in self.registry.names]

    def _sync_to_registry(self) -> None:
        """Push the current parameter values into the model and projection."""
        self.registry.scatter(self.parameter_list())

    def _data_output(self) -> Any:
        """``fom.compute_output`` of the training data, cached.

        The training trajectories are fixed for the whole optimization, so the
        output map is applied to them once instead of on every cost and
        gradient evaluation.  Keyed on the array's identity so that swapping
        the training data out invalidates the cache.
        """
        X = self.training_data.X
        cached = getattr(self, "_data_output_cache", None)
        if cached is None or cached[0] is not X:
            self._data_output_cache = (X, self.fom.compute_output(X))
        return self._data_output_cache[1]

    def _initial_states(self) -> Any:
        """Contiguous copy of the initial conditions ``X[:, :, 0]``, cached.

        ``X[:, :, 0]`` is a view with a stride of the whole time axis; reading
        it gathers one element per ``nt`` from the full data array, which at
        large ``N`` costs as much as the rest of a cost evaluation (1.7 s of
        3.4 s for 9 x 673800 x 200).  The data never changes, so gather once.
        """
        X = self.training_data.X
        cached = getattr(self, "_x0_cache", None)
        if cached is None or cached[0] is not X:
            self._x0_cache = (X, self.backend.ascontiguous(X[:, :, 0]))
        return self._x0_cache[1]

    def _gram_data(self) -> Any:
        """``||x_j(t)||^2`` of the training data, ``(ntraj, nt)``, cached."""
        X = self.training_data.X
        cached = getattr(self, "_gram_data_cache", None)
        if cached is None or cached[0] is not X:
            self._gram_data_cache = (X, self.backend.sum(X * X, axis=1))
        return self._gram_data_cache[1]

    def _gram_phi(self) -> tuple:
        r"""``(Phi^T X, Phi^T Phi)`` for the current :math:`\Phi`.

        ``Phi^T X`` is ``(ntraj, r, nt)``: one GEMM that reads the data once.
        It is reused for as long as :math:`\Phi` is unchanged -- checked
        exactly against a stored copy -- which is every evaluation while the
        bases are frozen.
        """
        bkend = self.backend
        X = self.training_data.X
        Phi = self.projection.Phi
        cached = getattr(self, "_gram_phi_cache", None)
        if (
            cached is not None
            and cached[0] is X
            and cached[1].shape == Phi.shape
            and bool(bkend.array_equal(cached[1], Phi))
        ):
            return cached[2], cached[3]
        PhiTX = bkend.stack([Phi.T @ X[j] for j in range(X.shape[0])], axis=0)
        PhiTPhi = Phi.T @ Phi
        self._gram_phi_cache = (X, bkend.copy(Phi), PhiTX, PhiTPhi)
        return PhiTX, PhiTPhi

    def _time_weights(self) -> Any:
        """Per-sample weights ``c`` as ``(ntraj, nt)`` (or ``None``), cached."""
        tw = getattr(self.training_data, "time_weights", None)
        if tw is None:
            return None
        cached = getattr(self, "_time_weights_cache", None)
        if cached is not None and cached[0] is tw:
            return cached[1]
        bkend = self.backend
        X = self.training_data.X
        ntraj, _, nt = X.shape
        c = bkend.asarray(tw, device=bkend.device_of(X), dtype=X.dtype)
        if c.ndim == 1:
            if c.shape[0] != nt:
                raise ValueError(
                    f"time_weights has {c.shape[0]} entries, the data {nt} snapshots"
                )
            c = bkend.stack([c] * ntraj, axis=0)
        if tuple(c.shape) != (ntraj, nt):
            raise ValueError(
                f"time_weights must be (nt,) or (ntraj, nt) = ({ntraj}, {nt})"
            )
        self._time_weights_cache = (tw, c)
        return c

    def _output_vjp(self, e: Any, Xhat: Any) -> Any:
        r"""Apply the transpose of the output Jacobian to ``e``.

        Maps ``(ntraj, no, nt) -> (ntraj, N, nt)``.  A FOM may supply
        ``apply_output_adjoint(e, q)`` to do this without ever forming the
        ``(no, N)`` matrix; that matters once the ambient dimension is large,
        and is free for the common case of an identity output map.  Without it
        we fall back to building the matrix and contracting.
        """
        apply_adjoint = getattr(self.fom, "apply_output_adjoint", None)
        if apply_adjoint is not None:
            return apply_adjoint(e, Xhat)
        C = self.fom.compute_output_derivative(Xhat)  # (no, N), constant
        return self.backend.einsum("on,bot->bnt", C, e)

    def _decode_trajectories(self, Z: Any) -> Any:
        r"""
        Decode a batch of latent trajectories to the ambient space.

        :param Z: latent trajectories of shape ``(ntraj, r, nt)``
        :returns: ambient trajectories of shape ``(ntraj, N, nt)``
        """
        bkend = self.backend
        ntraj, r, nt = Z.shape
        Z_flat = bkend.permute(Z, (0, 2, 1)).reshape(-1, r)  # (ntraj * nt, r)
        X_flat = self.projection.decode(Z_flat)  # (ntraj * nt, N)
        N = X_flat.shape[-1]
        return bkend.permute(X_flat.reshape(ntraj, nt, N), (0, 2, 1))

    def forward(self) -> Any:
        r"""
        Evaluate the NiTROM cost

        .. math::

            J = \sum_{j} \alpha_j^{-1}\sum_{i}
                \lVert y^{(j)}(t_i) - \hat{y}^{(j)}(t_i) \rVert^2,

        the weighted sum-of-squares output mismatch over trajectories and
        snapshots.

        :returns: scalar loss
        """
        bkend = self.backend
        self._sync_to_registry()

        # Encode the initial conditions to the latent space.
        z0 = self.projection.encode(self._initial_states())  # (ntraj, r)

        # Integrate the latent dynamics over the trajectory time grid: one
        # fixed-step solve on a uniform grid (or with rk45), else one per run
        # of equal intervals, n_substeps steps per interval throughout.
        Z = solve_at_measurements(self, z0)

        cost = trajectory_cost(self, Z)

        # Regularization on the quadratic tensor H.  It is a global term added
        # redundantly on every rank, so pre-dividing by the world size makes the
        # SUM all-reduce in the training loop recover it exactly once.
        if (
            self.reg > 0.0
            and hasattr(self.model, "poly_comp")
            and 2 in self.model.poly_comp
        ):
            reg = self.reg / self.world_size
            h_idx = self.model.poly_comp.index(2)
            H = self.model.inner_params()[h_idx]
            cost = cost + reg * bkend.vector_norm(H) ** 2

        return cost

    #: Cap on the retained dense forward trajectory (and stages).  Above this,
    #: ``gradient`` falls back to re-integrating each measurement interval.
    max_dense_bytes: int = 512 * 1024**2

    def _vjp_rhs(self, z: Any, lam: Any, t: float) -> list:
        """VJP of the latent RHS w.r.t. the inner model parameters (forwards forcing)."""
        if getattr(self.model, "forcing_exists", False) and self.forcing_fns:
            return self.model.inner_vjp_evaluate_rhs(
                z, lam, external_forcing=self.forcing_fns, t=t
            )
        return self.model.inner_vjp_evaluate_rhs(z, lam)

    def gradient(self) -> list:
        r"""
        Analytic gradient of the cost w.r.t. the trainable parameters, in
        registry order, by the selected adjoint method (dynamics via
        ``evaluate_adjoint_rhs``/``vjp_evaluate_rhs``, encoder via
        ``vjp_encode``, decoder via ``vjp_decode``).  Shared-parameter
        contributions are summed.

        :returns: list of gradient tensors, one per registered parameter
        """
        bkend = self.backend
        with bkend.no_grad():
            self._sync_to_registry()

            # --- forward solve at the measurement times --------------------
            X0 = self._initial_states()  # (ntraj, N), contiguous
            z0 = self.projection.encode(X0)  # (ntraj, r)

            Z, dense = solve_for_adjoint(self, z0)

            src, proj_grads = decoder_sources(self, Z)

            lam, model_grads = accumulate_adjoint(self, Z, dense, src)

            # Measurement at t_0, then encoder gradient seeded with lambda(0).
            lam = lam + src[:, :, 0]
            for k, g in enumerate(self.projection.vjp_encode(X0, lam)):
                proj_grads[k] = proj_grads[k] + g

            # Add regularization gradient on H if reg > 0.0 (see forward()).
            if (
                self.reg > 0.0
                and hasattr(self.model, "poly_comp")
                and 2 in self.model.poly_comp
            ):
                reg = self.reg / self.world_size
                h_idx = self.model.poly_comp.index(2)
                H = self.model.inner_params()[h_idx]
                model_grads[h_idx] = model_grads[h_idx] + 2.0 * reg * H

            # --- assemble in registry order, summing shared contributions --
            model_grads = self.model.project_inner_gradients(model_grads)

            grad_by_name: dict[str, Any] = {}
            for name, g in zip(self.projection.param_names, proj_grads, strict=True):
                grad_by_name[name] = g
            for name, g in zip(self.model.param_names, model_grads, strict=True):
                grad_by_name[name] = (
                    grad_by_name[name] + g if name in grad_by_name else g
                )
            grads = [grad_by_name[name] for name in self.registry.names]

        # Zero the gradient of any non-learnable parameter (base class).
        return self._apply_learnability(grads)
