from typing import Any

from nitrom.backend import distributed_rank_size
from nitrom.latent_space_models.model import regularized_tensor_index
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry
from nitrom.training_data import TrainingData

from .base import InferenceModule


class ObliqueOpInfModule(InferenceModule):
    r"""
    Oblique operator-inference module backed by a :class:`ParamRegistry`.

    Like :class:`NitromModule`, this module receives a :class:`ParamRegistry`
    holding both the latent-space :class:`Model` and the
    :class:`LinearProjection`, mirrors the registry's parameters as trainable
    parameters, and pushes them back through :meth:`ParamRegistry.scatter`.
    Unlike :class:`NitromModule`, the cost is a one-step (derivative-matching)
    residual rather than a trajectory misfit, so no time integration is
    involved.

    Minimizes the weighted least-squares cost

    .. math::

        J = \sum_{j=1}^{N_{traj}} \frac{1}{w_j} \sum_{i=1}^{N_{snaps}}
            \bigl\lVert
                S\,\Psi^\top \dot{x}^{(j)}(t_i)
                - S\,g\bigl(\Psi^\top x^{(j)}(t_i), u(t_i)\bigr)
            \bigr\rVert^2
            + \lambda \lVert H \rVert^2 ,
        \qquad
        S = (\Psi^\top \Phi)^{-1},

    where :math:`g` is the right-hand side of the latent-space model and
    :math:`w_j` are the per-trajectory training weights.  The residual is the
    one obtained by writing the Petrov--Galerkin ROM of the oblique projector
    :math:`\Phi S \Psi^\top` in the coordinates :math:`z = \Psi^\top x`.

    Unlike :class:`OpInfModule`, which projects the data once with a fixed
    basis, :math:`\Psi` is a trainable parameter here, so the latent data
    :math:`Z = \Psi^\top X`, :math:`\dot{Z} = \Psi^\top \dot{X}` and the
    weighting :math:`S` are recomputed at every cost/gradient evaluation.

    **Usage.** The intended setup optimizes the latent dynamics and the test
    basis only, with the trial basis :math:`\Phi` held fixed::

        registry = ParamRegistry(latent_space_model, LinearProjection([Phi, Psi]))
        module = ObliqueOpInfModule(training_data, registry)
        module.set_unlearnable("Phi")            # Phi is not optimized
        module.set_manifold_types(["Psi"], ["stiefel"])

    Gradients are returned for *every* registered parameter (including
    :math:`\Phi`); freezing is applied by the base class through
    :meth:`InferenceModule.set_unlearnable`, and the manifold each parameter
    is optimized on is chosen by the caller through
    :meth:`InferenceModule.set_manifold_types`, exactly as for
    :class:`NitromModule`.

    **Data layout.** The ambient data are stored once, at construction, in the
    snapshot-major layout ``(ntraj * nt, N)``, so that every per-iteration
    contraction is a single BLAS ``matmul`` and no array is transposed or
    copied inside the optimization loop.  This costs one extra copy of ``X``
    and ``dX``; the module never touches ``training_data.X`` / ``.dX`` again
    after construction, so a memory-bound caller may release them.

    **Distributed runs.** Trajectories are sharded across ranks by
    :class:`~nitrom.training_data.TrainingPool`, so :meth:`forward` and
    :meth:`gradient` return partial sums that
    :func:`~nitrom.optimization.train` reduces; the regularization term is
    divided by the world size so the reduction reconstructs it exactly once.

    :param training_data: training data with ``X``, ``dX``, ``weights``,
        ``time`` and (optionally) ``forcing_fns``
    :type training_data: TrainingData
    :param registry: registry of the ROM parameters spanning the latent-space
        model and the (linear, oblique) projection.  Any fixed input operator
        (e.g. ``B = Psi^T B_fom``) and any initial guess should already be
        baked into the model.
    :type registry: ParamRegistry
    :param reg: Tikhonov regularization weight on the quadratic tensor
    :type reg: float
    """

    def __init__(
        self,
        training_data: TrainingData,
        registry: ParamRegistry,
        reg: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(registry.projection, LinearProjection):
            raise TypeError(
                "ObliqueOpInfModule requires a LinearProjection (the cost is "
                f"defined by Phi and Psi), got "
                f"{type(registry.projection).__name__}."
            )

        self.registry = registry
        self.training_data = training_data
        self.reg = reg

        # Convenience handles into the registry's components
        self.rom = registry.model
        self.projection = registry.projection
        self.backend = self.rom.backend
        bkend = self.backend

        ntraj, N, nt = training_data.X.shape
        self.ntraj = ntraj
        self.N = N
        self.nt = nt

        # Psi is trainable, so the latent data Z = Psi^T X and dZ = Psi^T dX
        # change at every evaluation.  Pay the layout change once, here: rows
        # are ordered (trajectory, snapshot), so row j * nt + i holds snapshot
        # i of trajectory j and the slice [i::nt] gathers snapshot i across all
        # trajectories.
        self.X = self._flatten(training_data.X)  # (ntraj * nt, N)
        self.dX = self._flatten(training_data.dX)  # (ntraj * nt, N)

        # Per-row weights 1 / w_j, matching that ordering.
        self.weights = training_data.weights
        self.w_row = bkend.repeat_interleave(
            1.0 / self.weights.reshape(-1), nt
        ).reshape(-1, 1)  # (ntraj * nt, 1)

        # Store forcing callables and time grid
        self.forcing_fns = getattr(training_data, "forcing_fns", None)
        self.time = getattr(training_data, "time", None)

        # Register one parameter per registered parameter, in registry order.
        for name in self.registry.names:
            self.register_parameter(name, bkend.copy(self.registry.value(name)))

    def _flatten(self, A: Any) -> Any:
        """Reshape ``(ntraj, N, nt)`` trajectories to ``(ntraj * nt, N)``."""
        ntraj, N, nt = A.shape
        return self.backend.permute(A, (0, 2, 1)).reshape(ntraj * nt, N)

    @property
    def param_names(self) -> list[str]:
        """Names of the trainable parameters, in registry order."""
        return self.registry.names

    def parameter_list(self) -> list:
        """Current trainable parameter tensors, in registry order."""
        return [getattr(self, name) for name in self.registry.names]

    def _sync_to_registry(self) -> None:
        """
        Push the current parameter values into the model and projection.

        Always re-scatters: :math:`S = (\\Psi^\\top \\Phi)^{-1}` is cached by
        :meth:`LinearProjection.update`, so it must be refreshed whenever the
        bases change.
        """
        self.registry.scatter(self.parameter_list())

    # ------------------------------------------------------------------
    # Forward pieces
    # ------------------------------------------------------------------
    def _evaluate_rhs_all(self, Z: Any) -> Any:
        r"""
        Evaluate the ROM RHS at every ``(traj, snapshot)`` row of ``Z``.

        Without forcing, all rows are evaluated in a single batched call.
        With forcing the evaluation loops over the ``nt`` snapshots -- each
        call needs its own ``t`` -- but every call is still batched over the
        whole trajectory dimension.

        :param Z: latent states of shape ``(ntraj * nt, r)``
        :returns: ``g(Z, u)`` of shape ``(ntraj * nt, r)``
        """
        if self.forcing_fns is None or len(self.forcing_fns) == 0:
            return self.rom.evaluate_rhs(0.0, Z)

        nt = self.nt
        fZ = self.backend.zeros_like(Z)
        for i in range(nt):
            fZ[i::nt] = self.rom.evaluate_rhs(
                self.time[i], Z[i::nt], external_forcing=self.forcing_fns,
            )
        return fZ

    def _residual(self) -> tuple:
        r"""
        Assemble the oblique residual for the current parameters.

        :returns: ``(Z, R)`` where ``Z = X Psi`` (the rows of
            :math:`\Psi^\top x`) and ``R = (dZ - g(Z, u)) S^\top`` (the rows of
            :math:`S e`), both of shape ``(ntraj * nt, r)``
        """
        Psi = self.projection.Psi
        Z = self.X @ Psi  # rows of Psi^T x
        dZ = self.dX @ Psi  # rows of Psi^T xdot
        E = dZ - self._evaluate_rhs_all(Z)
        return Z, E @ self.projection.S.T  # rows of S e

    def _jacobian_transpose(self, t: float, v: Any, z: Any, scale: float) -> Any:
        r"""
        Apply the latent RHS Jacobian-transpose,
        :math:`(\partial g/\partial z)^\top v`, at the base state ``z``.

        :meth:`Model.evaluate_adjoint_rhs` is exactly linear in its adjoint
        argument, but zeroes every row whose norm exceeds the model's
        instability threshold.  That guard exists to protect an adjoint
        *integration*; here the adjoint argument is the weighted residual, whose
        magnitude carries no such meaning, so the seed is normalized before the
        call and the result rescaled.  This keeps the gradient correct even when
        a poorly conditioned iterate makes the residual very large.

        :param t: time instance (the Jacobian may depend on it through forcing)
        :param v: adjoint seed of shape ``(m, r)``
        :param z: base state of shape ``(m, r)``
        :param scale: normalization for ``v``, computed once by the caller so
            the reduction is not repeated per snapshot
        :returns: array of shape ``(m, r)``
        """
        if scale == 0.0:
            return self.backend.zeros_like(v)
        return scale * self.rom.evaluate_adjoint_rhs(t, v / scale, z)

    def _reg(self) -> float:
        """Regularization weight, rescaled for a distributed run."""
        _, world_size = distributed_rank_size()
        return self.reg / world_size

    def _regularized_tensor(self) -> Any | None:
        """The regularized nonlinear tensor, or ``None`` if the ROM has none."""
        if not hasattr(self.rom, "poly_comp"):
            return None
        rom = self.rom.model if hasattr(self.rom, "model") else self.rom
        idx = regularized_tensor_index(rom.poly_comp)
        return None if idx is None else rom.get_params()[idx]

    # ------------------------------------------------------------------
    # InferenceModule interface
    # ------------------------------------------------------------------
    def forward(self) -> Any:
        r"""
        Evaluate the weighted least-squares cost.

        :returns: scalar loss
        """
        bkend = self.backend
        self._sync_to_registry()

        _, R = self._residual()
        cost = bkend.sum(self.w_row * R * R)

        # Regularization on the quadratic tensor H
        H = self._regularized_tensor()
        if H is not None:
            cost = cost + self._reg() * bkend.vector_norm(H) ** 2

        return cost

    def gradient(self) -> list:
        r"""
        Analytic gradient of the cost w.r.t. the trainable parameters, in
        registry order.

        With :math:`e = \Psi^\top \dot{x} - g(\Psi^\top x)`, :math:`R = S e`,
        :math:`v = S^\top R` and :math:`u = (\partial g/\partial z)^\top v`,

        .. math::

            \frac{\partial J}{\partial \theta}
                = \sum_{j,i} -\frac{2}{w_j}
                  \left(\frac{\partial g}{\partial \theta}\right)^\top v ,

        .. math::

            \frac{\partial J}{\partial \Psi}
                = \sum_{j,i} \frac{2}{w_j}
                  \bigl(\dot{x}\, v^\top - x\, u^\top - (\Phi R)\, v^\top\bigr),
            \qquad
            \frac{\partial J}{\partial \Phi}
                = \sum_{j,i} -\frac{2}{w_j} (\Psi v)\, R^\top .

        The three :math:`\Psi` terms come from :math:`\Psi^\top\dot{x}`, from
        :math:`g(\Psi^\top x)`, and from :math:`S = (\Psi^\top\Phi)^{-1}`
        respectively; :math:`\Phi` enters through :math:`S` alone.  Shared
        parameters have their contributions summed.

        The two basis terms are contracted as :math:`\Phi (R^\top V_w)` and
        :math:`\Psi (V_w^\top R)`, keeping the intermediates ``(r, r)``: the
        ambient-sized products :math:`\Phi R` and :math:`\Psi v` are never
        formed.

        :returns: list of gradient tensors, one per registered parameter
        """
        bkend = self.backend
        with bkend.no_grad():
            self._sync_to_registry()

            nt = self.nt
            S = self.projection.S
            Phi, Psi = self.projection.Phi, self.projection.Psi

            Z, R = self._residual()  # both (ntraj * nt, r)

            # v = S^T R, with the trajectory weight 1/w_j folded in.
            V = R @ S
            Vw = self.w_row * V

            # One reduction for the whole batch: the Jacobian-transpose is
            # linear in its seed, so a single scale serves every snapshot.
            scale = float(bkend.vector_norm(Vw))

            reg = self._reg()
            if self.forcing_fns is None or len(self.forcing_fns) == 0:
                # --- latent-model parameters -----------------------------
                model_grads = self.rom.inner_vjp_evaluate_rhs(
                    Z, -2.0 * Vw, reg=reg
                )
                # --- U = (dg/dz)^T v, weighted ---------------------------
                U = self._jacobian_transpose(0.0, Vw, Z, scale)
            else:
                # Forcing makes the VJP depend on the per-snapshot time, but
                # each snapshot is still batched over all trajectories.
                model_grads = None
                U = bkend.zeros_like(Vw)
                for i in range(nt):
                    t_i = self.time[i]
                    z_i, vw_i = Z[i::nt], Vw[i::nt]
                    grads_i = self.rom.inner_vjp_evaluate_rhs(
                        z_i, -2.0 * vw_i,
                        reg=reg if i == 0 else 0.0,
                        external_forcing=self.forcing_fns, t=t_i,
                    )
                    if model_grads is None:
                        model_grads = grads_i
                    else:
                        for k in range(len(model_grads)):
                            model_grads[k] = model_grads[k] + grads_i[k]
                    U[i::nt] = self._jacobian_transpose(t_i, vw_i, z_i, scale)

            model_grads = list(self.rom.project_inner_gradients(model_grads))

            # --- projection parameters -----------------------------------
            grad_Psi = 2.0 * (
                self.dX.T @ Vw - self.X.T @ U - Phi @ (R.T @ Vw)
            )
            grad_Phi = -2.0 * (Psi @ (Vw.T @ R))
            # The projection maps the ambient basis cotangents onto gradients
            # w.r.t. whatever it actually parameterizes -- (Phi, Psi) for a
            # LinearProjection, N alone for an ObliqueChartProjection.
            grad_by_name = dict(zip(
                self.projection.param_names,
                self.projection.vjp_bases(grad_Phi, grad_Psi),
                strict=True,
            ))

            # --- assemble in registry order, summing shared contributions --
            for name, g in zip(self.rom.param_names, model_grads, strict=True):
                grad_by_name[name] = (
                    grad_by_name[name] + g if name in grad_by_name else g
                )
            grads = [grad_by_name[name] for name in self.registry.names]

        # Zero the gradient of any non-learnable parameter (base class).
        return self._apply_learnability(grads)
