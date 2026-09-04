from typing import Any

import numpy as np

from nitrom.latent_space_models.model import regularized_tensor_index
from nitrom.roms.param_registry import ParamRegistry
from nitrom.training_data import TrainingData

from ...time_steppers.time_stepper import solve_ivp, solve_adjoint_ivp_discrete
from ...utils import interp_quadratic
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
        forcing callables, weights)
    :type training_data: TrainingData
    :param registry: registry of the ROM parameters spanning the
        latent-space model and the projection
    :type registry: ParamRegistry
    :param fom: optional full-order model, e.g. for output evaluation
    :param reg: Tikhonov regularization weight
    :type reg: float
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
    ) -> None:
        super().__init__()

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
        if not self._check_sync():
            self.registry.scatter(self.parameter_list())

    def _check_sync(self) -> bool:
        """Check if the current parameters are synchronized with the registry."""
        try:
            for name, value in zip(self.registry.names, self.parameter_list()):
                if not self.backend.allclose(getattr(self.registry, name), value):
                    return False
            return True
        except AttributeError:
            return False

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
        z0 = self.projection.encode(self.training_data.X[:, :, 0])  # (ntraj, r)

        # Integrate the latent dynamics over the trajectory time grid.
        dt = (self.time[1] - self.time[0]) / self.n_substeps
        Z = solve_ivp(
            self.model.evaluate_rhs,
            z0,
            self.time[0],
            self.time[-1],
            dt,
            self.time,
            self.time_stepper,
            atol=self.atol,
            rtol=self.rtol,
            external_forcing=self.forcing_fns or None,
        )  # (ntraj, r, nt)

        # Weighted sum-of-squares output mismatch.
        e = self.fom.compute_output(self.training_data.X) - self.fom.compute_output(
            self._decode_trajectories(Z)
        )
        per_traj = bkend.sum(e * e, axis=(1, 2)) / self.weights.reshape(-1)
        cost = per_traj.sum()

        # Regularization on the quadratic tensor H
        from nitrom.backend import distributed_rank_size
        _, world_size = distributed_rank_size()
        reg = self.reg / world_size
        if reg > 0.0 and hasattr(self.model, "poly_comp"):
            h_idx = regularized_tensor_index(self.model.poly_comp)
            if h_idx is not None:
                H = self.model.inner_params()[h_idx]
                cost = cost + reg * bkend.vector_norm(H) ** 2

        return cost

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
        registry order, by the continuous adjoint method (dynamics via
        ``evaluate_adjoint_rhs``/``vjp_evaluate_rhs``, encoder via
        ``vjp_encode``, decoder via ``vjp_decode``).  Shared-parameter
        contributions are summed.

        :returns: list of gradient tensors, one per registered parameter
        """
        bkend = self.backend
        with bkend.no_grad():
            self._sync_to_registry()

            X = self.training_data.X  # (ntraj, N, nt)
            time = self.time  # (nt,)
            dev = bkend.device_of(X)
            dtype = X.dtype
            ntraj, _, nt = X.shape
            r = self.model.state_dimension
            ef = self.forcing_fns or None
            w = (1.0 / self.weights.reshape(-1)).reshape(-1, 1, 1)  # (ntraj, 1, 1)

            # --- forward solve at the measurement times --------------------
            z0 = self.projection.encode(X[:, :, 0])  # (ntraj, r)
            dt = (time[1] - time[0]) / self.n_substeps
            Z = solve_ivp(
                self.model.evaluate_rhs, z0, time[0], time[-1], dt, time,
                self.time_stepper, atol=self.atol, rtol=self.rtol,
                external_forcing=ef,
            )  # (ntraj, r, nt)

            # --- output residual and adjoint sources at each snapshot ------
            Xhat = self._decode_trajectories(Z)  # (ntraj, N, nt)
            e = self.fom.compute_output(X) - self.fom.compute_output(Xhat)
            C = self.fom.compute_output_derivative(Xhat)  # (no, N), constant
            # Weighted full-space output seed v_i = -2 alpha_j^{-1} C^T e_i.
            cw = -2.0 * w * bkend.einsum("on,bot->bnt", C, e)  # (ntraj, N, nt)
            N = cw.shape[1]

            # Flatten the (trajectory, snapshot) axes into a single batch.
            Z_flat = bkend.permute(Z, (0, 2, 1)).reshape(-1, r)  # (ntraj*nt, r)
            cw_flat = bkend.permute(cw, (0, 2, 1)).reshape(-1, N)  # (ntraj*nt, N)

            # Latent adjoint source: decoder Jacobian-transpose D_z^T applied to
            # the weighted output seed.
            src = bkend.permute(
                self.projection.vjp_decode_state(Z_flat, cw_flat).reshape(
                    ntraj, nt, r
                ),
                (0, 2, 1),
            )  # (ntraj, r, nt)

            # Decoder parameter gradient (vjp_decode sums over its batch).
            proj_grads = list(self.projection.vjp_decode(Z_flat, cw_flat))

            # --- backward adjoint sweep --------------------------------------
            model_grads = [bkend.zeros_like(p) for p in self.model.inner_params()]
            lam = bkend.zeros((ntraj, r), device=dev, dtype=dtype)

            if self.adjoint_method == "discrete":
                for k in range(nt - 1, 0, -1):
                    # Inject the measurement source at snapshot k.
                    lam = lam + src[:, :, k]

                    # Re-integrate the base flow over [t_{k-1}, t_k].
                    t0i, tfi = float(time[k - 1]), float(time[k])
                    delta = tfi - t0i
                    h = delta / self.n_substeps
                    sub_t = bkend.linspace(
                        t0i, tfi, self.n_substeps + 1, device=dev, dtype=dtype
                    )
                    Zint = solve_ivp(
                        self.model.evaluate_rhs, Z[:, :, k - 1], t0i, tfi,
                        h, sub_t, self.time_stepper,
                        atol=self.atol, rtol=self.rtol,
                        external_forcing=ef,
                    )  # (ntraj, r, n_substeps + 1)
                    
                    # Propagate adjoint and accumulate model grads
                    lam, step_grads = solve_adjoint_ivp_discrete(
                        self.model.evaluate_rhs,
                        self.model.evaluate_adjoint_rhs,
                        self._vjp_rhs,
                        Zint, sub_t, h, lam,
                        method=self.time_stepper,
                        external_forcing=ef,
                    )
                    
                    if step_grads is not None:
                        for idx, g in enumerate(step_grads):
                            model_grads[idx] = model_grads[idx] + g

            else:  # continuous adjoint
                xi, wq = self._gl_nodes, self._gl_weights

                for k in range(nt - 1, 0, -1):
                    # Inject the measurement source at snapshot k.
                    lam = lam + src[:, :, k]

                    # Re-integrate the base flow over [t_{k-1}, t_k].
                    t0i, tfi = float(time[k - 1]), float(time[k])
                    delta = tfi - t0i
                    a = 0.5 * delta
                    sub_t = bkend.linspace(
                        t0i, tfi, self.n_substeps + 1, device=dev, dtype=dtype
                    )
                    Zint = solve_ivp(
                        self.model.evaluate_rhs, Z[:, :, k - 1], t0i, tfi,
                        delta / self.n_substeps, sub_t, self.time_stepper,
                        atol=self.atol, rtol=self.rtol,
                        external_forcing=ef,
                    )  # (ntraj, r, n_substeps + 1)

                    # Adjoint in reversed time tau in [0, delta] (physical time
                    # t = tfi - tau): d(lam)/d(tau) = J_f(Z(t))^T lam.
                    def adj_rhs(tau, lam_, _Zint=Zint, _sub_t=sub_t, _tfi=tfi):
                        phys_t = _tfi - tau
                        tq = bkend.atleast_1d(
                            bkend.asarray(phys_t, device=dev, dtype=dtype)
                        )
                        Z_t = interp_quadratic(tq, _sub_t, _Zint)[..., 0]  # (ntraj, r)
                        return self.model.evaluate_adjoint_rhs(float(phys_t), lam_, Z_t)

                    # Integrate the adjoint onto the Gauss-Legendre nodes (reversed
                    # time), plus the interval end for the carry-forward.
                    tau_nodes = a * (1.0 - xi)  # (n_leggauss,)
                    order = bkend.argsort(tau_nodes)
                    tau_eval = bkend.concatenate(
                        [tau_nodes[order], bkend.asarray([delta], device=dev, dtype=dtype)]
                    )
                    Lam_sol = solve_ivp(
                        adj_rhs, lam, 0.0, delta, delta / self.n_substeps, tau_eval,
                        self.time_stepper, atol=self.atol, rtol=self.rtol,
                    )  # (ntraj, r, n_leggauss + 1)

                    # Adjoint at the GL nodes (undo the sort); carry lambda(t_{k-1}).
                    Lam_nodes = bkend.empty(
                        (ntraj, r, self.n_leggauss), device=dev, dtype=dtype
                    )
                    Lam_nodes[..., order] = Lam_sol[..., : self.n_leggauss]
                    lam = Lam_sol[..., -1]

                    # Base flow at the physical GL nodes.
                    t_gl = 0.5 * (tfi + t0i) + a * xi  # (n_leggauss,)
                    Z_nodes = interp_quadratic(t_gl, sub_t, Zint)  # (ntraj, r, n_leggauss)

                    # Gauss-Legendre quadrature of the model-parameter integral.
                    # Fold the quadrature weight a * w_i into the adjoint seed.
                    Lam_w = Lam_nodes * (a * wq).reshape(1, 1, -1)
                    if getattr(self.model, "forcing_exists", False) and self.forcing_fns:
                        # Forcing makes the VJP depend on the per-node time.
                        for i in range(self.n_leggauss):
                            g = self._vjp_rhs(
                                Z_nodes[..., i], Lam_w[..., i], float(t_gl[i])
                            )
                            for idx in range(len(model_grads)):
                                model_grads[idx] = model_grads[idx] + g[idx]
                    else:
                        # Flatten (trajectory, node) and reduce in one VJP call.
                        Zf = bkend.permute(Z_nodes, (0, 2, 1)).reshape(-1, r)
                        Lf = bkend.permute(Lam_w, (0, 2, 1)).reshape(-1, r)
                        for idx, g in enumerate(self.model.inner_vjp_evaluate_rhs(Zf, Lf)):
                            model_grads[idx] = model_grads[idx] + g

            # Measurement at t_0, then encoder gradient seeded with lambda(0).
            lam = lam + src[:, :, 0]
            for k, g in enumerate(self.projection.vjp_encode(X[:, :, 0], lam)):
                proj_grads[k] = proj_grads[k] + g

            # Add regularization gradient on H if reg > 0.0
            from nitrom.backend import distributed_rank_size
            _, world_size = distributed_rank_size()
            reg = self.reg / world_size
            if reg > 0.0 and hasattr(self.model, "poly_comp"):
                h_idx = regularized_tensor_index(self.model.poly_comp)
                if h_idx is not None:
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
