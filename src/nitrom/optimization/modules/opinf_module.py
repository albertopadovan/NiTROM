from typing import Any

from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.projections.projection import Projection
from nitrom.training_data import TrainingData

from .base import InferenceModule


class OpInfModule(InferenceModule):
    r"""
    Operator-inference module backed by :class:`PolynomialModel` or
    :class:`GasPolynomialModel`.

    Minimizes the weighted least-squares cost

    .. math::

        J = \sum_{i} w_i \lVert \dot{z}_i - f(t_i, z_i) \rVert^2
            + \lambda \sum_k \lVert A_k \rVert^2

    where :math:`z = \Phi^\top x`, :math:`\dot{z} = \Phi^\top \dot{x}`,
    and :math:`f` is evaluated by the underlying model.

    :param training_data: training data with ``X``, ``dX``, ``weights``, ``time``
    :type training_data: TrainingData
    :param latent_space_model: a pre-constructed latent-space dynamics model
        (e.g. :class:`PolynomialModel` or :class:`GasPolynomialModel`) whose
        parameters are optimized to fit the projected data.  Any fixed input
        operator (e.g. ``B = Phi^T B_fom``) and any initial guess should already
        be baked into the model (via its ``forcing_config`` / constructor).
    :type latent_space_model: PolynomialModel or GasPolynomialModel
    :param projection: a :class:`Projection` mapping the ambient state to the
        latent space; the training data are projected with
        :meth:`Projection.encode`
    :type projection: Projection
    :param reg: Tikhonov regularization weight
    :type reg: float
    """

    def __init__(
        self,
        training_data: TrainingData,
        latent_space_model: PolynomialModel | GasPolynomialModel,
        projection: Projection,
        reg: float = 0.0,
    ) -> None:
        super().__init__()

        self.reg = reg
        self.rom = latent_space_model
        self.projection = projection
        self.backend = latent_space_model.backend
        bkend = self.backend

        # Precompute projected data: Z, dZ of shape (ntraj, r, nt)
        self.Z = self._encode_trajectories(training_data.X)
        self.dZ = self._encode_trajectories(training_data.dX)

        ntraj, _, nt = self.Z.shape
        self.ntraj = ntraj
        self.nt = nt

        # Keep the snapshot axis leading in the hot cost/gradient path.  These
        # arrays do not depend on the ROM parameters, so repeatedly permuting
        # the trajectory representation in every objective evaluation is
        # unnecessary.
        r = self.rom.state_dimension
        self.Z_flat = bkend.permute(self.Z, (0, 2, 1)).reshape(-1, r)
        self.dZ_flat = bkend.permute(self.dZ, (0, 2, 1)).reshape(-1, r)

        # Store forcing callables and time grid.
        self.forcing_fns = getattr(training_data, "forcing_fns", None)
        self.time = getattr(training_data, "time", None)

        # For an unforced quadratic ROM, z \otimes z is also fixed throughout
        # OpInf.  It is built lazily on the first objective evaluation: a
        # closed-form-only OpInf solve does not need it, whereas GAS-OpInf then
        # reuses it for every RHS evaluation.  The threshold check preserves
        # PolynomialModel's blow-up-guard behaviour for unusual input data.
        poly_comp = getattr(self.rom, "poly_comp", ())
        self.Z_2_flat = None
        inner_model = getattr(self.rom, "model", self.rom)
        threshold_sq = getattr(inner_model, "_thresh_sq", None)
        self._quadratic_fastpath = (
            (self.forcing_fns is None or len(self.forcing_fns) == 0)
            and len(poly_comp) == 2
            and set(poly_comp) == {1, 2}
            and (
                threshold_sq is None
                or bool(((self.Z_flat * self.Z_flat).sum(axis=-1) < threshold_sq).all())
            )
        )

        # Per-snapshot weights.  Held as the (ntraj * nt,) diagonal rather than
        # the dense matrix: every use is a row scale of the residual, so
        # materializing the (ntraj*nt)^2 matrix costs O(n_s) times the memory
        # and turns an elementwise multiply into a GEMM against a diagonal.
        self.w = bkend.repeat_interleave(1 / training_data.weights.reshape(-1), nt)

        # The optimizer evaluates the cost and gradient at the same candidate
        # parameters.  Cache the assembly and residual for that candidate so
        # the gradient does not repeat the full batched RHS evaluation.
        self._synced_signature = None
        self._cached_signature = None
        self._cached_residual_flat = None

        # Register the model's parameters (wrapped per backend by the container)
        for name in self.rom.param_names:
            self.register_parameter(name, bkend.copy(getattr(self.rom, name)))

    def _encode_trajectories(self, A: Any) -> Any:
        r"""
        Encode a batch of ambient trajectories to the latent space.

        :meth:`Projection.encode` expects ``(N,)`` or ``(m, N)`` inputs, so the
        time axis is flattened into the batch dimension before encoding and
        restored afterwards.

        :param A: ambient trajectories of shape ``(ntraj, N, nt)``
        :returns: latent trajectories of shape ``(ntraj, r, nt)``
        """
        bkend = self.backend
        ntraj, N, nt = A.shape
        A_flat = bkend.permute(A, (0, 2, 1)).reshape(-1, N)  # (ntraj * nt, N)
        Z_flat = self.projection.encode(A_flat)  # (ntraj * nt, r)
        r = Z_flat.shape[-1]
        return bkend.permute(Z_flat.reshape(ntraj, nt, r), (0, 2, 1))

    def _sync_to_rom(self) -> None:
        """Push current parameters into the underlying ROM when they changed."""
        tensors = [getattr(self, name) for name in self.rom.param_names]
        signature = self._parameter_signature(tensors)
        if signature == self._synced_signature:
            return
        self.rom.update_params(tensors)
        self._synced_signature = signature
        self._cached_signature = None
        self._cached_residual_flat = None

    def _parameter_signature(self, tensors: list[Any]) -> tuple:
        """Return a cheap per-iterate identity/version signature.

        NumPy's optimizer replaces parameter arrays for every trial point;
        PyTorch increments a tensor's internal version counter for in-place
        optimizer updates.  Together these identify the normal training path
        without hashing the O(r^3) GAS tensors on every call.
        """
        if self.backend.is_torch:
            return tuple((id(t), t._version) for t in tensors)
        return tuple(id(t) for t in tensors)

    def _evaluate_rhs_flat(self) -> Any:
        """Evaluate all RHS values in ``(ntraj * nt, r)`` layout."""
        bkend = self.backend

        if self.forcing_fns is None or len(self.forcing_fns) == 0:
            # The common GAS-OpInf model has exactly linear and quadratic
            # physical operators.  Its already assembled tensors are exposed
            # through inner_params(), including for GasPolynomialModel.
            if self._quadratic_fastpath:
                if self.Z_2_flat is None:
                    self.Z_2_flat = (
                        self.Z_flat[:, :, None] * self.Z_flat[:, None, :]
                    ).reshape(self.Z_flat.shape[0], -1)
                tensors = self.rom.inner_params()
                i_A = self.rom.poly_comp.index(1)
                i_H = self.rom.poly_comp.index(2)
                A, H = tensors[i_A], tensors[i_H]
                return self.Z_flat @ A.T + self.Z_2_flat @ H.reshape(
                    self.rom.state_dimension, -1
                ).T
            return self.rom.evaluate_rhs(0.0, self.Z_flat)

        fZ = self._evaluate_rhs_all()
        return bkend.permute(fZ, (0, 2, 1)).reshape(-1, self.rom.state_dimension)

    def _residual_flat(self) -> Any:
        """Return the cached residual rows for the current parameter iterate."""
        if self._cached_signature == self._synced_signature:
            return self._cached_residual_flat
        residual = self.dZ_flat - self._evaluate_rhs_flat()
        self._cached_signature = self._synced_signature
        self._cached_residual_flat = residual
        return residual

    def _quadratic_inner_vjp(self, v: Any, reg: float) -> list[Any]:
        """VJP of an unforced linear--quadratic ROM using cached features."""
        r = self.rom.state_dimension
        tensors = self.rom.inner_params()
        i_A = self.rom.poly_comp.index(1)
        i_H = self.rom.poly_comp.index(2)
        grads = [None] * len(tensors)
        grads[i_A] = v.T @ self.Z_flat
        grads[i_H] = (v.T @ self.Z_2_flat).reshape(r, r, r)
        if reg > 0.0:
            grads[i_H] = grads[i_H] + 2.0 * reg * tensors[i_H]
        return grads

    def _evaluate_rhs_all(self) -> Any:
        r"""
        Evaluate the ROM RHS at every ``(traj, time)`` pair.

        When no forcing is present, all snapshots are batched in a single
        call.  When forcing callables exist, the evaluation loops over
        time snapshots so each call receives the correct ``t`` and
        ``external_forcing``.

        :returns: ``fZ`` of shape ``(ntraj, r, nt)``
        """
        bkend = self.backend
        r = self.rom.state_dimension

        if self.forcing_fns is None or len(self.forcing_fns) == 0:
            fZ = self.rom.evaluate_rhs(0.0, self.Z_flat)
            return bkend.permute(fZ.reshape(self.ntraj, self.nt, r), (0, 2, 1))

        # Loop over time snapshots to pass per-trajectory forcing
        fZ = bkend.zeros_like(self.dZ)  # (ntraj, r, nt)
        for j in range(self.nt):
            t_j = self.time[j]
            z_j = self.Z[:, :, j]  # (ntraj, r)
            fZ[:, :, j] = self.rom.evaluate_rhs(
                t_j, z_j, external_forcing=self.forcing_fns,
            )
        return fZ

    def forward(self) -> Any:
        r"""
        Evaluate the weighted least-squares cost.

        :returns: scalar loss
        """
        bkend = self.backend
        self._sync_to_rom()
        R_flat = self._residual_flat()
        cost = ((R_flat * self.w[:, None]) * R_flat).sum()

        # Regularization on the quadratic tensor H
        world_size = self.world_size  # communicator the pool was sharded on
        reg = self.reg / world_size
        if hasattr(self.rom, "poly_comp") and 2 in self.rom.poly_comp:
            if hasattr(self.rom, "model"):
                idx = self.rom.model.poly_comp.index(2)
                H = self.rom.model.get_params()[idx]
            else:
                idx = self.rom.poly_comp.index(2)
                H = self.rom.get_params()[idx]
            cost = cost + reg * bkend.vector_norm(H) ** 2

        return cost

    def gradient(self) -> list:
        r"""
        Compute analytic gradients of the cost w.r.t. the trainable
        parameters using the model's VJP.

        :returns: list of gradient tensors, one per parameter
        """
        self._sync_to_rom()
        r = self.rom.state_dimension

        R_flat = self._residual_flat()

        # VJP: adjoint seed v = -2 * R * W
        if self.forcing_fns is None or len(self.forcing_fns) == 0:
            v = -2.0 * R_flat * self.w[:, None]
            world_size = self.world_size  # communicator the pool was sharded on
            reg = self.reg / world_size
            if self._quadratic_fastpath:
                grads = self._quadratic_inner_vjp(v, reg)
            else:
                grads = self.rom.inner_vjp_evaluate_rhs(self.Z_flat, v, reg=reg)
        else:
            # Accumulate VJP over time snapshots
            grads = None
            world_size = self.world_size  # communicator the pool was sharded on
            reg_val = self.reg / world_size
            V = (-2.0 * R_flat * self.w[:, None]).reshape(self.ntraj, self.nt, r)
            for j in range(self.nt):
                t_j = self.time[j]
                z_j = self.Z[:, :, j]           # (ntraj, r)
                v_j = V[:, j, :]                 # (ntraj, r)
                reg_j = reg_val if j == 0 else 0.0
                grads_j = self.rom.inner_vjp_evaluate_rhs(
                    z_j, v_j,
                    reg=reg_j,
                    external_forcing=self.forcing_fns, t=t_j,
                )
                if grads is None:
                    grads = grads_j
                else:
                    for i in range(len(grads)):
                        grads[i] = grads[i] + grads_j[i]

        grads = self.rom.project_inner_gradients(grads)

        # Zero the gradient of any non-learnable parameter (base class).
        return self._apply_learnability(grads)
