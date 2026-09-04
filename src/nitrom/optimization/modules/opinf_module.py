from typing import Any

from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.model import regularized_tensor_index
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.projections.projection import Projection
from nitrom.training_data import TrainingData
from nitrom.backend import distributed_rank_size


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

        # Weight matrix
        W = bkend.repeat_interleave(1 / training_data.weights.reshape(-1), nt)
        self.W = bkend.diag(W)

        # Store forcing callables and time grid
        self.forcing_fns = getattr(training_data, "forcing_fns", None)
        self.time = getattr(training_data, "time", None)

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
        """Push current parameters into the underlying ROM."""
        tensors = [getattr(self, name) for name in self.rom.param_names]
        self.rom.update_params(tensors)

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
            Z_flat = bkend.permute(self.Z, (0, 2, 1)).reshape(-1, r)
            fZ = self.rom.evaluate_rhs(0.0, Z_flat)
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

        fZ = self._evaluate_rhs_all()

        R = self.dZ - fZ
        R_flat = bkend.permute(R, (1, 0, 2)).reshape(self.rom.state_dimension, -1)
        cost = ((R_flat @ self.W) * R_flat).sum()

        # Regularization on the quadratic tensor H
        _, world_size = distributed_rank_size()
        reg = self.reg / world_size
        if hasattr(self.rom, "poly_comp"):
            rom = self.rom.model if hasattr(self.rom, "model") else self.rom
            idx = regularized_tensor_index(rom.poly_comp)
            if idx is not None:
                H = rom.get_params()[idx]
                cost = cost + reg * bkend.vector_norm(H) ** 2

        return cost

    def gradient(self) -> list:
        r"""
        Compute analytic gradients of the cost w.r.t. the trainable
        parameters using the model's VJP.

        :returns: list of gradient tensors, one per parameter
        """
        bkend = self.backend
        self._sync_to_rom()
        r = self.rom.state_dimension

        # Compute residual
        fZ = self._evaluate_rhs_all()

        R = self.dZ - fZ
        R_flat = bkend.permute(R, (1, 0, 2)).reshape(r, -1)
        RW = bkend.permute(
            (R_flat @ self.W).reshape(r, self.ntraj, self.nt), (1, 0, 2)
        )

        # VJP: adjoint seed v = -2 * R * W
        if self.forcing_fns is None or len(self.forcing_fns) == 0:
            Z_flat = bkend.permute(self.Z, (0, 2, 1)).reshape(-1, r)
            v = -2.0 * bkend.permute(RW, (0, 2, 1)).reshape(-1, r)
            from nitrom.backend import distributed_rank_size
            _, world_size = distributed_rank_size()
            reg = self.reg / world_size
            grads = self.rom.inner_vjp_evaluate_rhs(Z_flat, v, reg=reg)
        else:
            # Accumulate VJP over time snapshots
            grads = None
            from nitrom.backend import distributed_rank_size
            _, world_size = distributed_rank_size()
            reg_val = self.reg / world_size
            for j in range(self.nt):
                t_j = self.time[j]
                z_j = self.Z[:, :, j]           # (ntraj, r)
                v_j = -2.0 * RW[:, :, j]        # (ntraj, r)
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
