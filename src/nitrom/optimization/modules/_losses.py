"""Trajectory losses and decoder VJPs, including the Gram formulation."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .nitrom_module import NitromModule


def trajectory_cost(module: "NitromModule", Z: Any) -> Any:
    """Weighted output mismatch, reduced over trajectories and snapshots."""
    bkend = module.backend
    # Weighted sum-of-squares output mismatch.
    if module.gram:
        SZ = (
            module.projection.S @ Z
            if module.projection.biorthogonalize == "decoder"
            else Z
        )  # (ntraj, r, nt)
        PhiTX, PhiTPhi = module._gram_phi()
        e2 = (
            module._gram_data()
            - 2.0 * bkend.sum(SZ * PhiTX, axis=1)
            + bkend.sum(SZ * (PhiTPhi @ SZ), axis=1)
        )  # (ntraj, nt)
    else:
        e = module._data_output() - module.fom.compute_output(
            module._decode_trajectories(Z)
        )
        e2 = bkend.sum(e * e, axis=1)  # (ntraj, nt)
    c = module._time_weights()
    if c is not None:
        e2 = e2 * c
    per_traj = bkend.sum(e2, axis=1) / module.weights.reshape(-1)
    cost = per_traj.sum()

    return cost


def decoder_sources(module: "NitromModule", Z: Any) -> tuple[Any, list]:
    """Return latent measurement seeds and decoder-parameter VJPs."""
    bkend = module.backend
    X = module.training_data.X
    ntraj, _, nt = X.shape
    r = module.model.state_dimension
    w = (1.0 / module.weights.reshape(-1)).reshape(-1, 1, 1)
    c = module._time_weights()
    if c is not None:
        w = w * c.reshape(ntraj, 1, nt)
    # --- output residual and adjoint sources at each snapshot ------
    if module.gram:
        # Gram form: with the seed v = -2 w (x - Phi S z),
        #   src = S^T Phi^T v = -2 w S^T (Phi^T x - Phi^T Phi S z),
        #   sum v (S z)^T = -2 [sum_j w_j X_j (S Z_j)^T
        #                       - Phi sum_j w_j (S Z_j)(S Z_j)^T].
        # For biorthogonalize="encoder", the decoder is Phi z: omit S here.
        S = (
            module.projection.S
            if module.projection.biorthogonalize == "decoder"
            else None
        )
        SZ = S @ Z if S is not None else Z  # (ntraj, r, nt)
        PhiTX, PhiTPhi = module._gram_phi()
        residual = PhiTX - PhiTPhi @ SZ
        src = -2.0 * w * (S.T @ residual if S is not None else residual)
        wSZ = w * SZ
        W_out = X[0] @ bkend.permute(wSZ[0], (1, 0))
        for j in range(1, ntraj):
            W_out = W_out + X[j] @ bkend.permute(wSZ[j], (1, 0))
        B = bkend.sum(wSZ @ bkend.permute(SZ, (0, 2, 1)), axis=0)
        W_out = -2.0 * (W_out - module.projection.Phi @ B)  # (N, r)
        proj_grads = list(module.projection.vjp_decode_from_outer(W_out))
    else:
        Xhat = module._decode_trajectories(Z)  # (ntraj, N, nt)
        e = module._data_output() - module.fom.compute_output(Xhat)
        # Weighted full-space output seed v_i = -2 alpha_j^{-1} C^T e_i.
        cw = -2.0 * w * module._output_vjp(e, Xhat)  # (ntraj, N, nt)
        N = cw.shape[1]

        # Flatten the (trajectory, snapshot) axes into a single batch.
        Z_flat = bkend.permute(Z, (0, 2, 1)).reshape(-1, r)  # (ntraj*nt, r)
        cw_flat = bkend.permute(cw, (0, 2, 1)).reshape(-1, N)  # (ntraj*nt, N)

        # Latent adjoint source: decoder Jacobian-transpose D_z^T
        # applied to the weighted output seed.
        src = bkend.permute(
            module.projection.vjp_decode_state(Z_flat, cw_flat).reshape(ntraj, nt, r),
            (0, 2, 1),
        )  # (ntraj, r, nt)

        # Decoder parameter gradient (vjp_decode sums over its batch).
        proj_grads = list(module.projection.vjp_decode(Z_flat, cw_flat))

    return src, proj_grads
