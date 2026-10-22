from math import isfinite
from numbers import Real
from typing import Any

import numpy as np

from nitrom.optimization.modules.opinf_module import OpInfModule


def _kronecker_power(bkend: Any, Z: Any, d: int) -> Any:
    """
    Compute the d-fold Kronecker product of each column of a matrix Z of shape (r, N_s).
    Returns a matrix of shape (r**d, N_s).
    """
    if d == 0:
        return bkend.ones((1, Z.shape[1]), dtype=Z.dtype, device=bkend.device_of(Z))
    out = Z
    for _ in range(d - 1):
        temp = out[:, None, :] * Z[None, :, :]
        out = temp.reshape(-1, Z.shape[1])
    return out


def _solve_lstsq(bkend, design, target, reg_diag, comm, rcond):
    """Solve an augmented system, using local QR factors in distributed jobs."""
    nrows = design.shape[0]
    if bkend.is_numpy:
        from nitrom.distributed import mpi_bcast, mpi_gather, mpi_rank_size

        _, size = mpi_rank_size(comm)
    else:
        import torch.distributed as dist

        size = dist.get_world_size(group=comm) if dist.is_initialized() else 1

    if size > 1:
        # QR preserves the least-squares minimizer without squaring the
        # condition number. Communicate small factors rather than snapshots.
        basis, factor = bkend.qr(design)
        projected = bkend.mH(basis) @ target
        local = (bkend.to_numpy(factor), bkend.to_numpy(projected), nrows)
        if bkend.is_numpy:
            summaries = mpi_bcast(mpi_gather(local, comm=comm), comm=comm)
        else:
            summaries = [None] * size
            dist.all_gather_object(summaries, local, group=comm)
        device, dtype = bkend.device_of(design), design.dtype
        design = bkend.concatenate(
            [bkend.asarray(s[0], dtype=dtype, device=device) for s in summaries], axis=0
        )
        target = bkend.concatenate(
            [bkend.asarray(s[1], dtype=dtype, device=device) for s in summaries], axis=0
        )
        nrows = sum(s[2] for s in summaries)

    if bool((reg_diag > 0).any()):
        design = bkend.concatenate((design, bkend.diag(bkend.sqrt(reg_diag))), axis=0)
        target = bkend.concatenate(
            (
                target,
                bkend.zeros(
                    (reg_diag.shape[0], target.shape[1]),
                    dtype=target.dtype,
                    device=bkend.device_of(target),
                ),
            ),
            axis=0,
        )
        nrows += reg_diag.shape[0]

    # Explicit SVD gives the same rank policy on NumPy, Torch CPU, and CUDA,
    # including underdetermined and rank-deficient polynomial feature sets.
    U, singular, Vh = bkend.svd(design, full_matrices=False)
    if rcond is None:
        rcond = bkend.xp.finfo(singular.dtype).eps * max(nrows, design.shape[1])
    keep = singular > rcond * singular[0]
    safe = bkend.where(keep, singular, bkend.ones_like(singular))
    inverse = bkend.where(keep, 1.0 / safe, bkend.zeros_like(singular))
    return bkend.mH(Vh) @ (inverse[:, None] * (bkend.mH(U) @ target))


def solve_opinf(
    module: OpInfModule, *, solver: str = "normal", rcond: float | None = None
) -> OpInfModule:
    """
    Analytically solve the Operator Inference (OpInf) weighted least-squares problem.

    Fits the polynomial ROM parameters to the projected trajectories using the
    exact closed-form regularized least-squares solution, taking into account
    any frozen (unlearnable) parameters and scaling weights.

    The optimal parameters S = [A_d1_flat, A_d2_flat, ..., B] satisfy:
    S (Q W Q.T + Lambda) = tilde_dZ W Q.T

    Lambda applies ``module.reg[d]`` to each coefficient of degree d;
    omitted degrees and the forcing operator are unpenalized.

    Updates the module's parameters in-place and syncs them to the ROM.

    :param module: the OpInfModule to solve
    :param solver: ``"normal"`` (default) uses the existing normal-equations
        eigensolve with a relative eigenvalue cutoff of 1e-12. ``"lstsq"``
        solves the weighted, regularized augmented system directly using SVD.
        Distributed least squares combines local QR factors over the module's
        communicator without gathering all snapshots.
    :param rcond: Relative singular-value cutoff for ``solver="lstsq"``.
        Defaults to machine precision times the larger global augmented-system
        dimension. Must be finite and nonnegative; unavailable for ``"normal"``.
    :returns: the updated OpInfModule (with optimized parameters)
    """
    if solver not in ("normal", "lstsq"):
        raise ValueError("solver must be 'normal' or 'lstsq'")
    if rcond is not None:
        if solver != "lstsq":
            raise ValueError("rcond is only supported with solver='lstsq'")
        if (
            isinstance(rcond, bool)
            or not isinstance(rcond, Real)
            or not isfinite(rcond)
            or rcond < 0
        ):
            raise ValueError("rcond must be finite and nonnegative")
    bkend = module.backend
    r = module.rom.state_dimension
    device = bkend.device_of(module.Z)
    dtype = module.Z.dtype

    # 1. Flatten the trajectories: (ntraj, r, nt) -> (r, ntraj * nt)
    Z_flat = bkend.permute(module.Z, (1, 0, 2)).reshape(r, -1)
    dZ_flat = bkend.permute(module.dZ, (1, 0, 2)).reshape(r, -1)

    # 2. Reconstruct the forcing input if forcing callables exist
    U_flat = None
    if module.forcing_fns is not None and len(module.forcing_fns) > 0:
        first_u = bkend.atleast_1d(module.forcing_fns[0](module.time[0]))
        m = first_u.shape[0]
        U_flat = bkend.zeros((m, module.ntraj * module.nt), dtype=dtype, device=device)
        for i in range(module.ntraj):
            for j in range(module.nt):
                idx = i * module.nt + j
                U_flat[:, idx] = bkend.atleast_1d(module.forcing_fns[i](module.time[j]))

    # 3. Handle unlearnable (fixed) parameters by subtracting their contributions from target derivatives
    tilde_dZ = bkend.copy(dZ_flat)
    param_names = module.rom.param_names
    is_learnable = module.is_learnable

    for name in param_names:
        if not is_learnable[name]:
            val = getattr(module, name)
            if name == "B":
                if U_flat is not None:
                    tilde_dZ = tilde_dZ - val @ U_flat
            else:
                d = int(name.split("_")[1])
                Z_d = _kronecker_power(bkend, Z_flat, d)
                val_flat = val.reshape(r, -1)
                tilde_dZ = tilde_dZ - val_flat @ Z_d

    # 4. Construct the feature matrix Q for all learnable parameters
    feature_list = []
    learnable_params = []
    for name in param_names:
        if is_learnable[name]:
            learnable_params.append(name)
            if name == "B":
                if U_flat is None:
                    raise ValueError(
                        "Forcing operator B is learnable, but no forcing functions were provided in training data."
                    )
                feature_list.append(U_flat)
            else:
                d = int(name.split("_")[1])
                Z_d = _kronecker_power(bkend, Z_flat, d)
                feature_list.append(Z_d)

    if len(feature_list) == 0:
        # All parameters are unlearnable, nothing to solve!
        return module

    Q = bkend.concatenate(feature_list, axis=0)
    Q_dim = Q.shape[0]

    # 5. Build the same degree-specific tensor penalty for either algorithm.
    reg_diag_np = np.zeros(Q_dim, dtype=float)
    offset = 0
    for name in learnable_params:
        if name == "B":
            num_cols = U_flat.shape[0]
        else:
            d = int(name.split("_")[1])
            num_cols = r**d
            reg_diag_np[offset : offset + num_cols] = module.reg.get(d, 0.0)
        offset += num_cols

    reg_diag_tensor = bkend.asarray(reg_diag_np, dtype=dtype, device=device)
    comm = getattr(module, "comm", None)
    if solver == "lstsq":
        with bkend.no_grad():
            sqrt_w = bkend.sqrt(module.w)
            S_T = _solve_lstsq(
                bkend,
                bkend.mH(Q) * sqrt_w[:, None],
                bkend.mH(tilde_dZ) * sqrt_w[:, None],
                reg_diag_tensor,
                comm,
                rcond,
            )
    else:
        # Normal equations communicate only Gram matrices in parallel runs.
        Q_w = Q * module.w[None, :]
        Q_w_Q_T = Q_w @ bkend.mH(Q)
        B_sys = tilde_dZ @ bkend.mH(Q_w)
        if bkend.is_numpy:
            from nitrom.distributed import mpi_allreduce_sum, mpi_rank_size

            _, size = mpi_rank_size(comm)
            if size > 1:
                Q_w_Q_T = mpi_allreduce_sum(Q_w_Q_T, comm=comm)
                B_sys = mpi_allreduce_sum(B_sys, comm=comm)
        else:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(Q_w_Q_T, op=dist.ReduceOp.SUM, group=comm)
                dist.all_reduce(B_sys, op=dist.ReduceOp.SUM, group=comm)
        A = Q_w_Q_T + bkend.diag(reg_diag_tensor)
        val, vec = bkend.eigh(A)
        tol = 1e-12 * bkend.max(bkend.abs(val))
        keep = val > tol
        safe = bkend.where(keep, val, bkend.ones_like(val))
        val_inv = bkend.where(keep, 1.0 / safe, bkend.zeros_like(val))
        A_inv = (vec * val_inv[None, :]) @ bkend.mH(vec)
        S_T = A_inv @ bkend.mH(B_sys)
    S = bkend.mH(S_T)

    # 6. Slice and reshape S back into individual learnable parameters
    offset = 0
    for name in learnable_params:
        if name == "B":
            m = U_flat.shape[0]
            param_shape = (r, m)
            num_cols = m
        else:
            d = int(name.split("_")[1])
            param_shape = (r,) * (d + 1)
            num_cols = r**d

        S_slice = S[:, offset : offset + num_cols]
        updated_tensor = S_slice.reshape(param_shape)
        if bkend.is_torch:
            updated_tensor = updated_tensor.detach()
        module.register_parameter(name, updated_tensor)
        offset += num_cols

    # 7. Sync the updated parameters to the ROM
    module._sync_to_rom()
    return module
