from typing import Any
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


def solve_opinf(module: OpInfModule) -> OpInfModule:
    """
    Analytically solve the Operator Inference (OpInf) weighted least-squares problem.

    Fits the polynomial ROM parameters to the projected trajectories using the
    exact closed-form regularized least-squares solution, taking into account
    any frozen (unlearnable) parameters and scaling weights.

    The optimal parameters S = [A_d1_flat, A_d2_flat, ..., B] satisfy:
    S (Q W Q.T + reg * I) = tilde_dZ W Q.T

    Updates the module's parameters in-place and syncs them to the ROM.

    :param module: the OpInfModule to solve
    :returns: the updated OpInfModule (with optimized parameters)
    """
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

    # 5. Formulate and solve the least-squares problem: S A = B_sys
    # A = Q_w @ Q.T + reg * I, where Q_w = Q @ W
    # B_sys = tilde_dZ @ Q_w.T
    w = module.w  # shape (N_s,)
    Q_w = Q * w[None, :]
    Q_w_Q_T = Q_w @ Q.T
    B_sys = tilde_dZ @ Q_w.T

    # Allreduce local contributions to get the global matrices in parallel runs
    if bkend.is_numpy:
        from nitrom.backend import mpi_allreduce_sum, mpi_rank_size
        _, size = mpi_rank_size()
        if size > 1:
            Q_w_Q_T = mpi_allreduce_sum(Q_w_Q_T)
            B_sys = mpi_allreduce_sum(B_sys)
    else:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(Q_w_Q_T, op=dist.ReduceOp.SUM)
                dist.all_reduce(B_sys, op=dist.ReduceOp.SUM)
        except ImportError:
            pass

    import numpy as np
    reg_diag_np = np.zeros(Q_dim, dtype=float)
    offset = 0
    for name in learnable_params:
        if name == "B":
            num_cols = U_flat.shape[0]
        else:
            d = int(name.split("_")[1])
            num_cols = r**d
            if d == 2:
                reg_diag_np[offset : offset + num_cols] = module.reg
        offset += num_cols

    reg_diag_tensor = bkend.asarray(reg_diag_np, dtype=dtype, device=device)
    reg_matrix = bkend.diag(reg_diag_tensor)
    A = Q_w_Q_T + reg_matrix

    # Solve via eigh: A = V @ diag(val) @ V.T -> A_inv = (V * val_inv[None, :]) @ V.T
    val, vec = bkend.eigh(A)
    max_val = bkend.max(bkend.abs(val))
    tol = 1e-12 * max_val
    val_inv = bkend.where(val > tol, 1.0 / val, 0.0)
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
