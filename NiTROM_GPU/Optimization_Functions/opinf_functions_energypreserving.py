import torch
import torch.distributed as dist

def opinf_ep(pool, phi, lambdas):
    # Assemble and preallocate tensors for OpInf EP
    m_local = pool.n_snapshots*pool.my_n_traj
    r = phi.shape[-1]
    Z = torch.zeros((r, m_local), device=phi.device, dtype=phi.dtype)
    dZ = torch.zeros((r, m_local), device=phi.device, dtype=phi.dtype)
    w = torch.zeros(m_local, device=phi.device, dtype=phi.dtype)
    for i in range(pool.my_n_traj):
        Z[:, i*pool.n_snapshots:(i+1)*pool.n_snapshots] = phi.T @ pool.X[i, :, :]
        dZ[:, i*pool.n_snapshots:(i+1)*pool.n_snapshots] = phi.T @ pool.dX[i, :, :]
        w[i*pool.n_snapshots:(i+1)*pool.n_snapshots] = 1./(pool.weights[i]*pool.n_traj)
    A = torch.zeros((r, r), device=phi.device, dtype=phi.dtype)
    H = torch.zeros((r, r*r), device=phi.device, dtype=phi.dtype)

    vkronf = torch.empty((r*r, m_local), device=phi.device, dtype=phi.dtype)
    for k in range(m_local):
        z = Z[:, k]
        vkronf[:, k] = torch.kron(z, z)

    # Iterate through rows to compute parts of A and H
    for i in range(r):
        F = dZ - H @ vkronf

        # Build vkron of remaining pairs
        rows = []
        for k in range(m_local):
            z = Z[:, k]
            rows.append(torch.concatenate([z[j] * z[i+1:r] for j in range(r)]))
        vkron = torch.column_stack(rows)
        qv = vkron.shape[0]
        D_local = torch.vstack([Z, vkron])
        sqrt_w = torch.sqrt(w).unsqueeze(0)
        D_w = D_local * sqrt_w
        f_w = F[i, :] * sqrt_w.squeeze(0)
        lhs_local = D_w @ D_w.T
        rhs_local = D_w @ f_w
        
        if pool.world_size > 1:
            dist.all_reduce(lhs_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(rhs_local, op=dist.ReduceOp.SUM)

        reg = torch.eye(r + qv, device=phi.device, dtype=phi.dtype)
        reg[:r, :r] *= lambdas[0]
        reg[r:, r:] *= lambdas[1]
        lhs_local += reg
        L = torch.linalg.cholesky(lhs_local)
        G = torch.cholesky_solve(rhs_local.unsqueeze(-1), L).squeeze(-1)

        A[i, :] = G[:r]
        offset = r
        # Fill inferred part of H from quadratic part of G
        for j in range(r):
            zstart = r * j
            zcount = j * (r - i - 1)
            cnt = r - i - 1
            if cnt > 0:
                H[i, (i+1) + zstart : (i+1) + zstart + cnt] = G[offset + zcount: offset + zcount + cnt]
        # Enforce skew-symmetry
        for j in range(r):
            zstart = r * j
            H[i:r, zstart + i] = -H[i, zstart + i : zstart + r]
    
    return A, H.reshape(r, r, r)