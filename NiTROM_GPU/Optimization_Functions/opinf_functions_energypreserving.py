import torch

def opinf_ep(pool, phi, lambdas):
    # Assemble and preallocate tensors for OpInf EP
    m = pool.n_snapshots*pool.n_traj
    r = phi.shape[-1]
    Z = torch.zeros((r, m), device=phi.device, dtype=phi.dtype)
    dZ = torch.zeros((r, m), device=phi.device, dtype=phi.dtype)
    W = torch.zeros(m, device=phi.device, dtype=phi.dtype)
    for i in range(pool.n_traj):
        Z[:, i*pool.n_snapshots:(i+1)*pool.n_snapshots] = phi.T @ pool.X[i, :, :]
        dZ[:, i*pool.n_snapshots:(i+1)*pool.n_snapshots] = phi.T @ pool.dX[i, :, :]
        W[i*pool.n_snapshots:(i+1)*pool.n_snapshots] = 1./pool.weights[i]*pool.n_traj
    W = torch.diag(W)
    A = torch.zeros((r, r), device=phi.device, dtype=phi.dtype)
    H = torch.zeros((r, r*r), device=phi.device, dtype=phi.dtype)

    vkronf = torch.empty((r*r, m), device=phi.device, dtype=phi.dtype)
    for k in range(m):
        z = Z[:, k]
        vkronf[:, k] = torch.kron(z, z)

    # Iterate through rows to compute parts of A and H
    for i in range(r):
        F = dZ - H @ vkronf

        # Build vkron of remaining pairs
        rows = []
        for k in range(m):
            z = Z[:, k]
            rows.append(torch.concatenate([z[j] * z[i+1:r] for j in range(r)]))
        vkron = torch.column_stack(rows)
        qv = vkron.shape[0]
        D = torch.vstack([Z, vkron])
        lhs = D @ W @ D.T
        rhs = D @ W @ F[i, :]
        reg = torch.eye(r + qv, device=phi.device, dtype=phi.dtype)
        reg[:r, :r] *= lambdas[0]
        reg[r:, r:] *= lambdas[1]

        G = torch.linalg.solve(lhs + reg, rhs)
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