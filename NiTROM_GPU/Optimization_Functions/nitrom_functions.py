import numpy as np
import scipy as sp
import torch
import torch.distributed as dist
from string import ascii_lowercase as ascii
import pymanopt

import time as tlib
from ..PyTorch_Functions.integrators import my_etdrk4, etdrk4_setup
from ..PyTorch_Functions.linear_interpolation import Interp1D

def create_objective_and_gradient(manifold,opt_obj,pool,fom):
    
    """
    opt_obj:        instance of class "optimization_objects" in file "classes.py"
    pool:           instance of the class "pool" in file "classes.py"
    fom:            instance of the full-order model class
    """

    euclidean_hessian = None

    @pymanopt.function.pytorch(manifold)
    def cost(*params):

        """ 
            Evaluate the cost function 
            Phi and Psi:    bases (size N x r) that define the projection operator
            tensors:        (A2,A3,...)
        """
        Phi, Psi = params[0], params[1]
        tensors = params[2:]

        Phi = Phi.to(pool.device); Psi = Psi.to(pool.device)
        tensors = tuple(tensor.to(pool.device) for tensor in tensors)
        PhiF = Phi@torch.linalg.inv(Psi.T@Phi)

        D, V = torch.linalg.eig(tensors[0])
        V_inv = torch.linalg.inv(V)
        linop = V, D, V_inv
        internal_steps = 1
        dt = (opt_obj.time[1] - opt_obj.time[0])/internal_steps
        etdrk4_coefs = etdrk4_setup(linop, dt)

        J = 0.0
        B = opt_obj.my_n_traj
        if B > 0:
            # z0_k = Psi.T @ X[k,:,0] => z0 = X0 @ Psi  -> (B, r)
            X0 = opt_obj.X[:, :, 0]                       # (B, N)
            z0 = X0 @ Psi                                  # (B, r)

            # u_k = Psi.T @ F[:,k] => u = F.T @ Psi -> (B, r)
            u_batch = opt_obj.F.T @ Psi                  # (B, r)

            sol = my_etdrk4(etdrk4_coefs, opt_obj.evaluate_rom_rhs_nonlinear, opt_obj.time, z0, internal_steps, args=(u_batch,)+tensors)  # (B, r, T)

            # Compute outputs in batch
            # y_true = C @ X  -> (B, 1, T); y_model = C @ (PhiF @ Z) -> (B, 1, T)
            Y_true = fom.compute_output(opt_obj.X)          # (B, 1, T)
            Y_model = fom.compute_output(torch.matmul(PhiF, sol))         # (B, 1, T)

            e = Y_true - Y_model                              # (B, 1, T)
            err_per_traj = (e * e).sum(dim=(1, 2))            # (B,)
            J = J + torch.sum(err_per_traj / opt_obj.weights) # scalar tensor

        if opt_obj.l2_pen is not None and pool.rank == 0:
            time_pen = torch.linspace(0,opt_obj.pen_tf,opt_obj.n_snapshots*opt_obj.nsave_rom,device=pool.device)
            Z = my_etdrk4(etdrk4_coefs,lambda t,z: 0*z,time_pen,opt_obj.randic)
            J += opt_obj.l2_pen*torch.dot(Z[:,-1],Z[:,-1])
        
        if pool.world_size > 1:
            dist.all_reduce(J, op=dist.ReduceOp.SUM)
        
        return J.cpu()
    
    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(*params):

        """ 
            Evaluate the euclidean gradient of the cost function with respect to the parameters
            Phi and Psi:    bases (size N x r) that define the projection operator
            tensors:        (A2,A3,...)
        """

        Phi, Psi = params[0], params[1]
        tensors = params[2:]

        Phi = torch.from_numpy(Phi).to(pool.device); Psi = torch.from_numpy(Psi).to(pool.device)
        tensors = tuple(torch.from_numpy(tensor).to(pool.device) for tensor in tensors)

        D, V = torch.linalg.eig(tensors[0])
        V_inv = torch.linalg.inv(V)
        linop = V, D, V_inv
        linop_T = V_inv.T, D, V.T
        internal_steps = 1
        dt = (opt_obj.time[1] - opt_obj.time[0])/internal_steps
        dt2 = dt / (opt_obj.nsave_rom-1)
        etdrk4_coefs = etdrk4_setup(linop, dt)
        etdrk4_coefs_2 = etdrk4_setup(linop, dt2)
        etdrk4_coefs_T2 = etdrk4_setup(linop_T, dt2)
        t_unit = torch.linspace(0.0, 1.0, steps=opt_obj.nsave_rom, device=pool.device, dtype=torch.float64)
        
        # Initialize arrays to store the gradients
        n, r = Phi.shape
        grad_Phi = torch.zeros((n,r), device=pool.device, dtype=Phi.dtype)
        grad_Psi = torch.zeros((n,r), device=pool.device, dtype=Phi.dtype)
        grad_tensors = [torch.zeros_like(tensor) for tensor in tensors]
        

        # Initialize arrays needed for future computations
        lam_j_0 = torch.zeros(r, device=pool.device, dtype=Phi.dtype)
        Int_lambda = torch.zeros(r, device=pool.device, dtype=Phi.dtype)
        
        # Biorthogonalize Phi and Psi
        F = torch.linalg.inv(Psi.T@Phi)
        PhiF = Phi@F
        
        # Gauss-Legendre quadrature points and weights
        # Cubic spline interpolation to compute integral
        tlg, wlg = np.polynomial.legendre.leggauss(opt_obj.leggauss_deg)
        tlg = torch.from_numpy(tlg).to(pool.device); wlg = torch.from_numpy(wlg).to(pool.device)

        B = opt_obj.my_n_traj
        if B > 0:
            # Forward ROM integration (batched)
            X0 = opt_obj.X[:, :, 0]                            # (B, N)
            z0 = X0 @ Psi                                      # (B, r)
            u_batch = opt_obj.F.T @ Psi                        # (B, r)
            sol = my_etdrk4(etdrk4_coefs, opt_obj.evaluate_rom_rhs_nonlinear,
                            opt_obj.time, z0, internal_steps, args=(u_batch,)+tensors)   # (B, r, T)
            Z = sol                                            # (B, r, T)

            # Outputs and errors (batched)
            # Y_true = C @ X, Y_model = C @ (PhiF @ Z)
            Cb = fom.C.to(device=pool.device, dtype=Phi.dtype).expand(B, -1, -1)  # (B, 1, N)
            Y_true = torch.bmm(Cb, opt_obj.X.to(Phi.dtype))                        # (B, 1, T)
            X_z = torch.einsum('nr,brt->bnt', PhiF, Z)                             # (B, N, T)
            Y_model = torch.bmm(Cb, X_z)                                           # (B, 1, T)
            e = Y_true - Y_model                                                   # (B, 1, T)

            # Cte, PCte, PsiPCte, C_minus, FZ (batched)
            CTb = fom.compute_output_derivative(X_z).T.to(Phi.dtype).expand(B, -1, -1)  # (B, N, 1)
            Cte = torch.bmm(CTb, e).squeeze(2)                                         # (B, N, T)
            PCte = torch.einsum('rn,bnT->brT', PhiF.T, Cte)                            # (B, r, T)
            PsiPCte = torch.einsum('nr,brT->bnT', Psi, PCte)                           # (B, N, T)
            C_minus = Cte - PsiPCte                                                    # (B, N, T)
            FZ = torch.einsum('rs,brT->bsT', F, Z)                                     # (B, r, T)

            # Accumulate grad_Psi and grad_Phi across batch with weights 2/alpha_k
            w = (2.0 / opt_obj.weights).view(B, 1, 1).to(Phi.dtype)                    # (B,1,1)
            grad_Psi.add_(torch.einsum('bnT,brT->nr', X_z * w, PCte))                  # (N, r)
            grad_Phi.add_(-torch.einsum('bnT,brT->nr', C_minus * w, FZ))               # (N, r)

            # Initialize batched adjoint state and integral
            lam_j_0 = torch.zeros((B, r), device=pool.device, dtype=Phi.dtype)
            Int_lambda = torch.zeros((B, r), device=pool.device, dtype=Phi.dtype)

            # Adjoint integration (not batched)
            for j in range(opt_obj.n_snapshots - 1):
                id1 = opt_obj.n_snapshots - 1 - j
                id0 = id1 - 1
                tf_j = opt_obj.time[id1]
                t0_j = opt_obj.time[id0]
                z0_j = Z[:, :, id0]                                                 # (B, r)

                delta = tf_j - t0_j
                time_rom_j = t0_j + t_unit * delta                                  # (nsave_rom,)
                if torch.abs(time_rom_j[-1] - tf_j) >= 1e-6:
                    print(time_rom_j[-1], tf_j)
                    raise ValueError("Error in euclidean_gradient() - final time is not correct!")

                # Forward ROM over [t0_j, tf_j] (batched)
                sol_j = my_etdrk4(etdrk4_coefs_2, opt_obj.evaluate_rom_rhs_nonlinear,
                                   time_rom_j, z0_j, internal_steps, args=(u_batch,)+tensors)  # (B, r, nsave_rom)
                Z_j = torch.fliplr(sol_j)                                           # (B, r, nsave_rom)

                # Update initial adjoint condition: lam_j_0 += (2/alpha)*PCtej
                PCtej = PCte[:, :, id1]                                             # (B, r)
                lam_j_0 = lam_j_0 + (2.0 / opt_obj.weights).to(Phi.dtype).view(B, 1) * PCtej

                a = (tf_j - t0_j) / 2
                b = (tf_j + t0_j) / 2
                time_j_lg = a * tlg + b

                # Batched adjoint integration using batched interpolant fZ
                fZ = Interp1D(time_rom_j, Z_j, extrapolate=True)                     # Z_j: (B, r, nsave_rom)
                sol_lam = my_etdrk4(etdrk4_coefs_T2, opt_obj.evaluate_rom_adjoint_nonlinear,
                                    time_rom_j, lam_j_0, internal_steps, args=(fZ,)+tensors)  # (B, r, nsave_rom)
                Lam = torch.fliplr(sol_lam)                                          # (B, r, nsave_rom)
                lam_j_0 = Lam[:, :, 0]                                               # (B, r)

                # Interpolants for quadrature (batched)
                fL = Interp1D(time_rom_j, Lam, extrapolate=True)
                Z_j_lg = fZ(time_j_lg)                                              # (B, r, leggauss_deg)
                Lam_lg = fL(time_j_lg)                                              # (B, r, leggauss_deg)

                # Accumulate Int_lambda across Gauss-Legendre points
                Int_lambda += a * torch.einsum('q,brq->br', wlg, Lam_lg)             # (B, r)

                # Accumulate grad_tensors across batch and quadrature
                for (count, p) in enumerate(opt_obj.poly_comp):
                    # Sum over quadrature points with weights, then reduce batch
                    acc = torch.zeros_like(grad_tensors[count])
                    for i_lg in range(opt_obj.leggauss_deg):
                        # Build batched einsum: operands are (B, r)
                        eq_parts = [f"...{ch}" for ch in ascii[:p+1]]                # ['...a','...b', ...]
                        equation = ",".join(eq_parts)                                 # e.g., '...a,...b,...c'
                        operands = [Lam_lg[..., i_lg]] + [Z_j_lg[..., i_lg] for _ in range(p)]
                        tmp = torch.einsum(equation, *operands)                       # (B, r, r, ..., r)
                        acc -= a * wlg[i_lg] * tmp.sum(dim=0)                         # sum over batch -> (r, r, ..., r)
                    grad_tensors[count].add_(acc)

        if opt_obj.l2_pen is not None and pool.rank == 0:
            idx = opt_obj.poly_comp.index(1)    # index of the linear tensor

            time_pen = torch.linspace(0,opt_obj.pen_tf,opt_obj.n_snapshots*opt_obj.nsave_rom,device=pool.device)
            Z = my_etdrk4(etdrk4_coefs,lambda t,z: 0*z,time_pen,opt_obj.randic)
            Mu = my_etdrk4(etdrk4_coefs,lambda t,z: 0*z,time_pen,-2*opt_obj.l2_pen*Z[:,-1])
            Mu = torch.fliplr(Mu)
            
            for k in range (opt_obj.n_snapshots - 1):
                
                k0, k1 = k*opt_obj.nsave_rom, (k+1)*opt_obj.nsave_rom
                fZ = Interp1D(time_pen[k0:k1],Z[:,k0:k1],extrapolate=True)
                fMu = Interp1D(time_pen[k0:k1],Mu[:,k0:k1],extrapolate=True)
            
                a = (time_pen[k1-1] - time_pen[k0])/2
                b = (time_pen[k1-1] + time_pen[k0])/2
                time_k_lg = a*tlg + b
                
                Zk = fZ(time_k_lg)
                Muk = fMu(time_k_lg)
                
                for i in range (opt_obj.leggauss_deg):
                    grad_tensors[idx] += -a*wlg[i]*torch.einsum('i,j',Muk[:,i],Zk[:,i])

        if pool.world_size > 1:
            grad_Phi = grad_Phi.contiguous()
            grad_Psi = grad_Psi.contiguous()
            dist.all_reduce(grad_Phi, op=dist.ReduceOp.SUM)
            dist.all_reduce(grad_Psi, op=dist.ReduceOp.SUM)
            for k in range (len(grad_tensors)):
                grad_tensors[k] = grad_tensors[k].contiguous()
                dist.all_reduce(grad_tensors[k], op=dist.ReduceOp.SUM)

        if opt_obj.which_fix == 'fix_bases':

            grad_Phi *= 0.0; grad_Psi *= 0.0

        elif opt_obj.which_fix == 'fix_tensors':    
            
            for k in range (len(grad_tensors)): grad_tensors[k] *= 0.0

        grad_Phi = grad_Phi.cpu().numpy()
        grad_Psi = grad_Psi.cpu().numpy()
        grad_tensors = tuple(tensor.cpu().numpy() for tensor in grad_tensors)

        return grad_Phi, grad_Psi, *grad_tensors
    

    return cost, euclidean_gradient, euclidean_hessian

