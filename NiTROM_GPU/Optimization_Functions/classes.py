import numpy as np
import torch
import torch.distributed as dist
from itertools import combinations
from string import ascii_lowercase as ascii

class pool:

    def __init__(self,n_traj,fname_traj,fname_time,device=None,rank=None,world_size=None,**kwargs):
        
        """ 
        This class contains all the info regarding the pool that will be used 
        during optimization. It also loads the training data from disk.
        
        n_traj:         total number of trajectories we wish to load from disk
        fname_traj:     e.g., 'traj_%03d.npy' (string used to load each trajectory)
        fname_time:     e.g., 'time.txt' (time vector at which we save snapshots)
        
        Optional keyword arguments:
            fname_weights:          e.g., 'weight_%03d.npy' 
            fname_steady_forcing:   e.g., 'forcing_%03d.npy'
            fname_derivs:           e.g., 'fname_derivs_%03d.npy'
        """

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.rank = 0
            self.world_size = 1
        else:
            self.device = device
            if rank is None or world_size is None:
                raise ValueError("If you provide a device, you also have to provide rank and world_size")
            self.rank = rank
            self.world_size = world_size

        self.n_traj = n_traj

        # Distribute trajectories across GPUs
        self.my_n_traj = n_traj // self.world_size
        if self.rank < n_traj % self.world_size:
            self.my_n_traj += 1

        start_idx = self.rank * (n_traj // self.world_size) + min(self.rank, n_traj % self.world_size)
        self.traj_indices = list(range(start_idx, start_idx + self.my_n_traj))       
        
        # Load data from file
        self.load_trajectories(fname_traj)
        time = np.load(fname_time)
        self.time = torch.from_numpy(time).to(self.device)
        self.load_weights(kwargs)
        self.load_steady_forcing(kwargs)
        self.load_time_derivatives(kwargs)
        
        
    def load_trajectories(self,fname_traj):
        
        self.fnames_traj = [fname_traj%k for k in self.traj_indices]
        X = [np.load(self.fnames_traj[k]) for k in range(len(self.fnames_traj))]
        self.N, self.n_snapshots = X[0].shape
        X2 = np.zeros((self.my_n_traj,self.N,self.n_snapshots))
        for k in range (self.my_n_traj): X2[k,] = X[k]
        self.X = torch.from_numpy(X2).to(self.device)

    def load_weights(self,kwargs):
        
        fname_weights = kwargs.get('fname_weights',None)
        weights2 = np.ones(self.my_n_traj)
        if fname_weights != None:
            self.fnames_weights = [fname_weights%k for k in self.traj_indices]
            weights = [np.load(self.fnames_weights[k]) for k in range(len(self.fnames_weights))]
            for k in range (self.my_n_traj): weights2[k] = weights[k]
        self.weights = torch.from_numpy(weights2).to(self.device)
            
    def load_steady_forcing(self,kwargs):
        
        fname_forcing = kwargs.get('fname_steady_forcing',None)
        F = np.zeros((self.N,self.my_n_traj))
        if fname_forcing != None:
            self.fnames_forcing = [(fname_forcing)%k for k in self.traj_indices]
            for k in range (self.my_n_traj):  F[:,k] = np.load(self.fnames_forcing[k])
        self.F = torch.from_numpy(F).to(self.device)
    
    def load_time_derivatives(self,kwargs):
        
        fname_deriv = kwargs.get('fname_derivs',None)
        if fname_deriv != None:
            self.fnames_deriv = [fname_deriv%k for k in self.traj_indices]
            dX = [np.load(self.fnames_deriv[k]) for k in range(len(self.fnames_deriv))]
            dX2 = np.zeros((self.my_n_traj,self.N,self.n_snapshots))
            for k in range (self.my_n_traj): dX2[k,] = dX[k]
            self.dX = torch.from_numpy(dX2).to(self.device)

class optimization_objects:

    def __init__(self,pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp,**kwargs):

        """ 
        This class contains the training data information that will get passed to pymanopt. 
        
        pool:           an instance of the pool class
        which_trajs:    array of integers to extract a subset of the trajectories contained in 
                        pool.X. Useful if we end up using stochastic gradient descent
        which_times:    arrays of integers to extract a subset of a trajectory owned by pool. Useful
                        if we want to start training on short trajectories and then progressively extend 
                        the length of the trajectories
        leggauss_deg:   number of Gauss-Legendre quadrature points used to approximate the integrals
                        in the gradient (see Prop. 2.1 in NiTROM arXiv paper)
        nsave_rom:      number of ROM snapshots to store in between two adjacent FOM snapshots
        poly_comp:      component of the polynomial ROM (e.g., [1,2] is a quadratic model with both first and
                                                         second order components) 

        Optional keyword arguments:
            which_fix:              one of fix_bases, fix_tensors or fix_none (default is fix_none)
            stab_promoting_pen:     value of L2 regularization coefficient
            stab_promoting_tf:      value of final time for stability promoting penalty
            stab_promoting_ic:      random (unit-norm) vector to probe the stability penalty
        """
        
        self.pool = pool

        local_which_trajs = self._global_to_local_indices(which_trajs,pool)
        self.global_trajs = which_trajs
        self.local_trajs = local_which_trajs

        if len(local_which_trajs) > 0:
            self.X = pool.X[local_which_trajs,:,:]
            self.X = self.X[:,:,which_times]
            self.F = pool.F[:,local_which_trajs]
            self.weights = pool.weights[local_which_trajs]
        else:
            self.X = torch.zeros((0,pool.N,len(which_times)), device=pool.device)
            self.F = torch.zeros((pool.N,0), device=pool.device)
            self.weights = torch.zeros((0,), device=pool.device)
        
        self.time = pool.time[which_times]
        self.my_n_traj, _, self.n_snapshots = self.X.shape
        self.leggauss_deg = leggauss_deg
        self.nsave_rom = nsave_rom
        self.poly_comp = poly_comp
        self.generate_einsum_subscripts()

        # Count the total number of trajectories in this batch and
        # scale the weight accordingly so that the cost function measures
        # the average error over snapshots and trajectories.
        self.weights *= pool.n_traj*self.n_snapshots
        
        # Parse the keyword arguments
        self.which_fix = kwargs.get('which_fix','fix_none')
        if self.which_fix not in ['fix_tensors','fix_bases','fix_none']:
            raise ValueError ("which_fix must be fix_none, fix_tensors or fix_bases")
            
        self.l2_pen = kwargs.get('stab_promoting_pen',None)
        self.pen_tf = kwargs.get('stab_promoting_tf',None)
        self.randic = kwargs.get('stab_promoting_ic',None)
        
        if self.l2_pen != None and self.pen_tf == None:
            raise ValueError ("If you provide a value for stab_promoting_pen you \
                              also have to provide a value for stab_promoting_tf")
                              
        if self.l2_pen != None and self.randic == None:
            raise ValueError ("If you provide a value for stab_promoting_pen you \
                              also have to provide a random ic vector of the same \
                              size as the ROM")
                              
        if self.l2_pen != None and 1 not in self.poly_comp:
            raise ValueError ("The penalty is currently implemented for the linear term \
                              in the rom dynamics. You have no linear term.")
                              
        if self.randic != None: 
            self.randic /= torch.linalg.vector_norm(self.randic)
            self.randic = self.randic.reshape(-1)
            

    def _global_to_local_indices(self,global_indices,pool):
        if len(global_indices) == 0:
            return torch.tensor([], device=pool.device, dtype=torch.long)
        
        if not isinstance(global_indices, torch.Tensor):
            global_indices = torch.tensor(global_indices, device=pool.device, dtype=torch.long)

        gpu_indices = torch.tensor(pool.traj_indices, device=pool.device)
        mask = torch.isin(global_indices, gpu_indices)
        requested_and_owned = global_indices[mask]

        if len(requested_and_owned) == 0:
            return torch.tensor([], device=pool.device, dtype=torch.long)
        
        local_indices = []
        for global_idx in requested_and_owned:
            local_idx = (gpu_indices == global_idx).nonzero(as_tuple=True)[0][0]
            local_indices.append(local_idx)
        
        return torch.tensor(local_indices, device=pool.device, dtype=torch.long)
    
    
    def generate_einsum_subscripts(self):
        """
            Generates the indices for the einsum evaluation of the 
            right-hand side and the adjoint
        """
        ss = []
        for k in self.poly_comp:
            ssk = ascii[:k+1]
            ssk = [ssk] + [s for s in ssk[1:]]
            ss.append(ssk)
        
        self.einsum_ss = tuple(ss)
        
        

    def evaluate_rom_rhs(self,t,z,u,*operators,**kwargs):
        """
            Function that can be fed into a PyTorch integrator routine. 
            t:          time instance
            z:          state vector
            u:          a steady forcing vector
            operators:  (A2,A3,A4,...)
            
            Optional keyword arguments:
                'forcing_interp':   a PyTorch interpolator f that gives us a forcing f(t)
        """
        if torch.linalg.vector_norm(z) >= 1e6:    
            dzdt = 0.0*z 
        else:
            f = kwargs.get('forcing_interp',None)
            f = f(t) if f != None else torch.zeros(len(z), device=z.device, dtype=z.dtype)
            u = u.clone() if isinstance(u,torch.Tensor) else u(t)
            dzdt = u + f
            for (i, k) in enumerate(self.poly_comp):
                equation = ",".join(self.einsum_ss[i])
                operands = [operators[i]] + [z for _ in range(k)]
                dzdt += torch.einsum(equation,*operands)
        
        return dzdt
    

    def evaluate_rom_rhs_nonlinear(self,t,z,u,*operators,**kwargs):
        """
            Function that can be fed into a PyTorch integrator routine (nonlinear terms only). 
            t:          time instance
            z:          state vector
            u:          a steady forcing vector
            operators:  (A3,A4,...)
            
            Optional keyword arguments:
                'forcing_interp':   a PyTorch interpolator f that gives us a forcing f(t)
        """
        if torch.linalg.vector_norm(z) >= 1e6:
            dzdt = 0.0*z 
        else:
            f = kwargs.get('forcing_interp',None)
            f = f(t) if f != None else torch.zeros(len(z), device=z.device, dtype=z.dtype)
            u = u.clone() if isinstance(u,torch.Tensor) else u(t)
            dzdt = u + f
            for (i, k) in enumerate(self.poly_comp[1:], start=1):
                equation = ",".join(self.einsum_ss[i])
                operands = [operators[i]] + [z for _ in range(k)]
                dzdt += torch.einsum(equation,*operands)
        
        return dzdt
    
    
    def evaluate_rom_adjoint(self,t,z,fq,*operators):
        """
            Function that can be fed into a PyTorch integrator routine. 
            t:          time instance
            z:          state vector
            fq:         PyTorch interpolator to evaluate the
                        base flow at time t
            operators:  (A2,A3,A4,...)
        """
        
        if torch.linalg.vector_norm(z) >= 1e6:
            dzdt = 0.0*z
        else:
            J = torch.zeros((len(z),len(z)),device=z.device,dtype=z.dtype)
            for (i, k) in enumerate(self.poly_comp):
                
                combs = list(combinations(self.einsum_ss[i][1:],r=k-1))
                operands = [operators[i]] + [fq(t) for _ in range(k-1)]
                for comb in combs:
                    equation = [self.einsum_ss[i][0]] + list(comb)
                    equation = ",".join(equation)
                    
                    J += torch.einsum(equation,*operands)
                    
            dzdt = J.T@z
            
        return dzdt
    

    def evaluate_rom_adjoint_nonlinear(self,t,z,fq,*operators):
        """
            Function that can be fed into a PyTorch integrator routine (nonlinear terms only). 
            t:          time instance
            z:          state vector
            fq:         PyTorch interpolator to evaluate the
                        base flow at time t
            operators:  (A3,A4,...)
        """

        if torch.linalg.vector_norm(z) >= 1e6:
            dzdt = 0.0*z
        else:
            J = torch.zeros((len(z),len(z)),device=z.device,dtype=z.dtype)
            for (i, k) in enumerate(self.poly_comp[1:], start=1):
                
                combs = list(combinations(self.einsum_ss[i][1:],r=k-1))
                operands = [operators[i]] + [fq(t).to(z.dtype) for _ in range(k-1)]
                for comb in combs:
                    equation = [self.einsum_ss[i][0]] + list(comb)
                    equation = ",".join(equation)
                    
                    J += torch.einsum(equation,*operands)
                    
            dzdt = J.T@z
            
        return dzdt