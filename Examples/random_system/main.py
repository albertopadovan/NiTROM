import numpy as np 
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import time

import pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers

from NiTROM_GPU.Optimization_Functions import classes as classes_batch, nitrom_functions as nitrom_functions_batch
from NiTROM_GPU_unbatched.Optimization_Functions import classes, nitrom_functions
from NiTROM_GPU_unbatched.PyTorch_Functions import gpu_utils
import fom_class

plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
plt.rc('text.latex',preamble=r'\usepackage{amsmath}')
torch.set_printoptions(precision=8)

cPOD, cOI, cTR, cOPT = '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'
lPOD, lOI, lTR, lOPT = 'solid', 'dotted', 'dashed', 'dashdot'

device, rank, world_size = gpu_utils.setup_distributed_gpus()
if rank == 0:
    print(f"Using {world_size} GPU(s) for distributed training.")
    verb = 2
else:
    verb = 0

n = 2000
n_traj = 30
C = torch.eye(n, device=device, dtype=torch.float64)
fom = fom_class.full_order_model(C)

# torch.backends.opt_einsum.is_available()

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_forcing = traj_path + "forcing_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

pool_inputs = (n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_steady_forcing':fname_forcing,
               'fname_weights':fname_weight,
               'fname_derivs':fname_deriv,
               'device':device,
               'rank':rank,
               'world_size':world_size
}
pool = classes.pool(*pool_inputs,**pool_kwargs)
pool_batch = classes_batch.pool(*pool_inputs,**pool_kwargs)

r = 250               # ROM dimension
poly_comp = [1,2]   # Model with a linear part and a quadratic part
print(f"{n_traj} trajectories, r={r}")

#%% Compute NiTROM model 

which_trajs = torch.arange(0,pool.n_traj,1,device=device)
which_times = torch.arange(0,pool.n_snapshots,1,device=device)
leggauss_deg = 5
nsave_rom = 2

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)
opt_obj_batch = classes_batch.optimization_objects(*opt_obj_inputs)

St = manifolds.Stiefel(n,r)
Gr = manifolds.Grassmann(n,r)
Euc_rr = manifolds.Euclidean(r,r)
Euc_rrr = manifolds.Euclidean(r,r,r)

M = manifolds.Product([Gr,St,Euc_rr,Euc_rrr])
cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
cost_batch, grad_batch, hess_batch = nitrom_functions_batch.create_objective_and_gradient(M,opt_obj_batch,pool_batch,fom)
# problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)

# line_searcher = myGPUAdaptiveLineSearcher(contraction_factor=0.5,sufficient_decrease=0.85,max_iterations=25,initial_step_size=1)
# optimizer = optimizers.ConjugateGradient(max_iterations=50,min_step_size=1e-20,max_time=3600,line_searcher=line_searcher,log_verbosity=1,verbosity=verb)

point = [None]*4
if rank == 0:
    point[0] = np.load(traj_path + "Phi.npy")
    point[1] = np.load(traj_path + "Psi.npy")
    point[2] = np.load(traj_path + "A2.npy")
    point[3] = np.load(traj_path + "A3.npy")
if world_size > 1: dist.broadcast_object_list(point,src=0)
point = tuple(point)
# point = tuple(torch.tensor(p,device=device) for p in point)

# result = optimizer.run(problem,initial_point=point)
# t1 = time.time()
# cost_val = cost_batch(*point)
# t2 = time.time()
# print(cost_val)
# print(t2 - t1)
# print()
# t1 = time.time()
# cost_val = cost(*point)
# t2 = time.time()
# print(cost_val)
# print(t2 - t1)

t1 = time.time()
grad_vals = grad_batch(*point)
t2 = time.time()
t1 = time.time()
grad_vals = grad_batch(*point)
t2 = time.time()
tb = t2 - t1
print("Batched", tb)

# print()
t1 = time.time()
grad_vals_unbatched = grad(*point)
t2 = time.time()
tu = t2 - t1
# t1 = time.time()
# grad_vals = grad(*point)
# t2 = time.time()
print("Not batched", tu)

print("Speedup = %1.5f"%(tu/tb))


for i in range(len(grad_vals)):
    g = grad_vals[i]
    gu = grad_vals_unbatched[i]
    error = np.linalg.norm(g - gu) / np.linalg.norm(gu) * 100
    print("Percent error = %1.15e"%error)

if world_size > 1: torch.distributed.barrier()
gpu_utils.cleanup_distributed()