import numpy as np
from scipy.integrate import solve_ivp
from mpi4py import MPI
import fom_class

from NiTROM.Optimization_Functions import classes, opinf_functions as opinf_fun

#%% Instantiate the full-order model class

n = 3
beta = 20
A2 = np.diag([-1,-2,-5])
A3 = np.zeros((3,3,3))
A3[:,:,-1] = np.diag([beta,beta,0.0])
B = np.ones((3,1))
C = np.ones((1,3))
time = np.linspace(0,10,num=100)

fom = fom_class.full_order_model(A2,A3,B,C)

#%% Generate training trajectories and save to file

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_forcing = traj_path + "forcing_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

n_traj = 4

max_val = 5/20

# betas = np.asarray([0.01,0.05,0.1,0.12,0.15,0.2,0.21,0.248])
betas = np.asarray([0.01,0.1,0.2,0.248])
n_traj = len(betas)
weights = np.zeros(len(betas))
for k in range (len(betas)):
    u = betas[k]*np.ones(3)
    
    sol = solve_ivp(fom.evaluate_fom_dynamics,[0,time[-1]],np.zeros(3),'RK45',t_eval=time,args=(u,))
    
    dX = np.zeros((3,len(time)))
    for j in range (sol.y.shape[-1]):
        dX[:,j] = fom.evaluate_fom_dynamics(time[j],sol.y[:,j],u) - u
    
    id_ss = np.asarray([-betas[k]/(-1+beta/5*betas[k]),-betas[k]/(-2+beta/5*betas[k]),betas[k]/5])
    weights[k] = np.linalg.norm(fom.compute_output(id_ss))**2
    
    np.save(fname_traj%k,sol.y)
    np.save(fname_deriv%k,dX)
    np.save(fname_weight%k,[weights[k]])
    np.save(fname_forcing%k,u)
    
np.save(traj_path + "time.npy",time)

pool_inputs = (MPI.COMM_WORLD, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_steady_forcing':fname_forcing,'fname_weights':fname_weight,'fname_derivs':fname_deriv}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)

r = 2               # ROM dimension

Phi_pod, _ = opinf_fun.perform_POD(pool,r)
Psi_pod = Phi_pod.copy()
tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(Phi_pod,Psi_pod)
np.save(traj_path + "Phi_pod.npy",Phi_pod)
np.save(traj_path + "Psi_pod.npy",Psi_pod)
np.save(traj_path + "A2.npy",tensors_pod[0])
np.save(traj_path + "A3.npy",tensors_pod[1])