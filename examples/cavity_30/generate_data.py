import matplotlib.pyplot as plt
import numpy as np 
import time_steppers as tstep
import post_process as pp
import classes_cavity as classes
import time as tlib

import scipy.linalg as sciplin

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams["legend.edgecolor"] = 'black'
plt.rcParams["legend.fontsize"] = 14
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2


Lx = 1
Ly = 1
Nx = 100
Ny = 100

dx = Lx/Nx
dy = Ly/Ny
Re = 8300

flow = classes.flow_class(Lx,Ly,Nx,Ny,Re)

n = 400
dt = 1.0/n

lops = classes.linear_operators_2D(flow,dt)
flow.q_sbf = np.load("bflow_Re%d_Nx%d_Ny%d.npy"%(Re,Nx,Ny))
fom = classes.fom_class(flow,lops)
fom.assemble_forcing_profile(0.95, 0.05)
B = fom.f.copy()


X, Y, fields = pp.output_fields(flow,B)

color_map = plt.cm.get_cmap('bwr')

idx = 1
vmin = np.min(fields[idx]) 
vmax = -vmin

plt.figure()
plt.contourf(X[idx],Y[idx],np.flipud(fields[idx]),levels=100,cmap=color_map,vmin=vmin,vmax=vmax)
ax = plt.gca()
ax.set_aspect('equal')
plt.colorbar()

u = fields[0]
v = fields[1]

print(np.max(fields[idx]),np.min(fields[idx]))


nsave = 100
time = dt*np.arange(0,n*40,1)
tsave = time[::nsave]

amps = [-1.0, -0.25, -0.05, 0.01, 0.05, 0.25, 1.0]
bc_coefs = [0,0,1,0,0,0,0,0]

Q = np.zeros((flow.szu + flow.szv,len(amps)*len(tsave)))
energy = np.zeros((len(amps),len(tsave)))

for k in range (len(amps)):
    
    t0 = tlib.time()
    print("Generating trajectory %d/%d"%(k+1,len(amps)))
    qic = flow.q_sbf + amps[k]*B
    data, _ = tstep.solver_2D(flow,lops,qic,time,nsave,bc_coefs)
    data -= flow.q_sbf.reshape(-1,1)
    
    Q[:,k*len(tsave):(k+1)*len(tsave)] = data 
    energy[k,] = np.linalg.norm(data,axis=0)**2
    t1 = tlib.time() - t0
    print("Execution time = %1.3f [min]"%(t1/60))
    

plt.figure()
for k in range (len(amps)):
    plt.plot(tsave,energy[k,],'k')


U, S, _ = sciplin.svd(Q,full_matrices=False)
Slo = 1 - np.cumsum(S**2)/np.sum(S**2)

plt.figure()
plt.plot(Slo,'o')

ax = plt.gca()
ax.set_yscale('log')


Phi_pre = U[:,:200]

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"


for k in range (len(amps)):
    
    print("Saving trajectory %d/%d"%(k+1,len(amps)))
    
    data = Q[:,k*len(tsave):(k+1)*len(tsave)]
    ddata = np.zeros_like(data)
    for j in range (data.shape[-1]):
        ddata[:,j] = fom.evaluate_fom_dynamics(data[:,j],bc_coefs,[0,0,0,0,0,0,0,0])
        
    data = Phi_pre.T@data
    ddata = Phi_pre.T@ddata
    weight = np.mean(np.linalg.norm(data,axis=0)**2)
    
    np.save(fname_traj%k,data)
    np.save(fname_deriv%k,ddata)
    np.save(fname_weight%k,[weight])
    
np.save(fname_time,tsave)
np.save(traj_path + "amps.npy",amps)
np.save(traj_path + "phi_pre.npy",Phi_pre)
    

X, Y, fields = pp.output_fields(flow,U[:,20])

color_map = plt.cm.get_cmap('bwr')

idx = 2
vmin = np.min(fields[idx]) 
vmax = -vmin

plt.figure()
plt.contourf(X[idx],Y[idx],np.flipud(fields[idx]),levels=100,cmap=color_map,vmin=vmin,vmax=vmax)
ax = plt.gca()
ax.set_aspect('equal')
plt.colorbar()

r = 30
Phi = U[:,:r]
tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(Phi, Phi, B, bc_coefs)

vec = np.random.randn(flow.szu + flow.szv)
vec /= np.linalg.norm(vec)
vec = Phi@(Phi.T@vec)

# Method 1
rhs = Phi@(Phi.T@fom.evaluate_fom_dynamics(vec,bc_coefs,[0,0,0,0,0,0,0,0]))

# Method 2
z = Phi.T@vec 
fz = Phi@(np.einsum('ij,j',tensors_pod[0],z) + np.einsum('ijk,j,k',tensors_pod[1],z,z))

diff = fz - rhs
print(np.linalg.norm(diff))

# Method 3

fq = Phi@(Phi.T@(fom.evaluate_full_fom_dynamics(vec,bc_coefs) - fom.evaluate_full_fom_dynamics(0*vec,bc_coefs)))

diff = fq - rhs
print(np.linalg.norm(diff))
plt.show()