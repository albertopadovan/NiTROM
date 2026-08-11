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


nsave = 100
time = dt*np.arange(0,n*40,1)
tsave = time[::nsave]

amps = np.random.uniform(-1, 1, size=25)
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

traj_path = "./trajectories/"
Phi_pre = np.load(traj_path + "phi_pre.npy")

fname_traj = traj_path + "traj_%03d_testing.npy"
fname_weight = traj_path + "weight_%03d_testing.npy"
fname_deriv = traj_path + "deriv_%03d_testing.npy"


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
    
np.save(traj_path + "amps_testing.npy",amps)