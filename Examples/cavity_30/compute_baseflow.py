#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 23:43:38 2023

@author: alberto
"""


import matplotlib.pyplot as plt
import numpy as np 
import time_steppers as tstep
import post_process as pp
import classes_cavity as classes

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


## Compute baseflow
nsave = 1000
time = dt*np.arange(0,n*1500,1)
q0 = np.zeros(flow.szu+flow.szv)
bc_coefs = [0,0,1,0,0,0,0,0] # ul, ur, ub, ut, vl, vr, vb, vt
data, tsave = tstep.solver_2D(flow,lops,q0,time,nsave,bc_coefs)

idx0 = 0
idx1 = data.shape[-1]

energy = pp.compute_energy(data[:,idx0:idx1],data[:,0],tsave[idx0:idx1])

plt.figure()
plt.plot(tsave[idx0:idx1]-tsave[idx0],energy)
plt.plot(tsave[idx0:idx1]-tsave[idx0],energy,'rx')

np.save("bflow_Re%d_Nx%d_Ny%d.npy"%(Re,Nx,Ny),data[:,-1])

ii = -2
X, Y, fields = pp.output_fields(flow,data[:,-1].real)

color_map = plt.cm.get_cmap('bwr')

idx = 1
vmin = np.min(fields[idx]) 
vmax = -vmin

plt.figure()
plt.contourf(X[idx],Y[idx],np.flipud(fields[idx]),levels=100,cmap=color_map,vmin=vmin,vmax=vmax)
ax = plt.gca()
ax.set_aspect('equal')
plt.colorbar()
plt.show()

u = fields[0]
v = fields[1]

print(np.max(fields[idx]),np.min(fields[idx]))