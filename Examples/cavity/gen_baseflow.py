"""Headless equivalent of examples/cavity/compute_baseflow.py."""
import numpy as np, time as tlib
import classes_cavity as classes, time_steppers as tstep
Lx = Ly = 1; Nx = Ny = 100; Re = 8300; n = 400; dt = 1.0/n
flow = classes.flow_class(Lx, Ly, Nx, Ny, Re)
lops = classes.linear_operators_2D(flow, dt)
q0 = np.zeros(flow.szu + flow.szv)
bc = [0, 0, 1, 0, 0, 0, 0, 0]           # ul, ur, ub, ut, vl, vr, vb, vt
t = dt*np.arange(0, n*150, 1)            # long transient -> steady state
t0 = tlib.time()
data, tsave = tstep.solver_2D(flow, lops, q0, t, 1000, bc)
print("elapsed %.1f min" % ((tlib.time()-t0)/60))
# convergence check: is the tail actually steady?
d = np.linalg.norm(data[:, -1] - data[:, -2])/np.linalg.norm(data[:, -1])
print("relative change over last saved interval: %.3e" % d)
np.save("bflow_Re%d_Nx%d_Ny%d.npy" % (Re, Nx, Ny), data[:, -1])
print("saved bflow_Re%d_Nx%d_Ny%d.npy" % (Re, Nx, Ny))
