import matplotlib.pyplot as plt
import numpy as np 
import time_steppers as tstep
import post_process as pp
import classes_cavity as classes

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": True,
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # "xtick.top": True,
        # "ytick.right": True,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "legend.handlelength": 2.8,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

FIG_WIDTH = 3.4
FIG_WIDTH_WIDE = 6.8
FIG_HEIGHT = 2.6

def make_figure(*, wide=False, nrows=1, ncols=1, height=None):
    width = FIG_WIDTH_WIDE if wide else FIG_WIDTH
    fig_height = height if height is not None else FIG_HEIGHT
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, fig_height), constrained_layout=True)


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

bflow = np.load("bflow_Re%d_Nx%d_Ny%d.npy"%(Re,Nx,Ny))

ii = -2
X, Y, fields = pp.output_fields(flow,bflow.real)

# color_map = plt.get_cmap('bwr')

idx = 2
vmin = np.min(fields[idx]) 
vmax = -vmin

fig, ax = make_figure()
plt.contourf(X[idx],Y[idx],np.flipud(fields[idx]),levels=100,cmap='RdBu_r',vmin=vmin,vmax=vmax)
ax = plt.gca()
ax.set_aspect('equal')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
# ax.set_xticks([0,0.25,0.5,0.75,1.0])
# ax.set_yticks([0,0.25,0.5,0.75,1.0])

plt.savefig("figures/cavity_baseflow.eps", format='eps')
plt.savefig("figures/cavity_baseflow.png")
# plt.show()

u = fields[0]
v = fields[1]

print(np.max(fields[idx]),np.min(fields[idx]))