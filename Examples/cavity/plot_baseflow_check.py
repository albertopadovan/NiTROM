"""
Plot the Re = 8300 cavity steady state for inspection.

Reproduces Figure 7(a) of Padovan, Vollmer & Bodony (SIADS 2024) (the vorticity
field) alongside the two velocity components, as a check on the base flow
produced by gen_baseflow.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import classes_cavity as classes
import post_process as pp

Lx = Ly = 1
Nx = Ny = 100
Re = 8300

flow = classes.flow_class(Lx, Ly, Nx, Ny, Re)
q = np.load("bflow_Re%d_Nx%d_Ny%d.npy" % (Re, Nx, Ny))
X, Y, fields = pp.output_fields(flow, q)

print(f"state dim {q.size},  ||q_sbf|| = {np.linalg.norm(q):.6e}")
for name, f in zip(("u", "v", "vorticity"), fields):
    print(f"  {name:10s} min {f.min():+.4f}  max {f.max():+.4f}")

fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
for ax, (name, x, y, f) in zip(
    axes, zip((r"$u$", r"$v$", r"vorticity $\omega_z$"), X, Y, fields)
):
    lim = float(np.max(np.abs(f)))
    cs = ax.contourf(x, y, np.flipud(f), levels=100, cmap="bwr",
                     vmin=-lim, vmax=lim)
    ax.set_aspect("equal")
    ax.set_title(name)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    fig.colorbar(cs, ax=ax, shrink=0.85)
fig.suptitle(rf"Lid-driven cavity steady state, $Re = {Re}$, "
             rf"${Nx}\times{Ny}$ grid")
out = "figures/cavity_baseflow.png"
import os
os.makedirs("figures", exist_ok=True)
fig.savefig(out, dpi=150)
print(f"saved -> {out}")
