"""
Validation of the cavity solver (cavity.py) at Re = 8300.

1. cavity.Cavity.integrate reproduces the original ../time_steppers.solver_2D.
2. The base flow (compute_baseflow.py) is steady.
3. Divergence: the actuator B, the unprojected Gaussian (control) and impulse-response
   snapshots; X0 may only contain divergence-free vectors.
4. Energy of the beta = +-1 impulse responses (cf. Fig. 7(b) of SIADS 2024: starts at 1,
   decays, then peaks around t ~ 5 at ~0.55 before decaying).
5. Time-step convergence (dt = 1/400 vs 1/800).
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cavity import Cavity, BC, NSTEP, HERE

sys.path.insert(0, os.path.dirname(HERE))
import time_steppers as tstep             # noqa: E402

DIV_TOL = 1e-10


def rel(a, b):
    return np.linalg.norm(a - b)/np.linalg.norm(b)


def main():
    cav = Cavity()
    flow, dt = cav.flow, cav.dt
    ok = True

    # 1. Original solver
    T, nsave = 2.0, 40
    time = dt*np.arange(int(round(T/dt)) + 1)
    ref, _ = tstep.solver_2D(flow, cav.lops, flow.q_sbf + cav.B, time, nsave, BC)
    _, mine = cav.integrate(cav.B, T, nsave)
    e = rel(mine, ref - flow.q_sbf[:, None])
    ok &= e < 1e-12
    print(f"1. Cavity.integrate vs ../time_steppers.solver_2D (beta = 1, t in [0, {T:g}]): rel. diff = {e:.1e}")

    # 2. Base flow
    res = np.linalg.norm(cav.fom.evaluate_full_fom_dynamics(0*flow.q_sbf, BC))/np.linalg.norm(flow.q_sbf)
    hist = np.load(os.path.join(HERE, "data", "bflow_history.npy"))
    ok &= res < 1e-6
    print(f"2. base flow: ||F(q_sbf)|| / ||q_sbf|| = {res:.1e}; relative change per unit time "
          f"at t = 1500: {hist[1, -1]:.1e}")

    # 3. Divergence
    raw = np.zeros(cav.N)
    for i in range(flow.rowsu):
        for j in range(flow.colsu):
            raw[i*flow.colsu + j] = np.exp(-5000*((flow.x[j] + flow.dx/2 - 0.95)**2 + (flow.y[i] - 0.05)**2))
    tsave, Q = cav.integrate(cav.B, 10.0, NSTEP//10)
    d_snap = max(cav.divergence(Q[:, k]) for k in range(1, Q.shape[1]))
    print(f"3. divergence dx||Dq||/||q||: B (projected actuator, = IC/beta) {cav.divergence(cav.B):.1e}; "
          f"unprojected Gaussian {cav.divergence(raw):.1e}; impulse-response snapshots (max) {d_snap:.1e}")
    ok &= cav.divergence(raw) > 1e-3 and d_snap < DIV_TOL

    # 4. Impulse-response energy (cf. Fig. 7(b))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for beta in [1.0, -1.0, 0.25, -0.25]:
        t, Q = cav.integrate(beta*cav.B, 40.0, NSTEP//10)
        E = np.linalg.norm(Q, axis=0)**2
        k = np.argmax(E*(t > 1.0))
        print(f"4. beta = {beta:+5.2f}: energy starts at {E[0]:.3f}, min {E[t < 3].min():.3f}, "
              f"peak {E[k]:.3f} at t = {t[k]:.1f}, energy at t = 40: {E[-1]:.1e}")
        ax.plot(t, E, "k", lw=1)
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\|q\|^2$")
    ax.set_title(r"Impulse responses, $\beta = \pm 1, \pm 0.25$ (cf. SIADS 2024, Fig. 7(b))", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "validation_energy.png"), dpi=150)

    # 5. Time step
    _, Q1 = cav.integrate(cav.B, 10.0, NSTEP//10)
    cav2 = Cavity(dt=dt/2)
    _, Q2 = cav2.integrate(cav.B, 10.0, 2*NSTEP//10)
    print(f"5. dt = 1/400 vs 1/800, beta = 1, t in [0, 10]: rel. diff = {rel(Q1, Q2):.1e}")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()
