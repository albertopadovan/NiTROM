"""
Lid-driven cavity at Re = 8300 (SIADS 2024, section 5), wrapping the solver in the
parent folder (classes_cavity, time_steppers, numba_operators; not modified):
second-order finite volumes on a 100 x 100 fully staggered grid, second-order
fractional-step (projection) time integration with dt = 1/400.

The state q is the perturbation about the steady base flow; the actuator is
B = exp(-5000((x - xc)^2 + (y - yc)^2)) in the x-momentum equation at (0.95, 0.95),
projected onto divergence-free fields and scaled to unit norm (eq. (5.3)).
(The solver's y axis is flipped: yc = 0.05 in solver coordinates is y = 0.95.)
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import classes_cavity as classes          # noqa: E402
import numba_operators as numbaops        # noqa: E402

Re, NX, NY, NSTEP = 8300, 100, 100, 400
BC = [0, 0, 1, 0, 0, 0, 0, 0]             # lid velocity u = 1 (ul, ur, ub, ut, vl, vr, vb, vt)
BFLOW = os.path.join(HERE, "data", f"bflow_Re{Re}_Nx{NX}_Ny{NY}.npy")


class Cavity:

    def __init__(self, dt=1.0/NSTEP):
        self.dt = dt
        self.flow = classes.flow_class(1, 1, NX, NY, Re)
        self.lops = classes.linear_operators_2D(self.flow, dt)
        self.flow.q_sbf = np.load(BFLOW)
        self.fom = classes.fom_class(self.flow, self.lops)
        self.fom.assemble_forcing_profile(0.95, 0.05)
        self.B = self.fom.f.copy()
        self.N = self.B.size

    def actuator(self, x, y):
        """Unit-norm divergence-free Gaussian exp(-5000((x' - x)^2 + (y' - y)^2)) in the
        x-momentum equation centred at the physical point (x, y): projected onto
        divergence-free fields first, then normalized (../classes_cavity.assemble_forcing_profile)."""
        self.fom.assemble_forcing_profile(x, 1.0 - y)       # solver y axis is flipped
        b = self.fom.f.copy()
        self.fom.assemble_forcing_profile(0.95, 0.05)       # restore the default actuator
        return b

    def divergence(self, q):
        """Relative discrete divergence dx ||D q|| / ||q|| of a perturbation q
        (zero boundary perturbation: no boundary contribution)."""
        return self.flow.dx*np.linalg.norm(self.lops.D @ q)/np.linalg.norm(q)

    def integrate(self, q0, T, nsave, forcing=None):
        """Advance the perturbation q0 (about the base flow) to time T with the
        scheme of time_steppers.solver_2D (AB2 on explicit terms, projection),
        saving every nsave steps. forcing(t) is a scalar w(t) multiplying B, treated
        like the volume forcing of solver_2D (explicitly, at t_{k-1}).
        Returns (tsave, perturbation snapshots)."""
        flow, lops = self.flow, self.lops
        nsteps = int(round(T/self.dt))
        q = flow.q_sbf + q0
        data = np.zeros((self.N, nsteps//nsave + 1))
        data[:, 0] = q
        qkm1 = np.zeros(self.N)
        for k in range(1, nsteps + 1):
            qbc = flow.populate_boundary_conditions(q, *BC)
            qlap_bc = numbaops.laplacian_boundary_conditions_2D(flow.Re, flow.x, flow.y, q, qbc)
            qdiv_bc = numbaops.divergence_boundary_conditions_2D(flow.Re, flow.x, flow.y, qbc)
            qrhs = qlap_bc - numbaops.evaluate_bilinearity_2D(flow.x, flow.y, q, q, qbc, qbc) \
                + lops.L.dot(q)
            if forcing is not None:
                qrhs = qrhs + forcing((k - 1)*self.dt)*self.B
            qfwd = 1.5*qrhs - 0.5*qkm1          # as in solver_2D (k starts at 1)
            qkm1 = qrhs
            qs = lops.LL.dot(lops.LR.dot(q) + qfwd)
            q = qs - lops.G.dot(lops.luP.solve(lops.D.dot(qs) - qdiv_bc))
            if k % nsave == 0:
                data[:, k//nsave] = q
                if not np.isfinite(q).all():
                    raise ValueError("solver blew up")
        return self.dt*nsave*np.arange(data.shape[1]), data - flow.q_sbf[:, None]
