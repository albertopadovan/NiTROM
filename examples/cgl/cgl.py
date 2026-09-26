"""
Complex Ginzburg-Landau (CGL) solver, following the setup of
Padovan, Vollmer & Bodony, SIADS 2024 (section 4):

    dq/dt = (-nu d/dx + gamma d^2/dx^2 + mu) q - a |q|^2 q,

with nu = 2 + 0.4i, gamma = 1 - i, mu = (mu0 - cu^2) + mu2 x^2/2, cu = Im(nu)/2,
mu0 = 0.38, mu2 = -0.01, a = 0.1.

Discretization (same as in the paper's code):
  - uniform grid of nx = 256 points on [-L/2, L/2], L = 100
  - 4th-order central finite differences in the interior, 2nd-order at the
    points adjacent to the boundaries (homogeneous Dirichlet)
  - real-valued state q = [Re(q); Im(q)] in R^{2 nx}
  - time integration: Crank-Nicolson for the linear part, 2nd-order
    Adams-Bashforth for the cubic nonlinearity (forward Euler on the first step)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class CGL:

    def __init__(self, nx=256, L=100.0, nu=2.0 + 0.4j, gamma=1.0 - 1.0j,
                 mu0=0.38, mu2=-0.01, a=0.1, s=1.6):

        self.nx = nx
        self.L = L
        self.x = np.linspace(-L/2, L/2, num=nx, endpoint=True)
        self.dx = self.x[1] - self.x[0]

        self.nu = nu
        self.gamma = gamma
        self.mu0 = mu0
        self.mu2 = mu2
        self.a = a
        self.s = s
        self.cu = nu.imag/2
        self.mu = (mu0 - self.cu**2) + mu2*self.x**2/2

        # Branch I of the disturbance-amplification region (xbar < 0)
        self.xbar = -np.sqrt(-2*(mu0 - self.cu**2)/mu2)

        self.D = self._first_derivative()
        self.DD = self._second_derivative()
        self.A = self._linear_operator()

    def _first_derivative(self):
        nx, dx = self.nx, self.dx
        f, s = 1./(12*dx), 1./(2*dx)
        rows, cols, data = [], [], []
        for i in range(2, nx - 2):
            rows += [i]*4
            cols += [i-2, i-1, i+1, i+2]
            data += [f, -8*f, 8*f, -f]
        i = 1
        rows += [i]*3; cols += [i-1, i+1, i+2]; data += [-8*f, 8*f, -f]
        i = nx - 2
        rows += [i]*3; cols += [i-2, i-1, i+1]; data += [f, -8*f, 8*f]
        rows += [0]; cols += [1]; data += [s]
        rows += [nx-1]; cols += [nx-2]; data += [-s]
        return sp.csc_array((data, (rows, cols)), shape=(nx, nx))

    def _second_derivative(self):
        nx, dx = self.nx, self.dx
        f, s = 1./(12*dx**2), 1./dx**2
        rows, cols, data = [], [], []
        for i in range(2, nx - 2):
            rows += [i]*5
            cols += [i-2, i-1, i, i+1, i+2]
            data += [-f, 16*f, -30*f, 16*f, -f]
        i = 1
        rows += [i]*4; cols += [i-1, i, i+1, i+2]; data += [16*f, -30*f, 16*f, -f]
        i = nx - 2
        rows += [i]*4; cols += [i-2, i-1, i, i+1]; data += [-f, 16*f, -30*f, 16*f]
        rows += [0]*2; cols += [0, 1]; data += [-2*s, s]
        rows += [nx-1]*2; cols += [nx-2, nx-1]; data += [s, -2*s]
        return sp.csc_array((data, (rows, cols)), shape=(nx, nx))

    def _linear_operator(self):
        muI = sp.diags(self.mu, offsets=0, format='csc')
        A11 = -self.nu.real*self.D + self.gamma.real*self.DD + muI
        A12 = self.nu.imag*self.D - self.gamma.imag*self.DD
        return sp.bmat([[A11, A12], [-A12, A11]], format='csc')

    def nonlinearity(self, q):
        nx = self.nx
        q2 = q[:nx]**2 + q[nx:]**2
        return -self.a*np.concatenate((q2*q[:nx], q2*q[nx:]))

    def rhs(self, q):
        return self.A @ q + self.nonlinearity(q)

    def gaussian(self, xbar):
        """Real-valued Gaussian exp(-((x + xbar)/s)^2) (the kernel in eq. (4.2)),
        embedded in the real part of the state."""
        g = np.exp(-((self.x + xbar)/self.s)**2)
        return np.concatenate((g, np.zeros(self.nx)))


class TimeStepper:
    """Crank-Nicolson (linear part) / Adams-Bashforth 2 (nonlinear part)."""

    def __init__(self, cgl, dt=1e-2):
        self.cgl = cgl
        self.dt = dt
        Id = sp.identity(2*cgl.nx, format='csc')
        self.lu = spla.splu(((1/dt)*Id - 0.5*cgl.A).tocsc())

    def integrate(self, q0, T, nsave, forcing=None):
        """Integrate from t = 0 to t = T, saving every nsave steps. forcing(t)
        (optional) returns the additive term B u(t), treated with the
        trapezoidal rule. Returns (time, Q) with Q of shape (2 nx, n_saved)."""
        cgl, dt = self.cgl, self.dt
        nsteps = int(round(T/dt))
        nsaved = nsteps//nsave + 1
        Q = np.zeros((2*cgl.nx, nsaved))
        Q[:, 0] = q0
        q = q0.copy()
        nl_old = None
        k_save = 1
        for k in range(1, nsteps + 1):
            nl = cgl.nonlinearity(q)
            nl_fwd = nl if nl_old is None else 1.5*nl - 0.5*nl_old
            nl_old = nl
            rhs = (1/dt)*q + 0.5*(cgl.A @ q) + nl_fwd
            if forcing is not None:
                rhs += 0.5*(forcing((k - 1)*dt) + forcing(k*dt))
            q = self.lu.solve(rhs)
            if k % nsave == 0:
                Q[:, k_save] = q
                k_save += 1
        time = dt*nsave*np.arange(nsaved)
        return time, Q
