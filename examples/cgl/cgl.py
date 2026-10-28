"""
Complex Ginzburg-Landau (CGL) solver, following the setup of
Padovan, Vollmer & Bodony, SIADS 2024 (section 4):

    dq/dt = (-nu d/dx + gamma d^2/dx^2 + mu) q - a |q|^2 q,

with nu = 2 + 0.4i, gamma = 1 - i, mu = (mu0 - cu^2) + mu2 x^2/2, cu = Im(nu)/2,
mu0 = 0.38, mu2 = -0.01, a = 0.1.

Discretization (restored from the repository's CGL implementation):
  - uniform grid of nx = 256 points on [-L/2, L/2], L = 100
  - 4th-order central finite differences in the interior, truncated at the
    near-boundary rows and 2nd-order at the endpoints, with zero exterior values
  - real-valued state q = [Re(q); Im(q)] in R^{2 nx}
  - time integration: Crank-Nicolson for the linear part, 2nd-order
    Adams-Bashforth for the cubic nonlinearity (forward Euler on the first step)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class CGL:
    def __init__(
        self,
        nx=256,
        L=100.0,
        nu=2.0 + 0.4j,
        gamma=1.0 - 1.0j,
        mu0=0.38,
        mu2=-0.01,
        a=0.1,
        s=1.6,
    ):

        if not isinstance(nx, (int, np.integer)) or nx < 5:
            raise ValueError("nx must be an integer of at least 5.")
        if not np.isfinite(L) or L <= 0 or not np.isfinite(s) or s <= 0:
            raise ValueError("L and s must be finite and positive.")

        self.nx = nx
        self.L = L
        self.x = np.linspace(-L / 2, L / 2, num=nx, endpoint=True)
        self.dx = self.x[1] - self.x[0]

        self.nu = nu
        self.gamma = gamma
        self.mu0 = mu0
        self.mu2 = mu2
        self.a = a
        self.s = s
        self.cu = nu.imag / 2
        self.mu = (mu0 - self.cu**2) + mu2 * self.x**2 / 2

        # Branch I of the disturbance-amplification region (xbar < 0)
        self.xbar = -np.sqrt(-2 * (mu0 - self.cu**2) / mu2)

        self.D = self._first_derivative()
        self.DD = self._second_derivative()
        self.A = self._linear_operator()

    def _first_derivative(self):
        # Truncated fourth-order stencil, with second-order endpoint rows.
        D = sp.diags(
            [1, -8, 8, -1],
            [-2, -1, 1, 2],
            shape=(self.nx, self.nx),
            format="lil",
            dtype=float,
        ) / (12 * self.dx)
        D[0, :], D[-1, :] = 0, 0
        D[0, 1], D[-1, -2] = 1 / (2 * self.dx), -1 / (2 * self.dx)
        return D.tocsc()

    def _second_derivative(self):
        DD = sp.diags(
            [-1, 16, -30, 16, -1],
            [-2, -1, 0, 1, 2],
            shape=(self.nx, self.nx),
            format="lil",
            dtype=float,
        ) / (12 * self.dx**2)
        DD[0, :], DD[-1, :] = 0, 0
        DD[0, :2] = np.array([-2, 1]) / self.dx**2
        DD[-1, -2:] = np.array([1, -2]) / self.dx**2
        return DD.tocsc()

    def _linear_operator(self):
        muI = sp.diags(self.mu, offsets=0, format="csc")
        A11 = -self.nu.real * self.D + self.gamma.real * self.DD + muI
        A12 = self.nu.imag * self.D - self.gamma.imag * self.DD
        return sp.bmat([[A11, A12], [-A12, A11]], format="csc")

    def nonlinearity(self, q):
        nx = self.nx
        q2 = q[:nx] ** 2 + q[nx:] ** 2
        return -self.a * np.concatenate((q2 * q[:nx], q2 * q[nx:]))

    def rhs(self, q):
        return self.A @ q + self.nonlinearity(q)

    def gaussian(self, xbar):
        """Real-valued Gaussian exp(-((x + xbar)/s)^2) (the kernel in eq. (4.2)),
        embedded in the real part of the state. Its centre is -xbar;
        use gaussian(-self.xbar) for an actuator at branch I."""
        g = np.exp(-(((self.x + xbar) / self.s) ** 2))
        return np.concatenate((g, np.zeros(self.nx)))


class TimeStepper:
    """Crank-Nicolson (linear part) / Adams-Bashforth 2 (nonlinear part)."""

    def __init__(self, cgl, dt=1e-2):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        self.cgl = cgl
        self.dt = dt
        Id = sp.identity(2 * cgl.nx, format="csc")
        self.lu = spla.splu(((1 / dt) * Id - 0.5 * cgl.A).tocsc())

    def integrate(self, q0, T, nsave, forcing=None):
        """Integrate from t = 0 to t = T, saving every nsave steps. forcing(t)
        (optional) returns the additive term B u(t), treated with the
        trapezoidal rule. Returns (time, Q) with Q of shape (2 nx, n_saved)."""
        cgl, dt = self.cgl, self.dt
        if not np.isfinite(T) or T < 0:
            raise ValueError("T must be finite and nonnegative.")
        if not isinstance(nsave, (int, np.integer)) or nsave < 1:
            raise ValueError("nsave must be a positive integer.")
        nsteps = round(T / dt)
        if not np.isclose(nsteps * dt, T, rtol=1e-12, atol=1e-12):
            raise ValueError("T must be an integer multiple of dt.")
        q0 = np.asarray(q0, dtype=float)
        if q0.shape != (2 * cgl.nx,) or not np.all(np.isfinite(q0)):
            raise ValueError(f"q0 must be a finite real vector of length {2 * cgl.nx}.")
        nsaved = nsteps // nsave + 1
        Q = np.zeros((2 * cgl.nx, nsaved))
        Q[:, 0] = q0
        q = q0.copy()
        nl_old = None
        k_save = 1
        for k in range(1, nsteps + 1):
            nl = cgl.nonlinearity(q)
            nl_fwd = nl if nl_old is None else 1.5 * nl - 0.5 * nl_old
            nl_old = nl
            rhs = (1 / dt) * q + 0.5 * (cgl.A @ q) + nl_fwd
            if forcing is not None:
                rhs += 0.5 * (forcing((k - 1) * dt) + forcing(k * dt))
            q = self.lu.solve(rhs)
            if k % nsave == 0:
                Q[:, k_save] = q
                k_save += 1
        time = dt * nsave * np.arange(nsaved)
        return time, Q


class OutputMap:
    """Gaussian sensor y = C q, with the constant Jacobian NiTROM needs."""

    def __init__(self, C):
        self.C = C

    def compute_output(self, q):
        return self.C @ q

    def compute_output_derivative(self, q):
        return self.C


def input_output(model, seed=0):
    """Unit actuator columns, random forcing direction, frequency, and sensor."""
    g = np.exp(-(((model.x - model.xbar) / model.s) ** 2))
    columns = np.array([np.r_[g, np.zeros(model.nx)], np.r_[np.zeros(model.nx), g]])
    # The random direction uses the original, unnormalized actuator matrix.
    v = np.random.default_rng(seed).standard_normal(2)
    direction = columns.T @ v
    direction /= np.linalg.norm(direction)
    columns = np.array([column / np.linalg.norm(column) for column in columns])
    eigenvalues = np.linalg.eigvals(model.A.toarray())
    omega = abs(eigenvalues[np.argmax(eigenvalues.real)].imag)
    sensor = np.exp(-(((model.x + model.xbar) / model.s) ** 2))
    C = np.zeros((2, 2 * model.nx))
    C[0, : model.nx], C[1, model.nx :] = sensor, sensor
    return columns, direction, omega, C
