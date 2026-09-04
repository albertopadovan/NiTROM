r"""
Full-order model for the complex Ginzburg--Landau (CGL) equation.

This is the benchmark of section 4 of

    A. Padovan, B. Vollmer and D. J. Bodony, "Non-intrusive optimization of
    reduced-order models", SIAM J. Appl. Dyn. Syst. 23(4), 2024.

The governing equation is

.. math::

    \frac{\partial q}{\partial t}
        = \left(-\nu \frac{\partial}{\partial x}
                + \gamma \frac{\partial^2}{\partial x^2}
                + \mu(x)\right) q - a |q|^2 q,
    \qquad x \in (-\infty, \infty),\ q(x,t) \in \mathbb{C},

with :math:`a = 0.1`, :math:`\gamma = 1 - i`, :math:`\nu = 2 + 0.4i` and
:math:`\mu(x) = (\mu_0 - c_u^2) + \mu_2 x^2 / 2`, where :math:`\mu_2 = -0.01`,
:math:`\mu_0 = 0.38` and :math:`c_u = 0.2` (so that :math:`\nu = U + 2 i c_u`
with :math:`U = 2`).  For these parameters the origin is linearly stable but
highly non-normal, producing large transient growth.

Inputs and outputs are the spatially localized pair

.. math::

    \mathbf{B}u = \exp\!\left\{-\left(\frac{x - \bar{x}}{s}\right)^2\right\} u,
    \qquad
    y = \mathbf{C}q = \int
        \exp\!\left\{-\left(\frac{x + \bar{x}}{s}\right)^2\right\} q \, dx,

with :math:`s = 1.6` and
:math:`\bar{x} = -\sqrt{-2(\mu_0 - c_u^2)/\mu_2}` the location of branch I of
the disturbance-amplification region.  The input therefore acts at branch I and
the sensor sits at branch II, downstream.

Discretizing in space on ``n`` nodes and splitting :math:`q` into its real and
imaginary parts gives a real-valued system with **cubic** dynamics

.. math::

    \frac{d\mathbf{q}}{dt}
        = \mathbf{A}\mathbf{q}
        + \mathbf{H} : (\mathbf{q} \otimes \mathbf{q} \otimes \mathbf{q})
        + \mathbf{B}\mathbf{u},
    \qquad \mathbf{y} = \mathbf{C}\mathbf{q},

with :math:`\mathbf{q} \in \mathbb{R}^{2n}`, :math:`\mathbf{u} \in
\mathbb{R}^2` and :math:`\mathbf{y} \in \mathbb{R}^2`.  The state is stored as
``q = [Re q; Im q]``.

The quartic tensor :math:`\mathbf{H}` is never formed: the cubic term is
pointwise in space, so it is applied matrix-free, and only its *reduced*
:math:`r \times r \times r \times r` projection is ever assembled.
"""

import numpy as np

# Parameters of section 4 of the paper.
NU = 2.0 + 0.4j
GAMMA = 1.0 - 1.0j
MU0 = 0.38
C_U = 0.2
MU2 = -0.01
A_CUBIC = 0.1
S_GAUSS = 1.6

#: Location of branch I, :math:`\bar{x} = -\sqrt{-2(\mu_0 - c_u^2)/\mu_2}`.
X_BAR = -np.sqrt(-2.0 * (MU0 - C_U**2) / MU2)


def analytical_eigenvalues(n_modes: int = 5) -> np.ndarray:
    r"""
    Exact eigenvalues of the linearized CGL operator,

    .. math::

        \lambda_k = (\mu_0 - c_u^2) - \frac{\nu^2}{4\gamma}
                    - \left(k + \tfrac{1}{2}\right)\sqrt{-2\mu_2\gamma}.

    Useful as a check on the spatial discretization; the imaginary part of
    :math:`\lambda_0` is the natural frequency :math:`\omega \approx 0.648`
    quoted in the paper.

    :param n_modes: number of eigenvalues to return
    :returns: complex array of shape ``(n_modes,)``
    """
    chi = np.sqrt(-2.0 * MU2 * GAMMA)
    k = np.arange(n_modes)
    return (MU0 - C_U**2) - NU**2 / (4.0 * GAMMA) - (k + 0.5) * chi


#: Natural frequency of the system, :math:`\omega \approx 0.648`.
OMEGA = float(abs(analytical_eigenvalues(1)[0].imag))


def build_operators(L: float = 30.0, n: int = 301, dtype=np.float64):
    r"""
    Assemble the discretized CGL operators on ``x`` in ``[-L, L]``.

    First and second derivatives use fourth-order central differences with
    homogeneous Dirichlet conditions at the truncated boundaries (the
    eigenfunctions are Gaussian-localized near the origin, so for ``L = 30``
    the truncation error is far below the discretization error).

    :param L: half-width of the truncated domain
    :param n: number of grid nodes
    :param dtype: floating-point type of the real-valued operators
    :returns: ``(x, dx, A, B, C)`` with ``A`` of shape ``(2n, 2n)``,
        ``B`` of shape ``(2n, 2)`` and ``C`` of shape ``(2, 2n)``
    """
    x = np.linspace(-L, L, n)
    dx = float(x[1] - x[0])

    D1 = np.zeros((n, n))
    for off, coef in zip((-2, -1, 1, 2), (1 / 12, -2 / 3, 2 / 3, -1 / 12)):
        D1 += np.diag(np.full(n - abs(off), coef), off)
    D2 = np.zeros((n, n))
    for off, coef in zip((-2, -1, 0, 1, 2), (-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12)):
        D2 += np.diag(np.full(n - abs(off), coef), off)
    D1 /= dx
    D2 /= dx**2

    mu = (MU0 - C_U**2) + MU2 * x**2 / 2.0
    Ac = -NU * D1 + GAMMA * D2 + np.diag(mu)  # complex (n, n)

    # Real representation of z -> Ac z acting on [Re z; Im z].
    A = np.block([[Ac.real, -Ac.imag], [Ac.imag, Ac.real]]).astype(dtype)

    b_field = np.exp(-(((x - X_BAR) / S_GAUSS) ** 2))
    c_field = np.exp(-(((x + X_BAR) / S_GAUSS) ** 2))
    zero = np.zeros(n)

    # Bu applies the real/imaginary parts of the complex input u to b(x).
    B = np.block(
        [[b_field[:, None], zero[:, None]], [zero[:, None], b_field[:, None]]]
    ).astype(dtype)
    # y = int c(x) q(x) dx, with dx the (uniform) quadrature weight.
    C = np.block(
        [[c_field[None, :] * dx, zero[None, :]], [zero[None, :], c_field[None, :] * dx]]
    ).astype(dtype)

    return x, dx, A, B, C


class full_order_model:
    r"""
    Discretized CGL full-order model with cubic dynamics.

    Mirrors the interface of the toy-model ``full_order_model`` so the same
    training and post-processing scripts apply, but the cubic tensor is
    applied matrix-free (it is pointwise in space, and a dense
    :math:`(2n)^4` tensor would be intractable).

    :param A: linear operator of shape ``(2n, 2n)``
    :param B: input matrix of shape ``(2n, 2)``
    :param C: output matrix of shape ``(2, 2n)``
    :param dtype: floating-point type
    """

    def __init__(self, A, B, C, device="cpu", dtype=np.float64):
        self.A = A
        self.B = B
        self.C = C
        self.n = A.shape[0] // 2  # number of spatial nodes
        self.device = device
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Cubic nonlinearity  -a |q|^2 q, pointwise in space
    # ------------------------------------------------------------------
    def cubic(self, q):
        r"""
        Evaluate :math:`-a|q|^2 q` for a state (or batch of states) ``q``.

        :param q: array of shape ``(..., 2n)``
        :returns: array of the same shape
        """
        n = self.n
        re, im = q[..., :n], q[..., n:]
        mag = re * re + im * im
        return np.concatenate([-A_CUBIC * mag * re, -A_CUBIC * mag * im], axis=-1)

    def cubic_trilinear(self, u, v, w):
        r"""
        Symmetric trilinear form :math:`T` with :math:`T(q, q, q) = -a|q|^2 q`.

        Writing :math:`q = (\alpha, \beta)` pointwise, the cubic term is
        :math:`-a(\alpha^3 + \alpha\beta^2, \beta^3 + \beta\alpha^2)`; the
        mixed monomials are symmetrized over their three argument slots.
        This is what gets projected to build the reduced quartic tensor.

        :param u: array of shape ``(..., 2n)``
        :param v: array of shape ``(..., 2n)``
        :param w: array of shape ``(..., 2n)``
        :returns: array of shape ``(..., 2n)``
        """
        n = self.n
        ur, ui = u[..., :n], u[..., n:]
        vr, vi = v[..., :n], v[..., n:]
        wr, wi = w[..., :n], w[..., n:]
        third = 1.0 / 3.0
        out_r = -A_CUBIC * (
            ur * vr * wr + third * (ur * vi * wi + ui * vr * wi + ui * vi * wr)
        )
        out_i = -A_CUBIC * (
            ui * vi * wi + third * (ui * vr * wr + ur * vi * wr + ur * vr * wi)
        )
        return np.concatenate([out_r, out_i], axis=-1)

    def cubic_jacobian_apply(self, Q, v):
        r"""
        Apply the Jacobian of the cubic term at base state ``Q`` to ``v``.

        Pointwise the Jacobian is the symmetric matrix
        :math:`-a\begin{pmatrix} 3\alpha^2 + \beta^2 & 2\alpha\beta \\
        2\alpha\beta & \alpha^2 + 3\beta^2\end{pmatrix}`, so it is its own
        transpose and serves both the tangent and the adjoint dynamics.

        :param Q: base state of shape ``(..., 2n)``
        :param v: perturbation of shape ``(..., 2n)``
        :returns: array of shape ``(..., 2n)``
        """
        n = self.n
        al, be = Q[..., :n], Q[..., n:]
        vr, vi = v[..., :n], v[..., n:]
        out_r = -A_CUBIC * ((3 * al * al + be * be) * vr + 2 * al * be * vi)
        out_i = -A_CUBIC * (2 * al * be * vr + (al * al + 3 * be * be) * vi)
        return np.concatenate([out_r, out_i], axis=-1)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def evaluate_fom_dynamics(self, t, q, u):
        r"""
        Evaluate :math:`\dot{q} = Aq + H:(q \otimes q \otimes q) + f(t)`.

        :param t: time
        :param q: state of shape ``(2n,)`` or ``(m, 2n)``
        :param u: full-space forcing field, either an array broadcastable to
            ``q`` or a callable ``u(t)`` returning one
        :returns: array of the same shape as ``q``
        """
        f = u if hasattr(u, "__len__") else u(t)
        return q @ self.A.T + self.cubic(q) + f

    def evaluate_fom_adjoint(self, t, q, fQ):
        r"""
        Evaluate the adjoint dynamics :math:`J(Q(t))^\top q`, where ``fQ`` is
        a callable returning the base flow :math:`Q(t)`.

        :param t: time
        :param q: adjoint state of shape ``(2n,)`` or ``(m, 2n)``
        :param fQ: callable returning the base flow at time ``t``
        :returns: array of the same shape as ``q``
        """
        # The cubic Jacobian is pointwise symmetric, so no transpose is needed.
        return q @ self.A + self.cubic_jacobian_apply(fQ(t), q)

    def compute_output(self, q):
        """Output ``y = C q``, batched over any leading axes."""
        return np.matmul(self.C, q)

    def compute_output_derivative(self, q):
        """Constant output Jacobian ``C``."""
        return self.C

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def assemble_petrov_galerkin_tensors(self, Phi, Psi):
        r"""
        Petrov--Galerkin projection of the FOM operators onto ``(Phi, Psi)``.

        With :math:`\tilde{\Phi} = \Phi(\Psi^\top\Phi)^{-1}`, the reduced
        operators are :math:`A_r = \Psi^\top A \tilde{\Phi}`,
        :math:`H_r[:,i,j,k] = \Psi^\top T(\tilde\Phi_i, \tilde\Phi_j,
        \tilde\Phi_k)`, :math:`B_r = \Psi^\top B` and
        :math:`C_r = C\tilde{\Phi}`.  Setting ``Psi = Phi`` recovers the
        POD-Galerkin model.

        :param Phi: trial basis of shape ``(2n, r)``
        :param Psi: test basis of shape ``(2n, r)``
        :returns: ``((A_r, H_r), (B_r, C_r))``
        """
        _, r = Phi.shape
        PhiF = Phi @ np.linalg.inv(Psi.T @ Phi)

        Ar = Psi.T @ self.A @ PhiF
        Hr = np.zeros((r, r, r, r), dtype=self.dtype)
        for i in range(r):
            for j in range(r):
                for k in range(r):
                    Hr[:, i, j, k] = Psi.T @ self.cubic_trilinear(
                        PhiF[:, i], PhiF[:, j], PhiF[:, k]
                    )

        Br = Psi.T @ self.B
        Cr = self.compute_output(PhiF)

        return (Ar, Hr), (Br, Cr)
