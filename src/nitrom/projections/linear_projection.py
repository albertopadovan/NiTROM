from typing import Any

from ..backend import get_backend
from .projection import Projection


class LinearProjection(Projection):
    r"""
    Oblique linear projection defined by trial basis :math:`\Phi` and
    test basis :math:`\Psi`:

    .. math::

        \text{encode}(q) = \Psi^\top q, \qquad
        \text{decode}(z) = \Phi (\Psi^\top \Phi)^{-1} z

    :param bases: list of two tensors ``[Phi, Psi]``, each of shape ``(N, r)``
    :type bases: list
    """

    def __init__(self, bases: list):
        Phi = bases[0]
        n, r = Phi.shape
        bkend = get_backend()
        super().__init__(
            n,
            r,
            param_names=["Phi", "Psi"],
            device=bkend.device_of(Phi),
            dtype=Phi.dtype,
        )
        self.Phi = bases[0]
        self.Psi = bases[1]
        self.S = bkend.inv(self.Psi.T @ self.Phi)

    def get_params(self) -> list[Any]:
        """Return ``[Phi, Psi]``."""
        return [self.Phi, self.Psi]

    def update(self, params: list) -> None:
        r"""
        Update the trial and test bases and recompute :math:`S = (\Psi^\top \Phi)^{-1}`.

        :param params: list of two tensors ``[Phi, Psi]``, each of shape ``(N, r)``
        :type params: list
        """
        self.Phi = params[0]
        self.Psi = params[1]
        self.S = self.backend.inv(self.Psi.T @ self.Phi)

    def encode(self, q: Any) -> Any:
        r"""
        Project from full space to reduced space: :math:`z = \Psi^\top q`.

        :param q: full-space vector of shape ``(N,)`` or ``(m, N)``
        :rtype: backend array
        """
        if q.ndim == 1:
            return self.Psi.T @ q
        return (self.Psi.T @ q.T).T

    def decode(self, z: Any) -> Any:
        r"""
        Reconstruct from reduced space to full space:
        :math:`q = \Phi (\Psi^\top \Phi)^{-1} z`.

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :rtype: backend array
        """
        if z.ndim == 1:
            return self.Phi @ (self.S @ z)
        return (self.Phi @ (self.S @ z.T)).T

    def vjp_encode(self, q: Any, v: Any) -> tuple:
        r"""
        VJP of the encoder :math:`z = \Psi^\top q` with respect to :math:`\Psi`.

        .. math::

            \frac{\partial J}{\partial \Psi} = q\, v^\top

        :param q: full-space vector of shape ``(N,)`` or ``(m, N)``
        :param v: upstream adjoint seed :math:`v = \partial J / \partial z`
            of shape ``(r,)`` or ``(m, r)``
        :returns: ``(grad_Phi, grad_Psi)``
        :rtype: tuple
        """
        if q.ndim == 1:
            grad_Psi = self.backend.outer(q, v)
        else:
            # q is (m, N), v is (m, r) -> q.T @ v sums over batch -> (N, r)
            grad_Psi = q.T @ v
        return (self.backend.zeros_like(grad_Psi), grad_Psi)

    def vjp_decode(self, z: Any, v: Any) -> tuple:
        r"""
        VJP of the decoder :math:`\hat{q} = \Phi\, S\, z` with respect to
        :math:`\Phi` and :math:`\Psi`, where :math:`S = (\Psi^\top \Phi)^{-1}`.

        .. math::

            \frac{\partial J}{\partial \Phi}
                = \bigl(v - \Psi\, S^\top \Phi^\top v\bigr)(S\,z)^\top,
            \qquad
            \frac{\partial J}{\partial \Psi}
                = -(\Phi\, S\, z)(S^\top \Phi^\top v)^\top

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :param v: upstream adjoint seed :math:`v = \partial J / \partial \hat{q}`
            of shape ``(N,)`` or ``(m, N)``
        :returns: ``(grad_Phi, grad_Psi)``
        :rtype: tuple
        """
        if v.ndim == 1:
            # Unbatched: v is (N,), z is (r,)
            w = self.backend.outer(v, self.S @ z)  # (N, r)
        else:
            # Batched: v is (m, N), z is (m, r) -> v.T @ (z @ S.T) -> (N, r)
            w = v.T @ (z @ self.S.T)  # (N, r)
        # grad_Phi = w - Psi @ S^T @ Phi^T @ w
        PhiTw = self.Phi.T @ w  # (r, r)
        grad_Phi = w - self.Psi @ (self.S.T @ PhiTw)  # (N, r)
        # grad_Psi = -Phi @ w^T @ Phi @ S
        wTPhi = w.T @ self.Phi  # (r, r)
        grad_Psi = -self.Phi @ (wTPhi @ self.S)  # (N, r)
        return (grad_Phi, grad_Psi)

    def vjp_bases(self, grad_Phi: Any, grad_Psi: Any) -> tuple:
        r"""
        Map cotangents with respect to the *bases* onto gradients with respect
        to this projection's parameters, in :attr:`param_names` order.

        A caller that differentiates a cost through :math:`\Phi` and
        :math:`\Psi` directly (rather than through :meth:`encode` /
        :meth:`decode`) obtains ambient ``(N, r)`` cotangents.  For a plain
        :class:`LinearProjection` the bases *are* the parameters, so this is
        the identity; a subclass that parameterizes them differently -- e.g.
        :class:`ObliqueChartProjection`, where
        :math:`\Psi = \Phi + W N` -- overrides it with the chain rule.

        :param grad_Phi: :math:`\partial J / \partial \Phi`, shape ``(N, r)``
        :param grad_Psi: :math:`\partial J / \partial \Psi`, shape ``(N, r)``
        :returns: gradients in :attr:`param_names` order
        :rtype: tuple
        """
        return (grad_Phi, grad_Psi)

    def vjp_decode_state(self, z: Any, v: Any) -> Any:
        r"""
        VJP of the decoder :math:`\hat{q} = \Phi\, S\, z` with respect to the
        latent state :math:`z`.  The decoder is linear in :math:`z`, so the
        Jacobian is the constant :math:`\Phi S` and

        .. math::

            \left(\frac{\partial \hat{q}}{\partial z}\right)^\top v
                = S^\top \Phi^\top v.

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)`` (unused;
            the Jacobian does not depend on the state)
        :param v: full-space cotangent of shape ``(N,)`` or ``(m, N)``
        :returns: latent-space vector of shape ``(r,)`` or ``(m, r)``
        :rtype: backend array
        """
        if v.ndim == 1:
            return self.S.T @ (self.Phi.T @ v)  # (r,)
        return (v @ self.Phi) @ self.S  # (m, r) = (S^T Phi^T v)^(j)
