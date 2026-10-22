from typing import Any, Literal

from ..backend import get_backend
from .projection import Projection


class LinearProjection(Projection):
    r"""
    Oblique linear projection defined by trial basis :math:`\Phi` and
    test basis :math:`\Psi`:

    .. math::

        \text{encode}(q) = \Psi^\top q, \qquad
        \text{decode}(z) = \Phi (\Psi^\top \Phi)^{-1} z

    With ``biorthogonalize="encoder"``, the maps instead are

    .. math::

        \text{encode}(q) = (\Psi^\top \Phi)^{-1}\Psi^\top q, \qquad
        \text{decode}(z) = \Phi z.

    :param bases: list of two tensors ``[Phi, Psi]``, each of shape ``(N, r)``
    :type bases: list
    :param biorthogonalize: which map to biorthogonalize, ``"decoder"`` (default)
        or ``"encoder"``
    """

    def __init__(
        self,
        bases: list,
        *,
        biorthogonalize: Literal["encoder", "decoder"] = "decoder",
        backend=None,
    ):
        if biorthogonalize not in ("encoder", "decoder"):
            raise ValueError(
                f"biorthogonalize must be 'encoder' or 'decoder', got {biorthogonalize!r}."
            )
        self.biorthogonalize = biorthogonalize
        Phi = bases[0]
        n, r = Phi.shape
        bkend = backend or get_backend()
        super().__init__(
            n,
            r,
            param_names=["Phi", "Psi"],
            device=bkend.device_of(Phi),
            dtype=Phi.dtype,
            backend=bkend,
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
        Project to reduced space using :math:`z = \Psi^\top q`, or
        :math:`z = S\Psi^\top q` when ``biorthogonalize="encoder"``.

        :param q: full-space vector of shape ``(N,)`` or ``(m, N)``
        :rtype: backend array
        """
        if q.ndim == 1:
            z = self.Psi.T @ q
            return self.S @ z if self.biorthogonalize == "encoder" else z
        z = self.Psi.T @ q.T
        return (self.S @ z).T if self.biorthogonalize == "encoder" else z.T

    def decode(self, z: Any) -> Any:
        r"""
        Reconstruct from reduced space to full space:
        :math:`q = \Phi (\Psi^\top \Phi)^{-1} z`.
        With ``biorthogonalize="encoder"``, use :math:`q = \Phi z`.

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :rtype: backend array
        """
        if self.biorthogonalize == "encoder":
            return self.Phi @ z if z.ndim == 1 else (self.Phi @ z.T).T
        if z.ndim == 1:
            return self.Phi @ (self.S @ z)
        return (self.Phi @ (self.S @ z.T)).T

    def vjp_encode(self, q: Any, v: Any) -> tuple:
        r"""
        VJP of the encoder with respect to :math:`\Phi` and :math:`\Psi`.
        By default, :math:`z = \Psi^\top q`, with zero gradient for :math:`\Phi`
        and

        .. math::

            \frac{\partial J}{\partial \Psi} = q\, v^\top

        With ``biorthogonalize="encoder"``, let :math:`z = S\Psi^\top q`
        and :math:`a = S^\top v`. The gradients are
        :math:`-\Psi a z^\top` and :math:`(q - \Phi z)a^\top`.

        :param q: full-space vector of shape ``(N,)`` or ``(m, N)``
        :param v: upstream adjoint seed :math:`v = \partial J / \partial z`
            of shape ``(r,)`` or ``(m, r)``
        :returns: ``(grad_Phi, grad_Psi)``
        :rtype: tuple
        """
        if self.biorthogonalize == "encoder":
            z = self.encode(q)
            if q.ndim == 1:
                a = self.S.T @ v
                return (
                    -self.backend.outer(self.Psi @ a, z),
                    self.backend.outer(q - self.Phi @ z, a),
                )
            a = v @ self.S
            return (-self.Psi @ (a.T @ z), (q - z @ self.Phi.T).T @ a)
        # The batched product sums the sample axis.
        grad_Psi = self.backend.outer(q, v) if q.ndim == 1 else q.T @ v
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

        With ``biorthogonalize="encoder"``, the decoder is :math:`\Phi z`
        and the gradients are :math:`v z^\top` and zero.

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :param v: upstream adjoint seed :math:`v = \partial J / \partial \hat{q}`
            of shape ``(N,)`` or ``(m, N)``
        :returns: ``(grad_Phi, grad_Psi)``
        :rtype: tuple
        """
        if self.biorthogonalize == "encoder":
            w = self.backend.outer(v, z) if v.ndim == 1 else v.T @ z
        else:
            w = (
                self.backend.outer(v, self.S @ z)
                if v.ndim == 1
                else v.T @ (z @ self.S.T)
            )
        return self.vjp_decode_from_outer(w)

    def vjp_decode_from_outer(self, w: Any) -> tuple:
        r"""
        Decoder parameter VJP given the accumulated outer product
        :math:`w = \sum_m v_m (S z_m)^\top`, shape ``(N, r)``.
        With ``biorthogonalize="encoder"``, supply
        :math:`w = \sum_m v_m z_m^\top` instead.

        :meth:`vjp_decode` forms ``w`` from full-space seeds ``v``; a caller
        that can assemble ``w`` more cheaply (e.g. without materializing the
        seeds) calls this directly, so both routes share the same formula.

        :returns: ``(grad_Phi, grad_Psi)``
        """
        if self.biorthogonalize == "encoder":
            return (w, self.backend.zeros_like(self.Psi))
        # grad_Phi = w - Psi @ S^T @ Phi^T @ w
        PhiTw = self.Phi.T @ w  # (r, r)
        grad_Phi = w - self.Psi @ (self.S.T @ PhiTw)  # (N, r)
        # grad_Psi = -Phi @ w^T @ Phi @ S
        wTPhi = w.T @ self.Phi  # (r, r)
        grad_Psi = -self.Phi @ (wTPhi @ self.S)  # (N, r)
        return (grad_Phi, grad_Psi)

    def vjp_decode_state(self, z: Any, v: Any) -> Any:
        r"""
        VJP of the decoder :math:`\hat{q} = \Phi\, S\, z` with respect to the
        latent state :math:`z`.  The decoder is linear in :math:`z`, so the
        Jacobian is the constant :math:`\Phi S` and

        .. math::

            \left(\frac{\partial \hat{q}}{\partial z}\right)^\top v
                = S^\top \Phi^\top v.

        With ``biorthogonalize="encoder"``, this is :math:`\Phi^\top v`.

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)`` (unused;
            the Jacobian does not depend on the state)
        :param v: full-space cotangent of shape ``(N,)`` or ``(m, N)``
        :returns: latent-space vector of shape ``(r,)`` or ``(m, r)``
        :rtype: backend array
        """
        if self.biorthogonalize == "encoder":
            return self.Phi.T @ v if v.ndim == 1 else v @ self.Phi
        if v.ndim == 1:
            return self.S.T @ (self.Phi.T @ v)  # (r,)
        return (v @ self.Phi) @ self.S  # (m, r) = (S^T Phi^T v)^(j)
