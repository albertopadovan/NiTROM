from typing import Any

from ..backend import get_backend
from .linear_projection import LinearProjection
from .projection import Projection


class ObliqueChartProjection(LinearProjection):
    r"""
    Oblique projection whose test basis is charted as
    :math:`\Psi = \Phi + W N`.

    The trial basis :math:`\Phi` (orthonormal, ``(N, r)``) and the chart
    :math:`W` (orthonormal, ``(N, k)``, spanning directions **orthogonal to**
    :math:`\Phi` -- in practice the next :math:`k` POD modes) are both
    **fixed**.  The single optimization parameter is the coefficient matrix
    :math:`N \in \mathbb{R}^{k \times r}`.

    Because :math:`W^\top \Phi = 0` and :math:`\Phi^\top \Phi = I`,

    .. math::

        \Psi^\top \Phi = \Phi^\top\Phi + N^\top W^\top \Phi = I_r

    holds *identically* for every :math:`N`.  Three things follow, and they are
    the whole point of the parameterization:

    * :math:`S = (\Psi^\top\Phi)^{-1} = I`, so the oblique projector is
      :math:`\Phi\Psi^\top` and :meth:`decode` is simply :math:`\Phi z` --
      there is no matrix inverse to condition, and no transversality constraint
      that the optimizer has to respect;
    * the parameterization is **Euclidean** in :math:`N`, with only
      :math:`kr` free parameters, so no Stiefel retraction is needed;
    * :math:`\Psi` can reach directions that :math:`\Phi` misses.  That
      matters for strongly non-normal systems, where the leading POD modes
      capture the energetic response but represent the *initial condition*
      poorly, and it is exactly what an oblique projection is for.

    :param bases: list ``[Phi, W, N]``.  ``Phi`` is ``(N, r)``, ``W`` is
        ``(N, k)`` with ``W^T Phi = 0``, and ``N`` is ``(k, r)`` -- pass
        zeros to start from the orthogonal projection :math:`\Psi = \Phi`.
    :type bases: list
    :raises ValueError: if ``W`` is not orthogonal to ``Phi``, or if the
        shapes are inconsistent
    """

    def __init__(self, bases: list):
        Phi, W, N = bases[0], bases[1], bases[2]
        bkend = get_backend()

        n, r = Phi.shape
        if W.shape[0] != n:
            raise ValueError(
                f"W must have {n} rows (the ambient dimension), got {W.shape[0]}."
            )
        k = W.shape[1]
        if tuple(N.shape) != (k, r):
            raise ValueError(
                f"N must have shape ({k}, {r}), got {tuple(N.shape)}."
            )
        leak = float(bkend.vector_norm(W.T @ Phi))
        scale = float(bkend.vector_norm(W)) * float(bkend.vector_norm(Phi))
        if leak > 1e-8 * max(scale, 1.0):
            raise ValueError(
                "W must be orthogonal to Phi so that Psi^T Phi = I identically; "
                f"||W^T Phi|| = {leak:.3e}."
            )

        # Skip LinearProjection.__init__: the parameter set is different.
        Projection.__init__(
            self,
            n,
            r,
            param_names=["N"],
            device=bkend.device_of(Phi),
            dtype=Phi.dtype,
        )
        self.Phi = Phi
        self.W = W
        self.update([N])

    def get_params(self) -> list[Any]:
        """Return ``[N]`` -- the only optimization parameter."""
        return [self.N]

    def update(self, params: list) -> None:
        r"""
        Set the chart coefficients and rebuild :math:`\Psi = \Phi + W N`.

        :math:`S` is the identity by construction, so it is set rather than
        inverted.

        :param params: list ``[N]`` with ``N`` of shape ``(k, r)``
        :type params: list
        """
        self.N = params[0]
        self.Psi = self.Phi + self.W @ self.N
        self.S = self.backend.eye(
            self.latent_space_dimension, dtype=self.dtype, device=self.device
        )

    def decode(self, z: Any) -> Any:
        r"""
        Reconstruct :math:`q = \Phi z` (the general oblique decode
        :math:`\Phi(\Psi^\top\Phi)^{-1}z` with :math:`S = I`).

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :rtype: backend array
        """
        if z.ndim == 1:
            return self.Phi @ z
        return z @ self.Phi.T

    def vjp_bases(self, grad_Phi: Any, grad_Psi: Any) -> tuple:
        r"""
        Chain rule through :math:`\Psi = \Phi + W N`.

        :math:`\Psi` is affine in :math:`N`, so
        :math:`\partial J/\partial N = W^\top \partial J/\partial \Psi`.
        ``grad_Phi`` is discarded: :math:`\Phi` is fixed.  Any contribution
        routed through :math:`S` vanishes on its own, since
        :math:`W^\top\Phi = 0`.

        :param grad_Phi: cotangent w.r.t. ``Phi`` (unused; ``Phi`` is fixed)
        :param grad_Psi: cotangent w.r.t. ``Psi``, shape ``(N, r)``
        :returns: ``(grad_N,)``, matching :attr:`param_names`
        :rtype: tuple
        """
        return (self.W.T @ grad_Psi,)

    def vjp_encode(self, q: Any, v: Any) -> tuple:
        r"""
        VJP of :math:`z = \Psi^\top q` with respect to :math:`N`:
        :math:`\partial J/\partial N = W^\top q\, v^\top`.

        :param q: full-space vector of shape ``(N,)`` or ``(m, N)``
        :param v: upstream adjoint seed of shape ``(r,)`` or ``(m, r)``
        :returns: ``(grad_N,)``
        :rtype: tuple
        """
        grad_Psi = (
            self.backend.outer(q, v) if q.ndim == 1 else q.T @ v
        )  # (N, r)
        return (self.W.T @ grad_Psi,)

    def vjp_decode(self, z: Any, v: Any) -> tuple:
        r"""
        VJP of :math:`\hat{q} = \Phi z` with respect to :math:`N`.  The
        decoder does not involve :math:`\Psi`, so this is zero.

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :param v: full-space cotangent of shape ``(N,)`` or ``(m, N)``
        :returns: ``(grad_N,)``, all zeros
        :rtype: tuple
        """
        return (self.backend.zeros_like(self.N),)

    def vjp_decode_state(self, z: Any, v: Any) -> Any:
        r"""
        VJP of :math:`\hat{q} = \Phi z` with respect to the latent state:
        :math:`\Phi^\top v`.

        :param z: reduced-space vector (unused; the decoder is linear)
        :param v: full-space cotangent of shape ``(N,)`` or ``(m, N)``
        :rtype: backend array
        """
        if v.ndim == 1:
            return self.Phi.T @ v
        return v @ self.Phi
