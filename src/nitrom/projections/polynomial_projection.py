from itertools import combinations
from string import ascii_lowercase
from typing import Any

from ..backend import get_backend
from .projection import Projection


class PolynomialProjection(Projection):
    r"""
    Polynomial projection with a linear encoder and a nonlinear decoder:

    .. math::

        \text{encode}(q) = \Psi^\top q, \qquad
        \text{decode}(z) = \Phi S\, z
            + \mathbb{P}\sum_{k} A_k\, z^{\otimes k},

    where :math:`S = (\Psi^\top \Phi)^{-1}` and
    :math:`\mathbb{P} = I - \Phi S\, \Psi^\top` is the complementary
    projector.  The projector ensures that
    :math:`\text{encode}(\text{decode}(z)) = z`.

    :param nonlin_poly_comp: nonlinear polynomial degrees in ascending
        order, e.g. ``[2, 3]``
    :type nonlin_poly_comp: list[int]
    :param tensors: ``[Phi, Psi, A_k1, A_k2, ...]`` where ``Phi`` and
        ``Psi`` have shape ``(N, r)`` and each ``A_k`` has shape
        ``(N,) + (r,) * k``
    :type tensors: list
    """

    def __init__(self, nonlin_poly_comp: list[int], tensors: list):
        if nonlin_poly_comp != sorted(nonlin_poly_comp):
            raise ValueError(
                f"nonlin_poly_comp must be in ascending order, got {nonlin_poly_comp}"
            )
        param_names = ["Phi", "Psi"] + [f"A{k}" for k in nonlin_poly_comp]
        Phi = tensors[0]
        n, r = Phi.shape
        bkend = get_backend()
        super().__init__(
            n,
            r,
            param_names=param_names,
            device=bkend.device_of(Phi),
            dtype=Phi.dtype,
        )
        self.nonlin_poly_comp = nonlin_poly_comp
        self._generate_einsum_subscripts()
        self.update(tensors)

    def _generate_einsum_subscripts(self) -> None:
        """Generate einsum subscripts for each nonlinear degree."""
        ss = []
        for k in self.nonlin_poly_comp:
            ssk = ascii_lowercase[: k + 1]
            ssk = [ssk] + [s for s in ssk[1:]]
            ss.append(ssk)
        self.einsum_ss = tuple(ss)

    def get_params(self) -> list[Any]:
        """Return ``[Phi, Psi, A_k1, A_k2, ...]``."""
        return [getattr(self, name) for name in self.param_names]

    def update(self, params: list) -> None:
        r"""
        Update all parameters and recompute :math:`S = (\Psi^\top \Phi)^{-1}`.

        :param params: ``[Phi, Psi, A_k1, A_k2, ...]`` matching :attr:`param_names`
        :type params: list
        """
        for name, tensor in zip(self.param_names, params, strict=True):
            setattr(self, name, tensor)
        self.S = self.backend.inv(self.Psi.T @ self.Phi)

    def encode(self, q: Any) -> Any:
        r"""
        Encode from full space to reduced space:

        .. math::

            z = \Psi^\top q

        :param q: full-space vector of shape ``(N,)`` or ``(m, N)``
        :returns: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :rtype: backend array
        """
        if q.ndim == 1:
            return self.Psi.T @ q
        return (self.Psi.T @ q.T).T

    def decode(self, z: Any) -> Any:
        r"""
        Decode from reduced space to full space:

        .. math::

            q = \Phi S\, z
                + \mathbb{P}\sum_{k} A_k\, z^{\otimes k},
            \quad
            \mathbb{P} = I - \Phi S\, \Psi^\top

        where :math:`S = (\Psi^\top \Phi)^{-1}`.

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :returns: full-space vector of shape ``(N,)`` or ``(m, N)``
        :rtype: backend array
        """
        bkend = self.backend
        # Linear part: Phi S z
        if z.ndim == 1:
            q = self.Phi @ (self.S @ z)
        else:
            q = (self.Phi @ (self.S @ z.T)).T

        # Nonlinear part: P @ sum_k A_k z^{otimes k}
        # P = I - Phi S Psi^T
        nonlin = bkend.zeros_like(q)
        for i, k in enumerate(self.nonlin_poly_comp):
            A_k = getattr(self, f"A{k}")
            if z.ndim == 1:
                equation = ",".join(self.einsum_ss[i])
                operands = [A_k] + [z for _ in range(k)]
                nonlin += bkend.einsum(equation, *operands)
            else:
                parts = self.einsum_ss[i]
                eq_parts = [parts[0]] + [f"...{p}" for p in parts[1:]]
                equation = ",".join(eq_parts)
                operands = [A_k] + [z for _ in range(k)]
                nonlin += bkend.einsum(equation, *operands)

        # Apply projector P = I - Phi S Psi^T
        if z.ndim == 1:
            q += nonlin - self.Phi @ (self.S @ (self.Psi.T @ nonlin))
        else:
            q += nonlin - (self.Phi @ (self.S @ (self.Psi.T @ nonlin.T))).T

        return q

    def vjp_encode(self, q: Any, v: Any) -> tuple:
        r"""
        VJP of the encoder :math:`z = \Psi^\top q` with respect to all
        parameters.  The encoder does not depend on :math:`A_k`, so those
        gradients are zero.

        :param q: full-space vector of shape ``(N,)`` or ``(m, N)``
        :param v: upstream adjoint seed :math:`v = \partial J / \partial z`
            of shape ``(r,)`` or ``(m, r)``
        :returns: ``(grad_Phi, grad_Psi, grad_A_k1, grad_A_k2, ...)``
        :rtype: tuple
        """
        bkend = self.backend
        if q.ndim == 1:
            grad_Psi = bkend.outer(q, v)
        else:
            grad_Psi = q.T @ v
        grads = (bkend.zeros_like(self.Phi), grad_Psi)
        for k in self.nonlin_poly_comp:
            grads += (bkend.zeros_like(getattr(self, f"A{k}")),)
        return grads

    def vjp_decode(self, z: Any, v: Any) -> tuple:
        r"""
        VJP of the decoder with respect to all parameters.

        Let :math:`g(z) = \sum_k A_k z^{\otimes k}`,
        :math:`P = I - \Phi S \Psi^\top`, :math:`h = z - \Psi^\top g`,
        and :math:`p = S^\top \Phi^\top v`.  Then:

        .. math::

            \frac{\partial J}{\partial A_k}
                = P^\top v \otimes \underbrace{z \otimes \cdots \otimes z}_{k},
            \quad
            \frac{\partial J}{\partial \Phi}
                = (v - \Psi p)(S h)^\top,
            \quad
            \frac{\partial J}{\partial \Psi}
                = -q\, p^\top

        where :math:`q = \text{decode}(z)`.

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :param v: upstream adjoint seed of shape ``(N,)`` or ``(m, N)``
        :returns: ``(grad_Phi, grad_Psi, grad_A_k1, grad_A_k2, ...)``
        :rtype: tuple
        """
        bkend = self.backend
        # Compute g(z) = Σ_k A_k z^{⊗k}
        if z.ndim == 1:
            g = bkend.zeros((self._n,), dtype=self.dtype, device=self.device)
        else:
            g = bkend.zeros(
                (z.shape[0], self._n), dtype=self.dtype, device=self.device
            )
        for i, k in enumerate(self.nonlin_poly_comp):
            A_k = getattr(self, f"A{k}")
            if z.ndim == 1:
                equation = ",".join(self.einsum_ss[i])
                operands = [A_k] + [z for _ in range(k)]
                g += bkend.einsum(equation, *operands)
            else:
                parts = self.einsum_ss[i]
                eq_parts = [parts[0]] + [f"...{p}" for p in parts[1:]]
                equation = ",".join(eq_parts)
                operands = [A_k] + [z for _ in range(k)]
                g += bkend.einsum(equation, *operands)

        # Projected adjoint: w = P^T v = v - Psi S^T Phi^T v
        if v.ndim == 1:
            p = self.S.T @ (self.Phi.T @ v)  # (r,)
            w = v - self.Psi @ p  # (N,)
        else:
            p = self.S.T @ (self.Phi.T @ v.T)  # (r, m)
            w = v - (self.Psi @ p).T  # (m, N)

        # --- grad_A_k: same VJP as PolynomialModel, using w as the seed ---
        grad_As = []
        for i, k in enumerate(self.nonlin_poly_comp):
            ss = self.einsum_ss[i]
            out_subscript = ss[0]
            if z.ndim == 1:
                in_subscripts = [ss[0][0], *ss[0][1:]]
            else:
                # Explicit summed batch index (numpy einsum rejects a broadcast
                # ellipsis dropped from an explicit output).
                batch = ascii_lowercase[len(out_subscript)]
                in_subscripts = [batch + s for s in out_subscript]
            equation = ",".join(in_subscripts) + "->" + out_subscript
            operands = [w] + [z for _ in range(k)]
            grad_As.append(bkend.einsum(equation, *operands))

        # --- grad_Phi, grad_Psi ---
        # Full decode: q = Phi S h + g, where h = z - Psi^T g.
        # Differentiating v^T q w.r.t. Phi:
        #   dq/dPhi: same structure as linear case with h instead of z.
        #   grad_Phi = (v - Psi S^T Phi^T v)(S h)^T
        # Differentiating v^T q w.r.t. Psi (through S and Psi^T g):
        #   v^T dq = -p^T dPsi^T q, where p = S^T Phi^T v, q = decode(z)
        #   grad_Psi = -q p^T  (or -q.T @ p.T for batched)
        if z.ndim == 1:
            h = z - self.Psi.T @ g
            Sh = self.S @ h  # (r,)
            W = bkend.outer(v, Sh)  # (N, r)
            q_decoded = self.Phi @ Sh + g  # (N,)
        else:
            h = z - (self.Psi.T @ g.T).T  # (m, r)
            W = v.T @ (h @ self.S.T)  # (N, r)
            q_decoded = (self.Phi @ (self.S @ h.T)).T + g  # (m, N)

        PhiTW = self.Phi.T @ W  # (r, r)
        grad_Phi = W - self.Psi @ (self.S.T @ PhiTW)  # (N, r)

        if v.ndim == 1:
            grad_Psi = -bkend.outer(q_decoded, p)  # (N, r)
        else:
            grad_Psi = -q_decoded.T @ p.T  # (N, r)

        return (grad_Phi, grad_Psi) + tuple(grad_As)

    def vjp_decode_state(self, z: Any, v: Any) -> Any:
        r"""
        VJP of the decoder with respect to the latent state :math:`z`.

        With :math:`\text{decode}(z) = \Phi S z + \mathbb{P} g(z)`,
        :math:`g(z) = \sum_k A_k z^{\otimes k}`,
        :math:`\mathbb{P} = I - \Phi S \Psi^\top`, the Jacobian-transpose is

        .. math::

            \left(\frac{\partial\,\text{decode}}{\partial z}\right)^\top v
                = S^\top \Phi^\top v
                + \left(\frac{\partial g}{\partial z}\right)^\top
                    \underbrace{(v - \Psi S^\top \Phi^\top v)}_{w = \mathbb{P}^\top v}.

        The contraction :math:`(\partial g/\partial z)^\top w` reuses the
        same combinatorial structure as :meth:`PolynomialModel.evaluate_adjoint_rhs`:
        for each degree :math:`k`, the free latent index is left out of the
        product over the remaining :math:`k-1` copies of :math:`z`, summed
        over which input slot is free.

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :param v: full-space cotangent of shape ``(N,)`` or ``(m, N)``
        :returns: latent-space vector of shape ``(r,)`` or ``(m, r)``
        :rtype: backend array
        """
        bkend = self.backend
        batched = z.ndim == 2

        # Linear part S^T Phi^T v, and w = P^T v.
        if not batched:
            p = self.S.T @ (self.Phi.T @ v)  # (r,)
            w = v - self.Psi @ p  # (N,)
        else:
            p = (v @ self.Phi) @ self.S  # (m, r)
            w = v - p @ self.Psi.T  # (m, N)
        out = bkend.copy(p)

        # Nonlinear part: sum_k (dg_k/dz)^T w.
        for i, k in enumerate(self.nonlin_poly_comp):
            A_k = getattr(self, f"A{k}")
            ss0 = self.einsum_ss[i][0]  # e.g. "abc"
            out_char = ss0[0]  # full-space output index
            input_chars = ss0[1:]  # latent input indices
            for comb in combinations(input_chars, k - 1):
                free = next(c for c in input_chars if c not in comb)
                if not batched:
                    eq = ",".join([ss0, out_char, *comb]) + "->" + free
                else:
                    eq = (
                        ",".join([ss0, "..." + out_char, *("..." + c for c in comb)])
                        + "->..."
                        + free
                    )
                out = out + bkend.einsum(eq, A_k, w, *([z] * (k - 1)))

        return out
