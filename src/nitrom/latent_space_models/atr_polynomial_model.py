from typing import Any

from .gas_polynomial_model import GasPolynomialModel


class AtrPolynomialModel(GasPolynomialModel):
    r"""
    Attracting-trapping-region (ATR) constrained polynomial ROM.

    Where :class:`GasPolynomialModel` makes the origin globally attracting --
    the wrong prior for a FOM that settles onto a limit cycle rather than an
    equilibrium -- this model applies the same structure-preserving maps to the
    *shifted* state :math:`\widehat{z} = z - m` and adds a constant
    :math:`\widehat{B} \in \mathbb{R}^r`:

    .. math::

        \dot{\widehat{z}} = \widehat{A}\,\widehat{z}
            + \widehat{H} : \widehat{z}\widehat{z}^\top + \widehat{B},
        \qquad
        \widehat{A} = \bigl((K - K^\top) - R^{-1}R^{-\top}\bigr)\,\tilde{Q},
        \qquad
        \widehat{H}_{ijk} = (S_{ilk} - S_{lik})\,\tilde{Q}_{lj},

    with :math:`\tilde{Q} = Q^{-1}Q^{-\top}` and
    :math:`\tilde{R} = R^{-1}R^{-\top}`.  With :math:`\widehat{V} =
    \tfrac12 \widehat{z}^\top \tilde{Q} \widehat{z}` and
    :math:`\widehat{y} = \tilde{Q}\widehat{z}`,

    .. math::

        \dot{\widehat{V}} = -\widehat{y}^\top \tilde{R} \widehat{y}
            + \widehat{y}^\top \widehat{B},

    which is negative whenever :math:`\lVert \widehat{y} \rVert >
    \lVert \widehat{B} \rVert / \lambda_{\min}(\tilde{R})`.  Every sufficiently
    large sublevel set of :math:`\widehat{V}` is therefore a monotonically
    attracting trapping region, contained in the ball of radius
    :meth:`trapping_region_radius` about :math:`m`.

    The free parameters are ``K``, ``R``, ``Q``, ``S``, ``Bhat``, ``m`` (plus an
    optional input operator ``B``).  The model assembles and stores the
    equivalent operators in the **unshifted** coordinates,

    .. math::

        \dot{z} = c + A z + H : zz^\top,
        \qquad
        A = \widehat{A} - H : (I \otimes m) - H : (m \otimes I),
        \qquad
        c = \widehat{B} - \widehat{A} m + H : mm^\top,

    so the inner :class:`~nitrom.latent_space_models.polynomial_model.PolynomialModel`
    (degrees ``[0, 1, 2]``) evaluates the dynamics directly, and ``A``, ``H``,
    ``c`` are immediately usable outside the model.

    :param r: reduced state dimension
    :type r: int
    :param poly_comp: polynomial degrees; must contain both ``1`` and ``2``
    :type poly_comp: list[int]
    :param device: device for array allocation (ignored by the NumPy backend)
    :type device: str
    :param dtype: data type for arrays; defaults to the backend's ``float64``
    :type dtype: backend dtype or None
    :param instability_threshold: norm threshold for blow-up guard
    :type instability_threshold: float
    :param atr_params: optional list of initial parameter tensors, ordered as
        :attr:`param_names` (``[K, R, Q, S, Bhat, m]``, plus ``B`` when forcing
        is present).  If ``None``, ``K``, ``R``, ``Q``, ``S`` are initialized
        randomly and ``Bhat``, ``m`` to zero.
    :type atr_params: list or None
    :param m0: optional initial shift, overriding the ``m`` entry of
        ``atr_params``
    :type m0: array-like or None
    :param forcing_config: optional dict with keys ``"forcing_exists"`` (bool)
        and ``"m"`` (int, forcing input dimension).  Note that this ``"m"`` is
        the input dimension of ``B``, unrelated to the shift ``m``.  See
        :class:`~nitrom.latent_space_models.polynomial_model.PolynomialModel`.
    :type forcing_config: dict or None
    """

    def __init__(
        self,
        r: int,
        poly_comp: list[int],
        device: str = "cpu",
        dtype: Any = None,
        instability_threshold: float = 1e6,
        atr_params: list | None = None,
        m0: Any = None,
        forcing_config: dict | None = None,
    ):
        if not (1 in poly_comp and 2 in poly_comp):
            raise ValueError(
                f"AtrPolynomialModel requires both a linear and a quadratic term, "
                f"i.e. 1 and 2 in poly_comp, got {poly_comp}."
            )

        super().__init__(
            r,
            poly_comp,
            device=device,
            dtype=dtype,
            instability_threshold=instability_threshold,
            gas_params=atr_params,
            forcing_config=forcing_config,
        )

        # A supplied m0 always overrides any value from atr_params.
        if m0 is not None:
            self.set_shift(m0)

    # ------------------------------------------------------------------
    # Parameterization hooks (see GasPolynomialModel)
    # ------------------------------------------------------------------
    @staticmethod
    def _gas_param_specs(
        r: int, poly_comp: list[int]
    ) -> list[tuple[str, tuple[int, ...], str]]:
        """GAS parameters, plus the constant ``Bhat`` and the shift ``m``."""
        specs = GasPolynomialModel._gas_param_specs(r, poly_comp)
        specs.extend([("Bhat", (r,), "zeros"), ("m", (r,), "zeros")])
        return specs

    @staticmethod
    def _inner_poly_comp(poly_comp: list[int]) -> list[int]:
        """Prepend the degree-0 (constant) term needed by the unshifted form."""
        return [0, *(k for k in poly_comp if k != 0)]

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------
    def assemble_gas_tensors(self) -> list[Any]:
        r"""
        Build the unshifted operator tensors from the current parameters.

        :returns: list of tensors ``[c, A, H, ..., B]`` matching the inner
            :class:`PolynomialModel` param order.  ``B`` is appended only when
            forcing is present.
        :rtype: list
        """
        bkend = self.backend
        Qtil = self._Qtil
        m = self.m

        # Shifted-frame operators (identical to the GAS parameterization).
        Ahat = ((self.K - self.K.T) - self._Rtil) @ Qtil
        H = bkend.einsum("ilk,lj->ijk", self._S_diff, Qtil)

        # Undo the shift: A = Ahat - H:(I x m) - H:(m x I).
        A = (
            Ahat
            - bkend.einsum("ijk,k->ij", H, m)
            - bkend.einsum("ikj,k->ij", H, m)
        )
        c = self.Bhat - Ahat @ m + bkend.einsum("ijk,j,k->i", H, m, m)

        # Cache the unshifted operators for the gradient projection.
        self._Ahat, self._A, self._H = Ahat, A, H

        tensors = [None] * len(self.poly_comp)
        tensors[self.poly_comp.index(0)] = c
        tensors[self.poly_comp.index(1)] = A
        tensors[self.poly_comp.index(2)] = H

        if self.forcing_exists:
            tensors.append(self.B)

        return tensors

    # ------------------------------------------------------------------
    # Gradients
    # ------------------------------------------------------------------
    def project_inner_gradients(self, inner_grads: list[Any]) -> list[Any]:
        r"""
        Project gradients w.r.t. the unshifted tensors ``(c, A, H, [B])`` back
        to the ATR parameters ``(K, R, Q, S, Bhat, m, [B])``.

        Differentiating the assembly in :meth:`assemble_gas_tensors` gives the
        effective shifted-frame gradients

        .. math::

            g_{\widehat{A}} &= g_A - g_c\, m^\top, \\
            (g_{\widehat{H}})_{ijk} &= (g_H)_{ijk} - (g_A)_{ij} m_k
                - (g_A)_{ik} m_j + (g_c)_i m_j m_k, \\
            g_{\widehat{B}} &= g_c, \\
            (g_m)_k &= -(g_A)_{ij}\bigl(H_{ijk} + H_{ikj}\bigr)
                - (A^\top g_c)_k,

        where the :math:`m` gradient through ``c`` collapses to
        :math:`-A^\top g_c` because :math:`\partial c/\partial m = -A`.  The
        remaining chain from :math:`(g_{\widehat{A}}, g_{\widehat{H}})` to
        ``K``, ``R``, ``Q``, ``S`` is the GAS one.

        :param inner_grads: gradients w.r.t. :meth:`inner_params`
        :type inner_grads: list
        :returns: gradients matching :attr:`param_names`
        :rtype: list
        """
        bkend = self.backend
        m = self.m

        i_c = self.poly_comp.index(0)
        i_A = self.poly_comp.index(1)
        i_H = self.poly_comp.index(2)
        grad_c, grad_A, grad_H = (
            inner_grads[i_c], inner_grads[i_A], inner_grads[i_H],
        )

        # Effective gradients w.r.t. the shifted-frame operators.
        shifted = list(inner_grads)
        shifted[i_A] = grad_A - bkend.outer(grad_c, m)
        shifted[i_H] = (
            grad_H
            - bkend.einsum("ij,k->ijk", grad_A, m)
            - bkend.einsum("ik,j->ijk", grad_A, m)
            + bkend.einsum("i,j,k->ijk", grad_c, m, m)
        )

        grad_m = (
            -bkend.einsum("ij,ijk->k", grad_A, self._H)
            - bkend.einsum("ij,ikj->k", grad_A, self._H)
            - self._A.T @ grad_c
        )

        # GAS chain rule for K, R, Q, S (and B, which it passes through).
        grads = super().project_inner_gradients(shifted)

        n_gas = len(grads) - (1 if self.forcing_exists else 0)
        return [*grads[:n_gas], grad_c, grad_m, *grads[n_gas:]]

    # ------------------------------------------------------------------
    # Shift handling
    # ------------------------------------------------------------------
    def set_shift(self, m: Any) -> None:
        """
        Set the shift ``m`` and reassemble the operator tensors.

        :param m: new shift of shape ``(r,)``
        """
        bkend = self.backend
        m = bkend.asarray(m, dtype=self.dtype, device=self.device).reshape(-1)
        params = [
            m if name == "m" else getattr(self, name) for name in self.param_names
        ]
        self.update_params(params)

    def set_shift_from_data(self, Z: Any) -> Any:
        r"""
        Initialize the shift ``m`` to the mean of the latent training data --
        a natural starting guess for the center of the trapping region.

        :param Z: latent data of shape ``(ntraj, r, nt)`` (mean taken over
            trajectories and time), ``(n, r)`` (mean over samples), or ``(r,)``
        :returns: the shift that was set
        :rtype: backend array
        """
        bkend = self.backend
        Z = bkend.asarray(Z, dtype=self.dtype, device=self.device)
        if Z.ndim == 3:
            n = Z.shape[0] * Z.shape[2]
            m = bkend.sum(Z, axis=(0, 2)) / n
        elif Z.ndim == 2:
            m = bkend.sum(Z, axis=0) / Z.shape[0]
        elif Z.ndim == 1:
            m = Z
        else:
            raise ValueError(
                f"Z must have 1, 2, or 3 dimensions, got {Z.ndim}."
            )
        self.set_shift(m)
        return self.m

    def trapping_region_radius(self) -> float:
        r"""
        Radius of a ball about ``m`` that contains the attracting trapping
        region,

        .. math::

            \rho = \frac{\lVert \widehat{B} \rVert}
                {\lambda_{\min}(\tilde{R})\,\lambda_{\min}(\tilde{Q})},

        a conservative bound obtained from
        :math:`\dot{\widehat{V}} \le -\lambda_{\min}(\tilde{R})
        \lVert \widehat{y} \rVert^2 + \lVert \widehat{B} \rVert
        \lVert \widehat{y} \rVert` with :math:`\widehat{y} = \tilde{Q}\widehat{z}`.

        :rtype: float
        """
        bkend = self.backend
        lam_R = float(bkend.eigh(0.5 * (self._Rtil + self._Rtil.T))[0].min())
        lam_Q = float(bkend.eigh(0.5 * (self._Qtil + self._Qtil.T))[0].min())
        return float(bkend.vector_norm(self.Bhat)) / (lam_R * lam_Q)

    # ------------------------------------------------------------------
    # Retraction
    # ------------------------------------------------------------------
    def retract_general_tensors_to_atr_tensors(
        self,
        tensors: list,
        m: Any = None,
        **kwargs,
    ) -> None:
        r"""
        Retract general (unshifted) polynomial operators onto the ATR parameter
        manifold and **set** the model's parameters -- the ATR analogue of
        :meth:`~nitrom.latent_space_models.gas_polynomial_model.GasPolynomialModel.retract_general_tensors_to_gas_tensors`,
        used to seed ATR training from an unconstrained operator-inference fit.

        The operators are first shifted by ``m``,

        .. math::

            \widehat{A} = A + H : (I \otimes m) + H : (m \otimes I),
            \qquad
            \widehat{B} = c + A m + H : mm^\top,

        after which the GAS retraction is applied to
        :math:`(\widehat{A}, \widehat{H}) = (\widehat{A}, H)`; ``Bhat`` and
        ``m`` are preserved by that call.  As for the GAS retraction,
        :math:`\widehat{A}` (and hence :math:`\widehat{B}`) is reproduced
        exactly when it is already Hurwitz, and its spectrum is shifted into the
        open left-half plane otherwise; ``H`` is reproduced exactly only when it
        is already skew-symmetric under the Lyapunov metric computed internally,
        and is otherwise replaced by its energy-preserving projection.

        :param tensors: ``[A, H]`` or ``[c, A, H]``, with ``c`` of shape
            ``(r,)``, ``A`` of shape ``(r, r)`` and ``H`` of shape
            ``(r, r, r)``.  ``c`` defaults to zero when omitted.
        :type tensors: list
        :param m: shift to retract about; defaults to the model's current ``m``
            (e.g. as set by :meth:`set_shift_from_data`)
        :type m: array-like or None
        :param kwargs: forwarded to
            :meth:`GasPolynomialModel.retract_general_tensors_to_gas_tensors`
            (``margin``, ``optimize_F``, ``F_cond_penalty``, ``use_P_I``)
        """
        bkend = self.backend

        if len(tensors) == 2:
            A, H = tensors
            c = bkend.zeros((self.state_dimension,), dtype=self.dtype, device=self.device)
        elif len(tensors) == 3:
            c, A, H = tensors
        else:
            raise ValueError(
                f"tensors must be [A, H] or [c, A, H], got {len(tensors)} entries."
            )

        A = bkend.asarray(A, dtype=self.dtype, device=self.device)
        H = bkend.asarray(H, dtype=self.dtype, device=self.device)
        c = bkend.asarray(c, dtype=self.dtype, device=self.device).reshape(-1)

        if m is None:
            m = self.m
        m = bkend.asarray(m, dtype=self.dtype, device=self.device).reshape(-1)

        # Shift into the frame centered on m.
        Ahat = (
            A
            + bkend.einsum("ijk,k->ij", H, m)
            + bkend.einsum("ikj,k->ij", H, m)
        )
        Bhat = c + A @ m + bkend.einsum("ijk,j,k->i", H, m, m)

        # Set the shift and constant first: the GAS retraction rebuilds only
        # K, R, Q, S and carries every other parameter through unchanged.
        params = []
        for name in self.param_names:
            if name == "m":
                params.append(m)
            elif name == "Bhat":
                params.append(Bhat)
            else:
                params.append(getattr(self, name))
        self.update_params(params)

        self.retract_general_tensors_to_gas_tensors([Ahat, H], **kwargs)
