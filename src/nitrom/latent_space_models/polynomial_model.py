from itertools import combinations
from string import ascii_lowercase
from typing import Any

from .model import Model


class PolynomialModel(Model):
    r"""
    Class for a polynomial ROM of the form

    .. math::

        f(z, u) = v + Az + H:z z^\top + \ldots + B u,

    where :math:`z` is the state and :math:`u` some external forcing.

    :param r: reduced state dimension
    :type r: int
    :param poly_comp: list of polynomial degrees, e.g. ``[1, 2]`` for linear + quadratic
    :type poly_comp: list[int]
    :param device: device for array allocation (ignored by the NumPy backend)
    :type device: str
    :param dtype: data type for arrays; defaults to the backend's ``float64``
    :type dtype: backend dtype or None
    :param instability_threshold: norm threshold above which the state is considered blown up
    :type instability_threshold: float
    :param tensors: optional list of operator tensors :math:`[A_1, A_2, \ldots]`.
        If ``None``, tensors are initialized to zero.  When
        ``forcing_config`` is set, the last tensor is ``B``.
    :type tensors: list or None
    :param forcing_config: optional dict with keys ``"forcing_exists"``
        (bool) and ``"m"`` (int, forcing input dimension).  When provided
        and ``forcing_exists`` is ``True``, the last entry of ``tensors``
        (or a zero-initialized ``(r, m)`` matrix) is treated as ``B``.
    :type forcing_config: dict or None
    """

    def __init__(
        self,
        r: int,
        poly_comp: list[int],
        device: str = "cpu",
        dtype: Any = None,
        instability_threshold: float = 1e6,
        tensors: list | None = None,
        forcing_config: dict | None = None,
    ):
        forcing_exists = forcing_config is not None and forcing_config.get(
            "forcing_exists", False
        )
        # An optional fixed input operator may be supplied via
        # ``forcing_config["B"]``; B remains a parameter but is flagged
        # non-learnable so that its gradient is zeroed.
        B_fixed = forcing_config.get("B") if forcing_config else None

        param_names = [f"A_{k}" for k in poly_comp]
        if forcing_exists:
            param_names.append("B")
        super().__init__(r, param_names, device, dtype)

        self.poly_comp = poly_comp
        self.thresh = instability_threshold
        # Compared against squared norms: sqrt() per RHS call is pure overhead on
        # a divergence guard, and np.linalg.norm carries __array_function__
        # dispatch that shows up in profiles of the adjoint sweep.
        self._thresh_sq = instability_threshold * instability_threshold
        self.forcing_exists = forcing_exists

        # Initialize tensors (zero, or the fixed B) if not provided
        if tensors is None:
            tensors = [
                self.backend.zeros(
                    (r,) * (k + 1), dtype=self.dtype, device=self.device
                )
                for k in poly_comp
            ]
            if forcing_exists:
                if B_fixed is not None:
                    tensors.append(
                        self.backend.asarray(
                            B_fixed, dtype=self.dtype, device=self.device
                        )
                    )
                else:
                    m = forcing_config["m"]
                    tensors.append(
                        self.backend.zeros(
                            (r, m), dtype=self.dtype, device=self.device
                        )
                    )
        self.update_params(tensors)
        # A supplied fixed B always overrides any value from tensors.
        if forcing_exists and B_fixed is not None:
            self.bkend = self.backend.asarray(
                B_fixed, dtype=self.dtype, device=self.device
            )
        self._generate_einsum_subscripts()

    def get_params(self) -> list[Any]:
        """Return the current parameter tensors as a list."""
        return [getattr(self, name) for name in self.param_names]

    def _generate_einsum_subscripts(self) -> None:
        """
        Generates the indices for the einsum evaluation of the
        right-hand side and the adjoint.

        The equations depend only on :attr:`poly_comp`, so they are built once
        here rather than re-derived (with ``combinations`` and string joins) on
        every one of the thousands of evaluations a single adjoint sweep makes.
        """
        ss = []
        for k in self.poly_comp:
            ssk = ascii_lowercase[: k + 1]
            ssk = [ssk] + [s for s in ssk[1:]]
            ss.append(ssk)
        self.einsum_ss = tuple(ss)

        # RHS:  A_k (z, ..., z)
        self._rhs_eq = tuple(
            ",".join(parts) for parts in self.einsum_ss
        )
        self._rhs_eq_batched = tuple(
            ",".join([parts[0]] + [f"...{p}" for p in parts[1:]])
            for parts in self.einsum_ss
        )

        # Adjoint:  J(Z)^T w, contracted directly rather than by assembling J.
        # For degree k, leaving one latent slot free and filling the other
        # k - 1 with Z gives one term of the Jacobian; contracting the ambient
        # index against the seed at the same time yields the free slot of the
        # result.
        adj, adj_b = [], []
        for i, k in enumerate(self.poly_comp):
            terms, terms_b = [], []
            if k > 0:
                ss0 = self.einsum_ss[i][0]
                out_char, input_chars = ss0[0], ss0[1:]
                for comb in combinations(input_chars, k - 1):
                    free = next(c for c in input_chars if c not in comb)
                    terms.append(
                        ",".join([ss0, out_char, *comb]) + "->" + free
                    )
                    terms_b.append(
                        ",".join(
                            [ss0, f"...{out_char}", *(f"...{c}" for c in comb)]
                        )
                        + f"->...{free}"
                    )
            adj.append(tuple(terms))
            adj_b.append(tuple(terms_b))
        self._adjoint_eq = tuple(adj)
        self._adjoint_eq_batched = tuple(adj_b)

        # Parameter VJP:  dJ/dA_k = v (x) z (x) ... (x) z
        vjp, vjp_b = [], []
        for parts in self.einsum_ss:
            out_subscript = parts[0]
            vjp.append(",".join(out_subscript) + "->" + out_subscript)
            # Explicit summed batch index (numpy einsum rejects a broadcast
            # ellipsis dropped from an explicit output).
            batch = ascii_lowercase[len(out_subscript)]
            vjp_b.append(
                ",".join(batch + s for s in out_subscript) + "->" + out_subscript
            )
        self._vjp_eq = tuple(vjp)
        self._vjp_eq_batched = tuple(vjp_b)

    def update_params(self, tensors: list) -> None:
        r"""
        Update the operator tensors of the polynomial model.

        The tensors are stored C-contiguous.  A caller that assembles them from
        contractions -- :class:`~nitrom.latent_space_models.gas_polynomial_model.GasPolynomialModel`
        and its subclasses do, once per optimizer step -- can otherwise hand over
        a permuted-stride view, which every RHS and adjoint evaluation until the
        next update then has to copy on reshape (see
        :meth:`~nitrom.backend.Backend.ascontiguous`).

        :param tensors: new list of operator tensors :math:`[A_1, A_2, \ldots]`,
            in the same order as :attr:`param_names`
        :type tensors: list
        """
        for name, tensor in zip(self.param_names, tensors, strict=True):
            setattr(self, name, self.backend.ascontiguous(tensor))

    def evaluate_rhs(self, t: float, z: Any, **kwargs) -> Any:
        r"""
        Evaluate the ROM right-hand side:

        .. math::

            \dot{z} = \sum_k A_k \underbrace{(z, \ldots, z)}_{k} + B f(t)

        :param t: time instance
        :type t: float
        :param z: state vector of shape ``(n,)`` or ``(m, n)``
        :keyword external_forcing: list of callables :math:`[f_0, f_1, \ldots]` where
            :math:`f_i(t)` returns the forcing for the *i*-th trajectory
        :type external_forcing: list[callable] or None
        :rtype: backend array
        """

        f_fun_lst = kwargs.get("external_forcing")
        tensors = self.get_params()
        bkend = self.backend

        # z is a vector
        if z.ndim == 1:
            # Guard against blow-up
            if (z * z).sum() >= self._thresh_sq:
                return bkend.zeros_like(z)

            # Compute the dynamics
            dzdt = bkend.zeros_like(z)
            for i, k in enumerate(self.poly_comp):
                operands = [tensors[i]] + [z for _ in range(k)]
                dzdt += bkend.einsum(self._rhs_eq[i], *operands)

            # Add the forcing
            if f_fun_lst is not None:
                f = bkend.atleast_1d(f_fun_lst[0](t))
                dzdt += self.B @ f if self.forcing_exists else f

        # z is a tensor (we use batching to evaluate all vectors at once)
        else:
            # Guard against blow-up. If all entries > thresh, then return zeros,
            # otherwise compute the rhs of the vectors that are < thresh.
            # Nothing having blown up is the overwhelmingly common case, and
            # there the boolean gather/scatter around the contraction is pure
            # overhead, so contract ``z`` whole when the mask is all-true.
            mask = (z * z).sum(axis=-1) < self._thresh_sq
            all_below = bool(mask.all())
            if not all_below and not mask.any():
                return bkend.zeros_like(z)

            # Compute the dynamics
            zk = z if all_below else z[mask]
            dzdt = bkend.zeros_like(z)
            for i, k in enumerate(self.poly_comp):
                operands = [tensors[i]] + [zk for _ in range(k)]
                term = bkend.einsum(self._rhs_eq_batched[i], *operands)
                if all_below:
                    dzdt += term
                else:
                    dzdt[mask] += term

            # Add the external forcing
            if f_fun_lst is not None:
                for i in range(len(f_fun_lst)):
                    if f_fun_lst[i] is None:
                        continue
                    if not all_below and not mask[i]:
                        continue
                    f = bkend.atleast_1d(f_fun_lst[i](t))
                    dzdt[i] += self.B @ f if self.forcing_exists else f

        return dzdt

    def batched_vjp_evaluate_rhs(
        self,
        Z: Any,
        V: Any,
        U: Any = None,
        out: list | None = None,
        max_bytes: int = 64 << 20,
    ) -> list[Any]:
        r"""
        Parameter VJP summed over a stack of ``(state, seed)`` records.

        Equivalent to summing :meth:`vjp_evaluate_rhs` over the records, i.e. for
        each degree :math:`k`

        .. math::

            \sum_d v_d \otimes \underbrace{z_d \otimes \cdots \otimes z_d}_{k},

        but evaluated as a single matrix product per degree instead of one
        outer-product-and-accumulate per record.  The adjoint sweep produces one
        record per Runge-Kutta stage per sub-step, so at ``r = 50`` this replaces
        several hundred rank-1 updates of an ``(r, r, r)`` accumulator -- which are
        memory-bound and dominate the sweep -- with one GEMM.

        This is the multilinear part only: unlike :meth:`vjp_evaluate_rhs` it takes
        no ``reg``, because a regularization term must be added once, not once per
        record.

        :param Z: stacked states, shape ``(D, r)``
        :param V: stacked upstream adjoint seeds, shape ``(D, r)``
        :param U: stacked forcing rows ``u(t_d)``, shape ``(D, m)``; required iff
            :attr:`forcing_exists`
        :param out: accumulators to add into, in :meth:`get_params` order; a new
            list is allocated when omitted
        :param max_bytes: soft cap on the ``(chunk, r**k)`` intermediate, which
            bounds memory independently of ``D``
        :returns: list of gradients ``[grad_A_0, ..., grad_A_K, grad_B]``
        :rtype: list
        """
        bkend = self.backend
        D, r = Z.shape

        if self.forcing_exists and U is None:
            raise ValueError(
                "batched_vjp_evaluate_rhs needs the stacked forcing U when "
                "forcing_exists is True"
            )

        # Bound the largest intermediate, (chunk, r**k_max), not the whole stack.
        k_max = max(self.poly_comp) if self.poly_comp else 1
        itemsize = getattr(Z.dtype, "itemsize", 8)
        per_record = max(1, itemsize * r ** max(k_max, 1))
        chunk = max(1, min(D, max_bytes // per_record))

        grads: list[Any] = [None] * len(self.poly_comp)
        grad_B = None

        for start in range(0, D, chunk):
            Zc = Z[start : start + chunk]
            Vc = V[start : start + chunk]
            Vt = bkend.permute(Vc, (1, 0))  # (r, C)

            for i, k in enumerate(self.poly_comp):
                if k == 0:
                    g = Vc.sum(axis=0)
                elif k == 1:
                    g = Vt @ Zc
                else:
                    # P[d, b*r + c + ...] = Z[d, b] * Z[d, c] * ...  so that
                    # (V^T @ P) reshaped is  sum_d V[d, a] Z[d, b] Z[d, c] ...
                    P = Zc
                    for _ in range(k - 1):
                        P = (P[:, :, None] * Zc[:, None, :]).reshape(Zc.shape[0], -1)
                    g = (Vt @ P).reshape((r,) * (k + 1))
                grads[i] = g if grads[i] is None else grads[i] + g

            if self.forcing_exists:
                gB = Vt @ U[start : start + chunk]
                grad_B = gB if grad_B is None else grad_B + gB

        if self.forcing_exists:
            grads.append(grad_B)

        if out is None:
            return grads
        return [o + g for o, g in zip(out, grads, strict=True)]

    def evaluate_adjoint_rhs(self, t: float, z: Any, Z: Any, **kwargs) -> Any:
        r"""
        Evaluate the adjoint right-hand side:

        .. math::

            \dot{z} = J(Z)^\top z

        where :math:`J(Z) = \nabla_z f(Z)` is the Jacobian of the RHS evaluated at
        the base flow :math:`Z`.

        :param t: time instance
        :type t: float
        :param z: adjoint state vector of shape ``(n,)`` or ``(m, n)``
        :param Z: base flow at which to evaluate the Jacobian, same shape as ``z``
        :rtype: backend array
        """
        tensors = self.get_params()
        bkend = self.backend

        # z is a vector
        if z.ndim == 1:
            # Guard against blow-up
            if (z * z).sum() >= self._thresh_sq:
                return bkend.zeros_like(z)

            # Contract J(Z)^T z directly: assembling J only to apply it once
            # costs an extra (n, n) intermediate per term.
            dzdt = bkend.zeros_like(z)
            for i, k in enumerate(self.poly_comp):
                operands = [tensors[i], z] + [Z for _ in range(k - 1)]
                for equation in self._adjoint_eq[i]:
                    dzdt += bkend.einsum(equation, *operands)

        # z is a tensor (we use batching to evaluate all vectors at once)
        else:
            # Guard against blow-up (see :meth:`evaluate_rhs` for the fast path)
            mask = (z * z).sum(axis=-1) < self._thresh_sq
            all_below = bool(mask.all())
            if not all_below and not mask.any():
                return bkend.zeros_like(z)

            # Contract J(Z)^T z directly, skipping the batched (B, n, n)
            # Jacobian the result is immediately contracted against.
            zk, Zk = (z, Z) if all_below else (z[mask], Z[mask])
            dzdt = bkend.zeros_like(z)
            for i, k in enumerate(self.poly_comp):
                operands = [tensors[i], zk] + [Zk for _ in range(k - 1)]
                for equation in self._adjoint_eq_batched[i]:
                    term = bkend.einsum(equation, *operands)
                    if all_below:
                        dzdt += term
                    else:
                        dzdt[mask] += term

        return dzdt

    def vjp_evaluate_rhs(self, z: Any, v: Any, reg: float = 0.0, **kwargs) -> list[Any]:
        r"""
        VJP of :meth:`evaluate_rhs` with respect to the operator tensors
        and, if forcing is present, with respect to :math:`B`.

        For each degree-:math:`k` tensor :math:`A_k`, the gradient is

        .. math::

            \frac{\partial J}{\partial A_k}
                = v \otimes \underbrace{z \otimes \cdots \otimes z}_{k}.

        For the input matrix :math:`B` (where the forward pass contributes
        :math:`B\,u(t)`), the gradient is

        .. math::

            \frac{\partial J}{\partial B} = v\, u(t)^\top.

        For the batched case, contributions are summed over the batch.

        :param z: state vector of shape ``(n,)`` or ``(m, n)``
        :param v: upstream adjoint seed, same shape as ``z``
        :keyword external_forcing: list of callables returning the forcing
        :keyword t: time at which to evaluate the forcing (required if
            ``external_forcing`` is provided)
        :returns: list of gradients ``[grad_A_0, ..., grad_A_K, grad_B]``
            where ``grad_B`` is only included when forcing is present
        :rtype: list
        """
        grads = []

        f_fun_lst = kwargs.get("external_forcing") if self.forcing_exists else None
        t = kwargs.get("t", 0.0)
        bkend = self.backend

        if z.ndim == 1:
            for i, k in enumerate(self.poly_comp):
                operands = [v] + [z for _ in range(k)]
                grads.append(bkend.einsum(self._vjp_eq[i], *operands))

            # grad_B = v @ u(t)^T
            if f_fun_lst is not None:
                u = bkend.atleast_1d(f_fun_lst[0](t))
                grads.append(bkend.outer(v, u))
        else:
            for i, k in enumerate(self.poly_comp):
                operands = [v] + [z for _ in range(k)]
                grads.append(bkend.einsum(self._vjp_eq_batched[i], *operands))

            # grad_B = sum_j v_j @ u_j(t)^T
            if f_fun_lst is not None:
                grad_B = bkend.zeros_like(self.B)
                for j in range(z.shape[0]):
                    u = bkend.atleast_1d(f_fun_lst[j](t))
                    grad_B += bkend.outer(v[j], u)
                grads.append(grad_B)

        if reg > 0.0:
            for i, k in enumerate(self.poly_comp):
                if k == 2:
                    grads[i] = grads[i] + 2.0 * reg * self.get_params()[i]

        return grads
