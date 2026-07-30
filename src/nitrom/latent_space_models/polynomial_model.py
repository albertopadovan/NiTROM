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
        right-hand side and the adjoint
        """
        ss = []
        for k in self.poly_comp:
            ssk = ascii_lowercase[: k + 1]
            ssk = [ssk] + [s for s in ssk[1:]]
            ss.append(ssk)
        self.einsum_ss = tuple(ss)

    def update_params(self, tensors: list) -> None:
        r"""
        Update the operator tensors of the polynomial model.

        :param tensors: new list of operator tensors :math:`[A_1, A_2, \ldots]`,
            in the same order as :attr:`param_names`
        :type tensors: list
        """
        for name, tensor in zip(self.param_names, tensors, strict=True):
            setattr(self, name, tensor)

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
            if bkend.vector_norm(z) >= self.thresh:
                return bkend.zeros_like(z)

            # Compute the dynamics
            dzdt = bkend.zeros_like(z)
            for i, k in enumerate(self.poly_comp):
                equation = ",".join(self.einsum_ss[i])
                operands = [tensors[i]] + [z for _ in range(k)]
                dzdt += bkend.einsum(equation, *operands)

            # Add the forcing
            if f_fun_lst is not None:
                f = bkend.atleast_1d(f_fun_lst[0](t))
                dzdt += self.B @ f if self.forcing_exists else f

        # z is a tensor (we use batching to evaluate all vectors at once)
        else:
            # Guard against blow-up. If all entries > thresh, then return zeros,
            # otherwise compute the rhs of the vectors that are < thresh
            norms = bkend.vector_norm(z, axis=-1)
            mask = norms < self.thresh
            if not mask.any():
                return bkend.zeros_like(z)

            # Compute the dynamics
            dzdt = bkend.zeros_like(z)
            for i, k in enumerate(self.poly_comp):
                parts = self.einsum_ss[i]
                eq_parts = [parts[0]] + [f"...{p}" for p in parts[1:]]
                equation = ",".join(eq_parts)
                operands = [tensors[i]] + [z[mask] for _ in range(k)]
                dzdt[mask] += bkend.einsum(equation, *operands)

            # Add the external forcing
            if f_fun_lst is not None:
                for i in range(len(f_fun_lst)):
                    if mask[i] and f_fun_lst[i] is not None:
                        f = bkend.atleast_1d(f_fun_lst[i](t))
                        dzdt[i] += self.B @ f if self.forcing_exists else f

        return dzdt

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
        n = z.shape[-1]
        tensors = self.get_params()
        bkend = self.backend

        # z is a vector
        if z.ndim == 1:
            # Guard against blow-up
            if bkend.vector_norm(z) >= self.thresh:
                return bkend.zeros_like(z)

            # Compute the Jacobian and adjoint dynamics
            J = bkend.zeros((n, n), dtype=self.dtype, device=self.device)
            for i, k in enumerate(self.poly_comp):
                if k == 0:
                    continue
                combs = list(combinations(self.einsum_ss[i][1:], r=k - 1))
                operands = [tensors[i]] + [Z for _ in range(k - 1)]
                for comb in combs:
                    equation = ",".join([self.einsum_ss[i][0], *comb])
                    J += bkend.einsum(equation, *operands)
            dzdt = J.T @ z

        # z is a tensor (we use batching to evaluate all vectors at once)
        else:
            # Guard against blow-up
            norms = bkend.vector_norm(z, axis=-1)
            mask = norms < self.thresh
            if not mask.any():
                return bkend.zeros_like(z)

            # Compute the Jacobian and adjoint dynamics
            dzdt = bkend.zeros_like(z)
            Jb = bkend.zeros(
                (int(mask.sum()), n, n), dtype=self.dtype, device=self.device
            )
            for i, k in enumerate(self.poly_comp):
                if k == 0:
                    continue
                combs = list(combinations(self.einsum_ss[i][1:], r=k - 1))
                for comb in combs:
                    eq_parts = [self.einsum_ss[i][0]] + [f"...{p}" for p in comb]
                    equation = ",".join(eq_parts)
                    operands = [tensors[i]] + [Z[mask] for _ in range(k - 1)]
                    Jb += bkend.einsum(equation, *operands)
            dzdt[mask] = bkend.einsum("bnm,bn->bm", Jb, z[mask])

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
                ss = self.einsum_ss[i]
                out_subscript = ss[0]
                in_subscripts = [ss[0][0], *ss[0][1:]]
                equation = ",".join(in_subscripts) + "->" + out_subscript
                operands = [v] + [z for _ in range(k)]
                grads.append(bkend.einsum(equation, *operands))

            # grad_B = v @ u(t)^T
            if f_fun_lst is not None:
                u = bkend.atleast_1d(f_fun_lst[0](t))
                grads.append(bkend.outer(v, u))
        else:
            for i, k in enumerate(self.poly_comp):
                ss = self.einsum_ss[i]
                out_subscript = ss[0]
                # Use an explicit batch index, omitted from the output so the
                # contribution is summed over the batch.  (Avoid the
                # "...a,...b->ab" ellipsis form: numpy einsum rejects a broadcast
                # ellipsis that is dropped from an explicit output.)
                batch = ascii_lowercase[len(out_subscript)]
                in_subscripts = [batch + s for s in out_subscript]
                equation = ",".join(in_subscripts) + "->" + out_subscript
                operands = [v] + [z for _ in range(k)]
                grads.append(bkend.einsum(equation, *operands))

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
