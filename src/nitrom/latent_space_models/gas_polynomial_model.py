from typing import Any
from scipy.linalg import solve_continuous_lyapunov
import numpy as np


from .model import Model
from .polynomial_model import PolynomialModel
from ..backend import mpi_rank_size


class GasPolynomialModel(Model):
    r"""
    GAS-constrained polynomial ROM.

    Holds free GAS parameters (``K``, ``R``, ``Q``, ``S``) and assembles
    physical operator tensors :math:`A` and :math:`H` via
    structure-preserving maps:

    .. math::

        A = \bigl((K - K^\top) - R^{-1}\,R^{-\top}\bigr)\,\tilde{Q},
        \qquad
        H_{ijk} = (S_{ilk} - S_{lik})\,\tilde{Q}_{lj},

    where :math:`\tilde{Q} = Q^{-1} Q^{-\top}`.

    All RHS, adjoint, and VJP evaluations are delegated to an internal
    :class:`PolynomialModel`.

    :param r: reduced state dimension
    :type r: int
    :param poly_comp: polynomial degrees, e.g. ``[1, 2]``
    :type poly_comp: list[int]
    :param device: device for array allocation (ignored by the NumPy backend)
    :type device: str
    :param dtype: data type for arrays; defaults to the backend's ``float64``
    :type dtype: backend dtype or None
    :param instability_threshold: norm threshold for blow-up guard
    :type instability_threshold: float
    :param gas_params: optional list of initial GAS parameter tensors
        (subset depending on ``poly_comp``).
        If ``None``, parameters are initialized randomly.
    :type gas_params: list or None
    :param forcing_config: optional dict with keys ``"forcing_exists"``
        (bool) and ``"m"`` (int).  See :class:`PolynomialModel`.
    :type forcing_config: dict or None
    """

    def __init__(
        self,
        r: int,
        poly_comp: list[int],
        device: str = "cpu",
        dtype: Any = None,
        instability_threshold: float = 1e6,
        gas_params: list | None = None,
        forcing_config: dict | None = None,
    ):
        # Determine GAS parameter names, shapes, and initializations
        specs = self._gas_param_specs(r, poly_comp)
        param_names: list[str] = [name for name, _, _ in specs]

        # Track forcing.  An optional fixed input operator may be supplied via
        # ``forcing_config["B"]``; B stays a parameter but is flagged
        # non-learnable so that its gradient is zeroed.
        forcing_exists = forcing_config is not None and forcing_config.get(
            "forcing_exists", False
        )
        B_fixed = forcing_config.get("B") if forcing_config else None
        if forcing_exists:
            param_names.append("B")

        super().__init__(r, param_names, device, dtype)
        bkend = self.backend
        self.poly_comp = self._inner_poly_comp(poly_comp)
        self.forcing_exists = forcing_exists

        # Set GAS parameters as attributes
        if gas_params is not None:
            for name, tensor in zip(param_names, gas_params, strict=True):
                setattr(
                    self, name,
                    bkend.asarray(tensor, dtype=self.dtype, device=self.device),
                )
        else:
            # Initialize the GAS parameters (K, R, Q, S, ...; B handled below)
            for name, shape, init in specs:
                alloc = bkend.zeros if init == "zeros" else bkend.randn
                setattr(
                    self, name,
                    alloc(shape, dtype=self.dtype, device=self.device),
                )
            # Initialize B (fixed value if supplied, else zeros)
            if forcing_exists:
                m = forcing_config["m"]
                self.B = (
                    bkend.asarray(B_fixed, dtype=self.dtype, device=self.device)
                    if B_fixed is not None
                    else bkend.zeros((r, m), dtype=self.dtype, device=self.device)
                )

        # A supplied fixed B always overrides any value from gas_params.
        if forcing_exists and B_fixed is not None:
            self.B = bkend.asarray(B_fixed, dtype=self.dtype, device=self.device)

        # Precompute inverses and products for Q, R, and S
        self._precompute_inverses()

        # Assemble physical tensors and create the inner PolynomialModel
        tensors = self.assemble_gas_tensors()
        self.model = PolynomialModel(
            r,
            self.poly_comp,
            device=device,
            dtype=self.dtype,
            instability_threshold=instability_threshold,
            tensors=tensors,
            forcing_config=forcing_config,
        )

    @staticmethod
    def _gas_param_specs(
        r: int, poly_comp: list[int]
    ) -> list[tuple[str, tuple[int, ...], str]]:
        r"""
        Specification of the free GAS parameters (everything except ``B``).

        Each entry is ``(name, shape, init)`` where ``init`` is ``"randn"`` or
        ``"zeros"``, used when no initial parameters are supplied.  Subclasses
        override this to extend the parameter set (see
        :class:`~nitrom.latent_space_models.atr_polynomial_model.AtrPolynomialModel`).

        :param r: reduced state dimension
        :param poly_comp: polynomial degrees requested by the caller
        :rtype: list[tuple[str, tuple[int, ...], str]]
        """
        specs: list[tuple[str, tuple[int, ...], str]] = []
        if 1 in poly_comp:
            specs.extend(
                [("K", (r, r), "randn"), ("R", (r, r), "randn"), ("Q", (r, r), "randn")]
            )
        if 2 in poly_comp:
            specs.append(("S", (r, r, r), "randn"))
        return specs

    @staticmethod
    def _inner_poly_comp(poly_comp: list[int]) -> list[int]:
        """
        Polynomial degrees carried by the inner :class:`PolynomialModel`.

        Identity for GAS; subclasses override it when the assembled dynamics
        need extra degrees (e.g. the ATR model, which adds a constant term).
        """
        return list(poly_comp)

    def _precompute_inverses(self) -> None:
        """Precompute and cache inverses and products for Q and R."""
        bkend = self.backend
        if hasattr(self, "Q"):
            self._Qinv = bkend.inv(self.Q)
            self._Qtil = self._Qinv @ self._Qinv.T
        if hasattr(self, "R"):
            self._Rinv = bkend.inv(self.R)
            self._Rtil = self._Rinv @ self._Rinv.T
        if hasattr(self, "S"):
            # S_diff_jik = S_jik - S_ijk
            self._S_diff = self.S - bkend.permute(self.S, (1, 0, 2))

    def get_params(self) -> list[Any]:
        """Return the current GAS parameter tensors as a list."""
        return [getattr(self, name) for name in self.param_names]

    def assemble_gas_tensors(self) -> list[Any]:
        r"""
        Build physical operator tensors from the current GAS parameters.

        :returns: list of tensors ``[A, H, ..., B]`` matching the inner
            :class:`PolynomialModel` param order.  ``B`` is appended
            only when forcing is present.
        :rtype: list
        """
        bkend = self.backend
        tensors = [None] * len(self.poly_comp)

        Qtil = self._Qtil

        if 1 in self.poly_comp:
            idx = self.poly_comp.index(1)
            tensors[idx] = ((self.K - self.K.T) - self._Rtil) @ Qtil

        if 2 in self.poly_comp:
            idx = self.poly_comp.index(2)
            tensors[idx] = bkend.einsum("ilk,lj->ijk", self._S_diff, Qtil)

        if self.forcing_exists:
            tensors.append(self.B)

        return tensors

    def update_params(self, params: list) -> None:
        r"""
        Update the GAS parameters (and B if present), reassemble
        physical tensors, and push them into the inner
        :class:`PolynomialModel`.

        :param params: parameter tensors matching :attr:`param_names`
            (e.g. ``[K, R, Q, S]`` or ``[K, R, Q, S, B]``)
        :type params: list
        """
        for name, tensor in zip(self.param_names, params, strict=True):
            setattr(self, name, tensor)
        self._precompute_inverses()
        self.model.update_params(self.assemble_gas_tensors())

    def retract_general_tensors_to_gas_tensors(
        self,
        tensors: list,
        margin: float = 1e-3,
        optimize_F: bool = False,
        F_cond_penalty: float = 1e-4,
        use_P_I: bool = False,
    ) -> None:
        r"""
        Retract general polynomial operator tensors ``[A, H]`` onto the GAS
        parameter manifold and **set** the model's GAS parameters.

        ``A`` is first shifted, if necessary, into the open left-half plane,
        :math:`A \leftarrow A - \alpha I` with

        .. math::

            \alpha = \begin{cases}
                0, & \max_i \mathrm{Re}\,\lambda_i(A) < 0,\\[2pt]
                \max_i \mathrm{Re}\,\lambda_i(A) + \texttt{margin}, &
                    \text{otherwise,}
            \end{cases}

        so that the (stabilized) ``A`` is Hurwitz.  The Lyapunov equation

        .. math::

            A^\top P + P A = -I

        is then solved for the SPD :math:`P`, and the GAS parameters are built
        from the skew/symmetric split of :math:`A P^{-1} = N - M`:

        .. math::

            N = \mathrm{skew}(A P^{-1}), \quad
            M = -\mathrm{sym}(A P^{-1}) = \tfrac12 P^{-2} \succ 0, \\
            K = \tfrac12 N, \quad
            R = \mathrm{chol}(M)^{-1}, \quad
            Q = \mathrm{chol}(P^{-1})^\top, \quad
            S_{:,:,k} = \tfrac14 \bigl(H_{:,:,k} P^{-1} - (H_{:,:,k} P^{-1})^\top\bigr),

        which give :math:`K - K^\top = N`, :math:`R^{-1}R^{-\top} = M` and
        :math:`Q^{-1}Q^{-\top} = P`.  The reconstruction of ``A`` is then exact,

        .. math::

            \bigl((K - K^\top) - R^{-1}R^{-\top}\bigr)\,Q^{-1}Q^{-\top}
                = (N - M)\,P = A P^{-1} P = A.

        If the quadratic operator ``H`` is structured (skew-symmetric under the
        :math:`P` metric, i.e., :math:`H_{:,:,k} P^{-1}` is skew-symmetric), then
        its reconstruction is also exact:

        .. math::

            S_{:,:,k} Q^{-1}Q^{-\top} - S_{:,:,k}^\top Q^{-1}Q^{-\top}
                = 2 S_{:,:,k} P = H_{:,:,k}.

        Using the Lyapunov solution :math:`P` (rather than :math:`P = I`)
        guarantees :math:`M` is SPD for *any* Hurwitz ``A``, so spectral
        stability alone suffices -- :math:`\mathrm{sym}(A)` need not be
        negative definite.  Any forcing operator ``B`` is left unchanged.

        :param tensors: ``[A, H]`` with ``A`` of shape ``(r, r)`` and ``H``
            of shape ``(r, r, r)``
        :type tensors: list
        :param margin: stability margin by which the spectrum is pushed into
            the left-half plane when ``A`` is not already strictly stable
        :type margin: float
        :param optimize_F: if True, optimizes :math:`F = L^{-1}L^{-\top}` to minimize
            :math:`\kappa(P) + \gamma\kappa(F)`
        :type optimize_F: bool
        :param F_cond_penalty: penalty weight :math:`\gamma` for the condition number of F
        :type F_cond_penalty: float
        :param use_P_I: if True, skips optimizing F and solving the Lyapunov equation,
            instead setting :math:`P = I`.
        :type use_P_I: bool
        :raises RuntimeError: if the assembled ``[K, R, Q, S]`` fail to
            reconstruct the (stabilized) ``A``, or if a structured ``H``
            fails to be reconstructed
        """
        rank, _ = mpi_rank_size()
        printable = rank == 0

        bkend = self.backend
        A = bkend.asarray(tensors[0], dtype=self.dtype, device=self.device)
        H = bkend.asarray(tensors[1], dtype=self.dtype, device=self.device)
        r = A.shape[0]
        eye = bkend.eye(r, dtype=self.dtype, device=self.device)

        # Shift the spectrum into the open left-half plane if A is not already
        # strictly stable (leave it unchanged otherwise). If P = I, we shift
        # so that sym(A) is negative definite (required to get an SPD M).
        if use_P_I:
            sym_A = A + A.T
            abscissa = float(bkend.eigh(sym_A)[0].max())
        else:
            abscissa = float(bkend.eigvals(A).real.max())
        shift = abscissa + margin if abscissa >= 0.0 else 0.0
        A = A - shift * eye

        if use_P_I:
            P = eye
            F = eye
        else:
            if optimize_F:
                from scipy.linalg import solve_triangular
                from scipy.optimize import approx_fprime
                from nitrom.optimization.manifold_optimization import riemannian_lbfgs

                A_np = bkend.to_numpy(A)
                
                def build_L(x):
                    L = np.zeros((r, r))
                    L[np.tril_indices(r)] = x
                    return L

                def cost_fn_flat(xs):
                    x = xs[0]
                    L = build_L(x)
                    try:
                        L_inv = solve_triangular(L, np.eye(r), lower=True)
                    except (np.linalg.LinAlgError, ValueError):
                        return 1e18
                    
                    F_np = L_inv @ L_inv.T
                    
                    try:
                        P_np_opt = solve_continuous_lyapunov(A_np.T, -F_np)
                    except ValueError:
                        return 1e18
                    
                    cond_P = np.linalg.cond(P_np_opt)
                    cond_F = np.linalg.cond(F_np)
                    
                    if not np.isfinite(cond_P) or not np.isfinite(cond_F):
                        return 1e18
                        
                    return cond_P + F_cond_penalty * cond_F

                def grad_fn_flat(xs):
                    x = xs[0]
                    eps = 1e-8
                    g = approx_fprime(x, lambda v: cost_fn_flat([v]), eps)
                    return [g]

                # Initial guess: L = I
                x0 = np.eye(r)[np.tril_indices(r)]
                
                res_xs, res_f = riemannian_lbfgs(
                    cost_fn_flat,
                    grad_fn_flat,
                    x0=[x0],
                    manifolds=["euclidean"],
                    max_iter=100,
                    callback=(
                        lambda it, f, gnorm: print(
                            f"F-Optimization Iteration {it+1} | Cost: {f:.4e} | gnorm: {gnorm:.2e}",
                            flush=True
                        )
                    ) if printable else None
                )
                
                L_opt = build_L(res_xs[0])
                L_inv_opt = solve_triangular(L_opt, np.eye(r), lower=True)
                F = bkend.asarray(L_inv_opt @ L_inv_opt.T, dtype=self.dtype, device=self.device)
            else:
                F = eye

            # Solve A^T P + P A = -F for the SPD Lyapunov solution P.
            P_np = solve_continuous_lyapunov(
                bkend.to_numpy(A.T), bkend.to_numpy(-F)
            )
            P = bkend.asarray(P_np, dtype=self.dtype, device=self.device)

        P = 0.5 * (P + P.T)  # symmetrize against round-off
        Pinv = bkend.inv(P)

        # Split A P^{-1} = N - M with N skew and M = -sym(A P^{-1}) = P^{-2}/2 SPD.
        APinv = A @ Pinv
        N = 0.5 * (APinv - APinv.T)
        M = -0.5 * (APinv + APinv.T)

        # K = N / 2 (so K - K^T = N);  R^{-1}R^{-T} = M;  Q^{-1}Q^{-T} = P.
        K = 0.5 * N
        R = bkend.inv(bkend.cholesky(M))
        Q = bkend.cholesky(Pinv).T

        # S_{:,:,k} = skew(H_{:,:,k} P^{-1}): transpose the leading (i, j) axes.
        H_Pinv = bkend.einsum("ijk,jl->ilk", H, Pinv)
        S = 0.25 * (H_Pinv - bkend.permute(H_Pinv, (1, 0, 2)))

        # Verify the reconstruction A = ((K - K^T) - R^{-1}R^{-T}) Q^{-1}Q^{-T}.
        Qinv = bkend.inv(Q)
        Rinv = bkend.inv(R)
        A_recon = ((K - K.T) - Rinv @ Rinv.T) @ (Qinv @ Qinv.T)
        err = float(bkend.vector_norm(A_recon - A) / bkend.vector_norm(A))
        if err > 1e-6:
            raise RuntimeError(
                f"GAS retraction failed to reconstruct A (rel. error {err:.2e})."
            )

        # Verify the reconstruction of H if the input was already structured.
        if 2 in self.poly_comp:
            Qtil = Qinv @ Qinv.T
            H_recon = bkend.einsum("ilk,lj->ijk", S, Qtil) - bkend.einsum(
                "lik,lj->ijk", S, Qtil
            )
            # Check if the input H was already structured (skew-symmetric under
            # the P metric).
            is_structured = float(
                bkend.vector_norm(H_Pinv + bkend.permute(H_Pinv, (1, 0, 2)))
                / (bkend.vector_norm(H_Pinv) + 1e-12)
            ) < 1e-6
            if is_structured:
                err_H = float(
                    bkend.vector_norm(H_recon - H)
                    / (bkend.vector_norm(H) + 1e-12)
                )
                if err_H > 1e-6:
                    raise RuntimeError(
                        f"GAS retraction failed to reconstruct H "
                        f"(rel. error {err_H:.2e})."
                    )

        # Set the GAS parameters (preserving B and any other current params).
        retracted = {"K": K, "R": R, "Q": Q, "S": S}
        params = [
            retracted.get(name, getattr(self, name)) for name in self.param_names
        ]
        self.update_params(params)

    def evaluate_rhs(self, t: float, z: Any, **kwargs) -> Any:
        """Delegate to the inner :class:`PolynomialModel`."""
        return self.model.evaluate_rhs(t, z, **kwargs)

    def evaluate_adjoint_rhs(self, t: float, z: Any, Z: Any, **kwargs) -> Any:
        """Delegate to the inner :class:`PolynomialModel`."""
        return self.model.evaluate_adjoint_rhs(t, z, Z, **kwargs)

    def inner_params(self) -> list[Any]:
        """Return the inner parameter tensors A, H, [B]."""
        return self.model.get_params()

    def project_inner_gradients(self, inner_grads: list[Any]) -> list[Any]:
        """
        Project the accumulated inner gradients (w.r.t A, H, [B]) back to the 
        GAS parameters (K, R, Q, S, [B]).
        """
        bkend = self.backend
        
        # Unpack inner gradients (indexed by position in poly_comp)
        grad_A = inner_grads[self.poly_comp.index(1)] if 1 in self.poly_comp else None
        grad_H = inner_grads[self.poly_comp.index(2)] if 2 in self.poly_comp else None
        grad_B = inner_grads[-1] if self.forcing_exists else None

        Qtil = self._Qtil

        grads = []

        # grad_K, grad_R (from linear term A = ((K - K^T) - R^{-1} R^{-T}) @ Qtil)
        if 1 in self.poly_comp:
            grad_A_Qtil = grad_A @ Qtil
            sym = grad_A_Qtil + grad_A_Qtil.T
            grad_K = grad_A_Qtil - grad_A_Qtil.T
            # M = R^{-1} R^{-T}; chain through M = P P^T and P = R^{-1}.
            grad_R = self._Rinv.T @ sym @ self._Rinv @ self._Rinv.T
            grads.extend([grad_K, grad_R])

        # grad_Q, grad_S from H_{ijk} = (S_{ilk} - S_{lik}) Qtil_{lj}
        if 2 in self.poly_comp:
            # grad_Qtil from linear term: A_pre^T @ grad_A
            if grad_A is not None:
                A_pre = (self.K - self.K.T) - self._Rtil
                grad_Qtil = A_pre.T @ grad_A
            else:
                grad_Qtil = bkend.zeros_like(Qtil)
            # grad_Qtil_{lj} from quadratic: Σ_{ik} grad_H_{ijk} (S_{ilk} - S_{lik})
            grad_Qtil += bkend.einsum("jik,jlk->il", self._S_diff, grad_H)

            # grad_Q from Qtil = Q^{-1} Q^{-T}
            grad_Q = -(self._Qinv.T @ (grad_Qtil @ Qtil) + self._Qinv.T @ (grad_Qtil.T @ Qtil))

            # grad_S_{abc} = Σ_j grad_H_{ajc} Qtil_{bj} - Σ_j grad_H_{bjc} Qtil_{aj}
            grad_S_pre = bkend.einsum("ijk,jl->ilk", grad_H, Qtil)
            grad_S = grad_S_pre - bkend.permute(grad_S_pre, (1, 0, 2))
            grads.extend([grad_Q, grad_S])

        if self.forcing_exists:
            grads.append(grad_B)

        return grads

    def inner_vjp_evaluate_rhs(self, z: Any, v: Any, reg: float = 0.0, **kwargs) -> list[Any]:
        r"""
        VJP of the RHS with respect to the inner model parameters.

        Calls the inner :class:`PolynomialModel` VJP to get gradients
        w.r.t. ``(A, H, [B])``.

        :param z: state vector of shape ``(n,)`` or ``(m, n)``
        :param v: upstream adjoint seed, same shape as ``z``
        :returns: list of gradients matching :meth:`inner_params`
        :rtype: list
        """
        return self.model.vjp_evaluate_rhs(z, v, reg=reg, **kwargs)

    def inner_batched_vjp_evaluate_rhs(
        self, Z: Any, V: Any, U: Any = None, out: list | None = None,
        max_bytes: int = 64 << 20,
    ) -> list[Any]:
        """Batched VJP w.r.t. the inner ``(A, H, [B])`` tensors.

        The GAS parameters are recovered from these by
        :meth:`project_inner_gradients`, which the caller applies once at the end
        of the sweep.
        """
        return self.model.batched_vjp_evaluate_rhs(
            Z, V, U=U, out=out, max_bytes=max_bytes
        )

    def vjp_evaluate_rhs(self, z: Any, v: Any, reg: float = 0.0, **kwargs) -> list[Any]:
        r"""
        VJP of the RHS with respect to the GAS parameters.

        :param z: state vector of shape ``(n,)`` or ``(m, n)``
        :param v: upstream adjoint seed, same shape as ``z``
        :returns: list of gradients matching :attr:`param_names`
        :rtype: list
        """
        inner_grads = self.inner_vjp_evaluate_rhs(z, v, reg=reg, **kwargs)
        return self.project_inner_gradients(inner_grads)
