"""Discrete/continuous adjoint accumulation and bulk parameter VJPs."""

from typing import TYPE_CHECKING, Any

import numpy as np

from ...numerics import interp_quadratic
from ...time_steppers import solve_adjoint_ivp_discrete, solve_ivp

if TYPE_CHECKING:
    from .nitrom_module import NitromModule


class _ParamVJPCollector:
    """Buffers the discrete adjoint's parameter-VJP records and reduces in bulk.

    The backward sweep produces one ``(state, seed)`` record per Runge-Kutta stage
    per sub-step.  Evaluating and accumulating the VJP per record means hundreds of
    rank-1 updates of an ``(r, r, r)`` tensor, which is memory-bound and dominates
    the sweep.  Because the VJP is multilinear in ``(state, seed)``, the sum over
    records equals a single contraction over the stacked records -- one GEMM per
    degree instead of one outer product per record.

    Records are flushed every ``max_records`` so the intermediate stays bounded
    regardless of trajectory count or horizon length.
    """

    def __init__(self, module: "NitromModule", max_records: int = 512):
        self.backend = module.backend
        self.model = module.model
        self.max_records = max_records
        self.grads = [self.backend.zeros_like(p) for p in self.model.inner_params()]

        forcing_fns = module.forcing_fns or None
        self.forcing_fns = (
            forcing_fns
            if getattr(self.model, "forcing_exists", False) and forcing_fns
            else None
        )
        self._m = (
            self.model.inner_params()[-1].shape[1]
            if self.forcing_fns is not None
            else 0
        )
        self._z: list = []
        self._w: list = []
        self._t: list = []

    def add(self, z: Any, w: Any, t: float) -> None:
        """Record one stage contribution; ``z``/``w`` are ``(ntraj, r)``."""
        self._z.append(z)
        self._w.append(w)
        self._t.append(t)
        if len(self._z) >= self.max_records:
            self.flush()

    def _forcing_block(self, ntraj: int) -> Any:
        """Stacked ``u_j(t_a)`` rows, flattened to match the record ordering."""
        bkend = self.backend
        rows = []
        for t in self._t:
            for j in range(ntraj):
                fn = self.forcing_fns[j] if j < len(self.forcing_fns) else None
                rows.append(
                    bkend.zeros(
                        (self._m,),
                        dtype=self.grads[-1].dtype,
                        device=bkend.device_of(self.grads[-1]),
                    )
                    if fn is None
                    else bkend.atleast_1d(fn(t))
                )
        return bkend.stack(rows, axis=0)  # (nrec * ntraj, m)

    def flush(self) -> None:
        """Reduce the buffered records into the running gradient accumulators."""
        if not self._z:
            return
        bkend = self.backend
        Z = bkend.stack(self._z, axis=0)  # (nrec, ntraj, r)
        W = bkend.stack(self._w, axis=0)
        nrec, ntraj, r = Z.shape

        # Flatten to (nrec * ntraj, r); record d then has trajectory index
        # d % ntraj, which is the indexing the forcing block above assumes.
        U = self._forcing_block(ntraj) if self.forcing_fns else None
        self.grads = self.model.inner_batched_vjp_evaluate_rhs(
            Z.reshape(nrec * ntraj, r),
            W.reshape(nrec * ntraj, r),
            U=U,
            out=self.grads,
        )
        self._z.clear()
        self._w.clear()
        self._t.clear()

    def result(self) -> list:
        """Flush any buffered records and return the accumulated gradients."""
        self.flush()
        return self.grads


def accumulate_adjoint(
    module: "NitromModule", Z: Any, dense: Any, src: Any
) -> tuple[Any, list]:
    """Sweep measurements backward, returning the initial seed and model VJPs."""
    bkend = module.backend
    X = module.training_data.X
    ntraj, _, nt = X.shape
    r = module.model.state_dimension
    dev, dtype = bkend.device_of(X), X.dtype
    time_h = np.asarray(bkend.to_numpy(module.time), dtype=float)
    ef = module.forcing_fns or None
    # --- backward adjoint sweep --------------------------------------
    model_grads = [bkend.zeros_like(p) for p in module.model.inner_params()]
    lam = bkend.zeros((ntraj, r), device=dev, dtype=dtype)

    if module.adjoint_method == "discrete":
        # The parameter VJP is multilinear in (state, seed), so the
        # per-stage contributions can be buffered and reduced in bulk.
        collector = _ParamVJPCollector(module)
        for k in range(nt - 1, 0, -1):
            # Inject the measurement source at snapshot k.
            lam = lam + src[:, :, k]

            # Base flow over [t_{k-1}, t_k]: slice the dense solve when
            # we have one, else re-integrate the interval.
            t0i, tfi = float(time_h[k - 1]), float(time_h[k])
            delta = tfi - t0i
            h = delta / module.n_substeps
            if dense is not None:
                j0 = (k - 1) * module.n_substeps
                j1 = j0 + module.n_substeps
                sub_t = dense.t[j0 : j1 + 1]
                Zint = dense.X[:, :, j0 : j1 + 1]
                stages_G = None if dense.stages is None else dense.stages[j0:j1]
            else:
                sub_t = bkend.linspace(
                    t0i, tfi, module.n_substeps + 1, device=dev, dtype=dtype
                )
                Zint = solve_ivp(
                    module.model.evaluate_rhs,
                    Z[:, :, k - 1],
                    t0i,
                    tfi,
                    h,
                    sub_t,
                    module.time_stepper,
                    atol=module.atol,
                    rtol=module.rtol,
                    external_forcing=ef,
                    backend=module.backend,
                )  # (ntraj, r, n_substeps + 1)
                stages_G = None

            # Propagate adjoint and accumulate model grads
            lam, step_grads = solve_adjoint_ivp_discrete(
                module.model.evaluate_rhs,
                module.model.evaluate_adjoint_rhs,
                module._vjp_rhs,
                Zint,
                sub_t,
                lam,
                method=module.time_stepper,
                collector=collector,
                stages_G=stages_G,
                external_forcing=ef,
                backend=module.backend,
            )

            if step_grads is not None:
                for idx, g in enumerate(step_grads):
                    model_grads[idx] = model_grads[idx] + g

        # One contraction over every buffered stage record.
        for idx, g in enumerate(collector.result()):
            model_grads[idx] = model_grads[idx] + g

    else:  # continuous adjoint
        xi, wq = module._gl_nodes, module._gl_weights

        for k in range(nt - 1, 0, -1):
            # Inject the measurement source at snapshot k.
            lam = lam + src[:, :, k]

            # Re-integrate the base flow over [t_{k-1}, t_k].
            t0i, tfi = float(time_h[k - 1]), float(time_h[k])
            delta = tfi - t0i
            a = 0.5 * delta
            if dense is not None:
                j0 = (k - 1) * module.n_substeps
                sub_t = dense.t[j0 : j0 + module.n_substeps + 1]
                Zint = dense.X[:, :, j0 : j0 + module.n_substeps + 1]
            else:
                sub_t = bkend.linspace(
                    t0i, tfi, module.n_substeps + 1, device=dev, dtype=dtype
                )
                Zint = solve_ivp(
                    module.model.evaluate_rhs,
                    Z[:, :, k - 1],
                    t0i,
                    tfi,
                    delta / module.n_substeps,
                    sub_t,
                    module.time_stepper,
                    atol=module.atol,
                    rtol=module.rtol,
                    external_forcing=ef,
                    backend=module.backend,
                )  # (ntraj, r, n_substeps + 1)

            # Adjoint in reversed time tau in [0, delta] (physical time
            # t = tfi - tau): d(lam)/d(tau) = J_f(Z(t))^T lam.
            def adj_rhs(tau, lam_, _Zint=Zint, _sub_t=sub_t, _tfi=tfi):
                phys_t = _tfi - tau
                tq = bkend.atleast_1d(bkend.asarray(phys_t, device=dev, dtype=dtype))
                Z_t = interp_quadratic(tq, _sub_t, _Zint, backend=module.backend)[
                    ..., 0
                ]  # (ntraj, r)
                return module.model.evaluate_adjoint_rhs(float(phys_t), lam_, Z_t)

            # Integrate the adjoint onto the Gauss-Legendre nodes (reversed
            # time), plus the interval end for the carry-forward.
            tau_nodes = a * (1.0 - xi)  # (n_leggauss,)
            order = bkend.argsort(tau_nodes)
            tau_eval = bkend.concatenate(
                [tau_nodes[order], bkend.asarray([delta], device=dev, dtype=dtype)]
            )
            Lam_sol = solve_ivp(
                adj_rhs,
                lam,
                0.0,
                delta,
                delta / module.n_substeps,
                tau_eval,
                module.time_stepper,
                atol=module.atol,
                rtol=module.rtol,
                backend=module.backend,
            )  # (ntraj, r, n_leggauss + 1)

            # Adjoint at the GL nodes (undo the sort); carry lambda(t_{k-1}).
            Lam_nodes = bkend.empty(
                (ntraj, r, module.n_leggauss), device=dev, dtype=dtype
            )
            Lam_nodes[..., order] = Lam_sol[..., : module.n_leggauss]
            lam = Lam_sol[..., -1]

            # Base flow at the physical GL nodes.
            t_gl = 0.5 * (tfi + t0i) + a * xi  # (n_leggauss,)
            Z_nodes = interp_quadratic(
                t_gl, sub_t, Zint, backend=module.backend
            )  # (ntraj, r, n_leggauss)

            # Gauss-Legendre quadrature of the model-parameter integral.
            # Fold the quadrature weight a * w_i into the adjoint seed.
            Lam_w = Lam_nodes * (a * wq).reshape(1, 1, -1)
            if getattr(module.model, "forcing_exists", False) and module.forcing_fns:
                # Forcing makes the VJP depend on the per-node time.
                for i in range(module.n_leggauss):
                    g = module._vjp_rhs(Z_nodes[..., i], Lam_w[..., i], float(t_gl[i]))
                    for idx in range(len(model_grads)):
                        model_grads[idx] = model_grads[idx] + g[idx]
            else:
                # Flatten (trajectory, node) and reduce in one VJP call.
                Zf = bkend.permute(Z_nodes, (0, 2, 1)).reshape(-1, r)
                Lf = bkend.permute(Lam_w, (0, 2, 1)).reshape(-1, r)
                for idx, g in enumerate(module.model.inner_vjp_evaluate_rhs(Zf, Lf)):
                    model_grads[idx] = model_grads[idx] + g

    return lam, model_grads
