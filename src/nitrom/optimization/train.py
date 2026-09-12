from collections.abc import Callable
from typing import Any

from ..backend import get_backend
from .manifold_optimization import project, to_manifold, transport
from .modules.base import InferenceModule


def train(
    model: InferenceModule,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    optimizer_type: str = "adam",
    print_every: int = 100,
    tol: float = 1e-10,
    n_restarts: int = 0,
    scheduler_creator: Callable[[Any], Any] | None = None,
    allow_increase: bool = False,
) -> InferenceModule:
    r"""
    Train an :class:`InferenceModule` by minimizing its cost using the module's
    analytic gradients.

    Works for any concrete inference module (operator inference,
    polynomial-manifold inference, NiTROM, ...) because it relies only on the
    :class:`InferenceModule` contract: ``model()`` returns the scalar cost and
    ``model.gradient()`` returns the analytic gradients in :meth:`parameters`
    order.

    The optimizer backend follows the active array backend: **torch** uses
    ``torch.optim`` (Adam/SGD/L-BFGS, with distributed all-reduce when
    ``torch.distributed`` is initialized); **numpy** uses a unified first-order
    driver in which every optimizer (Adam/SGD/L-BFGS) supplies only a search
    direction and the step always comes from a strong-Wolfe line search with an
    Armijo backtracking fallback (:func:`~nitrom.optimization.manifold_optimization.riemannian_optimize`).
    Riemannian (Grassmann/Stiefel) optimization -- configured per parameter via
    :meth:`InferenceModule.set_manifold_types` -- is applied on either backend;
    on numpy it uses retraction and vector transport throughout.

    :param model: the inference module (already configured with data, basis, etc.)
    :param n_epochs: number of optimization iterations (max iterations for scipy)
    :param lr: learning rate
    :param optimizer_type: ``"adam"``, ``"sgd"``, or ``"lbfgs"``
    :param print_every: print the loss every *print_every* epochs (rank 0)
    :param tol: convergence tolerance on the relative change in loss
    :param n_restarts: number of optimizer restarts when the loss stalls
    :param scheduler_creator: (torch backend only) callable mapping an optimizer
        to a learning-rate scheduler; ``None`` for no scheduling
    :param allow_increase: allow line search to accept steps where cost increases
    :returns: the trained model
    """
    manifolds = {
        name: mtype
        for (name, _), mtype in zip(
            model.named_parameters(), model.get_manifold_types(), strict=True
        )
        if mtype != "euclidean"
    }

    if get_backend().is_numpy:
        return _train_numpy(
            model, manifolds, n_epochs, lr, optimizer_type,
            print_every, tol, n_restarts, allow_increase,
        )
    return _train_torch(
        model, manifolds, n_epochs, lr, optimizer_type,
        print_every, tol, n_restarts, scheduler_creator,
    )


# ---------------------------------------------------------------------------
# NumPy path: manual Adam/SGD + scipy L-BFGS
# ---------------------------------------------------------------------------

def _train_numpy(
    model, manifolds, n_epochs, lr, optimizer_type, print_every, tol, n_restarts,
    allow_increase,
):
    import numpy as np

    from ..backend import mpi_allreduce_scalar, mpi_allreduce_sum, mpi_rank_size

    # Trajectory parallelism (MPI): each rank holds a shard of trajectories, so
    # model()/model.gradient() are partial sums; Allreduce(SUM) reconstructs the
    # global cost/gradient (the weights already carry the global normalization).
    # Every rank runs the identical optimizer on the reduced values, in lockstep.
    #
    # Reduce over the communicator the training pool was *sharded* on, not
    # COMM_WORLD: a pool built on a split communicator must be reduced over that
    # same communicator or the sum spans the wrong set of ranks.  ``None``
    # resolves to COMM_WORLD, so the common case is unchanged.
    comm = getattr(model, "comm", None)
    rank, size = mpi_rank_size(comm)
    distributed = size > 1
    printable = bool(print_every) and rank == 0
    names = [n for n, _ in model.named_parameters()]

    def set_p(name, val):
        model._params[name] = val
        object.__setattr__(model, name, val)

    def cost():
        c = float(model())
        return mpi_allreduce_scalar(c, comm=comm) if distributed else c

    def grads_projected():
        g = model.gradient()
        if distributed:
            g = [mpi_allreduce_sum(np.asarray(gi), comm=comm) for gi in g]
        for i, name in enumerate(names):
            if manifolds.get(name):
                g[i] = project(model._params[name], g[i], manifolds[name])
        return g

    if optimizer_type not in ("adam", "sgd", "lbfgs"):
        raise ValueError(
            f"optimizer_type must be 'adam', 'sgd', or 'lbfgs', "
            f"got '{optimizer_type}'"
        )

    # Unified optimizer: each type only supplies a search *direction*; the step
    # always comes from a strong-Wolfe line search with an Armijo backtracking
    # fallback (:func:`riemannian_optimize`).  Euclidean parameters are handled
    # as the trivial manifold, so this one path covers every case.
    from .manifold_optimization import (
        AdamDirection,
        LBFGSDirection,
        SGDDirection,
        riemannian_optimize,
    )

    mlist = [manifolds.get(n, "euclidean") for n in names]

    def cost_fn(xs):
        for name, arr in zip(names, xs, strict=True):
            set_p(name, arr)
        return cost()

    def rgrad_fn(xs):
        for name, arr in zip(names, xs, strict=True):
            set_p(name, arr)
        return grads_projected()

    def make_direction():
        if optimizer_type == "lbfgs":
            return LBFGSDirection(history_size=100)
        if optimizer_type == "adam":
            return AdamDirection(lr)
        return SGDDirection(lr)

    current_loss_history = []
    current_gradnorm_history = []

    def progress(it, f, gnorm):
        current_loss_history.append(f)
        current_gradnorm_history.append(gnorm)
        if printable and it % print_every == 0:
            print(f"iter {it:6d} | Loss: {f:.6e} | GradNorm: {gnorm:.6e}")

    x0 = [np.asarray(model._params[n], dtype=float) for n in names]
    # n_restarts > 0 -> multi-start: perturb the init to explore neighbouring
    # basins of a non-convex objective and keep the best.
    scale = 0.05 * (float(np.mean([np.abs(a).mean() for a in x0])) + 1e-8)
    rng = np.random.default_rng(0)
    best_xs, best_f = x0, float("inf")
    best_loss_history, best_gradnorm_history = [], []
    for attempt in range(n_restarts + 1):
        current_loss_history.clear()
        current_gradnorm_history.clear()
        xs0 = (
            [a.copy() for a in x0]
            if attempt == 0
            else [a + scale * rng.standard_normal(a.shape) for a in x0]
        )
        xk, fk = riemannian_optimize(
            cost_fn, rgrad_fn, xs0, mlist, make_direction(),
            max_iter=n_epochs, gtol=tol,
            ftol=max(tol, 1e-12), callback=progress,
            allow_increase=allow_increase,
        )
        if fk < best_f:
            best_xs, best_f = xk, fk
            best_loss_history = list(current_loss_history)
            best_gradnorm_history = list(current_gradnorm_history)
        if printable:
            tag = "run" if attempt == 0 else f"multistart {attempt}/{n_restarts}"
            print(
                f"{optimizer_type} (strong-Wolfe/Armijo, {tag}): "
                f"Loss {fk:.6e} | best {best_f:.6e}"
            )
    for name, arr in zip(names, best_xs, strict=True):
        set_p(name, arr)
    model.loss_history = best_loss_history
    model.gradnorm_history = best_gradnorm_history
    return model


# ---------------------------------------------------------------------------
# Torch path: torch.optim (+ distributed), unchanged
# ---------------------------------------------------------------------------

def _transport_optimizer_states(optimizer, model, manifold_types) -> None:
    """Vector-transport the optimizer's momentum buffers to the new tangent spaces."""
    bkend = get_backend()
    for name, param in model.named_parameters():
        if name in manifold_types:
            state = optimizer.state.get(param)
            if state is None:
                continue
            # First-moment / momentum buffers are tangent vectors: transport them.
            for key in ["exp_avg", "momentum_buffer"]:
                if key in state:
                    buf = state[key]
                    buf.copy_(transport(param, buf, manifold_types[name]))
            # Second moment (Adam v): per-coordinate variance -> transport and |.|.
            if "exp_avg_sq" in state:
                buf = state["exp_avg_sq"]
                buf.copy_(bkend.abs(transport(param, buf, manifold_types[name])))


def _train_torch(
    model, normalized_manifold_types, n_epochs, lr, optimizer_type,
    print_every, tol, n_restarts, scheduler_creator,
):
    import torch
    import torch.distributed as dist

    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    if is_distributed:
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    def _make_optimizer():
        if optimizer_type == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        if optimizer_type == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr)
        if optimizer_type == "lbfgs":
            return torch.optim.LBFGS(
                model.parameters(), lr=lr, max_iter=10,
                line_search_fn="strong_wolfe",
            )
        raise ValueError(
            f"optimizer_type must be 'adam', 'sgd', or 'lbfgs', "
            f"got '{optimizer_type}'"
        )

    optimizer = _make_optimizer()
    scheduler = scheduler_creator(optimizer) if scheduler_creator is not None else None

    def _compute_and_assign_grads():
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in normalized_manifold_types:
                    param.copy_(to_manifold(param, normalized_manifold_types[name]))

        optimizer.zero_grad()
        loss = model()
        grads = model.gradient()
        for param, grad in zip(model.parameters(), grads, strict=True):
            param.grad = grad.contiguous().clone()
        if is_distributed:
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            loss = loss.detach().clone()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in normalized_manifold_types and param.grad is not None:
                    projected = project(
                        param, param.grad, normalized_manifold_types[name]
                    )
                    param.grad.copy_(projected)

        return loss

    loss_prev = None
    loss_at_last_restart = float("inf")
    restarts_left = n_restarts

    model.loss_history = []
    model.gradnorm_history = []

    for epoch in range(n_epochs):
        if optimizer_type == "lbfgs":
            loss = optimizer.step(_compute_and_assign_grads)
        else:
            loss = _compute_and_assign_grads()
            optimizer.step()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in normalized_manifold_types:
                    param.copy_(to_manifold(param, normalized_manifold_types[name]))
                    if is_distributed:
                        dist.broadcast(param.data, src=0)

        _transport_optimizer_states(optimizer, model, normalized_manifold_types)

        loss_val = loss.item()

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss_val)
            else:
                scheduler.step()

        grad_norm = torch.sqrt(
            sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)
        ).item()

        model.loss_history.append(loss_val)
        model.gradnorm_history.append(grad_norm)
        if rank == 0 and (epoch % print_every == 0 or epoch == n_epochs - 1):
            lr_str = (
                f" | LR: {optimizer.param_groups[0]['lr']:.2e}"
                if scheduler is not None else ""
            )
            print(
                f"Epoch {epoch:6d} | Loss: {loss_val:.6e} "
                f"| GradNorm: {grad_norm:.6e}{lr_str}"
            )

        if loss_prev is not None and abs(loss_prev) > 0:
            rel_change = abs(loss_val - loss_prev) / abs(loss_prev)
            if rel_change < tol:
                improved = (
                    loss_at_last_restart == float("inf")
                    or loss_at_last_restart - loss_val
                    > tol * abs(loss_at_last_restart)
                )
                if restarts_left > 0 and improved:
                    restarts_left -= 1
                    loss_at_last_restart = loss_val
                    optimizer = _make_optimizer()
                    if scheduler_creator is not None:
                        scheduler = scheduler_creator(optimizer)
                    loss_prev = None
                    if rank == 0:
                        print(
                            f"Restart {n_restarts - restarts_left}/{n_restarts} "
                            f"at epoch {epoch} (loss {loss_val:.6e})"
                        )
                    continue
                if rank == 0:
                    print(
                        f"Converged at epoch {epoch} "
                        f"(rel. change {rel_change:.2e} < {tol:.2e}) "
                        f"loss value: {loss_val:.6e}"
                    )
                break
        loss_prev = loss_val

    return model
