"""Torch optimizer driver with process-group reductions and retractions."""

from ..distributed import torch_group_root
from .manifold_optimization import project, to_manifold, transport


def _transport_optimizer_states(optimizer, model, manifold_types) -> None:
    """Vector-transport the optimizer's momentum buffers to the new tangent spaces."""
    bkend = model.backend
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
    model,
    normalized_manifold_types,
    n_epochs,
    lr,
    optimizer_type,
    print_every,
    tol,
    n_restarts,
    scheduler_creator,
):
    import torch
    import torch.distributed as dist

    comm = model.comm
    is_distributed = dist.is_initialized()
    rank = dist.get_rank(group=comm) if is_distributed else 0
    root = torch_group_root(comm) if is_distributed else 0

    if is_distributed:
        for param in model.parameters():
            dist.broadcast(param.data, src=root, group=comm)

    def _make_optimizer():
        if optimizer_type == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        if optimizer_type == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr)
        if optimizer_type == "lbfgs":
            return torch.optim.LBFGS(
                model.parameters(),
                lr=lr,
                max_iter=10,
                line_search_fn="strong_wolfe",
            )
        raise ValueError(
            f"optimizer_type must be 'adam', 'sgd', or 'lbfgs', got '{optimizer_type}'"
        )

    optimizer = _make_optimizer()
    scheduler = scheduler_creator(optimizer) if scheduler_creator is not None else None

    def _compute_and_assign_grads():
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in normalized_manifold_types:
                    param.copy_(
                        to_manifold(
                            param,
                            normalized_manifold_types[name],
                            backend=model.backend,
                        )
                    )

        optimizer.zero_grad()
        loss = model()
        grads = model.gradient()
        for param, grad in zip(model.parameters(), grads, strict=True):
            param.grad = grad.contiguous().clone()
        if is_distributed:
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=comm)
            loss = loss.detach().clone()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=comm)

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
                    param.copy_(
                        to_manifold(
                            param,
                            normalized_manifold_types[name],
                            backend=model.backend,
                        )
                    )
                    if is_distributed:
                        dist.broadcast(param.data, src=root, group=comm)

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
        if (
            rank == 0
            and print_every
            and (epoch % print_every == 0 or epoch == n_epochs - 1)
        ):
            lr_str = (
                f" | LR: {optimizer.param_groups[0]['lr']:.2e}"
                if scheduler is not None
                else ""
            )
            print(
                f"Epoch {epoch:6d} | Loss: {loss_val:.6e} "
                f"| GradNorm: {grad_norm:.6e}{lr_str}"
            )

        if loss_prev is not None and abs(loss_prev) > 0:
            rel_change = abs(loss_val - loss_prev) / abs(loss_prev)
            if rel_change < tol:
                improved = loss_at_last_restart == float(
                    "inf"
                ) or loss_at_last_restart - loss_val > tol * abs(loss_at_last_restart)
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
