"""NumPy optimizer driver with MPI reductions and manifold line searches."""

from .manifold_optimization import project


def _train_numpy(
    model,
    manifolds,
    n_epochs,
    lr,
    optimizer_type,
    print_every,
    tol,
    n_restarts,
    allow_increase,
    callback=None,
):
    import numpy as np

    from ..distributed import mpi_allreduce_scalar, mpi_allreduce_sum, mpi_rank_size

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
            f"optimizer_type must be 'adam', 'sgd', or 'lbfgs', got '{optimizer_type}'"
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

        def at_iterate(it, xs, f, gnorm):
            # load the accepted iterate into the model, then hand it over
            for name, arr in zip(names, xs, strict=True):
                set_p(name, arr)
            callback(model, it, f, gnorm)

        xk, fk = riemannian_optimize(
            cost_fn,
            rgrad_fn,
            xs0,
            mlist,
            make_direction(),
            max_iter=n_epochs,
            gtol=tol,
            ftol=max(tol, 1e-12),
            callback=progress,
            allow_increase=allow_increase,
            iterate_callback=at_iterate if callback is not None else None,
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
