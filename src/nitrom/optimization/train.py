from collections.abc import Callable
from typing import Any

from ._train_numpy import _train_numpy
from ._train_torch import _train_torch
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
    callback: Callable[..., Any] | None = None,
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
    :param callback: (numpy backend only) ``callback(model, it, loss, gradnorm)``,
        called after every accepted iterate ``it`` (0-based) with the model's
        parameters set to that iterate -- e.g. to save checkpoints.  It does not
        touch the optimizer state (no L-BFGS restart); ``gradnorm`` is that of
        the iterate the step was taken from, as in ``model.gradnorm_history``.
    :returns: the trained model
    """
    manifolds = {
        name: mtype
        for (name, _), mtype in zip(
            model.named_parameters(), model.get_manifold_types(), strict=True
        )
        if mtype != "euclidean"
    }

    if model.backend.is_numpy:
        return _train_numpy(
            model,
            manifolds,
            n_epochs,
            lr,
            optimizer_type,
            print_every,
            tol,
            n_restarts,
            allow_increase,
            callback,
        )
    if callback is not None:
        raise NotImplementedError(
            "train(callback=...) is implemented for the numpy backend only"
        )
    return _train_torch(
        model,
        manifolds,
        n_epochs,
        lr,
        optimizer_type,
        print_every,
        tol,
        n_restarts,
        scheduler_creator,
    )
