"""Compare the continuous and discrete NiTROM adjoint gradients."""

import math

import numpy as np

from ._utils import as_numpy, percent_error


def _global_gradients(module) -> list[np.ndarray]:
    """Sum trajectory-sharded gradients before comparing, as training does."""
    gradients = module.gradient()
    if module.backend.is_numpy:
        from ..distributed import mpi_allreduce_sum, mpi_rank_size

        if mpi_rank_size(module.comm)[1] > 1:
            gradients = [mpi_allreduce_sum(g, comm=module.comm) for g in gradients]
    else:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            gradients = [module.backend.copy(g).contiguous() for g in gradients]
            for gradient in gradients:
                dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=module.comm)
    # Copy because a subsequent gradient evaluation may reuse work buffers.
    return [as_numpy(module, g).copy() for g in gradients]


def compare_gradients(module) -> dict:
    r"""Return continuous-versus-discrete gradient errors, without a pass/fail test.

    ``module`` is a :class:`~nitrom.NitromModule`. Both adjoints use its current
    parameters, training data, solver settings, and learnability flags. Frozen
    parameters therefore contribute zeros. In distributed jobs, call on every
    rank: gradients are summed over the module's communicator before comparison.

    The result has ``overall_percent_error`` and ``parameter_percent_errors``
    (keyed by parameter name). Each error is
    ``100 * norm(continuous - discrete) / norm(discrete)``; the overall norms
    treat all parameter gradients as one concatenated vector. A zero reference
    gives 0 if both gradients are zero, otherwise infinity. Nonfinite gradients
    give NaN. No discrepancy raises an error; evaluation errors still propagate.
    The original ``adjoint_method`` is restored even if evaluation fails.
    """
    original_method = module.adjoint_method
    try:
        module.adjoint_method = "continuous"
        continuous = _global_gradients(module)
        module.adjoint_method = "discrete"
        discrete = _global_gradients(module)
    finally:
        module.adjoint_method = original_method

    errors, references, per_parameter = [], [], {}
    for (name, parameter), gc, gd in zip(
        module.named_parameters(), continuous, discrete, strict=True
    ):
        if gc.shape != parameter.shape or gd.shape != parameter.shape:
            raise ValueError(f"Gradient shape does not match parameter {name!r}.")
        if not (np.isfinite(gc).all() and np.isfinite(gd).all()):
            error = reference = float("nan")
        else:
            error = float(np.linalg.norm(gc - gd))
            reference = float(np.linalg.norm(gd))
        errors.append(error)
        references.append(reference)
        per_parameter[name] = percent_error(error, reference)

    return {
        "overall_percent_error": percent_error(
            math.hypot(*errors), math.hypot(*references)
        ),
        "parameter_percent_errors": per_parameter,
    }
