"""Registered structural checks for unforced latent dynamics."""

from functools import singledispatch
from itertools import permutations

import numpy as np

from ..latent_space_models import AtrPolynomialModel, GasPolynomialModel
from ._utils import as_numpy, validate_tolerances


def check_model_properties(model, *, atol: float = 1e-8, rtol: float = 1e-5) -> dict:
    """Check the model's unforced structural guarantees and return scalar metrics.

    GAS checks a positive-definite Lyapunov metric, strict linear dissipation,
    and energy-conserving quadratic dynamics. ATR checks the same conditions
    in shifted coordinates and returns trapping bounds. Checks inspect the
    actual inner operators, without reassembling or changing the model. Input
    operators and external forcing are excluded. No trajectories are sampled.

    Failed conditions raise ``ValueError``. Unregistered models raise
    ``NotImplementedError`` rather than silently passing. Register extensions
    with ``@register_model_check(MyModel)`` on a function accepting
    ``(model, *, atol, rtol)`` and returning compact numerical results.
    Dispatch follows the model's Python MRO, so subclasses inherit checks unless
    a more specific check is registered.

    Algebraic identities use norm tolerances ``atol + rtol * scale``;
    positive-definiteness requires strictly positive minimum eigenvalues.
    These numerical checks do not establish stability of a time discretization.
    """
    validate_tolerances(atol, rtol)
    return _check_registered_model(model, atol=atol, rtol=rtol)


@singledispatch
def _check_registered_model(model, *, atol, rtol):
    raise NotImplementedError(
        f"No property check registered for {type(model).__name__}; "
        "use register_model_check to add one."
    )


register_model_check = _check_registered_model.register


def _finite_array(model, array, name, shape):
    value = as_numpy(model, array)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return value


def _metric_and_operators(model, allowed_degrees):
    degrees = model.model.poly_comp
    if 1 not in degrees or not set(degrees) <= allowed_degrees:
        raise ValueError(
            f"Structural check does not support polynomial degrees {degrees}."
        )
    r = model.state_dimension
    Q = _finite_array(model, model.Q, "Q", (r, r))
    try:
        Qinv = np.linalg.inv(Q)
    except np.linalg.LinAlgError as error:
        raise ValueError("Lyapunov metric requires an invertible Q.") from error
    metric = Qinv @ Qinv.T
    operators = {
        k: _finite_array(model, tensor, f"A_{k}", (r,) * (k + 1))
        for k, tensor in zip(degrees, model.inner_params()[: len(degrees)], strict=True)
    }
    return metric, operators


def _positive_eigenvalues(matrix, name):
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} has nonfinite entries.")
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    if eigenvalues[0] <= 0:
        raise ValueError(
            f"{name} must be positive definite; minimum eigenvalue = {eigenvalues[0]:.6g}."
        )
    return eigenvalues


def _check_structure(metric, linear, quadratic, *, atol, rtol):
    """Test V = z.T @ metric @ z / 2 using the actual operator coefficients."""
    metric_eigenvalues = _positive_eigenvalues(metric, "Lyapunov metric")
    product = metric @ linear
    dissipation = -0.5 * (product + product.T)
    dissipation_eigenvalues = _positive_eigenvalues(dissipation, "Linear dissipation")

    # z_i z_j z_k is fully symmetric: its cubic polynomial vanishes identically
    # iff the fully symmetric part of (metric @ H) vanishes. No state sampling.
    weighted = np.einsum("il,ljk->ijk", metric, quadratic)
    symmetric = sum(weighted.transpose(axes) for axes in permutations(range(3))) / 6
    residual = float(np.linalg.norm(symmetric))
    scale = float(np.linalg.norm(weighted))
    if not np.isfinite(residual) or residual > atol + rtol * scale:
        raise ValueError(
            f"Quadratic dynamics do not conserve Lyapunov energy; residual = {residual:.6g}."
        )
    results = {
        "metric_min_eigenvalue": float(metric_eigenvalues[0]),
        "dissipation_min_eigenvalue": float(dissipation_eigenvalues[0]),
        "quadratic_energy_residual": residual,
    }
    return results, metric_eigenvalues


@register_model_check(GasPolynomialModel)
def _check_gas(model, *, atol=1e-8, rtol=1e-5):
    metric, operators = _metric_and_operators(model, {1, 2})
    r = model.state_dimension
    quadratic = operators.get(2, np.zeros((r, r, r)))
    results, _ = _check_structure(metric, operators[1], quadratic, atol=atol, rtol=rtol)
    return results


@register_model_check(AtrPolynomialModel)
def _check_atr(model, *, atol=1e-8, rtol=1e-5):
    metric, operators = _metric_and_operators(model, {0, 1, 2})
    if set(operators) != {0, 1, 2}:
        raise ValueError(
            "ATR check requires constant, linear, and quadratic operators."
        )
    shift = _finite_array(model, model.m, "ATR shift", (model.state_dimension,))
    constant, linear, quadratic = (operators[k] for k in (0, 1, 2))
    shifted_linear = (
        linear
        + np.einsum("ijk,k->ij", quadratic, shift)
        + np.einsum("ijk,j->ik", quadratic, shift)
    )
    shifted_constant = (
        constant + linear @ shift + np.einsum("ijk,j,k->i", quadratic, shift, shift)
    )
    results, eigenvalues = _check_structure(
        metric, shifted_linear, quadratic, atol=atol, rtol=rtol
    )
    # In y = metric @ (z - m), Vdot = -y.T @ damping @ y + y.T @ b.
    # Recover damping from the checked operators, avoiding cached R products.
    damping = -shifted_linear @ np.linalg.inv(metric)
    damping = 0.5 * (damping + damping.T)
    min_damping = _positive_eigenvalues(damping, "ATR damping")[0]
    radius = float(np.linalg.norm(shifted_constant) / (min_damping * eigenvalues[0]))
    containing_radius = float(radius * np.sqrt(eigenvalues[-1] / eigenvalues[0]))
    if not np.isfinite(radius) or not np.isfinite(containing_radius):
        raise ValueError("ATR trapping bounds must be finite.")
    results.update(
        trapping_radius=radius,
        trapping_containing_radius=containing_radius,
    )
    return results
