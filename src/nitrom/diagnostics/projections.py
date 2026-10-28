"""Check the latent-space encoder/decoder identity on several states."""

import numpy as np

from ._utils import as_numpy, percent_error, validate_tolerances


def check_projection(
    projection,
    *,
    n_samples: int = 10,
    seed: int = 0,
    z=None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> dict[str, float]:
    """Check ``encode(decode(z)) == z`` and return maximum norm errors.

    Draw ``n_samples`` standard-normal latent states using a local seeded RNG
    (without advancing NumPy or Torch's global RNG). Alternatively, supply ``z``
    as a backend array of shape ``(r,)`` or ``(n_samples, r)``; then ``n_samples``
    and ``seed`` are ignored. Generated states match the projection dtype/device.

    Each sample must satisfy ``norm(recovered - z) <= atol + rtol * norm(z)``.
    Return ``max_absolute_error`` and ``max_percent_error`` using vector norms.
    A zero reference gives 0 percent if recovered exactly, otherwise infinity.
    Raise ``ValueError`` for failed checks, nonfinite values, or invalid inputs.
    """
    validate_tolerances(atol, rtol)
    r = projection.latent_space_dimension
    if z is None:
        if (
            isinstance(n_samples, bool)
            or not isinstance(n_samples, int)
            or n_samples < 1
        ):
            raise ValueError("n_samples must be a positive integer.")
        z = projection.backend.asarray(
            np.random.default_rng(seed).standard_normal((n_samples, r)),
            dtype=projection.dtype,
            device=projection.device,
        )
    else:
        projection.backend.validate_arrays(z=z)
        if z.ndim == 1:
            z = z[None, :]
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] != r:
        raise ValueError(f"z must have shape ({r},) or (n_samples, {r}).")

    reference = as_numpy(projection, z)
    if not np.isfinite(reference).all():
        raise ValueError("Projection check requires finite latent states.")
    with projection.backend.no_grad():
        recovered = as_numpy(projection, projection.encode(projection.decode(z)))
    if recovered.shape != reference.shape or not np.isfinite(recovered).all():
        raise ValueError(
            "Projection round trip returned an invalid shape or nonfinite values."
        )
    errors = np.linalg.norm(recovered - reference, axis=1)
    norms = np.linalg.norm(reference, axis=1)
    results = {
        "max_absolute_error": float(errors.max()),
        "max_percent_error": max(
            percent_error(float(error), float(norm))
            for error, norm in zip(errors, norms, strict=True)
        ),
    }
    if np.any(errors > atol + rtol * norms):
        raise ValueError(f"Projection encode(decode(z)) != z: {results}")
    return results
