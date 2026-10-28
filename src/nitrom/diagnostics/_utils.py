"""Small numerical helpers shared by the diagnostics."""

import numpy as np


def percent_error(error_norm: float, reference_norm: float) -> float:
    """Relative norm error in percent; zero/zero is 0, nonzero/zero is inf."""
    if reference_norm == 0:
        return 0.0 if error_norm == 0 else float("inf")
    return 100.0 * error_norm / reference_norm


def validate_tolerances(atol: float, rtol: float) -> None:
    if not (np.isfinite(atol) and atol >= 0 and np.isfinite(rtol) and rtol >= 0):
        raise ValueError("atol and rtol must be finite and nonnegative.")


def as_numpy(owner, array) -> np.ndarray:
    """Detach diagnostics from autograd and use double precision for reductions."""
    return np.asarray(owner.backend.to_numpy(array), dtype=np.float64)
