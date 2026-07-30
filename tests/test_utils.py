import numpy as np
import pytest
import torch

from nitrom.utils import interp_quadratic


# Use a cubic test function: the quadratic interpolant should converge
# with O(dt^3) in the max error (slope = 3 in log-log), but
# interp_quadratic is piecewise quadratic so the *interpolation order*
# is 3 (error ~ C * dt^3). We test with a smooth non-polynomial so
# the error doesn't vanish exactly.

def _test_function_1d(t: torch.Tensor) -> torch.Tensor:
    """sin(2 pi t) — smooth, non-polynomial."""
    return torch.sin(2 * torch.pi * t)


def _test_function_batched(t: torch.Tensor) -> torch.Tensor:
    """
    Returns shape (B, n, len(t)) for B=3, n=2.
    Each component is a different smooth function.
    """
    # (B, n, len(t))
    return torch.stack([
        torch.stack([torch.sin(2 * torch.pi * t),
                     torch.cos(3 * torch.pi * t)]),
        torch.stack([torch.exp(-t) * torch.sin(4 * t),
                     t ** 3 - 0.5 * t]),
        torch.stack([torch.sin(t) * torch.cos(2 * t),
                     torch.tanh(3 * t - 1.5)]),
    ])


DT_VALUES = [1 / 200, 1 / 400, 1 / 800, 1 / 1600, 1 / 3200]


class TestInterpolationConvergence:

    def _compute_errors(self, build_y_data, build_y_exact):
        """
        For each dt in DT_VALUES, sample the function on a uniform grid,
        interpolate at the evaluation points, and return (dt_array, error_array).
        """
        t_eval = torch.linspace(0.05, 0.95, 50, dtype=torch.float64)
        y_exact = build_y_exact(t_eval)

        errors = []
        for dt in DT_VALUES:
            n_pts = int(round(1.0 / dt)) + 1
            t_data = torch.linspace(0.0, 1.0, n_pts, dtype=torch.float64)
            y_data = build_y_data(t_data)
            y_interp = interp_quadratic(t_eval, t_data, y_data)
            err = torch.linalg.norm(y_interp - y_exact).item()
            errors.append(err)

        return np.array(DT_VALUES), np.array(errors)

    def _check_convergence_order(self, dts, errors, expected_order):
        """
        Check convergence order via successive error ratios.
        If dt is halved each time and error ~ dt^p, the ratio
        e_{i} / e_{i+1} should be 2^p.  We check each ratio
        is within 0.1% of the expected value.
        """
        expected_ratio = 2 ** expected_order
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            np.testing.assert_allclose(
                ratio, expected_ratio, rtol=1e-2,
                err_msg=f"Ratio errors[{i}]/errors[{i+1}] = {ratio:.6f}, "
                        f"expected {expected_ratio:.1f}",
            )

    def test_unbatched_convergence(self):
        """1-D (n_data,) input: error should decay as O(dt^3)."""
        dts, errors = self._compute_errors(
            build_y_data=_test_function_1d,
            build_y_exact=_test_function_1d,
        )
        self._check_convergence_order(dts, errors, expected_order=3)

    def test_unbatched_2d_convergence(self):
        """(n, n_data) input: error should decay as O(dt^3)."""
        def build_y(t):
            return torch.stack([
                torch.sin(2 * torch.pi * t),
                torch.cos(3 * torch.pi * t),
            ])  # (2, len(t))

        dts, errors = self._compute_errors(
            build_y_data=build_y,
            build_y_exact=build_y,
        )
        self._check_convergence_order(dts, errors, expected_order=3)

    def test_batched_convergence(self):
        """(B, n, n_data) input: error should decay as O(dt^3)."""
        dts, errors = self._compute_errors(
            build_y_data=_test_function_batched,
            build_y_exact=_test_function_batched,
        )
        self._check_convergence_order(dts, errors, expected_order=3)


class TestInterpolationExactForQuadratic:
    """A quadratic polynomial should be reproduced exactly."""

    def test_exact_quadratic_unbatched(self):
        t_data = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
        t_eval = torch.linspace(0.05, 0.95, 37, dtype=torch.float64)
        y_data = 3 * t_data ** 2 - 2 * t_data + 1
        y_exact = 3 * t_eval ** 2 - 2 * t_eval + 1
        y_interp = interp_quadratic(t_eval, t_data, y_data)
        np.testing.assert_allclose(
            y_interp.numpy(), y_exact.numpy(), atol=1e-12,
        )

    def test_exact_quadratic_batched(self):
        t_data = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
        t_eval = torch.linspace(0.05, 0.95, 37, dtype=torch.float64)
        # (B, n, n_data) with B=2, n=2
        coeffs = torch.tensor(
            [[[ 1.0, -2.0,  3.0],
              [ 0.5,  1.0, -1.0]],
             [[-1.0,  0.0,  2.0],
              [ 2.0, -3.0,  0.5]]],
            dtype=torch.float64,
        )  # (2, 2, 3) = (B, n, 3 coefficients)
        y_data = (
            coeffs[..., 0:1]
            + coeffs[..., 1:2] * t_data
            + coeffs[..., 2:3] * t_data ** 2
        )
        y_exact = (
            coeffs[..., 0:1]
            + coeffs[..., 1:2] * t_eval
            + coeffs[..., 2:3] * t_eval ** 2
        )
        y_interp = interp_quadratic(t_eval, t_data, y_data)
        np.testing.assert_allclose(
            y_interp.numpy(), y_exact.numpy(), atol=1e-12,
        )
