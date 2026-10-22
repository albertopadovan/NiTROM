"""Direct finite-difference checks of cached and reconstructed RK adjoints."""

import numpy as np
import pytest

from nitrom.time_steppers import solve_adjoint_ivp_discrete, solve_fixed_ivp


@pytest.mark.parametrize("method", ["rk2", "rk4", "backward_euler"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_adjoint_matches_discrete_loss(backend, method, batched, cached):
    theta = -0.3
    initial = [[0.7], [-0.3]] if batched else [0.7]
    initial = backend.asarray(initial, dtype=backend.float64)

    def rhs(t, z):
        return theta * z

    def solve(x):
        return solve_fixed_ivp(
            rhs, x, 0.0, 0.2, 0.05, method, with_stages=True, backend=backend
        )

    solution = solve(initial)
    final = solution.X[..., -1]
    seed, gradients = solve_adjoint_ivp_discrete(
        rhs,
        lambda t, w, z: theta * w,
        lambda z, w, t: [(z * w).sum()],
        solution.X,
        solution.t,
        2.0 * final,
        method,
        stages_G=solution.stages if cached else None,
        backend=backend,
    )
    eps = 1e-6

    def loss(x):
        z = solve(x).X[..., -1]
        return float((z * z).sum())

    theta += eps
    plus = loss(initial)
    theta -= 2.0 * eps
    minus = loss(initial)
    theta += eps
    np.testing.assert_allclose(
        float(gradients[0]), (plus - minus) / (2.0 * eps), rtol=2e-6
    )
    for index in np.ndindex(tuple(initial.shape)):
        perturbation = backend.zeros_like(initial)
        perturbation[index] = eps
        numeric = (loss(initial + perturbation) - loss(initial - perturbation)) / (
            2.0 * eps
        )
        np.testing.assert_allclose(float(seed[index]), numeric, rtol=2e-6)
