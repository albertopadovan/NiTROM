import numpy as np
import pytest
import torch

from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.time_steppers.time_stepper import evolve, solve_ivp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 4  # state dimension
B = 3  # batch size
T0 = 0.0
TF = 0.5
DT_BASE = 1 / 500  # coarsest dt; each level halves it
N_LEVELS = 5  # number of refinement levels


def _make_model(rng):
    """Create a PolynomialModel with poly_comp=[1, 2] and random tensors."""
    A1 = torch.tensor(rng.standard_normal((N, N)), dtype=torch.float64)
    # Make A1 strongly stable (shift eigenvalues well to the left)
    A1 = A1 - 10.0 * torch.eye(N, dtype=torch.float64)
    A2 = torch.tensor(
        0.01 * rng.standard_normal((N, N, N)), dtype=torch.float64
    )
    model = PolynomialModel(N, [1, 2], tensors=[A1, A2])
    return model


def _solve_at_tf(model, x0, dt, method):
    """
    Integrate the model from T0 to TF with step size dt using the
    evolve function directly and return the solution at TF.
    """
    nt = int(np.round((TF - T0) / dt))
    dt_actual = (TF - T0) / nt
    x = x0.clone()
    t = T0
    # Use tighter tolerance for implicit convergence order checks
    kwargs = {}
    if method == "backward_euler":
        kwargs["newton_tol"] = 1e-14
        kwargs["newton_max_iter"] = 30
    for _ in range(nt):
        x = evolve(model.evaluate_rhs, t, x, dt_actual, method, **kwargs)
        t += dt_actual
    return x


def _compute_successive_errors(model, x0, method, dt_base=DT_BASE):
    """
    Return arrays of dt values and successive-solution errors
    e_j = ||sol_j(tf) - sol_{j-1}(tf)||.
    """
    dts = [dt_base / (2 ** k) for k in range(N_LEVELS)]
    sols = []
    for dt in dts:
        sol = _solve_at_tf(model, x0, dt, method)
        sols.append(sol)

    errors = []
    for j in range(1, len(sols)):
        err = torch.linalg.norm(sols[j] - sols[j - 1]).item()
        errors.append(err)

    return dts, errors


def _check_order(errors, expected_order, rtol=1e-2):
    """
    Given successive errors e_1, e_2, ..., check that
    e_j / e_{j+1} ≈ 2^p.
    """
    expected_ratio = 2 ** expected_order
    # Use the last ratio (most asymptotic) and allow custom tolerance
    # because successive-difference convergence has higher-order bias
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    last_ratio = ratios[-1]
    np.testing.assert_allclose(
        last_ratio, expected_ratio, rtol=rtol,
        err_msg=f"Ratio = {last_ratio:.6f}, expected {expected_ratio:.1f}",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    rng = np.random.default_rng(42)
    return _make_model(rng)


@pytest.fixture
def x0_unbatched():
    rng = np.random.default_rng(100)
    x0 = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
    return 0.01 * x0 / torch.linalg.vector_norm(x0)


@pytest.fixture
def x0_batched():
    rng = np.random.default_rng(200)
    x0 = torch.tensor(rng.standard_normal((B, N)), dtype=torch.float64)
    norms = torch.linalg.vector_norm(x0, dim=-1, keepdim=True)
    return 0.01 * x0 / norms


# ---------------------------------------------------------------------------
# RK4 convergence tests
# ---------------------------------------------------------------------------

class TestRK4Convergence:

    def test_unbatched(self, model, x0_unbatched):
        _, errors = _compute_successive_errors(
            model, x0_unbatched, "rk4"
        )
        _check_order(errors, expected_order=4)

    def test_batched(self, model, x0_batched):
        _, errors = _compute_successive_errors(
            model, x0_batched, "rk4"
        )
        _check_order(errors, expected_order=4)


# ---------------------------------------------------------------------------
# RK2 convergence tests
# ---------------------------------------------------------------------------

class TestRK2Convergence:

    def test_unbatched(self, model, x0_unbatched):
        _, errors = _compute_successive_errors(
            model, x0_unbatched, "rk2"
        )
        _check_order(errors, expected_order=2)

    def test_batched(self, model, x0_batched):
        _, errors = _compute_successive_errors(
            model, x0_batched, "rk2"
        )
        _check_order(errors, expected_order=2)


# ---------------------------------------------------------------------------
# Backward Euler convergence tests
# ---------------------------------------------------------------------------

class TestBackwardEulerConvergence:

    def test_unbatched(self, model, x0_unbatched):
        _, errors = _compute_successive_errors(
            model, x0_unbatched, "backward_euler"
        )
        _check_order(errors, expected_order=1)

    def test_batched(self, model, x0_batched):
        _, errors = _compute_successive_errors(
            model, x0_batched, "backward_euler"
        )
        _check_order(errors, expected_order=1)


# ---------------------------------------------------------------------------
# Batched vs unbatched consistency
# ---------------------------------------------------------------------------

class TestBatchedConsistency:

    @pytest.mark.parametrize("method", ["rk4", "rk2", "backward_euler", "rk45"])
    def test_batched_matches_unbatched(self, model, method):
        """Each row of the batched solution should match the
        corresponding unbatched solve."""
        rng = np.random.default_rng(300)
        x0_batch = torch.tensor(
            rng.standard_normal((B, N)), dtype=torch.float64
        )
        norms = torch.linalg.vector_norm(x0_batch, dim=-1, keepdim=True)
        x0_batch = 0.01 * x0_batch / norms
        dt = 1e-3

        sol_batched = _solve_at_tf(model, x0_batch, dt, method)  # (B, n)

        rtol = 1e-8 if method == "backward_euler" else 1e-12
        atol = 1e-13 if method == "backward_euler" else 0.0
        for b in range(B):
            sol_single = _solve_at_tf(model, x0_batch[b], dt, method)  # (n,)
            np.testing.assert_allclose(
                sol_batched[b].detach().numpy(), sol_single.detach().numpy(),
                rtol=rtol, atol=atol,
                err_msg=f"Batch index {b} mismatch for method={method}",
            )


# ---------------------------------------------------------------------------
# Autograd verification through implicit solver
# ---------------------------------------------------------------------------

class TestImplicitAutograd:

    @pytest.mark.parametrize("method", ["backward_euler", "rk45"])
    def test_gradients_propagate(self, model, x0_unbatched, method):
        """Verify that gradients propagate back to model parameters through solve_ivp."""
        params = model.get_params()
        for p in params:
            p.requires_grad_(True)

        t_eval = torch.tensor([0.1, 0.2], dtype=torch.float64)
        sol = solve_ivp(
            model.evaluate_rhs,
            x0_unbatched,
            t0=0.0,
            tf=0.2,
            dt=0.02,
            t_eval=t_eval,
            method=method,
        )

        loss = sol.sum()
        loss.backward()

        for i, p in enumerate(params):
            if model.poly_comp[i] == 0:
                continue
            assert p.grad is not None
            assert not torch.allclose(p.grad, torch.zeros_like(p.grad))


# ---------------------------------------------------------------------------
# RK45 convergence tests (fixed step size)
# ---------------------------------------------------------------------------

class TestRK45Convergence:

    def test_unbatched(self, model, x0_unbatched):
        _, errors = _compute_successive_errors(
            model, x0_unbatched, "rk45", dt_base=0.04
        )
        _check_order(errors, expected_order=5, rtol=1e-1)

    def test_batched(self, model, x0_batched):
        _, errors = _compute_successive_errors(
            model, x0_batched, "rk45", dt_base=0.04
        )
        _check_order(errors, expected_order=5, rtol=1e-1)


# ---------------------------------------------------------------------------
# RK45 adaptive stepping tests
# ---------------------------------------------------------------------------

class TestRK45Adaptive:

    def test_adaptive_unbatched(self, model, x0_unbatched):
        t_eval = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)

        # Reference solution computed with a tiny fixed step size
        sol_ref = solve_ivp(
            model.evaluate_rhs, x0_unbatched, t0=T0, tf=TF,
            dt=1e-4, t_eval=t_eval, method="rk4"
        )

        # Adaptive solution with tight tolerances
        sol_adaptive_tight = solve_ivp(
            model.evaluate_rhs, x0_unbatched, t0=T0, tf=TF,
            dt=1e-2, t_eval=t_eval, method="rk45",
            atol=1e-10, rtol=1e-8
        )

        # Adaptive solution with loose tolerances
        sol_adaptive_loose = solve_ivp(
            model.evaluate_rhs, x0_unbatched, t0=T0, tf=TF,
            dt=1e-2, t_eval=t_eval, method="rk45",
            atol=1e-4, rtol=1e-3
        )

        # Check that both solutions match the reference
        # The tight solution should be closer to the reference than the loose one
        err_tight = torch.linalg.norm(sol_adaptive_tight - sol_ref).item()
        err_loose = torch.linalg.norm(sol_adaptive_loose - sol_ref).item()

        assert err_tight < err_loose
        assert err_tight < 1e-6
        assert err_loose < 1e-2

    def test_adaptive_batched(self, model, x0_batched):
        t_eval = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
        sol = solve_ivp(
            model.evaluate_rhs, x0_batched, t0=T0, tf=TF,
            dt=1e-2, t_eval=t_eval, method="rk45",
            atol=1e-6, rtol=1e-4
        )
        assert sol.shape == (B, N, len(t_eval))


