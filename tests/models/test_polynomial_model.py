"""Tests for the PolynomialModel class."""

import pytest
import numpy as np
import torch
from itertools import combinations
from string import ascii_lowercase

from nitrom.latent_space_models.polynomial_model import PolynomialModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(n, poly_comp, seed=42, B=None):
    """Create a PolynomialModel with random tensors for the given poly_comp."""
    rng = np.random.default_rng(seed)
    tensors = []
    for k in poly_comp:
        shape = (n,) * (k + 1)
        tensors.append(torch.tensor(rng.standard_normal(shape), dtype=torch.float64))
    if B is not None:
        tensors.append(B)
        forcing_config = {"forcing_exists": True, "m": B.shape[1]}
    else:
        forcing_config = None
    model = PolynomialModel(
        n, poly_comp, tensors=tensors, forcing_config=forcing_config,
    )
    return model


def _reference_rhs(tensors, poly_comp, z_np):
    """
    Compute the RHS via np.einsum for a single vector z_np of shape (n,).

    For degree k with tensor A of shape (n,)^(k+1):
        np.einsum('ab..., b, c, ... -> a', A, z, z, ...)
    """
    n = z_np.shape[-1]
    dzdt = np.zeros(n)
    for i, k in enumerate(poly_comp):
        subscripts = ascii_lowercase[: k + 1]
        # e.g. k=1 -> 'ab,b->a', k=2 -> 'abc,b,c->a', k=3 -> 'abcd,b,c,d->a'
        lhs = ",".join([subscripts] + list(subscripts[1:]))
        equation = f"{lhs}->{subscripts[0]}"
        A_np = tensors[i].numpy()
        operands = [A_np] + [z_np for _ in range(k)]
        dzdt += np.einsum(equation, *operands)
    return dzdt


def _reference_jacobian(tensors, poly_comp, Z_np):
    """
    Compute the Jacobian J(Z) via np.einsum for a single vector Z_np of shape (n,).

    For degree k with tensor A of shape (n,)^(k+1), the Jacobian contribution
    is the sum over all ways to leave one index free and contract the rest with Z.
    """
    n = Z_np.shape[-1]
    J = np.zeros((n, n))

    for i, k in enumerate(poly_comp):
        if k == 0:
            continue
        subscripts = ascii_lowercase[: k + 1]
        # The einsum subscript parts (same as model.einsum_ss)
        ss_full = subscripts
        ss_contracted = list(subscripts[1:])

        combs = list(combinations(ss_contracted, r=k - 1))
        A_np = tensors[i].numpy()
        for comb in combs:
            lhs = ",".join([ss_full] + list(comb))
            # Output has two free indices: the first (output dim) and the one not in comb
            equation = lhs  # np.einsum will figure out the free indices
            operands = [A_np] + [Z_np for _ in range(k - 1)]
            J += np.einsum(equation, *operands)
    return J


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 6  # state dimension
M = 4  # batch size


@pytest.fixture(params=[
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
    [0, 1],
    [0, 1, 2],
    [0, 2],
    [1, 2],
    [1, 3],
    [2, 3],
    [1, 2, 3],
    [1, 2, 3, 4],
    [1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4, 5],
    [2, 4, 5],
], ids=[
    "deg0",
    "deg1",
    "deg2",
    "deg3",
    "deg4",
    "deg5",
    "deg0+1",
    "deg0+1+2",
    "deg0+2",
    "deg1+2",
    "deg1+3",
    "deg2+3",
    "deg1+2+3",
    "deg1+2+3+4",
    "deg1+2+3+4+5",
    "deg0+1+2+3+4+5",
    "deg2+4+5",
])
def model_and_data(request):
    """Fixture that provides a model, random state vectors, and reference data."""
    poly_comp = request.param
    rng = np.random.default_rng(123)

    model = _make_model(N, poly_comp, seed=42)

    z_np = rng.standard_normal(N)
    z_torch = torch.tensor(z_np, dtype=torch.float64)

    z_batch_np = rng.standard_normal((M, N))
    z_batch_torch = torch.tensor(z_batch_np, dtype=torch.float64)

    return model, poly_comp, z_np, z_torch, z_batch_np, z_batch_torch


# ---------------------------------------------------------------------------
# evaluate_rhs tests
# ---------------------------------------------------------------------------

class TestEvaluateRhs:

    def test_unbatched_no_forcing(self, model_and_data):
        model, poly_comp, z_np, z_torch, _, _ = model_and_data

        result = model.evaluate_rhs(0.0, z_torch)
        expected = _reference_rhs(model.get_params(), poly_comp, z_np)

        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-12)

    def test_unbatched_with_forcing(self, model_and_data):
        _, poly_comp, z_np, z_torch, _, _ = model_and_data

        rng = np.random.default_rng(99)
        B_np = rng.standard_normal((N, N))
        B_torch = torch.tensor(B_np, dtype=torch.float64)
        model_f = _make_model(N, poly_comp, seed=42, B=B_torch)

        forcing_val = rng.standard_normal(N)
        forcing_torch = torch.tensor(forcing_val, dtype=torch.float64)
        f_fun = lambda t: forcing_torch

        result = model_f.evaluate_rhs(0.0, z_torch, external_forcing=[f_fun])
        expected = _reference_rhs(model_f.get_params(), poly_comp, z_np) + B_np @ forcing_val

        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-12)

    def test_batched_no_forcing(self, model_and_data):
        model, poly_comp, _, _, z_batch_np, z_batch_torch = model_and_data

        result = model.evaluate_rhs(0.0, z_batch_torch)

        for j in range(M):
            expected_j = _reference_rhs(model.get_params(), poly_comp, z_batch_np[j])
            np.testing.assert_allclose(
                result[j].numpy(), expected_j, rtol=1e-12,
                err_msg=f"Mismatch at batch index {j}",
            )

    def test_batched_with_forcing(self, model_and_data):
        _, poly_comp, _, _, z_batch_np, z_batch_torch = model_and_data

        rng = np.random.default_rng(77)
        B_np = rng.standard_normal((N, N))
        B_torch = torch.tensor(B_np, dtype=torch.float64)
        model_f = _make_model(N, poly_comp, seed=42, B=B_torch)

        forcing_vals = [rng.standard_normal(N) for _ in range(M)]
        forcing_funs = [
            (lambda v: lambda t: torch.tensor(v, dtype=torch.float64))(fv)
            for fv in forcing_vals
        ]

        result = model_f.evaluate_rhs(0.0, z_batch_torch, external_forcing=forcing_funs)

        for j in range(M):
            expected_j = _reference_rhs(model_f.get_params(), poly_comp, z_batch_np[j]) + B_np @ forcing_vals[j]
            np.testing.assert_allclose(
                result[j].numpy(), expected_j, rtol=1e-10,
                err_msg=f"Mismatch at batch index {j}",
            )

    def test_unbatched_blowup_returns_zeros(self, model_and_data):
        model, _, _, _, _, _ = model_and_data

        z_big = torch.ones(N, dtype=torch.float64) * 1e7
        result = model.evaluate_rhs(0.0, z_big)
        np.testing.assert_array_equal(result.numpy(), np.zeros(N))

    def test_batched_blowup_partial(self, model_and_data):
        model, poly_comp, _, _, _, _ = model_and_data

        rng = np.random.default_rng(55)
        z_batch = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)
        # Make the second row blow up
        z_batch[1] = 1e7

        result = model.evaluate_rhs(0.0, z_batch)

        # Blown-up row should be zeros
        np.testing.assert_array_equal(result[1].numpy(), np.zeros(N))

        # Other rows should match reference
        for j in [0, 2, 3]:
            expected_j = _reference_rhs(model.get_params(), poly_comp, z_batch[j].numpy())
            np.testing.assert_allclose(
                result[j].numpy(), expected_j, rtol=1e-12,
                err_msg=f"Mismatch at batch index {j}",
            )

    def test_batched_all_blowup_returns_zeros(self, model_and_data):
        model, _, _, _, _, _ = model_and_data

        z_batch = torch.ones((M, N), dtype=torch.float64) * 1e7
        result = model.evaluate_rhs(0.0, z_batch)
        np.testing.assert_array_equal(result.numpy(), np.zeros((M, N)))


# ---------------------------------------------------------------------------
# update tests
# ---------------------------------------------------------------------------

class TestUpdateModel:

    def test_update_changes_rhs(self, model_and_data):
        """After update, evaluate_rhs should use the new tensors."""
        model, poly_comp, z_np, z_torch, _, _ = model_and_data

        result_before = model.evaluate_rhs(0.0, z_torch).clone()

        # Create new random tensors
        rng = np.random.default_rng(999)
        new_tensors = []
        for k in poly_comp:
            shape = (N,) * (k + 1)
            new_tensors.append(
                torch.tensor(rng.standard_normal(shape), dtype=torch.float64)
            )

        model.update_params(new_tensors)
        result_after = model.evaluate_rhs(0.0, z_torch)

        expected = _reference_rhs(new_tensors, poly_comp, z_np)
        np.testing.assert_allclose(result_after.numpy(), expected, rtol=1e-12)
        assert not np.allclose(result_before.numpy(), result_after.numpy())

    def test_update_changes_adjoint(self, model_and_data):
        """After update, evaluate_adjoint_rhs should use the new tensors."""
        model, poly_comp, z_np, z_torch, _, _ = model_and_data

        rng = np.random.default_rng(400)
        Z_np = rng.standard_normal(N)
        Z_torch = torch.tensor(Z_np, dtype=torch.float64)

        result_before = model.evaluate_adjoint_rhs(0.0, z_torch, Z_torch).clone()

        # Create new random tensors
        new_tensors = []
        for k in poly_comp:
            shape = (N,) * (k + 1)
            new_tensors.append(
                torch.tensor(rng.standard_normal(shape), dtype=torch.float64)
            )

        model.update_params(new_tensors)
        result_after = model.evaluate_adjoint_rhs(0.0, z_torch, Z_torch)

        J = _reference_jacobian(new_tensors, poly_comp, Z_np)
        expected = J.T @ z_np
        np.testing.assert_allclose(result_after.numpy(), expected, rtol=1e-12)
        if any(k > 0 for k in poly_comp):
            assert not np.allclose(result_before.numpy(), result_after.numpy())

    def test_update_preserves_einsum_subscripts(self, model_and_data):
        """update should not affect the precomputed einsum subscripts."""
        model, poly_comp, _, _, _, _ = model_and_data

        ss_before = model.einsum_ss

        rng = np.random.default_rng(401)
        new_tensors = []
        for k in poly_comp:
            shape = (N,) * (k + 1)
            new_tensors.append(
                torch.tensor(rng.standard_normal(shape), dtype=torch.float64)
            )

        model.update_params(new_tensors)
        assert model.einsum_ss == ss_before


# ---------------------------------------------------------------------------
# evaluate_adjoint_rhs tests
# ---------------------------------------------------------------------------

class TestEvaluateAdjointRhs:

    def test_unbatched(self, model_and_data):
        model, poly_comp, z_np, z_torch, _, _ = model_and_data

        rng = np.random.default_rng(200)
        Z_np = rng.standard_normal(N)
        Z_torch = torch.tensor(Z_np, dtype=torch.float64)

        result = model.evaluate_adjoint_rhs(0.0, z_torch, Z_torch)

        J = _reference_jacobian(model.get_params(), poly_comp, Z_np)
        expected = J.T @ z_np

        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-12)

    def test_batched_shared_baseflow(self, model_and_data):
        """Batched z with a single (unbatched) base flow Z."""
        model, poly_comp, _, _, z_batch_np, z_batch_torch = model_and_data

        rng = np.random.default_rng(201)
        Z_np = rng.standard_normal(N)
        Z_torch = torch.tensor(Z_np, dtype=torch.float64)

        # Expand Z to (m, n) since model expects same shape as z
        Z_batch_torch = Z_torch.unsqueeze(0).expand(M, -1)
        Z_batch_np = np.broadcast_to(Z_np, (M, N))

        result = model.evaluate_adjoint_rhs(0.0, z_batch_torch, Z_batch_torch)

        J = _reference_jacobian(model.get_params(), poly_comp, Z_np)
        for j in range(M):
            expected_j = J.T @ z_batch_np[j]
            np.testing.assert_allclose(
                result[j].numpy(), expected_j, rtol=1e-10,
                err_msg=f"Mismatch at batch index {j}",
            )

    def test_batched_per_sample_baseflow(self, model_and_data):
        """Batched z with per-sample base flow Z."""
        model, poly_comp, _, _, z_batch_np, z_batch_torch = model_and_data

        rng = np.random.default_rng(202)
        Z_batch_np = rng.standard_normal((M, N))
        Z_batch_torch = torch.tensor(Z_batch_np, dtype=torch.float64)

        result = model.evaluate_adjoint_rhs(0.0, z_batch_torch, Z_batch_torch)

        for j in range(M):
            J_j = _reference_jacobian(model.get_params(), poly_comp, Z_batch_np[j])
            expected_j = J_j.T @ z_batch_np[j]
            np.testing.assert_allclose(
                result[j].numpy(), expected_j, rtol=1e-12,
                err_msg=f"Mismatch at batch index {j}",
            )

    def test_finite_difference_unbatched(self, model_and_data):
        """
        Adjoint consistency check via central finite differences.

        J(Z) = q^T f(Z)
        ⟨∇_Z J, δz⟩ = ⟨(∂f/∂Z)^T q, δz⟩ = ⟨evaluate_adjoint_rhs(t, q, Z), δz⟩
        ≈ [J(Z + ε δz) - J(Z - ε δz)] / (2ε)
        """
        model, _, _, _, _, _ = model_and_data

        rng = np.random.default_rng(300)
        Z = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
        q = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
        dz = torch.tensor(rng.standard_normal(N), dtype=torch.float64)

        # Directional derivative from adjoint
        adj = model.evaluate_adjoint_rhs(0.0, q, Z)
        dd_adjoint = torch.dot(adj, dz).item()

        # Directional derivative from central finite differences
        eps = 1e-7
        J_plus = torch.dot(q, model.evaluate_rhs(0.0, Z + eps * dz)).item()
        J_minus = torch.dot(q, model.evaluate_rhs(0.0, Z - eps * dz)).item()
        dd_fd = (J_plus - J_minus) / (2 * eps)

        np.testing.assert_allclose(dd_adjoint, dd_fd, rtol=1e-5)

    def test_finite_difference_batched(self, model_and_data):
        """Batched version of the adjoint finite difference consistency check."""
        model, _, _, _, _, _ = model_and_data

        rng = np.random.default_rng(301)
        Z_batch = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)
        q_batch = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)
        dz_batch = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)

        # Directional derivative from adjoint
        adj = model.evaluate_adjoint_rhs(0.0, q_batch, Z_batch)

        eps = 1e-7
        for j in range(M):
            dd_adjoint = torch.dot(adj[j], dz_batch[j]).item()

            J_plus = torch.dot(
                q_batch[j], model.evaluate_rhs(0.0, Z_batch[j] + eps * dz_batch[j])
            ).item()
            J_minus = torch.dot(
                q_batch[j], model.evaluate_rhs(0.0, Z_batch[j] - eps * dz_batch[j])
            ).item()
            dd_fd = (J_plus - J_minus) / (2 * eps)

            np.testing.assert_allclose(
                dd_adjoint, dd_fd, rtol=1e-5,
                err_msg=f"Mismatch at batch index {j}",
            )

    def test_unbatched_blowup_returns_zeros(self, model_and_data):
        model, _, _, _, _, _ = model_and_data

        z_big = torch.ones(N, dtype=torch.float64) * 1e7
        Z = torch.randn(N, dtype=torch.float64)
        result = model.evaluate_adjoint_rhs(0.0, z_big, Z)
        np.testing.assert_array_equal(result.numpy(), np.zeros(N))

    def test_batched_blowup_partial(self, model_and_data):
        model, poly_comp, _, _, _, _ = model_and_data

        rng = np.random.default_rng(203)
        z_batch = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)
        Z_batch = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)
        # Make the second row blow up
        z_batch[1] = 1e7

        result = model.evaluate_adjoint_rhs(0.0, z_batch, Z_batch)

        # Blown-up row should be zeros
        np.testing.assert_array_equal(result[1].numpy(), np.zeros(N))

        # Other rows should match reference
        for j in [0, 2, 3]:
            J_j = _reference_jacobian(model.get_params(), poly_comp, Z_batch[j].numpy())
            expected_j = J_j.T @ z_batch[j].numpy()
            np.testing.assert_allclose(
                result[j].numpy(), expected_j, rtol=1e-12,
                err_msg=f"Mismatch at batch index {j}",
            )

    def test_batched_all_blowup_returns_zeros(self, model_and_data):
        model, _, _, _, _, _ = model_and_data

        z_batch = torch.ones((M, N), dtype=torch.float64) * 1e7
        Z_batch = torch.randn((M, N), dtype=torch.float64)
        result = model.evaluate_adjoint_rhs(0.0, z_batch, Z_batch)
        np.testing.assert_array_equal(result.numpy(), np.zeros((M, N)))


# ---------------------------------------------------------------------------
# vjp_evaluate_rhs tests
# ---------------------------------------------------------------------------

class TestVjpEvaluateRhs:

    def test_finite_difference_unbatched(self, model_and_data):
        """
        For each tensor A_k, check:
        ⟨∂J/∂A_k, δA_k⟩ ≈ [v^T f(z; A_k + ε δA_k) - v^T f(z; A_k - ε δA_k)] / (2ε)
        """
        model, poly_comp, z_np, z_torch, _, _ = model_and_data

        rng = np.random.default_rng(700)
        v = torch.tensor(rng.standard_normal(N), dtype=torch.float64)

        grads = model.vjp_evaluate_rhs(z_torch, v)

        eps = 1e-7
        for idx, k in enumerate(poly_comp):
            dA = torch.tensor(
                rng.standard_normal(model.get_params()[idx].shape), dtype=torch.float64
            )

            dd_vjp = torch.sum(grads[idx] * dA).item()

            # Perturb tensor idx
            tensors_plus = [t.clone() for t in model.get_params()]
            tensors_minus = [t.clone() for t in model.get_params()]
            tensors_plus[idx] = model.get_params()[idx] + eps * dA
            tensors_minus[idx] = model.get_params()[idx] - eps * dA

            original_tensors = model.get_params()
            model.update_params(tensors_plus)
            J_plus = torch.dot(v, model.evaluate_rhs(0.0, z_torch)).item()
            model.update_params(tensors_minus)
            J_minus = torch.dot(v, model.evaluate_rhs(0.0, z_torch)).item()
            model.update_params(original_tensors)

            dd_fd = (J_plus - J_minus) / (2 * eps)

            np.testing.assert_allclose(
                dd_vjp, dd_fd, rtol=1e-5,
                err_msg=f"Mismatch for tensor index {idx} (degree {k})",
            )

    def test_finite_difference_batched(self, model_and_data):
        """
        Batched version: the VJP should sum contributions over the batch.
        """
        model, poly_comp, _, _, z_batch_np, z_batch_torch = model_and_data

        rng = np.random.default_rng(701)
        v_batch = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)

        grads = model.vjp_evaluate_rhs(z_batch_torch, v_batch)

        eps = 1e-7
        for idx, k in enumerate(poly_comp):
            dA = torch.tensor(
                rng.standard_normal(model.get_params()[idx].shape), dtype=torch.float64
            )

            dd_vjp = torch.sum(grads[idx] * dA).item()

            tensors_plus = [t.clone() for t in model.get_params()]
            tensors_minus = [t.clone() for t in model.get_params()]
            tensors_plus[idx] = model.get_params()[idx] + eps * dA
            tensors_minus[idx] = model.get_params()[idx] - eps * dA

            original_tensors = model.get_params()
            model.update_params(tensors_plus)
            rhs_plus = model.evaluate_rhs(0.0, z_batch_torch)
            model.update_params(tensors_minus)
            rhs_minus = model.evaluate_rhs(0.0, z_batch_torch)
            model.update_params(original_tensors)

            J_plus = torch.sum(v_batch * rhs_plus).item()
            J_minus = torch.sum(v_batch * rhs_minus).item()
            dd_fd = (J_plus - J_minus) / (2 * eps)

            np.testing.assert_allclose(
                dd_vjp, dd_fd, rtol=1e-5,
                err_msg=f"Mismatch for tensor index {idx} (degree {k})",
            )

    def test_gradient_shapes(self, model_and_data):
        """Each gradient tensor should have the same shape as the corresponding operator."""
        model, poly_comp, _, z_torch, _, _ = model_and_data

        v = torch.randn(N, dtype=torch.float64)
        grads = model.vjp_evaluate_rhs(z_torch, v)

        assert len(grads) == len(poly_comp)
        for idx, k in enumerate(poly_comp):
            assert grads[idx].shape == model.get_params()[idx].shape

    def test_finite_difference_B_unbatched(self, model_and_data):
        """FD check for grad_B: ⟨∂J/∂B, δB⟩ ≈ [v^T f(B+εδB) - v^T f(B-εδB)] / 2ε"""
        _, poly_comp, _, z_torch, _, _ = model_and_data

        rng = np.random.default_rng(800)
        P = 3  # forcing dimension
        B = torch.tensor(rng.standard_normal((N, P)), dtype=torch.float64)
        model_f = _make_model(N, poly_comp, seed=42, B=B)

        u_val = torch.tensor(rng.standard_normal(P), dtype=torch.float64)
        f_fun = lambda t: u_val
        v = torch.tensor(rng.standard_normal(N), dtype=torch.float64)

        grads = model_f.vjp_evaluate_rhs(z_torch, v, external_forcing=[f_fun], t=0.0)
        grad_B = grads[-1]
        assert grad_B.shape == (N, P)

        dB = torch.tensor(rng.standard_normal((N, P)), dtype=torch.float64)
        dd_vjp = torch.sum(grad_B * dB).item()

        eps = 1e-7
        params = model_f.get_params()
        params_plus = [t.clone() for t in params]
        params_minus = [t.clone() for t in params]
        params_plus[-1] = B + eps * dB
        params_minus[-1] = B - eps * dB

        model_f.update_params(params_plus)
        J_plus = torch.dot(v, model_f.evaluate_rhs(0.0, z_torch, external_forcing=[f_fun])).item()
        model_f.update_params(params_minus)
        J_minus = torch.dot(v, model_f.evaluate_rhs(0.0, z_torch, external_forcing=[f_fun])).item()
        model_f.update_params(params)

        dd_fd = (J_plus - J_minus) / (2 * eps)
        np.testing.assert_allclose(dd_vjp, dd_fd, rtol=1e-5)

    def test_finite_difference_B_batched(self, model_and_data):
        """Batched FD check for grad_B."""
        _, poly_comp, _, _, _, z_batch_torch = model_and_data

        rng = np.random.default_rng(801)
        P = 3
        B = torch.tensor(rng.standard_normal((N, P)), dtype=torch.float64)
        model_f = _make_model(N, poly_comp, seed=42, B=B)

        u_vals = [torch.tensor(rng.standard_normal(P), dtype=torch.float64) for _ in range(M)]
        f_funs = [(lambda u: lambda t: u)(u) for u in u_vals]
        v_batch = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)

        grads = model_f.vjp_evaluate_rhs(z_batch_torch, v_batch, external_forcing=f_funs, t=0.0)
        grad_B = grads[-1]
        assert grad_B.shape == (N, P)

        dB = torch.tensor(rng.standard_normal((N, P)), dtype=torch.float64)
        dd_vjp = torch.sum(grad_B * dB).item()

        eps = 1e-7
        params = model_f.get_params()
        params_plus = [t.clone() for t in params]
        params_minus = [t.clone() for t in params]
        params_plus[-1] = B + eps * dB
        params_minus[-1] = B - eps * dB

        model_f.update_params(params_plus)
        rhs_plus = model_f.evaluate_rhs(0.0, z_batch_torch, external_forcing=f_funs)
        model_f.update_params(params_minus)
        rhs_minus = model_f.evaluate_rhs(0.0, z_batch_torch, external_forcing=f_funs)
        model_f.update_params(params)

        J_plus = torch.sum(v_batch * rhs_plus).item()
        J_minus = torch.sum(v_batch * rhs_minus).item()
        dd_fd = (J_plus - J_minus) / (2 * eps)

        np.testing.assert_allclose(dd_vjp, dd_fd, rtol=1e-5)

    def test_no_grad_B_without_forcing(self, model_and_data):
        """Without forcing, grads should only contain tensor gradients, no grad_B."""
        model, poly_comp, _, z_torch, _, _ = model_and_data

        v = torch.randn(N, dtype=torch.float64)
        grads = model.vjp_evaluate_rhs(z_torch, v)
        assert len(grads) == len(poly_comp)
