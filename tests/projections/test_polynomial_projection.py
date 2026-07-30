"""Tests for the PolynomialProjection class."""

import pytest
import numpy as np
import torch
from string import ascii_lowercase

from nitrom.projections.polynomial_projection import PolynomialProjection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 10  # full-space dimension
R = 3   # reduced-space dimension
M = 5   # batch size


def _random_tensors(n, r, nonlin_poly_comp, seed=42):
    """Generate random Phi, Psi, and A_k tensors."""
    rng = np.random.default_rng(seed)
    Phi = torch.tensor(rng.standard_normal((n, r)), dtype=torch.float64)
    Psi = torch.tensor(rng.standard_normal((n, r)), dtype=torch.float64)
    tensors = [Phi, Psi]
    for k in nonlin_poly_comp:
        tensors.append(
            torch.tensor(
                0.1 * rng.standard_normal((n,) + (r,) * k), dtype=torch.float64
            )
        )
    return tensors


@pytest.fixture(params=[[2], [3], [2, 3]], ids=["deg2", "deg3", "deg2+3"])
def proj(request):
    nonlin = request.param
    tensors = _random_tensors(N, R, nonlin)
    return PolynomialProjection(nonlin, tensors), nonlin


# ---------------------------------------------------------------------------
# Basic / construction tests
# ---------------------------------------------------------------------------

class TestBasic:

    def test_param_names(self, proj):
        proj_obj, nonlin = proj
        expected = ["Phi", "Psi"] + [f"A{k}" for k in nonlin]
        assert proj_obj.param_names == expected

    def test_get_params_length(self, proj):
        proj_obj, nonlin = proj
        assert len(proj_obj.get_params()) == 2 + len(nonlin)

    def test_ascending_order_enforced(self):
        tensors = _random_tensors(N, R, [2, 3])
        with pytest.raises(ValueError, match="ascending"):
            PolynomialProjection([3, 2], tensors)

    def test_dimensions(self, proj):
        proj_obj, _ = proj
        assert proj_obj.ambient_space_dimension == N
        assert proj_obj.latent_space_dimension == R


# ---------------------------------------------------------------------------
# encode tests
# ---------------------------------------------------------------------------

class TestEncode:

    def test_unbatched_shape(self, proj):
        proj_obj, _ = proj
        q = torch.randn(N, dtype=torch.float64)
        assert proj_obj.encode(q).shape == (R,)

    def test_batched_shape(self, proj):
        proj_obj, _ = proj
        q = torch.randn(M, N, dtype=torch.float64)
        assert proj_obj.encode(q).shape == (M, R)

    def test_unbatched_value(self, proj):
        proj_obj, _ = proj
        q = torch.randn(N, dtype=torch.float64)
        expected = proj_obj.Psi.T @ q
        np.testing.assert_allclose(
            proj_obj.encode(q).numpy(), expected.numpy(), rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# decode tests
# ---------------------------------------------------------------------------

class TestDecode:

    def test_unbatched_shape(self, proj):
        proj_obj, _ = proj
        z = torch.randn(R, dtype=torch.float64)
        assert proj_obj.decode(z).shape == (N,)

    def test_batched_shape(self, proj):
        proj_obj, _ = proj
        z = torch.randn(M, R, dtype=torch.float64)
        assert proj_obj.decode(z).shape == (M, N)

    def test_reduces_to_linear_when_A_zero(self):
        """When all A_k = 0, decode should equal the linear decode."""
        rng = np.random.default_rng(99)
        Phi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        Psi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        A2 = torch.zeros(N, R, R, dtype=torch.float64)
        proj_obj = PolynomialProjection([2], [Phi, Psi, A2])

        z = torch.randn(R, dtype=torch.float64)
        S = torch.linalg.inv(Psi.T @ Phi)
        expected = Phi @ (S @ z)
        np.testing.assert_allclose(
            proj_obj.decode(z).numpy(), expected.numpy(), rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# encode-decode consistency
# ---------------------------------------------------------------------------

class TestEncodeDecode:

    def test_encode_decode_identity_in_reduced_space(self, proj):
        r"""encode(decode(z)) = z because Psi^T P = 0."""
        proj_obj, _ = proj
        z = torch.randn(R, dtype=torch.float64)
        z_rt = proj_obj.encode(proj_obj.decode(z))
        np.testing.assert_allclose(z_rt.numpy(), z.numpy(), rtol=1e-10)

    def test_encode_decode_identity_batched(self, proj):
        proj_obj, _ = proj
        z = torch.randn(M, R, dtype=torch.float64)
        z_rt = proj_obj.encode(proj_obj.decode(z))
        np.testing.assert_allclose(z_rt.numpy(), z.numpy(), rtol=1e-10)


# ---------------------------------------------------------------------------
# update tests
# ---------------------------------------------------------------------------

class TestUpdate:

    def test_update_changes_params(self, proj):
        proj_obj, nonlin = proj
        old_Phi = proj_obj.Phi.clone()
        new_tensors = _random_tensors(N, R, nonlin, seed=999)
        proj_obj.update(new_tensors)
        assert not torch.equal(proj_obj.Phi, old_Phi)
        assert torch.equal(proj_obj.Phi, new_tensors[0])

    def test_S_recomputed_after_update(self, proj):
        proj_obj, nonlin = proj
        new_tensors = _random_tensors(N, R, nonlin, seed=888)
        proj_obj.update(new_tensors)
        expected_S = torch.linalg.inv(new_tensors[1].T @ new_tensors[0])
        np.testing.assert_allclose(
            proj_obj.S.numpy(), expected_S.numpy(), rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# vjp_encode tests (autograd)
# ---------------------------------------------------------------------------

class TestVjpEncode:

    def _autograd_check(self, proj_obj, z_shape):
        rng = np.random.default_rng(500)
        Psi = proj_obj.Psi.clone().requires_grad_(True)

        if z_shape is None:
            q = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
            v = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
            J = torch.dot(v, Psi.T @ q)
        else:
            q = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)
            v = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
            J = torch.sum(v * (Psi.T @ q.T).T)

        J.backward()
        grad_auto = Psi.grad.clone()
        Psi.requires_grad_(False)

        grads = proj_obj.vjp_encode(q, v)
        # grad_Phi should be zero
        np.testing.assert_allclose(
            grads[0].numpy(), 0.0, atol=1e-15,
        )
        # grad_Psi
        np.testing.assert_allclose(
            grads[1].numpy(), grad_auto.numpy(), rtol=1e-10,
        )
        # grad_A_k should all be zero
        for g in grads[2:]:
            np.testing.assert_allclose(g.numpy(), 0.0, atol=1e-15)

    def test_unbatched(self, proj):
        proj_obj, _ = proj
        self._autograd_check(proj_obj, None)

    def test_batched(self, proj):
        proj_obj, _ = proj
        self._autograd_check(proj_obj, M)


# ---------------------------------------------------------------------------
# vjp_decode tests (autograd)
# ---------------------------------------------------------------------------

class TestVjpDecode:

    def _decode_autograd(self, Phi, Psi, A_tensors, nonlin, z, v):
        """Compute autograd gradients of v^T decode(z)."""
        Phi_ = Phi.clone().requires_grad_(True)
        Psi_ = Psi.clone().requires_grad_(True)
        As_ = [A.clone().requires_grad_(True) for A in A_tensors]

        S_ = torch.linalg.inv(Psi_.T @ Phi_)

        # g(z)
        if z.ndim == 1:
            g = torch.zeros(Phi.shape[0], dtype=torch.float64)
        else:
            g = torch.zeros(z.shape[0], Phi.shape[0], dtype=torch.float64)

        for i, k in enumerate(nonlin):
            ssk = ascii_lowercase[: k + 1]
            parts = [ssk] + [s for s in ssk[1:]]
            if z.ndim == 1:
                eq = ",".join(parts)
            else:
                eq = ",".join([parts[0]] + [f"...{p}" for p in parts[1:]])
            ops = [As_[i]] + [z for _ in range(k)]
            g = g + torch.einsum(eq, *ops)

        # q = Phi S z + P g
        if z.ndim == 1:
            q = Phi_ @ (S_ @ z) + g - Phi_ @ (S_ @ (Psi_.T @ g))
            J = torch.dot(v, q)
        else:
            q = (Phi_ @ (S_ @ z.T)).T + g - (Phi_ @ (S_ @ (Psi_.T @ g.T))).T
            J = torch.sum(v * q)

        J.backward()
        return [Phi_.grad.clone(), Psi_.grad.clone()] + [
            a.grad.clone() for a in As_
        ]

    def _check(self, proj_obj, nonlin, unbatched):
        rng = np.random.default_rng(600)
        if unbatched:
            z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
            v = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
        else:
            z = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
            v = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)

        A_tensors = [getattr(proj_obj, f"A{k}") for k in nonlin]
        grads_auto = self._decode_autograd(
            proj_obj.Phi, proj_obj.Psi, A_tensors, nonlin, z, v,
        )
        grads_analytic = proj_obj.vjp_decode(z, v)

        names = ["Phi", "Psi"] + [f"A{k}" for k in nonlin]
        for name, ga, gad in zip(names, grads_analytic, grads_auto):
            np.testing.assert_allclose(
                ga.detach().numpy(),
                gad.detach().numpy(),
                rtol=1e-8,
                atol=1e-12,
                err_msg=f"{name}: analytic vs autograd mismatch",
            )

    def test_unbatched(self, proj):
        proj_obj, nonlin = proj
        self._check(proj_obj, nonlin, unbatched=True)

    def test_batched(self, proj):
        proj_obj, nonlin = proj
        self._check(proj_obj, nonlin, unbatched=False)
