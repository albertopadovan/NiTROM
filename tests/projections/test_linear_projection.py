"""Tests for the LinearProjection class."""

import pytest
import numpy as np
import torch

from nitrom.projections.linear_projection import LinearProjection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 10  # full-space dimension
R = 3   # reduced-space dimension
M = 5   # batch size


def _random_bases(n, r, seed=42):
    """Generate random full-rank trial (Phi) and test (Psi) bases."""
    rng = np.random.default_rng(seed)
    Phi = torch.tensor(rng.standard_normal((n, r)), dtype=torch.float64)
    Psi = torch.tensor(rng.standard_normal((n, r)), dtype=torch.float64)
    return Phi, Psi


@pytest.fixture
def proj():
    Phi, Psi = _random_bases(N, R)
    return LinearProjection([Phi, Psi])


# ---------------------------------------------------------------------------
# encode tests
# ---------------------------------------------------------------------------

class TestEncode:

    def test_unbatched_shape(self, proj):
        q = torch.randn(N, dtype=torch.float64)
        z = proj.encode(q)
        assert z.shape == (R,)

    def test_batched_shape(self, proj):
        q = torch.randn(M, N, dtype=torch.float64)
        z = proj.encode(q)
        assert z.shape == (M, R)

    def test_unbatched_value(self, proj):
        """encode(q) = Psi^T q"""
        q = torch.randn(N, dtype=torch.float64)
        z = proj.encode(q)
        expected = proj.Psi.T @ q
        np.testing.assert_allclose(z.numpy(), expected.numpy(), rtol=1e-12)

    def test_batched_value(self, proj):
        """Each row of the batch should match the unbatched result."""
        q = torch.randn(M, N, dtype=torch.float64)
        z = proj.encode(q)
        for j in range(M):
            expected_j = proj.Psi.T @ q[j]
            np.testing.assert_allclose(z[j].numpy(), expected_j.numpy(), rtol=1e-12)


# ---------------------------------------------------------------------------
# decode tests
# ---------------------------------------------------------------------------

class TestDecode:

    def test_unbatched_shape(self, proj):
        z = torch.randn(R, dtype=torch.float64)
        q = proj.decode(z)
        assert q.shape == (N,)

    def test_batched_shape(self, proj):
        z = torch.randn(M, R, dtype=torch.float64)
        q = proj.decode(z)
        assert q.shape == (M, N)

    def test_unbatched_value(self, proj):
        """decode(z) = Phi S z where S = (Psi^T Phi)^{-1}"""
        z = torch.randn(R, dtype=torch.float64)
        q = proj.decode(z)
        S = torch.linalg.inv(proj.Psi.T @ proj.Phi)
        expected = proj.Phi @ (S @ z)
        np.testing.assert_allclose(q.numpy(), expected.numpy(), rtol=1e-12)

    def test_batched_value(self, proj):
        z = torch.randn(M, R, dtype=torch.float64)
        q = proj.decode(z)
        S = torch.linalg.inv(proj.Psi.T @ proj.Phi)
        for j in range(M):
            expected_j = proj.Phi @ (S @ z[j])
            np.testing.assert_allclose(q[j].numpy(), expected_j.numpy(), rtol=1e-12)


# ---------------------------------------------------------------------------
# encode-decode consistency tests
# ---------------------------------------------------------------------------

class TestEncodeDecode:

    def test_decode_encode_is_projection(self, proj):
        """
        For q in the column space of Phi, decode(encode(q)) should recover q
        (up to the oblique projection).
        Specifically: Phi S Psi^T (Phi c) = Phi S (Psi^T Phi) c = Phi c = q.
        """
        c = torch.randn(R, dtype=torch.float64)
        q = proj.Phi @ c  # q lives in col(Phi)
        z = proj.encode(q)
        q_reconstructed = proj.decode(z)
        np.testing.assert_allclose(
            q_reconstructed.numpy(), q.numpy(), rtol=1e-10
        )

    def test_decode_encode_is_projection_batched(self, proj):
        C = torch.randn(M, R, dtype=torch.float64)
        Q = (proj.Phi @ C.T).T  # (M, N), each row in col(Phi)
        Z = proj.encode(Q)
        Q_reconstructed = proj.decode(Z)
        np.testing.assert_allclose(
            Q_reconstructed.numpy(), Q.numpy(), rtol=1e-10
        )

    def test_encode_decode_is_identity_in_reduced_space(self, proj):
        """encode(decode(z)) = Psi^T Phi S z = (Psi^T Phi)(Psi^T Phi)^{-1} z = z"""
        z = torch.randn(R, dtype=torch.float64)
        z_roundtrip = proj.encode(proj.decode(z))
        np.testing.assert_allclose(
            z_roundtrip.numpy(), z.numpy(), rtol=1e-10
        )

    def test_encode_decode_is_identity_in_reduced_space_batched(self, proj):
        Z = torch.randn(M, R, dtype=torch.float64)
        Z_roundtrip = proj.encode(proj.decode(Z))
        np.testing.assert_allclose(
            Z_roundtrip.numpy(), Z.numpy(), rtol=1e-10
        )

    def test_projection_is_idempotent(self, proj):
        """Applying decode(encode(.)) twice should give the same result."""
        q = torch.randn(N, dtype=torch.float64)
        q1 = proj.decode(proj.encode(q))
        q2 = proj.decode(proj.encode(q1))
        np.testing.assert_allclose(q2.numpy(), q1.numpy(), rtol=1e-10)


# ---------------------------------------------------------------------------
# Orthogonal special case: Phi = Psi
# ---------------------------------------------------------------------------

class TestOrthogonalCase:

    def test_orthogonal_bases(self):
        """When Phi = Psi and columns are orthonormal, S = I."""
        rng = np.random.default_rng(99)
        A = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        Phi, _ = torch.linalg.qr(A)
        proj = LinearProjection([Phi, Phi])

        np.testing.assert_allclose(
            proj.S.numpy(), np.eye(R), atol=1e-12
        )

        # decode(encode(q)) = Phi Phi^T q (orthogonal projection)
        q = torch.randn(N, dtype=torch.float64)
        q_proj = proj.decode(proj.encode(q))
        expected = Phi @ (Phi.T @ q)
        np.testing.assert_allclose(q_proj.numpy(), expected.numpy(), rtol=1e-10)


# ---------------------------------------------------------------------------
# update tests
# ---------------------------------------------------------------------------

class TestUpdate:

    def test_update_changes_bases(self, proj):
        Phi_old = proj.Phi.clone()
        Psi_old = proj.Psi.clone()

        Phi_new, Psi_new = _random_bases(N, R, seed=999)
        proj.update([Phi_new, Psi_new])

        assert torch.equal(proj.Phi, Phi_new)
        assert torch.equal(proj.Psi, Psi_new)
        assert not torch.equal(proj.Phi, Phi_old)
        assert not torch.equal(proj.Psi, Psi_old)

    def test_update_recomputes_S(self, proj):
        Phi_new, Psi_new = _random_bases(N, R, seed=888)
        proj.update([Phi_new, Psi_new])

        expected_S = torch.linalg.inv(Psi_new.T @ Phi_new)
        np.testing.assert_allclose(
            proj.S.numpy(), expected_S.numpy(), rtol=1e-12
        )

    def test_encode_decode_consistent_after_update(self, proj):
        Phi_new, Psi_new = _random_bases(N, R, seed=777)
        proj.update([Phi_new, Psi_new])

        c = torch.randn(R, dtype=torch.float64)
        q = proj.Phi @ c
        q_roundtrip = proj.decode(proj.encode(q))
        np.testing.assert_allclose(
            q_roundtrip.numpy(), q.numpy(), rtol=1e-10
        )


# ---------------------------------------------------------------------------
# vjp_encode tests (autograd)
# ---------------------------------------------------------------------------

class TestVjpEncode:

    def _build_encode_fn(self, Phi, Psi, q):
        """Build a differentiable encode: Psi^T q."""
        def fn(psi):
            return psi.T @ q
        return fn

    def test_autograd_unbatched(self):
        rng = np.random.default_rng(500)
        Phi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        Psi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64, requires_grad=True)
        q = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal(R), dtype=torch.float64)

        # Autograd
        z = Psi.T @ q
        J = torch.dot(v, z)
        J.backward()
        grad_Psi_auto = Psi.grad.clone()
        Psi.grad = None
        Psi.requires_grad_(False)

        # Analytic
        proj = LinearProjection([Phi, Psi])
        _, grad_Psi_analytic = proj.vjp_encode(q, v)

        np.testing.assert_allclose(
            grad_Psi_analytic.numpy(), grad_Psi_auto.numpy(), rtol=1e-10,
        )

    def test_autograd_batched(self):
        rng = np.random.default_rng(502)
        Phi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        Psi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64, requires_grad=True)
        q = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)

        # Autograd
        z = (Psi.T @ q.T).T  # (M, R)
        J = torch.sum(v * z)
        J.backward()
        grad_Psi_auto = Psi.grad.clone()
        Psi.grad = None
        Psi.requires_grad_(False)

        # Analytic
        proj = LinearProjection([Phi, Psi])
        _, grad_Psi_analytic = proj.vjp_encode(q, v)

        np.testing.assert_allclose(
            grad_Psi_analytic.numpy(), grad_Psi_auto.numpy(), rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# vjp_decode tests (autograd)
# ---------------------------------------------------------------------------

class TestVjpDecode:

    def _autograd_decode(self, Phi, Psi, z, v):
        """Compute grad_Phi, grad_Psi of J = v^T decode(z) via autograd."""
        Phi_ = Phi.clone().requires_grad_(True)
        Psi_ = Psi.clone().requires_grad_(True)
        S = torch.linalg.inv(Psi_.T @ Phi_)
        if z.ndim == 1:
            q_hat = Phi_ @ (S @ z)
            J = torch.dot(v, q_hat)
        else:
            q_hat = (Phi_ @ (S @ z.T)).T  # (M, N)
            J = torch.sum(v * q_hat)
        J.backward()
        return Phi_.grad.clone(), Psi_.grad.clone()

    def test_autograd_Phi_unbatched(self):
        rng = np.random.default_rng(600)
        Phi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        Psi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal(N), dtype=torch.float64)

        grad_Phi_auto, _ = self._autograd_decode(Phi, Psi, z, v)
        proj = LinearProjection([Phi, Psi])
        grad_Phi_analytic, _ = proj.vjp_decode(z, v)

        np.testing.assert_allclose(
            grad_Phi_analytic.numpy(), grad_Phi_auto.numpy(), rtol=1e-8,
        )

    def test_autograd_Psi_unbatched(self):
        rng = np.random.default_rng(601)
        Phi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        Psi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal(N), dtype=torch.float64)

        _, grad_Psi_auto = self._autograd_decode(Phi, Psi, z, v)
        proj = LinearProjection([Phi, Psi])
        _, grad_Psi_analytic = proj.vjp_decode(z, v)

        np.testing.assert_allclose(
            grad_Psi_analytic.numpy(), grad_Psi_auto.numpy(), rtol=1e-8,
        )

    def test_autograd_Phi_batched(self):
        rng = np.random.default_rng(602)
        Phi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        Psi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        z = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)

        grad_Phi_auto, _ = self._autograd_decode(Phi, Psi, z, v)
        proj = LinearProjection([Phi, Psi])
        grad_Phi_analytic, _ = proj.vjp_decode(z, v)

        np.testing.assert_allclose(
            grad_Phi_analytic.numpy(), grad_Phi_auto.numpy(), rtol=1e-8,
        )

    def test_autograd_Psi_batched(self):
        rng = np.random.default_rng(603)
        Phi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        Psi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        z = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)

        _, grad_Psi_auto = self._autograd_decode(Phi, Psi, z, v)
        proj = LinearProjection([Phi, Psi])
        _, grad_Psi_analytic = proj.vjp_decode(z, v)

        np.testing.assert_allclose(
            grad_Psi_analytic.numpy(), grad_Psi_auto.numpy(), rtol=1e-8,
        )
