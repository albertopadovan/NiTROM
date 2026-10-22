"""Tests for the LinearProjection class."""

import numpy as np
import pytest

from nitrom.projections.linear_projection import LinearProjection

torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 10  # full-space dimension
R = 3  # reduced-space dimension
M = 5  # batch size


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
    def test_unbatched_value(self, proj):
        """encode(q) = Psi^T q"""
        q = torch.randn(N, dtype=torch.float64)
        z = proj.encode(q)
        assert z.shape == (R,)
        expected = proj.Psi.T @ q
        np.testing.assert_allclose(z.numpy(), expected.numpy(), rtol=1e-12)

    def test_batched_value(self, proj):
        """Each row of the batch should match the unbatched result."""
        q = torch.randn(M, N, dtype=torch.float64)
        z = proj.encode(q)
        assert z.shape == (M, R)
        for j in range(M):
            expected_j = proj.Psi.T @ q[j]
            np.testing.assert_allclose(z[j].numpy(), expected_j.numpy(), rtol=1e-12)


# ---------------------------------------------------------------------------
# decode tests
# ---------------------------------------------------------------------------


class TestDecode:
    def test_unbatched_value(self, proj):
        """decode(z) = Phi S z where S = (Psi^T Phi)^{-1}"""
        z = torch.randn(R, dtype=torch.float64)
        q = proj.decode(z)
        assert q.shape == (N,)
        S = torch.linalg.inv(proj.Psi.T @ proj.Phi)
        expected = proj.Phi @ (S @ z)
        np.testing.assert_allclose(q.numpy(), expected.numpy(), rtol=1e-12)

    def test_batched_value(self, proj):
        z = torch.randn(M, R, dtype=torch.float64)
        q = proj.decode(z)
        assert q.shape == (M, N)
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
        np.testing.assert_allclose(q_reconstructed.numpy(), q.numpy(), rtol=1e-10)

    def test_decode_encode_is_projection_batched(self, proj):
        C = torch.randn(M, R, dtype=torch.float64)
        Q = (proj.Phi @ C.T).T  # (M, N), each row in col(Phi)
        Z = proj.encode(Q)
        Q_reconstructed = proj.decode(Z)
        np.testing.assert_allclose(Q_reconstructed.numpy(), Q.numpy(), rtol=1e-10)

    def test_encode_decode_is_identity_in_reduced_space(self, proj):
        """encode(decode(z)) = Psi^T Phi S z = (Psi^T Phi)(Psi^T Phi)^{-1} z = z"""
        z = torch.randn(R, dtype=torch.float64)
        z_roundtrip = proj.encode(proj.decode(z))
        np.testing.assert_allclose(z_roundtrip.numpy(), z.numpy(), rtol=1e-10)

    def test_encode_decode_is_identity_in_reduced_space_batched(self, proj):
        Z = torch.randn(M, R, dtype=torch.float64)
        Z_roundtrip = proj.encode(proj.decode(Z))
        np.testing.assert_allclose(Z_roundtrip.numpy(), Z.numpy(), rtol=1e-10)

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

        np.testing.assert_allclose(proj.S.numpy(), np.eye(R), atol=1e-12)

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
        np.testing.assert_allclose(proj.S.numpy(), expected_S.numpy(), rtol=1e-12)

    def test_encode_decode_consistent_after_update(self, proj):
        Phi_new, Psi_new = _random_bases(N, R, seed=777)
        proj.update([Phi_new, Psi_new])

        c = torch.randn(R, dtype=torch.float64)
        q = proj.Phi @ c
        q_roundtrip = proj.decode(proj.encode(q))
        np.testing.assert_allclose(q_roundtrip.numpy(), q.numpy(), rtol=1e-10)


@pytest.mark.parametrize("biorthogonalize", ["encoder", "decoder"])
@pytest.mark.parametrize("batched", [False, True])
def test_inverse_placement_maps(backend, biorthogonalize, batched):
    rng = np.random.default_rng(43)
    Phi, Psi = (backend.asarray(rng.standard_normal((N, R))) for _ in range(2))
    q = backend.asarray(rng.standard_normal((M, N) if batched else N))
    z = backend.asarray(rng.standard_normal((M, R) if batched else R))
    proj = LinearProjection(
        [Phi, Psi], biorthogonalize=biorthogonalize, backend=backend
    )

    # A solve supplies an independent reference for the inverse application.
    phi, psi, q_np, z_np = map(backend.to_numpy, (Phi, Psi, q, z))
    gram = psi.T @ phi
    encoded = q_np @ psi
    decoded = z_np
    if biorthogonalize == "encoder":
        encoded = np.linalg.solve(gram, encoded.T).T
    else:
        decoded = np.linalg.solve(gram, decoded.T).T
    np.testing.assert_allclose(backend.to_numpy(proj.encode(q)), encoded, atol=1e-12)
    np.testing.assert_allclose(
        backend.to_numpy(proj.decode(z)), decoded @ phi.T, atol=1e-12
    )
    np.testing.assert_allclose(
        backend.to_numpy(proj.encode(proj.decode(z))), z_np, atol=1e-12
    )
    expected_projection = np.linalg.solve(gram, (q_np @ psi).T).T @ phi.T
    np.testing.assert_allclose(
        backend.to_numpy(proj.decode(proj.encode(q))), expected_projection, atol=1e-12
    )

    # Updating the bases must retain the chosen placement.
    proj.update([2 * Phi, Psi])
    np.testing.assert_allclose(
        backend.to_numpy(proj.encode(q)),
        encoded / 2 if biorthogonalize == "encoder" else encoded,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        backend.to_numpy(proj.decode(z)),
        2 * decoded @ phi.T if biorthogonalize == "encoder" else decoded @ phi.T,
        atol=1e-12,
    )


@pytest.mark.parametrize("biorthogonalize", ["encoder", "decoder"])
@pytest.mark.parametrize("batched", [False, True])
def test_inverse_placement_vjps(biorthogonalize, batched):
    Phi, Psi = _random_bases(N, R, seed=44)
    Phi.requires_grad_(True)
    Psi.requires_grad_(True)
    torch.manual_seed(44)
    q = torch.randn((M, N) if batched else (N,), dtype=torch.float64)
    z = torch.randn(
        (M, R) if batched else (R,), dtype=torch.float64, requires_grad=True
    )
    v_enc = torch.randn_like(z)
    v_dec = torch.randn_like(q)
    S = torch.linalg.inv(Psi.T @ Phi)
    encoded = q @ Psi
    decoded = z
    if biorthogonalize == "encoder":
        encoded = encoded @ S.T
    else:
        decoded = decoded @ S.T
    decoded = decoded @ Phi.T

    expected_encode = torch.autograd.grad(
        (encoded * v_enc).sum(), (Phi, Psi), retain_graph=True, allow_unused=True
    )
    expected_decode = torch.autograd.grad(
        (decoded * v_dec).sum(), (Phi, Psi, z), allow_unused=True
    )
    bases = [Phi.detach(), Psi.detach()]
    # The decoder cases also verify that omitting the option keeps its default.
    proj = (
        LinearProjection(bases)
        if biorthogonalize == "decoder"
        else LinearProjection(bases, biorthogonalize="encoder")
    )
    assert proj.biorthogonalize == biorthogonalize
    for index, (actual, expected) in enumerate(
        zip(
            (
                *proj.vjp_encode(q, v_enc),
                *proj.vjp_decode(z.detach(), v_dec),
                proj.vjp_decode_state(z.detach(), v_dec),
            ),
            (*expected_encode, *expected_decode),
            strict=True,
        )
    ):
        if expected is None:
            expected = torch.zeros_like(Phi if index == 0 else Psi)
        # Preserve the zero absolute tolerance of the former decoder-placement
        # parameter VJP tests; encoder placement and state VJPs keep theirs.
        atol = 0.0 if biorthogonalize == "decoder" and index < 4 else 1e-10
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=atol)


def test_invalid_inverse_placement():
    Phi, Psi = _random_bases(N, R)
    with pytest.raises(ValueError, match="biorthogonalize"):
        LinearProjection([Phi, Psi], biorthogonalize="invalid")


pytestmark = pytest.mark.usefixtures("torch_backend")
