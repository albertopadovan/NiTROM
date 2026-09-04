"""Tests for ObliqueChartProjection (Psi = Phi + W N)."""

import numpy as np
import pytest
import torch

from nitrom.projections import LinearProjection, ObliqueChartProjection

N_AMB, R, K = 12, 3, 4
DTYPE = torch.float64


def _bases(seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((N_AMB, R + K)))
    Phi = torch.tensor(np.ascontiguousarray(Q[:, :R]), dtype=DTYPE)
    W = torch.tensor(np.ascontiguousarray(Q[:, R:]), dtype=DTYPE)
    Nc = torch.tensor(rng.standard_normal((K, R)), dtype=DTYPE)
    return Phi, W, Nc


class TestConstruction:

    def test_only_n_is_a_parameter(self):
        proj = ObliqueChartProjection(list(_bases()))
        assert proj.param_names == ["N"]
        assert len(proj.get_params()) == 1

    def test_biorthogonality_holds_identically(self):
        Phi, W, _ = _bases()
        rng = np.random.default_rng(1)
        for _ in range(5):
            Nc = torch.tensor(10.0 * rng.standard_normal((K, R)), dtype=DTYPE)
            proj = ObliqueChartProjection([Phi, W, Nc])
            assert torch.allclose(proj.Psi.T @ Phi, torch.eye(R, dtype=DTYPE), atol=1e-12)
            assert torch.allclose(proj.S, torch.eye(R, dtype=DTYPE), atol=1e-12)

    def test_zero_n_is_the_orthogonal_projection(self):
        Phi, W, _ = _bases()
        proj = ObliqueChartProjection([Phi, W, torch.zeros((K, R), dtype=DTYPE)])
        assert torch.allclose(proj.Psi, Phi)

    def test_rejects_non_orthogonal_chart(self):
        Phi, W, Nc = _bases()
        with pytest.raises(ValueError, match="orthogonal to Phi"):
            ObliqueChartProjection([Phi, Phi.clone(), torch.zeros((R, R), dtype=DTYPE)])

    def test_rejects_bad_shapes(self):
        Phi, W, _ = _bases()
        with pytest.raises(ValueError, match="N must have shape"):
            ObliqueChartProjection([Phi, W, torch.zeros((K + 1, R), dtype=DTYPE)])
        with pytest.raises(ValueError, match="rows"):
            ObliqueChartProjection([Phi, W[:-1], torch.zeros((K, R), dtype=DTYPE)])


class TestMaps:

    def test_encode_decode_agree_with_linear_projection(self):
        """With the same Psi, the chart and a plain LinearProjection agree."""
        Phi, W, Nc = _bases()
        proj = ObliqueChartProjection([Phi, W, Nc])
        ref = LinearProjection([Phi, proj.Psi.clone()])
        rng = np.random.default_rng(2)
        q = torch.tensor(rng.standard_normal((7, N_AMB)), dtype=DTYPE)
        z = torch.tensor(rng.standard_normal((7, R)), dtype=DTYPE)
        assert torch.allclose(proj.encode(q), ref.encode(q))
        assert torch.allclose(proj.decode(z), ref.decode(z), atol=1e-12)
        assert torch.allclose(proj.encode(q[0]), ref.encode(q[0]))
        assert torch.allclose(proj.decode(z[0]), ref.decode(z[0]), atol=1e-12)

    def test_decode_is_phi_z(self):
        Phi, W, Nc = _bases()
        proj = ObliqueChartProjection([Phi, W, Nc])
        z = torch.tensor(np.random.default_rng(3).standard_normal((5, R)), dtype=DTYPE)
        assert torch.allclose(proj.decode(z), z @ Phi.T)

    def test_update_rebuilds_psi(self):
        Phi, W, Nc = _bases()
        proj = ObliqueChartProjection([Phi, W, torch.zeros((K, R), dtype=DTYPE)])
        proj.update([Nc])
        assert torch.allclose(proj.Psi, Phi + W @ Nc)
        assert torch.allclose(proj.get_params()[0], Nc)


class TestVJPs:
    """Analytic VJPs against autograd."""

    def _proj_with_grad(self):
        Phi, W, Nc = _bases(4)
        Nc = Nc.clone().requires_grad_(True)
        return ObliqueChartProjection([Phi, W, Nc]), Nc

    def test_vjp_encode(self):
        proj, Nc = self._proj_with_grad()
        rng = np.random.default_rng(5)
        q = torch.tensor(rng.standard_normal((6, N_AMB)), dtype=DTYPE)
        v = torch.tensor(rng.standard_normal((6, R)), dtype=DTYPE)
        (proj.encode(q) * v).sum().backward()
        np.testing.assert_allclose(
            proj.vjp_encode(q, v)[0].detach().numpy(), Nc.grad.numpy(),
            rtol=1e-10, atol=1e-12,
        )

    def test_vjp_decode_is_zero(self):
        """The decoder is Phi z, which does not involve N."""
        proj, Nc = self._proj_with_grad()
        rng = np.random.default_rng(6)
        z = torch.tensor(rng.standard_normal((6, R)), dtype=DTYPE)
        v = torch.tensor(rng.standard_normal((6, N_AMB)), dtype=DTYPE)
        grad = proj.vjp_decode(z, v)[0]
        assert grad.shape == (K, R)
        assert torch.allclose(grad, torch.zeros_like(grad))

    def test_vjp_decode_state(self):
        proj, _ = self._proj_with_grad()
        rng = np.random.default_rng(7)
        z = torch.tensor(rng.standard_normal((6, R)), dtype=DTYPE, requires_grad=True)
        v = torch.tensor(rng.standard_normal((6, N_AMB)), dtype=DTYPE)
        (proj.decode(z) * v).sum().backward()
        np.testing.assert_allclose(
            proj.vjp_decode_state(z.detach(), v).detach().numpy(),
            z.grad.numpy(), rtol=1e-10, atol=1e-12,
        )

    def test_vjp_bases_is_the_chain_rule(self):
        proj, _ = self._proj_with_grad()
        rng = np.random.default_rng(8)
        gPhi = torch.tensor(rng.standard_normal((N_AMB, R)), dtype=DTYPE)
        gPsi = torch.tensor(rng.standard_normal((N_AMB, R)), dtype=DTYPE)
        out = proj.vjp_bases(gPhi, gPsi)
        assert len(out) == 1
        # dJ/dN = W^T dJ/dPsi; the Phi cotangent is discarded (Phi is fixed).
        assert torch.allclose(out[0], proj.W.T @ gPsi)
        assert torch.allclose(proj.vjp_bases(torch.zeros_like(gPhi), gPsi)[0], out[0])
