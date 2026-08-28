"""Tests for the AtrPolynomialModel class."""

import numpy as np
import pytest
import torch

from nitrom.latent_space_models.atr_polynomial_model import AtrPolynomialModel
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.model import Model

R = 4  # state dimension
M = 3  # batch size
DTYPE = torch.float64


def _make_params(r, seed=42, forcing_dim=None, m_scale=1.0, bhat_scale=1.0):
    """Random ATR parameters ``[K, R, Q, S, Bhat, m]`` (+ ``B``)."""
    rng = np.random.default_rng(seed)
    eye = np.eye(r)
    params = [
        torch.tensor(rng.standard_normal((r, r)), dtype=DTYPE),              # K
        torch.tensor(eye + 0.1 * rng.standard_normal((r, r)), dtype=DTYPE),  # R
        torch.tensor(eye + 0.1 * rng.standard_normal((r, r)), dtype=DTYPE),  # Q
        torch.tensor(0.3 * rng.standard_normal((r, r, r)), dtype=DTYPE),     # S
        torch.tensor(bhat_scale * rng.standard_normal(r), dtype=DTYPE),      # Bhat
        torch.tensor(m_scale * rng.standard_normal(r), dtype=DTYPE),         # m
    ]
    if forcing_dim is not None:
        params.append(torch.tensor(rng.standard_normal((r, forcing_dim)), dtype=DTYPE))
    return params


def _shifted_operators(params):
    r"""Rebuild :math:`(\tilde{Q}, \tilde{R}, \widehat{A}, H)` from the parameters."""
    K, Rm, Q, S = params[0], params[1], params[2], params[3]
    Qinv = torch.linalg.inv(Q)
    Qtil = Qinv @ Qinv.T
    Rinv = torch.linalg.inv(Rm)
    Rtil = Rinv @ Rinv.T
    Ahat = ((K - K.T) - Rtil) @ Qtil
    H = torch.einsum("ilk,lj->ijk", S, Qtil) - torch.einsum("lik,lj->ijk", S, Qtil)
    return Qtil, Rtil, Ahat, H


@pytest.fixture()
def atr_model_and_data():
    params = _make_params(R, seed=42)
    model = AtrPolynomialModel(R, [1, 2], atr_params=params)

    rng = np.random.default_rng(123)
    z = torch.tensor(rng.standard_normal(R), dtype=DTYPE)
    z_batch = torch.tensor(rng.standard_normal((M, R)), dtype=DTYPE)
    return model, params, z, z_batch


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

class TestBasic:

    def test_is_model(self, atr_model_and_data):
        model, _, _, _ = atr_model_and_data
        assert isinstance(model, Model)
        assert isinstance(model, GasPolynomialModel)

    def test_param_names(self, atr_model_and_data):
        model, _, _, _ = atr_model_and_data
        assert model.param_names == ["K", "R", "Q", "S", "Bhat", "m"]

    def test_param_names_with_forcing(self):
        params = _make_params(R, seed=1, forcing_dim=2)
        model = AtrPolynomialModel(
            R, [1, 2], atr_params=params,
            forcing_config={"forcing_exists": True, "m": 2},
        )
        assert model.param_names == ["K", "R", "Q", "S", "Bhat", "m", "B"]

    def test_inner_poly_comp_has_constant(self, atr_model_and_data):
        """The inner model carries a degree-0 term for the constant."""
        model, _, _, _ = atr_model_and_data
        assert model.poly_comp == [0, 1, 2]
        assert model.model.poly_comp == [0, 1, 2]
        c, A, H = model.inner_params()
        assert c.shape == (R,)
        assert A.shape == (R, R)
        assert H.shape == (R, R, R)

    def test_requires_linear_and_quadratic(self):
        with pytest.raises(ValueError, match="linear and a quadratic"):
            AtrPolynomialModel(R, [1])
        with pytest.raises(ValueError, match="linear and a quadratic"):
            AtrPolynomialModel(R, [2])

    def test_default_shift_is_zero(self):
        model = AtrPolynomialModel(R, [1, 2])
        np.testing.assert_allclose(model.m.numpy(), np.zeros(R), atol=0.0)
        np.testing.assert_allclose(model.Bhat.numpy(), np.zeros(R), atol=0.0)

    def test_m0_overrides(self):
        params = _make_params(R, seed=5)
        m0 = torch.arange(R, dtype=DTYPE)
        model = AtrPolynomialModel(R, [1, 2], atr_params=params, m0=m0)
        np.testing.assert_allclose(model.m.numpy(), m0.numpy())


# ---------------------------------------------------------------------------
# Assembly: the unshifted operators must reproduce the shifted dynamics
# ---------------------------------------------------------------------------

class TestAssembly:

    def test_rhs_matches_shifted_form(self, atr_model_and_data):
        r""":math:`f(z) = \widehat{A}(z-m) + H:(z-m)(z-m)^\top + \widehat{B}`."""
        model, params, z, _ = atr_model_and_data
        _, _, Ahat, H = _shifted_operators(params)
        Bhat, m = params[4], params[5]

        zh = z - m
        expected = Ahat @ zh + torch.einsum("ijk,j,k->i", H, zh, zh) + Bhat
        np.testing.assert_allclose(
            model.evaluate_rhs(0.0, z).numpy(), expected.numpy(), rtol=1e-12,
            atol=1e-12,
        )

    def test_rhs_matches_shifted_form_batched(self, atr_model_and_data):
        model, params, _, z_batch = atr_model_and_data
        _, _, Ahat, H = _shifted_operators(params)
        Bhat, m = params[4], params[5]

        Zh = z_batch - m
        expected = Zh @ Ahat.T + torch.einsum("ijk,bj,bk->bi", H, Zh, Zh) + Bhat
        np.testing.assert_allclose(
            model.evaluate_rhs(0.0, z_batch).numpy(), expected.numpy(),
            rtol=1e-12, atol=1e-12,
        )

    def test_unshifted_operators(self, atr_model_and_data):
        """``A = Ahat - H:(I x m) - H:(m x I)`` and ``c = Bhat - Ahat m + H:mm``."""
        model, params, _, _ = atr_model_and_data
        _, _, Ahat, H = _shifted_operators(params)
        Bhat, m = params[4], params[5]

        c, A, H_inner = model.inner_params()
        A_expected = (
            Ahat
            - torch.einsum("ijk,k->ij", H, m)
            - torch.einsum("ikj,k->ij", H, m)
        )
        c_expected = Bhat - Ahat @ m + torch.einsum("ijk,j,k->i", H, m, m)

        np.testing.assert_allclose(A.numpy(), A_expected.numpy(), rtol=1e-12)
        np.testing.assert_allclose(c.numpy(), c_expected.numpy(), rtol=1e-12)
        np.testing.assert_allclose(H_inner.numpy(), H.numpy(), rtol=1e-12)

    def test_reduces_to_gas_when_unshifted(self):
        """With ``m = 0`` and ``Bhat = 0`` the model *is* the GAS model."""
        params = _make_params(R, seed=11)
        params[4] = torch.zeros(R, dtype=DTYPE)  # Bhat
        params[5] = torch.zeros(R, dtype=DTYPE)  # m
        atr = AtrPolynomialModel(R, [1, 2], atr_params=params)
        gas = GasPolynomialModel(R, [1, 2], gas_params=params[:4])

        c, A, H = atr.inner_params()
        A_gas, H_gas = gas.inner_params()
        np.testing.assert_allclose(c.numpy(), np.zeros(R), atol=1e-14)
        np.testing.assert_allclose(A.numpy(), A_gas.numpy(), rtol=1e-12)
        np.testing.assert_allclose(H.numpy(), H_gas.numpy(), rtol=1e-12)

        rng = np.random.default_rng(3)
        z = torch.tensor(rng.standard_normal(R), dtype=DTYPE)
        np.testing.assert_allclose(
            atr.evaluate_rhs(0.0, z).numpy(), gas.evaluate_rhs(0.0, z).numpy(),
            rtol=1e-12,
        )

    def test_update_params_reassembles(self, atr_model_and_data):
        model, _, z, _ = atr_model_and_data
        rhs_before = model.evaluate_rhs(0.0, z).clone()
        new_params = [p + 0.1 * torch.randn_like(p) for p in model.get_params()]
        model.update_params(new_params)
        assert not torch.allclose(rhs_before, model.evaluate_rhs(0.0, z))


# ---------------------------------------------------------------------------
# Trapping-region / Lyapunov properties
# ---------------------------------------------------------------------------

class TestTrappingRegion:

    def test_lyapunov_derivative_identity(self, atr_model_and_data):
        r""":math:`\dot{\widehat{V}} = -\hat{y}^\top \tilde{R}\hat{y}
        + \hat{y}^\top \widehat{B}` with :math:`\hat{y} = \tilde{Q}\hat{z}`."""
        model, params, _, _ = atr_model_and_data
        Qtil, Rtil, _, _ = _shifted_operators(params)
        Bhat, m = params[4], params[5]

        rng = np.random.default_rng(99)
        for _ in range(5):
            z = torch.tensor(2.0 * rng.standard_normal(R), dtype=DTYPE)
            y = Qtil @ (z - m)
            Vdot = y @ model.evaluate_rhs(0.0, z)
            expected = -y @ Rtil @ y + y @ Bhat
            np.testing.assert_allclose(
                Vdot.item(), expected.item(), rtol=1e-10, atol=1e-12,
            )

    def test_energy_conserving_quadratic(self, atr_model_and_data):
        r""":math:`\hat{z}^\top \tilde{Q}\,H(\hat{z},\hat{z}) = 0`."""
        model, params, _, _ = atr_model_and_data
        Qtil, _, _, _ = _shifted_operators(params)
        _, _, H = model.inner_params()

        rng = np.random.default_rng(555)
        for _ in range(5):
            zh = torch.tensor(rng.standard_normal(R), dtype=DTYPE)
            Hz = torch.einsum("ijk,j,k->i", H, zh, zh)
            np.testing.assert_allclose((zh @ Qtil @ Hz).item(), 0.0, atol=1e-10)

    def test_dissipative_outside_the_ball(self, atr_model_and_data):
        r""":math:`\dot{\widehat{V}} < 0` beyond :meth:`trapping_region_radius`."""
        model, params, _, _ = atr_model_and_data
        Qtil, _, _, _ = _shifted_operators(params)
        m = params[5]
        rho = model.trapping_region_radius()
        assert np.isfinite(rho) and rho > 0.0

        rng = np.random.default_rng(2024)
        for _ in range(10):
            d = rng.standard_normal(R)
            d = 1.01 * rho * d / np.linalg.norm(d)
            z = m + torch.tensor(d, dtype=DTYPE)
            y = Qtil @ (z - m)
            assert (y @ model.evaluate_rhs(0.0, z)).item() < 0.0

    def test_shifted_linear_operator_is_hurwitz(self, atr_model_and_data):
        model, _, _, _ = atr_model_and_data
        eigs = torch.linalg.eigvals(model._Ahat).real
        assert torch.all(eigs < 0), f"Ahat has non-negative eigenvalues: {eigs}"


# ---------------------------------------------------------------------------
# Shift helpers
# ---------------------------------------------------------------------------

class TestShift:

    def test_set_shift_reassembles(self, atr_model_and_data):
        model, params, z, _ = atr_model_and_data
        _, _, Ahat, H = _shifted_operators(params)
        Bhat = params[4]

        m_new = torch.tensor([0.5, -1.0, 2.0, 0.25], dtype=DTYPE)
        model.set_shift(m_new)
        np.testing.assert_allclose(model.m.numpy(), m_new.numpy())

        zh = z - m_new
        expected = Ahat @ zh + torch.einsum("ijk,j,k->i", H, zh, zh) + Bhat
        np.testing.assert_allclose(
            model.evaluate_rhs(0.0, z).numpy(), expected.numpy(), rtol=1e-12,
        )

    def test_set_shift_from_data_3d(self, atr_model_and_data):
        model, _, _, _ = atr_model_and_data
        rng = np.random.default_rng(17)
        Z = torch.tensor(rng.standard_normal((3, R, 7)), dtype=DTYPE)
        model.set_shift_from_data(Z)
        np.testing.assert_allclose(
            model.m.numpy(), Z.numpy().mean(axis=(0, 2)), rtol=1e-12,
        )

    def test_set_shift_from_data_2d(self, atr_model_and_data):
        model, _, _, _ = atr_model_and_data
        rng = np.random.default_rng(18)
        Z = torch.tensor(rng.standard_normal((11, R)), dtype=DTYPE)
        model.set_shift_from_data(Z)
        np.testing.assert_allclose(
            model.m.numpy(), Z.numpy().mean(axis=0), rtol=1e-12,
        )

    def test_set_shift_from_data_rejects_4d(self, atr_model_and_data):
        model, _, _, _ = atr_model_and_data
        with pytest.raises(ValueError, match="1, 2, or 3 dimensions"):
            model.set_shift_from_data(torch.zeros((2, 2, 2, 2), dtype=DTYPE))


# ---------------------------------------------------------------------------
# vjp_evaluate_rhs: finite differences and autograd
# ---------------------------------------------------------------------------

def _fd_check(model, z, v, seed):
    """Central FD check of ``vjp_evaluate_rhs`` against every parameter."""
    rng = np.random.default_rng(seed)
    grads = model.vjp_evaluate_rhs(z, v)
    params = [p.clone() for p in model.get_params()]

    eps = 1e-7
    for idx, name in enumerate(model.param_names):
        dp = torch.tensor(rng.standard_normal(params[idx].shape), dtype=DTYPE)
        dd_vjp = torch.sum(grads[idx] * dp).item()

        plus = [p.clone() for p in params]
        minus = [p.clone() for p in params]
        plus[idx] = params[idx] + eps * dp
        minus[idx] = params[idx] - eps * dp

        model.update_params(plus)
        J_plus = torch.sum(v * model.evaluate_rhs(0.0, z)).item()
        model.update_params(minus)
        J_minus = torch.sum(v * model.evaluate_rhs(0.0, z)).item()
        model.update_params(params)

        dd_fd = (J_plus - J_minus) / (2 * eps)
        np.testing.assert_allclose(
            dd_vjp, dd_fd, rtol=1e-4,
            err_msg=f"Mismatch for param '{name}' (index {idx})",
        )


class TestVjpEvaluateRhs:

    def test_finite_difference_unbatched(self, atr_model_and_data):
        model, _, z, _ = atr_model_and_data
        rng = np.random.default_rng(700)
        v = torch.tensor(rng.standard_normal(R), dtype=DTYPE)
        _fd_check(model, z, v, seed=701)

    def test_finite_difference_batched(self, atr_model_and_data):
        model, _, _, z_batch = atr_model_and_data
        rng = np.random.default_rng(702)
        v = torch.tensor(rng.standard_normal((M, R)), dtype=DTYPE)
        _fd_check(model, z_batch, v, seed=703)

    def test_finite_difference_with_forcing(self):
        params = _make_params(R, seed=21, forcing_dim=2)
        model = AtrPolynomialModel(
            R, [1, 2], atr_params=params,
            forcing_config={"forcing_exists": True, "m": 2},
        )
        rng = np.random.default_rng(704)
        z = torch.tensor(rng.standard_normal((M, R)), dtype=DTYPE)
        v = torch.tensor(rng.standard_normal((M, R)), dtype=DTYPE)

        def f(t):
            return torch.tensor([1.0, -0.5], dtype=DTYPE)

        forcing = [f] * M
        grads = model.vjp_evaluate_rhs(z, v, external_forcing=forcing, t=0.0)
        params_now = [p.clone() for p in model.get_params()]
        eps = 1e-7
        for idx, name in enumerate(model.param_names):
            dp = torch.tensor(
                rng.standard_normal(params_now[idx].shape), dtype=DTYPE
            )
            dd_vjp = torch.sum(grads[idx] * dp).item()

            plus = [p.clone() for p in params_now]
            minus = [p.clone() for p in params_now]
            plus[idx] = params_now[idx] + eps * dp
            minus[idx] = params_now[idx] - eps * dp

            model.update_params(plus)
            J_plus = torch.sum(
                v * model.evaluate_rhs(0.0, z, external_forcing=forcing)
            ).item()
            model.update_params(minus)
            J_minus = torch.sum(
                v * model.evaluate_rhs(0.0, z, external_forcing=forcing)
            ).item()
            model.update_params(params_now)

            np.testing.assert_allclose(
                dd_vjp, (J_plus - J_minus) / (2 * eps), rtol=1e-4,
                err_msg=f"Mismatch for param '{name}' (index {idx})",
            )

    def test_matches_autograd(self, atr_model_and_data):
        """Rebuild the forward pass with autograd and compare gradients."""
        model, params, z, _ = atr_model_and_data
        rng = np.random.default_rng(705)
        v = torch.tensor(rng.standard_normal(R), dtype=DTYPE)

        names = ["K", "R", "Q", "S", "Bhat", "m"]
        leaves = {
            n: p.clone().requires_grad_(True)
            for n, p in zip(names, params, strict=True)
        }
        Qinv = torch.linalg.inv(leaves["Q"])
        Qtil = Qinv @ Qinv.T
        Rinv = torch.linalg.inv(leaves["R"])
        Ahat = ((leaves["K"] - leaves["K"].T) - Rinv @ Rinv.T) @ Qtil
        H = torch.einsum("ilk,lj->ijk", leaves["S"], Qtil) - torch.einsum(
            "lik,lj->ijk", leaves["S"], Qtil
        )
        zh = z - leaves["m"]
        dzdt = Ahat @ zh + torch.einsum("ijk,j,k->i", H, zh, zh) + leaves["Bhat"]
        (v * dzdt).sum().backward()

        for grad, name in zip(model.vjp_evaluate_rhs(z, v), names, strict=True):
            np.testing.assert_allclose(
                grad.numpy(), leaves[name].grad.numpy(), rtol=1e-10, atol=1e-12,
                err_msg=f"Mismatch for param '{name}'",
            )


# ---------------------------------------------------------------------------
# Retraction
# ---------------------------------------------------------------------------

class TestRetraction:

    def test_shift_and_constant_are_set(self):
        rng = np.random.default_rng(7)
        A = torch.tensor(rng.standard_normal((R, R)), dtype=DTYPE)
        H = torch.tensor(0.3 * rng.standard_normal((R, R, R)), dtype=DTYPE)
        c = torch.tensor(rng.standard_normal(R), dtype=DTYPE)
        m = torch.tensor(0.5 * rng.standard_normal(R), dtype=DTYPE)

        model = AtrPolynomialModel(R, [1, 2])
        model.retract_general_tensors_to_atr_tensors([c, A, H], m=m)

        Bhat_expected = c + A @ m + torch.einsum("ijk,j,k->i", H, m, m)
        np.testing.assert_allclose(model.m.numpy(), m.numpy(), rtol=1e-12)
        np.testing.assert_allclose(
            model.Bhat.numpy(), Bhat_expected.numpy(), rtol=1e-12,
        )

    def test_result_is_stable_and_energy_conserving(self):
        """Even from an unstable input, the retracted model satisfies ATR."""
        rng = np.random.default_rng(8)
        A = torch.tensor(rng.standard_normal((R, R)), dtype=DTYPE)
        H = torch.tensor(0.3 * rng.standard_normal((R, R, R)), dtype=DTYPE)
        c = torch.tensor(rng.standard_normal(R), dtype=DTYPE)
        m = torch.tensor(0.5 * rng.standard_normal(R), dtype=DTYPE)
        assert torch.linalg.eigvals(A).real.max() > 0  # input is unstable

        model = AtrPolynomialModel(R, [1, 2])
        model.retract_general_tensors_to_atr_tensors([c, A, H], m=m)

        assert torch.all(torch.linalg.eigvals(model._Ahat).real < 0)
        Qtil = model._Qtil
        _, _, H_ret = model.inner_params()
        for _ in range(5):
            zh = torch.tensor(rng.standard_normal(R), dtype=DTYPE)
            Hz = torch.einsum("ijk,j,k->i", H_ret, zh, zh)
            # Normalized: the retracted metric can be poorly conditioned, so an
            # absolute tolerance on the raw energy would track its scale.
            scale = torch.linalg.norm(Qtil @ zh) * torch.linalg.norm(Hz)
            np.testing.assert_allclose(
                ((zh @ Qtil @ Hz) / scale).item(), 0.0, atol=1e-10,
            )
        assert np.isfinite(model.trapping_region_radius())

    def test_preserves_shifted_operators_of_an_atr_model(self):
        r"""Retracting an ATR model's own operators reproduces
        :math:`(\widehat{A}, \widehat{B}, m)` exactly."""
        src = AtrPolynomialModel(R, [1, 2], atr_params=_make_params(R, seed=31))
        dst = AtrPolynomialModel(R, [1, 2])
        dst.retract_general_tensors_to_atr_tensors(src.inner_params(), m=src.m)

        np.testing.assert_allclose(dst.m.numpy(), src.m.numpy(), rtol=1e-12)
        np.testing.assert_allclose(
            dst.Bhat.numpy(), src.Bhat.numpy(), rtol=1e-10,
        )
        np.testing.assert_allclose(
            dst._Ahat.numpy(), src._Ahat.numpy(), rtol=1e-8, atol=1e-10,
        )

    def test_uses_current_shift_by_default(self):
        rng = np.random.default_rng(9)
        A = torch.tensor(rng.standard_normal((R, R)), dtype=DTYPE)
        H = torch.tensor(0.3 * rng.standard_normal((R, R, R)), dtype=DTYPE)
        Z = torch.tensor(rng.standard_normal((2, R, 5)), dtype=DTYPE)

        model = AtrPolynomialModel(R, [1, 2])
        m = model.set_shift_from_data(Z)
        model.retract_general_tensors_to_atr_tensors([A, H])
        np.testing.assert_allclose(model.m.numpy(), m.numpy(), rtol=1e-12)

    def test_rejects_bad_tensor_list(self):
        model = AtrPolynomialModel(R, [1, 2])
        with pytest.raises(ValueError, match=r"\[A, H\] or \[c, A, H\]"):
            model.retract_general_tensors_to_atr_tensors(
                [torch.zeros(R, R, dtype=DTYPE)]
            )
