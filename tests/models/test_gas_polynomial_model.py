"""Tests for the GasPolynomialModel class."""

import numpy as np
import pytest
import torch

from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.model import Model

R = 4  # state dimension
M = 3  # batch size


def _make_gas_model(r, poly_comp, seed=42, forcing_config=None):
    """Create a GasPolynomialModel with random GAS parameters."""
    rng = np.random.default_rng(seed)
    gas_params = []
    if 1 in poly_comp:
        gas_params.append(torch.tensor(rng.standard_normal((r, r)), dtype=torch.float64))
        gas_params.append(torch.tensor(rng.standard_normal((r, r)), dtype=torch.float64))
    if 2 in poly_comp:
        gas_params.append(torch.tensor(rng.standard_normal((r, r)), dtype=torch.float64))
        gas_params.append(torch.tensor(rng.standard_normal((r, r, r)), dtype=torch.float64))
    if forcing_config is not None and forcing_config.get("forcing_exists", False):
        m = forcing_config["m"]
        gas_params.append(torch.tensor(rng.standard_normal((r, m)), dtype=torch.float64))
    return GasPolynomialModel(
        r, poly_comp, gas_params=gas_params, forcing_config=forcing_config,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[
    [1, 2],
], ids=["deg1+2"])
def gas_model_and_data(request):
    poly_comp = request.param
    rng = np.random.default_rng(123)
    model = _make_gas_model(R, poly_comp, seed=42)

    z_np = rng.standard_normal(R)
    z_torch = torch.tensor(z_np, dtype=torch.float64)
    z_batch_np = rng.standard_normal((M, R))
    z_batch_torch = torch.tensor(z_batch_np, dtype=torch.float64)

    return model, poly_comp, z_np, z_torch, z_batch_np, z_batch_torch


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

class TestBasic:

    def test_is_model(self, gas_model_and_data):
        model, _, _, _, _, _ = gas_model_and_data
        assert isinstance(model, Model)

    def test_param_names(self, gas_model_and_data):
        model, poly_comp, _, _, _, _ = gas_model_and_data
        expected = []
        if 1 in poly_comp:
            expected.extend(["K", "R"])
        if 2 in poly_comp:
            expected.extend(["Q", "S"])
        assert model.param_names == expected

    def test_get_params_shapes(self, gas_model_and_data):
        model, _, _, _, _, _ = gas_model_and_data
        params = model.get_params()
        assert len(params) == len(model.param_names)
        for name, param in zip(model.param_names, params):
            assert param is getattr(model, name)


# ---------------------------------------------------------------------------
# assemble_gas_tensors tests
# ---------------------------------------------------------------------------

class TestAssembleGasTensors:

    def test_assembly_matches_autograd(self, gas_model_and_data):
        """Verify assembled tensors match a direct autograd computation."""
        model, poly_comp, _, z_torch, _, _ = gas_model_and_data
        tensors = model.assemble_gas_tensors()

        # Recompute from scratch
        Qinv = torch.linalg.inv(model.Q)
        Qtil = Qinv @ Qinv.T
        if 1 in poly_comp:
            Rinv = torch.linalg.inv(model.R)
            A_expected = ((model.K - model.K.T) - Rinv @ Rinv.T) @ Qtil
            np.testing.assert_allclose(
                tensors[poly_comp.index(1)].numpy(), A_expected.numpy(), rtol=1e-12,
            )
        if 2 in poly_comp:
            H_expected = (
                torch.einsum("ilk,lj->ijk", model.S, Qtil)
                - torch.einsum("lik,lj->ijk", model.S, Qtil)
            )
            np.testing.assert_allclose(
                tensors[poly_comp.index(2)].numpy(), H_expected.numpy(), rtol=1e-12,
            )

    def test_A_eigenvalues_negative(self, gas_model_and_data):
        """All eigenvalues of A should be strictly negative."""
        model, poly_comp, _, _, _, _ = gas_model_and_data
        if 1 not in poly_comp:
            pytest.skip("No linear term")
        tensors = model.assemble_gas_tensors()
        A = tensors[poly_comp.index(1)]
        eigs = torch.linalg.eigvals(A).real
        assert torch.all(eigs < 0), f"A has non-negative eigenvalues: {eigs}"

    def test_H_energy_conservation(self, gas_model_and_data):
        r"""
        :math:`z^\top \tilde{Q}\, H(z, z) = 0` for all :math:`z`.
        """
        model, poly_comp, _, _, _, _ = gas_model_and_data
        if 2 not in poly_comp:
            pytest.skip("No quadratic term")
        tensors = model.assemble_gas_tensors()
        H = tensors[poly_comp.index(2)]
        Qinv = torch.linalg.inv(model.Q)
        Qtil = Qinv @ Qinv.T

        rng = np.random.default_rng(555)
        for _ in range(5):
            z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
            Hz = torch.einsum("ijk,j,k->i", H, z, z)
            energy = z @ Qtil @ Hz
            np.testing.assert_allclose(energy.item(), 0.0, atol=1e-10)

    def test_inner_model_uses_assembled_tensors(self, gas_model_and_data):
        model, poly_comp, _, _, _, _ = gas_model_and_data
        tensors = model.assemble_gas_tensors()
        inner_params = model.model.get_params()
        for i in range(len(poly_comp)):
            np.testing.assert_allclose(
                inner_params[i].numpy(), tensors[i].numpy(), rtol=1e-12,
            )


# ---------------------------------------------------------------------------
# evaluate_rhs / evaluate_adjoint_rhs delegation
# ---------------------------------------------------------------------------

class TestDelegation:

    def test_evaluate_rhs_unbatched(self, gas_model_and_data):
        model, _, _, z_torch, _, _ = gas_model_and_data
        result = model.evaluate_rhs(0.0, z_torch)
        expected = model.model.evaluate_rhs(0.0, z_torch)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-12)

    def test_evaluate_rhs_batched(self, gas_model_and_data):
        model, _, _, _, _, z_batch_torch = gas_model_and_data
        result = model.evaluate_rhs(0.0, z_batch_torch)
        expected = model.model.evaluate_rhs(0.0, z_batch_torch)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-12)

    def test_evaluate_adjoint_rhs_unbatched(self, gas_model_and_data):
        model, _, _, z_torch, _, _ = gas_model_and_data
        rng = np.random.default_rng(200)
        Z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        result = model.evaluate_adjoint_rhs(0.0, z_torch, Z)
        expected = model.model.evaluate_adjoint_rhs(0.0, z_torch, Z)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-12)


# ---------------------------------------------------------------------------
# update_params tests
# ---------------------------------------------------------------------------

class TestUpdateParams:

    def test_update_changes_inner_model(self, gas_model_and_data):
        model, poly_comp, _, z_torch, _, _ = gas_model_and_data
        rhs_before = model.evaluate_rhs(0.0, z_torch).clone()

        # Perturb all params
        new_params = [p + 0.1 * torch.randn_like(p) for p in model.get_params()]
        model.update_params(new_params)
        rhs_after = model.evaluate_rhs(0.0, z_torch)

        assert not torch.allclose(rhs_before, rhs_after)


# ---------------------------------------------------------------------------
# vjp_evaluate_rhs finite difference tests
# ---------------------------------------------------------------------------

class TestVjpEvaluateRhs:

    def test_finite_difference_unbatched(self, gas_model_and_data):
        """FD check: ⟨grad_p, δp⟩ ≈ [v^T f(p+εδp) - v^T f(p-εδp)] / 2ε"""
        model, poly_comp, _, z_torch, _, _ = gas_model_and_data

        rng = np.random.default_rng(700)
        v = torch.tensor(rng.standard_normal(R), dtype=torch.float64)

        grads = model.vjp_evaluate_rhs(z_torch, v)
        params = model.get_params()

        eps = 1e-7
        for idx, name in enumerate(model.param_names):
            dp = torch.tensor(
                rng.standard_normal(params[idx].shape), dtype=torch.float64
            )
            dd_vjp = torch.sum(grads[idx] * dp).item()

            params_plus = [p.clone() for p in params]
            params_minus = [p.clone() for p in params]
            params_plus[idx] = params[idx] + eps * dp
            params_minus[idx] = params[idx] - eps * dp

            model.update_params(params_plus)
            J_plus = torch.dot(v, model.evaluate_rhs(0.0, z_torch)).item()
            model.update_params(params_minus)
            J_minus = torch.dot(v, model.evaluate_rhs(0.0, z_torch)).item()
            model.update_params(params)

            dd_fd = (J_plus - J_minus) / (2 * eps)

            np.testing.assert_allclose(
                dd_vjp, dd_fd, rtol=1e-4,
                err_msg=f"Mismatch for param '{name}' (index {idx})",
            )

    def test_finite_difference_batched(self, gas_model_and_data):
        """Batched FD check."""
        model, poly_comp, _, _, _, z_batch_torch = gas_model_and_data

        rng = np.random.default_rng(701)
        v_batch = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)

        grads = model.vjp_evaluate_rhs(z_batch_torch, v_batch)
        params = model.get_params()

        eps = 1e-7
        for idx, name in enumerate(model.param_names):
            dp = torch.tensor(
                rng.standard_normal(params[idx].shape), dtype=torch.float64
            )
            dd_vjp = torch.sum(grads[idx] * dp).item()

            params_plus = [p.clone() for p in params]
            params_minus = [p.clone() for p in params]
            params_plus[idx] = params[idx] + eps * dp
            params_minus[idx] = params[idx] - eps * dp

            model.update_params(params_plus)
            rhs_plus = model.evaluate_rhs(0.0, z_batch_torch)
            model.update_params(params_minus)
            rhs_minus = model.evaluate_rhs(0.0, z_batch_torch)
            model.update_params(params)

            J_plus = torch.sum(v_batch * rhs_plus).item()
            J_minus = torch.sum(v_batch * rhs_minus).item()
            dd_fd = (J_plus - J_minus) / (2 * eps)

            np.testing.assert_allclose(
                dd_vjp, dd_fd, rtol=1e-4,
                err_msg=f"Mismatch for param '{name}' (index {idx})",
            )


# ---------------------------------------------------------------------------
# vjp_evaluate_rhs autograd verification
# ---------------------------------------------------------------------------

def _autograd_grads(K, R, Q, S, z, v, poly_comp, B=None, f_fun=None):
    """
    Compute gradients via torch.autograd by rebuilding the forward pass
    from scratch with requires_grad=True on each GAS parameter.
    """
    params = {}
    if 1 in poly_comp:
        params["K"] = K.clone().requires_grad_(True)
        params["R"] = R.clone().requires_grad_(True)
    if 2 in poly_comp:
        params["Q"] = Q.clone().requires_grad_(True)
        params["S"] = S.clone().requires_grad_(True)
    if B is not None:
        params["B"] = B.clone().requires_grad_(True)

    Qinv = torch.linalg.inv(params["Q"])
    Qtil = Qinv @ Qinv.T

    dzdt = torch.zeros_like(z)
    if 1 in poly_comp:
        Rinv = torch.linalg.inv(params["R"])
        A = ((params["K"] - params["K"].T) - Rinv @ Rinv.T) @ Qtil
        if z.ndim == 1:
            dzdt = dzdt + A @ z
        else:
            dzdt = dzdt + torch.einsum("ij,...j->...i", A, z)
    if 2 in poly_comp:
        H = (
            torch.einsum("ilk,lj->ijk", params["S"], Qtil)
            - torch.einsum("lik,lj->ijk", params["S"], Qtil)
        )
        if z.ndim == 1:
            dzdt = dzdt + torch.einsum("ijk,j,k->i", H, z, z)
        else:
            dzdt = dzdt + torch.einsum("ijk,...j,...k->...i", H, z, z)
    if B is not None and f_fun is not None:
        if z.ndim == 1:
            dzdt = dzdt + params["B"] @ f_fun(0.0)
        else:
            for i in range(z.shape[0]):
                dzdt[i] = dzdt[i] + params["B"] @ f_fun(0.0)

    loss = (v * dzdt).sum()
    loss.backward()

    ordered_names = []
    if 1 in poly_comp:
        ordered_names.extend(["K", "R"])
    if 2 in poly_comp:
        ordered_names.extend(["Q", "S"])
    if B is not None:
        ordered_names.append("B")

    return [params[n].grad for n in ordered_names]


class TestVjpAgainstAutograd:

    def test_unbatched(self, gas_model_and_data):
        model, poly_comp, _, z_torch, _, _ = gas_model_and_data
        rng = np.random.default_rng(900)
        v = torch.tensor(rng.standard_normal(R), dtype=torch.float64)

        analytic = model.vjp_evaluate_rhs(z_torch, v)
        autograd = _autograd_grads(
            model.K, model.R, model.Q, model.S,
            z_torch, v, poly_comp,
        )

        for name, ag, an in zip(model.param_names, autograd, analytic):
            np.testing.assert_allclose(
                an.numpy(), ag.numpy(), rtol=1e-10,
                err_msg=f"Autograd mismatch for '{name}'",
            )

    def test_batched(self, gas_model_and_data):
        model, poly_comp, _, _, _, z_batch_torch = gas_model_and_data
        rng = np.random.default_rng(901)
        v_batch = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)

        analytic = model.vjp_evaluate_rhs(z_batch_torch, v_batch)
        autograd = _autograd_grads(
            model.K, model.R, model.Q, model.S,
            z_batch_torch, v_batch, poly_comp,
        )

        for name, ag, an in zip(model.param_names, autograd, analytic):
            np.testing.assert_allclose(
                an.numpy(), ag.numpy(), rtol=1e-10,
                err_msg=f"Autograd mismatch for '{name}'",
            )

    def test_with_forcing(self):
        fc = {"forcing_exists": True, "m": 2}
        model = _make_gas_model(R, [1, 2], seed=42, forcing_config=fc)
        rng = np.random.default_rng(902)
        z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        u = torch.tensor(rng.standard_normal(2), dtype=torch.float64)
        f_fun = lambda t: u

        analytic = model.vjp_evaluate_rhs(z, v, external_forcing=[f_fun], t=0.0)
        autograd = _autograd_grads(
            model.K, model.R, model.Q, model.S,
            z, v, model.poly_comp, B=model.B, f_fun=f_fun,
        )

        for name, ag, an in zip(model.param_names, autograd, analytic):
            np.testing.assert_allclose(
                an.numpy(), ag.numpy(), rtol=1e-10,
                err_msg=f"Autograd mismatch for '{name}'",
            )


# ---------------------------------------------------------------------------
# Forcing tests
# ---------------------------------------------------------------------------

class TestForcing:

    def test_param_names_include_B(self):
        fc = {"forcing_exists": True, "m": 2}
        model = _make_gas_model(R, [1, 2], forcing_config=fc)
        assert "B" in model.param_names
        assert model.param_names[-1] == "B"

    def test_evaluate_rhs_with_forcing(self):
        fc = {"forcing_exists": True, "m": 2}
        model = _make_gas_model(R, [1, 2], seed=42, forcing_config=fc)
        rng = np.random.default_rng(99)
        z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        u = torch.tensor(rng.standard_normal(2), dtype=torch.float64)
        f_fun = lambda t: u

        rhs_with = model.evaluate_rhs(0.0, z, external_forcing=[f_fun])
        rhs_without = model.evaluate_rhs(0.0, z)
        diff = rhs_with - rhs_without
        expected_diff = model.B @ u
        np.testing.assert_allclose(diff.numpy(), expected_diff.numpy(), rtol=1e-12)

    def test_vjp_B_finite_difference(self):
        fc = {"forcing_exists": True, "m": 2}
        model = _make_gas_model(R, [1, 2], seed=42, forcing_config=fc)
        rng = np.random.default_rng(800)
        z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        u = torch.tensor(rng.standard_normal(2), dtype=torch.float64)
        f_fun = lambda t: u
        v = torch.tensor(rng.standard_normal(R), dtype=torch.float64)

        grads = model.vjp_evaluate_rhs(z, v, external_forcing=[f_fun], t=0.0)
        grad_B = grads[-1]

        dB = torch.tensor(rng.standard_normal(model.B.shape), dtype=torch.float64)
        dd_vjp = torch.sum(grad_B * dB).item()

        eps = 1e-7
        params = model.get_params()
        params_plus = [p.clone() for p in params]
        params_minus = [p.clone() for p in params]
        params_plus[-1] = params[-1] + eps * dB
        params_minus[-1] = params[-1] - eps * dB

        model.update_params(params_plus)
        J_plus = torch.dot(v, model.evaluate_rhs(0.0, z, external_forcing=[f_fun])).item()
        model.update_params(params_minus)
        J_minus = torch.dot(v, model.evaluate_rhs(0.0, z, external_forcing=[f_fun])).item()
        model.update_params(params)

        dd_fd = (J_plus - J_minus) / (2 * eps)
        np.testing.assert_allclose(dd_vjp, dd_fd, rtol=1e-5)


# ---------------------------------------------------------------------------
# Energy conservation tests
# ---------------------------------------------------------------------------

class TestEnergyConservation:

    def test_evaluate_rhs_tensors_energy_conservation(self, gas_model_and_data):
        """
        Test that the condition z^T Q^{-1} Q^{-T} H:zz^T + (H:zz^T)^T Q^{-1}Q^{-T} z = 0
        holds for the inner tensors actually used during evaluate_rhs.
        """
        model, poly_comp, _, _, _, _ = gas_model_and_data
        if 2 not in poly_comp:
            pytest.skip("No quadratic term")
        
        # Get tensors from the inner model which is used in evaluate_rhs
        inner_tensors = model.model.get_params()
        H_inner = inner_tensors[poly_comp.index(2)]
        
        Qinv = torch.linalg.inv(model.Q)
        Qtil = Qinv @ Qinv.T
        
        rng = np.random.default_rng(999)
        for _ in range(5):
            z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
            H_zz = torch.einsum("ijk,j,k->i", H_inner, z, z)
            
            term1 = z @ Qtil @ H_zz
            term2 = H_zz @ Qtil @ z
            val = term1 + term2
            np.testing.assert_allclose(val.item(), 0.0, atol=1e-7)

    def test_retracted_H_energy_conservation(self, gas_model_and_data):
        """
        Test that the reconstructed H based on the retraction done in 
        retract_general_tensors_to_gas_tensors satisfies the energy conservation condition:
        z^T Q^{-1} Q^{-T} H:zz^T + (H:zz^T)^T Q^{-1}Q^{-T} z = 0.
        """
        model, poly_comp, _, _, _, _ = gas_model_and_data
        if 2 not in poly_comp:
            pytest.skip("No quadratic term")
            
        rng = np.random.default_rng(1234)
        A_gen = torch.tensor(rng.standard_normal((R, R)), dtype=torch.float64)
        H_gen = torch.tensor(rng.standard_normal((R, R, R)), dtype=torch.float64)
        
        tensors = [A_gen]
        if 2 in poly_comp:
            tensors.append(H_gen)
            
        model.retract_general_tensors_to_gas_tensors(tensors)
        
        inner_tensors = model.model.get_params()
        H_recon = inner_tensors[poly_comp.index(2)]
        
        Qinv = torch.linalg.inv(model.Q)
        Qtil = Qinv @ Qinv.T
        
        for _ in range(5):
            z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
            H_zz = torch.einsum("ijk,j,k->i", H_recon, z, z)
            
            term1 = z @ Qtil @ H_zz
            term2 = H_zz @ Qtil @ z
            val = term1 + term2
            np.testing.assert_allclose(val.item(), 0.0, atol=1e-7)
