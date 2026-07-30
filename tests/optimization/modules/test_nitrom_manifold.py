"""Tests for NitromModule cost/gradients under non-Euclidean manifold constraints.

Phi lives on the Grassmann manifold, Psi lives on the Stiefel manifold,
and all other parameters live on the Euclidean manifold.
"""

import numpy as np
import pytest
import torch

from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import NitromModule, train
from nitrom.optimization.manifold_optimization import project, to_manifold
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry

from ._gradcheck import (
    DTYPE,
    Data,
    LinearOutputFOM,
    random_basis,
)

N, R, NTRAJ, NT, M, NO = 5, 2, 2, 4, 1, 3


def _data(seed: int = 0) -> Data:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(NTRAJ, N, NT, generator=g, dtype=DTYPE)
    weights = 1.0 + 0.5 * torch.rand(NTRAJ, generator=g, dtype=DTYPE)
    forcing_fns = [
        (lambda t, c=0.5 * (i + 1): torch.tensor([c], dtype=DTYPE))
        for i in range(NTRAJ)
    ]
    return Data(X, weights=weights, forcing_fns=forcing_fns)


def _projection() -> LinearProjection:
    # Use random bases that are orthonormal (so they start on the manifold)
    Phi = random_basis(N, R, seed=1)
    Psi = random_basis(N, R, seed=2)
    return LinearProjection([Phi, Psi])


def _fom(seed: int = 4) -> LinearOutputFOM:
    g = torch.Generator().manual_seed(seed)
    return LinearOutputFOM(torch.randn(NO, N, generator=g, dtype=DTYPE))


def _module(model) -> NitromModule:
    registry = ParamRegistry(model, _projection())
    return NitromModule(
        _data(),
        registry,
        fom=_fom(),
        reg=0.01,
        n_substeps=100,
        time_stepper="rk4",
        n_leggauss=5,
    )


def check_manifold_gradients(module: NitromModule):
    # Ensure Phi and Psi parameters are on the manifold initially
    with torch.no_grad():
        module.Phi.copy_(to_manifold(module.Phi, "grassmann"))
        module.Psi.copy_(to_manifold(module.Psi, "stiefel"))

    # Compute Euclidean gradients
    grads = module.gradient()
    grad_dict = dict(zip(module.param_names, grads, strict=True))

    I_r = torch.eye(R, dtype=DTYPE)
    assert torch.allclose(module.Phi.T @ module.Phi, I_r, atol=1e-12)
    assert torch.allclose(module.Psi.T @ module.Psi, I_r, atol=1e-12)

    eps = 1e-6
    g_seed = torch.Generator().manual_seed(42)

    for name in module.param_names:
        param = getattr(module, name)
        grad = grad_dict[name]

        # Determine manifold type
        if name == "Phi":
            mtype = "grassmann"
        elif name == "Psi":
            mtype = "stiefel"
        else:
            mtype = "euclidean"

        # 1. Project Euclidean gradient onto the tangent space of the manifold at param
        proj_grad = project(param, grad, mtype)

        # 2. Check tangent space algebraic properties for manifold parameters:
        if mtype == "grassmann":
            grassmann_check = param.T @ proj_grad
            assert torch.allclose(
                grassmann_check, torch.zeros(R, R, dtype=DTYPE), atol=1e-6
            )
        elif mtype == "stiefel":
            stiefel_check = param.T @ proj_grad
            assert torch.allclose(
                stiefel_check + stiefel_check.T,
                torch.zeros(R, R, dtype=DTYPE),
                atol=1e-6,
            )

        # 3. Manifold finite-difference check (Riemannian gradient consistency check):
        # Create a random direction of the same shape as param
        H = torch.randn(param.shape, generator=g_seed, dtype=DTYPE)

        # Project H onto the tangent space of the manifold at param
        tangent = project(param, H, mtype)

        # Normalize tangent to make sure we don't perturb too far
        norm_tangent = torch.linalg.norm(tangent)
        if norm_tangent > 0:
            tangent = tangent / norm_tangent

        with torch.no_grad():
            param_orig = param.clone()

            # Perturb forward: param_plus = retract(param + eps * tangent)
            if mtype in ("grassmann", "stiefel"):
                param.copy_(to_manifold(param_orig + eps * tangent, mtype))
            else:
                param.copy_(param_orig + eps * tangent)
            loss_plus = module()

            # Perturb backward: param_minus = retract(param - eps * tangent)
            if mtype in ("grassmann", "stiefel"):
                param.copy_(to_manifold(param_orig - eps * tangent, mtype))
            else:
                param.copy_(param_orig - eps * tangent)
            loss_minus = module()

            # Restore parameter
            param.copy_(param_orig)

        df_fd = (loss_plus - loss_minus) / (2.0 * eps)
        # Using proj_grad or grad yields the same value because tangent is in the tangent space
        df_analytic = torch.sum(proj_grad * tangent)

        np.testing.assert_allclose(
            df_fd.item(),
            df_analytic.item(),
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"Riemannian gradient mismatch for parameter '{name}' ({mtype})",
        )


def test_manifold_gradients_polynomial_rom():
    g = torch.Generator().manual_seed(2)
    tensors = [
        0.3 * torch.randn(R, R, generator=g, dtype=DTYPE),
        0.2 * torch.randn(R, R, R, generator=g, dtype=DTYPE),
        0.3 * torch.randn(R, M, generator=g, dtype=DTYPE),
    ]
    model = PolynomialModel(
        R,
        [1, 2],
        dtype=DTYPE,
        tensors=tensors,
        forcing_config={"forcing_exists": True, "m": M},
    )
    module = _module(model)
    check_manifold_gradients(module)


def test_manifold_gradients_gas_rom():
    g = torch.Generator().manual_seed(3)
    eye = torch.eye(R, dtype=DTYPE)
    gas_params = [
        0.3 * torch.randn(R, R, generator=g, dtype=DTYPE),  # K
        eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),  # R
        eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),  # Q
        0.2 * torch.randn(R, R, R, generator=g, dtype=DTYPE),  # S
        0.3 * torch.randn(R, M, generator=g, dtype=DTYPE),  # B
    ]
    model = GasPolynomialModel(
        R,
        [1, 2],
        dtype=DTYPE,
        gas_params=gas_params,
        forcing_config={"forcing_exists": True, "m": M},
    )
    module = _module(model)
    check_manifold_gradients(module)


@pytest.mark.parametrize("optimizer_type", ["sgd", "adam", "lbfgs"])
def test_nitrom_manifold_optimization_run(optimizer_type):
    # Test with a small Polynomial ROM
    g = torch.Generator().manual_seed(2)
    tensors = [
        0.3 * torch.randn(R, R, generator=g, dtype=DTYPE),
        0.2 * torch.randn(R, R, R, generator=g, dtype=DTYPE),
        0.3 * torch.randn(R, M, generator=g, dtype=DTYPE),
    ]
    model = PolynomialModel(
        R,
        [1, 2],
        dtype=DTYPE,
        tensors=tensors,
        forcing_config={"forcing_exists": True, "m": M},
    )
    module = _module(model)

    # Verify initial cost and that parameters are orthonormal initially
    initial_loss = module().item()
    I_r = torch.eye(R, dtype=DTYPE)

    # Enforce initial orthonormality
    with torch.no_grad():
        module.Phi.copy_(to_manifold(module.Phi, "grassmann"))
        module.Psi.copy_(to_manifold(module.Psi, "stiefel"))

    # Phi on the Grassmann manifold, Psi on the Stiefel manifold.
    module.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])

    # Train for 3 epochs with stable optimizer-specific learning rates
    lr = {"sgd": 1e-8, "adam": 1e-4, "lbfgs": 0.1}[optimizer_type]
    train(
        module,
        n_epochs=3,
        lr=lr,
        optimizer_type=optimizer_type,
        print_every=1,
    )

    final_loss = module().item()

    # Check that loss decreased or remained stable
    assert final_loss <= initial_loss + 1e-10

    # Check orthonormality constraints after training
    assert torch.allclose(module.Phi.T @ module.Phi, I_r, atol=1e-12)
    assert torch.allclose(module.Psi.T @ module.Psi, I_r, atol=1e-12)

    # Evaluate cost and gradients at the final parameter values to verify project_tangent
    grads = module.gradient()
    for param, grad in zip(module.parameters(), grads, strict=True):
        param.grad = grad.contiguous().clone()

    with torch.no_grad():
        for name, param in module.named_parameters():
            if name in ["Phi", "Psi"]:
                mtype = "grassmann" if name == "Phi" else "stiefel"
                projected = project(param, param.grad, mtype)
                param.grad.copy_(projected)

    # Check tangent space conditions of the final gradients
    assert module.Phi.grad is not None
    assert module.Psi.grad is not None

    # Grassmann check: Phi^T @ grad_Phi = 0
    grassmann_check = module.Phi.T @ module.Phi.grad
    assert torch.allclose(
        grassmann_check, torch.zeros(R, R, dtype=DTYPE), atol=1e-6
    )

    # Stiefel check: Psi^T @ grad_Psi + grad_Psi^T @ Psi = 0
    stiefel_check = module.Psi.T @ module.Psi.grad
    assert torch.allclose(
        stiefel_check + stiefel_check.T,
        torch.zeros(R, R, dtype=DTYPE),
        atol=1e-6,
    )

