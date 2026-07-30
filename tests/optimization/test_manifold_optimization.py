import pytest
import torch
import torch.nn as nn

from nitrom.optimization import train
from nitrom.optimization.modules.base import InferenceModule


class DummyManifoldModule(InferenceModule):
    """A simple InferenceModule for testing manifold optimization."""

    def __init__(self, N=5, r=2):
        super().__init__()
        self.Phi = nn.Parameter(torch.randn(N, r, dtype=torch.float64))
        self.Psi = nn.Parameter(torch.randn(N, r, dtype=torch.float64))
        self.theta = nn.Parameter(torch.randn(r, r, dtype=torch.float64))

    def forward(self) -> torch.Tensor:
        # Minimum at Phi=Psi=theta=0; Phi, Psi are constrained to be orthonormal.
        return (self.Phi**2).sum() + (self.Psi**2).sum() + (self.theta**2).sum()

    def gradient(self) -> list[torch.Tensor]:
        return [2.0 * self.Phi, 2.0 * self.Psi, 2.0 * self.theta]


def test_set_manifold_types_validation():
    """set_manifold_types rejects unknown parameters and invalid manifolds."""
    module = DummyManifoldModule()

    # Default: every parameter is Euclidean.
    assert module.get_manifold_types() == ["euclidean", "euclidean", "euclidean"]

    with pytest.raises(KeyError, match="Unknown parameter 'non_existent'"):
        module.set_manifold_types(["non_existent"], ["grassmann"])

    with pytest.raises(ValueError, match="manifold type for 'Phi'"):
        module.set_manifold_types(["Phi"], ["invalid_manifold"])

    with pytest.raises(ValueError, match="equal length"):
        module.set_manifold_types(["Phi", "Psi"], ["grassmann"])


def test_get_manifold_types_reflects_setting():
    """get_manifold_types returns the per-parameter list in parameters() order."""
    module = DummyManifoldModule()
    module.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])
    # parameters() order is Phi, Psi, theta.
    assert module.get_manifold_types() == ["grassmann", "stiefel", "euclidean"]


@pytest.mark.parametrize("optimizer_type", ["adam", "sgd", "lbfgs"])
def test_manifold_optimization_properties(optimizer_type):
    """Parameters stay orthonormal and gradients are projected to tangent spaces."""
    module = DummyManifoldModule(N=5, r=2)
    I_r = torch.eye(2, dtype=torch.float64)

    # Initial matrices are random and not orthonormal.
    assert not torch.allclose(module.Phi.T @ module.Phi, I_r, atol=1e-3)
    assert not torch.allclose(module.Psi.T @ module.Psi, I_r, atol=1e-3)

    module.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])
    train(module, n_epochs=5, lr=0.01, optimizer_type=optimizer_type, print_every=1)

    # 1. Orthonormality after training.
    assert torch.allclose(module.Phi.T @ module.Phi, I_r, atol=1e-12)
    assert torch.allclose(module.Psi.T @ module.Psi, I_r, atol=1e-12)

    # 2. Assigned gradients lie in the respective tangent spaces.
    assert module.Phi.grad is not None
    assert module.Psi.grad is not None

    # Grassmann constraint: Phi.T @ grad_Phi = 0
    assert torch.allclose(
        module.Phi.T @ module.Phi.grad,
        torch.zeros(2, 2, dtype=torch.float64),
        atol=1e-7,
    )
    # Stiefel constraint: Psi.T @ grad_Psi is skew-symmetric.
    skew = module.Psi.T @ module.Psi.grad
    assert torch.allclose(
        skew + skew.T, torch.zeros(2, 2, dtype=torch.float64), atol=1e-7
    )


def test_euclidean_is_default():
    """Without set_manifold_types, parameters are not constrained."""
    module = DummyManifoldModule(N=5, r=2)
    I_r = torch.eye(2, dtype=torch.float64)

    train(module, n_epochs=2, lr=0.01, optimizer_type="sgd")

    # Euclidean parameters are not orthonormalized.
    assert not torch.allclose(module.Phi.T @ module.Phi, I_r, atol=1e-3)
    assert not torch.allclose(module.Psi.T @ module.Psi, I_r, atol=1e-3)


def test_partial_manifold_assignment():
    """Only the parameters assigned a manifold are constrained; the rest stay free."""
    module = DummyManifoldModule(N=5, r=2)
    I_r = torch.eye(2, dtype=torch.float64)

    # Only Psi -> Stiefel; Phi is left Euclidean (the default).
    module.set_manifold_types(["Psi"], ["stiefel"])
    train(module, n_epochs=3, lr=0.01, optimizer_type="sgd")

    assert not torch.allclose(module.Phi.T @ module.Phi, I_r, atol=1e-3)
    assert torch.allclose(module.Psi.T @ module.Psi, I_r, atol=1e-12)

    skew = module.Psi.T @ module.Psi.grad
    assert torch.allclose(
        skew + skew.T, torch.zeros(2, 2, dtype=torch.float64), atol=1e-7
    )
