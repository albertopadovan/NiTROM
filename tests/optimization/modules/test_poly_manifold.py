"""Finite-difference gradient check for PolyManifoldInfModule.

Quartic manifold: the quadratic, cubic, and quartic tensors (A2, A3, A4) are
learned.  Ambient dimension N = 5, latent dimension r = 2.
"""

import torch

from nitrom.optimization import PolyManifoldInfModule

from ._gradcheck import (
    DTYPE,
    Data,
    assert_grad_close,
    finite_diff_grad,
    random_basis,
)

N, R, NTRAJ, NT = 5, 2, 2, 6
NONLIN_POLY_COMP = [2, 3, 4]


def test_gradient_quartic_manifold():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(NTRAJ, N, NT, generator=g, dtype=DTYPE)
    weights = 1.0 + 0.5 * torch.rand(NTRAJ, generator=g, dtype=DTYPE)
    data = Data(X, weights=weights)

    Phi = random_basis(N, R, seed=1)

    # Random initial quadratic/cubic/quartic tensors, A_k of shape (N, r, ..., r).
    initial_guess = [
        0.2 * torch.randn((N,) + (R,) * k, generator=g, dtype=DTYPE)
        for k in NONLIN_POLY_COMP
    ]

    module = PolyManifoldInfModule(
        data, NONLIN_POLY_COMP, Phi, Psi=Phi, reg=0.01,
        initial_guess=initial_guess,
    )

    # Sanity: exactly the three nonlinear tensors are trainable.
    assert len(list(module.parameters())) == len(NONLIN_POLY_COMP)

    assert_grad_close(module.gradient(), finite_diff_grad(module))
