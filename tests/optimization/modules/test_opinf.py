"""Finite-difference gradient checks for OpInfModule (polynomial, GAS, ATR ROMs).

Ambient dimension N = 5, latent dimension r = 2.
"""

import torch

from nitrom.latent_space_models.atr_polynomial_model import AtrPolynomialModel
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import OpInfModule
from nitrom.projections.linear_projection import LinearProjection

from ._gradcheck import (
    DTYPE,
    Data,
    assert_grad_close,
    finite_diff_grad,
    random_basis,
)

N, R, NTRAJ, NT, M = 5, 2, 2, 6, 1


def _data(seed: int = 0) -> Data:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(NTRAJ, N, NT, generator=g, dtype=DTYPE)
    dX = torch.randn(NTRAJ, N, NT, generator=g, dtype=DTYPE)
    weights = 1.0 + 0.5 * torch.rand(NTRAJ, generator=g, dtype=DTYPE)
    forcing_fns = [
        (lambda t, c=0.5 * (i + 1): torch.tensor([c], dtype=DTYPE))
        for i in range(NTRAJ)
    ]
    return Data(X, dX=dX, weights=weights, forcing_fns=forcing_fns)


def _projection() -> LinearProjection:
    Phi = random_basis(N, R, seed=1)
    return LinearProjection([Phi, Phi])


def test_gradient_polynomial_rom():
    g = torch.Generator().manual_seed(2)
    tensors = [
        torch.randn(R, R, generator=g, dtype=DTYPE),
        0.3 * torch.randn(R, R, R, generator=g, dtype=DTYPE),
        torch.randn(R, M, generator=g, dtype=DTYPE),
    ]
    rom = PolynomialModel(
        R, [1, 2], dtype=DTYPE, tensors=tensors,
        forcing_config={"forcing_exists": True, "m": M},
    )
    module = OpInfModule(_data(), rom, _projection(), reg=0.01)
    assert_grad_close(module.gradient(), finite_diff_grad(module))


def test_gradient_gas_rom():
    g = torch.Generator().manual_seed(3)
    eye = torch.eye(R, dtype=DTYPE)
    gas_params = [
        torch.randn(R, R, generator=g, dtype=DTYPE),               # K
        eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),    # R (invertible)
        eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),    # Q (invertible)
        0.3 * torch.randn(R, R, R, generator=g, dtype=DTYPE),       # S
        torch.randn(R, M, generator=g, dtype=DTYPE),               # B
    ]
    rom = GasPolynomialModel(
        R, [1, 2], dtype=DTYPE, gas_params=gas_params,
        forcing_config={"forcing_exists": True, "m": M},
    )
    module = OpInfModule(_data(), rom, _projection(), reg=0.01)
    assert_grad_close(module.gradient(), finite_diff_grad(module))


def test_gradient_atr_rom():
    g = torch.Generator().manual_seed(4)
    eye = torch.eye(R, dtype=DTYPE)
    atr_params = [
        torch.randn(R, R, generator=g, dtype=DTYPE),                # K
        eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),    # R (invertible)
        eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),    # Q (invertible)
        0.3 * torch.randn(R, R, R, generator=g, dtype=DTYPE),       # S
        torch.randn(R, generator=g, dtype=DTYPE),                   # Bhat
        torch.randn(R, generator=g, dtype=DTYPE),                   # m
        torch.randn(R, M, generator=g, dtype=DTYPE),                # B
    ]
    rom = AtrPolynomialModel(
        R, [1, 2], dtype=DTYPE, atr_params=atr_params,
        forcing_config={"forcing_exists": True, "m": M},
    )
    module = OpInfModule(_data(), rom, _projection(), reg=0.01)
    assert_grad_close(module.gradient(), finite_diff_grad(module))


def test_gradient_atr_rom_no_forcing():
    g = torch.Generator().manual_seed(5)
    eye = torch.eye(R, dtype=DTYPE)
    atr_params = [
        torch.randn(R, R, generator=g, dtype=DTYPE),
        eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),
        eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),
        0.3 * torch.randn(R, R, R, generator=g, dtype=DTYPE),
        torch.randn(R, generator=g, dtype=DTYPE),
        torch.randn(R, generator=g, dtype=DTYPE),
    ]
    rom = AtrPolynomialModel(R, [1, 2], dtype=DTYPE, atr_params=atr_params)
    data = Data(_data().X, dX=_data().dX, forcing_fns=[])
    module = OpInfModule(data, rom, _projection(), reg=0.01)
    assert_grad_close(module.gradient(), finite_diff_grad(module))
