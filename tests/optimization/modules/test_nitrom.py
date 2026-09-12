"""Finite-difference gradient checks for NitromModule (polynomial, GAS, ATR ROMs).

Ambient dimension N = 5, latent dimension r = 2.  The module's analytic gradient
is a continuous adjoint, so the comparison is against a finite difference of the
forward solve at a fairly fine sub-step resolution (the two agree to
discretization accuracy).
"""

import pytest
import torch

from nitrom.latent_space_models.atr_polynomial_model import AtrPolynomialModel
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import NitromModule
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry

from ._gradcheck import (
    DTYPE,
    Data,
    LinearOutputFOM,
    assert_grad_close,
    finite_diff_grad,
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
    # Oblique projection (Psi != Phi) so the encoder/decoder gradients for both
    # bases are exercised; both are trainable in NiTROM.
    Phi = random_basis(N, R, seed=1)
    Psi = random_basis(N, R, seed=2)
    return LinearProjection([Phi, Psi])


def _fom(seed: int = 4) -> LinearOutputFOM:
    g = torch.Generator().manual_seed(seed)
    return LinearOutputFOM(torch.randn(NO, N, generator=g, dtype=DTYPE))


# Sub-step resolution for the gradient checks.
#
# The *discrete* adjoint is backpropagation through the RK solver, so it is the
# exact gradient of the discretized cost -- precisely what differencing
# ``module()`` measures.  It agrees to roundoff at any resolution, and in fact
# slightly *better* on a coarse grid (fewer steps, less accumulated roundoff:
# 1.1e-9 at 5 sub-steps vs 5.6e-9 at 100).  10 is plenty.
#
# The *continuous* adjoint solves a different (continuous) equation and only
# converges to the discrete gradient as the grid refines, at the order of the
# stepper.  Measured deviation vs sub-steps (worst of the polynomial/GAS models):
#
#     rk4 (4th order)   25 -> 1.2e-06    50 -> 3.4e-07   100 -> 2.8e-07
#     rk2 (2nd order)   25 -> 2.8e-04    50 -> 7.0e-05   100 -> 1.8e-05
#     backward_euler    25 -> 3.6e-02    50 -> 1.8e-02   100 -> 8.8e-03
#
# against tolerances of rtol=1e-4 for rk2/rk4 and 3e-2 for backward_euler.  Each
# entry below is the coarsest grid that clears its tolerance with margin; the
# checks are deterministic, so the margin does not need to absorb run-to-run
# noise.  rk45 is adaptive -- its own atol/rtol set the accuracy and the sub-step
# count barely matters -- so it just follows rk4.
#
# This is what keeps the file at ~2 min rather than ~10.
_N_SUBSTEPS_CONTINUOUS = {"rk4": 25, "rk45": 25, "rk2": 100, "backward_euler": 50}


def _reference_grad(module, model_factory, time_stepper, adjoint_method):
    """Reference gradient for a gradient check.

    For the *discrete* adjoint this is a finite difference of the cost -- the
    ground truth, and the anchor for everything else.

    For the *continuous* adjoint it is the discrete adjoint on the same grid.
    That is the same assertion (the continuous adjoint should reproduce the
    gradient of the discretized cost, to discretization error) but it costs one
    adjoint solve instead of the ``2 * n_params`` forward solves a finite
    difference needs -- ~70x less work here.  The chain stays anchored:
    the discrete parametrization of every one of these tests still checks the
    discrete adjoint against a finite difference.
    """
    if adjoint_method == "discrete":
        return finite_diff_grad(module)
    ref = _module(model_factory(), time_stepper=time_stepper,
                  adjoint_method="discrete", atol=module.atol, rtol=module.rtol)
    ref.n_substeps = module.n_substeps
    return ref.gradient()


def _n_substeps(adjoint_method: str, time_stepper: str) -> int:
    if adjoint_method == "discrete":
        return 10
    return _N_SUBSTEPS_CONTINUOUS[time_stepper]


def _module(model, time_stepper="rk4", adjoint_method="discrete", atol=1e-6, rtol=1e-3) -> NitromModule:
    registry = ParamRegistry(model, _projection())
    return NitromModule(
        _data(), registry, fom=_fom(), reg=0.01,
        n_substeps=_n_substeps(adjoint_method, time_stepper),
        time_stepper=time_stepper,
        n_leggauss=5, adjoint_method=adjoint_method, atol=atol, rtol=rtol,
    )


@pytest.mark.parametrize("adjoint_method", ["discrete", "continuous"])
@pytest.mark.parametrize("time_stepper", ["rk4", "rk2", "backward_euler", "rk45"])
def test_gradient_polynomial_rom(adjoint_method, time_stepper):
    def make_model():
        g = torch.Generator().manual_seed(2)
        tensors = [
            0.3 * torch.randn(R, R, generator=g, dtype=DTYPE),
            0.2 * torch.randn(R, R, R, generator=g, dtype=DTYPE),
            0.3 * torch.randn(R, M, generator=g, dtype=DTYPE),
        ]
        return PolynomialModel(
            R, [1, 2], dtype=DTYPE, tensors=tensors,
            forcing_config={"forcing_exists": True, "m": M},
        )

    model = make_model()
    if time_stepper == "rk45":
        atol, rtol = 1e-12, 1e-10
    else:
        atol, rtol = 1e-6, 1e-3
    module = _module(model, time_stepper=time_stepper, adjoint_method=adjoint_method, atol=atol, rtol=rtol)
    
    # Continuous/discretization mismatch has larger tolerance requirements
    if adjoint_method == "continuous" and time_stepper == "backward_euler":
        rtol_check, atol_check = 3e-2, 3e-2
    elif time_stepper == "rk45":
        rtol_check, atol_check = 2e-3, 1e-6
    else:
        rtol_check, atol_check = 1e-4, 1e-6
        
    assert_grad_close(
        module.gradient(),
        _reference_grad(module, make_model, time_stepper, adjoint_method),
        rtol=rtol_check, atol=atol_check,
    )


@pytest.mark.parametrize("adjoint_method", ["discrete", "continuous"])
@pytest.mark.parametrize("time_stepper", ["rk4", "rk2", "backward_euler", "rk45"])
def test_gradient_gas_rom(adjoint_method, time_stepper):
    def make_model():
        g = torch.Generator().manual_seed(3)
        eye = torch.eye(R, dtype=DTYPE)
        gas_params = [
            0.3 * torch.randn(R, R, generator=g, dtype=DTYPE),        # K
            eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),  # R
            eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),  # Q
            0.2 * torch.randn(R, R, R, generator=g, dtype=DTYPE),     # S
            0.3 * torch.randn(R, M, generator=g, dtype=DTYPE),        # B
        ]
        return GasPolynomialModel(
            R, [1, 2], dtype=DTYPE, gas_params=gas_params,
            forcing_config={"forcing_exists": True, "m": M},
        )

    model = make_model()
    if time_stepper == "rk45":
        atol, rtol = 1e-12, 1e-10
    else:
        atol, rtol = 1e-6, 1e-3
    module = _module(model, time_stepper=time_stepper, adjoint_method=adjoint_method, atol=atol, rtol=rtol)
    
    # S has diagonal entries S[i, i, k] that cancel exactly in the H = S - S^T
    # assembly (assemble_gas_tensors), so their true gradient is 0 -- both
    # adjoint methods agree on this.  The finite-difference reference is not
    # exactly 0 there: backward_euler's implicit solve has its own ~1e-6
    # tolerance floor, and rk45's adaptive step controller can accept/reject
    # steps differently for the +-eps perturbed parameter, injecting noise
    # unrelated to the true (zero) sensitivity.  Continuous/discretization
    # mismatch separately needs a looser tolerance.
    if adjoint_method == "continuous" and time_stepper == "backward_euler":
        rtol_check, atol_check = 3e-2, 3e-2
    elif time_stepper == "rk45":
        rtol_check, atol_check = 2e-3, 1.5e-2
    elif time_stepper == "backward_euler":
        rtol_check, atol_check = 1e-4, 5e-6
    else:
        rtol_check, atol_check = 1e-4, 1e-6

    assert_grad_close(
        module.gradient(),
        _reference_grad(module, make_model, time_stepper, adjoint_method),
        rtol=rtol_check, atol=atol_check,
    )


@pytest.mark.parametrize("adjoint_method", ["discrete", "continuous"])
def test_gradient_atr_rom(adjoint_method):
    def make_model():
        g = torch.Generator().manual_seed(5)
        eye = torch.eye(R, dtype=DTYPE)
        atr_params = [
            0.3 * torch.randn(R, R, generator=g, dtype=DTYPE),         # K
            eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),   # R
            eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),   # Q
            0.2 * torch.randn(R, R, R, generator=g, dtype=DTYPE),      # S
            0.3 * torch.randn(R, generator=g, dtype=DTYPE),            # Bhat
            0.3 * torch.randn(R, generator=g, dtype=DTYPE),            # m
            0.3 * torch.randn(R, M, generator=g, dtype=DTYPE),         # B
        ]
        return AtrPolynomialModel(
            R, [1, 2], dtype=DTYPE, atr_params=atr_params,
            forcing_config={"forcing_exists": True, "m": M},
        )

    module = _module(make_model(), adjoint_method=adjoint_method)
    # As in the GAS check, the diagonal entries S[i, i, k] cancel exactly in the
    # H = S - S^T assembly, so their true gradient is 0 and the analytic adjoint
    # returns exactly 0 there.  The finite-difference reference is only
    # 0 +- ULP noise (the ATR cost is O(1e4) here, so central differencing at
    # eps=1e-6 has an absolute noise floor of ~2e-6), which trips atol=1e-6.
    assert_grad_close(
        module.gradient(),
        _reference_grad(module, make_model, "rk4", adjoint_method),
        rtol=1e-4, atol=5e-6,
    )


def test_nitrom_h_regularization():
    # Verify that the forward loss matches the expected mathematical definition
    g = torch.Generator().manual_seed(42)
    tensors = [
        0.3 * torch.randn(R, R, generator=g, dtype=DTYPE),
        0.2 * torch.randn(R, R, R, generator=g, dtype=DTYPE),
        0.3 * torch.randn(R, M, generator=g, dtype=DTYPE),
    ]
    model = PolynomialModel(
        R, [1, 2], dtype=DTYPE, tensors=tensors,
        forcing_config={"forcing_exists": True, "m": M},
    )
    
    proj = _projection()
    registry = ParamRegistry(model, proj)
    
    # Create module with reg = 0.0 and reg = 0.5
    module_no_reg = NitromModule(
        _data(), registry, fom=_fom(), reg=0.0,
        n_substeps=10, time_stepper="rk4", adjoint_method="discrete"
    )
    module_with_reg = NitromModule(
        _data(), registry, fom=_fom(), reg=0.5,
        n_substeps=10, time_stepper="rk4", adjoint_method="discrete"
    )
    
    loss_no_reg = module_no_reg()
    loss_with_reg = module_with_reg()
    
    # Get H and check loss difference
    H = model.A_2
    expected_reg_term = 0.5 * torch.sum(H * H)
    
    assert torch.allclose(loss_with_reg, loss_no_reg + expected_reg_term, rtol=1e-6)

