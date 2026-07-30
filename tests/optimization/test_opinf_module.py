"""Tests for OpInfModule with and without gas_flag."""

import numpy as np
import pytest
import torch

from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import OpInfModule
from nitrom.projections.linear_projection import LinearProjection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_module(
    opt_obj, Phi, poly_comp=(1, 2), reg=0.0, gas_flag=False,
    initial_guess=None, forcing_config=None,
):
    """Build a latent-space model then wrap it in an OpInfModule."""
    r = Phi.shape[-1]
    if gas_flag:
        rom = GasPolynomialModel(
            r, list(poly_comp), dtype=Phi.dtype,
            gas_params=initial_guess, forcing_config=forcing_config,
        )
    else:
        rom = PolynomialModel(
            r, list(poly_comp), dtype=Phi.dtype,
            tensors=initial_guess, forcing_config=forcing_config,
        )
    projection = LinearProjection([Phi, Phi])
    return OpInfModule(opt_obj, rom, projection, reg=reg)


class _MockOptObj:
    """Minimal stand-in for TrainingData with X, dX, weights."""

    def __init__(
        self, ntraj, N, nt, device="cpu", dtype=torch.float64, seed=0,
        forcing_fns=None,
    ):
        rng = np.random.default_rng(seed)
        self.X = torch.tensor(
            rng.standard_normal((ntraj, N, nt)), device=device, dtype=dtype
        )
        self.dX = torch.tensor(
            rng.standard_normal((ntraj, N, nt)), device=device, dtype=dtype
        )
        self.weights = torch.ones(ntraj, device=device, dtype=dtype)
        self.time = torch.linspace(0.0, 1.0, nt, device=device, dtype=dtype)
        self.forcing_fns = forcing_fns if forcing_fns is not None else []


NTRAJ, N, NT, R, M = 3, 6, 5, 3, 2
DTYPE = torch.float64


def _make_phi(N, r, seed=42):
    rng = np.random.default_rng(seed)
    Phi, _ = np.linalg.qr(rng.standard_normal((N, r)))
    return torch.tensor(Phi, dtype=DTYPE)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[False, True], ids=["standard", "gas"])
def model(request):
    gas_flag = request.param
    opt_obj = _MockOptObj(NTRAJ, N, NT, dtype=DTYPE, seed=0)
    Phi = _make_phi(N, R)
    poly_comp = [1, 2]
    return _make_module(opt_obj, Phi, poly_comp, reg=0.01, gas_flag=gas_flag)


@pytest.fixture()
def model_standard():
    opt_obj = _MockOptObj(NTRAJ, N, NT, dtype=DTYPE, seed=0)
    Phi = _make_phi(N, R)
    return _make_module(opt_obj, Phi, [1, 2], reg=0.01, gas_flag=False)


@pytest.fixture()
def model_gas():
    opt_obj = _MockOptObj(NTRAJ, N, NT, dtype=DTYPE, seed=0)
    Phi = _make_phi(N, R)
    return _make_module(opt_obj, Phi, [1, 2], reg=0.01, gas_flag=True)


def _make_forcing_fns(ntraj, m, seed=77):
    """Create simple linear forcing callables for testing."""
    rng = np.random.default_rng(seed)
    coeffs = [
        torch.tensor(rng.standard_normal((m,)), dtype=DTYPE)
        for _ in range(ntraj)
    ]
    offsets = [
        torch.tensor(rng.standard_normal((m,)), dtype=DTYPE)
        for _ in range(ntraj)
    ]
    return [
        (lambda t, c=c, o=o: c * t + o) for c, o in zip(coeffs, offsets)
    ]


@pytest.fixture()
def model_standard_forcing():
    forcing_fns = _make_forcing_fns(NTRAJ, M)
    opt_obj = _MockOptObj(NTRAJ, N, NT, dtype=DTYPE, seed=0, forcing_fns=forcing_fns)
    Phi = _make_phi(N, R)
    fc = {"forcing_exists": True, "m": M}
    return _make_module(opt_obj, Phi, [1, 2], reg=0.01, gas_flag=False, forcing_config=fc)


@pytest.fixture()
def model_gas_forcing():
    forcing_fns = _make_forcing_fns(NTRAJ, M)
    opt_obj = _MockOptObj(NTRAJ, N, NT, dtype=DTYPE, seed=0, forcing_fns=forcing_fns)
    Phi = _make_phi(N, R)
    fc = {"forcing_exists": True, "m": M}
    return _make_module(opt_obj, Phi, [1, 2], reg=0.01, gas_flag=True, forcing_config=fc)


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

class TestBasic:

    def test_forward_returns_scalar(self, model):
        loss = model()
        assert loss.ndim == 0

    def test_forward_nonnegative(self, model):
        loss = model()
        assert loss.item() >= 0.0

    def test_gradient_shapes_match_params(self, model):
        grads = model.gradient()
        params = list(model.parameters())
        assert len(grads) == len(params)
        for g, p in zip(grads, params):
            assert g.shape == p.shape, (
                f"grad shape {g.shape} != param shape {p.shape}"
            )

    def test_param_names(self, model):
        names = model.rom.param_names
        assert len(names) == len(list(model.parameters()))


# ---------------------------------------------------------------------------
# Gradient check via torch.autograd
# ---------------------------------------------------------------------------

class TestGradientAutograd:
    """Verify analytic gradient against torch.autograd."""

    def _autograd_check(self, model):
        # Compute analytic gradient
        grads_analytic = model.gradient()

        # Compute autograd gradient
        for p in model.parameters():
            p.requires_grad_(True)

        loss = model()
        loss.backward()
        grads_autograd = [p.grad.clone() for p in model.parameters()]

        for p in model.parameters():
            p.grad = None
            p.requires_grad_(False)

        for i, (ga, gad) in enumerate(zip(grads_analytic, grads_autograd)):
            np.testing.assert_allclose(
                ga.detach().numpy(),
                gad.detach().numpy(),
                rtol=1e-6,
                atol=1e-10,
                err_msg=f"Param {i}: analytic vs autograd mismatch",
            )

    def test_autograd_standard(self, model_standard):
        self._autograd_check(model_standard)

    def test_autograd_gas(self, model_gas):
        self._autograd_check(model_gas)

    def test_autograd_standard_forcing(self, model_standard_forcing):
        self._autograd_check(model_standard_forcing)

    def test_autograd_gas_forcing(self, model_gas_forcing):
        self._autograd_check(model_gas_forcing)


# ---------------------------------------------------------------------------
# Convergence: loss should decrease
# ---------------------------------------------------------------------------

class TestConvergence:

    def test_loss_decreases_adam(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss = model()
            grads = model.gradient()
            for param, grad in zip(model.parameters(), grads):
                param.grad = grad.contiguous().clone()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.6e} -> {losses[-1]:.6e}"
        )


# ---------------------------------------------------------------------------
# Initial guess
# ---------------------------------------------------------------------------

class TestInitialGuess:

    def test_standard_initial_guess(self):
        opt_obj = _MockOptObj(NTRAJ, N, NT, dtype=DTYPE, seed=0)
        Phi = _make_phi(N, R)
        poly_comp = [1, 2]

        init = [
            torch.eye(R, dtype=DTYPE),
            torch.zeros(R, R, R, dtype=DTYPE),
        ]
        model = _make_module(opt_obj, Phi, poly_comp, initial_guess=init)

        params = list(model.parameters())
        np.testing.assert_allclose(params[0].detach().numpy(), init[0].numpy())
        np.testing.assert_allclose(params[1].detach().numpy(), init[1].numpy())

    def test_gas_initial_guess(self):
        opt_obj = _MockOptObj(NTRAJ, N, NT, dtype=DTYPE, seed=0)
        Phi = _make_phi(N, R)
        poly_comp = [1, 2]

        init = [
            torch.eye(R, dtype=DTYPE),       # K
            torch.eye(R, dtype=DTYPE),       # R
            torch.eye(R, dtype=DTYPE),       # Q
            0.01 * torch.randn(R, R, R, dtype=DTYPE),  # S
        ]
        model = _make_module(opt_obj, Phi, poly_comp, gas_flag=True, initial_guess=init)

        params = list(model.parameters())
        assert len(params) == 4
        for p, ig in zip(params, init):
            np.testing.assert_allclose(p.detach().numpy(), ig.numpy())
