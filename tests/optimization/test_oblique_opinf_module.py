"""Tests for ObliqueOpInfModule (oblique operator inference, learnable Psi)."""

import numpy as np
import pytest
import torch

from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import ObliqueOpInfModule
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry

NTRAJ, N, NT, R, M = 3, 6, 5, 3, 2
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockOptObj:
    """Minimal stand-in for TrainingData with X, dX, weights."""

    def __init__(
        self, ntraj, N, nt, device="cpu", dtype=DTYPE, seed=0, forcing_fns=None,
    ):
        rng = np.random.default_rng(seed)
        self.X = torch.tensor(
            rng.standard_normal((ntraj, N, nt)), device=device, dtype=dtype
        )
        self.dX = torch.tensor(
            rng.standard_normal((ntraj, N, nt)), device=device, dtype=dtype
        )
        self.weights = torch.tensor(
            rng.uniform(0.5, 2.0, ntraj), device=device, dtype=dtype
        )
        self.time = torch.linspace(0.0, 1.0, nt, device=device, dtype=dtype)
        self.forcing_fns = forcing_fns if forcing_fns is not None else []


def _make_basis(N, r, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((N, r)))
    return torch.tensor(Q, dtype=DTYPE)


def _make_module(
    opt_obj, poly_comp=(1, 2), reg=0.0, gas_flag=False, forcing_config=None,
):
    """Build a latent-space model then wrap it in an ObliqueOpInfModule."""
    rng = np.random.default_rng(7)
    if gas_flag:
        # Seed the GAS parameters explicitly: the default init is random and
        # unseeded, and near-singular Q/R make the test wildly ill-conditioned.
        gas_params = [
            torch.tensor(np.eye(R) + 0.1 * rng.standard_normal((R, R)), dtype=DTYPE)
            for _ in ("K", "R", "Q")
        ]
        gas_params.append(
            0.1 * torch.tensor(rng.standard_normal((R, R, R)), dtype=DTYPE)
        )
        if forcing_config is not None and forcing_config.get("forcing_exists"):
            gas_params.append(
                0.1 * torch.tensor(rng.standard_normal((R, M)), dtype=DTYPE)
            )
        rom = GasPolynomialModel(
            R, list(poly_comp), dtype=DTYPE, gas_params=gas_params,
            forcing_config=forcing_config,
        )
    else:
        tensors = [
            0.1 * torch.tensor(rng.standard_normal((R,) * (k + 1)), dtype=DTYPE)
            for k in poly_comp
        ]
        if forcing_config is not None and forcing_config.get("forcing_exists"):
            tensors.append(
                0.1 * torch.tensor(rng.standard_normal((R, M)), dtype=DTYPE)
            )
        rom = PolynomialModel(
            R, list(poly_comp), dtype=DTYPE, tensors=tensors,
            forcing_config=forcing_config,
        )
    # Oblique: Phi and Psi are different (and non-orthogonal to each other).
    projection = LinearProjection([_make_basis(N, R, 42), _make_basis(N, R, 11)])
    registry = ParamRegistry(rom, projection)
    return ObliqueOpInfModule(opt_obj, registry, reg=reg)


def _forcing_fns(ntraj, m):
    return [
        (lambda t, k=k: torch.tensor(
            [np.cos((k + 1) * float(t)), np.sin((k + 2) * float(t))][:m],
            dtype=DTYPE,
        ))
        for k in range(ntraj)
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def model_standard():
    return _make_module(_MockOptObj(NTRAJ, N, NT, seed=0), reg=0.0)


@pytest.fixture()
def model_standard_reg():
    return _make_module(_MockOptObj(NTRAJ, N, NT, seed=1), reg=0.01)


@pytest.fixture()
def model_gas():
    return _make_module(_MockOptObj(NTRAJ, N, NT, seed=2), reg=0.01, gas_flag=True)


@pytest.fixture()
def model_standard_forcing():
    opt_obj = _MockOptObj(NTRAJ, N, NT, seed=3, forcing_fns=_forcing_fns(NTRAJ, M))
    return _make_module(
        opt_obj, reg=0.01,
        forcing_config={"forcing_exists": True, "m": M},
    )


@pytest.fixture()
def model_gas_forcing():
    opt_obj = _MockOptObj(NTRAJ, N, NT, seed=4, forcing_fns=_forcing_fns(NTRAJ, M))
    return _make_module(
        opt_obj, reg=0.01, gas_flag=True,
        forcing_config={"forcing_exists": True, "m": M},
    )


@pytest.fixture(params=["standard", "gas"])
def model(request, model_standard_reg, model_gas):
    return model_standard_reg if request.param == "standard" else model_gas


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

class TestBasics:

    def test_forward_returns_scalar(self, model):
        assert model().ndim == 0

    def test_forward_nonnegative(self, model):
        assert float(model().detach()) >= 0.0

    def test_param_names(self, model):
        assert model.param_names == list(model.rom.param_names) + ["Phi", "Psi"]
        assert len(model.param_names) == len(model.parameters())

    def test_gradient_shapes_match_params(self, model):
        grads = model.gradient()
        params = model.parameters()
        assert len(grads) == len(params)
        for g, p in zip(grads, params):
            assert g.shape == p.shape

    def test_manifolds_default_to_euclidean(self, model):
        """The caller chooses the manifolds, as for NitromModule."""
        assert set(model.get_manifold_types()) == {"euclidean"}
        model.set_manifold_types(["Psi"], ["stiefel"])
        types = dict(zip(
            [n for n, _ in model.named_parameters()], model.get_manifold_types()
        ))
        assert types["Psi"] == "stiefel"
        assert types["Phi"] == "euclidean"

    def test_cost_matches_definition(self, model_standard):
        """Recompute J from its definition and compare with forward()."""
        m = model_standard
        m()  # sync the registry
        Phi = m.projection.Phi.detach()
        Psi = m.projection.Psi.detach()
        S = torch.linalg.inv(Psi.T @ Phi)
        X, dX = m.training_data.X, m.training_data.dX
        total = 0.0
        for j in range(m.ntraj):
            acc = 0.0
            for i in range(m.nt):
                z = Psi.T @ X[j, :, i]
                dz = Psi.T @ dX[j, :, i]
                res = (S @ dz - S @ m.rom.evaluate_rhs(0.0, z)).detach()
                acc = acc + float(res @ res)
            total += acc / float(m.weights[j])
        np.testing.assert_allclose(
            float(model_standard().detach()), total, rtol=1e-10
        )

    def test_flat_layout_matches_trajectory_layout(self):
        """Row j * nt + i of the flat data is snapshot i of trajectory j."""
        opt_obj = _MockOptObj(NTRAJ, N, NT, seed=9)
        m = _make_module(opt_obj)
        for j in range(NTRAJ):
            for i in range(NT):
                assert torch.equal(m.X[j * NT + i], opt_obj.X[j, :, i])
                assert torch.equal(m.dX[j * NT + i], opt_obj.dX[j, :, i])
                # Snapshot slicing used by the forcing path.
                assert torch.equal(m.X[i::NT][j], opt_obj.X[j, :, i])
            assert float(m.w_row[j * NT]) == pytest.approx(
                1.0 / float(opt_obj.weights[j])
            )


# ---------------------------------------------------------------------------
# Gradient check via torch.autograd
# ---------------------------------------------------------------------------

class TestGradientAutograd:
    """Verify the analytic gradient against torch.autograd."""

    def _autograd_check(self, model):
        grads_analytic = model.gradient()

        for p in model.parameters():
            p.requires_grad_(True)

        loss = model()
        loss.backward()
        grads_autograd = [p.grad.clone() for p in model.parameters()]

        for p in model.parameters():
            p.grad = None
            p.requires_grad_(False)

        names = [n for n, _ in model.named_parameters()]
        for name, ga, gad in zip(names, grads_analytic, grads_autograd):
            np.testing.assert_allclose(
                ga.detach().numpy(), gad.detach().numpy(),
                rtol=1e-6, atol=1e-10,
                err_msg=f"Param '{name}': analytic vs autograd mismatch",
            )

    def test_autograd_standard(self, model_standard):
        self._autograd_check(model_standard)

    def test_autograd_standard_reg(self, model_standard_reg):
        self._autograd_check(model_standard_reg)

    def test_autograd_gas(self, model_gas):
        self._autograd_check(model_gas)

    def test_autograd_standard_forcing(self, model_standard_forcing):
        self._autograd_check(model_standard_forcing)

    def test_autograd_gas_forcing(self, model_gas_forcing):
        self._autograd_check(model_gas_forcing)


# ---------------------------------------------------------------------------
# Learnability
# ---------------------------------------------------------------------------

class TestLearnability:

    def test_freeze_phi(self, model_standard):
        """The intended setup: Phi fixed, Psi and the dynamics learned."""
        names = [n for n, _ in model_standard.named_parameters()]
        grads_free = model_standard.gradient()
        assert not torch.allclose(
            grads_free[names.index("Phi")],
            torch.zeros_like(grads_free[names.index("Phi")]),
        )

        model_standard.set_unlearnable("Phi")
        grads = model_standard.gradient()
        assert torch.allclose(
            grads[names.index("Phi")], torch.zeros_like(grads[names.index("Phi")])
        )
        assert not torch.allclose(
            grads[names.index("Psi")], torch.zeros_like(grads[names.index("Psi")])
        )

    def test_freeze_psi(self, model_standard):
        names = [n for n, _ in model_standard.named_parameters()]
        model_standard.set_unlearnable("Psi")
        grads = model_standard.gradient()
        idx = names.index("Psi")
        assert torch.allclose(grads[idx], torch.zeros_like(grads[idx]))


# ---------------------------------------------------------------------------
# Convergence: loss should decrease
# ---------------------------------------------------------------------------

class TestConvergence:

    def test_loss_decreases_adam(self, model_standard):
        model = model_standard
        model.set_unlearnable("Phi")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            loss = model()
            grads = model.gradient()
            for p, g in zip(model.parameters(), grads):
                p.grad = g
            optimizer.step()
            losses.append(float(loss.detach()))
        assert losses[-1] < losses[0]
