"""NiTROM on non-uniform time grids and with per-sample weights.

* The forward pass integrates every measurement interval with ``n_substeps``
  fixed steps of ``h_k = (t_k - t_{k-1}) / n_substeps`` (checked against the
  closed-form RK4 propagator of a linear model).
* The analytic gradient (discrete adjoint) is checked against torch autodiff of
  the cost, on uniform and non-uniform grids, with and without per-sample
  ``time_weights``, through the Gram and the general output paths, for a
  forced polynomial and an unforced GAS model, and with the dense solve and
  its per-interval fallback.  The continuous adjoint is checked against the
  discrete one to discretization error.
"""

import pytest
import torch

from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import NitromModule
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry

from ._gradcheck import DTYPE, Data, LinearOutputFOM, random_basis

N, R, NTRAJ, M, NO = 6, 2, 2, 1, 3

GRIDS = {
    "uniform": torch.linspace(0.0, 1.0, 9, dtype=DTYPE),
    # fine start, coarse tail (two runs of equal intervals)
    "two_runs": torch.tensor([0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=DTYPE),
    # three runs, the middle one a single interval
    "three_runs": torch.tensor([0.0, 0.05, 0.1, 0.25, 0.45, 0.65, 0.85], dtype=DTYPE),
}


class StateFOM:
    """y = x (enables the Gram form)."""

    output_is_state = True

    def compute_output(self, q):
        return q

    def apply_output_adjoint(self, e, q):
        return e


def trapezoid(t: torch.Tensor) -> torch.Tensor:
    c = torch.empty_like(t)
    c[1:-1] = 0.5 * (t[2:] - t[:-2])
    c[0], c[-1] = 0.5 * (t[1] - t[0]), 0.5 * (t[-1] - t[-2])
    return c / c.mean()


def _data(t, tw, forced, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = 0.5 * torch.randn(NTRAJ, N, len(t), generator=g, dtype=DTYPE)
    weights = 1.0 + 0.5 * torch.rand(NTRAJ, generator=g, dtype=DTYPE)
    fns = ([(lambda tt, c=0.5 * (i + 1): torch.tensor([c * float(torch.cos(torch.as_tensor(tt)))],
                                                     dtype=DTYPE)) for i in range(NTRAJ)]
           if forced else [])
    d = Data(X, weights=weights, time=t, forcing_fns=fns)
    if tw == "trapezoid":
        d.time_weights = trapezoid(t)
    elif tw == "per_traj":
        d.time_weights = torch.stack([trapezoid(t), 0.5 + torch.rand(len(t), generator=g, dtype=DTYPE)])
    return d


def _model(kind):
    g = torch.Generator().manual_seed(2)
    if kind == "poly_forced":
        return PolynomialModel(
            R, [1, 2], dtype=DTYPE,
            tensors=[0.3 * torch.randn(R, R, generator=g, dtype=DTYPE),
                     0.2 * torch.randn(R, R, R, generator=g, dtype=DTYPE),
                     0.3 * torch.randn(R, M, generator=g, dtype=DTYPE)],
            forcing_config={"forcing_exists": True, "m": M})
    eye = torch.eye(R, dtype=DTYPE)
    return GasPolynomialModel(
        R, [1, 2], dtype=DTYPE,
        gas_params=[0.3 * torch.randn(R, R, generator=g, dtype=DTYPE),
                    eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),
                    eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),
                    0.2 * torch.randn(R, R, R, generator=g, dtype=DTYPE)])


def _module(grid, tw, kind, output, adjoint="discrete", n_substeps=4):
    fom = StateFOM() if output == "state" else LinearOutputFOM(
        torch.randn(NO, N, generator=torch.Generator().manual_seed(4), dtype=DTYPE))
    proj = LinearProjection([random_basis(N, R, seed=1), random_basis(N, R, seed=2)])
    return NitromModule(_data(GRIDS[grid], tw, kind == "poly_forced"), ParamRegistry(_model(kind), proj),
                        fom=fom, reg=0.01, n_substeps=n_substeps, time_stepper="rk4",
                        adjoint_method=adjoint)


def autodiff_grad(module):
    # The Gram path caches Phi^T X; a copy built by gradient() (under no_grad)
    # would hide its dependence on Phi from autograd.
    module.__dict__.pop("_gram_phi_cache", None)
    params = list(module.parameters())
    J = module()
    return list(torch.autograd.grad(J, params, allow_unused=True))


def assert_close(analytic, reference, rtol, atol):
    for i, (a, b) in enumerate(zip(analytic, reference, strict=True)):
        b = torch.zeros_like(a) if b is None else b
        err = float((a - b).abs().max())
        scale = float(b.abs().max())
        assert err <= atol + rtol * scale, f"param {i}: |diff| {err:.2e}, |ref| {scale:.2e}"


@pytest.mark.parametrize("grid", list(GRIDS))
@pytest.mark.parametrize("tw", [None, "trapezoid", "per_traj"])
@pytest.mark.parametrize("kind", ["poly_forced", "gas"])
@pytest.mark.parametrize("output", ["state", "linear"])
def test_discrete_adjoint_matches_autodiff(grid, tw, kind, output):
    module = _module(grid, tw, kind, output)
    assert_close(module.gradient(), autodiff_grad(module), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("grid", ["two_runs", "three_runs"])
def test_per_interval_fallback_matches_dense(grid):
    a = _module(grid, "trapezoid", "poly_forced", "state")
    b = _module(grid, "trapezoid", "poly_forced", "state")
    b.max_dense_bytes = 0          # re-integrate every interval in the backward sweep
    assert_close(b.gradient(), a.gradient(), rtol=1e-11, atol=1e-13)
    assert abs(float(a()) - float(b())) <= 1e-13 * abs(float(a()))


@pytest.mark.parametrize("grid", ["two_runs", "three_runs"])
def test_continuous_adjoint_converges(grid):
    cont = _module(grid, "trapezoid", "gas", "state", adjoint="continuous", n_substeps=25)
    disc = _module(grid, "trapezoid", "gas", "state", adjoint="discrete", n_substeps=25)
    assert_close(cont.gradient(), disc.gradient(), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("grid", list(GRIDS))
def test_forward_is_per_interval_rk4(grid):
    """Linear model dz/dt = A z: z_k = P(h_k A)^n z_{k-1}, P the RK4 propagator."""
    t = GRIDS[grid]
    n = 3
    g = torch.Generator().manual_seed(5)
    A = 0.8 * torch.randn(R, R, generator=g, dtype=DTYPE)
    model = PolynomialModel(R, [1], dtype=DTYPE, tensors=[A])
    Phi = random_basis(N, R, seed=1)
    module = NitromModule(_data(t, None, False), ParamRegistry(model, LinearProjection([Phi, Phi])),
                          fom=StateFOM(), n_substeps=n, time_stepper="rk4")
    X = module.training_data.X
    z = (Phi.T @ X[:, :, 0].T)                       # (r, ntraj)
    I = torch.eye(R, dtype=DTYPE)
    J = 0.0
    for k in range(len(t)):
        if k > 0:
            hA = (t[k] - t[k - 1]) / n * A
            P = I + hA + hA @ hA / 2 + hA @ hA @ hA / 6 + hA @ hA @ hA @ hA / 24
            z = torch.linalg.matrix_power(P, n) @ z
        e = X[:, :, k].T - Phi @ z                   # (N, ntraj)
        J = J + ((e * e).sum(axis=0) / module.weights).sum()
    assert abs(float(module()) - float(J)) <= 1e-12 * abs(float(J))


def test_unit_time_weights_change_nothing():
    a = _module("two_runs", None, "poly_forced", "state")
    b = _module("two_runs", None, "poly_forced", "state")
    b.training_data.time_weights = torch.ones(len(GRIDS["two_runs"]), dtype=DTYPE)
    assert abs(float(a()) - float(b())) <= 1e-14 * abs(float(a()))
    assert_close(b.gradient(), a.gradient(), rtol=1e-13, atol=1e-15)
