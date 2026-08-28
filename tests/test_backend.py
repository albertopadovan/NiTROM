"""Backend shim + cross-backend agreement (numpy must match torch to ~eps).

For each ported class we build the *same* object (identical parameter values)
under both backends, run its methods, and assert the numpy and torch outputs
agree to machine precision.  An autouse fixture restores the default (torch)
backend so these tests never leak the active backend into the rest of the suite.
"""

from functools import partial

import numpy as np
import pytest
import torch

from nitrom.backend import get_backend, set_backend
from nitrom.latent_space_models.atr_polynomial_model import AtrPolynomialModel
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.projections.linear_projection import LinearProjection
from nitrom.projections.polynomial_projection import PolynomialProjection

N, R, MB = 5, 2, 4  # ambient, latent, batch


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    set_backend("torch")


def _cast(backend, a):
    return (
        torch.as_tensor(a, dtype=torch.float64)
        if backend == "torch"
        else np.asarray(a, dtype=np.float64)
    )


def _force(backend):
    if backend == "torch":
        return [
            (lambda t, j=j: torch.tensor([0.5 * (j + 1)], dtype=torch.float64))
            for j in range(MB)
        ]
    return [(lambda t, j=j: np.array([0.5 * (j + 1)])) for j in range(MB)]


def _to_np(x):
    if isinstance(x, dict):
        return {k: _to_np(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_to_np(e) for e in x]
    return x.detach().numpy() if hasattr(x, "detach") else np.asarray(x)


def _assert_agree(run, tol=1e-10):
    """Run a backend-parametrized callable under both backends; compare outputs."""
    set_backend("torch")
    t_out = _to_np(run("torch"))
    set_backend("numpy")
    n_out = _to_np(run("numpy"))

    def maxdiff(a, b):
        if isinstance(a, list):
            return max(maxdiff(x, y) for x, y in zip(a, b, strict=True))
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

    assert set(t_out) == set(n_out)
    for key in t_out:
        d = maxdiff(t_out[key], n_out[key])
        assert d < tol, f"{key}: numpy vs torch max|diff| = {d:.2e}"


# ---------------------------------------------------------------------------
# Backend shim basics
# ---------------------------------------------------------------------------

def test_set_get_backend():
    assert set_backend("numpy").is_numpy
    assert get_backend().name == "numpy"
    assert set_backend("torch").is_torch


def test_invalid_backend():
    with pytest.raises(ValueError, match="must be 'numpy' or 'torch'"):
        set_backend("jax")


def test_array_types():
    assert isinstance(set_backend("numpy").zeros((2, 2)), np.ndarray)
    assert torch.is_tensor(set_backend("torch").zeros((2, 2)))


# ---------------------------------------------------------------------------
# Cross-backend agreement
# ---------------------------------------------------------------------------

def test_polynomial_model_cross_backend():
    rng = np.random.default_rng(0)
    A1, A2, Bop = (
        rng.standard_normal((R, R)),
        0.3 * rng.standard_normal((R, R, R)),
        rng.standard_normal((R, 1)),
    )
    z, v = rng.standard_normal(R), rng.standard_normal(R)
    Z, V = rng.standard_normal((MB, R)), rng.standard_normal((MB, R))

    def run(b):
        c, f = partial(_cast, b), _force(b)
        m = PolynomialModel(
            R, [1, 2], tensors=[c(A1), c(A2), c(Bop)],
            forcing_config={"forcing_exists": True, "m": 1},
        )
        zc, vc, Zc, Vc = c(z), c(v), c(Z), c(V)
        return {
            "rhs_vec": m.evaluate_rhs(0.0, zc, external_forcing=[f[0]]),
            "rhs_batch": m.evaluate_rhs(0.0, Zc, external_forcing=f),
            "adj_vec": m.evaluate_adjoint_rhs(0.0, zc, zc),
            "adj_batch": m.evaluate_adjoint_rhs(0.0, Vc, Zc),
            "vjp_vec": m.vjp_evaluate_rhs(zc, vc, external_forcing=[f[0]], t=0.0),
            "vjp_batch": m.vjp_evaluate_rhs(Zc, Vc, external_forcing=f, t=0.0),
        }

    _assert_agree(run)


def test_gas_polynomial_model_cross_backend():
    rng = np.random.default_rng(1)
    eye = np.eye(R)
    K, Rm, Q = (
        rng.standard_normal((R, R)),
        eye + 0.1 * rng.standard_normal((R, R)),
        eye + 0.1 * rng.standard_normal((R, R)),
    )
    S, Bop = 0.3 * rng.standard_normal((R, R, R)), rng.standard_normal((R, 1))
    z, v = rng.standard_normal(R), rng.standard_normal(R)
    Z, V = rng.standard_normal((MB, R)), rng.standard_normal((MB, R))

    def run(b):
        c, f = partial(_cast, b), _force(b)
        m = GasPolynomialModel(
            R, [1, 2], gas_params=[c(K), c(Rm), c(Q), c(S), c(Bop)],
            forcing_config={"forcing_exists": True, "m": 1},
        )
        zc, vc, Zc, Vc = c(z), c(v), c(Z), c(V)
        return {
            "assembled": m.assemble_gas_tensors(),
            "rhs_vec": m.evaluate_rhs(0.0, zc, external_forcing=[f[0]]),
            "rhs_batch": m.evaluate_rhs(0.0, Zc, external_forcing=f),
            "vjp_vec": m.vjp_evaluate_rhs(zc, vc, external_forcing=[f[0]], t=0.0),
            "vjp_batch": m.vjp_evaluate_rhs(Zc, Vc, external_forcing=f, t=0.0),
        }

    _assert_agree(run)


def test_gas_retraction_cross_backend():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((R, R))
    H = 0.3 * rng.standard_normal((R, R, R))

    def run(b):
        c = partial(_cast, b)
        m = GasPolynomialModel(R, [1, 2])
        m.retract_general_tensors_to_gas_tensors([c(A), c(H)])
        return {name: getattr(m, name) for name in ["K", "R", "Q", "S"]}

    # The retraction chains eigvals -> Lyapunov -> cholesky -> inverse through
    # two different LAPACK stacks, so cross-backend agreement (~1e-8) sits at
    # the retraction's own reconstruction precision (1e-6) rather than at eps.
    _assert_agree(run, tol=1e-6)


def test_atr_polynomial_model_cross_backend():
    rng = np.random.default_rng(11)
    eye = np.eye(R)
    K, Rm, Q = (
        rng.standard_normal((R, R)),
        eye + 0.1 * rng.standard_normal((R, R)),
        eye + 0.1 * rng.standard_normal((R, R)),
    )
    S, Bop = 0.3 * rng.standard_normal((R, R, R)), rng.standard_normal((R, 1))
    Bhat, shift = rng.standard_normal(R), 0.5 * rng.standard_normal(R)
    z, v = rng.standard_normal(R), rng.standard_normal(R)
    Z, V = rng.standard_normal((MB, R)), rng.standard_normal((MB, R))

    def run(b):
        c, f = partial(_cast, b), _force(b)
        m = AtrPolynomialModel(
            R, [1, 2],
            atr_params=[c(K), c(Rm), c(Q), c(S), c(Bhat), c(shift), c(Bop)],
            forcing_config={"forcing_exists": True, "m": 1},
        )
        zc, vc, Zc, Vc = c(z), c(v), c(Z), c(V)
        return {
            "assembled": m.assemble_gas_tensors(),
            "radius": np.asarray(m.trapping_region_radius()),
            "rhs_vec": m.evaluate_rhs(0.0, zc, external_forcing=[f[0]]),
            "rhs_batch": m.evaluate_rhs(0.0, Zc, external_forcing=f),
            "vjp_vec": m.vjp_evaluate_rhs(zc, vc, external_forcing=[f[0]], t=0.0),
            "vjp_batch": m.vjp_evaluate_rhs(Zc, Vc, external_forcing=f, t=0.0),
        }

    _assert_agree(run)


def test_atr_retraction_cross_backend():
    rng = np.random.default_rng(12)
    A = rng.standard_normal((R, R))
    H = 0.3 * rng.standard_normal((R, R, R))
    cvec = rng.standard_normal(R)
    shift = 0.5 * rng.standard_normal(R)

    def run(b):
        c = partial(_cast, b)
        m = AtrPolynomialModel(R, [1, 2])
        m.retract_general_tensors_to_atr_tensors(
            [c(cvec), c(A), c(H)], m=c(shift)
        )
        return {name: getattr(m, name) for name in ["K", "R", "Q", "S", "Bhat", "m"]}

    # As for the GAS retraction, the eigvals -> Lyapunov -> cholesky -> inverse
    # chain limits cross-backend agreement to the retraction's own precision.
    _assert_agree(run, tol=1e-6)


def test_linear_projection_cross_backend():
    rng = np.random.default_rng(3)
    Phi, Psi = rng.standard_normal((N, R)), rng.standard_normal((N, R))
    qv, qb = rng.standard_normal(N), rng.standard_normal((MB, N))
    zv, zb = rng.standard_normal(R), rng.standard_normal((MB, R))
    vr, vrb = rng.standard_normal(R), rng.standard_normal((MB, R))
    vN, vNb = rng.standard_normal(N), rng.standard_normal((MB, N))

    def run(b):
        c = partial(_cast, b)
        p = LinearProjection([c(Phi), c(Psi)])
        return {
            "enc_v": p.encode(c(qv)), "enc_b": p.encode(c(qb)),
            "dec_v": p.decode(c(zv)), "dec_b": p.decode(c(zb)),
            "venc_v": p.vjp_encode(c(qv), c(vr)),
            "venc_b": p.vjp_encode(c(qb), c(vrb)),
            "vdec_v": p.vjp_decode(c(zv), c(vN)),
            "vdec_b": p.vjp_decode(c(zb), c(vNb)),
            "vdecs_v": p.vjp_decode_state(c(zv), c(vN)),
            "vdecs_b": p.vjp_decode_state(c(zb), c(vNb)),
        }

    _assert_agree(run)


def test_polynomial_projection_cross_backend():
    rng = np.random.default_rng(4)
    Phi, Psi = rng.standard_normal((N, R)), rng.standard_normal((N, R))
    A2 = 0.2 * rng.standard_normal((N, R, R))
    A3 = 0.2 * rng.standard_normal((N, R, R, R))
    qv, qb = rng.standard_normal(N), rng.standard_normal((MB, N))
    zv, zb = rng.standard_normal(R), rng.standard_normal((MB, R))
    vr, vrb = rng.standard_normal(R), rng.standard_normal((MB, R))
    vN, vNb = rng.standard_normal(N), rng.standard_normal((MB, N))

    def run(b):
        c = partial(_cast, b)
        p = PolynomialProjection([2, 3], [c(Phi), c(Psi), c(A2), c(A3)])
        return {
            "enc_v": p.encode(c(qv)), "enc_b": p.encode(c(qb)),
            "dec_v": p.decode(c(zv)), "dec_b": p.decode(c(zb)),
            "venc_v": p.vjp_encode(c(qv), c(vr)),
            "venc_b": p.vjp_encode(c(qb), c(vrb)),
            "vdec_v": p.vjp_decode(c(zv), c(vN)),
            "vdec_b": p.vjp_decode(c(zb), c(vNb)),
            "vdecs_v": p.vjp_decode_state(c(zv), c(vN)),
            "vdecs_b": p.vjp_decode_state(c(zb), c(vNb)),
        }

    _assert_agree(run)


# ---------------------------------------------------------------------------
# Inference modules end-to-end (forward + gradient) run on numpy and torch
# ---------------------------------------------------------------------------

class _Data:
    def __init__(self, X, dX, weights, time, forcing_fns):
        self.X, self.dX = X, dX
        self.weights, self.time = weights, time
        self.forcing_fns = forcing_fns


class _FOM:
    def __init__(self, C):
        self.C = C

    def compute_output(self, q):
        return self.C @ q

    def compute_output_derivative(self, q):
        return self.C


def test_opinf_module_cross_backend():
    from nitrom.latent_space_models.polynomial_model import PolynomialModel
    from nitrom.optimization import OpInfModule
    from nitrom.projections.linear_projection import LinearProjection

    rng = np.random.default_rng(5)
    ntraj, nt = 2, 4
    X = rng.standard_normal((ntraj, N, nt))
    dX = rng.standard_normal((ntraj, N, nt))
    weights = 1.0 + rng.random(ntraj)
    time = np.linspace(0.0, 1.0, nt)
    Phi = np.linalg.qr(rng.standard_normal((N, R)))[0]
    A1, A2 = rng.standard_normal((R, R)), 0.3 * rng.standard_normal((R, R, R))

    def run(b):
        c = partial(_cast, b)
        data = _Data(c(X), c(dX), c(weights), c(time), [])
        rom = PolynomialModel(R, [1, 2], tensors=[c(A1), c(A2)])
        proj = LinearProjection([c(Phi), c(Phi)])
        m = OpInfModule(data, rom, proj, reg=0.01)
        return {"cost": m(), "grad": m.gradient()}

    _assert_agree(run)


def test_nitrom_module_cross_backend():
    from nitrom.latent_space_models.polynomial_model import PolynomialModel
    from nitrom.optimization import NitromModule
    from nitrom.projections.linear_projection import LinearProjection
    from nitrom.roms.param_registry import ParamRegistry

    rng = np.random.default_rng(6)
    ntraj, nt, no = 2, 4, 3
    X = rng.standard_normal((ntraj, N, nt))
    weights = 1.0 + rng.random(ntraj)
    time = np.linspace(0.0, 1.0, nt)
    Phi = np.linalg.qr(rng.standard_normal((N, R)))[0]
    Psi = np.linalg.qr(rng.standard_normal((N, R)))[0]
    A1 = 0.3 * rng.standard_normal((R, R))
    A2 = 0.2 * rng.standard_normal((R, R, R))
    Bop = 0.3 * rng.standard_normal((R, 1))
    Cmat = rng.standard_normal((no, N))

    def run(b):
        c = partial(_cast, b)
        forcing = [(lambda t, j=j: c(np.array([0.5 * (j + 1)]))) for j in range(ntraj)]
        data = _Data(c(X), None, c(weights), c(time), forcing)
        rom = PolynomialModel(
            R, [1, 2], tensors=[c(A1), c(A2), c(Bop)],
            forcing_config={"forcing_exists": True, "m": 1},
        )
        proj = LinearProjection([c(Phi), c(Psi)])
        registry = ParamRegistry(rom, proj)
        m = NitromModule(
            data, registry, fom=_FOM(c(Cmat)), reg=0.01,
            n_substeps=50, n_leggauss=5,
        )
        return {"cost": m(), "grad": m.gradient()}

    _assert_agree(run, tol=1e-9)


# ---------------------------------------------------------------------------
# Training on the numpy backend (manual Adam/SGD + scipy L-BFGS + manifolds)
# ---------------------------------------------------------------------------

from nitrom.optimization.modules.base import InferenceModule  # noqa: E402


class _QuadModule(InferenceModule):
    """f = ||Phi||^2 + ||Psi||^2 (constant r on the manifold); tests retraction."""

    def __init__(self, Phi, Psi):
        super().__init__()
        self.register_parameter("Phi", Phi)
        self.register_parameter("Psi", Psi)

    def forward(self):
        return (self.Phi**2).sum() + (self.Psi**2).sum()

    def gradient(self):
        return [2.0 * self.Phi, 2.0 * self.Psi]


def test_train_numpy_manifold_retraction():
    from nitrom.optimization import train

    set_backend("numpy")
    rng = np.random.default_rng(7)
    m = _QuadModule(rng.standard_normal((N, R)), rng.standard_normal((N, R)))
    m.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])
    train(m, n_epochs=5, lr=0.05, optimizer_type="sgd", print_every=0)
    eye = np.eye(R)
    assert np.allclose(m.Phi.T @ m.Phi, eye, atol=1e-10)
    assert np.allclose(m.Psi.T @ m.Psi, eye, atol=1e-10)


def _numpy_opinf():
    from nitrom.latent_space_models.polynomial_model import PolynomialModel
    from nitrom.optimization import OpInfModule
    from nitrom.projections.linear_projection import LinearProjection

    rng = np.random.default_rng(8)
    ntraj, nt = 3, 6
    X = rng.standard_normal((ntraj, N, nt))
    dX = rng.standard_normal((ntraj, N, nt))
    weights = 1.0 + rng.random(ntraj)
    Phi = np.linalg.qr(rng.standard_normal((N, R)))[0]
    data = _Data(X, dX, weights, np.linspace(0.0, 1.0, nt), [])
    rom = PolynomialModel(R, [1, 2], tensors=[np.zeros((R, R)), np.zeros((R, R, R))])
    return OpInfModule(data, rom, LinearProjection([Phi, Phi]), reg=1e-8)


@pytest.mark.parametrize("optimizer_type", ["adam", "sgd", "lbfgs"])
def test_train_numpy_opinf_decreases(optimizer_type):
    from nitrom.optimization import train

    set_backend("numpy")
    m = _numpy_opinf()
    l0 = float(m())
    lr = {"adam": 0.05, "sgd": 1e-3, "lbfgs": 1.0}[optimizer_type]
    train(m, n_epochs=60, lr=lr, optimizer_type=optimizer_type, print_every=0, tol=1e-14)
    assert float(m()) < l0
