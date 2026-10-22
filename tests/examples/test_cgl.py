"""Numerical contracts shared by the stable and unstable CGL examples."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from tests.examples import _cgl_reference as opinf

from nitrom.latent_space_models import PolynomialModel
from nitrom.optimization import OpInfModule, solve_opinf
from nitrom.projections import LinearProjection

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "cgl"


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cgl = load_example("cgl")


def test_complex_equation_and_snapshot_rhs():
    model = cgl.CGL(nx=32)
    q = np.random.default_rng(3).normal(size=(64, 4))
    z = q[:32] + 1j * q[32:]
    dz = (
        -model.nu * (model.D @ z)
        + model.gamma * (model.DD @ z)
        + model.mu[:, None] * z
        - model.a * abs(z) ** 2 * z
    )
    np.testing.assert_allclose(model.rhs(q), np.concatenate((dz.real, dz.imag)))
    for j in range(q.shape[1]):
        np.testing.assert_allclose(model.rhs(q[:, j]), model.rhs(q)[:, j])


def test_forcing_with_known_linear_solution():
    model = cgl.CGL(nx=16, a=0)
    v = np.random.default_rng(4).normal(size=32)
    # q(t) = t*v solves q' = A*q + (v - t*A*v), q(0) = 0.
    t, Q = cgl.TimeStepper(model, 0.01).integrate(
        np.zeros(32), 0.2, 3, forcing=lambda t: v - t * (model.A @ v)
    )
    np.testing.assert_allclose(Q, v[:, None] * t, atol=2e-15)
    np.testing.assert_allclose(t, np.arange(7) * 0.03)


def test_nonlinear_time_convergence_and_phase_symmetry():
    model = cgl.CGL(nx=64)
    q0 = model.gaussian(-model.xbar)
    q0 /= np.linalg.norm(q0)
    reference = solve_ivp(
        lambda t, q: model.rhs(q),
        (0, 2),
        q0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    ).y[:, -1]
    errors = []
    for dt in (0.04, 0.02, 0.01):
        _, Q = cgl.TimeStepper(model, dt).integrate(q0, 2, 1)
        errors.append(np.linalg.norm(Q[:, -1] - reference))
    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.8

    t, Q = cgl.TimeStepper(model).integrate(q0, 2, 20)
    rotated = np.concatenate((-q0[64:], q0[:64]))
    tr, R = cgl.TimeStepper(model).integrate(rotated, 2, 20)
    np.testing.assert_array_equal(t, tr)
    np.testing.assert_allclose(R, np.concatenate((-Q[64:], Q[:64])), atol=1e-14)


def test_opinf_recovers_amplitude_scaled_cubic_dynamics():
    time = np.linspace(0, 1, 501)
    amps = np.array([-1.0, 0.1, 1.0])
    # Exact solution of x' = -x - x^3 from x(0) = beta.
    trajs = [
        (np.exp(-time) / np.sqrt(1 + a**2 * (1 - np.exp(-2 * time))))[None]
        for a in amps
    ]
    rom = opinf.opinf(
        trajs, np.eye(1), time[1] - time[0], "cubic", amps=amps, C=np.eye(1)
    )
    np.testing.assert_allclose(rom.Ar, [[-1]], atol=1e-8)
    np.testing.assert_allclose(rom.Hr, [[-1]], atol=1e-8)
    np.testing.assert_allclose(
        rom.integrate([1.0], time, c=0.1), trajs[1], rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize(("degree", "kind"), [(2, "quadratic"), (3, "cubic")])
def test_compact_fit_matches_package_tensor_regularization(numpy_backend, degree, kind):
    rng = np.random.default_rng(14)
    time = np.linspace(0.0, 2.0, 61)
    signals = np.array([np.cos(time), np.sin(1.3 * time), np.exp(-time), time**2])
    trajs = [rng.normal(size=(3, 4)) @ signals for _ in range(3)]
    amps = np.array([-1.0, 0.1, 1.0])
    psi = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    C = rng.normal(size=(2, 3))
    reg = 0.7
    compact = opinf.opinf(trajs, psi, time[1] - time[0], kind, reg, amps=amps, C=C)
    physical = np.stack([amp * X for amp, X in zip(amps, trajs, strict=True)])
    derivatives = np.stack(
        [
            amp * opinf.ddt_fd4(X, time[1] - time[0])
            for amp, X in zip(amps, trajs, strict=True)
        ]
    )
    data = SimpleNamespace(
        X=physical[:, :, 2:-2],
        dX=derivatives,
        weights=np.mean(np.sum((C @ physical) ** 2, axis=1), axis=1),
        time=time[2:-2],
        forcing_fns=[],
    )
    model = PolynomialModel(3, [1, degree])
    module = OpInfModule(data, model, LinearProjection([psi, psi]), reg={degree: reg})
    solve_opinf(module, solver="lstsq")
    np.testing.assert_allclose(model.A_1, compact.Ar, rtol=1e-9, atol=1e-10)
    probes = rng.normal(size=(20, 3))
    expected = np.stack([compact.rhs(0.0, z) for z in probes])
    np.testing.assert_allclose(
        model.evaluate_rhs(0.0, probes), expected, rtol=1e-9, atol=1e-10
    )


@pytest.mark.parametrize(("T", "nsave"), [(-1, 1), (0.015, 1), (0.1, 0), (0.1, 1.5)])
def test_invalid_time_grid(T, nsave):
    model = cgl.CGL(nx=8)
    with pytest.raises(ValueError):
        cgl.TimeStepper(model).integrate(np.zeros(16), T, nsave)


def test_full_grid_least_stable_eigenvalue():
    eigenvalues = np.linalg.eigvals(cgl.CGL().A.toarray())
    leading = eigenvalues[np.argmax(eigenvalues.real)]
    assert leading.real < 0
    assert abs(abs(leading.imag) - 0.648) < 1e-3


def test_nonlinearity_limits_transient_growth():
    nonlinear = cgl.CGL(nx=128)
    linear = cgl.CGL(nx=128, a=0.0)
    initial = nonlinear.gaussian(-nonlinear.xbar)
    initial *= 0.1 / np.linalg.norm(initial)
    _, Q = cgl.TimeStepper(nonlinear).integrate(initial, 40.0, 20)
    _, L = cgl.TimeStepper(linear).integrate(initial, 40.0, 20)
    nonlinear_gain = np.linalg.norm(Q, axis=0).max() / 0.1
    linear_gain = np.linalg.norm(L, axis=0).max() / 0.1
    assert 1.0 < nonlinear_gain < linear_gain
