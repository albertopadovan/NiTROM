"""Lorenz-96 physics, independent POD analysis, and checkpointed workflows."""

import importlib
import json
import os
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp as scipy_solve_ivp

from nitrom import AtrPolynomialModel, PolynomialModel, load_checkpoint, save_checkpoint
from nitrom.backend import get_backend
from nitrom.checkpointing import Checkpoint
from nitrom.projections import LinearProjection
from nitrom.time_steppers import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "examples" / "lorenz96"
KINDS = ("opinf", "atr_opinf", "nitrom", "atr_nitrom")


@pytest.fixture
def example(monkeypatch):
    names = ("settings", "model", "_utils", "pod_energy", "train_models")
    previous = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(FOLDER))
    try:
        yield {name: importlib.import_module(name) for name in names}
    finally:
        for name, module in previous.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module


def run(script, directory, *options, success=True):
    result = subprocess.run(
        [
            sys.executable,
            str(FOLDER / script),
            "--smoke",
            "--output-dir",
            str(directory),
            *options,
        ],
        cwd=directory.parent,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MPLCONFIGDIR": str(directory.parent / "mpl-cache"),
            "XDG_CACHE_HOME": str(directory.parent / "cache"),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    if success:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
    return result


def test_rhs_periodicity_forcing_and_energy(example):
    fom = example["model"].Lorenz96(n=8)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, 8))
    u = np.array([-2.0, 0.0, 2.0])
    result = fom.rhs(0, X, u[:, None])
    expected = np.array(
        [
            [
                x[(i - 1) % 8] * (x[(i + 1) % 8] - x[(i - 2) % 8]) - x[i] + 8 + a
                for i in range(8)
            ]
            for x, a in zip(X, u, strict=True)
        ]
    )
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(
        fom.rhs(0, np.roll(X, 1, axis=1), u[:, None]), np.roll(result, 1, axis=1)
    )
    np.testing.assert_allclose(
        np.sum(X * result, axis=1),
        -np.sum(X**2, axis=1) + (8 + u) * X.sum(axis=1),
    )
    np.testing.assert_allclose(fom.rhs(0, X[0], -2), result[0])
    functions = example["model"].input_functions([-1.5, 2], sinusoidal=[False, True])
    np.testing.assert_allclose(functions[0](0.4), [-1.5])
    np.testing.assert_allclose(functions[1](1.25), [2])


def test_reference_rk4_convergence(example, numpy_backend):
    fom = example["model"].Lorenz96(n=8)
    initial = 8 + np.random.default_rng(3).normal(size=8)
    time = np.linspace(0, 0.2, 11)
    reference = scipy_solve_ivp(
        lambda t, x: fom.rhs(t, x, 0.5),
        (0, time[-1]),
        initial,
        method="DOP853",
        t_eval=time,
        rtol=1e-12,
        atol=1e-13,
    ).y
    errors = []
    for dt in (0.01, 0.005, 0.0025):
        X = solve_ivp(fom.rhs, initial, 0, time[-1], dt, time, forcing=0.5)
        errors.append(np.linalg.norm(X - reference))
    assert errors[0] / errors[1] > 10
    assert errors[1] / errors[2] > 10
    assert errors[-1] / np.linalg.norm(reference) < 1e-6


def test_pod_energy_boundaries_and_zero_energy(example):
    pod = example["pod_energy"]
    X = np.array([[[3.0, -3, 0, 0], [0, 0, 1, -1], [0, 0, 0, 0]]])
    singular, energy = pod.energy_spectrum(X, np.zeros(3))
    np.testing.assert_allclose(singular**2, [18, 2, 0])
    np.testing.assert_allclose(energy, [0.9, 1, 1])
    assert pod.rank_for_energy(energy, energy[0]) == 1
    assert pod.rank_for_energy(energy, np.nextafter(energy[0], 1)) == 2
    assert pod.rank_for_energy(energy, 1) == 2
    for target in (0, -1, 1.01, np.inf, np.nan):
        with pytest.raises(ValueError, match="fractions"):
            pod.rank_for_energy(energy, target)
    with pytest.raises(ValueError, match="nonzero"):
        pod.energy_spectrum(np.zeros((2, 3, 4)), np.zeros(3))


def test_required_rank_and_profile_paths(example, tmp_path):
    arguments = example["settings"].arguments
    for options in ([], ["--r", "0"], ["--r", "41"]):
        with pytest.raises(SystemExit):
            arguments("test", options, training=True)
    settings, _, _ = arguments("test", ["--r", "10"], training=True)
    assert settings.r == 10
    _, full, _ = arguments("test", [])
    settings, smoke, _ = arguments("test", ["--smoke"])
    assert settings.r is None
    assert full == FOLDER / "outputs" / "full"
    assert smoke == FOLDER / "outputs" / "smoke"
    previous = get_backend()
    for path in FOLDER.glob("*.py"):
        runpy.run_path(str(path), run_name="import_check")
    assert get_backend() is previous


def test_centered_forced_atr_nitrom_gradient(example, numpy_backend):
    rng = np.random.default_rng(12)
    fom = example["model"].Lorenz96(n=6)
    time = np.linspace(0, 0.08, 5)
    amplitudes = np.array([-1.0, 1.0])
    physical = solve_ivp(
        fom.rhs,
        8 + rng.normal(size=(2, 6)),
        0,
        time[-1],
        0.001,
        time,
        forcing=amplitudes[:, None],
    )
    mean = physical.mean(axis=(0, 2))
    X = physical - mean[None, :, None]
    projection = LinearProjection([np.eye(6), np.eye(6)])
    np.testing.assert_allclose(
        projection.decode(projection.encode(X[0].T)).T + mean[:, None], physical[0]
    )
    r = 3
    phi = np.linalg.qr(rng.normal(size=(6, r)))[0]
    params = [
        0.1 * rng.normal(size=(r, r)),
        np.eye(r),
        np.eye(r),
        0.01 * rng.normal(size=(r, r, r)),
        rng.normal(size=r),
        0.1 * rng.normal(size=r),
        rng.normal(size=(r, 1)),
    ]
    model = AtrPolynomialModel(
        r, [1, 2], atr_params=params, forcing_config={"forcing_exists": True, "m": 1}
    )
    saved = Checkpoint(model, LinearProjection([phi.copy(), phi.copy()]))
    data = example["_utils"].training_view(X, np.zeros_like(X), time, amplitudes)
    settings = replace(example["settings"].Settings(), n=6, r=r)
    module = example["train_models"].make_nitrom(settings, data, saved)
    gradients = dict(zip(module.param_names, module.gradient(), strict=True))
    for name in module.param_names:
        original = getattr(module, name).copy()
        direction = rng.normal(size=original.shape)
        direction /= np.linalg.norm(direction)
        try:
            module.register_parameter(name, original + 1e-6 * direction)
            high = float(module())
            module.register_parameter(name, original - 1e-6 * direction)
            low = float(module())
        finally:
            module.register_parameter(name, original)
        np.testing.assert_allclose(
            np.sum(gradients[name] * direction),
            (high - low) / 2e-6,
            rtol=5e-5,
            atol=2e-7,
            err_msg=name,
        )


def test_input_aware_atr_bound_and_lyapunov_identity(example, numpy_backend):
    rng = np.random.default_rng(5)
    r = 3
    params = [
        rng.normal(size=(r, r)),
        np.diag([1.0, 1.2, 1.4]),
        np.diag([0.8, 1, 1.2]),
        0.1 * rng.normal(size=(r, r, r)),
        rng.normal(size=r),
        rng.normal(size=r),
        rng.normal(size=(r, 1)),
    ]
    model = AtrPolynomialModel(
        r, [1, 2], atr_params=params, forcing_config={"forcing_exists": True, "m": 1}
    )
    time = np.arange(3.0)
    amplitudes = np.linspace(-2, 2, 9)
    inputs = example["model"].input_functions(amplitudes)
    Z = rng.normal(size=(len(inputs), r, len(time)))
    diagnostics = example["_utils"].atr_diagnostics
    bound, _, _ = diagnostics(model, Z, time, inputs, 2)
    directions = rng.normal(size=Z.shape)
    directions /= np.linalg.norm(directions, axis=1)[:, None, :]
    Z = model.m[None, :, None] + 2 * bound["decrease_radius"] * directions
    bound, V, derivative = diagnostics(model, Z, time, inputs, 2)
    Q = np.array(bound["ellipsoid_metric"])
    y = Q @ (Z - model.m[None, :, None])
    # The model's forcing API takes callable inputs; evaluate each trajectory directly.
    rhs = np.array(
        [
            model.evaluate_rhs(0, z.T, external_forcing=[fn] * len(time)).T
            for z, fn in zip(Z, inputs, strict=True)
        ]
    )
    np.testing.assert_allclose(
        derivative, np.sum(y * rhs, axis=1), rtol=1e-12, atol=1e-10
    )
    np.testing.assert_allclose(
        V, 0.5 * np.sum((Z - model.m[None, :, None]) * y, axis=1)
    )
    assert bound["outside_samples"] == len(inputs) * len(time)
    assert bound["derivative_violations"] == 0
    assert np.all(derivative < 0)
    expected = max(
        np.linalg.norm(model.Bhat + sign * 2 * model.B[:, 0]) for sign in (-1, 1)
    )
    assert bound["effective_forcing_bound"] == pytest.approx(expected)


def test_statistics_failure_and_histogram_tails(example):
    statistics = example["_utils"].state_statistics
    assert statistics(np.full((3, 5), np.nan), np.linspace(-1, 1, 5), 2) is None
    stat = statistics(np.full((3, 5), 2.0), np.linspace(-1, 1, 5), 2)
    assert stat["histogram_outside_fraction"] == 1.0
    assert np.isnan(stat["autocorrelation"]).all()


def test_failed_rollout_stays_unavailable(example, numpy_backend):
    model = PolynomialModel(1, [0], tensors=[np.array([1e8])])
    saved = Checkpoint(model, LinearProjection([np.eye(1), np.eye(1)]))
    Z, prediction = example["_utils"].rollout(
        saved,
        np.zeros((1, 1)),
        np.zeros(1),
        np.linspace(0, 1, 11),
        [lambda t: np.zeros(1)],
        0.001,
    )
    valid = np.isfinite(prediction).all(axis=1)[0]
    assert valid[0] and not valid[-1]
    first_failure = np.flatnonzero(~valid)[0]
    assert np.isnan(prediction[:, :, first_failure:]).all()
    assert np.isnan(Z[:, :, first_failure:]).all()


def test_standalone_pod_and_complete_workflow(tmp_path):
    directory = tmp_path / "lorenz96"
    run("generate_data.py", directory)
    with np.load(directory / "data" / "training.npz") as data:
        assert data["X"].shape == (8, 40, 101)
        np.testing.assert_allclose(data["mean"], data["X"].mean(axis=(0, 2)))
        assert np.all(data["time"][:1] == 0)
        assert len(np.unique(data["X"][:, :, 0], axis=0)) == 8
        training_mean = data["mean"]
    # Analysis needs no models, and training must work without analysis outputs.
    run("pod_energy.py", directory)
    report = directory / "results" / "pod_energy.json"
    assert json.loads(report.read_text()) == {"targets": []}
    assert not list((directory / "models").iterdir())
    run("pod_energy.py", directory, "--energy", "0.9", "1")
    assert len(json.loads(report.read_text())["targets"]) == 2
    for target in ("0", "1.1", "nan"):
        run("pod_energy.py", directory, "--energy", target, success=False)
    (directory / "data" / "pod_energy.npz").unlink()
    report.unlink()
    for options in ((), ("--r", "0"), ("--r", "41")):
        run("train_models.py", directory, *options, success=False)
    run("train_models.py", directory, "--r", "10", "--fresh")
    run("plot_results.py", directory)
    saved = {
        kind: load_checkpoint(directory / "models" / f"{kind}.npz") for kind in KINDS
    }
    metrics = json.loads((directory / "results" / "metrics.json").read_text())
    assert set(metrics) == set(KINDS)
    for kind, checkpoint in saved.items():
        np.testing.assert_array_equal(checkpoint.metadata["mean"], training_mean)
        assert checkpoint.projection.Phi.shape == (40, 10)
        assert np.isfinite(checkpoint.metadata["final_loss"])
        assert all(np.isfinite(x).all() for x in checkpoint.model.get_params())
        assert all(np.isfinite(x).all() for x in checkpoint.history.values())
        if kind.startswith("atr"):
            assert isinstance(checkpoint.model, AtrPolynomialModel)
            assert metrics[kind]["atr"]["derivative_violations"] == 0
    with np.load(directory / "results" / "predictions.npz") as data:
        for kind in KINDS:
            assert np.isfinite(data[f"error_{kind}"][:, 0]).all()
            failed = ~np.isfinite(data[f"long_prediction_{kind}"]).all(axis=(1, 2))
            assert failed.sum() == metrics[kind]["long_failures"]
            for case, failure in enumerate(failed[:3]):
                assert (metrics[kind]["constant_statistics"][case] is None) == failure
    figures = list((directory / "figures").glob("*.pdf"))
    assert len(figures) == 5
    assert all(path.stat().st_size > 1000 for path in figures)

    run("train_models.py", directory, "--r", "10")
    for kind in ("nitrom", "atr_nitrom"):
        current = load_checkpoint(directory / "models" / f"{kind}.npz")
        assert current.metadata["runs"] == 2
        assert len(current.history["loss"]) > len(saved[kind].history["loss"])
        np.testing.assert_allclose(
            current.metadata["initial_loss"],
            saved[kind].metadata["final_loss"],
            rtol=1e-10,
            atol=1e-12,
        )

    baseline = directory / "models" / "opinf.npz"
    before = baseline.read_bytes()
    result = run("train_models.py", directory, "--r", "9", success=False)
    assert "Incompatible checkpoint" in result.stderr
    assert before == baseline.read_bytes()
    path = directory / "models" / "atr_nitrom.npz"
    incompatible = load_checkpoint(path)
    incompatible.metadata["mean"] += 1
    save_checkpoint(path, incompatible)
    result = run("train_models.py", directory, "--r", "10", success=False)
    assert "Incompatible checkpoint" in result.stderr
    assert before == baseline.read_bytes()
    run("train_models.py", directory, "--r", "10", "--fresh")
    for kind in ("nitrom", "atr_nitrom"):
        fresh = load_checkpoint(directory / "models" / f"{kind}.npz")
        assert fresh.metadata["runs"] == 1
        for expected, actual in zip(
            saved[kind].model.get_params(), fresh.model.get_params(), strict=True
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
