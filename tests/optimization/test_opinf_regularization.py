"""Degree-specific OpInf penalties agree across losses, gradients, and solves."""

from types import SimpleNamespace

import numpy as np
import pytest

from nitrom.latent_space_models import PolynomialModel
from nitrom.optimization import OpInfModule, solve_opinf
from nitrom.projections import LinearProjection


def make_problem(backend, forcing=False):
    rng = np.random.default_rng(31)
    X = backend.asarray(rng.normal(size=(2, 2, 12)))
    data = SimpleNamespace(
        X=X,
        dX=backend.asarray(rng.normal(size=(2, 2, 12))),
        weights=backend.asarray([1.0, 2.0], dtype=backend.float64),
        time=backend.linspace(0.0, 1.0, 12, dtype=backend.float64),
        forcing_fns=[],
    )
    tensors = [backend.asarray(rng.normal(size=(2,) * (d + 1))) for d in [3, 1]]
    config = None
    if forcing:
        config = {"forcing_exists": True, "m": 1}
        tensors.append(backend.asarray(rng.normal(size=(2, 1))))
        data.forcing_fns = [lambda t: backend.asarray([1.0], dtype=backend.float64)] * 2
    model = PolynomialModel(2, [3, 1], tensors=tensors, forcing_config=config)
    phi = backend.eye(2, dtype=backend.float64)
    return data, model, LinearProjection([phi, phi])


def numerical_gradients(module, eps=1e-6):
    """Replace arrays for each perturbation, as NumPy training does."""
    backend = module.backend
    gradients = []
    for name, parameter in module.named_parameters():
        values = backend.to_numpy(parameter).copy()
        gradient = np.zeros_like(values)
        try:
            for idx in np.ndindex(values.shape):
                shifted = values.copy()
                shifted[idx] += eps
                module.register_parameter(name, backend.asarray(shifted))
                plus = float(backend.to_numpy(module()))
                shifted = values.copy()
                shifted[idx] -= eps
                module.register_parameter(name, backend.asarray(shifted))
                minus = float(backend.to_numpy(module()))
                gradient[idx] = (plus - minus) / (2 * eps)
        finally:
            module.register_parameter(name, backend.asarray(values))
        gradients.append(gradient)
    module()
    return gradients


@pytest.mark.parametrize("forcing", [False, True])
def test_selected_degrees_loss_and_gradient(backend, forcing):
    data, model, projection = make_problem(backend, forcing)
    module = OpInfModule(data, model, projection, reg={1: 0.2, 3: 0.7})
    unpenalized = OpInfModule(data, model, projection)
    expected = 0.2 * backend.vector_norm(model.A_1) ** 2
    expected += 0.7 * backend.vector_norm(model.A_3) ** 2
    np.testing.assert_allclose(
        backend.to_numpy(module() - unpenalized()), backend.to_numpy(expected)
    )
    analytic_gradients = [backend.to_numpy(g).copy() for g in module.gradient()]
    for analytic, numerical in zip(
        analytic_gradients, numerical_gradients(module), strict=True
    ):
        np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("reg", [{3: 0.7}, {1: 0.2, 3: 0.7}])
@pytest.mark.parametrize("solver", ["normal", "lstsq"])
def test_solve_matches_augmented_least_squares(backend, reg, solver):
    data, model, projection = make_problem(backend)
    module = OpInfModule(data, model, projection, reg=reg)
    Z = backend.to_numpy(data.X).transpose(1, 0, 2).reshape(2, -1)
    target = backend.to_numpy(data.dX).transpose(1, 0, 2).reshape(2, -1)
    cubic = np.einsum("is,js,ks->ijks", Z, Z, Z).reshape(8, -1)
    design = np.concatenate([cubic, Z]).T
    weights = np.sqrt(backend.to_numpy(module.w))
    penalty = np.diag(np.sqrt([reg.get(3, 0.0)] * 8 + [reg.get(1, 0.0)] * 2))
    expected = np.linalg.lstsq(
        np.concatenate([design * weights[:, None], penalty]),
        np.concatenate([target.T * weights[:, None], np.zeros((10, 2))]),
        rcond=None,
    )[0].T
    solve_opinf(module, solver=solver)
    got = np.concatenate(
        [backend.to_numpy(tensor).reshape(2, -1) for tensor in model.get_params()],
        axis=1,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-10)
    for gradient in module.gradient():
        np.testing.assert_allclose(backend.to_numpy(gradient), 0.0, atol=1e-9)


@pytest.mark.parametrize(
    ("reg", "error"),
    [
        (0.0, TypeError),
        (0.1, TypeError),
        ({"3": 0.1}, TypeError),
        ({True: 0.1}, TypeError),
        ({2: 0.1}, ValueError),
        ({3: -0.1}, ValueError),
        ({3: np.nan}, ValueError),
        ({3: np.inf}, ValueError),
    ],
)
def test_invalid_regularization(numpy_backend, reg, error):
    data, model, projection = make_problem(numpy_backend)
    with pytest.raises(error, match="reg"):
        OpInfModule(data, model, projection, reg=reg)


@pytest.mark.parametrize("solver", ["normal", "lstsq"])
def test_empty_mapping_and_frozen_operator(numpy_backend, solver):
    data, model, projection = make_problem(numpy_backend)
    default = OpInfModule(data, model, projection)
    empty = OpInfModule(data, model, projection, reg={})
    np.testing.assert_allclose(default(), empty())
    original = model.A_3.copy()
    module = OpInfModule(data, model, projection, reg={3: 0.7})
    module.set_unlearnable("A_3")
    solve_opinf(module, solver=solver)
    np.testing.assert_array_equal(model.A_3, original)
    np.testing.assert_array_equal(module.gradient()[0], 0.0)


def test_lstsq_retains_weakly_identified_directions(backend):
    X = backend.asarray(np.diag([1.0, 1e-8])[None])
    data = SimpleNamespace(
        X=X, dX=2.0 * X, weights=backend.asarray(np.ones(1)), forcing_fns=[]
    )
    phi = backend.eye(2, dtype=backend.float64)

    def fit(solver, rcond=None):
        model = PolynomialModel(2, [1])
        module = OpInfModule(data, model, LinearProjection([phi, phi]))
        solve_opinf(module, solver=solver, rcond=rcond)
        return backend.to_numpy(model.A_1)

    np.testing.assert_allclose(fit("lstsq"), 2.0 * np.eye(2), atol=1e-12)
    np.testing.assert_allclose(fit("normal"), np.diag([2.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(fit("lstsq", 1e-6), np.diag([2.0, 0.0]), atol=1e-12)


@pytest.mark.parametrize("zero", [False, True])
def test_lstsq_rank_deficient_underdetermined_system(backend, zero):
    rng = np.random.default_rng(12)
    X = np.zeros((1, 2, 3)) if zero else rng.normal(size=(1, 2, 3))
    dX = rng.normal(size=X.shape)
    data = SimpleNamespace(
        X=backend.asarray(X),
        dX=backend.asarray(dX),
        weights=backend.asarray(np.ones(1)),
        forcing_fns=[],
    )
    phi = backend.eye(2, dtype=backend.float64)
    model = PolynomialModel(2, [1, 3])
    module = OpInfModule(data, model, LinearProjection([phi, phi]))
    solve_opinf(module, solver="lstsq")
    Z = X[0]
    features = np.concatenate(
        (Z, np.einsum("is,js,ks->ijks", Z, Z, Z).reshape(8, -1)), axis=0
    )
    expected = np.linalg.lstsq(features.T, dX[0].T, rcond=None)[0].T
    got = np.concatenate(
        [backend.to_numpy(t).reshape(2, -1) for t in model.get_params()], axis=1
    )
    np.testing.assert_allclose(got, expected, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize(
    "options",
    [
        {"solver": "unknown"},
        {"rcond": 1e-6},
        {"solver": "lstsq", "rcond": -1.0},
        {"solver": "lstsq", "rcond": np.nan},
        {"solver": "lstsq", "rcond": np.inf},
    ],
)
def test_invalid_solver_options(numpy_backend, options):
    data, model, projection = make_problem(numpy_backend)
    with pytest.raises(ValueError):
        solve_opinf(OpInfModule(data, model, projection), **options)
