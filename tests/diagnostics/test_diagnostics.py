"""On-demand diagnostics on both array backends, including broken setups."""

from types import SimpleNamespace

import numpy as np
import pytest

from nitrom import (
    AtrPolynomialModel,
    GasPolynomialModel,
    LinearProjection,
    NitromModule,
    ParamRegistry,
    PolynomialModel,
    PolynomialProjection,
    check_model_properties,
    check_projection,
    compare_gradients,
    register_model_check,
)


def _array(backend, x, single=False):
    return backend.asarray(x, dtype=backend.float32 if single else backend.float64)


def _model(backend, atr=False, single=False, forcing=False):
    rng = np.random.default_rng(7)
    params = [
        [[0, 0.2], [-0.3, 0]],
        [[1.1, 0.1], [0.1, 1]],
        [[2, 0.1], [0, 1]],
        rng.standard_normal((2, 2, 2)),
    ]
    if atr:
        params.extend([[0.5, -0.2], [1, -2]])
    if forcing:
        params.append([[10], [20]])
    params = [_array(backend, p, single) for p in params]
    kwargs = {
        "dtype": backend.float32 if single else backend.float64,
        "forcing_config": {"forcing_exists": True, "m": 1} if forcing else None,
    }
    if atr:
        return AtrPolynomialModel(2, [1, 2], atr_params=params, **kwargs)
    return GasPolynomialModel(2, [1, 2], gas_params=params, **kwargs)


@pytest.mark.parametrize("polynomial", [False, True])
@pytest.mark.parametrize("single", [False, True])
def test_projection_identity_and_rng_isolation(backend, polynomial, single):
    phi = _array(backend, [[1, 0], [0, 1], [0.2, 0.3]], single)
    psi = _array(backend, [[1, 0], [0, 1], [0, 0]], single)
    projection = (
        PolynomialProjection(
            [2], [phi, psi, _array(backend, np.ones((3, 2, 2)), single)]
        )
        if polynomial
        else LinearProjection([phi, psi])
    )
    np_state = np.random.get_state()
    if backend.is_torch:
        torch_state = backend.xp.random.get_rng_state().clone()
    result = check_projection(projection, n_samples=12)
    assert result["max_percent_error"] < 1e-3
    assert result == check_projection(projection, n_samples=12)
    restored = np.random.get_state()
    assert np_state[0] == restored[0]
    np.testing.assert_array_equal(np_state[1], restored[1])
    assert np_state[2:] == restored[2:]
    if backend.is_torch:
        assert backend.xp.equal(torch_state, backend.xp.random.get_rng_state())


def test_projection_failure_and_supplied_states(backend):
    phi = _array(backend, np.eye(2))
    projection = LinearProjection([phi, phi])
    assert check_projection(projection, z=_array(backend, [0, 0])) == {
        "max_absolute_error": 0.0,
        "max_percent_error": 0.0,
    }
    projection.S = 2 * projection.S
    with pytest.raises(ValueError, match="encode\\(decode"):
        check_projection(projection)
    with pytest.raises(ValueError, match="finite latent"):
        check_projection(projection, z=_array(backend, [np.nan, 0]))
    with pytest.raises(ValueError, match="shape"):
        check_projection(projection, z=_array(backend, [1, 2, 3]))
    with pytest.raises(ValueError, match="positive integer"):
        check_projection(projection, n_samples=0)


@pytest.mark.parametrize("atr", [False, True])
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("forcing", [False, True])
def test_stable_structure(backend, atr, single, forcing):
    model = _model(backend, atr, single, forcing)
    before = [backend.to_numpy(p).copy() for p in model.get_params()]
    inner_before = [backend.to_numpy(p).copy() for p in model.inner_params()]
    result = check_model_properties(model)
    assert result["metric_min_eigenvalue"] > 0
    assert result["dissipation_min_eigenvalue"] > 0
    assert result["quadratic_energy_residual"] < (1e-6 if single else 1e-12)
    if atr:
        np.testing.assert_allclose(
            result["trapping_radius"], model.trapping_region_radius(), rtol=1e-5
        )
        np.testing.assert_allclose(
            result["trapping_containing_radius"],
            model.trapping_region_containing_radius(),
            rtol=1e-5,
        )
    for old, new in zip(before, model.get_params(), strict=True):
        np.testing.assert_array_equal(old, backend.to_numpy(new))
    for old, new in zip(inner_before, model.inner_params(), strict=True):
        np.testing.assert_array_equal(old, backend.to_numpy(new))


@pytest.mark.parametrize("atr", [False, True])
@pytest.mark.parametrize("broken", ["linear", "quadratic", "metric", "nonfinite"])
def test_broken_structure_raises(backend, atr, broken):
    model = _model(backend, atr)
    tensors = [backend.copy(p) for p in model.inner_params()]
    if broken == "linear":
        tensors[model.poly_comp.index(1)] += 100 * _array(backend, np.eye(2))
        expected = "Linear dissipation"
    elif broken == "quadratic":
        tensors[model.poly_comp.index(2)][0, 0, 0] += 0.001
        expected = "conserve"
    elif broken == "metric":
        model.Q = _array(backend, np.zeros((2, 2)))
        expected = "invertible Q"
    else:
        tensors[model.poly_comp.index(1)][0, 0] = float("nan")
        expected = "finite"
    model.model.update_params(tensors)
    with pytest.raises(ValueError, match=expected):
        check_model_properties(model)


def test_linear_gas_and_zero_constant_atr(backend):
    gas = _model(backend)
    linear = GasPolynomialModel(2, [1], gas_params=gas.get_params()[:3])
    assert check_model_properties(linear)["quadratic_energy_residual"] == 0
    atr = _model(backend, atr=True)
    params = [backend.copy(p) for p in atr.get_params()]
    params[-2] *= 0
    atr.update_params(params)
    assert check_model_properties(atr)["trapping_radius"] < 1e-12


def test_property_registration(backend):
    class CustomModel:
        pass

    class ChildModel(CustomModel):
        pass

    with pytest.raises(NotImplementedError, match="No property check"):
        check_model_properties(CustomModel())
    with pytest.raises(NotImplementedError):
        check_model_properties(PolynomialModel(2, [1]))

    @register_model_check(CustomModel)
    def check_custom(model, *, atol, rtol):
        return {"margin": 0.5}

    assert check_model_properties(ChildModel()) == {"margin": 0.5}


def test_actual_nitrom_gradients_and_state_preservation(backend):
    phi = _array(backend, [[1, 0], [0, 1], [0.2, 0.3]])
    projection = LinearProjection([phi, backend.copy(phi)])
    model = PolynomialModel(2, [1], tensors=[_array(backend, [[-1, 0.2], [0, -0.5]])])
    time = _array(backend, np.linspace(0, 0.2, 4))
    data = SimpleNamespace(
        X=_array(backend, np.random.default_rng(0).standard_normal((1, 3, 4))),
        time=time,
        weights=_array(backend, [1]),
        forcing_fns=[],
    )
    fom = SimpleNamespace(
        compute_output=lambda q: q,
        compute_output_derivative=lambda q: _array(backend, np.eye(3)),
        output_is_state=True,
    )
    module = NitromModule(data, ParamRegistry(model, projection), fom=fom, n_substeps=8)
    module.set_unlearnable("Psi")
    before = [backend.to_numpy(p).copy() for p in module.parameters()]
    result = compare_gradients(module)
    assert 0 <= result["overall_percent_error"] < 0.01
    assert set(result["parameter_percent_errors"]) == set(module.param_names)
    assert result["parameter_percent_errors"]["Psi"] == 0
    assert module.adjoint_method == "discrete"
    for old, new in zip(before, module.parameters(), strict=True):
        np.testing.assert_array_equal(old, backend.to_numpy(new))
        if backend.is_torch:
            assert new.grad is None


def test_gradient_percentages_zero_references_and_failure_restore(backend):
    module = SimpleNamespace(
        backend=backend,
        comm=None,
        adjoint_method="continuous",
        named_parameters=lambda: [
            ("a", _array(backend, [0, 0])),
            ("b", _array(backend, [0])),
        ],
    )
    module.gradient = lambda: (
        [_array(backend, [3, 4]), _array(backend, [1])]
        if module.adjoint_method == "continuous"
        else [_array(backend, [0, 4]), _array(backend, [0])]
    )
    result = compare_gradients(module)
    assert result["parameter_percent_errors"] == {"a": 75.0, "b": float("inf")}
    assert result["overall_percent_error"] == pytest.approx(100 * np.sqrt(10) / 4)
    assert module.adjoint_method == "continuous"
    module.gradient = lambda: [_array(backend, [0, 0]), _array(backend, [0])]
    assert compare_gradients(module)["overall_percent_error"] == 0
    module.gradient = lambda: [_array(backend, [np.nan, 0]), _array(backend, [0])]
    assert np.isnan(compare_gradients(module)["overall_percent_error"])

    def failing_gradient():
        if module.adjoint_method == "discrete":
            raise RuntimeError("solver failed")
        return [_array(backend, [0, 0]), _array(backend, [0])]

    module.gradient = failing_gradient
    with pytest.raises(RuntimeError, match="solver failed"):
        compare_gradients(module)
    assert module.adjoint_method == "continuous"


def test_gradient_comparison_reduces_before_taking_norms(backend, monkeypatch):
    # The other rank has continuous=0 and discrete=1: global gradients match,
    # although this rank's continuous=2 and discrete=1 do not match.
    comm = object()
    module = SimpleNamespace(
        backend=backend,
        comm=comm,
        adjoint_method="discrete",
        named_parameters=lambda: [("a", _array(backend, [0]))],
    )
    module.gradient = lambda: [
        _array(backend, [2 if module.adjoint_method == "continuous" else 1])
    ]
    if backend.is_numpy:
        import nitrom.distributed as distributed

        monkeypatch.setattr(distributed, "mpi_rank_size", lambda group: (0, 2))

        def reduce_numpy(gradient, *, comm):
            assert comm is module.comm
            return gradient + (0 if module.adjoint_method == "continuous" else 1)

        monkeypatch.setattr(distributed, "mpi_allreduce_sum", reduce_numpy)
    else:
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_initialized", lambda: True)

        def reduce_torch(gradient, *, op, group):
            assert group is comm
            gradient += 0 if module.adjoint_method == "continuous" else 1

        monkeypatch.setattr(dist, "all_reduce", reduce_torch)
    assert compare_gradients(module)["overall_percent_error"] == 0
