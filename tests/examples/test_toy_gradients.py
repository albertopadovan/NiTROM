"""The toy's actual forced output problem supports both NiTROM adjoints."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nitrom.latent_space_models import GasPolynomialModel, PolynomialModel
from nitrom.optimization import NitromModule
from nitrom.projections import LinearProjection
from nitrom.roms import ParamRegistry
from nitrom.time_steppers import solve_ivp

PATH = Path(__file__).resolve().parents[2] / "examples" / "toymodel" / "model.py"
spec = importlib.util.spec_from_file_location("toy_physics", PATH)
physics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(physics)


@pytest.mark.parametrize("gas", [False, True])
@pytest.mark.parametrize("adjoint", ["discrete", "continuous"])
def test_toy_output_and_basis_gradients(numpy_backend, gas, adjoint):
    fom = physics.ToyModel()
    time = np.linspace(0, 0.2, 5)
    amplitudes = np.array([0.01, 0.1])
    X = solve_ivp(
        fom.rhs,
        np.zeros((2, 3)),
        0,
        time[-1],
        0.001,
        time,
        "rk4",
        forcing=amplitudes[:, None] * fom.B[:, 0],
    )
    phi = np.linalg.svd(np.concatenate(X, axis=1), full_matrices=False)[0][:, :2]
    tensors = fom.project(phi, phi)
    forcing = {"forcing_exists": True, "m": 1, "B": tensors[-1]}
    if gas:
        seed = GasPolynomialModel(2, [1, 2])
        seed.retract_general_tensors_to_gas_tensors(tensors[:2])
        model = GasPolynomialModel(
            2,
            [1, 2],
            gas_params=[*seed.get_params(), tensors[-1]],
            forcing_config=forcing,
        )
    else:
        model = PolynomialModel(2, [1, 2], tensors=tensors, forcing_config=forcing)
    data = SimpleNamespace(
        X=X,
        time=time,
        weights=fom.steady_weights(amplitudes) * 10,
        forcing_fns=[lambda t, b=b: np.array([b]) for b in amplitudes],
    )
    registry = ParamRegistry(model, LinearProjection([phi.copy(), phi.copy()]))
    module = NitromModule(
        data, registry, fom=fom, n_substeps=15, adjoint_method=adjoint
    )
    module.set_unlearnable("B")
    gradients = dict(zip(registry.names, module.gradient(), strict=True))
    names = ["K" if gas else "A_1", "Phi", "Psi"]
    for name in names:
        original = getattr(module, name).copy()
        idx = (0, 1) if name == "K" else (0, 0)
        try:
            plus = original.copy()
            plus[idx] += 1e-6
            module.register_parameter(name, plus)
            high = float(module())
            minus = original.copy()
            minus[idx] -= 1e-6
            module.register_parameter(name, minus)
            low = float(module())
        finally:
            module.register_parameter(name, original)
        numeric = (high - low) / 2e-6
        np.testing.assert_allclose(
            gradients[name][idx],
            numeric,
            rtol=1e-3 if adjoint == "continuous" else 1e-4,
            atol=1e-6,
        )
