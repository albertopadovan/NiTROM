"""Small deterministic helpers used by numerical regression tests."""

from types import SimpleNamespace

import numpy as np

from nitrom.latent_space_models import PolynomialModel
from nitrom.projections import LinearProjection


def finite_diff_grad(module, eps=1e-6):
    """Central differences of the scalar loss, restoring parameters on failure."""
    backend = module.backend
    result = []
    with backend.no_grad():
        for parameter in module.parameters():
            gradient = backend.zeros_like(parameter)
            for index in np.ndindex(tuple(parameter.shape)):
                original = parameter[index].item()
                try:
                    parameter[index] = original + eps
                    plus = float(module())
                    parameter[index] = original - eps
                    minus = float(module())
                finally:
                    parameter[index] = original
                gradient[index] = (plus - minus) / (2.0 * eps)
            result.append(gradient)
        # Restore derived model operators as well as the registered arrays.
        module()
    return result


class IdentityOutput:
    output_is_state = True

    def compute_output(self, q):
        return q

    def apply_output_adjoint(self, e, q):
        return e


def problem(backend):
    """Identical small forced-free ROM and data for either array backend."""
    rng = np.random.default_rng(93)
    phi = backend.asarray(np.linalg.qr(rng.standard_normal((4, 2)))[0])
    X = backend.asarray(rng.standard_normal((2, 4, 3)))
    data = SimpleNamespace(
        backend=backend,
        X=X,
        dX=-0.2 * X,
        time=backend.linspace(0.0, 0.2, 3, dtype=backend.float64),
        weights=backend.asarray(np.ones(2)),
        forcing_fns=[],
    )
    model = PolynomialModel(
        2, [1], tensors=[-0.2 * backend.eye(2, dtype=backend.float64)], backend=backend
    )
    projection = LinearProjection([phi, backend.copy(phi)], backend=backend)
    return data, model, projection
