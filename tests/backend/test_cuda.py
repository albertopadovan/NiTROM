"""Opt-in CUDA parity check for trajectory losses and analytic adjoints."""

from types import SimpleNamespace

import numpy as np
import pytest

from nitrom.latent_space_models import PolynomialModel
from nitrom.optimization import NitromModule
from nitrom.projections import LinearProjection
from nitrom.roms import ParamRegistry
from tests._support import IdentityOutput, problem

pytestmark = pytest.mark.gpu


def test_cuda_matches_cpu(torch_backend):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")
    data, model, projection = problem(torch_backend)
    cpu = NitromModule(
        data, ParamRegistry(model, projection), fom=IdentityOutput(), n_substeps=3
    )
    gpu_data = SimpleNamespace(**vars(data))
    for name in ["X", "dX", "time", "weights"]:
        setattr(gpu_data, name, getattr(data, name).cuda())
    gpu_model = PolynomialModel(
        2,
        [1],
        device="cuda",
        dtype=torch.float64,
        tensors=[value.cuda() for value in model.get_params()],
    )
    gpu_projection = LinearProjection(
        [value.cuda() for value in projection.get_params()]
    )
    gpu = NitromModule(
        gpu_data,
        ParamRegistry(gpu_model, gpu_projection),
        fom=IdentityOutput(),
        n_substeps=3,
    )
    np.testing.assert_allclose(gpu().item(), cpu().item(), atol=1e-12, rtol=1e-12)
    for actual, expected in zip(gpu.gradient(), cpu.gradient(), strict=True):
        assert actual.device.type == "cuda"
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected.detach().numpy(),
            atol=1e-11,
            rtol=1e-10,
        )
