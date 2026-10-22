"""Backend ownership and independent numerical checks on both array libraries."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import nitrom
from nitrom.backend import set_backend
from nitrom.bases import compute_data_driven_balancing, compute_pod
from nitrom.optimization import NitromModule, OpInfModule, PolyManifoldInfModule, train
from nitrom.projections import LinearProjection
from nitrom.roms import ParamRegistry
from tests._support import IdentityOutput, finite_diff_grad, problem


@pytest.mark.parametrize("adjoint_method", ["discrete", "continuous"])
def test_small_adjoint_matches_independent_differences(backend, adjoint_method):
    data, model, projection = problem(backend)
    module = NitromModule(
        data,
        ParamRegistry(model, projection),
        fom=IdentityOutput(),
        n_substeps=3,
        adjoint_method=adjoint_method,
    )
    analytic = module.gradient()
    numeric = finite_diff_grad(module)
    for actual, expected in zip(analytic, numeric, strict=True):
        np.testing.assert_allclose(
            backend.to_numpy(actual), backend.to_numpy(expected), atol=2e-8, rtol=2e-6
        )


def test_existing_modules_ignore_changes_to_default(backend):
    pytest.importorskip("torch")
    data, model, projection = problem(backend)
    registry = ParamRegistry(model, projection)
    nitrom = NitromModule(data, registry, fom=IdentityOutput(), n_substeps=3)
    loss = nitrom().item()
    gradients = [backend.to_numpy(g).copy() for g in nitrom.gradient()]
    set_backend("numpy" if backend.is_torch else "torch")
    # Constructing a module also uses its components, not the unrelated default.
    opinf = OpInfModule(data, model, projection)
    manifold = PolyManifoldInfModule(data, [2], projection.Phi)
    assert opinf.backend is backend
    assert manifold.proj.backend is backend
    assert nitrom().item() == loss
    for actual, expected in zip(nitrom.gradient(), gradients, strict=True):
        np.testing.assert_array_equal(backend.to_numpy(actual), expected)
    original = opinf().item()
    train(opinf, n_epochs=2, optimizer_type="sgd", lr=0.001, print_every=0)
    assert opinf().item() <= original


def test_mixed_components_are_rejected(backend):
    pytest.importorskip("torch")
    data, model, projection = problem(backend)
    other = set_backend("numpy" if backend.is_torch else "torch")
    wrong_projection = LinearProjection([other.asarray(np.eye(4, 2))] * 2)
    with pytest.raises(ValueError, match="Incompatible backends"):
        ParamRegistry(model, wrong_projection)
    wrong_data = SimpleNamespace(**vars(data))
    wrong_data.X = other.asarray(np.zeros((2, 4, 3)))
    with pytest.raises(TypeError, match="X must be"):
        OpInfModule(wrong_data, model, projection)


def test_basis_routines_follow_pool_backend(backend):
    pytest.importorskip("torch")
    data, _, _ = problem(backend)
    data.N, data.n_traj, data.n_snapshots = 4, 2, 3
    data.world_size, data.rank, data.comm = 1, 0, None
    expected = compute_pod(data)
    balanced = compute_data_driven_balancing(data, rcond=1e-3)
    set_backend("numpy" if backend.is_torch else "torch")
    actual = compute_pod(data)
    actual_balanced = compute_data_driven_balancing(data, rcond=1e-3)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(backend.to_numpy(a), backend.to_numpy(b))
    for a, b in zip(actual_balanced[:3], balanced[:3], strict=True):
        np.testing.assert_array_equal(backend.to_numpy(a), backend.to_numpy(b))


def test_numpy_session_never_imports_optional_libraries():
    # Block imports rather than relying on the current machine's installed extras.
    code = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mpi4py', 'matplotlib'}:
            raise ImportError('optional library blocked: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import nitrom
import numpy as np
from nitrom.distributed import distributed_rank_size
assert distributed_rank_size() == (0, 1)
model = nitrom.PolynomialModel(2, [1])
projection = nitrom.LinearProjection([np.eye(2), np.eye(2)])
registry = nitrom.ParamRegistry(model, projection)
solution = nitrom.solve_ivp(model.evaluate_rhs, np.ones(2), 0., 1., .1, np.linspace(0., 1., 3))
assert solution.shape == (2, 3)
assert not {'torch', 'mpi4py', 'matplotlib'}.intersection(sys.modules)
"""
    package_root = str(Path(nitrom.__file__).resolve().parent.parent)
    code = f"import sys; sys.path.insert(0, {package_root!r})\n" + code
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
