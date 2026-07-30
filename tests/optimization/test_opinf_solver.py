import numpy as np
import pytest
import torch

from nitrom.backend import set_backend, get_backend
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import OpInfModule, solve_opinf, train
from nitrom.projections.linear_projection import LinearProjection


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    set_backend("torch")


class _MockOptObj:
    """Minimal stand-in for TrainingData with X, dX, weights."""

    def __init__(
        self, ntraj, N, nt, device="cpu", dtype=torch.float64, seed=0,
        forcing_fns=None,
    ):
        rng = np.random.default_rng(seed)
        self.X = torch.tensor(
            rng.standard_normal((ntraj, N, nt)), device=device, dtype=dtype
        )
        self.dX = torch.tensor(
            rng.standard_normal((ntraj, N, nt)), device=device, dtype=dtype
        )
        self.weights = torch.ones(ntraj, device=device, dtype=dtype)
        self.time = torch.linspace(0.0, 1.0, nt, device=device, dtype=dtype)
        self.forcing_fns = forcing_fns if forcing_fns is not None else []


def _make_phi(N, r, seed=42):
    rng = np.random.default_rng(seed)
    Phi, _ = np.linalg.qr(rng.standard_normal((N, r)))
    return torch.tensor(Phi, dtype=torch.float64)


def _make_forcing_fns(ntraj, m, seed=77):
    rng = np.random.default_rng(seed)
    coeffs = [
        torch.tensor(rng.standard_normal((m,)), dtype=torch.float64)
        for _ in range(ntraj)
    ]
    offsets = [
        torch.tensor(rng.standard_normal((m,)), dtype=torch.float64)
        for _ in range(ntraj)
    ]
    return [
        (lambda t, c=c, o=o: c * t + o) for c, o in zip(coeffs, offsets)
    ]


def _to_numpy(bkend, x):
    if bkend.is_torch:
        return x.detach().cpu().numpy()
    return np.asarray(x)


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
@pytest.mark.parametrize("forcing", [False, True])
@pytest.mark.parametrize("fixed_B", [False, True])
def test_solve_opinf(backend_name, forcing, fixed_B):
    # Skip invalid configuration: fixed_B has no meaning if forcing is False
    if fixed_B and not forcing:
        return

    # 1. Set backend
    set_backend(backend_name)
    bkend = get_backend()
    
    ntraj, N, nt, r, m = 3, 5, 6, 2, 1
    dtype = bkend.float64

    # 2. Set up training data
    forcing_fns = _make_forcing_fns(ntraj, m) if forcing else None
    opt_obj_torch = _MockOptObj(ntraj, N, nt, dtype=torch.float64, seed=42, forcing_fns=forcing_fns)
    
    # Cast to appropriate backend types
    class BackendData:
        def __init__(self, torch_data):
            self.X = bkend.asarray(torch_data.X.numpy(), dtype=dtype)
            self.dX = bkend.asarray(torch_data.dX.numpy(), dtype=dtype)
            self.weights = bkend.asarray(torch_data.weights.numpy(), dtype=dtype)
            self.time = bkend.asarray(torch_data.time.numpy(), dtype=dtype)
            self.forcing_fns = torch_data.forcing_fns

    training_data = BackendData(opt_obj_torch)
    Phi_torch = _make_phi(N, r)
    Phi = bkend.asarray(Phi_torch.numpy(), dtype=dtype)
    projection = LinearProjection([Phi, Phi])

    # 3. Model config
    forcing_config = None
    if forcing:
        B_fixed = None
        if fixed_B:
            # Generate a fixed B operator
            rng = np.random.default_rng(99)
            B_fom = bkend.asarray(rng.standard_normal((N, m)), dtype=dtype)
            B_fixed = projection.encode(bkend.permute(B_fom, (1, 0))).T  # (r, m)
        forcing_config = {"forcing_exists": True, "m": m, "B": B_fixed}

    rom = PolynomialModel(r, [1, 2], dtype=dtype, forcing_config=forcing_config)
    module = OpInfModule(training_data, rom, projection, reg=1e-6)

    if forcing and fixed_B:
        module.set_unlearnable("B")

    # 4. Solves analytically
    solve_opinf(module)

    # 5. Check gradient at the analytical solution is zero
    grads = module.gradient()
    for name, grad in zip(module.rom.param_names, grads):
        if module.is_learnable[name]:
            grad_np = _to_numpy(bkend, grad)
            # The gradient norm should be extremely small at the minimum
            assert np.max(np.abs(grad_np)) < 1e-8

    # 6. Check that L-BFGS optimization does not improve the loss significantly
    loss_analytical = float(_to_numpy(bkend, module()))
    
    # Run optimizer for 10 epochs
    train(module, n_epochs=10, lr=1.0, optimizer_type="lbfgs", tol=1e-12)
    loss_optimized = float(_to_numpy(bkend, module()))
    
    # Assert loss did not decrease (within numerical tolerance)
    assert loss_optimized <= loss_analytical + 1e-12
    # Ensure they are extremely close
    assert np.abs(loss_analytical - loss_optimized) < 1e-8
